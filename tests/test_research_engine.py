"""CPU checks for training, benchmark isolation, and experiment artifacts."""

import json
import math
import os
import socket
import subprocess
import sys
import pytest
import torch

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from speedrunning_plms.models import PLM
from speedrunning_plms.research import benchmark, engine


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    tokens = torch.tensor(benchmark.encode_sequence("ACDEFGHIKLMNPQRSTVWY" * 8, 16))  # (n, 16)
    directory = tmp_path / "data"
    benchmark.write_dataset({"train": tokens, "valid": tokens.flip(0), "test": tokens.roll(1, 0)}, directory)
    return directory


def tiny_config(prepared: Path, tmp_path: Path, **kwargs: object) -> engine.ExperimentConfig:
    return engine.ExperimentConfig(data_dir=str(prepared), output_dir=str(tmp_path / "run"),
        device="cpu", hidden_size=8, heads=2, layers=2, batch_size=2,
        time_budget=30, max_steps=2, **kwargs)


@pytest.mark.parametrize("architecture", ["standard", "unet", "patch_unet"])
def test_cpu_train_save_reload_and_evaluate(prepared: Path, tmp_path: Path, architecture: str) -> None:
    config = tiny_config(prepared, tmp_path, architecture=architecture)
    result = engine.run_experiment(config)
    assert result["optimizer_steps"] == 2
    assert result["train_masked_tokens"] > 0
    assert result["masked_tokens"] > 0
    assert result["world_size"] == 1
    assert result["val_bits_per_masked_residue"] == pytest.approx(result["val_loss"] / math.log(2))
    assert 0 <= result["masked_accuracy"] <= 1
    assert result["wall_seconds"] >= result["train_seconds"] > 0
    assert result["peak_vram_mb"] == 0
    assert json.loads((tmp_path / "run/result.json").read_text())["benchmark_id"] == result["benchmark_id"]
    checkpoint = tmp_path / "run/checkpoint"
    model = PLM.from_pretrained(checkpoint, local_files_only=True)
    assert model.config.mlm and not model.config.masked_diffusion
    assert model.tokenizer is None
    rerun = engine.run_experiment(replace(config, evaluate_only=str(checkpoint), output_dir=str(tmp_path / "evaluation")))
    assert rerun["val_loss"] == pytest.approx(result["val_loss"], abs=1e-7)
    assert rerun["optimizer_steps"] == 0
    assert not (tmp_path / "evaluation/checkpoint").exists()
    with pytest.raises(FileExistsError):
        engine.run_experiment(config)


def test_validation_never_reads_test(prepared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    read_splits = []
    original = benchmark.load_split

    def tracked(directory: Path, split: str) -> torch.Tensor:
        read_splits.append(split)
        return original(directory, split)

    monkeypatch.setattr(benchmark, "load_split", tracked)
    engine.run_experiment(tiny_config(prepared, tmp_path))
    assert set(read_splits) == {"train", "valid"}


def test_test_split_requires_explicit_checkpoint(prepared: Path, tmp_path: Path) -> None:
    config = tiny_config(prepared, tmp_path, split="test")
    with pytest.raises(ValueError, match="evaluate-only"):
        engine.run_experiment(config)


def test_held_out_evaluation_reads_only_test_and_reports_checkpoint_architecture(
    prepared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_config(prepared, tmp_path, architecture="unet")
    engine.run_experiment(config)
    read_splits = []
    original = benchmark.load_split

    def tracked(directory: Path, split: str) -> torch.Tensor:
        read_splits.append(split)
        return original(directory, split)

    monkeypatch.setattr(benchmark, "load_split", tracked)
    output_dir = tmp_path / "held_out"
    result = engine.run_experiment(replace(config, split="test", architecture="standard",
        evaluate_only=str(tmp_path / "run/checkpoint"), output_dir=str(output_dir)))
    assert read_splits == ["test"]
    assert result["split"] == "test"
    assert math.isfinite(result["test_loss"])
    assert result["test_bits_per_masked_residue"] == pytest.approx(result["test_loss"] / math.log(2))
    assert "val_loss" not in result
    assert "val_bits_per_masked_residue" not in result
    assert result["optimizer_steps"] == 0
    assert result["train_masked_tokens"] == 0
    assert result["train_seconds"] == 0
    assert result["architecture"] == "unet"
    assert result["model_config"]["unet"] is True
    assert not (output_dir / "checkpoint").exists()
    assert json.loads((output_dir / "result.json").read_text())["test_loss"] == result["test_loss"]


def test_reproducible_training(prepared: Path, tmp_path: Path) -> None:
    config = tiny_config(prepared, tmp_path)
    first = engine.run_experiment(config)
    second = engine.run_experiment(replace(config, output_dir=str(tmp_path / "second")))
    assert first["val_loss"] == second["val_loss"]
    assert first["train_masked_tokens"] == second["train_masked_tokens"]


def test_training_shards_cover_global_epoch() -> None:
    tokens = torch.arange(12).reshape(12, 1)  # (12, 1)
    ranks = [engine.training_batches(tokens, 2, 42, rank, 3) for rank in range(3)]
    examples = torch.cat([next(iterator) for _ in range(2) for iterator in ranks]).flatten()  # (12,)
    assert sorted(examples.tolist()) == list(range(12))
    repeats = engine.training_batches(tokens[:1], 3, 42, 0, 2)
    assert next(repeats).shape == (3, 1)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.arange(33, dtype=torch.float32) / 33)  # (33,)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        # input_ids, attention_mask: (b, l); logits: (b, l, 33)
        return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, 33))


def fixed_corruption(tokens: torch.Tensor, *, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    # tokens: (b, l); positions with token >= 4 are all supervised for this unit test.
    labels = tokens.clone()  # (b, l)
    labels[tokens < 4] = -100  # (b, l)
    return tokens, labels


def test_gradient_accumulation_weights_masked_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "corrupt_tokens", fixed_corruption)
    tokens = torch.tensor([[4, 1, 1], [5, 6, 7], [8, 9, 1], [10, 11, 12]])  # (4, 3)
    accumulated, full_batch = TinyModel(), TinyModel()
    config = engine.ExperimentConfig(max_steps=1, time_budget=30, batch_size=1, grad_accum=4)
    engine._train(accumulated, tokens, config, torch.device("cpu"), 0, 1)
    engine._train(full_batch, tokens, replace(config, batch_size=4, grad_accum=1), torch.device("cpu"), 0, 1)
    torch.testing.assert_close(accumulated.logits.grad, full_batch.logits.grad)
    torch.testing.assert_close(accumulated.logits, full_batch.logits)


def test_empty_mask_batch_does_not_update_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "corrupt_tokens", fixed_corruption)
    model = TinyModel()
    before = model.logits.detach().clone()  # (33,)
    tokens = torch.ones((2, 3), dtype=torch.long)  # (2, 3)
    steps, masked, _ = engine._train(model, tokens, engine.ExperimentConfig(max_steps=2), torch.device("cpu"), 0, 1)
    assert steps == masked == 0
    torch.testing.assert_close(model.logits, before)


def test_time_budget_stops_before_any_unmetered_step(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([0.0, 1.0, 1.5])
    monkeypatch.setattr(engine.time, "perf_counter", lambda: next(clock))
    model = TinyModel()
    steps, masked, elapsed = engine._train(model, torch.ones((2, 3), dtype=torch.long),
        engine.ExperimentConfig(time_budget=0.5), torch.device("cpu"), 0, 1)
    assert (steps, masked, elapsed) == (0, 0, 1.5)


@pytest.mark.parametrize("grad_accum", [1, 4])
def test_deadline_discards_overtime_accumulation(grad_accum: int, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([0.0, 0.1, 0.2, 1.1, 1.2])
    monkeypatch.setattr(engine.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(benchmark, "corrupt_tokens", fixed_corruption)
    model = TinyModel()
    before = model.logits.detach().clone()  # (33,)
    tokens = torch.tensor([[4, 5, 1], [6, 7, 8]])  # (2, 3)
    steps, masked, elapsed = engine._train(model, tokens,
        engine.ExperimentConfig(time_budget=1, grad_accum=grad_accum), torch.device("cpu"), 0, 1)
    assert (steps, masked, elapsed) == (0, 0, 1.2)
    torch.testing.assert_close(model.logits, before)
    assert model.logits.grad is None


def test_deadline_broadcast_controls_nonzero_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    broadcasts = []

    def broadcast(stop: torch.Tensor, src: int) -> None:
        broadcasts.append(src)
        stop.fill_(1)  # (): rank zero reached the deadline

    monkeypatch.setattr(engine.dist, "broadcast", broadcast)
    monkeypatch.setattr(engine.time, "perf_counter", lambda: pytest.fail("Only rank zero decides the deadline"))
    assert engine._deadline_reached(0, 300, torch.device("cpu"), rank=1, world_size=2)
    assert broadcasts == [0]


@pytest.mark.parametrize("other", [("different_data", "code"), ("data", "different_code")])
def test_distributed_benchmark_mismatch_fails(other: tuple[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    def gather(identities: list[tuple[str, str]], local: tuple[str, str]) -> None:
        identities[:] = [local, other]

    monkeypatch.setattr(engine.dist, "all_gather_object", gather)
    with pytest.raises(ValueError, match="different benchmark"):
        engine._verify_distributed_benchmark("data", "code", 2)


def test_bf16_setting_does_not_change_evaluation(prepared: Path, tmp_path: Path) -> None:
    config = tiny_config(prepared, tmp_path)
    trained = engine.run_experiment(config)
    result = engine.run_experiment(replace(config, evaluate_only=str(tmp_path / "run/checkpoint"),
        output_dir=str(tmp_path / "bf16_eval"), bf16=True))
    assert result["val_loss"] == trained["val_loss"]
    assert result["eval_dtype"] == "float32"


@pytest.mark.parametrize("change", [
    {"time_budget": 0}, {"time_budget": float("nan")}, {"learning_rate": 0},
    {"weight_decay": -1}, {"batch_size": 0}, {"max_steps": -1},
    {"hidden_size": 7}, {"architecture": "diffusion"}, {"compile": "true"},
    {"architecture": "unet", "layers": 3}, {"architecture": "patch_unet", "patch_layers": 3},
    {"device": "tpu"}, {"split": "train"},
])
def test_invalid_settings_fail_before_artifacts(tmp_path: Path, change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        engine.run_experiment(replace(engine.ExperimentConfig(output_dir=str(tmp_path / "run")), **change))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("field", ["time_budget", "learning_rate", "weight_decay"])
def test_numeric_settings_reject_booleans(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        engine._validate(replace(engine.ExperimentConfig(), **{field: True}))


@pytest.mark.parametrize("seed", [True, 1.5, -(2**63) - 1, 2**64])
def test_seed_rejects_nonintegers_and_overflow(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        engine._validate(replace(engine.ExperimentConfig(), seed=seed))


@pytest.mark.parametrize("seed", [-(2**63), 2**64 - 1])
def test_training_supports_torch_seed_boundaries(
    seed: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "corrupt_tokens", fixed_corruption)
    tokens = torch.tensor([[4, 5, 6]])  # (1, 3)
    config = engine.ExperimentConfig(seed=seed, max_steps=1, batch_size=1)
    engine._validate(config)
    steps, masked, _ = engine._train(TinyModel(), tokens, config, torch.device("cpu"), 0, 1)
    assert steps == 1
    assert masked == 3


@pytest.mark.parametrize("rank,world_size,local_rank", [
    (1, 1, 0), (-1, 1, 0), (0, 0, 0), (0, 1, -1),
])
def test_invalid_distributed_environment_fails_before_artifacts(
    rank: int, world_size: int, local_rank: int,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in (("RANK", rank), ("WORLD_SIZE", world_size), ("LOCAL_RANK", local_rank)):
        monkeypatch.setenv(name, str(value))
    config = engine.ExperimentConfig(output_dir=str(tmp_path / "run"))
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        engine.run_experiment(config)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("field,value", [("grad_accum", 3), ("max_steps", 3), ("batch_size", 3)])
def test_distributed_configuration_mismatch_fails(
    field: str, value: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def gather(configurations: list[dict[str, object]], local: dict[str, object]) -> None:
        configurations[:] = [local, {**local, field: value}]

    monkeypatch.setattr(engine.dist, "all_gather_object", gather)
    with pytest.raises(ValueError, match="different experiment configurations"):
        engine._verify_distributed_config(engine.ExperimentConfig(), 2)


def test_distributed_configuration_excludes_machine_local_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def gather(configurations: list[dict[str, object]], local: dict[str, object]) -> None:
        assert "data_dir" not in local
        assert "output_dir" not in local
        configurations[:] = [local.copy(), local.copy()]

    monkeypatch.setattr(engine.dist, "all_gather_object", gather)
    engine._verify_distributed_config(engine.ExperimentConfig(), 2)


def test_cli_overrides_json_and_rejects_unknown_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    received = []
    monkeypatch.setattr(engine, "run_experiment", lambda config: received.append(config) or {})
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps({"batch_size": 3, "compile": True, "time_budget": 9}))
    engine.main(["--config", str(config_path), "--batch-size", "7", "--no-compile", "--time-budget", "5"])
    assert received[0].batch_size == 7
    assert received[0].time_budget == 5
    assert not received[0].compile
    config_path.write_text('{"mask_rate": 0.5}')
    with pytest.raises(SystemExit):
        engine.main(["--config", str(config_path)])


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="PyTorch lacks Gloo")
def test_two_process_cpu_training_and_uneven_evaluation(prepared: Path, tmp_path: Path) -> None:
    # Twelve examples split into uneven 7/6 shards after adding one sequence.
    train = benchmark.load_split(prepared, "train")  # (n, l)
    valid = torch.cat((train, train[:1]))  # (n + 1, l)
    assert len(valid) % 2 == 1
    distributed_data = tmp_path / "distributed_data"
    benchmark.write_dataset({"train": train, "valid": valid}, distributed_data)
    config = replace(tiny_config(distributed_data, tmp_path), architecture="unet", max_steps=1)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(engine.asdict(config)))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = os.environ | {
        "USE_LIBUV": "0", "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    command = [sys.executable, "-m", "speedrunning_plms.research.engine", "--config", str(config_path)]
    # Direct workers exercise torchrun's environment contract without its Windows
    # static rendezvous server, which forces unavailable libuv in PyTorch 2.6.
    workers = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=environment | {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2",
                           "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)}) for rank in range(2)]
    try:
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=60)
            assert worker.returncode == 0, stdout + stderr
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
    distributed_result = json.loads((tmp_path / "run/result.json").read_text())
    assert distributed_result["world_size"] == 2
    assert distributed_result["optimizer_steps"] == 1
    distributed_model = PLM.from_pretrained(tmp_path / "run/checkpoint", local_files_only=True)
    torch.manual_seed(config.seed)
    reference = PLM(distributed_model.config)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    rank_counts = []
    for rank in range(2):
        batch = next(engine.training_batches(train, config.batch_size, config.seed, rank, 2))  # (b, l)
        inputs, labels = benchmark.corrupt_tokens(batch, generator=torch.Generator().manual_seed(config.seed + 1 + rank))  # each (b, l)
        logits = reference(input_ids=inputs, attention_mask=inputs != 1).logits  # (b, l, c)
        engine._loss_sum(logits, labels).backward()
        rank_counts.append((labels != -100).sum().item())
    assert rank_counts[0] != rank_counts[1]
    for parameter in reference.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(sum(rank_counts))  # same shape as parameter
    optimizer.step()
    for actual, expected in zip(distributed_model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    single_result = engine.run_experiment(replace(config, evaluate_only=str(tmp_path / "run/checkpoint"),
                                                output_dir=str(tmp_path / "single_eval")))
    assert distributed_result["masked_tokens"] == single_result["masked_tokens"]
    assert distributed_result["val_loss"] == pytest.approx(single_result["val_loss"], rel=1e-6)
