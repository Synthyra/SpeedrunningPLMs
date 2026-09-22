"""Check configurable optimizations without changing the protein MLM benchmark."""

import json
import math
import pytest
import torch

from dataclasses import replace
from pathlib import Path

from speedrunning_plms.models import PLM
from speedrunning_plms.research import benchmark, engine


@pytest.mark.parametrize("progress,expected", [
    (0.0, 0.1), (0.05, 0.55), (0.1, 1.0), (0.5, 1.0), (0.75, 0.55), (1.0, 0.1), (2.0, 0.1),
])
def test_budget_lr_schedule(progress: float, expected: float) -> None:
    config = engine.ExperimentConfig(lr_schedule="warmup_cosine", warmup_fraction=0.1)
    assert engine.learning_rate_scale(progress, config) == pytest.approx(expected)
    assert engine.learning_rate_scale(progress, replace(config, lr_schedule="constant")) == 1.0


def test_zero_length_schedule_phases() -> None:
    config = engine.ExperimentConfig(lr_schedule="warmup_cosine", warmup_fraction=0, cooldown_fraction=0)
    assert engine.learning_rate_scale(0, config) == 1
    assert engine.learning_rate_scale(1, config) == 1


def test_accumulation_growth_keeps_microbatch_shape() -> None:
    config = engine.ExperimentConfig(grad_accum=2, grad_accum_final=4)
    assert [engine.accumulation_steps(p, config) for p in (0, 0.34, 0.67, 1)] == [2, 3, 4, 4]
    assert engine.accumulation_steps(0.9, replace(config, grad_accum_final=None)) == 2


def test_progress_uses_rank_zero_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    def broadcast(progress: torch.Tensor, src: int) -> None:
        assert src == 0
        progress.fill_(0.75)  # (): received from rank zero.

    monkeypatch.setattr(engine.dist, "broadcast", broadcast)
    monkeypatch.setattr(engine.time, "perf_counter", lambda: pytest.fail("Nonzero rank read its clock"))
    config = engine.ExperimentConfig(lr_schedule="warmup_cosine")
    assert engine._training_progress(0, config, torch.device("cpu"), 1, 2) == 0.75


@pytest.mark.parametrize("change", [
    {"optimizer": "sgd"}, {"muon_lr": 0}, {"muon_lr": True}, {"muon_steps": 0},
    {"muon_momentum": 1}, {"momentum_start": 1}, {"muon_backend": "svd"},
    {"muon_backend": "polar_express", "muon_steps": 3}, {"grad_accum_final": 0},
    {"grad_accum_final": True}, {"grad_accum": 4, "grad_accum_final": 2},
    {"min_lr_ratio": float("nan")}, {"warmup_fraction": True},
    {"warmup_fraction": 0.6, "cooldown_fraction": 0.6},
    {"lr_schedule": "unknown"}, {"attention_backend": "unknown"},
    {"prefetch": 1}, {"fused_adam": "true"}, {"value_embeddings": 1},
    {"embedding_residual": "true"}, {"value_embedding_gate": 1},
])
def test_invalid_optimization_settings(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        engine._validate(replace(engine.ExperimentConfig(), **change))


def test_optimization_cli_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    received = []
    monkeypatch.setattr(engine, "run_experiment", lambda config: received.append(config) or {})
    configuration = tmp_path / "experiment.json"
    configuration.write_text(json.dumps({"value_embeddings": True, "prefetch": True, "optimizer": "adamw"}))
    engine.main(["--config", str(configuration), "--no-value-embeddings", "--embedding-residual",
                 "--no-prefetch", "--optimizer", "muon", "--grad-accum-final", "3", "--fused-qkv"])
    config = received[0]
    assert config.value_embeddings is False and config.embedding_residual is True
    assert not config.prefetch and config.fused_qkv
    assert config.optimizer == "muon" and config.grad_accum_final == 3
    engine._validate(config)


@pytest.mark.parametrize("muon_backend", ["newton_schulz", "polar_express"])
def test_optimized_train_save_reload(tmp_path: Path, muon_backend: str) -> None:
    tokens = torch.tensor(benchmark.encode_sequence("ACDEFGHIKLMNPQRSTVWY" * 4, 16))  # (n, 16)
    directory = tmp_path / "data"
    benchmark.write_dataset({"train": tokens, "valid": tokens.flip(0)}, directory)
    config = engine.ExperimentConfig(
        data_dir=str(directory), output_dir=str(tmp_path / "run"), device="cpu", max_steps=2,
        time_budget=60, hidden_size=8, heads=2, layers=2, batch_size=2, grad_accum=2,
        optimizer="muon", muon_backend=muon_backend, fused_adam=True, prefetch=True,
        lr_schedule="warmup_cosine", momentum_warmup_fraction=0.05, grad_accum_final=3,
        attention_backend="sdpa", fused_qkv=True, value_embeddings=True,
        embedding_residual=True, value_embedding_gate=True,
    )
    result = engine.run_experiment(config)
    assert result["optimizer_steps"] == 2 and result["train_masked_tokens"] > 0
    assert math.isfinite(result["val_loss"])
    checkpoint = tmp_path / "run/checkpoint"
    model = PLM.from_pretrained(checkpoint, local_files_only=True)
    assert model.config.fused_qkv and model.config.value_embeddings
    assert model.config.attention_backend == "sdpa"
    evaluated = engine.run_experiment(replace(config, evaluate_only=str(checkpoint),
        output_dir=str(tmp_path / "evaluated")))
    assert evaluated["val_loss"] == pytest.approx(result["val_loss"], abs=1e-7)


def test_prefetch_does_not_change_training(tmp_path: Path) -> None:
    tokens = torch.tensor(benchmark.encode_sequence("ACDEFGHIKLMNPQRSTVWY" * 4, 16))  # (n, 16)
    directory = tmp_path / "data"
    benchmark.write_dataset({"train": tokens, "valid": tokens}, directory)
    config = engine.ExperimentConfig(
        data_dir=str(directory), output_dir=str(tmp_path / "serial"), device="cpu", max_steps=2,
        time_budget=60, hidden_size=8, heads=2, layers=2, batch_size=2, attention_backend="sdpa",
    )
    serial = engine.run_experiment(config)
    prefetched = engine.run_experiment(replace(config, prefetch=True, output_dir=str(tmp_path / "prefetched")))
    assert serial["val_loss"] == prefetched["val_loss"]
    assert serial["train_masked_tokens"] == prefetched["train_masked_tokens"]
