"""Offline tests of the fixed corruption protocol, data artifacts, and metrics."""

import hashlib
import json
import math
import sys
import pytest
import torch

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

from speedrunning_plms.research import benchmark


@pytest.fixture
def tokens() -> torch.Tensor:
    # (n=15, l=18), including a padded tail for variable eligible counts.
    return torch.tensor([
        window
        for sequence in ["LAGVSERTIDPKQNFYMHWCXBUZO", "AAA", "XXXXXX", "UOZB"] * 3
        for window in benchmark.encode_sequence(sequence, 18)
    ], dtype=torch.long)


@pytest.fixture
def data_dir(tmp_path: Path, tokens: torch.Tensor) -> Path:
    directory = tmp_path / "data"
    benchmark.write_dataset({"train": tokens, "valid": tokens[:3]}, directory)
    return directory


def test_encode_matches_esm_alphabet_and_preserves_long_tail() -> None:
    sequence = "LAGVSERTIDPKQNFYMHWCXBUZO.-"
    windows = benchmark.encode_sequence(sequence, 10)
    recovered = [token for row in windows for token in row if token not in (0, 1, 2)]
    assert recovered == list(range(4, 31))
    assert len(windows) == 4
    assert all(len(row) == 10 and row[0] == 0 and 2 in row for row in windows)
    assert windows[-1] == [0, 28, 29, 30, 2, 1, 1, 1, 1, 1]
    assert benchmark.encode_sequence(" a c\nD ", 5) == [[0, 5, 23, 13, 2]]


@pytest.mark.parametrize("sequence,length", [("", 8), ("  ", 8), ("A*", 8), ("A", 2)])
def test_encode_rejects_invalid_sequences(sequence: str, length: int) -> None:
    with pytest.raises(ValueError):
        benchmark.encode_sequence(sequence, length)


def test_corruption_is_exactly_mask_only_with_correct_targets(tokens: torch.Tensor) -> None:
    original = tokens.clone()  # (n, l)
    corrupted, labels = benchmark.corrupt_tokens(tokens, generator=torch.Generator().manual_seed(12))  # (n, l) each
    selected = labels.ne(-100)  # (n, l)
    assert selected.any()
    assert torch.equal(tokens, original)
    assert torch.equal(labels[selected], original[selected])
    assert torch.all(corrupted[selected] == 32)
    assert torch.equal(corrupted[~selected], original[~selected])
    assert torch.all((original[selected] >= 4) & (original[selected] <= 28))
    again = benchmark.corrupt_tokens(tokens, generator=torch.Generator().manual_seed(12))  # two (n, l) tensors
    assert all(torch.equal(left, right) for left, right in zip((corrupted, labels), again))


def test_corruption_masks_fifteen_percent_without_specials_or_gaps() -> None:
    # 250,000 eligible residues: sample error is below 0.2 percentage points.
    inputs = torch.arange(33).repeat(10_000, 1)  # (10000, 33)
    corrupted, labels = benchmark.corrupt_tokens(inputs, generator=torch.Generator().manual_seed(100))  # (10000, 33) each
    selected = labels.ne(-100)  # (10000, 33)
    assert abs(selected[:, 4:29].float().mean().item() - 0.15) < 0.002
    assert not selected[:, [0, 1, 2, 3, 29, 30, 31, 32]].any()
    assert torch.equal(corrupted[:, :4], inputs[:, :4])


def test_zero_masks_are_allowed_and_global_rng_is_untouched() -> None:
    before = torch.random.get_rng_state()  # (rng_state_bytes,)
    corrupted, labels = benchmark.corrupt_tokens(torch.tensor([[0, 5, 2]]), generator=torch.Generator().manual_seed(0))  # (1, 3) each
    assert torch.equal(corrupted, torch.tensor([[0, 5, 2]]))
    assert torch.all(labels == -100)
    assert torch.equal(before, torch.random.get_rng_state())


def test_evaluation_masks_are_invariant_to_batch_and_rank(tokens: torch.Tensor) -> None:
    expected = list(benchmark.evaluation_batches(tokens, 1, seed=91))
    batches = list(benchmark.evaluation_batches(tokens, 7, seed=91))
    for key in ("input_ids", "labels", "attention_mask"):
        assert torch.equal(torch.cat([row[key] for row in expected]), torch.cat([row[key] for row in batches]))
    for rank in range(4):
        shard = list(benchmark.evaluation_batches(tokens, 2, seed=91, rank=rank, world_size=4))
        for key in ("input_ids", "labels", "attention_mask"):
            assert torch.equal(torch.cat([row[key] for row in shard]), torch.cat([row[key] for row in expected[rank::4]]))
    assert torch.equal(torch.cat([row["attention_mask"] for row in expected]), tokens.ne(1).long())
    assert list(benchmark.evaluation_batches(tokens[:1], 8, rank=1, world_size=2)) == []


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": 1, "rank": -1}, {"batch_size": 1, "rank": 2, "world_size": 2}, {"batch_size": 1, "world_size": 0}])
def test_invalid_evaluation_partition(tokens: torch.Tensor, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        list(benchmark.evaluation_batches(tokens, **kwargs))


def test_dataset_roundtrip_content_hash_and_no_implicit_test(data_dir: Path, tokens: torch.Tensor) -> None:
    manifest = benchmark.load_manifest(data_dir)
    assert set(manifest["splits"]) == {"train", "valid"}
    assert torch.equal(benchmark.load_split(data_dir, "train"), tokens)
    assert torch.equal(benchmark.load_split(data_dir, "valid"), tokens[:3])
    fingerprint = benchmark.benchmark_id(data_dir)
    assert len(fingerprint) == 64
    # Formatting has no effect on benchmark identity.
    (data_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    assert benchmark.benchmark_id(data_dir) == fingerprint
    with pytest.raises(ValueError, match="not prepared"):
        benchmark.load_split(data_dir, "test")
    with pytest.raises(FileExistsError):
        benchmark.write_dataset({"train": tokens, "valid": tokens}, data_dir)


def test_data_checksums_are_enforced(data_dir: Path) -> None:
    with (data_dir / "valid.pt").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        benchmark.load_split(data_dir, "valid")


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True), ("objective", {}), ("tokenizer", {}),
    ("max_length", 2), ("max_length", True), ("dataset", {}), ("splits", {}),
])
def test_invalid_manifests(data_dir: Path, field: str, value: object) -> None:
    manifest = benchmark.load_manifest(data_dir)
    manifest[field] = value
    (data_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        benchmark.load_manifest(data_dir)


@pytest.mark.parametrize("field,value", [("file", "../valid.pt"), ("sha256", "invalid"), ("num_examples", 0), ("num_examples", True)])
def test_invalid_split_metadata(data_dir: Path, field: str, value: object) -> None:
    manifest = benchmark.load_manifest(data_dir)
    manifest["splits"]["valid"][field] = value
    (data_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        benchmark.load_manifest(data_dir)


@pytest.mark.parametrize("case", ["count", "width", "dtype", "range", "masked", "payload"])
def test_split_tensor_validation_even_with_matching_checksum(data_dir: Path, case: str) -> None:
    manifest = benchmark.load_manifest(data_dir)
    tokens = benchmark.load_split(data_dir, "valid").clone()  # (3, 18); release mapping before replacing file.
    if case == "count":
        tokens = tokens[:1]  # (1, 18)
    elif case == "width":
        tokens = tokens[:, :5]  # (3, 5)
    elif case == "dtype":
        tokens = tokens.float()  # (3, 18)
    elif case == "range":
        tokens[0, 0] = 33  # (3, 18)
    elif case == "masked":
        tokens[0, 1] = 32  # (3, 18)
    payload = {} if case == "payload" else {"input_ids": tokens}
    torch.save(payload, data_dir / "valid.pt")
    manifest["splits"]["valid"]["sha256"] = hashlib.sha256((data_dir / "valid.pt").read_bytes()).hexdigest()
    (data_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        benchmark.load_split(data_dir, "valid")


def test_write_validates_all_splits_before_creating_directory(tmp_path: Path, tokens: torch.Tensor) -> None:
    path = tmp_path / "bad"
    with pytest.raises(ValueError):
        benchmark.write_dataset({"train": tokens, "valid": tokens.float()}, path)
    assert not path.exists()


def test_saved_split_does_not_include_other_rows_in_shared_storage(data_dir: Path) -> None:
    valid = benchmark.load_split(data_dir, "valid")  # (3, 18)
    assert valid.untyped_storage().nbytes() == valid.numel() * valid.element_size()


def test_load_split_maps_storage_and_corruption_preserves_artifact(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_load = torch.load
    options = []

    def tracked_load(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        options.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    before = hashlib.sha256((data_dir / "train.pt").read_bytes()).hexdigest()
    mapped = benchmark.load_split(data_dir, "train")  # (n, l)
    original = mapped.clone()  # (n, l)
    corrupted, labels = benchmark.corrupt_tokens(mapped, generator=torch.Generator().manual_seed(42))  # (n, l) each
    assert options == [{"map_location": "cpu", "weights_only": True, "mmap": True}]
    assert labels.ne(-100).any()
    assert corrupted.data_ptr() != mapped.data_ptr()
    assert torch.equal(mapped, original)
    assert hashlib.sha256((data_dir / "train.pt").read_bytes()).hexdigest() == before


def test_manifest_and_train_loading_do_not_open_prepared_test_split(tmp_path: Path, tokens: torch.Tensor) -> None:
    benchmark.write_dataset({"train": tokens, "valid": tokens[:3], "test": tokens[-3:]}, tmp_path)
    (tmp_path / "test.pt").unlink()
    assert "test" in benchmark.load_manifest(tmp_path)["splits"]
    assert torch.equal(benchmark.load_split(tmp_path, "train"), tokens)
    with pytest.raises(FileNotFoundError):
        benchmark.load_split(tmp_path, "test")


@pytest.mark.parametrize("include_test", [False, True])
def test_prepare_streams_pinned_splits_and_keeps_tails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_test: bool) -> None:
    import datasets

    calls = []

    def load_dataset(repo_id: str, **kwargs: object) -> Iterator[dict[str, str]]:
        calls.append((repo_id, kwargs))
        yield {"sequence": "A" * 13}
        yield {"sequence": "C"}
        raise AssertionError("Read beyond requested source sequence bound")

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    output = tmp_path / "prepared"
    manifest = benchmark.prepare_dataset(output, max_length=8, train_sequences=2, eval_sequences=2, include_test=include_test)
    assert [call[1]["split"] for call in calls] == (["train", "valid", "test"] if include_test else ["train", "valid"])
    assert all(call[0] == "Synthyra/uniref50" and call[1]["streaming"] for call in calls)
    assert all(call[1]["revision"] == benchmark.DATASETS["uniref50"][1] for call in calls)
    assert manifest["splits"]["train"]["num_examples"] == 4
    train = benchmark.load_split(output, "train")  # (4, 8)
    assert train.eq(benchmark.RESIDUE_IDS["A"]).sum() == 13
    assert train.eq(benchmark.RESIDUE_IDS["C"]).sum() == 1


@pytest.mark.parametrize("kwargs", [{"dataset_name": "unknown"}, {"source_revision": "main"}, {"max_length": 2}, {"train_sequences": 0}, {"eval_sequences": 0}])
def test_prepare_rejects_invalid_settings_before_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, object]) -> None:
    import datasets

    def unexpected(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Unexpected dataset access")

    monkeypatch.setattr(datasets, "load_dataset", unexpected)
    with pytest.raises(ValueError):
        benchmark.prepare_dataset(tmp_path / "bad", **kwargs)


class FixedLogits(torch.nn.Module):
    def __init__(self, bad_logits: bool = False) -> None:
        super().__init__()
        self.bad_logits = bad_logits

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        # input_ids, attention_mask: (b, l).
        assert not self.training
        assert not torch.is_grad_enabled()
        logits = torch.arange(33, device=input_ids.device, dtype=torch.float32) / 10  # (33,)
        if self.bad_logits:
            logits[:] = float("nan")  # (33,)
        return SimpleNamespace(logits=logits.expand(*input_ids.shape, 33), loss=torch.tensor(-1000.0))


def test_evaluate_uses_token_weighted_logits_loss_and_bits(tokens: torch.Tensor) -> None:
    model = FixedLogits()
    expected_batches = list(benchmark.evaluation_batches(tokens, 1, seed=42))
    labels = torch.cat([batch["labels"] for batch in expected_batches])  # (n, l)
    targets = labels[labels.ne(-100)]  # (m,)
    weights = torch.arange(33, dtype=torch.float64) / 10  # (33,)
    expected_loss = (weights.logsumexp(0) - weights[targets]).mean().item()
    metrics = benchmark.evaluate_model(model, tokens, 7, "cpu")
    assert model.training
    assert metrics["loss"] == pytest.approx(expected_loss, abs=1e-6)
    assert metrics["bits_per_masked_residue"] == pytest.approx(expected_loss / math.log(2), abs=1e-6)
    assert metrics["masked_accuracy"] == 0
    assert metrics["masked_tokens"] == targets.numel()
    model.eval()
    assert benchmark.evaluate_model(model, tokens, 1, "cpu") == pytest.approx(metrics)
    assert not model.training


def test_evaluate_rejects_nonfinite_loss_and_restores_mode(tokens: torch.Tensor) -> None:
    model = FixedLogits(bad_logits=True)
    with pytest.raises(ValueError, match="non-finite"):
        benchmark.evaluate_model(model, tokens, 4, "cpu")
    assert model.training


def test_evaluate_rejects_no_masked_residues() -> None:
    with pytest.raises(ValueError, match="zero masked"):
        benchmark.evaluate_model(FixedLogits(), torch.tensor([[0, 2, 1]]), 1, "cpu")


def test_evaluate_rejects_uninitialized_distributed_group(tokens: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="initialized process group"):
        benchmark.evaluate_model(FixedLogits(), tokens, 1, "cpu", world_size=2)


def test_evaluate_reduces_global_totals_with_empty_local_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    reductions = []

    def all_reduce(totals: torch.Tensor, op: torch.distributed.ReduceOp) -> None:
        # totals: (3,); simulate rank 0 with 4 NLL, 1 correct, 2 masked.
        assert totals.tolist() == [0.0, 0.0, 0.0]
        reductions.append(op)
        totals += torch.tensor([4.0, 1.0, 2.0], dtype=torch.float64)  # (3,)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    metrics = benchmark.evaluate_model(FixedLogits(), torch.tensor([[0, 5, 2]]), 1, "cpu", rank=1, world_size=2)
    assert reductions == [torch.distributed.ReduceOp.SUM]
    assert metrics == {"loss": 2.0, "bits_per_masked_residue": 2 / math.log(2), "masked_accuracy": 0.5, "masked_tokens": 2}


def test_evaluate_checks_active_distributed_group(monkeypatch: pytest.MonkeyPatch, tokens: torch.Tensor) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    with pytest.raises(ValueError, match="active process group"):
        benchmark.evaluate_model(FixedLogits(), tokens, 1, "cpu")


def test_evaluate_positive_accuracy_ignores_model_loss() -> None:
    class PredictAlanine(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
            # input_ids/attention_mask: (b, l).
            logits = torch.zeros((*input_ids.shape, 33))  # (b, l, 33)
            logits[..., benchmark.RESIDUE_IDS["A"]] = 5  # (b, l, 33)
            return SimpleNamespace(logits=logits, loss=torch.tensor(float("nan")))

    inputs = torch.tensor(benchmark.encode_sequence("A" * 100, 20))  # (6, 20)
    metrics = benchmark.evaluate_model(PredictAlanine(), inputs, 2, "cpu")
    assert metrics["masked_accuracy"] == 1.0
    assert metrics["loss"] == pytest.approx(math.log(math.exp(5) + 32) - 5, abs=1e-6)


def test_evaluate_disables_ambient_autocast_and_preserves_context(tokens: torch.Tensor) -> None:
    class TinyMLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(33, 8)
            self.classifier = torch.nn.Linear(8, 33)
            self.output_dtypes = []

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
            # input_ids/attention_mask: (b, l).
            hidden = self.embedding(input_ids)  # (b, l, 8)
            logits = self.classifier(hidden)  # (b, l, 33)
            self.output_dtypes.append(logits.dtype)
            return SimpleNamespace(logits=logits)

    model = TinyMLM()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert model(tokens, tokens.ne(1)).logits.dtype == torch.bfloat16
    model.output_dtypes.clear()
    expected = benchmark.evaluate_model(model, tokens, 4, "cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = benchmark.evaluate_model(model, tokens, 4, "cpu")
        assert torch.is_autocast_enabled("cpu")
    assert not torch.is_autocast_enabled("cpu")
    assert actual == expected
    assert model.output_dtypes and set(model.output_dtypes) == {torch.float32}
    assert model.training


def test_prepare_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: [{"sequence": "LAGV"}])
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--output-dir", str(tmp_path / "cli"), "--dataset", "omg_prot50", "--train-sequences", "1", "--eval-sequences", "1"])
    benchmark.prepare_main()
    assert json.loads(capsys.readouterr().out)["benchmark_id"] == benchmark.benchmark_id(tmp_path / "cli")
    assert benchmark.load_manifest(tmp_path / "cli")["dataset"]["repo_id"] == "Synthyra/omg_prot50"
