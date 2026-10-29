"""The fixed benchmark: tokens, corruption, data preparation, the evaluator, the clock, the result schema, the probe."""

from __future__ import annotations

import json
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pathlib import Path
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from torch import Tensor

from speedrunning_plms import benchmark
from speedrunning_plms.benchmark import data, probe, tokens


def test_residue_encoding_round_trips() -> None:
    ids = tokens.encode_residues("acdw")
    assert [tokens.ESM2_TOKENS[index] for index in ids] == ["A", "C", "D", "W"]
    with pytest.raises(ValueError):
        tokens.encode_residues("ACDXW")
    framed = torch.tensor([[tokens.CLS_ID, *ids, tokens.ESM2_IDS["X"], tokens.EOS_ID, tokens.PAD_ID]])  # (1, 8)
    labels = tokens.residue_labels(framed)  # (1, 8)
    assert labels[0, 1:5].tolist() == [tokens.RESIDUE_CLASS[residue] for residue in "ACDW"]
    assert labels[0, 5] == tokens.IGNORE_LABEL  # X is not a standard residue
    assert labels[0, 0] == tokens.IGNORE_LABEL and labels[0, 6] == tokens.IGNORE_LABEL and labels[0, 7] == tokens.IGNORE_LABEL
    assert tokens.is_residue(framed).tolist() == [[False, True, True, True, True, False, False, False]]


def test_fixed_mask_is_deterministic_and_near_the_rate() -> None:
    first, second = data.fixed_mask(100_000, 7), data.fixed_mask(100_000, 7)
    assert torch.equal(first, second)
    assert not torch.equal(first, data.fixed_mask(100_000, 8))
    assert abs(first.float().mean().item() - data.MASK_RATE) < 0.01


def test_corruption_masks_only_standard_residues() -> None:
    letters = [tokens.ESM2_IDS[letter] for letter in "ACDEFGHIKLMNPQRSTVWYXB" * 40]
    ids = torch.tensor([[tokens.CLS_ID, *letters, tokens.EOS_ID, *([tokens.PAD_ID] * 5)]])  # (1, l)
    generator = torch.Generator().manual_seed(0)
    corrupted, labels = data.corrupt(ids, generator)
    changed = corrupted != ids
    assert torch.equal(changed, labels != tokens.IGNORE_LABEL)
    assert (corrupted[changed] == tokens.MASK_ID).all()
    assert not changed[0, 0] and not changed[ids == tokens.EOS_ID].any() and not changed[ids == tokens.PAD_ID].any()
    assert not changed[(ids == tokens.ESM2_IDS["X"]) | (ids == tokens.ESM2_IDS["B"])].any()
    again, _ = data.corrupt(ids, torch.Generator().manual_seed(0))
    assert torch.equal(again, corrupted)


def test_prepare_builds_a_verifiable_benchmark(tiny_data: Path) -> None:
    manifest = benchmark.load_manifest(tiny_data)
    assert manifest["train"]["proteins"] == 300 and manifest["splits"]["validation"]["proteins"] == 40
    assert manifest["splits"]["val_select"]["proteins"] == 20 and manifest["splits"]["val_select"]["families"] == 20
    assert len(benchmark.benchmark_id(manifest)) == 64
    stream = benchmark.load_train(tiny_data)
    assert len(stream.offsets) == 301 and stream.tokens[0] == tokens.CLS_ID
    assert int((stream.tokens == tokens.MASK_ID).sum()) == 0
    assert len(stream.probe_subset) == 60  # one per family, the subset size is above the family count
    for split in data.HELD_OUT_SPLITS:
        held_out = benchmark.load_held_out(tiny_data, split)
        assert int((held_out.input_ids == tokens.MASK_ID).sum()) == 0
        assert not held_out.masked[~tokens.is_residue(held_out.input_ids.long())].any()
        assert torch.equal(held_out.lengths, tokens.is_residue(held_out.input_ids.long()).sum(dim=1))
    corrupt_file = tiny_data / "val_select.pt"
    original = corrupt_file.read_bytes()
    corrupt_file.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    try:
        with pytest.raises(ValueError):
            benchmark.load_held_out(tiny_data, "val_select")
    finally:
        corrupt_file.write_bytes(original)


class ToyScorer(torch.nn.Module):
    """Logits from a fixed table over the ESM2 alphabet, a different vocabulary width to exercise the column map."""

    def __init__(self, vocab: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.table = torch.nn.Parameter(torch.randn(tokens.VOCAB_SIZE, vocab, generator=generator))  # (33, v)
        self.residue_columns = torch.arange(tokens.NUM_CLASSES) + 5  # the 20 classes live at columns 5..24

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # input_ids: (b, l); returns (b, l, v).
        return self.table[input_ids]

    def hidden_states(self, input_ids: Tensor, attention_mask: Tensor) -> list[Tensor]:
        return [self.table[input_ids][..., :8], self.table[input_ids][..., 8:16]]


def test_evaluator_matches_a_direct_computation(tiny_data: Path) -> None:
    split = benchmark.load_held_out(tiny_data, "test")
    scorer = ToyScorer(vocab=40, seed=1)
    metrics = benchmark.evaluate_split(scorer, split, torch.device("cpu"), tokens_per_batch=4096, autocast_dtype=None)
    input_ids = split.input_ids.long()  # (n, l)
    labels = tokens.residue_labels(input_ids).masked_fill(~split.masked, tokens.IGNORE_LABEL)  # (n, l)
    corrupted = input_ids.masked_fill(split.masked, tokens.MASK_ID)
    logits = scorer.table[corrupted][..., scorer.residue_columns]  # (n, l, 20)
    selected = labels != tokens.IGNORE_LABEL
    scored, targets = logits[selected], labels[selected]
    assert metrics["scored_residues"] == int(selected.sum()) == int(split.masked.sum())
    assert metrics["loss"] == pytest.approx(F.cross_entropy(scored, targets).item(), rel=1e-5)
    predictions = scored.argmax(dim=-1).numpy()
    assert metrics["accuracy"] == pytest.approx((predictions == targets.numpy()).mean())
    assert metrics["mcc"] == pytest.approx(matthews_corrcoef(targets.numpy(), predictions), abs=1e-6)
    for name, scorer_function in (("precision", precision_score), ("recall", recall_score), ("f1", f1_score)):
        for average in ("macro", "weighted"):
            expected = scorer_function(targets.numpy(), predictions, average=average, labels=list(range(20)), zero_division=0)
            assert metrics[f"{name}_{average}"] == pytest.approx(expected, abs=1e-6), (name, average)
    assert sum(metrics["residues_by_length"].values()) == metrics["scored_residues"]
    assert 0.0 <= metrics["roc_auc_macro"] <= 1.0 and 0.0 <= metrics["pr_auc_macro"] <= 1.0
    assert metrics["perplexity"] == pytest.approx(np.exp(metrics["loss"]))


def test_masked_cross_entropy_agrees_away_from_collapse() -> None:
    logits = torch.randn(4, 9, tokens.VOCAB_SIZE)
    labels = torch.randint(0, tokens.NUM_CLASSES, (4, 9))
    labels[:, ::3] = tokens.IGNORE_LABEL
    expected = F.cross_entropy(logits[..., tokens.ESM2_RESIDUE_COLUMNS].flatten(0, 1), labels.flatten(),
                               ignore_index=tokens.IGNORE_LABEL, reduction="sum")
    assert benchmark.masked_cross_entropy(logits, labels).item() == pytest.approx(expected.item(), rel=1e-5)


class FakeTimer:
    def __init__(self) -> None:
        self.time = 0.0

    def __call__(self) -> float:
        return self.time


def test_clock_counts_applied_steps_and_discards_the_overrun() -> None:
    timer = FakeTimer()
    clock = benchmark.Clock(torch.device("cpu"), budget=10.0, now=timer, synchronize=lambda: None)
    with clock.warming_up():
        timer.time += 2.0
    applied = 0
    while True:
        clock.start_step()
        timer.time += 3.0
        if clock.expired():
            break
        clock.applied()
        applied += 1
    receipt = clock.receipt()
    assert applied == receipt["steps"] == 3 and receipt["discarded_steps"] == 1
    assert receipt["train_seconds"] == pytest.approx(9.0) and receipt["warmup_seconds"] == pytest.approx(2.0)
    assert receipt["valid"] and clock.progress == pytest.approx(0.9)


def test_clock_caps_mark_a_run_invalid() -> None:
    timer = FakeTimer()
    clock = benchmark.Clock(torch.device("cpu"), budget=10.0, warmup_cap=5.0, eval_cap=4.0, now=timer, synchronize=lambda: None)
    with clock.warming_up():
        timer.time += 6.0
    assert not clock.warmup_valid and not clock.valid
    clock = benchmark.Clock(torch.device("cpu"), budget=10.0, warmup_cap=5.0, eval_cap=4.0, now=timer, synchronize=lambda: None)
    clock.start_step()
    timer.time += 1.0
    clock.applied()
    with clock.paused():
        timer.time += 5.0
    clock.start_step()
    timer.time += 1.0
    clock.applied()
    assert clock.train_seconds == pytest.approx(2.0)  # the pause is outside the clock
    assert clock.eval_seconds == pytest.approx(5.0) and not clock.eval_valid and not clock.valid


def test_result_is_written_once_and_rows_the_ledger(tmp_path: Path) -> None:
    clock_receipt = benchmark.Clock(torch.device("cpu"), now=lambda: 0.0, synchronize=lambda: None).receipt()
    key = benchmark.comparison_key("data", benchmark.evaluator_sha256(), clock_receipt)
    run_result = {"name": "x", "status": "completed", "seed": 0, "candidate_sha256": "c", "comparison_key": key,
              "clock": clock_receipt, "metrics": {"test": {"loss": 2.0, "accuracy": 0.3, "mcc": 0.1}}}
    path = benchmark.write_result(tmp_path, run_result)
    assert json.loads(path.read_text(encoding="utf-8"))["comparison_key"] == key
    with pytest.raises(FileExistsError):
        benchmark.write_result(tmp_path, run_result)
    row = benchmark.ledger_row(run_result)
    assert row["test_loss"] == 2.0 and row["valid"] and row["steps"] == 0
    other = {**row, "comparison_key": "different"}
    assert benchmark.comparable(row, row) and not benchmark.comparable(row, other)
    assert benchmark.comparison_key("data", benchmark.evaluator_sha256(), {**clock_receipt, "budget_seconds": 1.0}) != key


def test_probe_trains_and_scores_on_hidden_states(tiny_data: Path) -> None:
    stream = benchmark.load_train(tiny_data)
    model = ToyScorer(vocab=40, seed=2)
    head = probe.train_head(model, stream, torch.device("cpu"), hidden=8, num_hidden_states=2, autocast_dtype=None)
    assert head.layer_logits.shape == (2,)
    metrics = probe.score_head(model, head, benchmark.load_held_out(tiny_data, "validation"), torch.device("cpu"),
                               tokens_per_batch=4096, autocast_dtype=None)
    split = benchmark.load_held_out(tiny_data, "validation")
    assert metrics["scored_residues"] == int((split.three_di != tokens.NO_STATE).sum()) == int(split.lengths.sum())
    assert len(metrics["layer_weights"]) == 2 and sum(metrics["layer_weights"]) == pytest.approx(1.0)
    assert 0.0 <= metrics["accuracy"] <= 1.0 and "f1_macro" in metrics and "mcc" in metrics
