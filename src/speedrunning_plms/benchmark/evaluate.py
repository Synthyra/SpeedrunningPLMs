"""The fixed evaluator: 20-way masked-residue scoring of any model on a held-out split.

A scorer maps corrupted ESM2 ids to logits over its own vocabulary and names the 20 columns of its vocabulary that are
the standard residues, in the benchmark's class order. The evaluator corrupts each protein at its fixed mask positions,
gathers the 20 columns, and accumulates the collapse-free cross entropy, the confusion matrix, the per-class
probabilities and the loss by length bin. Logits are read in float32; the forward runs under the scorer's own precision.

Symbols: b proteins in a batch, l tokens, v the scorer's vocabulary, c = 20 classes, m scored residues.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol
from torch import Tensor

from foundry.layers.cross_entropy import cross_entropy
from speedrunning_plms.benchmark import tokens
from speedrunning_plms.benchmark.data import HeldOutSplit, length_bin


class Scorer(Protocol):
    """What the evaluator needs from a model."""

    residue_columns: Tensor  # (20,), the scorer's vocabulary ids of the standard residues in class order

    def __call__(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Logits over the scorer's vocabulary for ESM2 ids that may hold MASK."""
        # input_ids: (b, l) long; attention_mask: (b, l) bool; returns (b, l, v).
        ...


def masked_cross_entropy(logits: Tensor, labels: Tensor, residue_columns: Tensor = tokens.ESM2_RESIDUE_COLUMNS) -> Tensor:
    """Summed collapse-free cross entropy over the labelled positions, on the 20 residue columns; the training loss
    candidates import, the same 20-way problem the evaluator scores."""
    # logits: (..., v) any float dtype; labels: (...) long with IGNORE_LABEL where unscored; returns ().
    scored = logits[..., residue_columns.to(logits.device)].float().flatten(0, -2)  # (n, c)
    return cross_entropy(scored, labels.flatten(), ignore_index=tokens.IGNORE_LABEL, reduction="sum")


def batches_by_length(split: HeldOutSplit, tokens_per_batch: int) -> Iterator[Tensor]:
    """Row indices in descending-length order, each batch at most `tokens_per_batch` tokens after trimming padding."""
    # Rows are stored longest first, so a batch's width is its first row's width.
    n = len(split)
    start = 0
    while start < n:
        width = int(split.lengths[start]) + 2
        size = max(1, min(n - start, tokens_per_batch // min(width, split.input_ids.shape[1])))
        yield torch.arange(start, start + size)  # (b,)
        start += size


@dataclass
class _Accumulator:
    nll: float = 0.0
    count: int = 0
    confusion: Tensor | None = None  # (c, c), rows true, columns predicted
    bin_nll: dict[str, float] | None = None
    bin_count: dict[str, int] | None = None
    probabilities: list[Tensor] | None = None  # each (m_batch, c), float16
    labels: list[Tensor] | None = None  # each (m_batch,), uint8


def _macro_auc(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Macro one-vs-rest ROC AUC and average precision over the classes present in the labels."""
    # probabilities: (m, c); labels: (m,).
    from sklearn.metrics import average_precision_score, roc_auc_score

    present = np.unique(labels)
    roc = [roc_auc_score(labels == label, probabilities[:, label]) for label in present if (labels == label).sum() < len(labels)]
    pr = [average_precision_score(labels == label, probabilities[:, label]) for label in present if (labels == label).sum() < len(labels)]
    return float(np.mean(roc)), float(np.mean(pr))


def _classification_metrics(confusion: Tensor) -> dict[str, float]:
    """Accuracy, macro and weighted precision, recall and F1, and the multiclass MCC from a confusion matrix."""
    # confusion: (c, c) int64.
    C = confusion.double()  # (c, c)
    total = C.sum()  # ()
    true_positive = C.diagonal()  # (c,)
    predicted = C.sum(dim=0)  # (c,)
    actual = C.sum(dim=1)  # (c,)
    precision = torch.where(predicted > 0, true_positive / predicted.clamp(min=1), torch.zeros_like(predicted))  # (c,)
    recall = torch.where(actual > 0, true_positive / actual.clamp(min=1), torch.zeros_like(actual))  # (c,)
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall).clamp(min=1e-12), torch.zeros_like(precision))  # (c,)
    present = actual > 0  # (c,)
    weights = actual / total  # (c,)
    covariance = total * true_positive.sum() - (predicted * actual).sum()  # ()
    denominator = torch.sqrt((total**2 - (predicted**2).sum()) * (total**2 - (actual**2).sum()))  # ()
    return {
        "accuracy": float(true_positive.sum() / total),
        "precision_macro": float(precision[present].mean()), "recall_macro": float(recall[present].mean()),
        "f1_macro": float(f1[present].mean()),
        "precision_weighted": float((precision * weights).sum()), "recall_weighted": float((recall * weights).sum()),
        "f1_weighted": float((f1 * weights).sum()),
        "mcc": float(covariance / denominator) if float(denominator) > 0 else 0.0,
    }


def evaluate_split(
    scorer: Scorer, split: HeldOutSplit, device: torch.device, *, tokens_per_batch: int = 32_768,
    autocast_dtype: torch.dtype | None = torch.bfloat16, keep_probabilities: bool = True,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Score one held-out split; `keep_probabilities=False` skips the AUCs for a cheap in-run evaluation."""
    c = tokens.NUM_CLASSES
    accumulator = _Accumulator(confusion=torch.zeros((c, c), dtype=torch.long), bin_nll={}, bin_count={},
                               probabilities=[] if keep_probabilities else None, labels=[] if keep_probabilities else None)
    columns = scorer.residue_columns.to(device)  # (c,)
    enabled = autocast_dtype is not None and device.type == "cuda"
    with torch.inference_mode():
        for rows in batches_by_length(split, tokens_per_batch):
            width = int(split.lengths[rows[0]]) + 2
            width = min(width, split.input_ids.shape[1])
            input_ids = split.input_ids[rows, :width].long().to(device)  # (b, l)
            masked = split.masked[rows, :width].to(device)  # (b, l)
            attention_mask = input_ids != tokens.PAD_ID  # (b, l)
            labels = tokens.residue_labels(input_ids).masked_fill(~masked, tokens.IGNORE_LABEL)  # (b, l)
            corrupted = input_ids.masked_fill(masked, tokens.MASK_ID)  # (b, l)
            with torch.autocast(device.type, dtype=autocast_dtype or torch.float32, enabled=enabled):
                logits = scorer(corrupted, attention_mask)  # (b, l, v)
            selected = labels != tokens.IGNORE_LABEL  # (b, l)
            scored = logits[selected].float()[:, columns]  # (m, c)
            targets = labels[selected]  # (m,)
            losses = cross_entropy(scored, targets, reduction="none")  # (m,)
            accumulator.nll += float(losses.double().sum())
            accumulator.count += int(targets.numel())
            predictions = scored.argmax(dim=-1)  # (m,)
            accumulator.confusion += torch.bincount(targets * c + predictions, minlength=c * c).view(c, c).cpu()  # (c, c)
            per_row = selected.sum(dim=1)  # (b,)
            row_loss = torch.zeros(len(rows), dtype=torch.float64, device=device).index_add_(
                0, torch.repeat_interleave(torch.arange(len(rows), device=device), per_row), losses.double())  # (b,)
            for row, loss_sum, count in zip(rows.tolist(), row_loss.tolist(), per_row.tolist(), strict=True):
                name = length_bin(int(split.lengths[row]))
                accumulator.bin_nll[name] = accumulator.bin_nll.get(name, 0.0) + loss_sum
                accumulator.bin_count[name] = accumulator.bin_count.get(name, 0) + count
            if keep_probabilities:
                accumulator.probabilities.append(scored.softmax(dim=-1).half().cpu())  # (m, c)
                accumulator.labels.append(targets.to(torch.uint8).cpu())  # (m,)
            if progress is not None:
                progress(len(rows))
    if accumulator.count == 0:
        raise ValueError("the split has no scored residues")
    if not math.isfinite(accumulator.nll):
        raise ValueError("the evaluation produced a non-finite cross entropy")
    loss = accumulator.nll / accumulator.count
    metrics: dict[str, Any] = {
        "loss": loss, "bits_per_residue": loss / math.log(2), "perplexity": math.exp(loss),
        "scored_residues": accumulator.count, "proteins": len(split),
        **_classification_metrics(accumulator.confusion),
        "loss_by_length": {name: accumulator.bin_nll[name] / accumulator.bin_count[name] for name in sorted(accumulator.bin_nll)},
        "residues_by_length": dict(sorted(accumulator.bin_count.items())),
    }
    if keep_probabilities:
        probabilities = torch.cat(accumulator.probabilities).float().numpy()  # (m, c)
        labels_array = torch.cat(accumulator.labels).numpy().astype(np.int64)  # (m,)
        metrics["roc_auc_macro"], metrics["pr_auc_macro"] = _macro_auc(probabilities, labels_array)
    return metrics
