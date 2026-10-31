"""The 3Di probe: how much structure a model's hidden states carry, read the same way for every model.

The head is a softmax-weighted sum of every hidden state (the embedding and each block's output, 37 for the speedrun
shape) followed by one linear layer to the 20 3Di states, trained with the model frozen on the fixed probe subset
(20,000 training proteins, one per family, at most 512 tokens, one pass, uncorrupted input) and scored on every
residue of the full validation and test splits. It is reported beside the MLM metrics and never selects a checkpoint.

A model exposes `hidden_states(input_ids, attention_mask)` returning the list of per-layer states, each (b, l, d).

Symbols: b proteins, l tokens, d hidden width, k hidden states, s = 20 3Di states, m scored residues.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from collections.abc import Callable
from typing import Any, Protocol
from torch import Tensor

from foundry.layers.cross_entropy import cross_entropy
from speedrunning_plms.benchmark import tokens
from speedrunning_plms.benchmark.data import HeldOutSplit, TrainStream
from speedrunning_plms.benchmark.evaluate import _classification_metrics, _macro_auc, batches_by_length


PROBE_MAX_TOKENS = 512
PROBE_BATCH_PROTEINS = 64
PROBE_LEARNING_RATE = 1e-3
PROBE_SEED = 0
NUM_STATES = len(tokens.THREE_DI_STATES)  # s


class HiddenStateModel(Protocol):
    def hidden_states(self, input_ids: Tensor, attention_mask: Tensor) -> list[Tensor]:
        """Every hidden state of the forward, the embedding first; each (b, l, d)."""
        ...


class ThreeDiHead(torch.nn.Module):
    def __init__(self, num_states: int, hidden: int) -> None:
        super().__init__()
        self.layer_logits = torch.nn.Parameter(torch.zeros(num_states))  # (k,)
        self.classifier = torch.nn.Linear(hidden, NUM_STATES)  # d -> s

    def forward(self, hidden_states: list[Tensor]) -> Tensor:
        # hidden_states: k tensors of (b, l, d); returns (b, l, s).
        # Each state is RMS-normalized first, so the probe reads direction, not the scale of a model's residual stream:
        # the unnormalized head diverged on a Muon-trained candidate whose stream is hundreds of times wider than ESM2's.
        weights = self.layer_logits.softmax(dim=0)  # (k,)
        mixed = sum(weight * torch.nn.functional.rms_norm(state.float(), (state.shape[-1],))
                    for weight, state in zip(weights, hidden_states, strict=True))  # (b, l, d)
        return self.classifier(mixed)  # (b, l, s)


def _probe_batches(stream: TrainStream, batch_proteins: int) -> list[tuple[Tensor, Tensor, Tensor]]:
    """The probe subset as padded batches in a seeded order: ids (b, l), attention (b, l), labels (b, l)."""
    generator = np.random.default_rng(PROBE_SEED)
    order = generator.permutation(stream.probe_subset)  # (n_probe,)
    batches: list[tuple[Tensor, Tensor, Tensor]] = []
    for start in range(0, len(order), batch_proteins):
        indices = order[start : start + batch_proteins]  # (b,)
        proteins = [stream.protein(int(index))[:PROBE_MAX_TOKENS] for index in indices]
        states = [stream.three_di[stream.offsets[int(index)] : stream.offsets[int(index) + 1]][:PROBE_MAX_TOKENS] for index in indices]
        width = max(len(protein) for protein in proteins)
        input_ids = torch.full((len(proteins), width), tokens.PAD_ID, dtype=torch.long)  # (b, l)
        labels = torch.full((len(proteins), width), tokens.IGNORE_LABEL, dtype=torch.long)  # (b, l)
        for row, (protein, state_row) in enumerate(zip(proteins, states, strict=True)):
            protein_ids = torch.from_numpy(np.array(protein, dtype=np.int64))  # (length + 2,)
            input_ids[row, : len(protein)] = protein_ids
            state_ids = torch.from_numpy(np.array(state_row, dtype=np.int64))  # (length + 2,)
            labels[row, : len(protein)] = state_ids.masked_fill(state_ids == tokens.NO_STATE, tokens.IGNORE_LABEL)
            if len(protein) == PROBE_MAX_TOKENS and protein[-1] != tokens.EOS_ID:
                labels[row, PROBE_MAX_TOKENS - 1] = tokens.IGNORE_LABEL  # a truncated residue is not scored
        batches.append((input_ids, input_ids != tokens.PAD_ID, labels))
    return batches


def train_head(model: HiddenStateModel, stream: TrainStream, device: torch.device, *, hidden: int, num_hidden_states: int,
               autocast_dtype: torch.dtype | None = torch.bfloat16, progress: Callable[[int], None] | None = None) -> ThreeDiHead:
    """Fit the head on the probe subset, one pass, the model frozen; the model's forward runs under no grad."""
    torch.manual_seed(PROBE_SEED)
    head = ThreeDiHead(num_hidden_states, hidden).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=PROBE_LEARNING_RATE, weight_decay=0.0)
    enabled = autocast_dtype is not None and device.type == "cuda"
    for input_ids, attention_mask, labels in _probe_batches(stream, PROBE_BATCH_PROTEINS):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)  # (b, l) each
        with torch.no_grad(), torch.autocast(device.type, dtype=autocast_dtype or torch.float32, enabled=enabled):
            hidden_states = [state.detach() for state in model.hidden_states(input_ids, attention_mask)]  # k of (b, l, d)
        logits = head(hidden_states)  # (b, l, s)
        count = int((labels != tokens.IGNORE_LABEL).sum())
        loss = cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=tokens.IGNORE_LABEL, reduction="sum") / max(count, 1)  # ()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if progress is not None:
            progress(int(input_ids.shape[0]))
    return head


def score_head(model: HiddenStateModel, head: ThreeDiHead, split: HeldOutSplit, device: torch.device, *,
               tokens_per_batch: int = 32_768, autocast_dtype: torch.dtype | None = torch.bfloat16) -> dict[str, Any]:
    """Every residue of a held-out split, uncorrupted, scored against its true 3Di state."""
    s = NUM_STATES
    confusion = torch.zeros((s, s), dtype=torch.long)  # (s, s)
    nll, count = 0.0, 0
    probabilities: list[Tensor] = []
    labels_kept: list[Tensor] = []
    enabled = autocast_dtype is not None and device.type == "cuda"
    with torch.inference_mode():
        for rows in batches_by_length(split, tokens_per_batch):
            width = min(int(split.lengths[rows[0]]) + 2, split.input_ids.shape[1])
            input_ids = split.input_ids[rows, :width].long().to(device)  # (b, l)
            attention_mask = input_ids != tokens.PAD_ID  # (b, l)
            states = split.three_di[rows, :width].long().to(device)  # (b, l)
            with torch.autocast(device.type, dtype=autocast_dtype or torch.float32, enabled=enabled):
                logits = head(model.hidden_states(input_ids, attention_mask))  # (b, l, s)
            selected = states != tokens.NO_STATE  # (b, l)
            scored = logits[selected].float()  # (m, s)
            targets = states[selected]  # (m,)
            nll += float(cross_entropy(scored, targets, reduction="sum").double())
            count += int(targets.numel())
            confusion += torch.bincount(targets * s + scored.argmax(dim=-1), minlength=s * s).view(s, s).cpu()  # (s, s)
            probabilities.append(scored.softmax(dim=-1).half().cpu())  # (m, s)
            labels_kept.append(targets.to(torch.uint8).cpu())  # (m,)
    metrics: dict[str, Any] = {"loss": nll / count, "bits_per_residue": nll / count / math.log(2), "scored_residues": count,
                               **_classification_metrics(confusion)}
    metrics["roc_auc_macro"], metrics["pr_auc_macro"] = _macro_auc(torch.cat(probabilities).float().numpy(),
                                                                 torch.cat(labels_kept).numpy().astype(np.int64))
    metrics["layer_weights"] = head.layer_logits.softmax(dim=0).tolist()
    return metrics
