"""Training losses for the sequence and token heads of residue-embedding probes.

Protify's probes and InterpNet compute their loss with these. `SoftBCEWithLogitsLoss` and
`SoftBCELoss` skip targets equal to `ignore_index`, the value an unobserved label carries. A
`sigmoid_regression` head trains on its pre-sigmoid logits with `SoftBCEWithLogitsLoss`, since
binary cross entropy on probabilities loses a confident target's gradient in the sigmoid.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch import nn

from .cross_entropy import BCEWithLogitsLoss, CrossEntropyLoss, binary_cross_entropy_with_logits


def get_loss_fct(task_type: str, tokenwise: bool = False) -> nn.Module:
    """Return the training loss for a task type."""
    if task_type == "singlelabel":
        return CrossEntropyLoss()

    if task_type == "multilabel":
        return BCEWithLogitsLoss()

    if tokenwise and task_type != "regression":
        return CrossEntropyLoss()

    if task_type == "regression" and not tokenwise:
        return nn.MSELoss()

    if task_type == "sigmoid_regression":
        return SoftBCEWithLogitsLoss()

    raise ValueError(f"Unsupported task type: {task_type}")


class SoftBCEWithLogitsLoss(nn.Module):
    """Binary cross entropy on logits, with optional label smoothing and an ignored target value."""

    __constants__ = [
        "weight",
        "pos_weight",
        "reduction",
        "ignore_index",
        "smooth_factor",
    ]

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: float | None = -100.0,
        reduction: str = "mean",
        smooth_factor: float | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        # weight, pos_weight: (...) broadcastable to the targets, or None.
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Return the loss over the targets that do not equal `ignore_index`."""
        # y_pred, y_true: (...), the same shape; n_kept counts the targets not ignored
        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)  # (...)
        else:
            soft_targets = y_true  # (...)

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index  # (...)
            if not torch.any(not_ignored_mask):
                return y_pred[not_ignored_mask].sum()  # (); differentiable zero

            y_pred = y_pred[not_ignored_mask]  # (n_kept,)
            soft_targets = soft_targets[not_ignored_mask]  # (n_kept,)
            weight = torch.broadcast_to(self.weight, y_true.shape)[not_ignored_mask] if self.weight is not None else None  # (n_kept,) or None
            pos_weight = (
                torch.broadcast_to(self.pos_weight, y_true.shape)[not_ignored_mask] if self.pos_weight is not None else None
            )  # (n_kept,) or None
            loss = binary_cross_entropy_with_logits(
                y_pred,
                soft_targets,
                weight,
                pos_weight=pos_weight,
                reduction="none",
            )  # (n_kept,)
        else:
            loss = binary_cross_entropy_with_logits(
                y_pred,
                soft_targets,
                self.weight,
                pos_weight=self.pos_weight,
                reduction="none",
            )  # (...)

        if self.reduction == "mean":
            loss = loss.mean()  # ()

        if self.reduction == "sum":
            loss = loss.sum()  # ()

        return loss  # () when reduced, else (n_kept,) or (...)


class SoftBCELoss(nn.Module):
    """Binary cross entropy on probabilities, with optional label smoothing and an ignored target value."""

    __constants__ = [
        "weight",
        "pos_weight",
        "reduction",
        "ignore_index",
        "smooth_factor",
    ]

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: float | None = -100.0,
        reduction: str = "mean",
        smooth_factor: float | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        # weight, pos_weight: (...) broadcastable to the targets, or None.
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Return the loss over the targets that do not equal `ignore_index`."""
        # y_pred, y_true: (...), the same shape; n_kept counts the targets not ignored
        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)  # (...)
        else:
            soft_targets = y_true  # (...)

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index  # (...)
            if not torch.any(not_ignored_mask):
                return y_pred[not_ignored_mask].sum()  # (); differentiable zero

            y_pred = y_pred[not_ignored_mask]  # (n_kept,)
            soft_targets = soft_targets[not_ignored_mask]  # (n_kept,)
            weight = torch.broadcast_to(self.weight, y_true.shape)[not_ignored_mask] if self.weight is not None else None  # (n_kept,) or None

            pos_weight = (
                torch.broadcast_to(self.pos_weight, y_true.shape)[not_ignored_mask] if self.pos_weight is not None else None
            )  # (n_kept,) or None
        else:
            weight = self.weight  # broadcastable to (...)
            pos_weight = self.pos_weight  # broadcastable to (...)

        loss = F.binary_cross_entropy(y_pred, soft_targets, reduction="none")  # (n_kept,) or (...)
        if pos_weight is not None:
            # BCE clamps its logarithms at the probability boundaries; preserve that behavior.
            positive_loss = F.binary_cross_entropy(
                y_pred, torch.ones_like(y_pred), reduction="none",
            )  # (n_kept,) or (...)
            loss = loss + (pos_weight - 1) * soft_targets * positive_loss  # (n_kept,) or (...)
        if weight is not None:
            loss = loss * weight  # (n_kept,) or (...)

        if self.reduction == "mean":
            loss = loss.mean()  # ()

        if self.reduction == "sum":
            loss = loss.sum()  # ()

        return loss  # () when reduced, else (n_kept,) or (...)
