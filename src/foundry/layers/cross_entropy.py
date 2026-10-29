"""Cross entropy and log-softmax that keep the target's gradient at any confidence.

PyTorch's `log_softmax` adds the other classes' tiny exponentials to the largest one's `1` and
rounds the sum back to `1` once one logit leads by about 16.6 in float32, 5.5 in bfloat16 or 36.7
in float64. The target's log-probability, loss and gradient then round to exactly zero while the
other classes keep theirs, so the gradient no longer sums to zero: Softmax Collapse. Distillation
on teacher probabilities also leaves a nonzero gradient where the student equals the teacher:
Softmax Cancellation. Both are from Barron, Gnanapragasam and Tschannen,
[A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642) (2026), whose formulas
this module ports from JAX; PyTorch's tracking issues are pytorch/pytorch#113708 and #195678.

`cross_entropy` and `CrossEntropyLoss` take `torch.nn.functional.cross_entropy`'s and
`torch.nn.CrossEntropyLoss`'s arguments and match them wherever the stock version is right.
`binary_cross_entropy_with_logits` and `BCEWithLogitsLoss` do the same for the binary loss, whose
backward rounds `sigmoid(x) - 1` to zero for a confident positive label, the two-class case of the
same collapse. Binary cross entropy on probabilities cannot be repaired inside the loss, since the
sigmoid has already rounded; such a site computes from logits instead. Delete this module, and
point its importers back at PyTorch, once PyTorch ships the fix.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch import nn


LOW_PRECISION = frozenset({torch.float16, torch.bfloat16})


def log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log-softmax along `dim`, with the largest logit left out of the sum it is added to.

    `log(1 + e^x)` for the other classes' log-sum-exp `x` never forms `1 + e^x`, so the largest
    class's log-probability stays `-e^x` instead of rounding to zero. A 16-bit input is computed in
    float32 and returned in its own dtype.
    """
    # logits: (..., c, ...) with the c classes on dim
    working = logits.float() if logits.dtype in LOW_PRECISION else logits  # (..., c, ...)
    max_index = working.argmax(dim=dim, keepdim=True)  # (..., 1, ...)
    # The scattered 0 carries no gradient, so the argmax's +1 and -1 never cancel the small term in backward.
    shifted = (working - working.gather(dim, max_index)).scatter(dim, max_index, 0.0)  # (..., c, ...)
    others = shifted.scatter(dim, max_index, float("-inf"))  # (..., c, ...); the argmax left out
    others_lse = torch.logsumexp(others, dim=dim, keepdim=True)  # (..., 1, ...)
    log_partition = torch.logaddexp(others_lse, torch.zeros_like(others_lse))  # (..., 1, ...); log(1 + e^others_lse)
    return (shifted - log_partition).to(logits.dtype)  # (..., c, ...)


def cross_entropy(
    input: torch.Tensor,  # noqa: A002; torch.nn.functional.cross_entropy's name
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """`torch.nn.functional.cross_entropy` on `log_softmax` above.

    Class-index and class-probability targets, `weight`, `ignore_index`, `reduction` and
    `label_smoothing` mean what they mean in PyTorch; the arguments after `weight` are keyword-only,
    since PyTorch's fourth positional argument is the deprecated `size_average`. A 16-bit `input` is computed and returned in
    float32, as autocast runs PyTorch's. Probability targets fix Softmax Collapse but not Softmax
    Cancellation; distill from teacher logits with `soft_target_cross_entropy` for that.
    """
    # input: (b, c, ...) or (c,); target: (b, ...) or () class indices, or input's shape of probabilities; weight: (c,) or None
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"reduction must be 'none', 'mean' or 'sum', got {reduction!r}")
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError(f"label_smoothing must lie in [0, 1], got {label_smoothing}")
    class_dim = 1 if input.ndim > 1 else 0
    num_classes = input.shape[class_dim]  # c
    log_probabilities = log_softmax(input.float() if input.dtype in LOW_PRECISION else input, class_dim)  # (b, c, ...) or (c,)
    weight = None if weight is None else weight.to(log_probabilities.dtype)  # (c,) or None
    # weight broadcast along the class dim: (1, c, 1, ...) or (c,)
    class_weight = None if weight is None else weight.reshape([num_classes if axis == class_dim else 1 for axis in range(input.ndim)])

    if target.is_floating_point():
        probabilities = target if label_smoothing == 0.0 else target * (1 - label_smoothing) + label_smoothing / num_classes  # (b, c, ...) or (c,)
        weighted = log_probabilities * probabilities if class_weight is None else log_probabilities * probabilities * class_weight  # (b, c, ...) or (c,)
        losses = -weighted.sum(class_dim)  # (b, ...) or ()
        if reduction == "none":
            return losses  # (b, ...) or ()
        if reduction == "sum":
            return losses.sum()  # ()
        return losses.sum() / (input.numel() // num_classes)  # ()

    nll = F.nll_loss(log_probabilities, target, weight, ignore_index=ignore_index, reduction=reduction)  # () or (b, ...)
    if label_smoothing == 0.0:
        return nll  # () or (b, ...)
    ignored = target == ignore_index  # (b, ...) or ()
    smoothed = log_probabilities if class_weight is None else log_probabilities * class_weight  # (b, c, ...) or (c,)
    smooth_losses = (-smoothed.sum(class_dim)).masked_fill(ignored, 0.0)  # (b, ...) or ()
    if reduction == "none":
        smooth = smooth_losses  # (b, ...) or ()
    elif reduction == "sum":
        smooth = smooth_losses.sum()  # ()
    elif weight is None:
        smooth = smooth_losses.sum() / (~ignored).sum()  # ()
    else:
        smooth = smooth_losses.sum() / weight.gather(0, target.masked_select(~ignored).flatten()).sum()  # ()
    return (1 - label_smoothing) * nll + smooth * (label_smoothing / num_classes)  # () or (b, ...)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """`torch.nn.CrossEntropyLoss`, with its arguments and state dict, computed by `cross_entropy` above."""

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: A002
        """Return the loss of `input` logits against `target` classes or probabilities."""
        # input: (b, c, ...) or (c,); target: (b, ...) or () class indices, or input's shape of probabilities
        return cross_entropy(
            input, target, self.weight, ignore_index=self.ignore_index, reduction=self.reduction, label_smoothing=self.label_smoothing,
        )  # () or (b, ...)


def binary_cross_entropy_with_logits(
    input: torch.Tensor,  # noqa: A002; torch.nn.functional.binary_cross_entropy_with_logits's name
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    reduction: str = "mean",
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """`torch.nn.functional.binary_cross_entropy_with_logits` on `logsigmoid` of both signs.

    PyTorch's backward forms `sigmoid(x) - y`, which rounds to zero for a positive label once `x`
    passes 16.6 in float32 (37.5 in float64), and is off by half in bfloat16 at 6. Here the loss is
    `-(pos_weight * y * logsigmoid(x) + (1 - y) * logsigmoid(-x))`, whose gradient terms
    `-y * sigmoid(-x)` and `(1 - y) * sigmoid(x)` are each exact. `weight`, `pos_weight` and
    `reduction` mean what they mean in PyTorch, keyword-only after `weight`. A 16-bit `input` is
    computed and returned in float32, as autocast runs PyTorch's.
    """
    # input, target: (...) the same shape; weight, pos_weight: (...) broadcastable, pos_weight along the last dim, or None
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"reduction must be 'none', 'mean' or 'sum', got {reduction!r}")
    if target.shape != input.shape:
        raise ValueError(f"target shape {tuple(target.shape)} must equal input shape {tuple(input.shape)}")
    working = input.float() if input.dtype in LOW_PRECISION else input  # (...)
    target = target.to(working.dtype)  # (...)
    positive = target * F.logsigmoid(working)  # (...)
    if pos_weight is not None:
        positive = positive * pos_weight.to(working.dtype)  # (...)
    losses = -(positive + (1 - target) * F.logsigmoid(-working))  # (...)
    if weight is not None:
        losses = losses * weight.to(working.dtype)  # (...)
    if reduction == "none":
        return losses  # (...)
    return losses.sum() if reduction == "sum" else losses.mean()  # ()


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """`torch.nn.BCEWithLogitsLoss`, with its arguments and state dict, computed by `binary_cross_entropy_with_logits` above."""

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: A002
        """Return the loss of `input` logits against `target` labels or probabilities."""
        # input, target: (...) the same shape
        return binary_cross_entropy_with_logits(
            input, target, self.weight, reduction=self.reduction, pos_weight=self.pos_weight,
        )  # () or (...)


def soft_target_cross_entropy(student_logits: torch.Tensor, teacher_logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Cross entropy of the student against the teacher's softmax, one loss per row.

    The value is `-sum(p * log q)` for teacher probabilities `p` and student probabilities `q`. The
    student's gradient is `q - p` computed from the logit difference with `expm1` and `log1p`, so it
    is exactly zero where the student's logits equal the teacher's, and sums to zero along `dim`.
    The teacher's logits receive the ordinary gradient; detach them for a fixed teacher.
    """
    # student_logits, teacher_logits: (..., c, ...) with the c classes on dim
    student_log_probabilities = log_softmax(student_logits, dim)  # (..., c, ...)
    student_probabilities = student_log_probabilities.exp()  # (..., c, ...)
    teacher_probabilities = log_softmax(teacher_logits, dim).exp()  # (..., c, ...)
    logit_difference = student_logits - teacher_logits  # (..., c, ...)
    centered = logit_difference - (teacher_probabilities * logit_difference).sum(dim, keepdim=True)  # (..., c, ...); centered under p
    log_ratio = centered - torch.log1p((teacher_probabilities * torch.expm1(centered)).sum(dim, keepdim=True))  # (..., c, ...); log(q / p)
    probability_difference = teacher_probabilities * torch.expm1(log_ratio)  # (..., c, ...); q - p
    gradient = probability_difference - student_probabilities * probability_difference.sum(dim, keepdim=True)  # (..., c, ...); sums to 0
    loss = -(teacher_probabilities * student_log_probabilities.detach()).sum(dim)  # (..., ...)
    return loss + (gradient.detach() * (student_logits - student_logits.detach())).sum(dim)  # (..., ...); the gradient term is 0 forward
