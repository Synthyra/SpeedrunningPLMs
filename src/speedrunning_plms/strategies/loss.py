"""The loss family: the summed objective a training step differentiates, from the logits and the masked labels.

`plain` (the record): the summed collapse-free masked cross entropy over the 20 residue columns, the evaluator's
problem. `entropy_weighted` (MiLe Loss, arXiv 2310.19531, with exponent 1; `ideas/2026-10-30_selective_losses.md`,
idea 1): each masked residue's cross entropy weighted by its prediction's entropy under the 20-way softmax, detached,
with the weights normalized to mean 1 over the step's masked residues so the step size holds. The weight reads the
prediction and not the label, so the minimizer of the expected loss at a context is unchanged.

`z_weight` (PaLM's z-loss, arXiv 2204.02311; 0 is off) adds z_weight x (log Z)^2 summed over the masked residues, with
Z the sum of exp over the 20 residue logits: it pins the logit level and leaves every probability as it is.

Both act in `training_loss`, outside the compiled model and on both training paths; the logged loss and the evaluator
stay the record's plain cross entropy. The plain variant with no z-loss adds no hook.

Symbols: b windows, l tokens, v = 33 tokens, n masked residues in the step, c = 20 residues.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor

from foundry.layers.cross_entropy import log_softmax
from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import Strategy, StrategyConfig


@dataclass(frozen=True)
class LossConfig(StrategyConfig):
    variant: str = "plain"
    z_weight: float = 0.0  # the z-loss coefficient; 0 is off
    VARIANTS: ClassVar[tuple[str, ...]] = ("plain", "entropy_weighted")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"z_weight": (0.0, 1e-2)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.z_weight != 0.0 and self.z_weight < 1e-4:
            raise ValueError(f"z_weight is 0 (off) or 1e-4 to 1e-2, not {self.z_weight}")


def residue_scores(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """The masked residues' 20 residue logits in fp32 and their targets."""
    # logits: (b, l, v); labels: (b, l) with IGNORE_LABEL where unscored; returns (n, c), (n,).
    scored = labels != benchmark.IGNORE_LABEL  # (b, l)
    columns = benchmark.ESM2_RESIDUE_COLUMNS.to(logits.device)  # (c,)
    return logits[scored][:, columns].float(), labels[scored]  # (n, c), (n,)


def entropy_weighted_cross_entropy(Z: Tensor, targets: Tensor) -> Tensor:
    """The summed cross entropy with each residue weighted by its detached predictive entropy, mean weight 1."""
    # Z: (n, c) fp32; targets: (n,); returns ().
    log_p = log_softmax(Z, dim=-1)  # (n, c)
    cross_entropy = -log_p.gather(1, targets[:, None]).squeeze(1)  # (n,)
    entropy = -(log_p.exp() * log_p).sum(dim=-1).detach()  # (n,)
    weights = entropy * targets.numel() / entropy.sum().clamp(min=1e-12)  # (n,), mean 1
    return (weights * cross_entropy).sum()  # ()


def z_loss(Z: Tensor) -> Tensor:
    """The summed squared log normalizer of the 20-way softmax."""
    # Z: (n, c) fp32; returns ().
    return torch.logsumexp(Z, dim=-1).square().sum()  # ()


class Loss(Strategy):
    family = "loss"
    Config = LossConfig
    config: LossConfig

    def overrides(self, hook: str) -> bool:
        return hook == "training_loss" and (self.variant != "plain" or self.config.z_weight > 0)

    def training_loss(self, logits: Tensor, labels: Tensor, loss: Tensor) -> Tensor:
        # logits: (b, l, v); labels: (b, l); loss: (), the record's summed masked cross entropy; returns ().
        Z, targets = residue_scores(logits, labels)  # (n, c), (n,)
        objective = entropy_weighted_cross_entropy(Z, targets) if self.variant == "entropy_weighted" else loss  # ()
        if self.config.z_weight:
            objective = objective + self.config.z_weight * z_loss(Z)  # ()
        return objective  # ()
