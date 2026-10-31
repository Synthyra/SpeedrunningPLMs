"""The regularization family: noise on the training forward that the evaluated model never sees.

`early_dropout` (Liu et al., "Dropout Reduces Underfitting," arXiv 2303.01500; `2026-10-30_regularization_and_averaging.md`
idea 6): dropout on each sublayer's output before the residual add, at `rate` at the start of the clock, falling
linearly to 0 at `until`. The rate is a 0-d buffer the step callback refills, so one compiled graph serves every
phase: the mask is drawn in the graph as rand >= rate and the kept outputs scale by 1 / (1 - rate). At rate 0 every
draw in [0, 1) passes and the scale is 1, so after `until` the output is the sublayer's own, value for value, with
no branch and no recompile; the draw itself still costs its kernels for the rest of the run. Evaluation (the model in
eval mode) never drops. The only MLM evidence (Drop Dropout, arXiv 2505.24788) found no dropout better in one pass.

Symbols: b windows, l tokens, d = 768.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms.strategies.strategy import StepState, Strategy, StrategyConfig


@dataclass(frozen=True)
class RegularizationConfig(StrategyConfig):
    variant: str = "none"
    rate: float = 0.1  # early_dropout: the dropout rate at the start of the clock
    until: float = 0.15  # early_dropout: the share of the clock where the rate reaches 0
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "early_dropout")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.05, 0.2), "until": (0.05, 0.3)}

    def rate_at(self, progress: float) -> float:
        """The dropout rate at this share of the clock."""
        return self.rate * max(0.0, 1 - progress / self.until)


class Regularization(Strategy):
    family = "regularization"
    Config = RegularizationConfig
    config: RegularizationConfig

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.active:
            self.holder = nn.Module()
            self.holder.register_buffer("rate", torch.zeros(()), persistent=False)  # (), refilled each step
            model.extensions[self.family] = self.holder  # type: ignore[index]

    def initialize(self, model: nn.Module) -> None:
        if self.active:
            self.holder.rate.fill_(self.config.rate_at(0.0))

    def sublayer_output(self, index: int, sublayer: str, Y: Tensor) -> Tensor:
        # Y: (b, l, d), the attention's or the MLP's output before the residual update.
        if not self.holder.training:
            return Y  # (b, l, d)
        rate = self.holder.rate  # ()
        keep = torch.rand_like(Y, dtype=torch.float32) >= rate  # (b, l, d)
        return Y * keep.to(Y.dtype) / (1 - rate).to(Y.dtype)  # (b, l, d)

    def after_step(self, state: StepState) -> None:
        self.holder.rate.fill_(self.config.rate_at(state.progress))
