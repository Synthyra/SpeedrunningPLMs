"""The activation family: the MLP's nonlinearity between the up and down projections, which the skeleton's MLP calls.

- `relu2` (the record): ReLU squared.
- `xielu`: xIELU (arXiv 2411.13010; the reference implementation in Hugging Face's `XIELUActivation`):
  alpha_p x^2 + beta x for x > 0 and alpha_n (exp(x) - 1 - x) + beta x for x <= 0, with beta = 0.5 fixed and per-block
  trainable alpha_p = softplus(a_p) and alpha_n = beta + softplus(a_n), both starting at 0.8. Unlike ReLU squared it
  responds to negative inputs, so `backprop_free.unit_select`'s activity ranking means something else under it.

The `mlp_hidden` hook sees the activation's output; this family replaces the activation itself.

Symbols: b windows or proteins, l tokens, f = 2,048.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ParameterGroup, Strategy, StrategyConfig


XIELU_BETA = 0.5
XIELU_ALPHA_INIT = 0.8  # both alpha_p and alpha_n at the start, the paper's and the reference implementation's


def inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


@dataclass(frozen=True)
class ActivationConfig(StrategyConfig):
    variant: str = "relu2"
    VARIANTS: ClassVar[tuple[str, ...]] = ("relu2", "xielu")


class Activation(Strategy):
    family = "activation"
    Config = ActivationConfig
    config: ActivationConfig

    @property
    def active(self) -> bool:
        return False  # the MLP calls `apply` directly, never as a hook

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.variant != "xielu":
            return
        holder = nn.Module()
        holder.raw_positive = nn.Parameter(torch.full((benchmark.BLOCKS,), inverse_softplus(XIELU_ALPHA_INIT)))  # (36,)
        holder.raw_negative = nn.Parameter(torch.full((benchmark.BLOCKS,), inverse_softplus(XIELU_ALPHA_INIT - XIELU_BETA)))  # (36,)
        model.extensions[self.family] = holder  # type: ignore[index]
        self.holder = holder

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if self.variant != "xielu":
            return []
        parameters = [self.holder.raw_positive, self.holder.raw_negative]
        return [ParameterGroup("xielu", parameters, "adam", model.strategies.optimizer.config.scalar_lr)]  # type: ignore[union-attr]

    def apply(self, index: int, U: Tensor) -> Tensor:
        # U: (b, l, f), block `index`'s up projection.
        if self.variant == "relu2":
            return F.relu(U).square()  # (b, l, f)
        alpha_positive = F.softplus(self.holder.raw_positive[index]).to(U.dtype)  # ()
        alpha_negative = (XIELU_BETA + F.softplus(self.holder.raw_negative[index])).to(U.dtype)  # ()
        negative = U.clamp(max=0.0)  # (b, l, f); keeps expm1 finite on the branch `where` discards
        positive_branch = alpha_positive * U.square() + XIELU_BETA * U  # (b, l, f)
        negative_branch = alpha_negative * (torch.expm1(negative) - negative) + XIELU_BETA * U  # (b, l, f)
        return torch.where(U > 0, positive_branch, negative_branch)  # (b, l, f)
