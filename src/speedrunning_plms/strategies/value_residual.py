"""The value-residual family: every block after the first mixes its values with the first block's,
V = a V_own + (1 - a) V_first, a learned scalar a per block from 0.5 (028; ResFormer, modded-nanogpt record 14).

Symbols: b windows or proteins, l tokens, h heads, d_h per-head width.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ForwardContext, ParameterGroup, Strategy, StrategyConfig


@dataclass(frozen=True)
class ValueResidualConfig(StrategyConfig):
    variant: str = "none"
    initial_share: float = 0.5  # a at the start: the share of a block's own values
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "first_block")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"initial_share": (0.1, 1.0)}


class ValueResidual(Strategy):
    family = "value_residual"
    Config = ValueResidualConfig
    config: ValueResidualConfig

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.variant == "first_block":
            self.shares = nn.ParameterList(nn.Parameter(torch.tensor(self.config.initial_share))
                                           for _ in range(benchmark.BLOCKS))  # each (), block 0's unused
            model.extensions[self.family] = self.shares  # type: ignore[index]

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if self.variant == "none":
            return []
        return [ParameterGroup("value_residual", list(self.shares), "adam", model.strategies.optimizer.config.scalar_lr)]  # type: ignore[union-attr]

    def values(self, index: int, V: Tensor, context: ForwardContext) -> Tensor:
        # V: (b, l, h, d_h); the first block's values are stored for the rest.
        if index == 0:
            context.stored["first_values"] = V
            return V  # (b, l, h, d_h)
        share = self.shares[index]  # ()
        return share * V + (1 - share) * context.stored["first_values"]  # (b, l, h, d_h)
