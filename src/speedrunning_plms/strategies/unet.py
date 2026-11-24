"""The U-net family: each second-half block adds a learned scalar times the output of its mirrored first-half
block to its input (005, A4; modded-nanogpt record 11).

005 on the 003 incumbent was not lower (val_select 2.7568 against 2.7559) and cost 12 steps.

Symbols: b windows or proteins, l tokens, d = 768.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar
from torch import Tensor, nn

from speedrunning_plms.strategies.strategy import CandidateModel, ParameterGroup, Strategy, StrategyConfig


if TYPE_CHECKING:  # the registry imports this module
    from speedrunning_plms.strategies import Strategies


@dataclass(frozen=True)
class UNetConfig(StrategyConfig):
    variant: str = "none"
    initial_weight: float = 1.0
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "mirrored")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"initial_weight": (0.0, 1.0)}


class UNet(Strategy):
    family = "unet"
    Config = UNetConfig
    config: UNetConfig

    def conflicts(self, strategies: Strategies) -> list[str]:
        blocks = strategies.shape.config.blocks
        return [f"a mirror needs an even depth, not {blocks} blocks"] if self.active and blocks % 2 else []

    def build(self, model: CandidateModel, max_tokens: int) -> None:
        self.half = model.strategies.shape.config.blocks // 2
        if self.variant == "mirrored":
            self.weights = nn.Parameter(torch.full((self.half,), self.config.initial_weight))  # (blocks / 2,)
            holder = nn.Module()
            holder.weights = self.weights
            model.extensions[self.family] = holder

    def parameter_groups(self, model: CandidateModel) -> list[ParameterGroup]:
        if self.variant == "none":
            return []
        return [ParameterGroup("unet", [self.weights], "adam", model.strategies.optimizer.config.embed_lr)]

    def block_input(self, model: CandidateModel, index: int, X: Tensor, states: list[Tensor]) -> Tensor:
        # X: (b, l, d); states: (b, l, d) each, the stream after j blocks at j.
        half = self.half
        if index < half:
            return X  # (b, l, d)
        return X + self.weights[index - half] * states[half - (index - half)]  # (b, l, d)
