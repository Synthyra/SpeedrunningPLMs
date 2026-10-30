"""The U-net family: each second-half block adds a learned scalar times the output of its mirrored first-half
block to its input (005, A4; modded-nanogpt record 11).

005 on the 003 incumbent was not lower (val_select 2.7568 against 2.7559) and cost 12 steps.

Symbols: b windows or proteins, l tokens, d = 768.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ParameterGroup, Strategy, StrategyConfig


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

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.variant == "mirrored":
            self.weights = nn.Parameter(torch.full((benchmark.BLOCKS // 2,), self.config.initial_weight))  # (18,)
            holder = nn.Module()
            holder.weights = self.weights
            model.extensions[self.family] = holder  # type: ignore[index]

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if self.variant == "none":
            return []
        return [ParameterGroup("unet", [self.weights], "adam", model.strategies.optimizer.config.embed_lr)]  # type: ignore[union-attr]

    def block_input(self, model: nn.Module, index: int, X: Tensor, states: list[Tensor]) -> Tensor:
        # X: (b, l, d); states: (b, l, d) each, the stream after j blocks at j.
        half = benchmark.BLOCKS // 2
        if index < half:
            return X  # (b, l, d)
        return X + self.weights[index - half] * states[half - (index - half)]  # (b, l, d)
