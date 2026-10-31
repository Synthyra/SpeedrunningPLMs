"""The softcap family: a tanh cap on the head's logits, cap * tanh(z / cap) (003, A2; modded-nanogpt records 9, 18).

Symbols: b windows or proteins, l tokens, v = 33 tokens.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms.strategies.strategy import Strategy, StrategyConfig


@dataclass(frozen=True)
class SoftcapConfig(StrategyConfig):
    variant: str = "tanh"
    cap: float = 30.0
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "tanh")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"cap": (5.0, 100.0)}


class Softcap(Strategy):
    family = "softcap"
    Config = SoftcapConfig
    config: SoftcapConfig

    def logits(self, model: nn.Module, logits: Tensor) -> Tensor:
        # logits: (b, l, v).
        return self.config.cap * torch.tanh(logits / self.config.cap)  # (b, l, v)
