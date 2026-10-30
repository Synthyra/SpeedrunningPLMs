"""The neighbor family: a zero-initialized table over (left neighbor, right neighbor) token pairs added to the
embedding (010, A7 for masked prediction: attention is bidirectional, so a bigram needs both sides).

010 on the 003 incumbent was level with it to 270 s and behind after (val_select 2.7583 against 2.7559).

Symbols: b windows or proteins, l tokens, d = 768, v = 33 tokens.
"""

from __future__ import annotations

import torch.nn.functional as F

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ParameterGroup, Strategy, StrategyConfig


@dataclass(frozen=True)
class NeighborConfig(StrategyConfig):
    variant: str = "none"
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "pair_table")


class Neighbor(Strategy):
    family = "neighbor"
    Config = NeighborConfig

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.variant == "pair_table":
            self.table = nn.Embedding(benchmark.VOCAB_SIZE * benchmark.VOCAB_SIZE, benchmark.HIDDEN)  # (v * v, d)
            model.extensions[self.family] = self.table  # type: ignore[index]

    def initialize(self, model: nn.Module) -> None:
        if self.variant == "pair_table":
            nn.init.zeros_(self.table.weight)  # (v * v, d): the table starts silent

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if self.variant == "none":
            return []
        return [ParameterGroup("neighbor_table", [self.table.weight], "adam", model.strategies.optimizer.config.embed_lr)]  # type: ignore[union-attr]

    def embed(self, model: nn.Module, input_ids: Tensor, X: Tensor) -> Tensor:
        # input_ids: (b, l); X: (b, l, d).
        left = F.pad(input_ids[:, :-1], (1, 0), value=benchmark.PAD_ID)  # (b, l), the token before each position
        right = F.pad(input_ids[:, 1:], (0, 1), value=benchmark.PAD_ID)  # (b, l), the token after
        return X + self.table(left * benchmark.VOCAB_SIZE + right)  # (b, l, d)
