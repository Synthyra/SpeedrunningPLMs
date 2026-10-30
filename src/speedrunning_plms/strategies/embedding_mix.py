"""The embedding-mix family: every block's input is a learned mix of the stream and the embedding,
X = a X + c X_0, with a from 1 and c from `initial_share` per block (modded-nanogpt's x0 lambdas, records 4 onward).

The value residual (028) carries the first block's values to depth and was the first record to use context; this
carries the token identity in the stream itself. At c = 0 a run starts as the record.

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
class EmbeddingMixConfig(StrategyConfig):
    variant: str = "none"
    initial_share: float = 0.0  # c at the start: the embedding's weight in every block's input
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "every_block")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"initial_share": (0.0, 0.5)}


class EmbeddingMix(Strategy):
    family = "embedding_mix"
    Config = EmbeddingMixConfig
    config: EmbeddingMixConfig

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.variant == "every_block":
            holder = nn.Module()
            holder.stream = nn.Parameter(torch.ones(benchmark.BLOCKS))  # (36,): a, block 0's unused
            holder.embedding = nn.Parameter(torch.full((benchmark.BLOCKS,), self.config.initial_share))  # (36,): c
            self.holder = holder
            model.extensions[self.family] = holder  # type: ignore[index]

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if self.variant == "none":
            return []
        weights = [self.holder.stream, self.holder.embedding]
        return [ParameterGroup("embedding_mix", weights, "adam", model.strategies.optimizer.config.scalar_lr)]  # type: ignore[union-attr]

    def block_input(self, model: nn.Module, index: int, X: Tensor, states: list[Tensor]) -> Tensor:
        # X: (b, l, d); states[0]: (b, l, d), the embedding.
        if index == 0:
            return X  # (b, l, d): the stream is the embedding
        return self.holder.stream[index] * X + self.holder.embedding[index] * states[0]  # (b, l, d)
