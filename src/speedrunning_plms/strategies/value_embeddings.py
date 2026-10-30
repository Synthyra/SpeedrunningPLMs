"""The value-embeddings family: zero-initialized token tables added to the values of the first and last few blocks
(011, A5; modded-nanogpt records 14 and 15).

011 on the 003 incumbent was behind at every eval from 90 s (val_select 2.7594 against 2.7559).

Symbols: b windows or proteins, l tokens, d = 768, h heads, d_h per-head width, v = 33 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ForwardContext, ParameterGroup, Strategy, StrategyConfig


@dataclass(frozen=True)
class ValueEmbeddingsConfig(StrategyConfig):
    variant: str = "none"
    blocks_each_end: int = 3  # tables on blocks 0 .. k - 1 and 36 - k .. 35
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "ends")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"blocks_each_end": (1, 9)}


class ValueEmbeddings(Strategy):
    family = "value_embeddings"
    Config = ValueEmbeddingsConfig
    config: ValueEmbeddingsConfig

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.variant == "none":
            return
        k = int(self.config.blocks_each_end)
        tabled = tuple(range(k)) + tuple(range(benchmark.BLOCKS - k, benchmark.BLOCKS))
        self.position = {block: index for index, block in enumerate(tabled)}
        self.tables = nn.ModuleList(nn.Embedding(benchmark.VOCAB_SIZE, benchmark.HIDDEN) for _ in tabled)  # each (v, d)
        model.extensions[self.family] = self.tables  # type: ignore[index]

    def initialize(self, model: nn.Module) -> None:
        if self.variant == "ends":
            for table in self.tables:
                nn.init.zeros_(table.weight)  # (v, d)

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if self.variant == "none":
            return []
        tables = [table.weight for table in self.tables]
        return [ParameterGroup("value_embeddings", tables, "adam", model.strategies.optimizer.config.embed_lr)]  # type: ignore[union-attr]

    def values(self, index: int, V: Tensor, context: ForwardContext) -> Tensor:
        # V: (b, l, h, d_h).
        if index not in self.position:
            return V  # (b, l, h, d_h)
        return V + self.tables[self.position[index]](context.input_ids).view(V.shape)  # (b, l, h, d_h)
