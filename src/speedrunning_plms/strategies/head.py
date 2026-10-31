"""The head family: what reads the final state before the untied 33-way projection.

- `linear` (the record): the projection reads the RMS-normalized state directly.
- `two_layer`: ESM-2's masked-LM head (`RobertaLMHead` in `facebookresearch/esm`): a dense 768 to 768 map with bias,
  GELU and a LayerNorm with gain and bias, then the untied 33-way projection, all on the RMS-normalized state. The
  dense layer starts normal with std 0.02 (the record's head init), its bias and the LayerNorm at their defaults; all
  train with AdamW at the head's rate. About 0.6 million parameters and 0.3 percent of a step's FLOPs. Every reader
  of the head shares it: the loss, the evaluator, and readouts of intermediate states.

Symbols: b windows or proteins, l tokens, d = 768.
"""

from __future__ import annotations

import torch.nn.functional as F

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ParameterGroup, Strategy, StrategyConfig


DENSE_INIT_STD = 0.02  # the record's std for the head


@dataclass(frozen=True)
class HeadConfig(StrategyConfig):
    variant: str = "linear"
    VARIANTS: ClassVar[tuple[str, ...]] = ("linear", "two_layer")


class Head(Strategy):
    family = "head"
    Config = HeadConfig
    config: HeadConfig

    @property
    def active(self) -> bool:
        return self.variant == "two_layer"

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.variant != "two_layer":
            return
        holder = nn.Module()
        holder.dense = nn.Linear(benchmark.HIDDEN, benchmark.HIDDEN)
        holder.layer_norm = nn.LayerNorm(benchmark.HIDDEN)
        model.extensions[self.family] = holder  # type: ignore[index]
        self.holder = holder

    def initialize(self, model: nn.Module) -> None:
        if self.variant == "two_layer":
            nn.init.normal_(self.holder.dense.weight, std=DENSE_INIT_STD)  # (d, d)
            nn.init.zeros_(self.holder.dense.bias)  # (d,)

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if self.variant != "two_layer":
            return []
        head_lr = model.strategies.optimizer.config.head_lr  # type: ignore[union-attr]
        return [ParameterGroup("mlm_head", list(self.holder.parameters()), "adam", head_lr)]

    def head_input(self, model: nn.Module, X: Tensor) -> Tensor:
        # X: (b, l, d), the normalized final state.
        return self.holder.layer_norm(F.gelu(self.holder.dense(X)))  # (b, l, d)
