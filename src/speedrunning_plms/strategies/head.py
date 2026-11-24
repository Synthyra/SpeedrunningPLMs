"""The head family: what reads the final state before the untied 33-way projection, and what it starts from.

- `linear` (the record): the projection reads the RMS-normalized state directly.
- `two_layer`: ESM-2's masked-LM head (`RobertaLMHead` in `facebookresearch/esm`): a dense 768 to 768 map with bias,
  GELU and a LayerNorm with gain and bias, then the untied 33-way projection, all on the RMS-normalized state. The
  dense layer starts normal with std 0.02 (the record's head init), its bias and the LayerNorm at their defaults; all
  train with AdamW at the head's rate. About 0.6 million parameters and 0.3 percent of a step's FLOPs. Every reader
  of the head shares it: the loss, the evaluator, and readouts of intermediate states.

`bias` (round 9, no run yet; 0 is off): a learned bias on the 33 logits, the 20 residue columns from the log of the training stream's
shares, so the first step already predicts the unigram, 2.89 nats, instead of spending steps on it: the projection has no bias of its own and
has to find those shares through the state. AdamW at the head's rate, on either variant. The usual final-layer trick, with no measured gain here;
at 2,000 steps it can only be small, and it costs one add.

Symbols: b windows or proteins, l tokens, d = 768, v = 33 tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.residue_priors import unigram
from speedrunning_plms.strategies.strategy import CandidateModel, ParameterGroup, Strategy, StrategyConfig


DENSE_INIT_STD = 0.02  # the record's std for the head


@dataclass(frozen=True)
class HeadConfig(StrategyConfig):
    variant: str = "linear"
    bias: int = 0  # 1 adds a learned logit bias from the training stream's residue shares
    VARIANTS: ClassVar[tuple[str, ...]] = ("linear", "two_layer")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"bias": (0, 1)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bias not in (0, 1):
            raise ValueError(f"bias is 0 or 1, not {self.bias!r}")


class Head(Strategy):
    family = "head"
    Config = HeadConfig
    config: HeadConfig

    @property
    def active(self) -> bool:
        return self.variant == "two_layer"

    def overrides(self, hook: str) -> bool:
        return (hook == "head_input" and self.active) or (hook == "logits" and self.config.bias == 1)

    def build(self, model: CandidateModel, max_tokens: int) -> None:
        holder = nn.Module()
        if self.variant == "two_layer":
            d = model.strategies.shape.config.hidden
            holder.dense = nn.Linear(d, d)
            holder.layer_norm = nn.LayerNorm(d)
        if self.config.bias:
            start = torch.zeros(benchmark.VOCAB_SIZE)  # (v,)
            start[benchmark.ESM2_RESIDUE_COLUMNS] = unigram().log()  # (v,)
            holder.bias = nn.Parameter(start)  # (v,)
        if not list(holder.parameters()):
            return
        model.extensions[self.family] = holder
        self.holder = holder

    def initialize(self, model: CandidateModel) -> None:
        if self.variant == "two_layer":
            nn.init.normal_(self.holder.dense.weight, std=DENSE_INIT_STD)  # (d, d)
            nn.init.zeros_(self.holder.dense.bias)  # (d,)

    def parameter_groups(self, model: CandidateModel) -> list[ParameterGroup]:
        if self.variant != "two_layer" and not self.config.bias:
            return []
        head_lr = model.strategies.optimizer.config.head_lr
        return [ParameterGroup("mlm_head", list(self.holder.parameters()), "adam", head_lr)]

    def head_input(self, model: CandidateModel, X: Tensor) -> Tensor:
        # X: (b, l, d), the normalized final state.
        return self.holder.layer_norm(F.gelu(self.holder.dense(X)))  # (b, l, d)

    def logits(self, model: CandidateModel, logits: Tensor) -> Tensor:
        # logits: (b, l, v).
        return logits + self.holder.bias.to(logits.dtype)  # (b, l, v)
