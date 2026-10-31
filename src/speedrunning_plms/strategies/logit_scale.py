"""The logit-scale family: a learned per-head temperature on the attention scores.

- `ssmax`: Scalable-Softmax (arXiv 2501.19399). Each block's query heads carry a learned scalar s, and the queries are
  multiplied by s log(n) after QK-norm and RoPE, so the scores are s log(n) q.k / sqrt(d_h). The record's QK-norm is a
  plain RMS norm with no affine gain (`strategy.norm`), so this adds the only learned temperature rather than a second
  one. n is the training window (`batch.window`), fixed for the run and the same in evaluation: the padded width of an
  evaluation batch depends on its other proteins, and a protein's score must not. With a fixed n the multiplier is
  constant in length, so the variant is a learned per-head temperature in SSMax's parametrization; the paper's
  per-query n (the keys a query sees, here its protein's length) is the alternative. s starts at
  `initial_scale` / log(n), so the scores start at `initial_scale` times the record's (1 is the record's start).

Symbols: b windows or proteins, l tokens, q query heads, d_h per-head width, n the window length.
"""

from __future__ import annotations

import math
import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.attention import DIFFERENTIAL
from speedrunning_plms.strategies.strategy import ForwardContext, ParameterGroup, Strategy, StrategyConfig


@dataclass(frozen=True)
class LogitScaleConfig(StrategyConfig):
    variant: str = "none"
    initial_scale: float = 1.0  # s log(n) at the start: the scores' multiple of the record's
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "ssmax")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"initial_scale": (0.5, 2.0)}


class LogitScale(Strategy):
    family = "logit_scale"
    Config = LogitScaleConfig
    config: LogitScaleConfig

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.variant == "none":
            return
        strategies = model.strategies  # type: ignore[union-attr]
        attention = strategies.attention.config
        query_heads = 2 * attention.heads if attention.variant in DIFFERENTIAL else attention.heads  # q
        self.log_n = math.log(strategies.batch.config.window)
        self.scales = nn.Parameter(torch.full((benchmark.BLOCKS, query_heads), self.config.initial_scale / self.log_n))  # (36, q)
        holder = nn.Module()
        holder.scales = self.scales
        model.extensions[self.family] = holder  # type: ignore[index]

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if self.variant == "none":
            return []
        return [ParameterGroup("ssmax", [self.scales], "adam", model.strategies.optimizer.config.scalar_lr)]  # type: ignore[union-attr]

    def queries(self, index: int, Q: Tensor, context: ForwardContext) -> Tensor:
        # Q: (b, l, q, d_h).
        multiplier = self.scales[index] * self.log_n  # (q,)
        return Q * multiplier.to(Q.dtype)[:, None]  # (b, l, q, d_h)
