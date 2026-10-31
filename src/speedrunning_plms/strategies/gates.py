"""The gates family: per-head sigmoid gates on the attention output, before the output projection.

- `residual_channels`: gates read from the first h channels of the attention's normalized input, no new parameters
  (026; modded-nanogpt record 28): a token can mute heads whose context it does not need.
- `head_sigmoid`: the query-dependent head-wise gate of Gated Attention (Qwen, arXiv 2505.06708, position G1): a
  linear map per block from the attention's normalized input to one gate per head, sigmoid, multiplied into each
  head's output. The weight starts at zero and the bias at HEAD_GATE_BIAS, so every gate starts at sigmoid(4) = 0.982
  and step 0 is within 2 percent of the record's. Weight and bias train with AdamW at the scalar rate, as the
  differential kernel's zero-initialized lambda projection does: Muon's orthogonalized step would move all h rows by
  the same spectral amount from step 1 whatever their gradients, which suits a weight matrix, not a readout of h gates.

Symbols: b windows or proteins, l tokens, d = 768, h heads, d_h per-head width.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ParameterGroup, Strategy, StrategyConfig


HEAD_GATE_BIAS = 4.0  # sigmoid(4) = 0.982: the gate starts near open


@dataclass(frozen=True)
class GatesConfig(StrategyConfig):
    variant: str = "none"
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "residual_channels", "head_sigmoid")


class Gates(Strategy):
    family = "gates"
    Config = GatesConfig

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.variant != "head_sigmoid":
            return
        heads = model.strategies.attention.config.heads  # type: ignore[union-attr]
        self.projections = nn.ModuleList(nn.Linear(benchmark.HIDDEN, heads) for _ in range(benchmark.BLOCKS))
        model.extensions[self.family] = self.projections  # type: ignore[index]

    def initialize(self, model: nn.Module) -> None:
        if self.variant != "head_sigmoid":
            return
        for projection in self.projections:
            nn.init.zeros_(projection.weight)  # (h, d)
            nn.init.constant_(projection.bias, HEAD_GATE_BIAS)  # (h,)

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if self.variant != "head_sigmoid":
            return []
        scalar_lr = model.strategies.optimizer.config.scalar_lr  # type: ignore[union-attr]
        return [ParameterGroup("head_gates", list(self.projections.parameters()), "adam", scalar_lr)]

    def attention_output(self, index: int, X: Tensor, Y: Tensor) -> Tensor:
        # X: (b, l, d), the attention's normalized input; Y: (b, h, l, d_h).
        if self.variant == "head_sigmoid":
            gate = torch.sigmoid(self.projections[index](X)).transpose(1, 2).unsqueeze(-1)  # (b, h, l, 1)
        else:
            gate = torch.sigmoid(X[..., : Y.shape[1]]).transpose(1, 2).unsqueeze(-1)  # (b, h, l, 1)
        return Y * gate.to(Y.dtype)  # (b, h, l, d_h)
