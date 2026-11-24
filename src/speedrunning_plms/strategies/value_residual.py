"""The value-residual family: every block after the first mixes its values with the first block's,
V = a V_own + (1 - a) V_first, a learned scalar a per block from 0.5 (028; ResFormer, modded-nanogpt record 14).

The record's largest single gain, so round 9 opens the form around it (`ideas/2026-11-03_round9_strategies.md`; no runs yet), each off
at its default:
- `normalized`: IMU-1's form (arXiv 2602.02522, Eq. 4), V = s (a1 V_own + a2 V_first) / sqrt(a1^2 + a2^2), three learned scalars
  per block from (1, `initial_share`, 1 - `initial_share`), so the mix keeps its scale as the scalars move. IMU-1's ablation at 70M is
  -0.013 training loss, against -0.020 for QK-norm and -0.012 for LayerNorm scaling.
- `anchor_norm`: the first block's values RMS-normalized before the mix (NuResFormer and ExoFormer, arXiv 2601.08131,
  S = l1 RMSNorm(S_anchor) + l2 S), since the first block's value scale otherwise sets the anchor's share for all 35 later blocks.
  That paper's 450M perplexity: base 14.79, ResFormer 14.32, NuResFormer 14.15, ExoFormer 14.13.
- `per_head`: the scalars per block and head, not per block, the paper's head granularity.
- `qk_anchor`: the same mix on keys and queries (ExoFormer), a learned scalar per block from this share of the block's own; 1 is off.
  Keys mix before QK-norm and RoPE, and queries after both, where the hook is: RoPE turns both by the same position, so only the
  queries' RMS, slightly under 1 after the mix, differs from mixing before.

Symbols: b windows or proteins, l tokens, h heads, q query heads, d_h per-head width.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms.strategies.strategy import CandidateModel, ForwardContext, ParameterGroup, Strategy, StrategyConfig, norm


OFF_SHARE = 1.0  # `qk_anchor` at this share of the block's own Q and K leaves them alone


@dataclass(frozen=True)
class ValueResidualConfig(StrategyConfig):
    variant: str = "none"
    initial_share: float = 0.5  # a at the start: the share of a block's own values
    anchor_norm: int = 0  # 1 RMS-normalizes the first block's values before they mix
    per_head: int = 0  # 1 learns the shares per block and head
    qk_anchor: float = OFF_SHARE  # the block's own share of Q and K at the start; 1 is off
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "first_block", "normalized")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"initial_share": (0.1, 1.0), "anchor_norm": (0, 1), "per_head": (0, 1),
                                                        "qk_anchor": (0.3, OFF_SHARE)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.anchor_norm not in (0, 1) or self.per_head not in (0, 1):
            raise ValueError(f"anchor_norm and per_head are 0 or 1, not {self.anchor_norm!r}, {self.per_head!r}")


class ValueResidual(Strategy):
    family = "value_residual"
    Config = ValueResidualConfig
    config: ValueResidualConfig

    def overrides(self, hook: str) -> bool:
        """The values as ever; the keys and queries only when `qk_anchor` mixes them."""
        if hook == "values":
            return self.active
        return hook in ("keys", "queries") and self.active and self.config.qk_anchor != OFF_SHARE

    def build(self, model: CandidateModel, max_tokens: int) -> None:
        if self.variant == "none":
            return
        attention = model.strategies.attention.config
        shape = (attention.heads,) if self.config.per_head else ()
        share = self.config.initial_share
        blocks = model.strategies.shape.config.blocks
        self.shares = nn.ParameterList(nn.Parameter(torch.full(shape, share)) for _ in range(blocks))  # each () or (h,), block 0's unused
        model.extensions[self.family] = self.shares
        holder = nn.Module()
        if self.variant == "normalized":
            holder.anchors = nn.ParameterList(nn.Parameter(torch.full(shape, 1 - share)) for _ in range(blocks))  # each () or (h,)
            holder.gains = nn.ParameterList(nn.Parameter(torch.ones(shape)) for _ in range(blocks))  # each () or (h,)
        if self.config.qk_anchor != OFF_SHARE:
            holder.key_shares = nn.Parameter(torch.full((blocks,), self.config.qk_anchor))  # (blocks,), block 0's unused
            holder.query_shares = nn.Parameter(torch.full((blocks,), self.config.qk_anchor))  # (blocks,), block 0's unused
        if list(holder.parameters()):
            model.extensions[f"{self.family}_scalars"] = holder
        self.holder = holder

    def parameter_groups(self, model: CandidateModel) -> list[ParameterGroup]:
        if self.variant == "none":
            return []
        scalars = [*self.shares, *self.holder.parameters()]
        return [ParameterGroup("value_residual", scalars, "adam", model.strategies.optimizer.config.scalar_lr)]

    def _scalar(self, parameter: Tensor, like: Tensor) -> Tensor:
        """A block's scalar, or its per-head row shaped to broadcast over (b, l, h, d_h)."""
        # parameter: () or (h,); like: (b, l, h, d_h), for its dtype; returns () or (h, 1).
        return parameter.unsqueeze(-1).to(like.dtype) if parameter.ndim else parameter  # () or (h, 1)

    def values(self, index: int, V: Tensor, context: ForwardContext) -> Tensor:
        # V: (b, l, h, d_h); the first block's values are stored for the rest.
        if index == 0:
            context.stored["first_values"] = norm(V) if self.config.anchor_norm else V
            return V  # (b, l, h, d_h)
        first = context.stored["first_values"]  # (b, l, h, d_h)
        if self.variant == "normalized":
            own, anchor = self.shares[index], self.holder.anchors[index]  # each () or (h,)
            scale = self.holder.gains[index] / (own.square() + anchor.square()).sqrt().clamp(min=1e-6)  # () or (h,)
            return self._scalar(scale, V) * (self._scalar(own, V) * V + self._scalar(anchor, V) * first)  # (b, l, h, d_h)
        share = self._scalar(self.shares[index], V)  # () or (h, 1)
        return share * V + (1 - share) * first  # (b, l, h, d_h)

    def keys(self, index: int, K: Tensor, context: ForwardContext) -> Tensor:
        # K: (b, l, h, d_h), raw keys; a block that borrows K (`attention.kv_share`) never reaches this.
        if index == 0:
            context.stored["first_keys"] = K
            return K  # (b, l, h, d_h)
        share = self.holder.key_shares[index]  # ()
        return share * K + (1 - share) * context.stored["first_keys"]  # (b, l, h, d_h)

    def queries(self, index: int, Q: Tensor, context: ForwardContext) -> Tensor:
        # Q: (b, l, q, d_h), after QK-norm and RoPE.
        if index == 0:
            context.stored["first_queries"] = Q
            return Q  # (b, l, q, d_h)
        share = self.holder.query_shares[index]  # ()
        return share * Q + (1 - share) * context.stored["first_queries"]  # (b, l, q, d_h)
