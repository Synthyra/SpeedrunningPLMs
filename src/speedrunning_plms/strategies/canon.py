"""The canon family: a learned, per-channel mix of a token's neighbors in the stream (round 9, no run yet).

`neighbors` (Canon layers, arXiv 2512.17351, Canon-A and Canon-C, made bidirectional): the stream right before a sublayer's norm becomes
X_t + sum_k (w_-k . X_{t-k} + w_+k . X_{t+k}) for k up to `reach`, with w per block and site, one vector of d each, from zero, so step 0 is
the record's; AdamW at the scalar rate. A neighbor in another protein, or past the window, reads as zero. Attention sees every position, so
this is not new reach: it is a cheap local path that does not have to be learned through the rotation pairs, which is why the source finds that
it lifts weaker positional schemes to RoPE's level (1.3B pretraining), and gains 10 to 15 percent knowledge capacity and 2 to 4 times
reasoning depth on synthetic tasks. It gives no perplexity for RoPE plus Canon on language, so a tie with this RoPE-and-slopes record is a
live outcome. Its forward time at 1.3B, naive: about 5.8 percent for both sites.

`placement`: `ac` both sites, `a` before each attention but the first, `c` before each MLP. The sites use `updated_stream`, so Canon-A here
reads the stream before the norm rather than after it, the same mix up to that scale, and the first attention has no site (there is
nothing to mix but the embedding). `every`: only each `every`-th block's sites. The neighbors stay in bf16 on the GPU, since the backward
keeps one copy for each weight: in float32, at both sites of every block, they ran 94 GB out (`round9_pilot_02`).

Symbols: b windows or proteins, l tokens, d = 768, r = `reach`.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms.strategies.attention import documents
from speedrunning_plms.strategies.strategy import CandidateModel, ForwardContext, ParameterGroup, Strategy, StrategyConfig


SUBLAYER_SITE = {"attention": 0, "mlp": 1}  # the site after each sublayer's update, so before the MLP and before the next attention
PLACEMENTS = {"ac": ("attention", "mlp"), "a": ("mlp",), "c": ("attention",)}  # which update a site follows
SAME_PROTEIN = "canon_same_protein"  # the forward's stored mask


@dataclass(frozen=True)
class CanonConfig(StrategyConfig):
    variant: str = "none"
    reach: int = 1  # neighbors on each side
    placement: str = "ac"
    every: int = 1  # only each every-th block's sites
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "neighbors")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"reach": (1, 2), "every": (1, 6)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reach not in (1, 2):
            raise ValueError(f"reach is 1 or 2, not {self.reach!r}")
        if self.placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {tuple(PLACEMENTS)}, not {self.placement!r}")


def same_protein_at(input_ids: Tensor, reach: int) -> Tensor:
    """For each token and each distance on each side, whether that neighbor is in the token's protein and the window."""
    # input_ids: (b, l); returns (b, l, 2 r, 1) bool: distances 1 to r on the left, then on the right.
    document = documents(input_ids)  # (b, l)
    masks = []
    for distance in range(1, reach + 1):
        same = (document[:, distance:] == document[:, :-distance]) & (document[:, distance:] >= 0)  # (b, l - k)
        blank = torch.zeros_like(document[:, :distance], dtype=torch.bool)  # (b, k)
        masks += [torch.cat((blank, same), dim=1), torch.cat((same, blank), dim=1)]  # (b, l) each: the left, then the right neighbor
    return torch.stack(masks, dim=2)[..., None]  # (b, l, 2 r, 1)


def neighbor_mix(X: Tensor, same: Tensor, weights: Tensor) -> Tensor:
    """The stream plus its weighted neighbors inside each protein."""
    # X: (b, l, d); same: (b, l, 2 r, 1) bool; weights: (2 r, d), the left then right neighbor at each distance; returns (b, l, d).
    neighbors = X.to(torch.bfloat16) if X.device.type == "cuda" else X  # (b, l, d)
    mixed = X  # (b, l, d)
    for position in range(weights.shape[0]):
        distance = position // 2 + 1
        pad = torch.zeros_like(neighbors[:, :distance])  # (b, k, d)
        shifted = torch.cat((pad, neighbors[:, :-distance]), dim=1) if position % 2 == 0 else torch.cat((neighbors[:, distance:], pad), dim=1)  # (b, l, d)
        mixed = mixed + weights[position].to(neighbors.dtype) * shifted * same[:, :, position]  # (b, l, d)
    return mixed  # (b, l, d)


class Canon(Strategy):
    family = "canon"
    Config = CanonConfig
    config: CanonConfig

    def build(self, model: CandidateModel, max_tokens: int) -> None:
        if self.variant == "none":
            return
        holder = nn.Module()
        shape = model.strategies.shape.config
        holder.weights = nn.Parameter(torch.zeros(shape.blocks, len(SUBLAYER_SITE), 2 * self.config.reach, shape.hidden))  # (blocks, 2, 2 r, d)
        model.extensions[self.family] = holder
        self.holder = holder

    def parameter_groups(self, model: CandidateModel) -> list[ParameterGroup]:
        if self.variant == "none":
            return []
        return [ParameterGroup("canon", [self.holder.weights], "adam", model.strategies.optimizer.config.scalar_lr)]

    def updated_stream(self, index: int, sublayer: str, X: Tensor, context: ForwardContext) -> Tensor:
        # X: (b, l, d), the stream after the update, before the next sublayer's norm.
        if sublayer not in PLACEMENTS[self.config.placement] or index % self.config.every:
            return X  # (b, l, d)
        if SAME_PROTEIN not in context.stored:
            context.stored[SAME_PROTEIN] = same_protein_at(context.input_ids, self.config.reach)  # (b, l, 2 r, 1)
        return neighbor_mix(X, context.stored[SAME_PROTEIN], self.holder.weights[index, SUBLAYER_SITE[sublayer]])  # (b, l, d)
