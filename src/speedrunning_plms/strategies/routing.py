"""The routing family: the MLP's units in groups, each token's top groups chosen by a router, one router per application.

The record's MLP of 2,048 units is read as `experts` groups of 2,048 / experts units (rows of the up projection, columns
of the down projection), so the shape rule holds and no parameter is added beyond the routers. A router reads the MLP's
normalized input in float32 and picks each token's `top_k` groups by score plus a balancing bias; a chosen group's
activations are multiplied by 2 sigmoid(score), 1 at a zero score, and every other group's by 0 (LOOM, arXiv
2610.01153: per-loop sigmoid routers in float32). Routers are AdamW at the scalar rate from N(0, 0.02^2), a random draw
in `initialize` only when the family is on.

- `per_pass`: one router per block position, so in a tied loop every pass of a core block routes anew: LOOM's routing
  refresh. A router shared across passes picks nearly the same experts on every pass ("expert selection collapse"),
  which cost 1.21 of perplexity in LOOM's ablation (18.62 against 19.83 at 350M and 9 loops). Lee et al. (arXiv
  2605.09165) find looped MoE models scale and dense looped ones do not, for this reason.
- `shared`: one router per stored block, so the passes of a tied core block share one: the collapse control. Without a
  tied loop every block is its own stored block, and the two variants are one model.

The compute stays dense: the masked units are computed and zeroed, so a run tests what routing does to learning, not
what a grouped-GEMM kernel would do to the step time (top 2 of 8 groups would skip three quarters of the MLP's
matmuls). Balancing is LOOM's and DeepSeek-V3's auxiliary-loss-free bias: after each step a router's bias for a group
rises by `bias_rate` where the group took under its share of the step's choices and falls where it took more; the bias
picks groups and never weights them. Training forwards count choices into `load`; evaluation forwards count them by
position into `eval_load`, which a study reads for LOOM's cross-pass routing similarity.

Symbols: b windows or proteins, l tokens, d = 768, f = 2,048, e = experts, k = top_k, u = f / e units a group.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar
from torch import Tensor, nn

from speedrunning_plms.strategies.strategy import CandidateModel, ParameterGroup, StepState, Strategy, StrategyConfig


if TYPE_CHECKING:  # the registry imports this module
    from speedrunning_plms.strategies import Strategies


ROUTER_INIT_STD = 0.02


@dataclass(frozen=True)
class RoutingConfig(StrategyConfig):
    variant: str = "none"
    experts: int = 8
    top_k: int = 2
    bias_rate: float = 2e-3  # LOOM's load-balance bias rate; 0 leaves the choices unbalanced
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "per_pass", "shared")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"experts": (2, 64), "top_k": (1, 63), "bias_rate": (0.0, 0.05)}
    INTEGERS: ClassVar[tuple[str, ...]] = ("experts", "top_k")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.top_k >= self.experts:
            raise ValueError(f"top_k {self.top_k} must be under experts {self.experts}; all groups is the record's MLP")


def chosen_groups(scores: Tensor, bias: Tensor, top_k: int) -> Tensor:
    """Each token's top groups by score plus bias, as a mask."""
    # scores: (b, l, e); bias: (e,); returns (b, l, e) bool.
    chosen = (scores + bias).topk(top_k, dim=-1).indices  # (b, l, k)
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, chosen, True)  # (b, l, e)


def rebalanced(bias: Tensor, load: Tensor, rate: float) -> Tensor:
    """The bias after one step: up for a group under its share of a router's choices, down for one over it; a router
    that made no choice keeps its bias."""
    # bias, load: (r, e), r routers; returns (r, e).
    total = load.sum(dim=-1, keepdim=True)  # (r, 1)
    share = load / total.clamp(min=1)  # (r, e)
    step = rate * torch.sign(1.0 / load.shape[-1] - share)  # (r, e)
    return bias + step * (total > 0)  # (r, e)


class Routing(Strategy):
    family = "routing"
    Config = RoutingConfig
    config: RoutingConfig

    def conflicts(self, strategies: Strategies) -> list[str]:
        intermediate = strategies.shape.config.intermediate
        if self.active and intermediate % self.config.experts:
            return [f"experts {self.config.experts} must divide the MLP's {intermediate} units"]
        return []

    def build(self, model: CandidateModel, max_tokens: int) -> None:
        if not self.active:
            return
        shape, experts = model.strategies.shape.config, self.config.experts
        blocks = shape.blocks
        holder = nn.Module()
        holder.routers = nn.Parameter(torch.zeros(blocks, experts, shape.hidden))  # (blocks, e, d); drawn in `initialize`
        holder.register_buffer("bias", torch.zeros(blocks, experts))  # (blocks, e), part of the model: it picks the groups
        holder.register_buffer("load", torch.zeros(blocks, experts), persistent=False)  # (blocks, e), this step's choices
        holder.register_buffer("eval_load", torch.zeros(blocks, experts), persistent=False)  # (blocks, e), by position
        model.extensions[self.family] = holder
        self.holder = holder

    def initialize(self, model: CandidateModel) -> None:
        if not self.active:
            return
        nn.init.normal_(self.holder.routers, std=ROUTER_INIT_STD)  # (blocks, e, d)
        loop = model.strategies.loop
        shared = self.variant == "shared" and loop.variant == "tied"
        # The router each position uses: its own, or under `shared` the stored block's, so a core block's passes share one.
        self.router_of = [loop.config.source(position) if shared else position for position in range(len(model.blocks))]

    def parameter_groups(self, model: CandidateModel) -> list[ParameterGroup]:
        if not self.active:
            return []
        return [ParameterGroup("routers", [self.holder.routers], "adam", model.strategies.optimizer.config.scalar_lr)]

    def mlp_hidden(self, index: int, H: Tensor, X: Tensor) -> Tensor:
        # H: (b, l, f), the MLP's activation; X: (b, l, d), its normalized input.
        router = self.router_of[index]
        with torch.autocast(X.device.type, enabled=False):
            scores = X.float() @ self.holder.routers[router].T  # (b, l, e)
        mask = chosen_groups(scores, self.holder.bias[router], self.config.top_k)  # (b, l, e)
        gates = torch.where(mask, 2 * torch.sigmoid(scores), torch.zeros_like(scores))  # (b, l, e)
        counts = mask.sum(dim=(0, 1)).float()  # (e,)
        if self.holder.training:
            self.holder.load[router] += counts
        else:
            self.holder.eval_load[index] += counts
        b, l, f = H.shape
        grouped = H.view(b, l, self.config.experts, f // self.config.experts)  # (b, l, e, u)
        return (grouped * gates.to(H.dtype)[..., None]).view(b, l, f)  # (b, l, f)

    def after_step(self, state: StepState) -> None:
        self.holder.bias.copy_(rebalanced(self.holder.bias, self.holder.load, self.config.bias_rate))  # (36, e)
        self.holder.load.zero_()
