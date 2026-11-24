"""The sglr family: Self-Gated Latent Reasoners in the speedrun, small and experimental.

SGLR (github.com/lhallee/Self-Gated-Latent-Reasoners, on MNIST) carries a latent through a pool of heterogeneous
modules, a router choosing the next module at every step from the module before it. Here, after a prelude of `prelude`
blocks, the stream takes `steps` steps, and at each step every token takes exactly one module of a pool of three:

- `attention`: block `prelude`'s attention;
- `mlp`: block `prelude`'s MLP;
- `conv`: a depthwise convolution over each protein's residues, `reach` on each side, its taps from zero (so it starts
  as the identity tap), then ReLU squared and a 768 x 768 projection from zero (Muon). A neighbor in another protein or
  past the window reads as zero (`canon.same_protein_at`).

Every step position ties its attention's and MLP's matrices to block `prelude`'s (`loop.tie_matrices`), so the pool is
stored once; the coda of 36 - prelude - steps blocks follows as the record's. All three modules run on the step's state
(the attention's update is taken back before the MLP reads the stream), and each token adds only its chosen module's
output. The routers, one per previous module plus one for the first step (SGLR's per-source routers, a learned
transition graph), read the normalized step state in float32, AdamW at the scalar rate from N(0, 0.02^2). The forward
takes the argmax, a hard and auditable route; the gradient reaches the routers through the straight-through gate
one_hot + p - sg(p), which, with every module computed, is each module's output against the loss's gradient. Load
balance is Switch's auxiliary loss, 3 sum_m f_m P_m (arXiv 2101.03961), averaged over steps, times `balance`, per masked
residue like every auxiliary loss here (flex path only). Evaluation forwards count each step's routes over real
tokens into `routes` (steps x 3), for the route-trace study.

Two pieces from SGLR's own experiments on chained-digit sums (2026-11-23_chained_structure_search and
2026-11-23_residual_routing, where a router always held near uniform use lost to random routes and a fixed module order
beat both):

- `balance_floor`: when not 0, a step's balance term applies only while the hard usage entropy of its choices is below
  `balance_floor` x ln 3, an anti-collapse floor that otherwise leaves the router free to specialize (SGLR's floor is
  half of ln 24).
- `router`: `state` (the per-source routers above) or `program`, a learned table of logits per step and previous
  module with no read of the state: every token of a step takes the same module, a learned layer order (SGLR's
  transition router at rank 0). The table starts from N(0, 0.02^2), learns through the same straight-through gate, and
  has no balance term, since it routes all tokens of a step one way by design.

The compute is dense: a step costs about one record block plus the convolution, and a token takes one module where a
record block applies two, so at the same 36 positions the model is shallower per token. SGLR's exit action is left
out (the halting pieces are the loop family's, which this family excludes).

Symbols: b windows or proteins, l tokens, d = 768, r = reach, m = 3 modules.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.canon import neighbor_mix, same_protein_at
from speedrunning_plms.strategies.loop import tie_matrices, valid_tokens
from speedrunning_plms.strategies.strategy import CandidateModel, ForwardContext, ParameterGroup, Strategy, StrategyConfig, norm


if TYPE_CHECKING:  # the registry imports this module
    from speedrunning_plms.strategies import Strategies


MODULES = ("attention", "mlp", "conv")
FIRST_STEP = len(MODULES)  # the router of the first step, which follows no module
ROUTER_INIT_STD = 0.02
ROUTERS = ("state", "program")
STATE = "sglr_state"  # the stream where the current step began
PREVIOUS = "sglr_previous"  # (b, l) long, each token's last module, FIRST_STEP before the first step
ATTENTION_UPDATE = "sglr_attention_update"
SAME_PROTEIN = "sglr_same_protein"
BALANCE = "sglr_balance_"  # + step: that step's Switch balance term


@dataclass(frozen=True)
class SglrConfig(StrategyConfig):
    variant: str = "none"
    prelude: int = 6
    steps: int = 24
    reach: int = 2  # the convolution's neighbors on each side
    balance: float = 0.01  # Switch's load-balance weight
    balance_floor: float = 0.0  # 0 balances every step; else only below this share of ln 3 in hard usage entropy
    router: str = "state"
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "pool")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"prelude": (1, 34), "steps": (1, 35), "reach": (1, 4),
                                                        "balance": (0.0, 0.1), "balance_floor": (0.0, 1.0)}
    INTEGERS: ClassVar[tuple[str, ...]] = ("prelude", "steps", "reach")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.router not in ROUTERS:
            raise ValueError(f"router must be one of {ROUTERS}, not {self.router!r}")

    @property
    def balanced(self) -> bool:
        """Whether a balance term joins the loss: the state router's, at a nonzero weight."""
        return self.router == "state" and self.balance > 0

    def is_step(self, position: int) -> bool:
        return self.prelude <= position < self.prelude + self.steps


def routed_update(updates: tuple[Tensor, ...], probabilities: Tensor) -> tuple[Tensor, Tensor]:
    """Each token's chosen module's update, by the straight-through gate: the forward is the argmax module's update
    exactly, and the gradient reaches every module's probability."""
    # updates: (b, l, d) each, m of them; probabilities: (b, l, m); returns (b, l, d) and the choices (b, l) long.
    choices = probabilities.argmax(dim=-1)  # (b, l)
    # p - sg(p) is exactly 0 in the forward; added to the one-hot first, (1 + p) - p would round a gate off 1.
    gates = F.one_hot(choices, len(updates)).to(probabilities.dtype) + (probabilities - probabilities.detach())  # (b, l, m)
    routed = sum(gates[..., module, None].to(update.dtype) * update for module, update in enumerate(updates))  # (b, l, d)
    return routed, choices  # (b, l, d), (b, l)


def usage_entropy(choices: Tensor, modules: int) -> Tensor:
    """The entropy in nats of the share of tokens each module took: 0 when one takes all, ln m at an even split."""
    # choices: (b, l) long; returns ().
    shares = F.one_hot(choices, modules).float().mean(dim=(0, 1))  # (m,)
    return -(shares * shares.clamp_min(1e-12).log()).sum()  # ()


def switch_balance(probabilities: Tensor, choices: Tensor) -> Tensor:
    """Switch Transformer's load-balance term, m sum_i f_i P_i: 1 at a uniform route, m when one module takes all."""
    # probabilities: (b, l, m); choices: (b, l); returns ().
    modules = probabilities.shape[-1]
    shares = F.one_hot(choices, modules).float().mean(dim=(0, 1))  # (m,), f: the tokens each module took
    return modules * (shares * probabilities.float().mean(dim=(0, 1))).sum()  # ()


class Sglr(Strategy):
    family = "sglr"
    Config = SglrConfig
    config: SglrConfig

    def overrides(self, hook: str) -> bool:
        uses = {"updated_stream"} | ({"auxiliary_loss"} if self.config.balanced else set())
        return self.active and hook in uses

    def conflicts(self, strategies: Strategies) -> list[str]:
        if not self.active:
            return []
        blocks, reasons = strategies.shape.config.blocks, []
        if self.config.prelude + self.config.steps > blocks:
            reasons.append(f"prelude {self.config.prelude} + steps {self.config.steps} exceeds {blocks} blocks")
        if strategies.loop.active:
            reasons.append("sglr and loop each own the blocks after the prelude, so one must be off")
        if strategies.stream.variant != "single":
            reasons.append("the steps act on the stream itself, so sglr needs stream.variant 'single'")
        return reasons

    def build(self, model: CandidateModel, max_tokens: int) -> None:
        if not self.active:
            return
        d = model.strategies.shape.config.hidden
        holder = nn.Module()
        if self.config.router == "program":
            holder.program = nn.Parameter(torch.zeros(self.config.steps, FIRST_STEP + 1, len(MODULES)))  # (steps, 4, m); drawn in `initialize`
        else:
            holder.routers = nn.Parameter(torch.zeros(FIRST_STEP + 1, len(MODULES), d))  # (4, m, d): by previous module; drawn in `initialize`
        holder.taps = nn.Parameter(torch.zeros(2 * self.config.reach, d))  # (2 r, d), the convolution's neighbor taps
        holder.conv_out = nn.Linear(d, d, bias=False)  # (d, d), zeroed in `initialize`
        holder.register_buffer("routes", torch.zeros(self.config.steps, len(MODULES), dtype=torch.long), persistent=False)  # (steps, m)
        model.extensions[self.family] = holder
        self.holder = holder

    def initialize(self, model: CandidateModel) -> None:
        if not self.active:
            return
        nn.init.normal_(self._router(), std=ROUTER_INIT_STD)  # (4, m, d) or (steps, 4, m)
        nn.init.zeros_(self.holder.conv_out.weight)  # (d, d): the convolution starts silent, as every block's output does
        pool = model.blocks[self.config.prelude]
        for position in range(self.config.prelude + 1, self.config.prelude + self.config.steps):
            tie_matrices(model.blocks[position], pool)

    def parameter_groups(self, model: CandidateModel) -> list[ParameterGroup]:
        if not self.active:
            return []
        scalar_lr = model.strategies.optimizer.config.scalar_lr
        return [ParameterGroup("sglr_routers_and_taps", [self._router(), self.holder.taps], "adam", scalar_lr),
                ParameterGroup("sglr_conv", [self.holder.conv_out.weight], "muon", model.strategies.optimizer.config.muon_lr)]

    def _router(self) -> nn.Parameter:
        """The routing parameters: the per-source state routers (4, m, d), or the program's table (steps, 4, m)."""
        return self.holder.program if self.config.router == "program" else self.holder.routers

    def updated_stream(self, index: int, sublayer: str, X: Tensor, context: ForwardContext) -> Tensor:
        # X: (b, l, d), the stream after block `index`'s attention or MLP update.
        config = self.config
        if index == config.prelude - 1 and sublayer == "mlp":
            context.stored[STATE] = X  # (b, l, d)
            context.stored[PREVIOUS] = torch.full(X.shape[:2], FIRST_STEP, dtype=torch.long, device=X.device)  # (b, l)
            return X  # (b, l, d)
        if not config.is_step(index):
            return X  # (b, l, d)
        state = context.stored[STATE]  # (b, l, d)
        if sublayer == "attention":
            context.stored[ATTENTION_UPDATE] = X - state  # (b, l, d)
            return state  # (b, l, d): the MLP reads the step's state, not the attention's result
        return self._step(index - config.prelude, state, X - state, context)  # (b, l, d)

    def _step(self, step: int, state: Tensor, mlp_update: Tensor, context: ForwardContext) -> Tensor:
        """One step: score the pool from the state, add each token's chosen module's update."""
        # state, mlp_update: (b, l, d); returns the next step's state (b, l, d).
        H = norm(state)  # (b, l, d)
        if SAME_PROTEIN not in context.stored:
            context.stored[SAME_PROTEIN] = same_protein_at(context.input_ids, self.config.reach)  # (b, l, 2 r, 1)
        mixed = neighbor_mix(H, context.stored[SAME_PROTEIN], self.holder.taps)  # (b, l, d)
        conv_update = self.holder.conv_out(torch.relu(mixed).square())  # (b, l, d)
        previous = context.stored[PREVIOUS]  # (b, l)
        if self.config.router == "program":
            logits = self.holder.program[step][previous]  # (b, l, m), float32: the step's row for each token's last module
        else:
            with torch.autocast(state.device.type, enabled=False):
                every = H.float() @ self.holder.routers.flatten(0, 1).T  # (b, l, 4 m)
            b, l = state.shape[:2]
            logits = every.view(b, l, FIRST_STEP + 1, len(MODULES))  # (b, l, 4, m)
            logits = logits.gather(2, previous[:, :, None, None].expand(b, l, 1, len(MODULES)))[:, :, 0]  # (b, l, m)
        probabilities = logits.softmax(dim=-1)  # (b, l, m)
        update, choices = routed_update((context.stored[ATTENTION_UPDATE], mlp_update, conv_update), probabilities)  # (b, l, d), (b, l)
        if self.holder.training and context.labels is not None and self.config.balanced:
            term = switch_balance(probabilities, choices)  # ()
            if self.config.balance_floor:  # a floor, not a target: only a step whose routes are collapsing is balanced
                floor = self.config.balance_floor * math.log(len(MODULES))
                term = torch.where(usage_entropy(choices, len(MODULES)) < floor, term, torch.zeros_like(term))  # ()
            context.stored[f"{BALANCE}{step}"] = term
        if not self.holder.training:
            real = valid_tokens(context, state)  # (b, l)
            self.holder.routes[step] += F.one_hot(choices[real], len(MODULES)).sum(dim=0)  # (m,)
        context.stored[PREVIOUS] = choices  # (b, l)
        following = state + update  # (b, l, d)
        context.stored[STATE] = following
        return following  # (b, l, d)

    def auxiliary_loss(self, model: CandidateModel, states: list[Tensor], labels: Tensor, context: ForwardContext) -> Tensor:
        """The steps' mean balance term, per masked residue, so its weight reads as the cross entropy's does."""
        # states: (b, l, d) each; labels: (b, l); returns ().
        balance = sum(context.stored[f"{BALANCE}{step}"] for step in range(self.config.steps)) / self.config.steps  # ()
        masked = (labels != benchmark.IGNORE_LABEL).sum()  # ()
        return self.config.balance * balance * masked  # ()
