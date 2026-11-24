"""The residual family: how a block's sublayers update the stream.

- `prenorm` (the record): X <- X + f(norm(X)).
- `ngpt`: the stream lives on the unit-RMS sphere (nGPT, C1; arXiv 2410.01131). X <- norm(X + alpha (norm(f(norm(X)))
  - X)) with a learned per-channel eigen rate alpha per sublayer (from 0.05, the paper's Section 2.6); the embedding
  output is normalized; every matrix's vectors that live in the stream (the rows of the query, key, value, MLP up,
  embedding and head, the columns of the attention output and MLP down) are renormalized after each step and at
  initialization (the output projections start random instead of zero); the logits are cos(h, e) times a learned
  per-token scale s_z, stored at 1/sqrt(d) and read times sqrt(d) as in the paper's Section 2.5. QK-norm with the
  1/sqrt(d_h) softmax scale already equals nGPT's normalized attention without s_qk. The pre-sublayer RMS norm stays,
  so modules that push the stream off the sphere (U-net, hyperconnections) compose. Departures from the paper: ReLU
  squared instead of SwiGLU, no s_qk, s_u or s_nu, and Muon, not Adam, on the matrices.
- `perinorm`: Peri-LN (arXiv 2502.02732; Gemma 2, OLMo 2). X <- X + g norm(f(norm(X))), a learned per-channel gain g
  per sublayer from 1. An RMS norm of the record's zero-initialized output projections is undefined, so the attention
  output and MLP down projections start normal with std 0.02, the record's scale for its other matrices.
- `ln_scaling`: LayerNorm Scaling (arXiv 2502.05795). Block i's (0-based) normalized input is multiplied by
  (i + 1)^-1/2 before the attention and the MLP. QK-norm undoes the scale on the queries and keys, so the attention
  output scales by (i + 1)^-1/2 through the values (and against the first block's values the value residual mixes in
  unscaled); the ReLU-squared MLP is 2-homogeneous, so its output scales by 1 / (i + 1).

`branch_multiplier` (Depth-muP, arXiv 2310.02244; CompleteP, arXiv 2505.01618) multiplies every sublayer's output by a
constant before the update. Under Muon and Adam it acts as a cut of the branches' learning rate, so search it with
`muon_lr`. `ngpt` and `perinorm` normalize the output after it, which undoes it; it acts under `prenorm` and
`ln_scaling`. `branch_ramp` is the noise records' identity-path phase as a fade-in (`backprop_free.md` 2.3): the
branches' gain rises linearly from `branch_start` to `branch_multiplier` over the first `branch_ramp` share of the
clock, so the stream starts near the identity path from the embedding to the head and the blocks join it. The gain
is a 0-d buffer `after_step` refills from the clock's progress, so the compiled graph never sees a changing float. At
a start of 0 the zero-initialized output projections get no gradient on the first step and a small one after.

`learned_gain` (ReZero, arXiv 2003.04887; LayerScale, arXiv 2103.17239; `2026-10-30_init_and_parametrization.md` 3):
one learned scalar per block and sublayer, 72 in all, multiplies the sublayer's output before the update, from
`learned_gain` (never 0: the output projections already start at zero, and a zero gate on them would never start),
AdamW at the scalar rate. It composes with the multiplier and the ramp (it multiplies after them), and like them it is
undone by `ngpt` and `perinorm`, which normalize the output after it. Under Muon the output matrices move a fixed size
a step, so the gate learns each branch's overall scale beside them.

`momentum` (TMMFormer and YuriiFormer, arXiv 2605.24425; round 9, no run yet): a velocity stream beside the residual stream, one per
sublayer step. The sublayer reads the lookahead X + mu V; then V <- beta V + gamma Y, normalized when `velocity_norm` is 1, and X <- X + nu V.
The four scalars per sublayer are learned, AdamW at the scalar rate; mu and beta pass a sigmoid and start at `momentum_lookahead` and
`momentum_beta`, gamma and nu a softplus and start at 1, as in the paper, whose starts for mu and beta were not read. The velocity starts from a second
token embedding (its only large parameter there, 25,344 here), AdamW at the embedding rate. Compute-matched, 12 blocks of 768 under Muon, 30k
steps: OpenWebText 3.0078 to 2.9342, TinyStories 1.1569 to 1.1284, about 40 percent of that gap explained by the paper's extra embeddings.
Under pure AdamW the OpenWebText gap falls to 0.041. Its LN_v is a LayerNorm; RMS norm here, the stream's own. With `velocity_norm` the output projections
start normal, as `perinorm`'s do. With it at 0 the zero-initialized velocity table and output projections make step 0 the record's forward,
which the source does not test.

Symbols: b windows or proteins, l tokens, d = 768, v = 33 tokens.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import CandidateModel, ForwardContext, ParameterGroup, StepState, Strategy, StrategyConfig, norm


SUBLAYERS = ("attention", "mlp")
NGPT_HOOKS = ("embed", "logits", "after_step")
OUTPUT_INIT_STD = 0.02  # the output projections' start when a norm reads them, the record's std for its other matrices
VELOCITY = "velocity"  # the momentum stream's entry in the forward's stored tensors
LOOKAHEAD, DECAY, GAIN, REINJECTION = 0, 1, 2, 3  # the columns of the momentum scalars


@dataclass(frozen=True)
class ResidualConfig(StrategyConfig):
    variant: str = "prenorm"
    alpha_init: float = 0.05  # nGPT's eigen learning rate at the start
    branch_multiplier: float = 1.0  # every sublayer's output times this before the update
    branch_ramp: float = 0.0  # share of the clock over which the branches fade in; 0 is off
    branch_start: float = 0.0  # the branches' gain at the start of the ramp
    learned_gain: float = 0.0  # the start of a learned gain per block and sublayer; 0 is off
    momentum_beta: float = 0.5  # momentum: the velocity's decay at the start
    momentum_lookahead: float = 0.5  # momentum: how far ahead each sublayer reads at the start
    velocity_norm: int = 1  # momentum: 1 normalizes the velocity as the source does
    VARIANTS: ClassVar[tuple[str, ...]] = ("prenorm", "ngpt", "perinorm", "ln_scaling", "momentum")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"alpha_init": (0.01, 0.5), "branch_multiplier": (0.25, 1.0),
                                                        "branch_ramp": (0.0, 0.3), "branch_start": (0.0, 0.5),
                                                        "learned_gain": (0.0, 1.0), "momentum_beta": (0.02, 0.98),
                                                        "momentum_lookahead": (0.02, 0.98), "velocity_norm": (0, 1)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.learned_gain != 0.0 and self.learned_gain < 0.01:
            raise ValueError(f"learned_gain is 0 (off) or 0.01 to 1: a zero gate on a zero output projection never starts, "
                             f"not {self.learned_gain}")
        if self.velocity_norm not in (0, 1):
            raise ValueError(f"velocity_norm is 0 or 1, not {self.velocity_norm!r}")


def logit(share: float) -> float:
    """The sigmoid's inverse."""
    return math.log(share / (1 - share))


def unit_rows(W: Tensor, dim: int) -> None:
    """Scale every vector of W along `dim` to unit L2 norm, in place."""
    # W: (m, n).
    W.div_(W.norm(dim=dim, keepdim=True).clamp(min=1e-8))  # (m, n)


class Residual(Strategy):
    family = "residual"
    Config = ResidualConfig
    config: ResidualConfig

    def overrides(self, hook: str) -> bool:
        """The pre-norm residual is the skeleton's own arithmetic; each variant and the multiplier claim their hooks."""
        ramp = self.config.branch_ramp > 0
        if hook == "after_step":
            return self.variant == "ngpt" or ramp
        if hook in NGPT_HOOKS:
            return self.variant == "ngpt"
        if hook == "sublayer_output":
            return self.config.branch_multiplier != 1.0 or ramp or self.config.learned_gain > 0
        if hook == "normalized_input":
            return self.variant == "ln_scaling"
        return False

    def build(self, model: CandidateModel, max_tokens: int) -> None:
        d, blocks = model.strategies.shape.config.hidden, model.strategies.shape.config.blocks
        holder = nn.Module()
        if self.variant == "ngpt":
            holder.alphas = nn.ParameterList(nn.Parameter(torch.full((d,), self.config.alpha_init))
                                             for _ in range(blocks * len(SUBLAYERS)))  # each (d,)
            holder.logit_scale = nn.Parameter(torch.full((benchmark.VOCAB_SIZE,), 1 / math.sqrt(d)))  # (v,), read times sqrt(d)
        elif self.variant == "perinorm":
            holder.gains = nn.ParameterList(nn.Parameter(torch.ones(d)) for _ in range(blocks * len(SUBLAYERS)))  # each (d,)
        elif self.variant == "momentum":
            starts = torch.tensor([logit(self.config.momentum_lookahead), logit(self.config.momentum_beta),
                                   math.log(math.e - 1), math.log(math.e - 1)])  # (4,), the raw values at the configured starts
            holder.momentum_scalars = nn.Parameter(starts.repeat(blocks * len(SUBLAYERS), 1))  # (2 blocks, 4)
            holder.velocity_embedding = nn.Embedding(benchmark.VOCAB_SIZE, d, _weight=torch.zeros(benchmark.VOCAB_SIZE, d))  # (v, d)
        if self.config.branch_ramp > 0:
            # (), the branches' gain, refilled by `after_step` from the clock's progress: a tensor, so no recompile.
            holder.register_buffer("branch_gain", torch.tensor(self.config.branch_start), persistent=False)
        if self.config.learned_gain > 0:
            holder.learned_gains = nn.Parameter(torch.full((blocks, len(SUBLAYERS)), self.config.learned_gain))  # (blocks, 2)
        if not list(holder.parameters()) and not list(holder.buffers()):
            return
        model.extensions[self.family] = holder
        self.holder = holder

    @property
    def normalizes_output(self) -> bool:
        """True when a norm reads the output projections' result, so they cannot start at zero."""
        return self.variant in ("ngpt", "perinorm") or (self.variant == "momentum" and self.config.velocity_norm == 1)

    def initialize(self, model: CandidateModel) -> None:
        if self.variant == "momentum" and self.config.velocity_norm:
            nn.init.normal_(self.holder.velocity_embedding.weight, std=OUTPUT_INIT_STD)  # (v, d): a zero velocity has no direction
        if not self.normalizes_output:
            return
        for block in model.blocks:
            nn.init.normal_(block.attention.output.weight, std=OUTPUT_INIT_STD)  # (d, d): a zero matrix has no direction
            nn.init.normal_(block.mlp.down.weight, std=OUTPUT_INIT_STD)  # (d, f)
        if self.variant == "ngpt":
            self.normalize_weights(model)

    def parameter_groups(self, model: CandidateModel) -> list[ParameterGroup]:
        config = model.strategies.optimizer.config
        scalar_lr = config.scalar_lr
        groups = []
        if self.variant == "ngpt":
            groups.append(ParameterGroup("ngpt_scales", [*self.holder.alphas, self.holder.logit_scale], "adam", scalar_lr))
        if self.variant == "perinorm":
            groups.append(ParameterGroup("perinorm_gains", list(self.holder.gains), "adam", scalar_lr))
        if self.variant == "momentum":
            groups.append(ParameterGroup("momentum_scalars", [self.holder.momentum_scalars], "adam", scalar_lr))
            groups.append(ParameterGroup("velocity_embedding", [self.holder.velocity_embedding.weight], "adam", config.embed_lr))
        if self.config.learned_gain > 0:
            groups.append(ParameterGroup("branch_gains", [self.holder.learned_gains], "adam", scalar_lr))
        return groups

    def _velocity(self, context: ForwardContext) -> Tensor:
        """The velocity stream so far: the second embedding before the first sublayer's update."""
        # Returns (b, l, d).
        if VELOCITY not in context.stored:
            context.stored[VELOCITY] = self.holder.velocity_embedding(context.input_ids)  # (b, l, d)
        return context.stored[VELOCITY]  # (b, l, d)

    def sublayer_input(self, index: int, sublayer: str, X: Tensor, context: ForwardContext) -> Tensor:
        # X: (b, l, d), the stream.
        if self.variant == "momentum":
            position = index * len(SUBLAYERS) + SUBLAYERS.index(sublayer)
            X = X + torch.sigmoid(self.holder.momentum_scalars[position, LOOKAHEAD]).to(X.dtype) * self._velocity(context)  # (b, l, d)
        return norm(X)  # (b, l, d)

    def normalized_input(self, index: int, sublayer: str, X: Tensor) -> Tensor:
        # X: (b, l, d), block `index`'s normalized input; LayerNorm Scaling's 1 / sqrt(layer), layer = index + 1.
        return X * (index + 1) ** -0.5  # (b, l, d)

    def sublayer_output(self, index: int, sublayer: str, Y: Tensor) -> Tensor:
        # Y: (b, l, d).
        if self.config.branch_ramp > 0:
            Y = Y * self.holder.branch_gain.to(Y.dtype)  # (b, l, d)
        elif self.config.branch_multiplier != 1.0:
            Y = Y * self.config.branch_multiplier  # (b, l, d)
        if self.config.learned_gain > 0:
            Y = Y * self.holder.learned_gains[index, SUBLAYERS.index(sublayer)].to(Y.dtype)  # (b, l, d)
        return Y  # (b, l, d)

    def update(self, index: int, sublayer: str, X: Tensor, Y: Tensor, context: ForwardContext) -> Tensor:
        # X: (b, l, d), the stream; Y: (b, l, d), the sublayer's output; returns (b, l, d).
        if self.variant in ("prenorm", "ln_scaling"):
            return X + Y  # (b, l, d)
        position = index * len(SUBLAYERS) + SUBLAYERS.index(sublayer)
        if self.variant == "perinorm":
            return X + self.holder.gains[position].to(X.dtype) * norm(Y)  # (b, l, d)
        if self.variant == "momentum":
            scalars = self.holder.momentum_scalars[position]  # (4,)
            decay = torch.sigmoid(scalars[DECAY]).to(X.dtype)  # ()
            gain, reinjection = F.softplus(scalars[GAIN]).to(X.dtype), F.softplus(scalars[REINJECTION]).to(X.dtype)  # (), ()
            velocity = decay * self._velocity(context) + gain * Y  # (b, l, d)
            if self.config.velocity_norm:
                velocity = norm(velocity)  # (b, l, d)
            context.stored[VELOCITY] = velocity  # (b, l, d)
            return X + reinjection * velocity  # (b, l, d)
        alpha = self.holder.alphas[position]  # (d,)
        return norm(X + alpha.to(X.dtype) * (norm(Y) - X))  # (b, l, d)

    def embed(self, model: CandidateModel, input_ids: Tensor, X: Tensor) -> Tensor:
        # input_ids: (b, l); X: (b, l, d).
        return norm(X)  # (b, l, d)

    def logits(self, model: CandidateModel, logits: Tensor) -> Tensor:
        # logits: (b, l, v) = sqrt(d) cos(h, e) with unit-norm head rows and a unit-RMS state.
        return logits * self.holder.logit_scale.to(logits.dtype)  # (b, l, v), s_z cos(h, e)

    def after_step(self, state: StepState) -> None:
        if self.config.branch_ramp > 0:
            share = min(1.0, state.progress / self.config.branch_ramp)  # of the ramp spent
            start, end = self.config.branch_start, self.config.branch_multiplier
            self.holder.branch_gain.fill_(start + (end - start) * share)
        if self.variant == "ngpt":
            self.normalize_weights(state.model)

    @torch.no_grad()
    def normalize_weights(self, model: CandidateModel) -> None:
        """Every vector that lives in the 768-wide stream to unit norm."""
        reads = [model.embedding.weight, model.head.weight]
        writes: list[Tensor] = []
        for block in model.blocks:
            reads += [linear.weight for linear in (block.attention.query, block.attention.key, block.attention.value,
                                                   block.mlp.up) if linear is not None]  # no K or V under `kv_share`
            writes += [block.attention.output.weight, block.mlp.down.weight]
        for W in reads:
            unit_rows(W, dim=1)  # rows of (out, d)
        for W in writes:
            unit_rows(W, dim=0)  # columns of (d, in)
