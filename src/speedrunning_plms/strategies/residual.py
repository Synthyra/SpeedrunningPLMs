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

Symbols: b windows or proteins, l tokens, d = 768, v = 33 tokens.
"""

from __future__ import annotations

import math
import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ParameterGroup, StepState, Strategy, StrategyConfig, norm


SUBLAYERS = ("attention", "mlp")
NGPT_HOOKS = ("embed", "logits", "after_step")
OUTPUT_INIT_STD = 0.02  # the output projections' start when a norm reads them, the record's std for its other matrices


@dataclass(frozen=True)
class ResidualConfig(StrategyConfig):
    variant: str = "prenorm"
    alpha_init: float = 0.05  # nGPT's eigen learning rate at the start
    branch_multiplier: float = 1.0  # every sublayer's output times this before the update
    branch_ramp: float = 0.0  # share of the clock over which the branches fade in; 0 is off
    branch_start: float = 0.0  # the branches' gain at the start of the ramp
    VARIANTS: ClassVar[tuple[str, ...]] = ("prenorm", "ngpt", "perinorm", "ln_scaling")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"alpha_init": (0.01, 0.5), "branch_multiplier": (0.25, 1.0),
                                                        "branch_ramp": (0.0, 0.3), "branch_start": (0.0, 0.5)}


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
            return self.config.branch_multiplier != 1.0 or ramp
        if hook == "normalized_input":
            return self.variant == "ln_scaling"
        return False

    def build(self, model: nn.Module, max_tokens: int) -> None:
        d = benchmark.HIDDEN
        holder = nn.Module()
        if self.variant == "ngpt":
            holder.alphas = nn.ParameterList(nn.Parameter(torch.full((d,), self.config.alpha_init))
                                             for _ in range(benchmark.BLOCKS * len(SUBLAYERS)))  # each (d,)
            holder.logit_scale = nn.Parameter(torch.full((benchmark.VOCAB_SIZE,), 1 / math.sqrt(d)))  # (v,), read times sqrt(d)
        elif self.variant == "perinorm":
            holder.gains = nn.ParameterList(nn.Parameter(torch.ones(d)) for _ in range(benchmark.BLOCKS * len(SUBLAYERS)))  # each (d,)
        if self.config.branch_ramp > 0:
            # (), the branches' gain, refilled by `after_step` from the clock's progress: a tensor, so no recompile.
            holder.register_buffer("branch_gain", torch.tensor(self.config.branch_start), persistent=False)
        if not list(holder.parameters()) and not list(holder.buffers()):
            return
        model.extensions[self.family] = holder  # type: ignore[index]
        self.holder = holder

    def initialize(self, model: nn.Module) -> None:
        if self.variant not in ("ngpt", "perinorm"):
            return
        for block in model.blocks:  # type: ignore[union-attr]
            nn.init.normal_(block.attention.output.weight, std=OUTPUT_INIT_STD)  # (d, d): a zero matrix has no direction
            nn.init.normal_(block.mlp.down.weight, std=OUTPUT_INIT_STD)  # (d, f)
        if self.variant == "ngpt":
            self.normalize_weights(model)

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        scalar_lr = model.strategies.optimizer.config.scalar_lr  # type: ignore[union-attr]
        if self.variant == "ngpt":
            return [ParameterGroup("ngpt_scales", [*self.holder.alphas, self.holder.logit_scale], "adam", scalar_lr)]
        if self.variant == "perinorm":
            return [ParameterGroup("perinorm_gains", list(self.holder.gains), "adam", scalar_lr)]
        return []

    def sublayer_input(self, X: Tensor) -> Tensor:
        # X: (b, l, d).
        return norm(X)  # (b, l, d)

    def normalized_input(self, index: int, sublayer: str, X: Tensor) -> Tensor:
        # X: (b, l, d), block `index`'s normalized input; LayerNorm Scaling's 1 / sqrt(layer), layer = index + 1.
        return X * (index + 1) ** -0.5  # (b, l, d)

    def sublayer_output(self, index: int, sublayer: str, Y: Tensor) -> Tensor:
        # Y: (b, l, d).
        if self.config.branch_ramp > 0:
            return Y * self.holder.branch_gain.to(Y.dtype)  # (b, l, d)
        return Y * self.config.branch_multiplier  # (b, l, d)

    def update(self, index: int, sublayer: str, X: Tensor, Y: Tensor) -> Tensor:
        # X: (b, l, d), the stream; Y: (b, l, d), the sublayer's output; returns (b, l, d).
        if self.variant in ("prenorm", "ln_scaling"):
            return X + Y  # (b, l, d)
        position = index * len(SUBLAYERS) + SUBLAYERS.index(sublayer)
        if self.variant == "perinorm":
            return X + self.holder.gains[position].to(X.dtype) * norm(Y)  # (b, l, d)
        alpha = self.holder.alphas[position]  # (d,)
        return norm(X + alpha.to(X.dtype) * (norm(Y) - X))  # (b, l, d)

    def embed(self, model: nn.Module, input_ids: Tensor, X: Tensor) -> Tensor:
        # input_ids: (b, l); X: (b, l, d).
        return norm(X)  # (b, l, d)

    def logits(self, model: nn.Module, logits: Tensor) -> Tensor:
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
    def normalize_weights(self, model: nn.Module) -> None:
        """Every vector that lives in the 768-wide stream to unit norm."""
        reads = [model.embedding.weight, model.head.weight]  # type: ignore[union-attr]
        writes: list[Tensor] = []
        for block in model.blocks:  # type: ignore[union-attr]
            reads += [block.attention.query.weight, block.attention.key.weight, block.attention.value.weight,
                      block.mlp.up.weight]
            writes += [block.attention.output.weight, block.mlp.down.weight]
        for W in reads:
            unit_rows(W, dim=1)  # rows of (out, d)
        for W in writes:
            unit_rows(W, dim=0)  # columns of (d, in)
