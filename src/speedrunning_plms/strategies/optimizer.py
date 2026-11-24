"""The optimizer family: Muon on the block matrices, AdamW on the embedding, the head and every strategy's tables
and scalars.

- `normuon` (the record, 015): Muon with Polar Express, banked by parameter shape (001), with NorMuon's per-row
  second-moment normalization of the orthogonalized update and each matrix's norm restored (B1).
- `muon`: the same without the per-row normalization (001 to 012).
- `muon_plus`: Muon+ (arXiv 2602.21545) in place of NorMuon's step: the orthogonalized update of each matrix is
  normalized along `plus_axis` by this step's own norms, `col_row` (the paper's choice in its Tables 5 and 9: columns
  to unit norm, then rows), `row_col`, `row` or `col`, and each matrix's norm is restored as NorMuon's is, so
  `muon_lr` keeps its meaning. NorMuon divides each row by a running mean (beta 0.95) of its squared update across
  steps, an adaptive second moment with state; Muon+ divides by the current update's norms, stateless, and its
  `col_row` form also equalizes the columns.
- `anvil`: ANVIL's rails and lane equalizer (modded-nanogpt record 92a, `track_1_short/optim/anvil.py` on master, read
  2026-10-30) around this file's Polar Express. Two fp32 momentum rails, fast (beta 0.85) and slow
  (`anvil_slow_beta`, 0.98), blend as B = w v_fast + (1 - w) v_slow (`anvil_fast_weight`, 0.4385), and the lookahead
  g = G + momentum (B - G) is orthogonalized. Each lane (a row of a tall matrix, a column of a wide one: the
  reduction runs along the longer dimension) is divided by the square root of an EMA (beta ANVIL_LANE_BETA) of its
  mean squared update, and each matrix's norm is restored. Departures: the record engages the fast rail at step 514
  by a schedule its file does not show, here both rails run from step 0; its six-map quintic cascade, cautious decay
  gated by the slow rail and uint16 mantissa buffer are left out (Polar Express stays, the record has no decay).
- `amuse`: AMUSE (arXiv 2605.22432), Muon with Schedule-Free averaging, on the Muon matrices. Each matrix keeps a base
  iterate Z and an average X in fp32; the weights hold Y = (1 - beta_t) Z + beta_t X, where the gradient is taken.
  M <- mu M + (1 - mu) G with the schedule's momentum, Z <- Z - lr Polar(M) (the record's step scale), X <- X + (Z - X)
  / (t + 1), a uniform average of the Z iterates, then Y is written back. beta_t = `amuse_beta` for the first
  AMUSE_WARMUP_STEPS steps, then 1 - (T0 / t)^`amuse_rho` (1 - `amuse_beta`). Validation scores X through
  `evaluation_weights`, so the selected weights are X. No cooldown: `schedule_free` makes the loop ask the schedule
  for its multiplier without the cooldown, so the warmup runs and the plateau lasts to the end. AdamW's groups keep
  their own rule, and the Muon matrices take no weight decay of either kind (the record has none). The paper's LLM runs searched beta in {0.4, 0.6} and rho in {0.6, 0.8} (Figure 5 at 0.6 and 0.8);
  T0 is not stated, so it is the record's warmup, 2 percent of about 2,000 steps.
`cautious_weight_decay` decays a Muon weight only where the update and the weight share a sign (016, A3; modded-
nanogpt records 43 and 50); `weight_decay` is the plain decoupled decay on both optimizers (0 in the record).

Three numbers act on the schedule of the update rather than its form, all off at their defaults:
- `adam_beta2_end`: AdamW's beta2 moves linearly from `adam_beta2` to it over the schedule's cooldown (Dremov et al.,
  arXiv 2508.01483: higher beta2 in the cooldown helps). The betas are plain floats set on the param groups each step
  before the optimizers step, outside any compiled graph, as the learning rate is.
- `decay_timescale` (Power Lines, arXiv 2505.13738): weight decay on the Muon banks with a constant timescale of
  `decay_timescale` total runs, lambda_t = 1 / (timescale x total_steps x lr_t), so each step shrinks the weights by
  1 / (timescale x total_steps) whatever the learning rate. The run's total steps are estimated each step as
  (step + 1) / progress. When on, it replaces `cautious_weight_decay` on the banks. The unit is a 10-minute run: the
  total is divided by `StepState.run_scale` (the budget over 600 s), so one value is one per-step shrink on every track
  (at 3,600 s the run is about six times the steps, and "in total runs" would make each value six times weaker per
  step; the 1-hour scout file's idea 5). At the 10-minute track nothing changes.
- `depth_taper` (`2026-10-30_short_run_schedules.md` 7; Wu et al., arXiv 2602.07494, and "Optimal Scaling Needs Optimal
  Norm," arXiv 2510.03871, abstracts: hidden layers prefer lower rates, the rate falls with effective depth): Muon's
  rate at block i is multiplied by 1 + taper (1/2 - i / 35), a linear grade with mean 1 (taper 0.5: 1.25 at block 0,
  0.75 at block 35; negative reverses it), a (k, 1, 1) tensor per bank applied to the update, so no graph changes.
  A strategy's Muon matrices outside the blocks keep 1.
- `adamuon` (AdaMuon, arXiv 2507.11005, abstract): NorMuon's step with one second moment per entry of the
  orthogonalized update instead of per row (beta `normuon_beta`), each matrix's norm restored as NorMuon's is, so
  `muon_lr` keeps its meaning; the paper's own 0.2 sqrt(mn) RMS rescale is not used for that reason. The state grows
  from (k, m, 1) to (k, m, n): about 1 GB of fp32 for the record's block matrices.
- `fused_polar`: Polar Express under `torch.compile(dynamic=False)`, one graph per bank shape, so its elementwise
  parts fuse around the bf16 matmuls; the same iteration and coefficients, so the update matches up to the compiler's
  reordering. CUDA only: on the CPU the loop never compiles (the model's rule), and Windows CPU inductor needs MSVC.

Round 9 (`ideas/2026-11-03_round9_strategies.md`, no runs yet; all eager, so no compile cost). Each is off at its default:
- `muown` (Muown, arXiv 2605.10797, Algorithm 1): a variant. W = Diag(g / r) R, each row's magnitude g trained by Adam and its
  direction R by Muon on the gradient projected off the row, r the rows' norms. The Adam rate is `muown_ratio` times Muon's, whose
  default ties the two as the paper's 0.2 sqrt(max(m, n)) does at 768 x 768. A row that starts at zero (the output projections)
  has g = 0 and a random direction R, so W starts as the record's zero; its direction moves once g does. It takes no weight decay
  (the paper's default), holds R, so about 0.8 GB more fp32, and ignores the numbers below except `orthogonalizer` and `polar_steps`.
- `orthogonalizer`: `polar_express` (the record), or `mud1`, `mud2` (MUD, arXiv 2603.17970, Algorithm 2). p passes of Q <- tril(Q
  Q^T)^-1 Q with unit rows before and after: a Gauss-Seidel sweep in place of the polar factor, about 12 times fewer FLOPs. The paper's
  one protein number is ESM-2 150M on masked language modeling, where MUD reaches Muon's perplexity in about 64 to 71 percent of the
  time (Muon 16.840 in 997 s, MUD1 16.855 in 636 s, MUD2 16.795 in 710 s). Banked Muon is 26 ms of the record's 276 ms step, so the
  saving here is at most about 7 percent more steps. The triangular solve runs in fp32.
- `polar_steps`: the first n of Polar Express's five steps; fewer leaves small singular values under one and saves 2 ms a step per
  step dropped (HTMuon, arXiv 2603.10067, and Soft-Muon find small directions overweighted; these prefixes are not the papers' maps).
- `contra_start`, `contra_end` (Contra-Muon, modded-nanogpt PR 275; arXiv-free post, nilin.github.io/contra-muon-and-soft-muon):
  update = (1 + c) U - c M / s1, c moving linearly over the clock, M the Nesterov direction and s1 its top singular value by a
  warm-started power iteration. c > 0 is contra (it overweights small singular directions), c < 0 soft (it underweights them).
  The post's contra-to-soft schedule took track 3 from a record to 3,030 steps; its coefficients are not published.
- `radial_out`, `radial_in` (radial brake, modded-nanogpt PR 294: outward 0.5, inward 1, 2,990 steps over 11 seeds): per matrix, the
  update's radial part is scaled by its side's number and the matrix is rescaled to the norm that leaves, which also takes the
  tangential step's second-order growth away. Both at 0 hold every norm fixed, Hyperball's rule (arXiv 2606.16899: 20 to 30 percent
  token-equivalent speedup at 1.2B and at least 20 tokens per parameter, a tie on track 3); an outward scale of 0 is AdamP's. A
  matrix under RADIAL_REFERENCE_STD's norm steps plainly, so the zero-initialized output projections can grow.
- `variance_scale` (Muon-VS, arXiv 2601.14603, Eq. 13): the Nesterov direction is divided elementwise by the root of a bias-corrected
  EMA of (M - G)^2 before the orthogonalization; one elementwise buffer. 720M 2.7858 to 2.7798, 130M 3.2179 to 3.1982.
- `mona_beta` (MONA, arXiv 2605.26842, Algorithm 1 at its alpha = -1 / (2 (1 - beta))): the gradient joins momentum as G + alpha A,
  A the EMA of gradient differences, which at that alpha is (G + S) / 2 with S the EMA of earlier gradients, so one buffer holds it.
  Numbers only at 1B to 68B, on downstream benchmarks. It is a slow rail, as ANVIL's was, which lost its pilot by 0.014.
- `normuon_beta`, `nesterov`: two constants of the record's update, opened. `nesterov` 0 orthogonalizes M itself instead of the lookahead.
- `qk_lr_scale`, `vo_lr_scale`, `mlp_lr_scale`: Muon's rate per role, on query and key, value and output, MLP up and down
  (Prime Intellect's track 3 stack, primeintellect.ai/auto-nanogpt, reports per-role rates; no number per change), as `depth_taper`'s tensor.

Symbols: k matrices in a bank, m and n a matrix's dimensions, r = min(m, n), s = max(m, n).
"""

from __future__ import annotations

import contextlib
import functools
import torch

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar
from torch import Tensor

from speedrunning_plms.optim.muon import POLAR_EXPRESS_COEFFICIENTS
from speedrunning_plms.strategies.strategy import CandidateModel, ParameterGroup, StepState, Strategy, StrategyConfig


PLUS_AXES = ("col_row", "row_col", "row", "col")
ANVIL_FAST_BETA = 0.85  # the record's fast rail
ANVIL_LANE_BETA = 0.9  # the lane energy EMA: `lane_beta` in the noise code; anvil.py's own default is not in its file
AMUSE_WARMUP_STEPS = 40  # T0: the record's 2 percent warmup of about 2,000 steps
ORTHOGONALIZERS = {"polar_express": 0, "mud1": 1, "mud2": 2}  # MUD's passes; 0 is Polar Express
RADIAL_REFERENCE_STD = 0.02  # the record's std: a matrix under this entry RMS steps without the radial rule
RADIAL_OFF = 1.0
POWER_ITERATIONS = 2  # contra's s1, warm-started from the last step's vector
MUOWN_DEFAULT_RATIO = 0.18  # Adam's rate over Muon's: 1 / (0.2 sqrt(768)), the paper's tie at 768 x 768


@dataclass(frozen=True)
class OptimizerConfig(StrategyConfig):
    variant: str = "normuon"
    muon_lr: float = 0.04
    embed_lr: float = 3e-3  # 017 and 022
    head_lr: float = 1e-3
    scalar_lr: float = 3e-3  # the strategies' scalars and small matrices (028 put its mix scalars at the embedding rate)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    cautious_weight_decay: float = 0.0
    plus_axis: str = "col_row"  # muon_plus: the normalization's axes, in order
    anvil_slow_beta: float = 0.98  # anvil: the slow rail's EMA rate
    anvil_fast_weight: float = 0.4385  # anvil: the fast rail's share of the blend
    amuse_beta: float = 0.6  # amuse: the interpolation toward the average through the warmup
    amuse_rho: float = 0.8  # amuse: how fast beta_t rises toward 1 after it
    adam_beta2_end: float = 0.95  # AdamW's beta2 at the end of the cooldown
    decay_timescale: float = 0.0  # the Muon banks' decay timescale in total runs; 0 is off
    fused_polar: int = 0  # 1 compiles Polar Express per bank shape
    depth_taper: float = 0.0  # Muon's rate at block i times 1 + taper (1/2 - i / 35); 0 is the record's flat rate
    orthogonalizer: str = "polar_express"  # or mud1, mud2: the map from the Nesterov direction to the update
    polar_steps: int = 5  # the first n of Polar Express's steps
    contra_start: float = 0.0  # Contra-Muon's c at the start of the clock; below 0 it is soft
    contra_end: float = 0.0  # c at the end of the clock
    radial_out: float = RADIAL_OFF  # the radial part's scale when an update moves a matrix away from zero
    radial_in: float = RADIAL_OFF  # the scale when it moves the matrix toward zero
    variance_scale: int = 0  # 1 divides the direction by its running elementwise spread before the orthogonalization
    mona_beta: float = 0.0  # the slow gradient EMA's rate; 0 is off
    normuon_beta: float = 0.95  # EMA rate of NorMuon's per-row second moment
    nesterov: int = 1  # 1 orthogonalizes the lookahead, 0 the momentum itself
    qk_lr_scale: float = 1.0  # Muon's rate on the query and key matrices, relative to `muon_lr`
    vo_lr_scale: float = 1.0  # on the value and attention output matrices
    mlp_lr_scale: float = 1.0  # on the MLP's two matrices
    muown_ratio: float = MUOWN_DEFAULT_RATIO  # muown: the magnitudes' Adam rate over Muon's
    VARIANTS: ClassVar[tuple[str, ...]] = ("muon", "normuon", "muon_plus", "anvil", "amuse", "adamuon", "muown")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "muon_lr": (0.005, 0.12), "embed_lr": (3e-4, 3e-2), "head_lr": (1e-4, 1e-2), "scalar_lr": (3e-4, 3e-2),
        "adam_beta1": (0.7, 0.95), "adam_beta2": (0.9, 0.999), "weight_decay": (0.0, 0.2),
        "cautious_weight_decay": (0.0, 0.2), "anvil_slow_beta": (0.9, 0.995), "anvil_fast_weight": (0.1, 0.9),
        "amuse_beta": (0.2, 0.8), "amuse_rho": (0.0, 1.0), "adam_beta2_end": (0.9, 0.9995), "decay_timescale": (0.0, 10.0),
        "fused_polar": (0, 1), "depth_taper": (-1.0, 1.0), "polar_steps": (1, 5), "contra_start": (-1.0, 1.0),
        "contra_end": (-1.0, 1.0), "radial_out": (0.0, 1.0), "radial_in": (0.0, 1.0), "variance_scale": (0, 1),
        "mona_beta": (0.0, 0.999), "normuon_beta": (0.5, 0.999), "nesterov": (0, 1), "qk_lr_scale": (0.25, 4.0),
        "vo_lr_scale": (0.25, 4.0), "mlp_lr_scale": (0.25, 4.0), "muown_ratio": (0.02, 1.0),
    }

    def block_scale(self, index: int, blocks: int) -> float:
        """Muon's rate multiplier at block `index` of `blocks`: 1 + taper (1/2 - index / (blocks - 1)), mean 1 over them."""
        return 1 + self.depth_taper * (0.5 - index / (blocks - 1))

    @property
    def scales_members(self) -> bool:
        """True when some block matrix's rate differs from `muon_lr`'s, so the banks carry a rate tensor."""
        return bool(self.depth_taper) or (self.qk_lr_scale, self.vo_lr_scale, self.mlp_lr_scale) != (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.plus_axis not in PLUS_AXES:
            raise ValueError(f"plus_axis must be one of {PLUS_AXES}, not {self.plus_axis!r}")
        if self.decay_timescale != 0.0 and self.decay_timescale < 0.1:
            raise ValueError(f"decay_timescale is 0 (off) or 0.1 to 10, not {self.decay_timescale}")
        if self.fused_polar not in (0, 1):
            raise ValueError(f"fused_polar is 0 or 1, not {self.fused_polar!r}")
        if self.orthogonalizer not in ORTHOGONALIZERS:
            raise ValueError(f"orthogonalizer must be one of {tuple(ORTHOGONALIZERS)}, not {self.orthogonalizer!r}")
        if self.polar_steps not in range(1, len(POLAR_EXPRESS_COEFFICIENTS) + 1):
            raise ValueError(f"polar_steps is 1 to {len(POLAR_EXPRESS_COEFFICIENTS)}, not {self.polar_steps!r}")
        if self.variance_scale not in (0, 1) or self.nesterov not in (0, 1):
            raise ValueError(f"variance_scale and nesterov are 0 or 1, not {self.variance_scale!r}, {self.nesterov!r}")
        if self.mona_beta != 0.0 and self.mona_beta < 0.9:
            raise ValueError(f"mona_beta is 0 (off) or 0.9 to 0.999, not {self.mona_beta}")


def amuse_beta(step: int, beta: float, rho: float) -> float:
    """AMUSE's interpolation toward the average at step t (from 1): constant through T0, then rising toward 1."""
    if step < AMUSE_WARMUP_STEPS:
        return beta
    return 1 - (AMUSE_WARMUP_STEPS / step) ** rho * (1 - beta)


def plus_normalize(U: Tensor, axis: str) -> Tensor:
    """Muon+'s normalization of a bank of orthogonalized updates: unit rows, unit columns, or both in `axis`'s order."""
    # U: (k, m, n); returns (k, m, n).
    for single in axis.split("_"):
        dim = 2 if single == "row" else 1  # a row's entries run along n, a column's along m
        U = U / U.norm(dim=dim, keepdim=True).clamp(min=1e-8)  # (k, m, n)
    return U  # (k, m, n)

def polar_express(G: Tensor, safety_factor: float = 0.02, epsilon: float = 1e-6, steps: int = len(POLAR_EXPRESS_COEFFICIENTS)) -> Tensor:
    """The five-step Polar Express of modded-nanogpt, or its first `steps`, on a bank of matrices at once."""
    # G: (k, m, n) in fp32; returns (k, m, n), bf16 on CUDA.
    norms = G.flatten(1).norm(dim=1).view(-1, 1, 1)  # (k, 1, 1)
    X = G / (norms * (1 + safety_factor) + epsilon)  # (k, m, n)
    if G.device.type == "cuda":
        X = X.bfloat16()
    transposed = G.shape[1] > G.shape[2]
    if transposed:
        X = X.mT  # (k, r, s)
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS[:steps]:
        A = X @ X.mT  # (k, r, r)
        B = b * A + c * (A @ A)  # (k, r, r)
        X = a * X + B @ X  # (k, r, s)
    return X.mT if transposed else X  # (k, m, n)


def mud(G: Tensor, passes: int, epsilon: float = 1e-6) -> Tensor:
    """MUD (arXiv 2603.17970, Algorithm 2): `passes` forward sweeps of Q <- tril(Q Q^T)^-1 Q between unit-row
    normalizations, along the matrix's shorter dimension as Muon's convention has it. An all-zero row stays zero."""
    # G: (k, m, n) in fp32; returns (k, m, n) in fp32.
    transposed = G.shape[1] > G.shape[2]
    X = G.mT if transposed else G  # (k, r, s)
    X = X / X.norm(dim=2, keepdim=True).clamp(min=epsilon)  # (k, r, s), unit rows
    for _ in range(passes):
        L = (X @ X.mT).tril()  # (k, r, r), a unit diagonal but for zero rows
        L.diagonal(dim1=1, dim2=2).clamp_(min=epsilon)
        X = torch.linalg.solve_triangular(L, X, upper=False)  # (k, r, s)
        X = X / X.norm(dim=2, keepdim=True).clamp(min=epsilon)  # (k, r, s)
    return X.mT if transposed else X  # (k, m, n)


@functools.cache
def compiled_polar_express(steps: int = len(POLAR_EXPRESS_COEFFICIENTS)) -> Callable[[Tensor], Tensor]:
    """Polar Express under the compiler, one static graph per bank shape (three in the record)."""
    if steps == len(POLAR_EXPRESS_COEFFICIENTS):
        return torch.compile(polar_express, dynamic=False)  # (k, m, n) to (k, m, n)

    def truncated(G: Tensor) -> Tensor:
        # G: (k, m, n) in fp32; returns (k, m, n), bf16 on CUDA.
        return polar_express(G, steps=steps)  # (k, m, n)

    return torch.compile(truncated, dynamic=False)  # (k, m, n) to (k, m, n)


def top_singular_value(M: Tensor, vector: Tensor) -> Tensor:
    """Each matrix's top singular value by power iteration, `vector` updated in place for the next step's warm start. A
    zero matrix keeps its vector and reads zero."""
    # M: (k, m, n); vector: (k, n, 1); returns (k, 1, 1).
    left = M @ vector  # (k, m, 1)
    for _ in range(POWER_ITERATIONS):
        right = M.mT @ left  # (k, n, 1)
        norms = right.norm(dim=1, keepdim=True)  # (k, 1, 1)
        vector.copy_(torch.where(norms > 0, right / norms.clamp(min=1e-12), vector))  # (k, n, 1)
        left = M @ vector  # (k, m, 1)
    return left.norm(dim=1, keepdim=True)  # (k, 1, 1)


def muown_state(members: list[Tensor]) -> dict[str, Tensor]:
    """Muown's state for a bank: the rows' directions from the weights, a zero row's drawn at the record's std so its weights
    start at zero and can still turn, the magnitudes at the rows' norms, and Adam's moments at zero."""
    # members: (m, n) each.
    W = torch.stack(members).float()  # (k, m, n)
    g = W.norm(dim=2, keepdim=True)  # (k, m, 1)
    R = torch.where(g > 0, W, RADIAL_REFERENCE_STD * torch.randn_like(W))  # (k, m, n)
    return {"R": R, "g": g, "m_g": torch.zeros_like(g), "v_g": torch.zeros_like(g)}  # R: (k, m, n); the others: (k, m, 1)


def mona_accelerated(G: Tensor, slow: Tensor, beta: float) -> Tensor:
    """MONA's G + alpha A at alpha = -1 / (2 (1 - beta)): with S the EMA of the earlier gradients, A_k = (1 - beta) (G_k - S_{k-1}),
    so the sum is (G + S) / 2, and one buffer holds what the paper's two do. `slow` moves on to this gradient in place."""
    # G, slow: (k, m, n); returns (k, m, n).
    accelerated = (G + slow) / 2  # (k, m, n)
    slow.lerp_(G, 1 - beta)  # (k, m, n)
    return accelerated  # (k, m, n)


def radially_braked(P: Tensor, step: Tensor, outward: float, inward: float) -> Tensor:
    """The matrices after a step, its radial part scaled by the side it moves them to and their norms set to what that
    leaves (modded-nanogpt PR 294). A matrix under the reference norm takes the plain step."""
    # P, step: (k, m, n), the weights and the signed update; returns (k, m, n).
    norms = P.flatten(1).norm(dim=1).view(-1, 1, 1)  # (k, 1, 1)
    relative = (step * P).flatten(1).sum(dim=1).view(-1, 1, 1) / norms.square().clamp(min=1e-12)  # (k, 1, 1), the radial change over the norm
    kept = torch.where(relative > 0, outward, inward)  # (k, 1, 1)
    moved = P + step  # (k, m, n)
    target = norms * (1 + kept * relative)  # (k, 1, 1)
    rescaled = moved * (target / moved.flatten(1).norm(dim=1).view(-1, 1, 1).clamp(min=1e-12))  # (k, m, n)
    reference = RADIAL_REFERENCE_STD * (P.shape[1] * P.shape[2]) ** 0.5
    return torch.where(norms < reference, moved, rescaled)  # (k, m, n)


class Muon(torch.optim.Optimizer):
    """Muon with Polar Express, one batched orthogonalization per parameter shape.

    The matrices of one shape (144 of 768 x 768, 36 of 2,048 x 768, 36 of 768 x 2,048 in the record) share a stacked
    momentum buffer and one batched Polar Express, so a step launches tens of kernels instead of the thousands a
    matrix-by-matrix loop does. The update of each matrix equals the per-matrix one (checked on CPU to zero difference).
    """

    def __init__(self, params: Iterable[Tensor], lr: float, momentum: float, weight_decay: float,
                 normalize_rows: bool = True, cautious_weight_decay: float = 0.0, plus_axis: str | None = None,
                 rails: tuple[float, float] | None = None, schedule_free: tuple[float, float] | None = None,
                 fused_polar: bool = False, elementwise: bool = False, member_scales: dict[int, float] | None = None,
                 mud_passes: int = 0, polar_steps: int = len(POLAR_EXPRESS_COEFFICIENTS), contra: bool = False,
                 radial: tuple[float, float] = (RADIAL_OFF, RADIAL_OFF), variance_scale: bool = False, mona_beta: float = 0.0,
                 normuon_beta: float = 0.95, nesterov: bool = True, muown: tuple[float, float, float] | None = None) -> None:
        # rails: ANVIL's (slow beta, fast weight); schedule_free: AMUSE's (beta, rho); None for the record's Muon.
        # elementwise: AdaMuon's per-entry second moment; member_scales: a rate multiplier by parameter id, 1 if absent.
        # mud_passes: MUD's passes, 0 for Polar Express; contra: the strategy sets `contra_coefficient` each step.
        # radial: the (outward, inward) scales; muown: the (Adam rate over Muon's, beta1, beta2) of the row magnitudes.
        # params: (m, n) each, the matrices Muon updates.
        parameters = list(params)
        assert all(parameter.ndim == 2 and parameter.is_floating_point() for parameter in parameters)
        assert not (normalize_rows and elementwise), "NorMuon's rows and AdaMuon's entries are two variants"
        super().__init__(parameters, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))
        self.normalize_rows = normalize_rows
        self.elementwise = elementwise
        self.cautious_weight_decay = cautious_weight_decay
        self.plus_axis = plus_axis
        self.rails = rails
        self.schedule_free = schedule_free
        self.fused_polar = fused_polar
        self.mud_passes = mud_passes
        self.polar_steps = polar_steps
        self.radial = radial
        self.variance_scale = variance_scale
        self.mona_beta = mona_beta
        self.normuon_beta = normuon_beta
        self.nesterov = nesterov
        self.muown = muown
        self.contra_coefficient = 0.0  # Contra-Muon's c, set each step by the strategy; 0 is off
        self.decay_per_step = 0.0  # the constant-timescale decay, set each step by the strategy; 0 is off
        self.steps = 0
        self.spread_weight = 0.0  # `variance_scale`'s bias correction: the EMA of ones under the same momentum
        for group in self.param_groups:
            by_shape: dict[tuple[int, ...], list[Tensor]] = {}
            for parameter in group["params"]:
                by_shape.setdefault(tuple(parameter.shape), []).append(parameter)
            group["banks"] = [(members, torch.zeros(len(members), *shape, dtype=torch.float32, device=members[0].device),
                               torch.zeros(len(members), shape[0], shape[1] if elementwise else 1, dtype=torch.float32,
                                           device=members[0].device))
                              for shape, members in by_shape.items()]  # momentum buffer (k, m, n), second moment (k, m, 1 or n)
            group["depth"] = [None if member_scales is None else
                              torch.tensor([member_scales.get(id(member), 1.0) for member in members],
                                           device=members[0].device).view(-1, 1, 1)
                              for members, _, _ in group["banks"]]  # (k, 1, 1) rate multipliers, or None
            if rails is not None:  # the slow rail (k, m, n) and the lane energy, (k, m, 1) tall or (k, 1, n) wide
                group["anvil"] = [(torch.zeros_like(buffer), torch.zeros(len(members), *lane_shape(members[0].shape),
                                                                         dtype=torch.float32, device=members[0].device))
                                  for members, buffer, _ in group["banks"]]
            if schedule_free is not None:  # the base iterate Z and the average X, each (k, m, n) in fp32, from the weights
                group["amuse"] = [(torch.stack(members).float(), torch.stack(members).float()) for members, _, _ in group["banks"]]
            if muown is not None:
                group["muown"] = [muown_state(members) for members, _, _ in group["banks"]]
            if variance_scale:  # (k, m, n) each: the elementwise running spread of (M - G)
                group["spread"] = [torch.zeros_like(buffer) for _, buffer, _ in group["banks"]]
            if mona_beta:  # (k, m, n) each: the slow EMA of the earlier gradients
                group["slow"] = [torch.zeros_like(buffer) for _, buffer, _ in group["banks"]]
            if contra:  # (k, n, 1) each: the power iteration's unit vector, warm-started across steps
                group["top_vector"] = [torch.full((len(members), members[0].shape[1], 1), members[0].shape[1] ** -0.5,
                                                  dtype=torch.float32, device=members[0].device) for members, _, _ in group["banks"]]

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> None:
        # closure: () loss; never given, the loop computes the loss itself.
        assert closure is None
        self.steps += 1
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            if self.variance_scale:
                self.spread_weight = momentum * self.spread_weight + (1 - momentum)
            for bank, (members, buffer, second_moment) in enumerate(group["banks"]):
                if members[0].grad is None:
                    continue
                G = torch.stack([member.grad for member in members]).float()  # (k, m, n)
                polar = self._polar(G.device)
                depth = group["depth"][bank]  # (k, 1, 1) or None
                if self.schedule_free is not None:
                    self._amuse_update(members, G, buffer, *group["amuse"][bank], lr, momentum, polar, depth)
                    continue
                if self.muown is not None:
                    self._muown_update(members, G, buffer, group["muown"][bank], lr, momentum, polar, depth)
                    continue
                if self.rails is not None:
                    U = self._anvil_update(G, buffer, *group["anvil"][bank], momentum, polar)  # (k, m, n)
                else:
                    U = self._direction(group, bank, G, buffer, momentum, polar)  # (k, m, n)
                if self.normalize_rows:
                    second_moment.lerp_(U.square().mean(dim=2, keepdim=True), 1 - self.normuon_beta)  # (k, m, 1)
                    normalized = U / (second_moment.sqrt() + 1e-8)  # (k, m, n)
                    U = normalized * (U.flatten(1).norm(dim=1) / normalized.flatten(1).norm(dim=1).clamp(min=1e-8)).view(-1, 1, 1)  # (k, m, n), each matrix's norm restored
                if self.elementwise:  # AdaMuon: the same with one second moment per entry
                    second_moment.lerp_(U.square(), 1 - self.normuon_beta)  # (k, m, n)
                    normalized = U / (second_moment.sqrt() + 1e-8)  # (k, m, n)
                    U = normalized * (U.flatten(1).norm(dim=1) / normalized.flatten(1).norm(dim=1).clamp(min=1e-8)).view(-1, 1, 1)  # (k, m, n), each matrix's norm restored
                if self.plus_axis is not None:
                    normalized = plus_normalize(U, self.plus_axis)  # (k, m, n)
                    U = normalized * (U.flatten(1).norm(dim=1) / normalized.flatten(1).norm(dim=1).clamp(min=1e-8)).view(-1, 1, 1)  # (k, m, n), each matrix's norm restored
                scale = max(1, members[0].shape[0] / members[0].shape[1]) ** 0.5
                if group["weight_decay"]:
                    torch._foreach_mul_(members, 1 - lr * group["weight_decay"])
                if self.decay_per_step:  # the constant timescale, in place of the cautious decay
                    torch._foreach_mul_(members, 1 - self.decay_per_step)
                elif self.cautious_weight_decay:  # decay only where the update and the weight share a sign (record 43)
                    P = torch.stack(members)  # (k, m, n)
                    torch._foreach_sub_(members, list((P * (U * P >= 0)).unbind(0)), alpha=lr * self.cautious_weight_decay)
                if depth is not None:
                    U = U * depth  # (k, m, n), each block's rate
                if self.radial != (RADIAL_OFF, RADIAL_OFF):
                    moved = radially_braked(torch.stack(members).float(), -lr * scale * U, *self.radial)  # (k, m, n)
                    torch._foreach_copy_(members, list(moved.to(members[0].dtype).unbind(0)))
                else:
                    torch._foreach_add_(members, list(U.to(members[0].dtype).unbind(0)), alpha=-lr * scale)

    def _direction(self, group: dict[str, Any], bank: int, G: Tensor, buffer: Tensor, momentum: float,
                   polar: Callable[[Tensor], Tensor]) -> Tensor:
        """The orthogonalized update: momentum, the optional scalings of what is orthogonalized, and Contra-Muon's mix.
        At the record's numbers this is buffer.lerp_(G, 1 - momentum) and polar(G.lerp(buffer, momentum))."""
        # G, buffer: (k, m, n); polar: (k, m, n) to (k, m, n); returns (k, m, n) in fp32.
        if self.mona_beta:
            G = mona_accelerated(G, group["slow"][bank], self.mona_beta)  # (k, m, n)
        if self.variance_scale:
            spread = group["spread"][bank]  # (k, m, n)
            spread.mul_(momentum).add_((buffer - G).square(), alpha=momentum * (1 - momentum))  # (k, m, n), Muon-VS's Eq. 9
        buffer.lerp_(G, 1 - momentum)  # (k, m, n)
        M = G.lerp(buffer, momentum) if self.nesterov else buffer  # (k, m, n)
        if self.variance_scale:
            M = M / ((spread / self.spread_weight).sqrt() + 1e-12)  # (k, m, n), bias-corrected
        U = polar(M).float()  # (k, m, n)
        if self.contra_coefficient:
            s1 = top_singular_value(M, group["top_vector"][bank])  # (k, 1, 1)
            U = (1 + self.contra_coefficient) * U - self.contra_coefficient * M / s1.clamp(min=1e-12)  # (k, m, n)
        return U  # (k, m, n)

    def _muown_update(self, members: list[Tensor], G: Tensor, M: Tensor, state: dict[str, Tensor], lr: float, momentum: float,
                      polar: Callable[[Tensor], Tensor], depth: Tensor | None) -> None:
        """Muown (arXiv 2605.10797, Algorithm 1): Muon on each row's direction, Adam on its magnitude."""
        # members: (m, n) each; G, M: (k, m, n); state: (k, m, n) R and (k, m, 1) g, m_g, v_g; polar: (k, m, n) to (k, m, n); depth: (k, 1, 1) or None.
        assert self.muown is not None
        ratio, beta1, beta2 = self.muown
        R, g = state["R"], state["g"]  # (k, m, n), (k, m, 1)
        r = R.norm(dim=2, keepdim=True).clamp(min=1e-12)  # (k, m, 1)
        D = R / r  # (k, m, n), unit rows
        magnitude_gradient = (G * D).sum(dim=2, keepdim=True)  # (k, m, 1)
        direction_gradient = (g / r) * (G - magnitude_gradient * D)  # (k, m, n), the chain rule through W = (g / r) R
        M.lerp_(direction_gradient, 1 - momentum)  # (k, m, n)
        U = polar(direction_gradient.lerp(M, momentum)).float()  # (k, m, n), Nesterov
        scale = max(1, members[0].shape[0] / members[0].shape[1]) ** 0.5
        R.add_(U if depth is None else U * depth, alpha=-lr * scale)  # (k, m, n)
        state["m_g"].lerp_(magnitude_gradient, 1 - beta1)  # (k, m, 1)
        state["v_g"].lerp_(magnitude_gradient.square(), 1 - beta2)  # (k, m, 1)
        mean = state["m_g"] / (1 - beta1**self.steps)  # (k, m, 1), bias-corrected
        variance = state["v_g"] / (1 - beta2**self.steps)  # (k, m, 1)
        step = ratio * lr * mean / (variance.sqrt() + 1e-8)  # (k, m, 1)
        g.sub_(step if depth is None else step * depth)  # (k, m, 1)
        W = (g / R.norm(dim=2, keepdim=True).clamp(min=1e-12)) * R  # (k, m, n)
        torch._foreach_copy_(members, list(W.to(members[0].dtype).unbind(0)))

    def _polar(self, device: torch.device) -> Callable[[Tensor], Tensor]:
        """The orthogonalization: MUD, or Polar Express, compiled when `fused_polar` is on and the banks are on CUDA."""
        if self.mud_passes:
            return functools.partial(mud, passes=self.mud_passes)  # (k, m, n) to (k, m, n)
        if self.fused_polar and device.type == "cuda":
            return compiled_polar_express(self.polar_steps)  # (k, m, n) to (k, m, n)
        if self.polar_steps != len(POLAR_EXPRESS_COEFFICIENTS):
            return functools.partial(polar_express, steps=self.polar_steps)  # (k, m, n) to (k, m, n)
        return polar_express  # (k, m, n) to (k, m, n)

    def _amuse_update(self, members: list[Tensor], G: Tensor, M: Tensor, Z: Tensor, X: Tensor, lr: float, momentum: float,
                      polar: Callable[[Tensor], Tensor], depth: Tensor | None) -> None:
        """AMUSE: Muon steps the base iterate, the average follows it, the weights move to their interpolation."""
        # members: (m, n) each; G, M, Z, X: (k, m, n); polar: (k, m, n) to (k, m, n); depth: (k, 1, 1) or None; the gradient was taken at the weights, Y.
        assert self.schedule_free is not None
        beta, rho = self.schedule_free
        M.lerp_(G, 1 - momentum)  # (k, m, n), mu M + (1 - mu) G
        scale = max(1, members[0].shape[0] / members[0].shape[1]) ** 0.5
        U = polar(M).float()  # (k, m, n)
        Z.add_(U if depth is None else U * depth, alpha=-lr * scale)  # (k, m, n)
        X.lerp_(Z, 1 / (self.steps + 1))  # (k, m, n), the uniform average of the Z iterates
        Y = torch.lerp(Z, X, amuse_beta(self.steps, beta, rho))  # (k, m, n), (1 - beta_t) Z + beta_t X
        torch._foreach_copy_(members, list(Y.to(members[0].dtype).unbind(0)))

    @contextlib.contextmanager
    def averaged_weights(self) -> Iterator[None]:
        """AMUSE's average X in the weights while a validation pass scores them; Y returns afterwards."""
        assert self.schedule_free is not None
        live = [member.detach() for group in self.param_groups for members, _, _ in group["banks"] for member in members]
        saved = [member.clone() for member in live]
        averages = [average for group in self.param_groups for _, X in group["amuse"] for average in X.unbind(0)]
        torch._foreach_copy_(live, [average.to(member.dtype) for average, member in zip(averages, live, strict=True)])
        try:
            yield
        finally:
            torch._foreach_copy_(live, saved)

    def _anvil_update(self, G: Tensor, fast: Tensor, slow: Tensor, lane: Tensor, momentum: float,
                      polar: Callable[[Tensor], Tensor]) -> Tensor:
        """ANVIL: two rails blended into the lookahead, Polar Express, then each lane equalized by its energy EMA."""
        # G, fast, slow: (k, m, n); lane: (k, m, 1) or (k, 1, n); polar: (k, m, n) to (k, m, n); returns (k, m, n).
        assert self.rails is not None
        slow_beta, fast_weight = self.rails
        fast.lerp_(G, 1 - ANVIL_FAST_BETA)  # (k, m, n)
        slow.lerp_(G, 1 - slow_beta)  # (k, m, n)
        blend = fast_weight * fast + (1 - fast_weight) * slow  # (k, m, n)
        U = polar(G.lerp(blend, momentum)).float()  # (k, m, n)
        lanes = 2 if lane.shape[2] == 1 else 1  # the reduction runs along the longer dimension
        lane.lerp_(U.square().mean(dim=lanes, keepdim=True), 1 - ANVIL_LANE_BETA)  # (k, m, 1) or (k, 1, n)
        equalized = U * lane.clamp(min=1e-10).rsqrt()  # (k, m, n)
        return equalized * (U.flatten(1).norm(dim=1) / equalized.flatten(1).norm(dim=1).clamp(min=1e-10)).view(-1, 1, 1)  # (k, m, n), each matrix's norm restored


def lane_shape(shape: torch.Size) -> tuple[int, int]:
    """ANVIL's lanes: the rows of a tall or square matrix, the columns of a wide one."""
    rows, columns = shape
    return (rows, 1) if rows >= columns else (1, columns)


def distinct(parameters: list[Tensor]) -> list[Tensor]:
    """Each parameter once, at its first place: a shared matrix must not be stepped once per block that holds it."""
    # parameters: (...) each, any shapes.
    return list({id(parameter): parameter for parameter in parameters}.values())  # (...) each, the first of each identity


class OptimizerChoice(Strategy):
    family = "optimizer"
    Config = OptimizerConfig
    config: OptimizerConfig

    def build_optimizers(self, model: CandidateModel, momentum: float) -> list[torch.optim.Optimizer]:
        """Muon on every block matrix no strategy claimed, then AdamW on the embedding, the head and the claimed groups.
        Every trainable parameter lands in exactly one group."""
        groups: list[ParameterGroup] = [group for strategy in model.strategies
                                        for group in strategy.parameter_groups(model)]
        claimed = {id(parameter) for group in groups for parameter in group.parameters}
        # A tied loop (`loop`, `sglr`) shares matrices between blocks, so each list keeps a parameter once, at its first
        # place; with nothing shared the lists are the record's, in the record's order.
        block_matrices = distinct([parameter for block in model.blocks for parameter in block.parameters()
                                   if parameter.ndim == 2 and id(parameter) not in claimed])
        block_matrices += [parameter for group in groups if group.optimizer == "muon" for parameter in group.parameters]
        adam_groups = [{"params": [model.embedding.weight], "lr": self.config.embed_lr},
                       {"params": [model.head.weight], "lr": self.config.head_lr}]
        adam_groups += [{"params": distinct(group.parameters), "lr": group.lr} for group in groups if group.optimizer == "adam"]
        optimized = {id(parameter) for parameter in block_matrices} | {id(parameter) for group in adam_groups for parameter in group["params"]}
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and id(parameter) not in optimized]
        assert not missing, f"parameters no optimizer updates: {missing}"
        assert len(optimized) == len(trainable), "a parameter sits in two optimizer groups"
        config = self.config
        self.muon = Muon(block_matrices, lr=config.muon_lr, momentum=momentum, weight_decay=config.weight_decay,
                         normalize_rows=self.variant == "normuon", cautious_weight_decay=config.cautious_weight_decay,
                         plus_axis=config.plus_axis if self.variant == "muon_plus" else None,
                         rails=(config.anvil_slow_beta, config.anvil_fast_weight) if self.variant == "anvil" else None,
                         schedule_free=(config.amuse_beta, config.amuse_rho) if self.variant == "amuse" else None,
                         fused_polar=bool(config.fused_polar), elementwise=self.variant == "adamuon",
                         member_scales=self._member_scales(model) if config.scales_members else None,
                         mud_passes=ORTHOGONALIZERS[config.orthogonalizer], polar_steps=config.polar_steps,
                         contra=self.contra_active, radial=(config.radial_out, config.radial_in),
                         variance_scale=bool(config.variance_scale), mona_beta=config.mona_beta, normuon_beta=config.normuon_beta,
                         nesterov=bool(config.nesterov),
                         muown=(config.muown_ratio, config.adam_beta1, config.adam_beta2) if self.variant == "muown" else None)
        self.adam = torch.optim.AdamW(adam_groups, betas=(self.config.adam_beta1, self.config.adam_beta2),
                                      weight_decay=self.config.weight_decay, fused=model.embedding.weight.is_cuda)
        return [self.muon, self.adam]

    def _member_scales(self, model: CandidateModel) -> dict[int, float]:
        """Muon's rate multiplier for every block parameter, by id: `depth_taper`'s times its role's; a strategy's Muon
        matrices keep 1. The attention's other matrices and a matrix a strategy added to a block keep the depth's alone."""
        config = self.config
        scales: dict[int, float] = {}
        for index, block in enumerate(model.blocks):
            depth = config.block_scale(index, len(model.blocks))
            scales.update({id(parameter): depth for parameter in block.parameters()})
            attention, mlp = block.attention, block.mlp
            for linears, role in (((attention.query, attention.key), config.qk_lr_scale), ((attention.value, attention.output), config.vo_lr_scale),
                                  ((mlp.up, mlp.down), config.mlp_lr_scale)):
                scales.update({id(linear.weight): depth * role for linear in linears if linear is not None})  # no K or V under `kv_share`
        return scales

    @property
    def contra_active(self) -> bool:
        """True when Contra-Muon's coefficient is not 0 throughout, on a variant whose update goes through `_direction`."""
        return (self.config.contra_start, self.config.contra_end) != (0.0, 0.0) and self.variant not in ("anvil", "amuse", "muown")

    def after_backward(self, state: StepState) -> None:
        """AdamW's beta2 along the cooldown, the banks' constant-timescale decay and Contra-Muon's coefficient, before the
        optimizers step."""
        config = self.config
        if config.adam_beta2_end != config.adam_beta2:
            share = state.model.strategies.schedule.config.cooldown_share(state.progress)
            beta2 = config.adam_beta2 + share * (config.adam_beta2_end - config.adam_beta2)
            for group in self.adam.param_groups:
                group["betas"] = (group["betas"][0], beta2)
        if config.decay_timescale:
            total_steps = (state.step + 1) / state.progress if state.progress > 0 else 0.0  # the run's length, estimated
            reference_steps = total_steps / state.run_scale  # the run's length at the 10-minute track's budget
            self.muon.decay_per_step = 1 / (config.decay_timescale * reference_steps) if total_steps else 0.0
        if self.contra_active:
            self.muon.contra_coefficient = config.contra_start + state.progress * (config.contra_end - config.contra_start)

    @property
    def schedule_free(self) -> bool:
        """True when the optimizer averages its own iterates, so the schedule must not cool down."""
        return self.variant == "amuse"

    def overrides(self, hook: str) -> bool:
        if hook == "after_backward":
            return self.config.adam_beta2_end != self.config.adam_beta2 or self.config.decay_timescale > 0 or self.contra_active
        return hook == "evaluation_weights" and self.schedule_free

    @contextlib.contextmanager
    def evaluation_weights(self, model: CandidateModel) -> Iterator[None]:
        with self.muon.averaged_weights():
            yield
