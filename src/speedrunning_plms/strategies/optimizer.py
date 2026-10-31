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
  orthogonalized update instead of per row (beta NORMUON_BETA), each matrix's norm restored as NorMuon's is, so
  `muon_lr` keeps its meaning; the paper's own 0.2 sqrt(mn) RMS rescale is not used for that reason. The state grows
  from (k, m, 1) to (k, m, n): about 1 GB of fp32 for the record's block matrices.
- `fused_polar`: Polar Express under `torch.compile(dynamic=False)`, one graph per bank shape, so its elementwise
  parts fuse around the bf16 matmuls; the same iteration and coefficients, so the update matches up to the compiler's
  reordering. CUDA only: on the CPU the loop never compiles (the model's rule), and Windows CPU inductor needs MSVC.

Symbols: k matrices in a bank, m and n a matrix's dimensions, r = min(m, n), s = max(m, n).
"""

from __future__ import annotations

import contextlib
import functools
import torch

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.optim.muon import POLAR_EXPRESS_COEFFICIENTS
from speedrunning_plms.strategies.strategy import ParameterGroup, StepState, Strategy, StrategyConfig


NORMUON_BETA = 0.95  # EMA rate of the per-row second moment of the orthogonalized update
PLUS_AXES = ("col_row", "row_col", "row", "col")
ANVIL_FAST_BETA = 0.85  # the record's fast rail
ANVIL_LANE_BETA = 0.9  # the lane energy EMA: `lane_beta` in the noise code; anvil.py's own default is not in its file
AMUSE_WARMUP_STEPS = 40  # T0: the record's 2 percent warmup of about 2,000 steps


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
    VARIANTS: ClassVar[tuple[str, ...]] = ("muon", "normuon", "muon_plus", "anvil", "amuse", "adamuon")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "muon_lr": (0.005, 0.12), "embed_lr": (3e-4, 3e-2), "head_lr": (1e-4, 1e-2), "scalar_lr": (3e-4, 3e-2),
        "adam_beta1": (0.7, 0.95), "adam_beta2": (0.9, 0.999), "weight_decay": (0.0, 0.2),
        "cautious_weight_decay": (0.0, 0.2), "anvil_slow_beta": (0.9, 0.995), "anvil_fast_weight": (0.1, 0.9),
        "amuse_beta": (0.2, 0.8), "amuse_rho": (0.0, 1.0), "adam_beta2_end": (0.9, 0.9995), "decay_timescale": (0.0, 10.0),
        "fused_polar": (0, 1), "depth_taper": (-1.0, 1.0),
    }

    def block_scale(self, index: int) -> float:
        """Muon's rate multiplier at block `index`: 1 + taper (1/2 - index / 35), mean 1 over the 36 blocks."""
        return 1 + self.depth_taper * (0.5 - index / (benchmark.BLOCKS - 1))

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.plus_axis not in PLUS_AXES:
            raise ValueError(f"plus_axis must be one of {PLUS_AXES}, not {self.plus_axis!r}")
        if self.decay_timescale != 0.0 and self.decay_timescale < 0.1:
            raise ValueError(f"decay_timescale is 0 (off) or 0.1 to 10, not {self.decay_timescale}")
        if self.fused_polar not in (0, 1):
            raise ValueError(f"fused_polar is 0 or 1, not {self.fused_polar!r}")


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

def polar_express(G: Tensor, safety_factor: float = 0.02, epsilon: float = 1e-6) -> Tensor:
    """The five-step Polar Express of modded-nanogpt, on a bank of matrices at once."""
    # G: (k, m, n) in fp32; returns (k, m, n), bf16 on CUDA.
    norms = G.flatten(1).norm(dim=1).view(-1, 1, 1)  # (k, 1, 1)
    X = G / (norms * (1 + safety_factor) + epsilon)  # (k, m, n)
    if G.device.type == "cuda":
        X = X.bfloat16()
    transposed = G.shape[1] > G.shape[2]
    if transposed:
        X = X.mT  # (k, r, s)
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
        A = X @ X.mT  # (k, r, r)
        B = b * A + c * (A @ A)  # (k, r, r)
        X = a * X + B @ X  # (k, r, s)
    return X.mT if transposed else X  # (k, m, n)


@functools.cache
def compiled_polar_express() -> Callable[[Tensor], Tensor]:
    """Polar Express under the compiler, one static graph per bank shape (three in the record)."""
    return torch.compile(polar_express, dynamic=False)  # type: ignore[return-value]


class Muon(torch.optim.Optimizer):
    """Muon with Polar Express, one batched orthogonalization per parameter shape.

    The matrices of one shape (144 of 768 x 768, 36 of 2,048 x 768, 36 of 768 x 2,048 in the record) share a stacked
    momentum buffer and one batched Polar Express, so a step launches tens of kernels instead of the thousands a
    matrix-by-matrix loop does. The update of each matrix equals the per-matrix one (checked on CPU to zero difference).
    """

    def __init__(self, params: Iterable[Tensor], lr: float, momentum: float, weight_decay: float,
                 normalize_rows: bool = True, cautious_weight_decay: float = 0.0, plus_axis: str | None = None,
                 rails: tuple[float, float] | None = None, schedule_free: tuple[float, float] | None = None,
                 fused_polar: bool = False, elementwise: bool = False, member_scales: dict[int, float] | None = None) -> None:
        # rails: ANVIL's (slow beta, fast weight); schedule_free: AMUSE's (beta, rho); None for the record's Muon.
        # elementwise: AdaMuon's per-entry second moment; member_scales: a rate multiplier by parameter id, 1 if absent.
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
        self.decay_per_step = 0.0  # the constant-timescale decay, set each step by the strategy; 0 is off
        self.steps = 0
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

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> None:
        # closure: () loss; never given, the loop computes the loss itself.
        assert closure is None
        self.steps += 1
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for bank, (members, buffer, second_moment) in enumerate(group["banks"]):
                if members[0].grad is None:
                    continue
                G = torch.stack([member.grad for member in members]).float()  # (k, m, n)
                polar = self._polar(G.device)
                depth = group["depth"][bank]  # (k, 1, 1) or None
                if self.schedule_free is not None:
                    self._amuse_update(members, G, buffer, *group["amuse"][bank], lr, momentum, polar, depth)
                    continue
                if self.rails is not None:
                    U = self._anvil_update(G, buffer, *group["anvil"][bank], momentum, polar)  # (k, m, n)
                else:
                    buffer.lerp_(G, 1 - momentum)  # (k, m, n)
                    U = polar(G.lerp(buffer, momentum)).float()  # (k, m, n), Nesterov
                if self.normalize_rows:
                    second_moment.lerp_(U.square().mean(dim=2, keepdim=True), 1 - NORMUON_BETA)  # (k, m, 1)
                    normalized = U / (second_moment.sqrt() + 1e-8)  # (k, m, n)
                    U = normalized * (U.flatten(1).norm(dim=1) / normalized.flatten(1).norm(dim=1).clamp(min=1e-8)).view(-1, 1, 1)  # (k, m, n), each matrix's norm restored
                if self.elementwise:  # AdaMuon: the same with one second moment per entry
                    second_moment.lerp_(U.square(), 1 - NORMUON_BETA)  # (k, m, n)
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
                torch._foreach_add_(members, list(U.to(members[0].dtype).unbind(0)), alpha=-lr * scale)

    def _polar(self, device: torch.device) -> Callable[[Tensor], Tensor]:
        """Polar Express, compiled when `fused_polar` is on and the banks are on CUDA."""
        return compiled_polar_express() if self.fused_polar and device.type == "cuda" else polar_express

    def _amuse_update(self, members: list[Tensor], G: Tensor, M: Tensor, Z: Tensor, X: Tensor, lr: float, momentum: float,
                      polar: Callable[[Tensor], Tensor], depth: Tensor | None) -> None:
        """AMUSE: Muon steps the base iterate, the average follows it, the weights move to their interpolation."""
        # G, M, Z, X: (k, m, n); depth: (k, 1, 1) or None; the gradient was taken at the weights, Y.
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
        # G, fast, slow: (k, m, n); lane: (k, m, 1) or (k, 1, n); returns (k, m, n).
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


class OptimizerChoice(Strategy):
    family = "optimizer"
    Config = OptimizerConfig
    config: OptimizerConfig

    def build_optimizers(self, model: nn.Module, momentum: float) -> list[torch.optim.Optimizer]:
        """Muon on every block matrix no strategy claimed, then AdamW on the embedding, the head and the claimed groups.
        Every trainable parameter lands in exactly one group."""
        groups: list[ParameterGroup] = [group for strategy in model.strategies  # type: ignore[union-attr]
                                        for group in strategy.parameter_groups(model)]
        claimed = {id(parameter) for group in groups for parameter in group.parameters}
        block_matrices = [parameter for block in model.blocks for parameter in block.parameters()  # type: ignore[union-attr]
                          if parameter.ndim == 2 and id(parameter) not in claimed]
        block_matrices += [parameter for group in groups if group.optimizer == "muon" for parameter in group.parameters]
        adam_groups = [{"params": [model.embedding.weight], "lr": self.config.embed_lr},  # type: ignore[union-attr]
                       {"params": [model.head.weight], "lr": self.config.head_lr}]  # type: ignore[union-attr]
        adam_groups += [{"params": group.parameters, "lr": group.lr} for group in groups if group.optimizer == "adam"]
        optimized = {id(parameter) for parameter in block_matrices} | {id(parameter) for group in adam_groups for parameter in group["params"]}
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and id(parameter) not in optimized]
        assert not missing, f"parameters no optimizer updates: {missing}"
        assert len(optimized) == len(trainable), "a parameter sits in two optimizer groups"
        self.muon = Muon(block_matrices, lr=self.config.muon_lr, momentum=momentum, weight_decay=self.config.weight_decay,
                         normalize_rows=self.variant == "normuon", cautious_weight_decay=self.config.cautious_weight_decay,
                         plus_axis=self.config.plus_axis if self.variant == "muon_plus" else None,
                         rails=(self.config.anvil_slow_beta, self.config.anvil_fast_weight) if self.variant == "anvil" else None,
                         schedule_free=(self.config.amuse_beta, self.config.amuse_rho) if self.variant == "amuse" else None,
                         fused_polar=bool(self.config.fused_polar), elementwise=self.variant == "adamuon",
                         member_scales=self._member_scales(model) if self.config.depth_taper else None)
        self.adam = torch.optim.AdamW(adam_groups, betas=(self.config.adam_beta1, self.config.adam_beta2),
                                      weight_decay=self.config.weight_decay, fused=model.embedding.weight.is_cuda)  # type: ignore[union-attr]
        return [self.muon, self.adam]

    def _member_scales(self, model: nn.Module) -> dict[int, float]:
        """`depth_taper`'s rate multiplier for every block parameter, by id; a strategy's Muon matrices keep 1."""
        return {id(parameter): self.config.block_scale(index) for index, block in enumerate(model.blocks)  # type: ignore[union-attr]
                for parameter in block.parameters()}

    def after_backward(self, state: StepState) -> None:
        """AdamW's beta2 along the cooldown and the banks' constant-timescale decay, before the optimizers step."""
        config = self.config
        if config.adam_beta2_end != config.adam_beta2:
            share = state.model.strategies.schedule.config.cooldown_share(state.progress)  # type: ignore[union-attr]
            beta2 = config.adam_beta2 + share * (config.adam_beta2_end - config.adam_beta2)
            for group in self.adam.param_groups:
                group["betas"] = (group["betas"][0], beta2)
        if config.decay_timescale:
            total_steps = (state.step + 1) / state.progress if state.progress > 0 else 0.0  # the run's length, estimated
            reference_steps = total_steps / state.run_scale  # the run's length at the 10-minute track's budget
            self.muon.decay_per_step = 1 / (config.decay_timescale * reference_steps) if total_steps else 0.0

    @property
    def schedule_free(self) -> bool:
        """True when the optimizer averages its own iterates, so the schedule must not cool down."""
        return self.variant == "amuse"

    def overrides(self, hook: str) -> bool:
        if hook == "after_backward":
            return self.config.adam_beta2_end != self.config.adam_beta2 or self.config.decay_timescale > 0
        return hook == "evaluation_weights" and self.schedule_free

    @contextlib.contextmanager
    def evaluation_weights(self, model: nn.Module) -> Iterator[None]:
        with self.muon.averaged_weights():
            yield
