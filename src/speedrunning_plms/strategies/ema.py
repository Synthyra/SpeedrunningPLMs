"""The EMA family: a transform of the weights over the end of the run is what validation scores and selects; the raw
weights stay live for training.

- `tail` (006, A12; modded-nanogpt record 92c): an exponential average from `start` of the clock at `decay` per step.
  006 on the 003 incumbent (decay 0.995 from 60 percent of the clock) led mid-cooldown and lagged at the end
  (val_select 2.7578 against 2.7559); a shorter memory is untried.
- `latest_k`: LAWA, latest-weight averaging (arXiv 2306.03241 for LLM pre-training): the uniform mean of the latest
  LAWA_SNAPSHOTS weight snapshots, one every `window` / LAWA_SNAPSHOTS steps, so the mean spans about the last
  `window` steps. A true running mean over every step of the window would hold `window` copies of the weights; a ring
  of 8 snapshots holds 8 and gives the strided uniform mean LAWA itself takes over checkpoints. The mean is computed
  from the ring when a validation pass asks for it, so no running sum drifts.
- `drift_leap` (noise PR 27): validation scores w + `leap` (w_earlier - w), one move back along the recent drift, with
  w_earlier the oldest of DRIFT_SNAPSHOTS snapshots taken every `window` / DRIFT_SNAPSHOTS steps, so from 3/4 to all
  of `window` steps old once the ring is full (the oldest held before then). The PR also stops at 95 percent of its
  schedule; that is a schedule change and is not part of this variant.
- `two_decay` (noise PR 52): two exponential averages from `start`, at `decay` and `decay_fast`. Each validation pass
  scores both on val_select (the loop's `evaluation_options`), and the lower one is the pass's point and what selection
  keeps, so the choice reads validation only, never test; the final selected weights are whichever average scored
  lowest at the best pass. A pass costs two evaluations.
- `plateau_restart` (EWA, arXiv 2306.03241, and PMA, arXiv 2505.12082; `2026-10-30_regularization_and_averaging.md`
  idea 3): the `latest_k` ring of LAWA_SNAPSHOTS snapshots over the last `window` steps, and once, at the first step at
  or after `start` of the clock (0.6, the record's cooldown start), the live weights are replaced by the ring's
  uniform mean and training continues from there; the optimizers' moments carry on. Validation scores the live
  weights throughout. EWA's own protocol (Section 3) was not read.

The rings empty at the first applied step, so the compile warmup's steps never enter them.

`window`, `decay` and `decay_fast` are in the 10-minute track's steps and scale with the run (`StepState.run_scale`,
the budget over 600 s): a window of 400 is 400 steps at 600 s and 2,400 at 3,600 s, and a decay's memory
1 / (1 - decay) scales the same way (decay^(1 / scale)), so one value is one share of the run on every track (the 1-hour
scout file's idea 3). At the 10-minute track nothing changes.
"""

from __future__ import annotations

import contextlib
import torch

from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms.strategies.strategy import StepState, Strategy, StrategyConfig


LAWA_SNAPSHOTS = 8
DRIFT_SNAPSHOTS = 4


@dataclass(frozen=True)
class EmaConfig(StrategyConfig):
    variant: str = "none"
    start: float = 0.6  # share of the clock from which the averages run
    decay: float = 0.995  # per step; about 200 steps of memory
    window: int = 400  # latest_k and drift_leap: steps the snapshots span
    leap: float = 0.2  # drift_leap: the move back toward the earlier weights
    decay_fast: float = 0.95  # two_decay: the second, shorter average
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "tail", "latest_k", "drift_leap", "two_decay", "plateau_restart")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"start": (0.3, 0.95), "decay": (0.9, 0.999), "window": (100, 1000),
                                                        "leap": (0.05, 0.5), "decay_fast": (0.9, 0.99)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.window, int):
            raise ValueError(f"window must be an integer, not {self.window!r}")


def scaled_decay(decay: float, run_scale: float) -> float:
    """A per-step decay whose memory, 1 / (1 - decay) steps at the 10-minute track, scales with the run: decay^(1 / scale)."""
    return decay if run_scale == 1.0 else decay ** (1 / run_scale)


class Ema(Strategy):
    family = "ema"
    Config = EmaConfig
    config: EmaConfig

    def __init__(self, config: EmaConfig) -> None:
        super().__init__(config)
        self.averaged: list[Tensor] | None = None  # tail and two_decay: one tensor per parameter, from `start` on
        self.averaged_fast: list[Tensor] | None = None  # two_decay: the shorter average
        self.snapshots: list[list[Tensor]] = []  # latest_k, drift_leap and plateau_restart: the ring, oldest first
        self.restarted = False  # plateau_restart: the weights were replaced by the plateau mean

    @property
    def evaluation_options(self) -> int:
        return 2 if self.variant == "two_decay" and self.averaged is not None else 1

    def overrides(self, hook: str) -> bool:
        if self.variant == "plateau_restart":
            return hook == "after_step"  # validation scores the live weights
        return super().overrides(hook)

    def after_step(self, state: StepState) -> None:
        live = [parameter.detach() for parameter in state.model.parameters()]
        if self.variant in ("latest_k", "drift_leap"):
            self._record_snapshot(live, state, LAWA_SNAPSHOTS if self.variant == "latest_k" else DRIFT_SNAPSHOTS)
            return
        if self.variant == "plateau_restart":
            self._plateau_restart(live, state)
            return
        if state.progress < self.config.start:
            return
        if self.averaged is None:
            self.averaged = [parameter.clone() for parameter in live]
            self.averaged_fast = [parameter.clone() for parameter in live] if self.variant == "two_decay" else None
            return
        torch._foreach_lerp_(self.averaged, live, 1 - scaled_decay(self.config.decay, state.run_scale))
        if self.averaged_fast is not None:
            torch._foreach_lerp_(self.averaged_fast, live, 1 - scaled_decay(self.config.decay_fast, state.run_scale))

    def _record_snapshot(self, live: list[Tensor], state: StepState, count: int) -> None:
        """A ring of `count` snapshots, one every `window` / `count` steps (at the run's scale), reusing the oldest's
        memory once full."""
        step = state.step
        if step == 0:
            self.snapshots.clear()
        if step % max(1, round(self.config.window * state.run_scale) // count) != 0:
            return
        if len(self.snapshots) == count:
            oldest = self.snapshots.pop(0)
            torch._foreach_copy_(oldest, live)
            self.snapshots.append(oldest)
        else:
            self.snapshots.append([parameter.clone() for parameter in live])

    def _plateau_restart(self, live: list[Tensor], state: StepState) -> None:
        """Snapshots until `start`, then the live weights become their mean, once."""
        if state.step == 0:
            self.restarted = False
        if self.restarted:
            return
        if state.progress >= self.config.start and self.snapshots:
            torch._foreach_copy_(live, self._mean())
            self.snapshots.clear()  # the ring's memory goes back
            self.restarted = True
            return
        self._record_snapshot(live, state, LAWA_SNAPSHOTS)

    def _mean(self) -> list[Tensor]:
        """The uniform mean of the ring's snapshots."""
        mean = [parameter.clone() for parameter in self.snapshots[0]]
        for snapshot in self.snapshots[1:]:
            torch._foreach_add_(mean, snapshot)
        torch._foreach_div_(mean, len(self.snapshots))
        return mean

    def _scored(self, live: list[Tensor]) -> list[Tensor] | None:
        """The weights a validation pass scores, or None to score the live weights."""
        if self.variant == "tail":
            return self.averaged
        if self.variant == "two_decay":
            return self.averaged if self.option == 0 else self.averaged_fast
        if not self.snapshots:
            return None
        if self.variant == "drift_leap":
            earlier = self.snapshots[0]
            leaped = [parameter.clone() for parameter in live]
            torch._foreach_lerp_(leaped, earlier, self.config.leap)  # w + leap (w_earlier - w)
            return leaped
        return self._mean()

    @contextlib.contextmanager
    def evaluation_weights(self, model: nn.Module) -> Iterator[None]:
        live = [parameter.detach() for parameter in model.parameters()]
        scored = self._scored(live)
        if scored is None:
            yield
            return
        saved = [parameter.clone() for parameter in live]
        torch._foreach_copy_(live, scored)
        try:
            yield
        finally:
            torch._foreach_copy_(live, saved)
