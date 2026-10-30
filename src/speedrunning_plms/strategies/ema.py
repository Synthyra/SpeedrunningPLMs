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

The rings empty at the first applied step, so the compile warmup's steps never enter them.
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
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "tail", "latest_k", "drift_leap", "two_decay")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"start": (0.3, 0.95), "decay": (0.9, 0.999), "window": (100, 1000),
                                                        "leap": (0.05, 0.5), "decay_fast": (0.9, 0.99)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.window, int):
            raise ValueError(f"window must be an integer, not {self.window!r}")


class Ema(Strategy):
    family = "ema"
    Config = EmaConfig
    config: EmaConfig

    def __init__(self, config: EmaConfig) -> None:
        super().__init__(config)
        self.averaged: list[Tensor] | None = None  # tail and two_decay: one tensor per parameter, from `start` on
        self.averaged_fast: list[Tensor] | None = None  # two_decay: the shorter average
        self.snapshots: list[list[Tensor]] = []  # latest_k and drift_leap: the ring, oldest first

    @property
    def evaluation_options(self) -> int:
        return 2 if self.variant == "two_decay" and self.averaged is not None else 1

    def after_step(self, state: StepState) -> None:
        live = [parameter.detach() for parameter in state.model.parameters()]
        if self.variant in ("latest_k", "drift_leap"):
            self._record_snapshot(live, state.step, LAWA_SNAPSHOTS if self.variant == "latest_k" else DRIFT_SNAPSHOTS)
            return
        if state.progress < self.config.start:
            return
        if self.averaged is None:
            self.averaged = [parameter.clone() for parameter in live]
            self.averaged_fast = [parameter.clone() for parameter in live] if self.variant == "two_decay" else None
            return
        torch._foreach_lerp_(self.averaged, live, 1 - self.config.decay)
        if self.averaged_fast is not None:
            torch._foreach_lerp_(self.averaged_fast, live, 1 - self.config.decay_fast)

    def _record_snapshot(self, live: list[Tensor], step: int, count: int) -> None:
        """A ring of `count` snapshots, one every `window` / `count` steps, reusing the oldest's memory once full."""
        if step == 0:
            self.snapshots.clear()
        if step % max(1, self.config.window // count) != 0:
            return
        if len(self.snapshots) == count:
            oldest = self.snapshots.pop(0)
            torch._foreach_copy_(oldest, live)
            self.snapshots.append(oldest)
        else:
            self.snapshots.append([parameter.clone() for parameter in live])

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
        mean = [parameter.clone() for parameter in self.snapshots[0]]
        for snapshot in self.snapshots[1:]:
            torch._foreach_add_(mean, snapshot)
        torch._foreach_div_(mean, len(self.snapshots))
        return mean

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
