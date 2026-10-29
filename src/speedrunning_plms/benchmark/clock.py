"""The training clock: ten minutes of applied steps, with compile warmup and in-run evaluations outside it but capped.

The clock counts time from the start of the first training step. A candidate checks `expired()` after computing a
step's gradient and before applying it: once the budget is spent the step is discarded, so no update buys time past
the deadline. Warmup (`warming_up()`) and evaluations (`paused()`) stop the clock and are timed against their caps; a
cap broken marks the run invalid. Device synchronization makes every reading mean work finished.
"""

from __future__ import annotations

import time
import torch

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


BUDGET_SECONDS = 600.0
WARMUP_CAP_SECONDS = 180.0
EVAL_CAP_SECONDS = 60.0


class Clock:
    def __init__(
        self, device: torch.device, *, budget: float = BUDGET_SECONDS, warmup_cap: float = WARMUP_CAP_SECONDS,
        eval_cap: float = EVAL_CAP_SECONDS, now: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        self.device = device
        self.budget = budget
        self.warmup_cap = warmup_cap
        self.eval_cap = eval_cap
        self._now = now
        self._synchronize = synchronize if synchronize is not None else self._device_synchronize
        self.train_seconds = 0.0  # applied training time, the clock
        self.steps = 0
        self.discarded_steps = 0
        self.discarded_seconds = 0.0
        self.warmup_seconds = 0.0
        self.eval_seconds = 0.0
        self.eval_count = 0
        self._step_start: float | None = None
        self._paused_at: float | None = None

    def _device_synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def warming_up(self) -> Iterator[None]:
        """Compile and warm up on synthetic batches; the candidate restores its fresh state afterwards."""
        self._synchronize()
        start = self._now()
        yield
        self._synchronize()
        self.warmup_seconds += self._now() - start

    def start_step(self) -> None:
        """The first call starts the clock; each call marks a step's start after any pause."""
        if self._step_start is None:
            self._synchronize()
            self._step_start = self._now()

    def expired(self) -> bool:
        """True once the budget is spent: the caller discards the step it was about to apply."""
        self._synchronize()
        elapsed = self.train_seconds + (self._now() - self._step_start if self._step_start is not None else 0.0)
        if elapsed >= self.budget:
            self.discarded_steps += 1
            if self._step_start is not None:
                self.discarded_seconds += self._now() - self._step_start  # the overrun is wasted, not training
                self._step_start = None
            return True
        return False

    def applied(self) -> None:
        """The step's update was applied; its time counts."""
        self._synchronize()
        self.steps += 1
        self._close_step()

    def _close_step(self) -> None:
        if self._step_start is not None:
            self.train_seconds += self._now() - self._step_start
            self._step_start = None

    @property
    def progress(self) -> float:
        """The share of the budget spent, read without synchronization for schedules."""
        running = self._now() - self._step_start if self._step_start is not None else 0.0
        return min(1.0, (self.train_seconds + running) / self.budget)

    @contextmanager
    def paused(self) -> Iterator[None]:
        """An in-run evaluation: the clock stops, the evaluation's time counts against its cap."""
        self._synchronize()
        self._close_step()
        start = self._now()
        yield
        self._synchronize()
        self.eval_seconds += self._now() - start
        self.eval_count += 1
        self.start_step()

    @property
    def warmup_valid(self) -> bool:
        return self.warmup_seconds <= self.warmup_cap

    @property
    def eval_valid(self) -> bool:
        return self.eval_seconds <= self.eval_cap

    @property
    def valid(self) -> bool:
        return self.warmup_valid and self.eval_valid and self.train_seconds <= self.budget * 1.01

    def receipt(self) -> dict[str, Any]:
        return {
            "budget_seconds": self.budget, "train_seconds": self.train_seconds, "steps": self.steps,
            "discarded_steps": self.discarded_steps, "discarded_seconds": self.discarded_seconds,
            "warmup_seconds": self.warmup_seconds,
            "warmup_cap_seconds": self.warmup_cap, "eval_seconds": self.eval_seconds, "eval_count": self.eval_count,
            "eval_cap_seconds": self.eval_cap, "warmup_valid": self.warmup_valid, "eval_valid": self.eval_valid,
            "valid": self.valid,
        }
