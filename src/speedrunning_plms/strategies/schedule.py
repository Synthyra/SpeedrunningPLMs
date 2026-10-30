"""The schedule family: the learning-rate multiplier and Muon's momentum, both by share of the clock.

`trapezoid` (the record): linear warmup over `warmup_fraction`, a plateau, a linear cooldown over the last
`cooldown_fraction` to `final_lr_ratio`; Muon's momentum rises linearly from `momentum_start` to `momentum` over
`momentum_warmup` (019 tried a cooldown of 0.6, 023 a flat momentum).
`one_sqrt`: the same with the 1-sqrt cooldown (Hagele et al. 2024, arXiv 2405.18392): over the cooldown the multiplier
falls as 1 - sqrt(share of the cooldown spent), to `final_lr_ratio`. A schedule-free optimizer (`optimizer.amuse`)
turns the cooldown off: the loop asks for the multiplier without it, so the plateau runs to the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from speedrunning_plms.strategies.strategy import Strategy, StrategyConfig


@dataclass(frozen=True)
class ScheduleConfig(StrategyConfig):
    variant: str = "trapezoid"
    warmup_fraction: float = 0.02
    cooldown_fraction: float = 0.4
    final_lr_ratio: float = 0.0
    momentum: float = 0.95
    momentum_start: float = 0.85
    momentum_warmup: float = 0.15  # share of the budget
    VARIANTS: ClassVar[tuple[str, ...]] = ("trapezoid", "one_sqrt")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "warmup_fraction": (0.0, 0.1), "cooldown_fraction": (0.1, 0.8), "final_lr_ratio": (0.0, 0.3),
        "momentum": (0.85, 0.98), "momentum_start": (0.7, 0.98), "momentum_warmup": (0.01, 0.4),
    }


class Schedule(Strategy):
    family = "schedule"
    Config = ScheduleConfig
    config: ScheduleConfig

    def lr_scale(self, progress: float, cooldown: bool = True) -> float:
        """Linear warmup, a constant plateau, then the cooldown (linear, or 1-sqrt) unless `cooldown` is off."""
        config = self.config
        if progress < config.warmup_fraction:
            return progress / config.warmup_fraction
        if cooldown and progress > 1 - config.cooldown_fraction:
            phase = (progress - (1 - config.cooldown_fraction)) / config.cooldown_fraction
            if self.variant == "one_sqrt":
                phase = phase**0.5
            return 1 - (1 - config.final_lr_ratio) * phase
        return 1.0

    def momentum(self, progress: float) -> float:
        """Muon's momentum, rising from the start value over the momentum warmup."""
        config = self.config
        phase = min(1.0, progress / config.momentum_warmup)
        return config.momentum_start + phase * (config.momentum - config.momentum_start)
