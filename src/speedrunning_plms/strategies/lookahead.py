"""The lookahead family: slow weights that pull the live weights back every few steps (round 9, no run yet).

`periodic` (Lookahead, arXiv 1907.08610; Prime Intellect's track 3 stack carries it, primeintellect.ai/auto-nanogpt, with no number of
its own): from the first step past `start` of the clock, every `every` steps the slow weights move `alpha` of the way to the live weights and
the live weights become the slow weights. Muon's and AdamW's momenta carry on, as in the paper. The validation pass scores the live weights.
About 0.8 GB more fp32 for the slow weights and a small lerp and copy every `every` steps, outside any graph, so no compile cost. Under
`optimizer.amuse` and `optimizer.muown` the Muon matrices are rewritten from the optimizer's own state each step, so the pull
acts only on the AdamW groups there. `every` stays in steps on both tracks: it is the optimizer's own cadence, not a share of the run.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor

from speedrunning_plms.strategies.strategy import StepState, Strategy, StrategyConfig


@dataclass(frozen=True)
class LookaheadConfig(StrategyConfig):
    variant: str = "none"
    every: int = 5  # steps between pulls; the paper's k
    alpha: float = 0.5  # the slow weights' move toward the live weights
    start: float = 0.0  # share of the clock before the slow weights exist
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "periodic")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"every": (2, 50), "alpha": (0.1, 0.9), "start": (0.0, 0.9)}
    INTEGERS: ClassVar[tuple[str, ...]] = ("every",)


class Lookahead(Strategy):
    family = "lookahead"
    Config = LookaheadConfig
    config: LookaheadConfig

    def __init__(self, config: LookaheadConfig) -> None:
        super().__init__(config)
        self.slow: list[Tensor] | None = None  # one tensor per trainable parameter, from `start` on
        self.since_pull = 0

    def after_step(self, state: StepState) -> None:
        # The warmup's steps run at progress 0, so they never start the slow weights.
        if state.progress <= self.config.start:
            return
        live = [parameter.detach() for parameter in state.model.parameters() if parameter.requires_grad]
        if self.slow is None:
            self.slow = [parameter.clone() for parameter in live]
            return
        self.since_pull += 1
        if self.since_pull == self.config.every:
            self.since_pull = 0
            torch._foreach_lerp_(self.slow, live, self.config.alpha)
            torch._foreach_copy_(live, self.slow)
