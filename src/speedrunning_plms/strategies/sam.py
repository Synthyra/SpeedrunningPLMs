"""The sam family: sharpness-aware gradients late in the run (`2026-10-30_regularization_and_averaging.md` idea 4).

`late`: SAM (Foret et al., arXiv 2010.01412) on every `every`-th step from `start` of the clock. After the backward,
every trainable weight moves up its gradient by rho g / ||g|| (one global norm over all of them), the loop's own
forward and backward run again on the same batch and masks (`StepState.recompute_gradient`, the same compiled graph,
so no compile), the weights move back, and the optimizers step with the gradient taken at the perturbed point. The
logged loss stays the first pass's. Cost: one extra forward and backward per `every` steps of the late phase, about
10 percent of the clock at the defaults (every 4th step of the last 40 percent). The only evidence is on forgetting
(arXiv 2605.02105, abstract: a short SAM phase cut forgetting after post-training by 31 percent at 20M to 1B), none on
pretraining cross entropy; at 3,600 s the late phase holds six times the steps, the regime it aims at. The default
rho 0.05 is a vision value; the search sets it.

Symbols: none beyond the weights.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar

from speedrunning_plms.strategies.strategy import StepState, Strategy, StrategyConfig


@dataclass(frozen=True)
class SamConfig(StrategyConfig):
    variant: str = "none"
    rho: float = 0.05  # the perturbation's global norm
    every: int = 4  # one SAM gradient every this many steps
    start: float = 0.6  # the share of the clock from which SAM runs
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "late")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"rho": (0.005, 0.5), "every": (1, 16), "start": (0.0, 0.95)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.every, int):
            raise ValueError(f"every must be an integer, not {self.every!r}")


class Sam(Strategy):
    family = "sam"
    Config = SamConfig
    config: SamConfig

    def applies(self, state: StepState) -> bool:
        """True on the steps that take the SAM gradient."""
        return state.progress >= self.config.start and state.step % self.config.every == 0

    @torch.no_grad()
    def after_backward(self, state: StepState) -> None:
        if not self.applies(state):
            return
        assert state.recompute_gradient is not None, "SAM needs the loop's second forward and backward"
        weights = [parameter for parameter in state.model.parameters() if parameter.grad is not None]
        gradients = [parameter.grad for parameter in weights]
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients)))  # (), the global norm
        ascent = torch._foreach_mul(gradients, self.config.rho / (norm + 1e-12))  # rho g / ||g|| per weight
        ascent = [step.to(weight.dtype) for step, weight in zip(ascent, weights, strict=True)]
        torch._foreach_add_(weights, ascent)
        with torch.enable_grad():
            state.recompute_gradient()
        torch._foreach_sub_(weights, ascent)
