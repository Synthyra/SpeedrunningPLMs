"""The deep_supervision family: the record's own head on intermediate blocks, early in the run.

`mid_heads` (Al-Rfou et al., arXiv 1808.04444, decaying intermediate losses, search summary; LayerSkip, arXiv
2404.16710, an early-exit loss that needs a curriculum; `ideas/2026-10-31_capacity_and_data_at_one_hour.md` idea 2):
the streams after blocks READS (18 and 27, 0-based) pass through the head as the final state does (its norm, its
head-input hooks, the head, its logit hooks), and their summed masked cross entropy over the 20 residue columns joins
the training loss at a weight falling linearly from `weight` at the start to 0 at `until` of the clock (a 0-d buffer
the step callback refills, so one compiled graph serves every phase). No parameters: the head is shared, and only the
final state is read at evaluation. No source trains intermediate MLM heads on a bidirectional encoder for pretraining.
Auxiliary losses run on the flex training path only.

Symbols: b windows, l tokens, d = 768, v = 33 tokens.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ForwardContext, StepState, Strategy, StrategyConfig


READS = (18, 27)  # the blocks whose outputs the head reads in training


@dataclass(frozen=True)
class DeepSupervisionConfig(StrategyConfig):
    variant: str = "none"
    weight: float = 0.1  # the intermediate losses' weight at the start of the clock
    until: float = 0.5  # the share of the clock where the weight reaches 0
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "mid_heads")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"weight": (0.02, 0.5), "until": (0.1, 0.9)}

    def weight_at(self, progress: float) -> float:
        """The intermediate losses' weight at this share of the clock."""
        return self.weight * max(0.0, 1 - progress / self.until)


class DeepSupervision(Strategy):
    family = "deep_supervision"
    Config = DeepSupervisionConfig
    config: DeepSupervisionConfig

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.active:
            self.holder = nn.Module()
            self.holder.register_buffer("weight", torch.zeros(()), persistent=False)  # (), refilled each step
            model.extensions[self.family] = self.holder  # type: ignore[index]

    def initialize(self, model: nn.Module) -> None:
        if self.active:
            self.holder.weight.fill_(self.config.weight_at(0.0))

    def auxiliary_loss(self, model: nn.Module, states: list[Tensor], labels: Tensor, context: ForwardContext) -> Tensor:
        # states: (b, l, d) each, the embedding then every block's output; labels: (b, l); returns ().
        summed = sum(benchmark.masked_cross_entropy(model.logits(states[block + 1]), labels) for block in READS)  # type: ignore[operator]
        return self.holder.weight * summed  # ()

    def after_step(self, state: StepState) -> None:
        self.holder.weight.fill_(self.config.weight_at(state.progress))
