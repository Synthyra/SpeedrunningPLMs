"""The grouping family: an auxiliary loss over physico-chemical residue classes, early in the run.

`physchem` (`ideas/2026-10-30_protein_input_priors.md` idea 3; no source tests an output-side group loss in MLM, and
the nearest result, Ieremie et al., Bioinformatics 2024, found reduced input alphabets lose to the full one): a second
head, 768 to the GROUPS six classes (aliphatic AVILM, aromatic FWY, positive KRH, negative DE, polar STNQ, special
GPC), reads the head's input at every masked residue, and its summed cross entropy joins the training loss at a weight
that falls linearly from `weight` at the start to 0 at `until` of the clock (a 0-d buffer the step callback refills,
so one compiled graph serves every phase). The target is a coarse marginal of the 20-way one, so at the optimum it
adds nothing: any gain is an early curriculum on the output side. The head is built from zeros (uniform group
predictions at step 0, no generator draw) and trains with AdamW at the head rate. Auxiliary losses run on the flex
training path only, as every auxiliary loss here does.

Symbols: b windows, l tokens, d = 768, g = 6 groups.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from foundry.layers.cross_entropy import cross_entropy
from speedrunning_plms import benchmark
from speedrunning_plms.benchmark.tokens import STANDARD_RESIDUES
from speedrunning_plms.strategies.strategy import ForwardContext, ParameterGroup, StepState, Strategy, StrategyConfig


GROUPS = ("AVILM", "FWY", "KRH", "DE", "STNQ", "GPC")
GROUP_OF_CLASS = torch.tensor([next(group for group, members in enumerate(GROUPS) if residue in members)
                               for residue in STANDARD_RESIDUES])  # (20,), each residue class's group


@dataclass(frozen=True)
class GroupingConfig(StrategyConfig):
    variant: str = "none"
    weight: float = 0.1  # the auxiliary weight at the start of the clock
    until: float = 0.5  # the share of the clock where the weight reaches 0
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "physchem")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"weight": (0.02, 0.5), "until": (0.1, 0.8)}

    def weight_at(self, progress: float) -> float:
        """The auxiliary weight at this share of the clock."""
        return self.weight * max(0.0, 1 - progress / self.until)


def group_labels(labels: Tensor) -> Tensor:
    """Each masked residue's group, IGNORE_LABEL where unscored."""
    # labels: (b, l) residue classes or IGNORE_LABEL; returns (b, l).
    scored = labels != benchmark.IGNORE_LABEL  # (b, l)
    groups = GROUP_OF_CLASS.to(labels.device)[labels.clamp(min=0)]  # (b, l)
    return groups.masked_fill(~scored, benchmark.IGNORE_LABEL)  # (b, l)


class Grouping(Strategy):
    family = "grouping"
    Config = GroupingConfig
    config: GroupingConfig

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.active:
            self.holder = nn.Module()
            self.holder.head = nn.Parameter(torch.zeros(len(GROUPS), benchmark.HIDDEN))  # (g, d)
            self.holder.register_buffer("weight", torch.zeros(()), persistent=False)  # (), refilled each step
            model.extensions[self.family] = self.holder  # type: ignore[index]

    def initialize(self, model: nn.Module) -> None:
        if self.active:
            self.holder.weight.fill_(self.config.weight_at(0.0))

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if not self.active:
            return []
        return [ParameterGroup("group_head", [self.holder.head], "adam", model.strategies.optimizer.config.head_lr)]  # type: ignore[union-attr]

    def auxiliary_loss(self, model: nn.Module, states: list[Tensor], labels: Tensor, context: ForwardContext) -> Tensor:
        # states: (b, l, d) each, 37 of them; labels: (b, l); returns ().
        features = model.head_features(states[-1])  # type: ignore[operator]  # (b, l, d)
        logits = torch.nn.functional.linear(features, self.holder.head.to(features.dtype)).float()  # (b, l, g)
        summed = cross_entropy(logits.flatten(0, 1), group_labels(labels).flatten(), ignore_index=benchmark.IGNORE_LABEL,
                               reduction="sum")  # ()
        return self.holder.weight * summed  # ()

    def after_step(self, state: StepState) -> None:
        self.holder.weight.fill_(self.config.weight_at(state.progress))
