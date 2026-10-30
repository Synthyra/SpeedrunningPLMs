"""The latent family: an auxiliary latent-prediction loss inside the one training forward.

- `next_layer`: same-pass latent prediction (NextLat, arXiv 2511.05963) with SIGReg against collapse (LeJEPA, arXiv
  2511.08544). One linear map reads the normalized stream after block `source_block` and predicts the stop-gradient
  normalized stream after the last block, at the masked positions; the loss is 1 - cos(prediction, target), a cosine
  because the stream's scale grows with depth and the head reads it through a norm anyway. SIGReg's Epps-Pulley
  statistic on SIGREG_SLICES random unit projections, drawn fresh each step, pulls the normalized source states at
  the masked positions toward an isotropic Gaussian (an RMS-normalized state has unit variance along any direction
  when isotropic). The term is `weight` x ((1 - SIGREG_SHARE) x the summed cosine loss + SIGREG_SHARE x the statistic,
  which is N times its per-sample value), so after the loop divides by the N masked residues it reads as a mean.
  SIGREG_SHARE is LeJEPA's lambda. No teacher and no second forward; the target is detached, so the loss trains the
  source blocks and the predictor only. The loop adds auxiliary losses only on the flex path, so a CPU smoke of
  `train.py` does not compute it; the registry test's CPU step does.
- `neighbors`: neighbor-prediction heads (`ideas/2026-10-30_capacity_at_fixed_shape.md` 1). At each masked residue,
  one small linear head per offset in NEIGHBOR_OFFSETS reads the normalized last state and predicts the residue at
  that offset, the cross entropy of each scored like the main loss over the 20 residues and summed, times `weight`.
  A neighbor is scored only where it is a visible standard residue in the window: CLS, EOS, PAD, MASK and positions
  past the window's edge are skipped, so no offset reaches another protein (EOS and CLS sit between). The targets
  are in the input, so the heads ask the last state to keep its neighbors' identities. The heads train with AdamW at
  the head's rate. Like `next_layer` it acts only where the loop adds auxiliary losses.

Symbols: b windows, l tokens, d = 768, m projections, k integration points, v = 33 tokens.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ForwardContext, ParameterGroup, Strategy, StrategyConfig, norm


SIGREG_SHARE = 0.05  # LeJEPA's default lambda between the prediction loss and SIGReg
SIGREG_SLICES = 64  # m; fewer than LeJEPA's 256 to keep the (b, l, m, k) phase tensor small, redrawn every step
SIGREG_POINTS = 17  # k, on [-5, 5], LeJEPA's integration grid
PREDICTOR_INIT_STD = 0.02  # the record's std for its matrices; a zero map has no direction for the cosine
NEIGHBOR_OFFSETS = (-2, -1, 1, 2)


@dataclass(frozen=True)
class LatentConfig(StrategyConfig):
    variant: str = "none"
    source_block: int = 18  # the predictor reads the stream after this many blocks
    weight: float = 0.1
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "next_layer", "neighbors")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"source_block": (6, 30), "weight": (0.01, 0.5)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.source_block, int):
            raise ValueError(f"source_block must be an integer, not {self.source_block!r}")


def epps_pulley(Z: Tensor, weights: Tensor) -> Tensor:
    """SIGReg: the Epps-Pulley statistic of each random 1-D projection of Z against N(0, 1), averaged over
    projections, as N times the weighted integral of the squared gap between the characteristic functions."""
    # Z: (b, l, d), float; weights: (b, l), 1 at the samples; returns ().
    directions = torch.randn(Z.shape[-1], SIGREG_SLICES, device=Z.device)  # (d, m)
    directions = directions / directions.norm(dim=0, keepdim=True)  # (d, m), unit columns
    projected = Z @ directions  # (b, l, m)
    t = torch.linspace(-5.0, 5.0, SIGREG_POINTS, device=Z.device)  # (k,)
    gaussian = torch.exp(-0.5 * t.square())  # (k,), N(0, 1)'s characteristic function, also the integration weight
    phase = projected.unsqueeze(-1) * t  # (b, l, m, k)
    count = weights.sum().clamp(min=1.0)  # ()
    sample_weights = weights[..., None, None]  # (b, l, 1, 1)
    real = (phase.cos() * sample_weights).sum(dim=(0, 1)) / count  # (m, k)
    imaginary = (phase.sin() * sample_weights).sum(dim=(0, 1)) / count  # (m, k)
    gap = ((real - gaussian).square() + imaginary.square()) * gaussian  # (m, k)
    spacing = 10.0 / (SIGREG_POINTS - 1)  # the grid's step on [-5, 5]
    integral = (gap[:, 1:] + gap[:, :-1]).sum(dim=-1) * spacing / 2  # (m,), the trapezoid rule
    return count * integral.mean()  # ()


class Latent(Strategy):
    family = "latent"
    Config = LatentConfig
    config: LatentConfig

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.variant == "next_layer":
            self.predictor = nn.Linear(benchmark.HIDDEN, benchmark.HIDDEN, bias=False)
            model.extensions[self.family] = self.predictor  # type: ignore[index]
        elif self.variant == "neighbors":
            self.heads = nn.ModuleList(nn.Linear(benchmark.HIDDEN, benchmark.VOCAB_SIZE, bias=False) for _ in NEIGHBOR_OFFSETS)
            model.extensions[self.family] = self.heads  # type: ignore[index]

    def initialize(self, model: nn.Module) -> None:
        if self.variant == "next_layer":
            nn.init.normal_(self.predictor.weight, std=PREDICTOR_INIT_STD)  # (d, d)
        elif self.variant == "neighbors":
            for head in self.heads:
                nn.init.normal_(head.weight, std=PREDICTOR_INIT_STD)  # (v, d), the record's head init

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        config = model.strategies.optimizer.config  # type: ignore[union-attr]
        if self.variant == "next_layer":
            # AdamW, not Muon: the loop adds auxiliary losses only on the flex path, so on the CPU the predictor has no
            # gradient, which AdamW skips and Muon's bank of 768 x 768 matrices cannot hold.
            return [ParameterGroup("latent_predictor", [self.predictor.weight], "adam", config.scalar_lr)]
        if self.variant == "neighbors":
            return [ParameterGroup("neighbor_heads", list(self.heads.parameters()), "adam", config.head_lr)]
        return []

    def auxiliary_loss(self, model: nn.Module, states: list[Tensor], labels: Tensor,
                       context: ForwardContext) -> Tensor:
        # states: (b, l, d) each, 37 of them; labels: (b, l); returns ().
        if self.variant == "neighbors":
            return self.config.weight * self._neighbor_loss(states[-1], labels, context.input_ids)  # ()
        weights = (labels != benchmark.IGNORE_LABEL).float()  # (b, l), 1 at the masked residues
        source = norm(states[self.config.source_block]).float()  # (b, l, d)
        prediction = norm(self.predictor(source).float())  # (b, l, d)
        target = norm(states[-1].detach().float())  # (b, l, d)
        cosine = (prediction * target).mean(dim=-1)  # (b, l); unit-RMS vectors, so the mean product is the cosine
        latent = ((1.0 - cosine) * weights).sum()  # ()
        regularizer = epps_pulley(source, weights)  # ()
        return self.config.weight * ((1.0 - SIGREG_SHARE) * latent + SIGREG_SHARE * regularizer)  # ()

    def _neighbor_loss(self, final: Tensor, labels: Tensor, input_ids: Tensor) -> Tensor:
        """The summed cross entropy of every offset's head at the masked residues whose neighbor there is visible."""
        # final: (b, l, d), the last state; labels, input_ids: (b, l), input_ids corrupted; returns ().
        hidden = norm(final)  # (b, l, d)
        masked = labels != benchmark.IGNORE_LABEL  # (b, l)
        total = hidden.new_zeros((), dtype=torch.float32)  # ()
        for offset, head in zip(NEIGHBOR_OFFSETS, self.heads, strict=True):
            neighbors = torch.full_like(input_ids, benchmark.PAD_ID)  # (b, l); past the window's edge reads as PAD
            if offset > 0:
                neighbors[:, :-offset] = input_ids[:, offset:]
            else:
                neighbors[:, -offset:] = input_ids[:, :offset]
            targets = benchmark.residue_labels(neighbors).masked_fill(~masked, benchmark.IGNORE_LABEL)  # (b, l); MASK is no residue
            total = total + benchmark.masked_cross_entropy(head(hidden), targets)  # ()
        return total  # ()
