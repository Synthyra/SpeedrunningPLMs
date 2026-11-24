"""The protein-context family: a direct path for what a whole protein says (round 9, no run yet).

The record's gap to ESM2 8M is largest on long proteins, and the count measurement (`2026-11-03_kmer_hint_measure`) puts a protein's visible
composition alone at 2.8526 nats against the unigram's 2.8915, before any training.

- `composition`: at the embedding, every token gains a learned map of its protein's visible residue shares, centered on the training
  stream's: X + (f - p) W, W from zero, so step 0 is the record's; AdamW at the embedding rate. MASK is not a residue, so a masked residue
  never counts itself. In training a protein the window cuts counts its visible part; the evaluator's proteins are whole.
- `global_mean`: after the MLP of every `every`-th block, each token gains a learned map of its protein's mean normalized state (ProteinBERT's
  global track, Brandes et al. 2022, as a squeeze-and-excitation path; no loss number applies here), per site from zero, AdamW at the scalar
  rate. Attention can read a protein's CLS, so a tie is a live outcome; this shortcut does not have to be learned through it.
- `both`: the two together.

Symbols: b windows or proteins, l tokens, d = 768, c = 20 residues, s sites.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.residue_priors import unigram
from speedrunning_plms.strategies.strategy import CandidateModel, ForwardContext, ParameterGroup, Strategy, StrategyConfig, norm


@dataclass(frozen=True)
class ProteinContextConfig(StrategyConfig):
    variant: str = "none"
    every: int = 6  # the blocks whose index + 1 is a multiple of this gain the protein's mean state
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "composition", "global_mean", "both")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"every": (3, 18)}
    INTEGERS: ClassVar[tuple[str, ...]] = ("every",)


def protein_sums(values: Tensor, input_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Each token's protein's summed values over its non-padding tokens, and that count. A fixed table of l + 1 rows, so
    one graph serves every number of proteins in a window."""
    # values: (b, l, k); input_ids: (b, l); returns (b, l, k), (b, l, 1).
    b, l, k = values.shape
    document = (input_ids == benchmark.CLS_ID).cumsum(dim=1)[..., None]  # (b, l, 1) in [0, l]
    real = (input_ids != benchmark.PAD_ID)[..., None].to(values.dtype)  # (b, l, 1)
    sums = torch.zeros(b, l + 1, k, device=values.device, dtype=values.dtype).scatter_add(1, document.expand(-1, -1, k), values * real)  # (b, l + 1, k)
    counts = torch.zeros(b, l + 1, 1, device=values.device, dtype=values.dtype).scatter_add(1, document, real)  # (b, l + 1, 1)
    return sums.gather(1, document.expand(-1, -1, k)), counts.gather(1, document)  # (b, l, k), (b, l, 1)


def visible_composition(input_ids: Tensor) -> Tensor:
    """Each token's protein's shares of the 20 residues over its visible residues; a protein with none reads as zeros."""
    # input_ids: (b, l); returns (b, l, c).
    columns = benchmark.ESM2_RESIDUE_COLUMNS.to(input_ids.device)  # (c,)
    residues = (input_ids[..., None] == columns).float()  # (b, l, c)
    counts = protein_sums(residues, input_ids)[0]  # (b, l, c)
    return counts / counts.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (b, l, c)


class ProteinContext(Strategy):
    family = "protein_context"
    Config = ProteinContextConfig
    config: ProteinContextConfig

    @property
    def uses_composition(self) -> bool:
        return self.variant in ("composition", "both")

    @property
    def uses_mean(self) -> bool:
        return self.variant in ("global_mean", "both")

    def overrides(self, hook: str) -> bool:
        return (hook == "embed" and self.uses_composition) or (hook == "updated_stream" and self.uses_mean)

    def build(self, model: CandidateModel, max_tokens: int) -> None:
        if self.variant == "none":
            return
        holder = nn.Module()
        if self.uses_composition:
            holder.composition = nn.Parameter(torch.zeros(benchmark.NUM_CLASSES, model.strategies.shape.config.hidden))  # (c, d)
            holder.register_buffer("expected", unigram(), persistent=False)  # (c,)
        if self.uses_mean:
            shape = model.strategies.shape.config
            self.sites = {index: site for site, index in enumerate(range(self.config.every - 1, shape.blocks, self.config.every))}
            holder.mean_maps = nn.Parameter(torch.zeros(len(self.sites), shape.hidden, shape.hidden))  # (s, d, d)
        model.extensions[self.family] = holder
        self.holder = holder

    def parameter_groups(self, model: CandidateModel) -> list[ParameterGroup]:
        if self.variant == "none":
            return []
        config = model.strategies.optimizer.config
        groups = []
        if self.uses_composition:
            groups.append(ParameterGroup("protein_composition", [self.holder.composition], "adam", config.embed_lr))
        if self.uses_mean:
            groups.append(ParameterGroup("protein_mean", [self.holder.mean_maps], "adam", config.scalar_lr))
        return groups

    def embed(self, model: CandidateModel, input_ids: Tensor, X: Tensor) -> Tensor:
        # input_ids: (b, l); X: (b, l, d).
        shares = visible_composition(input_ids) - self.holder.expected  # (b, l, c)
        return X + (shares.to(X.dtype) @ self.holder.composition.to(X.dtype))  # (b, l, d)

    def updated_stream(self, index: int, sublayer: str, X: Tensor, context: ForwardContext) -> Tensor:
        # X: (b, l, d), the stream after the update.
        if sublayer != "mlp" or index not in self.sites:
            return X  # (b, l, d)
        sums, counts = protein_sums(norm(X), context.input_ids)  # (b, l, d), (b, l, 1)
        return X + (sums / counts.clamp(min=1.0)) @ self.holder.mean_maps[self.sites[index]].to(X.dtype)  # (b, l, d)
