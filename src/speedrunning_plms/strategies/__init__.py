"""The strategy registry and the genome: every family a candidate is built from, one choice each.

A genome is a JSON document `{"families": {"<family>": {"variant": "...", "<parameter>": value}}}`. A family absent from
it, or a field absent from a family, takes the default, and the defaults together are 022: local-bias
attention, NorMuon, the tanh softcap, table rates 3e-3 and 1e-3. `resolve` checks a document against every family's
config and every family's cross-family conflicts (`genome_conflicts`) and returns the complete genome; `Strategies` holds
one strategy per family, in registry order, which is the order their hooks run. To add a family, write one module with
a config and a strategy and list it in `FAMILIES`.
"""

from __future__ import annotations

import hashlib
import json

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from speedrunning_plms.strategies.activation import Activation
from speedrunning_plms.strategies.attention import AttentionKernel
from speedrunning_plms.strategies.backprop_free import BackpropFree
from speedrunning_plms.strategies.batch import Batch
from speedrunning_plms.strategies.canon import Canon
from speedrunning_plms.strategies.curriculum import Curriculum
from speedrunning_plms.strategies.deep_supervision import DeepSupervision
from speedrunning_plms.strategies.ema import Ema
from speedrunning_plms.strategies.embedding_mix import EmbeddingMix
from speedrunning_plms.strategies.gates import Gates
from speedrunning_plms.strategies.grouping import Grouping
from speedrunning_plms.strategies.head import Head
from speedrunning_plms.strategies.kernel import Kernel
from speedrunning_plms.strategies.latent import Latent
from speedrunning_plms.strategies.logit_scale import LogitScale
from speedrunning_plms.strategies.lookahead import Lookahead
from speedrunning_plms.strategies.loop import Loop
from speedrunning_plms.strategies.loss import Loss
from speedrunning_plms.strategies.memory import Memory
from speedrunning_plms.strategies.neighbor import Neighbor
from speedrunning_plms.strategies.optimizer import OptimizerChoice
from speedrunning_plms.strategies.positional import Positional
from speedrunning_plms.strategies.protein_context import ProteinContext
from speedrunning_plms.strategies.regularization import Regularization
from speedrunning_plms.strategies.residual import Residual
from speedrunning_plms.strategies.routing import Routing
from speedrunning_plms.strategies.sam import Sam
from speedrunning_plms.strategies.schedule import Schedule
from speedrunning_plms.strategies.sglr import Sglr
from speedrunning_plms.strategies.shape import Shape
from speedrunning_plms.strategies.softcap import Softcap
from speedrunning_plms.strategies.strategy import HOOKS, Strategy, StrategyConfig, config_from
from speedrunning_plms.strategies.stream import StreamMixer
from speedrunning_plms.strategies.unet import UNet
from speedrunning_plms.strategies.value_embeddings import ValueEmbeddings
from speedrunning_plms.strategies.value_residual import ValueResidual


GENOME_SCHEMA = 1
# Hook order: the embedding's table before nGPT normalizes it, the value tables before the value mix, nGPT's logit
# scale before the softcap, nGPT's weight normalization before the lookahead's pull, the pull before the average reads the
# weights, SAM's gradient before the backprop-free variants edit gradients. The activation and the latent loss act alone at their
# points, and the logit scale runs before the value residual's query mix. The loop and sglr tie block matrices in
# `initialize`, so they come before the backprop-free family, whose `unit_route` holds the MLP weights it finds there, and
# their stream boundaries act after the families that mix the stream (canon, neighbor, protein context); routing reads the
# loop's tying. The shape has no hooks; it comes first because every other family's `build` reads it.
FAMILIES: dict[str, type[Strategy]] = {strategy.family: strategy for strategy in (
    Shape, Positional, AttentionKernel, Kernel, LogitScale, Neighbor, ValueEmbeddings, ValueResidual, Gates, UNet, EmbeddingMix,
    Memory, ProteinContext, Canon, Residual, StreamMixer, Activation, Regularization, Head, Softcap, Loss, Sam, OptimizerChoice,
    Schedule, Batch, Curriculum, Lookahead, Ema, Routing, Loop, Sglr, BackpropFree, Latent, Grouping, DeepSupervision,
)}

Genome = dict[str, dict[str, Any]]  # family -> its config's fields


class GenomeConflict(ValueError):
    """Families whose configs are each valid but cannot be built together; the search redraws or skips such a genome."""

    def __init__(self, conflicts: list[tuple[str, str]]) -> None:
        super().__init__("; ".join(f"{family}: {reason}" for family, reason in conflicts))
        self.families = sorted({family for family, _ in conflicts})


def resolve(document: dict[str, Any]) -> Genome:
    """The complete genome a document describes; raises on an unknown family, field, variant or out-of-range value, and
    with `GenomeConflict` on families that do not compose."""
    families = document.get("families", {})
    unknown_keys = sorted(set(document) - {"families", "schema"})
    if unknown_keys:
        raise ValueError(f"a genome holds `families` (and `schema`), not {unknown_keys}")
    if document.get("schema", GENOME_SCHEMA) != GENOME_SCHEMA:
        raise ValueError(f"genome schema {document['schema']} is not {GENOME_SCHEMA}")
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown families {unknown}; the registry has {list(FAMILIES)}")
    complete = {family: config_from(strategy.Config, families.get(family, {})).as_document()
                for family, strategy in FAMILIES.items()}
    conflicts = genome_conflicts(complete)
    if conflicts:
        raise GenomeConflict(conflicts)
    return complete


def genome_conflicts(complete: Genome) -> list[tuple[str, str]]:
    """Every family's conflicts with the rest of a complete genome, as (family, reason); empty when they compose."""
    strategies = Strategies.unchecked(complete)
    return [(strategy.family, reason) for strategy in strategies for reason in strategy.conflicts(strategies)]


def default_genome() -> Genome:
    """The record's genome: every family at its default."""
    return resolve({})


def genome_document(genome: Genome) -> dict[str, Any]:
    """The JSON document of a genome, with only the fields that differ from the record, for files and the ledger."""
    record = default_genome()
    families = {family: {key: value for key, value in fields.items() if value != record[family][key]}
                for family, fields in resolve({"families": genome}).items()}
    return {"schema": GENOME_SCHEMA, "families": {family: fields for family, fields in families.items() if fields}}


def genome_sha256(genome: Genome) -> str:
    """The digest of a genome's complete canonical JSON."""
    return hashlib.sha256(json.dumps(resolve({"families": genome}), sort_keys=True).encode("utf-8")).hexdigest()


def load_genome(path: Path) -> Genome:
    return resolve(json.loads(path.read_text(encoding="utf-8")))


def save_genome(genome: Genome, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(genome_document(genome), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sources_sha256(directory: Path | None = None) -> str:
    """The digest of every strategy module's source, in file-name order: a genome means what this code makes of it.
    `directory` reads a run's `strategies/` snapshot instead of these modules."""
    digest = hashlib.sha256()
    for path in sorted((directory or Path(__file__).parent).glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Strategies:
    """One strategy per family, built from a genome; families are attributes (`strategies.attention`)."""

    # Each family's attribute, set from FAMILIES in `__init__` and typed here so a strategy reads another family's
    # config by name; a new family adds its line (a test checks the two agree).
    shape: Shape
    positional: Positional
    attention: AttentionKernel
    kernel: Kernel
    logit_scale: LogitScale
    neighbor: Neighbor
    value_embeddings: ValueEmbeddings
    value_residual: ValueResidual
    gates: Gates
    unet: UNet
    embedding_mix: EmbeddingMix
    memory: Memory
    protein_context: ProteinContext
    canon: Canon
    residual: Residual
    stream: StreamMixer
    activation: Activation
    regularization: Regularization
    head: Head
    softcap: Softcap
    loss: Loss
    sam: Sam
    optimizer: OptimizerChoice
    schedule: Schedule
    batch: Batch
    curriculum: Curriculum
    lookahead: Lookahead
    ema: Ema
    routing: Routing
    loop: Loop
    sglr: Sglr
    backprop_free: BackpropFree
    latent: Latent
    grouping: Grouping
    deep_supervision: DeepSupervision

    def __init__(self, genome: Genome) -> None:
        self._assemble(resolve({"families": genome}))

    @classmethod
    def unchecked(cls, complete: Genome) -> Strategies:
        """The strategies of a complete genome without the cross-family check: what `genome_conflicts` reads."""
        strategies = cls.__new__(cls)
        strategies._assemble(complete)
        return strategies

    def _assemble(self, complete: Genome) -> None:
        self.genome = complete
        self._ordered: list[Strategy] = []
        for family, strategy_type in FAMILIES.items():
            strategy = strategy_type(config_from(strategy_type.Config, complete[family]))
            setattr(self, family, strategy)
            self._ordered.append(strategy)
        self._hooked = {hook: [strategy for strategy in self._ordered if strategy.overrides(hook)] for hook in HOOKS}

    def __iter__(self) -> Iterator[Strategy]:
        return iter(self._ordered)

    def hooked(self, hook: str) -> list[Strategy]:
        """The strategies whose `hook` runs, in registry order."""
        return self._hooked[hook]

    def config(self, family: str) -> StrategyConfig:
        return getattr(self, family).config


__all__ = ["FAMILIES", "GENOME_SCHEMA", "Genome", "GenomeConflict", "Strategies", "default_genome", "genome_conflicts",
           "genome_document", "genome_sha256", "load_genome", "resolve", "save_genome", "sources_sha256"]
