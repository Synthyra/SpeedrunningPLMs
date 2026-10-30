"""The strategy registry and the genome: every family a candidate is built from, one choice each.

A genome is a JSON document `{"families": {"<family>": {"variant": "...", "<parameter>": value}}}`. A family absent from
it, or a field absent from a family, takes the default, and the defaults together are 022: local-bias
attention, NorMuon, the tanh softcap, table rates 3e-3 and 1e-3. `resolve` checks a document against every family's
config and returns the complete genome; `Strategies` holds one strategy per family, in registry order, which is the
order their hooks run. To add a family, write one module with a config and a strategy and list it in `FAMILIES`.
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
from speedrunning_plms.strategies.curriculum import Curriculum
from speedrunning_plms.strategies.ema import Ema
from speedrunning_plms.strategies.embedding_mix import EmbeddingMix
from speedrunning_plms.strategies.gates import Gates
from speedrunning_plms.strategies.head import Head
from speedrunning_plms.strategies.kernel import Kernel
from speedrunning_plms.strategies.latent import Latent
from speedrunning_plms.strategies.logit_scale import LogitScale
from speedrunning_plms.strategies.neighbor import Neighbor
from speedrunning_plms.strategies.optimizer import OptimizerChoice
from speedrunning_plms.strategies.positional import Positional
from speedrunning_plms.strategies.residual import Residual
from speedrunning_plms.strategies.schedule import Schedule
from speedrunning_plms.strategies.softcap import Softcap
from speedrunning_plms.strategies.strategy import HOOKS, Strategy, StrategyConfig, config_from
from speedrunning_plms.strategies.stream import StreamMixer
from speedrunning_plms.strategies.unet import UNet
from speedrunning_plms.strategies.value_embeddings import ValueEmbeddings
from speedrunning_plms.strategies.value_residual import ValueResidual


GENOME_SCHEMA = 1
# Hook order: the embedding's table before nGPT normalizes it, the value tables before the value mix, nGPT's logit
# scale before the softcap, nGPT's weight normalization before the average reads the weights. The activation and the
# latent loss act alone at their points, and the logit scale is the only query hook.
FAMILIES: dict[str, type[Strategy]] = {strategy.family: strategy for strategy in (
    Positional, AttentionKernel, Kernel, LogitScale, Neighbor, ValueEmbeddings, ValueResidual, Gates, UNet, EmbeddingMix,
    Residual, StreamMixer, Activation, Head, Softcap, OptimizerChoice, Schedule, Batch, Curriculum, Ema, BackpropFree, Latent,
)}

Genome = dict[str, dict[str, Any]]  # family -> its config's fields


def resolve(document: dict[str, Any]) -> Genome:
    """The complete genome a document describes; raises on an unknown family, field, variant or out-of-range value."""
    families = document.get("families", {})
    unknown_keys = sorted(set(document) - {"families", "schema"})
    if unknown_keys:
        raise ValueError(f"a genome holds `families` (and `schema`), not {unknown_keys}")
    if document.get("schema", GENOME_SCHEMA) != GENOME_SCHEMA:
        raise ValueError(f"genome schema {document['schema']} is not {GENOME_SCHEMA}")
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown families {unknown}; the registry has {list(FAMILIES)}")
    return {family: config_from(strategy.Config, families.get(family, {})).as_document()
            for family, strategy in FAMILIES.items()}


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


def sources_sha256() -> str:
    """The digest of every strategy module's source, in file-name order: a genome means what this code makes of it."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Strategies:
    """One strategy per family, built from a genome; families are attributes (`strategies.attention`)."""

    def __init__(self, genome: Genome) -> None:
        complete = resolve({"families": genome})
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


__all__ = ["FAMILIES", "GENOME_SCHEMA", "Genome", "Strategies", "default_genome", "genome_document", "genome_sha256",
           "load_genome", "resolve", "save_genome", "sources_sha256"]
