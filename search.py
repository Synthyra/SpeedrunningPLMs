"""The search over genomes: propose candidates, ship them through queue.sh, read the results into the ledger.

Three proposers share one interface (`propose(observations, count, rng)`):
- `screening`: a 12-run Plackett-Burman design over up to 11 binary module switches against the record, run before
  any adaptive search to estimate main effects and the largest interactions (`effects`).
- `bayes`: a Gaussian process with a noise term over the mixed space (one-hot variants, scaled numbers; numbers of a
  switched-off family held at the middle), with expected improvement over the best posterior mean, not the best
  observation, so a lucky seed is not chased; the noise variance never drops below one seed's (0.0015 nats on
  val_select). A batch is chosen by the kriging believer.
- `genetic`: a population of 8 by tournament from the observed genomes, uniform crossover over families and mutation of
  variants and numbers.
Beside them, `manual` wraps genome files the agent names, each with its hypothesis (a pilot of new modules).
Successive halving on the clock: an adaptive batch first runs at 180 s (scored on val_select alone) and only the top
half runs the full 600 s. Every proposal, its parents and its proposer's reason are recorded in the ledger
(`runs.jsonl`) when its result is collected, and the whole state lives in a JSON file beside the ledger.

    python search.py --state S propose --proposer screening --batch screen1 [--factors unet gates ...]
    python search.py --state S ship --batch screen1     # queue files, then $SPEEDRUN_LAUNCHER bash queue.sh --wait ...
    python search.py --state S collect --batch screen1 --results-dir R --ledger L   # $SPEEDRUN_FETCH fills R
    python search.py --state S effects --batch screen1  # the main-effect table in Markdown
    python search.py --state S status

Symbols: n observations, m candidates, D encoded dimensions, k factors.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import shlex
import subprocess
import numpy as np

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from speedrunning_plms.strategies import FAMILIES, Genome, default_genome, genome_document, genome_sha256, resolve


PROJECT = Path(__file__).resolve().parent
NOISE_SD = 0.0015  # one seed's spread on val_select: 004 against 003, and 015 at seeds 0, 1 and 2
FULL_BUDGET = 600
RUNG_BUDGET = 180
STATE_SCHEMA = 1
DEFAULT_RECORD = "022_embed_head_lr"  # the run the default genome reproduces


@dataclass(frozen=True)
class Dimension:
    """One searched coordinate of a genome: a family's variant or one numeric field."""

    key: str  # "family.field"
    choices: tuple[str | int, ...] = ()  # a variant's or a categorical number's allowed values; empty for a number
    low: float = 0.0
    high: float = 1.0
    log: bool = False
    integer: bool = False

    @property
    def family(self) -> str:
        return self.key.split(".")[0]

    @property
    def name(self) -> str:
        return self.key.split(".")[1]

    @property
    def categorical(self) -> bool:
        return bool(self.choices)


SPACE: tuple[Dimension, ...] = (
    Dimension("attention.variant", ("standard", "differential", "triangle", "triangle_differential", "paired")),
    Dimension("attention.heads", (12, 6)),
    Dimension("positional.variant", ("none", "alibi")),
    Dimension("positional.slope_scale", low=0.5, high=2.0, log=True),
    Dimension("positional.rope_base", low=1e4, high=1e5, log=True),
    Dimension("positional.rope_fraction", low=0.1, high=1.0, log=True),
    Dimension("positional.nope_every", low=0, high=6, integer=True),
    # Numbers whose 0 is off, as choices with 0 first, so a draw can leave them off.
    Dimension("positional.periodic_scale", (0.0, 0.25, 0.5, 1.0)),
    Dimension("attention.key_offset", (0.0, 0.125, 0.25, 0.5)),
    Dimension("attention.score_cap", (0.0, 4.0, 5.0, 6.0, 8.0)),
    Dimension("kernel.variant", ("default", "flash")),
    Dimension("head.variant", ("linear", "two_layer")),
    Dimension("logit_scale.variant", ("none", "ssmax")),
    Dimension("logit_scale.initial_scale", low=0.5, high=2.0, log=True),
    Dimension("residual.variant", ("prenorm", "ngpt", "perinorm", "ln_scaling")),
    Dimension("residual.alpha_init", low=0.02, high=0.2, log=True),
    Dimension("residual.branch_multiplier", low=0.25, high=1.0, log=True),
    Dimension("residual.branch_ramp", low=0.0, high=0.3),
    Dimension("residual.branch_start", low=0.0, high=0.5),
    Dimension("activation.variant", ("relu2", "xielu")),
    Dimension("latent.variant", ("none", "next_layer", "neighbors")),
    Dimension("latent.source_block", low=6, high=30, integer=True),
    Dimension("latent.weight", low=0.01, high=0.5, log=True),
    Dimension("stream.variant", ("single", "hyper", "dense")),
    Dimension("stream.streams", low=2, high=4, integer=True),
    Dimension("unet.variant", ("none", "mirrored")),
    Dimension("embedding_mix.variant", ("none", "every_block")),
    Dimension("embedding_mix.initial_share", low=0.0, high=0.3),
    Dimension("neighbor.variant", ("none", "pair_table")),
    Dimension("value_embeddings.variant", ("none", "ends")),
    Dimension("value_embeddings.blocks_each_end", low=1, high=6, integer=True),
    Dimension("value_residual.variant", ("none", "first_block")),
    Dimension("value_residual.initial_share", low=0.2, high=0.8),
    Dimension("gates.variant", ("none", "residual_channels", "head_sigmoid")),
    Dimension("softcap.variant", ("none", "tanh")),
    Dimension("softcap.cap", low=10.0, high=60.0, log=True),
    Dimension("optimizer.variant", ("muon", "normuon", "muon_plus", "anvil", "amuse")),
    Dimension("optimizer.plus_axis", ("col_row", "row_col", "row", "col")),
    Dimension("optimizer.anvil_slow_beta", low=0.95, high=0.995),
    Dimension("optimizer.anvil_fast_weight", low=0.2, high=0.7),
    Dimension("optimizer.amuse_beta", low=0.3, high=0.7),
    Dimension("optimizer.amuse_rho", low=0.4, high=1.0),
    Dimension("optimizer.adam_beta1", low=0.7, high=0.95),
    Dimension("optimizer.adam_beta2", low=0.95, high=0.999),
    Dimension("optimizer.muon_lr", low=0.02, high=0.08, log=True),
    Dimension("optimizer.embed_lr", low=1e-3, high=1e-2, log=True),
    Dimension("optimizer.head_lr", low=3e-4, high=3e-3, log=True),
    Dimension("optimizer.scalar_lr", low=1e-3, high=1e-2, log=True),
    Dimension("optimizer.cautious_weight_decay", low=0.0, high=0.1),
    Dimension("schedule.variant", ("trapezoid", "one_sqrt")),
    Dimension("schedule.warmup_fraction", low=0.0, high=0.05),
    Dimension("schedule.cooldown_fraction", low=0.25, high=0.65),
    Dimension("schedule.momentum_start", low=0.85, high=0.95),
    Dimension("batch.variant", ("packed", "whole")),
    Dimension("batch.window", (1024, 2048)),  # the count follows (`batch.windows_per_step` 0): 65,536 tokens a step
    Dimension("curriculum.variant", ("none", "short_first", "family_tempered", "length_tempered")),
    Dimension("curriculum.fraction", low=0.2, high=0.6),
    Dimension("curriculum.alpha", low=0.0, high=1.0),
    Dimension("curriculum.beta", low=-0.5, high=0.5),
    Dimension("ema.variant", ("none", "tail", "latest_k", "drift_leap", "two_decay")),
    Dimension("ema.start", low=0.4, high=0.9),
    Dimension("ema.decay", low=0.98, high=0.998),
    Dimension("ema.window", low=100, high=1000, log=True, integer=True),
    Dimension("ema.leap", low=0.05, high=0.5, log=True),
    Dimension("ema.decay_fast", low=0.9, high=0.99),
    Dimension("backprop_free.variant", ("none", "residual_es", "unit_select", "unit_random", "readout_warmstart",
                                        "readout_boost", "unit_route")),
    Dimension("backprop_free.readout_weight", low=0.05, high=1.0, log=True),
    Dimension("backprop_free.switch", low=0.1, high=1.0),
    Dimension("backprop_free.route_k", low=64, high=1024, log=True, integer=True),
)

# The binary switches a screening design may set high; low is the record's choice. Two attention switches combine:
# both high is the composed triangle_differential kernel.
SWITCHES: dict[str, dict[str, dict[str, Any]]] = {
    "unet": {"unet": {"variant": "mirrored"}},
    "embedding_mix": {"embedding_mix": {"variant": "every_block"}},
    "value_embeddings": {"value_embeddings": {"variant": "ends"}},
    "value_residual": {"value_residual": {"variant": "first_block"}},
    "gates": {"gates": {"variant": "residual_channels"}},
    "neighbor": {"neighbor": {"variant": "pair_table"}},
    "ema": {"ema": {"variant": "tail"}},
    "curriculum": {"curriculum": {"variant": "short_first"}},
    "ngpt": {"residual": {"variant": "ngpt"}},
    "differential": {"attention": {"variant": "differential"}},
    "triangle": {"attention": {"variant": "triangle"}},
    "hyper": {"stream": {"variant": "hyper"}},
    "dense": {"stream": {"variant": "dense"}},
    "residual_es": {"backprop_free": {"variant": "residual_es"}},
    "readout_warmstart": {"backprop_free": {"variant": "readout_warmstart"}},
    "unit_select": {"backprop_free": {"variant": "unit_select"}},
    # Switches that change numbers or shapes, not the graph's size, so they add little compile time: queue 3's two
    # near-record changes, the softcap off, plain Muon, and 6 heads of 128.
    "heads_6": {"attention": {"heads": 6}},
    "rope_1e5": {"positional": {"rope_base": 100_000.0}},
    "slopes_x2": {"positional": {"slope_scale": 2.0}},
    "no_softcap": {"softcap": {"variant": "none"}},
    "muon": {"optimizer": {"variant": "muon"}},
    # The 2026-10-30 modules: 2,048-token windows at the record's tokens a step, whole-protein packing, the tempered
    # reading orders at their defaults, SSMax, Peri-LN, LayerNorm Scaling, a halved branch, xIELU, the latent loss,
    # Muon+, the head-wise gate, a quarter of RoPE's pairs, and RoPE off in every fourth block.
    "window_2048": {"batch": {"window": 2048}},
    "whole": {"batch": {"variant": "whole"}},
    "family_tempered": {"curriculum": {"variant": "family_tempered"}},
    "length_tempered": {"curriculum": {"variant": "length_tempered", "beta": 0.5}},
    "ssmax": {"logit_scale": {"variant": "ssmax"}},
    "perinorm": {"residual": {"variant": "perinorm"}},
    "ln_scaling": {"residual": {"variant": "ln_scaling"}},
    "branch_half": {"residual": {"branch_multiplier": 0.5}},
    "xielu": {"activation": {"variant": "xielu"}},
    "latent": {"latent": {"variant": "next_layer"}},
    "muon_plus": {"optimizer": {"variant": "muon_plus"}},
    "head_gate": {"gates": {"variant": "head_sigmoid"}},
    "rope_quarter": {"positional": {"rope_fraction": 0.25}},
    "nope_4": {"positional": {"nope_every": 4}},
    # Round 2: the Dust, noise and Boost hybrids inside backprop, the noise track's Adam beta1, ANVIL and AMUSE, LAWA,
    # the 1-sqrt cooldown and the neighbor heads.
    "readout_boost": {"backprop_free": {"variant": "readout_boost"}},
    "unit_route": {"backprop_free": {"variant": "unit_route"}},
    "branch_ramp": {"residual": {"branch_ramp": 0.1}},
    "anvil": {"optimizer": {"variant": "anvil"}},
    "amuse": {"optimizer": {"variant": "amuse"}},
    "adam_beta1_0.7": {"optimizer": {"adam_beta1": 0.7}},
    "latest_k": {"ema": {"variant": "latest_k"}},
    "one_sqrt": {"schedule": {"variant": "one_sqrt"}},
    "neighbors": {"latent": {"variant": "neighbors"}},
    # Round 3: ESM-2's head, the bidirectional key offset, the helix and strand bias, a binding score cap, the drift
    # leap, the two-decay average, the FlashAttention-4 backend.
    "head_two_layer": {"head": {"variant": "two_layer"}},
    "key_offset": {"attention": {"key_offset": 0.25}},
    "periodic": {"positional": {"periodic_scale": 0.5}},
    "score_cap": {"attention": {"score_cap": 5.0}},
    "drift_leap": {"ema": {"variant": "drift_leap"}},
    "two_decay": {"ema": {"variant": "two_decay"}},
    "flash": {"kernel": {"variant": "flash"}},
}
BACKPROP_FREE_SWITCHES = ("residual_es", "readout_warmstart", "unit_select", "readout_boost", "unit_route")

# The compile warmup is capped at 180 s and a new graph compiles from scratch: the record's in 106 s on the GH200
# (pilot1_01). Each variant's seconds above it were measured alone on the record (pilot1 and pilot3 at the 180 s rung,
# and 005, 010, 011, 026, 028 by hand); a variant not listed, and every number, added none measurable. The sum predicts a
# genome's warmup; the margin covers compile costs that do not add exactly. The proposers skip a genome predicted over.
WARMUP_CAP = 180.0
RECORD_WARMUP = 106.0
WARMUP_MARGIN = 15.0
COMPILE_SECONDS: dict[str, float] = {
    "attention.variant=differential": 24.0,
    "attention.variant=triangle": 46.0,
    "attention.variant=triangle_differential": 76.0,
    "attention.variant=paired": 17.0,
    "residual.variant=ngpt": 36.0,
    "stream.variant=hyper": 37.0,
    "stream.variant=dense": 108.0,
    "backprop_free.variant=residual_es": 66.0,
    "backprop_free.variant=unit_select": 5.0,
    "backprop_free.variant=unit_random": 5.0,
    "backprop_free.variant=readout_warmstart": 7.0,
    "value_residual.variant=first_block": 11.0,
    "gates.variant=residual_channels": 12.0,
    "unet.variant=mirrored": 1.0,
    "neighbor.variant=pair_table": 1.0,
    "embedding_mix.variant=every_block": 5.0,
    # Not measured yet, conservative until a 180 s pilot replaces them: 10 s for an elementwise change in every block,
    # 25 s for a new module in every block, 10 s for a first compile at a new window shape.
    "batch.window=2048": 3.0,  # pilot4_01: warmup 125 s against the record genome's 122
    "logit_scale.variant=ssmax": 10.0,  # pilot4_02: 132 s
    "residual.variant=perinorm": 16.0,  # pilot4_03: 138 s
    "residual.variant=ln_scaling": 10.0,
    "activation.variant=xielu": 10.0,
    "latent.variant=next_layer": 20.0,
    "gates.variant=head_sigmoid": 25.0,
    "residual.branch_multiplier!=1.0": 10.0,  # `!=`: any value but the record's adds the hook
    "residual.branch_ramp!=0.0": 10.0,
    "backprop_free.variant=readout_boost": 30.0,  # 72 head readouts in the graph; the warm start's few cost 7 s
    "backprop_free.variant=unit_route": 25.0,  # a top-k and a scatter in every block
    "latent.variant=neighbors": 10.0,
    "head.variant=two_layer": 10.0,
    "attention.key_offset!=0.0": 10.0,  # two shifted copies and a select on the keys in every block
    "attention.score_cap!=0.0": 10.0,  # a tanh inside the flex score function
    "positional.periodic_scale!=0.0": 10.0,  # two cosines inside the flex score function
    "kernel.variant=flash": 30.0,  # the CuTe kernels' JIT; no published compile time, the pilot measures it
}
# A number that matters only under some of its family's variants; the rest follow `dimension_active`'s rule.
NUMBER_VARIANTS: dict[str, tuple[str, ...]] = {
    "positional.slope_scale": ("alibi",),
    "residual.alpha_init": ("ngpt",),
    "residual.branch_multiplier": ("prenorm", "ln_scaling"),  # ngpt and perinorm normalize the branch after it
    "curriculum.fraction": ("short_first",),
    "curriculum.alpha": ("family_tempered",),
    "curriculum.beta": ("length_tempered",),
    "residual.branch_ramp": ("prenorm", "ln_scaling"),
    "residual.branch_start": ("prenorm", "ln_scaling"),
    "latent.source_block": ("next_layer",),
    "optimizer.cautious_weight_decay": ("muon", "normuon", "muon_plus", "anvil"),  # amuse takes no decay
    "optimizer.anvil_slow_beta": ("anvil",),
    "optimizer.anvil_fast_weight": ("anvil",),
    "optimizer.amuse_beta": ("amuse",),
    "optimizer.amuse_rho": ("amuse",),
    "ema.start": ("tail", "two_decay"),
    "ema.decay": ("tail", "two_decay"),
    "ema.window": ("latest_k", "drift_leap"),
    "ema.leap": ("drift_leap",),
    "ema.decay_fast": ("two_decay",),
    "backprop_free.readout_weight": ("readout_boost",),
    "backprop_free.switch": ("readout_boost",),
    "backprop_free.route_k": ("unit_route",),
}


def read(genome: Genome, key: str) -> Any:
    family, name = key.split(".")
    return genome[family][name]


def written(genome: Genome, changes: dict[str, Any]) -> Genome:
    """A copy of the genome with `{"family.field": value}` changes, checked against the configs."""
    updated = {family: dict(fields) for family, fields in genome.items()}
    for key, value in changes.items():
        family, name = key.split(".")
        updated[family][name] = value
    return resolve({"families": updated})


def predicted_warmup(genome: Genome) -> float:
    """The genome's compile warmup in seconds on the GH200, from each variant's measured cost."""
    return RECORD_WARMUP + sum(COMPILE_SECONDS[key] for key in compile_costs(genome))


def compile_costs(genome: Genome) -> list[str]:
    """The COMPILE_SECONDS entries a genome pays: `field=value` when it holds the value, `field!=value` when it does
    not; the value compares as text (a variant, or a number such as a window)."""
    costs = []
    for key in COMPILE_SECONDS:
        field, value = key.split("!=") if "!=" in key else key.split("=")
        if (str(read(genome, field)) == value) != ("!=" in key):
            costs.append(key)
    return costs


def warmup_repaired(genome: Genome, rng: np.random.Generator) -> Genome:
    """The genome with its costly choices put back at the record's, one at a time in random order, until its predicted
    warmup fits: a uniform draw over the space is rarely under the cap (0.3 percent on 2026-10-30), a repaired draw
    keeps its cheap choices."""
    default = default_genome()
    costly = compile_costs(genome)
    for position in rng.permutation(len(costly)).tolist():
        if fits_warmup(genome):
            break
        field = costly[position].replace("!=", "=").split("=")[0]
        genome = written(genome, {field: read(default, field)})
    return genome


def fits_warmup(genome: Genome) -> bool:
    return predicted_warmup(genome) <= WARMUP_CAP - WARMUP_MARGIN


def dimension_active(genome: Genome, dimension: Dimension) -> bool:
    """A family's number matters only while the family is on (positional numbers need the prior or RoPE either way),
    or under the variants NUMBER_VARIANTS names."""
    if dimension.categorical:
        return True
    if dimension.key in NUMBER_VARIANTS:
        return read(genome, f"{dimension.family}.variant") in NUMBER_VARIANTS[dimension.key]
    if dimension.family in ("optimizer", "schedule", "positional"):
        return True
    return read(genome, f"{dimension.family}.variant") not in ("none", "prenorm", "single")


def open_space(retired: list[str]) -> list[Dimension]:
    """The space without retired variants (`family.variant=value`) or retired numbers (`family.field`)."""
    dimensions = []
    for dimension in SPACE:
        if dimension.key in retired:
            continue
        if dimension.categorical:
            choices = tuple(choice for choice in dimension.choices if f"{dimension.key}={choice}" not in retired)
            dimension = Dimension(dimension.key, choices)
        dimensions.append(dimension)
    return dimensions


def within(genome: Genome, space: list[Dimension]) -> Genome:
    """The genome with every retired choice back at the default: a variant outside its open choices, or a number the
    space no longer holds. Parents and crossovers carry retired choices; no proposal spends a run on one. The window's
    count goes back to tied, so a parent with a hand-set count never pairs it with the other window."""
    default = default_genome()
    open_keys = {dimension.key: dimension for dimension in space}
    changes = {"batch.windows_per_step": read(default, "batch.windows_per_step")}
    for dimension in SPACE:
        value = read(genome, dimension.key)
        if dimension.key not in open_keys:
            changes[dimension.key] = read(default, dimension.key)
        elif dimension.categorical and value not in open_keys[dimension.key].choices:
            assert read(default, dimension.key) in open_keys[dimension.key].choices, f"the default of {dimension.key} is retired"
            changes[dimension.key] = read(default, dimension.key)
    return written(genome, changes)


def encode(genome: Genome, space: list[Dimension]) -> np.ndarray:
    """A genome as a point in [0, 1]^D: variants one-hot, numbers scaled (log where marked), inactive numbers at 0.5."""
    coordinates: list[float] = []
    for dimension in space:
        value = read(genome, dimension.key)
        if dimension.categorical:
            coordinates += [1.0 if value == choice else 0.0 for choice in dimension.choices]
        elif not dimension_active(genome, dimension):
            coordinates.append(0.5)
        else:
            low, high, value = (np.log(dimension.low), np.log(dimension.high), np.log(value)) if dimension.log else (dimension.low, dimension.high, value)
            coordinates.append(float(np.clip((value - low) / (high - low), 0.0, 1.0)))
    return np.asarray(coordinates)  # (D,)


def sample_value(dimension: Dimension, rng: np.random.Generator) -> Any:
    if dimension.categorical:
        return dimension.choices[int(rng.integers(len(dimension.choices)))]
    if dimension.log:
        value = float(np.exp(rng.uniform(np.log(dimension.low), np.log(dimension.high))))
    else:
        value = float(rng.uniform(dimension.low, dimension.high))
    return round(value) if dimension.integer else float(f"{value:.4g}")


def random_genome(space: list[Dimension], rng: np.random.Generator) -> Genome:
    return written(default_genome(), {dimension.key: sample_value(dimension, rng) for dimension in space})


def perturbed(dimension: Dimension, value: Any, rng: np.random.Generator) -> Any:
    """A mutation of one coordinate: another variant, or a number moved by a fifth of its range (log-scaled where marked)."""
    if dimension.categorical:
        others = [choice for choice in dimension.choices if choice != value]
        return others[int(rng.integers(len(others)))] if others else value
    if dimension.log:
        moved = float(np.exp(np.log(value) + rng.normal(0.0, 0.2 * (np.log(dimension.high) - np.log(dimension.low)))))
    else:
        moved = float(value + rng.normal(0.0, 0.2 * (dimension.high - dimension.low)))
    moved = float(np.clip(moved, dimension.low, dimension.high))
    return round(moved) if dimension.integer else float(f"{moved:.4g}")


def mutate(genome: Genome, space: list[Dimension], rng: np.random.Generator, rate: float) -> tuple[Genome, list[str]]:
    """Each coordinate changes with probability `rate`; at least one always does."""
    chosen = [dimension for dimension in space if rng.random() < rate] or [space[int(rng.integers(len(space)))]]
    changes = {dimension.key: perturbed(dimension, read(genome, dimension.key), rng) for dimension in chosen}
    return written(genome, changes), sorted(changes)


def crossover(first: Genome, second: Genome, rng: np.random.Generator) -> tuple[Genome, list[str]]:
    """Each family's whole config from one parent or the other, so a family's variant keeps its own numbers."""
    from_second = [family for family in FAMILIES if rng.random() < 0.5]
    child = {family: dict(second[family] if family in from_second else first[family]) for family in FAMILIES}
    return resolve({"families": child}), from_second


@dataclass
class Observation:
    name: str
    genome: Genome
    value: float  # val_select of the selected weights at the full budget


@dataclass
class Proposal:
    genome: Genome
    parents: list[str]
    reason: str
    factors: dict[str, int] = field(default_factory=dict)  # a screening run's switch levels, +1 or -1


class Proposer(Protocol):
    name: str

    def propose(self, observations: list[Observation], count: int, rng: np.random.Generator) -> list[Proposal]:
        ...


def plackett_burman_12() -> np.ndarray:
    """The 12-run Plackett-Burman design: 11 orthogonal two-level columns."""
    generator = np.array([1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1])  # (11,)
    rows = [np.roll(generator, shift) for shift in range(11)] + [-np.ones(11, dtype=int)]
    return np.stack(rows)  # (12, 11)


def switched(factors: list[str], levels: np.ndarray, base: Genome) -> Genome:
    """The base with every factor at +1 switched on; the two attention switches together give the composed kernel."""
    # levels: (k,) of +1 and -1.
    families = {family: dict(fields) for family, fields in base.items()}
    on = {factor for factor, level in zip(factors, levels, strict=True) if level > 0}
    for factor in sorted(on):
        for family, fields in SWITCHES[factor].items():
            families[family].update(fields)
    if {"differential", "triangle"} <= on:
        families["attention"]["variant"] = "triangle_differential"
    if {"hyper", "dense"} <= on:
        raise ValueError("hyper and dense are two variants of one family; screen one of them")
    if len(on & set(BACKPROP_FREE_SWITCHES)) > 1:
        raise ValueError(f"{BACKPROP_FREE_SWITCHES} are variants of one family; screen one of them")
    variants: dict[str, set[str]] = {}
    for factor in on:
        for family, fields in SWITCHES[factor].items():
            if "variant" in fields:
                variants.setdefault(family, set()).add(fields["variant"])
    clashing = sorted(family for family, chosen in variants.items() if len(chosen) > 1 and family != "attention")
    if clashing:
        raise ValueError(f"switches set two variants of {clashing}; screen one of each family's")
    return resolve({"families": families})


def record_of(observations: list[Observation]) -> Observation:
    """The best observation, or the default genome (022) before any run is observed."""
    if not observations:
        return Observation(DEFAULT_RECORD, default_genome(), math.nan)
    return min(observations, key=lambda observation: observation.value)


class ScreeningProposer:
    """A Plackett-Burman design over the given switches, every run against the record: the best observation."""

    name = "screening"

    def __init__(self, factors: list[str]) -> None:
        unknown = sorted(set(factors) - set(SWITCHES))
        if unknown or not 2 <= len(factors) <= 11:
            raise ValueError(f"screen 2 to 11 switches from {sorted(SWITCHES)}; got {factors}")
        self.factors = factors

    def propose(self, observations: list[Observation], count: int, rng: np.random.Generator) -> list[Proposal]:
        record = record_of(observations)
        for factor in self.factors:
            alone = np.array([1 if other == factor else -1 for other in self.factors])  # (k,)
            if switched(self.factors, alone, record.genome) == record.genome:
                raise ValueError(f"{factor} is already on in the record {record.name}; it cannot be screened against it")
        design = plackett_burman_12()[:, : len(self.factors)]  # (12, k)
        proposals = []
        for row in design:
            on = [factor for factor, level in zip(self.factors, row, strict=True) if level > 0]
            genome = switched(self.factors, row, record.genome)
            if not fits_warmup(genome):
                raise ValueError(f"the row with {', '.join(on)} is predicted to compile in {predicted_warmup(genome):.0f} s, "
                                 f"over the {WARMUP_CAP - WARMUP_MARGIN:.0f} s the margin allows; screen fewer costly switches")
            proposals.append(Proposal(genome, [record.name],
                                      f"Plackett-Burman row: on {', '.join(on) or 'nothing'} against the record {record.name}",
                                      {factor: int(level) for factor, level in zip(self.factors, row, strict=True)}))
        return proposals


class ManualProposer:
    """Genomes the agent names, each with its hypothesis: a pilot of new modules, or an idea worth a run."""

    name = "manual"

    def __init__(self, genomes: list[Path], hypotheses: list[str]) -> None:
        if len(genomes) != len(hypotheses):
            raise ValueError("give one hypothesis per genome")
        self.genomes = genomes
        self.hypotheses = hypotheses

    def propose(self, observations: list[Observation], count: int, rng: np.random.Generator) -> list[Proposal]:
        record = record_of(observations)
        return [Proposal(resolve(json.loads(path.read_text(encoding="utf-8"))), [record.name], f"{path.stem}: {hypothesis}")
                for path, hypothesis in zip(self.genomes, self.hypotheses, strict=True)]


def main_effects(levels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each factor's mean response at +1 minus at -1."""
    # levels: (n, k) of +1 and -1; values: (n,); returns (k,).
    return np.array([values[column > 0].mean() - values[column < 0].mean() for column in levels.T])  # (k,)


def interactions(levels: np.ndarray, values: np.ndarray, factors: list[str], active: list[int], top: int) -> list[tuple[str, float]]:
    """The two-factor interactions with at least one active parent that most reduce the residual of a fit on the
    active main effects, each with its coefficient (half the interaction effect). Plackett-Burman columns alias every
    interaction partly with the other main effects, so these are leads to test, not estimates."""
    # levels: (n, k); values: (n,).
    base = np.column_stack([np.ones(len(values)), levels[:, active]])  # (n, 1 + a)
    residual = values - base @ np.linalg.lstsq(base, values, rcond=None)[0]  # (n,)
    found = []
    for first, second in itertools.combinations(range(levels.shape[1]), 2):
        if first not in active and second not in active:
            continue
        column = levels[:, first] * levels[:, second]  # (n,)
        coefficient = float(column @ residual / len(values))
        found.append((f"{factors[first]} x {factors[second]}", coefficient))
    return sorted(found, key=lambda item: -abs(item[1]))[:top]


def gp_posterior(X: np.ndarray, y: np.ndarray, X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """The posterior mean and sd of a squared-exponential GP with a constant mean and a noise term; the length scale,
    signal and noise come from a grid by marginal likelihood, the noise never below one seed's."""
    # X: (n, D); y: (n,); X_query: (m, D); returns (m,), (m,) and the chosen hyperparameters.
    mean = float(y.mean())
    centered = y - mean  # (n,)
    signal = max(float(centered.var()), NOISE_SD**2)
    squared = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)  # (n, n)
    best: tuple[float, float, float, float] | None = None
    for length, signal_scale, noise in itertools.product((0.5, 0.8, 1.2, 2.0, 3.0), (0.5, 1.0, 2.0), (1.0, 2.0, 4.0)):
        K = signal * signal_scale * np.exp(-squared / (2 * length**2)) + (noise * NOISE_SD) ** 2 * np.eye(len(y))  # (n, n)
        L = np.linalg.cholesky(K)  # (n, n)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, centered))  # (n,)
        evidence = -0.5 * centered @ alpha - np.log(np.diag(L)).sum()
        if best is None or evidence > best[0]:
            best = (evidence, length, signal * signal_scale, (noise * NOISE_SD) ** 2)
    assert best is not None
    _, length, variance, noise_variance = best
    K = variance * np.exp(-squared / (2 * length**2)) + noise_variance * np.eye(len(y))  # (n, n)
    K_query = variance * np.exp(-((X_query[:, None, :] - X[None, :, :]) ** 2).sum(-1) / (2 * length**2))  # (m, n)
    L = np.linalg.cholesky(K)  # (n, n)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, centered))  # (n,)
    V = np.linalg.solve(L, K_query.T)  # (n, m)
    posterior_mean = mean + K_query @ alpha  # (m,)
    posterior_sd = np.sqrt(np.clip(variance - (V**2).sum(0), 1e-12, None))  # (m,)
    hyperparameters = {"length": length, "signal_sd": math.sqrt(variance), "noise_sd": math.sqrt(noise_variance)}
    return posterior_mean, posterior_sd, hyperparameters  # (m,), (m,)


def expected_improvement(mean: np.ndarray, sd: np.ndarray, incumbent: float) -> np.ndarray:
    """Expected improvement below the incumbent's posterior mean (lower loss is better)."""
    # mean, sd: (m,); returns (m,).
    from scipy.stats import norm as gaussian

    gain = incumbent - mean  # (m,)
    z = gain / sd  # (m,)
    return gain * gaussian.cdf(z) + sd * gaussian.pdf(z)  # (m,)


class BayesianProposer:
    name = "bayes"

    def __init__(self, space: list[Dimension], pool: int = 4000) -> None:
        self.space = space
        self.pool = pool

    def propose(self, observations: list[Observation], count: int, rng: np.random.Generator) -> list[Proposal]:
        if len(observations) < 3:
            raise ValueError("the GP needs at least three observations at the full budget")
        X = np.stack([encode(observation.genome, self.space) for observation in observations])  # (n, D)
        y = np.array([observation.value for observation in observations])  # (n,)
        seen = {genome_sha256(observation.genome) for observation in observations}
        fitted_mean = gp_posterior(X, y, X)[0]  # (n,)
        ranked = np.argsort(fitted_mean)  # by posterior mean, not by the noisy observation
        incumbent = observations[int(ranked[0])]
        pool = [random_genome(self.space, rng) for _ in range(self.pool // 2)]
        for _ in range(self.pool // 2):
            parent = observations[int(ranked[int(rng.integers(min(5, len(ranked))))])]
            pool.append(mutate(parent.genome, self.space, rng, rate=2 / len(self.space))[0])
        distinct = {genome_sha256(genome): genome for genome in (warmup_repaired(within(genome, self.space), rng) for genome in pool)}
        pool = [genome for digest, genome in distinct.items() if digest not in seen and fits_warmup(genome)]
        X_pool = np.stack([encode(genome, self.space) for genome in pool])  # (m, D)
        proposals: list[Proposal] = []
        X_fit, y_fit = X, y
        for _ in range(count):
            mean, sd, hyper = gp_posterior(X_fit, y_fit, X_pool)  # (m,), (m,)
            best_mean = float(gp_posterior(X_fit, y_fit, X_fit)[0].min())
            gains = expected_improvement(mean, sd, best_mean)  # (m,)
            choice = int(np.argmax(gains))
            nearest = observations[int(np.argmin(((X - X_pool[choice]) ** 2).sum(-1)))]
            changes = genome_document(pool[choice])["families"]
            proposals.append(Proposal(pool[choice], [incumbent.name, nearest.name],
                                      f"GP expected improvement {gains[choice]:.5f}: predicted val_select {mean[choice]:.4f} "
                                      f"+- {sd[choice]:.4f} against the best posterior mean {best_mean:.4f} ({incumbent.name}); "
                                      f"nearest observed {nearest.name}; length {hyper['length']}, noise sd {hyper['noise_sd']:.4f}; "
                                      f"changes {json.dumps(changes, sort_keys=True)}"))
            # The kriging believer: the chosen point joins the data at its predicted mean, so the next pick looks elsewhere.
            X_fit = np.vstack([X_fit, X_pool[choice]])  # (n + 1, D)
            y_fit = np.append(y_fit, mean[choice])  # (n + 1,)
            X_pool = np.delete(X_pool, choice, axis=0)  # (m - 1, D)
            del pool[choice]
        return proposals


class GeneticProposer:
    name = "genetic"

    def __init__(self, space: list[Dimension], tournament: int = 3) -> None:
        self.space = space
        self.tournament = tournament

    def _select(self, observations: list[Observation], rng: np.random.Generator) -> Observation:
        entrants = rng.choice(len(observations), size=min(self.tournament, len(observations)), replace=False)
        return min((observations[int(index)] for index in entrants), key=lambda observation: observation.value)

    def propose(self, observations: list[Observation], count: int, rng: np.random.Generator) -> list[Proposal]:
        if len(observations) < 2:
            raise ValueError("the genetic proposer needs at least two observed genomes")
        seen = {genome_sha256(observation.genome) for observation in observations}
        proposals: list[Proposal] = []
        for _ in range(1000 * count):
            if len(proposals) == count:
                break
            first = self._select(observations, rng)
            second = self._select(observations, rng)
            child, from_second = crossover(first.genome, second.genome, rng)
            child, mutated = mutate(child, self.space, rng, rate=1.5 / len(self.space))
            child = within(child, self.space)
            digest = genome_sha256(child)
            if digest in seen or not fits_warmup(child):
                continue
            seen.add(digest)
            changes = genome_document(child)["families"]
            proposals.append(Proposal(child, [first.name, second.name],
                                      f"child of {first.name} ({first.value:.4f}) and {second.name} ({second.value:.4f}); "
                                      f"families from {second.name}: {', '.join(from_second) or 'none'}; mutated {', '.join(mutated)}; "
                                      f"changes {json.dumps(changes, sort_keys=True)}"))
        if len(proposals) < count:
            raise RuntimeError(f"{len(proposals)} new children within the warmup cap after {1000 * count} tries; widen the space")
        return proposals


# The state file and the ledger.

def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": STATE_SCHEMA, "batches": {}, "proposals": {}, "seeded": {}, "retired": []}
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["schema"] == STATE_SCHEMA
    return state


def save_state(state: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def observations_of(state: dict[str, Any]) -> list[Observation]:
    """Every genome with a valid result at the full budget: the search's own and the seeded hand-written runs."""
    found = [Observation(name, resolve({"families": seeded["genome"]}), seeded["val_select"])
             for name, seeded in state["seeded"].items()]
    for name, proposal in state["proposals"].items():
        full = proposal["results"].get(str(FULL_BUDGET))
        if full and full["valid"] and full["val_select"] is not None:
            found.append(Observation(name, resolve({"families": proposal["genome"]}), full["val_select"]))
    return found


def incumbent_of(state: dict[str, Any]) -> Observation:
    return min(observations_of(state), key=lambda observation: observation.value)


def run_name(batch: str, index: int, budget: int) -> str:
    return f"{batch}_{index:02d}" + ("" if budget == FULL_BUDGET else f"_r{budget}")


def write_queue_entry(name: str, genome: Genome, hypothesis: str, budget: int) -> None:
    """candidates/queue/<name>.json (the genome), .txt (the hypothesis) and .args (the budget and splits)."""
    queue = PROJECT / "candidates" / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    (queue / f"{name}.json").write_text(json.dumps(genome_document(genome), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (queue / f"{name}.txt").write_text(hypothesis.replace("\n", " ") + "\n", encoding="utf-8", newline="\n")
    extra = "" if budget == FULL_BUDGET else f"--budget {budget} --final-splits val_select"
    (queue / f"{name}.args").write_text(extra + "\n", encoding="utf-8", newline="\n")


def propose(state: dict[str, Any], proposer: Proposer, batch: str, count: int, seed: int, rungs: list[int]) -> list[str]:
    if batch in state["batches"]:
        raise ValueError(f"batch {batch} exists")
    rng = np.random.default_rng(seed)
    proposals = proposer.propose(observations_of(state), count, rng)
    names = []
    for index, proposal in enumerate(proposals, start=1):
        name = run_name(batch, index, FULL_BUDGET)
        state["proposals"][name] = {
            "batch": batch, "proposer": proposer.name, "parents": proposal.parents, "reason": proposal.reason,
            "genome": genome_document(proposal.genome)["families"], "genome_sha256": genome_sha256(proposal.genome),
            "factors": proposal.factors, "results": {}, "promoted": rungs[0] == FULL_BUDGET,
        }
        names.append(name)
    state["batches"][batch] = {"proposer": proposer.name, "rungs": rungs, "seed": seed, "names": names,
                               "created_utc": datetime.now(UTC).isoformat(), "shipments": []}
    return names


def next_rung(state: dict[str, Any], batch: str) -> int:
    """The batch's first rung with a run still to do."""
    entry = state["batches"][batch]
    return next(rung for rung in entry["rungs"] if any(
        str(rung) not in state["proposals"][name]["results"] and (rung != FULL_BUDGET or state["proposals"][name]["promoted"])
        for name in entry["names"]))


def ship(state: dict[str, Any], batches: list[str], budget: int | None = None) -> list[str]:
    """Write the batches' queue entries, each at its next rung (or `budget`), and run the launcher with one queue.sh
    over all of them, so the box runs them back to back; returns the run names."""
    names = []
    for batch in batches:
        rung = budget or next_rung(state, batch)
        shipped = []
        for name in state["batches"][batch]["names"]:
            proposal = state["proposals"][name]
            if (rung == FULL_BUDGET and not proposal["promoted"]) or str(rung) in proposal["results"]:
                continue
            run = run_name(batch, int(name.rsplit("_", 1)[1]), rung)
            hypothesis = f"{proposal['proposer']} proposal ({rung} s): {proposal['reason']}"
            write_queue_entry(run, resolve({"families": proposal["genome"]}), hypothesis, rung)
            shipped.append(run)
        state["batches"][batch]["shipments"].append({"budget": rung, "names": shipped, "utc": datetime.now(UTC).isoformat()})
        names += shipped
    command = ["bash", "queue.sh", "--wait", *names]
    launcher = os.environ.get("SPEEDRUN_LAUNCHER")
    if launcher is None:
        print("SPEEDRUN_LAUNCHER is unset; run on the box from the project:", shlex.join(command))
    else:
        subprocess.run([*shlex.split(launcher), *command], check=True)
    return names


def fetch(run: str, destination: Path) -> Path | None:
    """The run's result.json, copied by $SPEEDRUN_FETCH (a command with {name} and {destination}); None if absent."""
    target = destination / f"{run}.json"
    template = os.environ.get("SPEEDRUN_FETCH")
    if not target.exists() and template is not None:
        command = template.format(name=run, destination=shlex.quote(str(target)))
        subprocess.run(shlex.split(command), check=False, capture_output=True)
    return target if target.exists() else None


def broken_caps(run_result: dict[str, Any]) -> str:
    """Why a run is invalid, from its clock receipt."""
    clock = run_result["clock"]
    broken = [f"{part} {clock[f'{part}_seconds']:.0f} s over the {clock[f'{part}_cap_seconds']:.0f} s cap"
              for part in ("warmup", "eval") if clock[f"{part}_seconds"] > clock[f"{part}_cap_seconds"]]
    return "invalid: " + ("; ".join(broken) or f"status {run_result['status']}")


def collect(state: dict[str, Any], batch: str, results_dir: Path, ledger: Path) -> None:
    """Read every finished run of the batch into the state and the ledger, then promote the top half of a finished
    short rung."""
    import record

    rungs = state["batches"][batch]["rungs"]
    for name in state["batches"][batch]["names"]:
        proposal = state["proposals"][name]
        for budget in rungs:
            run = run_name(batch, int(name.rsplit("_", 1)[1]), budget)
            if str(budget) in proposal["results"]:
                continue
            path = fetch(run, results_dir)
            if path is None:
                continue
            run_result = json.loads(path.read_text(encoding="utf-8"))
            val_select = run_result["metrics"].get("val_select", {}).get("loss")
            valid = run_result["status"] == "completed" and bool(run_result["clock"]["valid"])
            incumbent = incumbent_of(state) if observations_of(state) else None  # before this result joins them
            proposal["results"][str(budget)] = {
                "run": run, "val_select": val_select, "test": run_result["metrics"].get("test", {}).get("loss"),
                "steps": run_result["clock"]["steps"], "warmup_seconds": run_result["clock"]["warmup_seconds"],
                "valid": valid, "candidate_sha256": run_result["candidate_sha256"],
            }
            if budget != FULL_BUDGET:
                incumbent = None  # a short rung decides promotion within its batch, never a record
            if not valid:
                verdict = "invalid"
            elif budget != FULL_BUDGET:
                verdict = "reference"
            else:
                verdict = "keep" if incumbent is None or val_select < incumbent.value else "revert"
            note = (f"{proposal['proposer']} batch {batch}, {budget} s rung, val_select {val_select:.4f}"
                    + (f" against {incumbent.name} {incumbent.value:.4f}" if incumbent else "")
                    + ("" if valid else f"; {broken_caps(run_result)}"))
            record.append_row(ledger, run_result, verdict, note, incumbent.name if incumbent else None, {
                "search_batch": batch, "proposer": proposal["proposer"], "parents": proposal["parents"],
                "reason": proposal["reason"], "genome": proposal["genome"], "rung_seconds": budget,
            })
    if len(rungs) > 1:
        short = str(rungs[0])
        scored = [(state["proposals"][name]["results"].get(short), name) for name in state["batches"][batch]["names"]]
        if all(rung is not None for rung, _ in scored) and not any(state["proposals"][name]["promoted"] for _, name in scored):
            ranked = sorted((rung["val_select"] if rung["valid"] and rung["val_select"] is not None else math.inf, name)
                            for rung, name in scored)  # type: ignore[index]
            for _, name in ranked[: max(1, len(ranked) // 2)]:
                state["proposals"][name]["promoted"] = True


def mark_crashed(state: dict[str, Any], run: str, note: str) -> None:
    """A run that left no result.json: its rung counts as invalid, so promotion ranks it last; the launch log says why."""
    match = re.fullmatch(r"(.+)_(\d+)(?:_r(\d+))?", run)
    if match is None:
        raise ValueError(f"{run} is not a search run name")
    batch, index, rung = match.groups()
    budget = int(rung) if rung else FULL_BUDGET
    name = run_name(batch, int(index), FULL_BUDGET)
    state["proposals"][name]["results"][str(budget)] = {"run": run, "val_select": None, "test": None, "steps": 0,
                                                        "warmup_seconds": None, "valid": False, "crash": note}


def effects_table(state: dict[str, Any], batch: str) -> str:
    """The screening batch's main effects and leading interactions on val_select at the full budget, in Markdown."""
    names = [name for name in state["batches"][batch]["names"] if str(FULL_BUDGET) in state["proposals"][name]["results"]]
    proposals = [state["proposals"][name] for name in names]
    factors = list(proposals[0]["factors"])
    levels = np.array([[proposal["factors"][factor] for factor in factors] for proposal in proposals])  # (n, k)
    values = np.array([proposal["results"][str(FULL_BUDGET)]["val_select"] for proposal in proposals])  # (n,)
    tests = [proposal["results"][str(FULL_BUDGET)]["test"] for proposal in proposals]
    effects = main_effects(levels, values)  # (k,)
    test_effects = main_effects(levels, np.array(tests)) if None not in tests else np.full(len(factors), np.nan)  # (k,)
    standard_error = NOISE_SD * math.sqrt(4 / len(values))
    active = [index for index, effect in enumerate(effects) if abs(effect) > 2 * standard_error]
    lines = [f"Main effects of switching each module on, {len(values)} runs, val_select in nats (negative helps); "
             f"one seed's sd {NOISE_SD} gives a standard error of {standard_error:.4f} per effect.", "",
             "| Switch | Effect on val_select | Effect on test | Beyond two standard errors |", "|---|---|---|---|"]
    for index in np.argsort(effects):
        lines.append(f"| {factors[index]} | {effects[index]:+.4f} | {test_effects[index]:+.4f} | {'yes' if index in active else 'no'} |")
    left = len(values) - 1 - len(active)  # residual degrees of freedom after the active main effects
    if left < 2:
        lines += ["", f"No interaction leads: the {len(active)} active main effects leave {left} of the {len(values) - 1} "
                      "degrees of freedom, too few to tell an interaction from noise; a follow-up run tests them."]
        return "\n".join(lines)
    lines += ["", "Leading two-factor interactions (half-effects, aliased with other main effects; leads, not estimates):", ""]
    lines += [f"- {pair}: {coefficient:+.4f}" for pair, coefficient in interactions(levels, values, factors, active, 3)]
    return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", type=Path, required=True, help="the search state JSON, beside the campaign ledger")
    commands = parser.add_subparsers(dest="command", required=True)
    proposing = commands.add_parser("propose")
    proposing.add_argument("--proposer", choices=("screening", "bayes", "genetic", "manual"), required=True)
    proposing.add_argument("--batch", required=True)
    proposing.add_argument("--count", type=int, default=8)
    proposing.add_argument("--seed", type=int, default=0)
    proposing.add_argument("--factors", nargs="+", default=None, help="screening: the switches, 2 to 11")
    proposing.add_argument("--genomes", nargs="+", type=Path, default=[], help="manual: genome files")
    proposing.add_argument("--hypotheses", nargs="+", default=[], help="manual: one sentence per genome")
    proposing.add_argument("--rungs", nargs="+", type=int, default=None,
                           help="budgets in order; screening 600 by default, the others 180 then 600")
    shipping = commands.add_parser("ship")
    shipping.add_argument("--batch", nargs="+", required=True, help="one or more batches, shipped as one queue")
    shipping.add_argument("--budget", type=int, default=None, help="the rung; each batch's next unfinished one by default")
    collecting = commands.add_parser("collect")
    collecting.add_argument("--batch", required=True)
    collecting.add_argument("--results-dir", type=Path, required=True)
    collecting.add_argument("--ledger", type=Path, required=True)
    collecting.add_argument("--crashed", nargs="+", default=[], help="runs that left no result.json, recorded as invalid")
    collecting.add_argument("--crash-note", default="", help="why they crashed, one line")
    seeding = commands.add_parser("seed", help="add a finished hand-written run on the record's base as an observation")
    seeding.add_argument("--name", required=True)
    seeding.add_argument("--genome", type=Path, required=True)
    seeding.add_argument("--val-select", type=float, required=True)
    retiring = commands.add_parser("retire", help="take a variant (family.variant=value) or a number out of the space")
    retiring.add_argument("keys", nargs="+")
    effects = commands.add_parser("effects")
    effects.add_argument("--batch", required=True)
    commands.add_parser("status")
    return parser.parse_args()


def proposer_named(arguments: argparse.Namespace, space: list[Dimension]) -> Proposer:
    if arguments.proposer == "screening":
        return ScreeningProposer(arguments.factors)
    if arguments.proposer == "bayes":
        return BayesianProposer(space)
    if arguments.proposer == "genetic":
        return GeneticProposer(space)
    return ManualProposer(arguments.genomes, arguments.hypotheses)


def main() -> None:
    arguments = parse_arguments()
    state = load_state(arguments.state)
    space = open_space(state["retired"])
    if arguments.command == "propose":
        rungs = arguments.rungs or ([FULL_BUDGET] if arguments.proposer == "screening" else [RUNG_BUDGET, FULL_BUDGET])
        proposer = proposer_named(arguments, space)
        names = propose(state, proposer, arguments.batch, arguments.count, arguments.seed, rungs)
        for name in names:
            print(name, state["proposals"][name]["reason"])
    elif arguments.command == "ship":
        print("shipped", ship(state, arguments.batch, arguments.budget))
    elif arguments.command == "collect":
        arguments.results_dir.mkdir(parents=True, exist_ok=True)
        for run in arguments.crashed:
            mark_crashed(state, run, arguments.crash_note)
        collect(state, arguments.batch, arguments.results_dir, arguments.ledger)
    elif arguments.command == "seed":
        genome = resolve(json.loads(arguments.genome.read_text(encoding="utf-8")))
        state["seeded"][arguments.name] = {"genome": genome_document(genome)["families"], "val_select": arguments.val_select}
    elif arguments.command == "retire":
        state["retired"] = sorted(set(state["retired"]) | set(arguments.keys))
    elif arguments.command == "effects":
        print(effects_table(state, arguments.batch))
    observations = observations_of(state)
    if observations:
        best = min(observations, key=lambda observation: observation.value)
        print(json.dumps({"observations": len(observations), "incumbent": best.name, "val_select": best.value,
                          "retired": state["retired"]}))
    save_state(state, arguments.state)


if __name__ == "__main__":
    main()
