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
Beside them, `random` draws uniformly over the open space: a track's first design, before the others have observations.
Successive halving on the clock: an adaptive batch first runs at 180 s (scored on val_select alone) and only the top
half runs the full 600 s. Every proposal, its parents and its proposer's reason are recorded in the ledger
(`runs.jsonl`) when its result is collected, and the whole state lives in a JSON file beside the ledger. What a genome
may vary, the screening switches and the compile costs are the catalog in `search_space.py`, read here as `search.<name>`.

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
import pandas as pd

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, NotRequired, Protocol, TypedDict

from search_space import (BACKPROP_FREE_SWITCHES, COMPILE_SECONDS, NUMBER_VARIANTS, PEAK_CAP_GIB, PEAK_PREDICTION_ERROR,
                          RECORD_PEAK_GIB, RECORD_WARMUP, SPACE, SWITCHES, WARMUP_CAP, WARMUP_MARGIN, Dimension, GenomeValue)
from speedrunning_plms.strategies import FAMILIES, Genome, GenomeConflict, default_genome, genome_document, genome_sha256, resolve
from speedrunning_plms.strategies.batch import STEP_TOKENS
from speedrunning_plms.strategies.shape import ShapeConfig


class Track(TypedDict):
    """One clock's budgets and its queue arguments; a state may override `group` and add `final_splits`."""

    budget: int  # the full run's seconds on the clock
    rung: int  # the short rung of successive halving
    rungs: list[int]  # the budgets an autosearch batch runs, in order
    eval_cap: float
    eval_interval: float
    prefix: str  # keeps the tracks' batch and run names apart in the one runs folder
    group: str  # the W&B group and experiment id
    final_splits: NotRequired[list[str]]


class RungResult(TypedDict):
    """One run's outcome at one budget, as `collect` or `mark_crashed` writes it."""

    run: str
    val_select: float | None
    test: float | None
    steps: int
    warmup_seconds: float | None
    valid: bool
    candidate_sha256: NotRequired[str]  # absent for a crash, which left no result.json
    crash: NotRequired[str]


class ProposalRecord(TypedDict):
    batch: str
    proposer: str
    parents: list[str]
    reason: str
    genome: dict[str, dict[str, Any]]  # the genome document's families
    genome_sha256: str
    factors: dict[str, int]  # a screening run's switch levels, +1 or -1
    results: dict[str, RungResult]  # by budget, as text
    promoted: bool  # whether it runs the full budget


class Shipment(TypedDict):
    budget: int
    names: list[str]
    utc: str


class BatchRecord(TypedDict):
    proposer: str
    rungs: list[int]
    seed: int
    names: list[str]  # its proposals' names, which are their full-budget run names
    created_utc: str
    shipments: list[Shipment]


class SeededRun(TypedDict):
    genome: dict[str, dict[str, Any]]
    val_select: float


class OtherTrackRow(TypedDict):
    name: str
    genome: dict[str, dict[str, Any]]
    budget: int
    val_select: float


class TrimmedPick(TypedDict):
    """A pick autosearch took out after a failed preflight or a crash, and what that retired."""

    run: str
    kind: str  # "preflight" or "crash"
    why: str
    retired: list[str]
    genome: dict[str, dict[str, Any]]
    reason: str
    utc: str


class SearchState(TypedDict):
    """The `search.json` document. `load_state` fills `track`; the rest of the optional keys are a state's own choices."""

    schema: int
    track: str
    batches: dict[str, BatchRecord]
    proposals: dict[str, ProposalRecord]
    seeded: dict[str, SeededRun]
    retired: list[str]  # `family.variant=value` or `family.field` keys out of the space
    proposers: NotRequired[list[str]]  # autosearch's turns
    rungs: NotRequired[list[int]]  # autosearch's budgets for a batch
    group: NotRequired[str]
    prefix: NotRequired[str]  # its batch names, apart from another search seeded from the same track's state
    final_splits: NotRequired[list[str]]
    other_track_rows: NotRequired[list[OtherTrackRow]]
    trimmed: NotRequired[list[TrimmedPick]]


class RunName(NamedTuple):
    """A run name read back: `<batch>_<index>` is the proposal's name and its full-budget run, `_r<budget>` a shorter rung."""

    proposal: str
    batch: str
    index: int
    budget: int


class DueRun(NamedTuple):
    """One run a shipment writes to the queue."""

    batch: str
    run: str
    rung: int


PROJECT = Path(__file__).resolve().parent
NOISE_SD = 0.0015  # one seed's spread on val_select: 004 against 003, and 015 at seeds 0, 1 and 2
# The tracks: a state names one, and `load_state` applies it to the module's budgets. The 1-hour track's rungs put
# every proposal through 600 s first; its evals are spaced and capped at six times the 10-minute track's interval
# (Logan, 2026-10-30: good selection without spending too much time on eval).
TRACKS: dict[str, Track] = {
    "10min": {"budget": 600, "rung": 180, "rungs": [600], "eval_cap": 60.0, "eval_interval": 90.0, "prefix": "",
              "group": "2026-10-28_afdb_v6_speedrun"},
    "1h": {"budget": 3600, "rung": 600, "rungs": [600, 3600], "eval_cap": 360.0, "eval_interval": 180.0, "prefix": "h_",
           "group": "2026-10-31_afdb_v6_speedrun_1h"},
}
TRACK = TRACKS["10min"]  # the loaded state's track, applied by `load_state`; the 10-minute track until one is loaded
FULL_BUDGET = TRACK["budget"]
RUNG_BUDGET = TRACK["rung"]
STATE_SCHEMA = 1
DEFAULT_RECORD = "022_embed_head_lr"  # the run the default genome reproduces
# A batch name may itself hold "_r" (h_auto01_random), so the index and the rung are read from the end.
RUN_NAME = re.compile(r"(?P<proposal>(?P<batch>.+)_(?P<index>\d+))(?:_r(?P<budget>\d+))?")


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


RECORD_SHAPE = ShapeConfig()
CONFLICT_PARTNERS = ("shape", "stream", "loop")  # the families every `Strategy.conflicts` reads besides its own


def predicted_warmup(genome: Genome) -> float:
    """The genome's compile warmup in seconds on the GH200: each variant's measured cost on the record's 36 blocks,
    scaled by the blocks the compiled graph unrolls (an estimate away from 36 until a run measures one)."""
    per_record_depth = RECORD_WARMUP + sum(COMPILE_SECONDS[key] for key in compile_costs(genome))
    return per_record_depth * read(genome, "shape.blocks") / RECORD_SHAPE.blocks


def predicted_peak_gib(genome: Genome) -> float:
    """The genome's peak training memory: the record's measured peak scaled by the tokens one compiled step holds, the
    depth, and a block's stored activations a token (about 7 d for the attention and its norms, 2 f for the MLP). An
    estimate until runs away from the record's shape measure it."""
    window = read(genome, "batch.window")
    tokens = window * (read(genome, "batch.windows_per_step") or STEP_TOKENS // window) / read(genome, "batch.micro_divisor")
    width = 7 * read(genome, "shape.hidden") + 2 * read(genome, "shape.intermediate")
    record_width = 7 * RECORD_SHAPE.hidden + 2 * RECORD_SHAPE.intermediate
    return RECORD_PEAK_GIB * tokens / STEP_TOKENS * read(genome, "shape.blocks") / RECORD_SHAPE.blocks * width / record_width


def fits_machine(genome: Genome) -> bool:
    """Predicted under the warmup cap and inside the GPU's memory, with the peak's worst measured under-prediction."""
    return fits_warmup(genome) and predicted_peak_gib(genome) * PEAK_PREDICTION_ERROR <= PEAK_CAP_GIB


def admissible(make: Callable[[], Genome]) -> Genome | None:
    """The genome `make` draws, or None when its choices do not compose (a family's numbers that contradict each other,
    or families that conflict): the feasible set is what resolves, so a proposer skips such a draw and draws again."""
    try:
        return make()
    except ValueError:  # GenomeConflict and a config's own refusal; every value a proposer draws is inside its bounds
        return None


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


def sample_value(dimension: Dimension, rng: np.random.Generator) -> GenomeValue:
    if dimension.categorical:
        return dimension.choices[int(rng.integers(len(dimension.choices)))]
    if dimension.log:
        value = float(np.exp(rng.uniform(np.log(dimension.low), np.log(dimension.high))))
    else:
        value = float(rng.uniform(dimension.low, dimension.high))
    return round(value) if dimension.integer else float(f"{value:.4g}")


def random_genome(space: list[Dimension], rng: np.random.Generator, tries: int = 200) -> Genome:
    """A uniform draw over the space, the families in a conflict drawn again until the genome composes (a loop on a
    mixing stream, a layout past the depth, heads that do not divide the width); a config's own refusal draws all again.
    Redrawing only the conflicting families keeps the rest of the draw: a whole new draw composes one time in ten."""
    changes = {dimension.key: sample_value(dimension, rng) for dimension in space}
    last = ""
    for attempt in range(tries):
        try:
            return written(default_genome(), changes)
        except GenomeConflict as conflict:
            # Every second redraw takes the families conflicts are with as well, so a family whose open choices cannot
            # fit the drawn shape (retirement can leave it so) gets a new shape instead of redrawing forever.
            families: list[str] | None = conflict.families + (list(CONFLICT_PARTNERS) if attempt % 2 else [])
            last = str(conflict)
        except ValueError as refusal:
            families, last = None, str(refusal)
        changes.update({dimension.key: sample_value(dimension, rng) for dimension in space
                        if families is None or dimension.family in families})
    raise RuntimeError(f"no composing genome after {tries} redraws; the last refusal: {last}")


def perturbed(dimension: Dimension, value: GenomeValue, rng: np.random.Generator) -> GenomeValue:
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
    """Each active coordinate changes with probability `rate`; at least one always does. A number inert under the
    genome's variants (`dimension_active`) is never the change, since a run would not feel it."""
    active = [dimension for dimension in space if dimension_active(genome, dimension)]
    chosen = [dimension for dimension in active if rng.random() < rate] or [active[int(rng.integers(len(active)))]]
    changes = {dimension.key: perturbed(dimension, read(genome, dimension.key), rng) for dimension in chosen}
    return written(genome, changes), sorted(changes)


def crossover(first: Genome, second: Genome, rng: np.random.Generator) -> tuple[Genome, list[str]]:
    """Each family's whole config from one parent or the other, so a family's variant keeps its own numbers."""
    from_second = [family for family in FAMILIES if rng.random() < 0.5]
    child = {family: dict(second[family] if family in from_second else first[family]) for family in FAMILIES}
    return resolve({"families": child}), from_second


class Child(NamedTuple):
    genome: Genome
    from_second: list[str]  # the families the second parent gave
    mutated: list[str]  # the coordinates the mutation moved
    repaired: bool  # costly choices were put back to fit the warmup cap


def breed(first: Genome, second: Genome, space: list[Dimension], rng: np.random.Generator) -> Child | None:
    """A child of two genomes: each family's config from one parent, a mutation, retired choices reset, and costly ones
    repaired toward the warmup cap (parents over it breed children over it); None when its choices do not compose."""
    try:
        child, from_second = crossover(first, second, rng)
        child, mutated = mutate(child, space, rng, rate=1.5 / len(space))
        child = within(child, space)
        repaired = not fits_warmup(child)
        if repaired:
            child = warmup_repaired(child, rng)
    except ValueError:  # GenomeConflict or a config's refusal, as in `admissible`
        return None
    return Child(child, from_second, mutated, repaired)


def candidate_pool(space: list[Dimension], parents: list[Observation], rng: np.random.Generator, size: int) -> list[Genome]:
    """Half uniform draws over the space, half mutations of the five best `parents` (best first); a draw that does not
    compose is skipped, so the pool can come out smaller than `size`."""
    draws = [admissible(lambda: random_genome(space, rng)) for _ in range(size // 2)]
    for _ in range(size // 2):
        parent = parents[int(rng.integers(min(5, len(parents))))]
        draws.append(admissible(lambda: mutate(parent.genome, space, rng, rate=2 / len(space))[0]))  # noqa: B023, called at once
    return [genome for genome in draws if genome is not None]


def feasible(pool: list[Genome], space: list[Dimension], seen: set[str], rng: np.random.Generator) -> list[Genome]:
    """The pool's distinct genomes with retired choices reset and costly ones repaired, unseen, composing and predicted
    to fit the warmup cap and the GPU's memory."""
    repaired = [admissible(lambda: warmup_repaired(within(genome, space), rng)) for genome in pool]  # noqa: B023, called at once
    distinct = {genome_sha256(genome): genome for genome in repaired if genome is not None}
    kept = [genome for digest, genome in distinct.items() if digest not in seen and fits_machine(genome)]
    if not kept:
        raise ValueError(f"none of {len(pool)} candidates composes, fits the machine and is new; widen the space")
    return kept


@dataclass
class Observation:
    name: str
    genome: Genome
    value: float  # val_select of the selected weights
    budget: int | None = None  # the run's seconds on the clock; None is the track's full budget
    other_track: bool = False  # a result the state carries from another track: read by its budget, never an incumbent

    @property
    def full(self) -> bool:
        return not self.other_track and (self.budget is None or self.budget == FULL_BUDGET)


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


class RandomProposer:
    """Uniform draws over the open space, warmup-repaired and distinct: a track's first design, before the Bayes and
    genetic proposers have the observations they need (three and two at the full budget)."""

    name = "random"

    def __init__(self, space: list[Dimension]) -> None:
        self.space = space

    def propose(self, observations: list[Observation], count: int, rng: np.random.Generator) -> list[Proposal]:
        seen = {genome_sha256(observation.genome) for observation in observations}
        parents = [observations[0].name] if observations else []
        proposals: list[Proposal] = []
        for _ in range(1000 * count):
            if len(proposals) == count:
                break
            genome = admissible(lambda: warmup_repaired(within(random_genome(self.space, rng), self.space), rng))
            if genome is None:
                continue
            digest = genome_sha256(genome)
            if digest in seen or not fits_machine(genome):
                continue
            seen.add(digest)
            changes = genome_document(genome)["families"]
            proposals.append(Proposal(genome, parents, f"random draw over the open space: {json.dumps(changes, sort_keys=True)}"))
        return proposals


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
    """A squared-exponential GP over the encoding; production reads full-budget results only, and the next pick is the pool's
    best expected improvement under its posterior."""

    name = "bayes"
    reads_every_budget = False

    def __init__(self, space: list[Dimension], pool: int = 4000) -> None:
        self.space = space
        self.pool = pool

    def candidates(self, parents: list[Observation], rng: np.random.Generator) -> list[Genome]:
        """The pool the posterior ranks: uniform draws, and mutations of the five best parents by posterior mean."""
        return candidate_pool(self.space, parents, rng, self.pool)

    def view(self, observations: list[Observation]) -> tuple[np.ndarray, np.ndarray, float, float | None]:
        """What the GP fits: the encoded rows, their targets, the level a prediction's target leaves out, and the extra
        column a full-budget query carries, None when the rows have no such column."""
        # Returns (n, D), (n,), and two numbers.
        X = np.stack([encode(observation.genome, self.space) for observation in observations])  # (n, D)
        return X, np.array([observation.value for observation in observations]), 0.0, None  # (n, D), (n,)

    def propose(self, observations: list[Observation], count: int, rng: np.random.Generator) -> list[Proposal]:
        full = np.array([observation.full for observation in observations], dtype=bool)  # (n,)
        if full.sum() < 3:
            raise ValueError("the GP needs at least three observations at the full budget")
        X, y, level, query_column = self.view(observations)  # (n, D), (n,)
        seen = {genome_sha256(observation.genome) for observation in observations}
        fitted_mean = gp_posterior(X, y, X)[0]  # (n,)
        rows = np.flatnonzero(full)  # (f,)
        ranked = rows[np.argsort(fitted_mean[rows])]  # by posterior mean, not by the noisy observation
        incumbent = observations[int(ranked[0])]
        pool = feasible(self.candidates([observations[int(row)] for row in ranked], rng), self.space, seen, rng)
        X_pool = np.stack([encode(genome, self.space) for genome in pool])  # (m, D)
        if query_column is not None:
            X_pool = np.column_stack([X_pool, np.full(len(pool), query_column)])  # (m, D + 1)
        proposals: list[Proposal] = []
        X_fit, y_fit, full_fit = X, y, full
        for _ in range(count):
            mean, sd, hyper = gp_posterior(X_fit, y_fit, X_pool)  # (m,), (m,)
            best_mean = float(gp_posterior(X_fit, y_fit, X_fit)[0][full_fit].min())
            gains = expected_improvement(mean, sd, best_mean)  # (m,)
            choice = int(np.argmax(gains))
            nearest = observations[int(np.argmin(((X - X_pool[choice]) ** 2).sum(-1)))]
            changes = genome_document(pool[choice])["families"]
            proposals.append(Proposal(pool[choice], [incumbent.name, nearest.name],
                                      f"GP expected improvement {gains[choice]:.5f}: predicted val_select {mean[choice] + level:.4f} "
                                      f"+- {sd[choice]:.4f} against the best posterior mean {best_mean + level:.4f} ({incumbent.name}); "
                                      f"nearest observed {nearest.name}; length {hyper['length']}, noise sd {hyper['noise_sd']:.4f}; "
                                      f"changes {json.dumps(changes, sort_keys=True)}"))
            # The kriging believer: the chosen point joins the data at its predicted mean, so the next pick looks elsewhere.
            X_fit = np.vstack([X_fit, X_pool[choice]])  # (n + 1, D)
            y_fit = np.append(y_fit, mean[choice])  # (n + 1,)
            full_fit = np.append(full_fit, True)  # (n + 1,)
            X_pool = np.delete(X_pool, choice, axis=0)  # (m - 1, D)
            del pool[choice]
        return proposals


class RungReadingProposer(BayesianProposer):
    """The same GP, reading every result on the track, its rungs too, with the budget one more dimension and each budget's
    level removed, since a stationary GP has no trend. The replay's `track` reading: on 54 adaptive rungs it ranks at rho
    0.640 against production's 0.089 (`2026-11-03_tabular_surrogate_replay`), and it still proposes for the full budget."""

    name = "bayes_rungs"
    reads_every_budget = True

    def view(self, observations: list[Observation]) -> tuple[np.ndarray, np.ndarray, float, float | None]:
        # Returns (n, D + 1), (n,), and two numbers.
        X, y, _, _ = super().view(observations)  # (n, D), (n,)
        return budget_view(X, y, [observation.budget or FULL_BUDGET for observation in observations])  # (n, D + 1), (n,), and two numbers


class HybridProposer(RungReadingProposer):
    """The genetic algorithm proposes and the GP disposes: tournament parents from the full-budget results ranked by
    posterior mean, family-wise crossover and mutation breed the pool, and the rung-reading GP's expected improvement
    picks the batch from it, the kriging believer spreading the picks. Structure (which modules, which shape) moves by
    crossover, numbers by mutation, and the surrogate spends runs only where it expects a gain. Logan, 2026-11-23: "some
    sort of genetic algorithm in combination with Bayesian hyperparameter search to search over every aspect of the
    speed run"."""

    name = "hybrid"
    tournament = 3

    def candidates(self, parents: list[Observation], rng: np.random.Generator) -> list[Genome]:
        # parents: best posterior mean first, so a tournament's winner is its lowest index.
        children = []
        for _ in range(self.pool):
            first, second = (parents[int(rng.integers(len(parents), size=self.tournament).min())] for _ in range(2))
            child = breed(first.genome, second.genome, self.space, rng)
            if child is not None:
                children.append(child.genome)
        return children


def budget_view(X: np.ndarray, y: np.ndarray, budgets: list[int]) -> tuple[np.ndarray, np.ndarray, float, float | None]:
    """The rows with their budget as a column on [0, 1], by log2, and each budget's mean removed from the targets; the
    full budget's mean and column are what a prediction there adds back and carries."""
    # X: (n, D); y: (n,); returns (n, D + 1), (n,), and two numbers.
    seconds = np.array(budgets, dtype=float)  # (n,)
    levels = {budget: float(y[seconds == budget].mean()) for budget in np.unique(seconds)}
    assert FULL_BUDGET in levels, "no row at the full budget"
    position = np.log2(seconds)  # (n,)
    low, high = position.min(), position.max()
    spread = high - low if high > low else 1.0
    centered = y - np.array([levels[budget] for budget in seconds])  # (n,)
    full_column = float((np.log2(FULL_BUDGET) - low) / spread)
    return np.column_stack([X, (position - low) / spread]), centered, levels[FULL_BUDGET], full_column  # (n, D + 1), (n,)


NOVELTY_EVERY = 4  # one pick in this many must hold something no result holds; a batch is 4


class Surrogate(Protocol):
    def quantiles(self, context: pd.DataFrame, values: np.ndarray, query: pd.DataFrame) -> np.ndarray:
        # values: (n,); returns (m, q), the predictive distribution on evenly spaced levels.
        ...


def genome_frame(genomes: list[Genome], budgets: list[int], space: list[Dimension]) -> pd.DataFrame:
    """The genomes as named columns: categories as text, numbers as floats, a number inert under its genome's variants
    missing, and the budget's log2. What a tabular foundation model reads, in place of the GP's one-hot encoding."""
    rows = []
    for genome, budget in zip(genomes, budgets, strict=True):
        row: dict[str, Any] = {"x:log2_budget": math.log2(budget)}
        for dimension in space:
            value = read(genome, dimension.key)
            if dimension.categorical:
                row[f"g:{dimension.key}"] = str(value)
            else:
                row[f"g:{dimension.key}"] = float(value) if dimension_active(genome, dimension) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def informative_frames(context: pd.DataFrame, query: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both tables without the columns constant in the context, categories shared between the two. A choice the context has never
    seen, in a column it varies, stays in the query as a category of its own: the model has to say it does not know."""
    kept = [column for column in context.columns if context[column].nunique(dropna=True) > 1]
    left, right = context[kept].copy(), query[kept].copy()
    for column in kept:
        if not pd.api.types.is_numeric_dtype(context[column]):
            known = sorted(set(left[column].dropna().astype(str)) | set(right[column].dropna().astype(str)))
            left[column], right[column] = pd.Categorical(left[column], categories=known), pd.Categorical(right[column], categories=known)
    return left, right


def novel_rows(context: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    """Which query rows hold what the context never has: a choice it never took, or another number in a column it never varied."""
    # Returns (m,).
    novel = np.zeros(len(query), dtype=bool)  # (m,)
    for column in context.columns:
        if not pd.api.types.is_numeric_dtype(context[column]):
            novel |= ~query[column].isin(context[column]).to_numpy()  # (m,)
            continue
        known = context[column].dropna().unique()
        if len(known) < 2:
            values = query[column].to_numpy()  # (m,)
            novel |= ~np.isnan(values) & ~np.isin(values, known)  # (m,)
    return novel  # (m,)


class TabularProposer:
    """A tabular foundation model reading every result, on this track and any other the state carries, named columns as they
    are. The same pool, warmup rule and believer as the GP's. The pick is the lowest predicted median, not the best expected
    improvement: on the real rows Kumo's quantiles span about 0.2 nats where a genome far from the data is concerned, against the GP's
    sd of 0.009, so its improvement picks were the widest distributions, with medians 0.05 nats behind the record. A pick joins the
    context at its predicted median.

    A model conditioned on the context has nothing to say about a column the context never varied, and it has no distance that
    grows toward the unseen, as the GP's kernel does. So every `NOVELTY_EVERY`-th pick, counted over its batches by `picks_made`, must
    hold something unseen, the best such genome by the same acquisition: mutated parents, in practice, as random draws rarely score
    well. SMAC alternates random configurations with its model's picks to give later models unbiased data (Hutter et al., 2011); this
    aims that idea at the unseen. Every other pick must hold nothing unseen: the model cannot see such a change, so it would ride along
    at random (the head-to-head's `auto12_kumo_01` and `auto13_kumo_01`, predicted 2.709, scored 2.7189 and 2.7524)."""

    name = "kumo"
    reads_every_budget = True
    reads_other_tracks = True

    def __init__(self, space: list[Dimension], surrogate: Surrogate, pool: int = 4000, picks_made: int = 0) -> None:
        self.space = space
        self.surrogate = surrogate
        self.pool = pool
        self.picks_made = picks_made

    def propose(self, observations: list[Observation], count: int, rng: np.random.Generator) -> list[Proposal]:
        full = [observation for observation in observations if observation.full]
        if len(full) < 3:
            raise ValueError("the tabular model needs at least three observations at the full budget")
        seen = {genome_sha256(observation.genome) for observation in observations}
        ranked = sorted(full, key=lambda observation: observation.value)
        incumbent = ranked[0]
        pool = feasible(candidate_pool(self.space, ranked, rng, self.pool), self.space, seen, rng)
        context = genome_frame([observation.genome for observation in observations],
                               [observation.budget or FULL_BUDGET for observation in observations], self.space)
        values = np.array([observation.value for observation in observations])  # (n,)
        proposals: list[Proposal] = []
        for index in range(count):
            query = genome_frame(pool, [FULL_BUDGET] * len(pool), self.space)
            informative_context, informative_query = informative_frames(context, query)
            quantiles = self.surrogate.quantiles(informative_context, values, informative_query)  # (m, q)
            medians = np.median(quantiles, axis=1)  # (m,)
            novel = novel_rows(context, query)  # (m,)
            exploring = (self.picks_made + index + 1) % NOVELTY_EVERY == 0 and bool(novel.any())
            order = np.argsort(medians, kind="stable")  # (m,), the best predicted first
            choice = int(order[novel[order] == (exploring or bool(novel.all()))][0])
            kind = f"unseen pick, among {int(novel.sum())} genomes holding something no result holds: " if exploring else ""
            changes = genome_document(pool[choice])["families"]
            proposals.append(Proposal(pool[choice], [incumbent.name],
                                      f"{kind}{self.name} lowest predicted median val_select {medians[choice]:.4f}, "
                                      f"5 to 95 percent {np.quantile(quantiles[choice], 0.05):.4f} to {np.quantile(quantiles[choice], 0.95):.4f}, "
                                      f"against the best observed {incumbent.value:.4f} ({incumbent.name}); changes {json.dumps(changes, sort_keys=True)}"))
            # The believer, as the GP's: the pick joins the context at its predicted median, so the next pick looks elsewhere.
            context = pd.concat([context, query.iloc[[choice]]], ignore_index=True)
            values = np.append(values, medians[choice])  # (n + 1,)
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
            bred = breed(first.genome, second.genome, self.space, rng)
            if bred is None:
                continue
            child, from_second, mutated, repaired = bred
            digest = genome_sha256(child)
            if digest in seen or not fits_machine(child):
                continue
            seen.add(digest)
            changes = genome_document(child)["families"]
            proposals.append(Proposal(child, [first.name, second.name],
                                      f"child of {first.name} ({first.value:.4f}) and {second.name} ({second.value:.4f}); "
                                      f"families from {second.name}: {', '.join(from_second) or 'none'}; mutated {', '.join(mutated)}; "
                                      f"{'repaired to the warmup cap; ' if repaired else ''}changes {json.dumps(changes, sort_keys=True)}"))
        if len(proposals) < count:
            raise RuntimeError(f"{len(proposals)} new children within the warmup cap after {1000 * count} tries; widen the space")
        return proposals


# The state file and the ledger.

def apply_track(state: SearchState) -> Track:
    """Set the module's budgets to the state's track (states written before the tracks are the 10-minute one). A state may
    name its own W&B group, as an experiment on a track does, its own batch-name `prefix`, and its own `final_splits` for a
    full-budget run."""
    global TRACK, FULL_BUDGET, RUNG_BUDGET
    TRACK = TRACKS[state.setdefault("track", "10min")]
    TRACK = {**TRACK, **{key: state[key] for key in ("group", "prefix", "final_splits") if key in state}}
    FULL_BUDGET = TRACK["budget"]
    RUNG_BUDGET = TRACK["rung"]
    return TRACK


def load_state(path: Path, track: str = "10min") -> SearchState:
    """The state at `path`, its track applied; a path without a state starts an empty one on `track`."""
    if not path.exists():
        state: SearchState = {"schema": STATE_SCHEMA, "track": track, "batches": {}, "proposals": {}, "seeded": {}, "retired": []}
    else:
        state = json.loads(path.read_text(encoding="utf-8"))
        assert state["schema"] == STATE_SCHEMA
    apply_track(state)
    return state


def save_state(state: SearchState, path: Path) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def observations_of(state: SearchState) -> list[Observation]:
    """Every genome with a valid result at the full budget: the search's own and the seeded hand-written runs."""
    found = [Observation(name, resolve({"families": seeded["genome"]}), seeded["val_select"])
             for name, seeded in state["seeded"].items()]
    for name, proposal in state["proposals"].items():
        full = proposal["results"].get(str(FULL_BUDGET))
        if full and full["valid"] and full["val_select"] is not None:
            found.append(Observation(name, resolve({"families": proposal["genome"]}), full["val_select"]))
    return found


def observations_at_every_budget(state: SearchState) -> list[Observation]:
    """`observations_of`, then every valid result of a shorter rung on this track, each at its own budget."""
    found = observations_of(state)
    for name, proposal in state["proposals"].items():
        for budget, outcome in proposal["results"].items():
            if int(budget) != FULL_BUDGET and outcome["valid"] and outcome["val_select"] is not None:
                found.append(Observation(name, resolve({"families": proposal["genome"]}), outcome["val_select"], int(budget)))
    return found


def observations_on_other_tracks(state: SearchState) -> list[Observation]:
    """The results of other tracks a state carries, frozen when it was seeded, each at its own budget. The clock and the split are the
    same benchmark's, so a model with budget as a column can read them: Kumo's replay lead (rho 0.801) came from this reading."""
    return [Observation(row["name"], resolve({"families": row["genome"]}), row["val_select"], row["budget"], other_track=True)
            for row in state.get("other_track_rows", [])]


def observations_read_by(proposer: Proposer, state: SearchState) -> list[Observation]:
    """What a proposer reads: production's full-budget results, or the rungs too, or other tracks' results as well."""
    if getattr(proposer, "reads_other_tracks", False):
        return observations_at_every_budget(state) + observations_on_other_tracks(state)
    return observations_at_every_budget(state) if getattr(proposer, "reads_every_budget", False) else observations_of(state)


def unproven_keys(state: SearchState, genome: Genome) -> list[str]:
    """What a genome holds that no valid result has, as `retire` keys: a choice none took, or another number in a column none varied.
    What a result has held compiled, fit the memory and ran; these may not, so the loop tries them before they take a turn.
    The default is the record's choice, which ran, so it is never unproven: retiring `loop.variant=none` beside
    `stream.variant=single` would leave no loop that composes."""
    rows = observations_at_every_budget(state) + observations_on_other_tracks(state)
    default = default_genome()
    keys = []
    for dimension in SPACE:
        value = read(genome, dimension.key)
        held = {read(row.genome, dimension.key) for row in rows} | {read(default, dimension.key)}
        if value in held or not dimension_active(genome, dimension):
            continue
        if dimension.categorical:
            keys.append(f"{dimension.key}={value}")
        elif len(held) < 2:
            keys.append(dimension.key)
    return keys


def incumbent_of(state: SearchState) -> Observation:
    return min(observations_of(state), key=lambda observation: observation.value)


def run_name(batch: str, index: int, budget: int) -> str:
    """A proposal's run at a budget: `<batch>_<index>`, with `_r<budget>` below the full budget."""
    return rung_run(f"{batch}_{index:02d}", budget)


def rung_run(proposal: str, budget: int) -> str:
    """The run of a named proposal at a budget."""
    return proposal + ("" if budget == FULL_BUDGET else f"_r{budget}")


def parse_run_name(run: str) -> RunName:
    """The proposal, batch, index and budget a run name holds; a name without the `_r` suffix is at the full budget."""
    match = RUN_NAME.fullmatch(run)
    if match is None:
        raise ValueError(f"{run} is not a search run name")
    budget = match["budget"]
    return RunName(match["proposal"], match["batch"], int(match["index"]), int(budget) if budget else FULL_BUDGET)


def due_at(proposal: ProposalRecord, rung: int) -> bool:
    """Whether a proposal still has a run to do at a rung: none recorded there, and promoted when it is the full budget."""
    return str(rung) not in proposal["results"] and (rung != FULL_BUDGET or proposal["promoted"])


def write_queue_entry(name: str, genome: Genome, hypothesis: str, budget: int) -> None:
    """candidates/queue/<name>.json (the genome), .txt (the hypothesis) and .args (the budget and splits)."""
    queue = PROJECT / "candidates" / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    (queue / f"{name}.json").write_text(json.dumps(genome_document(genome), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (queue / f"{name}.txt").write_text(hypothesis.replace("\n", " ") + "\n", encoding="utf-8", newline="\n")
    splits = TRACK.get("final_splits", []) if budget == FULL_BUDGET else ["val_select"]  # none: the benchmark's every split
    extra = (f"--budget {budget} --eval-cap {TRACK['eval_cap']} --eval-interval {TRACK['eval_interval']} "
             f"--experiment-group {TRACK['group']}" + (f" --final-splits {' '.join(splits)}" if splits else ""))
    (queue / f"{name}.args").write_text(extra + "\n", encoding="utf-8", newline="\n")


def propose(state: SearchState, proposer: Proposer, batch: str, count: int, seed: int, rungs: list[int]) -> list[str]:
    if batch in state["batches"]:
        raise ValueError(f"batch {batch} exists")
    rng = np.random.default_rng(seed)
    proposals = proposer.propose(observations_read_by(proposer, state), count, rng)
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


def next_rung(state: SearchState, batch: str) -> int | None:
    """The batch's first rung with a run still to do, or None when every rung is done."""
    entry = state["batches"][batch]
    return next((rung for rung in entry["rungs"] if any(due_at(state["proposals"][name], rung) for name in entry["names"])), None)


def due_runs(state: SearchState, batch: str, rung: int) -> list[DueRun]:
    """The batch's runs still to do at a rung, in proposal order."""
    return [DueRun(batch, rung_run(name, rung), rung) for name in state["batches"][batch]["names"] if due_at(state["proposals"][name], rung)]


def queue_runs(state: SearchState, runs: list[DueRun]) -> list[str]:
    """Write each run's queue entry, its proposal's genome and reason at the run's budget, and record one shipment per
    batch, at the batch's first listed rung; returns the run names in order."""
    for due in runs:
        proposal = state["proposals"][parse_run_name(due.run).proposal]
        hypothesis = f"{proposal['proposer']} proposal ({due.rung} s): {proposal['reason']}"
        write_queue_entry(due.run, resolve({"families": proposal["genome"]}), hypothesis, due.rung)
    for batch in dict.fromkeys(due.batch for due in runs):
        record_shipment(state, batch, next(due.rung for due in runs if due.batch == batch), [due.run for due in runs if due.batch == batch])
    return [due.run for due in runs]


def record_shipment(state: SearchState, batch: str, rung: int, runs: list[str]) -> None:
    state["batches"][batch]["shipments"].append({"budget": rung, "names": runs, "utc": datetime.now(UTC).isoformat()})


def ship(state: SearchState, batches: list[str], budget: int | None = None) -> list[str]:
    """Write the batches' queue entries, each at its next rung (or `budget`), and run the launcher with one queue.sh
    over all of them, so the box runs them back to back; returns the run names. A batch with nothing due still records
    its empty shipment."""
    names = []
    for batch in batches:
        rung = budget or next_rung(state, batch)
        if rung is None:
            raise ValueError(f"batch {batch} has no run left to do")
        due = due_runs(state, batch, rung)
        if not due:
            record_shipment(state, batch, rung, [])
        names += queue_runs(state, due)
    launch_queue(names)
    return names


def launch_queue(names: list[str]) -> None:
    """Run queue.sh over the runs through $SPEEDRUN_LAUNCHER, or print the command when it is unset."""
    command = ["bash", "queue.sh", "--wait", *names]
    launcher = os.environ.get("SPEEDRUN_LAUNCHER")
    if launcher is None:
        print("SPEEDRUN_LAUNCHER is unset; run on the box from the project:", shlex.join(command))
    else:
        subprocess.run([*shlex.split(launcher), *command], check=True)


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


def collect(state: SearchState, batch: str, results_dir: Path, ledger: Path) -> None:
    """Read every finished run of the batch into the state and the ledger, then promote the top half of a finished
    short rung."""
    for name in state["batches"][batch]["names"]:
        for budget in state["batches"][batch]["rungs"]:
            if str(budget) in state["proposals"][name]["results"]:
                continue
            path = fetch(rung_run(name, budget), results_dir)
            if path is not None:
                record_result(state, batch, name, budget, json.loads(path.read_text(encoding="utf-8")), ledger)
    promote_top_half(state, batch)


def record_result(state: SearchState, batch: str, name: str, budget: int, run_result: dict[str, Any], ledger: Path) -> None:
    """One run's result.json into its proposal's results and one ledger row, with its verdict against the incumbent."""
    import record  # a script beside this one, read only when a result is

    proposal = state["proposals"][name]
    val_select = run_result["metrics"].get("val_select", {}).get("loss")
    valid = run_result["status"] == "completed" and bool(run_result["clock"]["valid"])
    incumbent = incumbent_of(state) if observations_of(state) else None  # before this result joins them
    proposal["results"][str(budget)] = {
        "run": rung_run(name, budget), "val_select": val_select, "test": run_result["metrics"].get("test", {}).get("loss"),
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


def promote_top_half(state: SearchState, batch: str) -> None:
    """Once every run of a batch's short rung has a result and none is promoted, promote the lower half of its val_select
    losses, an invalid run ranked last."""
    rungs = state["batches"][batch]["rungs"]
    if len(rungs) > 1:
        short = str(rungs[0])
        scored = [(state["proposals"][name]["results"].get(short), name) for name in state["batches"][batch]["names"]]
        if all(rung is not None for rung, _ in scored) and not any(state["proposals"][name]["promoted"] for _, name in scored):
            ranked = sorted((rung["val_select"] if rung["valid"] and rung["val_select"] is not None else math.inf, name)
                            for rung, name in scored)  # type: ignore[index]
            for _, name in ranked[: max(1, len(ranked) // 2)]:
                state["proposals"][name]["promoted"] = True


def mark_crashed(state: SearchState, run: str, note: str) -> None:
    """A run that left no result.json: its rung counts as invalid, so promotion ranks it last; the launch log says why."""
    parsed = parse_run_name(run)
    state["proposals"][parsed.proposal]["results"][str(parsed.budget)] = {
        "run": run, "val_select": None, "test": None, "steps": 0, "warmup_seconds": None, "valid": False, "crash": note}


def effects_table(state: SearchState, batch: str, budget: int | None = None) -> str:
    """The screening batch's main effects and leading interactions on val_select at a budget, by default the full one, in Markdown."""
    key = str(budget or FULL_BUDGET)
    names = [name for name in state["batches"][batch]["names"] if key in state["proposals"][name]["results"]]
    proposals = [state["proposals"][name] for name in names]
    factors = list(proposals[0]["factors"])
    levels = np.array([[proposal["factors"][factor] for factor in factors] for proposal in proposals])  # (n, k)
    values = np.array([proposal["results"][key]["val_select"] for proposal in proposals])  # (n,)
    tests = [proposal["results"][key]["test"] for proposal in proposals]
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


PROPOSER_NAMES = ("screening", "hybrid", "bayes", "bayes_rungs", "kumo", "genetic", "random", "manual")
SPACE_PROPOSERS: dict[str, Callable[[list[Dimension]], Proposer]] = {
    "hybrid": HybridProposer, "bayes": BayesianProposer, "bayes_rungs": RungReadingProposer, "genetic": GeneticProposer,
    "random": RandomProposer}


def proposer_of(name: str, space: list[Dimension], picks_made: int = 0, factors: Sequence[str] = (),
                genomes: Sequence[Path] = (), hypotheses: Sequence[str] = ()) -> Proposer:
    """One proposer by name, for the command line and the loop alike; a tabular model's package loads only when its arm
    is asked for. `picks_made` is how many picks the tabular arm has made on this state, which counts toward its unseen
    picks across batches; `factors` are screening's, `genomes` and `hypotheses` manual's. An unknown name is a KeyError."""
    if name == "kumo":
        from tabular_surrogate import KumoSurrogate  # its package lives only in the proposer's own image

        return TabularProposer(space, KumoSurrogate(), picks_made=picks_made)
    if name == "screening":
        return ScreeningProposer(list(factors))
    if name == "manual":
        return ManualProposer(list(genomes), list(hypotheses))
    return SPACE_PROPOSERS[name](space)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line; `argv` None reads the process's arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", type=Path, required=True, help="the search state JSON, beside the campaign ledger")
    parser.add_argument("--track", choices=sorted(TRACKS), default="1h",
                        help="the track of a state started by this call; the 1-hour track is the headline since 2026-11-23")
    commands = parser.add_subparsers(dest="command", required=True)
    proposing = commands.add_parser("propose")
    proposing.add_argument("--proposer", choices=PROPOSER_NAMES, required=True)
    proposing.add_argument("--batch", required=True)
    proposing.add_argument("--count", type=int, default=8)
    proposing.add_argument("--seed", type=int, default=0)
    proposing.add_argument("--factors", nargs="+", default=[], help="screening: the switches, 2 to 11")
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
    effects.add_argument("--budget", type=int, default=None, help="the rung to read; the full budget by default")
    commands.add_parser("status")
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_arguments()
    state = load_state(arguments.state, arguments.track)
    space = open_space(state["retired"])
    if arguments.command == "propose":
        rungs = arguments.rungs or ([FULL_BUDGET] if arguments.proposer == "screening" else [RUNG_BUDGET, FULL_BUDGET])
        proposer = proposer_of(arguments.proposer, space, factors=arguments.factors, genomes=arguments.genomes,
                               hypotheses=arguments.hypotheses)
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
        print(effects_table(state, arguments.batch, arguments.budget))
    observations = observations_of(state)
    if observations:
        best = min(observations, key=lambda observation: observation.value)
        print(json.dumps({"observations": len(observations), "incumbent": best.name, "val_select": best.value,
                          "retired": state["retired"]}))
    save_state(state, arguments.state)


if __name__ == "__main__":
    main()
