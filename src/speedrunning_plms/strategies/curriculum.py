"""The curriculum family: what the run reads first, in time or in order.

`short_first` trains on windows of `short_ratio` times the batch's window for the first `fraction` of the clock,
then on the full window (027, A11; modded-nanogpt records 13, 46): cheaper attention early buys steps, and 009 lost on
long proteins only because it never saw 1,024. Both shapes run through one dynamic-shape graph, compiled in the warmup
outside the clock.

The two tempered orders draw one weighted random permutation of the training proteins on the CPU before the warmup,
so no protein repeats, none is dropped, and a step costs what it did; a 10-minute run reads about a quarter to a third
of the stream, so the order decides which proteins it sees. Each protein's weight w sets its place by an exponential
race: it arrives at E / w with E ~ Exp(1), and the proteins are read by arrival (Efraimidis and Spirakis's weighted
sampling without replacement). Weights read only what the benchmark's training files hold, the family labels and the
lengths, so no taxonomy enters beyond what family size carries.

- `family_tempered`: w = size(family)^-alpha (ESM-2 reads uniformly over clusters; ProtBFN weights a cluster by the
  square root of its size, which is alpha = 0.5 per protein). alpha = 0 is a uniform reshuffle, alpha = 1 gives every
  family the same total weight.
- `length_tempered`: w = residues^beta; beta > 0 reads long proteins earlier, beta < 0 short ones (Dataset
  Decomposition, arXiv 2405.13226, mixes lengths on purpose). beta = 0 is a uniform reshuffle.

Symbols: n proteins.
"""

from __future__ import annotations

import json
import numpy as np

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from speedrunning_plms.strategies.strategy import Strategy, StrategyConfig


FAMILIES_FILE = "train_families.json"  # one family label per training protein in stream order (`benchmark.prepare`)


@dataclass(frozen=True)
class CurriculumConfig(StrategyConfig):
    variant: str = "none"
    fraction: float = 0.4  # share of the clock on short windows
    short_ratio: float = 0.5  # the short window as a share of the full one
    alpha: float = 0.5  # family_tempered: weight size(family)^-alpha
    beta: float = 0.0  # length_tempered: weight residues^beta
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "short_first", "family_tempered", "length_tempered")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"fraction": (0.05, 0.9), "short_ratio": (0.125, 0.5),
                                                        "alpha": (0.0, 1.0), "beta": (-0.5, 0.5)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.short_ratio not in (0.125, 0.25, 0.5):
            raise ValueError(f"short_ratio must be 1/8, 1/4 or 1/2 so windows stay powers of two, not {self.short_ratio}")


def tempered_order(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random permutation in which a protein of weight w arrives at E / w, E ~ Exp(1): weighted sampling without
    replacement."""
    # weights: (n,) positive; returns (n,) protein indices in reading order.
    arrivals = rng.exponential(size=len(weights)) / weights  # (n,)
    return np.argsort(arrivals, kind="stable")  # (n,)


class Curriculum(Strategy):
    family = "curriculum"
    Config = CurriculumConfig
    config: CurriculumConfig

    @property
    def active(self) -> bool:
        return False  # read directly by the loop, never as a hook

    def windows(self, full: int) -> tuple[int, ...]:
        """Every window length a run trains on, each compiled in the warmup."""
        if self.variant != "short_first":
            return (full,)
        return (int(full * self.config.short_ratio), full)

    def window(self, full: int, progress: float) -> int:
        """The window length at this share of the clock."""
        if self.variant == "short_first" and progress < self.config.fraction:
            return int(full * self.config.short_ratio)
        return full

    def compile_options(self) -> dict[str, Any]:
        """One dynamic-shape graph for both windows; the record's default otherwise. 027 compiled a static graph per
        shape and its warmup took 249 s against the 180 s cap."""
        return {"dynamic": True} if self.variant == "short_first" else {}

    def protein_order(self, offsets: np.ndarray, data_dir: Path, seed: int) -> np.ndarray | None:
        """The tempered variants' reading order of the training proteins; None reads the stream as stored."""
        # offsets: (n + 1,); returns (n,) or None.
        rng = np.random.default_rng(seed)
        if self.variant == "family_tempered":
            families = json.loads((data_dir / FAMILIES_FILE).read_text(encoding="utf-8"))
            assert len(families) == len(offsets) - 1, "one family label per training protein"
            sizes = Counter(families)
            family_size = np.fromiter((sizes[family] for family in families), dtype=np.float64, count=len(families))  # (n,)
            return tempered_order(family_size ** -self.config.alpha, rng)  # (n,)
        if self.variant == "length_tempered":
            residues = (np.diff(offsets) - 2).astype(np.float64)  # (n,), CLS and EOS removed
            return tempered_order(residues ** self.config.beta, rng)  # (n,)
        return None
