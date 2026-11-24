"""A tiny synthetic data directory in the benchmark's shape, for CPU smokes and the tests; no download, no GPU.

    python tiny_data.py --output-dir <tiny>

Writes random proteins with random 3Di strings as the three source parquet files, then runs the benchmark's own
`prepare` on them, so `train.py --data-dir <tiny>` reads it like the real data. Its losses mean nothing.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from pathlib import Path

from speedrunning_plms import benchmark
from speedrunning_plms.benchmark.tokens import STANDARD_RESIDUES, THREE_DI_STATES


SPLITS = {"train": (300, 60, 1), "validation": (40, 20, 2), "test": (40, 20, 3)}  # proteins, families, seed per split


def synthetic_frame(count: int, families: int, seed: int, min_length: int = 50, max_length: int = 160) -> pd.DataFrame:
    """count random proteins in the source parquet's columns; the first of each of the families is its representative."""
    generator = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    for index in range(count):
        length = int(generator.integers(min_length, max_length))
        rows.append({"name": f"p{seed}_{index}", "seqs": "".join(generator.choice(list(STANDARD_RESIDUES), length)),
                     "three_di": "".join(generator.choice(list(THREE_DI_STATES), length)), "family": f"fam{index % families}",
                     "role": "rep" if index < families else "member"})
    return pd.DataFrame(rows)


def write_source(source_dir: Path) -> None:
    """The three split parquet files `benchmark.prepare` reads."""
    source_dir.mkdir(parents=True, exist_ok=True)
    for split, (count, families, seed) in SPLITS.items():
        synthetic_frame(count, families, seed).to_parquet(source_dir / f"{split}.parquet", index=False)


def build_tiny_data(output_dir: Path, source_dir: Path) -> None:
    """The prepared data directory at output_dir, from source files written to source_dir."""
    write_source(source_dir)
    benchmark.prepare(output_dir, source_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="the data directory to create; must be empty or absent")
    arguments = parser.parse_args()
    source_dir = arguments.output_dir.with_name(f"{arguments.output_dir.name}_source")
    build_tiny_data(arguments.output_dir, source_dir)
    print(f"tiny data -> {arguments.output_dir} (source parquet in {source_dir})")


if __name__ == "__main__":
    main()
