"""A tiny synthetic source dataset in the benchmark's parquet shape, prepared once for the test session."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from speedrunning_plms import benchmark


RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def synthetic_frame(count: int, families: int, seed: int, min_length: int = 50, max_length: int = 160) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    for index in range(count):
        length = int(generator.integers(min_length, max_length))
        rows.append({"name": f"p{seed}_{index}", "seqs": "".join(generator.choice(list(RESIDUES), length)),
                     "three_di": "".join(generator.choice(list(RESIDUES), length)), "family": f"fam{index % families}",
                     "role": "rep" if index < families else "member"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = tmp_path_factory.mktemp("source")
    synthetic_frame(300, 60, 1).to_parquet(source / "train.parquet", index=False)
    synthetic_frame(40, 20, 2).to_parquet(source / "validation.parquet", index=False)
    synthetic_frame(40, 20, 3).to_parquet(source / "test.parquet", index=False)
    return source


@pytest.fixture(scope="session")
def tiny_data(tiny_source: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    benchmark_dir = tmp_path_factory.mktemp("data") / "afdb_tiny"
    benchmark.prepare(benchmark_dir, tiny_source)
    return benchmark_dir
