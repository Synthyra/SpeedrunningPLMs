"""Build the benchmark data from the pinned source dataset; fixed, run once per box.

    python prepare.py --output-dir /data/afdb_v6 [--source-dir /data/source]
    python prepare.py --output-dir /data/afdb_v6_minus_uniprot_2026_04 --exclude-from /data/afdb_v6 --exclusion exclusion.json

Downloads the parquet files of the three splits of `Synthyra/alignment_atlas_families_afdb_v6` at the pinned revision
(HF_TOKEN in the environment: the source is private), joins each split into `{split}.parquet` in the source
directory, then builds the training stream, the held-out tensors with their fixed masks, and `manifest.json`. The
manifest's digests are the benchmark id every result carries.

The second form derives a data version from a prepared one without the training proteins an exclusion names
(`benchmark.exclude_training`): the held-out files are the same bytes, the benchmark id is new. The speedrun trains on
`afdb_v6_minus_uniprot_2026_04`, AFDB v6 without the relatives of `perplexity.py`'s new UniProt proteins.
"""

from __future__ import annotations

import argparse
import json
import re
import numpy as np
import pandas as pd

from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

from speedrunning_plms import benchmark
from speedrunning_plms.benchmark.data import SOURCE_REPO, SOURCE_REVISION


SPLITS = ("train", "validation", "test")
COLUMNS = ["name", "seqs", "three_di", "family", "role"]


def split_of(path: str) -> str | None:
    """The split a Hub parquet file belongs to, from the usual `data/{split}-00000-of-00010.parquet` layout."""
    match = re.search(r"(?:^|/)(train|validation|test)(?:-\d+-of-\d+)?\.parquet$", path)
    if match is None:
        match = re.search(r"(?:^|/)(train|validation|test)/[^/]+\.parquet$", path)
    return match.group(1) if match else None


def download(source_dir: Path) -> None:
    """Fetch the pinned revision's parquet files and join each split into one file; skip splits already present."""
    source_dir.mkdir(parents=True, exist_ok=True)
    missing = [split for split in SPLITS if not (source_dir / f"{split}.parquet").exists()]
    if not missing:
        print("source parquet files exist", flush=True)
        return
    files = HfApi().list_repo_files(SOURCE_REPO, repo_type="dataset", revision=SOURCE_REVISION)
    by_split = {split: sorted(path for path in files if split_of(path) == split) for split in SPLITS}
    for split, paths in by_split.items():
        if not paths:
            raise FileNotFoundError(f"no parquet file of split {split} in {SOURCE_REPO}@{SOURCE_REVISION}: {files}")
    snapshot = Path(snapshot_download(SOURCE_REPO, repo_type="dataset", revision=SOURCE_REVISION,
                                      allow_patterns=[path for paths in by_split.values() for path in paths],
                                      cache_dir=str(source_dir / "hub_cache")))
    for split in missing:
        frame = pd.concat([pd.read_parquet(snapshot / path, columns=COLUMNS) for path in by_split[split]], ignore_index=True)
        frame.to_parquet(source_dir / f"{split}.parquet", index=False)
        print(f"{split}: {len(frame):,} proteins from {len(by_split[split])} files -> {source_dir / f'{split}.parquet'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=None, help="where the parquet files land; default <output-dir>/../source")
    parser.add_argument("--exclude-from", type=Path, default=None,
                        help="derive from this prepared directory instead, without the training proteins --exclusion names")
    parser.add_argument("--exclusion", type=Path, default=None, help="JSON: `indices` (stream indices) and why; perplexity.py relatives writes one")
    arguments = parser.parse_args()
    if (arguments.exclude_from is None) != (arguments.exclusion is None):
        parser.error("--exclude-from and --exclusion go together")
    if arguments.exclude_from is not None:
        exclusion = json.loads(arguments.exclusion.read_text(encoding="utf-8"))
        indices = np.asarray(exclusion.pop("indices"), dtype=np.int64)  # (x,)
        manifest = benchmark.exclude_training(arguments.exclude_from, arguments.output_dir, indices, exclusion)
    else:
        source_dir = arguments.source_dir or arguments.output_dir.parent / "source"
        download(source_dir)
        manifest = benchmark.prepare(arguments.output_dir, source_dir)
    print(json.dumps({"benchmark_id": benchmark.benchmark_id(manifest), "train": manifest["train"], "splits": manifest["splits"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
