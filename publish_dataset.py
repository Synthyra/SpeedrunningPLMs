"""Put the prepared benchmark folder on the Hub as `Synthyra/plm_speedrun_afdb_v6`; fixed.

    python publish_dataset.py --data-dir /state/data/afdb_v6 --repo-id Synthyra/plm_speedrun_afdb_v6 [--public]

Uploads every file `manifest.json` names plus the manifest itself and a README built from the manifest's counts, into
a private dataset repository (public only with `--public`, on Logan's word). The upload is idempotent: files whose
content the Hub already holds are skipped. Needs HF_TOKEN in the environment.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from huggingface_hub import HfApi

from speedrunning_plms import benchmark


def card(manifest: dict[str, object], benchmark_id: str) -> str:
    source = manifest["source"]
    train = manifest["train"]
    splits = manifest["splits"]
    rows = "\n".join(f"| {name} | {split['proteins']:,} | {split['families']:,} | {split['residues']:,} "
                     f"| {split['scored_residues']:,} | {split['width']:,} |" for name, split in splits.items())
    files = "\n".join(f"| `{name}` | {entry['bytes']:,} |" for name, entry in manifest["files"].items())
    return f"""---
license: cc-by-4.0
pretty_name: PLM speedrun, AFDB v6
tags:
- protein
- masked-language-modeling
- benchmark
---

# PLM speedrun, AFDB v6

The fixed data of the SpeedrunningPLMs ten-minute protein masked language model benchmark
([code](https://github.com/Synthyra/SpeedrunningPLMs)): AlphaFold DB proteins from the family-disjoint split of the
Alignment Atlas (`{source['repo_id']}` at `{source['revision'][:12]}`), tokenized once in the ESM2 alphabet, with the
held-out splits' 15 percent masks materialized so every model is scored on the same hidden residues. No held-out
protein has a training relative at 30 percent identity over 80 percent coverage by sequence (MMseqs2) or by structure
(Foldseek). Benchmark id `{benchmark_id[:16]}`: the SHA-256 of this manifest, carried by every run's result.

## Objective

A model reads a protein (CLS, residues, EOS) with 15 percent of the standard residues replaced by MASK and nothing
else, and predicts the hidden residues; the loss is the cross entropy over the 20 standard residues on the masked
positions. `val_select` (one protein per validation family, truncated to {manifest['max_tokens']['val_select']:,} tokens) selects weights in a
run; validation and test are scored at full length (up to {manifest['max_tokens']['validation']:,} tokens). The train stream is one
uint8 token stream of {train['proteins']:,} proteins ({train['tokens']:,} tokens, {train['families']:,} families) in a seeded
shuffled order, with document offsets and the Foldseek 3Di state of every residue beside it; the 3Di states are never
an input, only the target of a reported structure probe on a fixed {train['probe_subset']:,}-protein subset.

## Splits

| Split | Proteins | Families | Residues | Scored (masked) residues | Width |
|---|---|---|---|---|---|
{rows}

## Files

| File | Bytes |
|---|---|
{files}

Every file's SHA-256 is in `manifest.json`; the loaders verify them.

`train_tokens.u8` and `train_three_di.u8` are flat uint8 streams, `train_offsets.i64` the int64 start of every
protein, `probe_subset.i64` the probe subset's indices, `train_families.json` the family of every training protein;
`val_select.pt`, `validation.pt` and `test.pt` hold padded `input_ids`, `masked`, `three_di`, `family` and `lengths`
tensors and the list of `families`. `speedrunning_plms.benchmark.load_train` and `load_held_out` read them and verify the digests.

## Source and terms

AlphaFold DB sequences (CC-BY-4.0) and Foldseek 3Di states, through the Alignment Atlas family split; the source
dataset's taxonomy is not audited here (see the workspace record).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="Synthyra/plm_speedrun_afdb_v6")
    parser.add_argument("--public", action="store_true", help="only on Logan's word; private otherwise")
    arguments = parser.parse_args()
    manifest = benchmark.load_manifest(arguments.data_dir)
    benchmark_id = benchmark.benchmark_id(manifest)
    api = HfApi()
    api.create_repo(arguments.repo_id, repo_type="dataset", private=not arguments.public, exist_ok=True)
    (arguments.data_dir / "README.md").write_text(card(manifest, benchmark_id), encoding="utf-8", newline="\n")
    names = ["README.md", "manifest.json", *manifest["files"]]
    api.upload_folder(repo_id=arguments.repo_id, repo_type="dataset", folder_path=str(arguments.data_dir),
                      allow_patterns=names, commit_message=f"benchmark {benchmark_id[:16]}")
    print(json.dumps({"repo_id": arguments.repo_id, "benchmark_id": benchmark_id, "files": names,
                      "private": not arguments.public}), flush=True)


if __name__ == "__main__":
    main()
