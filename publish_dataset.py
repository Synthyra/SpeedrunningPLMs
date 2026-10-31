"""Put the prepared benchmark folder on the Hub as `Synthyra/SpeedrunningPLMs`, with its card; fixed.

    python publish_dataset.py --data-dir /state/data/afdb_v6 [--public]
    python publish_dataset.py --card-only [--rename-from Synthyra/old_name] [--public]

The first form uploads every file `manifest.json` names plus the manifest itself and the card, into a private dataset
repository (public only with `--public`, on Logan's word); the upload is idempotent, files the Hub already holds are
skipped. The second form rewrites the card alone from the manifest the Hub holds, after an optional rename of the
repository. Needs HF_TOKEN in the environment.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any
from huggingface_hub import HfApi, hf_hub_download

from speedrunning_plms import benchmark
from speedrunning_plms.benchmark import tokens


REPO_ID = "Synthyra/SpeedrunningPLMs"
CODE_URL = "https://github.com/Synthyra/SpeedrunningPLMs"
# Test loss in nats per masked residue, the benchmark's evaluator on the same masks; the ten-minute record lives in the
# code repository's README and moves.
REFERENCE_TEST_LOSS = (
    ("unigram residue frequencies", "the floor", 2.8933),
    ("ESM2 8M", "UniRef50, days", 2.6253), ("ESM2 35M", "UniRef50, days", 2.4992), ("ESM2 150M", "UniRef50, days", 2.3303),
    ("ESM2 650M", "UniRef50, days", 2.0710), ("ESM2 3B", "UniRef50, days", 1.8705),
    ("ESMC 300M", "UniRef and more, days", 2.1269), ("ESMC 600M", "UniRef and more, days", 2.0510),
    ("ESMC 6B", "UniRef and more, days", 1.6891),
)
FILE_NOTES = {
    "train_tokens.u8": "one uint8 ESM2 token stream of every training protein, CLS, residues, EOS, in a seed-0 shuffled order",
    "train_offsets.i64": "int64: where each protein starts in the stream, plus the stream's end",
    "train_three_di.u8": "uint8: the Foldseek 3Di class of every stream position, 255 at CLS and EOS",
    "train_families.json": "the family of each training protein, in stream order",
    "probe_subset.i64": "int64: the stream indices of the structure probe's 20,000 training proteins, one per family",
    "val_select.pt": "the selection split: `input_ids`, `masked`, `three_di`, `lengths`, `family`, `families`",
    "validation.pt": "the full validation split, the same tensors at full length",
    "test.pt": "the full test split, the same tensors at full length",
    "manifest.json": "the source, the objective, the seeds, every count and every file's SHA-256",
}


def _size(byte_count: int) -> str:
    return f"{byte_count / 2**20:,.0f} MB" if byte_count >= 2**20 else f"{byte_count / 2**10:,.0f} KB"


def card(manifest: dict[str, Any], benchmark_id: str, repo_id: str) -> str:
    """The dataset card, from the manifest's counts."""
    source = manifest["source"]
    train = manifest["train"]
    splits = manifest["splits"]
    split_rows = "\n".join(
        f"| {name} | {split['proteins']:,} | {split['families']:,} | {split['residues']:,} | {split['scored_residues']:,} "
        f"| {manifest['max_tokens'][name]:,} |" for name, split in splits.items())
    sizes = {name: entry["bytes"] for name, entry in manifest["files"].items()}  # the manifest lists every file but itself
    file_rows = "\n".join(f"| `{name}` | {_size(sizes[name]) if name in sizes else 'small'} | {FILE_NOTES[name]} |" for name in FILE_NOTES)
    reference_rows = "\n".join(f"| {model} | {trained} | {loss:.4f} |" for model, trained, loss in REFERENCE_TEST_LOSS)
    residue_ids = ", ".join(f"{residue} {tokens.ESM2_IDS[residue]}" for residue in tokens.STANDARD_RESIDUES)
    return f"""---
license: cc-by-4.0
pretty_name: SpeedrunningPLMs
task_categories:
- fill-mask
tags:
- protein
- protein-language-model
- masked-language-modeling
- benchmark
- speedrun
- alphafold
- foldseek
size_categories:
- 1M<n<10M
---

# SpeedrunningPLMs: the benchmark data

**Train a protein language model in ten minutes on one GPU, and know exactly how good it is.** This is the fixed data
of [SpeedrunningPLMs]({CODE_URL}), a masked language model speedrun on proteins: {train['proteins']:,} training
proteins from the AlphaFold Database as one token stream, and held-out splits whose masks are drawn once, so every
run, every seed and every reference model is scored on the same hidden residues. No held-out protein has a training
relative at 30 percent sequence identity over 80 percent coverage, by sequence (MMseqs2) or by structure (Foldseek), so
a lower loss means a model that generalizes across protein families, not one that remembers them.

The benchmark id `{benchmark_id[:16]}` is the SHA-256 of `manifest.json`; every result carries it, and two results
compare only when it agrees.

## What a run does with it

| Step | Rule |
|---|---|
| Corruption | 15 percent of the standard residues are replaced by MASK, nothing else; CLS, EOS and padding are never masked. Training uses the same corruption. |
| Loss | Cross entropy over the 20 standard residues at the masked positions, in nats per masked residue. A model with a larger vocabulary (ESM2 has 33 tokens, ESMC 64) is scored on its 20 residue columns, so every model faces the same 20-way problem. |
| Clock | Ten minutes of applied training steps on one GPU. Compile warmup and in-run validation are outside the clock and capped. |
| Selection | The weights with the lowest loss on `val_select` (one protein per validation family, at most {manifest['max_tokens']['val_select']:,} tokens) are scored once on the full validation and test splits at full length (at most {manifest['max_tokens']['validation']:,} tokens). The headline is the test loss. |
| Structure probe | A linear probe from the hidden states to the 20 Foldseek 3Di states, trained on the fixed {train['probe_subset']:,}-protein subset and scored on validation and test. Reported beside the loss, never used for selection. |

## Splits

| Split | Proteins | Families | Residues | Scored (masked) residues | Max tokens |
|---|---|---|---|---|---|
| train | {train['proteins']:,} | {train['families']:,} | {train['residues']:,} | drawn each step | free |
{split_rows}

Families are disjoint across the three splits, inherited from the Alignment Atlas family split of the AlphaFold
Database (`{source['repo_id']}` at `{source['revision'][:12]}`). `val_select` is a subset of validation: each
family's longest representative, truncated, for the in-run selection; the full splits are scored once at the end.
Lengths run from 50 to 2,046 residues, and every result reports the loss by length bin (50 to 99, 100 to 249, 250 to
499, 500 to 999, 1,000 and more).

## Where models stand

Test loss on this data, every model scored by the same evaluator on the same masked residues. The reference models
trained for days on UniRef, which shares sequences with these proteins, so their rows are a comparator rather than
a fair baseline; the ten-minute record and the full table live in the [code repository]({CODE_URL}).

| Model | Trained on | Test loss |
|---|---|---|
{reference_rows}

## Loading

```bash
pip install git+{CODE_URL}
```

```python
from pathlib import Path

from huggingface_hub import snapshot_download
from speedrunning_plms import benchmark

data_dir = Path(snapshot_download("{repo_id}", repo_type="dataset"))
stream = benchmark.load_train(data_dir)                 # memory-mapped tokens, offsets, 3Di states, probe subset
protein = stream.protein(0)                            # (length + 2,) uint8: CLS, residues, EOS
test = benchmark.load_held_out(data_dir, "test")       # tensors sorted by descending length, digests verified
metrics = benchmark.evaluate_split(model, test, device)  # any model exposing logits and its 20 residue columns
```

The files need no library to read: the streams are flat NumPy arrays (`np.memmap("train_tokens.u8", dtype=np.uint8)`)
and the held-out files are `torch.load` dictionaries. Token ids are ESM2's: CLS 0, PAD 1, EOS 2, MASK 32, and the
residues {residue_ids}. The 3Di classes index Foldseek's alphabet `{tokens.THREE_DI_STATES}` in order, with
{tokens.NO_STATE} at positions that are not residues.

## Files

| File | Size | What it holds |
|---|---|---|
{file_rows}

Every file's SHA-256 is in `manifest.json` and the loaders verify it, so a run cannot read altered data unnoticed.

## Source and terms

AlphaFold Database sequences (CC BY 4.0) and their Foldseek 3Di states, grouped into families by sequence and
structure and split by family in Synthyra's Alignment Atlas. This release carries the tokens, the 3Di states and the
family index of each protein, no coordinates. The dataset is CC BY 4.0; cite the [code repository]({CODE_URL}) and the
AlphaFold Database.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None, help="the prepared folder; omitted with --card-only")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--rename-from", default=None, help="move this repository to --repo-id first")
    parser.add_argument("--card-only", action="store_true", help="rewrite README.md from the manifest the Hub holds")
    parser.add_argument("--public", action="store_true", help="only on Logan's word; private otherwise")
    arguments = parser.parse_args()
    if arguments.card_only == (arguments.data_dir is not None):
        raise ValueError("give --data-dir to upload the data, or --card-only to rewrite the card")
    api = HfApi()
    if arguments.rename_from is not None:
        api.move_repo(from_id=arguments.rename_from, to_id=arguments.repo_id, repo_type="dataset")
    if arguments.card_only:
        manifest = json.loads(Path(hf_hub_download(arguments.repo_id, "manifest.json", repo_type="dataset")).read_text(encoding="utf-8"))
        benchmark_id = benchmark.benchmark_id(manifest)
        api.upload_file(path_or_fileobj=card(manifest, benchmark_id, arguments.repo_id).encode("utf-8"), path_in_repo="README.md",
                        repo_id=arguments.repo_id, repo_type="dataset", commit_message=f"card for benchmark {benchmark_id[:16]}")
        names = ["README.md"]
    else:
        manifest = benchmark.load_manifest(arguments.data_dir)
        benchmark_id = benchmark.benchmark_id(manifest)
        api.create_repo(arguments.repo_id, repo_type="dataset", private=not arguments.public, exist_ok=True)
        (arguments.data_dir / "README.md").write_text(card(manifest, benchmark_id, arguments.repo_id), encoding="utf-8", newline="\n")
        names = ["README.md", "manifest.json", *manifest["files"]]
        api.upload_folder(repo_id=arguments.repo_id, repo_type="dataset", folder_path=str(arguments.data_dir),
                          allow_patterns=names, commit_message=f"benchmark {benchmark_id[:16]}")
    if arguments.public:
        api.update_repo_settings(arguments.repo_id, repo_type="dataset", private=False)
    print(json.dumps({"repo_id": arguments.repo_id, "benchmark_id": benchmark_id, "files": names,
                      "private": api.dataset_info(arguments.repo_id).private}), flush=True)


if __name__ == "__main__":
    main()
