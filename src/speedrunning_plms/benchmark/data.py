"""The benchmark's data: the pinned source, the prepared files, their manifest and the fixed corruption.

`prepare` reads the three split files of the pinned Hub dataset once and writes a data directory:

- `train_tokens.u8`, one uint8 ESM2 token stream of every training protein as CLS, residues, EOS, in a seed-0 shuffled
  order; `train_offsets.i64`, the start of each protein in the stream plus the stream's end; `train_three_di.u8`, the 3Di
  class of every stream position (NO_STATE at CLS and EOS); `train_families.json`, the family of each protein in stream
  order; `probe_subset.i64`, the stream indices of the 3Di probe's fixed training proteins.
- `val_select.pt`, `validation.pt` and `test.pt`: padded uint8 `input_ids`, bool `masked` (the fixed masks), uint8
  `three_di`, int64 `lengths` (residues), and the family index of each protein with the family names, every tensor
  sorted by descending length.
- `manifest.json`: the source, the objective, every file's SHA-256 and row count. Its digest is the benchmark id.

A candidate may read the training files any way it likes; the held-out files are read only by the evaluator.

Symbols: n proteins, l tokens (CLS, residues, EOS, PAD), c = 20 residue classes.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import torch

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from torch import Tensor

from speedrunning_plms.benchmark import tokens


SOURCE_REPO = "Synthyra/alignment_atlas_families_afdb_v6"
SOURCE_REVISION = "83737fa33e72b74faeb5b046a45a327f660ad116"
SCHEMA_VERSION = 2
MASK_RATE = 0.15
MASK_SEED = 20261028  # the fixed masks of the held-out splits
SHUFFLE_SEED = 0  # the training stream's order and the probe subset
VAL_SELECT_MAX_TOKENS = 1024  # CLS + 1,022 residues + EOS
FULL_MAX_TOKENS = 2048  # CLS + 2,046 residues + EOS, the dataset's longest protein
PROBE_SUBSET_SIZE = 20_000
LENGTH_BINS = ((50, 100), (100, 250), (250, 500), (500, 1000), (1000, 2047))  # residues, [low, high)
OBJECTIVE = {"mask_rate": MASK_RATE, "replacement": "mask_only", "masked": "standard residues", "loss": "masked residues",
             "classes": tokens.STANDARD_RESIDUES, "cross_entropy": "collapse_free"}
HELD_OUT_SPLITS = ("val_select", "validation", "test")


@dataclass(frozen=True)
class HeldOutSplit:
    """One held-out split's tensors, as the evaluator reads them."""

    input_ids: Tensor  # (n, l), uint8 ESM2 ids with PAD past each protein's EOS, uncorrupted
    masked: Tensor  # (n, l), bool: the fixed positions the evaluator corrupts and scores
    three_di: Tensor  # (n, l), uint8 3Di class, NO_STATE off the residues
    lengths: Tensor  # (n,), int64 residue counts
    family: Tensor  # (n,), int64 index into `families`
    families: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_mask(length: int, row_seed: int) -> Tensor:
    """The fixed mask of one held-out protein: each residue with probability MASK_RATE, no forced minimum."""
    generator = torch.Generator().manual_seed(row_seed)
    return torch.rand(length, generator=generator) < MASK_RATE  # (length,)


def corrupt(input_ids: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """The training corruption: MASK each standard residue with probability MASK_RATE; labels are 20-way classes."""
    # input_ids: (..., l) ESM2 ids on the generator's device; returns corrupted (..., l) and labels (..., l).
    eligible = tokens.is_residue(input_ids)  # (..., l)
    selected = (torch.rand(input_ids.shape, generator=generator, device=input_ids.device) < MASK_RATE) & eligible  # (..., l)
    corrupted = input_ids.masked_fill(selected, tokens.MASK_ID)  # (..., l)
    labels = tokens.residue_labels(input_ids).masked_fill(~selected, tokens.IGNORE_LABEL)  # (..., l)
    return corrupted, labels


def length_bin(length: int) -> str:
    for low, high in LENGTH_BINS:
        if low <= length < high:
            return f"{low}-{high - 1}"
    raise ValueError(f"length {length} is outside the dataset's 50 to 2,046 residues")


def _held_out_tensors(rows: list[tuple[str, str, str]], max_tokens: int, split_seed: int) -> dict[str, Any]:
    """Pad one held-out split's proteins to one width, sorted by descending length, with their fixed masks."""
    # rows: (sequence, three_di, family) per protein; every tensor below is (n, l) or (n,).
    order = sorted(range(len(rows)), key=lambda index: (-len(rows[index][0]), rows[index][2]))
    width = min(max_tokens, max(len(rows[index][0]) for index in order) + 2)
    n = len(order)
    input_ids = torch.full((n, width), tokens.PAD_ID, dtype=torch.uint8)  # (n, l)
    masked = torch.zeros((n, width), dtype=torch.bool)  # (n, l)
    three_di = torch.full((n, width), tokens.NO_STATE, dtype=torch.uint8)  # (n, l)
    lengths = torch.zeros(n, dtype=torch.long)  # (n,)
    families = sorted({family for _, _, family in rows})
    family_index = {family: index for index, family in enumerate(families)}
    family = torch.zeros(n, dtype=torch.long)  # (n,)
    for row, index in enumerate(order):
        sequence, states, name = rows[index]
        residues = tokens.encode_residues(sequence)[: width - 2]
        states_ids = tokens.encode_three_di(states, len(sequence))[: width - 2]
        length = len(residues)
        input_ids[row, 0] = tokens.CLS_ID
        input_ids[row, 1 : 1 + length] = torch.tensor(residues, dtype=torch.uint8)  # (length,)
        input_ids[row, 1 + length] = tokens.EOS_ID
        three_di[row, 1 : 1 + length] = torch.tensor(states_ids, dtype=torch.uint8)  # (length,)
        masked[row, 1 : 1 + length] = fixed_mask(length, split_seed + index)  # (length,)
        lengths[row] = len(sequence)  # the protein's full length, for the length bins
        family[row] = family_index[name]
    return {"input_ids": input_ids, "masked": masked, "three_di": three_di, "lengths": lengths, "family": family,
            "families": families}


def _split_seed(split: str) -> int:
    return MASK_SEED + {"val_select": 1_000_000, "validation": 2_000_000, "test": 3_000_000}[split]


def select_one_per_family(frame: Any) -> Any:
    """The validation subset: each family's longest representative (ties by name), so every family is scored once."""
    representatives = frame[frame["role"] == "rep"].copy()
    representatives["length"] = representatives["seqs"].str.len()
    representatives = representatives.sort_values(["family", "length", "name"], ascending=[True, False, True])
    return representatives.drop_duplicates("family", keep="first")


def _write_train(frame: Any, output_dir: Path) -> dict[str, Any]:
    """The training stream in shuffled order, its offsets, 3Di states, families and the probe subset."""
    generator = np.random.default_rng(SHUFFLE_SEED)
    order = generator.permutation(len(frame))  # (n_train,)
    sequences = frame["seqs"].to_numpy()[order]
    states = frame["three_di"].to_numpy()[order]
    families = frame["family"].to_numpy()[order]
    lengths = np.fromiter((len(sequence) for sequence in sequences), dtype=np.int64, count=len(sequences))  # (n_train,)
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)  # (n_train + 1,)
    np.cumsum(lengths + 2, out=offsets[1:])
    stream = np.empty(int(offsets[-1]), dtype=np.uint8)  # (t,), t tokens
    three_di = np.full(int(offsets[-1]), tokens.NO_STATE, dtype=np.uint8)  # (t,)
    for index, (sequence, state_string) in enumerate(zip(sequences, states, strict=True)):
        start, end = int(offsets[index]), int(offsets[index + 1])
        stream[start] = tokens.CLS_ID
        stream[start + 1 : end - 1] = np.asarray(tokens.encode_residues(sequence), dtype=np.uint8)  # (length,)
        stream[end - 1] = tokens.EOS_ID
        three_di[start + 1 : end - 1] = np.asarray(tokens.encode_three_di(state_string, len(sequence)), dtype=np.uint8)
    # The probe subset: the first protein of each family in stream order, the first PROBE_SUBSET_SIZE families met.
    seen: set[str] = set()
    subset: list[int] = []
    for index, family in enumerate(families):
        if family not in seen:
            seen.add(family)
            subset.append(index)
        if len(subset) == PROBE_SUBSET_SIZE:
            break
    stream.tofile(output_dir / "train_tokens.u8")
    offsets.tofile(output_dir / "train_offsets.i64")
    three_di.tofile(output_dir / "train_three_di.u8")
    np.asarray(subset, dtype=np.int64).tofile(output_dir / "probe_subset.i64")
    (output_dir / "train_families.json").write_text(json.dumps(families.tolist()), encoding="utf-8")
    return {"proteins": int(len(sequences)), "tokens": int(offsets[-1]), "residues": int(lengths.sum()),
            "families": int(len(set(families.tolist()))), "probe_subset": len(subset)}


def prepare(output_dir: Path, source_dir: Path) -> dict[str, Any]:
    """Build the data directory from the three split parquet files downloaded from the pinned source."""
    import pandas as pd

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite the nonempty data directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["name", "seqs", "three_di", "family", "role"]
    frames = {split: pd.read_parquet(source_dir / f"{split}.parquet", columns=columns)
              for split in ("train", "validation", "test")}
    held_out = {"val_select": select_one_per_family(frames["validation"]), "validation": frames["validation"],
                "test": frames["test"]}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {"repo_id": SOURCE_REPO, "revision": SOURCE_REVISION},
        "objective": OBJECTIVE, "tokenizer": "esm2", "vocab_size": tokens.VOCAB_SIZE,
        "mask_seed": MASK_SEED, "shuffle_seed": SHUFFLE_SEED,
        "max_tokens": {"val_select": VAL_SELECT_MAX_TOKENS, "validation": FULL_MAX_TOKENS, "test": FULL_MAX_TOKENS},
        "length_bins": [list(bounds) for bounds in LENGTH_BINS],
        "train": _write_train(frames["train"], output_dir),
        "splits": {},
        "files": {},
    }
    for split, frame in held_out.items():
        rows = list(zip(frame["seqs"], frame["three_di"], frame["family"], strict=True))
        tensors = _held_out_tensors(rows, manifest["max_tokens"][split], _split_seed(split))
        torch.save(tensors, output_dir / f"{split}.pt")
        manifest["splits"][split] = {
            "proteins": len(rows), "families": len(tensors["families"]),
            "residues": int(tensors["lengths"].sum()), "scored_residues": int(tensors["masked"].sum()),
            "width": int(tensors["input_ids"].shape[1]),
        }
    for path in sorted(output_dir.iterdir()):
        manifest["files"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_manifest(data_dir: Path) -> dict[str, Any]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("objective") != OBJECTIVE:
        raise ValueError("the data directory does not hold this benchmark's schema and objective")
    if set(manifest["splits"]) != set(HELD_OUT_SPLITS):
        raise ValueError("the manifest must describe val_select, validation and test")
    return manifest


def benchmark_id(manifest: dict[str, Any]) -> str:
    """The digest of the manifest: the data half of a run's comparison key."""
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_held_out(data_dir: Path, split: str, verify: bool = True) -> HeldOutSplit:
    """One held-out split, its file checked against the manifest's digest."""
    manifest = load_manifest(data_dir)
    path = data_dir / f"{split}.pt"
    if verify and sha256_file(path) != manifest["files"][path.name]["sha256"]:
        raise ValueError(f"{path.name} differs from the manifest")
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    if bool((tensors["input_ids"] == tokens.MASK_ID).any()):
        raise ValueError(f"{path.name} holds a MASK token; held-out files are uncorrupted")
    return HeldOutSplit(input_ids=tensors["input_ids"], masked=tensors["masked"], three_di=tensors["three_di"],
                        lengths=tensors["lengths"], family=tensors["family"], families=tuple(tensors["families"]))


@dataclass(frozen=True)
class TrainStream:
    """The training files, memory-mapped."""

    tokens: np.ndarray  # (t,), uint8
    offsets: np.ndarray  # (n_train + 1,), int64
    three_di: np.ndarray  # (t,), uint8
    probe_subset: np.ndarray  # (n_probe,), int64 protein indices

    def protein(self, index: int) -> np.ndarray:
        return self.tokens[self.offsets[index] : self.offsets[index + 1]]  # (length + 2,)


def load_train(data_dir: Path) -> TrainStream:
    return TrainStream(
        tokens=np.memmap(data_dir / "train_tokens.u8", dtype=np.uint8, mode="r"),
        offsets=np.fromfile(data_dir / "train_offsets.i64", dtype=np.int64),
        three_di=np.memmap(data_dir / "train_three_di.u8", dtype=np.uint8, mode="r"),
        probe_subset=np.fromfile(data_dir / "probe_subset.i64", dtype=np.int64),
    )
