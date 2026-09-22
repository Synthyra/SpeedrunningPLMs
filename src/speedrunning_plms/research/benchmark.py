"""Pinned protein data and the fixed 15% masked-residue benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import torch
import torch.nn.functional as F

from collections.abc import Iterator, Mapping
from itertools import islice
from pathlib import Path
from typing import Any
from torch import Tensor


MASK_RATE = 0.15
CLS_TOKEN_ID = 0
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2
MASK_TOKEN_ID = 32
VOCAB_SIZE = 33
# ESM-1b/ESM-2 alphabet: github.com/facebookresearch/esm/blob/main/esm/constants.py
RESIDUE_IDS = {residue: index + 4 for index, residue in enumerate("LAGVSERTIDPKQNFYMHWCXBUZO.-")}
TOKEN_IDS = {"cls": 0, "pad": 1, "eos": 2, "unk": 3, "null": 31, "mask": 32}
DATASETS = {
    "uniref50": ("Synthyra/uniref50", "36d67a647c4c596664ad2284ca9ab571baff08b9"),
    "omg_prot50": ("Synthyra/omg_prot50", "c5b07302de5fc0e2cac87933d9167e0b2d6f05c0"),
    "og_prot90": ("Synthyra/og_prot90", "322bcb78561007be855ccbf0b744f24bbec41c6b"),
}
OBJECTIVE = {"mask_rate": MASK_RATE, "replacement": "mask", "metric": "bits_per_masked_residue"}
TOKENIZER = {"name": "esm2", "vocab_size": VOCAB_SIZE, "residue_ids": RESIDUE_IDS, "special_ids": TOKEN_IDS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_tokens(tokens: Tensor, max_length: int | None = None) -> None:
    # tokens: (n, l), CPU integer ESM IDs.
    if not isinstance(tokens, Tensor) or tokens.dtype != torch.long or tokens.device.type != "cpu":
        raise ValueError("input_ids must be a CPU torch.long tensor")
    if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] < 3:
        raise ValueError("input_ids must have shape (n > 0, length >= 3)")
    if max_length is not None and tokens.shape[1] != max_length:
        raise ValueError("input_ids width differs from manifest max_length")
    rows_per_chunk = max(1, 1_048_576 // tokens.shape[1])
    for start in range(0, len(tokens), rows_per_chunk):
        chunk = tokens[start : start + rows_per_chunk]  # (n_chunk, l); bound validation temporaries.
        if bool(((chunk < 0) | (chunk >= VOCAB_SIZE)).any()):
            raise ValueError("input_ids contain IDs outside the ESM vocabulary")
        if bool((chunk == MASK_TOKEN_ID).any()):
            raise ValueError("Prepared data must contain uncorrupted tokens")


def encode_sequence(sequence: str, max_length: int) -> list[list[int]]:
    """Chunk a protein without dropping its tail; reserve CLS and EOS positions."""
    if max_length < 3:
        raise ValueError("max_length must be at least 3")
    sequence = "".join(sequence.split()).upper()
    if not sequence:
        raise ValueError("Protein sequences cannot be empty")
    invalid = set(sequence).difference(RESIDUE_IDS)
    if invalid:
        raise ValueError(f"Unsupported protein symbols: {sorted(invalid)}")
    residue_ids = [RESIDUE_IDS[residue] for residue in sequence]
    windows: list[list[int]] = []
    for start in range(0, len(residue_ids), max_length - 2):
        window = [CLS_TOKEN_ID, *residue_ids[start : start + max_length - 2], EOS_TOKEN_ID]
        windows.append(window + [PAD_TOKEN_ID] * (max_length - len(window)))
    return windows


def write_dataset(
    splits: Mapping[str, Tensor],
    output_dir: Path,
    *,
    dataset_name: str = "synthetic",
    source_revision: str = "local",
    repo_id: str | None = None,
) -> dict[str, Any]:
    """Write immutable local split files and their content-hash manifest."""
    # Each splits value: (n_split, l).
    output_dir = Path(output_dir)
    if not {"train", "valid"}.issubset(splits) or set(splits).difference({"train", "valid", "test"}):
        raise ValueError("Provide train and valid splits, with optional test")
    max_length = None
    for tokens in splits.values():  # (n_split, l)
        _validate_tokens(tokens, max_length)
        max_length = tokens.shape[1]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty dataset directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {"name": dataset_name, "repo_id": repo_id, "revision": source_revision},
        "max_length": max_length,
        "objective": dict(OBJECTIVE),
        "tokenizer": TOKENIZER,
        "splits": {},
    }
    for split, tokens in sorted(splits.items()):  # tokens: (n_split, l)
        filename = f"{split}.pt"
        # Clone views so a split cannot serialize another split's shared storage.
        torch.save(
            {"input_ids": tokens.clone(memory_format=torch.contiguous_format)},  # (n_split, l)
            output_dir / filename,
        )
        manifest["splits"][split] = {
            "file": filename,
            "sha256": _sha256(output_dir / filename),
            "num_examples": tokens.shape[0],
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_manifest(data_dir: Path) -> dict[str, Any]:
    """Check benchmark metadata without opening any sequence split."""
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("Unsupported benchmark manifest schema")
    if manifest.get("objective") != OBJECTIVE or manifest.get("tokenizer") != TOKENIZER:
        raise ValueError("Manifest does not describe the fixed 15% ESM benchmark")
    max_length = manifest.get("max_length")
    if type(max_length) is not int or max_length < 3:
        raise ValueError("Manifest max_length must be an integer >= 3")
    dataset = manifest.get("dataset")
    if (
        not isinstance(dataset, dict)
        or not isinstance(dataset.get("name"), str)
        or not dataset["name"]
        or not isinstance(dataset.get("revision"), str)
        or not dataset["revision"]
    ):
        raise ValueError("Manifest must identify the source dataset and revision")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not {"train", "valid"}.issubset(splits):
        raise ValueError("Manifest must define train and valid splits")
    for split, metadata in splits.items():
        if split not in {"train", "valid", "test"} or not isinstance(metadata, dict):
            raise ValueError("Invalid split metadata")
        if metadata.get("file") != f"{split}.pt":
            raise ValueError("Split filename must match its split name")
        if not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("sha256", ""))):
            raise ValueError("Invalid split SHA-256")
        if type(metadata.get("num_examples")) is not int or metadata["num_examples"] < 1:
            raise ValueError("Split num_examples must be a positive integer")
    return manifest


def benchmark_id(data_dir: Path) -> str:
    manifest = load_manifest(data_dir)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_split(data_dir: Path, split: str) -> Tensor:
    """Map a validated split into memory; callers must treat it as read-only.

    Indexed training batches and corruption produce separate tensors. Keep the
    immutable split file in place for the lifetime of this tensor.
    """
    manifest = load_manifest(data_dir)
    if split not in manifest["splits"]:
        raise ValueError(f"Split {split!r} was not prepared")
    metadata = manifest["splits"][split]
    path = Path(data_dir) / metadata["file"]
    if _sha256(path) != metadata["sha256"]:
        raise ValueError(f"Checksum mismatch for {split}")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or "input_ids" not in payload:
        raise ValueError("Split must contain input_ids")
    tokens = payload["input_ids"]  # (n, l)
    _validate_tokens(tokens, manifest["max_length"])
    if tokens.shape[0] != metadata["num_examples"]:
        raise ValueError("Split example count differs from manifest")
    return tokens  # (n, l)


def corrupt_tokens(input_ids: Tensor, *, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Mask independent residues with probability 0.15, without a forced minimum."""
    # input_ids: (..., l), CPU integer ESM IDs; outputs have the same shape.
    if input_ids.device.type != "cpu" or generator.device.type != "cpu":
        raise ValueError("Corruption requires CPU input and a CPU generator")
    eligible = (input_ids >= 4) & (input_ids <= 28)  # (..., l); exclude gaps and specials.
    selected = (torch.rand(input_ids.shape, generator=generator) < MASK_RATE) & eligible  # (..., l)
    corrupted = input_ids.masked_fill(selected, MASK_TOKEN_ID)  # (..., l)
    labels = input_ids.masked_fill(~selected, -100)  # (..., l)
    return corrupted, labels  # (..., l), (..., l)


def evaluation_batches(
    tokens: Tensor,
    batch_size: int,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[dict[str, Tensor]]:
    """Use a fixed mask per example, independent of batches and worker count."""
    # tokens: (n, l); batches: (b <= batch_size, l).
    if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid evaluation batch size or distributed rank")
    indices = range(rank, len(tokens), world_size)
    for start in range(0, len(indices), batch_size):
        corrupted_rows: list[Tensor] = []
        label_rows: list[Tensor] = []
        attention_rows: list[Tensor] = []
        for index in indices[start : start + batch_size]:
            generator = torch.Generator().manual_seed((seed + index) % (2**63))
            corrupted, labels = corrupt_tokens(tokens[index], generator=generator)  # (l), (l)
            corrupted_rows.append(corrupted)
            label_rows.append(labels)
            attention_rows.append(tokens[index].ne(PAD_TOKEN_ID).long())  # (l)
        yield {
            "input_ids": torch.stack(corrupted_rows),  # (b, l)
            "labels": torch.stack(label_rows),  # (b, l)
            "attention_mask": torch.stack(attention_rows),  # (b, l)
        }


def evaluate_model(
    model: torch.nn.Module,
    tokens: Tensor,
    batch_size: int,
    device: torch.device | str,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, float | int]:
    """Compute corpus-weighted masked-residue metrics from logits, never model loss."""
    # tokens: (n, l); logits: (b, l, vocab_size).
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        if world_size != torch.distributed.get_world_size() or rank != torch.distributed.get_rank():
            raise ValueError("Evaluation rank/world_size differs from the active process group")
    elif world_size != 1 or rank != 0:
        raise ValueError("Distributed evaluation requires an initialized process group")
    totals = torch.zeros(3, dtype=torch.float64, device=device)  # (3): NLL, correct, masked count.
    was_training = model.training
    model.eval()
    try:
        # Evaluation precision is fixed even when a caller enables training autocast.
        with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            for batch in evaluation_batches(tokens, batch_size, seed, rank, world_size):
                labels = batch["labels"].to(device)  # (b, l)
                selected = labels.ne(-100)  # (b, l)
                if not bool(selected.any()):
                    continue
                output = model(
                    input_ids=batch["input_ids"].to(device),  # (b, l)
                    attention_mask=batch["attention_mask"].to(device),  # (b, l)
                )
                logits = output.logits  # (b, l, vocab_size)
                if logits.shape != (*labels.shape, VOCAB_SIZE):
                    raise ValueError("Model logits must have shape (batch, length, 33)")
                masked_logits = logits[selected].float()  # (m, vocab_size); m selected residues.
                targets = labels[selected]  # (m)
                losses = F.cross_entropy(masked_logits, targets, reduction="none")  # (m)
                totals[0] += losses.double().sum()  # ()
                totals[1] += masked_logits.argmax(dim=-1).eq(targets).sum()  # ()
                totals[2] += targets.numel()  # ()
        if distributed:
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)  # (3)
        nll, correct, count = totals.tolist()
        if count == 0:
            raise ValueError("Evaluation selected zero masked residues; use a larger evaluation split")
        if not math.isfinite(nll):
            raise ValueError("Evaluation produced non-finite cross-entropy")
        loss = nll / count
        return {
            "loss": loss,
            "bits_per_masked_residue": loss / math.log(2),
            "masked_accuracy": correct / count,
            "masked_tokens": int(count),
        }
    finally:
        model.train(was_training)


def prepare_dataset(
    output_dir: Path,
    *,
    dataset_name: str = "uniref50",
    max_length: int = 256,
    train_sequences: int = 100_000,
    eval_sequences: int = 2048,
    include_test: bool = False,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Stream bounded source sequence counts; retain every chunk of each sequence."""
    from datasets import load_dataset

    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if max_length < 3 or train_sequences < 1 or eval_sequences < 1:
        raise ValueError("Require max_length >= 3 and positive sequence limits")
    repo_id, default_revision = DATASETS[dataset_name]
    revision = source_revision or default_revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Source revision must be an immutable 40-character commit SHA")
    if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty dataset directory: {output_dir}")
    splits: dict[str, Tensor] = {}
    split_limits = {"train": train_sequences, "valid": eval_sequences}
    if include_test:
        split_limits["test"] = eval_sequences
    for split, limit in split_limits.items():
        source = load_dataset(repo_id, split=split, revision=revision, streaming=True)
        windows: list[list[int]] = []
        for example in islice(source, limit):
            windows.extend(encode_sequence(example["sequence"], max_length))
        if not windows:
            raise ValueError(f"Source split {split} is empty")
        splits[split] = torch.tensor(windows, dtype=torch.long)  # (n_split, max_length)
    return write_dataset(splits, output_dir, dataset_name=dataset_name, source_revision=revision, repo_id=repo_id)


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="uniref50")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-sequences", type=int, default=100_000)
    parser.add_argument("--eval-sequences", type=int, default=2048)
    parser.add_argument("--include-test", action="store_true", help="Prepare the held-out test split explicitly")
    parser.add_argument("--source-revision", help="Override with an immutable dataset commit SHA")
    args = parser.parse_args()
    manifest = prepare_dataset(
        args.output_dir,
        dataset_name=args.dataset,
        max_length=args.max_length,
        train_sequences=args.train_sequences,
        eval_sequences=args.eval_sequences,
        include_test=args.include_test,
        source_revision=args.source_revision,
    )
    print(json.dumps({"benchmark_id": benchmark_id(args.output_dir), "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    prepare_main()
