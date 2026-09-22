"""Load benchmark assets only at manifest-pinned Hub commits."""

from __future__ import annotations

import json
import re

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _validate_asset(asset: Mapping[str, Any], *, label: str) -> None:
    repo_id = asset.get("repo_id")
    revision = asset.get("revision")
    if not isinstance(repo_id, str) or "/" not in repo_id:
        raise ValueError(f"{label}.repo_id must be a Hugging Face repository ID.")
    if not isinstance(revision, str) or not FULL_COMMIT_SHA.fullmatch(revision):
        raise ValueError(f"{label}.revision must be a full 40-character commit SHA.")


def load_benchmark_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate an immutable benchmark asset manifest."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark manifest schema_version.")

    tokenizer = manifest.get("tokenizer")
    models = manifest.get("models")
    datasets = manifest.get("datasets")
    if not isinstance(tokenizer, dict):
        raise ValueError("Benchmark manifest requires one tokenizer asset.")
    if not isinstance(models, list) or not models:
        raise ValueError("Benchmark manifest requires at least one model asset.")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Benchmark manifest requires at least one dataset asset.")

    _validate_asset(tokenizer, label="tokenizer")
    for index, model in enumerate(models):
        _validate_asset(model, label=f"models[{index}]")
        if not model.get("nickname"):
            raise ValueError(f"models[{index}].nickname is required.")
    for index, dataset in enumerate(datasets):
        _validate_asset(dataset, label=f"datasets[{index}]")
        if not dataset.get("name"):
            raise ValueError(f"datasets[{index}].name is required.")
        filename = dataset.get("filename")
        if not isinstance(filename, str) or "{split}" not in filename:
            raise ValueError(
                f"datasets[{index}].filename must contain the {{split}} placeholder."
            )

    model_names = [model["nickname"] for model in models]
    dataset_names = [dataset["name"] for dataset in datasets]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Model nicknames must be unique.")
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("Dataset names must be unique.")
    return manifest


def download_dataset_split(
    asset: Mapping[str, Any],
    split: str,
    *,
    downloader: Callable[..., str],
) -> str:
    """Download one dataset split at its pinned manifest revision."""
    return downloader(
        repo_id=asset["repo_id"],
        filename=asset["filename"].format(split=split),
        repo_type="dataset",
        revision=asset["revision"],
    )


def load_benchmark_model(asset: Mapping[str, Any], *, auto_model_cls: Any) -> Any:
    """Load model weights and remote code from the same immutable commit."""
    revision = asset["revision"]
    return auto_model_cls.from_pretrained(
        asset["repo_id"],
        trust_remote_code=True,
        revision=revision,
        code_revision=revision,
    )


def load_benchmark_tokenizer(asset: Mapping[str, Any], *, auto_tokenizer_cls: Any) -> Any:
    """Load the tokenizer from its immutable manifest commit."""
    return auto_tokenizer_cls.from_pretrained(
        asset["repo_id"],
        revision=asset["revision"],
    )
