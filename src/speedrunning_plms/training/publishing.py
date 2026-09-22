"""Opt-in publication of complete trained-model artifacts."""

import json

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


REMOTE_CODE_REQUIREMENTS = "torch>=2.5\ntransformers>=4.57.6,<5\n"


def _validate_model_weights(artifact_dir: Path, files: set[str]) -> None:
    """Require a weight file or an index whose referenced shards all exist."""
    weight_names = ("model.safetensors", "pytorch_model.bin")
    for name in weight_names:
        if name in files:
            return

        index_name = f"{name}.index.json"
        if index_name not in files:
            continue

        try:
            index = json.loads((artifact_dir / index_name).read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            raise RuntimeError(f"Invalid model weight index: {index_name}") from error

        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map or any(
            not isinstance(key, str)
            or not isinstance(shard, str)
            or not shard.endswith(Path(name).suffix)
            for key, shard in weight_map.items()
        ):
            raise RuntimeError(f"Invalid weight_map in model weight index: {index_name}")

        missing = set(weight_map.values()) - files
        if missing:
            raise RuntimeError(
                "Refusing to publish an incomplete model artifact; missing weight shards: "
                + ", ".join(sorted(missing))
            )
        return

    raise RuntimeError("Refusing to publish an artifact without model weights.")


def _unwrap_model(model: Any) -> Any:
    """Remove DDP and torch.compile wrappers before serialization."""
    seen: set[int] = set()
    while id(model) not in seen:
        seen.add(id(model))
        if hasattr(model, "module"):
            model = model.module
            continue
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        break
    return model


def publish_model_to_hub(
    model: Any,
    repo_id: str | None,
    *,
    enabled: bool = False,
    api_factory: Callable[[], Any] | None = None,
) -> Any:
    """Publish one complete model snapshot in a single Hub commit.

    Nothing is imported from or sent to the Hub unless ``enabled`` is true.
    The artifact is fully staged and validated before any external API call.
    """
    if not enabled:
        return None
    if not repo_id:
        raise ValueError("repo_id is required when Hub publication is enabled.")

    model = _unwrap_model(model)
    with TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)
        model.save_pretrained(artifact_dir, safe_serialization=True)
        (artifact_dir / "requirements.txt").write_text(
            REMOTE_CODE_REQUIREMENTS,
            encoding="utf-8",
        )

        files = {
            path.relative_to(artifact_dir).as_posix()
            for path in artifact_dir.rglob("*")
            if path.is_file()
        }
        required = {
            "config.json",
            "plm.py",
            "attention.py",
            "layers.py",
            "requirements.txt",
        }
        missing = required - files
        if missing:
            raise RuntimeError(
                "Refusing to publish an incomplete model artifact; missing: "
                + ", ".join(sorted(missing))
            )
        _validate_model_weights(artifact_dir, files)

        if api_factory is None:
            from huggingface_hub import HfApi

            api_factory = HfApi
        api = api_factory()
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        return api.upload_folder(
            folder_path=artifact_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Publish final trained model artifact",
        )


__all__ = ["REMOTE_CODE_REQUIREMENTS", "publish_model_to_hub"]
