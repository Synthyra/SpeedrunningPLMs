import json
import os
import subprocess
import sys
import pytest

from pathlib import Path

from speedrunning_plms.evaluation import (
    download_dataset_split,
    load_benchmark_manifest,
    load_benchmark_model,
    load_benchmark_tokenizer,
)
from speedrunning_plms.evaluation.benchmark_assets import FULL_COMMIT_SHA


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "evaluation" / "benchmark_manifest.json"


def test_manifest_pins_every_asset_to_a_full_commit_sha() -> None:
    manifest = load_benchmark_manifest(MANIFEST_PATH)

    assets = [manifest["tokenizer"], *manifest["models"], *manifest["datasets"]]
    assert len(assets) == 11
    for asset in assets:
        assert FULL_COMMIT_SHA.fullmatch(asset["revision"])


def test_manifest_rejects_mutable_revision(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["models"][0]["revision"] = "main"
    path = tmp_path / "mutable.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="full 40-character commit SHA"):
        load_benchmark_manifest(path)


def test_benchmark_entrypoint_loads_manifest_aware_code() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "evaluation.benchmark_esm", "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "--manifest" in completed.stdout


def test_full_shas_propagate_to_every_hub_loader() -> None:
    manifest = load_benchmark_manifest(MANIFEST_PATH)

    model_calls = []

    class RecordingModelLoader:
        @classmethod
        def from_pretrained(cls, repo_id: str, **kwargs: object) -> str:
            model_calls.append((repo_id, kwargs))
            return repo_id

    for asset in manifest["models"]:
        assert load_benchmark_model(
            asset,
            auto_model_cls=RecordingModelLoader,
        ) == asset["repo_id"]

    assert len(model_calls) == len(manifest["models"])
    for asset, (repo_id, kwargs) in zip(manifest["models"], model_calls):
        assert repo_id == asset["repo_id"]
        assert kwargs == {
            "trust_remote_code": True,
            "revision": asset["revision"],
            "code_revision": asset["revision"],
        }

    tokenizer_calls = []

    class RecordingTokenizerLoader:
        @classmethod
        def from_pretrained(cls, repo_id: str, **kwargs: object) -> str:
            tokenizer_calls.append((repo_id, kwargs))
            return repo_id

    tokenizer = manifest["tokenizer"]
    assert load_benchmark_tokenizer(
        tokenizer,
        auto_tokenizer_cls=RecordingTokenizerLoader,
    ) == tokenizer["repo_id"]
    assert tokenizer_calls == [
        (tokenizer["repo_id"], {"revision": tokenizer["revision"]})
    ]

    dataset_calls = []

    def recording_download(**kwargs: object) -> str:
        dataset_calls.append(kwargs)
        return kwargs["filename"]

    for asset in manifest["datasets"]:
        for split in ("valid", "test"):
            assert download_dataset_split(
                asset,
                split,
                downloader=recording_download,
            ) == asset["filename"].format(split=split)

    assert len(dataset_calls) == 2 * len(manifest["datasets"])
    for asset, calls in zip(
        manifest["datasets"],
        (dataset_calls[index:index + 2] for index in range(0, len(dataset_calls), 2)),
    ):
        for split, kwargs in zip(("valid", "test"), calls):
            assert kwargs == {
                "repo_id": asset["repo_id"],
                "filename": asset["filename"].format(split=split),
                "repo_type": "dataset",
                "revision": asset["revision"],
            }
