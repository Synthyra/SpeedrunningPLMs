"""Validate publication locally without contacting the Hub."""

import json
import pytest

from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

from speedrunning_plms.training.publishing import publish_model_to_hub


SOURCE_FILES = {"config.json": "{}", "plm.py": "", "attention.py": "", "layers.py": ""}


class ArtifactModel:
    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.saved_path: Path | None = None

    def save_pretrained(self, path: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization
        self.saved_path = path
        for name, content in self.files.items():
            destination = path / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")


def unexpected_api() -> NoReturn:
    pytest.fail("Invalid or disabled publications must never construct a Hub client")


@pytest.mark.parametrize("weight_name", ["model.safetensors", "pytorch_model.bin"])
@pytest.mark.parametrize("sharded", [False, True])
def test_complete_artifact_uploaded_once_and_staging_removed(weight_name: str, sharded: bool) -> None:
    files = dict(SOURCE_FILES)
    if sharded:
        suffix = Path(weight_name).suffix
        shards = [f"model-0000{index}-of-00002{suffix}" for index in (1, 2)]
        files.update(dict.fromkeys(shards, "weights"))
        files[f"{weight_name}.index.json"] = json.dumps(
            {"weight_map": {"a": shards[0], "b": shards[0], "c": shards[1]}}
        )
    else:
        files[weight_name] = "weights"

    model = ArtifactModel(files)
    calls = []

    class RecordingApi:
        def create_repo(self, **kwargs: object) -> None:
            calls.append(("create_repo", kwargs))

        def upload_folder(self, folder_path: Path, **kwargs: object) -> str:
            calls.append(("upload_folder", kwargs))
            assert {path.name for path in folder_path.iterdir()} == set(files) | {"requirements.txt"}
            for name, content in files.items():
                assert (folder_path / name).read_text(encoding="utf-8") == content
            return "published"

    # Exercise nested compile/DDP wrappers without needing distributed workers.
    wrapped = SimpleNamespace(module=SimpleNamespace(_orig_mod=model))
    assert publish_model_to_hub(
        wrapped, "test/model", enabled=True, api_factory=RecordingApi
    ) == "published"
    assert calls == [
        ("create_repo", {"repo_id": "test/model", "repo_type": "model", "exist_ok": True}),
        ("upload_folder", {
            "repo_id": "test/model", "repo_type": "model",
            "commit_message": "Publish final trained model artifact",
        }),
    ]
    assert model.saved_path is not None and not model.saved_path.exists()


@pytest.mark.parametrize("weight_name", ["model.safetensors", "pytorch_model.bin"])
@pytest.mark.parametrize("index", [
    "not json", "[]", "null", "{}", '{"weight_map": {}}',
    '{"weight_map": []}', '{"weight_map": {"a": null}}',
    '{"weight_map": {"a": 1}}', '{"weight_map": {"a": []}}',
    '{"weight_map": {"a": ""}}', '{"weight_map": {"a": "config.json"}}',
])
def test_invalid_shard_index_rejected_before_hub_access(weight_name: str, index: str) -> None:
    model = ArtifactModel({**SOURCE_FILES, f"{weight_name}.index.json": index})
    with pytest.raises(RuntimeError, match="Invalid .*index"):
        publish_model_to_hub(model, "test/model", enabled=True, api_factory=unexpected_api)
    assert model.saved_path is not None and not model.saved_path.exists()


@pytest.mark.parametrize("weight_name", ["model.safetensors", "pytorch_model.bin"])
@pytest.mark.parametrize("missing_name", ["missing", "../outside", "/absolute"])
def test_every_indexed_shard_must_exist_in_artifact(weight_name: str, missing_name: str) -> None:
    suffix = Path(weight_name).suffix
    present = f"model-00001-of-00002{suffix}"
    missing = missing_name + suffix
    files = {
        **SOURCE_FILES,
        present: "weights",
        f"{weight_name}.index.json": json.dumps({"weight_map": {"a": present, "b": missing}}),
    }
    model = ArtifactModel(files)
    with pytest.raises(RuntimeError, match="missing weight shards"):
        publish_model_to_hub(model, "test/model", enabled=True, api_factory=unexpected_api)


@pytest.mark.parametrize("missing", [*SOURCE_FILES, "model.safetensors"])
def test_missing_required_artifact_file_rejected_before_hub_access(missing: str) -> None:
    files = {**SOURCE_FILES, "model.safetensors": "weights"}
    del files[missing]
    with pytest.raises(RuntimeError, match="Refusing to publish"):
        publish_model_to_hub(
            ArtifactModel(files), "test/model", enabled=True, api_factory=unexpected_api
        )


def test_orphan_shards_without_an_index_are_not_a_complete_model() -> None:
    model = ArtifactModel({**SOURCE_FILES, "model-00001-of-00002.safetensors": "weights"})
    with pytest.raises(RuntimeError, match="without model weights"):
        publish_model_to_hub(model, "test/model", enabled=True, api_factory=unexpected_api)


def test_incomplete_safetensors_index_cannot_be_hidden_by_legacy_weights() -> None:
    model = ArtifactModel({
        **SOURCE_FILES,
        "pytorch_model.bin": "weights",
        "model.safetensors.index.json": json.dumps({"weight_map": {"a": "missing.safetensors"}}),
    })
    with pytest.raises(RuntimeError, match="missing weight shards"):
        publish_model_to_hub(model, "test/model", enabled=True, api_factory=unexpected_api)


@pytest.mark.parametrize("enabled,repo_id", [(False, None), (False, "test/model"), (True, None)])
def test_opt_in_and_destination_checked_before_serialization(enabled: bool, repo_id: str | None) -> None:
    model = ArtifactModel({})
    if enabled:
        with pytest.raises(ValueError, match="repo_id is required"):
            publish_model_to_hub(model, repo_id, enabled=enabled, api_factory=unexpected_api)
    else:
        assert publish_model_to_hub(
            model, repo_id, enabled=enabled, api_factory=unexpected_api
        ) is None
    assert model.saved_path is None


def test_upload_failure_propagates_and_removes_staging() -> None:
    model = ArtifactModel({**SOURCE_FILES, "model.safetensors": "weights"})

    class FailingApi:
        def create_repo(self, **kwargs: object) -> None:
            pass

        def upload_folder(self, **kwargs: object) -> NoReturn:
            raise ConnectionError("upload failed")

    with pytest.raises(ConnectionError, match="upload failed"):
        publish_model_to_hub(model, "test/model", enabled=True, api_factory=FailingApi)
    assert model.saved_path is not None and not model.saved_path.exists()
