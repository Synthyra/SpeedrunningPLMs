import json
import os
import subprocess
import sys
import textwrap
import pytest
import torch

from pathlib import Path
from typing import NoReturn

from speedrunning_plms.models import PLM, PLMConfig
from speedrunning_plms.training.publishing import publish_model_to_hub


def tiny_config(**overrides: object) -> PLMConfig:
    values = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "vocab_size": 33,
        "unet": False,
        "compile_flex_attention": False,
        "tokenizer_name": None,
        "cls_token_id": 0,
        "eos_token_id": 2,
        "pad_token_id": 1,
        "mask_token_id": 32,
    }
    values.update(overrides)
    return PLMConfig(**values)


@pytest.fixture
def tiny_model() -> PLM:
    torch.manual_seed(7)
    return PLM(tiny_config())


def test_config_save_has_canonical_autoclass_metadata(tmp_path: Path) -> None:
    config = tiny_config(auto_map={"AutoModel": "legacy.Unsupported"})
    config.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["model_type"] == "speedrunning_plm"
    assert saved["auto_map"] == {
        "AutoConfig": "plm.PLMConfig",
        "AutoModelForMaskedLM": "plm.PLM",
    }
    assert {"plm.py", "attention.py", "layers.py"}.issubset(
        path.name for path in tmp_path.iterdir()
    )

    for source_name in ("plm.py", "attention.py", "layers.py"):
        source = (tmp_path / source_name).read_text(encoding="utf-8")
        assert "from speedrunning_plms" not in source
        assert "model--PLM" not in source
        assert "torchinfo" not in source
        assert "huggingface_hub" not in source


def test_direct_pretrained_round_trip_preserves_config_and_weights(tiny_model: PLM, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    tiny_model.save_pretrained(checkpoint)

    restored = PLM.from_pretrained(checkpoint, local_files_only=True)

    assert restored.config.model_type == "speedrunning_plm"
    assert restored.config.tokenizer_name is None
    assert restored.tokenizer is None
    assert restored.state_dict().keys() == tiny_model.state_dict().keys()
    for key, expected in tiny_model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], expected)


def test_tied_embedding_round_trip_preserves_parameter_sharing(tmp_path: Path) -> None:
    model = PLM(tiny_config(tie_embeddings=True))
    checkpoint = tmp_path / "tied-checkpoint"

    assert model.embedding.weight is model.lm_head.decoder.weight
    model.save_pretrained(checkpoint)
    restored = PLM.from_pretrained(checkpoint, local_files_only=True)

    assert restored.config.tie_word_embeddings is True
    assert restored.embedding.weight is restored.lm_head.decoder.weight
    torch.testing.assert_close(restored.embedding.weight, model.embedding.weight)


def test_save_weights_local_uses_zero_padded_step_directory(tiny_model: PLM, tmp_path: Path) -> None:
    tiny_model.save_weights_local(tmp_path, step=42)

    checkpoint = tmp_path / "step_000042"
    assert (checkpoint / "config.json").is_file()
    restored = PLM.from_pretrained(checkpoint, local_files_only=True)
    torch.testing.assert_close(restored.embedding.weight, tiny_model.embedding.weight)


def test_masked_lm_contract_supports_batched_inference_attention_and_labels() -> None:
    model = PLM(tiny_config(num_hidden_layers=1))
    input_ids = torch.tensor(
        [
            [0, 5, 32, 2, 1, 1],
            [0, 7, 8, 32, 2, 1],
        ]
    )  # (2, 6)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ]
    )  # (2, 6)

    model.eval()
    inference = model(input_ids=input_ids, attention_mask=attention_mask)  # logits: (2, 6, 33)
    assert inference.loss is None
    assert inference.logits.shape == (2, 6, 33)

    labels = torch.full_like(input_ids, -100)  # (2, 6)
    labels[0, 2] = 9  # scalar target
    labels[1, 3] = 10  # scalar target
    model.train()
    training = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
    )  # logits: (2, 6, 33); hidden_states[0]: (2, 6, 8); loss: ()
    assert training.loss is not None
    assert training.loss.ndim == 0
    assert training.logits.shape == (2, 6, 33)
    assert training.hidden_states[0].shape == (2, 6, 8)
    training.loss.backward()
    assert model.embedding.weight.grad is not None

    tuple_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )  # logits: (2, 6, 33)
    assert tuple_output[0].shape == (2, 6, 33)


def test_autoclasses_load_saved_remote_code_without_installed_package(tmp_path: Path) -> None:
    checkpoint = tmp_path / "remote-checkpoint"
    PLM(tiny_config(num_hidden_layers=1)).save_pretrained(checkpoint)

    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        class BlockInstalledPackage(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "speedrunning_plms" or fullname.startswith("speedrunning_plms."):
                    raise ModuleNotFoundError("remote code imported the installed project package")
                if fullname == "torchinfo" or fullname.startswith("torchinfo."):
                    raise ModuleNotFoundError("remote code imported optional torchinfo")
                return None

        sys.meta_path.insert(0, BlockInstalledPackage())

        import torch
        from transformers import AutoConfig, AutoModelForMaskedLM

        checkpoint = {str(checkpoint)!r}
        config = AutoConfig.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            local_files_only=True,
        )
        assert config.__class__.__name__ == "PLMConfig"
        assert config.model_type == "speedrunning_plm"

        masked_lm = AutoModelForMaskedLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            local_files_only=True,
        )
        assert masked_lm.__class__.__name__ == "PLM"
        assert tuple(masked_lm.lm_head.decoder.weight.shape) == (33, 8)

        input_ids = torch.tensor([[0, 5, 32, 2, 1], [0, 6, 32, 2, 1]])  # (2, 5)
        attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])  # (2, 5)
        inference = masked_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        assert inference.loss is None
        assert tuple(inference.logits.shape) == (2, 5, 33)

        labels = torch.full_like(input_ids, -100)  # (2, 5)
        labels[:, 2] = torch.tensor([7, 8])  # (2,) selected targets
        training = masked_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        assert training.loss.ndim == 0
        assert tuple(training.logits.shape) == (2, 5, 33)
        """
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "HF_HOME": str(tmp_path / "hf-home"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_hub_publication_is_disabled_by_default(tiny_model: PLM) -> None:
    calls = []

    def unexpected_api_factory() -> NoReturn:
        calls.append("api_factory")
        raise AssertionError("The Hub API must not be constructed by default.")

    result = publish_model_to_hub(
        tiny_model,
        "Synthyra/test-model",
        api_factory=unexpected_api_factory,
    )

    assert result is None
    assert calls == []
    assert not hasattr(tiny_model, "push_code_and_config_to_hub")
    assert not hasattr(tiny_model, "push_weights_to_hub")


@pytest.mark.parametrize("arguments", [["--push-to-hub"], ["--masked-diffusion"], ["--mask-rate", "0.2"]])
def test_research_cli_rejects_publication_and_objective_overrides(arguments: list[str]) -> None:
    from speedrunning_plms.research.engine import main

    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("max_shard_size", ["5GB", "1KB"])
def test_opted_in_hub_publication_is_one_complete_artifact(tiny_model: PLM, monkeypatch: pytest.MonkeyPatch, max_shard_size: str) -> None:
    calls = []
    save_pretrained = tiny_model.save_pretrained

    def save_with_shard_limit(path: Path, **kwargs: object) -> None:
        save_pretrained(path, max_shard_size=max_shard_size, **kwargs)

    monkeypatch.setattr(tiny_model, "save_pretrained", save_with_shard_limit)

    class RecordingApi:
        def create_repo(self, **kwargs: object) -> None:
            calls.append(("create_repo", kwargs))

        def upload_folder(self, folder_path: Path, **kwargs: object) -> dict[str, str]:
            folder = Path(folder_path)
            requirements = (folder / "requirements.txt").read_text(encoding="utf-8")
            restored = PLM.from_pretrained(folder, local_files_only=True)
            assert restored.state_dict().keys() == tiny_model.state_dict().keys()
            for key, expected in tiny_model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], expected)
            calls.append(
                (
                    "upload_folder",
                    kwargs,
                    {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()},
                    json.loads((folder / "config.json").read_text(encoding="utf-8")),
                    requirements,
                )
            )
            return {"commit": "final-artifact"}

    result = publish_model_to_hub(
        tiny_model,
        "Synthyra/test-model",
        enabled=True,
        api_factory=RecordingApi,
    )

    assert result == {"commit": "final-artifact"}
    assert calls[0] == (
        "create_repo",
        {"repo_id": "Synthyra/test-model", "repo_type": "model", "exist_ok": True},
    )
    assert len(calls) == 2
    _, upload_kwargs, files, config, requirements = calls[1]
    assert upload_kwargs == {
        "repo_id": "Synthyra/test-model",
        "repo_type": "model",
        "commit_message": "Publish final trained model artifact",
    }
    assert {"config.json", "plm.py", "attention.py", "layers.py", "requirements.txt"} <= files
    if max_shard_size == "1KB":
        assert "model.safetensors.index.json" in files
        assert "model.safetensors" not in files
        assert len([name for name in files if name.endswith(".safetensors")]) > 1
    else:
        assert "model.safetensors" in files
    assert config["auto_map"]["AutoModelForMaskedLM"] == "plm.PLM"
    assert "AutoModel" not in config["auto_map"]
    assert requirements == "torch>=2.5\ntransformers>=4.57.6,<5\n"
