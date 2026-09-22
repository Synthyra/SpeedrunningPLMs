import pytest
import torch

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from speedrunning_plms.models import PLM, PLMConfig
from speedrunning_plms.models.attention import SelfAttention


def make_model(**overrides: object) -> PLM:
    values = dict(
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=3,
        num_unet_layers=4,
        max_sequence_length=8,
        compile_flex_attention=False,
        token_dropout=False,
        tokenizer_name=None,
        cls_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        mask_token_id=32,
    )
    values.update(overrides)
    torch.manual_seed(91)
    model = PLM(PLMConfig(**values))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, SelfAttention):
                torch.nn.init.normal_(module.Wo.weight, std=0.1)  # (d, d); expose attention changes.
    return model


@pytest.mark.parametrize("architecture", ["standard", "unet", "patch"])
def test_fused_sdpa_matches_flex_outputs_and_gradients(architecture: str) -> None:
    options = dict(unet=architecture == "unet", patch_unet=architecture == "patch", num_hidden_layers=2)
    reference = make_model(**options)
    optimized = make_model(**options, fused_qkv=True, attention_backend="sdpa")
    optimized.load_state_dict(reference.state_dict(), strict=True)
    input_ids = torch.tensor([[0, 32, 2, 1, 0, 7, 2, 1], [1, 1, 1, 1, 1, 1, 1, 1]])  # (2, 8)
    labels = torch.full_like(input_ids, -100)  # (2, 8)
    labels[0, 1] = 5  # ()
    expected = reference(input_ids, labels=labels, sliding_window_size=3)
    with patch("speedrunning_plms.models.plm.create_block_mask", side_effect=AssertionError("SDPA built a BlockMask")):
        actual = optimized(input_ids, labels=labels, sliding_window_size=3)
    torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=1e-5)
    assert torch.isfinite(actual.logits).all()
    expected.loss.backward()
    actual.loss.backward()
    for (name, parameter), (other_name, other_parameter) in zip(reference.named_parameters(), optimized.named_parameters()):
        assert name == other_name
        if parameter.grad is not None:
            assert torch.isfinite(other_parameter.grad).all(), name
            torch.testing.assert_close(other_parameter.grad, parameter.grad, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("value_embeddings,embedding_residual", [(True, False), (False, True), (True, True)])
def test_standard_features_are_independent_and_trainable(value_embeddings: bool, embedding_residual: bool) -> None:
    model = make_model(attention_backend="sdpa", value_embeddings=value_embeddings, embedding_residual=embedding_residual)
    assert hasattr(model, "value_embeds") == value_embeddings
    assert hasattr(model.transformer.layers[0], "lambdas") == embedding_residual
    input_ids = torch.tensor([[0, 32, 7, 2]])  # (1, 4)
    labels = torch.tensor([[-100, 5, -100, -100]])  # (1, 4)
    model(input_ids, labels=labels).loss.backward()
    if value_embeddings:
        assert len(model.value_embeds.embed) == 3
        for embedding in model.value_embeds.embed:
            assert embedding.weight.grad is not None
            assert embedding.weight.grad.abs().sum() > 0
    if embedding_residual:
        for layer in model.transformer.layers:
            assert layer.lambdas.grad is not None
            assert layer.lambdas.grad.abs().sum() > 0


def test_value_gate_starts_unchanged_and_receives_gradients() -> None:
    reference = make_model(attention_backend="sdpa", value_embeddings=True)
    gated = make_model(attention_backend="sdpa", value_embeddings=True, value_embedding_gate=True)
    missing, unexpected = gated.load_state_dict(reference.state_dict(), strict=False)
    assert len(missing) == 3 and all(name.endswith("value_gate.weight") for name in missing)
    assert not unexpected
    input_ids = torch.tensor([[0, 32, 7, 2]])  # (1, 4)
    labels = torch.tensor([[-100, 5, -100, -100]])  # (1, 4)
    expected = reference(input_ids, labels=labels)
    actual = gated(input_ids, labels=labels)
    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
    actual.loss.backward()
    for layer in gated.transformer.layers:
        assert layer.attn.value_gate.weight.grad.abs().sum() > 0


def test_sdpa_document_padding_and_window_isolation() -> None:
    model = make_model(attention_backend="sdpa", num_hidden_layers=1).eval()
    input_ids = torch.tensor([[0, 5, 6, 2, 0, 7, 8, 2]])  # (1, 8)
    changed = input_ids.clone()  # (1, 8)
    changed[0, 6] = 10  # ()
    expected = model(input_ids, sliding_window_size=2).logits  # (1, 8, 33)
    actual = model(changed, sliding_window_size=2).logits  # (1, 8, 33)
    torch.testing.assert_close(actual[:, :5], expected[:, :5], atol=0, rtol=0)
    assert not torch.allclose(actual[:, 5], expected[:, 5])
    valid_tokens = torch.ones_like(input_ids, dtype=torch.bool)  # (1, 8)
    valid_tokens[0, 6] = False  # ()
    masked = model(input_ids, attention_mask=valid_tokens).logits  # (1, 8, 33)
    masked_changed = model(changed, attention_mask=valid_tokens).logits  # (1, 8, 33)
    torch.testing.assert_close(masked[:, 5], masked_changed[:, 5], atol=0, rtol=0)


def test_optimized_features_survive_hf_roundtrip(tmp_path: Path) -> None:
    model = make_model(
        attention_backend="sdpa", fused_qkv=True, value_embeddings=True,
        embedding_residual=True, value_embedding_gate=True,
    ).eval()
    input_ids = torch.tensor([[0, 32, 7, 2]])  # (1, 4)
    expected = model(input_ids).logits  # (1, 4, 33)
    model.save_pretrained(tmp_path)
    restored = PLM.from_pretrained(tmp_path, local_files_only=True).eval()
    for name in ("fused_qkv", "value_embeddings", "embedding_residual", "value_embedding_gate"):
        assert getattr(restored.config, name) is True
    assert restored.config.attention_backend == "sdpa"
    torch.testing.assert_close(restored(input_ids).logits, expected, atol=0, rtol=0)


@pytest.mark.parametrize("options", [dict(unet=True), dict(patch_unet=True)])
def test_unet_defaults_and_explicit_disabling(options: dict[str, bool]) -> None:
    defaults = PLMConfig(**options)
    assert defaults.value_embeddings and defaults.embedding_residual
    model = make_model(**options, num_hidden_layers=2, value_embeddings=False, embedding_residual=False, attention_backend="sdpa")
    assert not hasattr(model, "value_embeds")
    input_ids = torch.tensor([[0, 32, 7, 2, 1, 1, 1, 1]])  # (1, 8)
    assert torch.isfinite(model(input_ids).logits).all()


def test_invalid_feature_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="attention_backend"):
        PLMConfig(attention_backend="unknown")
    with pytest.raises(ValueError, match="requires value_embeddings"):
        PLMConfig(value_embedding_gate=True)


@pytest.mark.parametrize("unet", [False, True])
def test_attention_accepts_legacy_config_namespace(unet: bool) -> None:
    legacy_config = SimpleNamespace(
        hidden_size=8, num_attention_heads=2, unet=unet, compile_flex_attention=False,
    )
    attention = SelfAttention(legacy_config)
    assert attention.value_embeddings == unet
    assert attention.fused_qkv is False
    assert attention.attention_backend == "flex"
    assert attention.value_gate is None
    inputs = torch.randn(2, 1, 8, requires_grad=True)  # (2, 1, 8)
    values = torch.randn_like(inputs) if unet else None  # Optional (2, 1, 8).
    output = attention(inputs, vi=values)  # (2, 1, 8)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert inputs.grad is not None
