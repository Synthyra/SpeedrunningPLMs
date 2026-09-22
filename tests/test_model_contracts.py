import pytest
import torch
import torch.nn.functional as F

from pathlib import Path

from speedrunning_plms.models import PLM, PLMConfig
from speedrunning_plms.models.attention import SelfAttention


def make_model(architecture: str = "standard") -> PLM:
    torch.manual_seed(19)
    model = PLM(
        PLMConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_hidden_layers=2,
            num_unet_layers=8 if architecture == "patch_bottleneck" else 4,
            num_extra_layers=1,
            max_sequence_length=4,
            vocab_size=33,
            unet=architecture == "unet",
            patch_unet=architecture.startswith("patch"),
            compile_flex_attention=False,
            tokenizer_name=None,
            cls_token_id=0,
            eos_token_id=2,
            pad_token_id=1,
            mask_token_id=32,
        )
    )
    # Fresh attention outputs are zero; activate them so mask tests detect leakage.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, SelfAttention):
                torch.nn.init.normal_(module.Wo.weight, std=0.1)  # (d, d)
    return model


@pytest.fixture(params=["standard", "unet", "patch", "patch_bottleneck"])
def model(request: pytest.FixtureRequest) -> PLM:
    return make_model(request.param)


def test_cpu_masked_loss_and_backward(model: PLM) -> None:
    input_ids = torch.tensor([[0, 32, 6, 2], [0, 7, 32, 2]])  # (2, 4)
    labels = torch.tensor([[-100, 5, -100, -100], [-100, -100, 8, -100]])  # (2, 4)
    output = model(input_ids, labels=labels, output_hidden_states=True)

    assert output.logits.shape == (2, 4, 33)
    assert output.hidden_states[0].shape == (2, 4, 8)
    assert output.logits.device.type == "cpu"
    assert torch.isfinite(output.logits).all()
    supervised_logits = torch.stack([output.logits[0, 1], output.logits[1, 2]])  # (2, 33)
    expected_loss = F.cross_entropy(supervised_logits, torch.tensor([5, 8]))  # ()
    torch.testing.assert_close(output.loss, expected_loss)

    output.loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    for parameter in (model.embedding.weight, model.lm_head.decoder.weight):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0
    attention = next(module for module in model.modules() if isinstance(module, SelfAttention))
    assert attention.Wq.weight.grad is not None
    assert attention.Wq.weight.grad.abs().sum() > 0


def test_batch_matches_individual_sequences(model: PLM) -> None:
    model.eval()
    input_ids = torch.tensor([[0, 5, 2, 1], [0, 7, 32, 2]])  # (2, 4)
    attention_mask = input_ids != 1  # (2, 4)
    with torch.no_grad():
        batched = model(input_ids, attention_mask=attention_mask).logits  # (2, 4, 33)
        individual = torch.cat(
            [
                model(row[None, :], attention_mask=mask[None, :]).logits
                for row, mask in zip(input_ids, attention_mask)
            ]
        )  # (2, 4, 33)
    torch.testing.assert_close(batched, individual, atol=1e-6, rtol=1e-5)


def test_sharded_save_preserves_predictions(model: PLM, tmp_path: Path) -> None:
    model.eval()
    input_ids = torch.tensor([[0, 5, 32, 2]])  # (1, 4)
    with torch.no_grad():
        expected = model(input_ids).logits  # (1, 4, 33)
    model.save_pretrained(tmp_path, max_shard_size="10KB")
    assert (tmp_path / "model.safetensors.index.json").is_file()
    restored = PLM.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        actual = restored(input_ids).logits  # (1, 4, 33)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("architecture", ["standard", "unet"])
def test_masked_tokens_cannot_change_visible_predictions(architecture: str) -> None:
    model = make_model(architecture).eval()
    input_ids = torch.tensor([[0, 5, 6, 2, 1, 1]])  # (1, 6)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])  # (1, 6)
    changed = torch.tensor([[0, 5, 6, 2, 9, 32]])  # (1, 6)
    with torch.no_grad():
        expected = model(input_ids, attention_mask=attention_mask).logits[:, :4]  # (1, 4, 33)
        actual = model(changed, attention_mask=attention_mask).logits[:, :4]  # (1, 4, 33)
        unmasked = model(changed, attention_mask=torch.ones_like(changed)).logits[:, :4]  # (1, 4, 33)
        automatic = model(input_ids).logits[:, :4]  # (1, 4, 33)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(automatic, expected, atol=0, rtol=0)
    assert not torch.allclose(unmasked, expected)


@pytest.mark.parametrize("architecture", ["standard", "unet"])
def test_packed_documents_are_isolated(architecture: str) -> None:
    model = make_model(architecture).eval()
    packed = torch.tensor([0, 5, 6, 2, 0, 7, 8, 2])  # (8,)
    changed = torch.tensor([0, 5, 6, 2, 0, 11, 12, 2])  # (8,)
    with torch.no_grad():
        expected = model(packed).logits  # (8, 33)
        actual = model(changed).logits  # (8, 33)
        isolated = model(packed[:4]).logits  # (4, 33)
    torch.testing.assert_close(actual[:4], expected[:4], atol=0, rtol=0)
    torch.testing.assert_close(isolated, expected[:4], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(actual[4:], expected[4:])


@pytest.mark.parametrize("architecture", ["standard", "unet"])
def test_unit_window_disables_cross_token_attention(architecture: str) -> None:
    model = make_model(architecture).eval()
    input_ids = torch.tensor([[0, 5, 6, 2]])  # (1, 4)
    changed = torch.tensor([[0, 11, 12, 2]])  # (1, 4)
    with torch.no_grad():
        expected = model(input_ids, sliding_window_size=1).logits[:, 0]  # (1, 33)
        actual = model(changed, sliding_window_size=1).logits[:, 0]  # (1, 33)
        full_original = model(input_ids).logits[:, 0]  # (1, 33)
        full_changed = model(changed).logits[:, 0]  # (1, 33)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.allclose(full_original, full_changed)


@pytest.mark.parametrize("field", ["input_ids", "attention_mask", "labels"])
def test_invalid_input_shapes_raise_clear_errors(field: str) -> None:
    model = make_model()
    input_ids = torch.tensor([[0, 5, 6, 2]])  # (1, 4)
    arguments = {"input_ids": input_ids}
    arguments[field] = torch.zeros((1, 1, 4), dtype=torch.long)  # (1, 1, 4)
    with pytest.raises(ValueError, match=field):
        model(**arguments)


@pytest.mark.parametrize("explicit_rate", [False, True])
def test_diffusion_loss_scaling_only_applies_during_training(explicit_rate: bool) -> None:
    model = make_model()
    model.masked_diffusion = True
    input_ids = torch.tensor([[0, 32, 2, 1]])  # (1, 4)
    labels = torch.tensor([[-100, 5, -100, -100]])  # (1, 4)
    mask_rate = torch.tensor(0.5) if explicit_rate else None  # () or None
    training = model(input_ids, labels=labels, mask_rate=mask_rate)
    cross_entropy = F.cross_entropy(training.logits[:, 1], torch.tensor([5]))  # ()
    rate = 0.5 if explicit_rate else 1 / 3
    torch.testing.assert_close(training.loss, cross_entropy / rate)

    model.eval()
    evaluation = model(input_ids, labels=labels, mask_rate=mask_rate)
    torch.testing.assert_close(evaluation.loss, cross_entropy)
