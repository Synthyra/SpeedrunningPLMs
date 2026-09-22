"""Optimizer partitioning, numerical updates, and checkpoint checks."""

import copy
import pytest
import torch

from speedrunning_plms.models import PLM, PLMConfig
from speedrunning_plms.optim.factory import build_optimizers
from speedrunning_plms.optim.muon import (
    Muon, zeropower_via_newtonschulz5, zeropower_via_polar_express,
)


def _optimizers(model: torch.nn.Module, **overrides: object) -> list[torch.optim.Optimizer]:
    options = dict(optimizer="muon", learning_rate=3e-4, muon_lr=0.02,
                   weight_decay=0.01, muon_momentum=0.95, muon_steps=5,
                   muon_backend="newton_schulz", fused_adam=False)
    options.update(overrides)
    return build_optimizers(model, **options)


@pytest.mark.parametrize("shape", [(3, 7), (7, 3), (4, 4)])
def test_newton_schulz_matches_singular_value_polynomial(shape: tuple[int, int]) -> None:
    generator = torch.Generator().manual_seed(7)
    G = torch.randn(shape, generator=generator)  # (m, n)
    U, s, Vh = torch.linalg.svd(G.double(), full_matrices=False)  # (m, r), (r,), (r, n)
    s = s / (torch.linalg.vector_norm(s) + 1e-7)  # (r,)
    for _ in range(5):
        s = 3.4445 * s - 4.7750 * s.pow(3) + 2.0315 * s.pow(5)  # (r,)
    expected = (U * s) @ Vh  # (m, n)
    actual = zeropower_via_newtonschulz5(G, 5)  # (m, n)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("backend", ["newton_schulz", "polar_express"])
def test_muon_decay_missing_gradients_and_world_size(
    backend: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "7")
    active = torch.nn.Parameter(torch.ones(3, 2))  # (3, 2)
    unused = torch.nn.Parameter(torch.ones(2, 4))  # (2, 4)
    optimizer = Muon([active, unused], lr=0.1, weight_decay=0.2, orthogonalization=backend)
    active.grad = torch.zeros_like(active)  # (3, 2)
    optimizer.step()
    torch.testing.assert_close(active, torch.full_like(active, 0.98))
    torch.testing.assert_close(unused, torch.ones_like(unused))
    assert unused not in optimizer.state
    assert torch.count_nonzero(active.grad) == 0


def test_muon_nesterov_shape_scale_and_gradient_preservation() -> None:
    parameter = torch.nn.Parameter(torch.zeros(6, 3))  # (6, 3)
    gradient = torch.cat((torch.eye(3), torch.zeros(3, 3)))  # (6, 3)
    parameter.grad = gradient.clone()  # (6, 3)
    optimizer = Muon([parameter], lr=0.02, momentum=0.8, ns_steps=1)
    optimizer.step()
    # First-step Nesterov direction is (1 - momentum**2) * gradient.
    singular_value = 0.36 / (0.36 * 3**0.5 + 1e-7)
    orthogonalized = 3.4445 * singular_value - 4.7750 * singular_value**3 + 2.0315 * singular_value**5
    torch.testing.assert_close(parameter, -0.02 * 2**0.5 * orthogonalized * gradient)
    torch.testing.assert_close(parameter.grad, gradient)
    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], 0.2 * gradient)


@pytest.mark.parametrize("backend", ["newton_schulz", "polar_express"])
def test_muon_checkpoint_continuation_and_bfloat16_state(backend: str) -> None:
    parameter = torch.nn.Parameter(torch.ones(3, 4))  # (3, 4)
    optimizer = Muon([parameter], orthogonalization=backend)
    parameter.grad = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12  # (3, 4)
    optimizer.step()
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())  # (3, 4)
    restored_optimizer = Muon([restored_parameter], orthogonalization=backend)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    restored_parameter.grad = parameter.grad.clone()  # (3, 4)
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored_parameter, parameter, atol=0, rtol=0)
    low_precision = torch.nn.Parameter(torch.ones(3, 4, dtype=torch.bfloat16))  # (3, 4)
    low_optimizer = Muon([low_precision], orthogonalization=backend)
    low_precision.grad = torch.ones_like(low_precision)  # (3, 4)
    low_optimizer.step()
    assert low_optimizer.state[low_precision]["momentum_buffer"].dtype == torch.float32
    assert torch.isfinite(low_precision).all()


def test_polar_express_preserves_orientation_and_improves_singular_values() -> None:
    G = torch.diag(torch.tensor([1.0, 0.5, 0.2]))  # (3, 3)
    actual = zeropower_via_polar_express(G)  # (3, 3)
    torch.testing.assert_close(actual, torch.diag(actual.diagonal()))
    assert torch.max(torch.abs(torch.linalg.svdvals(actual) - 1)) < 0.15
    torch.testing.assert_close(zeropower_via_polar_express(torch.zeros(3, 5)), torch.zeros(3, 5))
    with pytest.raises(ValueError, match="ns_steps=5"):
        Muon([torch.nn.Parameter(G)], ns_steps=4, orthogonalization="polar_express")


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_factory_partitions_plm_parameters_once(tied: bool, wrapped: bool) -> None:
    model = PLM(PLMConfig(hidden_size=8, num_attention_heads=2, num_hidden_layers=2,
                          tie_embeddings=tied))
    candidate = model
    if wrapped:
        candidate = torch.nn.Module()
        candidate.add_module("_orig_mod", model)
    muon, adam = _optimizers(candidate, fused_adam=True)
    assert isinstance(muon, Muon)
    assert isinstance(adam, torch.optim.AdamW)
    muon_ids = {id(p) for group in muon.param_groups for p in group["params"]}
    adam_ids = {id(p) for group in adam.param_groups for p in group["params"]}
    assert not muon_ids & adam_ids
    assert muon_ids | adam_ids == {id(p) for p in model.parameters()}
    assert id(model.embedding.weight) in adam_ids
    assert all(id(p) in adam_ids for p in model.lm_head.parameters())
    for layer in model.transformer.layers:
        assert all(id(projection.weight) in muon_ids
                   for projection in (layer.attn.Wq, layer.attn.Wk, layer.attn.Wv))
    assert adam.param_groups[0]["fused"] is None
    assert adam.param_groups[0]["foreach"] is None


@pytest.mark.parametrize("fused_adam", [False, True])
def test_adamw_factory_preserves_original_update_and_frozen_parameter_group(fused_adam: bool) -> None:
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    reference = copy.deepcopy(model)
    optimizer, = _optimizers(model, optimizer="adamw", fused_adam=fused_adam)
    expected = torch.optim.AdamW(reference.parameters(), lr=3e-4, weight_decay=0.01)
    model.weight.grad = torch.ones_like(model.weight)  # (2, 3)
    reference.weight.grad = model.weight.grad.clone()  # (2, 3)
    optimizer.step()
    expected.step()
    assert len(optimizer.param_groups[0]["params"]) == 2
    assert optimizer.param_groups[0]["fused"] is None
    assert optimizer.param_groups[0]["foreach"] is None
    torch.testing.assert_close(model.weight, reference.weight, atol=0, rtol=0)


def test_factory_keeps_value_embeddings_and_input_dependent_gates_in_adamw() -> None:
    model = PLM(PLMConfig(hidden_size=8, num_attention_heads=2, num_hidden_layers=2,
                          value_embeddings=True, value_embedding_gate=True))
    muon, adam = _optimizers(model)
    adam_ids = {id(p) for group in adam.param_groups for p in group["params"]}
    muon_ids = {id(p) for group in muon.param_groups for p in group["params"]}
    assert all(id(p) in adam_ids for p in model.value_embeds.parameters())
    for layer in model.transformer.layers:
        assert id(layer.attn.value_gate.weight) in adam_ids
        assert id(layer.attn.Wv.weight) in muon_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("backend", ["newton_schulz", "polar_express"])
def test_cuda_muon_matches_cpu_direction(backend: str) -> None:
    initial = torch.arange(48, dtype=torch.float32).reshape(8, 6) / 48  # (8, 6)
    cpu = torch.nn.Parameter(initial.clone())  # (8, 6)
    cuda = torch.nn.Parameter(initial.cuda())  # (8, 6)
    generator = torch.Generator().manual_seed(13)
    gradient = torch.randn(8, 6, generator=generator)  # (8, 6)
    cpu.grad = gradient  # (8, 6)
    cuda.grad = gradient.cuda()  # (8, 6)
    Muon([cpu], orthogonalization=backend).step()
    Muon([cuda], orthogonalization=backend).step()
    torch.testing.assert_close(cuda.cpu(), cpu, atol=2e-3, rtol=2e-3)
