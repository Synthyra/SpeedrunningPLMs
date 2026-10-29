"""The baseline candidate against the fixed benchmark: the shape rule, the forward contracts, and one end-to-end run."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import torch

from pathlib import Path
from types import ModuleType

from speedrunning_plms import benchmark


PROJECT = Path(__file__).resolve().parents[1]


def load_candidate() -> ModuleType:
    specification = importlib.util.spec_from_file_location("candidate_under_test", PROJECT / "train.py")
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module  # dataclasses resolve string annotations through sys.modules
    specification.loader.exec_module(module)
    return module


def test_baseline_candidate_passes_the_shape_rule() -> None:
    candidate = load_candidate()
    model = candidate.Candidate(candidate.Recipe(), max_tokens=64)
    receipt = benchmark.check_architecture(model)
    assert receipt["blocks"] == 36 and receipt["parameters"] > 190_000_000
    assert receipt["block_shapes"][0]["mlp"] == [(768, 2048), (2048, 768)]


def test_shape_rule_refuses_other_shapes() -> None:
    candidate = load_candidate()

    class Wrong(torch.nn.Module):
        def __init__(self, blocks: int, hidden: int, intermediate: int) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList()
            for _ in range(blocks):
                block = torch.nn.Module()
                block.attention = torch.nn.Linear(hidden, hidden, bias=False)
                block.mlp = torch.nn.Sequential(torch.nn.Linear(hidden, intermediate, bias=False), torch.nn.Linear(intermediate, hidden, bias=False))
                self.blocks.append(block)

    for blocks, hidden, intermediate in ((35, 768, 2048), (36, 512, 2048), (36, 768, 2304)):
        try:
            benchmark.check_architecture(Wrong(blocks, hidden, intermediate))
        except ValueError:
            continue
        raise AssertionError(f"{blocks} blocks, {hidden} wide, {intermediate} intermediate was accepted")
    benchmark.check_architecture(Wrong(36, 768, 2048))
    assert candidate.Recipe().window > 0


def test_candidate_forward_contracts() -> None:
    candidate = load_candidate()
    model = candidate.Candidate(candidate.Recipe(), max_tokens=32).eval()
    input_ids = torch.randint(4, 24, (2, 12))
    input_ids[:, 0] = benchmark.CLS_ID
    input_ids[0, 9:] = benchmark.PAD_ID
    attention_mask = input_ids != benchmark.PAD_ID
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        states = model.hidden_states(input_ids, attention_mask)
    assert logits.shape == (2, 12, benchmark.VOCAB_SIZE)
    assert len(states) == benchmark.BLOCKS + 1 and states[-1].shape == (2, 12, benchmark.HIDDEN)
    assert torch.equal(model.residue_columns, benchmark.ESM2_RESIDUE_COLUMNS)
    dense = candidate.document_dense_mask(torch.tensor([[0, 5, 5, 0, 6, 6]]))  # (1, 1, 6, 6)
    assert dense[0, 0, 1, 2] and not dense[0, 0, 1, 4] and dense[0, 0, 4, 5]


def test_candidate_muon_matches_the_per_matrix_muon() -> None:
    """The candidate's banked Muon applies the package's per-matrix Polar Express update to every matrix."""
    from speedrunning_plms.optim.muon import Muon

    candidate = load_candidate()
    torch.manual_seed(0)
    reference = [torch.nn.Parameter(torch.randn(shape) * 0.02) for shape in ((48, 48), (48, 48), (96, 48), (48, 96))]
    banked = [torch.nn.Parameter(parameter.detach().clone()) for parameter in reference]
    optimizers = (Muon(reference, lr=0.02, momentum=0.9, orthogonalization="polar_express"),
                  candidate.Muon(banked, lr=0.02, momentum=0.9, weight_decay=0.0))
    for _ in range(2):
        gradients = [torch.randn_like(parameter) for parameter in reference]
        for parameters in (reference, banked):
            for parameter, gradient in zip(parameters, gradients, strict=True):
                parameter.grad = gradient.clone()
        for optimizer in optimizers:
            optimizer.step()
    for expected, actual in zip(reference, banked, strict=True):
        assert torch.allclose(expected, actual, atol=1e-6), float((expected - actual).abs().max())


def test_end_to_end_cpu_run_writes_a_result(tiny_data: Path, tmp_path: Path) -> None:
    output = tmp_path / "run"
    command = [sys.executable, str(PROJECT / "train.py"), "--data-dir", str(tiny_data), "--output-dir", str(output),
               "--name", "test", "--max-steps", "1", "--no-compile", "--no-wandb", "--budget", "30", "--device", "cpu",
               "--window", "128", "--windows-per-step", "2", "--skip-final-evals"]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT)
    assert completed.returncode == 0, completed.stderr[-3000:]
    run_result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert run_result["architecture"]["blocks"] == 36 and run_result["clock"]["steps"] == 1
    assert run_result["selected_step"] == 1 and run_result["curve"] and (output / "best.pt").exists()
    assert run_result["comparison_key"] == benchmark.comparison_key(run_result["benchmark_id"], run_result["evaluator_sha256"], run_result["clock"])
    assert run_result["candidate_sha256"] == benchmark.candidate_sha256(PROJECT / "train.py")
