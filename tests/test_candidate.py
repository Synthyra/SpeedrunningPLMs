"""The record's candidate against the fixed benchmark: the shape rule, the forward contracts, and one end-to-end run."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import torch

from pathlib import Path
from types import ModuleType

from speedrunning_plms import benchmark
from speedrunning_plms.strategies import Strategies, default_genome, load_genome
from speedrunning_plms.strategies.attention import document_dense_mask
from speedrunning_plms.strategies.optimizer import Muon, polar_express


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
    model = candidate.build_candidate(default_genome(), max_tokens=64)
    receipt = benchmark.check_architecture(model)
    assert receipt["blocks"] == 36 and receipt["parameters"] > 190_000_000
    assert receipt["block_shapes"][0]["mlp"] == [(768, 2048), (2048, 768)]


def test_baseline_genome_is_000_with_every_strategy_off() -> None:
    """genomes/000_baseline.json, the 1-hour track's seed: 000's recipe in the registry, every family at its off value."""
    specification = importlib.util.spec_from_file_location("baseline_000", PROJECT / "candidates" / "000_baseline.py")
    assert specification is not None and specification.loader is not None
    baseline = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = baseline
    specification.loader.exec_module(baseline)
    recipe = baseline.Recipe()
    genome = load_genome(PROJECT / "genomes" / "000_baseline.json")
    record = default_genome()
    changed = {(family, field) for family, fields in genome.items() for field, value in fields.items() if value != record[family][field]}
    assert changed == {("positional", "variant"), ("softcap", "variant"), ("optimizer", "variant"), ("optimizer", "muon_lr"),
                       ("optimizer", "embed_lr"), ("optimizer", "head_lr")}
    assert genome["positional"]["variant"] == "none" and genome["positional"]["rope_base"] == recipe.rope_base
    assert genome["softcap"]["variant"] == "none" and genome["optimizer"]["variant"] == "muon"
    optimizer, schedule, batch = genome["optimizer"], genome["schedule"], genome["batch"]
    assert (optimizer["muon_lr"], optimizer["embed_lr"], optimizer["head_lr"]) == (recipe.muon_lr, recipe.embed_lr, recipe.head_lr)
    assert (optimizer["adam_beta1"], optimizer["adam_beta2"], optimizer["weight_decay"]) == (*recipe.adam_betas, recipe.weight_decay)
    assert (schedule["variant"], schedule["warmup_fraction"], schedule["cooldown_fraction"], schedule["final_lr_ratio"]) == \
        ("trapezoid", recipe.warmup_fraction, recipe.cooldown_fraction, recipe.final_lr_ratio)
    assert (schedule["momentum"], schedule["momentum_start"], schedule["momentum_warmup"], schedule["momentum_cooldown_end"]) == \
        (recipe.muon_momentum, recipe.muon_momentum_start, recipe.muon_momentum_warmup, recipe.muon_momentum)
    assert genome["attention"] == {"variant": "standard", "heads": recipe.heads, "key_offset": 0.0, "score_cap": 0.0, "local_every": 0,
                                   "kv_share": 0}
    assert batch["variant"] == "packed" and batch["window"] == recipe.window
    assert Strategies(genome).batch.step_tokens == recipe.window * recipe.windows_per_step  # 64 windows of 1,024
    switched_on = {"positional": "alibi", "attention": "standard", "kernel": "default", "residual": "prenorm", "stream": "single",
                   "activation": "relu2", "head": "linear", "softcap": "tanh", "loss": "plain", "optimizer": "normuon",
                   "schedule": "trapezoid", "batch": "packed"}  # families whose default is a choice, not "none"
    for family, fields in genome.items():
        if family not in switched_on:
            assert fields["variant"] == "none", family
    assert genome["residual"]["branch_multiplier"] == 1.0 and genome["residual"]["branch_ramp"] == 0.0
    assert genome["loss"]["z_weight"] == 0.0 and optimizer["cautious_weight_decay"] == 0.0 and optimizer["decay_timescale"] == 0.0
    assert genome["positional"]["periodic_scale"] == 0.0 and genome["positional"]["length_bins"] == 0
    assert genome["positional"]["terminus"] == 0 and schedule["muon_late_scale"] == 1.0
    assert genome["positional"]["nope_every"] == 0 and genome["positional"]["rope_fraction"] == 1.0


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
    assert candidate.Strategies(default_genome()).batch.config.window > 0


def test_candidate_forward_contracts() -> None:
    candidate = load_candidate()
    model = candidate.build_candidate(default_genome(), max_tokens=32).eval()
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
    dense = document_dense_mask(torch.tensor([[0, 5, 5, 0, 6, 6]]))  # (1, 1, 6, 6)
    assert dense[0, 0, 1, 2] and not dense[0, 0, 1, 4] and dense[0, 0, 4, 5]


def test_candidate_muon_banks_match_the_per_matrix_update() -> None:
    """Banking by shape changes nothing: a matrix's update equals the one a bank of itself alone produces, and the
    batched Polar Express equals the package's per-matrix one."""
    from speedrunning_plms.optim.muon import zeropower_via_polar_express

    torch.manual_seed(0)
    G = torch.randn(96, 48)  # (m, n)
    assert torch.allclose(polar_express(G[None])[0], zeropower_via_polar_express(G), atol=1e-5)
    banked = [torch.nn.Parameter(torch.randn(shape) * 0.02) for shape in ((48, 48), (48, 48), (96, 48), (48, 96))]
    alone = [torch.nn.Parameter(parameter.detach().clone()) for parameter in banked]
    optimizers = [Muon(banked, lr=0.02, momentum=0.9, weight_decay=0.0),
                  *(Muon([parameter], lr=0.02, momentum=0.9, weight_decay=0.0) for parameter in alone)]
    for _ in range(2):
        gradients = [torch.randn_like(parameter) for parameter in banked]
        for parameters in (banked, alone):
            for parameter, gradient in zip(parameters, gradients, strict=True):
                parameter.grad = gradient.clone()
        for optimizer in optimizers:
            optimizer.step()
    for expected, actual in zip(alone, banked, strict=True):
        difference = float((expected.detach() - actual.detach()).abs().max())
        assert torch.allclose(expected.detach(), actual.detach(), atol=1e-6), difference


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
    assert run_result["candidate_parts"]["train.py"] == benchmark.candidate_sha256(PROJECT / "train.py")
    assert run_result["genome_changes"] == {"batch": {"window": 128, "windows_per_step": 2}}


def test_eval_spacing_keeps_the_record_and_fits_slow_evals_under_the_cap() -> None:
    candidate = load_candidate()
    assert candidate.eval_interval(4.2, 600.0) == 90.0 and candidate.eval_interval(4.2, 180.0) == 90.0
    interval = candidate.eval_interval(12.0, 600.0)  # evals at 90, 260, 430 and the final one: 48 s
    evals = 1 + math.ceil((600.0 - 90.0) / interval)  # the first, the spaced ones before the end, and the final one
    assert interval == 170.0 and 12.0 * evals <= candidate.EVAL_TARGET_SECONDS
    assert candidate.eval_interval(30.0, 600.0) == math.inf
