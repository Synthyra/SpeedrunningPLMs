"""The strategy registry: every variant builds, passes the shape rule and trains a CPU step; the genome round-trips;
and the default genome reproduces the record's file exactly.

Symbols: b windows, l tokens, v = 33 tokens.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import json
import math
import subprocess
import sys
import typing
import numpy as np
import pytest
import torch

from pathlib import Path
from types import ModuleType
from typing import Any

from speedrunning_plms import benchmark
from speedrunning_plms.benchmark.tokens import STANDARD_RESIDUES
from speedrunning_plms.strategies import (
    FAMILIES, GenomeConflict, Strategies, default_genome, genome_document, genome_sha256, load_genome, resolve, save_genome,
)
from speedrunning_plms.strategies import memory as memory_module
from speedrunning_plms.strategies.attention import AttentionConfig, capped_attention, document_dense_mask, documents, shift_key_pairs, within_reach, without_own_value
from speedrunning_plms.strategies.batch import Batch, BatchConfig, best_fit
from speedrunning_plms.strategies.canon import neighbor_mix, same_protein_at
from speedrunning_plms.strategies.curriculum import Curriculum, CurriculumConfig
from speedrunning_plms.strategies.ema import Ema, EmaConfig, scaled_decay
from speedrunning_plms.strategies.grouping import GROUP_OF_CLASS, GROUPS, group_labels
from speedrunning_plms.strategies.kernel import Kernel, KernelConfig
from speedrunning_plms.strategies.lookahead import Lookahead, LookaheadConfig
from speedrunning_plms.strategies.loop import HELD, LoopConfig, gated_update, read_updates, reinjected
from speedrunning_plms.strategies.loss import Loss, LossConfig, residue_scores, substitution_difference
from speedrunning_plms.strategies.neighbor import pair_index, trigram_index
from speedrunning_plms.strategies.optimizer import Muon, OptimizerChoice, OptimizerConfig, amuse_beta, mona_accelerated, mud, polar_express, radially_braked, top_singular_value
from speedrunning_plms.strategies.positional import Positional, PositionalConfig, Rotary, length_bins, periodic_term, terminus_positions
from speedrunning_plms.strategies.protein_context import protein_sums, visible_composition
from speedrunning_plms.strategies.residual import Residual, ResidualConfig
from speedrunning_plms.strategies.residue_priors import BLOSUM62, UNIGRAM, substitution_targets, unigram
from speedrunning_plms.strategies.routing import chosen_groups, rebalanced
from speedrunning_plms.strategies.sam import Sam, SamConfig
from speedrunning_plms.strategies.schedule import Schedule, ScheduleConfig
from speedrunning_plms.strategies.sglr import BALANCE, routed_update, switch_balance, usage_entropy
from speedrunning_plms.strategies.shape import ShapeConfig
from speedrunning_plms.strategies.strategy import HOOKS, ForwardContext, StepState, norm as norm_rms


RECORD = ShapeConfig()  # the record's shape: 36 blocks, d = 768, f = 2,048


PROJECT = Path(__file__).resolve().parents[1]
# Run a candidate file with the clock's progress tied to applied steps, so its schedule is the same in every process.
DETERMINISTIC_RUN = """
import importlib.util, sys
from speedrunning_plms import benchmark
benchmark.Clock.progress = property(lambda clock: min(1.0, (clock.steps + 1) * 0.3))
specification = importlib.util.spec_from_file_location("candidate_under_test", sys.argv[1])
module = importlib.util.module_from_spec(specification)
sys.modules["candidate_under_test"] = module
specification.loader.exec_module(module)
sys.argv = sys.argv[1:]
module.main()
"""


def load_train() -> ModuleType:
    specification = importlib.util.spec_from_file_location("train_under_test", PROJECT / "train.py")
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def with_changes(changes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    genome = default_genome()
    for family, fields in changes.items():
        genome[family] = {**genome[family], **fields}
    return resolve({"families": genome})


VARIANTS = [(family, variant) for family, strategy in FAMILIES.items() for variant in strategy.Config.VARIANTS
            if variant != strategy.Config().variant]
COMPOSED = [
    {"attention": {"variant": "triangle_differential"}, "residual": {"variant": "ngpt"}, "stream": {"variant": "hyper"},
     "unet": {"variant": "mirrored"}, "value_embeddings": {"variant": "ends"}, "value_residual": {"variant": "first_block"},
     "gates": {"variant": "residual_channels"}, "neighbor": {"variant": "pair_table"}, "ema": {"variant": "tail", "start": 0.3},
     "embedding_mix": {"variant": "every_block", "initial_share": 0.1},
     "backprop_free": {"variant": "residual_es", "interval": 1}},
    {"attention": {"variant": "paired"}, "stream": {"variant": "dense", "span": 3}, "softcap": {"variant": "none"},
     "optimizer": {"variant": "muon", "cautious_weight_decay": 0.1}, "backprop_free": {"variant": "readout_warmstart", "stride": 12}},
    {"attention": {"variant": "differential", "heads": 6}, "stream": {"variant": "dense", "dynamic": 0},
     "backprop_free": {"variant": "unit_select", "interval": 1}},
    {"attention": {"variant": "triangle"}, "positional": {"rope_fraction": 0.1, "nope_every": 3},
     "logit_scale": {"variant": "ssmax", "initial_scale": 2.0}, "residual": {"variant": "ln_scaling", "branch_multiplier": 0.25},
     "activation": {"variant": "xielu"}, "latent": {"variant": "next_layer", "source_block": 6, "weight": 0.5},
     "optimizer": {"variant": "muon_plus", "plus_axis": "row"}, "gates": {"variant": "head_sigmoid"},
     "value_residual": {"variant": "first_block"}},
    {"residual": {"branch_ramp": 0.2, "branch_start": 0.1}, "optimizer": {"variant": "anvil", "adam_beta1": 0.7},
     "ema": {"variant": "latest_k", "window": 100}, "schedule": {"variant": "one_sqrt"}, "latent": {"variant": "neighbors"},
     "backprop_free": {"variant": "readout_boost", "readout_weight": 1.0, "switch": 0.1}, "stream": {"variant": "hyper"}},
    {"attention": {"variant": "paired", "key_offset": 0.5, "score_cap": 4.0}, "positional": {"periodic_scale": 1.0},
     "head": {"variant": "two_layer"}, "ema": {"variant": "two_decay", "start": 0.3}},
    {"attention": {"variant": "differential", "key_offset": 0.125, "score_cap": 8.0}, "positional": {"variant": "none", "periodic_scale": 0.25},
     "ema": {"variant": "drift_leap", "window": 100}},
    {"loss": {"variant": "entropy_weighted", "z_weight": 1e-3}, "neighbor": {"variant": "trigram_table"},
     "optimizer": {"adam_beta2_end": 0.999, "decay_timescale": 1.0, "fused_polar": 1, "cautious_weight_decay": 0.1},
     "schedule": {"momentum": 0.98, "momentum_cooldown_end": 0.97}, "stream": {"variant": "hyper"}},
    {"stream": {"variant": "attention_residual"}, "attention": {"variant": "triangle", "local_every": 2},
     "ema": {"variant": "plateau_restart", "start": 0.3, "window": 100}, "regularization": {"variant": "early_dropout"},
     "positional": {"length_bins": 8}, "kernel": {"variant": "cudagraphs"}},
    {"attention": {"variant": "paired", "local_every": 3}, "regularization": {"variant": "early_dropout", "rate": 0.2},
     "positional": {"variant": "none", "length_bins": 16}, "residual": {"variant": "ngpt"}},
    {"attention": {"variant": "differential", "kv_share": 2, "local_every": 3}, "value_residual": {"variant": "first_block"},
     "positional": {"terminus": 1, "length_bins": 8, "nope_every": 3}, "memory": {"variant": "product_keys", "block": 5},
     "grouping": {"variant": "physchem", "weight": 0.3}, "schedule": {"muon_late_scale": 2.0, "muon_late_start": 0.6}},
    {"attention": {"variant": "paired", "kv_share": 3}, "residual": {"variant": "ngpt"}, "value_embeddings": {"variant": "ends"},
     "memory": {"variant": "product_keys", "block": 1}},
    {"sam": {"variant": "late", "start": 0.0, "every": 1}, "optimizer": {"variant": "adamuon", "depth_taper": 0.5},
     "residual": {"learned_gain": 0.3, "branch_ramp": 0.1, "branch_start": 0.2}, "attention": {"kv_share": 2},
     "grouping": {"variant": "physchem"}},
    {"sam": {"variant": "late", "start": 0.0, "every": 2, "rho": 0.2}, "optimizer": {"variant": "amuse", "depth_taper": -0.5},
     "residual": {"variant": "ln_scaling", "learned_gain": 1.0, "branch_multiplier": 0.5}, "ema": {"variant": "tail", "start": 0.3}},
    {"memory": {"variant": "shared_pool"}, "deep_supervision": {"variant": "mid_heads", "weight": 0.3},
     "ema": {"variant": "latest_k", "window": 100}, "stream": {"variant": "attention_residual"}},
    {"attention": {"heads": 4, "qk_share": 0.75, "exclusive": 1, "key_offset": 0.25},
     "value_residual": {"variant": "first_block", "anchor_norm": 1, "per_head": 1, "qk_anchor": 0.7},
     "optimizer": {"orthogonalizer": "mud1", "radial_out": 0.5, "qk_lr_scale": 2.0, "depth_taper": 0.5}, "head": {"bias": 1},
     "loss": {"substitution_share": 0.1, "z_weight": 1e-3}, "lookahead": {"variant": "periodic", "every": 2},
     "canon": {"variant": "neighbors", "reach": 2}, "protein_context": {"variant": "both", "every": 12},
     "schedule": {"cooldown_power": 1.5}},
    {"attention": {"variant": "triangle", "heads": 8, "qk_share": 0.5, "tie_kv": 1, "kv_share": 2}, "residual": {"variant": "momentum"},
     "optimizer": {"variant": "muown", "polar_steps": 3}, "value_residual": {"variant": "normalized", "per_head": 1},
     "lookahead": {"variant": "periodic", "every": 2}},
    {"attention": {"variant": "paired", "qk_share": 0.75, "exclusive": 1}, "residual": {"variant": "momentum", "velocity_norm": 0},
     "optimizer": {"variance_scale": 1, "mona_beta": 0.99, "contra_start": 0.5, "contra_end": -0.25, "polar_steps": 3, "radial_out": 0.0,
                   "radial_in": 0.0, "nesterov": 0, "normuon_beta": 0.9}, "canon": {"variant": "neighbors", "placement": "c"}},
    {"attention": {"variant": "differential", "heads": 6, "tie_kv": 1, "exclusive": 1}, "head": {"variant": "two_layer", "bias": 1},
     "optimizer": {"orthogonalizer": "mud2", "variant": "muon_plus", "contra_start": 1.0, "vo_lr_scale": 0.5},
     "value_residual": {"variant": "first_block", "qk_anchor": 0.5}},
    {"loop": {"variant": "tied", "sample": 1, "readout_weight": 0.3, "inject": "learned", "residual_lambda": 0.5},
     "routing": {"variant": "per_pass"}, "attention": {"heads": 6}, "value_residual": {"variant": "first_block"},
     "residual": {"variant": "ln_scaling"}, "backprop_free": {"variant": "readout_warmstart"}, "head": {"variant": "two_layer"}},
    {"loop": {"variant": "tied", "prelude": 2, "core": 8, "passes": 4, "grad_passes": 2, "readout_weight": 0.1},
     "routing": {"variant": "shared", "experts": 16, "top_k": 4}, "backprop_free": {"variant": "unit_route"},
     "attention": {"variant": "differential"}},
    {"loop": {"variant": "untied", "sample": 1, "inject": "none", "exit": "confidence", "eval_passes": 2},
     "canon": {"variant": "neighbors"}, "unet": {"variant": "mirrored"}},
    {"sglr": {"variant": "pool", "steps": 12, "reach": 1}, "attention": {"heads": 6}, "value_embeddings": {"variant": "ends"},
     "latent": {"variant": "neighbors"}, "routing": {"variant": "per_pass", "experts": 4, "top_k": 1}},
]


def one_cpu_step(genome: dict[str, dict[str, Any]]) -> None:
    """Build, read the architecture receipt, then one training step with every hook and callback, and an evaluation forward."""
    train = load_train()
    torch.manual_seed(0)
    model = train.build_candidate(genome, max_tokens=64)
    benchmark.check_architecture(model)
    strategies: Strategies = model.strategies
    input_ids = torch.randint(4, 24, (2, 32))  # (b, l)
    input_ids[:, 0] = benchmark.CLS_ID
    input_ids[:, 16] = benchmark.CLS_ID  # two proteins a window
    corrupted, labels = benchmark.corrupt(input_ids, torch.Generator().manual_seed(0))  # each (b, l)
    optimizers = strategies.optimizer.build_optimizers(model, strategies.schedule.config.momentum)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    def forward_backward() -> None:
        logits, auxiliary = model(corrupted, labels=labels, dense_mask=document_dense_mask(corrupted))  # (b, l, v), ()
        loss = benchmark.masked_cross_entropy(logits, labels)  # ()
        for hook in strategies.hooked("training_loss"):
            loss = hook.training_loss(logits, labels, loss)  # ()
        loss = loss + auxiliary  # ()
        assert torch.isfinite(loss)
        loss.backward()

    def recompute_gradient() -> None:
        model.zero_grad(set_to_none=True)
        forward_backward()

    def loss_without_grad() -> torch.Tensor:
        with torch.no_grad():
            return benchmark.masked_cross_entropy(model(corrupted, dense_mask=document_dense_mask(corrupted)), labels)  # ()

    forward_backward()
    state = StepState(model, 0, 0.5, 1.0, corrupted, labels, loss_without_grad, recompute_gradient)
    for hook in strategies.hooked("after_backward"):
        hook.after_backward(state)
    for optimizer in optimizers:
        optimizer.step()
    for hook in strategies.hooked("after_step"):
        hook.after_step(state)
    changed = [name for name, parameter in model.named_parameters() if not torch.equal(before[name], parameter.detach())]
    assert any(name.startswith("blocks.") for name in changed) and "embedding.weight" in changed
    model.eval()
    padded = input_ids.clone()  # (b, l)
    padded[1, 20:] = benchmark.PAD_ID
    with torch.no_grad(), contextlib.ExitStack() as weights:
        for hook in strategies.hooked("evaluation_weights"):
            weights.enter_context(hook.evaluation_weights(model))
        evaluated = model(padded, padded != benchmark.PAD_ID)  # (b, l, v)
        states = model.hidden_states(padded, padded != benchmark.PAD_ID)
    assert evaluated.shape == (2, 32, benchmark.VOCAB_SIZE) and torch.isfinite(evaluated).all()
    assert len(states) == strategies.shape.config.blocks + 1


@pytest.mark.parametrize(("family", "variant"), VARIANTS, ids=[f"{family}-{variant}" for family, variant in VARIANTS])
def test_every_variant_builds_and_trains_a_cpu_step(family: str, variant: str) -> None:
    one_cpu_step(with_changes({family: {"variant": variant}}))


@pytest.mark.parametrize("changes", COMPOSED, ids=["triangle_differential-ngpt-hyper-and-more", "paired-dense-readout",
                                                   "differential6-dense-units", "triangle-partial-rope-and-2026-10-30",
                                                   "boost-ramp-anvil-lawa-neighbors", "paired-offset-cap-periodic-head-two-decay",
                                                   "differential-offset-cap-periodic-alone-leap",
                                                   "losses-trigram-timescale-beta2-momentum-hyper",
                                                   "attention-residual-local-triangle-restart-dropout-length",
                                                   "paired-local-dropout-length-ngpt",
                                                   "differential-kv2-local-terminus-memory-grouping-muon-late",
                                                   "paired-kv3-ngpt-value-tables-memory",
                                                   "sam-adamuon-taper-gains-kv2-grouping",
                                                   "sam-amuse-reverse-taper-ln-scaling-gains-tail",
                                                   "shared-pool-mid-heads-lawa-attention-residual",
                                                   "round9-narrow-qk-value-forms-mud-brake-canon-context",
                                                   "round9-triangle-tied-kv-momentum-muown",
                                                   "round9-paired-momentum-variance-mona-contra-hyperball",
                                                   "round9-differential-tied-exclusive-mud2-contra",
                                                   "loop-sampled-learned-scaled-readouts-routed-record-pieces",
                                                   "loop-recamplify-layout-truncated-shared-routers-unit-route",
                                                   "loop-untied-sampled-confidence-dial-canon-unet",
                                                   "sglr-pool-record-pieces-routed"])
def test_composed_genomes_train_a_cpu_step(changes: dict[str, dict[str, Any]]) -> None:
    one_cpu_step(with_changes(changes))


def test_default_genome_has_only_the_softcap_hook() -> None:
    strategies = Strategies(default_genome())
    hooked = {hook: [strategy.family for strategy in strategies.hooked(hook)] for hook in HOOKS}
    assert hooked == {hook: ["softcap"] if hook == "logits" else [] for hook in HOOKS}
    assert set(HOOKS) >= {"embed", "block_input", "normalized_input", "keys", "queries", "values", "attention_output",
                          "mlp_hidden", "sublayer_output", "updated_stream", "head_input", "logits", "auxiliary_loss",
                          "training_loss", "after_backward", "after_step", "evaluation_weights"}
    assert all(strategy.evaluation_options == 1 for strategy in strategies)


def test_rope_at_full_fraction_is_the_record_table_and_partial_rope_leaves_pairs_unrotated() -> None:
    full, partial = Rotary(64, 10_000.0, 16), Rotary(64, 10_000.0, 16, fraction=0.25)
    record_frequency = 10_000.0 ** (-torch.arange(0, 64, 2).float() / 64)  # (32,), the record's table
    assert torch.equal(full.cos, torch.outer(torch.arange(16).float(), record_frequency).cos())
    X = torch.randn(2, 16, 3, 64)  # (b, l, h, d_h)
    rotated = partial(X)  # (b, l, h, d_h)
    kept = torch.cat((torch.arange(8, 32), torch.arange(40, 64)))  # the 24 pairs after the first 8, both halves
    assert torch.equal(rotated[..., kept], X[..., kept]) and not torch.equal(rotated, X)


def stream_of(lengths: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """A stream of proteins with the given residue counts: CLS, residues 4 to 23, EOS."""
    offsets = np.concatenate(([0], np.cumsum(np.asarray(lengths) + 2))).astype(np.int64)  # (n + 1,)
    tokens = np.empty(int(offsets[-1]), dtype=np.uint8)  # (t,)
    for index, length in enumerate(lengths):
        start = int(offsets[index])
        tokens[start] = benchmark.CLS_ID
        tokens[start + 1 : start + 1 + length] = 4 + (np.arange(length) + index) % 20
        tokens[start + 1 + length] = benchmark.EOS_ID
    return tokens, offsets  # (t,), (n + 1,)


def test_whole_packing_keeps_proteins_whole_and_reads_every_token_once() -> None:
    lengths = [int(length) for length in np.random.default_rng(0).integers(10, 150, 300)] + [130, 200, 62]
    tokens, offsets = stream_of(lengths)
    window = 64
    packing = best_fit(offsets, np.arange(len(lengths)), window, np.random.default_rng(1))
    assert sorted(packing.sources.tolist()) == sorted(start for protein, size in enumerate(np.diff(offsets).tolist())
                                                      for start in range(int(offsets[protein]), int(offsets[protein]) + size, window))
    bounds = packing.window_starts
    for first, last in zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True):
        ends = packing.placements[first:last] + packing.lengths[first:last]  # (m,)
        assert int(ends.max()) <= window and np.array_equal(packing.placements[first + 1 : last], ends[:-1])
        cut = [tokens[source] != benchmark.CLS_ID for source in packing.sources[first:last].tolist()]
        assert not any(cut[1:]), "a piece without its CLS opens its window"
    steps = Batch(BatchConfig(variant="whole", window=window, windows_per_step=4)).steps(tokens, offsets, None, packing, 0)
    chunk = next(steps)  # (s,)
    whole = chunk.reshape(4, window)  # (b, l)
    real = whole != benchmark.PAD_ID  # (b, l)
    assert real.sum() == packing.lengths[: int(bounds[4])].sum() and (documents(torch.from_numpy(whole)) == -1).sum() == (~real).sum()
    # With windows several proteins long, as 1,024 is on the benchmark, best fit leaves little padding.
    assert best_fit(offsets, np.arange(len(lengths)), 1024, np.random.default_rng(1)).padding_share < 0.05


def test_tempered_orders_are_permutations_and_the_ordered_reader_follows_them(tmp_path: Path) -> None:
    lengths = [int(length) for length in np.random.default_rng(0).integers(10, 150, 200)]
    tokens, offsets = stream_of(lengths)
    (tmp_path / "train_families.json").write_text(json.dumps([f"f{index % 7 if index % 2 else 0}" for index in range(200)]))
    for changes in ({"variant": "family_tempered", "alpha": 1.0}, {"variant": "length_tempered", "beta": 0.5}):
        order = Curriculum(CurriculumConfig(**changes)).protein_order(offsets, tmp_path, seed=3)
        assert order is not None and sorted(order.tolist()) == list(range(200))
        steps = Batch(BatchConfig(window=32, windows_per_step=2)).steps(tokens, offsets, order, None, 0)
        first_protein = tokens[int(offsets[order[0]]) : int(offsets[order[0] + 1])]
        assert np.array_equal(next(steps)[: len(first_protein)], first_protein[:64])
    assert Curriculum(CurriculumConfig()).protein_order(offsets, tmp_path, seed=3) is None


def test_schedules_cool_down_linearly_or_by_one_minus_sqrt_and_amuse_turns_the_cooldown_off() -> None:
    linear, one_sqrt = Schedule(ScheduleConfig()), Schedule(ScheduleConfig(variant="one_sqrt"))
    for progress in (0.0, 0.01, 0.3, 0.7, 0.9, 1.0):
        assert linear.lr_scale(progress) == one_sqrt.lr_scale(progress) or progress > 0.6
    assert linear.lr_scale(0.8) == pytest.approx(0.5) and one_sqrt.lr_scale(0.8) == pytest.approx(1 - 0.5**0.5)
    assert one_sqrt.lr_scale(0.9, cooldown=False) == 1.0 and linear.lr_scale(0.01, cooldown=False) == linear.lr_scale(0.01)
    assert OptimizerChoice(OptimizerConfig(variant="amuse")).schedule_free and not OptimizerChoice(OptimizerConfig()).schedule_free
    assert amuse_beta(1, 0.6, 0.8) == 0.6 and amuse_beta(40, 0.6, 0.8) == pytest.approx(0.6) and 0.6 < amuse_beta(400, 0.6, 0.8) < 1


def test_momentum_tapers_over_the_cooldown_and_the_record_momentum_is_unchanged() -> None:
    record, tapered = Schedule(ScheduleConfig()), Schedule(ScheduleConfig(momentum=0.98, momentum_cooldown_end=0.95))
    for progress in (0.0, 0.05, 0.15, 0.5, 0.6, 0.8, 1.0):
        phase = min(1.0, progress / 0.15)
        assert record.momentum(progress) == 0.85 + phase * (0.95 - 0.85)  # the record's formula, exactly
    assert tapered.momentum(0.6) == pytest.approx(0.98) and tapered.momentum(0.8) == pytest.approx(0.965)
    assert tapered.momentum(1.0) == pytest.approx(0.95) and tapered.momentum(0.075) == pytest.approx(0.915)


def test_beta2_rises_over_the_cooldown_and_the_timescale_decay_replaces_the_cautious_decay() -> None:
    torch.manual_seed(0)
    model = load_train().build_candidate(with_changes({"optimizer": {"adam_beta2_end": 0.999, "decay_timescale": 2.0}}), max_tokens=64)
    choice: OptimizerChoice = model.strategies.optimizer
    assert [strategy.family for strategy in model.strategies.hooked("after_backward")] == ["optimizer"]
    choice.build_optimizers(model, 0.95)
    state = lambda step, progress: StepState(model, step, progress, 1.0, torch.zeros(1), torch.zeros(1), lambda: torch.zeros(()))  # noqa: E731
    choice.after_backward(state(99, 0.8))  # half the cooldown spent, 100 steps in 80 percent of the clock
    assert all(group["betas"] == pytest.approx((0.9, 0.9745)) for group in choice.adam.param_groups)
    assert choice.muon.decay_per_step == pytest.approx(1 / (2.0 * 125))
    choice.after_backward(state(0, 0.0))  # the compile warmup's steps: no estimate, no decay
    assert choice.muon.decay_per_step == 0.0
    # One bank stepped twice from the same weights and gradient: the timescale's shrink in place of the cautious decay.
    weights = [torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2)]
    gradients = [torch.randn(4, 4) for _ in weights]
    stepped = []
    for decay_per_step, cautious in ((0.0, 0.0), (0.01, 0.1)):
        members = [torch.nn.Parameter(weight.detach().clone()) for weight in weights]
        for member, gradient in zip(members, gradients, strict=True):
            member.grad = gradient.clone()
        muon = Muon(members, lr=0.05, momentum=0.95, weight_decay=0.0, normalize_rows=False, cautious_weight_decay=cautious)
        muon.decay_per_step = decay_per_step
        muon.step()
        stepped.append(torch.stack([member.detach() for member in members]))
    assert torch.allclose(stepped[1] - stepped[0], -0.01 * torch.stack([weight.detach() for weight in weights]), atol=1e-6)
    assert Muon(weights, lr=0.05, momentum=0.95, weight_decay=0.0, normalize_rows=False, fused_polar=True)._polar(torch.device("cpu")) is polar_express


def test_entropy_weights_average_one_and_the_z_loss_reads_only_the_normalizer() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 8, benchmark.VOCAB_SIZE)  # (b, l, v)
    labels = torch.full((2, 8), benchmark.IGNORE_LABEL)  # (b, l)
    labels[0, 2], labels[1, 5], labels[1, 6] = 3, 11, 19
    plain = benchmark.masked_cross_entropy(logits, labels)  # ()
    weighted, z_only = Loss(LossConfig(variant="entropy_weighted")), Loss(LossConfig(z_weight=1e-3))
    assert not Loss(LossConfig()).overrides("training_loss") and weighted.overrides("training_loss") and z_only.overrides("training_loss")
    flat = logits.clone()  # (b, l, v)
    flat[..., benchmark.ESM2_RESIDUE_COLUMNS] = 0.0
    flat[1, 6, benchmark.ESM2_RESIDUE_COLUMNS] = 5.0  # equal residue logits: equal entropies, so every weight is 1
    assert torch.allclose(weighted.training_loss(flat, labels, plain), benchmark.masked_cross_entropy(flat, labels))
    Z, targets = residue_scores(logits, labels)  # (3, c), (3,)
    entropy = -(Z.softmax(-1) * Z.log_softmax(-1)).sum(-1)  # (3,)
    per_residue = -Z.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)  # (3,), no logit leads by 16 here
    assert torch.allclose(weighted.training_loss(logits, labels, plain), (entropy / entropy.mean() * per_residue).sum(), atol=1e-5)
    log_normalizer = math.log(20.0)
    assert torch.allclose(z_only.training_loss(flat, labels, plain) - plain, torch.tensor(1e-3 * (2 * log_normalizer**2 + (5 + log_normalizer) ** 2)))


def test_context_tables_enter_after_the_last_block_from_a_silent_start() -> None:
    input_ids = torch.tensor([[benchmark.CLS_ID, 5, benchmark.MASK_ID, 7, benchmark.EOS_ID]])  # (1, 5)
    assert pair_index(input_ids)[0, 2].item() == 5 * benchmark.VOCAB_SIZE + 7
    code = ((benchmark.CLS_ID * 33 + 5) * 33 + 7) * 33 + benchmark.EOS_ID
    assert trigram_index(input_ids)[0, 2].item() == ((code * 2654435761) % 2**32) // 2**16
    every_code = torch.arange(33**4)  # (33^4,), every (left2, left, right, right2)
    rows = ((every_code * 2654435761) & 0xFFFFFFFF) >> 16  # (33^4,)
    assert rows.min() >= 0 and rows.max() < 2**16 and rows.unique().numel() == 2**16  # every row is used
    torch.manual_seed(0)
    model = load_train().build_candidate(with_changes({"neighbor": {"variant": "bigram_table"}}), max_tokens=64)
    neighbor = model.strategies.neighbor
    assert [strategy.family for strategy in model.strategies.hooked("updated_stream")] == ["neighbor"]
    assert not model.strategies.hooked("embed") and torch.sigmoid(neighbor.gated_table.gate).item() == 0.5
    context = ForwardContext(input_ids, None, None)
    X = torch.randn(1, 5, RECORD.hidden)  # (1, 5, d)
    assert torch.equal(neighbor.updated_stream(RECORD.blocks - 1, "mlp", X, context), X)  # the table starts at zero
    torch.nn.init.ones_(neighbor.table.weight)
    assert torch.equal(neighbor.updated_stream(RECORD.blocks - 2, "mlp", X, context), X)
    assert torch.equal(neighbor.updated_stream(RECORD.blocks - 1, "attention", X, context), X)
    assert torch.allclose(neighbor.updated_stream(RECORD.blocks - 1, "mlp", X, context), X + 0.5)


def test_attention_residuals_start_as_the_plain_sum_and_the_record_forward() -> None:
    train = load_train()
    input_ids = torch.randint(4, 24, (2, 32))  # (b, l)
    input_ids[:, 0] = benchmark.CLS_ID
    mask = document_dense_mask(input_ids)  # (b, 1, l, l)
    outputs = []
    for genome in (default_genome(), with_changes({"stream": {"variant": "attention_residual"}})):
        torch.manual_seed(0)
        model = train.build_candidate(genome, max_tokens=64)
        with torch.no_grad():
            outputs.append(model(input_ids, dense_mask=mask))  # (b, l, v)
    assert torch.equal(outputs[0], outputs[1])  # the logits draw nothing and the updates start at zero
    mixer = model.strategies.stream
    values = (torch.randn(2, 3, 8), torch.randn(2, 3, 8).bfloat16(), torch.randn(2, 3, 8).bfloat16())  # each (b, l, d)
    assert torch.allclose(mixer._attention_residual(values, 2), values[0] + values[1].float() + values[2].float())
    with torch.no_grad():
        mixer.holder.logits[2, :3] = torch.tensor([0.0, math.log(2.0), 0.0])
    weights = 3 * torch.tensor([0.25, 0.5, 0.25])  # (3,), n softmax
    expected = sum(weight * value.float() for weight, value in zip(weights, values, strict=True))  # (b, l, d)
    assert torch.allclose(mixer._attention_residual(values, 2), expected, atol=1e-6)


def test_local_blocks_see_only_their_reach_and_the_global_blocks_everything() -> None:
    torch.manual_seed(0)
    model = load_train().build_candidate(with_changes({"attention": {"local_every": 3}}), max_tokens=1024)
    assert [block.attention.local for block in model.blocks[:4]] == [False, True, True, False]
    input_ids = torch.randint(4, 24, (1, 600))  # (1, l), one protein longer than the reach
    input_ids[0, 0] = benchmark.CLS_ID
    context = ForwardContext(input_ids, None, document_dense_mask(input_ids))
    local, wide = model.blocks[1].attention._dense_mask(context, 600)[0, 0], model.blocks[0].attention._dense_mask(context, 600)[0, 0]  # (l, l) each
    assert wide.all() and local[300, 45] and not local[300, 44] and local[300, 555] and not local[300, 556]
    rule = within_reach(lambda batch, head, query, key: torch.tensor(True), paired=True)  # noqa: ARG005
    zero = torch.tensor(0)
    assert rule(zero, zero, torch.tensor(600), torch.tensor(2 * 45)) and not rule(zero, zero, torch.tensor(600), torch.tensor(2 * 44))
    assert Kernel(KernelConfig(variant="cudagraphs")).compile_options() == {"mode": "reduce-overhead"}
    assert Kernel(KernelConfig()).compile_options() == {} and not Kernel(KernelConfig()).marks_steps


def test_plateau_restart_continues_from_the_snapshot_mean_once() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    state = lambda step, progress: StepState(model, step, progress, 1.0, torch.zeros(1), torch.zeros(1), lambda: torch.zeros(()))  # noqa: E731
    restart = Ema(EmaConfig(variant="plateau_restart", start=0.5, window=800))  # a snapshot every 100 steps
    assert restart.overrides("after_step") and not restart.overrides("evaluation_weights")
    for step in range(0, 301, 50):
        model.weight.data.fill_(float(step))
        restart.after_step(state(step, 0.1))
    assert model.weight.item() == 300.0 and len(restart.snapshots) == 4  # steps 0, 100, 200 and 300
    model.weight.data.fill_(999.0)
    restart.after_step(state(301, 0.5))
    assert model.weight.item() == 150.0 and restart.restarted and not restart.snapshots
    model.weight.data.fill_(7.0)
    restart.after_step(state(302, 0.9))
    assert model.weight.item() == 7.0


def test_early_dropout_falls_to_the_record_arithmetic() -> None:
    torch.manual_seed(0)
    model = load_train().build_candidate(with_changes({"regularization": {"variant": "early_dropout", "rate": 0.2, "until": 0.1}}),
                                         max_tokens=64)
    dropout = model.strategies.regularization
    assert dropout.config.rate_at(0.05) == pytest.approx(0.1) and dropout.config.rate_at(0.2) == 0.0
    assert dropout.holder.rate.item() == pytest.approx(0.2)
    Y = torch.randn(4, 64, RECORD.hidden)  # (b, l, d)
    dropped = dropout.sublayer_output(0, "mlp", Y)  # (b, l, d)
    kept = dropped != 0  # (b, l, d)
    assert 0.75 < kept.float().mean() < 0.85 and torch.allclose(dropped[kept], Y[kept] / 0.8)
    dropout.after_step(StepState(model, 9, 0.2, 1.0, torch.zeros(1), torch.zeros(1), lambda: torch.zeros(())))
    assert dropout.holder.rate.item() == 0.0 and torch.equal(dropout.sublayer_output(0, "mlp", Y), Y)
    dropout.holder.rate.fill_(0.5)
    model.eval()
    assert torch.equal(dropout.sublayer_output(0, "mlp", Y), Y)


def test_length_bins_count_each_proteins_tokens_without_padding() -> None:
    input_ids = torch.full((1, 400), 5)  # (1, l)
    input_ids[0, 0], input_ids[0, 40] = benchmark.CLS_ID, benchmark.CLS_ID  # a 40-token protein, then 360 tokens
    input_ids[0, 300:] = benchmark.PAD_ID  # the second protein's 260 tokens, then padding
    bins = length_bins(input_ids, 16)[0]  # (l,)
    assert bins[:40].unique().tolist() == [math.floor((math.log2(40) - 5) * 16 / 6)]
    assert bins[40:300].unique().tolist() == [math.floor((math.log2(260) - 5) * 16 / 6)]
    assert length_bins(torch.full((1, 4), benchmark.CLS_ID), 8).tolist() == [[0, 0, 0, 0]]
    torch.manual_seed(0)
    model = load_train().build_candidate(with_changes({"positional": {"variant": "none", "length_bins": 16}}), max_tokens=64)
    assert [strategy.family for strategy in model.strategies.hooked("embed")] == ["positional"]
    assert not model.extensions["positional"].length_table.weight.any()


def test_product_key_memory_reads_the_exact_top_rows_and_starts_silent() -> None:
    train = load_train()
    input_ids = torch.randint(4, 24, (2, 16))  # (b, l)
    input_ids[:, 0] = benchmark.CLS_ID
    mask = document_dense_mask(input_ids)  # (b, 1, l, l)
    outputs = []
    for genome in (default_genome(), with_changes({"memory": {"variant": "product_keys", "block": 3}})):
        torch.manual_seed(0)
        model = train.build_candidate(genome, max_tokens=64)
        with torch.no_grad():
            outputs.append(model(input_ids, dense_mask=mask))  # (b, l, v)
    assert torch.equal(outputs[0], outputs[1])  # built empty, drawn after the record, values at zero
    assert benchmark.check_architecture(model)["blocks"] == 36
    memory = model.strategies.memory.memory
    with torch.no_grad():
        memory.values.normal_()
        X = torch.randn(1, 3, RECORD.hidden)  # (1, 3, d)
        read = memory(X)[0]  # (3, d)
        Q = (X[0] @ memory.query.T).view(3, 2, -1)  # (3, 2, d / 2)
        half = torch.einsum("nhe,hse->nhs", Q, memory.keys)  # (3, 2, s)
        every = (half[:, 0, :, None] + half[:, 1, None, :]).flatten(1)  # (3, s^2), every row's score
        best, rows = every.topk(memory_module.TOP, dim=-1)  # (3, t) each
        expected = (torch.softmax(best, -1)[..., None] * memory.values[rows]).sum(1)  # (3, d)
    assert torch.allclose(read, expected, atol=1e-4)


def test_borrowing_blocks_reuse_the_lenders_keys_and_values() -> None:
    torch.manual_seed(0)
    model = load_train().build_candidate(with_changes({"attention": {"kv_share": 2}}), max_tokens=64)
    record = load_train().build_candidate(default_genome(), max_tokens=64)
    lender, borrower = model.blocks[0].attention, model.blocks[1].attention
    assert lender.lends_kv and borrower.borrows_kv and borrower.key is None and borrower.value is None
    assert benchmark.check_architecture(record)["parameters"] - benchmark.check_architecture(model)["parameters"] == 18 * 2 * 768**2
    with torch.no_grad():
        borrower.query.weight.copy_(lender.query.weight)
        lender.output.weight.normal_(std=0.02)
        borrower.output.weight.copy_(lender.output.weight)
    input_ids = torch.randint(4, 24, (2, 16))  # (b, l)
    input_ids[:, 0] = benchmark.CLS_ID
    context = ForwardContext(input_ids, None, document_dense_mask(input_ids))
    X = torch.randn(2, 16, RECORD.hidden)  # (b, l, d)
    with torch.no_grad():
        lent = lender(X, context)  # (b, l, d), stores its K and V
        assert set(context.stored) >= {"shared_keys", "shared_values"}
        assert torch.equal(borrower(X, context), lent)  # the same queries read the lender's K and V


def test_group_loss_terminus_vectors_and_muon_widening() -> None:
    assert [GROUPS[group] for group in GROUP_OF_CLASS.tolist()][STANDARD_RESIDUES.index("K")] == "KRH"
    assert sorted("".join(GROUPS)) == sorted(STANDARD_RESIDUES)
    labels = torch.tensor([[benchmark.IGNORE_LABEL, 0, 8, benchmark.IGNORE_LABEL]])  # (1, 4): A and K
    assert group_labels(labels).tolist() == [[benchmark.IGNORE_LABEL, 0, 2, benchmark.IGNORE_LABEL]]
    torch.manual_seed(0)
    model = load_train().build_candidate(with_changes({"grouping": {"variant": "physchem", "weight": 0.2, "until": 0.5}}), max_tokens=64)
    grouping = model.strategies.grouping
    states = [torch.randn(1, 4, RECORD.hidden)] * (RECORD.blocks + 1)  # (1, 4, d) each
    context = ForwardContext(torch.zeros(1, 4, dtype=torch.long), None, None)
    assert torch.allclose(grouping.auxiliary_loss(model, states, labels, context), torch.tensor(0.2 * 2 * math.log(6.0)))
    grouping.after_step(StepState(model, 9, 0.25, 1.0, torch.zeros(1), torch.zeros(1), lambda: torch.zeros(())))
    assert grouping.holder.weight.item() == pytest.approx(0.1)
    input_ids = torch.tensor([[benchmark.CLS_ID, 5, 6, 7, benchmark.EOS_ID, benchmark.CLS_ID, 8, benchmark.EOS_ID, 9]])  # (1, 9)
    first, last = terminus_positions(input_ids)  # (1, 9) each
    assert first[0].nonzero().flatten().tolist() == [1, 6] and last[0].nonzero().flatten().tolist() == [3, 6]
    model = load_train().build_candidate(with_changes({"positional": {"terminus": 1}}), max_tokens=64)
    with torch.no_grad():
        model.extensions["positional"].terminus.copy_(torch.stack([torch.ones(RECORD.hidden), 2 * torch.ones(RECORD.hidden)]))
    added = model.strategies.positional.embed(model, input_ids, torch.zeros(1, 9, RECORD.hidden))[0, :, 0]  # (9,)
    assert added.tolist() == [0, 1, 0, 2, 0, 0, 3, 0, 0]
    record, widened = Schedule(ScheduleConfig()), Schedule(ScheduleConfig(muon_late_scale=2.0, muon_late_start=0.8))
    assert not record.widens_muon and widened.widens_muon
    assert [widened.muon_lr_factor(progress) for progress in (0.5, 0.8, 0.9, 1.0)] == pytest.approx([1.0, 1.0, 1.5, 2.0])


def test_sam_steps_with_the_gradient_at_the_perturbed_weights_and_puts_them_back() -> None:
    model = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0, 0.5]]))
    target = torch.tensor([[0.3, 0.1, -0.2]])  # (1, 3); loss = ||W - target||^2 / 2 * c, gradient c (W - target)

    def recompute_gradient() -> None:
        model.zero_grad(set_to_none=True)
        (2.0 * (model.weight - target).square().sum() / 2).backward()

    recompute_gradient()
    start = model.weight.detach().clone()
    gradient = model.weight.grad.clone()
    sam = Sam(SamConfig(variant="late", rho=0.1, every=2, start=0.5))
    state = lambda step, progress: StepState(model, step, progress, 1.0, torch.zeros(1), torch.zeros(1), lambda: torch.zeros(()),  # noqa: E731
                                             recompute_gradient)
    sam.after_backward(state(4, 0.4))  # before the start: the plain gradient stays
    assert torch.equal(model.weight.grad, gradient)
    sam.after_backward(state(5, 0.6))  # an odd step: off
    assert torch.equal(model.weight.grad, gradient)
    sam.after_backward(state(6, 0.6))
    perturbed = start + 0.1 * gradient / gradient.norm()  # (1, 3)
    assert torch.allclose(model.weight.grad, 2.0 * (perturbed - target)) and torch.allclose(model.weight, start)
    assert not Sam(SamConfig()).overrides("after_backward")


def test_adamuon_depth_taper_and_learned_gains() -> None:
    torch.manual_seed(0)
    weights = [torch.randn(4, 6) for _ in range(2)]
    gradients = [torch.randn(4, 6) for _ in weights]

    def one_step(**options: Any) -> tuple[torch.Tensor, Muon]:
        members = [torch.nn.Parameter(weight.clone()) for weight in weights]
        for member, gradient in zip(members, gradients, strict=True):
            member.grad = gradient.clone()
        scales = options.pop("scales", None)
        muon = Muon(members, lr=0.05, momentum=0.95, weight_decay=0.0,
                    member_scales=None if scales is None else {id(member): scale for member, scale in zip(members, scales, strict=True)},
                    **options)
        muon.step()
        return torch.stack([member.detach() for member in members]), muon  # (k, m, n), and the optimizer

    plain, _ = one_step(normalize_rows=False)
    adamuon, muon = one_step(normalize_rows=False, elementwise=True)
    assert muon.param_groups[0]["banks"][0][2].shape == (2, 4, 6)  # one second moment per entry
    start = torch.stack(weights)  # (2, 4, 6)
    assert torch.allclose((adamuon - start).flatten(1).norm(dim=1), (plain - start).flatten(1).norm(dim=1), atol=1e-5)
    assert not torch.allclose(adamuon, plain)
    with pytest.raises(AssertionError):
        one_step(normalize_rows=True, elementwise=True)
    tapered, _ = one_step(normalize_rows=False, scales=[1.25, 0.75])
    assert torch.allclose(tapered - start, (plain - start) * torch.tensor([1.25, 0.75]).view(-1, 1, 1), atol=1e-6)
    config = OptimizerConfig(depth_taper=0.5)
    assert (config.block_scale(0, RECORD.blocks), config.block_scale(35, RECORD.blocks)) == pytest.approx((1.25, 0.75))
    assert sum(config.block_scale(index, RECORD.blocks) for index in range(RECORD.blocks)) == pytest.approx(RECORD.blocks)
    assert (config.block_scale(0, 12), config.block_scale(11, 12)) == pytest.approx((1.25, 0.75))  # the taper spans any depth
    torch.manual_seed(0)
    model = load_train().build_candidate(with_changes({"residual": {"learned_gain": 0.3}}), max_tokens=64)
    residual = model.strategies.residual
    assert residual.overrides("sublayer_output") and not Residual(ResidualConfig()).overrides("sublayer_output")
    with torch.no_grad():
        residual.holder.learned_gains[2, 1] = 2.0
    Y = torch.randn(2, 4, RECORD.hidden)  # (b, l, d)
    assert torch.equal(residual.sublayer_output(2, "mlp", Y), Y * 2.0)
    assert torch.allclose(residual.sublayer_output(2, "attention", Y), Y * 0.3)
    with pytest.raises(ValueError):
        ResidualConfig(learned_gain=0.001)


def test_shared_pool_reads_and_intermediate_heads() -> None:
    torch.manual_seed(0)
    model = load_train().build_candidate(with_changes({"memory": {"variant": "shared_pool"},
                                                       "deep_supervision": {"variant": "mid_heads", "weight": 0.2, "until": 0.5}}),
                                         max_tokens=64)
    memory = model.strategies.memory
    pool = memory.memory
    assert pool.query.shape == (3, RECORD.hidden, RECORD.hidden) and pool.keys.shape[0] == 3
    X = torch.randn(1, 3, RECORD.hidden)  # (1, 3, d)
    with torch.no_grad():
        assert torch.equal(memory.block_input(model, 18, X, []), X)  # the pool starts at zero
        pool.values.normal_()
        assert torch.equal(memory.block_input(model, 10, X, []), X)
        read = memory.block_input(model, 27, X, [])[0] - X[0]  # (3, d), the third read
        Q = norm_rms(X[0]) @ pool.query[2].T  # (3, d)
        half = torch.einsum("nhe,hse->nhs", Q.view(3, 2, -1), pool.keys[2])  # (3, 2, s)
        every = (half[:, 0, :, None] + half[:, 1, None, :]).flatten(1)  # (3, s^2)
        best, rows = every.topk(memory_module.TOP, dim=-1)  # (3, t) each
        expected = (torch.softmax(best, -1)[..., None] * pool.values[rows]).sum(1)  # (3, d)
    assert torch.allclose(read, expected, atol=1e-4)
    deep = model.strategies.deep_supervision
    states = [torch.randn(1, 4, RECORD.hidden) for _ in range(RECORD.blocks + 1)]  # (1, 4, d) each
    labels = torch.tensor([[benchmark.IGNORE_LABEL, 3, 7, benchmark.IGNORE_LABEL]])  # (1, 4)
    context = ForwardContext(torch.zeros(1, 4, dtype=torch.long), None, None)
    with torch.no_grad():
        expected_loss = 0.2 * sum(benchmark.masked_cross_entropy(model.logits(states[block]), labels) for block in (19, 28))
        assert torch.allclose(deep.auxiliary_loss(model, states, labels, context), expected_loss)
    deep.after_step(StepState(model, 9, 0.25, 1.0, torch.zeros(1), torch.zeros(1), lambda: torch.zeros(())))
    assert deep.holder.weight.item() == pytest.approx(0.1)


def test_step_units_scale_with_the_run() -> None:
    model = torch.nn.Linear(1, 1, bias=False)

    def state(step: int, progress: float, run_scale: float) -> StepState:
        return StepState(model, step, progress, 1.0, torch.zeros(1), torch.zeros(1), lambda: torch.zeros(()), run_scale=run_scale)

    for run_scale, kept in ((1.0, [0, 50, 100]), (2.0, [0, 100])):  # window 400 over 8 snapshots: every 50 steps a 600 s run
        lawa = Ema(EmaConfig(variant="latest_k", window=400))
        for step in range(0, 101, 10):
            model.weight.data.fill_(float(step))
            lawa.after_step(state(step, 0.5, run_scale))
        assert [snapshot[0].item() for snapshot in lawa.snapshots] == kept
    assert scaled_decay(0.99, 1.0) == 0.99 and (1 - scaled_decay(0.99, 6.0)) == pytest.approx(0.01 / 6, rel=0.01)
    torch.manual_seed(0)
    candidate = load_train().build_candidate(with_changes({"optimizer": {"decay_timescale": 1.0}}), max_tokens=64)
    choice = candidate.strategies.optimizer
    choice.build_optimizers(candidate, 0.95)
    for run_scale, steps in ((1.0, 2000), (6.0, 12000)):  # one per-step shrink for a run six times the steps at an hour
        choice.after_backward(StepState(candidate, steps // 2 - 1, 0.5, 1.0, torch.zeros(1), torch.zeros(1),
                                        lambda: torch.zeros(()), run_scale=run_scale))
        assert choice.muon.decay_per_step == pytest.approx(1 / 2000)


def test_neighbor_heads_score_only_visible_residues_at_masked_positions() -> None:
    torch.manual_seed(0)
    model = load_train().build_candidate(with_changes({"latent": {"variant": "neighbors"}}), max_tokens=64)
    latent = model.strategies.latent
    input_ids = torch.tensor([[benchmark.CLS_ID, 5, benchmark.MASK_ID, 7, benchmark.EOS_ID, benchmark.CLS_ID]])  # (1, 6)
    labels = torch.full((1, 6), benchmark.IGNORE_LABEL)  # (1, 6)
    labels[0, 2] = 0
    final = torch.randn(1, 6, RECORD.hidden)  # (1, 6, d)
    hidden = torch.nn.functional.rms_norm(final, (RECORD.hidden,))[0, 2]  # (d,), the masked position's state
    # At position 2: -2 is CLS, -1 is residue 5, +1 is residue 7, +2 is EOS, so only the -1 and +1 heads score.
    expected = sum(benchmark.masked_cross_entropy(latent.heads[head](hidden)[None], benchmark.residue_labels(torch.tensor([token])))
                   for head, token in ((1, 5), (2, 7)))  # ()
    assert torch.allclose(latent._neighbor_loss(final, labels, input_ids), expected)


def score_grid(score_mod: Any, heads: int, length: int) -> torch.Tensor:
    """A flex score function evaluated eagerly on every (head, query, key) at raw score 1."""
    head, query, key = torch.meshgrid(torch.arange(heads), torch.arange(length), torch.arange(length), indexing="ij")  # each (k, n, n)
    return score_mod(torch.ones(heads, length, length), torch.zeros_like(head), head, query, key)  # (k, n, n)


def test_periodic_bias_and_score_cap_leave_the_record_score_function_at_zero() -> None:
    slopes = torch.tensor([0.5, 0.25])  # (k,)
    record = Positional(PositionalConfig())
    record_mod, zero_mod = record.score_mod(slopes, 1, False), Positional(PositionalConfig(periodic_scale=0.0)).score_mod(slopes, 1, False, 0.0)
    assert record_mod.__code__ is zero_mod.__code__ and record_mod.__name__ == "local_bias"  # the record's function, so its cost
    distance = (torch.arange(6)[:, None] - torch.arange(6)[None, :]).abs()  # (n, n)
    expected = 1 - slopes[:, None, None] * distance + 0.5 * periodic_term(distance)  # (k, n, n)
    periodic = Positional(PositionalConfig(periodic_scale=0.5))
    assert torch.allclose(score_grid(periodic.score_mod(slopes, 1, False), 2, 6), expected)
    assert torch.allclose(periodic.dense_bias(slopes, 1, 6, False, torch.float32, torch.device("cpu"))[0], expected - 1)
    assert periodic_term(torch.tensor([0, 2])).tolist() == pytest.approx([1.0, (math.cos(4 * math.pi / 3.6) + 1) / 2])
    capped = score_grid(record.score_mod(slopes, 1, False, 4.0), 2, 6)  # (k, n, n)
    assert torch.allclose(capped, 4 * math.tanh(1 / 4) - slopes[:, None, None] * distance)  # the cap before the slopes


def test_capped_attention_is_attention_with_the_scaled_scores_capped() -> None:
    Q, K, V = torch.randn(2, 4, 5, 8), torch.randn(2, 2, 5, 8), torch.randn(2, 2, 5, 8)  # (b, q, n, e), (b, k, n, e) twice
    allowed = (torch.rand(2, 1, 5, 5) > 0.3) | torch.eye(5, dtype=torch.bool)  # (b, 1, n, n), every row keeps itself
    mask = torch.zeros(2, 1, 5, 5).masked_fill(~allowed, float("-inf"))  # (b, 1, n, n)
    reference = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask, enable_gqa=True)  # (b, q, n, e)
    assert torch.allclose(capped_attention(Q, K, V, mask, 1e6), reference, atol=1e-5)
    scores = 3 * torch.tanh((Q @ K.repeat_interleave(2, dim=1).transpose(-2, -1)) * 8**-0.5 / 3) + mask  # (b, q, n, n)
    assert torch.allclose(capped_attention(Q, K, V, mask, 3.0), torch.softmax(scores, -1) @ V.repeat_interleave(2, dim=1), atol=1e-5)


def test_key_offset_moves_whole_pairs_within_each_protein() -> None:
    input_ids = torch.tensor([[benchmark.CLS_ID, 5, 6, benchmark.EOS_ID, benchmark.CLS_ID, 7]])  # (1, 6), two proteins
    K = torch.arange(6.0)[None, :, None, None].expand(1, 6, 1, 8).clone()  # (1, 6, 1, 8): each key holds its position
    shifted = shift_key_pairs(K, input_ids, pairs=1)[0, :, 0]  # (6, 8); d_h 8, 4 pairs, pair i is channels i and i + 4
    # Pair 3 (channels 3, 7) from the previous token, pair 2 (channels 2, 6) from the next, within each protein.
    assert shifted[:, 3].tolist() == [0, 0, 1, 2, 4, 4] and torch.equal(shifted[:, 3], shifted[:, 7])
    assert shifted[:, 2].tolist() == [1, 2, 3, 3, 5, 5] and torch.equal(shifted[:, 2], shifted[:, 6])
    assert torch.equal(shifted[:, [0, 1, 4, 5]], K[0, :, 0][:, [0, 1, 4, 5]])
    assert Kernel(KernelConfig()).kernel_options() is None and Kernel(KernelConfig(variant="flash")).kernel_options() == {"BACKEND": "FLASH"}


def test_two_decay_scores_both_averages_and_the_drift_leap_moves_back() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    state = lambda step, progress: StepState(model, step, progress, 1.0, torch.zeros(1), torch.zeros(1), lambda: torch.zeros(()))  # noqa: E731
    two = Ema(EmaConfig(variant="two_decay", start=0.3, decay=0.95, decay_fast=0.9))
    assert two.evaluation_options == 1
    for step, value in enumerate((1.0, 3.0)):
        model.weight.data.fill_(value)
        two.after_step(state(step, 0.5))
    seen = []
    for option in range(two.evaluation_options):
        two.option = option
        with two.evaluation_weights(model):
            seen.append(model.weight.item())
    assert seen == pytest.approx([1.1, 1.2]) and model.weight.item() == 3.0  # 0.95 and 0.9 decays; live weights back
    leap = Ema(EmaConfig(variant="drift_leap", window=100, leap=0.2))
    for step in range(0, 101, 25):
        model.weight.data.fill_(float(step))
        leap.after_step(state(step, 0.5))
    with leap.evaluation_weights(model):
        assert model.weight.item() == pytest.approx(100 + 0.2 * (25 - 100))  # the oldest of 4 snapshots, 75 steps back


def test_genome_round_trip(tmp_path: Path) -> None:
    record = default_genome()
    assert genome_document(record) == {"schema": 1, "families": {}}
    changed = with_changes({"attention": {"variant": "triangle"}, "optimizer": {"muon_lr": 0.05}})
    save_genome(changed, tmp_path / "genome.json")
    assert load_genome(tmp_path / "genome.json") == changed
    assert json.loads((tmp_path / "genome.json").read_text())["families"] == {"attention": {"variant": "triangle"},
                                                                              "optimizer": {"muon_lr": 0.05}}
    assert genome_sha256(changed) == genome_sha256(load_genome(tmp_path / "genome.json")) != genome_sha256(record)
    for bad in ({"families": {"nonsense": {}}}, {"families": {"attention": {"variant": "nonsense"}}},
                {"families": {"attention": {"width": 3}}}, {"families": {"optimizer": {"muon_lr": 5.0}}}, {"genes": {}}):
        with pytest.raises(ValueError):
            resolve(bad)


def test_the_registry_types_every_family_attribute() -> None:
    """`Strategies` declares one attribute per registered family, with the family's strategy class."""
    assert typing.get_type_hints(Strategies) == dict(FAMILIES)


def test_integer_fields_refuse_other_numbers() -> None:
    """Each config's INTEGERS names an int field, and a float in one raises the message the genome's author reads."""
    for strategy in FAMILIES.values():
        config_type = strategy.Config
        field_types = {item.name: item.type for item in dataclasses.fields(config_type)}
        for name in config_type.INTEGERS:
            assert field_types[name] == "int", (config_type.__name__, name)
            value = float(getattr(config_type(), name))
            with pytest.raises(ValueError, match=f"^{name} must be an integer, not {value!r}$"):
                config_type(**{name: value})


def test_record_genomes_on_disk_resolve() -> None:
    for path in sorted((PROJECT / "genomes").glob("*.json")):
        load_genome(path)
    assert load_genome(PROJECT / "genomes" / "022_record.json") == default_genome()


def test_default_genome_reproduces_the_record_file_exactly(tiny_data: Path, tmp_path: Path) -> None:
    """train.py without a genome and the frozen 022 snapshot give the same metrics and weights on a CPU smoke."""
    arguments = ["--data-dir", str(tiny_data), "--name", "smoke", "--device", "cpu", "--no-compile", "--no-wandb",
                 "--window", "32", "--windows-per-step", "2", "--max-steps", "2", "--skip-final-evals"]
    outputs = {}
    for label, path in (("record", PROJECT / "candidates" / "022_embed_head_lr.py"), ("registry", PROJECT / "train.py")):
        outputs[label] = tmp_path / label
        completed = subprocess.run([sys.executable, "-c", DETERMINISTIC_RUN, str(path), *arguments, "--output-dir", str(outputs[label])],
                                   capture_output=True, text=True, cwd=PROJECT)
        assert completed.returncode == 0, completed.stderr[-3000:]
    record, registry = (json.loads((outputs[label] / "result.json").read_text()) for label in ("record", "registry"))
    assert [point["loss"] for point in record["curve"]] == [point["loss"] for point in registry["curve"]]
    assert record["architecture"] == registry["architecture"]
    record_weights, registry_weights = (torch.load(outputs[label] / "best.pt") for label in ("record", "registry"))
    assert sorted(record_weights) == sorted(registry_weights)
    assert all(torch.equal(record_weights[name], registry_weights[name]) for name in record_weights)


def test_a_genome_run_writes_its_genome(tiny_data: Path, tmp_path: Path) -> None:
    genome_path = tmp_path / "genome.json"
    save_genome(with_changes({"gates": {"variant": "residual_channels"}}), genome_path)
    command = [sys.executable, str(PROJECT / "train.py"), "--data-dir", str(tiny_data), "--output-dir", str(tmp_path / "run"),
               "--name", "genome", "--genome", str(genome_path), "--max-steps", "1", "--no-compile", "--no-wandb",
               "--budget", "180", "--device", "cpu", "--window", "32", "--windows-per-step", "2", "--final-splits", "val_select"]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT)
    assert completed.returncode == 0, completed.stderr[-3000:]
    run_result = json.loads((tmp_path / "run" / "result.json").read_text())
    assert run_result["genome"]["gates"]["variant"] == "residual_channels" and set(run_result["metrics"]) == {"val_select"}
    assert run_result["clock"]["budget_seconds"] == 180


def test_an_optimizer_step_can_sum_micro_batches(tiny_data: Path, tmp_path: Path) -> None:
    """Two optimizer steps read four one-window micro-batches on one shape, and the clock counts what was applied."""
    genome_path = tmp_path / "genome.json"
    save_genome(with_changes({"batch": {"micro_divisor": 2, "accumulate_early": 2, "accumulate_late": 2}}), genome_path)
    command = [sys.executable, str(PROJECT / "train.py"), "--data-dir", str(tiny_data), "--output-dir", str(tmp_path / "run"),
               "--name", "micro", "--genome", str(genome_path), "--max-steps", "2", "--no-compile", "--no-wandb",
               "--device", "cpu", "--window", "32", "--windows-per-step", "2", "--skip-final-evals"]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert json.loads((tmp_path / "run" / "result.json").read_text())["clock"]["steps"] == 2


def test_micro_batches_sum_to_the_gradient_of_the_whole_step() -> None:
    train = load_train()

    class Table(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.table = torch.nn.Embedding(benchmark.VOCAB_SIZE, benchmark.VOCAB_SIZE)

        def forward(self, tokens: torch.Tensor, block_masks: tuple[()]) -> torch.Tensor:
            # tokens: (b, l); returns (b, l, v).
            return self.table(tokens)  # (b, l, v)

    torch.manual_seed(0)
    tokens = torch.randint(4, 24, (4, 8))  # (b, l)
    labels = torch.full_like(tokens, benchmark.IGNORE_LABEL)  # (b, l)
    labels[:, ::2] = benchmark.residue_labels(tokens)[:, ::2]  # each window holds four, so the micro-batches weigh the same

    def gradient_of(parts: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        # parts: (b, l) pairs of tokens and labels; returns (v, v).
        torch.manual_seed(1)
        model = Table()
        for part_tokens, part_labels in parts:
            train.train_step(model, part_tokens, part_labels, (), None, torch.device("cpu"), False, [], len(parts))
        return model.table.weight.grad  # (v, v)

    whole, halves = gradient_of([(tokens, labels)]), gradient_of([(tokens[:2], labels[:2]), (tokens[2:], labels[2:])])  # (v, v) each
    assert whole.abs().sum() > 0 and torch.allclose(halves, whole)


def test_the_batch_schedule_switches_on_the_clock_and_the_record_never_accumulates() -> None:
    ramp = Strategies(with_changes({"batch": {"micro_divisor": 2, "accumulate_late": 2, "accumulate_switch": 0.25}})).batch
    assert ramp.step_tokens == 32_768 and [ramp.accumulation(progress) for progress in (0.0, 0.24, 0.25, 1.0)] == [1, 1, 2, 2]
    record = Strategies(default_genome()).batch
    assert record.step_tokens == 65_536 and {record.accumulation(progress) for progress in (0.0, 0.5, 1.0)} == {1}
    with pytest.raises(ValueError, match="micro-batches"):
        BatchConfig(windows_per_step=2, micro_divisor=4)


# Round 9: each new number at its own arithmetic, on small tensors; the composed genomes above train them together.

def test_mud_decorrelates_rows_keeps_unit_rows_and_leaves_a_zero_matrix_alone() -> None:
    torch.manual_seed(0)
    G = torch.randn(3, 8, 32) + 0.5 * torch.randn(3, 8, 1)  # (k, r, s), correlated rows

    def off_diagonal(X: torch.Tensor) -> float:
        # X: (k, r, s) of unit rows; returns the largest absolute correlation between rows.
        gram = X @ X.mT  # (k, r, r)
        return float((gram - torch.eye(8)).abs().max())

    start = off_diagonal(G / G.norm(dim=2, keepdim=True))
    one, two = mud(G, 1), mud(G, 2)  # (k, r, s) each
    assert off_diagonal(two) < off_diagonal(one) < start
    assert torch.allclose(one.norm(dim=2), torch.ones(3, 8), atol=1e-5)
    tall = mud(G.mT, 1)  # (k, s, r): the rows of the shorter side
    assert torch.allclose(tall, one.mT, atol=1e-5)
    assert torch.equal(mud(torch.zeros(2, 4, 8), 2), torch.zeros(2, 4, 8))
    assert torch.equal(polar_express(G, steps=5), polar_express(G))


def test_polar_prefixes_stop_short_of_the_full_map() -> None:
    torch.manual_seed(0)
    G = torch.randn(2, 16, 32)  # (k, r, s)

    def distance(steps: int) -> float:
        X = polar_express(G, steps=steps)  # (k, r, s)
        return float((X @ X.mT - torch.eye(16)).abs().max())

    assert distance(5) < distance(3)


def test_mona_closed_form_matches_the_papers_recursion() -> None:
    torch.manual_seed(0)
    beta = 0.9
    alpha = -1 / (2 * (1 - beta))
    gradients = torch.randn(30, 2, 3, 4)  # (t, k, m, n)
    A, previous, slow = torch.zeros(2, 3, 4), torch.zeros(2, 3, 4), torch.zeros(2, 3, 4)
    for G in gradients:
        A = beta * A + (1 - beta) * (G - previous)  # (k, m, n), Algorithm 1's EMA of differences
        previous = G
        assert torch.allclose(mona_accelerated(G, slow, beta), G + alpha * A, atol=1e-4)


def test_radial_brake_scales_each_side_and_both_at_zero_hold_the_norm() -> None:
    torch.manual_seed(0)
    P = torch.ones(1, 4, 4)  # (k, m, n), norm 4, over the reference norm
    outward = radially_braked(P, 0.1 * P, 0.5, 1.0)  # (k, m, n)
    assert math.isclose(float(outward.norm()), 4 * 1.05, rel_tol=1e-5)
    inward = radially_braked(P, -0.1 * P, 0.5, 1.0)  # (k, m, n)
    assert math.isclose(float(inward.norm()), 4 * 0.9, rel_tol=1e-5)
    held = radially_braked(P, torch.randn(1, 4, 4), 0.0, 0.0)  # (k, m, n)
    assert math.isclose(float(held.norm()), 4.0, rel_tol=1e-5)
    small = torch.zeros(1, 4, 4)  # (k, m, n): a zero-initialized output projection steps plainly
    step = torch.randn(1, 4, 4)  # (k, m, n)
    assert torch.equal(radially_braked(small, step, 0.0, 0.0), step)


def test_contra_overweights_small_directions_and_soft_underweights_them() -> None:
    M = torch.diag(torch.tensor([3.0, 1.0]))[None]  # (k, m, n)
    assert math.isclose(float(top_singular_value(M, torch.full((1, 2, 1), 2**-0.5))), 3.0, rel_tol=1e-3)

    def step_under(coefficient: float) -> torch.Tensor:
        W = torch.nn.Parameter(torch.zeros(2, 2))  # (m, n)
        W.grad = M[0].clone()  # (m, n)
        muon = Muon([W], lr=1.0, momentum=0.0, weight_decay=0.0, normalize_rows=False, contra=True)
        muon.contra_coefficient = coefficient
        muon.step()
        return -W.detach().diagonal()  # (2,), the update along each singular direction

    plain = step_under(0.0)  # (2,), Polar Express's map of the singular values, near 1
    relative = torch.tensor([1.0, 1 / 3])  # (2,), each singular value over the top one
    assert torch.allclose(step_under(1.0), 2 * plain - relative, atol=2e-3)  # the small direction gains
    assert torch.allclose(step_under(-0.5), 0.5 * plain + 0.5 * relative, atol=2e-3)  # the small direction loses
    assert step_under(1.0)[1] > plain[1] > step_under(-0.5)[1]


def test_variance_scale_equalizes_the_entries_before_the_orthogonalization() -> None:
    G = torch.tensor([[[100.0, -1.0], [0.01, 5.0]]])  # (k, m, n)
    W = torch.nn.Parameter(torch.zeros(2, 2))  # (m, n)
    muon = Muon([W], lr=1.0, momentum=0.9, weight_decay=0.0, variance_scale=True)
    muon.spread_weight = 1 - 0.9
    group = muon.param_groups[0]
    direction = muon._direction(group, 0, G, torch.zeros_like(G), 0.9, lambda M: M)  # (k, m, n), the scaled direction, unorthogonalized
    assert torch.allclose(direction.abs(), direction.abs()[0, 0, 0] * torch.ones(1, 2, 2), rtol=1e-4)
    assert torch.equal(direction.sign(), G.sign())


def test_muown_keeps_the_weights_at_the_start_a_zero_row_stays_zero_and_it_descends() -> None:
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(4, 4))  # (m, n)
    Z = torch.nn.Parameter(torch.zeros(4, 4))  # (m, n)
    start = W.detach().clone()  # (m, n)
    muon = Muon([W, Z], lr=0.02, momentum=0.9, weight_decay=0.0, normalize_rows=False, muown=(0.18, 0.9, 0.95))
    state = muon.param_groups[0]["muown"][0]
    R, g = state["R"], state["g"]  # (k, m, n), (k, m, 1)
    assert torch.allclose((g / R.norm(dim=2, keepdim=True) * R)[0], start)
    assert torch.equal((g / R.norm(dim=2, keepdim=True) * R)[1], torch.zeros(4, 4)) and bool((R[1].norm(dim=1) > 0).all())
    W.grad, Z.grad = torch.zeros(4, 4), torch.zeros(4, 4)
    muon.step()
    assert torch.allclose(W.detach(), start) and torch.equal(Z.detach(), torch.zeros(4, 4))  # no gradient, no move
    target = torch.randn(4, 4)  # (m, n)
    first = float((W.detach() - target).square().sum() + (Z.detach() - target).square().sum())
    for _ in range(200):
        W.grad, Z.grad = 2 * (W.detach() - target), 2 * (Z.detach() - target)
        muon.step()
    assert float((W.detach() - target).square().sum() + (Z.detach() - target).square().sum()) < 0.5 * first


def test_role_scales_multiply_the_depth_taper_and_the_defaults_carry_no_tensor() -> None:
    assert not OptimizerConfig().scales_members and OptimizerConfig(mlp_lr_scale=0.5).scales_members
    train = load_train()
    genome = with_changes({"optimizer": {"qk_lr_scale": 2.0, "mlp_lr_scale": 0.5, "depth_taper": 0.5}})
    model = train.build_candidate(genome, max_tokens=16)
    scales = model.strategies.optimizer._member_scales(model)
    first, last = model.blocks[0], model.blocks[35]
    assert math.isclose(scales[id(first.attention.query.weight)], 1.25 * 2.0) and math.isclose(scales[id(first.mlp.up.weight)], 1.25 * 0.5)
    assert math.isclose(scales[id(last.attention.value.weight)], 0.75)


def test_cooldown_power_is_the_line_at_one_and_its_power_elsewhere() -> None:
    line, squared = Schedule(ScheduleConfig()), Schedule(ScheduleConfig(cooldown_power=2.0))
    assert line.lr_scale(0.8) == 1 - (0.8 - 0.6) / 0.4 and math.isclose(squared.lr_scale(0.8), 0.25, rel_tol=1e-9)
    assert squared.lr_scale(0.5) == 1.0


def test_lookahead_pulls_the_live_weights_back_on_its_cadence() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    lookahead = Lookahead(LookaheadConfig(variant="periodic", every=2, alpha=0.5))
    state = StepState(model, 0, 0.0, 1.0, torch.zeros(1), torch.zeros(1), lambda: torch.zeros(()))
    lookahead.after_step(state)
    assert lookahead.slow is None  # the warmup's progress 0 starts nothing
    state.progress = 0.5
    lookahead.after_step(state)
    start = model.weight.detach().clone()  # (2, 2), the slow weights
    with torch.no_grad():
        model.weight.add_(1.0)
    lookahead.after_step(state)
    assert torch.equal(model.weight.detach(), start + 1.0)  # one step past the snapshot, no pull yet
    lookahead.after_step(state)
    assert torch.allclose(model.weight.detach(), start + 0.5)  # alpha of the way, and live is slow


def test_canon_mixes_neighbors_only_inside_a_protein() -> None:
    input_ids = torch.tensor([[benchmark.CLS_ID, 5, 6, benchmark.EOS_ID, benchmark.CLS_ID, 7, 8, benchmark.PAD_ID]])  # (b, l)
    same = same_protein_at(input_ids, 2)  # (b, l, 2 r, 1)
    X = torch.arange(1.0, 9.0).view(1, 8, 1)  # (b, l, d)
    assert torch.equal(neighbor_mix(X, same, torch.zeros(4, 1)), X)
    left = neighbor_mix(X, same, torch.tensor([[1.0], [0.0], [0.0], [0.0]])) - X  # (b, l, d), the distance-1 left neighbor
    assert left.squeeze().tolist() == [0.0, 1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0]  # no read across the protein edge, nor into or from padding
    right = neighbor_mix(X, same, torch.tensor([[0.0], [0.0], [0.0], [1.0]])) - X  # (b, l, d), the distance-2 right neighbor
    assert right.squeeze().tolist() == [3.0, 4.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0]
    moved = X.clone()  # (b, l, d)
    moved[0, 5] = 100.0  # the second protein
    weights = torch.ones(4, 1)  # (2 r, d)
    assert torch.equal(neighbor_mix(moved, same, weights)[0, :4], neighbor_mix(X, same, weights)[0, :4])


def test_canon_every_leaves_the_other_blocks_alone() -> None:
    torch.manual_seed(0)
    model = scaffold(with_changes({"canon": {"variant": "neighbors", "every": 2}}))
    canon = model.strategies.canon
    canon.build(model, 16)
    with torch.no_grad():
        canon.holder.weights.fill_(0.5)
    context = ForwardContext(torch.randint(4, 24, (1, 6)), None, None)
    X = torch.randn(1, 6, RECORD.hidden)  # (b, l, d)
    assert torch.equal(canon.updated_stream(1, "attention", X, context), X) and not torch.equal(canon.updated_stream(2, "attention", X, context), X)


def test_protein_context_counts_each_protein_and_never_a_masked_residue() -> None:
    A, C = int(benchmark.ESM2_RESIDUE_COLUMNS[0]), int(benchmark.ESM2_RESIDUE_COLUMNS[1])
    input_ids = torch.tensor([[benchmark.CLS_ID, A, A, benchmark.MASK_ID, benchmark.EOS_ID, benchmark.CLS_ID, C, A, benchmark.PAD_ID]])  # (b, l)
    shares = visible_composition(input_ids)  # (b, l, c)
    assert shares[0, 3, 0] == 1.0 and shares[0, 3].sum() == 1.0  # the masked position reads only the two visible A
    assert shares[0, 6, 0] == 0.5 and shares[0, 6, 1] == 0.5
    sums, counts = protein_sums(torch.ones(1, 9, 2), input_ids)  # (b, l, k), (b, l, 1)
    assert counts[0, :, 0].tolist() == [5.0] * 5 + [3.0] * 3 + [3.0] and sums[0, 0].tolist() == [5.0, 5.0]
    length_cut = torch.tensor([[A, C, benchmark.EOS_ID, benchmark.CLS_ID, C]])  # (b, l): a window opens inside a protein
    assert visible_composition(length_cut)[0, 0, 1] == 0.5 and visible_composition(length_cut)[0, 4, 1] == 1.0


def test_gated_exclusive_attention_starts_at_the_identity_and_can_remove_the_own_value() -> None:
    torch.manual_seed(0)
    Y, V = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 8)  # (b, h, l, d_h) each
    assert torch.equal(without_own_value(Y, V, torch.zeros(3)), Y)
    removed = without_own_value(Y, V, torch.full((3,), 20.0))  # (b, h, l, d_h)
    assert float((removed * V).sum(dim=-1).abs().max()) < 1e-3


def test_qk_width_keeps_rotation_pairs_whole_at_every_head_count() -> None:
    assert AttentionConfig().qk_width(RECORD.hidden) == 64
    for heads in (4, 6, 8, 12):
        for share in (1.0, 0.75, 0.5):
            width = AttentionConfig(heads=heads, qk_share=share).qk_width(RECORD.hidden)
            assert width % 4 == 0 and width <= RECORD.hidden // heads


def scaffold(genome: dict[str, dict[str, Any]]) -> torch.nn.Module:
    """A bare module holding only what a strategy's `build` reads, so one family's arithmetic runs without the 36 blocks."""
    model = torch.nn.Module()
    model.strategies = Strategies(genome)
    model.extensions = torch.nn.ModuleDict()
    return model


def test_value_residual_forms_mix_as_their_docstring_says() -> None:
    torch.manual_seed(0)
    V0, V1 = torch.randn(2, 4, 12, 64), torch.randn(2, 4, 12, 64)  # (b, l, h, d_h) each

    def after_two_blocks(fields: dict[str, Any]) -> torch.Tensor:
        model = scaffold(with_changes({"value_residual": fields}))
        strategy = model.strategies.value_residual
        strategy.build(model, 16)
        context = ForwardContext(torch.zeros(2, 4, dtype=torch.long), None, None)
        assert torch.equal(strategy.values(0, V0, context), V0)
        return strategy.values(1, V1, context).detach()  # (b, l, h, d_h)

    plain = 0.5 * V1 + 0.5 * V0  # (b, l, h, d_h)
    assert torch.equal(after_two_blocks({"variant": "first_block"}), plain)
    assert torch.equal(after_two_blocks({"variant": "first_block", "per_head": 1}), plain)
    assert torch.allclose(after_two_blocks({"variant": "first_block", "anchor_norm": 1}), 0.5 * V1 + 0.5 * norm_rms(V0))
    assert torch.allclose(after_two_blocks({"variant": "normalized"}), plain / 0.5**0.5)


def test_momentum_stream_runs_its_scalars_and_starts_at_the_record_without_its_norm() -> None:
    torch.manual_seed(0)
    model = scaffold(with_changes({"residual": {"variant": "momentum", "velocity_norm": 0, "momentum_beta": 0.25, "momentum_lookahead": 0.5}}))
    residual = model.strategies.residual
    residual.build(model, 16)
    context = ForwardContext(torch.zeros(1, 3, dtype=torch.long), None, None)
    X, Y = torch.randn(1, 3, 768), torch.randn(1, 3, 768)  # (b, l, d) each
    assert torch.equal(residual.sublayer_input(0, "attention", X, context), norm_rms(X))  # the velocity starts at zero
    assert torch.equal(residual.update(0, "attention", X, torch.zeros_like(Y), context), X)  # a zero output adds nothing
    after = residual.update(0, "mlp", X, Y, context).detach()  # (b, l, d)
    assert torch.allclose(after, X + Y, atol=1e-5)  # gamma and nu start at 1
    read = residual.sublayer_input(1, "attention", X, context).detach()  # (b, l, d), from the lookahead X + 0.5 Y
    assert torch.allclose(read, norm_rms(X + 0.5 * Y), atol=1e-5)
    second = residual.update(1, "attention", X, Y, context).detach()  # (b, l, d), V = 0.25 Y + Y
    assert torch.allclose(second, X + 1.25 * Y, atol=1e-4)


def test_head_bias_starts_at_the_training_shares() -> None:
    model = scaffold(with_changes({"head": {"bias": 1}}))
    head = model.strategies.head
    assert head.overrides("logits") and not Strategies(default_genome()).head.overrides("logits")
    head.build(model, 16)
    scored = benchmark.ESM2_RESIDUE_COLUMNS
    shifted = head.logits(model, torch.zeros(1, 1, benchmark.VOCAB_SIZE))[0, 0, scored].detach()  # (c,)
    assert torch.allclose(shifted.softmax(dim=0), unigram(), atol=1e-6)


def test_residue_priors_are_what_they_say() -> None:
    table = torch.tensor(BLOSUM62)  # (c, c)
    assert torch.equal(table, table.T) and table.shape == (20, 20)
    assert math.isclose(sum(UNIGRAM), 1.0, abs_tol=1e-3)
    soft = substitution_targets()  # (c, c)
    assert torch.allclose(soft.sum(dim=1), torch.ones(20))
    residue = {name: index for index, name in enumerate(STANDARD_RESIDUES)}
    assert soft[residue["I"], residue["V"]] > soft[residue["I"], residue["D"]]
    assert soft[residue["W"]].argmax() == residue["W"]


def test_substitution_share_changes_the_objective_and_fades_out() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 4, benchmark.VOCAB_SIZE)  # (b, l, v)
    labels = torch.tensor([[-100, 2, 7, -100]])  # (b, l)
    loss = benchmark.masked_cross_entropy(logits, labels)  # ()
    strategy = Loss(LossConfig(substitution_share=0.2, substitution_until=0.5))
    assert strategy.overrides("training_loss") and strategy.overrides("after_step") and not Loss(LossConfig()).overrides("after_step")
    assert not torch.equal(strategy.training_loss(logits, labels, loss), loss)
    state = StepState(torch.nn.Linear(1, 1), 0, 0.25, 1.0, torch.zeros(1), torch.zeros(1), lambda: torch.zeros(()))
    strategy.after_step(state)
    assert math.isclose(strategy.share, 0.1)
    state.progress = 0.5
    strategy.after_step(state)
    assert strategy.share == 0.0 and torch.equal(strategy.training_loss(logits, labels, loss), loss)
    uniform = substitution_difference(torch.zeros(2, 20), torch.tensor([2, 7]), substitution_targets())  # ()
    assert abs(float(uniform)) < 1e-5


def two_proteins(windows: int = 2) -> torch.Tensor:
    """Windows of 32 random residues holding two proteins each."""
    input_ids = torch.randint(4, 24, (windows, 32))  # (b, l)
    input_ids[:, 0] = benchmark.CLS_ID
    input_ids[:, 16] = benchmark.CLS_ID
    return input_ids  # (b, l)


def perturbed(model: torch.nn.Module) -> None:
    """Draw every block's output projections, so no block is the identity it starts as and a pass moves the stream."""
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.attention.output.weight, std=0.02)  # (d, d)
            torch.nn.init.normal_(block.mlp.down.weight, std=0.02)  # (d, f)


def test_a_tied_loop_stores_each_core_matrix_once_and_muon_steps_it_once() -> None:
    train = load_train()
    torch.manual_seed(0)
    model = train.build_candidate(with_changes({"loop": {"variant": "tied"}}), max_tokens=64)
    record = train.build_candidate(default_genome(), max_tokens=64)
    config = model.strategies.loop.config  # prelude 6, core 8, passes 3: positions 14 to 29 repeat blocks 6 to 13
    for position in range(14, 30):
        source = 6 + (position - 6) % 8
        assert config.source(position) == source
        assert model.blocks[position].attention.query.weight is model.blocks[source].attention.query.weight
        assert model.blocks[position].mlp.down.weight is model.blocks[source].mlp.down.weight
    assert model.blocks[30].mlp.up.weight is not model.blocks[14].mlp.up.weight and config.stored_blocks(RECORD.blocks) == 20
    matrices = sum(parameter.numel() for parameter in record.blocks[0].parameters() if parameter.ndim == 2)
    assert benchmark.check_architecture(model)["parameters"] == benchmark.check_architecture(record)["parameters"] - 16 * matrices
    muon = model.strategies.optimizer.build_optimizers(model, 0.95)[0]
    members = [parameter for group in muon.param_groups for parameter in group["params"]]
    assert len(members) == len({id(parameter) for parameter in members}) == 20 * 6  # q, k, v, o, up, down of each stored block


def test_a_one_pass_loop_is_the_record_forward() -> None:
    train = load_train()
    input_ids = two_proteins()  # (b, l)
    outputs = []
    for genome in (default_genome(), with_changes({"loop": {"variant": "tied", "passes": 1}})):
        torch.manual_seed(0)
        model = train.build_candidate(genome, max_tokens=64)
        perturbed(model)
        model.eval()
        with torch.no_grad():
            outputs.append(model(input_ids, input_ids != benchmark.PAD_ID))  # (b, l, v)
    assert torch.equal(outputs[0], outputs[1])


def test_the_dial_and_the_exits_stop_passes_and_keep_a_stopped_tokens_state() -> None:
    train = load_train()
    torch.manual_seed(0)
    # No re-injection, so the stream at a pass's end is the pass's own result in every run (with it, a running token's
    # stream there already holds the next pass's injection, which a stopped token rightly skips).
    model = train.build_candidate(with_changes({"loop": {"variant": "tied", "inject": "none"}}), max_tokens=64)
    perturbed(model)
    with torch.no_grad():  # a quiet core: a pass moves each token by about a percent of its state, so a fixed point is near
        for block in model.blocks[6:14]:
            block.attention.output.weight.mul_(0.01)  # (d, d)
            block.mlp.down.weight.mul_(0.01)  # (d, f)
    model.eval()
    loop = model.strategies.loop
    input_ids = two_proteins()  # (b, l), 64 real tokens

    def states() -> list[torch.Tensor]:
        loop.holder.pass_tokens.zero_()
        with torch.no_grad():
            return model.hidden_states(input_ids, input_ids != benchmark.PAD_ID)  # 37 of (b, l, d); pass j ends at state 6 + 8 j

    full = states()
    assert not torch.equal(full[14], full[30]) and loop.holder.pass_tokens.tolist() == [64, 64, 64, 64]
    loop.eval_passes = 1
    dialed = states()
    assert torch.equal(dialed[14], full[14]) and torch.equal(dialed[22], dialed[14]) and torch.equal(dialed[30], dialed[14])
    assert loop.holder.pass_tokens.tolist() == [64, 64, 0, 0]
    loop.eval_passes = 3
    for exit, threshold, expected in (("confidence", 0.0, dialed), ("confidence", 1.0, full), ("fixed_point", 0.1, dialed),
                                      ("fixed_point", 0.0, full)):
        loop.exit, loop.exit_threshold = exit, threshold
        assert all(torch.equal(state, reference) for state, reference in zip(states(), expected, strict=True)), (exit, threshold)


def test_a_sampled_depth_discards_the_passes_past_each_windows_draw() -> None:
    train = load_train()
    torch.manual_seed(0)
    model = train.build_candidate(with_changes({"loop": {"variant": "tied", "sample": 1}}), max_tokens=64)
    perturbed(model)
    model.train()
    input_ids = two_proteins(windows=8)  # (b, l)
    torch.manual_seed(1)
    depths = torch.randint(1, 4, (8, 1, 1))[:, 0, 0].tolist()  # the draw the forward makes first
    assert len(set(depths)) == 3
    torch.manual_seed(1)
    with torch.no_grad():
        states = model._states(input_ids, ForwardContext(input_ids, None, document_dense_mask(input_ids)))[0]  # 37 of (b, l, d)
    for window, depth in enumerate(depths):
        end = 6 + 8 * depth  # the state after pass `depth`
        assert torch.equal(states[30][window], states[end][window])
        assert not torch.equal(states[end][window], states[end - 8][window])


def test_truncated_passes_cut_the_stream_keep_the_forward_and_train_the_prelude_through_e() -> None:
    train = load_train()
    corrupted, labels = benchmark.corrupt(two_proteins(), torch.Generator().manual_seed(0))  # each (b, l)
    finals, cut, gradients, embedding_gradients = [], [], [], []
    for grad_passes in (0, 1):
        torch.manual_seed(0)
        model = train.build_candidate(with_changes({"loop": {"variant": "tied", "grad_passes": grad_passes}}), max_tokens=64)
        perturbed(model)
        context = ForwardContext(corrupted, None, document_dense_mask(corrupted))
        final = model._states(corrupted, context)[1]  # (b, l, d)
        benchmark.masked_cross_entropy(model.head(model.head_features(final)), labels).backward()
        finals.append(final.detach())
        cut.append(not context.stored[HELD].requires_grad)  # where the last pass began
        gradients.append(model.blocks[6].mlp.up.weight.grad.clone())  # (f, d)
        embedding_gradients.append(model.embedding.weight.grad.abs().sum())
    assert torch.equal(finals[0], finals[1]) and cut == [False, True]
    assert not torch.equal(gradients[0], gradients[1]) and float(embedding_gradients[1]) > 0


def test_a_free_shape_trains_a_cpu_step_with_a_loop_and_routing() -> None:
    genome = with_changes({"shape": {"blocks": 16, "hidden": 512, "intermediate": 1536}, "attention": {"heads": 8},
                           "loop": {"variant": "tied", "prelude": 2, "core": 4, "passes": 3, "sample": 1, "readout_weight": 0.3},
                           "routing": {"variant": "per_pass", "experts": 4, "top_k": 2}})
    one_cpu_step(genome)
    model = load_train().build_candidate(genome, max_tokens=64)
    receipt = benchmark.check_architecture(model)
    assert receipt["blocks"] == 16 and receipt["hidden"] == 512 and model.blocks[15].mlp.up.weight.shape == (1536, 512)
    assert model.extensions["routing"].routers.shape == (16, 4, 512)
    assert model.blocks[10].mlp.up.weight is model.blocks[2].mlp.up.weight  # the third pass of core block 2


@pytest.mark.parametrize("changes, family", [
    ({"shape": {"hidden": 640}}, "attention"),  # 12 heads of 53.3
    ({"shape": {"hidden": 1024}, "attention": {"heads": 6}}, "attention"),
    ({"shape": {"hidden": 2048}, "attention": {"heads": 4}}, "attention"),  # 512 a head, past the kernels' 256
    ({"shape": {"blocks": 20}, "loop": {"variant": "tied"}}, "loop"),  # 6 + 8 x 3 > 20
    ({"loop": {"variant": "tied"}, "stream": {"variant": "hyper"}}, "loop"),
    ({"sglr": {"variant": "pool"}, "loop": {"variant": "tied"}}, "sglr"),
    ({"shape": {"blocks": 24}, "sglr": {"variant": "pool"}}, "sglr"),  # 6 + 24 > 24
    ({"routing": {"variant": "per_pass", "experts": 3, "top_k": 1}}, "routing"),
    ({"shape": {"blocks": 35}, "unet": {"variant": "mirrored"}}, "unet"),
    ({"shape": {"blocks": 12}, "memory": {"variant": "product_keys", "block": 12}}, "memory"),
    ({"shape": {"blocks": 12}, "latent": {"variant": "next_layer", "source_block": 12}}, "latent"),
    ({"shape": {"blocks": 8}, "value_embeddings": {"variant": "ends", "blocks_each_end": 5}}, "value_embeddings"),
])
def test_families_that_cannot_compose_refuse_as_a_genome_conflict(changes: dict[str, dict[str, Any]], family: str) -> None:
    with pytest.raises(GenomeConflict, match=f"{family}: "):
        resolve({"families": changes})


def test_reinjection_and_residual_scale_are_looms_and_bad_loops_refuse() -> None:
    config = LoopConfig(variant="tied", residual_lambda=0.5)  # prelude 6, core 8, passes 3
    assert math.isclose(config.share(2), 0.5 / (2 * math.sqrt(8))) and math.isclose(config.share(3), 0.5 / (3 * math.sqrt(8)))
    assert math.isclose(config.residual_scale, 0.5 / (3 * math.sqrt(8))) and LoopConfig(variant="tied").residual_scale == 1.0
    X, E = torch.randn(2, 4, 8), torch.randn(2, 4, 8)  # (b, l, d) each
    assert torch.allclose(reinjected(X, E, 0.25), 0.75 * X + 0.25 * E)
    assert config.readout_passes == (1, 2) and LoopConfig(variant="tied", prelude=2, passes=4, grad_passes=2).readout_passes == (3,)
    for bad in ({"variant": "untied", "grad_passes": 1}, {"grad_passes": 1, "sample": 1}, {"eval_passes": 4},
                {"inject": "gated"}, {"grad_passes": 3}, {"readout_weight": 0.1, "passes": 1}, {"exit": "entropy"}):
        with pytest.raises(ValueError):
            LoopConfig(**{"variant": "tied", **bad})
    model = scaffold(with_changes({"loop": {"variant": "tied", "inject": "learned"}}))
    strategy = model.strategies.loop
    strategy.build(model, 64)
    assert torch.allclose(torch.sigmoid(strategy.holder.inject_logits[:, 0]), torch.tensor([config.share(2), config.share(3)]))


def test_the_residual_read_weighs_updates_by_content_and_age_and_starts_as_their_sum() -> None:
    E = torch.randn(2, 3, 8)  # (b, l, d), 2 heads of 4
    updates = [torch.randn(2, 3, 8) for _ in range(3)]  # 3 of (b, l, d), oldest first
    plain = read_updates(E, updates, torch.zeros(2, 4), torch.zeros(2))  # (b, l, d)
    assert torch.allclose(plain, E + sum(updates), atol=1e-6)
    decayed = read_updates(E, updates, torch.zeros(2, 4), torch.tensor([0.55, 0.0]))  # (b, l, d)
    weights = 3 * torch.softmax(-0.55 * torch.tensor([2.0, 1.0, 0.0]), dim=0)  # (n,): the newest weighs most
    expected = E[..., :4] + sum(weight * update[..., :4] for weight, update in zip(weights, updates, strict=True))  # (b, l, 4)
    assert torch.allclose(decayed[..., :4], expected, atol=1e-6) and torch.allclose(decayed[..., 4:], plain[..., 4:], atol=1e-6)
    query = torch.zeros(2, 4)
    query[0, 0] = 50.0  # head 0 reads the update whose normalized first channel is largest, nearly alone
    sharp = read_updates(E, updates, query, torch.zeros(2))  # (b, l, d)
    U = torch.stack(updates, dim=2)  # (b, l, n, d)
    pick = norm_rms(U[..., :4])[..., 0].argmax(dim=2)  # (b, l)
    chosen = U[..., :4].gather(2, pick[:, :, None, None].expand(2, 3, 1, 4))[:, :, 0]  # (b, l, 4)
    assert torch.allclose(sharp[..., :4], E[..., :4] + 3 * chosen, atol=1e-2)
    update = torch.randn(2, 3, 8)  # (b, l, d)
    assert torch.equal(gated_update(update, torch.zeros(2, 4), torch.zeros(2)), update)  # 2 sigmoid(0) is exactly 1
    halved = gated_update(update, torch.zeros(2, 4), torch.tensor([0.0, -math.log(3.0)]))  # 2 sigmoid(-ln 3) = 1/2
    assert torch.equal(halved[..., :4], update[..., :4]) and torch.allclose(halved[..., 4:], 0.5 * update[..., 4:])


def test_a_loop_with_the_read_and_the_write_gate_starts_as_the_carried_loop_and_trains() -> None:
    train = load_train()
    input_ids = two_proteins()  # (b, l)
    outputs = []
    for fields in ({"inject": "none"}, {"inject": "read"}, {"inject": "read", "write_gate": 1}, {"inject": "none", "write_gate": 1}):
        torch.manual_seed(0)
        model = train.build_candidate(with_changes({"loop": {"variant": "tied", **fields}}), max_tokens=64)
        perturbed(model)
        model.eval()
        with torch.no_grad():
            outputs.append(model(input_ids, input_ids != benchmark.PAD_ID))  # (b, l, v)
    assert all(torch.allclose(output, outputs[0], atol=1e-4) for output in outputs[1:])
    genome = with_changes({"loop": {"variant": "tied", "inject": "read", "read_decay": 0.55, "write_gate": 1, "sample": 1,
                                    "readout_weight": 0.3}})
    one_cpu_step(genome)
    torch.manual_seed(0)
    model = train.build_candidate(genome, max_tokens=64)
    perturbed(model)
    corrupted, labels = benchmark.corrupt(two_proteins(), torch.Generator().manual_seed(0))  # each (b, l)
    final = model._states(corrupted, ForwardContext(corrupted, None, document_dense_mask(corrupted)))[1]  # (b, l, d)
    benchmark.masked_cross_entropy(model.head(model.head_features(final)), labels).backward()
    holder = model.extensions["loop"]
    assert holder.read_queries.shape == (3, 12, 64) and torch.equal(holder.read_decay.detach(), torch.full((12,), 0.55))
    for name in ("read_queries", "read_decay", "write_weights", "write_bias"):
        assert getattr(holder, name).grad.abs().sum() > 0, name
    groups = {group.name: group for group in model.strategies.loop.parameter_groups(model)}
    assert len(groups["loop_injection"].parameters) == 4
    with pytest.raises(ValueError, match="grad_passes"):
        LoopConfig(variant="tied", inject="read", grad_passes=1)


def test_sglr_programs_route_every_token_of_a_step_one_way_and_the_floor_balances_only_collapse() -> None:
    assert math.isclose(float(usage_entropy(torch.tensor([[0, 1, 2]]), 3)), math.log(3), rel_tol=1e-6)
    assert float(usage_entropy(torch.zeros(1, 4, dtype=torch.long), 3)) == 0.0
    train = load_train()
    torch.manual_seed(0)
    model = train.build_candidate(with_changes({"sglr": {"variant": "pool", "steps": 12, "router": "program"}}), max_tokens=64)
    perturbed(model)
    model.eval()
    input_ids = two_proteins()  # (b, l)
    with torch.no_grad():
        model(input_ids, input_ids != benchmark.PAD_ID)
    routes = model.strategies.sglr.holder.routes  # (steps, m)
    assert routes.max(dim=1).values.tolist() == [64] * 12  # one module takes every token at each step
    assert not hasattr(model.extensions["sglr"], "routers") and model.extensions["sglr"].program.shape == (12, 4, 3)
    assert not model.strategies.sglr.overrides("auxiliary_loss")  # a program has no balance term
    one_cpu_step(with_changes({"sglr": {"variant": "pool", "steps": 12, "router": "program"}}))
    one_cpu_step(with_changes({"sglr": {"variant": "pool", "steps": 12, "balance_floor": 0.5}}))
    corrupted, labels = benchmark.corrupt(two_proteins(), torch.Generator().manual_seed(0))  # each (b, l)
    terms = {}
    for floor in (0.0, 1.0, 1e-6):  # never floored; floored at ln 3, which 64 tokens over 3 modules never reach; only a collapse
        torch.manual_seed(0)
        model = train.build_candidate(with_changes({"sglr": {"variant": "pool", "steps": 12, "balance_floor": floor}}), max_tokens=64)
        perturbed(model)
        model.train()
        context = ForwardContext(corrupted, None, document_dense_mask(corrupted), labels=labels)
        with torch.no_grad():  # a training forward, so the terms are kept; FlexAttention has no CPU backward
            model._states(corrupted, context)
        terms[floor] = [float(context.stored[f"{BALANCE}{step}"]) for step in range(12)]
    assert terms[1.0] == terms[0.0] and all(term > 0 for term in terms[0.0])
    assert all(term == 0.0 for term in terms[1e-6])  # no step collapsed onto one module, so none is balanced
    with pytest.raises(ValueError, match="router"):
        resolve({"families": {**default_genome(), "sglr": {**default_genome()["sglr"], "router": "tokens"}}})


def test_routers_pick_each_tokens_top_groups_and_the_bias_balances_them() -> None:
    scores = torch.tensor([[[0.3, -1.0, 2.0, 0.1]]])  # (b, l, e)
    assert chosen_groups(scores, torch.zeros(4), 2).tolist() == [[[True, False, True, False]]]
    assert chosen_groups(scores, torch.tensor([0.0, 5.0, 0.0, 0.0]), 2).tolist() == [[[False, True, True, False]]]
    load = torch.tensor([[6.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])  # (r, e): one router over its share, one unused
    assert torch.allclose(rebalanced(torch.zeros(2, 4), load, 0.1), torch.tensor([[-0.1, 0.0, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0]]))
    train = load_train()
    torch.manual_seed(0)
    for variant, routers in (("shared", 20), ("per_pass", 36)):
        model = train.build_candidate(with_changes({"loop": {"variant": "tied"}, "routing": {"variant": variant}}), max_tokens=64)
        assert len(set(model.strategies.routing.router_of)) == routers
    routing = model.strategies.routing
    H, X = torch.randn(2, 5, 2048), torch.randn(2, 5, 768)  # (b, l, f), (b, l, d)
    gated = routing.mlp_hidden(3, H, X)  # (b, l, f)
    scores = X @ routing.holder.routers[3].detach().T  # (b, l, e)
    gates = torch.where(chosen_groups(scores, routing.holder.bias[3], 2), 2 * torch.sigmoid(scores), 0.0)  # (b, l, e)
    assert torch.allclose(gated, (H.view(2, 5, 8, 256) * gates[..., None]).view(2, 5, 2048), atol=1e-6)
    assert gated.view(2, 5, 8, 256).ne(0).any(dim=-1).sum(dim=-1).eq(2).all()
    assert routing.holder.load[3].sum() == 2 * 5 * 2  # each token's two choices, counted for the step's rebalancing


def test_sglr_adds_exactly_one_modules_update_per_token_and_step() -> None:
    updates = tuple(torch.randn(2, 3, 4) for _ in range(3))  # 3 of (b, l, d)
    probabilities = torch.tensor([[0.2, 0.5, 0.3], [0.6, 0.3, 0.1], [0.1, 0.1, 0.8]]).expand(2, 3, 3).clone().requires_grad_()  # (b, l, m)
    routed, choices = routed_update(updates, probabilities)  # (b, l, d), (b, l)
    assert choices.tolist() == [[1, 0, 2], [1, 0, 2]]
    assert torch.equal(routed.detach(), torch.stack(updates, dim=2)[torch.arange(2)[:, None], torch.arange(3)[None], choices])
    routed.sum().backward()
    assert probabilities.grad is not None and probabilities.grad.abs().sum() > 0
    assert math.isclose(float(switch_balance(torch.full((1, 3, 3), 1 / 3), torch.tensor([[0, 1, 2]]))), 1.0, rel_tol=1e-6)
    assert math.isclose(float(switch_balance(torch.tensor([[[1.0, 0.0, 0.0]] * 3]), torch.zeros(1, 3, dtype=torch.long))), 3.0)
    train = load_train()
    torch.manual_seed(0)
    model = train.build_candidate(with_changes({"sglr": {"variant": "pool", "steps": 12}}), max_tokens=64)
    assert all(model.blocks[position].mlp.up.weight is model.blocks[6].mlp.up.weight for position in range(7, 18))
    assert model.blocks[18].mlp.up.weight is not model.blocks[6].mlp.up.weight
    perturbed(model)
    model.eval()
    input_ids = two_proteins()  # (b, l)
    with torch.no_grad():
        model(input_ids, input_ids != benchmark.PAD_ID)  # (b, l, v)
    assert model.strategies.sglr.holder.routes.sum(dim=1).tolist() == [64] * 12
    with pytest.raises(ValueError):
        train.build_candidate(with_changes({"sglr": {"variant": "pool"}, "loop": {"variant": "tied"}}), max_tokens=64)
