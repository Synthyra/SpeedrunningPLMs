"""The strategy registry: every variant builds, passes the shape rule and trains a CPU step; the genome round-trips;
and the default genome reproduces the record's file exactly.

Symbols: b windows, l tokens, v = 33 tokens.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import subprocess
import sys
import numpy as np
import pytest
import torch

from pathlib import Path
from types import ModuleType
from typing import Any

from speedrunning_plms import benchmark
from speedrunning_plms.strategies import (
    FAMILIES, Strategies, default_genome, genome_document, genome_sha256, load_genome, resolve, save_genome,
)
from speedrunning_plms.strategies.attention import capped_attention, document_dense_mask, documents, shift_key_pairs
from speedrunning_plms.strategies.batch import Batch, BatchConfig, best_fit
from speedrunning_plms.strategies.curriculum import Curriculum, CurriculumConfig
from speedrunning_plms.strategies.ema import Ema, EmaConfig
from speedrunning_plms.strategies.kernel import Kernel, KernelConfig
from speedrunning_plms.strategies.optimizer import OptimizerChoice, OptimizerConfig, amuse_beta
from speedrunning_plms.strategies.positional import Positional, PositionalConfig, Rotary, periodic_term
from speedrunning_plms.strategies.schedule import Schedule, ScheduleConfig
from speedrunning_plms.strategies.strategy import HOOKS, StepState


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
]


def one_cpu_step(genome: dict[str, dict[str, Any]]) -> None:
    """Build, check the shape rule, then one training step with every hook and callback, and an evaluation forward."""
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
    logits, auxiliary = model(corrupted, labels=labels, dense_mask=document_dense_mask(corrupted))  # (b, l, v), ()
    loss = benchmark.masked_cross_entropy(logits, labels) + auxiliary  # ()
    assert torch.isfinite(loss)
    loss.backward()

    def loss_without_grad() -> torch.Tensor:
        with torch.no_grad():
            return benchmark.masked_cross_entropy(model(corrupted, dense_mask=document_dense_mask(corrupted)), labels)  # ()

    state = StepState(model, 0, 0.5, 1.0, corrupted, labels, loss_without_grad)
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
    assert len(states) == benchmark.BLOCKS + 1


@pytest.mark.parametrize(("family", "variant"), VARIANTS, ids=[f"{family}-{variant}" for family, variant in VARIANTS])
def test_every_variant_builds_and_trains_a_cpu_step(family: str, variant: str) -> None:
    one_cpu_step(with_changes({family: {"variant": variant}}))


@pytest.mark.parametrize("changes", COMPOSED, ids=["triangle_differential-ngpt-hyper-and-more", "paired-dense-readout",
                                                   "differential6-dense-units", "triangle-partial-rope-and-2026-10-30",
                                                   "boost-ramp-anvil-lawa-neighbors", "paired-offset-cap-periodic-head-two-decay",
                                                   "differential-offset-cap-periodic-alone-leap"])
def test_composed_genomes_train_a_cpu_step(changes: dict[str, dict[str, Any]]) -> None:
    one_cpu_step(with_changes(changes))


def test_default_genome_has_only_the_softcap_hook() -> None:
    strategies = Strategies(default_genome())
    hooked = {hook: [strategy.family for strategy in strategies.hooked(hook)] for hook in HOOKS}
    assert hooked == {hook: ["softcap"] if hook == "logits" else [] for hook in HOOKS}
    assert set(HOOKS) >= {"embed", "block_input", "normalized_input", "keys", "queries", "values", "attention_output",
                          "mlp_hidden", "sublayer_output", "updated_stream", "head_input", "logits", "auxiliary_loss",
                          "after_backward", "after_step", "evaluation_weights"}
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
    return tokens, offsets


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


def test_neighbor_heads_score_only_visible_residues_at_masked_positions() -> None:
    torch.manual_seed(0)
    model = load_train().build_candidate(with_changes({"latent": {"variant": "neighbors"}}), max_tokens=64)
    latent = model.strategies.latent
    input_ids = torch.tensor([[benchmark.CLS_ID, 5, benchmark.MASK_ID, 7, benchmark.EOS_ID, benchmark.CLS_ID]])  # (1, 6)
    labels = torch.full((1, 6), benchmark.IGNORE_LABEL)  # (1, 6)
    labels[0, 2] = 0
    final = torch.randn(1, 6, benchmark.HIDDEN)  # (1, 6, d)
    hidden = torch.nn.functional.rms_norm(final, (benchmark.HIDDEN,))[0, 2]  # (d,), the masked position's state
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
