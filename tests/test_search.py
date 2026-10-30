"""The search driver on toy objectives: the screening design recovers planted effects, the GP and the genetic
proposers improve on their starting genomes, and a batch flows from proposal to the ledger with its successive halving.

Symbols: n runs, k factors.
"""

from __future__ import annotations

import json
import sys
import numpy as np
import pytest

from pathlib import Path
from typing import Any

from speedrunning_plms.strategies import Strategies, default_genome, resolve


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))  # search.py and record.py are scripts beside the package

import search  # noqa: E402


def toy_loss(genome: dict[str, dict[str, Any]], rng: np.random.Generator | None = None) -> float:
    """A planted landscape: the triangle kernel and the gates help, U-net hurts, and Muon's rate is best at 0.05."""
    loss = 2.75
    loss -= 0.010 * (genome["attention"]["variant"] == "triangle")
    loss -= 0.004 * (genome["gates"]["variant"] == "residual_channels")
    loss += 0.006 * (genome["unet"]["variant"] == "mirrored")
    loss += 0.02 * abs(np.log(genome["optimizer"]["muon_lr"] / 0.05))
    return float(loss + (rng.normal(0.0, search.NOISE_SD) if rng is not None else 0.0))


def test_plackett_burman_is_balanced_and_orthogonal() -> None:
    design = search.plackett_burman_12()  # (12, 11)
    assert design.shape == (12, 11) and (design.sum(0) == 0).all()
    assert (design.T @ design == 12 * np.eye(11)).all()


def test_screening_recovers_planted_main_effects() -> None:
    factors = ["unet", "gates", "triangle", "ema", "rope_1e5", "slopes_x2", "no_softcap", "muon"]
    proposals = search.ScreeningProposer(factors).propose([], 12, np.random.default_rng(0))
    assert len(proposals) == 12
    rng = np.random.default_rng(1)
    levels = np.array([[proposal.factors[factor] for factor in factors] for proposal in proposals])  # (n, k)
    values = np.array([toy_loss(proposal.genome, rng) for proposal in proposals])  # (n,)
    effects = dict(zip(factors, search.main_effects(levels, values), strict=True))
    tolerance = 3 * search.NOISE_SD
    assert abs(effects["triangle"] + 0.010) < tolerance and abs(effects["gates"] + 0.004) < tolerance
    assert abs(effects["unet"] - 0.006) < tolerance and all(abs(effects[factor]) < tolerance for factor in ("ema", "rope_1e5", "muon"))


def test_screening_runs_against_the_best_observation() -> None:
    record = search.written(default_genome(), {"value_residual.variant": "first_block"})
    observations = [search.Observation("022", default_genome(), 2.749), search.Observation("028", record, 2.732)]
    proposals = search.ScreeningProposer(["unet", "gates", "muon"]).propose(observations, 12, np.random.default_rng(0))
    assert all(proposal.parents == ["028"] for proposal in proposals)
    assert all(proposal.genome["value_residual"]["variant"] == "first_block" for proposal in proposals)
    assert proposals[-1].genome == record  # the all-low row is the record itself
    with pytest.raises(ValueError, match="already on"):
        search.ScreeningProposer(["unet", "value_residual"]).propose(observations, 12, np.random.default_rng(0))



def test_proposers_never_propose_a_retired_choice() -> None:
    retired = ["unet.variant=mirrored", "optimizer.muon_lr"]
    space = search.open_space(retired)
    rng = np.random.default_rng(4)
    observations = [search.Observation(f"start_{index}", search.written(search.random_genome(space, rng), {"unet.variant": "mirrored"}), 2.75 - 0.001 * index)
                    for index in range(6)]  # every parent holds the retired U-net
    for proposer in (search.BayesianProposer(space, pool=400), search.GeneticProposer(space)):
        for proposal in proposer.propose(observations, 4, np.random.default_rng(5)):
            assert proposal.genome["unet"]["variant"] == "none"
            assert proposal.genome["optimizer"]["muon_lr"] == default_genome()["optimizer"]["muon_lr"]


def test_proposers_keep_to_the_warmup_cap() -> None:
    assert search.predicted_warmup(default_genome()) == search.RECORD_WARMUP
    costly = search.written(default_genome(), {"attention.variant": "triangle", "residual.variant": "ngpt"})
    assert search.predicted_warmup(costly) == search.RECORD_WARMUP + 46 + 36 and not search.fits_warmup(costly)
    with pytest.raises(ValueError, match="compile"):
        search.ScreeningProposer(["triangle", "ngpt", "hyper"]).propose([], 12, np.random.default_rng(0))
    space = search.open_space([])
    observations = observed(6, np.random.default_rng(2), space)
    for proposer in (search.BayesianProposer(space, pool=400), search.GeneticProposer(space)):
        proposals = proposer.propose(observations, 4, np.random.default_rng(3))
        assert len(proposals) == 4 and all(search.fits_warmup(proposal.genome) for proposal in proposals)


def observed(count: int, rng: np.random.Generator, space: list[search.Dimension]) -> list[search.Observation]:
    """Starting genomes that never hold the triangle kernel, so a proposer must find it."""
    genomes = [default_genome()] + [search.written(search.random_genome(space, rng), {"attention.variant": "standard"})
                                    for _ in range(count - 1)]
    return [search.Observation(f"start_{index}", genome, toy_loss(genome, rng)) for index, genome in enumerate(genomes)]


def test_the_window_sets_the_count_so_a_step_holds_65536_tokens() -> None:
    assert "batch.windows_per_step" not in {dimension.key for dimension in search.SPACE}
    for window in (1024, 2048):
        genome = search.switched(["window_2048", "unet"], np.array([1 if window == 2048 else -1, -1]), default_genome())
        assert Strategies(genome).batch.step_tokens == 65_536 and Strategies(genome).batch.config.count == 65_536 // window
    hand_set = search.written(default_genome(), {"batch.windows_per_step": 64})
    assert search.within(search.written(hand_set, {"batch.window": 2048}), list(search.SPACE))["batch"]["windows_per_step"] == 0


# The planted landscape's own gates choices, so the toy does not change when the search space gains a gate.
TOY_SPACE = [dimension for dimension in search.SPACE if dimension.key in
             ("attention.variant", "unet.variant", "optimizer.muon_lr", "ema.variant")] + [
    search.Dimension("gates.variant", ("none", "residual_channels"))]


@pytest.mark.parametrize("proposer_type", [search.BayesianProposer, search.GeneticProposer])
def test_adaptive_proposers_improve_on_the_start(proposer_type: type) -> None:
    """Over ten proposal seeds, at least seven find the triangle kernel within six rounds. One seed's outcome moves
    with every random draw the registry adds (crossover draws one number per family), a rate does not."""
    found = []
    for seed in range(10):
        observations = observed(6, np.random.default_rng(1), TOY_SPACE)
        start = min(toy_loss(observation.genome) for observation in observations)
        rng = np.random.default_rng(seed)
        proposer = search.BayesianProposer(TOY_SPACE, pool=600) if proposer_type is search.BayesianProposer else search.GeneticProposer(TOY_SPACE)
        for round_index in range(6):
            proposals = proposer.propose(observations, 4, rng)
            assert len({search.genome_sha256(proposal.genome) for proposal in proposals}) == 4 and all(proposal.reason for proposal in proposals)
            observations += [search.Observation(f"round{round_index}_{index}", proposal.genome, toy_loss(proposal.genome, rng))
                             for index, proposal in enumerate(proposals)]
        found.append(min(toy_loss(observation.genome) for observation in observations) < start - 0.005)
    assert sum(found) >= 7, found  # the triangle kernel found


def test_gp_does_not_chase_one_seed_of_noise() -> None:
    """Two observations of nearly the same genome a seed's noise apart: the posterior means sit closer than the data."""
    genome = default_genome()
    X = np.stack([search.encode(genome, list(search.SPACE))] * 2 + [search.encode(search.written(genome, {"unet.variant": "mirrored"}), list(search.SPACE))])  # (3, D)
    y = np.array([2.7490, 2.7470, 2.7560])  # (3,)
    mean, _, hyper = search.gp_posterior(X, y, X[:2])
    assert hyper["noise_sd"] >= search.NOISE_SD and abs(mean[0] - mean[1]) < 1e-9


def write_result(directory: Path, name: str, val_select: float, budget: int) -> None:
    run_result = {"name": name, "status": "completed", "seed": 0, "candidate_sha256": "c" * 64, "comparison_key": "k" * 64,
              "clock": {"steps": 10, "train_seconds": float(budget), "warmup_seconds": 30.0, "eval_seconds": 5.0, "valid": True},
              "metrics": {"val_select": {"loss": val_select}, **({"test": {"loss": val_select - 0.003}} if budget == 600 else {})},
              "hypothesis": "h", "selected_step": 10, "architecture": {"parameters": 1}, "environment": {"gpu": None}}
    (directory / f"{name}.json").write_text(json.dumps(run_result), encoding="utf-8")


def test_a_batch_flows_from_proposal_to_the_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search, "PROJECT", tmp_path)
    monkeypatch.delenv("SPEEDRUN_LAUNCHER", raising=False)
    monkeypatch.delenv("SPEEDRUN_FETCH", raising=False)
    state = search.load_state(tmp_path / "search.json")
    state["seeded"]["022_embed_head_lr"] = {"genome": {}, "val_select": 2.7490}
    for name, changes in (("023_x", {"gates": {"variant": "residual_channels"}}), ("024_x", {"unet": {"variant": "mirrored"}})):
        state["seeded"][name] = {"genome": changes, "val_select": toy_loss(resolve({"families": changes}))}
    names = search.propose(state, search.GeneticProposer(list(search.SPACE)), "ga1", 4, 0, [search.RUNG_BUDGET, search.FULL_BUDGET])
    shipped = search.ship(state, ["ga1"], search.RUNG_BUDGET)
    assert shipped == [f"{name}_r180" for name in names]
    assert (tmp_path / "candidates" / "queue" / f"{names[0]}_r180.args").read_text().split() == ["--budget", "180", "--final-splits", "val_select"]
    results, ledger = tmp_path / "results", tmp_path / "runs.jsonl"
    results.mkdir()
    for index, name in enumerate(names):
        write_result(results, f"{name}_r180", 2.76 - 0.001 * index, 180)
    search.collect(state, "ga1", results, ledger)
    promoted = [name for name in names if state["proposals"][name]["promoted"]]
    assert promoted == names[2:]  # the lower half of the short rung's losses
    assert search.ship(state, ["ga1"]) == promoted
    write_result(results, promoted[0], 2.7400, 600)
    search.collect(state, "ga1", results, ledger)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["rung_seconds"] for row in rows] == [180] * 4 + [600]
    assert rows[-1]["verdict"] == "keep" and rows[-1]["proposer"] == "genetic" and len(rows[-1]["parents"]) == 2
    assert search.incumbent_of(state).name == promoted[0]
