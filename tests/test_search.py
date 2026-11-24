"""The search driver on toy objectives: the screening design recovers planted effects, the GP and the genetic
proposers improve on their starting genomes, and a batch flows from proposal to the ledger with its successive halving.

Symbols: n runs, k factors.
"""

from __future__ import annotations

import json
import math
import sys
import numpy as np
import pandas as pd
import pytest

from pathlib import Path
from typing import Any
from sklearn.ensemble import ExtraTreesRegressor

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
    observations = [search.Observation(f"start_{index}", search.written(search.random_genome(space, rng), {"unet.variant": "mirrored", "shape.blocks": 36}),
                                       2.75 - 0.001 * index)
                    for index in range(6)]  # every parent holds the retired U-net, on a depth a mirror takes
    for proposer in (search.BayesianProposer(space, pool=400), search.GeneticProposer(space), search.RandomProposer(space)):
        for proposal in proposer.propose(observations, 4, np.random.default_rng(5)):
            assert proposal.genome["unet"]["variant"] == "none"
            assert proposal.genome["optimizer"]["muon_lr"] == default_genome()["optimizer"]["muon_lr"]


def test_random_proposer_starts_a_track_without_observations() -> None:
    """A fresh track has nothing the Bayes and genetic proposers can use; random draws over the open space fill the
    first batches, distinct, inside the warmup cap, and never a genome already observed."""
    space = search.open_space([])
    proposals = search.RandomProposer(space).propose([], 6, np.random.default_rng(7))
    digests = {search.genome_sha256(proposal.genome) for proposal in proposals}
    assert len(proposals) == len(digests) == 6
    assert all(search.fits_warmup(proposal.genome) for proposal in proposals)
    assert all(proposal.parents == [] for proposal in proposals)
    seen = search.Observation("seen", proposals[0].genome, 2.8)
    again = search.RandomProposer(space).propose([seen], 3, np.random.default_rng(7))
    assert search.genome_sha256(proposals[0].genome) not in {search.genome_sha256(proposal.genome) for proposal in again}
    assert all(proposal.parents == ["seen"] for proposal in again)


def test_the_hybrid_breeds_children_the_gp_ranks_and_every_pick_composes_and_fits() -> None:
    """GA children ranked by the rung-reading GP: distinct, unseen, each resolving (no family conflict) and predicted
    inside the warmup cap and the GPU's memory, over a space whose shape, loop, routing and SGLR are free."""
    space = search.open_space([])
    assert {"shape.blocks", "loop.variant", "routing.variant", "sglr.variant"} <= {dimension.key for dimension in space}
    observations = observed(6, np.random.default_rng(2), space)
    proposals = search.HybridProposer(space, pool=300).propose(observations, 4, np.random.default_rng(3))
    digests = {search.genome_sha256(proposal.genome) for proposal in proposals}
    assert len(proposals) == len(digests) == 4 and not digests & {search.genome_sha256(item.genome) for item in observations}
    for proposal in proposals:
        assert resolve({"families": proposal.genome}) == proposal.genome and search.fits_machine(proposal.genome)
        assert proposal.reason.startswith("GP expected improvement")
    assert search.predicted_peak_gib(default_genome()) == search.RECORD_PEAK_GIB
    wide = search.written(default_genome(), {"shape.hidden": 1536, "shape.intermediate": 6144})
    assert not search.fits_machine(wide) and search.fits_machine(search.written(wide, {"batch.micro_divisor": 4}))
    deep = search.written(default_genome(), {"shape.blocks": 46})  # predicted 84.8 GiB, under the cap but not its margin
    assert search.fits_warmup(deep) and search.predicted_peak_gib(deep) < search.PEAK_CAP_GIB and not search.fits_machine(deep)
    assert search.admissible(lambda: search.written(default_genome(), {"shape.blocks": 20, "loop.variant": "tied"})) is None


def test_a_state_carries_its_track(tmp_path: Path) -> None:
    """A new state names its track and `load_state` applies the track's budgets; the 10-minute track is the default,
    and a state written before the tracks is the 10-minute one."""
    hour = search.load_state(tmp_path / "hour.json", "1h")
    assert hour["track"] == "1h" and search.FULL_BUDGET == 3600 and search.RUNG_BUDGET == 600
    assert search.run_name("h_auto01_random", 1, 600) == "h_auto01_random_01_r600"
    (tmp_path / "old.json").write_text(json.dumps({"schema": search.STATE_SCHEMA, "batches": {}, "proposals": {}, "seeded": {}, "retired": []}), encoding="utf-8")
    old = search.load_state(tmp_path / "old.json")
    assert old["track"] == "10min" and search.FULL_BUDGET == 600 and search.RUNG_BUDGET == 180


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
    assert (tmp_path / "candidates" / "queue" / f"{names[0]}_r180.args").read_text().split() == [
        "--budget", "180", "--eval-cap", "60.0", "--eval-interval", "90.0", "--experiment-group", "2026-10-28_afdb_v6_speedrun",
        "--final-splits", "val_select"]  # the track's clock rides with every queue entry
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


def test_the_loop_ships_a_batch_whose_name_holds_the_rung_letter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The 1-hour track's first batch, h_auto01_random, crashed the loop's name parse on 2026-10-31 ("_r" split)."""
    import autosearch  # noqa: PLC0415  # the loop's driver, a script beside search.py

    monkeypatch.setattr(search, "PROJECT", tmp_path)
    monkeypatch.delenv("SPEEDRUN_LAUNCHER", raising=False)
    monkeypatch.delenv("SPEEDRUN_FETCH", raising=False)
    state = search.load_state(tmp_path / "search.json", track="1h")
    names = search.propose(state, search.RandomProposer(list(search.SPACE)), "h_auto01_random", 2, 0, [600, 3600])
    pending = autosearch.pending_runs(state)
    assert [run for _, run, _ in pending] == [f"{name}_r600" for name in names]
    assert search.queue_runs(state, pending) == [f"{name}_r600" for name in names]
    assert state["batches"]["h_auto01_random"]["shipments"][-1]["names"] == [f"{name}_r600" for name in names]
    assert search.parse_run_name(f"{names[0]}_r600") == search.RunName(names[0], "h_auto01_random", 1, 600)
    assert (tmp_path / "candidates" / "queue" / f"{names[0]}_r600.args").read_text().split()[:2] == ["--budget", "600"]


class TreesSurrogate:
    """A stand-in for the tabular foundation model, so `search.TabularProposer`'s arithmetic runs without a GPU or its package."""

    def quantiles(self, context: pd.DataFrame, values: np.ndarray, query: pd.DataFrame) -> np.ndarray:
        # values: (n,); returns (m, q).
        joined = pd.concat([context, query], ignore_index=True)
        numbers = [column for column in joined if pd.api.types.is_numeric_dtype(joined[column])]
        parts = [joined[numbers].fillna(0.5), pd.get_dummies(joined.drop(columns=numbers).astype(object), dtype=float)]
        matrix = pd.concat(parts, axis=1).to_numpy(dtype=float)  # (n + m, D)
        trees = ExtraTreesRegressor(30, random_state=0).fit(matrix[: len(context)], values)
        per_tree = np.stack([tree.predict(matrix[len(context):]) for tree in trees.estimators_], axis=1)  # (m, T)
        return np.quantile(per_tree, np.arange(1, 200) / 200, axis=1).T  # (m, q)


def test_budget_view_removes_each_budgets_level_and_adds_its_column() -> None:
    search.apply_track({"track": "10min"})
    X, y, level, column = search.budget_view(np.zeros((4, 1)), np.array([2.8, 2.9, 2.7, 2.75]), [180, 180, 600, 600])  # (n, D + 1), (n,)
    assert X[:, 1].tolist() == [0.0, 0.0, 1.0, 1.0] and column == 1.0 and math.isclose(level, 2.725)
    assert np.allclose(y, [-0.05, 0.05, -0.025, 0.025])


def test_every_budget_reads_the_rungs_and_production_does_not() -> None:
    search.apply_track({"track": "10min"})
    state = {"seeded": {}, "proposals": {
        "a_01": {"genome": {}, "results": {"180": {"valid": True, "val_select": 2.78}, "600": {"valid": True, "val_select": 2.72}}},
        "a_02": {"genome": {}, "results": {"180": {"valid": True, "val_select": 2.80}}},
        "a_03": {"genome": {}, "results": {"180": {"valid": False, "val_select": None}}}}}
    assert [(item.name, item.budget) for item in search.observations_of(state)] == [("a_01", None)]
    every = search.observations_at_every_budget(state)
    assert [(item.name, item.budget, item.full) for item in every] == [("a_01", None, True), ("a_01", 180, False), ("a_02", 180, False)]


@pytest.mark.parametrize("arm", ["bayes_rungs", "tabular"])
def test_arms_reading_every_budget_find_what_production_can_on_the_toy(arm: str) -> None:
    """Five proposal seeds, at least three finding the triangle kernel within six rounds, with half of every batch seen only at its rung."""
    search.apply_track({"track": "10min"})
    found = []
    for seed in range(5):
        observations = observed(6, np.random.default_rng(1), TOY_SPACE)
        start = min(toy_loss(observation.genome) for observation in observations)
        rng = np.random.default_rng(seed)
        proposer = search.RungReadingProposer(TOY_SPACE, pool=300) if arm == "bayes_rungs" else search.TabularProposer(TOY_SPACE, TreesSurrogate(), pool=300)
        for round_index in range(6):
            proposals = proposer.propose(observations, 4, rng)
            assert len({search.genome_sha256(proposal.genome) for proposal in proposals}) == 4 and all(proposal.reason for proposal in proposals)
            for index, proposal in enumerate(proposals):
                rung = toy_loss(proposal.genome, rng) + 0.05  # the rung's own level, higher than the full budget's
                observations.append(search.Observation(f"round{round_index}_{index}", proposal.genome, rung, 180))
                if index % 2 == 0:
                    observations.append(search.Observation(f"round{round_index}_{index}", proposal.genome, toy_loss(proposal.genome, rng)))
        found.append(min(toy_loss(observation.genome) for observation in observations) < start - 0.005)
    assert sum(found) >= 3, found


def test_a_state_names_its_own_arms_and_rungs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import autosearch  # noqa: PLC0415  # the loop's driver, a script beside search.py

    monkeypatch.setattr(search, "PROJECT", tmp_path)
    state = search.load_state(tmp_path / "search.json", track="10min")
    assert autosearch.next_batch_name(state) == ("auto01_hybrid", "hybrid")  # the loop's default since 2026-11-23
    state["proposers"], state["rungs"] = ["bayes_rungs", "bayes"], [180, 600]
    assert autosearch.next_batch_name(state) == ("auto01_bayes_rungs", "bayes_rungs")
    names = autosearch.propose_new(state, 2, 0)  # no observations yet, so the random draws start it, named as what they were
    assert names[0].startswith("auto01_random") and state["batches"]["auto01_random"]["rungs"] == [180, 600]
    assert autosearch.next_batch_name(state) == ("auto02_bayes", "bayes")
    assert isinstance(autosearch.proposer_of("bayes_rungs", list(search.SPACE)), search.RungReadingProposer)


def test_a_state_without_proposers_runs_what_is_pending_and_nothing_more(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import autosearch  # noqa: PLC0415  # the loop's driver, a script beside search.py

    monkeypatch.setattr(search, "PROJECT", tmp_path)
    state = search.load_state(tmp_path / "search.json", track="10min")
    state["proposers"] = []
    autosearch.command_next(state, tmp_path, 4)
    assert state["batches"] == {} and (tmp_path / "queue_next.txt").read_text(encoding="utf-8").strip() == ""
    state["proposers"] = ["random"]
    autosearch.command_next(state, tmp_path, 2)
    assert len((tmp_path / "queue_next.txt").read_text(encoding="utf-8").split()) == 2


def test_unproven_keys_name_what_no_valid_result_held() -> None:
    search.apply_track({"track": "10min"})
    base = default_genome()
    document = lambda genome: search.genome_document(genome)["families"]  # noqa: E731
    state = {"seeded": {}, "proposals": {
        "a_01": {"genome": document(base), "results": {"600": {"valid": True, "val_select": 2.72}}},
        "a_02": {"genome": document(search.written(base, {"optimizer.muon_lr": 0.03})), "results": {"180": {"valid": True, "val_select": 2.78}}},
        "a_03": {"genome": document(search.written(base, {"attention.block_size": 256})), "results": {"600": {"valid": False, "val_select": None}}}}}
    assert search.unproven_keys(state, base) == []
    assert search.unproven_keys(state, search.written(base, {"optimizer.muon_lr": 0.02})) == []  # a number in a column a result varied
    assert search.unproven_keys(state, search.written(base, {"attention.heads": 8})) == ["attention.heads=8"]
    assert search.unproven_keys(state, search.written(base, {"schedule.cooldown_power": 1.5})) == ["schedule.cooldown_power"]
    assert search.unproven_keys(state, search.written(base, {"attention.block_size": 256})) == ["attention.block_size=256"]  # a crash proves nothing


def test_a_failed_preflight_trims_the_pick_and_keeps_the_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import autosearch  # noqa: PLC0415  # the loop's driver, a script beside search.py

    monkeypatch.setattr(search, "PROJECT", tmp_path)
    monkeypatch.setattr(autosearch, "PREFLIGHT_ROOT", tmp_path / "preflight")
    state = search.load_state(tmp_path / "search.json", track="10min")
    state["proposers"] = ["random"]
    queue = tmp_path / "queue_next.txt"

    def next_run() -> str:
        autosearch.command_next(state, tmp_path, 1)
        run = queue.read_text(encoding="utf-8").strip()
        assert (tmp_path / "preflight_next.txt").read_text(encoding="utf-8").strip() == run  # no result has proven anything yet
        return run

    def preflight(run: str, warmup: float) -> None:
        (tmp_path / "preflight" / run).mkdir(parents=True)
        (tmp_path / "preflight" / run / "result.json").write_text(json.dumps({"clock": {"warmup_seconds": warmup, "warmup_cap_seconds": 180.0}}), encoding="utf-8")

    run = next_run()
    name = search.parse_run_name(run).proposal
    (tmp_path / "preflight.log").write_text("", encoding="utf-8")
    preflight(run, 150.0)
    autosearch.command_trim(state, tmp_path, run, 0)
    assert queue.read_text(encoding="utf-8").strip() == run and name in state["proposals"] and "trimmed" not in state and not (tmp_path / "preflight" / run).exists()

    preflight(run, 186.0)  # the real run's warm compile cache would hide this
    autosearch.command_trim(state, tmp_path, run, 0)
    assert queue.read_text(encoding="utf-8").strip() == "" and name not in state["proposals"] and state["batches"] == {}
    assert autosearch.next_batch_name(state)[0] == name.rsplit("_", 1)[0]  # the turn stays
    assert "over the 180 s cap" in state["trimmed"][0]["why"] and set(state["trimmed"][0]["retired"]) <= set(state["retired"]) and state["trimmed"][0]["retired"]

    run = next_run()
    (tmp_path / "preflight.log").write_text("Traceback\ntorch.compile: ValueError: Q and KV block size\nend\n", encoding="utf-8")
    autosearch.command_trim(state, tmp_path, run, 1)
    assert "exit code 1: torch.compile: ValueError" in state["trimmed"][1]["why"] and queue.read_text(encoding="utf-8").strip() == ""
    assert state["batches"] == {} and len(search.open_space(state["retired"])) > 0


def test_a_crashed_run_keeps_its_turn_taken_and_retires_what_it_alone_held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import autosearch  # noqa: PLC0415  # the loop's driver, a script beside search.py

    monkeypatch.setattr(search, "PROJECT", tmp_path)
    state = search.load_state(tmp_path / "search.json", track="10min")
    state |= {"proposers": ["random"], "rungs": [600]}
    autosearch.command_next(state, tmp_path, 1)
    run = (tmp_path / "queue_next.txt").read_text(encoding="utf-8").strip()
    autosearch.command_collect(state, tmp_path, [run])
    assert state["proposals"][run]["results"]["600"]["valid"] is False and state["trimmed"][0]["kind"] == "crash"
    retired = state["trimmed"][0]["retired"]
    assert retired and set(retired) <= set(state["retired"])
    space = search.open_space(state["retired"])
    genome = search.resolve({"families": state["proposals"][run]["genome"]})
    assert not any(dimension.categorical and search.read(genome, dimension.key) in dimension.choices and f"{dimension.key}={search.read(genome, dimension.key)}" in retired for dimension in space)


def test_a_state_names_its_own_wandb_group_and_the_track_keeps_its_own() -> None:
    search.apply_track({"track": "10min", "group": "2026-11-03_surrogate_head_to_head"})
    assert search.TRACK["group"] == "2026-11-03_surrogate_head_to_head" and search.TRACK["budget"] == 600
    assert search.TRACKS["10min"]["group"] == "2026-10-28_afdb_v6_speedrun"
    search.apply_track({"track": "10min"})
    assert search.TRACK["group"] == "2026-10-28_afdb_v6_speedrun"


def test_a_state_seeded_from_a_track_names_its_batches_apart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import autosearch  # noqa: PLC0415  # the loop's driver, a script beside search.py

    monkeypatch.setattr(search, "PROJECT", tmp_path)
    state = search.load_state(tmp_path / "search.json", track="1h")
    state["batches"]["h_auto12_genetic"] = {"created_utc": "2026-11-02T00:00:00+00:00"}  # the seed state's last batch
    assert autosearch.next_batch_name(state) == ("h_auto02_hybrid", "hybrid")
    state["prefix"] = "fa_"
    search.apply_track(state)
    assert autosearch.next_batch_name(state) == ("fa_auto01_hybrid", "hybrid") and search.TRACK["budget"] == 3600
    search.apply_track({"track": "10min"})
    assert search.TRACK["prefix"] == ""


def test_a_state_names_the_final_splits_of_its_full_runs_and_a_rung_keeps_its_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search, "PROJECT", tmp_path)
    search.apply_track({"track": "10min", "final_splits": ["val_select", "test"]})
    search.write_queue_entry("full", default_genome(), "why", 600)
    search.write_queue_entry("rung", default_genome(), "why", 180)
    search.apply_track({"track": "10min"})
    search.write_queue_entry("every_split", default_genome(), "why", 600)
    queue = tmp_path / "candidates" / "queue"
    assert (queue / "full.args").read_text().split()[-3:] == ["--final-splits", "val_select", "test"]
    assert (queue / "rung.args").read_text().split()[-2:] == ["--final-splits", "val_select"]
    assert "--final-splits" not in (queue / "every_split.args").read_text()


def test_novel_rows_find_what_no_context_row_holds() -> None:
    context = pd.DataFrame({"g:constant": ["a", "a"], "g:varied": ["b", "c"], "g:still": [0.5, 0.5], "g:moved": [0.1, 0.2], "g:inert": [math.nan, math.nan]})
    query = pd.DataFrame({"g:constant": ["a", "new", "a", "a", "a", "a"], "g:varied": ["b", "b", "new", "c", "b", "b"], "g:still": [0.5, 0.5, 0.5, 0.9, 0.5, 0.5],
                          "g:moved": [0.15, 0.1, 0.1, 0.1, 9.0, 0.1], "g:inert": [math.nan, math.nan, math.nan, math.nan, math.nan, 0.3]})
    assert search.novel_rows(context, query).tolist() == [False, True, True, True, False, True]  # a number in a column it varied is not unseen


class FlatSurrogate:
    """Every genome the same, with the seat's quantile shape: only the arm's rules can tell them apart."""

    def quantiles(self, context: pd.DataFrame, values: np.ndarray, query: pd.DataFrame) -> np.ndarray:
        # values: (n,); returns (m, q).
        return np.tile(np.linspace(values.min() - 0.01, values.min() + 0.01, 199), (len(query), 1))  # (m, q)


def test_every_fourth_tabular_pick_holds_what_no_result_holds() -> None:
    search.apply_track({"track": "10min"})
    observations = observed(6, np.random.default_rng(1), TOY_SPACE)
    assert {observation.genome["attention"]["variant"] for observation in observations} == {"standard"}
    proposals = search.TabularProposer(TOY_SPACE, FlatSurrogate(), pool=200).propose(observations, 4, np.random.default_rng(0))
    assert proposals[3].reason.startswith("unseen pick") and not any(proposal.reason.startswith("unseen pick") for proposal in proposals[:3])
    earlier = [observation.genome for observation in observations] + [proposal.genome for proposal in proposals[:3]]
    context = search.genome_frame(earlier, [search.FULL_BUDGET] * len(earlier), TOY_SPACE)
    assert search.novel_rows(context, search.genome_frame([proposals[3].genome], [search.FULL_BUDGET], TOY_SPACE)).tolist() == [True]
    for index in range(3):  # what the model cannot see never rides along on a pick it chose
        earlier = [observation.genome for observation in observations] + [proposal.genome for proposal in proposals[:index]]
        context = search.genome_frame(earlier, [search.FULL_BUDGET] * len(earlier), TOY_SPACE)
        assert search.novel_rows(context, search.genome_frame([proposals[index].genome], [search.FULL_BUDGET], TOY_SPACE)).tolist() == [False]


def test_the_unseen_pick_counts_across_batches_of_one() -> None:
    search.apply_track({"track": "10min"})
    observations = observed(6, np.random.default_rng(1), TOY_SPACE)
    unseen = [search.TabularProposer(TOY_SPACE, FlatSurrogate(), pool=200, picks_made=made).propose(observations, 1, np.random.default_rng(0))[0].reason.startswith("unseen pick")
              for made in range(8)]
    assert unseen == [False, False, False, True, False, False, False, True]


def test_only_the_tabular_arm_reads_other_tracks_and_they_never_lead_it() -> None:
    search.apply_track({"track": "10min"})
    record = search.genome_document(default_genome())["families"]
    state = {"seeded": {}, "other_track_rows": [{"name": "h_a@600", "genome": record, "budget": 600, "val_select": 2.60}],
             "proposals": {f"a_0{index}": {"genome": search.genome_document(search.written(default_genome(), {"optimizer.muon_lr": 0.02 + 0.01 * index}))["families"],
                                           "results": {"600": {"valid": True, "val_select": 2.72 + 0.01 * index}}} for index in range(3)}}
    tabular = search.TabularProposer(TOY_SPACE, FlatSurrogate(), pool=50)
    assert len(search.observations_read_by(tabular, state)) == 4
    for proposer in (search.BayesianProposer(TOY_SPACE), search.RungReadingProposer(TOY_SPACE)):
        assert not any(item.other_track for item in search.observations_read_by(proposer, state))
    other = search.observations_on_other_tracks(state)[0]
    assert other.budget == 600 and not other.full
    proposal = tabular.propose(search.observations_read_by(tabular, state), 1, np.random.default_rng(0))[0]
    assert proposal.parents == ["a_00"] and "best observed 2.7200" in proposal.reason


@pytest.mark.parametrize(("argv", "expected"), [
    (["propose", "--proposer", "screening", "--batch", "s1", "--factors", "unet", "gates"],
     {"command": "propose", "proposer": "screening", "batch": "s1", "count": 8, "seed": 0, "factors": ["unet", "gates"], "genomes": [],
      "hypotheses": [], "rungs": None}),
    (["propose", "--proposer", "random", "--batch", "r1", "--count", "3", "--seed", "2", "--rungs", "180", "600"],
     {"proposer": "random", "count": 3, "seed": 2, "factors": [], "rungs": [180, 600]}),
    (["propose", "--proposer", "manual", "--batch", "m1", "--genomes", "a.json", "b.json", "--hypotheses", "one", "two"],
     {"proposer": "manual", "genomes": [Path("a.json"), Path("b.json")], "hypotheses": ["one", "two"]}),
    (["ship", "--batch", "a", "b"], {"command": "ship", "batch": ["a", "b"], "budget": None}),
    (["ship", "--batch", "a", "--budget", "180"], {"batch": ["a"], "budget": 180}),
    (["collect", "--batch", "a", "--results-dir", "r", "--ledger", "l.jsonl", "--crashed", "a_01", "--crash-note", "oom"],
     {"command": "collect", "batch": "a", "results_dir": Path("r"), "ledger": Path("l.jsonl"), "crashed": ["a_01"], "crash_note": "oom"}),
    (["collect", "--batch", "a", "--results-dir", "r", "--ledger", "l.jsonl"], {"crashed": [], "crash_note": ""}),
    (["seed", "--name", "n", "--genome", "g.json", "--val-select", "2.73"], {"command": "seed", "name": "n", "genome": Path("g.json"), "val_select": 2.73}),
    (["retire", "unet.variant=mirrored", "optimizer.muon_lr"], {"command": "retire", "keys": ["unet.variant=mirrored", "optimizer.muon_lr"]}),
    (["effects", "--batch", "s1"], {"command": "effects", "batch": "s1", "budget": None}),
    (["effects", "--batch", "s1", "--budget", "180"], {"budget": 180}),
    (["status"], {"command": "status"}),
])
def test_search_command_line_parses_each_subcommand(argv: list[str], expected: dict[str, Any]) -> None:
    arguments = search.parse_arguments(["--state", "s.json", *argv])
    assert arguments.state == Path("s.json") and arguments.track == "1h"  # the headline track since 2026-11-23
    assert {key: getattr(arguments, key) for key in expected} == expected


def test_search_command_line_refuses_what_it_does_not_know() -> None:
    for argv in (["status"], ["--state", "s.json"], ["--state", "s.json", "propose", "--proposer", "nope", "--batch", "b"],
                 ["--state", "s.json", "--track", "2h", "status"]):
        with pytest.raises(SystemExit):
            search.parse_arguments(argv)
    assert search.parse_arguments(["--state", "s.json", "--track", "1h", "status"]).track == "1h"


@pytest.mark.parametrize(("argv", "expected"), [
    (["next"], {"command": "next", "state_dir": Path("/state/search"), "count": 4, "crashed": [], "run": None, "exit_code": 0}),
    (["next", "--state-dir", "s", "--count", "2"], {"state_dir": Path("s"), "count": 2}),
    (["collect", "--crashed", "a_01", "a_02_r180"], {"command": "collect", "crashed": ["a_01", "a_02_r180"]}),
    (["trim", "--run", "a_01", "--exit-code", "1"], {"command": "trim", "run": "a_01", "exit_code": 1}),
])
def test_autosearch_command_line_parses_each_command(argv: list[str], expected: dict[str, Any]) -> None:
    import autosearch  # noqa: PLC0415  # the loop's driver, a script beside search.py

    arguments = autosearch.parse_arguments(argv)
    assert {key: getattr(arguments, key) for key in expected} == expected
    with pytest.raises(SystemExit):
        autosearch.parse_arguments(["propose"])


def test_one_factory_names_every_proposer() -> None:
    import autosearch  # noqa: PLC0415  # the loop's driver, a script beside search.py

    space = list(search.SPACE)
    assert autosearch.proposer_of is search.proposer_of
    for name, kind in (("bayes", search.BayesianProposer), ("bayes_rungs", search.RungReadingProposer),
                       ("genetic", search.GeneticProposer), ("random", search.RandomProposer)):
        proposer = search.proposer_of(name, space)
        assert type(proposer) is kind and proposer.name == name
    assert isinstance(search.proposer_of("screening", space, factors=["unet", "gates"]), search.ScreeningProposer)
    assert isinstance(search.proposer_of("manual", space, genomes=[Path("g.json")], hypotheses=["h"]), search.ManualProposer)
    with pytest.raises(ValueError, match="screen 2 to 11"):
        search.proposer_of("screening", space)
    with pytest.raises(KeyError):
        search.proposer_of("nope", space)
    assert set(search.PROPOSER_NAMES) == {*search.SPACE_PROPOSERS, "kumo", "screening", "manual"}


def test_run_names_read_back_and_a_finished_batch_ships_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    search.apply_track({"track": "10min"})
    assert search.parse_run_name("screen1_05") == search.RunName("screen1_05", "screen1", 5, 600)
    assert search.parse_run_name("h_auto01_random_12_r180") == search.RunName("h_auto01_random_12", "h_auto01_random", 12, 180)
    with pytest.raises(ValueError, match="not a search run name"):
        search.parse_run_name("no_index")
    monkeypatch.setattr(search, "PROJECT", tmp_path)
    monkeypatch.delenv("SPEEDRUN_LAUNCHER", raising=False)
    state = search.load_state(tmp_path / "search.json")
    names = search.propose(state, search.RandomProposer(list(search.SPACE)), "r1", 2, 0, [600])
    for name in names:
        state["proposals"][name]["results"]["600"] = {"run": name, "val_select": 2.75, "test": 2.74, "steps": 1, "warmup_seconds": 1.0, "valid": True}
    assert search.next_rung(state, "r1") is None
    with pytest.raises(ValueError, match="no run left"):
        search.ship(state, ["r1"])
    assert search.ship(state, ["r1"], 600) == [] and state["batches"]["r1"]["shipments"][-1]["names"] == []  # an empty shipment is still recorded


def test_fetch_copies_a_result_through_the_command_and_keeps_one_it_has(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, destination = tmp_path / "runs", tmp_path / "results"
    source.mkdir()
    destination.mkdir()
    (source / "a_01.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("SPEEDRUN_FETCH", raising=False)
    assert search.fetch("a_01", destination) is None
    python = Path(sys.executable).as_posix()
    monkeypatch.setenv("SPEEDRUN_FETCH", f"{python} -c 'import shutil, sys; shutil.copy(sys.argv[1], sys.argv[2])' {source.as_posix()}/{{name}}.json {{destination}}")
    assert search.fetch("a_01", destination) == destination / "a_01.json" and (destination / "a_01.json").read_text(encoding="utf-8") == "{}"
    assert search.fetch("a_02", destination) is None  # the command fails and the run stays absent
    monkeypatch.setenv("SPEEDRUN_FETCH", "a-command-that-must-not-run")
    assert search.fetch("a_01", destination) == destination / "a_01.json"  # a result already fetched is not fetched again


def test_effects_table_reads_a_screening_batch(tmp_path: Path) -> None:
    """The planted landscape without noise: the triangle, the gates and U-net are the active effects, the rest zero."""
    search.apply_track({"track": "10min"})
    state = search.load_state(tmp_path / "search.json")
    factors = ["unet", "gates", "triangle", "ema", "muon"]
    names = search.propose(state, search.ScreeningProposer(factors), "screen1", 12, 0, [180, 600])
    for name in names:
        loss = toy_loss(resolve({"families": state["proposals"][name]["genome"]}))
        state["proposals"][name]["results"]["600"] = {"run": name, "val_select": loss, "test": loss - 0.003, "steps": 1, "warmup_seconds": 1.0, "valid": True}
        state["proposals"][name]["results"]["180"] = {"run": f"{name}_r180", "val_select": loss + 0.05, "test": None, "steps": 1, "warmup_seconds": 1.0, "valid": True}
    table = search.effects_table(state, "screen1").splitlines()
    assert table[0].startswith("Main effects of switching each module on, 12 runs") and "standard error of 0.0009" in table[0]
    rows = {line.split(" | ")[0].lstrip("| "): line for line in table if line.startswith("| ") and "Switch" not in line}
    assert list(rows)[:2] == ["triangle", "gates"] and list(rows)[-1] == "unet" and set(rows) == set(factors)  # the most helpful first
    assert rows["triangle"] == "| triangle | -0.0100 | -0.0100 | yes |" and rows["unet"] == "| unet | +0.0060 | +0.0060 | yes |"
    assert rows["muon"].endswith("| no |")
    assert "Leading two-factor interactions" in "\n".join(table)  # 8 degrees of freedom left after three active effects
    rung = search.effects_table(state, "screen1", 180).splitlines()
    assert "| triangle | -0.0100 | +nan | yes |" in rung  # the short rung has no test loss


def test_round9_switches_resolve_alone_and_the_record_pays_no_new_compile_cost() -> None:
    search.apply_track({"track": "10min"})
    assert search.compile_costs(default_genome()) == []
    record = search.written(default_genome(), {"value_residual.variant": "first_block"})
    for factor in ("heads_8", "heads_4", "qk_75", "tie_kv", "exclusive", "mud1", "mud2", "polar_3", "muown", "radial_brake", "hyperball",
                   "contra_to_soft", "variance_scale", "mona", "lookahead", "mlp_lr_half", "cooldown_power", "anchor_norm", "value_per_head",
                   "qk_anchor", "composition", "protein_mean", "substitution_targets", "head_bias"):
        genome = search.switched([factor, "unet"], np.array([1, -1]), record)
        assert genome != record and search.fits_warmup(genome), factor
    for factor in ("canon", "momentum_stream"):  # the momentum stream's measured 61 s, and Canon's estimate scaled by the measured ones, leave no room
        genome = search.switched([factor, "unet"], np.array([1, -1]), record)
        assert genome != record and not search.fits_warmup(genome), factor
