"""The search's own driver on the box: propose a batch, hand it to the queue, read the results back; fixed.

`autosearch.sh` runs this inside the image between queues, so the GPU never waits for a laptop. The search state and
the ledger live on the box (`/state/search/search.json` and `runs.jsonl`, seeded once from the workspace copies, then
authoritative until the loop stops) and every run's result is read straight from `/state/runs/<name>/result.json`.

    python autosearch.py next --state-dir /state/search --count 4
        ship up to `count` runs still to do, newest batch first (a pilot of new modules proposed at a restart runs
        before the backlog), else propose a new batch: the Bayes and genetic proposers alternate, straight at the full
        budget; writes the run names to `queue_next.txt` for the shell loop
    python autosearch.py collect --state-dir /state/search [--crashed name ...]
        read every finished run of every batch into the state and the ledger

Everything the benchmark leaves free is theirs to optimize (Logan, 2026-10-30): the two proposers draw from the whole
open space, and a retired variant stays out. A new strategy module needs a new shipment of the tree; the loop that
ships with it takes over from the running one at its next cycle.
"""

from __future__ import annotations

import argparse
import json
import os

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import search

BATCH_PREFIX = "auto"
PROPOSERS = ("bayes", "genetic")  # alternating, batch by batch


def pending_runs(state: dict[str, Any]) -> list[tuple[str, str, int]]:
    """(batch, run, rung) for every promoted run without a result at its batch's next rung, newest batch first."""
    pending = []
    newest_first = sorted(state["batches"], key=lambda batch: state["batches"][batch]["created_utc"], reverse=True)
    for batch in newest_first:
        entry = state["batches"][batch]
        try:
            rung = search.next_rung(state, batch)
        except StopIteration:
            continue
        for name in entry["names"]:
            proposal = state["proposals"][name]
            if str(rung) in proposal["results"] or (rung == search.FULL_BUDGET and not proposal["promoted"]):
                continue
            pending.append((batch, search.run_name(batch, int(name.rsplit("_", 1)[1]), rung), rung))
    return pending


def ship_runs(state: dict[str, Any], selected: list[tuple[str, str, int]]) -> list[str]:
    """Queue entries for the selected runs, recorded as one shipment per batch; returns the run names in order."""
    names = []
    for batch, run, rung in selected:
        name = search.run_name(batch, int(run.split("_r")[0].rsplit("_", 1)[1]), search.FULL_BUDGET)
        proposal = state["proposals"][name]
        hypothesis = f"{proposal['proposer']} proposal ({rung} s): {proposal['reason']}"
        search.write_queue_entry(run, search.resolve({"families": proposal["genome"]}), hypothesis, rung)
        names.append(run)
    for batch in dict.fromkeys(batch for batch, _, _ in selected):
        shipped = [run for other, run, _ in selected if other == batch]
        rung = next(rung for other, _, rung in selected if other == batch)
        state["batches"][batch]["shipments"].append({"budget": rung, "names": shipped, "utc": datetime.now(UTC).isoformat()})
    return names


def next_batch_name(state: dict[str, Any]) -> tuple[str, str]:
    """`auto<k>_<proposer>`, the proposers alternating by k."""
    index = 1 + sum(batch.startswith(BATCH_PREFIX) for batch in state["batches"])
    proposer = PROPOSERS[(index - 1) % len(PROPOSERS)]
    return f"{BATCH_PREFIX}{index:02d}_{proposer}", proposer


def propose_new(state: dict[str, Any], count: int, seed: int) -> list[str]:
    """One batch from the next proposer in turn, the other proposer when it cannot propose."""
    space = search.open_space(state["retired"])
    batch, proposer_name = next_batch_name(state)
    for attempt in (proposer_name, *[other for other in PROPOSERS if other != proposer_name]):
        proposer = search.BayesianProposer(space) if attempt == "bayes" else search.GeneticProposer(space)
        try:
            return search.propose(state, proposer, batch.replace(proposer_name, attempt), count, seed, [search.FULL_BUDGET])
        except ValueError as error:
            print(f"autosearch: {attempt} could not propose: {error}")
    raise RuntimeError("neither proposer could propose a batch")


def command_next(state: dict[str, Any], state_dir: Path, count: int) -> None:
    selected = pending_runs(state)[:count]
    if not selected:
        seed = len(state["batches"]) + 1  # a fresh draw per batch, reproducible from the state
        propose_new(state, count, seed)
        selected = pending_runs(state)[:count]
    names = ship_runs(state, selected)
    (state_dir / "queue_next.txt").write_text(" ".join(names) + "\n", encoding="utf-8")
    print(f"autosearch: next {' '.join(names)}")


def command_collect(state: dict[str, Any], state_dir: Path, crashed: list[str]) -> None:
    os.environ["SPEEDRUN_FETCH"] = "cp /state/runs/{name}/result.json {destination}"
    results = state_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    for run in crashed:
        search.mark_crashed(state, run, "no result.json after the queue finished; see the queue log")
    for batch in state["batches"]:
        search.collect(state, batch, results, state_dir / "runs.jsonl")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("next", "collect"))
    parser.add_argument("--state-dir", type=Path, default=Path("/state/search"))
    parser.add_argument("--count", type=int, default=4, help="runs per cycle, and per proposed batch")
    parser.add_argument("--crashed", nargs="*", default=[], help="collect: runs that left no result.json")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    state_path = arguments.state_dir / "search.json"
    if not state_path.exists():
        raise SystemExit(f"{state_path} is missing: seed it from the workspace's search.json first")
    state = search.load_state(state_path)
    if arguments.command == "next":
        command_next(state, arguments.state_dir, arguments.count)
    else:
        command_collect(state, arguments.state_dir, arguments.crashed)
    observations = search.observations_of(state)
    best = min(observations, key=lambda observation: observation.value)
    print(json.dumps({"utc": datetime.now(UTC).isoformat(timespec="seconds"), "observations": len(observations),
                      "incumbent": best.name, "val_select": round(best.value, 4), "batches": len(state["batches"]),
                      "pending": len(pending_runs(state))}))
    search.save_state(state, state_path)


if __name__ == "__main__":
    main()
