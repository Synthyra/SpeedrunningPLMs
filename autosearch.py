"""The search's own driver on the box: propose a batch, hand it to the queue, read the results back; fixed.

`autosearch.sh` runs this inside the image between queues, so the GPU never waits for a laptop. The search state and
the ledger live on the box (`/state/search/search.json` and `runs.jsonl`, seeded once from the workspace copies, then
authoritative until the loop stops) and every run's result is read straight from `/state/runs/<name>/result.json`.

    python autosearch.py next --state-dir /state/search --count 4
        ship up to `count` runs still to do, newest batch first (a pilot of new modules proposed at a restart runs
        before the backlog), else propose a new batch: the Bayes and genetic proposers alternate, straight at the full
        budget; writes the run names to `queue_next.txt` for the shell loop
    python autosearch.py collect --state-dir /state/search [--crashed name ...]
        read every finished run of every batch into the state and the ledger; what a crashed run alone held is retired
    python autosearch.py trim --state-dir /state/search --run name --exit-code N
        a preflight's verdict: a pick that crashed, or compiled over the warmup cap, leaves the queue and the state, what it alone held
        is retired, and its proposer's turn stays, so it proposes again

Everything the benchmark leaves free is theirs to optimize (Logan, 2026-10-30): the two proposers draw from the whole
open space, and a retired variant stays out. A new strategy module needs a new shipment of the tree; the loop that
ships with it takes over from the running one at its next cycle.

A state may name its own `proposers`, taking turns batch by batch on one shared state, and its own `rungs` for a batch; with an empty list
of proposers the loop runs what is pending and ends. The head-to-head of surrogates
(`experiments/representation/speedrunning_plms/2026-11-03_surrogate_head_to_head/`) runs that way, in a state folder of its own.

Runs that will crash are trimmed automatically (Logan, 2026-11-04). A crashed run is no observation, so a proposer cannot learn from it: both arms of
the head-to-head picked `attention.block_size` 64 in one pair, which cannot compile. So a pick holding what no valid result has held
(`search.unproven_keys`) compiles and takes two steps first, between timed runs, in `autosearch.sh`; the real run then warms up from the compile cache.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import search

from datetime import UTC, datetime
from pathlib import Path


BATCH_PREFIX = "auto"
# The genetic algorithm's children ranked by the rung-reading GP (`search.HybridProposer`, 2026-11-23); random draws until
# it has three full-budget observations. A state names its own `proposers` to alternate others.
PROPOSERS = ("hybrid",)
PREFLIGHT_ROOT = Path("/state/preflight")
proposer_of = search.proposer_of  # the one factory; the head-to-head's scripts read it here


def pending_runs(state: search.SearchState) -> list[search.DueRun]:
    """Every promoted run without a result at its batch's next rung, newest batch first."""
    pending = []
    newest_first = sorted(state["batches"], key=lambda batch: state["batches"][batch]["created_utc"], reverse=True)
    for batch in newest_first:
        rung = search.next_rung(state, batch)
        if rung is not None:
            pending += search.due_runs(state, batch, rung)
    return pending


def proposers_of(state: search.SearchState) -> tuple[str, ...]:
    """The proposers taking turns: the state's own, or the two the loop has always alternated."""
    return tuple(state.get("proposers", PROPOSERS))


def next_batch_name(state: search.SearchState) -> tuple[str, str]:
    """`auto<k>_<proposer>`, the proposers taking turns by k."""
    prefix = search.TRACK["prefix"] + BATCH_PREFIX
    index = 1 + sum(batch.startswith(prefix) for batch in state["batches"])
    proposers = proposers_of(state)
    proposer = proposers[(index - 1) % len(proposers)]
    return f"{prefix}{index:02d}_{proposer}", proposer


def propose_new(state: search.SearchState, count: int, seed: int) -> list[str]:
    """One batch from the next proposer in turn, the others when it cannot propose, and random draws while none has the
    observations it needs (a track's start). The batch's name says which proposer it really was."""
    space = search.open_space(state["retired"])
    batch, proposer_name = next_batch_name(state)
    for attempt in (proposer_name, *[other for other in proposers_of(state) if other != proposer_name], "random"):
        picks_made = sum(proposal["proposer"] == attempt for proposal in state["proposals"].values())
        try:
            return search.propose(state, proposer_of(attempt, space, picks_made), batch.replace(proposer_name, attempt), count, seed,
                                  state.get("rungs", search.TRACK["rungs"]))
        except ValueError as error:
            print(f"autosearch: {attempt} could not propose: {error}")
    raise RuntimeError("no proposer could propose a batch")


def genome_of_run(state: search.SearchState, run: str) -> search.Genome:
    return search.resolve({"families": state["proposals"][search.parse_run_name(run).proposal]["genome"]})


def retire_what_it_held(state: search.SearchState, run: str, kind: str, why: str) -> None:
    """Retire what a failed run alone held, so no proposer repeats it, and keep the run and the reason. An innocent choice held beside the culprit goes too."""
    keys = search.unproven_keys(state, genome_of_run(state, run))
    proposal = state["proposals"][search.parse_run_name(run).proposal]
    state["retired"] = sorted(set(state["retired"]) | set(keys))
    state.setdefault("trimmed", []).append({"run": run, "kind": kind, "why": why, "retired": keys, "genome": proposal["genome"], "reason": proposal["reason"],
                                            "utc": datetime.now(UTC).isoformat()})
    print(f"autosearch: {kind} trims {run}: {why}; retired {keys}")


def command_next(state: search.SearchState, state_dir: Path, count: int) -> None:
    selected = pending_runs(state)[:count]
    if not selected and proposers_of(state):
        seed = len(state["batches"]) + 1 + len(state.get("trimmed", []))  # a fresh draw per batch and per trimmed pick, reproducible from the state
        propose_new(state, count, seed)
        selected = pending_runs(state)[:count]
    names = search.queue_runs(state, selected)  # none when nothing is pending and the state names no proposer: the loop ends
    unproven = [run for run in names if search.unproven_keys(state, genome_of_run(state, run))]
    (state_dir / "queue_next.txt").write_text(" ".join(names) + "\n", encoding="utf-8")
    (state_dir / "preflight_next.txt").write_text(" ".join(unproven) + "\n", encoding="utf-8")
    print(f"autosearch: next {' '.join(names)}" + (f"; preflight first: {' '.join(unproven)}" if unproven else ""))


def preflight_failure(run: str, exit_code: int, log: Path) -> str | None:
    """Why a preflight failed, or None: a crash, or a cold compile over the warmup cap, which the real run's warm compile cache would hide."""
    receipt = PREFLIGHT_ROOT / run / "result.json"
    if exit_code or not receipt.exists():
        errors = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if "Error" in line]
        return f"exit code {exit_code}: {errors[-1][:200] if errors else 'no error line in the log'}"
    clock = json.loads(receipt.read_text(encoding="utf-8"))["clock"]
    if clock["warmup_seconds"] > clock["warmup_cap_seconds"]:
        return f"compiled in {clock['warmup_seconds']:.0f} s, over the {clock['warmup_cap_seconds']:.0f} s cap"
    return None


def command_trim(state: search.SearchState, state_dir: Path, run: str, exit_code: int) -> None:
    why = preflight_failure(run, exit_code, state_dir / "preflight.log")
    shutil.rmtree(PREFLIGHT_ROOT / run, ignore_errors=True)
    if why is None:
        print(f"autosearch: preflight passes {run}")
        return
    retire_what_it_held(state, run, "preflight", why)
    drop_pick(state, state_dir, run)


def drop_pick(state: search.SearchState, state_dir: Path, run: str) -> None:
    """Take a trimmed run's proposal out of the state and its run out of the queue; an emptied batch goes too."""
    name = search.parse_run_name(run).proposal
    batch = state["proposals"].pop(name)["batch"]
    state["batches"][batch]["names"].remove(name)
    if not state["batches"][batch]["names"]:
        del state["batches"][batch]  # the turn stays: `next_batch_name` counts this proposer's batch again
    queue = state_dir / "queue_next.txt"
    queue.write_text(" ".join(other for other in queue.read_text(encoding="utf-8").split() if other != run) + "\n", encoding="utf-8")


def command_collect(state: search.SearchState, state_dir: Path, crashed: list[str]) -> None:
    os.environ["SPEEDRUN_FETCH"] = "cp /state/runs/{name}/result.json {destination}"
    results = state_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    for run in crashed:
        search.mark_crashed(state, run, "no result.json after the queue finished; see the queue log")
        retire_what_it_held(state, run, "crash", "no result.json after the queue finished; see the queue log")
    for batch in state["batches"]:
        search.collect(state, batch, results, state_dir / "runs.jsonl")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line; `argv` None reads the process's arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("next", "collect", "trim"))
    parser.add_argument("--state-dir", type=Path, default=Path("/state/search"))
    parser.add_argument("--count", type=int, default=4, help="runs per cycle, and per proposed batch")
    parser.add_argument("--crashed", nargs="*", default=[], help="collect: runs that left no result.json")
    parser.add_argument("--run", help="trim: the run a preflight tried")
    parser.add_argument("--exit-code", type=int, default=0, help="trim: the preflight's")
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_arguments()
    state_path = arguments.state_dir / "search.json"
    if not state_path.exists():
        raise SystemExit(f"{state_path} is missing: seed it from the workspace's search.json first")
    state = search.load_state(state_path)
    if arguments.command == "next":
        command_next(state, arguments.state_dir, arguments.count)
    elif arguments.command == "trim":
        command_trim(state, arguments.state_dir, arguments.run, arguments.exit_code)
    else:
        command_collect(state, arguments.state_dir, arguments.crashed)
    observations = search.observations_of(state)
    best = min(observations, key=lambda observation: observation.value, default=None)  # none before a track's first result
    print(json.dumps({"utc": datetime.now(UTC).isoformat(timespec="seconds"), "observations": len(observations),
                      "incumbent": best.name if best else None, "val_select": round(best.value, 4) if best else None,
                      "track": state["track"], "batches": len(state["batches"]), "pending": len(pending_runs(state))}))
    search.save_state(state, state_path)


if __name__ == "__main__":
    main()
