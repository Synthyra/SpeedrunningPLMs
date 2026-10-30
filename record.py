"""Append a finished run to the campaign ledger, with its verdict; fixed.

    python record.py runs/000_baseline/result.json --ledger runs.jsonl --verdict keep --note "the incumbent"

The ledger is one JSON row per run (`benchmark.ledger_row` plus the verdict, the note and the comparison run); a run
appears once. The verdict is the driver's call after reading the result: `keep` (the change becomes the incumbent),
`revert`, `invalid` (a cap or the clock broke), `crash`, or `reference` (a baseline, a confirmation seed or a short
search rung). The search driver adds its proposal's fields (`search.py`: the batch, the proposer, the parents, its
reason, the genome and the rung) through `append_row`.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any

from speedrunning_plms import benchmark


VERDICTS = ("keep", "revert", "invalid", "crash", "reference")


def append_row(ledger: Path, run_result: dict[str, Any], verdict: str, note: str, incumbent: str | None,
               extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Append one run's row; refuse a second row for a name and a record verdict for a broken run."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict {verdict!r} is not one of {VERDICTS}")
    if run_result["status"] != "completed" and verdict not in ("invalid", "crash"):
        raise ValueError(f"a {run_result['status']} run takes the verdict invalid or crash, not {verdict}")
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()] if ledger.exists() else []
    if any(row["name"] == run_result["name"] for row in rows):
        raise ValueError(f"{run_result['name']} is already in {ledger}")
    row = {**benchmark.ledger_row(run_result), "verdict": verdict, "note": note, "incumbent": incumbent,
           "wandb": run_result.get("wandb"), **(extra or {})}
    with ledger.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result", type=Path, help="the run's result.json")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--verdict", choices=VERDICTS, required=True)
    parser.add_argument("--note", default="", help="one line: what the result means")
    parser.add_argument("--incumbent", default=None, help="the run this one was compared against")
    arguments = parser.parse_args()
    run_result = json.loads(arguments.result.read_text(encoding="utf-8"))
    row = append_row(arguments.ledger, run_result, arguments.verdict, arguments.note, arguments.incumbent)
    print(json.dumps({key: row[key] for key in ("name", "verdict", "validation_loss", "test_loss", "steps", "valid")}), flush=True)


if __name__ == "__main__":
    main()
