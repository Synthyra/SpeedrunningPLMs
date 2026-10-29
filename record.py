"""Append a finished run to the campaign ledger, with its verdict; fixed.

    python record.py runs/000_baseline/result.json --ledger runs.jsonl --verdict keep --note "the incumbent"

The ledger is one JSON row per run (`benchmark.ledger_row` plus the verdict and the note); a run appears once. The
verdict is the driver's call after reading the result: `keep` (the change stays in train.py), `revert`, `invalid`
(a cap or the clock broke), `crash`, or `reference` (a baseline or a confirmation seed).
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from speedrunning_plms import benchmark


VERDICTS = ("keep", "revert", "invalid", "crash", "reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result", type=Path, help="the run's result.json")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--verdict", choices=VERDICTS, required=True)
    parser.add_argument("--note", default="", help="one line: what the result means")
    parser.add_argument("--incumbent", default=None, help="the run this one was compared against")
    arguments = parser.parse_args()
    run_result = json.loads(arguments.result.read_text(encoding="utf-8"))
    if run_result["status"] != "completed" and arguments.verdict not in ("invalid", "crash"):
        raise ValueError(f"a {run_result['status']} run takes the verdict invalid or crash, not {arguments.verdict}")
    rows = [json.loads(line) for line in arguments.ledger.read_text(encoding="utf-8").splitlines() if line.strip()] if arguments.ledger.exists() else []
    if any(row["name"] == run_result["name"] for row in rows):
        raise ValueError(f"{run_result['name']} is already in {arguments.ledger}")
    row = {**benchmark.ledger_row(run_result), "verdict": arguments.verdict, "note": arguments.note,
           "incumbent": arguments.incumbent, "wandb": run_result.get("wandb")}
    with arguments.ledger.open("a", encoding="utf-8", newline="\n") as ledger:
        ledger.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({key: row[key] for key in ("name", "verdict", "validation_loss", "test_loss", "steps", "valid")}), flush=True)


if __name__ == "__main__":
    main()
