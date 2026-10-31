"""The campaign figures from the run and reference JSON files; fixed.

    python plots.py --runs <folder of result.json copies> --output-dir <figure set folder>

Writes, at 300 dpi and without titles (the captions live in the figure set's README):
`selection_curves.png` (A: every run's `val_select` loss against the clock, the record holders in color; B: the
record's test loss by candidate), `references.png` (A: test loss against parameters for the record holder and the
reference models; B: test loss by length bin for the same). Reads any JSON with `metrics.val_select` as a run
(a `clock` field) or a reference model (a `model_id` field); the verdicts come from the campaign ledger when given.
One folder is one track: every run in it must share one `clock.budget_seconds` (the 10-minute and 1-hour tracks get
their own folders and figure sets), and the record's label and the clock axis read that budget.
"""

from __future__ import annotations

import argparse
import json
import matplotlib
import numpy as np

from pathlib import Path
from typing import Any

matplotlib.use("Agg")  # the backend is fixed before pyplot loads: this script only writes files

import matplotlib.pyplot as plt  # noqa: E402


REFERENCE_LABELS = {"unigram": "unigram", "esm2_8m": "ESM2 8M", "esm2_35m": "ESM2 35M", "esm2_150m": "ESM2 150M",
                    "esm2_650m": "ESM2 650M", "esm2_3b": "ESM2 3B", "esmc_small": "ESMC 300M", "esmc_large": "ESMC 600M",
                    "esmc_6b": "ESMC 6B"}
LENGTH_BINS = ["50-99", "100-249", "250-499", "500-999", "1000-2046"]
RECORD_COLOR = "#c0392b"
KEPT_COLOR = "#e67e22"
REVERTED_COLOR = "#b0b0b0"
REFERENCE_COLORS = {"esm2": "#2e86c1", "esmc": "#239b56", "unigram": "#7f8c8d"}


def load(folder: Path, ledger: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    runs, references = [], []
    for path in sorted(folder.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if "val_select" not in report.get("metrics", {}):
            continue
        (runs if "clock" in report else references).append(report)
    verdicts: dict[str, str] = {}
    if ledger is not None:
        for line in ledger.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            verdicts[row["name"]] = row.get("verdict") or ""
        order = {name: index for index, name in enumerate(verdicts)}  # the ledger's order is the campaign's
        runs.sort(key=lambda run: order.get(run["name"], len(order)))
    return runs, references, verdicts


def track_budget(runs: list[dict[str, Any]]) -> float:
    """The one training budget, in seconds, every run of the folder shares; raises on a folder that mixes tracks."""
    budgets = sorted({float(run["clock"]["budget_seconds"]) for run in runs})
    if len(budgets) != 1:
        raise ValueError(f"a runs folder holds one track, one budget; this one holds {budgets or 'no runs'} s")
    return budgets[0]


def budget_label(budget: float) -> str:
    """The budget as the figures name it: 600 s is "10 min"."""
    return f"{budget / 60:g} min"


def record_name(runs: list[dict[str, Any]], verdicts: dict[str, str]) -> str | None:
    """The kept run with the lowest `val_select` loss: the selection rule's record holder."""
    kept = [run for run in runs if verdicts.get(run["name"]) == "keep"]
    return min(kept, key=lambda run: run["metrics"]["val_select"]["loss"])["name"] if kept else None


def letter(axis: plt.Axes, text: str) -> None:
    axis.text(-0.12, 1.04, text, transform=axis.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")


def selection_curves(runs: list[dict[str, Any]], verdicts: dict[str, str], budget: float, output: Path) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    record = record_name(runs, verdicts)
    for run in runs:
        seconds = [point["train_seconds"] for point in run["curve"]]
        losses = [point["loss"] for point in run["curve"]]
        verdict = verdicts.get(run["name"], "")
        if run["name"] == record:
            left.plot(seconds, losses, color=RECORD_COLOR, linewidth=2.4, zorder=5, label=f"{run['name']} (record)")
        elif verdict == "keep":
            left.plot(seconds, losses, color=KEPT_COLOR, linewidth=1.4, alpha=0.9, zorder=4, label=run["name"])
        else:
            left.plot(seconds, losses, color=REVERTED_COLOR, linewidth=0.9, alpha=0.8, zorder=2)
    left.plot([], [], color=REVERTED_COLOR, linewidth=0.9, label="reverted or reference")
    left.set_xlabel("training clock (s)")
    left.set_ylabel("val_select loss (nats per masked residue)")
    left.set_xlim(budget / 10, budget + budget / 30)  # from the first eval's neighborhood to just past the budget
    left.legend(fontsize=7.5, frameon=False, loc="upper right")
    left.grid(alpha=0.25)
    letter(left, "A")

    names = [run["name"] for run in runs]
    tests = [run["metrics"]["test"]["loss"] for run in runs]
    colors = [RECORD_COLOR if name == record else KEPT_COLOR if verdicts.get(name) == "keep" else REVERTED_COLOR for name in names]
    positions = np.arange(len(runs))
    best = np.minimum.accumulate(np.where([verdicts.get(name) == "keep" for name in names], tests, np.inf))
    right.step(positions, best, where="post", color=RECORD_COLOR, linewidth=1.2, alpha=0.7, zorder=2, label="record")
    right.scatter(positions, tests, color=colors, s=[90 if name == record else 34 for name in names],
                  marker="o", zorder=4, edgecolor="white", linewidth=0.6)
    right.scatter([], [], color=KEPT_COLOR, label="kept")
    right.scatter([], [], color=REVERTED_COLOR, label="reverted")
    right.set_xticks(positions)
    right.set_xticklabels([name if name.split("_")[0].isdigit() is False else name.split("_")[0] for name in names],
                          fontsize=5.5, rotation=90)
    right.set_xlabel("candidate, in campaign order")
    right.set_ylabel("test loss (nats per masked residue)")
    low, high = min(tests), max(tests)
    right.set_ylim(low - 0.003, high + 0.003)
    right.legend(fontsize=8, frameon=False)
    right.grid(axis="y", alpha=0.25)
    letter(right, "B")
    figure.tight_layout()
    figure.savefig(output / "selection_curves.png", dpi=300)
    plt.close(figure)


def references_figure(runs: list[dict[str, Any]], references: list[dict[str, Any]], verdicts: dict[str, str], budget: float,
                      output: Path) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    track = budget_label(budget)
    name = record_name(runs, verdicts)
    record = next(run for run in runs if run["name"] == name) if name else runs[-1]
    unigram = next((reference for reference in references if reference["model_id"] == "unigram"), None)
    for reference in references:
        if reference["model_id"] == "unigram":
            continue
        family = "esm2" if reference["model_id"].startswith("esm2") else "esmc"
        left.scatter(reference["parameters"], reference["metrics"]["test"]["loss"], color=REFERENCE_COLORS[family], s=36, zorder=4)
        offset = (6, -11) if reference["model_id"] == "esmc_large" else (5, 4)
        left.annotate(REFERENCE_LABELS[reference["model_id"]], (reference["parameters"], reference["metrics"]["test"]["loss"]),
                      textcoords="offset points", xytext=offset, fontsize=7.5)
    left.scatter(record["architecture"]["parameters"], record["metrics"]["test"]["loss"], color=RECORD_COLOR, marker="*", s=140, zorder=5)
    left.annotate(f"{record['name']}, {track}",(record["architecture"]["parameters"], record["metrics"]["test"]["loss"]),
                  textcoords="offset points", xytext=(6, -10), fontsize=7.5, color=RECORD_COLOR)
    if unigram is not None:
        left.axhline(unigram["metrics"]["test"]["loss"], color=REFERENCE_COLORS["unigram"], linestyle="--", linewidth=1)
        left.text(1.1e7, unigram["metrics"]["test"]["loss"] + 0.01, "unigram floor", fontsize=7.5, color=REFERENCE_COLORS["unigram"])
    left.set_xscale("log")
    left.set_xlabel("parameters")
    left.set_ylabel("test loss (nats per masked residue)")
    left.scatter([], [], color=REFERENCE_COLORS["esm2"], label="ESM2 (UniRef50, days)")
    left.scatter([], [], color=REFERENCE_COLORS["esmc"], label="ESMC (UniRef and more, days)")
    left.legend(fontsize=7.5, frameon=False, loc="lower left")
    left.grid(alpha=0.25)
    letter(left, "A")

    positions = np.arange(len(LENGTH_BINS))
    shown = [reference for reference in references if reference["model_id"] in ("esm2_8m", "esm2_650m", "esmc_large")]
    for reference in shown:
        family = "esm2" if reference["model_id"].startswith("esm2") else "esmc"
        by_length = reference["metrics"]["test"]["loss_by_length"]
        right.plot(positions, [by_length[bin_name] for bin_name in LENGTH_BINS], marker="o", markersize=4,
                   color=REFERENCE_COLORS[family], alpha=0.5 + 0.5 * (reference["model_id"] != "esm2_8m"),
                   label=REFERENCE_LABELS[reference["model_id"]])
    by_length = record["metrics"]["test"]["loss_by_length"]
    right.plot(positions, [by_length[bin_name] for bin_name in LENGTH_BINS], marker="*", markersize=9, color=RECORD_COLOR,
               linewidth=2, label=f"{record['name']}, {track}")
    if unigram is not None:
        by_length = unigram["metrics"]["test"]["loss_by_length"]
        right.plot(positions, [by_length[bin_name] for bin_name in LENGTH_BINS], linestyle="--", color=REFERENCE_COLORS["unigram"],
                   linewidth=1, label="unigram")
    right.set_xticks(positions)
    right.set_xticklabels(LENGTH_BINS)
    right.set_xlabel("protein length (residues)")
    right.set_ylabel("test loss (nats per masked residue)")
    right.legend(fontsize=7.5, frameon=False)
    right.grid(alpha=0.25)
    letter(right, "B")
    figure.tight_layout()
    figure.savefig(output / "references.png", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="a folder of result.json copies and reference JSON files")
    parser.add_argument("--ledger", type=Path, default=None, help="runs.jsonl, for the verdicts")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    runs, references, verdicts = load(arguments.runs, arguments.ledger)
    budget = track_budget(runs)  # before any filter, so no run of another track hides in the folder
    runs = [run for run in runs if run.get("seed", 0) == 0 and run["clock"]["valid"]]  # seeds are references; an invalid run is no curve
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    selection_curves(runs, verdicts, budget, arguments.output_dir)
    references_figure(runs, references, verdicts, budget, arguments.output_dir)
    print(f"{len(runs)} runs, {len(references)} reference models -> {arguments.output_dir}")


if __name__ == "__main__":
    main()
