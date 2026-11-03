"""The reference comparison figure, from the tables of results.md.

    python summary_plot.py --results results.md --output <figure set folder>/reference_comparison.png

Writes, at 300 dpi and without a title (the caption lives in the figure set's README): A, test loss against parameters
for the ESM2 and ESMC reference models and each track's record; B, test loss by length bin for the same models.
"""

from __future__ import annotations

import argparse
import matplotlib
import numpy as np

from pathlib import Path

from plots import KEPT_COLOR, LENGTH_BINS, RECORD_COLOR, REFERENCE_COLORS, letter


matplotlib.use("Agg")  # the backend is fixed before pyplot loads: this script only writes files

import matplotlib.pyplot as plt  # noqa: E402


ESM2 = ["ESM2 8M", "ESM2 35M", "ESM2 150M", "ESM2 650M", "ESM2 3B"]
ESMC = ["ESMC 300M", "ESMC 600M", "ESMC 6B"]
UNIGRAM = "unigram floor"
RECORD_10MIN = "record, 10 min"
RECORD_1H = "record, 1 hour"
LABEL_COLOR = "#52514e"
LABEL_GAP = 0.052  # nats; the least vertical distance between two labels at the end of panel B's lines


def parse_tables(text: str) -> tuple[dict[str, tuple[float, float]], dict[str, list[float]]]:
    """Each model's (parameters, test loss) and its loss by length bin; raises when a table of results.md moved."""
    overall: dict[str, tuple[float, float]] = {}
    by_length: dict[str, list[float]] = {}
    header: list[str] = []
    for line in text.splitlines():
        cells = [cell.strip().strip("*") for cell in line.strip().strip("|").split("|")]
        if not line.startswith("|"):
            header = []
        elif not header:
            header = cells
        elif set(cells[0]) == {"-"}:
            continue
        elif header[:3] == ["Model", "Parameters", "`val_select`"]:
            parameters = float(cells[1].replace(",", "")) if cells[1] else float("nan")  # the floor has no parameters
            overall[cells[0]] = (parameters, float(cells[header.index("Test loss")]))
        elif header[1:] == LENGTH_BINS:
            by_length[cells[0]] = [float(cell) for cell in cells[1:]]
    expected = {*ESM2, *ESMC, RECORD_10MIN, RECORD_1H, UNIGRAM}
    assert expected <= overall.keys(), f"results.md lacks {sorted(expected - overall.keys())}"
    assert expected <= by_length.keys(), f"results.md lacks {sorted(expected - by_length.keys())}"
    return overall, by_length


def spread_labels(positions: list[float], gap: float) -> list[float]:
    """Move sorted label positions apart by at least gap, keeping their mean near the original."""
    placed = list(positions)
    for _ in range(200):
        for index in range(len(placed) - 1):
            overlap = gap - (placed[index + 1] - placed[index])
            if overlap > 0:
                placed[index] -= overlap / 2
                placed[index + 1] += overlap / 2
    return placed


def style(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(True, color="#e6e5e0", linewidth=0.8)
    axis.set_axisbelow(True)


def panel_parameters(axis: plt.Axes, overall: dict[str, tuple[float, float]]) -> None:
    floor = overall[UNIGRAM][1]
    axis.axhline(floor, color=REFERENCE_COLORS["unigram"], linestyle="--", linewidth=1.2)
    axis.text(2.0e9, floor + 0.015, "unigram floor", color=LABEL_COLOR, fontsize=8.5, ha="center", va="bottom")
    families = [("ESM2", ESM2, REFERENCE_COLORS["esm2"], "o"), ("ESMC", ESMC, REFERENCE_COLORS["esmc"], "s")]
    for family, names, color, marker in families:
        parameters = [overall[name][0] for name in names]
        losses = [overall[name][1] for name in names]
        axis.plot(parameters, losses, color=color, marker=marker, markersize=7, linewidth=2, label=family, zorder=3)
        for name, x, y in zip(names, parameters, losses, strict=True):
            offset = (-9, -14) if name == "ESMC 600M" else (6, 6)  # 600M and 650M sit almost on one point
            axis.annotate(name.split()[1], (x, y), xytext=offset, textcoords="offset points", color=LABEL_COLOR, fontsize=8.5)
    records = [(RECORD_10MIN, "10 min record", KEPT_COLOR, (9, 5)), (RECORD_1H, "1 hour record", RECORD_COLOR, (9, -11))]
    for name, label, color, offset in records:
        x, y = overall[name]
        axis.plot([x], [y], color=color, marker="*", markersize=15, markeredgecolor="white", markeredgewidth=0.7,
                  linestyle="none", label=label, zorder=4)
        axis.annotate(f"{label}, {y:.4f}", (x, y), xytext=offset, textcoords="offset points", color=LABEL_COLOR, fontsize=8.5)
    axis.set_xscale("log")
    axis.set_xlabel("parameters")
    axis.set_ylabel("test loss (nats per masked residue)")
    axis.legend(loc="lower left", frameon=False, fontsize=9)


def panel_lengths(axis: plt.Axes, by_length: dict[str, list[float]]) -> None:
    bins = np.arange(len(LENGTH_BINS))  # (B,)
    series = [(UNIGRAM, REFERENCE_COLORS["unigram"], "", "--", 1.2, "unigram floor")]
    series += [(name, REFERENCE_COLORS["esm2"], "o", "-", 1.6, name) for name in ESM2]
    series += [(name, REFERENCE_COLORS["esmc"], "s", "-", 1.6, name) for name in ESMC]
    series += [(RECORD_10MIN, KEPT_COLOR, "*", "-", 2.4, "10 min record"), (RECORD_1H, RECORD_COLOR, "*", "-", 2.4, "1 hour record")]
    for name, color, marker, linestyle, width, _ in series:
        is_record = marker == "*"
        axis.plot(bins, by_length[name], color=color, marker=marker or None, markersize=11 if is_record else 4.5, linestyle=linestyle,
                  linewidth=width, markeredgecolor="white", markeredgewidth=0.5, zorder=4 if is_record else 3,
                  markevery=[len(bins) - 1] if is_record else None)  # one star ends the line: a star per bin hides the gap
    ends = sorted((by_length[entry[0]][-1], entry[5]) for entry in series)
    placed = spread_labels([end for end, _ in ends], LABEL_GAP)
    label_x = bins[-1] + 0.3
    for (end, label), y in zip(ends, placed, strict=True):
        axis.annotate(label, xy=(bins[-1] + 0.06, end), xytext=(label_x, y), color=LABEL_COLOR, fontsize=8.5, va="center",
                      annotation_clip=False, arrowprops={"arrowstyle": "-", "color": "#b5b4ad", "linewidth": 0.6, "shrinkA": 1, "shrinkB": 0})
    axis.set_xticks(bins, LENGTH_BINS)
    axis.set_xlim(-0.15, bins[-1] + 0.15)
    axis.set_xlabel("protein length (residues)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="results.md, which holds the numbers")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    text = arguments.results.read_text(encoding="utf-8")
    overall, by_length = parse_tables(text)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True, gridspec_kw={"width_ratios": [1, 1.05]})
    panel_parameters(axes[0], overall)
    panel_lengths(axes[1], by_length)
    axes[1].tick_params(labelleft=True)
    for axis, panel in zip(axes, "AB", strict=True):
        style(axis)
        letter(axis, panel)
    figure.tight_layout()
    figure.subplots_adjust(right=0.89, wspace=0.17)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=300)
    plt.close(figure)
    print(f"{len(overall)} models -> {arguments.output}")


if __name__ == "__main__":
    main()
