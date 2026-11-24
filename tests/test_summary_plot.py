"""The comparison figure's reader of results.md: both tables parsed by their headers, and a moved table refused."""

from __future__ import annotations

import math
import sys
import pytest

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))  # summary_plot.py and plots.py are scripts beside the package

import summary_plot  # noqa: E402


MODELS = [summary_plot.UNIGRAM, summary_plot.RECORD_10MIN, summary_plot.RECORD_1H, *summary_plot.ESM2, *summary_plot.ESMC]
OVERALL_HEADER = "| Model | Parameters | `val_select` | Validation | Test loss | Bits |\n|---|---|---|---|---|---|"
LENGTH_HEADER = "| Model | 50-99 | 100-249 | 250-499 | 500-999 | 1000-2046 |\n|---|---|---|---|---|---|"


def overall_row(index: int, name: str) -> str:
    """A row whose numbers encode its index, the records bold as results.md writes them, the floor without parameters."""
    shown = f"**{name}**" if name.startswith("record") else name
    parameters = "" if name == summary_plot.UNIGRAM else f"{1_000_000 * (index + 1):,}"
    return f"| {shown} | {parameters} | 9.{index:04d} | 8.{index:04d} | 2.{index:04d} | 4.000 |"


def length_row(index: int, name: str) -> str:
    return f"| {name} | " + " | ".join(f"{bin_index}.{index:03d}" for bin_index in range(5)) + " |"


def results_text(models: list[str]) -> str:
    """Two tables in results.md's layout, prose between them."""
    overall = "\n".join(overall_row(index, name) for index, name in enumerate(models))
    by_length = "\n".join(length_row(index, name) for index, name in enumerate(models))
    return f"## References\n\n{OVERALL_HEADER}\n{overall}\n\nBy length:\n\n{LENGTH_HEADER}\n{by_length}\n"


def test_parse_tables_reads_both_tables_by_their_headers() -> None:
    overall, by_length = summary_plot.parse_tables(results_text(MODELS))
    assert overall.keys() == by_length.keys() == set(MODELS)
    parameters, test_loss = overall[summary_plot.RECORD_10MIN]  # index 1, bold in the table
    assert (parameters, test_loss) == (2_000_000.0, 2.0001)  # the comma-grouped count, and the Test loss column, not val_select
    assert math.isnan(overall[summary_plot.UNIGRAM][0]) and overall[summary_plot.UNIGRAM][1] == 2.0
    assert by_length["ESMC 6B"] == [0.010, 1.010, 2.010, 3.010, 4.010]  # ESMC 6B is index 10


def test_parse_tables_refuses_a_missing_model() -> None:
    with pytest.raises(AssertionError, match="ESMC 6B"):
        summary_plot.parse_tables(results_text(MODELS[:-1]))


def test_parse_tables_reads_the_project_results() -> None:
    """The tables of the committed results.md still parse, so the figure can be redrawn from them."""
    overall, by_length = summary_plot.parse_tables((PROJECT / "results.md").read_text(encoding="utf-8"))
    assert set(MODELS) <= overall.keys() and set(MODELS) <= by_length.keys()
    assert all(len(losses) == len(summary_plot.LENGTH_BINS) for losses in by_length.values())
