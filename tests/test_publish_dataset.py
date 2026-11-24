"""The dataset card, built offline from a small manifest: its counts, files and reference losses; nothing touches the Hub."""

from __future__ import annotations

import sys

from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))  # publish_dataset.py and summary_plot.py are scripts beside the package

import publish_dataset  # noqa: E402
import summary_plot  # noqa: E402


BENCHMARK_ID = "0123456789abcdef" + "f" * 48


def small_manifest() -> dict[str, Any]:
    """The fields `card` reads, with counts large enough to need thousands separators."""
    splits = {name: {"proteins": proteins, "families": families, "residues": residues, "scored_residues": scored}
              for name, proteins, families, residues, scored in (("val_select", 1_200, 1_200, 300_000, 45_000),
                                                                  ("validation", 2_500, 1_200, 900_000, 135_000),
                                                                  ("test", 2_600, 1_300, 950_000, 142_500))}
    files = {name: {"bytes": 3 * 2**20} for name in publish_dataset.FILE_NOTES if name != "manifest.json"}
    files["train_families.json"] = {"bytes": 5 * 2**10}
    return {"source": {"repo_id": "Synthyra/source_families", "revision": "abcdef0123456789abcdef"},
            "train": {"proteins": 1_754_034, "families": 400_000, "residues": 500_000_000, "probe_subset": 20_000},
            "splits": splits, "max_tokens": {"val_select": 1_024, "validation": 2_048, "test": 2_048}, "files": files}


def test_card_carries_the_manifest_counts_and_ids() -> None:
    text = publish_dataset.card(small_manifest(), BENCHMARK_ID, "Synthyra/test_repo")
    assert text.startswith("---\nlicense: cc-by-4.0\n")
    assert "`0123456789abcdef`" in text and BENCHMARK_ID[:17] not in text  # the id shows its first 16 characters only
    assert "| train | 1,754,034 | 400,000 | 500,000,000 | drawn each step | free |" in text
    assert "| validation | 2,500 | 1,200 | 900,000 | 135,000 | 2,048 |" in text
    assert "(`Synthyra/source_families` at `abcdef012345`)" in text  # the source revision cut to 12 characters
    assert 'snapshot_download("Synthyra/test_repo", repo_type="dataset")' in text
    assert "fixed 20,000-protein subset" in text and "at most 1,024 tokens" in text
    assert "residues A 5, C 23, D 13" in text  # ESM2's ids, in the benchmark's residue order


def test_card_lists_every_file_with_its_size() -> None:
    text = publish_dataset.card(small_manifest(), BENCHMARK_ID, publish_dataset.REPO_ID)
    assert f"| `train_tokens.u8` | 3 MB | {publish_dataset.FILE_NOTES['train_tokens.u8']} |" in text
    assert "| `train_families.json` | 5 KB |" in text
    assert "| `manifest.json` | small |" in text  # the manifest does not list itself


def test_card_reference_losses_agree_with_results() -> None:
    """The card's reference rows are the test losses of results.md's reference table, the floor under its own name."""
    text = publish_dataset.card(small_manifest(), BENCHMARK_ID, publish_dataset.REPO_ID)
    overall, _ = summary_plot.parse_tables((PROJECT / "results.md").read_text(encoding="utf-8"))
    for model, trained, loss in publish_dataset.REFERENCE_TEST_LOSS:
        name = summary_plot.UNIGRAM if model == "unigram residue frequencies" else model
        assert round(overall[name][1], 4) == loss, model
        assert f"| {model} | {trained} | {loss:.4f} |" in text
    assert len(publish_dataset.REFERENCE_TEST_LOSS) == 1 + len(summary_plot.ESM2) + len(summary_plot.ESMC)
