"""CPU tests of perplexity.py's own arithmetic: the k-mer references, the relatives summary and the figure's label spread."""

from __future__ import annotations

import dataclasses
import functools
import json
import sys
import types
import numpy as np
import perplexity
import pytest
import torch

from pathlib import Path
from types import SimpleNamespace

from speedrunning_plms import benchmark


def write_state(state: Path, data_dir: Path, rows: int = 30) -> list[dict]:
    """dev.jsonl and corpus.jsonl drawn from the training stream itself (standard residues only), and corpus.json."""
    stream = benchmark.load_train(data_dir)
    letters = {index: token for index, token in enumerate(benchmark.tokens.ESM2_TOKENS) if len(token) == 1}
    rng = np.random.default_rng(0)
    proteins = []
    for index in range(len(stream.offsets) - 1):
        sequence = "".join(letters.get(int(token), "?") for token in stream.protein(index)[1:-1])
        if set(sequence) <= set(perplexity.STANDARD_RESIDUES) and len(sequence) >= perplexity.POSITIONS:
            positions = sorted(int(position) for position in rng.choice(len(sequence), perplexity.POSITIONS, replace=False))
            proteins.append({"accession": f"T{index}", "sequence": sequence, "positions": positions, "cluster": f"T{index}",
                             "cluster_size": 1, "organism": "x", "taxid": 1})
        if len(proteins) == rows:
            break
    state.mkdir(parents=True, exist_ok=True)
    for name, part in (("dev", proteins[:10]), ("corpus", proteins[10:])):
        (state / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in part), encoding="utf-8")
    (state / "corpus.json").write_text(json.dumps({"corpus_sha256": perplexity.sha256(state / "corpus.jsonl"), "release": "test",
                                                   "query": "test", "dev_sha256": perplexity.sha256(state / "dev.jsonl")}), encoding="utf-8")
    return proteins


def test_kmer_tables_count_what_a_direct_count_finds(tiny_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proteins = write_state(tmp_path, tiny_data)
    monkeypatch.setattr(perplexity, "KMER_BETAS", (1.0, 4.0))  # every scored window is in the counts, so the dev set wants beta -> 0
    with pytest.raises(AssertionError, match="widen KMER_BETAS"):
        perplexity.kmer_references(tmp_path, tiny_data)
    monkeypatch.setattr(perplexity, "KMER_BETAS", tuple(4.0 ** power for power in range(-12, 2)))
    perplexity.kmer_references(tmp_path, tiny_data)
    report = json.loads((tmp_path / "kmers.json").read_text(encoding="utf-8"))
    stream = benchmark.load_train(tiny_data)
    residues = [perplexity.residue_symbols(np.asarray(stream.protein(index)[1:-1])) for index in range(len(stream.offsets) - 1)]
    counts = np.bincount(np.concatenate(residues), minlength=perplexity.OTHER + 1)[:20]  # (c,)
    corpus = proteins[10:]
    truth = np.array([perplexity.residue_symbols(protein["sequence"])[position] for protein in corpus for position in protein["positions"]])
    add_one = (counts + 1) / (counts.sum() + 20)  # (c,)
    expected = -np.log2(add_one[truth]).reshape(len(corpus), -1).mean(axis=1).mean()
    assert report["models"]["1-mer"]["bits_per_residue"] == pytest.approx(expected, rel=1e-9)
    assert report["models"]["1-mer"]["training_windows"] == int(counts.sum())
    # Each scored residue's own 5-residue window is in the training counts, so context can only help here.
    bits = [report["models"][f"{2 * a + 1}-mer"]["bits_per_residue"] for a in perplexity.KMER_HALF_WIDTHS]
    assert bits[2] < bits[1] < bits[0]


def test_a_context_code_reads_each_side_with_boundaries_past_the_ends() -> None:
    padded, positions = perplexity.padded_stream([np.array([0, 1, 2]), np.array([3, 4])])
    assert padded[positions].tolist() == [0, 1, 2, 3, 4]
    base, boundary = perplexity.CONTEXT_BASE, perplexity.BOUNDARY
    first, second = perplexity.context_codes(padded, positions[[0, 3]], 2)  # residue 0 of each protein
    assert first == ((boundary * base + 1) * base + boundary) * base + 2  # left 1, right 1, then left 2, right 2
    assert second == ((boundary * base + 4) * base + boundary) * base + boundary  # nothing of the first protein leaks in


def test_the_relatives_summary_takes_the_union_and_each_cutoff(tiny_data: Path, tmp_path: Path) -> None:
    write_state(tmp_path, tiny_data, rows=12)
    directory = tmp_path / "relatives"
    directory.mkdir()
    # Training 0 passes the rule on query coverage alone; 1 passes every cutoff; 2 fails on e-value; 3 only the forward search finds.
    (directory / "reverse.tsv").write_text("0\tP1\t0.350\t0.600\t0.300\t1e-10\n1\tP2\t0.950\t0.900\t0.850\t1e-50\n"
                                           "2\tP1\t0.900\t0.900\t0.900\t0.01\n", encoding="ascii")
    (directory / "forward.tsv").write_text("P1\t0\t0.400\t0.700\t0.700\t1e-12\nP3\t3\t0.310\t0.200\t0.550\t1e-5\n", encoding="ascii")
    (directory / "exhaustive.tsv").write_text("", encoding="ascii")
    perplexity.summarize_relatives(tmp_path, tiny_data, 300, 12, "test-build")
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert report["cutoffs"]["rule"]["training_flagged"] == 3 and report["cutoffs"]["rule"]["benchmark_with_relative"] == 3
    assert report["cutoffs"]["cluster_rule"]["training_flagged"] == 1 and report["cutoffs"]["identity_90"]["training_flagged"] == 1
    assert report["by_search"]["forward"]["rule"] == 2 and report["by_search"]["exhaustive"]["rule"] == 0
    assert (directory / "train_relatives.tsv").read_text(encoding="ascii") == "0\t0.400\tP1\n1\t0.950\tP2\n3\t0.310\tP3\n"
    assert not (tmp_path / "exclusion.json").exists()  # relatives are measured, never removed


def test_exact_copies_name_identical_training_proteins_only(tiny_data: Path, tmp_path: Path) -> None:
    proteins = write_state(tmp_path, tiny_data, rows=12)  # every benchmark protein is a training protein's sequence
    corpus = [json.loads(line) for line in (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    corpus[0]["sequence"] = corpus[0]["sequence"][:-1]  # one residue short: no longer a copy
    (tmp_path / "corpus.jsonl").write_text("".join(json.dumps(row) + "\n" for row in corpus), encoding="utf-8")
    perplexity.exact_copies(tmp_path, tiny_data)
    exclusion = json.loads((tmp_path / "exclusion.json").read_text(encoding="utf-8"))
    copied = {int(protein["accession"][1:]) for protein in proteins} - {int(corpus[0]["accession"][1:])}
    assert exclusion["indices"] == sorted(copied) and exclusion["copied_benchmark_proteins"] == 11
    assert exclusion["benchmark"]["proteins"] == 12 and set(exclusion["copies"]) == {str(index) for index in copied}


class Vocabulary:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}


def test_residue_ids_read_each_models_own_form_and_refuse_two() -> None:
    specials = ["<pad>", "</s>", "<unk>"]
    prot_t5 = Vocabulary(specials + ["▁" + residue for residue in perplexity.STANDARD_RESIDUES])  # ProtT5's word-initial pieces
    ankh = Vocabulary(specials + ["[NLU]"] + list(reversed(perplexity.STANDARD_RESIDUES)))  # ANKH's bare letters, its own order
    assert perplexity.residue_token_ids(prot_t5) == list(range(3, 23))
    assert perplexity.residue_token_ids(ankh) == [4 + 19 - index for index in range(20)]
    with pytest.raises(AssertionError, match="ambiguous"):
        perplexity.residue_token_ids(Vocabulary(specials + list(perplexity.STANDARD_RESIDUES) + ["▁A"]))


def test_restoring_past_key_values_undoes_a_rename_of_the_field(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers.utils import ModelOutput

    @dataclasses.dataclass
    class Output(ModelOutput):
        last_hidden_state: torch.Tensor | None = None
        past_key_values: object | None = None

    constructor = Output.__init__

    @functools.wraps(constructor)
    def renamed(self: Output, *args: object, **kwargs: object) -> None:  # what transformers 5.24's metaclass does
        kwargs["cache"] = kwargs.pop("past_key_values")
        constructor(self, *args, **kwargs)

    type.__setattr__(Output, "__init__", renamed)
    module = types.ModuleType("fastplms.test_output")
    module.Output = Output
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(TypeError, match="cache"):
        Output(past_key_values=1)
    assert perplexity.restore_past_key_values("fastplms.") == 1
    # The scorers read the logits and hidden states alone; transformers 5.24 itself files a set past_key_values under cache.
    assert Output(last_hidden_state=torch.ones(1), past_key_values=1).last_hidden_state.tolist() == [1.0]
    assert perplexity.restore_past_key_values("fastplms.") == 0  # nothing left wrapped


class Recorder(torch.nn.Module):
    """Uniform logits over a 28-token vocabulary, keeping the last input it saw."""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        self.seen = input_ids.clone()  # (K, T)
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 28))  # (K, T, 28)


@pytest.mark.parametrize(("layout", "expected"), [
    ("aa_only", [2, 18, 16, 24, 4, 3]),
    ("structure_first", [5, 6, 6, 6, 6, 7, 2, 18, 16, 24, 4, 3]),
    ("aa_first", [2, 18, 16, 24, 4, 3, 5, 6, 6, 6, 6, 7]),
])
def test_dplm2_layouts_mask_the_residue_on_its_own_track(layout: str, expected: list[int]) -> None:
    scorer = perplexity.MaskedScorer.__new__(perplexity.MaskedScorer)  # the input layout alone, no weights
    scorer.spec, scorer.device, scorer.layout = perplexity.MODELS["dplm2_150m"], torch.device("cpu"), layout
    scorer.ids, scorer.structure = {"cls": 2, "eos": 3, "mask": 4}, [5, 6, 7]
    scorer.residue_ids = torch.arange(8, 28)  # A to Y at 8 to 27: M 18, K 16, T 24, A 8
    scorer.model = Recorder()
    log_probs = scorer("MKTA", (0, 3))
    assert scorer.model.seen[1].tolist() == expected  # the second row masks residue 3, A
    assert np.allclose(log_probs, np.log(1 / 20))


def test_the_table_sorts_by_bits_and_writes_each_format_as_given() -> None:
    reports = [{"model_id": "ankh3_large", "label": "ANKH3 large", "parameters": 1_879_049_216, "bits_per_residue": 3.1,
                "ci95": [3.0, 3.2], "format": ["sentinel", "<extra_id_0>", "[NLU]"]},
               {"model_id": "dplm2_150m", "label": "DPLM2 150M", "parameters": 150_000_000, "bits_per_residue": 3.4,
                "ci95": [3.3, 3.5], "format": ["aa_only", "", ""]}]
    star = {"model_id": "run", "parameters": 199_000_000, "bits_per_residue": 3.2, "ci95": [3.1, 3.3], "run_finished_utc": "2026-11-23T20:00:00Z"}
    kmers = {"models": {"1-mer": {"bits_per_residue": 4.19, "ci95": [4.18, 4.2]}}}
    lines = perplexity.markdown_table(reports, star, kmers).splitlines()
    assert lines[2].startswith("| ANKH3 large (sentinel, `<extra_id_0>`, `[NLU]`) | 2025-05-26 | 1.9B | 3.100 |")
    assert lines[3].startswith(f"| **{perplexity.STAR_LABEL}** (`run`) | 2026-11-23 | 199M | 3.200 |")
    assert lines[4].startswith("| DPLM2 150M (aa only) |") and lines[5].startswith("| 1-mer lookup table (reference) |")


def test_reference_labels_keep_their_gap_and_order() -> None:
    values = [4.32, 4.19, 4.14, 4.12]
    heights = perplexity.spread(values, 0.05)
    assert sorted(heights, reverse=True) == heights  # the order of the lines
    assert min(a - b for a, b in zip(heights, heights[1:])) >= 0.05 - 1e-12
    assert np.mean(heights) == pytest.approx(np.mean(values))
    assert perplexity.spread([1.0, 2.0], 0.5) == [1.0, 2.0]  # far enough apart: untouched


def test_the_served_scorer_masks_one_residue_reads_its_row_retries_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import email.message
    import io
    import urllib.error

    monkeypatch.setenv("FOUNDRY_SECRETS", str(tmp_path / "absent.env"))  # loads nothing: the test's token stands
    monkeypatch.setenv("BIOHUB_TOKEN", "test-token")
    monkeypatch.setattr(perplexity.time, "sleep", lambda seconds: None)
    sequence, positions = "MKTAYIAKQR", (1, 8)
    bodies, failures = [], [1]  # the first request is rate limited once

    class Response:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *exception: object) -> None:
            return None

        def read(self) -> bytes:
            return self.payload

    def fake_urlopen(request: object, timeout: float) -> Response:
        assert request.headers["Authorization"] == "Bearer test-token"  # type: ignore[attr-defined]
        if failures:
            failures.pop()
            headers = email.message.Message()
            headers["Retry-After"] = "0"
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", headers, io.BytesIO(b"slow down"))  # type: ignore[attr-defined, arg-type]
        body = json.loads(request.data)  # type: ignore[attr-defined]
        bodies.append(body)
        rows = [[0.0] * 64 for _ in body["inputs"]["sequence"]]
        masked = body["inputs"]["sequence"].index(perplexity.ESM3_MASK)
        rows[masked] = [0.1 * column for column in range(64)]
        return Response(json.dumps({"model": body["model"], "tokens_used": len(sequence), "logits": {"sequence": rows}}).encode())

    monkeypatch.setattr(perplexity.urllib.request, "urlopen", fake_urlopen)
    spec = perplexity.MODELS["esm3_large"]
    cache = tmp_path / "esm3_large.partial.jsonl"
    scorer = perplexity.BiohubScorer(spec, cache, workers=2)
    log_probs = scorer(sequence, positions)  # (K,)
    assert scorer.retries == {"HTTP 429": 1}  # the rate limit was waited out and counted
    vocabulary = list(perplexity.ESM3_SEQUENCE_VOCAB)
    for body, position in zip(sorted(bodies, key=lambda body: body["inputs"]["sequence"].index(perplexity.ESM3_MASK)), positions, strict=True):
        ids = body["inputs"]["sequence"]  # (L + 2,)
        assert body["model"] == "esm3-large-2024-03" and ids[0] == perplexity.ESM3_BOS and ids[-1] == perplexity.ESM3_EOS
        assert ids.index(perplexity.ESM3_MASK) == position + 1 and ids.count(perplexity.ESM3_MASK) == 1
        assert [vocabulary[token] for token in ids[1:-1] if token != perplexity.ESM3_MASK] == list(sequence[:position] + sequence[position + 1:])
    residue_logits = torch.tensor([0.1 * vocabulary.index(residue) for residue in perplexity.STANDARD_RESIDUES])  # (c,)
    expected = [float(residue_logits.log_softmax(0)[perplexity.STANDARD_RESIDUES.index(sequence[position])]) for position in positions]
    assert log_probs == pytest.approx(expected, abs=1e-6)
    resumed = perplexity.BiohubScorer(spec, cache)
    calls = len(bodies)
    assert resumed(sequence, positions) == pytest.approx(log_probs) and len(bodies) == calls  # read back, nothing requested
    assert resumed.tokens_used == len(positions) * len(sequence)
