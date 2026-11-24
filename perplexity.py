"""Bits per residue of protein language models on proteins new to UniProt, with the speedrun's best run beside them.

OpenAI's "Perplexity and intelligence" (2026-11-22) plots one fixed held-out set's log-perplexity across its models by
release date. This is that plot for protein language models:

    python perplexity.py corpus --state-dir /state/perplexity --threads 48
    python perplexity.py fetch --state-dir /state/perplexity
    python perplexity.py score --state-dir /state/perplexity --models esm2_650m prot_t5_xl_uniref50 ...
    python perplexity.py check-served --state-dir /state/perplexity --model esm3-open-2024-03 --against esm3_small
    python perplexity.py score --state-dir /state/perplexity --models esm3_large
    python perplexity.py star --state-dir /state/perplexity --run h_auto11_bayes_01
    python perplexity.py copies --state-dir /state/perplexity --data-dir /state/data/afdb_v6
    python perplexity.py relatives --state-dir /state/perplexity --data-dir /state/data/afdb_v6 --threads 48
    python perplexity.py kmers --state-dir /state/perplexity --data-dir /state/data/afdb_v6_minus_uniprot_2026_04
    python perplexity.py plot --state-dir runs/perplexity --output assets/perplexity_frontier.png

`copies` names the training proteins identical to a benchmark protein, which the speedrun's data leaves out
(`prepare.py --exclude-from`); `relatives` measures the homology overlap, reported only; `kmers` scores lookup tables
counted on the speedrun's training stream, the figure's reference lines.

The measure. Each sampled residue is masked alone with every other residue visible, the model's probability of the true
residue is renormalized over the 20 standard residues, and -log2 of it is averaged over the corpus's sampled residues:
bits per residue, which is bits per byte of a FASTA sequence. Masking one residue gives every model the same context
whatever its objective: a masked language model reads the mask, a T5 denoiser (ProtT5, ANKH) reads it through its
decoder, so all are scored on one problem. Autoregressive models see only the left context and are left out. ESM3
large (98B) has no public weights: the Biohub API serves it, one request a masked residue (`BiohubScorer`, token
`BIOHUB_TOKEN`), and `check-served` first scores the open ESM3 that way beside its local port on the dev set.

The corpus. Every UniProtKB entry first released on or after 2026-11-01 (release 2026_04, 2026-11-04), 50 to 1,022
residues, not a fragment, of the 20 standard residues only, is clustered by MMseqs2 at 30% identity over 80% coverage.
2,000 clusters are drawn uniformly and one member uniformly from each (not the representative), then 8 positions
uniformly from each member. A disjoint 100 clusters, drawn the same way, are the dev set: it chooses a T5 model's input
format and smoke-tests every model, and never enters a reported number. Every scored model was public before
2026-06-01, so none trained on these entries; a sequence may still have close relatives in a model's training data.

The star is a speedrun run (`star`), rebuilt from its own `train.py` and `strategies/` snapshot in the private Hub
repository `Synthyra/SpeedrunningPLMs-runs` and scored the same way; rerun it when the record moves.

Symbols: K positions a protein, L residues, V a model's vocabulary, c = 20 standard residues.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
import torch

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from torch import Tensor

from probe import load_run_snapshot
from speedrunning_plms import benchmark
from speedrunning_plms.benchmark.tokens import STANDARD_RESIDUES


QUERY = "date_created:[2026-11-01 TO *] AND length:[50 TO 1022] AND fragment:false"
STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
RELEASE_DATE = "2026-11-04"  # release 2026_04, every entry the query takes
# The query's lengths in six ranges of about 0.7M entries, streamed side by side, each inside the stream's 600 s.
LENGTH_PIECES = ("50 TO 150", "151 TO 250", "251 TO 350", "351 TO 500", "501 TO 700", "701 TO 1022")
SEED = 20261123
CORPUS_CLUSTERS = 2000
DEV_CLUSTERS = 100
POSITIONS = 8  # K, residues scored a protein
MIN_IDENTITY = 0.3
COVERAGE = 0.8
BOOTSTRAP = 2000  # resamples of proteins for the 95% interval
RUNS_REPO = "Synthyra/SpeedrunningPLMs-runs"
# What `fetch` leaves on the Hub: other frameworks' weights and alternative checkpoints (ProtT5's 600k and 723k steps).
UNUSED_FILES = ["*.h5", "*.msgpack", "*.onnx", "tf_model*", "flax_model*", "*_600k.bin", "*_723k.bin", "*.ckpt"]
RESIDUE_SET = frozenset(STANDARD_RESIDUES)


@dataclass(frozen=True)
class ModelSpec:
    """How to load and query one public model. `kind` picks the scorer; `formats` are a T5 model's candidate inputs,
    the dev set choosing one."""

    model_id: str
    label: str
    family: str
    repo: str
    kind: str  # "masked", "e1", "t5" or "biohub" (served by the Biohub API, `repo` its model name)
    # "masked_lm" (AutoModelForMaskedLM), "auto" (AutoModel, ESM3's sequence head) or "bert" (BertForMaskedLM, for
    # ProtBERT, whose config.json has no model_type for the Auto classes to read)
    loader: str = "masked_lm"
    logits_field: str = "logits"
    tokenizer: str = "auto"  # "auto" (AutoTokenizer) or "vocab_file" (`VocabularyFile`, the repository's vocab.txt)
    special: dict[str, str] = field(default_factory=dict)  # cls, eos and mask token strings the tokenizer leaves unset
    formats: tuple[tuple[str, str, str], ...] = ()  # T5: (mode, mask token, prefix), mode "bart" or "sentinel"; DPLM2: (layout, "", "")


# ProtT5's card: BART-like denoising, masked residues replaced by "[MASK]", which its vocabulary may not hold.
T5_PROTT5 = (("bart", "[MASK]", ""), ("bart", "<extra_id_0>", ""), ("bart", "<unk>", ""), ("sentinel", "<extra_id_0>", ""))
T5_ANKH = (("sentinel", "<extra_id_0>", ""), ("bart", "<extra_id_0>", ""))
T5_ANKH3 = (("sentinel", "<extra_id_0>", "[NLU]"), ("bart", "<extra_id_0>", "[NLU]"), ("sentinel", "<extra_id_0>", ""))
DPLM2_SPECIAL = {"cls": "<cls_aa>", "eos": "<eos_aa>", "mask": "<mask_aa>"}
# DPLM2's card shows the structure track (all masks here: no structure is given) before the amino-acid track, its
# runtime's packed-layout check reads amino acids first, and it runs a lone amino-acid track as a plain protein.
DPLM2_LAYOUTS = (("aa_only", "", ""), ("structure_first", "", ""), ("aa_first", "", ""))

MODELS: dict[str, ModelSpec] = {spec.model_id: spec for spec in (
    ModelSpec("esm1b", "ESM-1b", "ESM", "biohub/esm1b_t33_650M_UR50S", "masked"),
    ModelSpec("esm1v", "ESM-1v", "ESM", "biohub/esm1v_t33_650M_UR90S_1", "masked"),
    ModelSpec("esm2_8m", "ESM2 8M", "ESM", "Synthyra/ESM2-8M", "masked"),
    ModelSpec("esm2_35m", "ESM2 35M", "ESM", "Synthyra/ESM2-35M", "masked"),
    ModelSpec("esm2_150m", "ESM2 150M", "ESM", "Synthyra/ESM2-150M", "masked"),
    ModelSpec("esm2_650m", "ESM2 650M", "ESM", "Synthyra/ESM2-650M", "masked"),
    ModelSpec("esm2_3b", "ESM2 3B", "ESM", "Synthyra/ESM2-3B", "masked"),
    ModelSpec("esm2_15b", "ESM2 15B", "ESM", "biohub/esm2_t48_15B_UR50D", "masked"),
    ModelSpec("esmc_300m", "ESMC 300M", "ESMC", "Synthyra/ESMplusplus_small", "masked"),
    ModelSpec("esmc_600m", "ESMC 600M", "ESMC", "Synthyra/ESMplusplus_large", "masked"),
    ModelSpec("esmc_6b", "ESMC 6B", "ESMC", "Synthyra/ESMplusplus_6B", "masked"),
    ModelSpec("esm3_small", "ESM3 open", "ESMC", "Synthyra/ESM3_small", "masked", loader="auto", logits_field="sequence_logits"),
    ModelSpec("esm3_large", "ESM3 large", "ESMC", "esm3-large-2024-03", "biohub"),
    ModelSpec("prot_bert", "ProtBERT", "ProtTrans", "Rostlab/prot_bert", "masked", loader="bert", tokenizer="vocab_file"),
    ModelSpec("prot_bert_bfd", "ProtBERT-BFD", "ProtTrans", "Rostlab/prot_bert_bfd", "masked", loader="bert", tokenizer="vocab_file"),
    ModelSpec("prot_t5_xl_uniref50", "ProtT5-XL-U50", "ProtTrans", "Rostlab/prot_t5_xl_uniref50", "t5", formats=T5_PROTT5),
    ModelSpec("prot_t5_xl_bfd", "ProtT5-XL-BFD", "ProtTrans", "Rostlab/prot_t5_xl_bfd", "t5", formats=T5_PROTT5),
    ModelSpec("ankh_base", "ANKH base", "ANKH", "ElnaggarLab/ankh-base", "t5", formats=T5_ANKH),
    ModelSpec("ankh_large", "ANKH large", "ANKH", "ElnaggarLab/ankh-large", "t5", formats=T5_ANKH),
    ModelSpec("ankh2_large", "ANKH2 large", "ANKH", "ElnaggarLab/ankh2-ext2", "t5", formats=T5_ANKH),
    ModelSpec("ankh3_large", "ANKH3 large", "ANKH", "ElnaggarLab/ankh3-large", "t5", formats=T5_ANKH3),
    ModelSpec("ankh3_xl", "ANKH3 XL", "ANKH", "ElnaggarLab/ankh3-xl", "t5", formats=T5_ANKH3),
    ModelSpec("dplm_150m", "DPLM 150M", "DPLM", "Synthyra/DPLM-150M", "masked"),
    ModelSpec("dplm_650m", "DPLM 650M", "DPLM", "Synthyra/DPLM-650M", "masked"),
    ModelSpec("dplm_3b", "DPLM 3B", "DPLM", "Synthyra/DPLM-3B", "masked"),
    ModelSpec("dplm2_150m", "DPLM2 150M", "DPLM", "Synthyra/DPLM2-150M", "masked", special=DPLM2_SPECIAL, formats=DPLM2_LAYOUTS),
    ModelSpec("dplm2_650m", "DPLM2 650M", "DPLM", "Synthyra/DPLM2-650M", "masked", special=DPLM2_SPECIAL, formats=DPLM2_LAYOUTS),
    ModelSpec("dplm2_3b", "DPLM2 3B", "DPLM", "Synthyra/DPLM2-3B", "masked", special=DPLM2_SPECIAL, formats=DPLM2_LAYOUTS),
    ModelSpec("e1_150m", "E1 150M", "E1", "Synthyra/Profluent-E1-150M", "e1"),
    ModelSpec("e1_300m", "E1 300M", "E1", "Synthyra/Profluent-E1-300M", "e1"),
    ModelSpec("e1_600m", "E1 600M", "E1", "Synthyra/Profluent-E1-600M", "e1"),
)}
# When each model was introduced, read on 2026-11-23 from the source beside it: a paper's first version, or the release
# note of the weights where the paper predates them. Month precision where the source gives no day.
RELEASES: dict[str, tuple[str, str]] = {
    "esm1b": ("2020-12", "github.com/facebookresearch/esm news: Dec 2020, ESM-1b added"),
    "esm1v": ("2021-07", "github.com/facebookresearch/esm news: July 2021, ESM-1v released"),
    **{f"esm2_{size}": ("2022-08", "github.com/facebookresearch/esm news: August 2022, ESM-2 released, 15B included")
       for size in ("8m", "35m", "150m", "650m", "3b", "15b")},
    **{f"esmc_{size}": ("2024-12-04", "evolutionaryscale.ai/blog/esm-cambrian, dated 12.4.24 (the site writes month first)")
       for size in ("300m", "600m", "6b")},
    "esm3_small": ("2024-06-25", "evolutionaryscale.ai/blog/esm3-release, dated 6.25.24: open weights released"),
    "esm3_large": ("2024-06-25", "evolutionaryscale.ai/blog/esm3-release, dated 6.25.24: the 98B model announced, served only by API"),
    "prot_bert": ("2020-07-12", "bioRxiv 10.1101/2020.07.12.199554 v1 (BERT on UniRef100 and BFD)"),
    "prot_bert_bfd": ("2020-07-12", "bioRxiv 10.1101/2020.07.12.199554 v1 (BERT on UniRef100 and BFD)"),
    "prot_t5_xl_uniref50": ("2021-05-04", "bioRxiv 10.1101/2020.07.12.199554 v3, the first version naming T5"),
    "prot_t5_xl_bfd": ("2021-05-04", "bioRxiv 10.1101/2020.07.12.199554 v3, the first version naming T5"),
    "ankh_base": ("2023-01-16", "arXiv 2301.06568 v1"),
    "ankh_large": ("2023-01-16", "arXiv 2301.06568 v1"),
    "ankh2_large": ("2024-07-07", "no paper; the Hub repository ElnaggarLab/ankh2-ext2 was created 2024-07-07"),
    "ankh3_large": ("2025-05-26", "arXiv 2505.20052 v1"),
    "ankh3_xl": ("2025-05-26", "arXiv 2505.20052 v1"),
    **{f"dplm_{size}": ("2024-02-28", "arXiv 2402.18567 v1") for size in ("150m", "650m", "3b")},
    **{f"dplm2_{size}": ("2024-10-17", "arXiv 2410.13782 v1") for size in ("150m", "650m", "3b")},
    **{f"e1_{size}": ("2025-11", "bioRxiv DOI 10.1101/2025.11.12, linked from github.com/Profluent-AI/E1") for size in ("150m", "300m", "600m")},
}
assert RELEASES.keys() == MODELS.keys(), sorted(RELEASES.keys() ^ MODELS.keys())
# A served model's size as its provider states it, read on 2026-11-24 from docs.biohub.ai/models/esm3 (Variants).
SERVED_PARAMETERS = {"esm3_large": 98_000_000_000}
assert SERVED_PARAMETERS.keys() == {model_id for model_id, spec in MODELS.items() if spec.kind == "biohub"}

# Left out: AMPLIFY (its remote code imports xformers, which has no aarch64 wheel for the GH200), SaProt (its tokens pair
# each residue with a structure letter), ProGen2 and ProtGPT2 (autoregressive: left context only).


@dataclass(frozen=True)
class Protein:
    accession: str
    sequence: str
    positions: tuple[int, ...]  # K 0-based residue indices
    cluster: str  # the cluster's MMseqs2 representative
    cluster_size: int
    organism: str
    taxid: int


# The corpus.

def piece_query(lengths: str) -> str:
    return QUERY.replace("length:[50 TO 1022]", f"length:[{lengths}]")


def expected_entries(query: str) -> tuple[int, str]:
    """The search endpoint's count of a query, and the release it counts in; the stream endpoint sends neither."""
    url = f"{SEARCH_URL}?{urllib.parse.urlencode({'query': query, 'size': '1', 'fields': 'accession'})}"
    with urllib.request.urlopen(url, timeout=120) as response:  # nosec: a fixed https URL
        return int(response.headers["X-Total-Results"]), response.headers["X-UniProt-Release"]


def download_piece(lengths: str, raw: Path) -> dict[str, object]:
    """One length range's entries as gzipped FASTA, its entry count checked against the search endpoint's: the stream
    can end at a gzip member boundary (2026-11-23: 2,650,300 of 4,204,117 entries after 603 s) and still read whole."""
    query = piece_query(lengths)
    expected, release = expected_entries(query)
    url = f"{STREAM_URL}?{urllib.parse.urlencode({'query': query, 'format': 'fasta', 'compressed': 'true'})}"
    partial = raw.with_suffix(".partial")
    started = time.time()
    with urllib.request.urlopen(url, timeout=600) as response, partial.open("wb") as handle:  # nosec: a fixed https URL
        assert response.headers["X-UniProt-Release"] == release, "the release changed between the count and the stream"
        while chunk := response.read(1 << 22):
            handle.write(chunk)
    entries = sum(1 for _ in fasta_records(partial))
    if entries != expected:
        raise RuntimeError(f"lengths {lengths}: streamed {entries} entries, the query has {expected}")
    partial.rename(raw)
    return {"lengths": lengths, "query": query, "entries": entries, "release": release, "seconds": round(time.time() - started, 1)}


def fasta_records(path: Path) -> Iterator[tuple[str, str]]:
    """(header, sequence) pairs of a FASTA file, gzipped or not (read from its first bytes, not its name)."""
    with path.open("rb") as handle:
        opener = gzip.open if handle.read(2) == b"\x1f\x8b" else open
    header, parts = None, []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts)


def header_fields(header: str) -> tuple[str, str, int]:
    """Accession, organism and taxonomy id of a UniProtKB FASTA header (`db|ACC|NAME desc OS=... OX=... ...`)."""
    accession = header.split("|")[1]
    organism = header.split(" OS=")[1].split(" OX=")[0]
    taxid = int(header.split(" OX=")[1].split(" ")[0])
    return accession, organism, taxid


def build_corpus(state_dir: Path, threads: int) -> None:
    from concurrent.futures import ThreadPoolExecutor

    state_dir.mkdir(parents=True, exist_ok=True)
    raws = [state_dir / f"uniprot_new_{index}.fasta.gz" for index in range(len(LENGTH_PIECES))]
    pool = state_dir / "pool.fasta"
    download_path = state_dir / "download.json"
    if not download_path.exists():
        expected, release = expected_entries(QUERY)
        with ThreadPoolExecutor(len(LENGTH_PIECES)) as executor:  # each piece well under the stream's 600 s
            pieces = list(executor.map(download_piece, LENGTH_PIECES, raws))
        assert sum(piece["entries"] for piece in pieces) == expected, f"the pieces hold {sum(piece['entries'] for piece in pieces)}, the query {expected}"
        assert {piece["release"] for piece in pieces} == {release}
        download_path.write_text(json.dumps({"query": QUERY, "url": STREAM_URL, "release": release, "entries": expected,
                                             "pieces": pieces, "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                            indent=2) + "\n", encoding="utf-8")
    headers = json.loads(download_path.read_text(encoding="utf-8"))
    meta: dict[str, tuple[str, int]] = {}
    sequences: dict[str, str] = {}
    entries, counts = 0, np.zeros(len(STANDARD_RESIDUES), dtype=np.int64)  # (c,)
    index = {residue: column for column, residue in enumerate(STANDARD_RESIDUES)}
    for raw in raws:
        for header, sequence in fasta_records(raw):
            entries += 1
            if not set(sequence) <= RESIDUE_SET:
                continue
            accession, organism, taxid = header_fields(header)
            meta[accession] = (organism, taxid)
            sequences[accession] = sequence
            counts += np.bincount([index[residue] for residue in sequence], minlength=len(STANDARD_RESIDUES))
    assert entries == headers["entries"], f"read {entries} entries, downloaded {headers['entries']}"
    with pool.open("w", encoding="utf-8") as handle:
        for accession in sorted(sequences):
            handle.write(f">{accession}\n{sequences[accession]}\n")
    clusters_tsv = state_dir / "clusters_cluster.tsv"
    if not clusters_tsv.exists():
        subprocess.run(["mmseqs", "easy-cluster", str(pool), str(state_dir / "clusters"), str(state_dir / "mmseqs_tmp"),
                        "--min-seq-id", str(MIN_IDENTITY), "-c", str(COVERAGE), "--threads", str(threads)], check=True)
    members: dict[str, list[str]] = {}
    with clusters_tsv.open(encoding="utf-8") as handle:
        for line in handle:
            representative, member = line.split()
            members.setdefault(representative, []).append(member)
    assert sum(len(group) for group in members.values()) == len(sequences), "the clusters do not cover the pool"
    rng = np.random.default_rng(SEED)
    representatives = sorted(members)
    order = rng.permutation(len(representatives))  # (clusters,)
    chosen = [representatives[int(position)] for position in order[:DEV_CLUSTERS + CORPUS_CLUSTERS]]
    proteins = []
    for representative in chosen:
        group = sorted(members[representative])
        accession = group[int(rng.integers(len(group)))]  # uniform over the cluster's members, the representative no likelier
        sequence = sequences[accession]
        positions = tuple(sorted(int(position) for position in rng.choice(len(sequence), POSITIONS, replace=False)))
        organism, taxid = meta[accession]
        proteins.append(Protein(accession, sequence, positions, representative, len(group), organism, taxid))
    write_proteins(state_dir / "dev.jsonl", proteins[:DEV_CLUSTERS])
    write_proteins(state_dir / "corpus.jsonl", proteins[DEV_CLUSTERS:])
    version = subprocess.run(["mmseqs", "version"], check=True, capture_output=True, text=True).stdout.strip()
    frequencies = counts / counts.sum()  # (c,)
    manifest = {
        "query": QUERY, "release": headers["release"], "release_date": RELEASE_DATE,
        "downloaded_utc": headers["downloaded_utc"], "entries": entries, "standard_only": len(sequences),
        "clusters": len(members), "min_identity": MIN_IDENTITY, "coverage": COVERAGE, "mmseqs": version, "seed": SEED,
        "corpus_proteins": CORPUS_CLUSTERS, "dev_proteins": DEV_CLUSTERS, "positions": POSITIONS,
        "pool_residue_frequency": dict(zip(STANDARD_RESIDUES, frequencies.round(6).tolist(), strict=True)),
        "pool_unigram_bits": float(-(frequencies * np.log2(frequencies)).sum()),
        "corpus_sha256": sha256(state_dir / "corpus.jsonl"), "dev_sha256": sha256(state_dir / "dev.jsonl"),
    }
    (state_dir / "corpus.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("release", "entries", "standard_only", "clusters", "pool_unigram_bits")}), flush=True)


def write_proteins(path: Path, proteins: list[Protein]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for protein in proteins:
            handle.write(json.dumps({"accession": protein.accession, "sequence": protein.sequence, "positions": list(protein.positions),
                                     "cluster": protein.cluster, "cluster_size": protein.cluster_size, "organism": protein.organism,
                                     "taxid": protein.taxid}) + "\n")


def read_proteins(path: Path) -> list[Protein]:
    proteins = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        proteins.append(Protein(row["accession"], row["sequence"], tuple(row["positions"]), row["cluster"], row["cluster_size"],
                                row["organism"], row["taxid"]))
    return proteins


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The speedrun's training data and this benchmark's 2,100 proteins (dev and corpus). `copies` names the training
# proteins whose sequence is one of the benchmark's, the ones the speedrun's data leaves out: no compared model's
# training data is homology-disjoint from these proteins, so the speedrun's need not be either (Logan, 2026-11-23).
# `relatives` measures the homology overlap, reported and never removed: the workspace's rule for a benchmark kept
# clean (docs/conventions/dataset_biases.md), identity at least 30% over half of either sequence at e-value 1e-3,
# searched in both directions and once with no prefilter, the union flagged, and stricter cutoffs beside it.

def exact_copies(state_dir: Path, data_dir: Path) -> None:
    """Write `exclusion.json`: the stream indices of training proteins identical to a benchmark protein, for
    `prepare.py --exclude-from`."""
    proteins = read_proteins(state_dir / "dev.jsonl") + read_proteins(state_dir / "corpus.jsonl")
    accessions: dict[str, list[str]] = {}
    for protein in proteins:
        accessions.setdefault(protein.sequence, []).append(protein.accession)
    stream = benchmark.load_train(data_dir)
    lookup = np.full(benchmark.tokens.VOCAB_SIZE, ord("X"), dtype=np.uint8)  # (V,)
    for token_id_, token in enumerate(benchmark.tokens.ESM2_TOKENS):
        if len(token) == 1:
            lookup[token_id_] = ord(token)
    letters = lookup[np.asarray(stream.tokens)]  # (t,)
    copies: dict[int, list[str]] = {}
    for index in range(len(stream.offsets) - 1):
        sequence = letters[int(stream.offsets[index]) + 1 : int(stream.offsets[index + 1]) - 1].tobytes().decode("ascii")
        if sequence in accessions:
            copies[index] = accessions[sequence]
    manifest = json.loads((state_dir / "corpus.json").read_text(encoding="utf-8"))
    exclusion = {"rule": "an exact copy: a training protein whose sequence is a temporal benchmark protein's",
                 "copied_benchmark_proteins": len({accession for found in copies.values() for accession in found}),
                 "copies": {str(index): found for index, found in sorted(copies.items())},  # stream index -> benchmark accessions
                 "benchmark": {"name": "new UniProt proteins", "release": manifest["release"], "query": manifest["query"],
                               "proteins": len(proteins), "corpus_sha256": manifest["corpus_sha256"], "dev_sha256": manifest["dev_sha256"]},
                 "indices": sorted(copies)}
    (state_dir / "exclusion.json").write_text(json.dumps(exclusion) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in exclusion.items() if key != "indices"} | {"training_copies": len(copies)}), flush=True)


RELATIVE_IDENTITY = 0.3
RELATIVE_COVERAGE = 0.5  # of either sequence: a long chain holding a short benchmark protein still counts
RELATIVE_EVALUE = 1e-3
RELATIVE_SEARCHES = ("reverse", "forward", "exhaustive")  # training -> benchmark; benchmark -> training, k-mer; and exhaustive
HIT_FIELDS = "query,target,fident,qcov,tcov,evalue"
# Other cutoffs reported beside the rule, so the cost of each is on record: (name, identity, coverage, of both).
RELATIVE_CUTOFFS = (("rule", RELATIVE_IDENTITY, RELATIVE_COVERAGE, False), ("cluster_rule", MIN_IDENTITY, COVERAGE, True),
                    ("identity_50", 0.5, RELATIVE_COVERAGE, False), ("identity_90", 0.9, RELATIVE_COVERAGE, False))


def write_training_fasta(data_dir: Path, path: Path) -> int:
    """The prepared training stream decoded to FASTA, each protein named by its stream index; returns the count."""
    stream = benchmark.load_train(data_dir)
    lookup = np.full(benchmark.tokens.VOCAB_SIZE, ord("X"), dtype=np.uint8)  # (V,)
    for token_id, token in enumerate(benchmark.tokens.ESM2_TOKENS):
        if len(token) == 1:
            lookup[token_id] = ord(token)
    letters = lookup[np.asarray(stream.tokens)]  # (t,)
    proteins = len(stream.offsets) - 1
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for index in range(proteins):
            start, end = int(stream.offsets[index]), int(stream.offsets[index + 1])
            handle.write(f">{index}\n{letters[start + 1 : end - 1].tobytes().decode('ascii')}\n")  # CLS and EOS dropped
    return proteins


def search_relatives(state_dir: Path, data_dir: Path, threads: int, searches: list[str]) -> None:
    """Run the named searches, then flag every training protein whose best hit passes each cutoff (the union of searches)."""
    directory = state_dir / "relatives"
    directory.mkdir(exist_ok=True)
    proteins = read_proteins(state_dir / "dev.jsonl") + read_proteins(state_dir / "corpus.jsonl")
    benchmark_fasta, training_fasta = directory / "benchmark.fasta", directory / "train.fasta"
    if not benchmark_fasta.exists():
        benchmark_fasta.write_text("".join(f">{protein.accession}\n{protein.sequence}\n" for protein in proteins), encoding="ascii")
    if not training_fasta.exists():
        partial = training_fasta.with_suffix(".partial")
        write_training_fasta(data_dir, partial)
        partial.rename(training_fasta)
    training_count = int(np.fromfile(data_dir / "train_offsets.i64", dtype=np.int64).size - 1)
    for name in searches:
        hits = directory / f"{name}.tsv"
        if hits.exists():
            continue
        query, target = (training_fasta, benchmark_fasta) if name == "reverse" else (benchmark_fasta, training_fasta)
        command = ["mmseqs", "easy-search", str(query), str(target), str(hits.with_suffix(".partial")), str(directory / f"tmp_{name}"),
                   "--threads", str(threads), "-e", str(RELATIVE_EVALUE), "-s", "7.5", "--alignment-mode", "3",
                   "--format-output", HIT_FIELDS,
                   "--max-seqs", str(len(proteins) if name == "reverse" else training_count)]  # never cap a family's relatives
        if name == "exhaustive":
            command += ["--exhaustive-search", "1"]
        started = time.perf_counter()
        subprocess.run(command, check=True)
        hits.with_suffix(".partial").rename(hits)
        print(f"{name}: {time.perf_counter() - started:.0f} s", flush=True)
    version = subprocess.run(["mmseqs", "version"], check=True, capture_output=True, text=True).stdout.strip()
    summarize_relatives(state_dir, data_dir, training_count, len(proteins), version)


def summarize_relatives(state_dir: Path, data_dir: Path, training_count: int, benchmark_count: int, mmseqs: str) -> None:
    directory = state_dir / "relatives"
    offsets = np.fromfile(data_dir / "train_offsets.i64", dtype=np.int64)  # (n_train + 1,)
    residues = np.diff(offsets) - 2  # (n_train,)
    done = [name for name in RELATIVE_SEARCHES if (directory / f"{name}.tsv").exists()]
    flagged: dict[str, dict[str, set]] = {cutoff: {"training": set(), "benchmark": set()} for cutoff, *_ in RELATIVE_CUTOFFS}
    by_search: dict[str, dict[str, int]] = {}
    best: dict[int, tuple[float, str]] = {}  # training index -> (identity, benchmark accession) of the rule's best hit
    for name in done:
        reverse = name == "reverse"
        found = {cutoff: set() for cutoff, *_ in RELATIVE_CUTOFFS}
        with (directory / f"{name}.tsv").open(encoding="ascii") as handle:
            for line in handle:
                query, target, identity, query_cover, target_cover, evalue = line.split("\t")
                training, accession = (int(query), target) if reverse else (int(target), query)
                identity, covers = float(identity), (float(query_cover), float(target_cover))
                if float(evalue) > RELATIVE_EVALUE:
                    continue
                for cutoff, minimum_identity, minimum_cover, both in RELATIVE_CUTOFFS:
                    cover = min(covers) if both else max(covers)
                    if identity >= minimum_identity and cover >= minimum_cover:
                        found[cutoff].add(training)
                        flagged[cutoff]["training"].add(training)
                        flagged[cutoff]["benchmark"].add(accession)
                        if cutoff == "rule" and identity > best.get(training, (0.0, ""))[0]:
                            best[training] = (identity, accession)
        by_search[name] = {cutoff: len(indices) for cutoff, indices in found.items()}
    report = {"searches": done, "mmseqs": mmseqs, "training_proteins": training_count, "training_residues": int(residues.sum()),
              "benchmark_proteins": benchmark_count, "by_search": by_search, "cutoffs": {}}
    for cutoff, minimum_identity, minimum_cover, both in RELATIVE_CUTOFFS:
        indices = np.fromiter(flagged[cutoff]["training"], dtype=np.int64)  # (flagged,)
        report["cutoffs"][cutoff] = {
            "identity": minimum_identity, "coverage": minimum_cover, "coverage_of": "both" if both else "either",
            "training_flagged": int(indices.size), "training_fraction": indices.size / training_count,
            "residue_fraction": float(residues[indices].sum() / residues.sum()) if indices.size else 0.0,
            "benchmark_with_relative": len(flagged[cutoff]["benchmark"]),
        }
    (directory / "train_relatives.tsv").write_text(
        "".join(f"{index}\t{identity:.3f}\t{accession}\n" for index, (identity, accession) in sorted(best.items())), encoding="ascii")
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


# The scorers: each returns ln p(true residue) over the 20 standard residues at a protein's K positions.

def residue_log_probs(logits: Tensor, residue_ids: Tensor, sequence: str, positions: tuple[int, ...]) -> np.ndarray:
    """The K masked rows' log-probabilities of their true residues, renormalized over the 20 residue columns."""
    # logits: (K, V), one row a masked position; residue_ids: (c,); returns (K,).
    rows = logits[:, residue_ids].float().log_softmax(dim=-1)  # (K, c)
    truth = torch.tensor([STANDARD_RESIDUES.index(sequence[position]) for position in positions], device=rows.device)  # (K,)
    return rows[torch.arange(len(positions), device=rows.device), truth].cpu().double().numpy()  # (K,)


class MaskedScorer:
    """An encoder whose tokenizer holds one token a residue: [cls] residues [eos], one residue masked a row."""

    def __init__(self, spec: ModelSpec, revision: str, device: torch.device, form: tuple[str, str, str] | None = None) -> None:
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, BertForMaskedLM

        # DPLM2's layout: the amino-acid track alone, or beside a structure track of masks, structure or amino acids first.
        self.spec, self.device, self.layout = spec, device, form[0] if form else "aa_only"
        if spec.tokenizer == "vocab_file":
            tokenizer: object = VocabularyFile(spec.repo, revision)
        else:
            tokenizer = AutoTokenizer.from_pretrained(spec.repo, revision=revision, trust_remote_code=True)
        model_class = {"masked_lm": AutoModelForMaskedLM, "auto": AutoModel, "bert": BertForMaskedLM}[spec.loader]
        # FP32 parameters under BF16 autocast for every model: DPLM2's inference contract, and one precision for all.
        self.model = model_class.from_pretrained(spec.repo, revision=revision, trust_remote_code=True, dtype=torch.float32).to(device).eval()
        special = {"cls": tokenizer.cls_token or tokenizer.bos_token, "eos": tokenizer.eos_token or tokenizer.sep_token,
                   "mask": tokenizer.mask_token, **spec.special}
        self.ids = {name: token_id(tokenizer, token) for name, token in special.items()}
        if self.layout != "aa_only":
            self.structure = [token_id(tokenizer, token) for token in ("<cls_struct>", "<mask_struct>", "<eos_struct>")]
        self.residue_ids = torch.tensor([token_id(tokenizer, residue) for residue in STANDARD_RESIDUES], device=device)  # (c,)
        assert len(set(self.residue_ids.tolist())) == len(STANDARD_RESIDUES), f"{spec.model_id}: residues share tokens"

    def __call__(self, sequence: str, positions: tuple[int, ...]) -> np.ndarray:
        lookup = dict(zip(STANDARD_RESIDUES, self.residue_ids.tolist(), strict=True))
        residues = [self.ids["cls"], *(lookup[residue] for residue in sequence), self.ids["eos"]]  # L + 2
        offset = 0
        if self.layout != "aa_only":
            cls_struct, mask_struct, eos_struct = self.structure
            structure = [cls_struct, *([mask_struct] * len(sequence)), eos_struct]  # L + 2, every structure token unknown
            offset = len(structure) if self.layout == "structure_first" else 0
            residues = structure + residues if self.layout == "structure_first" else residues + structure
        ids = torch.tensor(residues, device=self.device)  # (T,), T = L + 2, or 2 (L + 2) with the structure track
        rows = torch.arange(len(positions), device=self.device)  # (K,)
        columns = torch.tensor(positions, device=self.device) + 1 + offset  # (K,), past the cls token
        batch = ids.repeat(len(positions), 1)  # (K, L + 2)
        batch[rows, columns] = self.ids["mask"]
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            output = self.model(input_ids=batch, attention_mask=torch.ones_like(batch))
        logits = getattr(output, self.spec.logits_field)  # (K, T, V)
        return residue_log_probs(logits[rows, columns], self.residue_ids, sequence, positions)  # (K,)


def restore_past_key_values(prefix: str) -> int:
    """Give back the plain dataclass constructor to every model-output class under modules named `prefix...` that still
    declares a `past_key_values` field; returns how many.

    Transformers 5.24 wraps each ModelOutput subclass's constructor to rename `past_key_values` to `cache`, which breaks
    remote code that declares the old field (the FastPLMs E1 port: "unexpected keyword argument 'cache'"). The wrapper
    keeps the dataclass constructor as `__wrapped__`; `type.__setattr__` sets it back without the metaclass rewrapping it.
    ModelOutput's own `__setattr__` then files the value under `cache`; the scorers read only logits and hidden states.
    """
    import dataclasses
    import inspect
    import sys

    from transformers.utils import ModelOutput

    restored = 0
    for name, module in list(sys.modules.items()):
        if not name.startswith(prefix):
            continue
        for value in list(vars(module).values()):
            if isinstance(value, type) and issubclass(value, ModelOutput) and dataclasses.is_dataclass(value) \
                    and "past_key_values" in {field.name for field in dataclasses.fields(value)} and hasattr(value.__init__, "__wrapped__"):
                type.__setattr__(value, "__init__", inspect.unwrap(value.__init__))
                restored += 1
    return restored


class E1Scorer:
    """Profluent E1 in single-sequence mode: its own preparation (<bos> 1 residues 2 <eos>), "?" the mask."""

    def __init__(self, spec: ModelSpec, revision: str, device: torch.device) -> None:
        from transformers import AutoModelForMaskedLM

        self.device = device
        self.model = AutoModelForMaskedLM.from_pretrained(spec.repo, revision=revision, trust_remote_code=True,
                                                          dtype=torch.float32).to(device).eval()
        restore_past_key_values("fastplms.")
        self.prepare = self.model.prep_tokens
        self.residue_ids = torch.tensor([self.prepare.vocab[residue] for residue in STANDARD_RESIDUES], device=device)  # (c,)

    def __call__(self, sequence: str, positions: tuple[int, ...]) -> np.ndarray:
        masked = [sequence[:position] + self.prepare.mask_token + sequence[position + 1:] for position in positions]
        batch = self.prepare.get_batch_kwargs(masked, device=self.device)
        rows = torch.arange(len(positions), device=self.device)  # (K,)
        columns = torch.tensor(positions, device=self.device) + 2  # (K,), past <bos> and "1"
        assert bool((batch["input_ids"][rows, columns] == self.prepare.mask_token_id).all()), "E1's layout moved"
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            output = self.model(input_ids=batch["input_ids"], within_seq_position_ids=batch["within_seq_position_ids"],
                                global_position_ids=batch["global_position_ids"], sequence_ids=batch["sequence_ids"])
        return residue_log_probs(output.logits[rows, columns], self.residue_ids, sequence, positions)  # (K,)


class T5Scorer:
    """A T5 denoiser. `bart`: the decoder rebuilds the whole sequence, teacher-forced, and reads the masked residue's
    step; `sentinel`: the decoder writes the sentinel then the residue. Either way only the one residue is hidden."""

    def __init__(self, spec: ModelSpec, revision: str, device: torch.device, form: tuple[str, str, str]) -> None:
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.device, (self.mode, mask, prefix) = device, form
        tokenizer = AutoTokenizer.from_pretrained(spec.repo, revision=revision)
        self.mask_id = token_id(tokenizer, mask)  # before the weights load: a format whose mask the vocabulary lacks ends here
        self.model = T5ForConditionalGeneration.from_pretrained(spec.repo, revision=revision, dtype=torch.float32).to(device).eval()
        self.residue_ids = torch.tensor(residue_token_ids(tokenizer), device=device)  # (c,)
        self.prefix = [token_id(tokenizer, prefix)] if prefix else []  # one token ("[NLU]"), read as residues are
        self.eos_id, self.start_id = tokenizer.eos_token_id, self.model.config.decoder_start_token_id

    def __call__(self, sequence: str, positions: tuple[int, ...]) -> np.ndarray:
        lookup = dict(zip(STANDARD_RESIDUES, self.residue_ids.tolist(), strict=True))
        residues = [lookup[residue] for residue in sequence]
        k, offset = len(positions), len(self.prefix)
        encoder = torch.tensor([*self.prefix, *residues, self.eos_id], device=self.device).repeat(k, 1)  # (K, P + L + 1)
        rows = torch.arange(k, device=self.device)  # (K,)
        encoder[rows, torch.tensor(positions, device=self.device) + offset] = self.mask_id
        if self.mode == "bart":
            decoder = torch.tensor([self.start_id, *residues], device=self.device).repeat(k, 1)  # (K, L + 1)
            steps = torch.tensor(positions, device=self.device)  # (K,): step i's logits predict residue i
        else:
            decoder = torch.tensor([self.start_id, self.mask_id], device=self.device).repeat(k, 1)  # (K, 2)
            steps = torch.ones(k, dtype=torch.long, device=self.device)  # (K,): after the sentinel
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            logits = self.model(input_ids=encoder, attention_mask=torch.ones_like(encoder), decoder_input_ids=decoder).logits  # (K, T, V)
        return residue_log_probs(logits[rows, steps], self.residue_ids, sequence, positions)  # (K,)


class VocabularyFile:
    """A BERT vocabulary read from its `vocab.txt`, one token a line, the line its id: what the scorers need of a
    tokenizer, for ProtBERT, whose fast tokenizer transformers 5 cannot build from that file alone."""

    unk_token, cls_token, eos_token, mask_token, bos_token, sep_token = "[UNK]", "[CLS]", "[SEP]", "[MASK]", None, "[SEP]"

    def __init__(self, repo: str, revision: str) -> None:
        from huggingface_hub import hf_hub_download

        lines = Path(hf_hub_download(repo, "vocab.txt", revision=revision)).read_text(encoding="utf-8").splitlines()
        self.ids = {token: index for index, token in enumerate(lines)}
        assert len(self.ids) == len(lines), f"{repo}'s vocab.txt repeats a token"
        self.unk_token_id = self.ids[self.unk_token]

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.ids.get(token)


def token_id(tokenizer: object, token: str) -> int:
    """A token string's id; raises when the vocabulary lacks it instead of reading it as unknown."""
    identifier = tokenizer.convert_tokens_to_ids(token)  # type: ignore[attr-defined]
    unknown = tokenizer.unk_token_id  # type: ignore[attr-defined]
    if identifier is None or (identifier == unknown and token != tokenizer.unk_token):  # type: ignore[attr-defined]
        raise KeyError(f"the vocabulary has no {token!r}")
    return int(identifier)


def residue_token_ids(tokenizer: object) -> list[int]:
    """Each residue's own vocabulary entry: sentencepiece's word-initial "▁A" (ProtT5) or the bare letter (ANKH 1 to 3).
    Read from the vocabulary, not by tokenizing: transformers 5 puts an <unk> before each of ANKH's bare letters."""
    vocabulary = tokenizer.get_vocab()  # type: ignore[attr-defined]
    ids = []
    for residue in STANDARD_RESIDUES:
        forms = [form for form in ("▁" + residue, residue) if form in vocabulary]
        assert len(forms) == 1, f"the vocabulary holds {forms or 'no form'} of {residue}, so a residue's token is ambiguous"
        ids.append(vocabulary[forms[0]])
    return ids


# ESM3's sequence vocabulary (`esm/utils/constants/esm3.py` in github.com/Biohub/esm): cls, pad, eos, unk, the residues.
ESM3_SEQUENCE_VOCAB = ("<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N",
                       "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "|", "<mask>")
ESM3_BOS, ESM3_EOS, ESM3_MASK = 0, 2, 32
# The sequence track only; the service refuses or ignores the others (docs.biohub.ai/api/protein/models/logits).
SERVED_LOGITS_CONFIG = {"sequence": True, "structure": False, "secondary_structure": False, "sasa": False, "function": False,
                        "residue_annotations": False, "return_embeddings": False, "return_mean_embedding": False,
                        "return_hidden_states": False, "return_mean_hidden_states": False, "ith_hidden_layer": -1}
SERVED_ATTEMPTS = 8


class BiohubScorer:
    """A model the Biohub API serves and nobody can download (ESM3 large): one `logits` request a masked residue, every
    other residue visible, the row a local model's scorer reads. Each protein's K log-probabilities are appended to
    `cache` as they land, so an interrupted run resumes without paying for a request twice; `tokens_used` sums what the
    service reported billing (one token a residue, measured 2026-11-24).

    `workers` proteins are requested at once (`prefetch`), so K x workers requests are in flight. Biohub publishes no
    limit for `logits`, so its answers set the pace: a 429 or a server error pauses every request for its Retry-After
    (else a doubling wait), each is printed and counted in `retries`, and a run that sees them should use fewer workers."""

    def __init__(self, spec: ModelSpec, cache: Path, workers: int = 1) -> None:
        from foundry.biohub import BIOHUB_TOKEN_ENV_KEY, DEFAULT_BIOHUB_URL
        from foundry.secrets_env import load_secrets_env

        load_secrets_env(names=[BIOHUB_TOKEN_ENV_KEY])
        assert os.environ.get(BIOHUB_TOKEN_ENV_KEY), f"{BIOHUB_TOKEN_ENV_KEY} is not set"
        assert workers >= 1, workers
        self.spec, self.cache, self.token_name, self.workers = spec, cache, BIOHUB_TOKEN_ENV_KEY, workers
        self.url = f"{DEFAULT_BIOHUB_URL}/api/v1/logits"
        self.residue_ids = torch.tensor([ESM3_SEQUENCE_VOCAB.index(residue) for residue in STANDARD_RESIDUES])  # (c,)
        self.lock = threading.Lock()  # the cache file, the counts and the shared pause
        self.paused_until = 0.0  # time.monotonic() before which no request is sent
        self.retries: dict[str, int] = {}
        self.done: dict[str, list[float]] = {}
        self.tokens_used = 0
        if cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.done[row["key"]] = row["log_probs"]
                self.tokens_used += row["tokens_used"]

    @staticmethod
    def key(sequence: str, positions: tuple[int, ...]) -> str:
        return f"{hashlib.sha256(sequence.encode()).hexdigest()[:16]}:{','.join(map(str, positions))}"

    def __call__(self, sequence: str, positions: tuple[int, ...]) -> np.ndarray:
        key = self.key(sequence, positions)
        if key not in self.done:
            self.fetch(sequence, positions)
        return np.asarray(self.done[key], dtype=np.float64)  # (K,)

    def prefetch(self, proteins: list[Protein]) -> None:
        """Request every protein not yet cached, `workers` at a time, printing the pace every 100 proteins."""
        missing = [protein for protein in proteins if self.key(protein.sequence, protein.positions) not in self.done]
        started, finished = time.perf_counter(), 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for _ in pool.map(lambda protein: self.fetch(protein.sequence, protein.positions), missing):
                finished += 1
                if finished % 100 == 0 or finished == len(missing):
                    minutes = (time.perf_counter() - started) / 60
                    print(f"{self.spec.model_id}: {finished}/{len(missing)} proteins, {finished / minutes:.1f} a minute, "
                          f"{self.tokens_used:,} tokens, retries {self.retries}", flush=True)

    def fetch(self, sequence: str, positions: tuple[int, ...]) -> None:
        with ThreadPoolExecutor(max_workers=len(positions)) as pool:  # a protein's K requests side by side
            replies = list(pool.map(lambda position: self.request(sequence, position), positions))
        rows = torch.tensor([row for row, _ in replies], dtype=torch.float32)  # (K, V)
        log_probs = residue_log_probs(rows, self.residue_ids, sequence, positions)  # (K,)
        tokens = sum(billed for _, billed in replies)
        with self.lock:
            with self.cache.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": self.key(sequence, positions), "log_probs": log_probs.tolist(), "tokens_used": tokens}) + "\n")
            self.done[self.key(sequence, positions)] = log_probs.tolist()
            self.tokens_used += tokens

    def wait_out_pause(self) -> None:
        while (remaining := self.paused_until - time.monotonic()) > 0:
            time.sleep(remaining)

    def pause(self, wait: float, reason: str) -> None:
        with self.lock:
            self.retries[reason] = self.retries.get(reason, 0) + 1
            self.paused_until = max(self.paused_until, time.monotonic() + wait)
        print(f"{self.spec.model_id}: {reason}, every request waits {wait:.0f} s", flush=True)

    def request(self, sequence: str, position: int) -> tuple[list[float], int]:
        """The masked row's logits over ESM3's vocabulary and the tokens billed; a rate limit or a server error pauses
        every request (its Retry-After, else doubling) and tries again, anything else fails the run."""
        from foundry.biohub import RETRY_STATUS_CODES, BiohubAPIError

        ids = [ESM3_BOS, *(ESM3_SEQUENCE_VOCAB.index(residue) for residue in sequence), ESM3_EOS]  # L + 2
        ids[position + 1] = ESM3_MASK
        body = json.dumps({"model": self.spec.repo, "inputs": {"sequence": ids}, "logits_config": SERVED_LOGITS_CONFIG}).encode()
        headers = {"Authorization": f"Bearer {os.environ[self.token_name]}", "Content-Type": "application/json", "Accept": "application/json"}
        for attempt in range(SERVED_ATTEMPTS):
            self.wait_out_pause()
            last = attempt == SERVED_ATTEMPTS - 1
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url, data=body, headers=headers, method="POST"), timeout=300) as response:  # nosec: the service's fixed https URL
                    data = json.loads(response.read())
                break
            except urllib.error.HTTPError as error:
                if error.code not in RETRY_STATUS_CODES or last:
                    raise BiohubAPIError(error.code, error.read().decode(errors="replace")[:500]) from error
                self.pause(min(float(error.headers.get("Retry-After") or 2 ** attempt), 120.0), f"HTTP {error.code}")
            except (urllib.error.URLError, TimeoutError) as error:
                if last:
                    raise
                self.pause(min(float(2 ** attempt), 120.0), type(error).__name__)
        assert data["model"] == self.spec.repo, f"asked for {self.spec.repo}, served {data['model']}"
        logits = data["logits"]["sequence"]  # (L + 2, V)
        assert len(logits) == len(ids), f"{len(logits)} rows of logits for {len(ids)} tokens"
        return logits[position + 1], int(data["tokens_used"])


def build_scorer(spec: ModelSpec, revision: str, device: torch.device, form: tuple[str, str, str] | None = None,
                 cache: Path | None = None, served_workers: int = 1) -> object:
    if spec.kind == "biohub":
        assert cache is not None, f"{spec.model_id} is served, and its requests are cached"
        return BiohubScorer(spec, cache, served_workers)
    if spec.kind == "masked":
        return MaskedScorer(spec, revision, device, form)
    if spec.kind == "e1":
        return E1Scorer(spec, revision, device)
    assert form is not None, f"{spec.model_id} needs a T5 input format"
    return T5Scorer(spec, revision, device, form)


def score_proteins(scorer: object, proteins: list[Protein]) -> np.ndarray:
    """ln p of every protein's K true residues."""
    if isinstance(scorer, BiohubScorer):
        scorer.prefetch(proteins)  # every request in flight `workers` proteins at a time; the stack below reads the cache
    return np.stack([scorer(protein.sequence, protein.positions) for protein in proteins])  # type: ignore[operator]  # (n, K)


def summary(log_probs: np.ndarray) -> dict[str, object]:
    """Bits per residue and its 95% interval over proteins."""
    # log_probs: (n, K) natural logs.
    bits = -log_probs / np.log(2)  # (n, K)
    per_protein = bits.mean(axis=1)  # (n,)
    rng = np.random.default_rng(SEED)
    resampled = per_protein[rng.integers(len(per_protein), size=(BOOTSTRAP, len(per_protein)))].mean(axis=1)  # (B,)
    return {"bits_per_residue": float(per_protein.mean()), "ci95": [float(np.quantile(resampled, 0.025)), float(np.quantile(resampled, 0.975))],
            "residues_scored": int(bits.size), "per_protein_bits": per_protein.round(5).tolist()}


def fetch_models(state_dir: Path, model_ids: list[str]) -> None:
    """Download every model's repository at its current revision into the Hub cache, and pin that revision for `score`."""
    from huggingface_hub import HfApi, snapshot_download

    pins_path = state_dir / "revisions.json"
    pins = json.loads(pins_path.read_text(encoding="utf-8")) if pins_path.exists() else {}
    for model_id in model_ids:
        spec = MODELS[model_id]
        pins.setdefault(model_id, HfApi().model_info(spec.repo).sha)
        started = time.perf_counter()
        snapshot_download(spec.repo, revision=pins[model_id], ignore_patterns=UNUSED_FILES)
        pins_path.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{model_id}: {spec.repo}@{pins[model_id][:12]} in {time.perf_counter() - started:.0f} s", flush=True)


def score_models(state_dir: Path, model_ids: list[str], device: torch.device, served_workers: int = 1) -> None:
    corpus, dev = read_proteins(state_dir / "corpus.jsonl"), read_proteins(state_dir / "dev.jsonl")
    manifest = json.loads((state_dir / "corpus.json").read_text(encoding="utf-8"))
    assert sha256(state_dir / "corpus.jsonl") == manifest["corpus_sha256"], "corpus.jsonl changed since it was built"
    revisions = json.loads((state_dir / "revisions.json").read_text(encoding="utf-8"))  # `fetch` pinned them
    output_dir = state_dir / "scores"
    output_dir.mkdir(exist_ok=True)
    failed = []
    for model_id in model_ids:
        output = output_dir / f"{model_id}.json"
        if output.exists():
            print(f"{model_id}: done", flush=True)
            continue
        # A served model's name carries its version (`esm3-large-2024-03`); a downloaded one is pinned by `fetch`.
        revision = MODELS[model_id].repo if MODELS[model_id].kind == "biohub" else revisions[model_id]
        try:
            score_model(model_id, revision, corpus, dev, manifest, output, device, served_workers)
            (output_dir / f"{model_id}.error.txt").unlink(missing_ok=True)  # an earlier attempt's traceback no longer holds
        except Exception:  # one model's failure is its record, and the list goes on; the command still fails at the end
            (output_dir / f"{model_id}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"{model_id}: FAILED", traceback.format_exc(), sep="\n", flush=True)
            failed.append(model_id)
        torch.cuda.empty_cache()
    if failed:
        raise RuntimeError(f"{len(failed)} models failed, their tracebacks in {output_dir}: {failed}")


def score_model(model_id: str, revision: str, corpus: list[Protein], dev: list[Protein], manifest: dict[str, object],
                output: Path, device: torch.device, served_workers: int = 1) -> None:
    """One model's bits per residue on the corpus at a pinned revision, its input format (a T5 model's, DPLM2's layout)
    first chosen on the dev set."""
    spec = MODELS[model_id]
    started = time.perf_counter()
    form: tuple[str, str, str] | None = None
    dev_formats: list[dict[str, object]] = []
    if spec.formats:
        for candidate in spec.formats:
            try:
                scorer = build_scorer(spec, revision, device, candidate)
            except KeyError as missing:  # a mask token the vocabulary lacks is not a format of this model; recorded
                dev_formats.append({"format": list(candidate), "dev_bits": None, "absent": str(missing)})
                continue
            dev_formats.append({"format": list(candidate), "dev_bits": summary(score_proteins(scorer, dev))["bits_per_residue"]})
            del scorer
        usable = [row for row in dev_formats if row["dev_bits"] is not None]
        assert usable, f"{model_id}: no candidate format loads"
        form = tuple(min(usable, key=lambda row: row["dev_bits"])["format"])  # type: ignore[arg-type, return-value]
    cache = output.with_suffix(".partial.jsonl")  # a served model's requests so far; the figure reads only *.json
    scorer = build_scorer(spec, revision, device, form, cache, served_workers)
    dev_summary = summary(score_proteins(scorer, dev))
    log_probs = score_proteins(scorer, corpus)  # (n, K)
    served = spec.kind == "biohub"
    parameters = SERVED_PARAMETERS[model_id] if served else sum(parameter.numel() for parameter in scorer.model.parameters())  # type: ignore[attr-defined]
    report = {"model_id": model_id, "label": spec.label, "family": spec.family, "repo": spec.repo, "revision": revision,
              "kind": spec.kind, "format": list(form) if form else None, "dev_formats": dev_formats,
              "dev_bits_per_residue": dev_summary["bits_per_residue"], "parameters": parameters,
              "corpus_sha256": manifest["corpus_sha256"], **summary(log_probs),
              "log_probs": log_probs.round(6).tolist(), "seconds": round(time.perf_counter() - started, 1),
              **({"tokens_used": scorer.tokens_used, "served_retries": scorer.retries} if served else {}),  # type: ignore[attr-defined]
              "environment": benchmark.environment(device)}
    output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    cache.unlink(missing_ok=True)  # the report holds every log-probability the cache did
    print(json.dumps({model_id: {key: report[key] for key in ("bits_per_residue", "ci95", "dev_bits_per_residue", "format", "seconds")}}),
          flush=True)
    del scorer
    torch.cuda.empty_cache()


def check_served(state_dir: Path, model: str, against: str) -> None:
    """A served model's dev-set bits beside a downloaded model's: the open ESM3 served by Biohub against the local port
    of its weights checks the request, the vocabulary and the row read before a model only the service has is scored."""
    spec = ModelSpec(f"check_{model}", model, "", model, "biohub")
    scorer = BiohubScorer(spec, state_dir / f"check_{model}.partial.jsonl")
    served = summary(score_proteins(scorer, read_proteins(state_dir / "dev.jsonl")))["bits_per_residue"]
    local = json.loads((state_dir / "scores" / f"{against}.json").read_text(encoding="utf-8"))["dev_bits_per_residue"]
    report = {"served": model, "local": against, "served_dev_bits": served, "local_dev_bits": local,
              "difference": served - local, "tokens_used": scorer.tokens_used}
    (state_dir / f"check_{model}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


# The star: a speedrun run rebuilt from its own snapshot (`probe.load_run_snapshot`).

class SpeedrunScorer:
    """A speedrun candidate on the ESM2 alphabet: CLS residues EOS, MASK at one residue a row."""

    def __init__(self, run_dir: Path, device: torch.device) -> None:
        run_result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        module = load_run_snapshot(run_dir)  # its own train.py and strategies, digests checked
        # The benchmark's longest held-out window, as probe.py builds it; a corpus protein is at most 1,024 tokens.
        self.model = module.build_candidate(run_result["genome"], max_tokens=2048).to(device)  # type: ignore[attr-defined]
        state = torch.load(run_dir / "best.pt", map_location=device)
        self.model.load_state_dict({name: tensor.float() if tensor.is_floating_point() else tensor for name, tensor in state.items()})
        self.model.eval()
        self.device, self.run_result = device, run_result
        self.residue_ids = benchmark.ESM2_RESIDUE_COLUMNS.to(device)  # (c,)

    def __call__(self, sequence: str, positions: tuple[int, ...]) -> np.ndarray:
        ids = torch.tensor([benchmark.CLS_ID, *(benchmark.tokens.ESM2_IDS[residue] for residue in sequence), benchmark.EOS_ID],
                           device=self.device)  # (L + 2,)
        rows = torch.arange(len(positions), device=self.device)  # (K,)
        columns = torch.tensor(positions, device=self.device) + 1  # (K,)
        batch = ids.repeat(len(positions), 1)  # (K, L + 2)
        batch[rows, columns] = benchmark.MASK_ID
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            logits = self.model(batch, torch.ones_like(batch, dtype=torch.bool))  # (K, L + 2, 33)
        return residue_log_probs(logits[rows, columns], self.residue_ids, sequence, positions)  # (K,)


def score_star(state_dir: Path, run: str | None, run_dir: Path | None, output: str, device: torch.device) -> None:
    """A speedrun run scored as a model: by name from the private runs repository, or a run folder on this machine."""
    from huggingface_hub import HfApi, snapshot_download

    assert (run is None) != (run_dir is None), "name a run in the Hub repository or a local run folder, not both"
    if run_dir is None:
        revision = HfApi().model_info(RUNS_REPO).sha
        # The repository keeps each run under runs/<run>/ (best.pt, result.json, train.py, strategies/).
        local = Path(snapshot_download(RUNS_REPO, revision=revision, allow_patterns=[f"runs/{run}/*", f"runs/{run}/strategies/*"],
                                       local_dir=str(state_dir / "hub_runs")))
        run_dir, repo = local / "runs" / run, RUNS_REPO
    else:
        run, repo, revision = run_dir.name, "local", sha256(run_dir / "SHA256SUMS")  # the run folder's seal
    corpus = read_proteins(state_dir / "corpus.jsonl")
    manifest = json.loads((state_dir / "corpus.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    scorer = SpeedrunScorer(run_dir, device)
    log_probs = score_proteins(scorer, corpus)  # (n, K)
    run_result = scorer.run_result
    report = {"model_id": run, "label": f"speedrun {run_result['clock']['budget_seconds'] / 3600:g} h", "family": "speedrun",
              "repo": repo, "revision": revision, "kind": "speedrun", "parameters": run_result["architecture"]["parameters"],
              "budget_seconds": run_result["clock"]["budget_seconds"], "test_loss": run_result["metrics"]["test"]["loss"],
              "benchmark_id": run_result["benchmark_id"], "run_finished_utc": run_result["finished_utc"],
              "corpus_sha256": manifest["corpus_sha256"], **summary(log_probs),
              "log_probs": log_probs.round(6).tolist(), "seconds": round(time.perf_counter() - started, 1),
              "environment": benchmark.environment(device)}
    (state_dir / output).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({run: {key: report[key] for key in ("bits_per_residue", "ci95", "parameters", "seconds")}}), flush=True)


# The k-mer references: the masked residue predicted from the a residues on each side of it, by counts over the
# speedrun's training stream, so each line is what a lookup table learns from the data the speedrun trains on. Order a
# backs off to order a - 1 (the same residue with one fewer neighbor each side) through a Dirichlet prior of weight
# beta, chosen on the dev set: p_a(x | c) = (n_a(c, x) + beta_a p_{a-1}(x | c')) / (n_a(c) + beta_a), and p_0 is the
# add-one residue frequency. Only contexts the dev set and corpus hold are counted.

KMER_HALF_WIDTHS = (0, 1, 2)  # a: the 1-, 3- and 5-mer models (Logan, 2026-11-23: "just do 1, 3, and 5")
KMER_BETAS = tuple(float(4.0 ** power) for power in range(-3, 9))  # 1/64 to 65,536; a pick on either edge widens it
BOUNDARY, OTHER = 20, 21  # context symbols past either end of a protein and for a nonstandard residue
CONTEXT_BASE = 22
KMER_CHUNK = 50_000  # training proteins a pass


def residue_symbols(tokens_or_sequence: np.ndarray | str) -> np.ndarray:
    """Residue classes 0 to 19, OTHER for anything else; from a sequence or from ESM2 token ids."""
    if isinstance(tokens_or_sequence, str):
        lookup = {residue: index for index, residue in enumerate(STANDARD_RESIDUES)}
        return np.array([lookup.get(residue, OTHER) for residue in tokens_or_sequence], dtype=np.int64)  # (L,)
    table = np.full(benchmark.tokens.VOCAB_SIZE, OTHER, dtype=np.int64)  # (V,)
    for index, residue in enumerate(STANDARD_RESIDUES):
        table[benchmark.tokens.ESM2_IDS[residue]] = index
    return table[tokens_or_sequence]  # (L,)


def padded_stream(proteins: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Proteins laid end to end with as many BOUNDARY symbols between them as the widest context, and each residue's
    position in that array."""
    pad = max(KMER_HALF_WIDTHS)
    lengths = np.fromiter((len(protein) for protein in proteins), dtype=np.int64, count=len(proteins))  # (n,)
    starts = pad + np.concatenate([[0], np.cumsum(lengths + pad)[:-1]])  # (n,)
    padded = np.full(int(starts[-1] + lengths[-1] + pad), BOUNDARY, dtype=np.int64)  # (P,)
    positions = np.concatenate([start + np.arange(length) for start, length in zip(starts, lengths, strict=True)])  # (R,)
    padded[positions] = np.concatenate(proteins)
    return padded, positions


def context_codes(padded: np.ndarray, centers: np.ndarray, half_width: int) -> np.ndarray:
    """One integer for the half_width symbols on each side of every center, nearest first, left then right."""
    codes = np.zeros(len(centers), dtype=np.int64)  # (R,)
    for distance in range(1, half_width + 1):
        codes = (codes * CONTEXT_BASE + padded[centers - distance]) * CONTEXT_BASE + padded[centers + distance]
    return codes


def kmer_references(state_dir: Path, data_dir: Path) -> None:
    corpus, dev = read_proteins(state_dir / "corpus.jsonl"), read_proteins(state_dir / "dev.jsonl")
    manifest = json.loads((state_dir / "corpus.json").read_text(encoding="utf-8"))
    targets: dict[str, tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]] = {}
    for name, proteins in (("dev", dev), ("corpus", corpus)):
        padded, positions = padded_stream([residue_symbols(protein.sequence) for protein in proteins])
        starts = np.concatenate([[0], np.cumsum([len(protein.sequence) for protein in proteins])[:-1]])  # (n,)
        scored = np.concatenate([positions[start + np.asarray(protein.positions)] for start, protein in zip(starts, proteins, strict=True)])  # (n K,)
        targets[name] = (padded[scored], scored, {a: context_codes(padded, scored, a) for a in KMER_HALF_WIDTHS})
    needed = {a: np.unique(np.concatenate([targets[name][2][a] for name in targets])) for a in KMER_HALF_WIDTHS}  # each (m_a,)
    counts = {a: np.zeros((len(needed[a]), len(STANDARD_RESIDUES)), dtype=np.int64) for a in KMER_HALF_WIDTHS}  # each (m_a, c)
    stream = benchmark.load_train(data_dir)
    proteins_total = len(stream.offsets) - 1
    started = time.perf_counter()
    for first in range(0, proteins_total, KMER_CHUNK):
        last = min(first + KMER_CHUNK, proteins_total)
        symbols = residue_symbols(np.asarray(stream.tokens[stream.offsets[first] : stream.offsets[last]]))  # (t_chunk,)
        local = stream.offsets[first : last + 1] - stream.offsets[first]  # (chunk + 1,)
        padded, positions = padded_stream([symbols[local[index] + 1 : local[index + 1] - 1] for index in range(last - first)])  # CLS, EOS dropped
        centers = positions[padded[positions] < BOUNDARY]  # (R,) standard residues only
        residues = padded[centers]  # (R,)
        for a in KMER_HALF_WIDTHS:
            codes = context_codes(padded, centers, a)  # (R,)
            slots = np.minimum(np.searchsorted(needed[a], codes), len(needed[a]) - 1)  # (R,)
            hit = needed[a][slots] == codes  # (R,)
            counts[a] += np.bincount(slots[hit] * len(STANDARD_RESIDUES) + residues[hit],
                                     minlength=counts[a].size).reshape(counts[a].shape)
        print(f"kmers: {last:,} of {proteins_total:,} proteins, {time.perf_counter() - started:.0f} s", flush=True)

    def log_probs(name: str, betas: dict[int, float], order: int) -> np.ndarray:
        """ln p of the true residue at every scored position of `name`, at context order `order`."""
        residue, _, codes = targets[name]
        probability = None
        for a in range(order + 1):
            rows = counts[a][np.searchsorted(needed[a], codes[a])].astype(np.float64)  # (n K, c)
            prior = np.full_like(rows, 1.0 / len(STANDARD_RESIDUES)) if probability is None else probability  # (n K, c)
            beta = float(len(STANDARD_RESIDUES)) if probability is None else betas[a]  # add-one at order 0
            probability = (rows + beta * prior) / (rows.sum(axis=1, keepdims=True) + beta)  # (n K, c)
        return np.log(probability[np.arange(len(residue)), residue])  # (n K,)

    betas: dict[int, float] = {}
    report: dict[str, object] = {"data_dir": str(data_dir), "training_proteins": proteins_total,
                                 "corpus_sha256": manifest["corpus_sha256"], "models": {}}
    for a in KMER_HALF_WIDTHS:
        dev_bits = {}
        for beta in KMER_BETAS if a else (float(len(STANDARD_RESIDUES)),):
            dev_bits[beta] = float(-log_probs("dev", {**betas, a: beta}, a).mean() / np.log(2))
        betas[a] = min(dev_bits, key=dev_bits.__getitem__)
        if a and betas[a] in (KMER_BETAS[0], KMER_BETAS[-1]):  # an edge pick is fine only where the dev curve has flattened
            neighbor = KMER_BETAS[1] if betas[a] == KMER_BETAS[0] else KMER_BETAS[-2]
            assert dev_bits[neighbor] - dev_bits[betas[a]] < 1e-4, f"order {a} still improves at beta {betas[a]}: widen KMER_BETAS"
        corpus_log_probs = log_probs("corpus", betas, a).reshape(len(corpus), POSITIONS)  # (n, K)
        report["models"][f"{2 * a + 1}-mer"] = {"half_width": a, "beta": betas[a], "dev_bits_by_beta": dev_bits,
                                                "contexts_counted": int(len(needed[a])), "training_windows": int(counts[a].sum()),
                                                **{key: value for key, value in summary(corpus_log_probs).items() if key != "per_protein_bits"}}
        print(json.dumps({f"{2 * a + 1}-mer": {key: report["models"][f"{2 * a + 1}-mer"][key] for key in ("bits_per_residue", "ci95", "beta")}}),
              flush=True)
    (state_dir / "kmers.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


# The figure.

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LINEAGE = {"ESM": "#2a78d6", "ESMC": "#2a78d6", "ProtTrans": "#eb6834", "ANKH": "#eb6834", "DPLM": "#1baf7a", "E1": "#1baf7a"}
MARKER = {"ESM": "o", "ESMC": "s", "ProtTrans": "o", "ANKH": "s", "DPLM": "o", "E1": "s"}  # shape tells a lineage's families apart
LEGEND = {"ESM": "ESM-1, ESM2", "ESMC": "ESMC, ESM3", "ProtTrans": "ProtBERT, ProtT5", "ANKH": "ANKH, ANKH2", "DPLM": "DPLM, DPLM2",
          "E1": "E1"}
# Scored and kept in the result, but left out of the figure and its table: the one-mask rule misreads these models.
# Checked on the dev set with the authors' own likelihood code (ankh/likelihood.py) under transformers 4.46.3 and 5.24.0,
# which agree to four decimals and load every checkpoint tensor as stored. With 15% of residues masked at once, the
# condition ANKH3 trained under, ANKH3 XL reads 2.97 bits against 3.58 with one mask; ANKH3 large stays near uniform
# (4.22) and predicts the top residue 17% of the time against ANKH large's 36%.
FIGURE_EXCLUDED = {
    "ankh3_large": "near uniform under every masking tried, its released checkpoint does not predict masked residues",
    "ankh3_xl": "out of its training distribution with one mask (3.58 bits, against 2.97 with 15% masked)",
}
# The sizes of one model that panel B joins with a line, smallest first; a model in none is a lone point.
SERIES: tuple[tuple[str, ...], ...] = (
    ("esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3b", "esm2_15b"),
    ("esmc_300m", "esmc_600m", "esmc_6b"),
    ("esm3_small", "esm3_large"),
    ("ankh_base", "ankh_large"),
    ("dplm_150m", "dplm_650m", "dplm_3b"),
    ("dplm2_150m", "dplm2_650m", "dplm2_3b"),
    ("e1_150m", "e1_300m", "e1_600m"),
)
assert all(model_id in MODELS for series in SERIES for model_id in series)
assert FIGURE_EXCLUDED.keys() <= MODELS.keys() and not FIGURE_EXCLUDED.keys() & {model_id for series in SERIES for model_id in series}
STAR_LABEL = "speedrun, 1 h"


def release_date(text: str) -> np.datetime64:
    return np.datetime64(text if len(text) == 10 else f"{text}-01")


def spread(values: list[float], gap: float) -> list[float]:
    """Label heights as close to their values as a minimum gap between neighbors allows, the stack centered on them."""
    order = np.argsort(values)
    placed = np.asarray(values, dtype=np.float64)[order]  # (r,)
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + gap)
    placed -= (placed - np.sort(values)).mean()
    result = np.empty_like(placed)
    result[order] = placed
    return result.tolist()


def plot(state_dir: Path, output: Path) -> None:
    """A: bits per residue by release; B: by parameters, each model's sizes joined. Color is the lineage (three hues that
    stay apart for every pair), shape the family within it. Only the star carries a label: model names crowd the
    low-bits corner wherever they go, so the README's table names every point. Reference lines: uniform over the 20
    residues (dashed) and the k-mer lookup tables (dotted), labeled once in panel B's right margin."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((state_dir / "corpus.json").read_text(encoding="utf-8"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((state_dir / "scores").glob("*.json"))]
    for report in reports:
        if report["model_id"] in FIGURE_EXCLUDED:
            print(f"left out: {report['model_id']} ({report['bits_per_residue']:.3f} bits), {FIGURE_EXCLUDED[report['model_id']]}")
    reports = [report for report in reports if report["model_id"] not in FIGURE_EXCLUDED]
    star = json.loads((state_dir / "star.json").read_text(encoding="utf-8"))
    kmers = json.loads((state_dir / "kmers.json").read_text(encoding="utf-8"))
    for report in [*reports, star, kmers]:
        assert report["corpus_sha256"] == manifest["corpus_sha256"], f"{report.get('model_id', 'kmers')} scored another corpus"
    references = [("uniform", float(np.log2(len(STANDARD_RESIDUES))), (0, (4, 3)))]
    references += [(name, model["bits_per_residue"], (0, (1, 2))) for name, model in kmers["models"].items()]
    assert {report["family"] for report in reports} <= LINEAGE.keys(), "a family the figure has no color for"
    by_id = {report["model_id"]: report for report in reports}
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, constrained_layout=True)
    star_date = np.datetime64(star["run_finished_utc"][:10])
    for axis, letter, x_of in ((axes[0], "A", lambda report: release_date(RELEASES[report["model_id"]][0])),
                               (axes[1], "B", lambda report: report["parameters"])):
        for _, reference, style in references:
            axis.axhline(reference, color=MUTED, linewidth=0.7, linestyle=style, zorder=1)
        if letter == "B":
            for series in SERIES:
                members = [by_id[model_id] for model_id in series if model_id in by_id]
                axis.plot([report["parameters"] for report in members], [report["bits_per_residue"] for report in members],
                          color=LINEAGE[members[0]["family"]], linewidth=1.0, alpha=0.55, solid_capstyle="round", zorder=2)
        for family in LINEAGE:
            members = [report for report in reports if report["family"] == family]
            axis.scatter([x_of(report) for report in members], [report["bits_per_residue"] for report in members], s=24,
                         marker=MARKER[family], color=LINEAGE[family], edgecolors="white", linewidths=0.7, zorder=3,
                         label=LEGEND[family] if letter == "A" else None)
        star_x = star_date if letter == "A" else star["parameters"]
        axis.scatter([star_x], [star["bits_per_residue"]], s=120, marker="*", color=INK, edgecolors="white", linewidths=0.6,
                     zorder=4, label=STAR_LABEL if letter == "A" else None)
        left = letter == "A"  # the newest point sits at panel A's right edge, so its label reads leftward there
        axis.annotate(STAR_LABEL, (star_x, star["bits_per_residue"]), xytext=(-8 if left else 8, 0), textcoords="offset points",
                      ha="right" if left else "left", va="center", fontsize=6.5, color=INK)
        axis.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.02, 1.02, letter, transform=axis.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
    axes[0].set_ylabel("Bits per residue")
    axes[0].set_xlabel("Release date")
    axes[1].set_xlabel("Parameters")
    axes[1].set_xscale("log")
    low, high = axes[1].get_ylim()
    heights = spread([reference for _, reference, _ in references], 0.055 * (high - low))  # 6-point labels never touch
    for (text, reference, _), height in zip(references, heights, strict=True):
        axes[1].annotate(text, (1.0, reference), xycoords=("axes fraction", "data"), xytext=(1.04, height),
                         textcoords=("axes fraction", "data"), ha="left", va="center", color=MUTED, fontsize=6, annotation_clip=False,
                         arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.4, "shrinkA": 0, "shrinkB": 1})
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=7, frameon=False, fontsize=6.5,
                  handletextpad=0.2, columnspacing=1.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    print(f"wrote {output}")
    print(markdown_table(reports, star, kmers))


def parameter_text(parameters: int) -> str:
    return f"{parameters / 1e9:.1f}B" if parameters >= 1e9 else f"{parameters / 1e6:.0f}M"


def markdown_table(reports: list[dict], star: dict, kmers: dict) -> str:
    """Every scored point and reference, best first: the README's table of the figure."""
    rows = [(report["bits_per_residue"], report["ci95"], report["label"], RELEASES[report["model_id"]][0], parameter_text(report["parameters"]),
             report.get("format")) for report in reports]
    rows.append((star["bits_per_residue"], star["ci95"], f"**{STAR_LABEL}** (`{star['model_id']}`)", star["run_finished_utc"][:10],
                 parameter_text(star["parameters"]), None))
    rows += [(model["bits_per_residue"], model["ci95"], f"{name} lookup table (reference)", "", "", None) for name, model in kmers["models"].items()]
    lines = ["| Model | Released | Parameters | Bits per residue | 95% CI |", "|---|---|---|---|---|"]
    for bits, (low, high), label, released, parameters, form in sorted(rows, key=lambda row: row[0]):
        # The mode or layout in words ("aa_only" as "aa only"), then the mask token and prefix as written.
        parts = [form[0].replace("_", " "), *(f"`{part}`" for part in form[1:] if part)] if form else []
        note = f" ({', '.join(parts)})" if parts else ""
        lines.append(f"| {label}{note} | {released} | {parameters} | {bits:.3f} | {low:.3f} to {high:.3f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    corpus = commands.add_parser("corpus")
    corpus.add_argument("--state-dir", type=Path, required=True)
    corpus.add_argument("--threads", type=int, default=32)
    score = commands.add_parser("score")
    score.add_argument("--state-dir", type=Path, required=True)
    # A served model spends the account's credits, so only naming it scores it.
    score.add_argument("--models", nargs="+", default=[model_id for model_id, spec in MODELS.items() if spec.kind != "biohub"],
                       choices=list(MODELS))
    score.add_argument("--served-workers", type=int, default=1, help="a served model's proteins requested at once, K requests each")
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--state-dir", type=Path, required=True)
    fetch.add_argument("--models", nargs="+", default=[model_id for model_id, spec in MODELS.items() if spec.kind != "biohub"],
                       choices=[model_id for model_id, spec in MODELS.items() if spec.kind != "biohub"])
    served = commands.add_parser("check-served")
    served.add_argument("--state-dir", type=Path, required=True)
    served.add_argument("--model", default="esm3-open-2024-03", help="a model name the Biohub API serves")
    served.add_argument("--against", default="esm3_small", choices=list(MODELS), help="the scored local model of the same weights")
    star = commands.add_parser("star")
    star.add_argument("--state-dir", type=Path, required=True)
    source = star.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help=f"a run's name in {RUNS_REPO}")
    source.add_argument("--run-dir", type=Path, help="a run folder on this machine")
    star.add_argument("--output", default="star.json", help="the report's name in the state directory; the figure reads star.json")
    for command in (score, star):
        command.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    figure = commands.add_parser("plot")
    figure.add_argument("--state-dir", type=Path, required=True)
    figure.add_argument("--output", type=Path, required=True)
    copies = commands.add_parser("copies")
    copies.add_argument("--state-dir", type=Path, required=True)
    copies.add_argument("--data-dir", type=Path, required=True, help="the prepared data directory the exclusion's indices refer to")
    relatives = commands.add_parser("relatives")
    relatives.add_argument("--state-dir", type=Path, required=True)
    relatives.add_argument("--data-dir", type=Path, required=True, help="the speedrun's prepared data directory")
    relatives.add_argument("--threads", type=int, default=32)
    relatives.add_argument("--searches", nargs="+", default=list(RELATIVE_SEARCHES), choices=RELATIVE_SEARCHES)
    kmers = commands.add_parser("kmers")
    kmers.add_argument("--state-dir", type=Path, required=True)
    kmers.add_argument("--data-dir", type=Path, required=True, help="the speedrun's prepared data directory, whose counts the tables hold")
    arguments = parser.parse_args()
    # cuDNN's attention builds an execution plan for every new sequence length: 610 ms a forward at a new length against
    # 13 ms without it, ESM2 8M on the GH200 (2026-11-23); the corpus has hundreds of lengths, so no model uses it.
    torch.backends.cuda.enable_cudnn_sdp(False)
    if arguments.command == "corpus":
        build_corpus(arguments.state_dir, arguments.threads)
    elif arguments.command == "kmers":
        kmer_references(arguments.state_dir, arguments.data_dir)
    elif arguments.command == "copies":
        exact_copies(arguments.state_dir, arguments.data_dir)
    elif arguments.command == "relatives":
        search_relatives(arguments.state_dir, arguments.data_dir, arguments.threads, arguments.searches)
    elif arguments.command == "fetch":
        fetch_models(arguments.state_dir, arguments.models)
    elif arguments.command == "check-served":
        check_served(arguments.state_dir, arguments.model, arguments.against)
    elif arguments.command == "score":
        score_models(arguments.state_dir, arguments.models, torch.device(arguments.device), arguments.served_workers)
    elif arguments.command == "star":
        score_star(arguments.state_dir, arguments.run, arguments.run_dir, arguments.output, torch.device(arguments.device))
    else:
        plot(arguments.state_dir, arguments.output)


if __name__ == "__main__":
    main()
