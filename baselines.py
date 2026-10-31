"""The reference models through the same evaluator and the same probe; fixed.

    python baselines.py --data-dir /data/afdb_v6 --output-dir runs/baselines --models esm2_8m esmc_small

Each model loads from FastPLMs' pinned repository and revision. The benchmark's ESM2 ids are remapped to the model's
tokenizer by token string, so ESMC (64 tokens) and ESM2 (33 tokens) read the same proteins; the evaluator then scores
every model on its own 20 standard-residue columns. Writes `<model>.json` per model: MLM metrics on `val_select`,
validation and test, and the 3Di probe on validation and test.

Symbols: b proteins, l tokens, v the model's vocabulary, k hidden states.
"""

from __future__ import annotations

import argparse
import json
import time
import numpy as np
import torch

from pathlib import Path
from torch import Tensor

from speedrunning_plms import benchmark
from speedrunning_plms.benchmark import probe, tokens


MODEL_IDS = ("unigram", "esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3b", "esmc_small", "esmc_large", "esmc_6b")
PINNED = Path(__file__).with_name("baselines.json")  # repository and revision per id, from FastPLMs' registry


class ReferenceModel:
    """A FastPLMs masked language model behind the evaluator's and the probe's interfaces."""

    def __init__(self, model_id: str, device: torch.device) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        pinned = json.loads(PINNED.read_text(encoding="utf-8"))[model_id]
        self.model_id = model_id
        self.repo_id, self.revision = pinned["repo_id"], pinned["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, revision=self.revision, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(self.repo_id, revision=self.revision, trust_remote_code=True,
                                                          dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).to(device).eval()
        self.device = device
        self.id_map = self._id_map().to(device)  # (33,), benchmark id -> model id
        self.residue_columns = self.id_map[tokens.ESM2_RESIDUE_COLUMNS].cpu()  # (20,)
        self.pad_id = int(self.id_map[tokens.PAD_ID])
        self.hidden = int(self.model.config.hidden_size)
        self.num_hidden_states = int(self.model.config.num_hidden_layers) + 1  # k

    def _id_map(self) -> Tensor:
        lookup = {tokens.ESM2_TOKENS[index]: index for index in range(tokens.VOCAB_SIZE)}
        ids = []
        for token in tokens.ESM2_TOKENS:
            model_id = self.tokenizer.convert_tokens_to_ids(token)
            if model_id is None or model_id == self.tokenizer.unk_token_id and token != "<unk>":
                model_id = self.tokenizer.unk_token_id  # tokens the model lacks (null_1, rare letters) read as unknown
            ids.append(int(model_id))
        for residue in tokens.STANDARD_RESIDUES:
            assert ids[lookup[residue]] != self.tokenizer.unk_token_id, f"{self.model_id} lacks residue {residue}"
        assert ids[lookup["<mask>"]] == self.tokenizer.mask_token_id and ids[lookup["<pad>"]] == self.tokenizer.pad_token_id
        return torch.tensor(ids, dtype=torch.long)

    def __call__(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # input_ids, attention_mask: (b, l); returns (b, l, v) in the model's vocabulary.
        return self.model(input_ids=self.id_map[input_ids], attention_mask=attention_mask).logits

    def hidden_states(self, input_ids: Tensor, attention_mask: Tensor) -> list[Tensor]:
        # input_ids, attention_mask: (b, l); returns k tensors of (b, l, d).
        output = self.model(input_ids=self.id_map[input_ids], attention_mask=attention_mask, output_hidden_states=True)
        states = list(output.hidden_states)
        assert len(states) == self.num_hidden_states, (len(states), self.num_hidden_states)
        return states


class UnigramModel:
    """The floor: every masked residue predicted from the training stream's residue frequencies."""

    def __init__(self, data_dir: Path, device: torch.device) -> None:
        stream = benchmark.load_train(data_dir)
        counts = torch.bincount(torch.from_numpy(np.asarray(stream.tokens)).long(), minlength=tokens.VOCAB_SIZE).double()  # (v,)
        self.log_frequency = (counts / counts.sum()).clamp(min=1e-12).log().float().to(device)  # (v,)
        self.residue_columns = tokens.ESM2_RESIDUE_COLUMNS  # (20,)
        self.model_id, self.repo_id, self.revision = "unigram", None, None
        self.residue_frequency = {residue: float(counts[tokens.ESM2_IDS[residue]] / counts[tokens.ESM2_RESIDUE_COLUMNS].sum())
                                  for residue in tokens.STANDARD_RESIDUES}

    def __call__(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # input_ids: (b, l); returns (b, l, v), the same row everywhere.
        return self.log_frequency.expand(*input_ids.shape, -1)


def evaluate_model(model_id: str, data_dir: Path, output_dir: Path, device: torch.device, *, tokens_per_batch: int,
                   with_probe: bool) -> dict[str, object]:
    started = time.perf_counter()
    reference: ReferenceModel | UnigramModel = UnigramModel(data_dir, device) if model_id == "unigram" else ReferenceModel(model_id, device)
    metrics = {split: benchmark.evaluate_split(reference, benchmark.load_held_out(data_dir, split), device,
                                               tokens_per_batch=tokens_per_batch)
               for split in benchmark.data.HELD_OUT_SPLITS}
    report: dict[str, object] = {
        "model_id": model_id, "repo_id": reference.repo_id, "revision": reference.revision,
        "residue_columns": reference.residue_columns.tolist(),
        "benchmark_id": benchmark.benchmark_id(benchmark.load_manifest(data_dir)), "evaluator_sha256": benchmark.evaluator_sha256(),
        "metrics": metrics, "environment": benchmark.environment(device),
    }
    if isinstance(reference, UnigramModel):
        report["residue_frequency"] = reference.residue_frequency
        with_probe = False
    else:
        report["parameters"] = sum(parameter.numel() for parameter in reference.model.parameters())
        report["vocab_size"] = int(reference.model.config.vocab_size)
    if with_probe:
        report["probe"] = run_probe(reference, data_dir, device, tokens_per_batch=tokens_per_batch)
    report["seconds"] = time.perf_counter() - started
    output = output_dir / f"{model_id}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({model_id: {split: {key: values[key] for key in ("loss", "accuracy", "mcc")} for split, values in metrics.items()}}), flush=True)
    return report


def run_probe(reference: ReferenceModel, data_dir: Path, device: torch.device, *, tokens_per_batch: int) -> dict[str, object]:
    stream = benchmark.load_train(data_dir)
    head = probe.train_head(reference, stream, device, hidden=reference.hidden, num_hidden_states=reference.num_hidden_states)
    return {split: probe.score_head(reference, head, benchmark.load_held_out(data_dir, split), device, tokens_per_batch=tokens_per_batch)
            for split in ("validation", "test")}


def probe_model(model_id: str, data_dir: Path, output_dir: Path, device: torch.device, *, tokens_per_batch: int) -> None:
    """Replace a scored model's probe section, the metrics untouched; for a change to the probe itself."""
    output = output_dir / f"{model_id}.json"
    report = json.loads(output.read_text(encoding="utf-8"))
    started = time.perf_counter()
    report["probe"] = run_probe(ReferenceModel(model_id, device), data_dir, device, tokens_per_batch=tokens_per_batch)
    report["probe_seconds"] = time.perf_counter() - started
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({model_id: {split: {key: values[key] for key in ("loss", "accuracy", "mcc")} for split, values in report["probe"].items()}}),
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODEL_IDS), choices=MODEL_IDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens-per-batch", type=int, default=16_384)
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--probe-only", action="store_true", help="re-run the probe on a scored model, keeping its metrics")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(arguments.device)
    for model_id in arguments.models:
        if arguments.probe_only:
            if model_id != "unigram":
                probe_model(model_id, arguments.data_dir, arguments.output_dir, device, tokens_per_batch=arguments.tokens_per_batch)
            continue
        if (arguments.output_dir / f"{model_id}.json").exists():
            print(f"{model_id}: done", flush=True)
            continue
        evaluate_model(model_id, arguments.data_dir, arguments.output_dir, device, tokens_per_batch=arguments.tokens_per_batch,
                       with_probe=not arguments.no_probe)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
