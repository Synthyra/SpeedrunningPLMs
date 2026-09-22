"""Run the actual staged training engine from a workstation launcher on CPU."""

import json
import sys
import torch

from pathlib import Path

from speedrunning_plms.research.benchmark import encode_sequence, write_dataset
from speedrunning_plms.research.runner import Host, Target, run_experiment


def test_staged_cpu_training_returns_a_loadable_result(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    tokens = torch.tensor(encode_sequence("ACDEFGHIKLMNPQRSTVWY" * 6, 8))  # (n, 8)
    data_dir = tmp_path / "data"
    write_dataset({"train": tokens, "valid": tokens.flip(0)}, data_dir)
    config = tmp_path / "experiment.json"
    config.write_text(json.dumps({
        "device": "cpu", "hidden_size": 8, "heads": 2, "layers": 2,
        "batch_size": 2, "max_steps": 1,
    }), encoding="utf-8")
    target = Target("cpu-smoke", (Host(None, str(tmp_path / "staging"), sys.executable),))

    record = run_experiment(
        target, root, tmp_path / "runs", "integration", str(data_dir),
        time_budget=30, timeout=90, config=config,
    )

    assert record["status"] == "completed"
    assert record["comparable"] is False
    assert record["result"]["optimizer_steps"] == 1
    assert record["result"]["val_bits_per_masked_residue"] > 0
    assert record["result"]["eval_dtype"] == "float32"
    local_run = tmp_path / "runs" / record["run_id"]
    assert (local_run / "source.zip").is_file()
    assert json.loads((local_run / "result.json").read_text())["benchmark_id"] == record["result"]["benchmark_id"]
    checkpoint = Path(record["node_dirs"][0]) / "output" / "checkpoint"
    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "config.json").is_file()
    assert (local_run / "node-0.log").stat().st_size > 0
    ledger = [json.loads(line) for line in (tmp_path / "runs/results.jsonl").read_text().splitlines()]
    assert len(ledger) == 1 and ledger[0]["source_sha256"] == record["source_sha256"]
