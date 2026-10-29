"""The result of a run and the campaign ledger's row.

`result.json` holds everything a reader needs to audit a run: the benchmark id, the evaluator's digest, the
candidate's digest, the clock and architecture receipts, the selection curve, and the metrics of the selected weights
on every split. The comparison key joins the data, the evaluator and the clock settings; two runs compare only when
their keys agree.
"""

from __future__ import annotations

import hashlib
import json
import platform
import torch

from pathlib import Path
from typing import Any

from speedrunning_plms.benchmark import architecture, clock, data, evaluate, probe, tokens


SCHEMA_VERSION = 2
EVALUATOR_MODULES = (tokens, data, evaluate, clock, architecture, probe)


def evaluator_sha256() -> str:
    """The digest of the fixed modules' source, in order."""
    digest = hashlib.sha256()
    for module in EVALUATOR_MODULES:
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def comparison_key(benchmark_id: str, evaluator: str, clock_receipt: dict[str, Any]) -> str:
    settings = {"benchmark_id": benchmark_id, "evaluator_sha256": evaluator,
                "budget_seconds": clock_receipt["budget_seconds"], "warmup_cap_seconds": clock_receipt["warmup_cap_seconds"],
                "eval_cap_seconds": clock_receipt["eval_cap_seconds"]}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()


def candidate_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(device: torch.device) -> dict[str, Any]:
    cuda = device.type == "cuda"
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda if cuda else None,
        "gpu": torch.cuda.get_device_name(device) if cuda else None,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else 0.0,
        "python": platform.python_version(), "platform": platform.platform(),
    }


def write_result(output_dir: Path, run_result: dict[str, Any]) -> Path:
    """Write result.json atomically; refuse a second result in one directory."""
    path = output_dir / "result.json"
    if path.exists():
        raise FileExistsError(f"{path} exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "result.json.tmp"
    temporary.write_text(json.dumps(run_result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def ledger_row(run_result: dict[str, Any]) -> dict[str, Any]:
    """The compact row `runs.jsonl` keeps for every run."""
    splits = run_result.get("metrics", {})
    return {
        "name": run_result["name"], "status": run_result["status"], "verdict": run_result.get("verdict"),
        "hypothesis": run_result.get("hypothesis"), "seed": run_result["seed"],
        "candidate_sha256": run_result["candidate_sha256"], "comparison_key": run_result["comparison_key"],
        "steps": run_result["clock"]["steps"], "train_seconds": run_result["clock"]["train_seconds"],
        "warmup_seconds": run_result["clock"]["warmup_seconds"], "eval_seconds": run_result["clock"]["eval_seconds"],
        "valid": run_result["clock"]["valid"], "selected_step": run_result.get("selected_step"),
        "val_select_loss": splits.get("val_select", {}).get("loss"),
        "validation_loss": splits.get("validation", {}).get("loss"),
        "test_loss": splits.get("test", {}).get("loss"),
        "test_accuracy": splits.get("test", {}).get("accuracy"),
        "test_mcc": splits.get("test", {}).get("mcc"),
        "parameters": run_result.get("architecture", {}).get("parameters"),
        "gpu": run_result.get("environment", {}).get("gpu"),
        "finished_utc": run_result.get("finished_utc"),
    }


def comparable(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["comparison_key"] == b["comparison_key"] and a["status"] == b["status"] == "completed" and a["valid"] and b["valid"]
