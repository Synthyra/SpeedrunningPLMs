"""Run evidence: what a run leaves beside its result, so a run made on someone else's GH200 can be confirmed.

`train.py` makes three calls, each outside the clock and outside the capped evals: `RunEvidence.start` once the run's
start time is taken and before data loads, `mark_training` after the compile warmup and before the first timed step,
and `finish` after `result.json` is written. They write in the run folder:

- `machine.json`: the GPU inventory (or `"none: cpu run"`), torch, CUDA, cuDNN and triton versions, every installed
  package, the host's processor and memory, an allowlisted slice of the environment with the compile caches' entry
  counts, the command, the candidate's digest and parts, the evaluator's digest, the benchmark id, a source revision,
  and the SHA-256 of every training data file, checked against the data manifest.
- `nvidia-smi_start.txt` and `nvidia-smi_end.txt`: `nvidia-smi` and `nvidia-smi -q`, whole.
- `gpu_samples.csv`: `nvidia-smi` readings every 500 ms from its own process (`foundry.provenance.GpuSampler`).
- `phases.jsonl`: start, training and finish, each with UTC and local time (the sampler's clock), the last with the
  sampler's summary over the training window.

A header and a footer a human reads in the raw log go to standard error: standard output stays one JSON object a
line, its last the ledger row. `run_gh200.sh train` adds `host.json`, `console.log`, `source.patch` and `SHA256SUMS`;
`verify_run.py` checks the folder.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
import torch

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from foundry.provenance import (
    NVIDIA_SMI, GpuSampler, SamplerSummary, allowlisted_environment, compute_apps, gpu_inventory, host_hardware,
    package_versions, query_report, source_revision, text_report,
)
from foundry.provenance.nvidia_smi import TIMESTAMP_FORMAT
from foundry.serialization.atomic import write_json_atomic, write_text_atomic
from speedrunning_plms import benchmark
from speedrunning_plms.benchmark.data import HELD_OUT_SPLITS, sha256_file


PROJECT = Path(__file__).resolve().parents[2]
ENVIRONMENT_NAMES = (
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_MAX_AUTOTUNE", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
    "PYTHONHASHSEED", "WANDB_MODE",
)
CACHE_NAMES = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")  # paths: recorded as set or not and their file counts
SOURCE_PATHS = ("train.py", "src/speedrunning_plms", "record.py", "probe.py", "prepare.py", "verify_run.py", "run_gh200.sh",
                "Dockerfile", "requirements.txt", "pyproject.toml")
CPU_RUN = "none: cpu run"
MACHINE_FILE, PHASES_FILE, SAMPLES_FILE = "machine.json", "phases.jsonl", "gpu_samples.csv"
START_REPORT, END_REPORT = "nvidia-smi_start.txt", "nvidia-smi_end.txt"


def training_data_digests(data_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    """The SHA-256 of every training file the manifest lists, each checked against it; a mismatch raises. The held-out
    files are the evaluator's to check (`benchmark.load_held_out`)."""
    held_out = {f"{split}.pt" for split in HELD_OUT_SPLITS}
    digests: dict[str, str] = {}
    for name, entry in sorted(manifest["files"].items()):
        if name in held_out:
            continue
        digest = sha256_file(data_dir / name)
        if digest != entry["sha256"]:
            raise ValueError(f"{data_dir / name} has SHA-256 {digest}, the manifest {entry['sha256']}: the training data "
                             "is not the benchmark's; rebuild it with prepare.py")
        digests[name] = digest
    return digests


def library_versions() -> dict[str, str | None]:
    """torch, its CUDA and cuDNN, and triton; a library this install lacks says so."""
    try:
        triton = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton = "not installed"
    cudnn = torch.backends.cudnn.version()
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": None if cudnn is None else str(cudnn), "triton": triton}


def local_now() -> str:
    """The local time in `nvidia-smi`'s timestamp format, so phases and samples compare on one clock."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


@dataclass
class RunEvidence:
    """The evidence files of one run, opened by `start`."""

    output_dir: Path
    device_type: str
    nvidia_smi: tuple[str, ...]
    sampler: GpuSampler | None
    training_local: datetime | None = None

    @classmethod
    def start(cls, output_dir: Path, data_dir: Path, device: torch.device, candidate: tuple[str, dict[str, str]],
              nvidia_smi: tuple[str, ...] = NVIDIA_SMI) -> RunEvidence:
        """Write `machine.json` and the first `nvidia-smi` report, start the sampler and print the header. On a CUDA
        run a missing or failing `nvidia-smi` raises here, before any training work."""
        if device.type not in ("cuda", "cpu"):
            raise ValueError(f"run evidence knows cuda and cpu runs, not {device.type}")
        cuda = device.type == "cuda"
        report = text_report(nvidia_smi) + "\n" + query_report(nvidia_smi) if cuda else ""
        gpus: Any = gpu_inventory(nvidia_smi) if cuda else CPU_RUN
        manifest = benchmark.load_manifest(data_dir)
        digest, parts = candidate
        machine = {
            "schema_version": 1, "gpu": gpus, "compute_apps": compute_apps(nvidia_smi) if cuda else [],
            "record_eligibility": "decided by verify_run.py" if cuda else f"not eligible: {CPU_RUN}",
            "versions": library_versions(), "host": host_hardware(), "packages": package_versions(),
            "environment": allowlisted_environment(ENVIRONMENT_NAMES, CACHE_NAMES), "argv": sys.argv,
            "candidate_sha256": digest, "candidate_parts": parts, "evaluator_sha256": benchmark.evaluator_sha256(),
            "benchmark_id": benchmark.benchmark_id(manifest),
            "training_data": training_data_digests(data_dir, manifest), "source": source_revision(PROJECT, SOURCE_PATHS),
        }
        write_json_atomic(output_dir / MACHINE_FILE, machine, allow_nan=False)
        sampler = None
        if cuda:
            write_text_atomic(output_dir / START_REPORT, report)
            sampler = GpuSampler.start(output_dir / SAMPLES_FILE, nvidia_smi)
        evidence = cls(output_dir, device.type, nvidia_smi, sampler)
        evidence.append_phase({"phase": "start"})
        print(header(machine, report), file=sys.stderr, flush=True)
        return evidence

    def append_phase(self, fields: dict[str, Any]) -> None:
        line = {**fields, "utc": datetime.now(UTC).isoformat(), "local": local_now()}
        with (self.output_dir / PHASES_FILE).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(line, sort_keys=True, allow_nan=False) + "\n")

    def mark_training(self) -> None:
        """The compile warmup is over; the next step is the first on the clock."""
        self.training_local = datetime.now()
        self.append_phase({"phase": "training"})

    def finish(self, run_result: dict[str, Any]) -> None:
        """Stop the sampler, write the last `nvidia-smi` report and the sampler's summary, and print the footer."""
        assert self.training_local is not None, "finish before mark_training"
        clock = run_result["clock"]
        # The clocked steps and the in-run evals ran back to back from the training mark; the final evals follow.
        end = self.training_local + timedelta(seconds=clock["train_seconds"] + clock["eval_seconds"] + clock["discarded_seconds"])
        summary = None
        if self.sampler is not None:
            summary = self.sampler.stop(window=(self.training_local, end))
            write_text_atomic(self.output_dir / END_REPORT, text_report(self.nvidia_smi) + "\n" + query_report(self.nvidia_smi))
        self.append_phase({"phase": "finish", "training_window_end_local": end.strftime(TIMESTAMP_FORMAT), "sampler": summary})
        print(footer(run_result, summary), file=sys.stderr, flush=True)


def header(machine: dict[str, Any], report: str) -> str:
    """The lines a human reads at the top of the raw log."""
    versions, parts, source = machine["versions"], machine["candidate_parts"], machine["source"]
    git = source["git"]
    git_line = f"git {git['commit']}{' (dirty)' if git['dirty'] else ''}" if git["commit"] else f"git unavailable ({git['answer']})"
    gpu = machine["gpu"] if isinstance(machine["gpu"], str) else "; ".join(f"{record['index']} {record['name']} {record['uuid']}"
                                                                           for record in machine["gpu"])
    lines = [
        f"== run evidence: start {datetime.now(UTC).isoformat(timespec='seconds')} ==",
        f"python {versions['python']} | torch {versions['torch']} | cuda {versions['cuda']} | cudnn {versions['cudnn']} | triton {versions['triton']}",
        f"candidate {machine['candidate_sha256']} (train.py {parts['train.py']}, strategies {parts['strategies']}, genome {parts['genome']})",
        f"evaluator {machine['evaluator_sha256']} | benchmark {machine['benchmark_id']} | "
        f"{len(machine['training_data'])} training data files match the manifest",
        f"source {source['paths_sha256']} over {source['files']} files | {git_line}",
        f"command: {' '.join(machine['argv'])}",
        f"gpu: {gpu}",
    ]
    return "\n".join(lines) + ("\n" + report.rstrip() if report else "") + "\n== end of run evidence header =="


def tokens_per_step(genome: dict[str, Any]) -> tuple[int, int]:
    """An optimizer step's tokens before and after the batch family's accumulation switch (`strategies.batch`)."""
    batch = genome["batch"]
    windows = batch["windows_per_step"] or 65_536 // batch["window"]  # batch.STEP_TOKENS when the count is tied
    micro = batch["window"] * (windows // batch["micro_divisor"])
    return micro * batch["accumulate_early"], micro * batch["accumulate_late"]


def footer(run_result: dict[str, Any], summary: SamplerSummary | None) -> str:
    """The lines a human reads at the end of the raw log: the clock receipt, the rates and the sampler's summary."""
    clock = run_result["clock"]
    seconds = clock["train_seconds"]
    rate = clock["steps"] / seconds if seconds else 0.0
    early, late = tokens_per_step(run_result["genome"])
    tokens = f"{rate * early:,.0f}" if early == late else f"{rate * early:,.0f} to {rate * late:,.0f}"
    lines = [
        "== run evidence: finish ==",
        f"clock: {seconds:.1f} s of {clock['budget_seconds']:.0f} s, {clock['steps']} steps, {clock['discarded_steps']} discarded "
        f"({clock['discarded_seconds']:.2f} s) | warmup {clock['warmup_seconds']:.1f} s of {clock['warmup_cap_seconds']:.0f} s | "
        f"evals {clock['eval_seconds']:.1f} s of {clock['eval_cap_seconds']:.0f} s in {clock['eval_count']} | "
        f"{'valid' if clock['valid'] else 'INVALID'}",
        f"rate: {rate:.3f} steps/s, {tokens} tokens/s",
    ]
    if summary is None:
        lines.append(f"gpu samples: {CPU_RUN}")
    else:
        lines.append(
            f"gpu samples: {summary['samples']} ({summary['window_samples']} while training), utilization median "
            f"{summary['utilization_percent_median']} %, power median {summary['power_watts_median']} W max "
            f"{summary['power_watts_max']} W, SM clock {summary['sm_clock_mhz_min']} to {summary['sm_clock_mhz_max']} MHz, "
            f"temperature max {summary['temperature_c_max']} C, largest gap {summary['largest_gap_seconds']} s, event reasons "
            f"{', '.join(summary['event_reasons_seen']) or 'none'} | sampler {summary['status']}")
    return "\n".join(lines) + "\n== end of run evidence =="
