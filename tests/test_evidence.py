"""Run evidence and the validator on a CPU smoke: what `train.py` writes beside its result, what `verify_run.py` says of
the folder, and three tamperings it must catch."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import pytest
import torch
import verify_run

from pathlib import Path
from typing import Any

from foundry.provenance import NvidiaSmiError, write_sha256sums
from speedrunning_plms import benchmark, strategies
from speedrunning_plms.evidence import CPU_RUN, ENVIRONMENT_NAMES, RunEvidence


PROJECT = Path(__file__).resolve().parents[1]
DUMMY_SECRET = "dummy-secret-value-7f3a"
SMOKE = ["--name", "smoke", "--device", "cpu", "--no-compile", "--no-wandb", "--window", "32", "--windows-per-step", "2",
         "--max-steps", "1", "--budget", "600"]


def seal(run_dir: Path) -> None:
    (run_dir / verify_run.SUMS_NAME).unlink(missing_ok=True)
    write_sha256sums(run_dir)


@pytest.fixture(scope="module")
def smoke(tiny_data: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A CPU run folder as `run_gh200.sh train` assembles it: the snapshots, the run, its console, host.json, the sums."""
    run_dir = tmp_path_factory.mktemp("smoke") / "run"
    run_dir.mkdir()
    shutil.copy2(PROJECT / "train.py", run_dir / "train.py")
    shutil.copytree(PROJECT / "src" / "speedrunning_plms" / "strategies", run_dir / "strategies",
                    ignore=shutil.ignore_patterns("__pycache__"))
    environment = {**os.environ, "SPEEDRUN_DUMMY_TOKEN": DUMMY_SECRET, "HF_TOKEN": DUMMY_SECRET, "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run([sys.executable, str(PROJECT / "train.py"), "--data-dir", str(tiny_data), "--output-dir", str(run_dir),
                                *SMOKE], capture_output=True, text=True, cwd=PROJECT, env=environment)
    assert completed.returncode == 0, completed.stderr[-3000:]
    stdout = completed.stdout.splitlines()
    rows = [json.loads(line) for line in stdout]  # standard output stays one JSON object a line
    assert rows[-1] == benchmark.ledger_row(json.loads((run_dir / "result.json").read_text(encoding="utf-8")))
    assert "== run evidence: start" in completed.stderr and "== run evidence: finish ==" in completed.stderr
    (run_dir / "console.log").write_text(completed.stderr + completed.stdout, encoding="utf-8")
    host = {"started_utc": "2026-11-14T00:00:00Z", "exit_status": 0, "compute_apps_before": "", "compute_apps_after": "",
            "finished_utc": "2026-11-14T00:01:00Z"}
    (run_dir / "host.json").write_text(json.dumps(host), encoding="utf-8")
    seal(run_dir)
    return run_dir


def tampered(smoke: Path, tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    run_dir = tmp_path / "run"
    shutil.copytree(smoke, run_dir)
    return run_dir, json.loads((run_dir / "result.json").read_text(encoding="utf-8"))


def write_result(run_dir: Path, run_result: dict[str, Any]) -> None:
    (run_dir / "result.json").write_text(json.dumps(run_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_a_cuda_start_without_the_tool_raises_before_any_training_work(tiny_data: Path, tmp_path: Path) -> None:
    with pytest.raises(NvidiaSmiError, match="not on PATH"):
        RunEvidence.start(tmp_path, tiny_data, torch.device("cuda"), ("digest", {}), nvidia_smi=("no-such-nvidia-smi",))
    assert list(tmp_path.iterdir()) == []


def test_the_smoke_writes_machine_json_and_phases(smoke: Path, tiny_data: Path) -> None:
    machine = json.loads((smoke / "machine.json").read_text(encoding="utf-8"))
    assert machine["gpu"] == CPU_RUN and machine["record_eligibility"] == f"not eligible: {CPU_RUN}"
    assert list(machine["environment"]) == sorted(ENVIRONMENT_NAMES) and DUMMY_SECRET not in json.dumps(machine)
    assert "SPEEDRUN_DUMMY_TOKEN" not in json.dumps(machine) and "HF_TOKEN" not in json.dumps(machine)
    assert isinstance(machine["environment"]["TORCHINDUCTOR_CACHE_DIR"], dict)  # a path is recorded by its count only
    manifest = benchmark.load_manifest(tiny_data)
    assert machine["training_data"] == {name: entry["sha256"] for name, entry in manifest["files"].items() if not name.endswith(".pt")}
    assert machine["benchmark_id"] == benchmark.benchmark_id(manifest)
    phases = [json.loads(line) for line in (smoke / "phases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [phase["phase"] for phase in phases] == ["start", "training", "finish"] and phases[-1]["sampler"] is None
    assert not (smoke / "gpu_samples.csv").exists() and not (smoke / "nvidia-smi_start.txt").exists()


def test_the_validator_fails_the_smoke_only_on_hardware(smoke: Path, tiny_data: Path) -> None:
    findings = verify_run.verify(smoke, tiny_data)
    assert verify_run.verdict(findings) == "FAIL" and findings
    assert {finding.check for finding in findings} == {"hardware"}, findings


def test_the_rescore_reproduces_the_smoke_on_the_cpu(smoke: Path, tiny_data: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "dont_write_bytecode", False)  # as in the image, where nothing turns bytecode off
    monkeypatch.setattr(sys, "pycache_prefix", sys.pycache_prefix)  # the rescore sets it; restored after the test
    findings = verify_run.verify(smoke, tiny_data, torch.device("cpu"))
    assert [finding for finding in findings if finding.check != "hardware"] == []
    assert not (smoke / "__pycache__").exists()  # the rescore added nothing to the sealed folder


def test_the_probe_s_files_after_sealing_are_noted_and_any_other_fails(smoke: Path, tmp_path: Path) -> None:
    run_dir, _ = tampered(smoke, tmp_path)
    (run_dir / "probe.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "__pycache__").mkdir()
    (run_dir / "__pycache__" / "train.cpython-312.pyc").write_bytes(b"\x00")
    findings = verify_run.verify(run_dir)
    assert {finding.reason.split()[0] for finding in findings if finding.level == "NOTE"} == {"probe.json", "__pycache__/train.cpython-312.pyc"}
    assert {finding.check for finding in findings if finding.level != "NOTE"} == {"hardware"}
    (run_dir / "notes.txt").write_text("added by hand\n", encoding="utf-8")
    assert "sums" in {finding.check for finding in verify_run.verify(run_dir) if finding.level == "FAIL"}


def test_an_edited_curve_breaks_the_sums(smoke: Path, tmp_path: Path) -> None:
    run_dir, run_result = tampered(smoke, tmp_path)
    run_result["curve"][0]["loss"] -= 0.5
    write_result(run_dir, run_result)
    findings = verify_run.verify(run_dir)
    assert "sums" in {finding.check for finding in findings if finding.level == "FAIL"}


def test_a_resealed_warmup_over_the_cap_fails_the_cap(smoke: Path, tmp_path: Path) -> None:
    run_dir, run_result = tampered(smoke, tmp_path)
    run_result["clock"]["warmup_seconds"] = run_result["clock"]["warmup_cap_seconds"] + 20.0  # the flags still say valid
    write_result(run_dir, run_result)
    seal(run_dir)
    failed = {finding.check for finding in verify_run.verify(run_dir) if finding.level == "FAIL"}
    assert "clock" in failed and "sums" not in failed


def test_a_changed_genome_breaks_the_digest(smoke: Path, tmp_path: Path) -> None:
    run_dir, run_result = tampered(smoke, tmp_path)
    run_result["genome"]["optimizer"]["muon_lr"] *= 2
    write_result(run_dir, run_result)
    seal(run_dir)
    failed = {finding.check for finding in verify_run.verify(run_dir) if finding.level == "FAIL"}
    assert "candidate" in failed and "sums" not in failed


def test_the_validator_digests_equal_the_package_digests() -> None:
    genome = strategies.default_genome()
    assert verify_run.genome_sha256(genome) == strategies.genome_sha256(genome)
    assert verify_run.strategies_sha256(PROJECT / "src" / "speedrunning_plms" / "strategies") == strategies.sources_sha256()
