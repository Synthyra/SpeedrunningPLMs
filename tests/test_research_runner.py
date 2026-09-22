"""Offline launcher tests use tiny stand-in workers, never GPUs or SSH hosts."""

import hashlib
import io
import json
import os
import shlex
import signal
import subprocess
import sys
import zipfile
import pytest

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

from speedrunning_plms.research import runner


BENCHMARK_SOURCE = b"# fixed benchmark\n"


def valid_result(**changes: object) -> dict[str, object]:
    return {"schema_version": 1, "status": "completed", "objective": "masked15",
            "eval_dtype": "float32", "seed": 42, "device": "cpu", "gpu_names": [None],
            "cpu_name": "test-cpu", "torch_version": "2.6.0", "transformers_version": "4.57.6",
            "benchmark_id": "dataset-v1", "benchmark_code_sha256": hashlib.sha256(BENCHMARK_SOURCE).hexdigest(),
            "split": "valid", "val_bits_per_masked_residue": 2.5,
            "world_size": 1, "time_budget": 1, "train_seconds": 1, "config": {}, **changes}


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    package = root / "src" / "speedrunning_plms" / "research"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package.parent / "__init__.py").write_text("")
    (package / "benchmark.py").write_bytes(BENCHMARK_SOURCE)
    (package / "engine.py").write_text(
        "import argparse,json,pathlib\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--output-dir');p.add_argument('--data-dir');p.add_argument('--time-budget',type=float)\n"
        "a=p.parse_args()\n"
        "out=pathlib.Path(a.output_dir);out.mkdir(parents=True)\n"
        f"result={valid_result()!r}\n"
        "result['time_budget']=a.time_budget\n"
        "(out/'result.json').write_text(json.dumps(result))\n"
        "print('worker finished',flush=True)\n", encoding="utf-8")
    return root


@pytest.fixture
def target(tmp_path: Path) -> runner.Target:
    return runner.Target("cpu-smoke", (runner.Host(None, str(tmp_path / "staging"), sys.executable),))


def test_local_execution_stages_snapshot_collects_result_and_ledger(source_root: Path, target: runner.Target, tmp_path: Path) -> None:
    output = tmp_path / "runs"
    record = runner.run_experiment(target, source_root, output, "baseline", str(tmp_path), 1, 30)
    assert record["status"] == "completed"
    assert record["comparable"] is True
    assert record["result"]["val_bits_per_masked_residue"] == 2.5
    run_dir = output / record["run_id"]
    assert json.loads((run_dir / "launcher.json").read_text()) == json.loads((output / "results.jsonl").read_text())
    assert "worker finished" in (run_dir / "node-0.log").read_text()
    assert (run_dir / "source.zip").exists()
    assert (Path(record["node_dirs"][0]) / "source/src/speedrunning_plms/research/engine.py").exists()


def test_snapshot_is_reproducible_and_excludes_non_source(source_root: Path, tmp_path: Path) -> None:
    (source_root / ".env").write_text("private")
    (source_root / "data").mkdir()
    (source_root / "data" / "secret.py").write_text("private")
    (source_root / "src" / "speedrunning_plms" / "credential.json").write_text("private")
    first, digest = runner.source_snapshot(source_root)
    assert (first, digest) == runner.source_snapshot(source_root)
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert set(archive.namelist()) == {"src/speedrunning_plms/__init__.py", "src/speedrunning_plms/research/__init__.py", "src/speedrunning_plms/research/engine.py", "src/speedrunning_plms/research/benchmark.py"}
    config = tmp_path / "config.json"
    config.write_text('{"hidden_size":32}')
    snapshot, other_digest = runner.source_snapshot(source_root, config)
    assert other_digest != digest
    with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
        assert json.loads(archive.read("experiment.json")) == {"hidden_size": 32}


@pytest.mark.parametrize("candidate", [{"split": "test"}, {"evaluate_only": True}, []])
def test_runner_rejects_held_out_evaluation_before_launch(source_root: Path, tmp_path: Path, candidate: object) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(candidate))
    with pytest.raises(ValueError):
        runner.source_snapshot(source_root, config)


def test_dry_run_does_not_stage_or_connect(source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = runner.Target("remote", (runner.Host("gpu-a", "/scratch/experiments"),))
    forbidden = Mock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    record = runner.run_experiment(target, source_root, tmp_path / "runs", "candidate", "/data", dry_run=True)
    assert record["status"] == "planned"
    assert not (tmp_path / "runs").exists()
    assert record["commands"][0][:3] == ["python", "-m", "speedrunning_plms.research.engine"]


@pytest.mark.parametrize("change", [
    {"hosts": []}, {"hosts": [{"host": "-ProxyCommand=bad", "workdir": "/tmp"}]},
    {"hosts": [{"host": "gpu;echo bad", "workdir": "/tmp"}]},
    {"hosts": [{"host": "gpu", "workdir": "relative"}]},
    {"hosts": [{"host": "gpu", "workdir": "/tmp", "gpus": 0}]},
    {"hosts": [{"host": "gpu", "workdir": "/tmp", "gpus": True}]},
    {"master_port": 65536}, {"master_addr": "gpu;bad"},
    {"hosts": [{"host": "a", "workdir": "/tmp"}, {"host": "b", "workdir": "/tmp"}]},
    {"hosts": [{"host": "a", "workdir": "/tmp", "gpus": 1}, {"host": "b", "workdir": "/tmp", "gpus": 2}], "master_addr": "a"},
])
def test_target_rejects_invalid_hosts_resources_and_rendezvous(tmp_path: Path, change: dict[str, object]) -> None:
    path = tmp_path / "target.json"
    path.write_text(json.dumps({"name": "gpu", "hosts": [{"host": "gpu", "workdir": "/tmp"}], **change}))
    with pytest.raises(ValueError):
        runner.load_target(path)


def test_multinode_target_constructs_each_rank_and_resource_count(tmp_path: Path) -> None:
    path = tmp_path / "target.json"
    path.write_text(json.dumps({"name": "cluster", "hosts": [
        {"host": "gpu-a", "workdir": "/scratch/experiments", "gpus": 4},
        {"host": "gpu-b", "workdir": "/scratch/experiments", "gpus": 4}], "master_addr": "10.0.0.1"}))
    target = runner.load_target(path)
    command = runner.engine_command(target, 1, "/data", "/output", 300, True)
    assert command[:3] == ["python", "-m", "torch.distributed.run"]
    assert "--nproc-per-node=4" in command and "--nnodes=2" in command
    assert "--node-rank=1" in command and "--master-addr=10.0.0.1" in command
    assert command[-2:] == ["--config", "experiment.json"]


@pytest.mark.parametrize("changes", [{"split": "test"}, {"val_bits_per_masked_residue": float("nan")},
    {"val_bits_per_masked_residue": float("inf")}, {"val_bits_per_masked_residue": -1},
    {"val_bits_per_masked_residue": True}, {"benchmark_id": ""}, {"world_size": 8},
    {"time_budget": 30}, {"time_budget": True}, {"world_size": True},
    {"benchmark_code_sha256": None}, {"config": []},
    {"schema_version": True}, {"schema_version": "1"}, {"schema_version": 2},
    {"status": "failed"}, {"objective": "diffusion"}, {"eval_dtype": "bfloat16"},
    {"seed": True}, {"seed": "42"}, {"torch_version": ""}, {"transformers_version": 5},
    {"cpu_name": None}, {"device": "auto"}, {"gpu_names": []},
    {"train_seconds": float("nan")}, {"train_seconds": float("inf")},
    {"train_seconds": -1}, {"train_seconds": True}, {"train_seconds": "1"},
    {"gpu_names": ["GPU"]}, {"device": "cuda:0", "gpu_names": [None]}])
def test_invalid_results_never_receive_scores(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        runner.validate_result(valid_result(**changes), 1, 1)


@pytest.mark.parametrize("field", ["schema_version", "status", "objective", "eval_dtype", "config",
    "seed", "device", "gpu_names", "cpu_name", "torch_version", "transformers_version", "train_seconds"])
def test_result_contract_requires_reproducibility_fields(field: str) -> None:
    result = valid_result()
    del result[field]
    with pytest.raises(ValueError):
        runner.validate_result(result, 1, 1)


def test_result_accepts_rank_ordered_cuda_hardware_metadata() -> None:
    runner.validate_result(valid_result(device="cuda:0", gpu_names=["A100", "A100"], world_size=2), 2, 1)


def test_wrong_imported_benchmark_cannot_enter_ledger(source_root: Path, target: runner.Target, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock()
    process.poll.return_value = 0
    monkeypatch.setattr(runner, "_launch", Mock(return_value=process))
    monkeypatch.setattr(runner, "_stop", Mock())
    monkeypatch.setattr(runner, "_fetch_result", Mock(return_value=valid_result(benchmark_code_sha256="a" * 64)))
    with pytest.raises(ValueError, match="differs from the staged snapshot"):
        runner.run_experiment(target, source_root, tmp_path / "runs", "wrong-import", str(tmp_path), 1, 30)
    record = json.loads((tmp_path / "runs/results.jsonl").read_text())
    assert record["comparable"] is False and record["status"] == "failed"
    assert "comparison_key" not in record


@pytest.mark.parametrize("changes", [{"cpu_name": "different-cpu"}, {"torch_version": "2.7.0"},
    {"transformers_version": "4.58.0"}, {"seed": 43},
    {"device": "cuda:0", "gpu_names": ["A100"]}])
def test_hardware_software_and_seed_define_comparison_tracks(source_root: Path, target: runner.Target, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]) -> None:
    process = Mock()
    process.poll.return_value = 0
    monkeypatch.setattr(runner, "_launch", Mock(return_value=process))
    monkeypatch.setattr(runner, "_fetch_result", Mock(side_effect=[valid_result(), valid_result(**changes)]))
    first = runner.run_experiment(target, source_root, tmp_path / "runs", "baseline", str(tmp_path), 1, 30)
    second = runner.run_experiment(target, source_root, tmp_path / "runs", "changed", str(tmp_path), 1, 30)
    assert first["comparison_key"] != second["comparison_key"]


@pytest.mark.parametrize("train_seconds,comparable", [(0, True), (1, True), (1.05, True), (1.050001, False), (20, False)])
def test_training_overrun_preserves_artifacts_but_excludes_unfair_scores(source_root: Path, target: runner.Target, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, train_seconds: float, comparable: bool) -> None:
    process = Mock()
    process.poll.return_value = 0
    monkeypatch.setattr(runner, "_launch", Mock(return_value=process))
    monkeypatch.setattr(runner, "_fetch_result", Mock(return_value=valid_result(train_seconds=train_seconds)))
    record = runner.run_experiment(target, source_root, tmp_path / "runs", "timed", str(tmp_path), 1, 30)
    assert record["status"] == "completed"
    assert record["comparable"] is comparable
    assert (tmp_path / "runs" / record["run_id"] / "result.json").is_file()
    assert record["result"]["val_bits_per_masked_residue"] == 2.5
    if comparable:
        assert "comparison_exclusion_reason" not in record
    else:
        assert record["comparison_exclusion_reason"] == "Training budget exceeded by more than 5%"
    assert json.loads((tmp_path / "runs/results.jsonl").read_text())["comparable"] is comparable


def test_remote_stage_uses_stdin_and_quotes_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    call = Mock()
    monkeypatch.setattr(runner.subprocess, "run", call)
    runner._stage(runner.Host("gpu-box", "/scratch/my runs"), "/scratch/my runs/run", b"source archive")
    argv = call.call_args.args[0]
    assert argv[:7] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "--", "gpu-box"]
    assert "'/scratch/my runs/run'" in argv[-1]
    assert call.call_args.kwargs["input"] == b"source archive"
    assert call.call_args.kwargs["timeout"] == 60


def test_remote_launch_has_independent_timeout_and_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    launch = Mock()
    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    runner._launch(runner.Host("gpu-box", "/scratch"), "/scratch/run/node-0", ["python", "-m", "engine"], 600, io.BytesIO())
    command = launch.call_args.args[0][-1]
    assert "setsid --wait" in command and "timeout --signal=TERM --kill-after=45s 600" in command
    assert "process-group.pid" in command and "PYTHONPATH=/scratch/run/node-0/source/src" in command
    assert "cancel.requested" in command and "exit 130" in command


def test_remote_cancellation_targets_remote_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    execute = Mock()
    monkeypatch.setattr(runner.subprocess, "run", execute)
    process = Mock()
    process.poll.return_value = 1
    runner._stop(runner.Host("gpu-box", "/scratch"), "/scratch/run/node-0", process)
    command = execute.call_args.args[0]
    assert command[-2] == "gpu-box"
    assert "os.killpg" in command[-1] and "signal.SIGKILL" in command[-1]
    assert "process-group.pid" in command[-1]


@pytest.mark.parametrize("exits_after_term", [False, True])
def test_remote_cancellation_allows_worker_cleanup_before_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exits_after_term: bool,
) -> None:
    execute = Mock()
    monkeypatch.setattr(runner.subprocess, "run", execute)
    process = Mock()
    process.poll.return_value = 0
    runner._stop(runner.Host("gpu-box", "/scratch"), "/scratch/run", process)
    command = shlex.split(execute.call_args.args[0][-1])
    assert command[1] == "-c"
    pid_path = tmp_path / "process-group.pid"
    pid_path.write_text("1234", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["-c", str(pid_path)])
    proc_command = Path("/proc") / "1234" / "cmdline"
    original_exists = Path.exists
    original_read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "exists", lambda path: path == proc_command or original_exists(path))
    monkeypatch.setattr(
        Path, "read_bytes",
        lambda path: str(tmp_path).encode() if path == proc_command else original_read_bytes(path),
    )
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    signals = []

    def killpg(group: int, received_signal: int) -> None:
        assert group == 1234
        signals.append(received_signal)
        if exits_after_term and received_signal == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", killpg, raising=False)
    monkeypatch.setattr(runner.time, "monotonic", Mock(side_effect=[0, 1, 41]))
    monkeypatch.setattr(runner.time, "sleep", Mock())
    exec(compile(command[2], "remote-cancellation", "exec"), {})

    expected = [signal.SIGTERM, 0]
    if not exits_after_term:
        expected.append(signal.SIGKILL)
    assert signals == expected
    assert execute.call_args.kwargs["check"] is True
    assert (tmp_path / "cancel.requested").exists()


def test_cancellation_before_remote_start_leaves_durable_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    execute = Mock()
    monkeypatch.setattr(runner.subprocess, "run", execute)
    process = Mock()
    process.poll.return_value = 0
    runner._stop(runner.Host("gpu-box", "/scratch"), "/scratch/run", process)
    command = shlex.split(execute.call_args.args[0][-1])
    monkeypatch.setattr(sys, "argv", ["-c", str(tmp_path / "process-group.pid")])
    killpg = Mock(side_effect=AssertionError("No process exists to cancel"))
    monkeypatch.setattr(os, "killpg", killpg, raising=False)

    with pytest.raises(SystemExit) as stopped:
        exec(compile(command[2], "remote-cancellation", "exec"), {})

    assert stopped.value.code == 0
    assert (tmp_path / "cancel.requested").is_file()
    killpg.assert_not_called()


@pytest.mark.parametrize("changes", [
    {"max_steps": None, "config": {"max_steps": 1}},
    {"max_steps": 1, "config": {}},
    {"evaluate_only": None, "config": {"evaluate_only": "checkpoint"}},
])
def test_comparison_excludes_limited_runs_from_either_metadata_level(
    target: runner.Target, changes: dict[str, object],
) -> None:
    comparison = runner._comparison_metadata(valid_result(**changes), target, 1)
    assert comparison["comparable"] is False


def test_concurrent_results_preserve_every_ledger_record(tmp_path: Path) -> None:
    def save(index: int) -> None:
        run_dir = tmp_path / str(index)
        run_dir.mkdir()
        runner._save_record({"run_id": index, "message": "x" * 1000}, run_dir, tmp_path)

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(save, range(12)))
    records = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert sorted(record["run_id"] for record in records) == list(range(12))
    for record in records:
        assert json.loads((tmp_path / str(record["run_id"]) / "launcher.json").read_text()) == record


def test_invalid_completed_result_is_not_scored(source_root: Path, target: runner.Target, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock()
    process.poll.return_value = 0
    monkeypatch.setattr(runner, "_launch", Mock(return_value=process))
    monkeypatch.setattr(runner, "_stop", Mock())
    monkeypatch.setattr(runner, "_fetch_result", Mock(return_value=valid_result(split="test")))
    with pytest.raises(ValueError, match="validation split"):
        runner.run_experiment(target, source_root, tmp_path / "runs", "invalid", str(tmp_path), 1, 30)
    record = json.loads((tmp_path / "runs/results.jsonl").read_text())
    assert record["status"] == "failed" and record["comparable"] is False
    assert "result" not in record


def test_runner_requires_repository_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository root"):
        runner.source_snapshot(tmp_path)


@pytest.mark.parametrize("failure", [RuntimeError("worker failure"), TimeoutError("deadline"), subprocess.CalledProcessError(1, "ssh")])
def test_failures_are_recorded_without_comparable_score(source_root: Path, target: runner.Target, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    monkeypatch.setattr(runner, "_launch", Mock(side_effect=failure))
    with pytest.raises(type(failure)):
        runner.run_experiment(target, source_root, tmp_path / "runs", "failed", str(tmp_path), 1, 30)
    record = json.loads((tmp_path / "runs/results.jsonl").read_text())
    assert record["status"] == "failed" and record["comparable"] is False
    assert "result" not in record and "comparison_key" not in record


def test_failed_rank_stops_other_ranks(source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = runner.Target("cluster", (runner.Host("a", "/scratch"), runner.Host("b", "/scratch")), "a")
    processes = [Mock(), Mock()]
    processes[0].poll.return_value = 1
    processes[1].poll.return_value = None
    monkeypatch.setattr(runner, "_stage", Mock())
    monkeypatch.setattr(runner, "_launch", Mock(side_effect=processes))
    stop = Mock()
    monkeypatch.setattr(runner, "_stop", stop)
    with pytest.raises(RuntimeError, match="worker failed"):
        runner.run_experiment(target, source_root, tmp_path / "runs", "failed", "/data", 1, 30)
    assert stop.call_count == 2
    assert stop.call_args_list[1].args[-1] is processes[1]


def test_timeout_cancels_worker_and_records_failure(source_root: Path, target: runner.Target, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(runner, "_launch", Mock(return_value=process))
    monkeypatch.setattr(runner.time, "monotonic", Mock(side_effect=[0, 31]))
    stop = Mock()
    monkeypatch.setattr(runner, "_stop", stop)
    with pytest.raises(TimeoutError):
        runner.run_experiment(target, source_root, tmp_path / "runs", "timeout", str(tmp_path), 1, 30)
    stop.assert_called_once()
    assert json.loads((tmp_path / "runs/results.jsonl").read_text())["comparable"] is False


def test_smoke_runs_are_recorded_but_not_comparable(source_root: Path, target: runner.Target, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock()
    process.poll.return_value = 0
    monkeypatch.setattr(runner, "_launch", Mock(return_value=process))
    monkeypatch.setattr(runner, "_fetch_result", Mock(return_value=valid_result(config={"max_steps": 1})))
    record = runner.run_experiment(target, source_root, tmp_path / "runs", "smoke", str(tmp_path), 1, 30)
    assert record["status"] == "completed" and record["comparable"] is False


def test_comparison_key_excludes_candidate_source_but_includes_budget(source_root: Path, target: runner.Target, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock()
    process.poll.return_value = 0
    monkeypatch.setattr(runner, "_launch", Mock(return_value=process))
    monkeypatch.setattr(runner, "_fetch_result", Mock(side_effect=[valid_result(), valid_result(), valid_result(time_budget=2)]))
    first = runner.run_experiment(target, source_root, tmp_path / "runs", "a", str(tmp_path), 1, 30)
    (source_root / "src/speedrunning_plms/research/engine.py").write_text("# changed architecture")
    second = runner.run_experiment(target, source_root, tmp_path / "runs", "b", str(tmp_path), 1, 30)
    third = runner.run_experiment(target, source_root, tmp_path / "runs", "c", str(tmp_path), 2, 30)
    assert first["source_sha256"] != second["source_sha256"]
    assert first["comparison_key"] == second["comparison_key"]
    assert second["comparison_key"] != third["comparison_key"]
