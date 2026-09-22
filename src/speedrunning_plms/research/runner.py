"""Stage and run isolated experiments locally or on existing SSH GPU hosts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shlex
import signal
import subprocess
import time
import uuid
import zipfile

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


JsonRecord = dict[str, Any]


@dataclass(frozen=True)
class Host:
    host: str | None
    workdir: str
    python: str = "python"
    gpus: int = 1


@dataclass(frozen=True)
class Target:
    name: str
    hosts: tuple[Host, ...]
    master_addr: str | None = None
    master_port: int = 29500


def load_target(path: Path) -> Target:
    """Read explicit host settings without probing the network."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("hosts"), list):
        raise ValueError("Target requires a hosts list")
    hosts = tuple(Host(**entry) for entry in payload.pop("hosts"))
    target = Target(hosts=hosts, **payload)
    if not target.name or not hosts:
        raise ValueError("Target requires a name and at least one host")
    for host in hosts:
        if host.host is not None and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", host.host):
            raise ValueError("host must be a hostname or SSH configuration alias")
        absolute = PurePosixPath(host.workdir).is_absolute() if host.host else Path(host.workdir).is_absolute()
        if not absolute:
            raise ValueError("workdir must be absolute")
        if not isinstance(host.gpus, int) or isinstance(host.gpus, bool) or host.gpus < 1:
            raise ValueError("gpus must be a positive integer")
        if not host.python or "\x00" in host.python or "\n" in host.python:
            raise ValueError("python must be an executable name or path")
    if len({host.gpus for host in hosts}) != 1:
        raise ValueError("All hosts must use the same GPUs per node")
    if len(hosts) > 1 and not target.master_addr:
        raise ValueError("Multi-node targets require master_addr reachable from every node")
    if target.master_addr and not re.fullmatch(r"[A-Za-z0-9_.:-]+", target.master_addr):
        raise ValueError("Invalid master_addr")
    if type(target.master_port) is not int or not 1 <= target.master_port <= 65535:
        raise ValueError("master_port must be between 1 and 65535")
    return target


def source_snapshot(root: Path, config: Path | None = None) -> tuple[bytes, str]:
    """Archive source only; datasets, credentials, caches, and Git stay local."""
    if not all((root / "src/speedrunning_plms/research" / name).is_file() for name in ("engine.py", "benchmark.py")):
        raise ValueError("Run the launcher from the repository root containing the research engine and benchmark")
    paths = sorted((root / "src" / "speedrunning_plms").rglob("*.py"))
    paths += [root / name for name in ("train.py", "research.py", "pyproject.toml") if (root / name).is_file()]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Source symlinks outside the snapshot are unsupported: {path}")
            name = path.relative_to(root).as_posix()
            # Fixed metadata gives identical source bytes an identical digest.
            archive.writestr(zipfile.ZipInfo(name), path.read_bytes())
        if config is not None:
            candidate = json.loads(config.read_text(encoding="utf-8"))
            if not isinstance(candidate, dict):
                raise ValueError("Experiment config must be a JSON object")
            if candidate.get("split", "valid") != "valid" or candidate.get("evaluate_only", False):
                raise ValueError("Research runner accepts validation training experiments only")
            archive.writestr(zipfile.ZipInfo("experiment.json"), json.dumps(candidate, sort_keys=True))
    contents = stream.getvalue()
    return contents, hashlib.sha256(contents).hexdigest()


def _ssh(host: str, command: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "--", host, command]


def _node_dir(host: Host, run_id: str, rank: int) -> str:
    path_type = PurePosixPath if host.host else Path
    return str(path_type(host.workdir) / run_id / f"node-{rank}")


def engine_command(
    target: Target, rank: int, data_dir: str, output_dir: str,
    time_budget: float, has_config: bool,
) -> list[str]:
    host = target.hosts[rank]
    command = [host.python]
    if len(target.hosts) * host.gpus > 1:
        command += ["-m", "torch.distributed.run", f"--nproc-per-node={host.gpus}",
                    f"--nnodes={len(target.hosts)}", f"--node-rank={rank}"]
        if len(target.hosts) == 1:
            command += ["--standalone"]
        else:
            command += [f"--master-addr={target.master_addr}", f"--master-port={target.master_port}"]
    command += ["-m", "speedrunning_plms.research.engine", "--data-dir", data_dir,
                "--output-dir", output_dir, "--time-budget", str(time_budget)]
    if has_config:
        command += ["--config", "experiment.json"]
    return command


def _stage(host: Host, node_dir: str, snapshot: bytes) -> None:
    if host.host:
        script = ("import io,pathlib,sys,zipfile; "
                  "p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=False); "
                  "zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read())).extractall(p/'source')")
        command = shlex.join([host.python, "-c", script, node_dir])
        subprocess.run(_ssh(host.host, command), input=snapshot, check=True, timeout=60,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        directory = Path(node_dir)
        directory.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
            archive.extractall(directory / "source")


def _launch(
    host: Host, node_dir: str, command: list[str], timeout: float, log: BinaryIO,
) -> subprocess.Popen[bytes]:
    source_dir = str((PurePosixPath(node_dir) if host.host else Path(node_dir)) / "source")
    working_directory = None
    environment = None
    if host.host:
        # GNU timeout bounds the GPU process even if the workstation disconnects.
        pidfile = str(PurePosixPath(node_dir) / "process-group.pid")
        cancellation = str(PurePosixPath(node_dir) / "cancel.requested")
        child = (f"echo $$ > {shlex.quote(pidfile)}; "
                 f"if [ -e {shlex.quote(cancellation)} ]; then exit 130; fi; exec " +
                 shlex.join(["timeout", "--signal=TERM", "--kill-after=45s", str(timeout),
                             "env", f"PYTHONPATH={source_dir}/src", *command]))
        remote = f"cd {shlex.quote(source_dir)} && exec setsid --wait sh -c {shlex.quote(child)}"
        command = _ssh(host.host, remote)
    else:
        working_directory = source_dir
        environment = {**os.environ, "PYTHONPATH": str(Path(source_dir) / "src")}
    return subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, cwd=working_directory, env=environment,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )


def _stop(host: Host, node_dir: str, process: subprocess.Popen[bytes]) -> None:
    remote_error = None
    if host.host:
        # torchrun forwards TERM to separate worker groups and allows 30s to stop.
        script = """import os,pathlib,signal,sys,time
p = pathlib.Path(sys.argv[1])
p.with_name('cancel.requested').touch()
if not p.exists():
    sys.exit(0)
group = int(p.read_text())
command = pathlib.Path('/proc') / str(group) / 'cmdline'
if not command.exists() or str(p.parent).encode() not in command.read_bytes():
    sys.exit(0)
try:
    os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        os.killpg(group, 0)
        time.sleep(0.1)
    os.killpg(group, signal.SIGKILL)
except ProcessLookupError:
    pass
"""
        command = shlex.join([host.python, "-c", script, str(PurePosixPath(node_dir) / "process-group.pid")])
        try:
            subprocess.run(_ssh(host.host, command), timeout=55, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as error:
            remote_error = error
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=20)
    if remote_error is not None:
        # Preserve the failure in launcher.json; the remote timeout still applies.
        raise remote_error


def _fetch_result(host: Host, node_dir: str) -> JsonRecord:
    result_path = str((PurePosixPath(node_dir) if host.host else Path(node_dir)) / "output" / "result.json")
    if host.host:
        command = shlex.join([host.python, "-c", "import pathlib,sys; sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())", result_path])
        result = subprocess.run(_ssh(host.host, command), check=True, capture_output=True, timeout=30)
        payload = json.loads(result.stdout)
    else:
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Engine result must be a JSON object")
    return payload


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_result(
    result: JsonRecord, world_size: int, time_budget: float,
    benchmark_sha256: str | None = None,
) -> None:
    if type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        raise ValueError("Engine result requires integer schema_version=1")
    for field, expected in (("status", "completed"), ("objective", "masked15"), ("eval_dtype", "float32")):
        if result.get(field) != expected:
            raise ValueError(f"Engine result requires {field}={expected}")
    if result.get("split") != "valid":
        raise ValueError("Research runs must report the validation split")
    score = result.get("val_bits_per_masked_residue")
    if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or score < 0:
        raise ValueError("Engine result requires finite val_bits_per_masked_residue")
    if not isinstance(result.get("benchmark_id"), str) or not result["benchmark_id"]:
        raise ValueError("Engine result requires benchmark_id")
    if not isinstance(result.get("benchmark_code_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", result["benchmark_code_sha256"]):
        raise ValueError("Engine result requires benchmark_code_sha256")
    if benchmark_sha256 is not None and result["benchmark_code_sha256"] != benchmark_sha256:
        raise ValueError("Engine imported benchmark code that differs from the staged snapshot")
    if not isinstance(result.get("config"), dict):
        raise ValueError("Engine result config must be a JSON object")
    if type(result.get("world_size")) is not int or isinstance(result.get("time_budget"), bool):
        raise ValueError("Engine result requires numeric resource metadata")
    if result.get("world_size") != world_size or result.get("time_budget") != time_budget:
        raise ValueError("Engine result resource budget does not match the requested run")
    train_seconds = result.get("train_seconds")
    if isinstance(train_seconds, bool) or not isinstance(train_seconds, (float, int)) or not math.isfinite(train_seconds) or train_seconds < 0:
        raise ValueError("Engine result requires finite nonnegative train_seconds")
    if type(result.get("seed")) is not int:
        raise ValueError("Engine result requires an integer seed")
    for field in ("torch_version", "transformers_version"):
        if not isinstance(result.get(field), str) or not result[field]:
            raise ValueError(f"Engine result requires {field}")
    if not isinstance(result.get("cpu_name"), str):
        raise ValueError("Engine result requires cpu_name")
    device = result.get("device")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
        raise ValueError("Engine result requires resolved cpu or cuda device")
    gpu_names = result.get("gpu_names")
    if not isinstance(gpu_names, list) or len(gpu_names) != world_size:
        raise ValueError("Engine result requires one gpu_names entry per rank")
    if device == "cpu" and any(name is not None for name in gpu_names):
        raise ValueError("CPU result gpu_names must contain null entries")
    if device.startswith("cuda") and any(not isinstance(name, str) or not name for name in gpu_names):
        raise ValueError("CUDA result requires a GPU name for every rank")


def _wait_for_workers(processes: Sequence[subprocess.Popen[bytes]], deadline: float, run_dir: Path) -> None:
    while True:
        codes = [process.poll() for process in processes]
        if any(code is not None and code != 0 for code in codes):
            raise RuntimeError(f"A worker failed: exit codes {codes}; see {run_dir}")
        if all(code == 0 for code in codes):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Experiment exceeded its timeout; see {run_dir}")
        time.sleep(0.1)


def _comparison_metadata(result: JsonRecord, target: Target, time_budget: float) -> JsonRecord:
    configuration = result["config"]
    smoke_run = result.get("max_steps") is not None or configuration.get("max_steps") is not None
    evaluation_only = bool(result.get("evaluate_only") or configuration.get("evaluate_only"))
    comparison: JsonRecord = {"comparable": not smoke_run and not evaluation_only}
    if result["train_seconds"] > time_budget * 1.05:
        comparison["comparable"] = False
        comparison["comparison_exclusion_reason"] = "Training budget exceeded by more than 5%"
    fields = (
        "benchmark_id", "benchmark_code_sha256", "seed", "world_size", "device", "gpu_names",
        "cpu_name", "torch_version", "transformers_version", "eval_dtype",
    )
    conditions = {field: result[field] for field in fields}
    comparison["comparison_key"] = _digest({**conditions, "target": asdict(target), "time_budget": time_budget})
    return comparison


@contextmanager
def _ledger_lock(path: Path) -> Iterator[None]:
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _save_record(record: JsonRecord, run_dir: Path, output_root: Path) -> None:
    temporary = run_dir / "launcher.json.tmp"
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(run_dir / "launcher.json")
    with _ledger_lock(output_root / "results.lock"):
        with (output_root / "results.jsonl").open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(record) + "\n")


def run_experiment(
    target: Target, root: Path, output_root: Path, name: str, data_dir: str,
    time_budget: float = 300, timeout: float | None = None,
    config: Path | None = None, dry_run: bool = False, description: str = "",
) -> JsonRecord:
    """Run all ranks and save status, source, logs, and validation results."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name):
        raise ValueError("name must be 1-80 letters, digits, dots, underscores, or hyphens")
    if not math.isfinite(time_budget) or time_budget <= 0:
        raise ValueError("time_budget must be finite and positive")
    timeout = time_budget + 300 if timeout is None else timeout
    if not math.isfinite(timeout) or timeout <= time_budget:
        raise ValueError("timeout must exceed time_budget to allow startup and evaluation")
    if not all((PurePosixPath(data_dir).is_absolute() if host.host else Path(data_dir).is_absolute()) for host in target.hosts):
        raise ValueError("data-dir must be absolute and accessible at the same path on every host")
    snapshot, source_sha = source_snapshot(root, config)
    with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
        benchmark_sha = hashlib.sha256(archive.read("src/speedrunning_plms/research/benchmark.py")).hexdigest()
    run_id = f"{name}-{uuid.uuid4().hex[:12]}"
    node_dirs = [_node_dir(host, run_id, rank) for rank, host in enumerate(target.hosts)]
    commands = []
    for rank, (host, directory) in enumerate(zip(target.hosts, node_dirs)):
        node_path = PurePosixPath(directory) if host.host else Path(directory)
        commands.append(engine_command(
            target, rank, data_dir, str(node_path / "output"), time_budget, config is not None,
        ))
    record: JsonRecord = {
        "schema_version": 1, "run_id": run_id, "name": name, "status": "planned",
        "description": description, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_sha, "benchmark_code_sha256": benchmark_sha,
        "target": asdict(target), "data_dir": data_dir, "time_budget": time_budget,
        "timeout": timeout, "node_dirs": node_dirs, "commands": commands,
    }
    if dry_run:
        return record
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    run_dir.mkdir()
    (run_dir / "source.zip").write_bytes(snapshot)
    manifest = run_dir / "launcher.json"
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    processes: list[subprocess.Popen[bytes]] = []
    logs: list[BinaryIO] = []
    try:
        for host, directory in zip(target.hosts, node_dirs):
            _stage(host, directory, snapshot)
        deadline = time.monotonic() + timeout
        for rank, (host, directory, command) in enumerate(zip(target.hosts, node_dirs, commands)):
            log = (run_dir / f"node-{rank}.log").open("wb")
            logs.append(log)
            processes.append(_launch(host, directory, command, timeout, log))
        _wait_for_workers(processes, deadline, run_dir)
        result = _fetch_result(target.hosts[0], node_dirs[0])
        validate_result(result, sum(host.gpus for host in target.hosts), time_budget, benchmark_sha)
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.update(status="completed", result=result, **_comparison_metadata(result, target, time_budget))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        record.update(status="failed", error=str(error), comparable=False)
        raise
    finally:
        for host, directory, process in zip(target.hosts, node_dirs, processes):
            if record["status"] != "completed":
                try:
                    _stop(host, directory, process)
                except (OSError, subprocess.SubprocessError) as error:
                    record.setdefault("cleanup_errors", []).append(str(error))
        for log in logs:
            log.close()
        _save_record(record, run_dir, output_root)
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one isolated local or SSH experiment")
    run.add_argument("--target", type=Path, required=True)
    run.add_argument("--name", required=True)
    run.add_argument("--description", default="", help="Hypothesis recorded in the experiment ledger")
    run.add_argument("--data-dir", required=True)
    run.add_argument("--time-budget", type=float, default=300)
    run.add_argument("--timeout", type=float)
    run.add_argument("--config", type=Path)
    run.add_argument("--output-root", type=Path, default=Path("runs"))
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    record = run_experiment(
        load_target(args.target), Path.cwd(), args.output_root, args.name, args.data_dir,
        args.time_budget, args.timeout, args.config, args.dry_run, args.description,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
