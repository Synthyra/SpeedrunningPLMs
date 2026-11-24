"""`nvidia-smi` as run evidence: the full text report, the GPU inventory, the processes on the GPUs, and a sampler that
writes readings to a file from its own process.

Every function that runs the tool takes its command as a sequence (`NVIDIA_SMI` by default), so a test runs a small
stand-in script in its place on any platform. A required answer that the tool cannot give raises `NvidiaSmiError`
with what to do; an optional sampler field the driver does not know is left out of the sampler, and the tool's own
answer for it is kept in the summary.
"""

from __future__ import annotations

import atexit
import csv
import ctypes
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import IO, TypedDict


__all__ = [
    "INVENTORY_FIELDS", "NVIDIA_SMI", "OPTIONAL_SAMPLE_FIELDS", "SAMPLE_FIELDS", "TIMESTAMP_FORMAT", "GpuRecord",
    "GpuSample", "GpuSampler", "NvidiaSmiError", "SamplerSummary", "compute_apps", "gpu_inventory", "probe_fields",
    "query_report", "read_samples", "summarize_samples", "text_report",
]

NVIDIA_SMI: tuple[str, ...] = ("nvidia-smi",)
TOOL_TIMEOUT_SECONDS = 60.0
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S.%f"  # the `timestamp` field, in the local time of the process that runs the tool
INVENTORY_FIELDS = ("index", "uuid", "name", "driver_version", "memory.total")
SAMPLE_FIELDS = ("timestamp", "uuid", "utilization.gpu", "memory.used")  # what a sample cannot do without
OPTIONAL_SAMPLE_FIELDS = ("power.draw", "clocks.sm", "clocks.mem", "temperature.gpu", "pstate", "clocks_event_reasons.active")
DEFAULT_INTERVAL_MS = 500
STOP_TIMEOUT_SECONDS = 10.0
PR_SET_PDEATHSIG = 1  # linux/prctl.h
WHAT_TO_DO = ("A CUDA run needs the NVIDIA driver's nvidia-smi on PATH: inside Docker run with `--gpus all` (the NVIDIA "
              "container toolkit mounts it), or run through `run_gh200.sh`; check that `nvidia-smi` alone works first.")


class NvidiaSmiError(RuntimeError):
    """`nvidia-smi` is missing, failed, or gave no answer a run can be confirmed by."""


class GpuRecord(TypedDict):
    """One GPU as the inventory query lists it; `memory_total` keeps the tool's unit (`97871 MiB`)."""

    index: int
    uuid: str
    name: str
    driver_version: str
    memory_total: str


class GpuSample(TypedDict):
    """One sampler reading; a value the device reports as `[N/A]` or `[Not Supported]`, or a field left out, is None."""

    time: datetime
    uuid: str
    utilization_percent: float | None
    memory_used_mib: float | None
    power_watts: float | None
    sm_clock_mhz: float | None
    memory_clock_mhz: float | None
    temperature_c: float | None
    pstate: str | None
    event_reasons: int | None


class SamplerSummary(TypedDict):
    """What the sampler saw. The value fields are over the window (the whole file when there is none)."""

    status: str
    fields: list[str]
    unsupported_fields: dict[str, str]
    samples: int
    window_samples: int
    uuids: list[str]
    utilization_percent_median: float | None
    power_watts_median: float | None
    power_watts_max: float | None
    sm_clock_mhz_min: float | None
    sm_clock_mhz_max: float | None
    temperature_c_max: float | None
    largest_gap_seconds: float | None
    event_reasons_seen: list[str]
    truncated_last_row: bool


def _executable(command: Sequence[str]) -> None:
    if shutil.which(command[0]) is None:
        raise NvidiaSmiError(f"{command[0]} is not on PATH. {WHAT_TO_DO}")


def _run(command: Sequence[str], arguments: Sequence[str]) -> str:
    """The tool's standard output; a missing tool or a non-zero exit raises with the tool's own message."""
    _executable(command)
    completed = subprocess.run([*command, *arguments], capture_output=True, text=True, timeout=TOOL_TIMEOUT_SECONDS)
    if completed.returncode != 0:
        answer = (completed.stderr.strip() or completed.stdout.strip())[:2000]
        raise NvidiaSmiError(f"`{' '.join([command[-1], *arguments])}` exited with code {completed.returncode}: {answer}. {WHAT_TO_DO}")
    return completed.stdout


def text_report(command: Sequence[str] = NVIDIA_SMI) -> str:
    """The plain `nvidia-smi` table: driver, CUDA version, every GPU's state and its processes."""
    return _run(command, [])


def query_report(command: Sequence[str] = NVIDIA_SMI) -> str:
    """`nvidia-smi -q`: every attribute the driver reports for every GPU, serial numbers, clocks and limits included."""
    return _run(command, ["-q"])


def gpu_inventory(command: Sequence[str] = NVIDIA_SMI) -> list[GpuRecord]:
    """Every GPU the driver lists: index, UUID, name, driver version and total memory. No GPU raises."""
    text = _run(command, [f"--query-gpu={','.join(INVENTORY_FIELDS)}", "--format=csv,noheader"])
    records: list[GpuRecord] = []
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(row) != len(INVENTORY_FIELDS):
            raise NvidiaSmiError(f"the inventory query printed {row!r}, not {len(INVENTORY_FIELDS)} fields. {WHAT_TO_DO}")
        index, uuid, name, driver, memory = (value.strip() for value in row)
        records.append(GpuRecord(index=int(index), uuid=uuid, name=name, driver_version=driver, memory_total=memory))
    if not records:
        raise NvidiaSmiError(f"nvidia-smi lists no GPU. {WHAT_TO_DO}")
    return records


def compute_apps(command: Sequence[str] = NVIDIA_SMI) -> list[str]:
    """The processes using a GPU, one `gpu_uuid, pid, process_name, used_memory` line each; empty when none."""
    text = _run(command, ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"])
    return [line.strip() for line in text.splitlines() if line.strip()]


def probe_fields(command: Sequence[str], fields: Sequence[str]) -> dict[str, str | None]:
    """Each field's standing with this driver: None when the query accepts it, else the tool's own answer."""
    _executable(command)
    standing: dict[str, str | None] = {}
    for field in fields:
        completed = subprocess.run([*command, f"--query-gpu={field}", "--format=csv,noheader"], capture_output=True,
                                   text=True, timeout=TOOL_TIMEOUT_SECONDS)
        answer = (completed.stdout.strip() + "\n" + completed.stderr.strip()).strip()
        standing[field] = None if completed.returncode == 0 else f"exit code {completed.returncode}: {answer[:500]}"
    return standing


def _die_with_parent() -> None:
    """Linux only, run in the child before it executes: the kernel sends it SIGTERM when the thread that started it
    ends, by any means, SIGKILL included (prctl(2), PR_SET_PDEATHSIG)."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM)) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")


class GpuSampler:
    """`nvidia-smi --query-gpu=... --format=csv -lms <interval> -f <file>` as its own process, writing straight to the
    file, so the readings survive whatever happens to the process that started it.

    On Linux the sampler dies with the thread that started it (`_die_with_parent`); elsewhere, and on a clean exit
    everywhere, an `atexit` hook ends it.
    """

    def __init__(self, path: Path, fields: tuple[str, ...], unsupported: dict[str, str], process: subprocess.Popen[bytes],
                 errors: IO[bytes]) -> None:
        self.path = path
        self.fields = fields
        self.unsupported = unsupported
        self._process = process
        self._errors = errors
        atexit.register(self._terminate)

    @classmethod
    def start(cls, path: Path, command: Sequence[str] = NVIDIA_SMI, interval_ms: int = DEFAULT_INTERVAL_MS) -> GpuSampler:
        """Probe the optional fields, then start sampling every GPU into `path` (CSV with a header and units)."""
        if path.exists():
            raise FileExistsError(f"{path} exists; a sampler never appends to an earlier one's readings")
        standing = probe_fields(command, OPTIONAL_SAMPLE_FIELDS)
        unsupported = {field: answer for field, answer in standing.items() if answer is not None}
        fields = SAMPLE_FIELDS + tuple(field for field in OPTIONAL_SAMPLE_FIELDS if field not in unsupported)
        errors = tempfile.TemporaryFile()  # noqa: open-without-context  (it outlives this call; `stop` closes it)
        arguments = [f"--query-gpu={','.join(fields)}", "--format=csv", "-lms", str(interval_ms), "-f", str(path)]
        process = subprocess.Popen([*command, *arguments], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
                                   preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None)
        return cls(path, fields, unsupported, process, errors)

    def _terminate(self) -> int | None:
        """End the sampler; returns its exit code if it had already exited on its own, else None."""
        exited = self._process.poll()
        if exited is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        return exited

    def stop(self, window: tuple[datetime, datetime] | None = None) -> SamplerSummary:
        """End the sampler and summarize its file over `window` (local times, as the tool writes them)."""
        atexit.unregister(self._terminate)
        exited = self._terminate()
        self._errors.seek(0)
        message = self._errors.read().decode("utf-8", errors="replace").strip()
        self._errors.close()
        if exited is None:
            status = "stopped by the run"
        else:
            status = f"exited with code {exited} before the run stopped it" + (f": {message[:500]}" if message else "")
        if not self.path.exists():
            raise NvidiaSmiError(f"the sampler wrote no {self.path.name} ({status}). {WHAT_TO_DO}")
        samples, truncated = read_samples(self.path)
        return summarize_samples(samples, window, status, list(self.fields), self.unsupported, truncated)


def _number(value: str) -> float | None:
    """A reading such as `98 %`, `612.30 W` or `55`; a bracketed answer such as `[N/A]` is None."""
    text = value.strip()
    if not text or text.startswith("["):
        return None
    return float(text.split()[0])


def _text(value: str) -> str | None:
    text = value.strip()
    return None if not text or text.startswith("[") else text


def read_samples(path: Path) -> tuple[list[GpuSample], bool]:
    """The sampler file's readings, and whether a last row cut short by the stop was dropped; any other malformed row
    raises."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        return [], False
    header = [name.strip().split(" [")[0] for name in rows[0]]  # `utilization.gpu [%]` -> `utilization.gpu`
    for required in SAMPLE_FIELDS:
        if required not in header:
            raise ValueError(f"{path.name} has no {required} column; its header is {rows[0]!r}")
    samples: list[GpuSample] = []
    truncated = False
    body = rows[1:]
    for position, row in enumerate(body):
        if [value.strip() for value in row] == [value.strip() for value in rows[0]]:
            continue  # the tool repeats its header when it reopens the file
        if len(row) != len(header):
            if position == len(body) - 1:
                truncated = True
                continue
            raise ValueError(f"{path.name} row {position + 2} has {len(row)} fields, not {len(header)}: {row!r}")
        values = dict(zip(header, row, strict=True))

        def number(field: str) -> float | None:
            return _number(values[field]) if field in values else None

        reasons = _text(values.get("clocks_event_reasons.active", ""))
        samples.append(GpuSample(
            time=datetime.strptime(values["timestamp"].strip(), TIMESTAMP_FORMAT), uuid=values["uuid"].strip(),
            utilization_percent=number("utilization.gpu"), memory_used_mib=number("memory.used"),
            power_watts=number("power.draw"), sm_clock_mhz=number("clocks.sm"), memory_clock_mhz=number("clocks.mem"),
            temperature_c=number("temperature.gpu"), pstate=_text(values.get("pstate", "")),
            event_reasons=None if reasons is None else int(reasons, 16),
        ))
    return samples, truncated


def summarize_samples(samples: list[GpuSample], window: tuple[datetime, datetime] | None, status: str, fields: list[str],
                      unsupported: dict[str, str], truncated_last_row: bool) -> SamplerSummary:
    """The medians, extremes and gaps of the readings in `window`, with the sampler's status and its field list."""
    inside = [sample for sample in samples if window is None or window[0] <= sample["time"] <= window[1]]
    utilization = [value for sample in inside if (value := sample["utilization_percent"]) is not None]
    power = [value for sample in inside if (value := sample["power_watts"]) is not None]
    clock = [value for sample in inside if (value := sample["sm_clock_mhz"]) is not None]
    temperature = [value for sample in inside if (value := sample["temperature_c"]) is not None]
    times = sorted(sample["time"] for sample in inside)
    gaps = [(later - earlier).total_seconds() for earlier, later in zip(times, times[1:], strict=False)]  # consecutive pairs
    reasons = sorted({sample["event_reasons"] for sample in inside if sample["event_reasons"]})
    return SamplerSummary(
        status=status, fields=fields, unsupported_fields=unsupported, samples=len(samples), window_samples=len(inside),
        uuids=sorted({sample["uuid"] for sample in samples}),
        utilization_percent_median=statistics.median(utilization) if utilization else None,
        power_watts_median=statistics.median(power) if power else None, power_watts_max=max(power, default=None),
        sm_clock_mhz_min=min(clock, default=None), sm_clock_mhz_max=max(clock, default=None),
        temperature_c_max=max(temperature, default=None), largest_gap_seconds=max(gaps, default=None),
        event_reasons_seen=[f"0x{reason:016x}" for reason in reasons], truncated_last_row=truncated_last_row,
    )
