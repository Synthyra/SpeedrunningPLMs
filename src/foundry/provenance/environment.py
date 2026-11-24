"""The software and host a run executed on: installed package versions, an allowlisted slice of the environment and
the host's processor and memory.

Nothing here records the whole environment, the host name or the user name. A variable outside the caller's allowlist
is never read, and a directory variable is recorded as whether it is set and how many files its directory holds,
never as its path, which may name a home folder.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import psutil

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict


__all__ = ["DirectoryVariable", "HostHardware", "allowlisted_environment", "host_hardware", "package_versions"]


class DirectoryVariable(TypedDict):
    """A variable that names a directory, such as a compile cache: set or not, and the files under it."""

    set: bool
    exists: bool
    entries: int


class HostHardware(TypedDict):
    """The host's processor and memory; `usable_cores` is this process's CPU affinity, read on Linux only."""

    machine: str
    cpu_model: str
    logical_cores: int | None
    usable_cores: int | None
    memory_total_bytes: int
    operating_system: str
    python: str


def package_versions() -> dict[str, str]:
    """Every installed distribution's version by its normalized name; two installs of one name list both versions."""
    versions: dict[str, set[str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            versions.setdefault(name.lower().replace("_", "-"), set()).add(distribution.version)
    return {name: " | ".join(sorted(found)) for name, found in sorted(versions.items())}


def allowlisted_environment(names: Sequence[str], directory_names: Sequence[str] = (),
                            environ: Mapping[str, str] | None = None) -> dict[str, str | None | DirectoryVariable]:
    """The value of each allowlisted variable (None when unset); a directory variable as a `DirectoryVariable`."""
    unlisted = sorted(set(directory_names) - set(names))
    if unlisted:
        raise ValueError(f"directory variables {unlisted} are not in the allowlist")
    source = os.environ if environ is None else environ
    recorded: dict[str, str | None | DirectoryVariable] = {}
    for name in names:
        value = source.get(name)
        if name in directory_names:
            directory = Path(value) if value else None
            exists = directory is not None and directory.is_dir()
            entries = sum(1 for path in directory.rglob("*") if path.is_file()) if exists and directory is not None else 0
            recorded[name] = DirectoryVariable(set=value is not None, exists=exists, entries=entries)
        else:
            recorded[name] = value
    return recorded


def _cpu_model() -> str:
    """The processor's model: `model name` from /proc/cpuinfo on x86, its implementer and part codes on Arm (Grace
    is implementer 0x41, part 0xd4f, the Neoverse V2), and `platform.processor()` off Linux."""
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return platform.processor()
    fields: dict[str, set[str]] = {}
    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        key, _, value = line.partition(":")
        fields.setdefault(key.strip(), set()).add(value.strip())
    if "model name" in fields:
        return " | ".join(sorted(fields["model name"]))
    return f"implementer {' | '.join(sorted(fields.get('CPU implementer', {'?'})))} part {' | '.join(sorted(fields.get('CPU part', {'?'})))}"


def host_hardware() -> HostHardware:
    """The host's architecture, processor, core counts, memory and operating system, without its name."""
    affinity = getattr(os, "sched_getaffinity", None)
    return HostHardware(
        machine=platform.machine(), cpu_model=_cpu_model(), logical_cores=os.cpu_count(),
        usable_cores=len(affinity(0)) if affinity is not None else None,
        memory_total_bytes=psutil.virtual_memory().total, operating_system=platform.platform(),
        python=platform.python_version(),
    )
