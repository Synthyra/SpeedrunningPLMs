---
id: foundry.provenance
kind: library_module
status: active
aliases: [foundry.provenance]
---
# foundry.provenance

The evidence a job leaves so that a run made on someone else's machine can be confirmed from its folder: what
`nvidia-smi` says of the GPUs before, during and after, what software and host ran it, what code it was, and a
`SHA256SUMS` that seals the folder. Importable without torch; it needs `psutil` and the standard library.

Every function that runs `nvidia-smi` takes the command as a sequence (`NVIDIA_SMI = ("nvidia-smi",)` by default), so a
test passes a small Python stand-in in its place on any platform.

| Name | Does |
|---|---|
| `text_report(command)`, `query_report(command)` | `nvidia-smi` and `nvidia-smi -q`, whole. A missing tool or a non-zero exit raises `NvidiaSmiError` with the tool's message and what to do |
| `gpu_inventory(command)` | Index, UUID, name, driver version and total memory of every GPU (`GpuRecord`); no GPU raises |
| `compute_apps(command)` | The processes on the GPUs, one `gpu_uuid, pid, process_name, used_memory` line each |
| `probe_fields(command, fields)` | None for each `--query-gpu` field the driver accepts, else the tool's own answer |
| `GpuSampler.start(path, command, interval_ms=500)` | `nvidia-smi --query-gpu=... --format=csv -lms <ms> -f <path>` as its own process writing straight to the file: timestamp, UUID, utilization, memory used, power, SM and memory clocks, temperature, performance state, clock event (throttle) reasons. An optional field the driver refuses is left out and its answer kept. On Linux the sampler dies with the thread that started it (`PR_SET_PDEATHSIG`); everywhere an `atexit` hook ends it |
| `GpuSampler.stop(window)` | Ends it and returns a `SamplerSummary` over the window (local times, the tool's clock): sample counts, UUIDs, median utilization, median and maximum power, SM clock range, maximum temperature, largest gap, the non-zero event reasons seen, the sampler's status |
| `read_samples(path)`, `summarize_samples(...)` | The same summary from a sampler file, for a validator that trusts no stored summary |
| `package_versions()` | Every installed distribution's version through `importlib.metadata` |
| `allowlisted_environment(names, directory_names, environ=None)` | Only the named variables; a directory variable (a compile cache) as set or not, exists, and its file count, never its path |
| `host_hardware()` | Architecture, CPU model (Grace reads as implementer 0x41 part 0xd4f), cores, memory, OS and Python; no host or user name |
| `source_revision(root, paths, git=("git",))` | A digest over the named paths' files (`paths_sha256`), with git's commit and dirty state of them, or git's answer when it is absent or refuses the directory |
| `write_sha256sums(directory)`, `read_sha256sums(path)`, `directory_digests(directory)` | `SHA256SUMS` in `sha256sum`'s format, both ways |

First consumer: speedrunning_plms' run evidence (`src/speedrunning_plms/evidence.py`) and its validator
(`verify_run.py`), made 2026-11-14 so outside collaborators' record attempts on their own GH200 can be confirmed.

## Known issues

- TODO(provenance_migration): capture of this kind exists in five other places, each left as it is for now:
  `foundry.pilot._NvidiaSmi` (a piped sampler); FastPLMs' `tools/remote/run.py` (the GPU identity query),
  `tools/remote/biohub_reference_environment.py` and `biohub_reference_lock.py` (driver, packages), and
  `benchmarks/run.py` (`environment_fingerprint`); and `modal/suite_capacity.py` (name, memory, driver). FastPLMs'
  tools vendor no foundry, so theirs move only with that projection's vendoring.

Tests: `tests/tier1_unit/test_provenance.py`, with a stand-in `nvidia-smi`; the GH200 behavior of the real tool
(flushing `-f` per reading, `clocks_event_reasons.active` on driver 570, the death signal in Docker) is not yet
measured.
