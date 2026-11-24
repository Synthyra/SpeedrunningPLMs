"""Run provenance: the evidence a job leaves so that a run on someone else's machine can be confirmed from its folder.

`nvidia_smi` holds the full text report, the GPU inventory, the processes on the GPUs and a sampler process writing
readings to a file; `environment` the installed packages, an allowlisted slice of the environment and the host's
hardware; `source` a source revision over named paths and `SHA256SUMS` over a folder. Importable without torch.
"""

from __future__ import annotations

from foundry.provenance.environment import (
    DirectoryVariable, HostHardware, allowlisted_environment, host_hardware, package_versions,
)
from foundry.provenance.nvidia_smi import (
    NVIDIA_SMI, GpuRecord, GpuSample, GpuSampler, NvidiaSmiError, SamplerSummary, compute_apps, gpu_inventory, probe_fields,
    query_report, read_samples, summarize_samples, text_report,
)
from foundry.provenance.source import (
    SUMS_NAME, GitState, SourceRevision, directory_digests, git_state, paths_sha256, read_sha256sums, source_revision,
    write_sha256sums,
)


__all__ = [
    "NVIDIA_SMI", "SUMS_NAME", "DirectoryVariable", "GitState", "GpuRecord", "GpuSample", "GpuSampler", "HostHardware",
    "NvidiaSmiError", "SamplerSummary", "SourceRevision", "allowlisted_environment", "compute_apps", "directory_digests",
    "git_state", "gpu_inventory", "host_hardware", "package_versions", "paths_sha256", "probe_fields", "query_report",
    "read_samples", "read_sha256sums", "source_revision", "summarize_samples", "text_report", "write_sha256sums",
]
