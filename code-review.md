# Code review and standards coverage

This pass inspected all 64 extant first-party Python files, including compatibility
entry points and tests. The inventory uses tracked and untracked Python files,
excluding deleted modules and generated environments. Classifications are relative
to the working tree at the start of the follow-up review.

| Scope | Files | Mechanical | Structural or new | Already compliant |
| --- | ---: | ---: | ---: | ---: |
| Root entry points, `model/` wrappers, package and research `__init__.py` | 12 | 9 | 0 | 3 |
| `src/speedrunning_plms/models/` | 7 | 7 | 0 | 0 |
| `src/speedrunning_plms/data/` and `data/` wrappers | 15 | 14 | 1 | 0 |
| `src/speedrunning_plms/optim/`, `flex/`, and `training/` | 8 | 7 | 0 | 1 |
| Research benchmark, package evaluation, and legacy `evaluation/` | 6 | 4 | 0 | 2 |
| Research engine and runner | 2 | 0 | 2 | 0 |
| `tests/*.py` | 14 | 11 | 2 | 1 |
| Total | 64 | 52 | 5 | 7 |

Mechanical work includes imports, function annotations, numerical notation and
shape traces, spacing, and concise comments. Regression coverage and small bug
fixes accompany some mechanical classifications. Structural work reuses the
existing chunk packer, separates launcher phases, validates distributed settings,
and replaces manual test cleanup with fixtures. The new Python file tests scalar
training schedules.

The review also covered the historical HTML/JavaScript viewer, both shell entry
points, Dockerfile, deployment workflow, package configuration, and ignore rules.
CPU CI and offline JavaScript regressions were added. Historical figures, datasets,
results, the exploratory notebook, and generated environments were excluded from
source conversion. No source files were blocked.

## Correctness fixes

- Preserve complete documents in partial training batches across shard boundaries.
- Record CUDA consumer-stream ownership for asynchronously transferred tensors.
- Handle an unavailable CPU count during tokenization.
- Update scalar schedules without passing Python numbers to `Tensor.copy_()`.
- Reject invalid distributed ranks, mismatched rank configurations, and invalid
  numerical settings; wrap derived mask seeds within PyTorch's seed range.
- Cancel distributed workers gracefully, handle cancellation before remote startup,
  and avoid signaling a recycled process ID. Preserve cancellation errors in logs.
- Exclude smoke runs despite conflicting metadata; serialize concurrent ledger
  appends and replace launcher manifests atomically.
- Escape historical table content and report loading failures without indefinite
  retries. Label historical scores separately from the current benchmark.

## Preserved interfaces and behavior

Compatibility wrappers retain wildcard re-exports and initialization-sensitive
import order. Lazy package exports remain lazy. Public parameter names such as
`x` and `target_L`, serialized model fields, and state-dictionary names are retained.
Model code stays together where Transformers remote-code serialization requires it.
`Any` remains at dynamic JSON, YAML, model-output, and injected API boundaries.

The legacy ESM evaluator retains its forced minimum mask and batch-averaged score.
It is not the fixed-15% research evaluator. Changing historical score semantics was
rejected because it would silently change comparisons with stored results.

## Verification

Baseline: 285 CPU tests passed. Final local verification: 317 Python tests passed
in 73.20 seconds on CPU; four JavaScript tests passed in 82 milliseconds. Dependency
and whitespace checks passed. Commands:

```bash
python -m pytest -q --durations=10
node --test tests/test_hub.cjs
python -m pip check
git diff --check
```

Independent inspection found no missing function annotations, import-order
violations, or material numerical-shape errors across the Python inventory.
Language-audit findings retained only technical terms and literal source strings.
Model runtime syntax trees matched the baseline after excluding annotations,
docstrings, import organization, and mechanical variable renames.

Additional differential checks preserved chunked evaluation outputs in 150 seeded
cases, historical masking outputs and random-number state in 60 cases, and
Newton-Schulz optimizer outputs in nine CPU cases. Shell and JavaScript syntax
checks passed. CPU regressions simulate CUDA stream ownership and remote process
control; physical CUDA training, live SSH hosts, container builds, and browser/CDN
integration remain unverified.
