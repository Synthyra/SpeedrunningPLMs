# Contributing

This file is the how-to: install, test, add a strategy module, run a genome and submit a record attempt. The rules
those steps follow (the fixed benchmark, the files a change may touch, the hooks, the search) are in
[`program.md`](program.md); the records and every lesson so far are in [`results.md`](results.md); how the Python is
written is in [`CODING_STANDARDS.md`](CODING_STANDARDS.md). The terms are defined in the [glossary](#glossary).

## Install

Python 3.12 or newer (`requires-python` in `pyproject.toml`) and PyTorch 2.11 or newer. From a clone:

```bash
pip install -e ".[test]"
```

The `foundry` modules the code imports are vendored under `src/foundry` and install with the package. Everything in
this file up to [Run a genome on a GH200](#run-a-genome-on-a-gh200) runs on a laptop CPU with no GPU and no download.

## Build the tiny data

The CPU smoke and the tests read a tiny synthetic data directory in the benchmark's shape: 380 random proteins with
random 3Di strings, prepared by the benchmark's own `prepare`, so `train.py` reads it like the real data. Its losses
mean nothing. It took about 53 s on a laptop:

```bash
python tiny_data.py --output-dir runs/tiny
```

The directory must be empty or absent; the source parquet files land beside it in `runs/tiny_source`. `runs/` is
ignored by git. The tests build their own copy (`tests/conftest.py` calls the same code).

## Smoke a genome on the CPU

```bash
python train.py --data-dir runs/tiny --output-dir runs/smoke --name smoke --device cpu --no-compile --no-wandb \
  --window 64 --windows-per-step 2 --max-steps 3 --skip-final-evals --genome genomes/auto01_bayes_02.json
```

Leave out `--genome` to train the default genome, or name your own. A run refuses an output directory that already
holds a `result.json`, so give each smoke a new one. This smoke took 722 s on a laptop CPU: 326 s of compile warmup
(the warmup's synthetic steps, even without compilation), 219 s for the three steps and 84 s of in-run evals. Those
break the 180 s and 60 s caps, so the smoke ends with status `invalid`, as it should; it shows that the genome
builds, trains and writes `result.json`, nothing about its loss.

## Run the tests

The suite holds about 220 tests (many are parametrized) and takes on the order of an hour on a laptop CPU, because
many tests build the full 198M-parameter model; it has not been timed end to end in one run. Run a narrow selection
while you work and the whole suite before you propose a change. The tests cover every fixed part on the CPU, build
every variant of every family and train it one CPU step, check that the default genome reproduces the record's frozen
`train.py` exactly, check the banked optimizer against the per-matrix one, check the run evidence and the run
validator, and run each search proposer on a toy objective.

| Command | Tests | Time on a laptop CPU |
|---|---|---|
| `python -m pytest -q tests/test_documents.py` | the documents against the code | seconds |
| `python -m pytest -q tests/test_search.py` | 50: proposers, state, shipping, collection | about 11 minutes |
| `python -m pytest -q tests/test_candidate.py tests/test_summary_plot.py tests/test_publish_dataset.py` | 13: the record's candidate end to end, two scripts | about 7 minutes |
| `python -m pytest -q tests/test_strategies.py -k "<family> or registry or default_genome"` | one family's variants, the registry, the default genome against the record | not timed |
| `python -m pytest -q` | everything | about an hour |

A change that fails a test is not shipped, and no test is skipped or marked to fail to get a green run.

## Add a strategy module

A new idea is one module under `src/speedrunning_plms/strategies/`, a line in the registry and its entries in the
search's catalog. The worked example below is modelled on the smallest real module, `softcap.py`; it is an
illustration and is not in the registry. It multiplies the attention and MLP outputs of the first `blocks` blocks by
`scale` before the residual update.

### 1. The module

`src/speedrunning_plms/strategies/early_scale.py`:

```python
"""The early_scale family: the first blocks' sublayer outputs times a constant before the residual update.

Symbols: b windows or proteins, l tokens, d = 768.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor

from speedrunning_plms.strategies.strategy import Strategy, StrategyConfig


@dataclass(frozen=True)
class EarlyScaleConfig(StrategyConfig):
    variant: str = "none"
    scale: float = 1.0
    blocks: int = 6
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "early")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"scale": (0.25, 4.0), "blocks": (1, 36)}
    INTEGERS: ClassVar[tuple[str, ...]] = ("blocks",)


class EarlyScale(Strategy):
    family = "early_scale"
    Config = EarlyScaleConfig
    config: EarlyScaleConfig

    def sublayer_output(self, index: int, sublayer: str, Y: Tensor) -> Tensor:
        # Y: (b, l, d). scale and blocks are constants, so the compiled graph may bake them in.
        if index >= self.config.blocks:
            return Y  # (b, l, d)
        return self.config.scale * Y  # (b, l, d)
```

The config is a frozen dataclass whose fields default to the record's behavior:

- `VARIANTS` lists every variant. Make the default the one that changes nothing (here `none`): the default genome
  must still equal the record bit for bit, and `tests/test_strategies.py` checks that only the softcap acts in it.
- Give the family at least one variant besides the default. The registry test builds and trains a CPU step on every
  non-default variant of every family, so a family with only its default variant gets no such test.
- `BOUNDS` holds the inclusive range of each number a genome may set; a genome outside it raises.
- `INTEGERS` names the fields a genome must give as integers (step counts, block indices); a float there raises.
- A stored genome is a diff from these defaults, so a field's name and default never change once a run has used them.
- An auxiliary loss whose weight fades over the clock can take the `FadingWeight` mixin from `strategy.py`.

The strategy overrides only the hooks it needs; the [hook reference](program.md#the-hook-reference) says when each
runs and which run inside the compiled model. A strategy whose family is at `none` is never called. A value that
changes during the run goes in a 0-d buffer that a callback refills, never a Python number a forward hook reads. A
module that adds parameters registers them in `build` under `model.extensions[family]`, initializes them in
`initialize`, and hands them to the optimizer in `parameter_groups`; draw no random numbers in `build` or `initialize`
while the family is at its default, since both run for every family and a draw shifts the initialization of everything
built after it.

### 2. The registry

In `src/speedrunning_plms/strategies/__init__.py`, import the class, add it to the `FAMILIES` tuple and add its typed
attribute to `Strategies` (`early_scale: EarlyScale`; a test checks the two agree).

Where it sits in `FAMILIES` matters: that order is the order the hooks run. Two families acting at the same hook point
compose in registry order, and every family's `build` and `initialize` run in that order. The comment above `FAMILIES`
records the orderings that matter today; add yours to it when your family must act before or after another one at a
shared hook.

### 3. The search's catalog

In `search_space.py`:

```python
Dimension("early_scale.variant", ("none", "early")),            # in SPACE
Dimension("early_scale.scale", low=0.25, high=4.0, log=True),   # in SPACE
Dimension("early_scale.blocks", low=1, high=36, integer=True),  # in SPACE
"early_scale": {"early_scale": {"variant": "early"}},           # in SWITCHES, for screening
```

A searched range may be narrower than the config's `BOUNDS`. Add the variant's compile seconds above the record's to
`COMPILE_SECONDS` once a pilot measures them (a module that adds nothing to the compiled graph needs no entry), and add
a number to `NUMBER_VARIANTS` when it matters only under some of the family's variants; a family's numbers already
count only while the family is not at `none`.

### 4. Check it

Run the module's tests, then smoke a genome that turns it on:

```bash
python -m pytest -q tests/test_strategies.py -k "early_scale or registry or default_genome"
```

```json
{"schema": 1, "families": {"early_scale": {"variant": "early", "scale": 0.5}}}
```

Then record its evidence in the catalog, [`ideas/README.md`](ideas/README.md), as the other families do, and follow
[program.md](program.md#your-place-in-the-loop) for the pilot and the screening.

## Write a genome

A genome is a JSON document that names only what differs from the defaults:

```json
{"schema": 1, "families": {"attention": {"variant": "triangle"}, "optimizer": {"muon_lr": 0.05}}}
```

A family left out, or a field left out of a family, takes its default. An unknown family, field or variant, a number
outside `BOUNDS` or a float in an integer field raises when the genome loads. Check a file before you spend a run on
it:

```bash
python -c "from pathlib import Path; from speedrunning_plms.strategies import load_genome; load_genome(Path('genomes/my_genome.json'))"
```

The genomes of note, the records among them, are in `genomes/`.

## Run a genome on a GH200

Every GPU workload runs in the Docker image through `run_gh200.sh`, which needs Docker with GPU support and `sudo` on
the machine. It keeps data, runs and caches under `$SPEEDRUN_STATE` (`~/speedrun` by default), mounted in the image as
`/state`.

```bash
bash run_gh200.sh build
bash run_gh200.sh python -c "from huggingface_hub import snapshot_download; snapshot_download('Synthyra/SpeedrunningPLMs', repo_type='dataset', local_dir='/state/data/afdb_v6')"
bash run_gh200.sh derive
bash run_gh200.sh pilot
bash run_gh200.sh train --name my_run --hypothesis "what this run tests" --genome genomes/my_genome.json
```

The second line downloads the public prepared data, the files its `manifest.json` names. `bash run_gh200.sh prepare`
builds the same data from its source dataset instead, which is private, so only a maintainer can run it. `derive`
builds the data runs read since 2026-11-23, `afdb_v6_minus_uniprot_2026_04`: the prepared data without the training
proteins that are exact copies of the new UniProt proteins `perplexity.py` scores, named by the exclusion list
`perplexity.py copies` writes (`/state/perplexity/exclusion.json`); the held-out files stay the same bytes. `SPEEDRUN_DATA=/state/data/afdb_v6` runs on the data before. The pilot
is a 60 s clock run that reports step time, compile time and memory. `train` takes `--name` first and a genome path
relative to the project folder. It logs to Weights and Biases (project `speedrunning-plms`) when `WANDB_API_KEY` is in
the environment; add `--no-wandb` to run without it. The run folder is `~/speedrun/runs/<name>/`; a name that already
holds a result is refused.

## Read `result.json`

`result.json` (schema in `src/speedrunning_plms/benchmark/result.py`, written by `train.py`) holds:

| Field | What it is |
|---|---|
| `name`, `hypothesis`, `seed` | what you passed to `train.py` |
| `status` | `completed`, or `invalid` when a cap broke |
| `metrics` | per final split (`val_select`, `validation`, `test`): `loss` (the headline is `test`), `bits_per_residue`, `perplexity`, `accuracy`, `mcc`, the precision, recall, F1, ROC AUC and PR AUC, and `loss_by_length` |
| `selected_step`, `selected_val_select_loss` | the evaluation whose weights were kept and scored |
| `curve` | every in-run evaluation: `step`, `train_seconds`, `loss` and `accuracy` on `val_select` |
| `clock` | the receipt: `budget_seconds`, `train_seconds`, `steps`, the discarded step, `warmup_seconds` and `eval_seconds` against their caps, `eval_count`, and the `valid` flags |
| `benchmark_id`, `evaluator_sha256`, `comparison_key` | the data's digest, the fixed evaluator's source digest, and the key two runs must share to compare |
| `candidate_sha256`, `candidate_parts` | the digest of `train.py`, the strategy modules and the genome, and each part's own digest |
| `genome`, `genome_changes` | the complete genome, and only what differs from the defaults |
| `architecture` | blocks, width, the parameter count and every block's shapes, from the architecture receipt |
| `environment` | torch, CUDA, the GPU's name and peak memory, Python, the platform |
| `wandb` | the W&B run's URL, or null |
| `max_steps`, `compiled`, `final_eval_seconds`, `started_utc`, `finished_utc` | the remaining receipts |

The last line `train.py` prints is the ledger row (`benchmark.ledger_row`), the compact form the campaign ledger keeps.

## Submit a record attempt

Outside collaborators may submit record attempts they ran themselves on their own GH200, with the run's evidence (the
owner, 2026-11-14). A
record attempt is a run at a track's full budget, with status `completed`, whose `val_select` loss is below the
current record's in [`results.md`](results.md); the keep rule is in [program.md](program.md#the-loop-by-hand).

The run folder is the evidence, and a maintainer confirms the run from it. `bash run_gh200.sh train` fills it:

- from `train.py`: `result.json`, `best.pt` (the selected weights, in bf16), and the run evidence of
  `src/speedrunning_plms/evidence.py`: `machine.json` (the GPU inventory, versions, packages, the command, the
  candidate's digests and the SHA-256 of every training data file, checked against the data's manifest before
  training), `nvidia-smi_start.txt` and `nvidia-smi_end.txt`, `gpu_samples.csv` (a reading every 500 ms for the whole
  run) and `phases.jsonl` (start, training and finish times);
- from the script: the `train.py` and `strategies/` snapshots of the candidate and, when the command names
  `--genome`, its `genome.json`; `console.log` (the run's whole output); `host.json` (the image, the git revision and
  `git status` of the project folder, the container's exit status, and the processes on the GPU before and after);
  `source.patch` when that status is not clean, the diff of tracked files against the revision (an untracked file is
  named in `host.json`'s status, not copied); and `SHA256SUMS` over every file.

Check the folder before you submit it:

```bash
bash run_gh200.sh python verify_run.py /state/runs/my_run --data-dir /state/data/afdb_v6
```

It prints one verdict, then one finding a line:

- `PASS` (exit 0): the folder is internally consistent, came from this benchmark's evaluator and the candidate it
  names, broke no cap, and shows no sign of another GPU or a shared one.
- `FAIL` (exit 1): a hard check failed, such as a missing or altered file against `SHA256SUMS`, a broken cap, a
  comparison key, evaluator or candidate digest that does not recompute, a GPU that is not exactly one GH200, a curve
  that disagrees with the selection, or a console log that disagrees with `result.json`. A failed folder is not a
  record attempt.
- `REVIEW` (exit 3): only soft checks fired (GPU utilization under 90% while training, a gap over 5 s in the GPU
  readings, throttling the driver reported, another process on the GPU), and a person reads the findings.

Probing a finished run is fine: `probe.json` and the bytecode `probe.py` caches in `__pycache__/` are added after
`SHA256SUMS`, and the validator lists them as `NOTE` lines that never change the verdict. Any other file added after
the run, and any listed file changed, fails the sums check.

Add `--rescore` to rebuild the model from the snapshot, load `best.pt` and score the splits again; it needs a GPU to
finish in reasonable time. A PASS does not prove the run happened as recorded: a folder fabricated whole and
consistently passes every check but `--rescore`, which shows only that the weights score what the result claims.
`verify_run.py`'s docstring lists every check.

Then propose the attempt on the GitHub repository: the genome file, any new strategy module, the run's name and track,
and a link to the whole run folder, `best.pt` included (`SHA256SUMS` covers it, so a folder missing it fails). A public
W&B run is encouraged, not required.

## Glossary

- **Genome**: a JSON document choosing one variant and its numbers for each family; what it leaves out is the default.
- **Family**: one composable choice, one module under `src/speedrunning_plms/strategies/` with a config and a strategy.
- **Variant**: one of a family's choices, its config's `variant` field; `none` switches the family off.
- **Hook**: a point in the forward, the loss or the training loop where a strategy may act
  ([reference](program.md#the-hook-reference)).
- **Track**: a training budget with its own search state, ledger and record: `10min` (600 s) and `1h` (3,600 s).
- **Rung**: a shorter clock a batch runs first, 180 s on the 10-minute track and 600 s on the 1-hour track; only the
  top half by `val_select` goes on to the full budget.
- **`val_select`**: the held-out split of one protein per validation family that selects the weights during a run.
- **Record**: the kept run with the lowest `val_select` on a track; its test loss is the headline.
- **Keep**: the verdict of a valid run at the track's full budget whose `val_select` beats the best so far.
- **Revert**: the verdict of a valid full-budget run that does not.
- **Reference**: the verdict of a rung, or of a run kept for comparison (another seed, a reference model).
- **Invalid**: a run that broke the warmup or eval cap; never a record.
- **Comparison key**: the digest of the data, the evaluator and the clock settings; two runs compare only when their
  keys agree.

## Where to change what

The [code map](PUBLIC_README.md#code-map) lists every file. By task:

| To | Change |
|---|---|
| try a combination of existing modules | a genome under `genomes/` |
| add an idea | a module under `src/speedrunning_plms/strategies/`, the registry, `search_space.py`, `ideas/README.md` |
| widen or narrow what the search may propose | `search_space.py` |
| change a proposer | `search.py` |
| add a hook point | `train.py`, under the rule in [program.md](program.md#what-a-change-may-touch) |
| change the benchmark | nothing in an ordinary change; see [program.md](program.md#what-a-change-may-touch) |
