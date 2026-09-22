# Protein MLM speedruns

A small research loop for improving protein masked-language modeling under a fixed
GPU budget. UniRef50 is the default dataset. Standard transformers, U-Nets, and patch
U-Nets share one objective: independently mask 15% of eligible residues and predict
the original residues. Every selected residue is replaced by `<mask>`.

The workflow follows [Karpathy's autoresearch](https://github.com/karpathy/autoresearch):
keep the benchmark fixed, change the experiment, measure, and retain improvements.
This repository adds protein data and local, SSH, and multi-node execution. It grew
out of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

## Four files to start with

| File | Purpose |
| --- | --- |
| `prepare.py` | Prepare pinned data once; fixed tokenizer and evaluation protocol. |
| `train.py` | Run one bounded training experiment or evaluate a saved checkpoint. |
| `experiment.json` | Small, editable set of architecture and optimization settings. |
| `program.md` | Instructions for an autonomous coding agent on the workstation. |

`research.py` stages candidates on configured machines and retrieves results.
Implementation lives under `src/speedrunning_plms/research`. Reusable models remain
under `src/speedrunning_plms/models` and support Transformers `save_pretrained()`.

## Install and prepare

Use Python 3.10+ and a virtual environment. Install the PyTorch build appropriate
for the GPU hosts, then install this package:

```bash
python -m pip install -e ".[test,evaluation]"
python prepare.py --dataset uniref50 --output-dir data/uniref50 \
  --max-length 256 --train-sequences 100000 --eval-sequences 2048
```

Preparation streams the explicitly pinned source revision, writes local tensors,
and records their hashes and tokenizer/objective definitions in `manifest.json`.
Long sequences are divided into chunks with CLS/EOS; their tails are retained.
The limits count source sequences, so long sequences can produce multiple examples.
Downloads happen only during preparation. Training reads these local files.

Use `--dataset omg_prot50` or `--dataset og_prot90` for separate data tracks.
The tokenizer is the fixed ESM residue alphabet; no tokenizer download is needed.
Preparation fetches train and validation only unless `--include-test` is explicit.
Prepare a new directory to change data volume, sequence length, or dataset revision.
These choices change the benchmark identity and require a new baseline.

## Run one experiment

```bash
python train.py --data-dir data/uniref50 --config experiment.json \
  --output-dir runs/baseline --time-budget 300 --device cuda
```

For several GPUs on one machine:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-dir data/uniref50 --config experiment.json \
  --output-dir runs/baseline-4gpu --time-budget 300 --device cuda
```

The training budget includes training steps and their compilation overhead. Data
loading, model construction, final evaluation, and checkpoint writing are reported
in total wall time separately. Rank zero checks the deadline between microbatches
and before each optimizer update; unfinished accumulation is discarded. An in-flight
operation can finish after the deadline, so actual training time is recorded and
runs exceeding the budget by more than 5% are excluded from comparisons. Distributed
ranks stop together, and losses are weighted by masked-residue count, not batch count.

Each successful run writes `result.json` and a loadable `checkpoint/` in a new output
directory. Results include the benchmark identity, model/configuration, seed,
hardware, world size, timing, and metric. Existing results are not overwritten.
There is no automatic model publication or experiment-tracking login.

For a short CPU smoke run:

```bash
python train.py --data-dir data/uniref50 --output-dir runs/smoke \
  --device cpu --hidden-size 8 --heads 2 --layers 2 --batch-size 2 --max-steps 2
```

Step-limited runs are for debugging and do not qualify for fixed-budget comparisons.
Explicit CLI values override JSON configuration; unknown fields are rejected.
Use `--architecture unet` or `--architecture patch_unet` to change architecture.
Use `--compile` and `--bf16` only on hardware that supports them.

## Metric and held-out evaluation

The primary score is **validation bits per masked residue**, lower is better:

```text
sum(cross_entropy over selected residues) / (number of selected residues * ln(2))
```

This is a conditional MLM score, not autoregressive bits per byte. Cross-entropy in
nats and masked accuracy are also reported. Evaluation masks are deterministic per
example and do not change with batch size or GPU partitioning. CLS, EOS, padding,
unknown/null/mask tokens, and alignment gaps are never selected. Residues use
independent Bernoulli(0.15) selection; no extra residue is forced into short sequences.
All ranks contribute sums and counts once, without duplicated evaluation examples.
Evaluation always uses float32, independent of the candidate's training precision.

Use validation for search. Compare the same benchmark, seed, time budget, and hardware
allocation. Confirm small gains across multiple seeds rather than choosing a lucky
seed. Never feed test results back into the search loop.

After selecting a model, explicitly prepare a separate dataset directory with
`--include-test` and evaluate its checkpoint:

```bash
python train.py --evaluate-only runs/baseline/checkpoint --split test \
  --data-dir data/uniref50-final --output-dir runs/final-test --device cuda
```

Use the same dataset revision, tokenizer, and sequence length as the selection
benchmark. Final evaluation cannot be requested as part of a training run.

## Workstation to GPU hosts

Copy an example from `targets/` to `targets.local.json` and set the actual SSH
aliases, absolute staging paths, Python executables, and GPUs per node. The local
target uses `host: null`; Windows local paths may use `C:/...`. Remote hosts use
Linux/POSIX paths and require Python, PyTorch/package dependencies, SSH, and
GNU `timeout` and `setsid`. Provision dependencies and prepared data before starting a session.
The runner does not provision machines or download data.

```bash
python research.py run --target targets.local.json --name baseline \
  --data-dir /absolute/path/to/data/uniref50 --config experiment.json \
  --time-budget 300 --dry-run

python research.py run --target targets.local.json --name baseline \
  --data-dir /absolute/path/to/data/uniref50 --config experiment.json \
  --time-budget 300
```

The first command shows the plan without connecting or launching compute. The second
snapshots the candidate source, stages it in a unique directory, starts the run,
and retrieves results and logs to the workstation. Checkpoints stay at the recorded
execution location. Credentials, `.git`, caches, and datasets are excluded from the
source archive. Existing remote checkouts are not reset or modified.
Cancellation first lets `torchrun` stop its workers, then forces termination if
needed. Remote timeouts include a 45-second grace period before forced termination.

For multiple nodes, use the cluster target example. Nodes need equal GPU counts,
the same prepared data at the same absolute path, and a reachable rank-zero address
and rendezvous port. Use an existing allocation if the cluster has a scheduler.
Each node runs `torchrun`; this is not a Slurm provisioning layer. One workstation
process owns an experiment and its local results ledger. Run separate sessions in
separate output directories and allocate disjoint devices when searching in parallel.

## Autonomous agents

Give the agent `program.md`, a target, a prepared-data path, a session name, a per-run
budget, and a maximum experiment count. It edits candidates, invokes the runner,
reads returned results, and keeps or discards its own changes. The runner records
source hashes and comparison keys; the agent records hypotheses and decisions.
Comparison keys distinguish data, evaluator, hardware, framework versions, seed,
and training budget. The launcher verifies the reported evaluator against its
staged source before accepting a score.
The evaluation protocol stays fixed. No separate inference API integration is needed.

Examples using already authenticated coding clients:

```bash
codex exec --sandbox workspace-write -m gpt-6-astra \
  "Follow program.md. Target targets.local.json; data /data/uniref50; session astra-01; 300 seconds per run; at most 20 experiments."

codex exec --sandbox workspace-write -m gpt-5.6-sol \
  "Follow program.md. Target targets.local.json; data /data/uniref50; session sol-01; 300 seconds per run; at most 20 experiments."

claude -p --model claude-opus-5-5 \
  "Follow program.md. Target targets.local.json; data /data/uniref50; session opus-01; 300 seconds per run; at most 20 experiments."
```

Configure the client to permit the project runner and the specified SSH targets
before unattended use. These commands preserve the client's permission controls.
The maximum experiment count is an agent instruction; the launcher enforces each
job's timeout. Model availability depends on the client/account. See the official
[Codex CLI guidance](https://learn.chatgpt.com/docs/non-interactive-mode),
[Codex models](https://learn.chatgpt.com/docs/models), and
[Claude model configuration](https://code.claude.com/docs/en/model-config).

## CPU tests

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test,evaluation]"
python -m pytest -q
```

The tests use tiny synthetic data, run offline, disable CUDA, and limit CPU threads.
They check corruption and metric semantics, architecture gradients and masking,
sharded serialization/publication, local training, distributed behavior, transport
command construction, data integrity, and built-package installation. SSH command
tests do not establish performance or connectivity on your GPU hosts.
The CPU CI workflow runs this suite on Python 3.10 and 3.12.
Run the historical viewer's offline checks with `node --test tests/test_hub.cjs`.
See [code review coverage](docs/code-review.md) for the repository standards pass
and its verification limits.

## Migration from the earlier training scripts

`--yaml_path`, diffusion, masking schedules, interactive token prompts, and the large
legacy trainer have been retired from the training entry point. Use experiment.json
and the fixed benchmark instead. Packed-data tools and the ESM baseline evaluator
are retained for historical comparisons; they are not the new
search protocol. Historical scores should not be compared directly with this one.
The historical ESM evaluator retains its forced minimum mask and batch-averaged
score for compatibility; the research evaluator uses residue-weighted scoring.
Library checkpoint loading and explicit `publish_model_to_hub()` remain available.
This project retains its existing license; see LICENSE.
