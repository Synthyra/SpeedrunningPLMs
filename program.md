# The program: the benchmark's rules and the search

You improve one number: the test cross entropy, in nats per masked residue, of a protein masked language model of any
shape after one hour of training on one GH200, its weights chosen by `val_select` (the headline since 2026-11-23; the
10-minute track is the history before it and the hour's first rung). A candidate is a genome: one
choice from each family of composable strategy modules plus their numbers. A search proposes genomes, the GPU runs
them, and you reason on top of it: you read what the search has learned, add the modules the data asks for, retire
what it shows inert, and let the proposers run between your interventions. Everything outside the modules is fixed and
audited. This file holds the rules, the hook reference and the search; the commands to install, test, add a module and
submit a run are in [`CONTRIBUTING.md`](CONTRIBUTING.md), and every record and lesson is in [`results.md`](results.md).

## The benchmark (fixed, `src/speedrunning_plms/benchmark/`)

- **Data.** `Synthyra/SpeedrunningPLMs` (public), built by `prepare.py` from the family-disjoint AlphaFold DB split: no
  held-out protein has a training relative at 30% identity over 80% coverage by sequence (MMseqs2) or structure
  (Foldseek). Train: 1,754,034 proteins as one shuffled uint8 token stream in the ESM2 alphabet (33 tokens: CLS,
  residues, EOS; MASK is 32), 392.6M tokens. Lengths run 50 to 2,046 residues. Held out: `val_select` (one protein per
  validation family, 4,751 proteins, at most 1,024 tokens), `validation` (25,459 proteins) and `test` (25,778
  proteins), both at full length (at most 2,048 tokens), each with fixed masks. A run checks every training file
  against the data's `manifest.json` before it trains, and every held-out file as it loads. Since 2026-11-23 runs train
  on `afdb_v6_minus_uniprot_2026_04` (`prepare.py --exclude-from`): the same data without the training proteins that
  are exact copies of the 2,100 new UniProt proteins `perplexity.py` scores (2 of them), so no run trains on a sequence
  the plot scores it on; relatives stay, as they do in every model the plot compares. The held-out files are unchanged
  and the benchmark id is new. The records in `results.md` before that date ran on `afdb_v6`.
- **Objective.** 15% of standard residues replaced by MASK, nothing else; the loss is the collapse-free cross entropy
  ([arXiv 2610.31642](https://arxiv.org/abs/2610.31642), `foundry.layers.cross_entropy`) on the masked residues over
  the 20 standard residues. Training uses the same corruption (`benchmark.corrupt`).
- **Hardware.** One NVIDIA GH200 (the H100 die, 96 GB HBM3). A run on other hardware is a different clock and is
  recorded as such.
- **Clock.** 10 minutes of applied training steps, measured with device synchronization at step boundaries; the step
  that overruns is discarded. A compile warmup before the clock is capped at 3 minutes; in-run validation evals are
  outside the clock and capped at 60 seconds in total. Breaking a cap makes the run `invalid`.
- **Selection.** The weights with the lowest `val_select` loss during the run are scored once on full validation and
  full test. The headline is the test loss; the weights are chosen by validation, never by test.
- **Shape.** Free since 2026-11-23: compute is matched by the clock alone
  ([decision](../../../docs/decisions/representation/speedrunning_plms/2026-11-23_compute_matched_free_architecture.md)).
  The `shape` family sets the depth, the stream width and the MLP width (the record's 36, 768 and 2,048 by default),
  and a looped core, routing or a module pool may spend them differently; `benchmark.check_architecture` checks only
  the skeleton's structure (`blocks` of `attention` and `mlp`, an `embedding`) and writes the receipt. Test-time compute
  is bounded by the in-run eval cap: a model scores `val_select` at every interval inside it.
- **Metrics.** Loss, bits, perplexity, accuracy, macro and weighted precision, recall and F1, MCC, macro ROC AUC and
  PR AUC, loss by length bin. Every model is scored on its own 20 residue columns, so ESM2 (33 tokens) and ESMC (64
  tokens) are scored on the same problem. The 3Di probe (`probe.py`) reports how much structure the hidden states
  carry: a softmax-weighted sum of all hidden states, each RMS-normalized, then one linear layer to the 20 Foldseek 3Di
  states, trained on 20,000 training proteins and scored on validation and test. It never selects.
- **Comparison.** `result.json` carries the benchmark id (the data's digests), the evaluator's source digest, the
  candidate's digest (`train.py`, the strategy modules and the genome) and the clock receipts. Two runs compare only
  when their comparison keys agree; the budget is part of the key, so a 180 s rung never compares with a 600 s run.

## What a run may not do

- Read the held-out tensors except through `benchmark.evaluate_split`.
- Change the corruption, the clock, the evaluator, the architecture receipt or the result writer.
- Train on validation or test proteins, or select weights by test.
- Exceed the warmup or eval caps to buy training time.

## What a change may touch

A run changes a genome. A new idea is a new module under `src/speedrunning_plms/strategies/`, listed in the registry
and in the search's catalog (`search_space.py`); [`CONTRIBUTING.md`](CONTRIBUTING.md#add-a-strategy-module) walks
through one.

These files define the benchmark: the modules under `src/speedrunning_plms/benchmark/` (data, tokens, the evaluator,
the clock, the architecture receipt, the result schema, the probe), `prepare.py`, `baselines.py`, `probe.py`, `record.py` and
`plots.py`. A change to them changes what every result means and is a separate, deliberate commit with its reason.

`train.py` is the skeleton. It changes when the skeleton must offer a new hook point, or when the benchmark's own
machinery needs a call inside the loop, as the three run evidence calls did (2026-11-14, at the owner's request).
After any change to it, the default genome must still equal the record's frozen `train.py` bit for bit on a CPU smoke
(`tests/test_strategies.py`). `candidates/` holds frozen snapshots of past record holders: their bytes identify runs
in the ledger, so they are never edited.

## The candidate: a skeleton and a genome

`train.py` is the skeleton: an embedding, the `shape` family's blocks of attention and MLP, a head, the training loop
and the result.
Every choice the search makes lives in a family under `src/speedrunning_plms/strategies/`, one module each: a frozen
config dataclass (`variant` plus numbers, each defaulting to the record) and a strategy class that overrides only the
hooks it needs (the [hook reference](#the-hook-reference) below). A genome is a JSON document,
`{"families": {"attention": {"variant": "triangle"}, "optimizer": {"muon_lr": 0.05}}}`; what it leaves out takes the
default, so `python train.py` with no `--genome` is 022, bit for bit on a CPU smoke. The families, their variants and
the evidence for each are the catalog in [`ideas/README.md`](ideas/README.md); the record genomes are named in
[`results.md`](results.md).

### The hook reference

The model calls a hook only on the strategies that override it, in registry order (the order of `FAMILIES` in
`src/speedrunning_plms/strategies/__init__.py`), so where a family sits in the registry decides which of two families
acting at one hook point acts first. On the GPU, `torch.compile` compiles the training model with every forward hook
inside it: a number such a hook reads from Python is baked into the graph, so a value that changes during the run lives
in a 0-d buffer a callback refills. The signatures and shapes are in `strategy.py`; `tests/test_documents.py` checks
that this table names exactly the hooks `strategy.HOOKS` lists, in its order.

| Hook | When it runs | In the compiled model |
|---|---|---|
| `embed` | once a forward, on the token embedding before block 0 | yes |
| `block_input` | as each block starts, with the embedding and every earlier block's output | yes |
| `normalized_input` | on each block's normalized stream entering its attention or its MLP | yes |
| `keys` | on each block's raw keys from the key projection, before QK-norm and RoPE | yes |
| `queries` | on each block's queries after QK-norm and RoPE, before attention | yes |
| `values` | on each block's values, before attention | yes |
| `attention_output` | on the heads' outputs, before the output projection | yes |
| `mlp_hidden` | on each block's MLP activation, before the down projection, with the MLP's normalized input beside it | yes |
| `sublayer_output` | on the attention's or the MLP's output, before the residual update | yes |
| `updated_stream` | on the stream right after each residual update | yes |
| `head_input` | on the normalized final state the head reads | yes |
| `logits` | on the head's logits; not on the loop's CPU training forward | yes |
| `auxiliary_loss` | at the end of a training forward given labels, which the loop passes on the GPU (flex) path alone; the summed loss joins the training loss | yes |
| `training_loss` | after the forward, turning the summed masked cross entropy into the objective the backward differentiates, on both training paths; the logged loss stays the record's | no |
| `after_backward` | after the backward, before the optimizers step, inside the clock | no |
| `after_step` | after the optimizers step and the gradients are cleared, inside the clock | no |
| `evaluation_weights` | around each in-run validation pass, outside the clock; the live weights return afterwards | no |

The two callbacks also run on the compile warmup's synthetic steps (steps 0 to 2 of each window shape, progress 0),
whose weights the loop then restores, so a callback that keeps state resets it at step 0. Four more methods are not
hooks:

| Method | When it runs |
|---|---|
| `build` | inside the candidate's construction, after the embedding exists and before the blocks and the head do; adds modules under `model.extensions[family]` |
| `initialize` | after the record's own initialization, on the whole model |
| `parameter_groups` | each time the loop builds its optimizers (before the compile warmup and again after it); every trainable parameter lands in exactly one group |
| `evaluation_options` | read before each validation pass when the strategy overrides `evaluation_weights`: how many weight options the pass scores; the loop keeps the lowest `val_select` loss |

Every strategy's `build` and `initialize` run, in registry order, whether or not its family is active, so a random
draw in either shifts the initialization of everything built after it and breaks the default genome's equality with
the record.

## The search (`search.py`)

Eight proposers behind one interface (`--proposer`), and one state file beside the campaign ledger. What a genome may
vary, the screening switches and the compile costs are the catalog in `search_space.py`, read as `search.<name>`.

- **Hybrid** (`--proposer hybrid`, the loop's default since 2026-11-23): genetic children ranked by the rung-reading
  Gaussian process, described under [the proposer](#the-proposer-genetic-children-ranked-by-the-gp).

- **Screening** (`--proposer screening --factors ...`): a 12-run Plackett-Burman design over up to 11 binary module
  switches (`search.SWITCHES`) on top of the record (the best observation), every run at 600 s; its all-low row is
  the record again. `search.py effects` prints each switch's main effect on `val_select` and test with its standard
  error (one seed's sd, 0.0015 nats, gives 0.0009 per effect) and the leading two-factor interactions among active
  switches. Plackett-Burman aliases each interaction partly with the other main effects, so an interaction it names is
  a lead to test, not an estimate.
- **Bayes** (`--proposer bayes`): a Gaussian process with a noise term over the mixed space (variants one-hot, numbers
  scaled, a switched-off family's numbers held at the middle), its noise never below one seed's; expected improvement
  over the best posterior mean, not the best observation, so it does not chase a lucky seed; a batch by the kriging
  believer.
- **Genetic** (`--proposer genetic`): a population of 8 by tournament over every observed genome, uniform crossover
  over families (a family's variant keeps its own numbers) and mutation of variants and numbers.
- **Random** (`--proposer random`): uniform draws over the open space, a track's first design before the Bayes and
  genetic proposers have observations.
- **Manual** (`--proposer manual --genomes ... --hypotheses ...`): genome files you name, one hypothesis each, such as
  a pilot of new modules.
- **In a head-to-head, not yet in the loop**: `--proposer bayes_rungs`, the same Gaussian process reading rung results
  with budget a dimension, and `--proposer kumo`, Kumo Tabular (`tabular_surrogate.py`) on those and any other track's
  results a state carries, in its own image (`Dockerfile.kumo`, `run_gh200.sh build-kumo` and `python-kumo`). The
  comparison is a workspace experiment whose record is not in this repository.
- **The warmup cap.** A new graph compiles from scratch, the record's in 106 s against the 180 s cap. Each variant's
  extra seconds, measured alone, are `search.COMPILE_SECONDS`; every proposer skips a genome whose sum comes within
  15 s of the cap, the Bayes pool and the genetic children first put a genome's costly choices back at the record's
  until it fits (few uniform draws do; a repaired child says so in its reason), and the screening refuses a design with
  such a row.
- **Runs that will crash are trimmed automatically** (Logan, 2026-11-04). A crashed run is no observation, so a
  proposer cannot learn from it. A pick holding what no valid result has held (`search.unproven_keys`) first compiles
  and takes two steps at the real batch size, between timed runs (`autosearch.sh`); if it crashes or compiles over the
  cap it leaves the state, what it alone held is retired and kept in the state's `trimmed`, and its proposer proposes
  again. A run that still crashes retires what it alone held. A proven choice beside the culprit cannot be told from
  it, so an innocent choice can go too. Two steps do not reach what acts late in the clock, and a memory blowup from
  proven choices together is not tried.
- **Successive halving.** An adaptive batch runs at 180 s first (`--budget 180 --final-splits val_select`: the whole
  schedule compressed to the shorter clock, a complete valid run under its own comparison key), and only the top half
  by `val_select` runs the full 600 s. Both rungs are recorded.

### The loop by hand

`ship` writes each run's queue entry under `candidates/queue/` and runs `bash queue.sh --wait <runs>` through the
command in `SPEEDRUN_LAUNCHER`; `collect` copies each run's `result.json` into `--results-dir` with the command in
`SPEEDRUN_FETCH`. Both are environment variables you set, not files in this repository:

- `SPEEDRUN_LAUNCHER` is a command prefix: `search.py` splits it like a shell and appends the `queue.sh` command. Set
  it to the empty string when `search.py` runs in the project folder on the GPU machine, or to something like
  `ssh <box> cd <project on the box> &&` when it runs on another computer whose queue entries reach the box's
  `candidates/queue/` (a shared or copied folder). Unset, `ship` only prints the `queue.sh` command for you to run on
  the box.
- `SPEEDRUN_FETCH` is a command template with `{name}` (the run) and `{destination}` (the file to write), for example
  `cp /state/runs/{name}/result.json {destination}` inside the image, which is what `autosearch.py` sets, or an `scp`
  from the box. Unset, `collect` reads only the files already in `--results-dir`.

From the project folder, with the state and the ledger (`runs.jsonl`) in one folder:

```bash
python search.py --state <state dir>/search.json propose --proposer bayes --batch bayes1 --count 8
python search.py --state <state dir>/search.json ship --batch bayes1           # the 180 s rung, through queue.sh
python search.py --state <state dir>/search.json collect --batch bayes1 --results-dir <dir> --ledger <state dir>/runs.jsonl
python search.py --state <state dir>/search.json ship --batch bayes1           # the promoted half at 600 s
python search.py --state <state dir>/search.json collect --batch bayes1 --results-dir <dir> --ledger <state dir>/runs.jsonl
python search.py --state <state dir>/search.json status
```

A path without a state starts an empty one on `--track` (the 10-minute track by default). Every proposal is in the
state file the moment it is made (its genome, its parents, the proposer's reason); its ledger row, written by `collect`
through `record.append_row`, carries the same fields with the verdict. A run at the track's full budget that beats the
best observation so far is `keep`, else `revert`; a shorter rung is `reference`; a run that breaks a cap is `invalid`.
Seed 0 decides.

### Two tracks

A state names its track (`search.TRACKS`): `10min` (600 s, the 180 s rung, evals every 90 s under a 60 s cap) or `1h`
(3,600 s, the 600 s rung first and the top half promoted, evals every 180 s under a 360 s cap, batch names prefixed
`h_`, its own W&B group). `load_state` applies the track's budgets, so every command and the loop read them from the
state; `--track` names the track of a state a command starts. The 1-hour track started on 2026-10-31 from the 000
baseline (`genomes/000_baseline.json`) with nothing of the 10-minute track in its state: the first batches are random
draws over the open space (`RandomProposer`) until the Bayes and genetic proposers have their three and two
observations. Each track has its own state folder on the box (`search`, `search_1h`), its own ledger, results record
and figure set; `plots.py` refuses a folder that mixes budgets. Since 2026-11-23 the 1-hour track is the headline and
the commands' default (`--track 1h`).

### The proposer: genetic children, ranked by the GP

`HybridProposer` (`hybrid`, the loop's default) is the genetic algorithm and the Bayesian search in one: tournament
parents from the full-budget results, ranked by the GP's posterior mean; family-wise crossover and mutation breed a
pool, so structure (which modules, which shape) moves by crossover and numbers by mutation; the GP, reading every rung
with the budget as a dimension, picks the batch by expected improvement, the kriging believer spreading it. A child
whose families conflict (`strategies.genome_conflicts`: a loop on a mixing stream, a layout past the depth, heads that
do not divide the width) is skipped, and one predicted over the warmup cap or the GPU's memory
(`search.predicted_warmup`, `search.predicted_peak_gib`, both scaled from the record's measurements) is repaired or
skipped. A uniform draw redraws only its conflicting families (`random_genome`).

### The unattended loop

The search runs itself on the GPU machine. `autosearch.sh` loops until a deadline or a `stop` file: `autosearch.py next`
ships up to four runs still to do, newest batch first, else proposes four from the state's proposers in turn (the
hybrid alone by default), straight at the full budget; `queue.sh` runs them one after another; `autosearch.py collect` reads them into the
state and the ledger, which live beside the runs on the machine (`$SPEEDRUN_STATE/search`, `~/speedrun/search` by
default) while the loop runs. New strategy modules ship as a new tree with the same command: the new loop takes over
at the running cycle's end, and a pilot batch of the new modules runs ahead of the backlog. A state may name its own
`proposers`, `rungs` and W&B `group`, for an experiment on a track in a state folder of its own; with an empty list of
proposers the loop runs what is pending and ends.

## Your place in the loop

Your job is to supply strategies and ideas; the genetic and Bayesian searches find the combinations and the numbers.

1. **Read the state.** `search.py status` (the incumbent, the observations, what is retired), `search.py effects`
   for each screening batch, the ledger's latest rows and the catalog. Look at the curves, not only the final numbers:
   a module behind at 90 s and ahead at 600 s wants a longer horizon, one ahead early and behind late wants a schedule.
2. **Intervene where the data asks**, one kind of change at a time, and record why in the ledger note or the catalog:
   - *Add a module* when an idea is not yet expressible ([how](CONTRIBUTING.md#add-a-strategy-module)). Smoke it, pilot
     it on the box at the 180 s rung, add its measured compile seconds to `search.COMPILE_SECONDS`, then its switch to
     `search.SWITCHES` and its dimensions to `search.SPACE`.
   - *Screen* a set of new switches together with the old ones whose effects are least settled.
   - *Retire* a variant or number the screening shows inert or harmful beyond two standard errors
     (`search.py retire attention.variant=paired`), or one that crashes or breaks a cap, so the proposers stop
     spending runs on it. Retiring is a recorded decision with its numbers, not deletion: the module stays.
   - *Narrow or widen* a number's range in `search.SPACE` when the results pile up at an edge.
3. **Let the proposers run** between interventions: hybrid batches (Bayes and genetic in turn before 2026-11-23);
   each batch is eight proposals, four of which reach 600 s. Do not hand-pick genomes except to seed a batch with a
   hypothesis worth a run (`search.py seed` records a finished hand-written run as an observation).
4. **Promote a settled record.** When a genome holds the record across later batches, write it to `genomes/` and,
   in its own commit, move the defaults of the families it changes to its values, with a CPU smoke showing the new
   defaults equal the genome.
5. Seed 0 decides. No seed confirmations while the search runs: they return once the record is near a good
   benchmark, about 2.3 nats. The 015 seeds are the one noise measurement so far.

Hand-written candidates still work: a complete `train.py` as `candidates/queue/<name>.py` beside `<name>.txt` runs
through `queue.sh` as before, and a genome as `<name>.json`. One run on the GPU at a time.
