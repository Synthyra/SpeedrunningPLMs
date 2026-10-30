# The program: an agent's brief for the speedrun

You improve one number: the test cross entropy, in nats per masked residue, of a 36-block protein masked language
model after 10 minutes of training on one GH200. A candidate is a genome: one choice from each family of composable
strategy modules plus their numbers. A search proposes genomes, the GPU runs them, and you reason on top of it:
you read what the search has learned, add the modules the data asks for, retire what it shows inert, and let the
proposers run between your interventions. Everything outside the modules is fixed and audited.

## The benchmark (fixed, `src/speedrunning_plms/benchmark/`)

- **Data.** `Synthyra/SpeedrunningPLMs` (public), built by `prepare.py` from the family-disjoint AlphaFold DB split: no
  held-out protein has a training relative at 30 percent identity over 80 percent coverage by sequence (MMseqs2) or
  structure (Foldseek). Train: 1,754,034 proteins as one shuffled uint8 token stream in the ESM2 alphabet (33 tokens:
  CLS, residues, EOS; MASK is 32). Held out: `val_select` (one protein per validation family, 4,751 proteins, at most
  1,024 tokens), `validation` (25,459 proteins) and `test` (25,778 proteins), both at full length (at most 2,048
  tokens), each with fixed masks.
- **Objective.** 15 percent of standard residues replaced by MASK, nothing else; the loss is the collapse-free cross
  entropy (`foundry.layers.cross_entropy`) on the masked residues over the 20 standard residues. Training uses the
  same corruption (`benchmark.corrupt`).
- **Clock.** 10 minutes of applied training steps, measured with device synchronization at step boundaries; the step
  that overruns is discarded. A compile warmup before the clock is capped at 3 minutes; in-run validation evals are
  outside the clock and capped at 60 seconds in total. Breaking a cap makes the run `invalid`.
- **Selection.** The weights with the lowest `val_select` loss during the run are scored once on full validation and
  full test. The headline is the test loss; the weights are chosen by validation, never by test.
- **Shape.** Exactly 36 blocks on a 768-wide residual stream, each one attention sublayer and one MLP with intermediate
  2,048 (`benchmark.check_architecture`). Heads, grouped or differential attention, value embeddings, skips, gates,
  scalars, normalization, tables and the optimizer are free.
- **Metrics.** Loss, bits, perplexity, accuracy, macro and weighted precision, recall and F1, MCC, macro ROC AUC and
  PR AUC, loss by length bin. The 3Di probe (`probe.py`) reports how much structure the hidden states carry; it never
  selects.
- **Comparison.** `result.json` carries the benchmark id (the data's digests), the evaluator's source digest, the
  candidate's digest (`train.py`, the strategy modules and the genome) and the clock receipts. Two runs compare only
  when their comparison keys agree; the budget is part of the key, so a 180 s rung never compares with a 600 s run.

## The candidate: a skeleton and a genome

`train.py` is the skeleton: an embedding, 36 blocks of attention and MLP, a head, the training loop and the result.
Every choice the search makes lives in a family under `src/speedrunning_plms/strategies/`, one module each: a frozen
config dataclass (`variant` plus numbers, each defaulting to the record) and a strategy class that may add modules at
construction, initialize them, claim optimizer parameter groups, transform tensors at the forward's hook points
(`embed`, `block_input`, `normalized_input`, `keys`, `queries`, `values`, `attention_output`, `mlp_hidden`,
`sublayer_output`, `updated_stream`, `head_input`, `logits`, `auxiliary_loss`), reweight the summed training loss
outside the compiled model (`training_loss`), and act through callbacks (`after_backward`, `after_step`,
`evaluation_weights`). A genome is a JSON document,
`{"families": {"attention": {"variant": "triangle"}, "optimizer": {"muon_lr": 0.05}}}`; what it leaves out takes the
default, so `python train.py` with no `--genome` is 022, bit for bit on a CPU smoke; the record's genome is in
`genomes/`. The families, their variants and the evidence for each are the catalog in `ideas/README.md`.

## The search (`search.py`)

Three proposers behind one interface, and one state file beside the campaign ledger:

- **Screening** (`--proposer screening --factors ...`): a 12-run Plackett-Burman design over up to 11 binary module
  switches (`search.SWITCHES`) on top of the record (the best observation), every run at 600 s; its all-low row is
  the record again. `search.py effects` prints each switch's main effect on
  `val_select` and test with its standard error (one seed's sd, 0.0015 nats, gives 0.0009 per effect) and the
  leading two-factor interactions among active switches. Plackett-Burman aliases each interaction partly with the
  other main effects, so an interaction it names is a lead to test, not an estimate.
- **Bayes** (`--proposer bayes`): a Gaussian process with a noise term over the mixed space (variants one-hot, numbers
  scaled, a switched-off family's numbers held at the middle), its noise never below one seed's; expected improvement
  over the best posterior mean, not the best observation, so it does not chase a lucky seed; a batch by the kriging
  believer.
- **Genetic** (`--proposer genetic`): a population of 8 by tournament over every observed genome, uniform crossover
  over families (a family's variant keeps its own numbers) and mutation of variants and numbers.
- **The warmup cap.** A new graph compiles from scratch, the record's in 106 s against the 180 s cap. Each variant's
  extra seconds, measured alone, are `search.COMPILE_SECONDS`; every proposer skips a genome whose sum comes within
  15 s of the cap, the Bayes pool and the genetic children first put a genome's costly choices back at the record's until
  it fits (few uniform draws do; a repaired child says so in its reason), and the screening refuses a design with such
  a row.
- **Successive halving.** An adaptive batch runs at 180 s first (`--budget 180 --final-splits val_select`: the whole
  schedule compressed to the shorter clock, a complete valid run under its own comparison key), and only the top half
  by `val_select` runs the full 600 s. Both rungs are recorded.

The loop, from the project folder (the workspace sets `SPEEDRUN_LAUNCHER` and `SPEEDRUN_FETCH`; the launch log has
them):

```bash
python search.py --state <ledger dir>/search.json propose --proposer bayes --batch bayes1 --count 8
python search.py --state <ledger dir>/search.json ship --batch bayes1           # the 180 s rung, through queue.sh
python search.py --state <ledger dir>/search.json collect --batch bayes1 --results-dir <dir> --ledger <ledger>
python search.py --state <ledger dir>/search.json ship --batch bayes1           # the promoted half at 600 s
python search.py --state <ledger dir>/search.json collect --batch bayes1 --results-dir <dir> --ledger <ledger>
python search.py --state <ledger dir>/search.json status
```

Every proposal is in the state file the moment it is made (its genome, its parents, the proposer's reason); its
ledger row, written by `collect` through `record.append_row`, carries the same fields with the verdict. A 600 s run
that beats the best observation so far is `keep`, else `revert`; a 180 s rung is `reference`.

## Your place in the loop

Your job is to supply strategies and ideas; the genetic and Bayesian searches find the combinations and the numbers.

1. **Read the state.** `search.py status` (the incumbent, the observations, what is retired), `search.py effects`
   for each screening batch, the ledger's latest rows and the catalog. Look at the curves, not only the final numbers:
   a module behind at 90 s and ahead at 600 s wants a longer horizon, one ahead early and behind late wants a schedule.
2. **Intervene where the data asks**, one kind of change at a time, and record why in the ledger note or the catalog:
   - *Add a module* when an idea is not yet expressible: one module file with its config and strategy, listed in
     `FAMILIES`; the registry tests then build it, check the shape rule and train a CPU step on it. Smoke it, pilot it
     on the box at the 180 s rung, add its measured compile seconds to `search.COMPILE_SECONDS`, then its switch to
     `search.SWITCHES` and its dimensions to `search.SPACE`.
   - *Screen* a set of new switches together with the old ones whose effects are least settled.
   - *Retire* a variant or number the screening shows inert or harmful beyond two standard errors
     (`search.py retire attention.variant=paired`), or one that crashes or breaks a cap, so the proposers stop
     spending runs on it. Retiring is a recorded decision with its numbers, not deletion: the module stays.
   - *Narrow or widen* a number's range in `search.SPACE` when the results pile up at an edge.
3. **Let the proposers run** between interventions: alternate Bayes and genetic batches on the same observations;
   each batch is eight proposals, four of which reach 600 s. Do not hand-pick genomes except to seed a batch with a
   hypothesis worth a run (`search.py seed` records a finished hand-written run as an observation).
4. **Promote a settled record.** When a genome holds the record across later batches, write it to `genomes/` and,
   in its own commit, move the defaults of the families it changes to its values, with a CPU smoke showing the new
   defaults equal the genome.
5. Seed 0 decides. No seed confirmations while the search runs: they return once the record is near a good
   benchmark, about 2.3 nats. The 015 seeds are the one noise measurement so far.

Hand-written candidates still work: a complete `train.py` as `candidates/queue/<name>.py` beside `<name>.txt` runs
through `queue.sh` as before, and a genome as `<name>.json`. One run on the GPU at a time.

## What a run may not do

- Read the held-out tensors except through `benchmark.evaluate_split`.
- Change the corruption, the clock, the evaluator, the shape check or the result writer.
- Train on validation or test proteins, or select weights by test.
- Exceed the warmup or eval caps to buy training time.
