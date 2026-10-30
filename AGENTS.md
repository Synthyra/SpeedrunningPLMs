# SpeedrunningPLMs: how to work here

Read `program.md` first: it is the brief for running the loop. The short form:

- A candidate is a genome (a JSON file of family choices) built by `train.py`'s skeleton from the strategy modules in
  `src/speedrunning_plms/strategies/`. A new idea is a module file listed in the registry, not a fork of `train.py`.
  Everything under `src/speedrunning_plms/benchmark/` (data, tokens, the evaluator, the clock, the shape rule, the
  result schema, the probe), `prepare.py`, `baselines.py`, `probe.py`, `record.py` and `plots.py` are fixed; a change
  to them changes the benchmark and is a separate, deliberate commit.
- `search.py` proposes genomes (screening, Bayes, genetic), ships them through `queue.sh`, and reads the results into
  the campaign ledger with their proposer, parents and reason; its state file sits beside the ledger. The agent
  reasons on top: reads the state and the screening effects, adds modules, retires inert ones, lets the proposers run.
- A 600 s run is kept when its `val_select` loss is the lowest so far; seed 0 decides. Every run is recorded with its
  verdict and reason. `ideas/README.md` is the module catalog: each family's variants and evidence, and the ideas
  not yet a module.
- With no genome `train.py` equals 022 bit for bit on a CPU smoke (`tests/test_strategies.py`). Smoke a
  new module on the CPU before it goes to the GPU: `python train.py --data-dir <tiny> --output-dir <out> --name smoke
  --device cpu --no-compile --no-wandb --window 64 --windows-per-step 2 --max-steps 3 --skip-final-evals --genome g.json`.
- The clock measures applied training steps only; compile warmup and in-run validation are outside it and capped.
  A candidate that breaks a cap is `invalid`, not a record.
- Python follows `CODING_STANDARDS.md`: import order, precise types, intent comments, and a shape trace on every
  tensor line. `candidates/` snapshots are frozen bytes and exempt.
