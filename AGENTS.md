# SpeedrunningPLMs: how to work here

Read `program.md` first: it is the brief for running the loop. The short form:

- `train.py` is the only file a run edits. Everything under `src/speedrunning_plms/benchmark/` (data, tokens, the
  evaluator, the clock, the shape rule, the result schema, the probe), `prepare.py`, `baselines.py`, `probe.py`,
  `record.py` and `plots.py` are fixed; a change to them changes the benchmark and is a separate, deliberate commit.
- One candidate at a time, one change per candidate, a one-sentence hypothesis on the command line. Keep a
  candidate when its `val_select` loss is lower than the incumbent's; otherwise restore the incumbent's bytes from
  `candidates/`. Snapshot every record holder there.
- Every run is recorded with `record.py` in the campaign ledger, kept or reverted, with the verdict and the reason.
  Read `ideas/README.md` before inventing: it holds what was tried, what it did and the ranked queue of what is next.
- Smoke a candidate on the CPU with the tiny data (`tests/`) before it goes to the GPU:
  `python train.py --data-dir <tiny> --output-dir <out> --name smoke --device cpu --no-compile --no-wandb --window 64
  --windows-per-step 2 --max-steps 3 --skip-final-evals`.
- The clock measures applied training steps only; compile warmup and in-run validation are outside it and capped.
  A candidate that breaks a cap is `invalid`, not a record.
- Python follows `CODING_STANDARDS.md`: import order, precise types, intent comments, and a shape trace on every
  tensor line. `candidates/` snapshots are frozen bytes and exempt; `train.py` is the example the rest copies.
