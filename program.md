# The program: an agent's brief for the speedrun

You improve one number: the test cross entropy, in nats per masked residue, of a 36-block protein masked language
model after 10 minutes of training on one GH200. You do it by editing one file, `train.py`, and running it. Everything
else is fixed and audited.

## The benchmark (fixed, `src/speedrunning_plms/benchmark/`)

- **Data.** `Synthyra/plm_speedrun_afdb_v6`, built by `prepare.py` from the family-disjoint AlphaFold DB split: no
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
  candidate's digest and the clock receipts. Two runs compare only when their comparison keys agree.

## The loop

1. Take the top idea from `ideas/README.md`, or one of your own. Write one sentence: what changes, why it should help.
2. Edit `train.py`. Keep the baseline's structure: `Candidate` exposes `blocks`, each with `attention` and `mlp`, a
   `residue_columns` tensor, `forward(input_ids, attention_mask)` for the evaluator and `hidden_states(...)` for the
   probe. Everything inside is yours.
3. Smoke it on the CPU: `python train.py --data-dir <tiny data> --output-dir /tmp/smoke --name smoke --max-steps 2
   --no-compile --no-wandb --skip-final-evals --budget 5`.
4. Run it: `bash run_gh200.sh train --name <nnn>_<slug> --hypothesis "..."` on the box (from the workspace,
   `ws remote run speedrun_gh200 speedrunning_plms --detach -- bash run_gh200.sh train ...`). One run at a time.
   Several independent candidates on the same incumbent go in one shipment instead: `candidates/queue/<name>.py`
   (a complete train.py) beside `<name>.txt` (the hypothesis), then `-- bash queue.sh`, which runs them back
   to back on the box, so the GPU never waits for the driver. Still one run on the GPU at a time.
5. Read `result.json` (`/state/runs/<name>/` on the box, a copy under `index/outputs/<experiment>/`). A lower
   `metrics.val_select.loss` than the incumbent keeps the change: snapshot the shipped bytes to
   `candidates/<name>.py` and run `probe.py` on it when convenient; otherwise restore the incumbent's bytes from
   `candidates/`. Record every run with `record.py` in the campaign ledger, with the verdict and the hypothesis.
6. Seed 0 decides. Confirm a milestone with seeds 1 and 2 before calling it a record.

## What a run may not do

- Read the held-out tensors except through `benchmark.evaluate_split`.
- Change the corruption, the clock, the evaluator, the shape check or the result writer.
- Train on validation or test proteins, or select weights by test.
- Exceed the warmup or eval caps to buy training time.
