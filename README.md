# SpeedrunningPLMs

**How good a protein masked language model can one GPU train in ten minutes, and what makes it better?**

A fixed benchmark and a searched candidate, in the spirit of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt):
the data, the objective, the clock, the evaluator and the shape rule are fixed; a candidate is a genome, one choice
from each family of composable strategy modules, built by `train.py`'s skeleton. A search proposes genomes (a
screening design, a Bayesian optimizer, a genetic algorithm) and an agent reasons on top of it. A candidate is kept
when it lowers the selection loss of the incumbent, every run carries the digests of the data, the evaluator and the
candidate (the skeleton, the modules and the genome) that produced it, and the record holders before the search are
snapshotted under `candidates/`.

## The record

| # | Test loss (nats) | Bits | `val_select` | Steps | What changed | Candidate |
|---|---|---|---|---|---|---|
| 0 | 2.7671 | 3.992 | 2.7680 | 1,412 | the baseline: pre-norm 36 x 768, RoPE, QK-norm, ReLU-squared MLP, flex attention with a document mask over packed 1,024-token windows, per-matrix Muon with Polar Express, AdamW on the tables, warmup and a 40 percent cooldown by clock fraction | [`000_baseline.py`](candidates/000_baseline.py) |
| 1 | 2.7554 | 3.975 | 2.7570 | 2,157 | Muon banked by parameter shape: one batched Polar Express per shape instead of 216 per-matrix calls (the host had been 76 percent of the step) | [`001_batched_muon.py`](candidates/001_batched_muon.py) |
| 2 | 2.7546 | 3.974 | 2.7564 | 2,161 | Muon learning rate 0.02 to 0.04 | [`002_muon_lr_0.04.py`](candidates/002_muon_lr_0.04.py) |
| 3 | 2.7539 | 3.973 | 2.7559 | 2,157 | tanh logit softcap at 30 | [`003_softcap.py`](candidates/003_softcap.py) |
| 4 | 2.7501 | 3.968 | 2.7531 | 2,059 | a fixed ALiBi-style local bias on the attention scores beside RoPE, per-head slopes 2^(-8 i / h): every head is local from step 0 | [`012_local_bias.py`](candidates/012_local_bias.py) |
| 5 | 2.7457 | 3.961 | 2.7495 | 2,051 | NorMuon: the orthogonalized update normalized per output neuron by an EMA of its mean square, rescaled to the Muon norm (confirmed at seeds 1 and 2: 2.7447 and 2.7437) | [`015_normuon.py`](candidates/015_normuon.py) |
| 6 | 2.7455 | 3.961 | 2.7490 | 2,055 | embedding lr 3e-3 and head lr 1e-3, three times the tables' rates; kept by the rule, inside one seed's band | [`022_embed_head_lr.py`](candidates/022_embed_head_lr.py) |
| 7 | 2.7291 | 3.937 | 2.7319 | 1,994 | a value residual: every block's values mixed with the first block's, V = a V_l + (1 - a) V_0 with a learned scalar a per block from 0.5 (ResFormer); the gain is on long proteins, 0.025 nats over 500 residues | [`028_value_residual.py`](candidates/028_value_residual.py) |
| 8 | 2.7175 | 3.921 | 2.7217 | 2,147 | the first screen's best row on 7: 6 heads of 128 instead of 12 of 64, value embeddings on blocks 0 to 2 and 33 to 35, no softcap, doubled local slopes, U-net skips and a tail weight average | [`screen1_05.json`](genomes/screen1_05.json) |
| 9 | 2.7119 | 3.912 | 2.7177 | 2,148 | the screen's five helpful switches together on 7: 6 heads of 128, value embeddings, RoPE base 100,000, unit selection (only the more active half of the MLP units get gradients each step) and no softcap | [`confirm1_01.json`](genomes/confirm1_01.json) |
| 10 | **2.7078** | 3.907 | 2.7146 | 2,149 | 9 with the shared-head readout warm start in place of unit selection: early in the run the head also reads the stream after every sixth block, so the blocks learn from a signal that does not wait for the end-to-end gradient; the Gaussian process proposed it and the single-switch batch found the same genome at 2.7147 | [`bayes2_01.json`](genomes/bayes2_01.json) |

![The selection curves of every candidate and the record progression](assets/selection_curves.png)

**A.** The `val_select` loss of every seed-0 candidate against the training clock: kept candidates in orange, the
record holder in red, reverted candidates in gray. **B.** The test loss of every candidate in the order tried, with the
record as a step line. One seed resolves about 0.0015 nats (the NorMuon record at seeds 0, 1 and 2: 2.7457, 2.7447
and 2.7437 on test), so a gain inside that band is kept by the rule but not claimed.

The hardware of record is one NVIDIA GH200 (the H100 die, 96 GB HBM3). A run on other hardware is a different clock
and is recorded as such.

## Where it stands

![The record holder against the reference models](assets/references.png)

**A.** Test loss against parameters for the record holder (red star) and the reference models, every model scored on
its own 20 residue columns on the same masked positions; the unigram floor is the loss of predicting the training
residue frequencies. **B.** Test loss by protein length: the ten-minute model is flat across lengths while the
reference models gain on long proteins, so use of context is what the short clock has not yet bought.

| Model | Parameters | `val_select` | Validation | Test loss | Bits | Accuracy | F1 macro | MCC | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| unigram floor | | 2.8934 | 2.8938 | 2.8933 | 4.174 | 0.098 | 0.009 | 0.000 | 0.500 | 0.050 |
| **record, 10 min** | 198,383,652 | 2.7146 | 2.7105 | 2.7078 | 3.907 | 0.160 | 0.123 | 0.094 | 0.663 | 0.126 |
| ESM2 8M | 7,523,034 | 2.6393 | 2.6291 | 2.6253 | 3.788 | 0.186 | 0.157 | 0.124 | 0.695 | 0.157 |
| ESM2 35M | 33,517,234 | 2.5164 | 2.4965 | 2.4992 | 3.606 | 0.225 | 0.206 | 0.168 | 0.735 | 0.208 |
| ESM2 150M | 148,161,274 | 2.3490 | 2.3176 | 2.3303 | 3.362 | 0.275 | 0.267 | 0.224 | 0.780 | 0.275 |
| ESM2 650M | 651,085,494 | 2.0971 | 2.0534 | 2.0710 | 2.988 | 0.354 | 0.355 | 0.309 | 0.836 | 0.380 |
| ESM2 3B | 2,839,090,114 | 1.8921 | 1.8537 | 1.8705 | 2.699 | 0.416 | 0.422 | 0.376 | 0.871 | 0.460 |
| ESMC 300M | 332,997,184 | 2.1463 | 2.1118 | 2.1269 | 3.068 | 0.339 | 0.337 | 0.293 | 0.824 | 0.359 |
| ESMC 600M | 575,036,992 | 2.0676 | 2.0304 | 2.0510 | 2.959 | 0.361 | 0.363 | 0.317 | 0.838 | 0.390 |
| ESMC 6B | 6,352,005,184 | 1.7061 | 1.6617 | 1.6891 | 2.437 | 0.472 | 0.480 | 0.437 | 0.897 | 0.531 |

The reference models trained for days on UniRef50 and more, which holds relatives of the held-out families; they
bound the benchmark from above, they are not a fair race. The ten-minute model sees about 135M tokens, a third of
one pass over the training stream, so no protein is seen twice.

## Everything tried

Candidates are tried one at a time on the incumbent; a candidate is kept when its `val_select` loss is lower. The
ranked queue, with the evidence behind each idea, is [`ideas/README.md`](ideas/README.md); the literature notes it draws
on sit beside it.

| # | Change | Verdict | `val_select` | Test |
|---|---|---|---|---|
| 000 | the baseline | incumbent | 2.7680 | 2.7671 |
| 001 | Muon banked by parameter shape (1.53 times the steps) | kept | 2.7570 | 2.7554 |
| 002 | Muon lr 0.04 | kept, within noise | 2.7564 | 2.7546 |
| 003 | tanh logit softcap at 30 | kept, within noise | 2.7559 | 2.7539 |
| 004 | the incumbent at seed 1 | reference: one seed resolves about 0.0015 nats | 2.7574 | 2.7551 |
| 005 | U-net skips with learned scalars | reverted | 2.7568 | 2.7545 |
| 006 | weight EMA (decay 0.995) over the cooldown | reverted: led mid-cooldown, lagged at the end | 2.7578 | 2.7560 |
| 007 | learnable exclusive self-attention | reverted: 4 percent fewer steps, behind throughout | 2.7610 | 2.7593 |
| 008 | 32 windows a step | reverted: 1.8 times the steps, ahead at 90 s, behind from 180 s | 2.7579 | 2.7557 |
| 009 | 512-token windows, 128 a step | reverted: 6 percent more steps, long proteins lose | 2.7600 | 2.7580 |
| 010 | a (left, right) neighbor-pair table added to the embedding | reverted | 2.7583 | 2.7563 |
| 011 | zero-initialized value embeddings on blocks 0 to 2 and 33 to 35 | reverted | 2.7594 | 2.7573 |
| 012 | a fixed ALiBi-style local bias on every head beside RoPE | kept: ahead at every eval from 90 s | 2.7531 | 2.7501 |
| 013 | one protein per family per pass (family-balanced order) | reverted: the same curve; the budget is not family-limited | 2.7532 | 2.7502 |
| 014 | the local bias on six heads, six free | reverted: loses most of 012's gain | 2.7548 | 2.7515 |
| 015 | NorMuon | kept: ahead from 180 s | 2.7495 | 2.7457 |
| 016 | cautious weight decay 0.1 on the block matrices | reverted: behind from 180 s | 2.7564 | 2.7535 |
| 017 | embedding lr 3e-3, head lr 1e-3 | reverted: within noise of 015 | 2.7527 | 2.7495 |
| 018 | Muon lr 0.06 | reverted: behind from 270 s | 2.7551 | 2.7521 |
| 019 | cooldown over the last 60 percent of the clock | reverted: within noise of 015 | 2.7534 | 2.7498 |
| 015 s1, s2 | the record at seeds 1 and 2 | reference: three-seed test mean 2.7447, span 0.002 | 2.7482, 2.7480 | 2.7447, 2.7437 |
| 020 | mimetic V and O initialization (W_O W_V = 0.25 I at the start) | reverted: the same curve | 2.7496 | 2.7453 |
| 021 | the local-bias slopes learned per head inside the flex attention score function | invalid: compile warmup 264 s against the 180 s cap, then 28 s a step | | |
| 022 | embedding lr 3e-3, head lr 1e-3, on 015 | kept by the rule: lower by 0.0005, as it was on 012 | 2.7490 | 2.7455 |
| 023 | Muon momentum 0.95 from step 0, no warmup from 0.85 | reverted: lost 0.004 | 2.7527 | 2.7491 |
| 024 | RoPE base 100,000 | reverted: under 022 by 0.0011, inside the band, not under 028 from the same queue; retest on 028 | 2.7479 | 2.7443 |
| 025 | the local-bias slopes doubled | reverted: under 022 by 0.0005, inside the band; retest on 028 | 2.7485 | 2.7450 |
| 026 | per-head sigmoid attention gates from the residual's first h channels | reverted: lost 0.002 | 2.7510 | 2.7476 |
| 027 | 512-token windows for the first 40 percent of the clock, then 1,024 | invalid: two window shapes compiled twice, warmup 249 s against the 180 s cap; its 2.7479 does not count | | |
| 028 | a value residual to the first block's values, a learned scalar per block from 0.5 | kept: lower by 0.017, eleven times the band; 0.025 nats on proteins over 500 residues | 2.7319 | 2.7291 |
| pilot | the MLP's matmuls in FP8 (`torch._scaled_mm`, per-tensor scales) | not run: 0.284 s a step against 0.276 in bf16 | | |

What the campaign has learned so far:

- **The step is GPU-bound at 0.276 s.** After the banked Muon, a profile puts matmuls at 109 ms near the bf16 peak,
  flex attention at 80 ms, Muon at 26 ms and the host idle. More steps now come from attention or from precision,
  not from the host; FP8 did not pay at width 768.
- **Context use is the open gap, and the value residual opened it.** Through 022 the test loss of the ten-minute
  model was flat from 50 to 2,046 residues, 0.15 nats under the unigram floor, while the reference models gain most
  on long proteins. A local prior on attention (012) was the first change beyond noise after the optimizer, and
  freeing half the heads (014) gave most of it back. Mixing the first block's values into every block's (028) is
  the first gain that grows with length: 0.004 nats under 100 residues, 0.025 over 500.
- **The first screen found five gains on the value residual.** Twelve runs switching 11 modules on and off together
  (a Plackett-Burman design, standard error 0.0009 per effect): 6 heads of 128 instead of 12 of 64 (-0.0129, with
  9 percent more steps), value embeddings at both ends (-0.0050, a loss before the value residual), RoPE base 100,000,
  unit selection and no softcap (each about -0.0025). The neighbor table, gates, U-net skips and a tail weight
  average hurt; plain Muon in place of NorMuon costs 0.0167. The five gains together (record 9) gave 0.014 on
  `val_select`, less than the 0.025 their sum predicts, so they overlap.
- **The adventurous modules cost more steps than they give at three minutes.** Alone on 022 at a 180 s clock,
  Differential Transformer V2, Dual Triangle Attention, paired heads, nGPT, hyperconnections and residual-scalar
  evolution strategies each trail the record, every one with slower steps; MUDD-style dense connections and the
  composed triangle-differential kernel break the compile cap.
- **Local k-mer context is worth 0.026 nats.** Count models fitted on the training stream and scored on the test
  masks: unigram 2.8933, one neighbor each side 2.8745, two each side 2.8674. The record passes all of them before
  its first eval, so tables, priors or auxiliary losses built on k-mer statistics have nothing left to add; the
  gains come from context beyond three residues.
- **The budget is not family-limited.** Covering every training family in the first half of the clock (013) gave
  the same curve as the shuffled stream.
- **The optimizer's fine settings are inside the noise.** Three times the table rates (022) and a longer cooldown
  (019) moved `val_select` by 0.0005 or less; the Muon rate past 0.04 (018) and cautious decay (016) lost 0.003 to
  0.007. The record's three seeds span 0.002 on test.
- **A learnable tensor inside the flex attention score function is not free.** Slopes as parameters (021) put the
  kernel on a path a hundred times slower (28 s a step) and broke the compile cap; the fixed slopes keep the gain.
- **Structure follows.** A probe from the hidden states to Foldseek 3Di states (reported, never selected on) reaches
  test accuracy 0.367 and MCC 0.245 on the NorMuon record, against 0.384 and 0.266 for ESM2 8M and 0.514 and 0.437
  for ESMC 6B through the same probe.

## The genome and the search

A genome is a JSON file of family choices; what it leaves out takes the default, so an empty genome is 022 (bit for bit
on a CPU smoke), and the record is [`genomes/bayes2_01.json`](genomes/bayes2_01.json). A genome reads:

```json
{"schema": 1, "families": {"attention": {"variant": "triangle"}, "stream": {"variant": "hyper"}, "optimizer": {"muon_lr": 0.05}}}
```

| Family | Variants (the default first) |
|---|---|
| `positional` | `alibi` (fixed per-head distance slopes beside RoPE), `none`; numbers `slope_scale`, `rope_base` |
| `attention` | `standard`, `differential` (Differential Transformer V2), `triangle` (Dual Triangle Attention), `triangle_differential`, `paired` (paired-head); `heads` 12 or 6 |
| `residual` | `prenorm`, `ngpt` (the stream on the unit-RMS sphere with learned eigen rates and weight normalization) |
| `stream` | `single`, `hyper` (static hyperconnections), `dense` (MUDD-style dense depth connections) |
| `unet`, `neighbor`, `value_embeddings`, `value_residual`, `gates`, `embedding_mix` | off, or U-net skips, a neighbor-pair table, value tables, a value residual, per-head output gates, a learned mix of the embedding into every block's input |
| `softcap` | `tanh`, `none` |
| `optimizer` | `normuon`, `muon`; rates and cautious weight decay |
| `schedule`, `batch`, `curriculum`, `ema` | the warmup, cooldown and momentum; the window and batch; short windows first; a tail weight average |
| `backprop_free` | `none`, `residual_es` (sublayer scales trained by evolution strategies), `unit_select` and `unit_random` (forward-ranked MLP units train, against a random control), `readout_warmstart` (shared-head readouts of every sixth block early) |

`search.py` holds three proposers. A 12-run Plackett-Burman screening design over up to 11 module switches, each run
against the record, estimates main effects and leads on interactions before any adaptive search. A Gaussian process with a noise term (never below
one seed's spread) proposes by expected improvement over its best posterior mean, so a lucky seed is not chased. A
genetic algorithm (population 8) recombines families and mutates numbers. An adaptive batch runs 180 s first and only
its better half the full 600 s. Each variant's compile seconds were measured on the GPU, and no proposer ships a genome
whose sum comes within 15 s of the warmup cap. Every proposal, its parents and the proposer's reason go into the campaign ledger
beside the result. The catalog of families and the evidence for each is [`ideas/README.md`](ideas/README.md).

The search runs itself on the GPU box. `autosearch.sh` loops until a deadline or a `stop` file: `autosearch.py next`
ships up to four runs still to do, newest batch first, else proposes four from the Bayes and genetic proposers in
turn, straight at the full budget; `queue.sh` runs them one after another; `autosearch.py collect` reads them into the
state and the ledger, which live beside the runs on the box while the loop runs. Everything the benchmark leaves free
is the two proposers' to optimize. New strategy modules ship as a new tree with the same command: the new loop takes
over at the running cycle's end, and a pilot batch of the new modules runs ahead of the backlog.

A second track, the 1-hour speedrun, started on 2026-10-31: the same benchmark at 3,600 s of training (evals every
180 s of clock under a 360 s cap, the compile cap unchanged), searched from the 000 baseline by random draws and then
the Bayes and genetic proposers alone, with its own state, ledger and figures; nothing of the 10-minute track seeds it.
Its first batches go through a 600 s rung and the top half on to the full hour. Its results have their own section
once the first runs land.

## The benchmark

| Rule | What it is |
|---|---|
| Data | [`Synthyra/SpeedrunningPLMs`](https://huggingface.co/datasets/Synthyra/SpeedrunningPLMs): 1,754,034 training proteins (392.6M tokens) from the AlphaFold DB family split of the Alignment Atlas, 25,459 validation and 25,778 test proteins. No held-out protein has a training relative at 30 percent identity over 80 percent coverage by sequence (MMseqs2) or by structure (Foldseek). Lengths 50 to 2,046 residues. ESM2 alphabet. |
| Objective | 15 percent of standard residues replaced by MASK, nothing else; collapse-free cross entropy ([arXiv 2610.31642](https://arxiv.org/abs/2610.31642)) on the masked residues over the 20 standard residues. Training uses the same corruption. |
| Clock | 10 minutes of applied training steps on one GPU, measured with device synchronization; the overrunning step is discarded. Compile warmup is outside the clock, capped at 3 minutes; in-run validation is outside the clock, capped at 60 seconds in total. A broken cap makes the run `invalid`. |
| Selection | The weights with the lowest loss on `val_select` (one protein per validation family, 4,751 proteins, at most 1,024 tokens, fixed masks) are scored once on the full validation and test splits at full length (at most 2,048 tokens, fixed masks). The headline is the test loss, in nats per masked residue. |
| Shape | 36 blocks on a 768-wide residual stream, each one attention sublayer and one MLP with intermediate 2,048 (about 198M parameters with the embedding and head). Heads, grouped or differential attention, biases, value embeddings, skips, gates, scalars, normalization, tables and the optimizer are free. The shape is checked on every run. |
| Metrics | Loss, bits, perplexity, accuracy, macro and weighted precision, recall and F1, MCC, macro ROC AUC and PR AUC, loss by length bin; every model scored on its own 20 residue columns, so ESM2 (33 tokens) and ESMC (64 tokens) are scored on the same problem. |
| Structure probe | A softmax-weighted sum of all hidden states (each RMS-normalized), then one linear layer to the 20 Foldseek 3Di states, trained on 20,000 training proteins and scored on validation and test. Reported, never used for selection. |
| Seeds | The search runs at seed 0 and keeps single-seed wins. Seed confirmations are off until the record is near 2.3 nats; the one noise measurement so far is the NorMuon record at three seeds (test 2.7457, 2.7447, 2.7437). |

## Running

```bash
pip install -e ".[test]"
python -m pytest -q                      # CPU tests of every fixed part, a few minutes
bash run_gh200.sh build                  # the image (CUDA 12.8, torch 2.11; aarch64 or x86_64)
HF_TOKEN=... bash run_gh200.sh prepare   # the data, once per machine
bash run_gh200.sh pilot                  # a 60 s clock: step time, compile time, memory
bash run_gh200.sh train --name 000_baseline --hypothesis "the baseline"
bash run_gh200.sh probe --run-dir /state/runs/000_baseline
bash run_gh200.sh baselines              # the reference models through the same evaluator and probe
bash queue.sh                            # every candidates/queue/<name>.py or <name>.json (a genome) back to back
python train.py --genome genomes/024_rope_base_1e5.json ...   # one genome; without --genome, the record
python search.py --state search.json propose --proposer screening --batch screen1 --factors unet gates triangle ...
python search.py --state search.json ship --batch screen1      # through queue.sh ($SPEEDRUN_LAUNCHER runs it remotely)
python search.py --state search.json collect --batch screen1 --results-dir results --ledger runs.jsonl
python search.py --state search.json effects --batch screen1   # main effects and interaction leads
```

A run writes `result.json` (the metrics, the clock and architecture receipts, the selection curve, the digests) and
`best.pt` (the selected weights). `record.py` appends a run to a campaign ledger with its verdict; `plots.py` draws
the figures above from the ledger and the result files. `program.md` is the brief an agent follows to run the loop.

## Contributing

A contribution is a module or a genome. A module is one file under `src/speedrunning_plms/strategies/`: a config
dataclass whose defaults are the record's, and a strategy that overrides only the hooks it needs (modules at
construction, optimizer groups, forward hooks, step and evaluation callbacks), listed in the registry; the tests then
build it, check the shape rule and train a CPU step on it. A genome is a run on the clock with its verdict recorded. `AGENTS.md` is the short form of the loop; `CODING_STANDARDS.md` is how the Python is written
(import order, precise types, intent comments, a shape trace on every tensor line), and `python -m pytest -q` checks
the fixed parts and the working candidate on the CPU. The benchmark modules change only with a reason, in their own
commit.

## Layout

```
train.py             the skeleton (embedding, 36 blocks, head), the training loop and the result; builds a genome
src/speedrunning_plms/strategies/  one module per family a genome chooses from, and the genome registry
genomes/             genomes of note, the record's first
search.py            the screening, Bayesian and genetic proposers, successive halving, shipping and collection
candidates/          every record holder's train.py before the search, and candidates/queue/ for the next shipment
ideas/               the ranked queue of what to try and the literature it rests on
prepare.py           builds the benchmark data from the source dataset
baselines.py         the reference models through the same evaluator and probe
probe.py             the 3Di probe on a run's selected weights
record.py, plots.py  the campaign ledger and its figures
run_gh200.sh         the Docker entrypoint on the GPU machine; queue.sh runs queued candidates
program.md           the brief an agent follows; AGENTS.md its short form; CODING_STANDARDS.md how the Python is written
src/speedrunning_plms/benchmark/   data, tokens, evaluator, clock, shape rule, result schema, probe (fixed)
src/speedrunning_plms/optim/       the per-matrix Muon (Newton-Schulz or Polar Express) the candidate's banked one is checked against
tests/               CPU tests of every fixed part, every module, the genome, the record's reproduction and the proposers
```

## License

MIT.
