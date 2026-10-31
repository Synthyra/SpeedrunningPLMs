# Budget scaling: what changes in the best recipe at one hour

Date: 2026-10-31. A literature scout for SpeedrunningPLMs (36 blocks, 768 stream, 2,048 MLP, 15 percent MASK-only
corruption, one GH200). It asks what changes when the clock goes from 600 s to 3,600 s, with the data stream (392.6M
tokens) read by a few passes rather than fewer than one. It read `program.md`, `ideas/README.md`, the format of
`ideas/2026-10-30_short_run_schedules.md`, the `TRACKS`, `SPACE` and `SWITCHES` in `search.py`, the 1-hour
`TRACKS` entry, and the modules it cites (`schedule.py`, `optimizer.py`, `regularization.py`, `ema.py`, `batch.py`,
`attention.py`). The 10-30 files already cover the optimizers, noise transfer, regularization and averaging,
capacity at fixed shape, MLM objectives, data order and init; their sources are cited where they bear on the
1-hour question and not re-ranked. Shorthand: TPP is tokens per parameter (198M non-embedding parameters here); CBS
is the critical batch size, above which a larger batch stops cutting steps; LR is learning rate; WD is weight decay;
EMA is exponential moving average; MLM is masked language modeling; "nats" is validation cross entropy. Nothing here
is measured on this benchmark. Each gain is a source's number or our estimate, and the text says which. One seed
resolves about 0.0015 nats (README).

## Outcome

- **The premise needs one correction.** At the record's step time (0.276 s a step, catalog profile), a 3,600 s run is
  about 13,000 steps of 65,536 tokens: about 0.85B tokens, or 2.2 passes over the 392.6M-token stream. The 600 s run
  is about 2,150 steps (0.14B tokens, 0.36 of a pass). So the one-hour run repeats data, starting at about 46 percent of
  the clock. TPP goes from 0.7 to 4.3. Repetition is a weak concern at this count: Muennighoff et al. (arXiv
  2305.16264, abstract) report that up to 4 epochs of repeated data change the loss little against unique data.
- **The main change is scale, not a new trick.** Several ranges in `SPACE` were set for about 2,150 steps and are in
  units that do not transfer: the Muon and AdamW learning-rate floors (set where the 10-minute optimum sat), the
  cooldown floor, the EMA window (in steps), and `decay_timescale` (in total runs). Four of the top ideas are range
  edits that cost under an hour each; the last measurement is the batch count, which is not in `SPACE` at all.
- **Top three, by expected gain per implementation hour:** (1) lower the learning-rate floors for the 1-hour track, since
  both sources that give a horizon law say the optimal rate falls as the run grows; (2) lower the cooldown floor from
  0.25 to 0.15; (3) scale the EMA window to the run's steps. The batch count (idea 4) has the widest sign uncertainty
  and is the one to measure.
- **Dropout is last.** The repetition source (Xue et al.) finds dropout helps with repeated data, but the repetition here
  is about 1.2 extra passes, and the same paper reports dropout slowing learning when data is ample.
- **Out of the rules.** Masking rate and span masking at larger budgets are the protein-MLM question the brief names;
  the corruption is fixed by the rules, so it is listed under "Outside the rules" with no change proposed.

## Ranked table

Gain is the expected change in validation loss on this benchmark (our estimate; negative is better). "Read" says how
the source was read: fetched (the page text, with the sections named), search summary, or the 10-30 file's reading.
Cost is implementation hours for this setting. Compile is the warmup cost of a new graph shape, against the 180 s
cap. Step cost is per step at the record's profile.

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats) | Compile | Step cost | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | Learning-rate floors fall with the 1-hour horizon (Muon and AdamW groups) | arXiv 2409.19913 (2024, ICLR 2025); arXiv 2501.18965 (2025) | fetched (targeted re-read, Section 3.2, Tables 5 and 10); fetched (Section 5.1) | 0.5 | 0 to -0.005 | none (bounds only) | 0 | the direction is in two sources; the size is an extrapolation about 30 times below the source's token range |
| 2 | Cooldown floor from 0.25 to 0.15 at one hour | arXiv 2405.18392 (2024, NeurIPS 2024) | fetched (Sections 3.2 and 5, Tables 5 and 6) | 0.1 | 0 to -0.003 | none | 0 | the cooldown needed to match cosine shrinks as runs lengthen; the 1-sqrt versus linear gap is at most 0.03 points |
| 3 | EMA window scaled to the run's steps | arXiv 2209.14981 (2022); arXiv 2306.03241 (2023) (cited in the 10-30 averaging file) | the 10-30 file's reading; the scaling rule is ours | 0.5 | 0 to -0.004 | none (snapshots, not graph) | 0 (snapshot copies only) | no source gives an averaging-horizon law at this scale |
| 4 | Batch count as a searched dimension (32 and 128 windows) | arXiv 2410.21676 (2024, ICLR 2025); arXiv 2507.01598 (2025) | fetched (Sections 1, 3.2, 3.3, Appendix E); search summary | 1 | -0.005 to +0.005 (sign open) | a new shape: 3 to 10 s, estimate | 0; 128 windows needs about twice the activation memory (pilot) | CBS grows with data size; no Muon rule for LR against batch found |
| 5 | Decay timescale grid corrected for the run's steps | arXiv 2505.13738 (2025, revised 2025); arXiv 2405.13698 (2024) | partial HTML (the 10-30 file's reading); code | 1 | -0.004 to +0.002 | none | 0 | the grid is in total runs, which makes the same value six times weaker per step at one hour |
| 6 | Late attention-logit guard: log the logit maximum and screen `score_cap` | arXiv 2309.14322 (2023, ICLR 2024) | search summary of the abstract and PDF | 1 | 0 to -0.005 (conditional on growth) | none (the switch exists; logging is eval-side) | 0 | the mechanism is a snippet; it is a guard, not a measured gain |
| 7 | Second-pass dropout: extend `early_dropout` to the repeat | arXiv 2305.13230 (2023, NeurIPS 2023); arXiv 2305.16264 (2023) | fetched (ar5iv, Section 4 Insights 8 to 10, Section 5); fetched (abstract only) | 0.5 | -0.002 to +0.003 | none (one graph) | the draw's kernels for the whole run (module docstring); pilot | the sources measure masked-token accuracy, not loss; at about 2.2 passes the repeat cost is small |
| - | Masking rate and span masking at a larger budget | arXiv 2202.08005 (2022, EACL 2023); arXiv 2411.02142 (2024) | 10-30 files (2202.08005 summary); abstract (2411.02142) | not ranked | not estimated | - | - | out of the rules: the corruption is fixed |

## 1. Learning-rate floors fall with the 1-hour horizon

- **Source:** Bjorck, Benhaim, Chaudhary, Wei and Song, "Scaling Optimal LR Across Token Horizons," arXiv 2409.19913
  (posted 30 September 2024; ICLR 2025), https://arxiv.org/abs/2409.19913. Read: fetched the v3 HTML with a targeted
  prompt; Section 3.2 gives the horizon law LR\*(D) = B·D^(-β), with β fitted per model size (Table 5: 0.32 to 0.42 at
  760M to 2.7B; 0.65 to 0.70 at 50M and 125M) and a joint fit β ≈ 0.32, α ≈ 0.23 (Equation 3, models at 760M and up).
  The horizons are 25B to 800B tokens (never below 25B), batch 0.5M tokens, Adam-family settings (β2 0.95, WD 0.1).
  The paper reports that the optimal LR still falls with the horizon under muP (50M model, 25B to 100B tokens). A
  first extraction of the same paper labeled Equation 1 a loss law; the LR reading is the paper's title and the
  targeted re-read, so check the PDF before the bounds are committed.
- **Second source:** Schaipp et al., "The Surprising Agreement Between Convex Optimization Theory and Learning-Rate
  Scheduling for Large Model Training," arXiv 2501.18965 (January 2025), https://arxiv.org/abs/2501.18965. Its theory
  gives an optimal base LR that scales as 1/√T in the horizon T (Corollary 3.3); the fit is read off Figure 3(b) and
  was not checked by us. Its experiments are 124M and 210M Llama-style models on SlimPajama, not MLM.
- **Evidence:** the two laws give LR multipliers for six times the steps of 6^(-0.32) = 0.56 (Bjorck, joint fit) and
  6^(-0.5) = 0.41 (theory). The record's Muon LR 0.04 therefore moves to about 0.016 to 0.022, below the current
  `SPACE` floor of 0.02 (the code allows 0.005, `optimizer.py` BOUNDS). The AdamW groups' record values are
  `embed_lr` 3e-3, `head_lr` 1e-3 and `scalar_lr` 3e-3 (`optimizer.py` defaults, the 022 record), which move to about
  1.2e-3 to 1.7e-3, 4e-4 to 5.6e-4 and 1.2e-3 to 1.7e-3.
- **What changes here (number ranges only, the `optimizer` family; no new switch):** at the 1-hour track, edit the
  bounds in `search.SPACE`:
  - `optimizer.muon_lr`: (0.02, 0.08) to (0.01, 0.05), log.
  - `optimizer.embed_lr`: (1e-3, 1e-2) to (5e-4, 5e-3), log.
  - `optimizer.head_lr`: (3e-4, 3e-3) to (1.5e-4, 1.5e-3), log.
  - `optimizer.scalar_lr`: (1e-3, 1e-2) to (3e-4, 5e-3), log.
  First check the 10-minute state's observations: if the best `muon_lr` already sits at the 0.02 edge, the floor is
  binding now and the case is stronger.
- **Expected cost:** 0.5 hours (bounds, plus a 600 s rung pilot to show the optimum is interior). Compile: none. Step: 0.
- **Gain estimate:** 0 to -0.005 nats. The direction is supported by two sources and a muP check; the size is our
  estimate, because the fit is about 30 times below the source's token range and Muon was not tested.
- **Risk:** Muon and Adam-family rates may not share the exponent; a bound that is too low wastes rungs; the paper's
  batch (0.5M tokens) is 8 times ours.

## 2. Cooldown floor from 0.25 to 0.15 at one hour

- **Source:** Hägele, Bakouch, Kosson, Ben Allal, von Werra and Jaggi, "Scaling Laws and Compute-Optimal Training Beyond
  Fixed Training Durations," arXiv 2405.18392 (May 2024; NeurIPS 2024), https://arxiv.org/abs/2405.18392. Read:
  fetched the HTML (Sections 3.2 and 5; Tables 5 and 6, first 100,000 characters).
- **Evidence:** the cooldown needed to match cosine decay shrinks as training gets longer (Figure 6 caption); in a
  200,000-step run, about 10,000 cooldown steps almost match cosine. A cooldown under 20 percent of the steps reaches
  cosine's final loss when it is long enough in absolute steps (Takeaway 3). At 1 billion parameters, 460 billion
  tokens (Table 6), a 5 percent and a 20 percent cooldown score within 0.2 points of each other, and the 1-sqrt
  versus linear gap at 100 billion tokens (Table 5) is 0.03 points. The paper does not give a formula for the duration.
- **Counter-evidence, kept:** Schaipp et al. (arXiv 2501.18965, Section 5.1) find that extending a run calls for a
  larger cooldown, about 0.6 when the length doubles. That is a different setting (extending a trained run, not
  training one from the start), so the ceiling of 0.65 stays.
- **What changes here (number range, the `schedule` family):** `schedule.cooldown_fraction` from (0.25, 0.65) to
  (0.15, 0.65) on the 1-hour track. At 13,000 steps, 15 percent is about 1,950 steps, which is a fifth of the 10,000
  steps the paper reports as enough at 200,000; a 600 s rung at 0.15 should be checked for the sign.
- **Expected cost:** 0.1 hours. Compile: none. Step: 0.
- **Gain estimate:** 0 to -0.003 nats. Our estimate.
- **Risk:** the absolute-steps claim is not a law; the search may waste a few runs in a floor region the record does not
  need.

## 3. EMA window scaled to the run's steps

- **Source:** the averaging horizon is in the 10-30 file `2026-10-30_regularization_and_averaging.md` (latest-k averaging,
  LAWA: arXiv 2209.14981 (2022); early weight averaging and PMA: arXiv 2306.03241 (2023)). Not re-read here. The
  scaling rule below is ours; no averaging-horizon law was found at this scale.
- **Evidence (code):** `ema.py` takes `window` in steps, with BOUNDS (100, 1000), and `search.SPACE` searches
  `ema.window` (100 to 1,000, log, integer). At 2,150 steps that is 5 to 47 percent of the run. The same share at
  13,000 steps is about 600 to 6,000 steps.
- **What changes here (number range, the `ema` family):** at the 1-hour track set `ema.window` to (600, 6000) log
  integer, and in `ema.py` BOUNDS to match. `ema.decay` (0.98 to 0.998) is per step, so its timescale of 50 to 500 steps
  is unchanged; leave it.
- **Expected cost:** 0.5 hours. Compile: none (snapshots are outside the graph; check the snapshot copy). Step: a copy of
  the weights every `window / count` steps, under 1 percent of a step at the record's size (our estimate).
- **Gain estimate:** 0 to -0.004 nats. Our estimate.
- **Risk:** the share rule is an assumption. A window of 6,000 steps averages in weights from the high-LR plateau; the
  `ema.start` range (0.4 to 0.9, the share of the clock where averaging begins) is left unchanged and may need its own
  check.

## 4. Batch count as a searched dimension

- **Sources:** Zhang, Morwani, Vyas et al., "How Does Critical Batch Size Scale in Pre-training?," arXiv 2410.21676
  (October 2024; ICLR 2025), https://arxiv.org/abs/2410.21676. Read: fetched the HTML (Sections 1, 1.1, 3.2, 3.3,
  Appendix E). Sato and Naganuma, "Convergence Bound and Critical Batch Size of Muon," arXiv 2507.01598 (July 2025).
  Read: search summary only.
- **Evidence:** CBS tracks data size more than model size (Abstract, Section 1.1). At a fixed model, CBS rises with the
  token count (Section 3.3). Appendix E gives B\* = 22.91 · D^0.47 at 20 percent step overhead, with D in billions; the
  units of B\* are as the paper states them and were not checked by us. Below CBS, doubling the batch roughly halves the
  steps (Section 1). Sato and Naganuma report that Muon's CBS is below AdamW's on ResNet-18 (summary); no Muon LR rule
  against batch was found. Our own catalog (`batch.py` docstring) tried 32 windows (008): 1.8 times the steps, and
  behind from the 180 s rung. The file read does not say whether the LR was re-tuned for that run; check `candidates/008`.
- **What changes here (a dimension, the `batch` family):** add `batch.windows_per_step` to `search.SPACE` with choices
  (0, 32, 128). The code already accepts powers of two in (0, 256) (`batch.py` BOUNDS), so no strategy change. At one
  hour, 32 windows is 32,768 tokens a step (about 26,000 steps), and 128 windows is 131,072 tokens (about 6,500 steps).
  The Muon LR is searched jointly through `muon_lr`; the 1-hour search should not assume a batch-LR rule.
- **Expected cost:** 1 hour (the `SPACE` edit, a `COMPILE_SECONDS` entry, and a 180 s pilot of each new count). Compile:
  a new static shape per count; estimate 3 to 10 s (the 2,048-window entry is 3 s, pilot to replace). Step: 128
  windows needs about twice the activation memory on the GH200; a pilot checks the fit.
- **Gain estimate:** -0.005 to +0.005 nats. The sign is open. At a fixed token budget, a batch below CBS costs little
  token efficiency and gains updates, and a batch above CBS wastes tokens. CBS rises with the data consumed, so the
  one-hour run's CBS is higher than the 10-minute run's, which is why the 10-minute result (32 windows behind) may not
  carry over.
- **Risk:** a larger count is a new graph shape per genome, and a 128-window genome may break the warmup cap or the memory
  budget. Both are measurable in a pilot before any full run.

## 5. Decay timescale grid corrected for the run's steps

- **Source:** Bergsma et al., "Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training," arXiv
  2505.13738 (May 2025, revised November 2025), https://arxiv.org/abs/2505.13738. Read: partial HTML (the 10-30 file's
  reading). Wang and Aitchison, arXiv 2405.13698 (2024): excerpt, as in the 10-30 file.
- **Evidence (code, `optimizer.py` lines 37 to 39 and 335 to 337):** `decay_timescale` is in total runs, and each step
  shrinks the Muon banks by 1 / (timescale x total_steps). At 10 minutes, the grid value 0.1 is a per-step shrink of
  1 / (0.1 x 2,150) = 0.0047, a timescale of about 215 steps. The same value at one hour, 1 / (0.1 x 13,000) = 0.00077,
  is six times weaker per step. The 1-hour track cannot reach the 10-minute strength on this grid, and the code floor of
  0.1 (`optimizer.py` line 108) blocks the value that would.
- **Power Lines' constant-timescale form** (the 10-30 file's idea 3, tau = 1 / (eta x lambda) in steps, 250 steps at
  Muon LR 0.04 and lambda 0.1) corresponds to a timescale of 250 / 13,000 = 0.019 total runs at one hour.
- **What changes here (number range and one floor, the `optimizer` family):** at the 1-hour track, the grid becomes
  (0, 0.02, 0.05, 0.1, 0.3, 1, 3) in total runs, and the floor in `optimizer.py` line 108 drops to 0.01. This is the
  units fix; the constant-timescale schedule of the 10-30 file is a separate switch.
- **Expected cost:** 1 hour (code floor, grid, a 600 s rung). Compile: none. Step: 0.
- **Gain estimate:** -0.004 to +0.002 nats (the 10-30 file's estimate, unchanged). The sign is open, and the search
  decides which units the optimum follows.
- **Risk:** Power Lines measured timescale in steps against batch size, not against total steps; whether the optimum
  is fixed in steps or in fractions of the run is the question this grid leaves to the search.

## 6. Late attention-logit guard

- **Source:** Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities," arXiv 2309.14322
  (September 2023; ICLR 2024), https://arxiv.org/abs/2309.14322. Read: search summary of the abstract and PDF.
  Claims: attention-logit growth and output-logit drift appear in small models trained at high LR; mitigations used at
  scale work in that regime too; the cause is linked to gradient norms that shrink with scale and LR, which makes the
  default AdamW epsilon too large, and to parameter-norm growth; the instability can be predicted from activation and
  gradient norms.
- **Evidence and reading:** the one-hour run takes about six times more steps at a lower LR (idea 1). Parameter-norm
  growth accumulates with steps, so a late logit growth is plausible. This is a mechanism from a summary, not a measured
  effect here.
- **What changes here:** (a) log the largest attention score per block at each in-run eval (outside the clock; the eval
  cap is 360 s at one hour), so the late trend is on record; (b) screen the existing `score_cap` switch
  (`attention.score_cap` choices 0, 4, 5, 6, 8; `SWITCHES` "score_cap") in the 1-hour batches instead of leaving it to
  chance. AdamW's epsilon is not a `SPACE` dimension; a search over it is a candidate only after the code check
  (not done here).
- **Expected cost:** 1 hour (logging, and the screen's extra runs). Compile: none for the switch's graph (its cost is
  not in `COMPILE_SECONDS`; check). Step: 0 to a few percent for the logging, pilot.
- **Gain estimate:** 0 to -0.005 nats, only if a 1-hour curve shows the growth. Our estimate.
- **Risk:** a score cap can cost early learning. Only screen it if the logged maxima grow.

## 7. Second-pass dropout

- **Sources:** Xue, Fu, Zhou, Zheng and You, "To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis,"
  arXiv 2305.13230 (May 2023; NeurIPS 2023), https://arxiv.org/abs/2305.13230. Read: fetched the ar5iv HTML (Section 4,
  Insights 8 to 10; Section 5). Muennighoff et al., "Scaling Data-Constrained Language Models," arXiv 2305.16264 (May
  2023; NeurIPS 2023), https://arxiv.org/abs/2305.16264. Read: abstract only.
- **Evidence:** Xue et al. find that dropout alone is highly effective against multi-epoch degradation, that dropout
  introduced later performs about as well as dropout from the start, and that dropout can slow learning when data is
  ample (Insight 8). Their measures are masked-token accuracy and SQuAD, not loss; the typical rate is 0.1, and 0.2 or
  0.3 was best at scale (Section 5). Muennighoff et al. (abstract) report that up to 4 epochs of repeated data cost
  little loss against unique data. The repeat here is about 1.2 extra passes. The code's docstring cites a one-pass MLM
  finding (Drop Dropout, arXiv 2505.24788) that no dropout was best; that is not re-read here.
- **What changes here (number range, the `regularization` family):** `regularization.until` from (0.05, 0.3) to
  (0.3, 0.7) at one hour. The ramp in `early_dropout` ends at `until`, so at (0.05, 0.3) it finishes before the repeat
  starts (about 46 percent of the clock). A second-pass-only variant (rate starting at the repeat) is a new module and
  is not proposed here (2 hours).
- **Expected cost:** 0.5 hours (range and a 600 s pilot). Compile: none (the rate is a buffer; one graph). Step: the
  draw's kernels run for the whole run, as the module docstring says, so pilot the step time.
- **Gain estimate:** -0.002 to +0.003 nats. Our estimate; the sources do not measure loss at this repetition.
- **Risk:** dropout slows learning on the fresh first pass, which is most of the run. The range change applies it there
  too, unless the second-pass variant is built.

## Outside the rules, or waiting on Logan's word

- **Masking rate and span masking at a larger budget.** Wettig et al., "Should You Mask 15% in Masked Language Modeling?,"
  arXiv 2202.08005 (February 2022; EACL 2023), https://arxiv.org/abs/2202.08005: 40 percent beat 15 percent for a
  BERT-large-sized model on GLUE and SQuAD, and span and PMI masking need a lower rate (fetched in
  `protein_pretraining.md` section 2.1; not re-read here). Cheng et al., "Training Compute-Optimal Protein Language
  Models," arXiv 2411.02142 (November 2024), https://arxiv.org/abs/2411.02142: MLM overfit at 3B parameters when
  UniRef50 was repeated, and metagenomic data removed the plateau (abstract, read in the 10-30 data file). Both are
  out of the rules: `program.md` fixes the corruption at 15 percent MASK-only and forbids changing it. The question for
  Logan: may the 1-hour track use a different corruption rate or span masking? It changes the objective, so the
  comparison with the 10-minute track would need its own key.

## Checked, not found

- **modded-nanogpt longer runs.** The GitHub README (fetched 2026-10-31, https://github.com/KellerJordan/modded-nanogpt)
  has no 50-billion-token folder, no Contra-Muon and no power-law cooldown, though a search summary named all three.
  None is cited. Its record table has the relevant dated changes: LR floor 0.1 (#19, 2025-01-13), cooldown fraction 0.45
  (#26, 2025-07-13), cautious weight decay (#43, 2025-11-10; #50, 2025-12-18), batch-size schedule (#46, 2025-11-29) and
  momentum warmup (#9, 2024-11-06). These are speedrun records, not longer-budget runs with numbers. A 2026 discussion by
  Keller Jordan on longer budgets was searched and not found.
- **Power Scheduler,** arXiv 2408.13359 (August 2024), https://arxiv.org/abs/2408.13359: a learning-rate schedule whose
  peak rate follows a power law in batch size and tokens. Search summaries only, and they disagree on the formula. It
  would be a schedule shape, not a number range; fetch the PDF before any use.
- **A Muon learning-rate law against horizon.** Not found; Bjorck et al. have no Muon run.
- **Other 2026 modded-nanogpt records on longer horizons.** Not found in the README.

## Limits

- Every gain above is ours unless a source number is quoted. The one-hour step time (0.276 s) is the 600 s profile; the
  eval cap (360 s) and compile warmup change the clock and the step count, so the 1-hour counts are estimates.
- The horizon laws were fit at 25B tokens and up (Bjorck) or on other architectures (Schaipp, Hägele); our run is 0.85B
  tokens at 4.3 TPP, so each law is an extrapolation.
- Sources read in summary (Sato and Naganuma, Wortsman et al., Muennighoff et al. beyond the abstract, Power Scheduler)
  carry that uncertainty; the table marks them.
- The batch-count estimate (idea 4) and the EMA share rule (idea 3) are the two places where our own reasoning carries
  the most weight.

## Sources

- Bjorck et al., arXiv 2409.19913 (30 September 2024; ICLR 2025): https://arxiv.org/abs/2409.19913
- Schaipp et al., arXiv 2501.18965 (January 2025): https://arxiv.org/abs/2501.18965
- Hägele et al., arXiv 2405.18392 (May 2024; NeurIPS 2024): https://arxiv.org/abs/2405.18392
- Zhang, Morwani, Vyas et al., arXiv 2410.21676 (October 2024; ICLR 2025): https://arxiv.org/abs/2410.21676
- Sato and Naganuma, arXiv 2507.01598 (July 2025): https://arxiv.org/abs/2507.01598 (summary)
- Xue et al., arXiv 2305.13230 (May 2023; NeurIPS 2023): https://arxiv.org/abs/2305.13230
- Muennighoff et al., arXiv 2305.16264 (May 2023; NeurIPS 2023): https://arxiv.org/abs/2305.16264
- Wortsman et al., arXiv 2309.14322 (September 2023; ICLR 2024): https://arxiv.org/abs/2309.14322 (summary)
- Bergsma et al., arXiv 2505.13738 (May 2025): https://arxiv.org/abs/2505.13738
- Wang and Aitchison, arXiv 2405.13698 (May 2024): https://arxiv.org/abs/2405.13698
- Shen et al., Power Scheduler, arXiv 2408.13359 (August 2024): https://arxiv.org/abs/2408.13359 (summary)
- Wettig et al., arXiv 2202.08005 (February 2022; EACL 2023): https://arxiv.org/abs/2202.08005
- Cheng et al., arXiv 2411.02142 (November 2024): https://arxiv.org/abs/2411.02142
- modded-nanogpt repository, fetched 2026-10-31: https://github.com/KellerJordan/modded-nanogpt
