# Schedules for very short runs, not yet in the catalog

Date: 2026-10-30. A literature scout for SpeedrunningPLMs (36 blocks, 768 stream, 2,048 MLP, 15 percent MASK, 65,536
tokens a step in 64 windows of 1,024, Muon at lr 0.04 with NorMuon, AdamW on the embedding, head and scalars, one GH200,
600 s, about 2,150 steps of 0.28 s; record 2.7146 nats). It read `ideas/README.md`, `ideas/2026-10-30_speedrun_optimizers.md`
and `ideas/2026-10-30_regularization_and_averaging.md`, and does not repeat them. Nothing here is measured on this
benchmark. Each gain is a source's number or our estimate, and the text says which. Shorthand: TPP is tokens per
parameter (0.3 to 0.7 here); CBS is the critical batch size, above which a larger batch stops cutting steps; LR is
learning rate; WD is weight decay; beta2 is Adam's second-moment decay; ln is the natural log; "nats" is validation
cross entropy. One seed resolves about 0.0015 nats (README). The yardstick: one unit of ln(steps) is worth about 0.026
nats (the catalog's 001, where 1.53 times the steps gave 0.011 nats). A 20 percent step saving is therefore about 0.006
nats. That slope comes from a different change, so treat every conversion as rough.

## Outcome

- Nine ideas, ranked by expected loss gain per implementation hour. The top five are raising Adam beta2 through the
  cooldown, a Muon momentum end and taper, a constant-timescale weight decay, a Muon learning-rate search range set by
  the horizon law, and a late gradient-norm correction. Compile risk is none for all five, with one check on the Adam
  betas (idea 1). Each costs 0.5 to 2 hours (idea 5 after its full text). Each estimate is 0.004 nats or less in size,
  and the sign of idea 3 is not known.
- The token-per-step ramp has the largest single estimate on paper (about 0.007 nats from the step count), but the
  benchmark fixes 65,536 tokens a step, so it needs Logan's word, and any real saving changes the compiled shape. The
  catalog's two-shape attempt (027) reached a 249 s warmup against the 180 s cap. Net, about 0.002 nats at best.
- Most sources here measure at 20 TPP or more, or on vision. Our run sees 0.3 to 0.7 TPP. The one result from a
  speedrun-scale schedule is modded-nanogpt's batch-schedule record, about 2 percent of wall clock, and the record does
  not isolate it.
- Not found: a Muon-specific rule for the learning rate against the batch size, a schedule that sets the global rate
  from the gradient norm or the loss with numbers, a warm-restart study at a short LM budget, and a short-budget study
  of per-depth learning rates. The "Checked and rejected" section lists what was looked at.

## Ranked table

Gain is the expected change in validation loss on this benchmark (our estimate; negative is better). "Read" says how
much of the source was read. Cost is implementation hours for this setting. Step cost and compile cost are estimates
against the catalog's profile (276 ms a step; banked Muon 26 ms; 180 s warmup cap). Each "Read" entry names the form
the numbers came from: abstract, partial HTML (the first part of the page, read through a summary), search summary.

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats) | Compile | Step cost | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | Adam beta2 raised through the cooldown | arXiv 2508.01483 (2025, ICLR 2026 poster); arXiv 2406.19146 (2024) | abstract; Porian et al. by search summary | 1 | 0 to -0.004 | none, if the betas are tensors or the step is uncompiled (check `train.py`) | 0 | consistent gain reported; no size |
| 2 | Muon momentum end at 0.98, tapered back in the cooldown | arXiv 2410.23922 (2024); reference code in arXiv 2602.11948 (2026) | partial HTML (Sections 3 and 6.1) | 1 | 0 to -0.003 | none | 0 | high momentum removed warmup for a non-Muon optimizer; no Muon value |
| 3 | Constant-timescale weight decay: lambda_t = 1 / (tau * lr_t), capped | arXiv 2505.13738 (2025); arXiv 2405.13698 (2024) | partial HTML (first part); excerpt | 1 | -0.004 to +0.002 | none | 0 | fit over 20 to 1,280 TPP; our budget is below the range |
| 4 | muon_lr search range set by the token-horizon law (a search change) | arXiv 2409.19913 (2024, ICLR 2025); arXiv 2505.02222 (2025) | partial HTML (first part); abstract | 0.5 | 0 (information) | none | 0 | fit for AdamW at 25B tokens and up; no Muon fit |
| 5 | Late gradient-norm correction (weight decay and normalization interaction) | arXiv 2506.02285 (2025) | abstract; snippet for the variant names | 2 after the full text | 0 to -0.003 | none | 0 | "lower loss throughout training"; no number in the abstract |
| 6 | Warm-up-free or adaptive warm-up for Muon | arXiv 2602.05813 (2026); arXiv 2410.23922 (2024) | abstract; partial HTML | 2 to 3 | 0 to -0.002 | none | 0 | adaptive matches or beats tuned warm-up; no numbers in the abstract |
| 7 | Depth-graded learning rate across the 36 blocks | arXiv 2602.07494 (2026); arXiv 2510.03871 (2025) | abstract | 1 to 2 | +0.005 to -0.005, sign unknown | none (a broadcast scale tensor) | 0 | laws across depths, not across block index |
| 8 | Token-per-step ramp: 32 windows early, 64 late | arXiv 1711.00489 (2018); arXiv 2601.05034 (2026); modded-nanogpt record 46 | abstract; partial (fetch of the repository) | 4 to 6 | -0.002 net (0 to -0.004); 0 with two shapes | two shapes invalid (catalog 027); one dynamic shape 13 percent slower a step | -50 percent early, +13 percent with one dynamic shape | record 46: about 2 percent of wall clock |
| 9 | Polar Express iteration schedule: 3 iterations early, 5 late | arXiv 2505.16932 (2025, ICLR 2026) | partial HTML (Sections 1.3, 4.5, 5.1, 5.2, Eq. 17) | 2 | 0 to -0.001 | a second graph (cap risk); identity coefficients save nothing | -4 percent early | no iteration-count ablation |

Compile costs are not listed per row beyond the risk named: each new module adds a few seconds, and the 180 s cap
applies. Gains assume the catalog's 2,150-step run. A row's gain is not additive with its neighbors without a test.

## 1. Adam beta2 raised through the cooldown

- Source: Dremov, Hägele, Kosson and Jaggi, "Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay Learning
  Rate Scheduler," arXiv 2508.01483 (TMLR 2025; ICLR 2026 poster). Read: abstract. Porian, Wortsman, Jitsev, Schmidt and
  Carmon, "Resolving Discrepancies in Compute-Optimal Scaling of Language Models," arXiv 2406.19146 (NeurIPS 2024): the
  search summary only.
- Evidence: the abstract reports "consistent improvements with higher values of β2 during cooldown". Tuning the AdamW
  settings moves performance about as much as choosing the cooldown shape, and shapes that balance exploration and
  exploitation beat the alternatives. The abstract gives no beta2 values, sizes or token counts. The Porian et al.
  summary says beta2 tuning matters most at small batch sizes. A benchmarking study (arXiv 2509.01440, summary) found
  beta2 0.999 best at its scale and 0.9999 divergent, so the upper end is a stability limit.
- What changes here: the record's AdamW groups (embedding, head, scalars) run beta2 0.95, which averages the second
  moment over about 20 steps. At 0.99 the window is about 100 steps. Over the cooldown (about 860 steps) the change is
  gradual, so the step size of the Adam groups stays close to the record's until late. The schedule family computes
  beta2 from the clock. Search `beta2_end` from 0.95 to 0.99, with the cooldown start as the ramp's start.
- Expected cost: 1 hour (one schedule field and one parameter-group update per step). Compile: none if the betas are
  tensors or the optimizer step is outside the compiled graph. If a compiled step reads Python floats, a change will
  recompile, so check the code path before the first GPU run. `train.py` was not read for this scout.
- Gain estimate: 0 to -0.004 nats. The source gives the direction, not a size.
- Sketch (names follow `strategies/strategy.py` and the schedule family; the group field names are illustrative):

```python
def beta2(self, progress: float) -> float:        # schedule family field; the record's 0.95 before the cooldown
    cool_start = 1.0 - self.cfg.cooldown_fraction
    if progress < cool_start:
        return self.cfg.beta2
    frac = (progress - cool_start) / self.cfg.cooldown_fraction
    return self.cfg.beta2 + (self.cfg.beta2_end - self.cfg.beta2) * frac

def after_step(self, state: StepState) -> None:   # writes the Adam groups' beta2 once per step
    for group in self.adam_groups(state.optimizer):
        group["betas"] = (group["betas"][0], self.beta2(state.progress))
```

- Risk: the size is unknown and the abstract's gain may come from the shape interaction. A larger beta2 slows the
  final fit of the embedding (lr 3e-3); test `beta2_end` at 0.98 and 0.99 first.

## 2. Muon momentum end at 0.98, tapered back in the cooldown

- Source: Kosson, Messmer and Jaggi, "Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training," arXiv
  2410.23922 (NeurIPS 2024). Read: partial HTML (Section 3, Section 6.1 and the Figure 5 caption). Reference code:
  "Insights on Muon from Simple Quadratics," arXiv 2602.11948 (2026): search snippet only.
- Evidence: for LionAR, a rotational Lion variant (not Muon), on GPT-2 124M (batch 480 of 1,024 tokens, 5,000 steps,
  about 20 TPP), momentum beta 0.98 with Nesterov and an inverse early bias correction left no warmup as the best
  setting. The text says that "at higher momentum values LionAR no longer benefits from warmup". The text read gives no
  loss numbers for this result. The catalog's 023 (Muon momentum 0.95 from step 0) lost 0.004 nats against the warm
  start from 0.85, so the warm start is worth keeping. The 2602.11948 reference ramps momentum from 0.85 to 0.95 over the
  first 300 steps and tapers it at the end (summary; no ablation).
- What changes here: the schedule family already computes the momentum from the clock. Add `momentum_end` (0.95 to
  0.98) for the plateau and a taper back toward 0.95 through the cooldown. Momentum 0.98 averages about 50 steps of
  gradient against about 20 at 0.95, so it lags the cooldown's changes unless the taper ends the lag.
- Expected cost: 1 hour. Compile: none (a scalar; check that it is a tensor if the step is compiled). Step: 0.
- Gain estimate: 0 to -0.003 nats.
- Sketch (in the schedule family; the warm start is the record's):

```python
def momentum(self, progress: float) -> float:
    if progress < self.cfg.warm_end:                                   # the record's warm start, 0.85 to 0.95
        return 0.85 + (0.95 - 0.85) * progress / self.cfg.warm_end
    cool_start = 1.0 - self.cfg.cooldown_fraction
    if progress < cool_start:
        return self.cfg.momentum_end                                   # the plateau value, 0.95 to 0.98
    frac = (progress - cool_start) / self.cfg.cooldown_fraction
    return self.cfg.momentum_end + (self.cfg.momentum_taper - self.cfg.momentum_end) * frac
```

- Risk: the source's value is for a different optimizer at 20 TPP. The catalog's 023 says the warm start earns its
  keep, but nothing says the end value does.

## 3. Constant-timescale weight decay

- Sources: Bergsma et al., "Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training," arXiv
  2505.13738 (2025; revised November 2025). Read: partial HTML (the first part, through a
  summary). Wang and Aitchison, "How to set AdamW's weight decay as you scale model and dataset size," arXiv 2405.13698
  (2024): excerpt, as read for the catalog.
- Evidence (Power Lines): the optimal AdamW timescale, tau = 1 / (eta * lambda) in steps, stays roughly constant as the
  batch grows at fixed model and data (a 610M model at 20 TPP, batch 63 to 2,016), so the optimal lambda scales linearly
  with the batch up to about the critical batch. The timescale is computed at the peak LR, because the effective
  timescale changes as the LR decays. The study covers 20 to 1,280 TPP; the fitted exponent of tau in TPP is about
  -0.52. Our 0.3 to 0.7 TPP is below the studied range, so the law is an extrapolation.
- What changes here: at Muon lr 0.04 and lambda 0.1 (the catalog's cautious-decay value), tau is 1 / (0.04 x 0.1) = 250
  steps, about 12 percent of the run. A constant-timescale schedule sets lambda_t = 1 / (tau x lr_t), so the decay per
  step stays at 1 / tau through the cooldown while lambda rises as the LR falls. That is the opposite of a decay that
  shrinks with the LR. The catalog's 016 (constant cautious decay 0.1 on the Muon banks) lost 0.0033 nats, but that
  decay acted through the whole run. This form acts mostly in the cooldown, which 016 did not test.
- Expected cost: 1 hour (one schedule field). Compile: none if the decay factor is a tensor. Step: 0.
- Gain estimate: -0.004 to +0.002 nats. The sign is unknown; a 016-like loss is possible.
- Sketch (in the schedule family, with a cap so the rate does not blow up as the LR goes to zero):

```python
def weight_decay(self, lr_factor: float, peak_lr: float) -> float:   # lambda_t from the constant timescale
    lr_t = max(peak_lr * lr_factor, 1e-12)
    return min(1.0 / (self.cfg.tau_steps * lr_t), self.cfg.weight_decay_max)
```

- Risk: the fit is for AdamW and the catalog's Muon banks use their own update scale (the Moonlight convention in
  `speedrun_optimizers.md`). Start with the banks only, then the Adam groups.

## 4. muon_lr search range set by the token-horizon law (a search change, not a module)

- Source: Bjorck, Benhaim, Chaudhary, Wei and Song, "Scaling Optimal LR Across Token Horizons," arXiv 2409.19913 (2024;
  ICLR 2025). Read: partial HTML (the first part, through a summary). Essential AI, "Practical Efficiency of Muon for
  Pretraining," arXiv 2505.02222 (2025): abstract.
- Evidence: the optimal AdamW learning rate falls as a power of the token horizon, LR* proportional to D^(-beta), with
  beta = 0.32 recommended for models of 760M parameters and up (per-model fits 0.32 to 0.42 for 760M to 2.7B). The larger fits
  start at about 25B tokens (up to 200B for 760M and 1.3B, 100B for 2.7B). The paper finds the learning rate does not transfer across
  horizons under muP. For Llama-1 7B, the paper's extrapolated optimum at 1T tokens is about 1.1 to 1.15 x 10^-4 against
  the 3 x 10^-4 used, and its upper bound on the loss penalty at 100B tokens is 0.027 nats. Essential AI reports (abstract)
  pretraining up to 4B parameters, and (summary) that the muP scaling used for AdamW also works for Muon; it gives no
  rule for Muon's LR.
- What changes here: our horizon is 65 to 131 million tokens, about 200 times below the 25B start of the larger fits. Carried
  over with the -0.32 exponent, that is a factor of about 5 above the optimum at 25B, which is far outside any fit for
  Muon. The record's runs do not show a gain above 0.04: 002 moved the rate to 0.04 within noise of 001, and 018 (lr 0.06)
  lost from 270 s. The search range is 0.02 to 0.08. The source does not decide the range; it says the direction is up at short
  horizons, which the record has not confirmed.
- Expected cost: 0.5 hour (widen the search bound and add a run to the queue). Compile: none. Step: 0.
- Gain estimate: 0 nats; the value is information. Propose a search bound of 0.12 for one batch to see whether the
  search finds an optimum above 0.08. Do not move the record's lr on this basis.
- Sketch: a bound change in the search space (`muon_lr` from 0.02 to 0.12, log scale). No module.

## 5. Late gradient-norm correction (weight decay and normalization interaction)

- Source: Defazio, "Why Gradients Rapidly Increase Near the End of Training," arXiv 2506.02285 (2025). Read: abstract.
  The variant names (AdamC, SGDC) come from a search snippet and are not confirmed against the paper.
- Evidence: the abstract says gradient norms rise sharply near the end of long LLM runs because of "an unintended
  interaction between weight decay, normalization layers, and the learning rate schedule", and that a "simple correction"
  stops the rise and lowers loss throughout training. It gives no numbers, model sizes or token counts.
- What changes here: our model has QK-norm and RMS norms, and our decay is the search's `weight_decay` (default 0) on
  the Adam groups. The cooldown is 40 percent of the run. The source studies long runs, and a 2,150-step run may not show
  the rise at all. Check the gradient-norm trace of the record's existing runs (already logged; no new run) before
  building anything.
- Expected cost: 2 hours after the full text. Compile: none if the correction is an optimizer-side scale. Step: 0.
- Gain estimate: 0 to -0.003 nats, and near 0 if the norm does not rise at 2,150 steps.
- Sketch: depends on the correction in the full text; no sketch until then.

## 6. Warm-up-free or adaptive warm-up for Muon

- Sources: Riabinin, Veprikov, Bolatov, Takáč and Beznosikov, "Where Does Warm-Up Come From? Adaptive Scheduling for
  Norm-Constrained Optimizers," arXiv 2602.05813 (2026; NeurIPS 2026 poster). Read: abstract. Kosson et al., arXiv
  2410.23922 (2024): see idea 2.
- Evidence: Riabinin et al. derive warm-up followed by decay from a convergence proof under a generalized smoothness
  condition. Their practical scheduler sets the warm-up length at the start with standard hyperparameters, and "consistently
  outperforms or at least matches the best manually tuned warm-up schedules" on LLaMA pretraining. The abstract gives no
  numbers, model sizes or token counts. Kosson et al. find that warm-up mainly limits the early update size. An SNR-scaled
  update "acts similar to an automatic warmup" and "helps close the gap", but overall it is "slightly worse" (no value
  given). Their GPT-2 124M test used batch 480 and 5,000 steps (about 20 TPP).
- What changes here: the record's warm-up is 2 percent of the clock, about 43 steps. That is already short. The catalog
  has not run a zero warm-up on the Muon banks at the record's lr (023 tested momentum, not the LR), so the first run
  is 0 percent warm-up with everything else fixed, which is a single queue entry.
- Expected cost: 2 to 3 hours after the rule is read in the full text (the abstract does not state the warm-up length
  rule). Compile: none. Step: 0.
- Gain estimate: 0 to -0.002 nats. The record's short warm-up leaves little to take.
- Sketch: the schedule hook only, until the full text gives the rule:

```python
def lr_factor(self, progress: float) -> float:
    if self.cfg.warmup_fraction > 0 and progress < self.cfg.warmup_fraction:   # 0 is the first test
        return progress / self.cfg.warmup_fraction
    return self.trapezoid(progress)                                          # the record's plateau and cooldown
```

## 7. Depth-graded learning rate across the 36 blocks

- Sources: Wu, Zhang, Ma, Bian, Zhang, Chen and Lin, "Hyperparameter Transfer Laws for Non-Recurrent Multi-Path Neural
  Networks," arXiv 2602.07494 (2026). Read: abstract. "Optimal Scaling Needs Optimal Norm," arXiv 2510.03871 (2025; v2 January 2026). Read: abstract. Depth-muP (arXiv 2310.02244) is in the catalog.
- Evidence: Wu et al. argue, under stabilizing initializations and a maximal-update criterion, that the optimal learning
  rate decays with effective depth (the minimal path length, counting layers and residual additions) following a
  universal -3/2 power law, checked across CNNs, ResNets and Transformers. The abstract gives the exponent, not the
  prefactor or the training budgets. "Optimal Scaling Needs Optimal Norm" (Adam and Scion, up to 1.3B parameters and 138B
  tokens) reports that per-layer-group rates help, that the output layer is the most sensitive, and that hidden layers do
  better with lower rates. Neither source tests a rate that varies with block index inside one stack. Depth-muP's
  per-weight rate (catalog) is set by total depth, which is fixed here at 36.
- What changes here: the banked Muon applies one rate per bank, and each bank stacks one matrix shape across all blocks.
  A depth-graded rate therefore needs a per-slice multiplier, a broadcast tensor applied in `Muon.step`. That is cheap and
  does not change the graph. Test a linear taper of the Muon rate from 1.25 at block 0 to 0.75 at block 35 (and the
  reverse). Block 0 feeds the value residual, so its rate matters more.
- Expected cost: 1 to 2 hours. Compile: none (a tensor). Step: 0.
- Gain estimate: plus or minus 0.005 nats; the sign is unknown, and a tie is the likely result.
- Sketch (the multiplier is a buffer, so one graph):

```python
def parameter_groups(self, model):
    depth = torch.linspace(0.0, 1.0, num_blocks)                      # 0 at block 0, 1 at block 35
    self.block_scale = 1.0 + self.cfg.taper * (0.5 - depth)           # taper > 0: earlier blocks faster
    return self.muon_banks_with_slice_scale(model, self.block_scale)  # applied per slice in Muon.step
```

- Risk: the zero-initialized output projections mean a block's rate does not change step 0, so the early effect is
  small. Expect a tie unless the taper is large.

## 8. Token-per-step ramp: 32 windows early, 64 late

- Sources: Smith, Kindermans, Ying and Le, "Don't Decay the Learning Rate, Increase the Batch Size," arXiv 1711.00489
  (ICLR 2018). Read: abstract and summary. It reports that the same epochs give equivalent accuracy with fewer updates,
  and ResNet-50 on ImageNet to 76.1 percent in under 30 minutes. McCandlish, Kaplan, Amodei and the OpenAI Dota team,
  "An Empirical Model of Large-Batch Training," arXiv 1812.06162 (2018): the gradient noise scale predicts the largest
  useful batch and grows as the loss falls (summary). Zhang, Morwani, Vyas et al., "How Does Critical Batch Size Scale in
  Pre-training?," arXiv 2410.21676 (ICLR 2025): the critical batch size tracks data size more than model size (summary).
  Bergsma et al. (arXiv 2505.13738): B_crit proportional to D^0.47 (the fit exponent; the coefficient was not read).
  Zhou et al., "How to Set the Batch Size for Large-Scale Pre-training?," arXiv 2601.05034 (2026): a dynamic batch
  scheduler for WSD (abstract; no numbers). modded-nanogpt record 46, "Batch size schedule": 2.248 to 2.203 minutes, about
  2.0 percent of wall clock. The README table names only this change and states no gain.
- Evidence: the closest measure is record 46, about 2 percent of wall clock. By the yardstick that is about 0.0005 nats.
  The sources' gains are at fixed epochs on vision or at larger budgets.
- What changes here: the benchmark fixes 65,536 tokens a step, so a ramp that uses fewer tokens early changes the
  benchmark's count and needs Logan's word. Under one compiled shape a ramp can only mask windows, which saves no compute.
  A saving needs a smaller shape early: two compiled shapes (catalog 027: warmup 249 s, invalid) or one dynamic shape
  (the catalog's dynamic-shape pilot: 13 percent slower steps). Estimate with half-cost steps (0.14 s) for the first 30 percent
  of the clock at equal tokens: about 1,290 steps early and 1,500 late (0.28 s), against about 2,140 at the fixed size,
  a step ratio of 1.30. By the yardstick that is about 0.007 nats. A 13 percent slower step costs about 0.003, so the net is about
  -0.004 at best. The estimate assumes the per-step quality of a 32-window step holds, and small batches add noise.
- Expected cost: 4 to 6 hours including a compile test. Compile: two shapes invalid under the cap (027); one dynamic shape
  costs 13 percent a step. Step: -50 percent early in the first form, +13 percent in the dynamic form.
- Gain estimate: -0.002 nats net (range 0 to -0.004), and 0 if it needs two compiled shapes. Logan's word first.
- Sketch (the windows are a schedule value; the one-shape form needs a dynamic dimension, and the cap decides):

```python
def windows_for(self, progress: float) -> int:          # schedule family; the record's 64 after the switch
    return self.cfg.early_windows if progress < self.cfg.switch else 64
# one graph: pad to 64 windows and zero the unused ones (correct, no saving)
# two graphs: the saving, and the 027 cap failure
# one dynamic dimension: the saving less the dynamic overhead (13 percent in the catalog's pilot)
```

- Risk: the benchmark rule, the cap, and the per-step quality of small batches. Record 46 is the only batch schedule with a
  measured time, and it is about 0.0005 nats.

## 9. Polar Express iteration schedule: fewer iterations early

- Source: Amsel, Persson, Musco and Gower, "The Polar Express: Optimal Matrix Sign Methods and Their Application to the
  Muon Algorithm," arXiv 2505.16932 (2025; ICLR 2026). Read: partial HTML (Sections 1.3, 4.5, 5.1, 5.2 and Equation 17).
- Evidence: the paper runs five iterations in 16-bit bfloat (its "float16b") for Muon, with degree-5 polynomials per
  iteration; it recommends d = 5 and T = 5 or 6 (Section 4.5); Section 5.1 says that about 11 iterations give excellent
  accuracy, and that most GPT-2 weight matrices need more than 10 to converge in spectral norm. It gives no ablation of
  the iteration count against validation loss and no schedule over training.
- What changes here: the banked polar step is about 26 ms of the 276 ms step (catalog profile). Three iterations early
  instead of five saves about 10 ms, about 4 percent of a step, for the early part of the clock. Whether early Muon
  updates, which are noisy, lose anything from a rougher polar step is untested.
- Compile: the iteration count is a Python loop bound, so a switch recompiles, and identity coefficients keep one graph
  but save nothing. Two graphs are the cap risk again.
- Expected cost: 2 hours. Step: about -4 percent early, about -2 percent over the run (estimate).
- Gain estimate: 0 to -0.001 nats (2 percent of steps is about 0.0005 nats by the yardstick).

## Checked and rejected

- Cooldown shapes beyond 1-sqrt: `one_sqrt` (Hägele et al.) is in the catalog. Dremov et al. say shapes that balance
  exploration and exploitation win, without naming a shape in the abstract, so nothing new. Schaipp, Hägele, Taylor,
  Şimşekli and Bach, "The Surprising Agreement Between Convex Optimization Theory and Learning-Rate Scheduling for Large
  Model Training," arXiv 2501.18965 (ICML 2025): the summary says a convex bound reproduces the cosine and WSD match and
  the cooldown's necessity and optimal length. It adds no shape to test.
- Cooldown fraction: the search range (0.25 to 0.65) covers modded-nanogpt's record 26, which raised the cooldown
  fraction to 0.45 alongside another change (about 1.1 percent of wall clock), and record 19, which set the final LR to 0.1 instead of 0.0. A summary of
  Dremov et al. says cooldowns of about 20 percent are usual (not checked in the paper). No change to the range.
- Warm restarts and cyclic schedules inside 2,000 steps: SGDR (Loshchilov and Hutter, arXiv 1608.03983, ICLR 2017) reports
  2 to 4 times fewer epochs on CIFAR (vision; summary), and notes that the restarts "temporarily hurt performance". WSD-S
  (arXiv 2410.05192) reuses decayed checkpoints to serve several budgets from one run; it does not help a single final
  checkpoint like ours. No short-budget LM restart study was found. Reject.
- Power Scheduler, Shen et al., arXiv 2408.13359 (2024): an LR set from the tokens seen and the batch, with no total step
  count. Our clock fixes the budget, so it adds nothing here. Its 3B results are not compared with cosine at a short
  budget (summary). Reject for this benchmark.
- Sequence-length warmup, Li, Zhang and He, arXiv 2108.06084 (NeurIPS 2022): up to 2.2 times fewer tokens and 3.7 times less
  wall clock at matched zero-shot results, for GPT-2 with 8 times the batch and 4 times the LR (abstract). The catalog's
  window family, records 13, 29 and 72, and its 009 and 027 tests cover it. Not added.
- Embedding learning rate in the large-vocabulary regime, Hayou and Liu, arXiv 2506.15025 (2025): the optimal embedding-to-hidden
  rate ratio scales as the square root of width in a large-vocabulary regime, and as width under muP (abstract). Our
  vocabulary is about 25 tokens, far from large, so the muP prior is the relevant one, and `embed_lr` is already searched
  (catalog 017 and 022). The "300 times" rule from the noise code was not read here and is not a source, so it is not
  adopted.
- Gradient-grouped learning rates, arXiv 2506.01049 (2025): per-parameter-group scaling from gradient statistics. It is
  not a schedule, and the abstract gives no numbers. Reject for now.
- Gradient-norm or loss-driven global schedules: no source found that schedules the global rate from the gradient norm or
  the loss, with numbers. The closest was idea 5, which is a weight-decay correction. Reject as unsupported.
- Muon learning rate against batch size: Sato and Naganuma, arXiv 2507.01598 (2025): Muon's critical batch size is below
  AdamW's on ResNet-18 and the learning rate and weight decay must satisfy eta at most 1 / lambda (summary). Essential AI
  (idea 4) gives no rule for the rate. No Muon law was found. Used as context for idea 8 only.
- Adam epsilon: the NVIDIA Emerging Optimizers primer says 1e-8 is not negligible against second moments in modern LLMs
  (summary). No short-run LM number was found. Not ranked; a zero-cost check of the scalar groups at a smaller epsilon is
  possible if wanted.
- Weight decay on the Muon banks, constant: the catalog's 016 lost 0.0033 nats, and idea 3 differs only in when the decay
  acts.
- Momentum as a warm start: covered by the catalog's `momentum_start` search and 023.
- Schedule-free averaging and AMUSE: being added separately, so excluded here.
- Straight to Zero (arXiv 2502.15938) and WSD's river-valley account (arXiv 2410.05192): the linear decay to zero is already
  the record's shape (`final_lr_ratio` 0.0), and the river-valley account is context, not a schedule.
- Kimi K2 and Moonlight update-RMS rules: covered in `speedrun_optimizers.md` (idea 6 there).

## Gaps and read status

- Read in partial form (the first 100,000 characters of the HTML, returned through a summary; no table was checked against
  the PDF): Kosson et al., Power Lines, Bjorck et al., Polar Express, the modded-nanogpt repository page.
- Abstract only: Riabinin et al., Defazio, Dremov et al., Wu et al., "Optimal Scaling Needs Optimal Norm," Essential AI, Zhou
  et al., Smith et al., Hayou and Liu, the Porian et al. abstract, arXiv 2509.01440, arXiv 2506.01049.
- Search summary or snippet only: McCandlish et al., Zhang et al. (critical batch size), Schaipp et al., Power Scheduler,
  SGDR, Sato and Naganuma, the NVIDIA primer, arXiv 2602.11948 (reference code), the Porian et al. beta2 claim and the
  Dremov cooldown fraction. Their numbers are not verified here.
- Not found: a Muon learning-rate law against batch or horizon; a Muon momentum-schedule ablation; a per-block-index
  learning-rate study in LM pretraining; a short-budget (under 2,000 steps) LM study of restarts.
- Every source number is at 20 TPP or more or on vision or BERT, except the modded-nanogpt times. Our 0.3 to 0.7 TPP is below
  every fit here. The conversions use the catalog's slope, which is rough.

## Sources

- Smith et al., Don't Decay the Learning Rate, Increase the Batch Size: https://arxiv.org/abs/1711.00489
- McCandlish et al., An Empirical Model of Large-Batch Training: https://arxiv.org/abs/1812.06162
- Zhang et al., How Does Critical Batch Size Scale in Pre-training?: https://arxiv.org/abs/2410.21676
- Bergsma et al., Power Lines: https://arxiv.org/abs/2505.13738
- Zhou et al., How to Set the Batch Size for Large-Scale Pre-training?: https://arxiv.org/abs/2601.05034
- Li et al., The Stability-Efficiency Dilemma (sequence length warmup): https://arxiv.org/abs/2108.06084
- Wen et al., Understanding Warmup-Stable-Decay Learning Rates (WSD, WSD-S): https://arxiv.org/abs/2410.05192
- MiniCPM (WSD scheduler): https://arxiv.org/abs/2404.06395
- Schaipp et al., The Surprising Agreement Between Convex Optimization Theory and Learning-Rate Scheduling: https://arxiv.org/abs/2501.18965
- Shen et al., Power Scheduler: https://arxiv.org/abs/2408.13359
- Dremov et al., Training Dynamics of the Cooldown Stage in WSD: https://arxiv.org/abs/2508.01483
- Porian et al., Resolving Discrepancies in Compute-Optimal Scaling of Language Models: https://arxiv.org/abs/2406.19146
- Benchmarking optimizers for LLM pretraining: https://arxiv.org/abs/2509.01440
- Kosson et al., Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training: https://arxiv.org/abs/2410.23922
- Insights on Muon from Simple Quadratics (reference momentum ramp): https://arxiv.org/abs/2602.11948
- Riabinin et al., Where Does Warm-Up Come From? Adaptive Scheduling for Norm-Constrained Optimizers: https://arxiv.org/abs/2602.05813
- Amsel et al., The Polar Express: https://arxiv.org/abs/2505.16932
- Bjorck et al., Scaling Optimal LR Across Token Horizons: https://arxiv.org/abs/2409.19913
- Essential AI, Practical Efficiency of Muon for Pretraining: https://arxiv.org/abs/2505.02222
- Sato and Naganuma, Convergence Bound and Critical Batch Size of Muon: https://arxiv.org/abs/2507.01598
- Hayou and Liu, Optimal Embedding Learning Rate in LLMs: https://arxiv.org/abs/2506.15025
- Wu et al., Hyperparameter Transfer Laws for Non-Recurrent Multi-Path Neural Networks: https://arxiv.org/abs/2602.07494
- Optimal Scaling Needs Optimal Norm: https://arxiv.org/abs/2510.03871
- Defazio, Why Gradients Rapidly Increase Near the End of Training: https://arxiv.org/abs/2506.02285
- Gradient grouping (SGG): https://arxiv.org/abs/2506.01049
- Loshchilov and Hutter, SGDR: https://arxiv.org/abs/1608.03983
- Wang and Aitchison, AdamW weight decay timescale: https://arxiv.org/abs/2405.13698
- Straight to Zero: https://arxiv.org/abs/2502.15938
- modded-nanogpt records and README: https://github.com/KellerJordan/modded-nanogpt
