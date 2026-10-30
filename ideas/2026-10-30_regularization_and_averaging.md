# Regularization, noise and weight averaging for the 10-minute speedrun, 2023 to 2026

Date: 2026-10-30. A literature scout for SpeedrunningPLMs (198M parameters, 36 blocks, 768 stream, about 2,100 steps of
65,536 tokens in 600 s on one GH200; test cross entropy on masked residues). Read-only: no code was run and nothing here
is measured on this benchmark. Each gain is a source's number or our estimate, and the text says which.
Read first: `ideas/README.md`, `ideas/2026-10-30_speedrun_optimizers.md`, `ideas/2026-10-30_init_and_parametrization.md`,
and by search `ideas/2026-10-30_mlm_pretraining.md` and `ideas/modded_nanogpt_2026.md` (dropout, label smoothing, z-loss,
Schedule-Free and tail averaging are already there and are not repeated as ideas).

Shorthand: CE is cross entropy in nats; TPP is tokens per parameter; LAWA is LAtest Weight Averaging; EWA is early weight
averaging; PMA is pre-trained model averaging; SAM is sharpness-aware minimization; SR is stochastic rounding; PLD is
progressive layer dropping; LR is learning rate. One seed resolves about 0.0015 nats (README).

## Outcome

- Eight ideas, ranked by expected loss gain per implementation hour. The top two (latest-k averaging and a 1-sqrt
  cooldown) are one-hour changes that leave training untouched. Expected gains are small, most under 0.005 nats.
- Every number in the averaging and cooldown sources comes from runs at 70 to 460 billion tokens or 20 to 100 TPP.
  Our run sees 0.3 to 0.7 TPP. Discount every gain here for that. The one short-budget result in this scope is the
  2020 PLD wall-clock claim, and it is for BERT, not a CE number.
- The strongest prior in our own record: EMA (catalog 006, decay 0.995 from 60 percent of the clock) led mid-cooldown
  and lagged at the end, retired at +0.0041. Averaging the tail therefore helps only if it stays short. Idea 1 is the
  uniform window of about 500 steps, not a shorter EMA; the README's untried item is a smaller EMA decay.
- Two sources that look new are already in the catalog and are not repeated: Schedule-Free AdamW
  (`architecture_optimization.md`, rank 10) and z-loss (`architecture_optimization.md`, `mlm_pretraining.md`).
- Dropout, label smoothing and tail averaging are covered in `mlm_pretraining.md` and `modded_nanogpt_2026.md`. Idea 6
  is the early-only dropout form, which those files do not cover.

## Ranked table

Gain is the expected change in test CE on this benchmark (our estimate; negative is better). "Read" says how much of the
source was read. Step cost and compile cost are estimates against the record's profile (276 ms a step, 106 to 130 s
compile, 180 s warmup cap).

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats) | Step cost | Evidence strength |
|---|---|---|---|---|---|---|---|
| 1 | Uniform latest-k averaging over the late run, scored at evaluation only (LAWA, uniform window) | arXiv 2209.14981 (2022); arXiv 2306.03241 (2023) | Figure 1 caption and setup (HTML) | 0.5 | 0 to -0.005 | none; one copy every 100 steps | moderate at 72 TPP (GPT-2 small: 2.963 to 2.917); out of regime |
| 2 | 1-sqrt cooldown in the existing trapezoid | arXiv 2405.18392 (2024, NeurIPS) | HTML, Tables 5 and 6 | under 1 | 0 to -0.003 | none | weak: +0.03 and +0.07 benchmark points over linear |
| 3 | Plateau averaging: average the high-LR checkpoints and continue from the average (EWA, PMA) | arXiv 2306.03241 (2023); arXiv 2505.12082 (2025) | abstracts and captions; protocol not read | 1 to 2 | 0 to -0.01 | none | moderate at 70 TPP; the protocol decides which variant this is |
| 4 | SAM every 4 steps in the final 40 percent of the clock | arXiv 2605.02105 (2026); arXiv 2110.08529 (2022) | abstract only | 2 | -0.002 to +0.003 | +10 percent GPU time (about 210 steps) | weak: forgetting, not CE |
| 5 | bf16 weights with stochastic rounding in the optimizer update | arXiv 2010.06192 (2020); arXiv 2405.03637 (2024) | abstracts only | 2 to 3 | 0 to -0.001 | -1 to -3 percent | moderate for training accuracy; the record is fp32 where it matters |
| 6 | Early-only dropout (p 0.1 for the first 10 to 20 percent of the clock) | arXiv 2303.01500 (2023, ICML) | abstract and search summary | 1 | 0 to +0.005 | +2 to 4 percent for the whole run unless the mask is skipped | weak: vision; single-pass MLM says no dropout |
| 7 | LookSAM: one extra gradient every 5 steps over the whole run | arXiv 2203.02714 (2022) | search summary | 2 | +0.000 to +0.010 | +20 percent GPU time (about 420 steps) | weak: vision only |
| 8 | Progressive layer dropping (keep rate falls with depth and the clock) | arXiv 2010.13369 (2020, NeurIPS) | abstract (two versions) | 4 to 6 | 0 to -0.010 | fewer FLOPs, but a graph per pattern | moderate for BERT wall-clock; graph cost likely breaks the warmup cap |

## 1. Latest-k averaging over the late run, scored and never trained on

- Sources: Kaddour, "Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging,"
  arXiv 2209.14981 (2022); Sanyal, Neerkaje, Kaddour, Kumar and Sanghavi, "Early Weight Averaging meets High Learning Rates
  for LLM Pre-training," arXiv 2306.03241 (2023; NeurIPS 2023 workshop). Read: Figure 1 caption and experimental setup
  from the HTML (v2); abstract of 2209.14981; the 2022 paper's masked LM result (RoBERTa-base, WikiText-103) by search summary.
- Evidence (2306.03241, Figure 1 caption, validation loss at 70K steps on OpenWebText):

  | Model | Original | EMA (decay 0.9, each step) | SWA (from 75 percent, every 10 steps) | LAWA, best setting |
  |---|---|---|---|---|
  | GPT-2 small, 125M | 2.963 | 2.949 | 2.952 | 2.917 |
  | GPT-2 medium, 335M (caption says 355M) | 2.855 | 2.845 | 2.837 | 2.819 |
  | GPT-2 large, 770M | 2.977 | 2.968 | 2.961 | 2.908 |

  LAWA used a window of k = 5 checkpoints spaced 200 or 1K steps apart; the text does not say which spacing gave the
  "best" row. OpenWebText is about 9B tokens (the dataset size in the abstract; the trained token count was not read), so
  the regime is far above our 0.3 to 0.7 TPP.
- What changes here: the record already anneals to zero, so the last checkpoints differ little and the averaged model
  may gain little. The catalog's EMA (006, decay 0.995) has a memory of about 200 steps and lagged at the end. A uniform window of
  five checkpoints 100 steps apart spans 500 steps, about 24 percent of the run, so it is a longer memory than 006,
  not the shorter one the README lists as untried (a smaller EMA decay). The
  modded-nanogpt tail average (record 92c) is a lerp blend with rates per shard; it is not separable there and not copied.
- Selection: the benchmark selects by validation (`val_select`) and reports test. Score the raw final weights and the
  averaged weights on validation, keep the lower, and report test beside both. Never ship the average without that
  comparison (the rule in `modded_nanogpt_2026.md` 92c).
- Expected cost: 0.5 hour. Compile: 0 s. Step: one device copy of 198M parameters every 100 steps, under 1 ms on GH200
  (estimate). Memory: five fp32 copies, about 4 GB (estimate from 198M times 4 bytes). Evaluation: one extra validation
  pass when the averaged model is scored, outside the clock but capped.
- Sketch (hooks `after_step` and `evaluation_weights` as in `strategies/strategy.py`; `snapshot` and `load_mean` are
  helpers to write, no existing one was read):

```python
class LatestAverage(Strategy):  # family "averaging"; variant "latest_k" keeps the last k snapshots after `start`
    family = "averaging"
    class Config(StrategyConfig):
        VARIANTS = ("none", "latest_k"); variant: str = "none"; k: int = 5; spacing: int = 100; start: float = 0.6
        BOUNDS = {"k": (2, 8), "spacing": (25, 400), "start": (0.4, 0.95)}
    def after_step(self, state: StepState) -> None:      # state.progress is the clock share; snapshot after the update
        if self.active and state.progress >= self.cfg.start and (state.step + 1) % self.cfg.spacing == 0:
            self.snaps = (self.snaps + [snapshot(state.model)])[-self.cfg.k:]
    @contextlib.contextmanager
    def evaluation_weights(self, model: nn.Module) -> Iterator[None]:
        if not self.snaps:
            yield; return
        live = snapshot(model); load_mean(model, self.snaps)    # uniform mean, fp32
        try: yield
        finally: load_state(model, live)                         # the live weights return after the pass
```

- Risk: the gain is near the band (one seed resolves about 0.0015 nats), and the sources' gains are at 70 TPP. The EMA
  lag in 006 says averaging into the cooldown's end can cost. Test with k and spacing searched, `start` at 0.6 and 0.8.
  Keep the control (raw weights) in every run's record.

## 2. A 1-sqrt cooldown in the existing trapezoid

- Source: Hägele et al., "Scaling Laws and Compute-Optimal Training Beyond Fixed
  Training Durations," arXiv 2405.18392 (NeurIPS 2024 spotlight). Read: HTML v3, Tables 5 and 6 (aggregated benchmark
  scores, not loss), the text of Section 3.2.
- Evidence (benchmark average, 1B model, FineWeb):
  - 100B tokens (Table 5): cosine to 0 45.88; cosine to 10 percent 46.26; 1-sqrt 20 percent cooldown 46.23; linear 20
    percent 46.20. The 1-sqrt minus linear difference is +0.03 points.
  - 460B tokens (Table 6): cosine to 0 48.03; 1-sqrt 5 percent 47.91; linear 5 percent 47.84; linear 10 percent 47.98;
    linear 20 percent 47.92. The difference is +0.07 points. The caption says longer cooldowns do not necessarily improve.
  - Section 3.2 says 1-sqrt beats linear more as runs get longer (Figure 4, no values in the text).
  These are benchmark points, not loss, and the gap is inside benchmark noise, so the evidence for CE is weak.
- What changes here: `schedule.py` has only the linear cooldown (`lr_scale` returns 1 minus the phase). The 1-sqrt shape
  falls fast at the start of the cooldown and flattens to zero at its end: at half the cooldown it is at 0.29 of the
  peak, against 0.5 for linear. The catalog's `cooldown_fraction` search has a code bound of 0.1 to 0.8 (`schedule.py`
  BOUNDS), while `ideas/README.md` says 0.25 to 0.65 was searched; the 5 to 10 percent cooldowns of the source are below
  both. Check which range the search uses before reading the screening effects.
- Expected cost: under 1 hour. Compile 0 s. Step 0. Memory 0.
- Sketch (one variant in `schedule.py`, a new `shape` field; hook `lr_scale` on the schedule family):

```python
def lr_scale(self, progress: float) -> float:            # variant "sqrt_cooldown"; the record's linear branch unchanged
    if progress > 1 - cf:
        phase = (progress - (1 - cf)) / cf                # 0 at the cooldown's start, 1 at the end
        shape = math.sqrt(phase) if self.config.shape == "sqrt" else phase
        return 1 - (1 - self.config.final_lr_ratio) * shape
    return ...                                            # warmup and plateau unchanged
```

- Risk: the effect is likely inside the band, so expect a tie. The record stays linear until a genome is promoted.

## 3. Plateau averaging: average the high-LR checkpoints, then continue (EWA, PMA)

- Sources: Sanyal et al., arXiv 2306.03241 (2023): early averaging while LR is high speeds convergence, and the gain grows
  with checkpoint spacing; it compares EMA and SWA and uses GPT-2 125M to 770M on OpenWebText plus Pythia 1B to 12B. Read:
  Figure 1 caption and the abstract (see idea 1 for the table). Li et al., "Model Merging in Pre-training of Large Language
  Models," arXiv 2505.12082 (2025), PMA: merging checkpoints trained at a constant LR gives gains and predicts the annealed
  result; scales from millions to over 100B parameters. Read: abstract and one summary. The summary says merging about 10
  checkpoints is best up to 70B; this was not verified in the paper.
- Evidence: the EWA rows in idea 1 (2.963 to 2.917 at 125M). EWA's claim is about convergence in fewer steps, which is the
  clock we measure. PMA's numbers were not read.
- What changes here: our plateau runs from about 2 percent to 60 percent of the clock (trapezoid), roughly 1,200 steps at a
  constant LR (the record's). An average of plateau checkpoints scored at the end is a mid-run point, so the scoring-only
  form of idea 1 cannot reach the sources' gain. The variant the sources imply is training-side: at the end of the plateau,
  replace the live weights with the average of the last k plateau snapshots and continue the cooldown from there. The
  protocol (whether EWA resets or re-enters training) is in 2306.03241, Section 3, which was not read. Read it before
  building the training-side form.
- Expected cost: 1 to 2 hours after the protocol is read. Compile 0 s. Step 0. Memory: k snapshots (idea 1's cost).
- Sketch (a training-side variant; `after_step` replaces the weights once, at the plateau's end):

```python
def after_step(self, state: StepState) -> None:
    if not self.done and state.progress >= self.cfg.plateau_end:   # 0.6 of the clock, the cooldown's start
        load_mean(state.model, self.snaps)                           # continue from the plateau average
        self.done = True
    elif not self.done and (state.step + 1) % self.cfg.spacing == 0 and state.progress >= self.cfg.start:
        self.snaps = (self.snaps + [snapshot(state.model)])[-self.cfg.k:]
```

- Risk: the average of high-LR snapshots sits in a flatter, possibly higher-loss region; a restart at 0.6 costs the
  cooldown's first steps. Two variants, "replace" and "score only", should be run separately and never stacked with idea 1.
  The gain at 70 TPP may be gone at 0.5 TPP.

## 4. SAM in the final 40 percent, one extra gradient every 4 steps

- Sources: "Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting," arXiv 2605.02105 (May 2026; authors not read). Read:
  abstract only. Small models of 20M to 150M parameters and an OLMo-2-1B checkpoint; a short SAM mid-training phase cut
  forgetting by 31 percent after MetaMath post-training and by 40 percent after 4-bit quantization. These are forgetting
  numbers, not CE. Bahri, Mobahi and Tay, "Sharpness-aware minimization improves language model generalization," arXiv
  2110.08529: SAM on T5 fine-tuning; numbers not read. Liu et al., "Towards Efficient and Scalable Sharpness-Aware
  Minimization" (LookSAM), arXiv 2203.02714 (CVPR 2022): see idea 7.
- Evidence on CE: none. The 2026 preprint's summary says full SAM pretraining roughly doubles training time; this is a
  search summary and was not checked in the body.
- What changes here: SAM needs a second forward and backward at the perturbed weights. Every 4 steps in the last 40 percent
  is one extra step per 4, about 840 / 4 = 210 extra steps of 0.276 s, which is about 58 s or 10 percent of the clock
  (estimate). The second pass has the same shapes, so the graph is reused and compile does not change. The extra cost is
  about 210 steps, and any gain has to exceed that loss.
- Expected cost: 2 hours. Compile 0 s. Step: +10 percent of the clock. Memory: one fp32 copy of the gradients, 0.8 GB.
- Sketch (uses a grad-enabled loss closure that the loop does not yet expose; `StepState` has `loss_without_grad`):

```python
def after_backward(self, state: StepState) -> None:      # SAM gradient on every k-th step of the final phase
    if state.progress < self.cfg.start or state.step % self.cfg.k: return
    grads = [p.grad.detach().clone() for p in self.params]               # the gradient at w, kept
    norm = torch.sqrt(sum(g.float().square().sum() for g in grads)) + 1e-12
    eps = [self.cfg.rho * g / norm for g in grads]                       # ascent step, global norm
    with torch.no_grad():
        for p, e in zip(self.params, eps): p.add_(e)
    for p in self.params: p.grad = None
    state.loss_with_grad().backward()                                    # gradient at w + eps (loop must provide)
    with torch.no_grad():
        for p, e in zip(self.params, eps): p.sub_(e)                     # restore w; the step uses the SAM gradient
```

- Risk: rho (0.05 is a vision default, not an LM value) must be scaled to the Muon update's RMS, which is not the Adam
  scale; the effect on CE is unknown in sign; and the sources measure forgetting, not pretraining loss. Expect the 10
  percent step loss to dominate unless the gain is large.

## 5. Stochastic rounding for bf16 weights in the optimizer update

- Sources: Zamirai et al., "Revisiting BFloat16 Training," arXiv 2010.06192 (2020). Read: abstract.
  Stochastic rounding and Kahan summation "enable up to 7% absolute validation accuracy gain in 16-bit-FPU training" over
  naive 16-bit, and reach "0.1% lower to 0.2% higher validation accuracy compared to 32-bit training across seven deep
  learning applications". The PDF tables could not be read (binary in the fetch). Collage, Yu et al., arXiv 2405.03637
  (ICML 2024). Read: text. Up to 3.7 times training throughput against FP32 master weights for GPT-6.7B and OpenLLaMA-7B
  (their claim); peak memory 22.8 percent lower for GPT 125M to 6.7B; RoBERTa-base GLUE within 0.85 percent of the best
  baseline.
- What changes here: `optimizer.py` keeps the Muon momentum in fp32 (lines 107 and 108) and the polar step runs in bf16
  (`muon.py`). I did not read `train.py` for the dtype of the weights themselves, so whether the weight update is already
  fp32 is unverified; check it first. The banked Muon is 26 ms of the 276 ms step, so the best step saving is 1 to 3
  percent (estimate). A 1 to 3 percent step gain is 20 to 60 steps, inside one seed's band (our estimate), so the CE gain is 0 to -0.001.
- Expected cost: 2 to 3 hours. Compile 0 s. Step: -1 to -3 percent (estimate). Memory: weights in bf16 halve their
  storage, which the record does not need.
- Sketch (inside the update in `optimizer.py`, not a hook; the rounding is a bit operation that keeps the bf16 mantissa):

```python
def sr_bf16(x: Tensor) -> Tensor:                         # stochastic rounding fp32 to bf16 (Zamirai et al. 2020)
    bits = x.view(torch.int32)
    noise = torch.randint_like(bits, 0, 1 << 16)           # uniform low 16 bits
    return ((bits + noise) & ~0xFFFF).view(torch.float32).bfloat16()

# in the update: p.copy_(sr_bf16(p.float() + delta))       # delta: the fp32 update of this step
```

- Risk: the gain is speed, not loss, and a bf16 weight path with nearest rounding loses accuracy (the 7 percent gap). The
  sketch needs a CPU check against fp32 on random tensors before any run.

## 6. Early-only dropout

- Source: Liu, Xu, Jin, Shen and Darrell, "Dropout Reduces Underfitting," arXiv 2303.01500 (ICML 2023, PMLR 202:22233 to
  22248). Read: abstract and search summary. Early dropout lowers the final training loss against no dropout and against
  standard dropout, and the authors attribute it to lower gradient variance across mini-batches early on. The evaluations
  are image classification and downstream tasks; no LM numbers were read.
- Counter-evidence: Drop Dropout, arXiv 2505.24788 (ACL Findings 2025), reports that no dropout gave lower MLM loss in a
  single-pass BERT-base run (Figure 1 only, per `mlm_pretraining.md`). That is the closest regime to ours.
- What changes here: our run is single-pass and underfit, the regime where early dropout's mechanism applies, but the only
  MLM evidence says none. The change is a dropout on the attention output and the MLP hidden state for the first 10 to 20
  percent of the clock, at p = 0.1. The rate must be a tensor buffer: a Python float that changes would recompile the
  graph. The random mask is also drawn every step, so the step cost stays for the whole run unless the mask is skipped
  when the rate is zero (a Python branch would recompile). Keep both costs in mind.
- Expected cost: 1 hour. Compile 0 s if the rate is a buffer. Step: +2 to 4 percent for the whole run (estimate).
- Sketch (hook `mlp_hidden`, with the rate set in `after_step` from the clock share):

```python
class EarlyDropout(Strategy):  # family "dropout"; variant "early" applies p until `until` of the clock
    family = "dropout"
    class Config(StrategyConfig):
        VARIANTS = ("none", "early"); variant: str = "none"; p: float = 0.1; until: float = 0.15
        BOUNDS = {"p": (0.02, 0.2), "until": (0.05, 0.3)}
    def build(self, model: nn.Module, max_tokens: int) -> None:
        self.rate = torch.zeros((), device=model_device(model))      # a buffer: one compiled graph for both phases
    def after_step(self, state: StepState) -> None:
        self.rate.fill_(self.cfg.p if state.progress < self.cfg.until else 0.0)
    def mlp_hidden(self, index: int, H: Tensor) -> Tensor:
        keep = (torch.rand_like(H) >= self.rate).to(H.dtype)
        return H * keep / (1 - self.rate)
```

- Risk: expected sign is unknown and may be positive (harm). The only MLM result says no dropout. Run once as a control
  against the record's seed band before any search.

## 7. LookSAM over the whole run (one extra gradient every 5 steps)

- Source: Liu et al., "Towards Efficient and Scalable Sharpness-Aware Minimization,"
  arXiv 2203.02714 (CVPR 2022). Read: search summary. On ViT-B/16, LookSAM with k = 5 matched SAM's 79.8 percent top-1
  at 68.6 s an epoch against 103.1 s for SAM (summary); the gap to naive periodic SAM widens with model size (summary).
  No LM result was found; the search for LM LookSAM found none.
- What changes here: a full-run schedule costs one extra gradient every 5 steps, +20 percent GPU time, about 400 fewer
  steps at 0.276 s (estimate). That loss is about 20 percent of steps, which the catalog's step sensitivity (008: 1.8 times
  the steps, behind from 180 s) suggests is larger than any SAM gain we could expect.
- Expected cost: 2 hours. Compile 0 s. Step +20 percent. Memory: one gradient copy, 0.8 GB.
- Sketch: idea 4's `after_backward` with `start = 0.0` and `k = 5`; no other change.
- Risk: the vision result is not an LM result; the step cost is larger than the one we would pay in idea 4. Rank last
  among the SAM forms and expect a loss.

## 8. Progressive layer dropping (keep rate falls with depth and the clock)

- Source: Zhang and He, "Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping,"
  arXiv 2010.13369 (NeurIPS 2020). Read: arXiv abstract and the NeurIPS listing. The arXiv abstract reports a 24 percent
  reduction in average per-sample wall-clock time and a 2.5 times speedup to reach a similar downstream accuracy on BERT
  pretraining; the NeurIPS listing reports a 25 percent FLOP reduction and a 24 percent wall-clock reduction with GLUE 1.1
  points higher at the same sample count. No CE number was read. Their keep schedule is theta(t) = (1 - theta_bar) exp(-gamma
  t) + theta_bar with a depth-scaled keep rate; the values of theta_bar and gamma were not read.
- What changes here: a dropped block is a different computation, so under `torch.compile` each drop pattern is a new graph.
  The record's compile is 106 to 130 s against the 180 s warmup cap, and the catalog's own two-graph attempt (027) reached
  249 s and went invalid. A mask that keeps one graph (multiply the block output by zero) saves no FLOPs and gives no
  speedup, so only a small fixed set of patterns could fit the cap, and none was measured.
- Expected cost: 4 to 6 hours including the compile test. Step: -20 to -25 percent in eager BERT (the paper), unknown here.
  Compile: a graph per pattern (likely invalid).
- Sketch (the schedule only; the block skip itself needs one graph per pattern, which is why this is last):

```python
def keep_rates(self, progress: float, n_blocks: int) -> list[float]:   # PLD's depth-scaled keep probabilities
    theta = (1 - self.cfg.theta_bar) * math.exp(-self.cfg.gamma * progress) + self.cfg.theta_bar
    return [1 - (i + 1) / n_blocks * (1 - theta) for i in range(n_blocks)]   # block 36 keeps the least
```

- Risk: the paper's time saving is for eager BERT on other hardware; the step saving may vanish under the fused record
  kernels; and the compile cap probably blocks it. Only worth a test if a single fixed pattern is possible.

## Checked, not ideas

- Label smoothing: Müller, Kornblith and Hinton, "When Does Label Smoothing Help?," arXiv 1906.02629 (NeurIPS 2019). Read:
  summaries and slides. It improves calibration and translation, and the authors find it impairs distillation. A smoothed
  target cannot reach the CE minimum on the true labels, and the score here is plain CE on masked residues. Already
  rejected in `mlm_pretraining.md`; not repeated.
- Z-loss: covered by the PaLM result in `architecture_optimization.md`, and rejected in `mlm_pretraining.md` as an
  objective change.
- Schedule-Free AdamW: Defazio et al., "The Road Less Scheduled," arXiv 2405.15682 (NeurIPS 2024 oral). The GPT-2 124M
  OpenWebText figure is said to match or exceed cosine (caption; no numbers read). Listed in
  `architecture_optimization.md` (rank 10). Muon compatibility is unknown.
- Tail averaging with blends (modded-nanogpt record 92c): covered in `modded_nanogpt_2026.md`. Idea 1 is the uniform,
  evaluation-only form; the record's blend is not separable and is not copied.
- Clipping under Muon: the polar step is invariant to the gradient's scale, so clipping the raw gradient changes only the
  relative weight of steps inside the momentum, not the direction of one step. The only source is a DeepSpeed bug report
  (issue 8439), and it states the no-op argument without a loss result. MuCon (arXiv 2605.26459) clips singular values
  with no LM result, and it is the same family as Musec in `speedrun_optimizers.md` (rank 12). Not proposed. If the record
  ever spikes, clip the Adam group only, and read `train.py`'s current clipping first.
- Weight noise, gradient-noise injection and gradient-norm penalties (GAM): no LM pretraining CE number was found. The
  closest source (arXiv 2605.02105) reports forgetting. Not proposed.
- Stochastic depth as LayerDrop (Fan et al., 2019): a pruning regularizer applied at inference for smaller models; no
  pretraining speed or CE number was read. The training-speed claim is PLD (idea 8).
- Dropout on the whole run, dropout on the embedding, and token dropout: covered by `mlm_pretraining.md` (Drop Dropout;
  ESM-2 token dropout, untested and small).
- Muon momentum and cooldown interactions with averaging: covered by the catalog's `ema` family and by catalog run 019.
- MuCon and singular-value clipping: see Muon clipping above.
- The PMA "about 10 checkpoints" rule: a summary claim, not verified. Idea 3 says so.

## Gaps and limits

- Not read in full: 2306.03241's training protocol (idea 3 depends on it), 2505.12082's tables, 2010.06192's PDF tables
  (the fetch returned binary), 2303.01500's numbers, 2605.02105's body, 2010.13369's body. Each dependent claim is marked.
- Not searched in depth: gradient-norm penalties (GAM), Lookahead-style slow weights, the power scheduler, and a 2025 to
  2026 study of cooldown length for short MLM runs. A search for each is the next pass if idea 1 or 2 shows a gain.
- Regime: every averaging and cooldown number is at 20 TPP or more. Our 0.3 to 0.7 TPP is below every point where the
  averaging sources report gains. Nothing here is a protein or MLM result except the BERT PLD claim and the single-pass
  dropout observation.
- Costs are estimates from the record's profile (276 ms a step, 26 ms of banked Muon, compile 106 to 130 s). No code was
  run.

## Sources

- LAWA: https://arxiv.org/abs/2209.14981
- Early weight averaging (EWA, LAWA for LLM pretraining): https://arxiv.org/abs/2306.03241 ; HTML v2: https://arxiv.org/html/2306.03241v2
- Scaling laws beyond fixed durations (1-sqrt cooldown, SWA note): https://arxiv.org/abs/2405.18392 ; HTML v3: https://arxiv.org/html/2405.18392v3
- Model merging in pre-training (PMA): https://arxiv.org/abs/2505.12082
- Sharpness-aware pretraining and forgetting: https://arxiv.org/abs/2605.02105
- SAM for language models (Bahri et al.): https://arxiv.org/abs/2110.08529
- LookSAM: https://arxiv.org/abs/2203.02714
- Revisiting BFloat16 training: https://arxiv.org/abs/2010.06192
- Collage (low-precision LLM training): https://arxiv.org/abs/2405.03637
- Dropout reduces underfitting: https://arxiv.org/abs/2303.01500
- Drop Dropout (single-pass MLM): https://arxiv.org/abs/2505.24788
- Progressive layer dropping: https://arxiv.org/abs/2010.13369
- When does label smoothing help: https://arxiv.org/abs/1906.02629
- The Road Less Scheduled (Schedule-Free): https://arxiv.org/abs/2405.15682
- DeepSpeed Muon clipping issue: https://github.com/deepspeedai/DeepSpeed/issues/8439
- MuCon: https://arxiv.org/abs/2605.26459
- Local: `src/speedrunning_plms/strategies/strategy.py` (hooks, `StepState`), `schedule.py` (`lr_scale`, BOUNDS), `ema.py` (006 note), `optimizer.py` (fp32 momentum), `ideas/2026-10-30_mlm_pretraining.md` (rejections), `ideas/modded_nanogpt_2026.md` (92c)
