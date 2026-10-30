# Loss weighting and token selection for the 10-minute protein MLM

Date: 2026-10-30. A literature scout for SpeedrunningPLMs (36 blocks, 768 wide, 2,048 MLP, 15 percent MASK, one GH200,
600 s, about 2,150 steps of 65,536 tokens, about 9,800 masked residues a step). It reads `ideas/README.md`, the MLM,
capacity and beyond-backprop files, and does not repeat their ideas (neighbor heads, token dropping, the optimizers,
the backprop-free families). Scope: what to do with the per-residue losses before the backward. Nothing here is
measured on this benchmark. Each gain is our estimate unless the text says it is a source's number. Shorthand: MLM is
masked language modeling; CE is cross entropy over the 20 residues; RHO is the reducible-loss rule of Mindermann et al.
(2022) and RHO-1 is its language-model form, selective language modeling (SLM); EMA is an exponential moving average of
the weights; KL is Kullback-Leibler divergence; z-loss is a penalty on the log of the softmax normalizer; ppl is
perplexity; "nats" is validation CE; one seed resolves about 0.0015 nats (README).

Yardstick: one unit of ln(steps) is worth about 0.026 nats (README). A change that adds a fraction c to the step time
costs about 0.026 x ln(1 + c) nats; a saving of s is worth about 0.026 x ln(1 / (1 - s)) nats. The slope comes from a
different change, so treat the conversions as rough.

## Outcome

- No candidate has a measured gain on this benchmark, and no source measures a loss weighting in the first few thousand
  steps of an MLM. Most estimates are ties within plus or minus 0.003 nats, which is two seed bands; each needs two seeds.
- The test metric is the plain CE over the masked residues of fixed masks, and training sees fresh proteins (under 0.1
  epoch). Two consequences set the scope:
  - A weight that reads only the model's prediction, not the true label, keeps the expected-CE minimizer. For a fixed
    context x, the weighted loss is w(x) times the expected CE over labels, which a flexible model minimizes at the
    posterior for any w(x) > 0. A weight that reads the label (loss, excess loss, focal weights, clipping) moves the
    optimum toward the residues the model finds easy or hard. That bias must pay off in speed inside 2,150 steps.
  - At 0.1 epoch nothing overfits. Regularizers (label smoothing, confidence penalties, consistency terms, noise) have
    nothing to fix, and label smoothing has a measurable cost on the test metric (section 6).
- Selection by loss does not save compute. The forward and backward run on every position, and masking a loss term
  removes gradient, not matmuls. A selection idea is judged on gradient quality, not on step time. The only selection
  that saves time removes tokens from the blocks (token dropping, in the MLM file).
- The strongest sourced evidence is RHO-1's self-reference (no external data, downstream averages only). The protein
  evidence is a 2025 residue selection that is post-training with a reference model, not pretraining.
- Our RHO-1 runs are about 140 million tokens (2,150 steps at 65,536 tokens). RHO-1's runs are 9 billion to 36 billion
  tokens, 64 to 255 times more. That gap is the main reason to expect any selection gain here to be smaller.
- Yardstick conversions: a 2x step cost is worth about 0.018 nats (0.026 x ln 2); a 33 percent step cost about 0.0075;
  a 10 percent saving about -0.0027.
- Caveat: the brief gives the record as 2.7146 nats, and the README's record 028 row is 2.7319 val_select. The record may
  have moved since 028; the arithmetic here does not depend on it.

## Ranked table

Gain is the expected change in validation CE on this benchmark (estimate; negative is better). "Read" says how much of
the source was read. Step cost and compile cost are our estimates; the record compiles in 106 to 122 s against the 180 s
cap.

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats) | Compile | Step cost | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | Entropy-weighted masked CE, weights from the model's own predictive entropy | MiLe Loss, arXiv 2310.19531 (2023; NAACL 2024) | abstract | 1 | -0.002 to +0.002 | 0 to 5 s | +0.5 percent | 468M, 1.2B and 6.7B LMs on the Pile; "consistent" downstream gains, no numbers read |
| 2 | Excess-loss selection against the model's own EMA (RHO-1 self-reference), keep 70 percent | RHO-1, arXiv 2404.07965 (2024) | HTML; Sections 2 to 3.4, Tables 3 and 4 | 4 | net 0 to +0.005 (gross -0.002 to -0.008) | a second no-grad graph: 15 to 30 s (estimate) | +33 percent (estimate) | +3.3 percent as the text states, with 40 percent fewer tokens; the table averages do not reproduce the percentage |
| 3 | z-loss on the output logits | PaLM, arXiv 2204.02311 (2022); OLMo 2 (2024, snippet) | abstract; z-loss from secondary sources | 0.5 | 0 to -0.001 (stability) | 0 s | 0 | stability reports only; redundant with the record's tanh softcap |
| 4 | Clipping the highest per-token losses (loss truncation) | Kang and Hashimoto (ACL 2020); Error Norm Truncation, arXiv 2310.00840 (2023) | search summary | 0.5 | 0 to +0.003 | 0 s | 0 | generation and noisy-web settings; no MLM number |
| 5 | Focal weights (1 - p)^gamma on the masked CE | Lin et al., arXiv 1708.02002 (2017) | not read | 1 | +0.005 to +0.02 (estimate, worse) | 0 to 5 s | 0 | no MLM pretraining result found |
| 6 | Self-distillation from an EMA teacher on masked positions (mean teacher) | Tarvainen and Valpola, arXiv 1703.01780 (2017); SAS, arXiv 2106.07176 (2021) | not read; snippet | 3 | net 0 to +0.006 (gross -0.002 to -0.008) | a second no-grad graph: 15 to 30 s (estimate) | +33 percent (estimate) | no MLM pretraining number found |
| 7 | Label smoothing (epsilon 0.1) or a confidence penalty | Mueller et al. (NeurIPS 2019), arXiv 1906.02629; Malagutti et al., arXiv 2403.17240 (2024) | search summary; abstract | 0.5 | +0.003 to +0.010 (estimate, worse) | 0 s | 0 | no held-out CE result for LM pretraining found |
| 8 | Consistency between two corruptions of one window (R-drop style) | R-Drop, arXiv 2106.14448 (2021) | search summary | 2 | 0 to +0.003 at equal tokens; about +0.018 at 2x tokens | 0 to 5 s | 0 (equal tokens) or +100 percent | fine-tuning and translation; no pretraining cost reported |
| 9 | Inverse-frequency or class-balanced weights on the 20 residues | Cui et al., arXiv 1901.05555 (2019) | not read | 0.5 | 0 to +0.002 | 0 s | 0 | none in protein MLM; the skew is mild |
| 10 | Easy-first or hard-later order inside a batch | curriculum learning (Bengio et al., 2009); no token-level LM result | search summary | 1 | exactly 0 within a step | 0 s | 0 | none |
| 11 | Pairwise or listwise ranking loss on the 20-way softmax | RankT5, arXiv 2210.10634 (2022), retrieval only | snippet | 2 | 0 or worse (estimate: +0.005 or more) | 0 to 5 s | 0 to 1 percent | no vocabulary-level LM result |
| 12 | Auxiliary losses on unmasked positions (identity, or ELECTRA-style detection) | ELECTRA, arXiv 2003.10555 (2020) | search summary | 2 | 0 to +0.003 | 0 to 10 s | +1 to 2 percent | the rules fix the input; identity on visible residues is trivial |

## 1. Entropy-weighted masked CE

- Source: MiLe Loss, "MiLe Loss: a New Entropy-Weighed Loss for Mitigating the Bias of Learning Difficulties in Large
  Language Models" (arXiv 2310.19531; NAACL 2024). Read: abstract. The search summary gives the weight's bounds as 1 to
  1 + log N for a vocabulary of N; the formula and constants were not read.
- Evidence: the abstract reports models at 468M, 1.2B and 6.7B parameters trained on the Pile, with "consistent
  performance improvement on downstream benchmarks". No table was read. ESLM (arXiv 2505.19893, 2025) is a selective
  relative: it keeps the tokens above a value-at-risk threshold on loss or entropy, and its abstract claims fewer FLOPs at
  matched perplexity. No numbers were read.
- What changes here: each masked residue's weight is a function of its own prediction's entropy, computed from logits
  that already exist. Because the weight reads the prediction and not the label, it is the same for every label at a
  context, so the expected-CE minimizer is unchanged (see the first Outcome bullet). Its effect is on how fast the model fits
  high-entropy contexts, which carry most of the CE.
- Expected cost: 1 hour. Step: +0.5 percent (estimate: a softmax over 20 residues for about 9,800 masked positions).
  Compile: one logits hook and one weights hook with static shapes, 0 to 5 s (estimate). Memory: none.
- Gain estimate: -0.002 to +0.002 nats. The sign is unknown: the weight may speed fitting on uncertain contexts, or spend
  the gradient on contexts that are uncertain for good reason.
- Sketch (no hook covers the per-token loss, so `token_weights` is new: a one-line change where `train.py` reduces the
  masked CE):

```python
class EntropyWeighted(Strategy):                            # family "token_weight" (new); variant "entropy" or "none"
    def logits(self, model, Z, state):                      # Z: (b, l, 20) residue logits, before the loss
        p = Z.detach().float().softmax(-1)                  # (b, l, 20), no label read
        state["entropy"] = -(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(20.0)   # (b, l) in [0, 1]
        return Z
    def token_weights(self, state, masked):                 # NEW hook at the loss site: (b, l) weights
        w = (1.0 + state["entropy"]) * masked.float()       # in [1, 2] on masked positions, 0 elsewhere
        return w * masked.float().sum() / w.sum().clamp_min(1.0)   # mean 1 over masked, so the step size holds
```

- Risk: a tie is the likely result. Normalizing to mean 1 keeps the update scale, but the weights change the gradient
  direction, and the Muon normalization may absorb part of the change.

## 2. Excess-loss selection against the model's own EMA (RHO-1 self-reference)

- Source: RHO-1, "Rho-1: Not All Tokens Are What You Need" (arXiv 2404.07965; NeurIPS 2024). Read: the HTML, its
  Section 3.4, Table 3 and Table 4; the fetch covered the first 100,000 of 107,000 characters, so Appendix H is partly
  read. The rule: score each token by the excess of the training model's loss over a reference's, keep the top tokens,
  and train only on those. Table 4 (appendix) sweeps the keep ratio for the loss score from 90 to 50 percent; the best
  average, 23.9, is at 70 percent.
- Parent rule: RHO-loss (arXiv 2206.07137, ICML 2022; search summary). Its reference is a small model trained on held-out
  data. The brief forbids a reference model, so the allowed form is a self-reference.
- Evidence (self-reference, from the fetch's reading of Table 3): the reference is trained on OpenWebMath, a subset of
  Proof-Pile-2, and no external high-quality data is used. The intersection of the loss and entropy scores trains on 9
  billion of 15 billion tokens and averages 24.8 downstream, against 21.5 for continual pretraining on all tokens. The
  text states +3.3 percent for that row, which does not match 24.8 against 21.5 (a 15 percent ratio); check the text before
  quoting it. On the larger PPile setting the text states +1.8 percent with 30 percent fewer tokens (averages 26.5
  against 24.7). Only downstream averages are reported for these runs; no validation perplexity was read for them.
- Protein support (not pretraining): Structure-aligned protein language model (arXiv 2505.16896, 2025; abstract) scores
  residues by an excess loss against a small model trained on high-quality structures and keeps the top fraction per
  residue type. Its abstract reports a 59 percent increase in P at L for ESM2 650M on CASP16 contact prediction, and gains
  in deep mutational scanning, with models from 8M to 650M. The reference model and the post-training step differ from
  ours, so this supports that residue-level excess selection transfers to proteins, not that it speeds MLM pretraining.
- What changes here: the reference is an EMA of the live weights, updated each step. Each step runs one no-grad forward on
  the same 65,536 tokens, which costs about a third of a training step (estimate: a forward is about a third of a forward
  plus backward). Excess is the live CE minus the EMA CE per masked residue; the top 70 percent of about 9,800 masked
  targets keep weight 1, and the rest weight 0.
- Expected cost: 4 hours (an EMA copy, the no-grad path, the top-k mask). Step: +33 percent (estimate), which costs about
  +0.0075 nats by the yardstick before any selection gain. Compile: the no-grad forward is a second graph unless the EMA
  shares the compiled module; 15 to 30 s of extra warmup (estimate) against the record's 106 to 122 s and the 180 s cap.
  A pilot must confirm the warmup. If the no-grad pass runs in eager mode it costs more per step.
- Gain estimate: gross -0.002 to -0.008 nats before the cost (estimate; RHO-1's downstream gain does not convert to our
  units); net 0 to +0.005 nats. A loss is more likely than a gain.
- Sketch (the EMA and top-k are the new parts; `token_weights` is the loss-site hook of section 1):

```python
class ExcessSelect(Strategy):                               # family "selection" (new); variant "ema_excess" or "none"
    def build(self, model, max_tokens):
        self.ema = copy.deepcopy(model).requires_grad_(False)     # the reference: an EMA of the live weights
    def after_step(self, state):
        for e, w in zip(self.ema.parameters(), state.model.parameters()):
            e.lerp_(w.detach(), 1.0 - self.decay)           # decay is a search value (estimate 0.999)
    def token_weights(self, state, masked, cur_ce):          # cur_ce, ref_ce: (b, l) per-residue CE
        excess = (cur_ce - state["ref_ce"]).detach().flatten()           # (b * l,) reducible-loss proxy
        score = excess.masked_fill(~masked.flatten(), float("-inf"))
        k = self.k_fixed                                    # 70 percent of masked targets per window, fixed before the run
        keep = torch.zeros_like(score, dtype=torch.bool)
        keep[score.topk(k).indices] = True                  # (b * l,) selection, static k keeps the graph fixed
        return (keep & masked.flatten()).view_as(masked).float()
```

- Risk: the EMA lags the live model, so the excess is partly the last few hundred steps of change, measured on 9,800
  noisy residues a step. The text-table mismatch means the RHO-1 gain is not yet sized.

## 3. z-loss on the output logits

- Source: PaLM (arXiv 2204.02311, 2022) uses a z-loss for training stability, and OLMo 2 uses it too (search summary). Read:
  the PaLM landing page only (the fetch did not return the stability section), so the formula and the coefficient are from
  secondary sources (a T5X code note; a 2025 report on stability). Treat the coefficient as unverified.
- Formula: z_loss x (log Z)^2 added to the CE, where Z is the sum over residues of exp(logit). The penalty is averaged over
  masked positions.
- Key point: the softmax is invariant to a shift of all logits, while log Z is not. The penalty pins the logit level and
  leaves every probability unchanged, so the minimizer of the test CE is untouched. The gain is stability: fewer loss
  spikes. A run that already trains stably at 2,150 steps has little to gain.
- Redundancy: the record's logits pass a tanh softcap at 30, which already bounds log Z. The expected effect is zero.
- Expected cost: 0.5 hour. Step: 0 (one logsumexp over logits already computed). Compile: 0 s.
- Gain estimate: 0 to -0.001 nats (a stability insurance; no loss number read).
- Sketch:

```python
class ZLoss(Strategy):                                      # family "logit_norm" (new); variant "z" or "none"
    def auxiliary_loss(self, model, states, labels, logits):     # logits: (b, l, 20) at masked positions
        log_z = torch.logsumexp(logits.float(), dim=-1)          # (b, l) log normalizer per residue
        return self.coefficient * (log_z ** 2).mean()            # coefficient: a search value, 1e-4 in the secondary source
```

## 4. Clipping the highest per-token losses (loss truncation)

- Source: Kang and Hashimoto, "Improved Natural Language Generation via Loss Truncation" (ACL 2020; search summary):
  adaptively drops high-loss examples as noise. Error Norm Truncation (arXiv 2310.00840, 2023; search snippet) truncates
  high-error tokens in generation. Neither is a masked-LM or protein result.
- What changes here: drop the masked targets whose CE is above a per-step quantile (the top 2 percent, say), so they carry
  zero weight and leave the mean.
- Objective effect: the weight reads the label, so the optimum is biased against the hard residues. The test CE counts
  every masked residue, so a clipped tail is a test loss on those residues. It helps only if the tail is mostly
  unlearnable noise, plausible for disordered or low-complexity segments but not measured here.
- Expected cost: 0.5 hour. Step: 0. Compile: 0 s (a quantile over about 9,800 values per step, under 1 ms, estimate).
- Gain estimate: 0 to +0.003 nats (a loss is the likely sign).
- Sketch:

```python
def token_weights(self, state, masked, cur_ce):             # NEW hook at the loss site; cur_ce: (b, l)
    cut = torch.quantile(cur_ce.detach()[masked], 0.98)      # per-step threshold on the masked CE
    return (cur_ce.detach() <= cut).float() * masked.float() # drop the top 2 percent of masked targets
```

## 5. Focal weights on the masked CE

- Source: Lin et al., "Focal Loss for Dense Object Detection" (arXiv 1708.02002, 2017). Read: not read; cited for the
  definition. MLM evidence: none found. The only related result in the MLM file is a 2022 BERT fine-tuning calibration
  study.
- What changes here: weight (1 - p_y)^gamma on each masked CE term, with gamma 1 to 2, where p_y is the model's probability
  of the true residue.
- Objective effect: the weight reads the label, and the focal optimum is not the posterior. It down-weights confident
  correct predictions and up-weights hard labels, so the test CE carries a bias. At gamma 2 a residue at p_y 0.9 gets
  weight 0.01 and one at p_y 0.3 gets weight 0.49, a large reweighting. Estimate +0.005 to +0.02 nats (worse); the magnitude
  is not measured.
- Expected cost: 1 hour. Step: 0. Compile: 0 to 5 s.
- Gain estimate: +0.005 to +0.02 nats (worse, estimate). Rejected unless a pilot shows a fast early drop that survives to 600 s.

## 6. Label smoothing and confidence penalties

- Sources: Mueller et al. (NeurIPS 2019; arXiv 1906.02629), as summarized in a 2020 AACL paper: label smoothing improves
  calibration and hurts knowledge distillation (search summary). Malagutti et al. (arXiv 2403.17240; NAACL 2024; abstract):
  label smoothing is formally equivalent to add-lambda smoothing, and their derived regularizers are "comparable to and
  sometimes outperform" label smoothing on language modeling and translation. Self-evolution learning (arXiv 2305.15275;
  Findings of ACL 2023; abstract) uses token-specific label smoothing for discriminative pretraining, with +1.43 to +2.12
  average on 10 tasks; those are not CE numbers.
- Arithmetic (ours): with epsilon 0.1 over 20 residues, the model learns the smoothed target, so its output is
  p = 0.9 p* + 0.005 for a calibrated posterior p*. The excess test CE is then KL(p* || p). For a true residue at
  posterior 0.3, the excess is 0.0015 nats; at 0.8, 0.016; at 0.95, 0.043. Most masked residues of a model at 2.7 nats
  sit in the uncertain range, so the mean excess is estimated at +0.003 to +0.010 nats at epsilon 0.1. A confidence
  penalty (an entropy bonus) moves the optimum toward uniform by the same kind of amount (estimate).
- Why it is rejected here: the score is plain CE, and at 0.1 epoch the model is underfit, so smoothing buys no
  regularization. The smoothed target is a biased CE target.
- Expected cost: 0.5 hour. Step: 0. Compile: 0 s (a target change).
- Gain estimate: +0.003 to +0.010 nats (worse). Not sketched; the change is one line in the target.

## 7. Self-distillation from an EMA teacher on masked positions

- Source: mean teacher, Tarvainen and Valpola (arXiv 1703.01780, 2017; not read). SAS, arXiv 2106.07176 (2021; search
  snippet), uses a teacher-student consistency for language-model pretraining with no number read here. No 2023 to 2026
  MLM result with an EMA teacher was found.
- What changes here: add lambda times KL(teacher || student) at the masked positions, with the teacher an EMA of the weights.
  The teacher forward is the same no-grad pass as in section 2.
- Objective effect: the KL target is a smoothed version of the model's own predictions, so the minimizer moves toward the
  teacher. It does not read the label, so the expected-CE minimizer holds only if the teacher tracks the posterior. Early
  in training the teacher lags, so its targets are poor, which is the regime this clock covers.
- Expected cost: 3 hours. Step: +33 percent (estimate, +0.0075 nats). Compile: 15 to 30 s (estimate; second graph).
- Gain estimate: gross -0.002 to -0.008 nats (soft targets carry the teacher's uncertainty per residue, estimate); net 0 to
  +0.006 nats.
- Sketch:

```python
def auxiliary_loss(self, model, states, labels, masked, ids):     # student logits from the live forward
    with torch.no_grad():
        t = self.ema_logits(ids).float().softmax(-1)              # (b, l, 20) teacher, same masked input
    s = self.student_logits.float().log_softmax(-1)               # (b, l, 20) student
    kl = (t * (t.clamp_min(1e-12).log() - s)).sum(-1)             # (b, l) KL(teacher || student)
    return self.lam * (kl * masked.float()).sum() / masked.float().sum().clamp_min(1.0)
```

## 8. Consistency between two corruptions of one window (R-drop style)

- Source: R-Drop, arXiv 2106.14448 (NeurIPS 2021; search summary): two forward passes per sample with a KL term between
  their outputs. Its reported gains are on fine-tuning and translation, and no pretraining cost was found.
- Why the form matters here: R-drop needs disagreement between the two passes. The usual source is dropout, and
  `train.py` has none. The only other disagreement the rules allow is a second, independent mask draw on the same window.
  Those draws mark different residues, so a residue scored in one draw is visible in the other, and the consistency term
  covers only residues masked in both draws (about 2 percent of residues at a 15 percent rate, estimate). A same-mask
  second forward with no dropout is identical, so the term is zero.
- What changes here: two mask draws per window, with a KL term at residues masked in both draws. At equal tokens per step
  this halves the windows per draw.
- Expected cost: 2 hours. Step: 0 at equal tokens; +100 percent at 2x tokens. Compile: 0 to 5 s.
- Gain estimate: 0 to +0.003 nats at equal tokens; about +0.018 nats of step cost at 2x tokens (by the yardstick). Rejected
  as a regularizer at 0.1 epoch.

## 9. Inverse-frequency and class-balanced weights on the 20 residues

- Source: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" (arXiv 1901.05555; CVPR 2019). Read: not
  read. The brief says the residue skew is mild, so the reweighting has little to correct.
- Objective effect: the test CE weights each residue by its natural frequency. Inverse-frequency weights move the optimum to
  a uniform mix over residues, so the frequency-weighted test CE is biased. A gain needs rare residues to be learned faster
  and to generalize, which a mild skew does not force.
- Expected cost: 0.5 hour. Step: 0. Compile: 0 s.
- Gain estimate: 0 to +0.002 nats. Rejected for the test metric unless a per-residue CE table shows a skewed gap.

## 10. Easy-first or hard-later order inside a batch

- Source: curriculum learning (Bengio et al., 2009; not read). Token-level curricula in the search were masking curricula,
  which change which residues are masked: concept-based curriculum masking (arXiv 2212.07617; search summary) and a
  second-hand citation of an easy-to-predict-first masking schedule. None orders the loss in time within a step. The
  protein-level curriculum is the catalog's `curriculum` family.
- Argument: within a step the update is the gradient of one summed loss over the whole batch, and a sum does not depend on
  order. So reordering residues inside a step has exactly zero effect. Ordering matters only across steps, which is the
  data-order file's territory, or when some residues are dropped at a step, which is selection (sections 2 and 4).
- Expected cost: 1 hour. Step: 0. Compile: 0 s.
- Gain estimate: exactly 0 within a step. Not a candidate.

## 11. Pairwise or listwise ranking losses on the 20-way softmax

- Source: RankT5 (arXiv 2210.10634, 2022; snippet) compares pointwise CE, pairwise logistic and listwise softmax CE for
  text ranking. No vocabulary-level LM result for 2024 to 2026 was found.
- Argument: the test metric is CE, and a pairwise or margin loss does not minimize CE, so its minimizer is off-target. The
  listwise top-one softmax loss is CE itself, which is the record's loss.
- Cost: 2 hours. Step: +0 to 1 percent (190 residue pairs per masked position, estimate). Compile: 0 to 5 s.
- Gain estimate: 0 or worse (estimate: +0.005 nats or more). Rejected.

## 12. Auxiliary losses on unmasked positions

- Source: ELECTRA (arXiv 2003.10555, ICLR 2020; search summary): replaced-token detection gives a signal at every position,
  and its generator is an MLM trained on masked positions. No 2024 to 2026 all-token MLM result was found.
- Argument: the rules fix the input at 15 percent MASK and nothing else. An identity loss on visible residues is solved by
  copying the embedding, which the stream already carries (the catalog's `embedding_mix` and value residual), so the term
  adds gradient the model already satisfies. ELECTRA's replacement corruption changes the input and is out of rules. The
  only non-trivial variant is the teacher term of section 7 applied to unmasked positions, which inherits its cost.
- Expected cost: 2 hours. Step: +1 to 2 percent (identity head over all 65,536 positions, estimate). Compile: 0 to 10 s.
- Gain estimate: 0 to +0.003 nats. Rejected as an identity loss; the teacher variant is section 7.

## Checked and rejected

- Mixup or token-level mixup for MLM: no 2023 to 2026 MLM source found. The search returned MAE-LM (arXiv 2302.02060, ICLR
  2024; the [MASK] token takes up representation space), which is not mixup. Out of rules anyway, since mixing changes the input.
- RHO-loss with a held-out small reference model (arXiv 2206.07137): the brief forbids a reference model. Section 2 is the
  self-reference form.
- Entropy-aware masking (arXiv 2605.28526, 2026) and weighted sampling for MLM (WSBERT, arXiv 2302.14225): both choose which
  residues are masked, which the rules fix. WSBERT's abstract says it masks by token frequency and training loss and reports
  STS and GLUE gains; no numbers were read.
- Adaptive per-residue masking (arXiv 2510.20475, 2025): a masking rule, already idea 4 of the MLM file.
- Logit normalization (Wei et al., arXiv 2205.09310, ICML 2022; not read): rescales the logits before the softmax, so the
  probabilities the CE reads change. Off-target for the test metric.
- Confidence penalty (Pereyra et al., arXiv 1701.06548, 2017; not read): same bias as section 6.
- Self-paced learning (Kumar et al., NeurIPS 2010; not read): loss-thresholded selection, the same objective bias as
  section 4, and not a 2023 to 2026 source.
- Learning to reweight examples by a validation-gradient meta step (Ren et al., ICML 2018; not read): a second backward
  per step, so about 2x step cost, and the sign on a 0.1-epoch run is unknown. Rejected on cost.
- Token dropping by loss (Hou et al., 2022; Zhong et al., 2023): covered in the MLM file (idea 3) and the capacity file.

## Hooks the registry does not have

- `token_weights(state, masked, cur_ce)` at the loss site (sections 1, 2, 4, 5): the main masked CE is not a hook. The
  change is one line where the loss is reduced in `train.py`, and it is a loop change, so it needs a deliberate commit.
- `logits` (listed) sees the logits before the loss: enough for the entropy and log Z.
- A second no-grad forward for the EMA (sections 2 and 7): a second compiled module or an eager pass. The compile cost
  decides both.
- `auxiliary_loss` sees the context with labels: enough for z-loss, the teacher KL and an identity head.

## Gaps and read status

- Read in substance: RHO-1 (HTML, most of it; the self-reference rows of Table 3 as the fetch reported them).
- Abstract only: MiLe (arXiv 2310.19531), ESLM (arXiv 2505.19893), self-evolution learning (arXiv 2305.15275), the
  structure-aligned protein model (arXiv 2505.16896), Malagutti et al. (arXiv 2403.17240), WSBERT (arXiv 2302.14225), PaLM
  (arXiv 2204.02311, landing page only).
- Search summaries or snippets only: RHO-loss (arXiv 2206.07137), Kang and Hashimoto (2020), Error Norm Truncation (arXiv
  2310.00840), R-Drop (arXiv 2106.14448), ELECTRA (arXiv 2003.10555), Mueller et al. (2019, via a 2020 summary), SAS (arXiv
  2106.07176), RankT5 (arXiv 2210.10634), concept-based curriculum masking (arXiv 2212.07617).
- Not read: focal loss (arXiv 1708.02002), class-balanced loss (arXiv 1901.05555), logit normalization (arXiv 2205.09310),
  confidence penalty (arXiv 1701.06548), mean teacher (arXiv 1703.01780), the curriculum and self-paced papers, and the
  meta-reweighting paper.
- No masked-LM or protein number exists in the read set for any loss weighting. No source measures one in the first few
  thousand steps of an MLM. The one stated number that conflicts with its own table is RHO-1's +3.3 percent.
- The gains here are language-model numbers at 0.3 to 25 TPP and our arithmetic. The protein MLM at 0.3 to 0.7 TPP is
  not measured by any source read.

## Sources

- RHO-1, Rho-1: Not All Tokens Are What You Need, arXiv 2404.07965: https://arxiv.org/abs/2404.07965 (HTML: https://arxiv.org/html/2404.07965v4)
- RHO-loss, Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt, arXiv 2206.07137: https://arxiv.org/abs/2206.07137
- MiLe Loss, arXiv 2310.19531: https://arxiv.org/abs/2310.19531
- ESLM, arXiv 2505.19893: https://arxiv.org/abs/2505.19893
- Structure-aligned protein language model, arXiv 2505.16896: https://arxiv.org/abs/2505.16896
- Weighted sampling for masked language modeling (WSBERT), arXiv 2302.14225: https://arxiv.org/abs/2302.14225
- Adaptive MLM for BabyLMs, arXiv 2510.20475: https://arxiv.org/abs/2510.20475
- Self-evolution learning for discriminative pretraining, arXiv 2305.15275: https://arxiv.org/abs/2305.15275
- Label smoothing and add-lambda smoothing (Malagutti et al.), arXiv 2403.17240: https://arxiv.org/abs/2403.17240
- When does label smoothing help (Mueller et al.), arXiv 1906.02629: https://arxiv.org/abs/1906.02629
- PaLM, arXiv 2204.02311: https://arxiv.org/abs/2204.02311
- Error Norm Truncation, arXiv 2310.00840: https://arxiv.org/abs/2310.00840
- Kang and Hashimoto, Improved Natural Language Generation via Loss Truncation (ACL 2020): https://aclanthology.org (search summary; no link checked)
- Focal loss, arXiv 1708.02002: https://arxiv.org/abs/1708.02002
- Class-balanced loss, arXiv 1901.05555: https://arxiv.org/abs/1901.05555
- Logit normalization, arXiv 2205.09310: https://arxiv.org/abs/2205.09310
- Confidence penalty, arXiv 1701.06548: https://arxiv.org/abs/1701.06548
- Mean teacher, arXiv 1703.01780: https://arxiv.org/abs/1703.01780
- SAS, self-augmentation for language model pretraining, arXiv 2106.07176: https://arxiv.org/abs/2106.07176
- R-Drop, arXiv 2106.14448: https://arxiv.org/abs/2106.14448
- ELECTRA, arXiv 2003.10555: https://arxiv.org/abs/2003.10555
- RankT5, arXiv 2210.10634: https://arxiv.org/abs/2210.10634
- MAE-LM (representation deficiency in MLM), arXiv 2302.02060: https://arxiv.org/abs/2302.02060
- Concept-based curriculum masking, arXiv 2212.07617: https://arxiv.org/abs/2212.07617
- Focal loss, Lin et al. (ICCV 2017) and the other original papers above are cited for their definitions only.
