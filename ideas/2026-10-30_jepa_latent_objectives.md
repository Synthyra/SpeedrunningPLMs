# JEPA-like and latent-prediction objectives for the 10-minute protein MLM

Date: 2026-10-30

Scope: auxiliary or replacement objectives in representation space for the fixed speedrun (36 pre-norm blocks, 198M
parameters, 15% MASK-only masked LM, 20-way cross entropy (CE), one GH200, 600 s of applied training steps). The module
catalog (`ideas/README.md`) already holds the architecture and optimizer families; nothing here repeats them. Score is
test CE on masked residues, so every idea is judged on whether it lowers that number, not on downstream probes.

Shorthand: JEPA joint-embedding predictive architecture; EMA exponential moving average teacher; SIGReg sketched
isotropic Gaussian regularizer (LeJEPA); KD knowledge distillation; PLM protein language model; ESM2-8M the 8M-parameter
ESM-2 checkpoint, cited in the catalog at test CE 2.625 against the record's 2.7119.

## How the evidence was read

- Read from the primary paper (fetched HTML or PDF): ProteinJEPA (arXiv 2605.07554, v1 and v2); LLM-JEPA (arXiv
  2509.14252 v2); NextLat (arXiv 2511.05963 v1); data2vec (arXiv 2202.03555 v3); I-JEPA (arXiv 2301.08243 v3);
  hierarchical self-distillation for encoders (arXiv 2503.03008 v1); hidden-layer distillation for LLM pretraining
  (arXiv 2605.11513 v1).
- Search summaries or abstracts only, so numbers are unverified: DLLM-JEPA (arXiv 2606.00091, numbers from a secondary
  summary); LeJEPA (arXiv 2511.08544); Self-Representation Alignment (arXiv 2505.02831); Repr-Align (arXiv 2605.06885,
  no numbers retrieved); Bootleg (arXiv 2603.15553, one relative figure); "Strong Teacher Not Needed?" (arXiv
  2605.23857, abstract listing); self-distillation for data-scarce pretraining (ICML 2026 workshop page, no numbers).
- Not found: a sequence or protein result for a same-pass (no EMA, no extra forward) latent target; a protein result
  for a consistency loss between two masks of one sequence with a number; a protein data2vec result.

## Costs under the clock (my arithmetic unless marked)

- A training step is about three forward-equivalents. One clean forward through all 36 blocks adds about 33% step
  time; a clean forward through the first 18 adds about 17%. ProteinJEPA reports 17% for its half-depth target
  (2605.07554 v2, Sec. 3 and 4.2). Its 8 h all-position study puts the JEPA branch at about 1.8 times the cost per step
  (v1 Sec. 3.5; 88,024 steps against 160,793 for MLM-only, v1 Sec. 5).
- The record runs about 1,400 to 2,160 steps (catalog, 001). A 17% step tax is roughly 240 to 370 fewer steps.
- A predictor on the 15% masked positions is cheap. A two-layer SwiGLU at width 768 with hidden 2,048 (ProteinJEPA's
  8/3 expansion) is about 9.4M parameters and costs about 0.7% of block multiply-accumulates, counting only masked
  positions. A one-layer version is half that.
- A frozen ESM2-8M teacher costs about 1.3% of a step by parameter count (forward only, ignoring attention).
- Compile is the binding constraint. The catalog records 106 s for the baseline, 117 s with the value residual, 142 s
  for nGPT and 143 s for hyperconnections, against the 180 s cap. Any idea that adds a second compiled graph must be
  measured; the record leaves about 60 s.
- The break-even rule for any step-costing idea: the latent or teacher gain must beat the loss from fewer steps.
  Price that from the record's own loss curve in `runs.jsonl` (the per-eval losses), not from a literature number.

## Ranked ideas

Ranking is by expected gain per implementation hour under the clock, with cost and metric alignment weighed in. The
two teacher ideas rank highest only if the leakage question in the risk notes is cleared.

### 1. Frozen ESM2-8M soft-label distillation at masked positions, annealed to zero

- Source: KD baseline in hidden-layer distillation for LLM pretraining, Guigon, Dixon and Sander, arXiv 2605.11513
  (2026); the weak-teacher result in "Strong Teacher Not Needed?", arXiv 2605.23857 (2026, abstract only).
- Evidence: in that study, logit KD from a 3.4B teacher trained on 106B tokens lowered C4 log-perplexity from 3.063 to
  3.005 at 123M parameters (-0.058) and from 2.639 to 2.609 at 735M (-0.030) (Table 2). The abstract says a weaker,
  undertrained teacher still helps a larger student when the loss mix is tuned. Here the teacher is better than the
  record on our metric (2.625 against 2.7119, catalog), so the headroom is real. No protein number exists.
- What it changes: the student also learns the teacher's distribution over the 20 residues at masked positions, the
  quantity the score measures. The score is CE against the true residue, so the KD weight must go to zero by the end
  of the run.
- Cost: one extra no-grad forward of ESM2-8M on the clean window (about 1.3% of a step); one extra compiled graph
  (measure; catalog compile increments run from 5 s to 40 s). Implementation about 2 hours, including the tokenizer map.
- Sketch:

```python
class TeacherLogitKD:
    """Frozen small PLM soft labels over the 20 standard residues at masked positions."""
    def build(self, model, cfg):  # build: load by id, freeze, map residues
        self.teacher = load_esm2(cfg.teacher_id).eval().requires_grad_(False)
        self.aa_ids = teacher_ids_for_20_aa(self.teacher)  # assert all 20 found; check specials
    def forward(self, clean_tokens):  # forward: extra no-grad pass on the unmasked window
        with torch.no_grad():
            return self.teacher(clean_tokens).logits[..., self.aa_ids]  # [B, L, 20]
    def loss(self, s_logits, labels, mask, t_logits, w):  # loss: CE plus KL(teacher || student)
        ce = F.cross_entropy(s_logits[mask], labels[mask])
        kl = F.kl_div(F.log_softmax(s_logits[mask], -1), F.softmax(t_logits[mask], -1), reduction="batchmean")
        return ce + w * kl
    def schedule(self, frac):  # schedule: ramp in over 30%, fade to zero over the last 20%
        return self.cfg.kd_weight * min(1.0, frac / 0.3) * min(1.0, (1.0 - frac) / 0.2)
```

- Risk: the leakage question. ESM-2 was trained on UniRef50 (verify in Lin et al. 2022 before relying on it), and
  UniRef50 overlaps the sequence families behind AFDB. A teacher that has seen test-family sequences would make a
  test gain partly memorization. Check the identity of the test set against the teacher's training set before any
  run; this is a benchmark-validity call for Logan, not a rule I can settle from the literature. Second risk: soft
  labels from a teacher with its own miscalibration can raise CE on true labels in the middle of training; the anneal
  limits that.

### 2. Multi-exit deep supervision through the shared MLM head

- Source: Hierarchical self-distillation for encoders, arXiv 2503.03008 (2025), read from v1.
- Evidence: a 1B, 36-layer MLM encoder with exits at layers 4, 9, 18, 27 and 36 sharing one head, trained on 1T tokens
  (about 450,000 GPU hours in total). On CodeSearchNet mean reciprocal rank (MRR) (Table 5): layer 4 70.5 to 73.2
  (+2.7), layer 9 73.8 to 77.3 (+3.5), layer 18 80.7 to 81.0 (+0.3), layer 27 79.1 to 80.3 (+1.2), final layer 79.5
  to 79.6 (+0.1). The paper reports no MLM loss. The final-layer gain is the one that matters here and it is tiny.
- What it changes: the shared MLM head reads blocks 9, 18 and 27 at masked positions with weights i/36, so the
  shallower blocks get direct token supervision. No new parameters; the head is already in the record.
- Cost: the extra head matmul is on 20 classes at masked positions, effectively zero step time. Implementation about
  one hour. Compile unchanged to first order.
- Sketch:

```python
class MultiExitDeepSupervision:
    """The shared MLM head also reads intermediate blocks at masked positions; no new parameters."""
    exits = (9, 18, 27)
    def build(self, model, cfg):  # build: reuse the record's head; weights i/36
        self.head = model.lm_head
        self.alpha = {e: e / 36 for e in self.exits}
    def forward(self, model, x):  # forward: keep the block outputs at the exits
        hs = {}
        for i, blk in enumerate(model.blocks, start=1):
            x = blk(x)
            if i in self.exits:
                hs[i] = x
        return x, hs
    def loss(self, final_ce, labels, mask, hs, cfg):  # loss: final CE plus weighted exit CEs
        extra = sum(self.alpha[e] * F.cross_entropy(self.head(model.norm(hs[e][mask])), labels[mask]) for e in self.exits)
        return final_ce + self.schedule_w * extra
    def optimizer(self):  # optimizer: nothing new; the head stays in its group
        return []
    def schedule(self, frac):  # schedule: ramp in, then fade to zero so the final head ends on pure CE
        self.schedule_w = min(1.0, frac / 0.3) * min(1.0, (1.0 - frac) / 0.2)
```

- Risk: the source's gains concentrate on early exits and are retrieval scores, not MLM CE, so expected final-layer
  gain is near zero. Auxiliary CE on shallow blocks may also compete with the value residual, which already feeds the
  first block's values into the stream. Run it first only because it is nearly free; keep it if it does not move the
  final CE beyond the noise band (0.0015 nats).

### 3. Same-pass stop-gradient latent prediction with SIGReg

- Source: next-latent prediction, NextLat, Teoh et al., arXiv 2511.05963 (2025), read from v1; the SIGReg regularizer
  from LeJEPA, arXiv 2511.08544 (2025, abstract level); the protein use of SIGReg in ProteinJEPA's 8 h study
  (2605.07554 v1, Sec. 3.2, lambda 0.45, 256 projections).
- Evidence: NextLat adds a latent prediction loss with a stop-gradient target and no extra inference cost; on
  TinyStories (57M, 100k steps) its one-step variant ran at 3.26 against 3.72 iterations per second for the baseline
  on one H100 NVL (about 12% slower; Table 2). It "matches" next-token performance; the numbers sit in Figure 10 and
  were not read. In protein JEPA, latent loss alone collapses: JEPA-only versus MLM-only is 2 wins, 17 losses,
  median -0.124 on ESM2-35M, and the MLM CE rose to 11.4 nats before settling (2605.07554 v2, App. A.1). So the latent
  term must ride on MLM, and it needs an anti-collapse term.
- What it changes: no teacher and no second forward. A predictor reads block 18 at masked positions and predicts the
  stop-gradient block 36 output at the same positions. The stop-gradient target from the same pass is untested in
  protein work; the nearest tested design (ProteinJEPA) uses a clean forward or an EMA teacher.
- Cost: under 1% of step FLOPs for a one-layer predictor on masked positions; SIGReg on 256 projections adds a sort per
  projection on the masked batch (measure). Implementation about two hours.
- Sketch:

```python
class SamePassLatent:
    """Stop-gradient latent prediction inside the one masked forward; SIGReg guards against collapse."""
    def build(self, model, cfg):  # build: one-layer SwiGLU predictor, 768 wide
        self.pred = SwiGLU(768, 2048, layers=1)
        self.src, self.tgt = 18, 36
    def forward(self, hs):  # forward: no extra pass; the target is detached
        return hs[self.src], hs[self.tgt].detach()
    def loss(self, z_src, z_tgt, mask, lam):  # loss: cosine latent term plus SIGReg on predictions
        p = F.layer_norm(self.pred(z_src[mask]), (768,))
        t = F.layer_norm(z_tgt[mask], (768,))
        return lam * (1 - (p * t).sum(-1)).mean() + self.cfg.reg * sigreg(p, n_proj=256)
    def optimizer(self):  # optimizer: predictor in its own group, no weight decay
        return [{"params": self.pred.parameters(), "weight_decay": 0.0}]
    def schedule(self, frac):  # schedule: ramp lambda in over the first 30%
        return self.cfg.lam * min(1.0, frac / 0.3)
```

- Risk: the same-pass target may be trivially satisfiable from block 18 if block 36 carries little beyond block 18;
  check the cosine on a held-out batch. The gain may be below the noise band. Expect the MLM CE to rise slightly, as
  ProteinJEPA's did (about 0.002 nats), so this needs a real gain elsewhere to survive the score.

### 4. Frozen ESM2-8M hidden-state regression at masked positions

- Source: hidden-layer distillation (2605.11513, Sec. 3 and 4): normalized hidden states matched through a learned
  regressor, HLDC (joint) and HLDF (sequential); Repr-Align, arXiv 2605.06885 (2026), where a layer-wise cosine
  alignment at masked positions beat an L2 alignment (summary, no number retrieved).
- Evidence: on C4 at 123M, HLDF beat KD by 0.005 log-perplexity (3.000 against 3.005) and HLDC matched KD (3.005); at
  735M the gap was 0.002 (2.607 against 2.609). The teacher was 3.4B on 106B tokens. Hidden-state targets add little
  over logit KD in that study; the authors call the effect within what one seed resolves.
- What it changes: the student's masked-position states are pulled toward a frozen PLM's states, which encode more
  context than 20-way logits. The adapter maps ESM2-8M's width (320, verify) to 768.
- Cost: same teacher forward as idea 1 (about 1.3%), plus a linear adapter. Only run it on top of idea 1 or
  alone if idea 1 fails. Implementation about 2 to 3 hours including the adapter and the layer choice.
- Sketch:

```python
class TeacherHiddenRegression:
    """Frozen small PLM hidden states as targets at masked positions; cosine, after a linear adapter."""
    def build(self, model, cfg):  # build: frozen teacher; adapter outside the 36 blocks
        self.teacher = load_esm2(cfg.teacher_id).eval().requires_grad_(False)
        self.adapter = nn.Linear(self.teacher.width, 768)
        self.layer = cfg.teacher_layer  # chosen by a screen, not guessed
    def forward(self, clean_tokens):  # forward: no-grad teacher states on the unmasked window
        with torch.no_grad():
            return self.teacher.hidden_states(clean_tokens)[self.layer]  # [B, L, 320]
    def loss(self, s_states, t_states, mask, lam):  # loss: cosine at masked positions
        a = self.adapter(t_states[mask])
        return lam * (1 - F.cosine_similarity(s_states[mask], a, dim=-1)).mean()
    def optimizer(self):  # optimizer: adapter in the AdamW group
        return [{"params": self.adapter.parameters(), "weight_decay": 0.0}]
    def schedule(self, frac):  # schedule: fade to zero over the last 20%
        return self.cfg.lam * min(1.0, (1.0 - frac) / 0.2)
```

- Risk: the same leakage question as idea 1 applies. The adapter can absorb the teacher-student width gap and hide a
  poor match, so log the cosine at each eval.

### 5. ProteinJEPA masked-position latent loss against a half-depth EMA teacher

- Source: ProteinJEPA, arXiv 2605.07554 v2 (2026), main recipe; no independent replication found.
- Evidence (continued pretraining, 35M to 150M, UniRef50): MLM+JEPA against step-matched MLM-only, 76 wins, 15 losses,
  23 ties, median +0.0051 (p < 0.0001); against compute-matched, 78 / 14 / 22, median +0.0049. The MLM loss itself was
  about 0.002 nats higher with the JEPA term. On the structure and homology subset the median gain is +0.0106. The
  8 h wall-clock study on masked positions gave ESM2-35M a macro gain of +0.010 (10 wins, 3 losses, 3 ties) and
  AMPLIFY-120M 0.000 (v1 Table 1). The all-position version lost on ESM2-35M at 8 h (-0.042, v1 Table 3). Caveat:
  the gains are on downstream tasks at about one pass over the data, not on MLM CE.
- What it changes: the 18-block EMA teacher (decay ramped from 0.996 to 0.9995) sees the unmasked window and supplies
  a target at half depth; the student's final-layer states at masked positions predict it through a predictor.
- Cost: about 17% per step (paper). That is roughly 240 to 370 steps in the clock. Implementation about three to four
  hours including the EMA copy of the first 18 blocks, the predictor and the cooldown interplay.
- Sketch:

```python
class HalfDepthEMATarget:
    """EMA copy of the first 18 blocks sees the clean window; the student predicts its states at masked positions."""
    def build(self, model, cfg):  # build: EMA copy of blocks 1 to 18; predictor on the student side
        self.teacher = copy_blocks(model, upto=18)
        self.pred = SwiGLU(768, 2048, layers=1)
    def forward(self, clean_tokens):  # forward: extra no-grad pass, about 17% of a step
        with torch.no_grad():
            return self.teacher(clean_tokens)  # [B, L, 768]
    def loss(self, s_final, t_half, mask, lam):  # loss: cosine after layer norm
        p = F.layer_norm(self.pred(s_final[mask]), (768,))
        t = F.layer_norm(t_half[mask], (768,))
        return lam * (1 - (p * t).sum(-1)).mean()
    def post_step(self, model):  # post_step: EMA update of the teacher
        ema_update(self.teacher, model, upto=18, decay=self.decay)
    def schedule(self, frac):  # schedule: decay ramps 0.996 to 0.9995; lambda 0.45 (wall-clock setting)
        self.decay = 0.996 + (0.9995 - 0.996) * frac
        return 0.45
```

- Risk: the expected MLM CE is slightly worse, and the 17% step tax is paid in full. Break-even only if the latent gain
  is bigger than the loss from about 240 to 370 fewer steps. Read that slope from the record before building this.

### 6. data2vec-style top-K normalized targets from an EMA teacher on the clean window

- Source: data2vec, Baevski et al., arXiv 2202.03555 (ICML 2022), read from v3; data2vec 2.0 (arXiv 2212.07525, ICML 2023)
  for the cheaper targets.
- Evidence: on GLUE dev, data2vec averages 82.7 against a retrained RoBERTa at 82.5 and BERT at 80.7 (Table 4), so the
  gain over the retrained baseline is 0.2 points, within noise. NLP targets average the top K = 10 blocks after a
  parameter-free layer norm and regress the feed-forward output before the last residual. EMA tau runs from 0.999 to
  0.9999 over 100,000 updates. NLP pretraining compute is 16 GPUs for 1M updates; GPU-hours are not stated.
- What it changes: the target is a contextualized average of the top blocks rather than one layer. For our 36-block
  model, K = 10 needs the teacher's full forward, so the cost is the highest of the teacher options.
- Cost: about 33% per step (full teacher forward). Implementation about three hours.
- Important scaling: tau = 0.999 has a memory of 1,000 updates, which is half this run's length. Use tau near 0.99 to
  0.996 here, not the paper's schedule.
- Sketch:

```python
class Data2VecTopK:
    """EMA teacher over the clean window; target is the mean of normalized top-K block outputs at masked positions."""
    def build(self, model, cfg):  # build: EMA copy of the whole model; K = 10 of 36
        self.teacher = ema_copy(model)
        self.K = cfg.topk
    def forward(self, clean_tokens):  # forward: full no-grad teacher pass, about 33% of a step
        with torch.no_grad():
            hs = self.teacher.all_block_outputs(clean_tokens)
        return torch.stack([F.layer_norm(h, (768,)) for h in hs[-self.K:]]).mean(0)  # [B, L, 768]
    def loss(self, s_pred, tgt, mask):  # loss: smooth L1 at masked positions
        return F.smooth_l1_loss(s_pred[mask], tgt[mask])
    def post_step(self, model):  # post_step: EMA update with tau near 0.99 to 0.996 (not 0.999)
        ema_update(self.teacher, model, tau=self.tau)
    def schedule(self, frac):  # schedule: tau ramps from 0.99 to 0.996 over the run
        self.tau = 0.99 + 0.006 * frac
```

- Risk: the GLUE gain is within noise, and the full teacher pass is the most expensive target on this list. Keep it
  for a later pass if idea 5 shows a gain.

### 7. Next-layer chain prediction with stop-gradient targets

- Source: NextLat (2511.05963, Sec. 3.2), applied layer-wise instead of over time. Nothing tested for a masked
  protein model; the hierarchical self-distillation result (2503.03008) is the nearest encoder evidence.
- Evidence: none on this task. In NextLat the one-step latent dynamics model adds a stop-gradient latent term to the
  next-token loss at about 12% step cost (Table 2), and matches next-token performance on TinyStories (figure only).
  Treat as a design to screen, not a result.
- What it changes: a small linear predictor per block predicts the stop-gradient output of block l+1 at masked
  positions from block l. This is deep supervision in representation space, a cheap cousin of idea 3.
- Cost: 36 linear predictors of 768 by 768 on 15% of positions is about 1.6% of block multiply-accumulates (my
  arithmetic). Parameters sit outside the 36 blocks, which the rules allow. Implementation about two hours.
- Sketch:

```python
class NextLayerChain:
    """Each block's masked-position state predicts the stop-gradient state of the next block."""
    def build(self, model, cfg):  # build: one linear predictor per block, outside the blocks
        self.preds = nn.ModuleList([nn.Linear(768, 768, bias=False) for _ in range(35)])
    def forward(self, hs):  # forward: pairs of adjacent block outputs; target detached
        return [(hs[l], hs[l + 1].detach()) for l in range(35)]
    def loss(self, pairs, mask, lam):  # loss: cosine at masked positions, averaged over layers
        terms = [(1 - F.cosine_similarity(p(a[mask]), b[mask], dim=-1)).mean() for p, (a, b) in zip(self.preds, pairs)]
        return lam * torch.stack(terms).mean()
    def optimizer(self):  # optimizer: predictors with weight decay off
        return [{"params": self.preds.parameters(), "weight_decay": 0.0}]
    def schedule(self, frac):  # schedule: constant after a 10% ramp
        return self.cfg.lam * min(1.0, frac / 0.1)
```

- Risk: adjacent blocks are close by design, so the target is easy and the gradient may be small. It may also pull
  the value residual's early-block signal toward later blocks. Screen with a short pilot and the step-time check.

### 8. Two-view latent consistency with a lower-noise view

- Source: LLM-JEPA, Huang, LeCun and Balestriero, arXiv 2509.14252 (2025), read from v2; DLLM-JEPA, arXiv 2606.00091
  (2026, secondary summary only, numbers unverified); the lower-noise teacher branch in Self-Representation
  Alignment (arXiv 2505.02831, ICLR 2026, abstract only).
- Evidence: LLM-JEPA fine-tuning with the cross-view term, Llama-3.2-1B: GSM8K 32.36 to 36.36 (Table 13), NL-RX-SYNTH 57.29
  to 71.46 (Table 12). From random init on NL-RX-SYNTH it went 54.38 to 60.59 (Table 2). The views are natural
  pairs (text and code, question and answer), which proteins lack. DLLM-JEPA uses two masking rates of one input and
  reports 33% fewer FLOPs than LLM-JEPA's two-view form (secondary summary).
- What it changes: a protein has no natural second view. Two candidates: the same sequence at 15% and at a second
  mask rate, or the two halves of one 1,024-token window (a crop pair). The predicted view takes the stop-gradient
  lower-noise state of the other.
- Cost: a second forward. LLM-JEPA reports compute roughly doubling and mitigates it by random loss dropout
  (Table 6: dropout 0.75 with lambda 4 reached 73.08 at the final compute point). Under the clock that is +33% to
  +100% step time.
- Implementation about four hours, including the choice of views and the dropout schedule.
- Sketch:

```python
class TwoViewConsistency:
    """Predict the stop-gradient state of a second view (other mask or clean) at the first view's masked positions."""
    def build(self, model, cfg):  # build: predictor; second view is clean or re-masked at cfg.view_b_rate
        self.pred = SwiGLU(768, 2048, layers=1)
        self.view_b_rate = cfg.view_b_rate
    def forward(self, model, view_a, view_b):  # forward: second pass, no grad on view b
        h_a = model.hidden(view_a)
        with torch.no_grad():
            h_b = model.hidden(view_b)
        return h_a, h_b
    def loss(self, h_a, h_b, mask_a, lam):  # loss: cosine after layer norm
        p = F.layer_norm(self.pred(h_a[mask_a]), (768,))
        t = F.layer_norm(h_b[mask_a], (768,))
        return lam * (1 - (p * t).sum(-1)).mean()
    def schedule(self, frac):  # schedule: loss dropout as in LLM-JEPA, keep rate falling with the clock
        return self.cfg.lam * min(1.0, frac / 0.3)
```

- Risk: the two views share most of their information, so the consistency term may be satisfied by copying the
  context; the gains that LLM-JEPA reports come from views with different content. The cost is the largest on the list,
  so this is a last resort.

## Considered and dropped

| Idea | Source | Reason dropped |
|---|---|---|
| I-JEPA multi-block span targets | arXiv 2301.08243 (CVPR 2023): ViT-H/14 linear probe 79.3 at 300 epochs against MAE 77.2 at 1,600 epochs; under 1,200 GPU hours | Span masking is fixed out by the catalog's masking rule; the target encoder processes the full input, so no step saving; vision only |
| V-JEPA 2 | arXiv 2506.09985 (2025) | Video; no sequence evidence |
| Self-Representation Alignment | arXiv 2505.02831 (ICLR 2026) | Diffusion transformers on images; no number retrieved; the lower-noise branch is folded into idea 8 |
| Bootleg, self-distilled hidden layers | arXiv 2603.15553 (2026) | Vision; one relative figure ("+10%" against I-JEPA); no method details in the abstract |
| Two-mask agreement, masked autoencoder self-consistency | arXiv 2302.14431 (2023) | Vision; no number retrieved; two forward passes |
| Self-distillation for data-scarce LM pretraining | ICML 2026 workshop page | The regime is data-scarce against Chinchilla. This run sees about 92M to 141M tokens of 393M (64 windows of 1,024 tokens times 1,400 to 2,160 steps), so it is compute-bound, not data-bound |
| Hidden-layer distillation from a self-trained teacher | arXiv 2605.11513 | Needs a teacher pretrained on 106B tokens; the hidden-state gain over KD was 0.002 to 0.005 |
| 3Di structure latents as targets | catalog, `ideas/README.md` (outside the rules) | Changes the objective; blocked until Logan opens it. Probably the most informative latent target for proteins, so raise it with him |

## Cross-cutting findings

- The auxiliary gain on a metric is rarely the gain on MLM CE. ProteinJEPA's latent term raised MLM CE by about 0.002
  while improving downstream tasks; the exit heads' gains were on retrieval, with a final-layer gain of 0.1 points.
  Every idea here must be judged on test CE, with the 0.0015-nat noise band.
- Latent-only objectives collapse (JEPA-only in ProteinJEPA, median -0.124). Every latent term must ride on MLM and
  carry an anti-collapse term (stop-gradient plus predictor, or SIGReg, or an EMA with a predictor).
- Under 600 s, teacher costs are a step tax. An EMA or clean forward costs 17% to 33% of the steps, and a frozen small
  teacher costs about 1%. The cheapest latent terms (ideas 2 and 3) cost almost nothing per step but have the weakest
  evidence; the frozen teacher (idea 1) has the best combination of cost and headroom, subject to the leakage check.
- The EMA horizon must match the run. A tau of 0.999 is a 1,000-step memory, half the record's length. Use a shorter
  memory (for ProteinJEPA's schedule, 250 to 2,000 steps; see idea 5).

## Measure before building

1. Price the step tax. Fit the record's val loss against step count from `runs.jsonl` (the per-eval points) over the
   last 40% of the clock; a 17% step tax costs about 0.19 times that slope in log steps. This turns idea 5's break-even
   into a number.
2. Leakage: measure sequence identity between the test split and UniRef50 (or the teacher's training set) before any
   run with idea 1, 4 or 6. Escalate the result to Logan; he decides whether a teacher trained on that set is allowed.
3. Compile: one 60 s pilot per idea with its graph, against the 180 s cap; record the compile seconds with the
   genome (`search.COMPILE_SECONDS`).
4. Step time: a 60 s pilot per idea for step cost, so the proposers can price it in the genome.
