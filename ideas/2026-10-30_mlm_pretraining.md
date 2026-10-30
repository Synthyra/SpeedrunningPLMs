# MLM pretraining ideas for the 10-minute speedrun, 2024 to 2026 literature

Date: 2026-10-30

## Bottom line

The 2024 to 2026 encoder literature moved hard on masking rate (20 to 50 percent in the recent encoders checked), gated MLPs,
and local attention layers. Those are fixed or unlikely to help here. Inside the rules, the sourced gains are small:
Muon+ (an optimizer normalization step, decoder-only evidence) and a per-head attention temperature (SSMax, causal
evidence, training loss only). No source in the window reports a clear masked-LM gain at 198M parameters on a
10-minute clock. Expect the best ideas to land inside the 0.0015 nats band that one seed resolves, so each needs two
seeds before it counts.

Shorthand used below: MLM (masked language modeling), CE (cross entropy), GLUE (General Language Understanding
Evaluation, the encoder benchmark suite), BLiMP (a grammaticality benchmark), SSMax (Scalable-Softmax), AMLM (adaptive
MLM), ELBO (evidence lower bound), CGLS (Curriculum-Guided Layer Scaling), NS (Newton-Schulz orthogonalization inside
Muon). Logan's rule for this file: 15 percent masking, MASK-only replacement, the 20-way loss, the data, the 36 blocks
with intermediate 2,048 and the 198M parameters are fixed. Ideas that change them are marked out of rules.

## What I read

- Catalogue: `ideas/README.md` in full; `ideas/protein_pretraining.md` and `ideas/architecture_optimization.md` by search
  for masking, dropout, local windows, z-loss and token dropping, so overlaps are flagged.
- Code seams (read only): `strategies/strategy.py` (hook names), `attention.py` (`training_masks`, `Attention._attend`),
  `optimizer.py` (`polar_express`, `Muon.step`, `OptimizerChoice.build_optimizers`), `batch.py`, `schedule.py`.
  `train.py` has no dropout (searched).
- Primary sources fetched (HTML or PDF text, sections quoted): ModernBERT (arXiv 2412.13663), NeoBERT (arXiv 2502.19587),
  Ettin (arXiv 2507.11412), EuroBERT (arXiv 2503.05500), Drop Dropout (arXiv 2505.24788), Muon+ (arXiv 2602.21545),
  SSMax (arXiv 2501.19399), AMLM (arXiv 2510.20475), entropy-aware masking (arXiv 2605.28526), CGLS (arXiv 2506.11389).
- Search summaries only, marked where used: Wettig et al. 2022 (arXiv 2202.08005), token dropping (arXiv 2203.13240 and
  2305.15273), progressive stacking (arXiv 2011.13635), MDLM (arXiv 2406.07524), AF-Muon (NeurIPS 2026 listing, no arXiv
  ID found), perplexity-based data pruning (arXiv 2409.05816).

## Clock facts the estimates use

- Step 0.276 s (record profile): matmuls 109 ms, flex attention 80 ms (55 ms of it backward), Muon 26 ms, RMS norms
  17 ms, ReLU-squared 8 ms.
- Kept runs took 1,412 to 2,157 steps (idea 001 in the README), at 65,536 tokens a step. That is 92 to 141M tokens, a
  third or less of the 393M. Each protein is seen at most once, so no score or cache from an earlier pass can be reused.
- Mean protein length is about 225 residues (393M tokens over 1.75M proteins). A 1,024-token window holds four or five
  proteins, and the document mask already removes cross-protein work.

## Ranked ideas

Ranking is by expected gain per implementation hour on this clock. Ideas 1 to 7 are in rules unless marked. Idea 8 is
out of rules.

### 1. Muon+ normalization after orthogonalization (in rules)

- **Source:** Muon+: Towards Better Muon via One Additional Normalization Step, arXiv 2602.21545 (Feb 2026).
- **Evidence:** decoder-only validation perplexity, seeds not reported. GPT-Small (124M, 3.0B tokens) 29.66 to 27.64.
  GPT-Base (362M, 7.2B tokens) 21.70 to 19.98. LLaMA-130M (134M, 2.2B tokens) 19.06 to 18.65. LLaMA-350M (368M, 6.4B
  tokens) 14.02 to 13.41. The gain holds across three NS variants (Table 6). In its NorMuon table (Table 15), Muon+ 27.91
  against NorMuon with beta2 = 0 at 28.29, on GPT-Small. No MLM experiment.
- **What it changes:** the record's NorMuon already normalizes each neuron after orthogonalization (README family
  `optimizer`, 015). So Muon+ is a replacement or a near relative, not an addition. Two tests: Muon+ `col_row` in place
  of NorMuon's second-moment step, and Muon+ stacked on NorMuon. The paper's best axis changes with model size, so run
  the axis as a field, not a fixed choice.
- **Cost:** 1 hour. Compile: no new graph expected. Step: one norm reduction and one divide per banked matrix, expected
  under 1 ms (the whole Muon step is 26 ms; not measured). Memory: none.
- **Sketch:**

```python
# optimizer.py: a variant field on the Muon branch of OptimizerChoice (family "optimizer", variant "normuon_plus")
def plus_normalize(O: Tensor, axis: str) -> Tensor:  # O: (m, n) orthogonalized update
    rows = O / O.norm(dim=1, keepdim=True).clamp_min(1e-8)  # per row, shape (m, n)
    cols = O / O.norm(dim=0, keepdim=True).clamp_min(1e-8)  # per column, shape (m, n)
    return {"row": rows, "col": cols, "row_col": plus_normalize(rows, "col"), "col_row": plus_normalize(cols, "row")}[axis]
# Muon.step: O = polar_express(M); O = plus_normalize(O, axis); then NorMuon's second moment (or none, per the test)
# Config: plus_axis: str = "col_row" (Muon+ Table 5 and 9), with the update scaled by sqrt(m / n) as the paper writes it
```

- **Risk:** the gain may be what NorMuon already buys (the README says NorMuon is worth four times more at the 028 base
  than at 012, so axis effects may differ). Decoder-only, and the paper's LR sweep picked the same LR for Muon and Muon+
  in four of the five listed rows (Table 9). Expected 0.001 to 0.004 nats, which is the size of one seed band. Needs two seeds.

### 2. Per-protein attention temperature, SSMax style (in rules)

- **Source:** Scalable-Softmax Is Superior for Attention, arXiv 2501.19399 (Nakanishi, Jan 2025).
- **Evidence:** a 162M causal LM (12 blocks, 768 wide, SlimPajama, 419B tokens, 1,024 context). Training loss about 0.008
  lower than softmax (Sec 3.1, Fig 4, no table). No validation loss, no overhead measured, no bidirectional or masked
  experiment. Weak. The learned per-layer, per-head scale adds 144 parameters (its variant b).
- **What it changes:** query i's scores are scaled by s_h times log(n_i), where n_i is the number of keys query i may
  attend to. For us n_i is the same-protein key count inside the window (the document mask defines it). Short proteins
  get a softer scale than long ones: log 225 is 5.4 and log 1,024 is 6.9. That targets the long-protein context gap the
  README names as open. With n fixed to the window, the term reduces to a learned per-head temperature. First check
  whether QK-norm already carries a learnable per-head gain in `attention.py`. If it does, the fixed-n form is already
  in the record and only the per-protein term is new.
- **Cost:** 1 to 1.5 hours. Compile: the per-token log-length tensor has a fixed shape, so no recompile. Step: one
  multiply on Q, expected under 0.5 percent. Memory: one float per token plus 216 scalars (6 heads, 36 blocks).
- **Sketch:**

```python
# No hook covers the attention score, so the edit sits in Attention._attend (attention.py), with the length tensor
# built in AttentionKernel.training_masks from the same input_ids that make the document mask.
def same_protein_lengths(input_ids: Tensor, boundary: Tensor) -> Tensor:  # (b, l) log n_i per query
    doc = boundary.cumsum(-1)  # (b, l) protein index inside each window, from the document boundaries
    counts = torch.zeros_like(doc, dtype=torch.float).scatter_add_(1, doc, torch.ones_like(doc, dtype=torch.float))
    return counts.gather(1, doc).clamp_min(2.0).log()  # (b, l), log of the same-protein length
# Attention._attend: Q = Q * (s[None, :, None, None] * log_n[:, None, :, None])  # s: (heads,), init 1 / mean(log n)
# AttentionKernel.parameter_groups: s joins the AdamW group with no weight decay
```

- **Risk:** the paper's effect is 0.3 percent of training loss, so it may be zero here. The per-protein term is the only
  part with a mechanism for this data; the rest may already be in QK-norm. Document boundaries must match the packer
  (check `prepare.py` before using the cumsum).

### 3. Fixed-count token dropping in middle blocks (in rules)

- **Source:** Token Dropping for Efficient BERT Pretraining, arXiv 2203.13240 (Hou et al., 2022): a 25 percent cut in
  pretraining cost at similar fine-tuning. Revisiting Token Dropping Strategy in Efficient BERT Pretraining, arXiv
  2305.15273 (Zhong et al., ACL 2023): up to 57 percent time saved and up to 1.56 points of average gain over vanilla
  dropping (search summaries, not read in full). I found no 2024 to 2026 token-dropping MLM paper.
- **Evidence:** BERT-base scale on GLUE. Compute saving, not a loss curve. The vanilla method drops tokens by MLM-loss or
  attention importance and the 2023 paper says it hurts semantic tasks. Protein windows are about 85 percent unmasked, so
  the dropped share here would be high (not measured).
- **What it changes:** in blocks 6 to 29 each window keeps every masked residue plus a fixed count K of unmasked residues.
  Fixed K keeps the shapes static, so no recompile. Dropped residues carry their block-6 state through the middle and are
  restored at block 30. Upper bound from the profile: at K = 50 percent over two thirds of the blocks, matmuls save about
  36 ms and attention about 40 ms, a total of 76 ms of 276 ms (27 percent), before gather, scatter and the second block
  mask. That is the largest step-time lever in this list. The cost is unmasked context in the middle of the network,
  which is where the README says use of context is the open gap.
- **Cost:** 4 to 6 hours. Compile: a second block mask on the subset document ids. The single-variant probes compile in
  106 to 152 s against the 180 s cap, so compile is the first thing to measure. Step: predicted 15 to 25 percent less.
  Memory: lower activations in the middle blocks.
- **Sketch:**

```python
# block_input hook: gather at the first dropped block, scatter back before the last one (strategy.py hooks)
class TokenDrop(Strategy):  # family "token_drop" (new module), variant "fixed_k" or "none"
    class Config(StrategyConfig):
        variant: str = "none"; first: int = 6; last: int = 30; keep: float = 0.5
    def block_input(self, model, index, X, states):
        if index == self.cfg.first:  # keep: (b, K) = all masked positions plus top unmasked by ||X||, K fixed per window
            self.idx, self.skipped = select_keep(X, self.masked, self.cfg.keep); return X.gather(1, self.idx[..., None].expand(-1, -1, X.size(-1)))
        if index == self.cfg.last:  # restore: skipped residues keep their block-6 state, kept ones take the middle output
            return self.skipped.scatter(1, self.idx[..., None].expand(-1, -1, X.size(-1)), X)
        return X
# AttentionKernel.training_masks: a second block mask over the kept positions' document ids, used in blocks first..last
```

- **Risk:** the largest chance of a real loss. The middle blocks see fewer unmasked neighbors, and long proteins lose the
  most (the value residual's gain grows with length: 0.025 nats over 500 residues). Also: top-K by norm is a guess; a
  random K is the control and should run first.

### 4. Per-residue adaptive masking (border: needs Logan's word)

- **Source:** Mask and You Shall Receive: Optimizing Masked Language Modeling For Pretraining BabyLMs, arXiv 2510.20475
  (Edman and Fraser, Oct 2025; BabyLM at EMNLP 2025).
- **Evidence:** BabyLM strict-small, a DeBERTa-v2 model at 384 hidden, LAMB, 3 runs, no variance reported. Constant 15
  percent masking: the hard adaptive variant raises the (Super)GLUE finetune average from 68.5 to 70.0 (+1.5) and lowers
  BLiMP from 70.7 to 70.0 (-0.7) (Table 1). The paper also tests a decaying 40 to 15 percent rate, which is out of rules.
  Its n-hot sub-token embeddings have no analog for 20 residues; skip them.
- **What it changes:** each residue type gets a masking weight, updated from its recent accuracy on masked positions:
  the target is 0.15 times (1 minus accuracy), smoothed with lambda = 0.2 every 200 steps, then rescaled so each protein
  keeps a mean rate of 15 percent. The sign on test CE is unknown, and BLiMP's mixed signal says the metric is
  contested.
- **Cost:** 1.5 hours. Compile: none. Step: none; the update is a host operation on 20 numbers every 200 steps.
  Memory: none.
- **Sketch:**

```python
# Needs the corruption in benchmark/data.py (MASK_RATE per residue) to read a weight vector instead of a scalar.
# That file is fixed by AGENTS.md, so this is a benchmark change and a separate commit if Logan approves.
class AdaptiveMask(Strategy):  # family "masking" (new), hooks: build, after_step
    def build(self, model, max_tokens):
        self.rate = torch.full((20,), 0.15); self.hits = torch.zeros(20); self.total = torch.zeros(20)
    def after_step(self, state):  # accumulate per-residue accuracy on masked positions; every 200 steps:
        if state.step % 200 == 0:
            acc = (self.hits + 0.5) / (self.total + 1.0)                   # smoothed accuracy per residue type
            target = 0.15 * (1.0 - acc)
            self.rate = 0.8 * self.rate + 0.2 * target                     # lambda = 0.2
            self.rate = self.rate * 0.15 / self.rate.mean()                # per-protein mean stays 0.15 (approximate)
```

- **Risk:** the effect sizes are small and come from a model trained on about 10M words. Rare residues (W, C, H) get more
  masks and small per-type counts make the accuracy noisy. Needs a clear decision from Logan because it changes a fixed
  benchmark file.

### 5. Entropy-aware masking (border; expected negative on this clock)

- **Source:** Entropy-aware Masking for Masked Language Modeling, arXiv 2605.28526 (Srinivasagan, Hartung and Georges;
  *SEM 2026, May 2026).
- **Evidence:** BERT-base (109M) and a 4-layer BERTlet, pretrained on wikitext-103 (about 30k samples, 25 epochs, LR
  1e-4). GLUE total over 7 tasks: baseline 73.27, teacher-entropy masking 75.02, self-entropy masking with a teacher warm
  start 76.47 (Table 2). A mid-entropy control scores 57.56, which is 15.7 points lower, and I could not explain that
  gap from the text. No compute accounting, two seeds, no variance, and three "Total" values in Table 1 do not match
  their per-task means. The abstract's "5 percent" is not defined.
- **What it changes:** the 15 percent of each protein's residues with the highest predicted entropy are masked (a fixed
  count per protein, not a Bernoulli draw). Teacher-entropy needs a frozen reference protein MLM, which this budget does
  not have. Self-entropy needs a forward pass on the unmasked protein before masking. With each protein seen once, no
  stale score exists.
- **Cost:** 2 hours. Step: one extra no-grad forward per step. A forward is about a third of a training step, so the
  clock gets about 25 percent fewer training steps. Compile: a second graph for the scoring forward, which carries the
  027 warmup risk. Memory: the scoring forward's activations under no-grad.
- **Sketch:**

```python
def entropy_scores(model, input_ids: Tensor) -> Tensor:  # no-grad forward on the unmasked protein
    with torch.no_grad():
        probs = model(input_ids).softmax(-1)                               # (b, l, v)
    return -(probs * probs.clamp_min(1e-12).log()).sum(-1)                 # (b, l) entropy per residue

def top_fraction_mask(scores: Tensor, eligible: Tensor, rate: float = 0.15) -> Tensor:
    k = (rate * eligible.sum(-1, keepdim=True)).round()                    # (b, 1) residues to mask per protein
    ranks = scores.masked_fill(~eligible, float("-inf")).argsort(-1, descending=True).argsort(-1)
    return (ranks < k) & eligible                                          # (b, l) boolean mask
```

- **Risk:** predicted negative: the scoring forward costs about 25 percent of steps, and the paper's gain is not measured
  against compute. The fixed count per protein also changes the sampler, which is a benchmark change.

### 6. Local and global attention layers, the ModernBERT pattern (in rules; catalogued as B4)

- **Source:** Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder (ModernBERT), arXiv 2412.13663 (Warner
  et al., Dec 2024).
- **Evidence (checked in the text):** global attention every third layer with RoPE base 160,000, and a 128-token
  sliding window in the other layers with RoPE base 10,000 (Sec 2.1.2, Table 4). Appendix D says the alternating
  pattern matched all-global downstream at 100B tokens with major speedups, but the fetched text has no numbers.
- **What it changes:** two static block masks from `training_masks`: the same-protein mask, and the same-protein mask
  intersected with |i - j| at most 128. Layers 2, 5, 8 and so on take the global mask. The saving is smaller than it looks.
  Pairs more than 128 residues apart are 9 percent of all pairs in a 225-residue protein and 28 percent in a 512-residue
  one (my arithmetic). Across the mix that is about 2 percent of step time, and the context it removes is the long-protein
  context the record needs.
- **Cost:** 2 hours. Compile: two static masks in one graph, warmup change expected under 10 s (not measured). Step:
  under 2 percent on the median mix. Memory: none.
- **Sketch:**

```python
# AttentionKernel.training_masks returns one mask per layer kind; _attend picks by layer index
def training_masks(self, input_ids: Tensor) -> tuple[BlockMask, BlockMask]:
    same = same_protein_mask(input_ids)                                    # the existing document mask
    local = create_block_mask(lambda b, h, q, k: same(b, h, q, k) & ((q - k).abs() <= 128), ...)  # 128-residue band
    return same, local
# Attention.forward: block_mask = masks[0] if self.index % 3 == 2 else masks[1]
```

- **Risk:** small gain on this length mix, and the ALiBi local bias already on every head (README 012) already prefers
  nearby keys, so the hard band may add little either way. Its main value is the speed case for long proteins, which the
  current data has few of.

### 7. Batch-size warmup with fixed micro-batches (in rules; a train.py loop change)

- **Source:** Seq vs Seq: An Open Suite of Paired Encoders and Decoders (Ettin), arXiv 2507.11412 (Weller et al., Jul 2025,
  revised Mar 2026). Its trapezoid schedule has a batch-size warmup per model size (Table 1: 125B tokens at 17M, 10B at 400M,
  3B at 1B). No ablation of the batch-size warmup alone is in the fetched text.
- **Evidence:** none isolated. The record's own run 008 (32 windows a step, 1.8 times the steps) was behind from 180 s,
  which argues against smaller batches on this clock. An early-small, late-full mix is different, but untested.
- **What it changes:** the first 5 percent of the clock runs 16 windows a step, ramping to 64, built from fixed
  16-window micro-batches so every compiled shape is the same. Early updates per token rise by 4. Cost per token rises
  too: Muon's 26 ms is paid per optimizer step, so a 16-window step costs about 1.3 times as much per token as a
  64-window step. The net sign is uncertain.
- **Cost:** 1.5 to 2 hours. It touches the loop in `train.py` (outside the strategy registry), so it is a deliberate
  change. Compile: none if the micro shape is fixed. Step: one micro-step is a quarter of a 64-window step.
- **Sketch:**

```python
class BatchRamp(Strategy):  # family "batch" (batch.py), variant "ramp" or "none"
    class Config(StrategyConfig):
        variant: str = "none"; start_windows: int = 16; ramp_end: float = 0.05
    def windows_per_step(self, progress: float) -> int:     # 16 at 0, 64 from ramp_end on
        if progress >= self.cfg.ramp_end: return 64
        return int(16 + (64 - 16) * progress / self.cfg.ramp_end)
# train.py: accumulate windows_per_step // 16 micro-batches of shape (16, 1024) per optimizer step
```

- **Risk:** likely small or negative; the LR for Muon was tuned at 64 windows and would need a check at 16. Logan should
  see the loop change before it lands.

### 8. Masking rate 30 percent, decaying to 15 (OUT OF RULES)

- **Source:** ModernBERT, arXiv 2412.13663 (30 percent pretraining, Sec 2.2.2, checked in the text); Ettin, arXiv
  2507.11412 (30 percent pretraining, 15 percent in the decay phase, footnote 6); EuroBERT, arXiv 2503.05500 (50 percent
  pretraining, 10 percent in annealing; the ablation is in a figure); NeoBERT, arXiv 2502.19587 (20 percent, Sec 3.3;
  its M5 run lowers GLUE by 0.7 percent at its scale); Wettig et al., arXiv 2202.08005 (40 percent beats 15 percent for
  BERT-large on GLUE and SQuAD, numbers from a search summary); Chinese ModernBERT, arXiv 2510.12285 (15 to 30 to 15, no
  ablation read).
- **Evidence:** fine-tuning metrics, not CE. The direction for test CE on masked protein residues is unknown. A higher
  rate gives more targets per step for the same compute and a harder task, and the fixed 15 percent is in the rules.
- **What it changes:** the training corruption rate becomes a schedule. The held-out evaluation mask must stay at 15
  percent, or the benchmark changes.
- **Cost:** 0.5 hour. Compile: none. Step: none. This changes `benchmark/data.py` and the objective, both fixed. Logan's
  decision, not a search candidate.
- **Sketch:**

```python
# benchmark/data.py (fixed): the training corruption draws Bernoulli(MASK_RATE) per standard residue.
def train_rate(progress: float) -> float:  # OUT OF RULES: needs Logan's word
    if progress < 0.5: return 0.30
    if progress < 0.9: return 0.30 - (0.30 - 0.15) * (progress - 0.5) / 0.4   # linear 30 to 15 percent
    return 0.15
# The held-out mask (benchmark evaluator) keeps MASK_RATE = 0.15.
```

- **Risk:** out of rules. The Wettig and NeoBERT results point in opposite directions at fine-tuning, so the sign is open.

## Rejected, with the reason

- **Label smoothing.** The score is plain CE on the masked residues, so a smoothed target is scored against a biased
  target. No 2024 to 2026 MLM pretraining result found; the Cerebras BERT model has an option that defaults to zero.
- **Focal loss and unlikelihood.** Reweight or penalize away from CE, so out of rules as an objective change. The only
  focal result I found is a 2022 BERT fine-tuning calibration study.
- **Z-loss.** Adds a term to the loss, so out of rules as an objective change. The PaLM result is already in
  `architecture_optimization.md`.
- **Progressive layer growth.** CGLS (arXiv 2506.11389, Jun 2025) is compute-matched only, with no FLOP or wall-clock
  speedup; its own 1B run was about 4 percent slower, and per-stage recompilation is not reported. Its results trail
  the randomized baseline on Pile perplexity at 2.5B tokens. The 2020 stacking claim of more than 110 percent speedup
  (arXiv 2011.13635) I read only from a snippet. The record's 027 went invalid with two static graphs (warmup 249 s).
  Rejected for this clock.
- **Auxiliary order, deshuffling or multi-token heads.** Out of rules (the objective is fixed). The 2025 order-prediction
  result (arXiv 2508.19228) is decoder-only at 340M to 7B. MEXMA's token-level objectives (ACL 2025) are sentence-encoder
  work. No MLM-encoder result found.
- **Gated MLP (SwiGLU) as in NeoBERT.** Out of rules: NeoBERT shrinks the hidden width to 2/3 to hold the parameter
  count, which changes the fixed intermediate width of 2,048.
- **Dropout.** `train.py` has none. Drop Dropout (arXiv 2505.24788, ACL Findings 2025) reports that no dropout gave lower
  MLM loss in a single-pass BERT-base run (Fig 1; the numbers are only in the figure). Already true, no action.
- **Two-stage context extension** (NeoBERT 1,024 to 4,096; EuroBERT's annealing to random crops of 12 to 8,192 with RoPE
  base 250,000). It changes the window the clock is measured on. The short-first curriculum and 512-token windows already
  lost (027, 009). Out of clock.
- **Masked diffusion with an ELBO** (MDLM, arXiv 2406.07524, NeurIPS 2024). Its loss is a weighted average of MLM losses
  over a variable rate (search summary); out of rules, and I did not read its results.
- **Online data selection by per-protein loss** (perplexity-based pruning, arXiv 2409.05816, ICLR 2025; causal LMs). It
  changes which proteins train. No MLM result. Out unless Logan treats sampling as free.
- **Muon for encoders.** No encoder-specific result found. AF-Muon (NeurIPS 2026 listing, no arXiv ID found) reports a
  protein language model gain for tied-embedding models. This model's embedding is untied, so it does not apply directly.
- **ESM-2 token dropout** (rescale embeddings when tokens are masked). A config flag only, per `protein_pretraining.md`;
  untested and small.

## Notes for the catalogue

- `protein_pretraining.md` lists the original ModernBERT masking rate as unverified. It is 30 percent (ModernBERT Sec
  2.2.2, checked for this report). The "open" row can close.
- `architecture_optimization.md` already lists local and global layers as a transfer (B4 and the context table). Idea 6
  here adds the verified layout and the document-length arithmetic that lowers its expected value.
- `ideas/README.md` already puts masking rate outside the rules; this report adds the four encoders' rates for Logan's
  decision (idea 8).

## Limits

- No code was run and no step time was measured. Every cost and speed figure above is arithmetic on the record profile.
- Several sources were read from abstracts or the first 100,000 characters. Items from search summaries are marked.
- The ranking assumes the record's current genome. A change to NorMuon or to the document mask changes ideas 1 and 2.
