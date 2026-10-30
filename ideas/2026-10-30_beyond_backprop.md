# Training signals beyond backprop for the 10-minute protein MLM

Date: 2026-10-30. A literature scout for SpeedrunningPLMs (36 blocks, 768 wide, 2,048 MLP, 15 percent MASK, one GH200,
600 s, about 2,100 steps of 0.29 s). It read `ideas/README.md`, `ideas/backprop_free.md` (Dust, noise tracks, Boost,
Cut, the interpretability post) and `ideas/2026-10-30_speedrun_optimizers.md`, and does not repeat them. Nothing here is
measured on this benchmark. Each gain is a source's number or our estimate, and the text says which. Shorthand: MLM is
masked language modeling; TPP is tokens per parameter; ppl is perplexity; "nats" is validation cross entropy;
one seed resolves about 0.0015 nats (README). Ppl converts to nats as ln(ppl ratio).

## Outcome

- No candidate has a measured gain on this benchmark. The best expected gains are estimates of 0 to -0.008 nats, and
  each is a step-time or averaging change that must clear the record's 0.0015 nat band in a run.
- Yardstick (from the catalog's `001`, where 1.53x steps gave about 0.011 nats): one unit of log steps is worth about
  0.026 nats, so a 20 percent step saving is worth about 0.006 nats. A loss gap of 0.03 nats cancels a 1.5x step gain.
  Every gain below is converted with this slope. The slope comes from a different change, so treat it as rough.
- Forward gradients, forward-forward, zeroth-order and Hebbian updates have no LM-pretraining evidence that survives a
  10-minute clock. The one measured forward-gradient LM result is a clear loss (section "Checked and rejected").
- Local learning has one LM-scale result (SOLO, 2026). On one GPU it is slower and worse by about 0.03 nats.
- Interpretability-driven speedups: no reachable primary source. Nothing here is an interpretability idea.

## Ranked table

Gain is the expected change in validation loss on this benchmark (estimate; negative is better). "Read" says how much of
the source was read. Step cost is our estimate from the catalog profile (276 ms a step: matmuls 109 ms, flex attention 80 ms,
banked Muon 26 ms, RMS norms 17 ms, ReLU squared 8 ms). The 180 s warmup cap applies to every compile.

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats) | Compile | Step cost | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | AMUSE: Muon with schedule-free averaging, no cooldown | arXiv 2605.22432 (2026) | full HTML | 2 | 0 to -0.008 | none, if the forward reads the averaged point | +1 to 2 percent | 720M Llama, 1.51x fewer steps than cosine Muon |
| 2 | Progressive layer dropping (PLD) | arXiv 2010.13369 (2020) | full HTML | 3 | -0.002 to -0.008 | invalid if the whole model recompiles per drop pattern; needs per-block compile | -15 to -25 percent on average | BERT-base: 24 percent less time at matched samples; GLUE 83.2 against 82.1 |
| 3 | Token dropping in middle blocks, fixed K | arXiv 2203.13240 (2022); arXiv 2305.15273 (2023) | abstract; HTML of 2305.15273 | 4 | -0.001 to -0.008 | none with a fixed K | -10 to -20 percent | BERT-base: vanilla -15 percent time, GLUE -0.76; ScTD -9 percent time, GLUE +0.61 |
| 4 | Early-block freezing, progressive unfreezing | arXiv 2312.02194 (2023; vision, summary only) | summary | 2 | 0 to -0.002 | a recompile at each switch: invalid under the cap | about -5 percent of total compute | no LM pretraining source found |
| 5 | Staged depth growth | arXiv 2203.06211 (2022, ICML) | search summary | 8 | -0.002 to -0.01 | a second graph at the switch: invalid (catalog 027) | early steps about half cost | GPT-2: 5 to 30 percent compute (summary; 22 percent headline) |
| 6 | Top-block query restriction (NarrowBERT, SparseQueries) | arXiv 2301.04761 (2023) | full HTML | 5 | 0 to +0.02 net; likely a loss | fixed-K gather; flex mask rework | -25 to -35 percent | 2.34x pretraining speed; MLM loss 1.47 against 1.41 at matched steps |
| 7 | Neuron-level 2:4 activation sparsity | arXiv 2602.06183 (2026) | abstract | 3 to read, 10 or more to build | unknown | kernels; unknown | claimed 1.4 to 1.7x, unverified | abstract only |
| 8 | 2:4 squared-ReLU FFN with FP8 kernels | arXiv 2503.16672 (2025) | full HTML | 10 to 12 | about 0 | custom kernels | FFN up to 1.3x at kernel level | 1.5B on 63B tokens (emulated): 2.652 against 2.651 dense |
| 9 | Shared-readout local learning (SOLO) | arXiv 2609.35440 (2026) | full HTML | 8 | +0.03 to +0.04 (worse) | detached blocks: one graph per module, risky | 0.84x throughput on one GPU at K = 2 | 340M ppl 28.04 to 29.05; 2B 20.93 to 21.57 |

## 1. AMUSE: Muon with schedule-free averaging, no cooldown

- Source: arXiv 2605.22432, "AMUSE: Anytime Muon with Stable Gradient Evaluation" (2026). Read: the HTML, its
  pretraining section and update equations (not the figure values).
- Evidence: Llama-style decoders on FineWeb. 124M (2.1B tokens), 720M (16k iterations, batch 1,984, sequence 512,
  about 16.3B tokens, about 23 TPP) and 1.3B (26B tokens). At 720M it needs 1.51x fewer steps than Muon to reach Muon's
  final perplexity. Baselines use cosine schedules; AMUSE uses linear warmup and no decay. The main text gives no
  perplexity values (only the figure). The vision numbers (1.34x, 2.57x) do not count here.
- Equations, as read. Gradients are taken at Y_t = (1 - b_t) Z_t + b_t X_t; M_t = 0.95 M_{t-1} + 0.05 grad L(Y_t);
  Z_{t+1} = Z_t - eta Muon(M_t); X_{t+1} = (1 - c) X_t + c Z_{t+1} with c = 1/(t+1). b_t = b_init in warmup, then
  b_t = 1 - (T0/t)^rho (1 - b_init); the tuning grid is b_init in {0.4, 0.6}, rho in {0.6, 0.8}.
- What changes here: the record's 40 percent cooldown and NorMuon are the baseline the paper does not test. Expect
  AMUSE to replace the cooldown's role, not add to it. Our 0.3 to 0.7 TPP is far below its 23 TPP. ScheduleFree+
  (arXiv 2605.19095) reports its gains at 1,000 TPP, outside our regime.
- Expected cost: compile 0 s (the forward reads the same parameter tensors); step +1 to 2 percent (estimate: lerps on the
  Muon banks); memory one fp32 copy of the banks, about 1 GB (estimate from about 254M matrix parameters), fine on a GH200.
- Gain estimate: 0 to -0.008 nats, with the top of the range only if the cooldown is switched off in the same run.
- Sketch (hooks in `strategy.py`; the Muon step must act on `z`, not on the live parameter, which is an optimizer.py change):

```python
class AnytimeMuon(Strategy):
    def build(self, model, max_tokens):                       # z: the Muon iterate the optimizer steps; x: the average
        self.z = [w.detach().float().clone() for w in banks(model)]   # banks(): helper, the Muon matrices
        self.x = [z.clone() for z in self.z]
    def after_step(self, state: StepState):                   # X_{t+1}, then Y_{t+1} into the live weights the next forward reads
        c, beta = 1.0 / (state.step + 1), beta_at(state.step)  # beta_at: beta_init in warmup, then the rho rule
        for x, z, w in zip(self.x, self.z, banks(state.model)):
            x.lerp_(z, c); w.data.copy_((1 - beta) * z + beta * x)
    @contextlib.contextmanager
    def evaluation_weights(self, model):                      # score X, the average, then restore Y
        saved = [w.detach().clone() for w in banks(model)]
        for w, x in zip(banks(model), self.x): w.data.copy_(x)
        yield
        for w, s in zip(banks(model), saved): w.data.copy_(s)
```
- Risk: the 1.51x is against cosine Muon at 16B tokens, not NorMuon with a cooldown at 0.3 TPP. The extra state is a
  second copy of each bank and must be kept in fp32 for the average. Test with the cooldown kept and then off.

## 2. Progressive layer dropping (PLD)

- Source: arXiv 2010.13369, "Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping"
  (Zhang and He, NeurIPS 2020). Read: the HTML, its schedule and Table 1.
- Evidence: 12-layer BERT-base, batch 4K, 200K steps on 64 V100s. At matched samples PLD takes 24 percent less wall-clock
  (38.45 h to 29.22 h, 1.3x at checkpoint 186). PLD at checkpoint 100, with 46 percent fewer samples, scores GLUE 82.3
  against the baseline's 82.1 at checkpoint 186, in 2.5x less time. GLUE medians over five inits: PLD at 186 scores 83.2
  against the baseline's 82.1, so per-sample quality held in the paper's setting. The keep ratio starts at 1 and decays to
  theta_bar (0.5 to 0.9) with gamma = 100 / T; the drop probability of block l is (l / L)(1 - keep). The paper's equations
  are inconsistent in notation; its algorithm is the reference.
- What changes here: with 36 blocks, a drop of 25 to 50 percent of blocks in the first half of the clock is a step saving.
  Keep block 0 (it feeds the value residual and the first values) and the last block always. The expected depth at
  theta_bar 0.5 is about three quarters of L.
- Expected cost: compile: a drop pattern changes the graph, so the whole model cannot compile per pattern (the 027 failure
  again). Compile each block once and call or skip it from Python; a 60 s pilot must show no per-pattern recompile. Step
  -15 to -25 percent on average (estimate, from the 24 percent wall-clock at matched samples). Memory down.
- Gain estimate: -0.002 to -0.008 nats; a 20 percent step saving is worth about 0.006 by the slope above.
- Sketch (needs a `skip` decision in the block loop, which is not a current hook):

```python
class LayerDrop(Strategy):
    def build(self, model, max_tokens):
        self.theta_bar, self.gamma = 0.5, 100.0 / total_steps()   # PLD: the keep ratio decays from 1 to theta_bar
    def skip(self, index: int, step: int, num_blocks: int) -> bool:   # the loop calls this per block per step
        if index in (0, num_blocks - 1): return False             # block 0 feeds the value residual; the last block reads out
        keep = (1 - self.theta_bar) * math.exp(-self.gamma * step) + self.theta_bar
        return random.random() < (index / num_blocks) * (1 - keep)  # drop probability, growing with depth
# a skipped block is the identity: X passes through; the value residual (block 0) is never skipped
```
- Risk: a random skip changes the compute path of each batch, so step time is not constant. An identity pass saves no time
  unless the block call itself is skipped. The gain at 600 s is a bet on PLD's time-to-samples ratio, measured on BERT-base,
  not on a 36-block decoder.

## 3. Token dropping in middle blocks, fixed K

- Source: Hou et al., "Token Dropping for Efficient BERT Pretraining" (ACL 2022), arXiv 2203.13240. Read: abstract.
  Follow-up: Zhong et al., "Revisiting Token Dropping Strategy in Efficient BERT Pretraining", arXiv 2305.15273 (2023).
  Read: HTML, Table 2.
- Evidence: Hou et al. report 25 percent less BERT pretraining cost with similar fine-tuning (abstract). Zhong et al.'s
  table (BERT-base, 250K steps, GLUE): baseline 15.17 h and GLUE 81.11; vanilla token dropping 12.92 h (-15 percent) and
  GLUE 80.35 (-0.76); their semantic-consistent variant (ScTD, extra KL passes every 10 steps) 13.78 h (-9.2 percent) and
  GLUE 81.72 (+0.61); at 160K steps, -42 percent time and GLUE 81.28. Their "up to 57 percent" headline is not in the
  table. So the vanilla method costs quality; the 25 percent figure is the optimistic one.
- What changes here: in the middle blocks (say 12 to 23) keep a fixed K per window (K = 0.5 T), chosen as all masked
  residues plus the highest-loss unmasked ones, where loss is the 20-way CE of an intermediate head. The head is a 768 by 20
  matmul, so the score is cheap. Dropped tokens skip the middle blocks and rejoin at the output (Hou et al.: "picked up by
  the last layer"), so the restore is a scatter. Positions (RoPE, local bias) and document ids must be gathered too.
- Expected cost: compile none with a fixed K (one graph); a gather and a scatter per boundary (about 1 ms, estimate); step
  -10 to -20 percent (estimate: 12 blocks at half the tokens is -17 percent of block compute, before attention's quadratic
  saving). Memory down. The ScTD extra pass every 10 steps adds about 2 percent.
- Gain estimate: -0.001 to -0.008 nats. Vanilla dropping may lose more than its saving at 600 s, as the GLUE table shows.
- Sketch (needs `keep` and `restore` hooks, which are not in `strategy.py`):

```python
class TokenDrop(Strategy):
    def build(self, model, max_tokens):
        self.first, self.last, self.k_frac = 12, 24, 0.5           # blocks 12..23 see half the tokens
    def keep(self, index, X, labels, positions):                  # fixed K: masked residues always kept, then highest-loss
        if not (self.first <= index < self.last): return X, None, positions
        ce = intermediate_ce(X, labels)                            # (b, l): the 20-way head at this depth, every position
        ce = ce + 1e4 * is_masked(labels)                          # masked residues forced in; labels hold the original residue
        kept = ce.topk(int(self.k_frac * X.shape[1]), dim=1).indices.sort(dim=1).values
        return gather_tokens(X, kept), kept, gather_tokens(positions, kept)
    def restore(self, index, X_kept, kept, X_pre):                # dropped tokens keep their pre-drop state; rejoin at the output
        return X_pre.scatter(1, kept[..., None].expand_as(X_kept), X_kept)
```
- Risk: the document mask (flex) must index the gathered tokens, and the data path must put the original residue in
  `labels` at every position (the loss uses masked positions only). Vanilla dropping lost 0.76 GLUE in the one table read.

## 4. Early-block freezing, progressive unfreezing

- Source: no LM pretraining source found. Vision evidence: arXiv 2312.02194, "Local Masking Meets Progressive Freezing"
  (2023), read as a search summary only; it reports a per-iteration speedup from freezing front to back, proportional to
  the frozen layers' cost. "Freeze Deep, Train Shallow" (arXiv 2605.11416, 2026) is continued pretraining and was not read.
- What changes here: freezing blocks 0 to 11 for the first quarter of the clock removes their backward pass, about 5
  percent of total compute (estimate: a third of the blocks, two thirds of a step, a quarter of the steps). The value
  residual reads block 0's values, so block 0 must stay trainable or have its values stored.
- Expected cost: compile: `requires_grad` changes the graph, so each switch recompiles, which the 180 s cap does not allow
  (the 027 failure). A switch that multiplies by a 0-d tensor saves no compute. Step -5 percent (estimate). Memory down.
- Gain estimate: 0 to -0.002 nats, most likely a tie.
- Sketch (freeze at build, one switch at a fixed step; the switch is the risk):

```python
class FreezeEarly(Strategy):
    def build(self, model, max_tokens):
        self.switch = int(0.25 * total_steps())
        for block in model.blocks[:12]: block.requires_grad_(False)   # frozen from the start, so one graph until the switch
    def after_step(self, state: StepState):
        if state.step == self.switch:                               # a graph change: measure the recompile in a pilot first
            for block in state.model.blocks[:12]: block.requires_grad_(True)
```
- Risk: no LM evidence, and freezing the bottom blocks may cap what the top learns. The always-frozen form compiles once but
  is not progressive; the progressive form needs a pilot that shows the recompile fits the cap.

## 5. Staged depth growth

- Source: Shen et al., "Staged Training for Transformer Language Models" (ICML 2022), arXiv 2203.06211. Read: the search
  summary and the abstract; not read in full.
- Evidence (summary): GPT-2-style models; the abstract's headline is 22 percent compute saving against a baseline trained
  from scratch; the main text gives 5 to 30 percent, measured by validation loss and zero-shot perplexity; some width-only
  configurations were negative. Code is public (allenai/staged-training).
- What changes here: a stage at 18 blocks, then a copy into 36 blocks (depth growth). The first stage costs about half a
  step, so early steps are cheap. The record's early steps cost the same as its late ones.
- Expected cost: compile: a second graph at the switch; the catalog's 027 (two compiled shapes) reached 249 s and was
  invalid. A single graph that takes the stage as an input tensor saves no compute. Step saving about 25 percent over the
  first stage (estimate). Memory the same at the end.
- Gain estimate: -0.002 to -0.01 nats, if it compiles inside the cap.
- Sketch (a stage switch in the training loop; the cap decides):

```python
class StagedDepth(Strategy):
    def build(self, model, max_tokens):
        self.switch = int(0.5 * total_steps())                     # grow at half the clock
    def after_step(self, state: StepState):
        if state.step == self.switch:                              # copy blocks 0..17 into 18..35; the new blocks start empty
            for i in range(18): copy_block(state.model.blocks[i], state.model.blocks[i + 18])
```
- Risk: the new blocks start with empty optimizer state, and the staged gain comes from the cheaper first stage. Compile
  is the decider.

## 6. Top-block query restriction (NarrowBERT, SparseQueries)

- Source: arXiv 2301.04761, "NarrowBERT: Accelerating Masked Language Model Pretraining and Inference" (2023). Read: the
  full HTML, Table 1 and the MLM and GLUE numbers.
- Evidence: BERT-base, 512-token sequences, 1,728 sequences a step, 70k steps, 8 A100s. Pretraining speedup against
  BERT: {2 full, 10 restricted} 2.34x; {1, 11} 2.47x; ContextFirst (restricts only the FFN) 1.33x. End-of-training MLM loss:
  BERT 1.41; ContextFirst 1.43; {2, 10} 1.47. GLUE: {2, 10} is within about 3 points on each task. The paper gives no
  matched-step table beyond the end of training.
- What changes here: the record's loss is on masked residues only, so in the top blocks the queries need only the masked
  positions, while keys and values stay full. The objective is the same as the record's. The catch is the loss: 0.06 nats
  at matched steps is far more than any step gain here (0.006 for 20 percent, by the slope above).
- Expected cost: compile: a fixed-K gather (K = masked count per window, exact if the corruption is exact, otherwise padded)
  keeps one graph; the flex document mask must index the query subset. Step -25 to -35 percent for the top 12 blocks
  (estimate). Memory down.
- Gain estimate: 0 to +0.02 nats net; a loss is the likely result. Test only the ContextFirst-like FFN-only form if the flex
  mask rework is cheap.
- Sketch (hooks `keep`, `attention_output` for the scatter back; full keys and values):

```python
class SparseQueries(Strategy):
    def build(self, model, max_tokens):
        self.first_narrow = 24                                    # blocks 24..35 compute queries at masked residues only
    def keep(self, index, X, labels, positions):                  # K = masked residues per window, fixed by the corruption
        if index < self.first_narrow: return X, None
        return gather_tokens(X, masked_index(labels, K)), masked_index(labels, K)
    def attention_output(self, index, X, Y):                      # scatter the restricted queries back into the stream
        return scatter_tokens(X, Y, self.kept)
```
- Risk: the MLM loss gap of 0.06 is the main risk. Flex attention with a query subset and a document mask is untested here.

## 7. Neuron-level 2:4 activation sparsity (abstract only)

- Source: arXiv 2602.06183, "To 2:4 Sparsity and Beyond: Neuron-level Activation Function to Accelerate LLM Pre-Training"
  (2026). Read: abstract only; the full text was not reachable.
- Evidence (abstract): sparsity on all FFN matmuls, 2:4 weights and v:n:m activation sparsity; sparse steps for most of
  training and dense steps near the end; the FFN up to 50 percent of pretraining FLOPs in larger models; end-to-end 1.4 to
  1.7x with "the same performance on our quality benchmarks"; applicable from A100 on. No dense baseline timings, benchmark
  names or model sizes in what was read.
- What changes here: the record's ReLU squared gives 84 to 98 percent natural activation sparsity (per arXiv 2503.16672).
  The 2:4 tensor-core path needs a 2:4 pattern, so the activation must be pruned to it, not used as is.
- Expected cost: unknown; a kernel build of at least a day; compile unknown; memory down. Read the full text first (3 h).
- Gain estimate: unknown. Not ranked on a number.
- Sketch: not written. The activation pruning would be a new `mlp_hidden` returning a 2:4 pattern, and the down projection a
  custom kernel. Build only after the full text gives baseline numbers.
- Risk: the speedup is unverified and the claims are abstract-level. Our FFN is 768 by 2,048, a small GEMM where catalog
  B7 (FP8) saw no gain (0.284 s against 0.276 s).

## 8. 2:4 squared-ReLU FFN with FP8 kernels

- Source: arXiv 2503.16672, "Accelerating Transformer Inference and Training with 2:4 Activation Sparsity" (2025). Read:
  full HTML.
- Evidence: a 1.5B model on 63B tokens with emulated 2:4 activations: final values 2.651 (dense squared ReLU), 2.652 (the
  2:4 recipe with 5 percent of features dense in the backward pass), 2.682 (naive backward sparsity), 2.919 (no row
  permutation). The table header says perplexity, but the values look like loss; treat the units as unclear. FFN kernel
  speedup up to 1.3x forward and backward, FFN forward up to 30 percent; no end-to-end training number; the accuracy runs
  used 64 H100s (1B) and 256 (7B). The 2:4 FP8 GEMM is 1.5 to 1.7x a dense FP8 GEMM on H100 at large shapes.
- What changes here: the recipe needs FP8 and custom fused kernels. Catalog B7 (FP8 MLP) measured no gain at 768 by 2,048,
  the same shape class. The 2:4 pattern is applied to the squared-ReLU output (about 1 percent of nonzeros dropped). The
  first 1,000 steps are dense.
- Expected cost: a custom kernel build, 10 to 12 hours; compile unknown (the paper does not say); step about 0 at our width
  (estimate); memory about the same.
- Gain estimate: about 0 nats. Ranked 8 because the evidence is at large width.
- Sketch: as in idea 7, a 2:4 pattern on the output of `mlp_hidden` and a custom GEMM for the down projection.
- Risk: the width mismatch and the FP8 overhead the catalog already measured.

## 9. Shared-readout local learning (SOLO)

- Source: arXiv 2609.35440, "SOLO: Pretraining Billion-Parameter Language Models with Shared-Output Local Learning"
  (2026). Read: the HTML, the loss definitions and the tables.
- Evidence: 15B SlimPajama tokens. WikiText ppl, backprop against SOLO with K = 2 and K = 4 modules: 340M 28.04 against
  29.05 and 31.19; 1.3B 21.61 against 22.29 and 23.59; 2B 20.93 against 21.57 and 22.21. In nats, that is about +0.03 to
  +0.04 at K = 2. Throughput on four A100s, 340M: 247k tokens a second for backprop against 208k (K = 2) and 156k (K = 4).
  Memory 29.6 GB against 15.8 GB. The large gain is pipeline-parallel memory (1.44x the best 1F1B throughput), not
  single-GPU speed. The shared readout is a stop-gradient copy of the final head, refreshed each step.
- What changes here: on one GPU with a fixed batch, the memory saving buys nothing; the step is slower and the loss higher.
  It is the one local-learning result at LM scale, so it is the reference for the family.
- Expected cost: 8 hours; detaching blocks and adding a loss per module; compile: one graph per module boundary (risky);
  step about -16 percent at K = 2 (the paper's throughput ratio).
- Gain estimate: +0.03 to +0.04 nats (worse). Do not build for one GPU.
- Sketch (for the record only):

```python
class SharedReadoutLocal(Strategy):
    def block_input(self, model, index, X, states):             # detach at each cut: no backprop across modules
        return X.detach() if index in self.cuts else X
    def auxiliary_loss(self, model, states, labels):            # per module: CE(softmax(tau_k * sg(W) @ phi_k(h_k)), y)
        return sum(local_ce(states[k], labels, self.tau[k], sg(model.head.weight)) for k in self.cuts)
```

## Checked and rejected

- Split forward gradient (arXiv 2607.16612, 2026): the one forward-gradient LM result read. WikiText-103, a 16.1M-parameter
  4-layer causal GPT, one seed, 119M tokens. Ppl: backprop with Adam 149.6; pure forward gradient (K = 4) 2,885; frozen
  trunk 668; Split-FG with trunk step scale 0.03, 387. Split-FG's step is 133.8 ms against 41.7 ms for backprop. Reject: it
  is 2.6x worse than backprop at the same tokens and 3.2x slower a step.
- Forward gradient for images (Ren et al., ICLR 2023, arXiv 2210.03310): vision only (MNIST, CIFAR-10, ImageNet); the
  search found no LM result. A second search cited Fournier et al. (ICML 2023) with the direction of the perturbation
  result disputed between snippets; not used.
- Forward-Forward for language: two searches; no LM result found. Reject for lack of evidence.
- Zeroth-order pretraining: DP zeroth-order pretraining (arXiv 2402.07818) and evolution-strategy pretraining (EA4LLM,
  arXiv 2510.10603, 2025; claims 4x less memory at 4B and above, sample efficiency trails gradient methods). No numbers read.
  At about 2,000 steps the sample cost is the problem. Reject.
- Hebbian updates: the 2025 Hebbian results are a sparse autoencoder on embeddings (ICLR 2025 workshop; reconstruction
  behind backprop), HebbFF (OPT 2025 workshop; vision) and CHT (arXiv 2501.19107; sparse topology regrowth, not Hebbian
  weights). No pretraining-scale LM. Reject.
- Closed-form readout or embedding updates by least squares: nothing at pretraining scale; the catalog's
  `readout_warmstart` (a shared readout trained by backprop) is the nearest.
- Dead-neuron revival: no LM training source found. A 2024 snippet reports dead neurons falling from 23.8 to 3.7 percent in
  GPT-2 small with a neuron-embedding loss (likely arXiv 2411.08166, unverified). A lead for a later read, not a speedrun
  result. Skip now.
- Interpretability-driven speedups ("clean units", concentrated credit): no reachable primary source. METR's April 2026
  NanoGPT note lists 77 records, and the search found none credited to interpretability. The 6.7 s n-gram claim uses
  external data (the catalog rejects it). Skip.
- MONA (arXiv 2605.26842, 2026): Nesterov-style acceleration in Muon's gradient path, MoE from 1B to 68B. Abstract only; no
  loss numbers, and the Nesterov term is not described. Revisit after the full text.
- ScheduleFree+ (arXiv 2605.19095, 2026): 31 percent over WSD at 1,000 TPP; outside our 0.3 to 0.7 TPP.
- Zero or one layer progressive training (arXiv 2511.04981, 2025): depth growth at scale, abstract snippet only. Idea 5
  covers the same question at 10 minutes.

## Hooks the registry does not have

The `strategy.py` hooks cover the forward path, the auxiliary loss, after_backward and after_step. These ideas need more:

- `keep(index, X, labels, positions)` and `restore(...)`: a token subset at block boundaries and a scatter back (ideas 3
  and 6). The positions and the flex document mask must follow the subset.
- `skip(index, step)` in the block loop (idea 2): a block call that is not made, compiled per block.
- An optimizer change (idea 1): the Muon step applies to the iterate `z`, while the forward reads the live parameters.
- A requires_grad switch (idea 4) is a graph change; only a pilot can say whether it fits the cap.

## Gaps and read status

- Read in full (HTML or PDF text): AMUSE, PLD, Zhong et al. (token dropping follow-up), NarrowBERT, 2503.16672 (the
  squared-ReLU 2:4 recipe), Split-FG, SOLO.
- Abstract or summary only: Hou et al. (token dropping), 2602.06183, Staged Training, 2312.02194, 2605.11416, MONA,
  ScheduleFree+, 2511.04981. Their numbers are not verified here.
- No LM-scale source found: forward-forward, zeroth-order pretraining with numbers, Hebbian pretraining, dead-neuron
  revival, interpretability-driven speedups, freezing for LM pretraining.
- Every number is for language models or BERT at 0.3 to 25 TPP, not for protein MLM at 0.3 to 0.7 TPP. Nothing here is a
  protein result.

## Sources

- AMUSE: https://arxiv.org/abs/2605.22432 (HTML: https://arxiv.org/html/2605.22432)
- Progressive layer dropping: https://arxiv.org/abs/2010.13369
- Token dropping: https://arxiv.org/abs/2203.13240 ; Revisiting token dropping (ScTD): https://arxiv.org/abs/2305.15273
- Staged training: https://arxiv.org/abs/2203.06211
- Freezing, vision: https://arxiv.org/abs/2312.02194 ; Freeze Deep, Train Shallow: https://arxiv.org/abs/2605.11416
- NarrowBERT: https://arxiv.org/abs/2301.04761
- Neuron-level 2:4 activations: https://arxiv.org/abs/2602.06183
- 2:4 activation sparsity, squared ReLU: https://arxiv.org/abs/2503.16672
- SOLO: https://arxiv.org/abs/2609.35440
- Split forward gradient: https://arxiv.org/abs/2607.16612
- Forward gradient, vision: https://arxiv.org/abs/2210.03310
- Zeroth-order: https://arxiv.org/abs/2402.07818 ; EA4LLM: https://arxiv.org/abs/2510.10603
- Hebbian and sparse topology: https://arxiv.org/abs/2501.19107
- MONA: https://arxiv.org/abs/2605.26842 ; ScheduleFree+: https://arxiv.org/abs/2605.19095
- Zero or one layer progressive training: https://arxiv.org/abs/2511.04981
- Dead-neuron lead: https://arxiv.org/abs/2411.08166 (unverified)
