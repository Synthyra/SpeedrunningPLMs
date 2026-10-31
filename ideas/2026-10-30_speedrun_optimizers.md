# Speedrun optimizer and architecture ideas, not yet in the catalog

Date: 2026-10-30. A literature scout for SpeedrunningPLMs (198M parameters, 36 blocks, 768 wide, 10 minutes on one
GH200). It checked `ideas/README.md` and the four files beside it, then read primary sources where the tools allowed.
Nothing here is measured on this benchmark. Each gain is a source's number or our estimate, and the table says which.
Shorthand: MLM is masked language modeling; LMO is linear minimization oracle; TPP is tokens per parameter; "nats" is
validation cross entropy in nats; one seed resolves about 0.0015 nats (README).

## Outcome

- Ranked by our estimate of expected loss gain per implementation hour. The top five are SSMax, Peri-LN, xIELU,
  partial RoPE and the copy pointer. Expected gains are small: every estimate is under 0.03 nats and most are under
  0.01.
- Two items I expected to be new are already in the incumbent. The cooldown already decays linearly to zero
  (`schedule.py`, `final_lr_ratio` defaults to 0.0), which is the shape Straight to Zero recommends. Plain decoupled
  weight decay is already a search field (`optimizer.py`, `weight_decay` 0 to 0.2, default 0).
- The main caveat is the token budget. About 2,000 steps of 0.29 s in 600 s, at 32 to 64 windows of 1,024 tokens,
  is 65 to 131 million tokens, or 0.3 to 0.7 TPP. Almost every paper below measures at 2 to 200 TPP, mostly 20 or
  more. One source shows the sign of a schedule effect reversing at 2 TPP. Discount every gain here for this.
- modded-nanogpt: the README records table ends at record 92 (PR 360, 2026-08-30). Nothing later is merged as of
  2026-10-30. PR 379 (copy-pointer head, opened 2026-10-02) is open. The "6.7 s" n-gram claim is rejected (see below).
- FP8 is not a lever at this width. The catalog pilot measured 0.284 s a step against 0.276 s in bf16, and torchao's
  docs (May 2026) say small GEMM shapes on H100 can run slower in FP8.

## Ranked table

Gain is the expected change in validation loss on this benchmark (our estimate, negative is better). "Read" says how
much of the source was read: full, partial (HTML sections), abstract, or snippet.

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats) | Step cost | Evidence strength |
|---|---|---|---|---|---|---|---|
| 1 | SSMax: learnable per-head query temperature scaled by log n | arXiv 2501.19399 (2025) | partial | under 1 | 0 to 0.003 | none | weak: training loss -0.008 at 162M, causal |
| 2 | Partial RoPE: rotate about 10 percent of head channels | arXiv 2603.11611 (2026) | partial | 1 | -0.005 to +0.005 | none | mixed: better at 1B, worse at 8B |
| 3 | Peri-LN: a norm on each sublayer's output | arXiv 2502.02732 (2025) | partial | 1 to 2 | 0 to -0.03 | +5 to 7 percent | strong numbers, out of regime (-0.09 eval loss at 400M, 75 TPP) |
| 4 | xIELU: trainable piecewise activation replacing ReLU squared | arXiv 2411.13010 (2024) | partial | 2 to 3 | -0.005 to -0.015 | +1 to 3 percent | moderate: -0.014 at 1.1B, 125B tokens; -3.3 percent perplexity at 4B tokens |
| 5 | Muon+: one normalization after polar orthogonalization | arXiv 2602.21545 (2026) | abstract | 0.5 to 1 after the full text | unknown | none | weak: 29 to 37 percent claims from a snippet, metric unknown |
| 6 | Uniform update RMS across shapes (Moonlight convention) | arXiv 2502.16982 (2025) | partial | under 1, plus a learning-rate re-search | -0.005 to +0.005 | none | moderate at 16B MoE; no MLM test |
| 7 | AdaMuon: element-wise second moment on the Muon update | arXiv 2507.11005 (2025) | abstract | 2 | 0 to -0.005 | +1 to 2 percent | author claims only |
| 8 | Scion: sign and spectral norms per layer, no AdamW on embedding | arXiv 2502.07529 (2025) | partial | 2 to 3 | 0 to -0.01 | none | moderate: 3B, Muon 2.909 vs Scion 2.890 |
| 9 | CPLM copy-pointer head over visible residues | GitHub PR 379 (2026, open) | PR page | 8 to 12 | 0 to -0.005 | +1 percent | weak: equal loss at 12 percent fewer steps, p = 0.66 |
| 10 | Newton-Muon: input-covariance preconditioned Muon | arXiv 2604.01472 (2026) | partial | 4 to 6 | 0 to -0.005 | +2 percent | weak: one 124M run, single seed; total time +1.8 percent |
| 11 | Dion3: orthogonalize a fraction of momentum rows | arXiv 2608.11612 (2026) | snippet | 3 to 4 | 0 to -0.003 | -5 to -7 percent | secondary only |
| 12 | Spectral clipping of the update (SPECTRA, Musec) | arXiv 2603.14315; arXiv 2609.11655 (2026) | abstract | 2 to 4 | unknown | +1 to 2 percent | abstract only, no numbers |
| 13 | Weight-decay timescale parameterization | arXiv 2405.13698 (2024, ICML 2025) | excerpt | under 1 | under 0.002 | none | a rule; no numbers read |

Cost is implementation hours for this setting. Step cost is our estimate from the catalog profile (a 276 ms step, of
which matmuls 109 ms, flex attention 80 ms, banked Muon 26 ms, RMS norms 17 ms, ReLU squared 8 ms). Compile cost is
not listed per row: each new module adds a few to twenty seconds of compile, and the 180 s warmup cap applies.

## 1. SSMax: learnable per-head query temperature scaled by log n

- Source: arXiv 2501.19399, "Scalable-Softmax" (2025). Read: partial.
- Evidence: 162M causal LM with 12 layers, hidden size 768 (our width), 1,024-token training length, batch 2M
  tokens per update, about 419B tokens in the learning-curve run. Training loss is 0.008 lower than the standard
  transformer. Long-context test loss stays lower to about 10 times the training length. The paper gives no GPU
  cost. It adds 144 parameters (one per layer and head). The bias variant lowered early loss but hurt long-context
  retrieval, so do not add the bias.
- What changes here: QK-norm already fixes the query and key norms, so softmax sharpness is set only by a constant
  scale. SSMax multiplies the query by s times log n. At 1,024 the multiplier is 1 by construction. At 2,048 residues
  it is log 2048 / log 1024, about 1.10, a 10 percent sharper softmax. That is a small change at our eval lengths, so
  expect little unless evaluation goes far past 1,024 tokens. The catalog's length bins are flat (2.746 to 2.762),
  which suggests long-protein cost is not large in the current model.
- Expected cost: under 1 hour. No step or compile cost (a scalar scale per head). No memory cost.
- Sketch (names follow `strategies/strategy.py`; `queries` is not yet a hook, so this needs one or a change inside
  `attention.py`):

```python
# ssmax.py: per-head query temperature s * log(n) / log(n_ref), n = keys visible to the query (arXiv 2501.19399)
class SSMax(Strategy):
    def build(self, model, max_tokens):          # one scalar per block and head; s = 1 / log(n_ref) so the start matches the record
        self.scale = nn.Parameter(torch.full((num_blocks, heads), 1.0 / math.log(self.n_ref)))
    def parameter_groups(self, model):           # a scalar group on the Adam rate, no weight decay
        ...
    def queries(self, index, Q, n_keys):         # NEW hook; Q is (B, H, T, D) after QK-norm and RoPE
        return Q * (self.scale[index] * math.log(n_keys))[None, :, None, None]
```
- Risk: the gain is near the noise band (0.0015 nats per seed) and the paper's large claims are about 10 times
  extrapolation. With document masks, "n" must be defined per query (the document length) or fixed (the window). Test
  both. Our evidence for the mechanism is one 162M causal run.

## 2. Partial RoPE: rotate about 10 percent of head channels

- Source: arXiv 2603.11611, "Fractional Rotation, Full Potential? Investigating Performance and Convergence of Partial
  RoPE" (March 2026). Read: partial (HTML sections).
- Evidence: fractions 0 (NoPE), 1, 4, 10, 25, 50, 75 and 100 percent. LAMBADA perplexity at 2,048 tokens, zero shot:
  Llama-3.2-1B on FineWeb, 10 percent 19.89 versus full RoPE 20.26 and NoPE 26.03; Llama-3.1-8B, 10 percent 8.25
  versus full 7.57 (worse by 9 percent) and NoPE 8.91; Pythia-1B on FineWeb-Edu, 10 percent 74.64 versus full 77.89.
  Training curves at 10 percent or more look like full RoPE. The study is causal decoders trained on 100B-token
  subsets. It has no bidirectional or MLM test.
- What changes here: each head already has a fixed local bias (`positional`, `alibi`). Partial RoPE removes positional
  phase from most channels, so those channels match on content only. Head dimension is 128, so 10 percent is about 12
  channels. The sources read do not say which channels should keep the rotation. arXiv 2502.11276 finds the last (low
  frequency) channels matter for long-distance retrieval, so test rotating the first 12 and the last 12.
- Expected cost: 1 hour. No step or compile cost. The rotation table shrinks.
- Sketch (a variant in `positional.py`, not a hook):

```python
def partial_rope(Q, K, cos, sin, rotary_dims: int):   # Q, K: (B, H, T, D); cos, sin: (1, 1, T, rotary_dims)
    Qr, Qp = Q[..., :rotary_dims], Q[..., rotary_dims:]   # the rotated and the position-free channels
    Kr, Kp = K[..., :rotary_dims], K[..., rotary_dims:]
    Qr = Qr * cos + rotate_half(Qr) * sin                  # standard RoPE on the first rotary_dims channels only
    Kr = Kr * cos + rotate_half(Kr) * sin
    return torch.cat([Qr, Qp], dim=-1), torch.cat([Kr, Kp], dim=-1)
# config: rotary_fraction in [0.1, 1.0] added to the positional family's BOUNDS; build() sizes cos and sin to rotary_dims
```
- Risk: the 8B result is worse, and the evidence is causal. Our model is bidirectional, so the position-free channels
  get no ordering signal beyond the local bias.

## 3. Peri-LN: a norm on each sublayer's output

- Source: arXiv 2502.02732, "Peri-LN: Revisiting Normalization Layer in the Transformer Architecture" (2025). Read:
  partial. Gemma 2 and OLMo 2 use the placement.
- Evidence: 400M, 1.5B and 3.2B parameters (excluding embeddings), 30B tokens of DCLM-baseline, sequence 8,192, Adam,
  cosine schedule. Evaluation loss on C4: 400M Pre-LN 3.43 versus Peri-LN 3.34; 1.5B 3.29 versus 3.18; 3.2B 3.20 versus
  3.11. Gradient-norm spikes were rare under Peri-LN. The paper reports no compute overhead.
- What changes here: our blocks are pre-norm, with zero-initialized output projections, QK-norm and a value residual.
  The paper's Pre-LN baseline has neither QK-norm nor a value residual, so the gap may shrink. Two specific risks.
  (a) RMSNorm of an exactly zero branch is undefined in the limit, so the output projections must not be zero at
  initialization. (b) Peri-LN sets each branch's size from its norm gain, not from the weights, so the gain should start
  at zero to keep the record's start.
- Expected cost: 1 to 2 hours. Step: RMS norms are 17 ms of 276 ms, so doubling them adds about 17 ms, or 6 percent
  (estimate). Memory negligible. Compile: a few seconds more (estimate).
- Sketch (hooks: `sublayer_output`, `initialize`, `build`, `parameter_groups`):

```python
class PeriLN(Strategy):
    def build(self, model, max_tokens):          # one gain per block and sublayer, zero at the start (the record's zero branch)
        self.out_gain = nn.Parameter(torch.zeros(num_blocks, 2))   # attention, MLP
    def initialize(self, model):                 # keep output projections nonzero: RMSNorm of an exact zero is undefined
        for block in model.blocks: nn.init.normal_(block.attn_out.weight, std=0.02); nn.init.normal_(block.mlp_down.weight, std=0.02)
    def sublayer_output(self, index, sublayer, Y):
        return rms_norm(Y) * self.out_gain[index, SUBLAYER_INDEX[sublayer]]
    def parameter_groups(self, model):           # the gains on the scalar group, no weight decay
        ...
```
- Risk: the zero-initialized output projection in the record is replaced by a small random one, which changes step 0
  and is the first thing to check against the record's own run.

## 4. xIELU: trainable piecewise activation replacing ReLU squared

- Source: arXiv 2411.13010, "Deriving Activation Functions Using Integration" (Huang and Schlag, 2024). Read: partial,
  with the formula and Table 2.
- Formula: for x > 0, alpha_p x^2 + x/2; for x <= 0, alpha_n (exp(x) - 1) - alpha_n x + x/2. Alpha_p and alpha_n are
  trainable per layer, beta = 0.5 is fixed, and both start at 0.8. Alpha_p is kept positive by softplus; alpha_n is
  softplus plus beta.
- Evidence: Llama models on FineWeb-Edu, 125B tokens. 1.1B: SwiGLU 2.353 nats, ReLU squared 2.337, xIELU 2.323 (-0.014
  against ReLU squared). 3B: SwiGLU 2.273, xIELU 2.258 (-0.015 against SwiGLU; no ReLU squared at 3B). At 1.1B on 4B
  tokens (3.6 TPP, the closest point to our regime): perplexity 17.52 for xIELU, 17.58 for SwiGLU, 18.12 for ReLU
  squared, so -3.3 percent against ReLU squared.
- What changes here: it replaces the record's ReLU squared at the `mlp_hidden` hook. The record's ReLU squared is a
  fused kernel, so an unfused xIELU costs more per step. The negative branch uses exp, so it needs care in fusion. The
  `backprop_free` family's `unit_select` ranks units by activity under ReLU squared, and xIELU has a nonzero negative
  response, so that ranking changes. Retest unit selection with xIELU rather than reusing its evidence.
- Expected cost: 2 to 3 hours. Step: +1 to 3 percent with compile (estimate). Memory: two scalars per block.
- Sketch:

```python
class XIELU(Strategy):
    def build(self, model, max_tokens):          # raw parameters per block; effective alpha_p, alpha_n start at 0.8
        self.raw_p = nn.Parameter(torch.full((num_blocks,), inv_softplus(0.8)))
        self.raw_n = nn.Parameter(torch.full((num_blocks,), inv_softplus(0.8 - 0.5)))   # alpha_n = softplus(raw_n) + beta
    def parameter_groups(self, model):           # scalar group, no weight decay
        ...
    def mlp_hidden(self, index, H):              # replaces relu(H) ** 2
        ap = F.softplus(self.raw_p[index]); an = F.softplus(self.raw_n[index]) + 0.5
        neg = torch.clamp(H, max=0.0)            # clamp keeps expm1 finite on the unused branch
        return torch.where(H > 0, ap * H.square() + 0.5 * H, an * (torch.expm1(neg) - neg) + 0.5 * H)
```
- Risk: the constants are from the paper's text; check them against its code before trusting the init. No MLM evidence.
  The 125B-token gain is out of regime; the 4B-token row is the one to match.

## 5. Muon+: one normalization after polar orthogonalization

- Source: arXiv 2602.21545, "MUON+: Towards More Effective Muon via One Additional Normalization Step for LLM
  Pre-training" (February 2026, revised May 2026). Read: abstract only.
- Evidence (abstract): a one-line fix after polar orthogonalization; no optimizer state; GPT and LLaMA models from
  60M to 7B parameters at up to about 200 TPP; "consistently outperforms Muon in terms of training and validation
  perplexity". The motivation is that polar steps amplify imbalance between column and row norms. The 37.1 percent
  (GPT-Base) and 29.1 percent (LLaMA-350M) figures came from a search snippet, and I could not tell whether they are
  steps or wall-clock. Treat them as unverified.
- What changes here: a cheap ablation. Test it on plain Muon, not on NorMuon: the record's row normalization may
  duplicate the same correction, and the record 015 gain (2.7531 to 2.7495, 0.0036 nats) is NorMuon's.
- Expected cost: 0.5 to 1 hour once the axis and form are read. No step cost.
- Sketch (in `Muon.step`, after the polar call; the exact form must come from the full paper):

```python
U = polar_express(G.lerp(buffer, momentum)).float()
U = muon_plus_normalize(U)     # the one added normalization; axis and form from arXiv 2602.21545 (not read)
```
- Risk: the speedup numbers are not verified, and the axis is unknown.

## 6. Uniform update RMS across shapes (Moonlight convention)

- Source: arXiv 2502.16982, "Muon is Scalable for LLM Training" (2025). Read: partial (body text).
- Evidence: the update is scaled to RMS 0.2 times sqrt(max(m, n)), so Muon can reuse AdamW's learning rate and weight
  decay. Muon needs about 52 percent of AdamW's training FLOPs to match it at compute-optimal settings (the paper's
  "~2x"). The model is a 16B-total mixture of experts (2.24B active) trained on 5.7T tokens. Decoupled weight decay was
  required, and decay on the RMSNorm gains was "crucial" in its appendix.
- What changes here: the incumbent's per-matrix update RMS depends on shape. For square and tall matrices with
  NorMuon's norm restoration and Muon's sqrt(max(1, m/n)) scale, the RMS is 1/sqrt(n) (0.036 for 768 by 768 and
  2048 by 768). For the wide 768 by 2048 matrices it is 1/sqrt(2048), about 0.022. The Moonlight rule gives 0.2 for
  every shape. So the wide down-projection gets about 0.6 times the incumbent's step relative to the square matrices.
  A single `muon_lr` cannot express that relative change. The rule also needs about 5 times smaller `muon_lr` than the
  incumbent's (0.02 to 0.08 searched), so the search range does not reach it.
- Expected cost: under 1 hour of code, plus a `muon_lr` search in roughly 0.002 to 0.005. Plain decoupled decay is
  already a field.
- Sketch (in `Muon.step`):

```python
scale = 0.2 * max(members[0].shape[0], members[0].shape[1]) ** 0.5   # was max(1, m/n) ** 0.5; drop NorMuon row restore when testing alone
```
- Risk: no MLM test. The gain is a reparameterization of step sizes, so expect a tie unless the relative shape sizes
  matter, which the search cannot tell us today.

## 7. AdaMuon: element-wise second moment on the Muon update

- Source: arXiv 2507.11005, "AdaMuon: Adaptive Muon Optimizer" (Si, Zhang and Shen; first version July 2025, latest
  December 2025). Read: abstract and search summary.
- Evidence (authors' claims): consistently beats Muon on GPT-2 across sizes and learning rates, and beats Muon and AdamW
  on Qwen2.5 across the trajectory. I found no independent replication. The method keeps an element-wise second moment
  V = beta V + (1 - beta) O squared on the orthogonalized update O, divides by sqrt(V), then rescales each matrix to
  0.2 sqrt(mn) over its Frobenius norm. The record's NorMuon normalizes per row (record 015, 0.0036 nats). AdaMuon
  normalizes per element, an axis this catalog has not tested.
- What changes here: the optimizer state grows from one vector per row to one entry per weight. At fp32 for about
  254M non-embedding parameters, that is about 1 GB, which fits on the GH200. The step cost is element-wise, so
  small.
- Expected cost: 2 hours. Step: +1 to 2 percent (estimate).
- Sketch (in `Muon.step`, replacing the row second-moment lines; state shape changes from (k, m, 1) to (k, m, n)):

```python
second_moment.lerp_(U.square(), 1 - ADAMUON_BETA)        # element-wise, (k, m, n)
U = U / (second_moment.sqrt() + 1e-8)
U = U * (0.2 * math.sqrt(m * n) / U.flatten(1).norm(dim=1)).view(-1, 1, 1)   # RMS-aligned rescale
```
- Risk: element-wise scaling can add noise at a 600 s budget. The authors' gains are unverified here, and the catalog's
  NorMuon gain may not carry over to the element axis.

## 8. Scion: sign and spectral norms per layer

- Source: arXiv 2502.07529, "Training Deep Learning Models with Norm-Constrained LMOs" (Pethick et al., ICML 2025).
  Read: partial (tables and text). Gluon (arXiv 2505.13416) generalizes it; its abstract gives no numbers.
- Evidence: 3B on FineWeb: Adam 3.024, Muon 2.909, Scion 2.890, unconstrained Scion 2.882 (Table 5). 124M: Muon 2.988
  against unconstrained Scion 2.984. Speedrun-style configuration: "slightly lower validation loss (across 3 runs) than
  Muon", no numbers. Large batch 6144 at 124M: matches Muon with about 25 percent less wall-clock. The paper says the
  optimal stepsize transfers across width (a figure; no values read). Norm choices in GPT runs: sign for the first
  layer, spectral for hidden layers, sign for the last. For one-hot inputs, column-normalized, then spectral, then sign.
- What changes here: the record uses AdamW on the embedding and head (rates 3e-3 and 1e-3, from search 022). Scion
  replaces that with a sign LMO and keeps the spectral LMO on the blocks, which is Muon's polar step. The 25-token
  embedding is a small matrix, so a sign step on it is coarse.
- Expected cost: 2 to 3 hours. Memory: one momentum per matrix, less than AdamW's two. Step: the spectral step is the
  existing polar call; the sign step is element-wise.
- Sketch (variant in `optimizer.py`; the per-layer radii are in the paper's Tables 3 and 4, which I did not read):

```python
def lmo_update(G, kind: str):                 # kind: "sign" for embedding and head, "spectral" for block matrices
    if kind == "sign":
        return torch.sign(G)
    return polar_express(G)                   # the spectral LMO is the existing polar call

# step: W <- W - lr * radius[layer] * lmo_update(momentum, kind); radius from the paper's Tables 3 and 4
```
- Risk: the stepsize transfer depends on those radii, which I did not read. The 124M speedrun claim has no numbers.

## 9. CPLM copy-pointer head over visible residues

- Source: GitHub `KellerJordan/modded-nanogpt` PR 379, "Track 1: CPLM on #360" (NathanGodey, opened 2026-10-02, open at
  the time of reading). Read: the PR page.
- Evidence: record 92 baseline 40.575 plus or minus 0.024 s, 1,194 steps, 13 runs, validation cross entropy 3.2765
  plus or minus 0.0016. CPLM: 36.009 plus or minus 0.039 s, 1,050 steps, 8 runs, validation cross entropy 3.2769 plus
  or minus 0.0016 (+0.33 millinats, Welch p = 0.66). Per step the time is 0.0340 s against 0.0343 s, about 1 percent
  more. The time saving comes from the schedule's step count, cut from 1,194 to 1,050. The author chose that count on a
  4-GPU GH200 dev node. On 8 H100s a 1,035-step run came out about 3 millinats higher, unverified.
- Mechanism: a gate, taken from a padding row of `lm_head`, mixes the LM softmax with a pointer head of width 128 that
  attends over the document's earlier tokens with a learned sink. Queries and keys are RMS-normed, with a learned query
  gain. It adds 196,737 parameters and uses a fused Triton kernel.
- What changes here: for MLM the pointer should look at any visible (unmasked) position of the same document, not only
  earlier ones. It places probability on the 20 residue identities seen elsewhere in the window, so it can copy a
  residue from a repeat or a low-complexity segment. The catalog says use of context is the open gap, and this is the
  one candidate here aimed at it.
- Expected cost: 8 to 12 hours (flex attention `score_mod`, the gate and the 20-class scatter). Compile: 10 to 20 s more
  (estimate). Step: about +1 percent, from the PR.
- Sketch (hook `sublayer_output` stashes the final residual, `logits` mixes):

```python
class CopyPointer(Strategy):
    def build(self, model, max_tokens):          # pointer projections of width 128; gate logit near -6 so the start matches the record
        self.wq, self.wk = nn.Linear(D, 128), nn.Linear(D, 128); self.gate = nn.Parameter(torch.tensor(-6.0))
    def sublayer_output(self, index, sublayer, Y):   # stash the final residual at the last block
        if index == num_blocks - 1 and sublayer == LAST: self.final = Y
        return Y
    def logits(self, model, logits):             # log of the mixture; cross entropy on it gives the mixture
        a = masked_softmax(self.wq(self.final) @ self.wk(self.final).transpose(1, 2) / math.sqrt(128), visible_same_doc)  # (B, T, T)
        p_ptr = scatter_add_by_token(a, token_ids, vocab=20)                   # (B, T, 20)
        g = torch.sigmoid(self.gate)
        return torch.log((1 - g) * logits.softmax(-1) + g * p_ptr)
```
- Risk: the 12 percent step saving is one author's run, not significant, and the step count was chosen on a different
  machine. The expected gain is near zero until our own runs show it.

## 10. Newton-Muon: input-covariance preconditioned Muon

- Source: arXiv 2604.01472, "The Newton-Muon Optimizer" (April 2026). Read: partial (method, setup, Table 1).
- Update: W <- W - eta msgn(G (ZZ^T)^-1), with Z the layer inputs. An exponential average K of ZZ^T/N is refreshed every
  k steps (k = 32 for record 4, 16 for record 28), damped by gamma tr(K)/n, inverted, and applied to G before Muon's
  momentum and polar call.
- Evidence: record 4 (one H100, 124M GPT-2, 3.25B tokens, one run per method): final loss 3.2611 for Newton-Muon, 3.2793
  for Muon, 3.3801 for AdamW. Total time 7,443 s against 7,314 s for Muon (+1.8 percent). Time to Muon's loss is about 4
  percent less, from a single run. Record 28 (one L40S, 4 runs each): 3.2739 against 3.2787, total time +0.85 percent.
  Record 17 (3 runs): 2.9183 against 2.9196, not significant.
- What changes here: Z must be captured for every bank, which means hooks on the block and MLP inputs. The down
  projection has a 2048 by 2048 covariance, and its inverse is refreshed every 16 to 32 steps outside the compiled
  graph. The paper's own totals show a net time cost at equal wall-clock, so the gain is not established.
- Expected cost: 4 to 6 hours. Step: +2 percent (estimate).
- Sketch (variant in `optimizer.py`; the capture hook is `block_input`):

```python
def capture(self, index, Z):                     # from block_input or mlp_hidden: Z (B*T, in_dim) for each bank
    self.K[index].mul_(BETA).add_((1 - BETA) * (Z.T @ Z) / Z.shape[0])
def step(self):
    if self.step_count % self.k == 0:            # outside the compiled graph
        self.inv = torch.linalg.inv(self.K + self.gamma * self.K.diagonal().sum() / self.n * eye)
    G = G @ self.inv                             # right-multiply the gradient before momentum and polar
```
- Risk: single-run evidence, and a per-step cost that the authors' own totals show is not paid back at equal time.

## 11. Dion3: orthogonalize a fraction of momentum rows

- Source: arXiv 2608.11612, "Dion3: Full-Stack Orthogonal Updates" (August 2026). Read: Microsoft Research page and
  search summaries only.
- Evidence (secondary): matches or improves Muon's loss while step time falls by up to 6 times. The update
  orthogonalizes only a fraction of momentum rows each step, and reduces to Muon when the fraction is 1. A fraction of
  1/4 is reported to beat NorMuon at 3B to 14B. The 6 times figure is for sharded multi-GPU training.
- What changes here: on one GPU the banked orthogonalization is 26 ms of a 276 ms step (about 9 percent). A quarter
  fraction would save about three quarters of that, or about 7 percent of step time, if the cost is linear in
  orthogonalized rows. That is worth 0 to 0.003 nats by our estimate. The multi-GPU speedups do not apply.
- Expected cost: 3 to 4 hours after the full text. Step: -5 to -7 percent (estimate).
- Sketch (rule from arXiv 2608.11612, which I did not read; placeholder):

```python
rows = select_rows(U_momentum, fraction=0.25)       # the paper's selection rule
U_out = torch.zeros_like(G); U_out[:, rows] = polar_express(G_momentum[:, rows]).float()
```
- Risk: the selection rule and the 1/4 quality claim come from secondary summaries. At 600 s the partial update may
  cost quality that the step saving does not repay.

## 12. Spectral clipping of the update (SPECTRA, Musec)

- Sources: arXiv 2603.14315, "Enhancing LLM training via spectral clipping" (accepted to ICML 2026 per a search
  summary); arXiv 2609.11655, "Musec: MomentUm SpEctral Clipping for Stable Muon-type Training" (September 2026).
  Read: abstracts only.
- Evidence (abstracts, no numbers): SPECTRA applies post-spectral clipping to updates to enforce a spectral norm bound,
  and reports that it "uniformly improves validation loss for various optimizers, including AdamW, Signum, Mars, and
  AdEMAMix". Musec clips momentum singular values above a threshold in place of Muon's flattening of all of them to one,
  and claims greater stability across learning rates and model sizes.
- What changes here: the polar step sets every singular value to about one. A threshold keeps the spectral structure of
  the momentum. Whether that helps at 600 s is unknown, and the abstracts give no loss numbers.
- Expected cost: 2 to 4 hours after the full text. Step: one power iteration per bank for the top singular value,
  +1 to 2 percent (estimate).
- Sketch (in `Muon.step`, after the polar call):

```python
sigma = top_singular_value(U, iters=5)           # power iteration; not in the catalog's profile
U = U * torch.clamp(TAU / sigma, max=1.0)        # clip the largest singular direction at TAU
```
- Risk: the threshold rule is not known, and it overlaps with NorMuon's row restore.

## 13. Weight-decay timescale parameterization

- Source: arXiv 2405.13698, "How to set AdamW's weight decay as you scale model and dataset size" (Wang and Aitchison,
  2024, ICML 2025). Read: excerpt.
- Evidence (excerpt): AdamW's weights are an exponential average of recent updates, with timescale tau = 1/(eta lambda)
  in steps. The optimal timescale in epochs is roughly constant across scales. Under muP, the optimal decay grows with
  width. I did not read the fitted constants.
- What changes here: the search varies `weight_decay` at a fixed learning rate. Parameterizing by tau keeps the
  decay's meaning fixed when `muon_lr` changes. Whether the rule applies to a single pass is unclear: the paper's
  timescale is in epochs, and our run may see less than one pass of its data.
- Expected cost: under 1 hour. No step cost.
- Sketch (in `OptimizerChoice`):

```python
weight_decay = 1.0 / (lr * tau_steps)            # tau_steps in steps, searched in place of weight_decay
```
- Risk: a rule with no numbers read, derived for multi-epoch training. Expect a tie.

## Checked and rejected

- Straight to Zero (arXiv 2502.15938): the best shape in the paper (linear decay to zero) is the incumbent's cooldown
  (`final_lr_ratio` defaults to 0.0; `cooldown_fraction` is searched from 0.1 to 0.8). The paper's 610M result (0.77
  percent) is at 20 TPP, and it reports that decay to zero loses at 2 TPP for 111M. Our 0.3 to 0.7 TPP is below every
  point where it wins. No change.
- Plain decoupled weight decay: already a search field (`weight_decay` 0 to 0.2, default 0).
- Dion (arXiv 2504.05295) and Orth-Dion (arXiv 2605.16341): built for sharded training. Dion's FLOP count per matrix at
  full rank (14.5 m n^2 + 2.17 n^3) is below Muon's (20 m n^2 + 10 n^3), but its QR is 14.5 percent slower in wall-clock
  at 350M, where validation loss is 2.919 to 2.921 against Muon's 2.920 (the paper's Table 6). On one GPU, with banked
  Muon at about 9 percent of the step, there is no gain to take.
- QK-Clip and MuonClip (Kimi K2, arXiv 2507.20534): a per-head clip on the maximum attention logit with threshold 100,
  for a mixture of experts with latent attention. Our QK-norm already bounds the logits, so the clip is redundant. I
  read the section; no gain expected.
- The n-gram hint claim: "NanoGPT Speedrun in 6.7s" (Herman Brunborg, blog and social post, 2026-10-28). It reports a
  retrieval-augmented n-gram hint built over a 37B-token corpus from the web. That is external data, outside this
  benchmark's fixed data. A protein version (retrieving homolog sequences) is also external data. No merged PR was
  found. Not for this benchmark.
- FP8: the catalog's pilot (0.284 s against 0.276 s in bf16) and torchao's docs (small GEMM shapes can be slower in
  FP8 on H100 with cuBLAS). No FP8 path shows a gain at 768 width.
- Adaptive warmup (arXiv 2602.05813, February 2026): the abstract claims it matches or beats tuned warmup for Muon and
  Lion. No numbers read. Warmup fraction is already a searched field (0 to 0.1), so the gain is small.
- muP for Muon (arXiv 2505.02222): transfers learning rates across width. Our width is fixed, so nothing transfers.
- Gluon (arXiv 2505.13416): the abstract says its stepsize theory matches Pethick et al.'s tuned values. No numbers read.
- MONA (arXiv 2605.26842, Nesterov-style Muon, MoE from 1B to 68B) and Hyperball (arXiv 2606.16899, a Frobenius-norm
  wrapper): not read. Listed for the next pass.
- Attention sinks in masked encoders: searched. The sources found are diffusion language models and causal-to-bidirectional
  conversion, with no encoder pretraining numbers. No idea.

## Gaps and read status

- Read in full or in sections: Dion, Scion, xIELU, Peri-LN, SSMax, partial RoPE, Straight to Zero, Moonlight,
  Newton-Muon, the Kimi K2 QK-Clip section, PR 379, and the README records table.
- Abstract or snippet only: Gluon, AdaMuon, Muon+, Musec, SPECTRA, Dion3, Orth-Dion, the adaptive warmup paper, MONA,
  Hyperball. Their numbers are not verified.
- Gradient-noise and second-order tricks beyond Newton-Muon: one search, nothing cheap with a number.
- Warmup length and batch ramps: the catalog's records 46 and 72 cover ramps. No 2025 to 2026 study of warmup length
  for short MLM budgets was found.
- Every number is for language models, almost all causal, and mostly at 20 TPP or more. Our benchmark is bidirectional
  MLM at under 1 TPP. Nothing here is a protein result.

## Sources

- SSMax: https://arxiv.org/abs/2501.19399
- Partial RoPE: https://arxiv.org/abs/2603.11611 ; channel-use note: https://arxiv.org/abs/2502.11276
- Peri-LN: https://arxiv.org/abs/2502.02732
- xIELU: https://arxiv.org/abs/2411.13010
- Muon+: https://arxiv.org/abs/2602.21545
- Moonlight (Muon scalable): https://arxiv.org/abs/2502.16982
- AdaMuon: https://arxiv.org/abs/2507.11005
- Scion: https://arxiv.org/abs/2502.07529 ; Gluon: https://arxiv.org/abs/2505.13416
- CPLM, modded-nanogpt PR 379: https://github.com/KellerJordan/modded-nanogpt/pull/379
- Newton-Muon: https://arxiv.org/abs/2604.01472
- Dion3: https://arxiv.org/abs/2608.11612 ; Orth-Dion: https://arxiv.org/abs/2605.16341 ; Dion: https://arxiv.org/abs/2504.05295
- SPECTRA: https://arxiv.org/abs/2603.14315 ; Musec: https://arxiv.org/abs/2609.11655
- Weight-decay timescale: https://arxiv.org/abs/2405.13698
- Straight to Zero: https://arxiv.org/abs/2502.15938
- Kimi K2 (QK-Clip): https://arxiv.org/abs/2507.20534
- n-gram claim (not adopted): https://brunborg.com/n-gram-hinted-lm
- modded-nanogpt records table: https://github.com/KellerJordan/modded-nanogpt
