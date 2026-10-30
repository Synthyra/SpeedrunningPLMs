# Attention internals and the output head, not yet in the catalog

Date: 2026-10-30. A literature scout for SpeedrunningPLMs (36 blocks, 768 wide, a 2,048 MLP, 15 percent MASK, 20-way
masked cross entropy on the 33-token ESM2 alphabet scored on its 20 standard-residue columns, one GH200, 600 s, about
2,150 steps, record 2.7146 nats). It reads `ideas/README.md`, `2026-10-30_long_protein_attention.md` and
`2026-10-30_mlm_pretraining.md` first and cites the sibling scout files where they already cover an idea. Nothing here
is measured on this benchmark. Each gain is a source's number or our estimate, and the text says which.

Shorthand: MLM is masked language modeling; CE is cross entropy; QK-norm is the RMS normalization of queries and keys
(no learned gain); RoPE is rotary position embedding; ALiBi is the record's fixed per-head local bias; flex is PyTorch's
flex attention; GQA is grouped-query attention; MLA is multi-head latent attention; GELU is the Gaussian error linear
unit; LN is LayerNorm; PR is a GitHub pull request; ppl is perplexity; TPP is tokens per parameter (our run is 0.3 to
0.7); "nats" is validation cross entropy; one seed resolves about 0.0015 nats (README). The yardstick, rough because it
comes from another change, is 0.026 nats per unit of log(steps): a 10 percent step saving is worth about 0.0025 nats,
and a 3 percent saving about 0.0008.

## Outcome

- No candidate has a measured gain here. The best estimates are under 0.005 nats each, so every one of them needs two
  seeds before it counts. The top candidate is a two-layer MLM head (dense, GELU, LN, then the 33-way head), the design
  of ESM-2's own head, at about an hour of work.
- Two facts from our code change the output-head question. Training already scores the 20 columns: `masked_cross_entropy`
  gathers `ESM2_RESIDUE_COLUMNS` before the cross entropy (`benchmark/evaluate.py` line 42, used by `train.py` line 209),
  so the 13 non-standard head rows get no gradient from the loss. The head's RMS norm has no gain (`strategy.py` line 177),
  so a learned gain is a reparametrization of the head matrix.
- QK-norm bounds the attention scores. With each vector RMS-normalized, |q| = |k| = sqrt(d_h), so the scaled score
  q.k / sqrt(d_h) lies in [-sqrt(d_h), sqrt(d_h)], which is [-8, 8] for 12 heads of 64. Gemma 2's attention cap of 50
  would never bind here; a cap of 4 to 6 would bind on aligned pairs.
- Speed ideas are worth little on this yardstick. GQA and MLA only save compute, so their net gains are a few
  thousandths of a nat at most, and the sign of any quality cost decides them.
- Several sources need a kernel that flex attention does not expose: the normalizer is fixed to softmax, and a
  post-softmax clip or a sigmoid normalizer is not a score function. Those ideas are priced at 8 to 12 hours or left out.

## Ranked table

Gain is the expected change in validation loss on this benchmark (estimate; negative is better). "Read" says how much
of the source was read: full, partial (the first part of an HTML or PDF text), abstract, code excerpt, PR description,
or blog text. Compile is against the 180 s cap; the record compiles in 106 to 122 s. Step cost is our estimate.

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats) | Compile | Step cost | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | Two-layer MLM head: dense, GELU, LN, then the untied 33-way head | ESM-2 code, `esm/modules.py` (2023 release) | code excerpt (class forward) | 1 | -0.000 to -0.005 | 0 to 5 s (one new static matmul) | +0.3 percent (768 by 768 on every position) | the base family's design; no ablation read |
| 2 | Partial key offset, both directions, within each protein | modded-nanogpt record 49, PR 169 (2025), adapted to bidirectional | PR description and the shift line | 1 to 2 | 0 to -0.003 | 0 s | 0 (a slice copy) | causal, one-layer induction; the record's gain is time only (1.1 percent, about 0.0003 nats) |
| 3 | Attention logit softcap that binds: cap 4 to 6 on the scaled scores | Gemma Team (2024), arXiv 2408.00118 | full HTML (values; no effect reported) | 1 | 0 to -0.003 | pilot (a Python constant in `score_mod`) | +1 to 2 percent | the cap values are Gemma 2's; the bound and the binding are our arithmetic |
| 4 | Learned window sink: one reserved token per window, readable by every protein | Xiao et al. (StreamingLLM, 2024), arXiv 2309.17453 | abstract | 2 | 0 to -0.003 | 0 to 5 s (a mask-rule change, same shape) | 0 to +1 percent | causal streams; the sink mechanism in masked models is untested here |
| 5 | Clipped softmax: a stretched and clipped softmax with exact zeros | Bondarenko et al. (2023), arXiv 2306.12929 | partial (first 100,000 of about 141,000 characters) | 8 to 12 | 0 to -0.005 | unknown (post-softmax clip needs a kernel) | unknown; the dense path is the only route without a kernel | BERT-base FP16 MLM ppl 4.49 to 4.39, 128 tokens (about 0.022 nats, one table) |
| 6 | Prediction from the mean of the last k blocks' states | Peters et al. (ELMo, 2018), arXiv 1802.05365 | abstract | 2 | 0 to -0.003 | 0 s | 0 | no pretraining number; needs a head-input hook |
| 7 | Asymmetric head cap, 23 sigmoid((z + 5) / 7.5) | modded-nanogpt record 60 (2026), PR 199 | PR notes via the record file | 0.5 | 0 to -0.002 | 0 to 3 s | 0 | a time record; no MLM number |
| 8 | Per-head RoPE frequency spread (a head-wise base) | Barbero et al. (2024), arXiv 2410.06205 | abstract | 1 | 0 +/- 0.003 | 0 s | 0 | the per-layer base of long-protein idea 3 at head granularity |
| 9 | Grouped-query attention, 3 key-value heads for 12 queries | Ainslie et al. (2023), arXiv 2305.13245 | abstract | 1 | -0.002 to +0.002 (net of about 0.0015 from the step saving) | 0 s | -5 to -6 percent | uptrained GQA "close to" MHA (abstract); the kernel call already takes `enable_gqa` |
| 10 | MLA-style low-rank keys and values (rank 256) | DeepSeek-V2 (2024), arXiv 2405.04434 | abstract | 3 | -0.001 to +0.002 (net of about 0.0008) | 0 s | -3 percent | a decoder KV-cache result (93.3 percent cut against a 67B model, whole model) |
| 11 | Muon on the head instead of AdamW | Jordan (2024), Muon post | blog text | 0.5 | 0 to +0.010 (worse) | 0 s | 0 | the post says AdamW for the head; the record's head rate is tuned on AdamW |
| 12 | Sigmoid attention with bias -log n | Ramapuram et al. (2024), arXiv 2409.04431 | partial (first 100,000 of about 253,000 characters) | 10 to 12 | 0 (their 1B table: 49.4 against 49.5) | unknown; no flex normalizer | their kernel: 0.34 against 0.38 s at 1B | needs a custom kernel |

## 1. Two-layer MLM head

- Source: the ESM-2 masked-LM head, `RobertaLMHead` in `facebookresearch/esm`, `esm/modules.py`. Read: the class forward
  (dense, then GELU, then LN, then the decoder as `F.linear(x, self.weight) + self.bias`). The constructor takes the
  decoder weight; the code that builds the head was not read, so whether ESM-2 ties it is inferred, not confirmed.
- Evidence: no ablation was read. The design is the base family's own readout, which is the strongest argument we have.
- What changes here: our head is one linear map (`train.py` line 117, 768 to 33, no bias, std 0.02 init), applied to
  `norm(X)` at line 155. The change adds a 768 by 768 dense layer, GELU and LN before it. Its parameters are about 0.6
  million, 0.3 percent of the 198 million; its FLOPs are about 0.3 percent of a step, since the head runs on every
  position. Muon is not used on it (the AdamW head group takes it, at the head rate from table 022).
- Expected cost: 1 hour, one new module. Compile: a new static matmul, 0 to 5 s (estimate). Step: +0.3 percent.
- Expected gain: -0.000 to -0.005 nats (estimate). A gain above the one-seed band needs two seeds.
- Risk: the head input is `norm(X)` at `train.py` line 155, and the `logits` hook sees only logits, so the change needs a
  `readout` hook or one edited line in `train.py`. The head's dense layer starts at a random init, not the zero-start the
  record uses for its output projections, so the step-0 network differs from the record.

```python
class MLMHead(Strategy):
    def build(self, model, max_tokens):                      # 768 -> 768 dense, LN; ESM-2 order: dense, GELU, LN, decoder
        self.dense = nn.Linear(HIDDEN, HIDDEN, bias=False)   # (768, 768)
        self.ln = nn.LayerNorm(HIDDEN)                        # (768,)
    def readout(self, model, X):                              # X: (b, l, 768), norm(X) of the final state
        return self.ln(F.gelu(self.dense(X)))                 # (b, l, 768), then the untied 33-way head
```

## 2. Partial key offset, adapted to bidirectional attention

- Source: modded-nanogpt record 49 (2025-12-14), PR 169, read through the PR description and its shift line. The
  PR shifts, by one token, the stationary quarters of each 128-dim head's keys (dims 32 to 64 and 96 to 128), so each
  key carries its predecessor's content in those dims; it is applied in the long sliding-window layers only, and the PR
  reports that applying it to every layer gives the lowest loss at higher cost. Its headline is a 2.4 s time gain from
  five changes, of which the offset is the part that let the record drop 50 steps (PR text).
- Evidence: the PR's mean loss over 11 runs is 3.2780 (std 0.0014) at 128.88 s. The record's 1.1 percent is a time
  gain, which is about 0.0003 nats on the yardstick. The quality effect of the offset on a masked model is not measured.
- What changes here: for our 12 heads of 64, the stationary quarters are dims 16 to 32 and 48 to 64. Shift the keys
  within each protein only: token j takes key j-1 in those dims when j-1 is in the same protein, and keeps its own key
  at each protein's first token. Two directions are free: a forward shift (the PR's, which is the induction argument),
  a backward shift (the mirror, which is the bidirectional form), and both as separate head groups. The induction
  argument is causal; the bidirectional case is an empirical question, so test the forward shift, the backward shift and
  no shift on one seed each. The input of a masked key is never the label, so the shift carries no target.
- Expected cost: 1 to 2 hours, including the protein-boundary mask. Step: 0 (a slice copy on K). Compile: 0 s, since the
  shape is unchanged.
- Expected gain: 0 to -0.003 nats (estimate, quality unknown).
- Risk: the shift must not cross a protein boundary, or the first token of each protein reads a neighbor's key. The
  placement relative to RoPE is a choice. The PR's shift is on the key path after its norm; here QK-norm is at
  `attention.py` line 96 and RoPE at line 98, so shifting before RoPE gives the neighbor's content without its rotation,
  and shifting after RoPE gives the neighbor's rotation too. Test one placement; the PR does not settle it.

```python
def shift_stationary(K: Tensor, same_protein: Tensor) -> Tensor:
    # K: (b, l, h, d_h) keys after QK-norm; same_protein: (b, l) bool, True where token l - 1 is in l's protein
    shifted = torch.cat([K[:, :1], K[:, :-1]], dim=1)                  # (b, l, h, d_h), the previous key
    quarter = torch.zeros(K.shape[-1], dtype=torch.bool, device=K.device)
    quarter[K.shape[-1] // 4 : K.shape[-1] // 2] = True                # dims 16 to 32 of 64
    quarter[3 * K.shape[-1] // 4 :] = True                              # dims 48 to 64 of 64
    return torch.where(same_protein[:, :, None, None] & quarter, shifted, K)   # (b, l, h, d_h)
```

## 3. Attention logit softcap that binds

- Source: Gemma Team (2024), arXiv 2408.00118, read in full HTML. The report sets the soft_cap to 50.0 for the
  self-attention layers and to 30.0 for the final layer. It reports no effect of the attention cap on loss or training,
  so the value is a design choice, not a measured gain.
- Our arithmetic: with RMS-normalized q and k in 64 dims, the scaled score is 8 times their cosine, so it lies in
  [-8, 8] before the ALiBi-style local bias, which is negative and only lowers scores. A cap of 50 never binds. A cap of
  4 binds above cosine 0.5, and a cap of 6 above cosine 0.75. The 6-heads-of-128 variant has the scaled bound 11.3, so
  the same caps bind less there. Our head's logit cap is 30, the same as Gemma 2's final cap, and the catalog's `softcap`
  family searched 10 to 60 on the head only.
- What changes here: the cap is applied to the full score inside the flex `score_mod`, as `cap * tanh(score / cap)` with
  `cap` a Python constant. A constant is not the captured-parameter path that made catalog 021 invalid, but the compile
  is unmeasured here, so it needs a pilot.
- Expected cost: 1 hour. Compile: a pilot, expected small. Step: +1 to 2 percent (estimate, an extra tanh per score).
- Expected gain: 0 to -0.003 nats (estimate). The cap has a bite only on near-aligned pairs, so most of the score matrix
  is untouched; expect a small effect either way.

```python
def capped(score_mod, cap: float):                       # cap: a Python float, a constant, not a captured tensor
    def mod(score, b, h, q, k):
        return cap * torch.tanh(score_mod(score, b, h, q, k) / cap)   # the full score, local bias included
    return mod
```

## 4. Learned window sink

- Source: Xiao et al. (StreamingLLM, 2024), arXiv 2309.17453, read from the abstract. Its finding: keeping the keys and
  values of the first few tokens recovers most of window attention's lost performance, because models put strong
  attention on initial tokens as sinks; its abstract also reports that a dedicated placeholder token as a sink during
  pretraining improves streaming. The long-protein scout's idea 2 cites the sink reduction in gated attention (first-token
  attention from 0.467 to 0.048 in a dense 1.7B model), which is the same mechanism from the other side.
- Evidence: causal streaming with up to 4 million tokens. Nothing for a bidirectional masked model in our reading.
- What changes here: each window's first token becomes a sink slot with the PAD embedding (`tokens.PAD_ID` = 1), its
  own document id, and a mask rule that lets every token read it. The window keeps its 1,024 slots, so the shape is
  unchanged; the cost is one residue per window (about 0.1 percent of training tokens). The sink is not a residue and is
  not scored.
- Expected cost: 2 hours, including the mask rule and a check that no residue reads a padded sink. Compile: the mask is a
  different function of the same shapes, 0 to 5 s (estimate). Step: 0 to +1 percent.
- Expected gain: 0 to -0.003 nats (estimate). It needs no new block-mask shape, so it is cheaper than a learned null key,
  which would add a key column to every block mask.
- Risk: a sink that every protein reads is a shared state across proteins in the window, which the document mask exists
  to prevent. The mask rule must make the sink readable and keep it from carrying one protein's content into another.

```python
def sink_aware(doc: Tensor, sink: Tensor):              # doc: (b, l) protein id; sink: (b, l) bool, True at the sink slot
    def rule(b, h, q, k):
        return (doc[b, q] == doc[b, k]) | sink[b, k]    # same protein, or any query reads the sink
    return rule
```

## 5. Clipped softmax

- Source: Bondarenko et al. (2023), arXiv 2306.12929, read in part (the first 100,000 of about 141,000 characters of the
  HTML). The method is clip((zeta - gamma) softmax(x) + gamma, 0, 1) with zeta at least 1 and gamma at most 0, so exact
  zeros are possible. Their BERT ablation found gamma = -0.03 with zeta = 1 best.
- Evidence (BERT-base, Table 2): MLM perplexity on Wikipedia, FP16, vanilla 4.49 (plus or minus 0.01) against clipped
  softmax 4.39 (plus or minus 0.00); sequences of 128 tokens. ln(4.49 / 4.39) is about 0.022 nats, one table. Their
  target is outliers and 8-bit quantization (the W8A8 column): vanilla 1,294 against 4.52 for clipped. The kurtosis and
  the max infinity norm fall sharply, so the gain here is probably the outlier effect, which we do not measure.
- What changes here: the clip sits after the normalization, so flex attention cannot express it (its normalizer is
  internal). Without a kernel the only route is the dense path (`attention.py` line 127), which materializes one
  (1,024 by 1,024) score matrix per window; that is far slower and not measured. A kernel change is the cost.
- Expected cost: 8 to 12 hours. Compile: unknown. Step: unknown, likely a loss on the dense path. Memory: higher.
- Expected gain: 0 to -0.005 nats (estimate; the BERT-base gain is 0.022 in one table, which we scale down because our
  attention is not quantized and our loss is the FP16 loss at 1,024 tokens).
- Risk: the BERT number is for a different model and sequence length, and the step cost may eat the gain. Read the rest of
  the paper (about 40,000 characters unread) before any build.

## 6. Prediction from the mean of the last k blocks

- Source: Peters et al. (ELMo, 2018), arXiv 1802.05365, read from the abstract only: the task model takes a softmax-weighted
  sum of the layers' representations with a scalar. No weights or pretraining numbers were read, and the abstract does not
  say which layers are best.
- What changes here: the head reads the mean of the last k blocks' normalized states (for example k = 4) instead of the
  final state. The head is a readout of the final block only, so the change means a head-input hook; `train.py` builds
  the head input at line 155.
- Expected cost: 2 hours. Compile: 0 s. Step: 0.
- Expected gain: 0 to -0.003 nats (estimate). The last block is trained for the readout, so an average over the last k
  may blur it. Expect a tie; the sign is unknown.
- Risk: the value residual (a record gain) already passes the first block's values forward, so the last block is not the
  only route to the head. The average may fight the value residual's gain at the output. One seed per k.

## 7. Asymmetric head cap

- Source: modded-nanogpt record 60 (2026-01-16), PR 199, through the record file's notes: the softcap in its fused
  kernel is 23 times sigmoid((z + 5) / 7.5), an asymmetric cap centered on a positive offset, not tanh. The record's gain
  is 0.90 percent of time, about 0.0002 nats on the yardstick. No MLM number.
- What changes here: a head cap of the record's shape in place of tanh at 30. It is a one-line change to the logit hook
  (`softcap.py` line 31), which the registry already allows.
- Expected cost: 0.5 hour. Compile: 0 to 3 s. Step: 0.
- Expected gain: 0 to -0.002 nats (estimate). The asymmetric cap leaves negative logits less bounded, which matters only
  for the 20 scored columns' tails; the catalog's softcap search says the head cap is worth about 0.002 in either
  direction.

## 8. Per-head RoPE frequency spread

- Source: Barbero et al. (2024), arXiv 2410.06205, read from the abstract: in Gemma 7B the highest RoPE frequencies build
  positional attention patterns and the lowest carry what the authors suspect is semantic information. The abstract gives
  no per-head numbers.
- What changes here: the record's RoPE has one base for all heads. A per-head base (or a per-head frequency multiplier)
  spreads the frequency bands across heads, so some heads read near positions and some far ones. The table is built once
  in `positional.py`, so the cost is zero.
- Expected cost: 1 hour. Compile: 0 s. Step: 0.
- Expected gain: 0 +/- 0.003 nats (estimate). This is the head-granularity form of long-protein idea 3 (a per-layer base),
  and the catalog's `rope_base` search (1e4 to 1e5) found base 1e5 about 0.0025 better in screen1 (a lower loss, a main
  effect in the screen's sign convention) and inside the band in run 024. Treat it as a variant of `positional`, not a
  new family.
- Risk: the partial-RoPE and NoPE results in the long-protein and init scouts are mixed. Expect a tie.

## 9. Grouped-query attention

- Source: Ainslie et al. (2023), arXiv 2305.13245, read from the abstract: uptrained grouped-query attention reaches quality
  "close to" multi-head attention at speed comparable to multi-query attention; the uptraining cost is 5 percent of the
  original pretraining compute for the MQA recipe. No table numbers were read, and the abstract gives no GQA-only cost.
- What changes here: 3 key-value heads serving 12 query heads, each group of 4 queries reading one key-value head. The
  kernel call already takes the group size (`attention.py` lines 120 and 123, `enable_gqa=group > 1`), so the change is
  the key and value projection shape.
- Expected cost: 1 hour. Compile: 0 s. Step: -5 to -6 percent (estimate: the key and value projections are about 21
  percent of the block's matmul weight, about 23 ms of the 109 ms of matmuls; cutting them to a quarter saves about 17 ms
  of the 276 ms step). On the yardstick, 5.5 percent is about 0.0014 nats.
- Expected gain: -0.002 to +0.002 nats net of the step saving (estimate). The quality cost is the unknown: an encoder
  with bidirectional attention has no decoding cache to save, so the case for GQA is only the step count.
- Risk: the encoder literature does not test GQA for masked models in what we read. The step saving is the only benefit.

## 10. Multi-head latent attention, low-rank keys and values

- Source: DeepSeek-V2 (2024), arXiv 2405.04434, read from the abstract only: MLA compresses the key-value cache into a
  latent vector and reduces the cache by 93.3 percent against DeepSeek 67B (a whole-model comparison, not an isolated
  MLA number). The abstract does not give the latent width, the decoupled RoPE key, or any quality number.
- What changes here: keys and values come from a rank-256 latent (a 768 to 256 down projection and a 256 to 768 up
  projection each), so the key and value projections drop from 2 by 768 by 768 to about 2 by 0.4 million weights per
  block. The step saving is about a third of the key and value matmuls, about 3 percent of the step (estimate).
- Expected cost: 3 hours, including the decoupled RoPE handling if the rope acts on the latent. Compile: 0 s. Step: -3
  percent (estimate). On the yardstick, that is about 0.0008 nats.
- Expected gain: 0 to +0.003 nats net (estimate). A rank-256 bottleneck on keys and values is a constraint on the
  attention, so the quality cost is more likely than a gain here.
- Risk: the rope on a latent is a design detail the abstract does not give. Read the paper's architecture section before a
  build.

## 11. Muon on the head

- Source: Keller Jordan's Muon post (2024), read as blog text. It says AdamW should be used for the embedding and the final
  classifier head layers, and reports Muon's 35 percent speedup on the NanoGPT speedrun, which is for the block matrices.
- What changes here: the head moves from the AdamW group (table 022's rates, 1e-3 for the head) to the Muon group. Our
  head is 768 by 33, a short matrix, which the orthogonalization treats as a 33-row update, and the post's point is that
  the post gives an empirical reason for the head's rule, not a theoretical one.
- Expected cost: 0.5 hour. Compile: 0 s. Step: 0.
- Expected gain: 0 to +0.010 nats (worse; estimate). The guidance and the record both point to AdamW, so a test would
  spend a run to confirm a known direction. Run it only if the proposers have no other head test.

## 12. Sigmoid attention

- Source: Ramapuram et al. (2024), arXiv 2409.04431, read in part (the first 100,000 of about 253,000 characters of the
  HTML). The attention is sigmoid(q.k / sqrt(d) + b) V with b = -log(n), in place of softmax.
- Evidence: an autoregressive LM with ALiBi, average accuracy over English benchmarks: 1B at 2k, softmax 49.4 and sigmoid
  49.5; 1B at 4k, 49.2 and 48.2; 1B with hybrid norm at 4k, 49.4 and 50.2; 7B with hybrid norm at 4k, 59.0 and 59.6.
  Step time at 1B and 2k is 0.34 s against 0.38 s on their FlashSigmoid kernel. These are downstream accuracies, not loss.
- What changes here: flex attention normalizes with a softmax, so sigmoid attention needs a custom kernel, which their
  FlashSigmoid is. A flex score function cannot replace the normalizer.
- Expected cost: 10 to 12 hours. Compile: unknown. Step: unknown.
- Expected gain: 0 nats (their 1B table is a tie). Not ranked above the others for that reason.
- Risk: a kernel build for a tie is the poorest use of the budget here.

## Checked and rejected

- Restricted or sampled softmax over the 20 scored columns. Already in the record: the training loss gathers the 20
  columns before the cross entropy (`benchmark/evaluate.py` line 42; `train.py` line 209). The 13 non-standard head rows
  get no gradient from it, so the head carries 13 by 768 (9,984) inert weights, and cutting them changes parameter count,
  not the loss. Sampled softmax is not needed at 33 classes: the record's own sampled-softmax ablation (record 92b) was
  for a 50,257-token vocabulary and reported a bias. The non-standard columns could still serve as negatives in a 33-way
  training loss, but that is the opposite of the scored quantity and is not a candidate.
- An output RMS norm with a learned gain. `norm` has no affine (`strategy.py` line 177). A per-channel gain before a
  linear head is a reparametrization of the head matrix; the function class is the same, and only the optimizer's
  geometry changes. Expected 0.
- Tied input and output embeddings, with or without a scale. The capacity scout rejects a tie on capacity grounds; the
  ESM-2 excerpt shows its decoder constructor takes the embedding weight, which suggests the base family ties. Our tuned rates differ (embedding
  3e-3, head 1e-3, table 022), and a tie forces one rate, which is the stronger argument against it here.
- Mixture of softmaxes. The softmax bottleneck binds when the vocabulary exceeds the hidden width plus one; here 33 is
  below 769 (the capacity scout's point). The abstract's perplexities (Penn Treebank 47.69; WikiText-2 40.68) are for word
  vocabularies of thousands, and no softmax baseline was in the abstract.
- Label smoothing. The scored cross entropy is against the hard label, so a smoothed target trains a different
  quantity. Muller et al. (2019), arXiv 1906.02629, read from the abstract, report better calibration and a weaker
  teacher for distillation; neither is the scored loss. The MLM and regularization scouts reject it.
- Z-loss. PaLM's term (architecture scout) keeps log Z from growing; our head's softcap at 30 already bounds the logits,
  so there is little to control, and the term changes the objective (`mlm_pretraining.md`).
- Logit adjustment (Menon et al., arXiv 2007.07314), read from the abstract only; its formula was not read. The abstract
  frames it around rare labels and margins. The evaluator's loss is a summed per-residue cross entropy over the masked
  positions, which is a micro-average, not class-balanced, so a log-prior shift would score a different quantity.
- Softmax-plus-one (softmax1, Miller 2023, blog): an argument with no experiments, and a kernel change. The in-mask sink
  (idea 4) is its cheap form.
- Talking-heads attention (Shazeer et al., 2020, arXiv 2003.02436), read from the abstract. It mixes heads before and after
  the softmax; flex's `score_mod` sees one head at a time, so the mixing needs every head's score matrix at once, which
  means the dense path and its materialized scores (not measured here). The abstract reports only qualitative MLM gains.
- A per-head learned gain on the QK-norm, which is a learned temperature. That is the SSMax family already in the catalog
  (`logit_scale`, arXiv 2501.19399), so it is a duplicate.
- A norm on the keys only, or on the queries only. No source was read, and an unnormalized query gives a per-token
  temperature. Not ranked.
- Query and key biases. Qwen2 uses QKV bias (its report, section 2.2.1, states no reason for it; arXiv 2407.10671, read in
  HTML). A bias before the QK-norm moves the direction of q and k, and no masked-LM result was found. Expected 0 +/- 0.002.
- Value-side norms. Not searched in this pass.
- Muon orthogonalizing query and key head pairs (modded-nanogpt record 80, PR 253). An optimizer grouping, measured at 0.35
  percent of time (about 0.0001 nats on the yardstick) on a record with unequal step counts. Belongs to the optimizer
  scout.
- The head on masked positions only. The head costs 33 by 768 per token, about 0.01 percent of a step, so restricting it
  to the 15 percent masked positions saves nothing that matters.
- Asymmetric or unbounded head caps beyond idea 7; a head bias (`train.py` line 117 has none). A 33-value output bias is a
  unigram prior that the head can already approximate; expected 0. The capacity scout's unigram key tables are the related
  item.
- Learned relative tables (T5 buckets, FIRE, CoPE), head-specific context masks (DuoAttention), cross-layer key and value
  sharing, local and global layer patterns, native sparse attention, and registers. Covered in
  `2026-10-30_long_protein_attention.md` (ideas 3, 4, 5, 7, 8, 9 and 10); not repeated here.
- Gated attention on the output (the head-sigmoid gate), partial RoPE, Peri-LN and SSMax. Covered by the catalog
  (`gates`, `positional`, `residual`, `logit_scale`) and `2026-10-30_long_protein_attention.md` idea 2.
- Copy-pointer heads over the visible residues (speedrun optimizers scout, item 9). The output-head alternative with
  the most source-level detail; not repeated.
- Mimetic V and O initialization (record 64; catalog 020, the same curve) and the orthogonal zero-start variants. Covered
  by `2026-10-30_init_and_parametrization.md`.

## Sources

- ESM-2 masked-LM head, code: https://github.com/facebookresearch/esm/blob/main/esm/modules.py
- modded-nanogpt records and PRs: https://github.com/KellerJordan/modded-nanogpt (README); PR 169 (record 49):
  https://github.com/KellerJordan/modded-nanogpt/pull/169; PR 199 (record 60): https://github.com/KellerJordan/modded-nanogpt/pull/199;
  PR 253 (record 80): https://github.com/KellerJordan/modded-nanogpt/pull/253
- Gemma 2: https://arxiv.org/abs/2408.00118 (HTML: https://arxiv.org/html/2408.00118)
- Clipped softmax: https://arxiv.org/abs/2306.12929 (HTML: https://arxiv.org/html/2306.12929)
- StreamingLLM (attention sinks): https://arxiv.org/abs/2309.17453
- Sigmoid attention: https://arxiv.org/abs/2409.04431 (HTML: https://arxiv.org/html/2409.04431)
- Softmax-plus-one: https://www.evanmiller.org/attention-is-off-by-one.html
- Muon post: https://kellerjordan.github.io/posts/muon/
- RoPE frequencies (Barbero et al.): https://arxiv.org/abs/2410.06205
- Grouped-query attention: https://arxiv.org/abs/2305.13245
- DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434
- Qwen2 technical report: https://arxiv.org/html/2407.10671
- ELMo: https://arxiv.org/abs/1802.05365
- Talking-heads attention: https://arxiv.org/abs/2003.02436
- Mixture of softmaxes: https://arxiv.org/abs/1711.03953
- Label smoothing (Muller et al.): https://arxiv.org/abs/1906.02629
- Logit adjustment: https://arxiv.org/abs/2007.07314
- Tied embeddings: https://arxiv.org/abs/1608.05859
- Gated attention (the head gate and the sink reduction, cited via the long-protein scout): https://arxiv.org/abs/2505.06708
- Repository: `src/speedrunning_plms/strategies/attention.py` lines 95 to 127; `strategies/softcap.py` lines 25 to 32;
  `strategies/strategy.py` lines 177 to 179; `benchmark/evaluate.py` lines 38 to 42; `benchmark/tokens.py` lines 16 to 28;
  `train.py` lines 117, 129, 155 and 209.
