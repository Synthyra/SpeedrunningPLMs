# Long-protein attention and context ideas, not yet in the catalog

Date: 2026-10-30. A literature scout for SpeedrunningPLMs (36 blocks, 768 wide, 2,048 MLP, 10 minutes on one GH200,
record 028: val_select 2.7319). It read `ideas/README.md` and `ideas/2026-10-30_speedrun_optimizers.md` first, then
the packing and mask code (read only, nothing run). Nothing here is measured on this benchmark. Each gain is a source's
number or our estimate, and each row says which. Shorthand: MLM is masked language modeling; SDPA is scaled dot-product
attention; TPP is tokens per parameter (our run is 0.3 to 0.7); "nats" is validation cross entropy; one seed resolves
about 0.0015 nats (README); RoPE is rotary position embedding; ALiBi is the fixed per-head local bias already in the
record (`positional`, `alibi`).

## Outcome

- The record's known gap is use of context on long proteins. Two of the ideas target the mechanism more directly than
  any architecture change: the training windows cut proteins at their edges (see "Repo facts"), while the evaluator
  scores whole proteins. Idea 1 fixes that mismatch with no new kernel. It is ranked first on that basis, not on a
  published number, since no MLM source measures it.
- The best-evidenced architecture change is the query-dependent head-wise sigmoid gate on the SDPA output (Qwen, NeurIPS
  2025). It is cheap, but the catalog's residual-channel gate (`gates`, 026) lost, so expect a small gain at most.
- Most long-context designs in the 2024 to 2026 literature are inference-memory or extrapolation results (KV cache,
  RULER at 128k to 256k). They do not transfer to a 1,024-token MLM at 0.3 to 0.7 TPP. Discount them accordingly.
- Expected gains are small: every estimate is under 0.01 nats, and most are under 0.005, which is near the one-seed band.

## Ranked table

Gain is the expected change in validation loss (our estimate; negative is better). "Read" says how much of the source was
read: full, partial (HTML sections), abstract, or snippet. Step and compile are our estimates unless marked from the
catalog.

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats) | Step / compile | Evidence strength |
|---|---|---|---|---|---|---|---|
| 1 | Whole-protein packing (1,024 windows, then a 2,048-window variant) | repo code; arXiv 2501.07747 (2025); arXiv 2107.02027 (2021) | code full; 2501.07747 partial; 2107.02027 title and claim | 1 to 2 | -0.002 to -0.010 | 0 step at 1,024; +0 to +10 percent at 2,048; compile 0 at 1,024, +20 to 50 s at 2,048 (catalog's new-mask cost) | moderate mechanism, no MLM number |
| 2 | Query-dependent head-wise sigmoid gate on the SDPA output | arXiv 2505.06708 (2025, NeurIPS 2025) | partial (Table 1 to 5) | under 1 | -0.002 to -0.007 | +1 to 2 percent step; compile +a few s; memory negligible | strong on LM (dense 1.7B, 400B tokens: -0.7 percent PPL); near-tested by 026 |
| 3 | RoPE in some layers only (NoPE every fourth block) and a per-layer RoPE base | arXiv 2501.18795 (2025, NeurIPS 2025); arXiv 2503.19786 (2025) | partial (HTML) | 1 | 0 +/- 0.003 | 0 step; 0 compile; needs one new hook | weak here: causal 8B, long-context results only |
| 4 | Local and global layers in a fixed pattern (local window 512 on two of every three blocks) | arXiv 2412.13663 (2024); arXiv 2503.19786 (2025) | ModernBERT partial (HTML); Gemma 3 partial | 2 to 3 | 0 to -0.003 quality; step -3 to -7 percent | compile +20 to 50 s for the local mask shape | qualitative tie in ModernBERT appendix; no table |
| 5 | Cross-layer K and V sharing (cross-layer attention) | arXiv 2405.12981 (2024, NeurIPS 2024) | abstract and summaries | 2 | 0 to +0.004 quality; step -3 to -5 percent | 0 compile; needs a new keys hook | memory result only; no training-loss number read |
| 6 | Attention Residuals: softmax over earlier layer outputs (Block variant) | arXiv 2603.15031 (2026) | abstract only | 3 to 5 | 0 to -0.005 | step +3 to +6 percent; about 3.6 GB of extra bf16 state at 65,536 tokens a step (estimate) | abstract only; no numbers read |
| 7 | Learned relative bias (FIRE, T5 buckets, CoPE) added to the local bias | arXiv 2310.04418 (2024, ICLR 2024); arXiv 2405.18719 (2024) | FIRE partial; CoPE abstract | 3 to 4 | 0 to -0.003 | invalid risk: a captured parameter in `score_mod` (catalog 021: 264 s warmup, 28 s a step) | SCROLLS +1 point average (FIRE); text and code only (CoPE) |
| 8 | Head-specific context: a few local heads, the rest global (retrieval and streaming heads) | arXiv 2410.10819 (2024, ICLR 2025) | partial (abstract and summaries) | 3 | 0 to +0.005 (catalog 014 points the wrong way) | step -5 to -10 percent if two of six heads are local (estimate); compile +20 to 50 s for H-dependent masks | inference memory and decode only |
| 9 | Native sparse attention: compression, block selection and a sliding window, gated | arXiv 2502.11089 (2025, ACL 2025) | not read (HTTP 429); summaries only | 10 to 15 | unknown | selection is not a flex mask; new kernel; compile risk | no numbers read |
| 10 | Encoder registers or sink tokens | arXiv 2309.16588 (2023, ICLR 2024) | abstract and summaries | 1 | about 0; blocked by the shape rule | adds positions to the window | vision only; no MLM evidence |

## 1. Whole-protein packing, then 2,048-token windows

- Source: the repository (`train.py` `windows()`, lines 171 to 181; `attention.py` `training_masks`, lines 196 to 220).
  Literature: arXiv 2501.07747, "Scaling Up ESM2 Architectures for Long Protein Sequences" (2025), read partially; its
  2,048 model beats standard ESM2 on proteins over 1,024 residues in function prediction (Table 5, BPO test: T6 0.501
  standard, 0.516 long; T12 0.493 to 0.533; T33 0.517 to 0.511, a loss). It is a function-prediction result, not MLM.
  Packing without leakage: arXiv 2107.02027, Krell et al. (2021), title and claim only read.
- Evidence: the training stream is cut into windows at fixed token offsets, so a protein that straddles a window edge
  loses its other part. The document index is the running count of CLS tokens (`attention.py` line 197), so the tail of a
  straddling protein sits in document 0 and attends only to itself. At evaluation every protein is scored whole (padded to
  one width per split). Training and scoring therefore differ on context. Under a random offset, a protein of n tokens
  (CLS and EOS included) crosses a 1,024-token edge with probability about (n - 1) / 1,024. With a mean training length of
  250 residues (an assumption; measure it from the train offsets), about a quarter of proteins straddle. The catalog's
  "length bins are flat" (2.746 to 2.762) does not rule this out, since the bins are per whole protein at evaluation.
- What changes here: windows hold whole proteins only. Pack by first fit within a window; fill the remainder with pad
  tokens. Two variants:
  - 1a: 1,024 windows, whole proteins. No shape change, no new compile. Costs the padding fraction in real tokens per step
    (estimate 3 to 8 percent with first fit; measure).
  - 1b: 2,048 windows, 32 a step, whole proteins. Same tokens per step (65,536). Straddling falls to about a twelfth
    of proteins at a mean of 250 residues. Flex cost per token scales with the size-weighted mean protein length (the sum
    of squared document lengths over the window length), so for short proteins the step should be near flat (estimate).
    The catalog's 008 (32 windows of 1,024, 1.8 times the steps) changed the step count, not the window, so it does not
    test 1b.
- Expected cost: 1 to 2 hours. Step: 0 for 1a; +0 to +10 percent for 1b (estimate). Compile: 0 for 1a; 1b is a new
  static shape, so expect the catalog's 20 to 50 s new-mask cost and check it against the 180 s cap (028 compiles in 117 s).
  Memory: unchanged.
- Sketch (a `batch` variant of `train.py`'s window builder; `TrainStream.offsets` gives each protein's span):

```python
def whole_protein_windows(stream: benchmark.TrainStream, window: int, per_step: int, seed: int) -> Iterator[Tensor]:
    """Whole proteins only: first fit into windows of `window` tokens; a protein never straddles an edge."""
    spans = np.diff(stream.offsets)                                   # (n,) CLS + residues + EOS
    rows: list[np.ndarray] = []                                       # each (window,) uint8, PAD in the tail
    order = np.random.default_rng(seed).permutation(len(spans))       # shuffle at protein level
    # first-fit packing over `order`; a protein longer than the window is cut at the window edge and asserted rare
    ...
    # the mask needs one more rule: a PAD position gets its own document id, so it neither sees nor is seen by a protein
```

- Risk: the gain is a train-test context mismatch, so it should show as a lower loss on straddling proteins, not on the
  whole mean. Padding costs real tokens; for 1b the mask's document rule needs the PAD clause or pads leak into
  neighbors. Check the test and validation `max_tokens` in the manifest before claiming anything above 1,024: the
  `val_select` split is capped at 1,024 tokens (`benchmark/data.py` line 43, 1,022 residues plus CLS and EOS). Whether
  packing is a "fixed data" matter or the `batch` family's free field is for Logan to confirm; we read it as free because
  the shape rule (`benchmark/architecture.py`) fixes the blocks, not the window.

## 2. Query-dependent head-wise sigmoid gate on the SDPA output

- Source: arXiv 2505.06708, "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"
  (Qwen team, 2025; NeurIPS 2025 oral). Read: partial (Tables 1, 2, 4, 5 in HTML).
- Evidence: 15B-total, 2.5B-active MoE, 400B tokens, Table 1 (average perplexity over their eval sets): baseline 6.026;
  SDPA head-wise gate 5.792 (1.6 million added parameters); SDPA element-wise 5.761 (201 million). The head-wise gate is
  within 0.03 of element-wise at 0.8 percent of the parameters, and the paper says the granularity and the activation
  "have relatively minor impacts" once heads get distinct scores. Dense 1.7B (48 layers, 400B tokens): baseline 7.421,
  head-wise 7.370 (-0.7 percent). Attention to the first token falls from 0.467 to 0.048; mean max activation from 1,053
  to 94 (Table 4). Long-context RULER improves at 4k to 32k without YaRN (Table 5). No per-token nat loss is reported;
  the relative drop converts to about 0.007 nats in the dense run, a rough conversion because their average is over sets.
- Repo context: the catalog's `gates` family (026, `residual_channels`) gates from the residual's first h channels and
  lost 0.002 on 022 (screen1 +0.0074 on 028). That is not the paper's gate, which reads the block input through a learned
  linear map and sits on the SDPA output before the output projection. It is a near repeat, so expect a small effect.
- What changes here: a per-block 768 to 6 linear gate, applied per head to the SDPA output. Initialize the bias at +4
  so the sigmoid starts near 1 and the step-0 network matches the record.
- Expected cost: under 1 hour. Step: +1 to 2 percent (estimate; one (b, l, 6) gate and a broadcast multiply). Compile: a
  few seconds. Memory: 6 floats per token per block.
- Sketch (hook `attention_output(index, X, Y)`; check in `attention.py` whether `Y` is before or after the output projection):

```python
class HeadGate(Strategy):
    def build(self, model, max_tokens):              # one 768 -> 6 gate per block; bias +4 so sigmoid starts near 1
        self.gates = nn.ModuleList(nn.Linear(HIDDEN, HEADS) for _ in range(NUM_BLOCKS))
        for gate in self.gates:
            nn.init.zeros_(gate.weight); nn.init.constant_(gate.bias, 4.0)
    def attention_output(self, index, X, Y):         # X: (b, l, 768) block input; Y: (b, l, 768) SDPA output
        scale = torch.sigmoid(self.gates[index](X))                           # (b, l, 6)
        heads = Y.unflatten(-1, (HEADS, -1))                                  # (b, l, 6, 128)
        return (heads * scale.unsqueeze(-1)).flatten(-2)                      # (b, l, 768)
    def parameter_groups(self, model):               # gate weights on the matrix group or AdamW; no weight decay on bias
        ...
```

- Risk: the gain may be near zero because the catalog already tested a gate (026) and the window is short. The paper's
  stability and sink claims are for 1.7B to 15B models at 400B to 3.5T tokens; we are at 65 to 131 million tokens. If the
  gate is placed after `o_proj`, the paper's G1 position is not tested, so check the placement before reading the result.

## 3. RoPE in some layers only, and a per-layer RoPE base

- Source: arXiv 2501.18795, "Rope to Nope and Back Again: A New Hybrid Attention Strategy" (2025, NeurIPS 2025), read in
  HTML. Gemma 3 technical report, arXiv 2503.19786 (2025): RoPE base 10,000 on local layers and 1,000,000 on global
  layers (read in the report's architecture section).
- Evidence: in an 8B observation model trained on 750B tokens, RoPE alone gives validation loss 1.52, NoPE alone 1.58, and
  RoPE is better on every short benchmark (Table 2). The hybrid (NoPE every fourth layer, sliding-window RoPE in the rest)
  gives RULER at 256k of 74.8 against 57.1 for the baseline (Table 6), and the paper reports short-context parity. All of
  it is causal and at 8k to 256k lengths. Gemma 3 gives no table of its base split in what we read.
- What changes here: NoPE in every fourth block (a fixed set of layers), keeping the record's ALiBi local bias on all
  heads, so the distance prior is not removed. The RoPE base can differ per layer: 10,000 on the local-looking layers and
  1,000,000 on the rest. The catalog's `rope_base` (1e4 to 1e5) is one base for all layers, so the split is new.
- Expected cost: 1 hour. Step: 0. Compile: 0 (static per-layer flags). Needs a new hook that rotates queries and keys
  per layer, since the base `Strategy` has no query or key hook.
- Sketch (a new hook `rotate(index, Q, K)`, called inside `attention.py` after QK-norm; `positional.py` provides the table):

```python
class MixedPositions(Strategy):
    NOPE_BLOCKS = frozenset(range(3, NUM_BLOCKS, 4))  # every fourth block: no rotation, the ALiBi bias stays
    def build(self, model, max_tokens):                # two tables: base 1e4 for the rest, base 1e6 for the global ones
        self.rotary_local = Rotary(HEAD_DIM, 1e4, max_tokens)
        self.rotary_global = Rotary(HEAD_DIM, 1e6, max_tokens)
    def rotate(self, index, Q, K):                     # Q, K: (b, h, l, 128) after QK-norm
        if index in self.NOPE_BLOCKS: return Q, K
        rotary = self.rotary_global if index % 4 == 0 else self.rotary_local
        return rotary(Q), rotary(K)
```

- Risk: at 1,024 tokens the long-context benefit is absent, and the 8B result favors RoPE at short context. With a bidirectional
  MLM the NoPE layers get no ordering signal beyond the ALiBi prior. Expect a tie.

## 4. Local and global layers in a fixed pattern

- Source: arXiv 2412.13663, "Smarter, Better, Faster, Longer: ModernBERT" (2024). Read: abstract and HTML sections (Section
  2.1.2 and Table 4 for the pattern; Appendix D for the ablation). Gemma 3 (arXiv 2503.19786, 2025): five local layers
  with a 1,024-token span for each global layer.
- Evidence: ModernBERT uses global attention every third layer and a 128-token local window, with RoPE base 160,000 on
  global and 10,000 on local layers. Its Appendix D reports "identical downstream performance" to all-global attention, with
  no table or number. Gemma 3 gives no ablation of the window in what we read. Both are long-context encoders or
  decoders at 1,024 to 8,192 training length, not 65 to 131 million tokens of MLM training.
- What changes here: a sliding local window (512 tokens inside each document) on two of every three blocks, global on the
  rest. A 512 window changes only proteins longer than 512, which are the ones the gap concerns. A 128 window would cut
  the long-protein context the record needs, so it is not a candidate. The step saving grows with the share of long
  proteins; the catalog's length distribution is not read here.
- Expected cost: 2 to 3 hours. Step: -3 to -7 percent (estimate). Compile: +20 to 50 s for the second block-mask shape
  (catalog's new-mask cost), which puts 028's 117 s compile close to the 180 s cap; the proposers' 15 s margin rule
  applies. Memory: unchanged.
- Sketch (a new hook `masks(index, masks)` that picks the layer's block mask; `attention.py` `training_masks` builds the
  two masks):

```python
class LocalGlobal(Strategy):
    def build(self, model, max_tokens):
        self.local_window = 512                        # 128 cuts the long-protein context; not a candidate
    def training_masks(self, corrupted, masks):        # masks: (global_doc_mask, local_doc_window_mask) from attention.py
        return masks
    def masks(self, index, masks):                     # picks per layer; global every third block
        return masks[0] if index % 3 == 2 else masks[1]
```

- Risk: the ModernBERT result is a tie with no numbers, and the local window's cost to long proteins is exactly the gap.
  Expect 0 quality change and a step saving that buys few steps at 0.3 to 0.7 TPP. Low expected value; worth it only with
  idea 1 or 2 already in.

## 5. Cross-layer K and V sharing

- Source: arXiv 2405.12981, "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention" (Brandon et al., 2024,
  NeurIPS 2024). Read: abstract and summaries. Systematic follow-up: arXiv 2410.14442 (2024).
- Evidence: CLA halves the KV cache beyond multi-query attention at nearly the same accuracy (the paper's headline; the
  per-model tables were not read). It is a memory result. A review reports that CLA with MQA is best at sharing factor 2.
  No training-loss number was read.
- What changes here: the record's K and V projections are 2 of the 5.5 million weights per block (about 21 percent of the
  block's matmul work). Sharing K and V between adjacent blocks in half the layers saves about 10 percent of matmul time,
  which is roughly 11 of the 109 milliseconds of matmuls in the profile (estimate, about 4 percent of the step). The
  quality cost is the risk. Queries stay per block, so flex attention still runs once per block.
- Expected cost: 2 hours. Step: -3 to -5 percent (estimate). Compile: 0 (no new shape). Memory: about 10 percent fewer
  weights.
- Sketch (new hook `keys_values(index, X)`; the second block in each pair reuses the first block's K and V):

```python
class SharedKV(Strategy):
    SHARED = frozenset(range(1, NUM_BLOCKS, 2))       # odd blocks reuse the even block's K and V
    def build(self, model, max_tokens):
        for index in self.SHARED: model.blocks[index].attn.k_proj = None; model.blocks[index].attn.v_proj = None
    def keys_values(self, index, X, cache):           # cache: dict of the even block's K and V for this step
        if index in self.SHARED: return cache["K"], cache["V"]
        K, V = project(X); cache["K"], cache["V"] = K, V
        return K, V
```

- Risk: sharing removes capacity that the paper says is cheap at decoder scale; for a 1,024-token bidirectional MLM the
  effect is unknown. Expect a tie or a small loss. Lower priority than the gate.

## 6. Attention Residuals: softmax over earlier layer outputs

- Source: arXiv 2603.15031, "Attention Residuals" (Kimi Team, March 2026). Read: abstract only; the HTML page returned no
  body. Follow-ups read only as titles: arXiv 2605.18855 (Delta Attention Residuals), 2607.27230 (Multi-Head AttnRes),
  2610.13630 (Multi-Stream AttnRes).
- Evidence (abstract): replaces the fixed unit-weight residual sum with softmax attention over earlier layer outputs; a
  Block variant attends over block summaries; the authors report consistent gains across model sizes and integration into
  Kimi Linear (48B total, 3B active, 1.4T tokens). No validation numbers were readable. The delta variant reports
  1.7 to 8.2 percent validation perplexity gains at 220M to 7.6B; those figures are from a follow-up's abstract, not
  checked.
- Repo context: the catalog's `dense` stream (DenseFormer-like span, retired by the warmup cap at span 8) and the value
  residual (028, the record's gain) are the nearest relatives. A shorter span might return, per the catalog.
- What changes here: every block reads a softmax over the outputs of earlier blocks, with a per-block query. The extra
  compute is small per token (a softmax over up to 36 vectors of 768 at each block; about 5 percent of block FLOPs,
  estimate). The memory is the layer outputs kept for the softmax, about 3.6 GB in bf16 for 65,536 tokens a step (estimate).
- Expected cost: 3 to 5 hours after the full text. Step: +3 to +6 percent (estimate). Compile: a few seconds more.
- Sketch (hook `block_input(index, X, states)`, which already sees the prior states):

```python
class AttentionResidual(Strategy):
    def build(self, model, max_tokens):
        self.query = nn.Parameter(torch.zeros(NUM_BLOCKS, HIDDEN))     # one pseudo-query per block, zero at the start
    def block_input(self, model, index, X, states):   # states: the list of earlier block outputs, each (b, l, 768)
        stack = torch.stack(states + [X], dim=2)                       # (b, l, k, 768)
        scores = (stack * self.query[index]).sum(-1) / HIDDEN ** 0.5   # (b, l, k)
        return (scores.softmax(-1).unsqueeze(-1) * stack).sum(2)       # (b, l, 768)
```

- Risk: no numbers read, and the zero-start pseudo-query gives a uniform average over earlier states, which is not the
  record's residual at step 0 (the record adds each output with unit weight). Check the paper's initialization before
  using this sketch.

## 7. Learned relative bias (FIRE, T5 buckets, CoPE) beside the local bias

- Source: arXiv 2310.04418, "Functional Interpolation for Relative Positions Improves Long Context Transformers" (Li et al.,
  2024, ICLR 2024); read: partial (abstract and summaries). arXiv 2405.18719, "Contextual Position Encoding" (Golovneva et
  al., 2024); read: abstract and summaries.
- Evidence: FIRE beats the competing methods by over 1 point on average on SCROLLS, and represents T5's relative bias,
  ALiBi and Kerple as special cases (paper claims). No protein number. CoPE counts selected tokens; it reports gains on
  text and code, and a separate CoPE (NeurIPS 2025 workshop) is unrelated.
- Repo context: the catalog's learned slopes inside the flex `score_mod` (021) were invalid: a captured parameter took a
  far slower path (264 s warmup, then 28 s a step). A FIRE or T5 bucket table is also a captured tensor in `score_mod`, so
  it inherits that failure unless the bias avoids the captured-parameter path.
- What changes here: a learned relative bias added to the score for each head, on top of the fixed ALiBi bias.
- Expected cost: 3 to 4 hours, most of it finding a kernel path that avoids the captured parameter. Step: invalid under
  the current kernel (021); 0 to +10 percent if it works (estimate). Compile: over the cap in the 021 form.
- Sketch (new `score_mod` term; the table is a buffer, not a parameter, and the gradient is taken through a
  precomputed bias, not the score function):

```python
def learned_bias(slopes: Tensor, table: Tensor):   # table: (heads, buckets) learned; a buffer per graph step
    def score_mod(score, batch, head, query, key):
        return score + table[head, bucket(query - key)]    # captured tensor: the 021 slow path, unless precomputed
    return score_mod
```

- Risk: the same failure as 021 is likely. Expected gain on the record is small; a tie is the most likely result.

## 8. Head-specific context: a few local heads, the rest global

- Source: arXiv 2410.10819, "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads" (Xiao,
  Tang, Han and co-authors, 2024, ICLR 2025). Read: abstract and summaries.
- Evidence: inference memory down 2.55 times for MHA and 1.67 times for GQA; decoding up to 2.18 times faster. Retrieval
  heads keep a full cache; streaming heads keep sinks and recent tokens. The heads are found by optimization on synthetic
  data. No training-quality number. The catalog's 014 is the nearest training result: the local bias on six heads, with the
  other six free, lost most of 012's gain. That suggests every head wants the local prior, which points against giving some
  heads no local prior and others a hard window.
- What changes here: two of six heads on a 128 or 256 window (hard local), four global. A mask that depends on the head
  index needs one block mask per head group, so it is a new shape and a new compile.
- Expected cost: 3 hours. Step: -5 to -10 percent (estimate). Compile: +20 to 50 s (catalog). Memory: unchanged.
- Sketch (the mask rule takes the head index, so the local rule is a second block mask for two heads):

```python
def head_windows(document: Tensor, local_heads: tuple[int, ...], window: int):
    def rule(batch, head, query, key):
        local = (query - key).abs() < window
        return (document[batch, query] == document[batch, key]) & ((head not in local_heads) | local)  # ()
    return create_block_mask(rule, B, HEADS, L, L, device=device, _compile=True)
```

- Risk: 014 says the local prior is wanted on all heads, so expect a loss. This is a speed idea, not a quality idea.

## 9. Native sparse attention

- Source: arXiv 2502.11089, "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention" (2025, ACL
  2025). Not read: the paper fetch returned HTTP 429. Summaries only.
- Evidence: none read. Summaries describe three branches (compressed blocks, selected blocks, a sliding window) with a
  learned gate; block sizes of 32 and 64 and 16 selected blocks appear only in a blog post. Treat all numbers as unknown.
- What changes here: selection is a top-k over blocks per query, which a flex block mask does not express. It needs a new
  kernel, which the catalog's C3 (fused kernels, days of work) already rules out for a 10-minute budget.
- Expected cost: 10 to 15 hours. Step and compile: unknown and likely over the cap.
- Risk: out of budget. Listed for completeness.

## 10. Encoder registers or sink tokens

- Source: arXiv 2309.16588, "Vision Transformers Need Registers" (Darcet et al., ICLR 2024). Read: abstract and summaries.
  Their artifact findings are for vision transformers; a search for encoder-language-model register or sink results found
  nothing usable.
- Evidence: learnable register tokens remove high-norm artifact tokens in ViT feature maps (paper's claim). No MLM or
  protein number.
- Repo context: this adds positions to every window. The shape rule fixes the blocks, not the window, but a register adds a
  token per window, so the window's real token count changes. The gated attention (idea 2) already reports a sink reduction
  (first-token attention 0.467 to 0.048 in its Table 4), which covers the same concern more cheaply.
- Expected cost: 1 hour; blocked or redundant. Recommendation: skip unless idea 2 fails and a sink is measured.

## Checked and rejected

- Learned-state or state-space long-context models (LC-PLM, arXiv 2411.08909, 2024; TMLR 2025): a bidirectional Mamba with
  shared projections trained with MLM on UniRef50. The paper reports an average of 13.5 percent lower evaluation loss than
  ESM-2 at similar sizes, but that comparison is a whole model, not an attention change, and it replaces the fixed
  attention skeleton. Out of scope for this speedrun. Its length-bin figure is plotted, not tabulated, so per-length
  numbers are unread.
- ESM C's context schedule: a GitHub issue (Biohub/esm #179) quotes the release blog: 512 tokens for the first 1 million
  steps, then 2,048 for the last 500,000. This is a curriculum of the kind the catalog retired (`short_first`, warmup cap).
  Its one-graph form is the only viable version. A maintainer says correctness drops past 2,048. Not a primary source.
- ProGen3 (8,192 context) and xTrimoPGLM (context length not reported): the sources read are blog summaries. No attention
  ablation was found for either.
- Dilated and strided attention (LongNet, arXiv 2307.02486, 2023): before the 2024 to 2026 window, and it needs a strided
  mask per layer, which is idea 4's cost without a protein result.
- Attention sinks for masked encoders: searched; the only hits were diffusion language models and causal-to-bidirectional
  conversion, with no encoder pretraining numbers.
- Llama 4 iRoPE (interleaved RoPE and NoPE layers with chunked local attention and attention temperature): the reference
  for idea 3, read only through secondary write-ups and the vLLM source. Not a separate idea.

## Repo facts the ideas depend on

- `train.py` `windows()` (lines 171 to 181): contiguous reshape of the shuffled token stream into `(b, window)`. No
  protein-boundary handling. Idea 1 depends on this.
- `strategies/attention.py` line 197: the document index is the running count of CLS tokens. Tokens before the first CLS
  in a window share document 0. A protein cut at a window edge keeps only its own tail in context.
- `benchmark/data.py` line 43: `VAL_SELECT_MAX_TOKENS = 1024` (CLS, 1,022 residues, EOS). The evaluator scores at most
  1,022 residues on `val_select`. Check the other splits' `max_tokens` in the manifest before claiming anything above
  1,024.
- `benchmark/architecture.py`: the shape rule fixes 36 blocks, a 2,048 MLP and a 768 stream. It does not fix the window
  or the batch.
- Hooks in the base `Strategy` (`strategy.py`): `build`, `initialize`, `parameter_groups`, `embed`, `block_input`,
  `values`, `attention_output`, `mlp_hidden`, `sublayer_output`, `logits`, `auxiliary_loss`, `after_backward`,
  `after_step`, `evaluation_weights`. Ideas 3, 4, 5 and 8 each need one new hook (`rotate`, `masks`, `keys_values`, a
  per-head mask); the table above notes which. Idea 2 uses `attention_output` as it stands.
- `attention.py` `training_masks`: one block mask per variant (a tuple for the triangle kernels). A second mask shape costs
  about 20 to 50 s of compile here, per the catalog.

## Gaps and read status

- Read in full or in sections: `ideas/README.md`, the optimizer scout, the repo's packing, mask, hook and shape code, the
  Gated Attention tables (2505.06708), RNoPE (2501.18795), the ModernBERT architecture and appendix (2412.13663), ESM2-long
  (2501.07747), FIRE (2310.04418, partial).
- Abstract or summaries only: Attention Residuals (2603.15031), CLA (2405.12981), DuoAttention (2410.10819), CoPE
  (2405.18719), ViT registers (2309.16588), Krell packing (2107.02027), LC-PLM (2411.08909, partial).
- Not read: NSA (2502.11089; HTTP 429), the biologically induced sparse attention protein model (bioRxiv 10.1101/2025.05.29.656907;
  HTTP 429), the Gemma 3 window ablation (not located), the full FIRE tables.
- Every number above is for language models or vision transformers, almost all causal and at 20 or more TPP. The
  benchmark is bidirectional MLM at under 1 TPP. Nothing here is a protein MLM result except ESM2-long's Fmax (a function
  task) and LC-PLM's loss comparison (a different architecture).
- The most useful measurement before any build: the mean protein length and the straddle fraction in the train offsets
  (idea 1), and the step time of the record with a 512-token local mask on two of every three blocks (idea 4), as a
  profile on the GH200. Neither was run here.

## Sources

- ModernBERT: https://arxiv.org/abs/2412.13663 (HTML: https://arxiv.org/html/2412.13663v2)
- Gemma 3 technical report: https://arxiv.org/abs/2503.19786
- Rope to Nope and Back Again (RNoPE): https://arxiv.org/abs/2501.18795 (HTML: https://arxiv.org/html/2501.18795v2)
- Gated Attention for LLMs: https://arxiv.org/abs/2505.06708 (HTML: https://arxiv.org/html/2505.06708v1)
- Attention Residuals: https://arxiv.org/abs/2603.15031 ; Delta AttnRes: https://arxiv.org/abs/2605.18855 ; Multi-Head AttnRes: https://arxiv.org/abs/2607.27230 ; Multi-Stream AttnRes: https://arxiv.org/abs/2610.13630
- Cross-layer attention: https://arxiv.org/abs/2405.12981 ; cross-layer KV sharing study: https://arxiv.org/abs/2410.14442
- ESM2 for long sequences: https://arxiv.org/abs/2501.07747
- LC-PLM: https://arxiv.org/abs/2411.08909
- ESM C context schedule (secondary): https://github.com/Biohub/esm/issues/179
- FIRE: https://arxiv.org/abs/2310.04418
- CoPE: https://arxiv.org/abs/2405.18719
- DuoAttention: https://arxiv.org/abs/2410.10819
- Native Sparse Attention: https://arxiv.org/abs/2502.11089 (not read)
- Vision Transformers Need Registers: https://arxiv.org/abs/2309.16588
- Efficient sequence packing without cross-contamination (Krell et al.): https://arxiv.org/abs/2107.02027
- Repository: `train.py` lines 171 to 181; `src/speedrunning_plms/strategies/attention.py` lines 194 to 227;
  `src/speedrunning_plms/strategies/strategy.py`; `src/speedrunning_plms/benchmark/data.py` line 43;
  `src/speedrunning_plms/benchmark/architecture.py`; `ideas/README.md` rows 013, 014, 021, 026, 027, 028.
