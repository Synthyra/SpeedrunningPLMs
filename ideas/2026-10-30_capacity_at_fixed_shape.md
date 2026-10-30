# Capacity at a fixed shape: literature scout

Date: 2026-10-30. A literature scout for SpeedrunningPLMs (36 blocks, 768 wide, 2,048 MLP, 10 minutes on one GH200,
record 028: val_select 2.7319). It read `ideas/README.md` and `ideas/2026-10-30_long_protein_attention.md` first and
skipped every idea they already list (the long-context set, the optimizer set, the catalog's families). Nothing here is
measured on this benchmark and nothing was run. Shorthand: MLM is masked language modeling; MoD is mixture-of-depths;
PKM is product-key memory; MTP is multi-token prediction; TPP is tokens per parameter (our run is 0.3 to 0.7); "nats" is
validation cross entropy; one seed resolves about 0.0015 nats (README). Each gain is a source's number or our estimate,
and each row says which. Hook names follow the list in the long-attention file; the signatures in the sketches are
assumed, not read from `strategy.py`.

## Outcome

- Only four ideas pass the rule of a source and a number for this scope. I did not pad the list to eight; the rejected
  items with their numbers are in the table after the ideas.
- The mechanism that matters most for this MLM is the MASK-only corruption (the catalog's "outside the rules" section
  fixes the objective so). A masked residue enters the model as the MASK token, so any table indexed by its own identity
  adds the same vector at every masked position. Only tables indexed by neighbors, or the model's own context, carry
  information about the scored residue. This rules out most of the token-indexed ideas in the brief (per-token MLP
  biases, per-layer token tables, unigram key tables) before any run.
- The catalog says step time is the lever left (the profile puts 109 ms of 276 ms in matmuls and 80 ms in flex
  attention). Token dropping is the only idea here that cuts step time materially, so it is the one with a large
  possible effect; it is also the most expensive to build.
- Expected gains are small: every estimate is under 0.005 nats, which is three seed bands. Expect ties.

## Ranked table

Ranked by expected gain per implementation hour. "Read" says how much of the source was read.

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats; our estimate) | Step / compile / params | Evidence strength |
|---|---|---|---|---|---|---|---|
| 1 | Neighbor-prediction auxiliary heads on masked residues | arXiv 2409.01482 (2024); arXiv 2404.19737 (2024, ICML 2024); NVIDIA MTP doc (2025) | search snippets; ICML poster | 1 | -0.000 to -0.003 | +1 to 2 percent step; compile +0 to 5 s; 25k params per offset | weak: decoder MTP results and a two-token gap of 2.0 to 1.8 percent on a mixer; no masked-LM number |
| 2 | Exact bigram and trigram context tables with a context gate | arXiv 2601.07372 (2026); modded-nanogpt records 62 and 83 (2026) | abstract; README rows | 2 | -0.000 to -0.004; the catalog's pair table (010) was harmful, so the sign is uncertain | +2 to 3 percent step; compile +0 to 10 s; 28.4M params (27.6M trigram, 0.8M bigram) | medium for the mechanism at scale; the speedrun numbers are record times, not nats |
| 3 | Token dropping of unmasked context in middle blocks (MLM mixture-of-depths) | arXiv 2203.13240 (2022, ACL 2022); arXiv 2305.15273 (2023); arXiv 2404.02258 (2024) | abstracts; search summaries | 3 to 4 | -0.000 to -0.004, carried by more steps | step -20 to -25 percent (estimate); compile +20 to 50 s (new static shape); 0 params | medium: a 25 percent BERT pretraining cost cut at matched fine-tuning; no loss number read |
| 4 | Product-key memory side path on one block | arXiv 2412.09764 (2024, ICML 2025); arXiv 1907.05242 (2019) | abstract; search summaries | 4 to 6 | 0 to +0.005 (expected tie) | step +1 to 2 percent (estimate); compile +10 to 20 s; 50.3M params outside the blocks | weak at this scale: the evidence is 1 trillion tokens at 8B |

## 1. Neighbor-prediction auxiliary heads on masked residues

- Source: Masked Mixers, arXiv 2409.01482 (2024): two-token prediction by two forward passes with summed losses, the
  loss gap to a transformer going from 2.0 to 1.8 percent. Gloeckle et al., "Better and Faster Large Language Models via
  Multi-token Prediction", arXiv 2404.19737 (ICML 2024, PMLR 235): 13B models solve 12 percent more HumanEval and 17
  percent more MBPP problems than next-token baselines (poster and summaries; full PDF not read); up to 3 times faster
  inference; a third-party review finds mixed or regressive results on likelihood-based benchmarks. NVIDIA's Megatron
  docs describe the auxiliary MTP loss at a scale of 0.05 to 0.2 of the main loss (search snippet).
- Evidence: the MTP evidence is for causal decoders. We found no MTP result for a bidirectional masked encoder (the
  search for one returned nothing usable). The catalog's `latent` family (NextLat with SIGReg) competes for the same
  auxiliary-loss budget, so the two should not both ride unless each is tested alone.
- What changes here: at each masked residue, two linear heads (768 to 33) predict the original identity of its left and
  right neighbors, scored only where the neighbor is in the same protein. The main objective is untouched, so the
  scored loss is still the masked-residue cross entropy. Neighbor identities are the only training signal about a
  masked residue's flanks that the head can use; they are cheap to predict from the context, so the expected effect is
  small.
- Expected cost: 1 hour. Step: +1 to 2 percent (estimate). Compile: 0 to 5 s. Params: 25k per offset.
- Sketch (hooks `build`, `initialize`, `auxiliary_loss`; the masked mean uses weights, not boolean indexing, so the
  shape stays static):

```python
class NeighborHeads(Strategy):
    OFFSETS = (-1, 1)                                   # each offset is one linear head on each masked residue
    WEIGHT = 0.1                                        # share of the main loss (NVIDIA's 0.05 to 0.2)
    def build(self, model, max_tokens):
        self.heads = nn.ModuleList(nn.Linear(HIDDEN, VOCAB, bias=False) for _ in self.OFFSETS)  # (768, 33) each
    def auxiliary_loss(self, model, X, targets, masked, doc):   # X (b, l, 768); targets (b, l); doc (b, l)
        terms = []
        for offset, head in zip(self.OFFSETS, self.heads):
            goal = targets.roll(-offset, dims=1)                       # neighbor identity, (b, l); wraps at row ends
            same = (doc == doc.roll(-offset, dims=1)).float()          # neighbor in the same protein, (b, l)
            weight = masked.float() * same                             # score masked residues with a neighbor
            ce = F.cross_entropy(head(X).flatten(0, 1), goal.flatten(), reduction="none")   # (b * l,)
            terms.append((ce * weight.flatten()).sum() / weight.sum().clamp_min(1.0))
        return self.WEIGHT * torch.stack(terms).sum()
```

- Risk: the expected gain is the small end of the table. The catalog's `latent` family uses the same budget. The wrap
  at row ends is not handled by the sketch: the last column wraps, and the `same` test only drops it when the document
  ids differ, so mask the wrapped column explicitly before a run.

## 2. Exact bigram and trigram context tables with a context gate

- Source: Cheng et al., "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models",
  arXiv 2601.07372 (DeepSeek and Peking University, submitted 12 January 2026, revised 12 July 2026). Read: abstract
  only. It reports a U-shaped scaling law between MoE experts and static memory, Engram scaled to 27B parameters, and
  gains of MMLU +3.4, CMMLU +4.0, BBH +5.0, ARC-Challenge +3.7, HumanEval +3.0 and MATH +2.4; the baseline for those
  deltas is not stated in the abstract. Multi-query needle-in-a-haystack rises from 84.2 to 97.0. Secondary write-ups
  place the module in layers 2 and 15 and gate the retrieved vector by context before the residual add; these are
  unverified. In the speedrun community, modded-nanogpt record 62 "Bigram Hash Embedding" (1.748 to 1.655 minutes,
  19 January 2026, record-time delta 5.3 percent) and record 83 "Sign Trick on Bigram Embed" (1.353 to 1.328 minutes,
  20 May 2026); record 92 adds an "84.6M-row hashed n-gram table" (65B parameters in the README's technique list) with
  nine other changes, so its table is not isolated. The README does not say where the bigram vector enters.
- Evidence: the catalog's `neighbor` family tried the same kind of table: a (left, right) pair table, 1,089 by 768,
  added to the embedding (run 010 on 003: level to 270 s, behind after, 2.7583 against 2.7559; screen1 +0.0114). That is
  a strong prior against a table added to the embedding. The two ideas that differ are the context gate (Engram's point)
  and where the vector enters.
- What changes here: a bigram table (1,089 rows) and a trigram table (35,937 rows) indexed by the residue and its
  neighbors. With 33 tokens, hashing is unnecessary: the index is exact and collision-free, so the table is the full
  n-gram statistic. At a masked residue the center is MASK, so the trigram row is the flank-conditioned distribution of
  the hidden residue, the only token-indexed signal that reaches a masked position. The table enters on two blocks
  through a per-token sigmoid gate, so an uninformative flank can be switched off.
- Expected cost: 2 hours. Step: +2 to 3 percent (estimate). Compile: 0 to 10 s. Params: 28.4M (27.6M trigram, 0.8M
  bigram), outside the blocks. Memory: the parameters and Adam state of 28M rows are about 0.3 GB in fp32.
- Sketch (hooks `build`, `initialize`, `embed`, `block_input`; the memory is computed once per forward and carried in a
  per-forward state, not stored on the module, so the compiled graph does not capture it):

```python
class ContextTables(Strategy):
    BLOCKS = (2, 15)                                     # read-in blocks; the secondary-source placement, unverified
    def build(self, model, max_tokens):
        self.bigram = nn.Embedding(VOCAB ** 2, HIDDEN)   # 1,089 rows, exact index
        self.trigram = nn.Embedding(VOCAB ** 3, HIDDEN)  # 35,937 rows, 27.6M params, exact index
        self.gate = nn.ModuleDict({str(b): nn.Linear(HIDDEN, 1) for b in self.BLOCKS})
    def initialize(self, model):
        nn.init.zeros_(self.bigram.weight); nn.init.zeros_(self.trigram.weight)   # step 0 equals the record
        for gate in self.gate.values(): nn.init.zeros_(gate.weight); nn.init.zeros_(gate.bias)
    def embed(self, model, tokens, state):               # tokens (b, l), MASK included; returns the embedding unchanged
        left = F.pad(tokens, (1, 0), value=PAD)[:, :-1]  # previous residue, (b, l)
        right = F.pad(tokens, (0, 1), value=PAD)[:, 1:]  # next residue, (b, l)
        state["memory"] = self.bigram(left * VOCAB + tokens) + self.trigram((left * VOCAB + tokens) * VOCAB + right)
        return model.token_embedding(tokens)             # (b, l, 768)
    def block_input(self, model, index, X, states, state):
        if index not in self.BLOCKS: return X
        return X + torch.sigmoid(self.gate[str(index)](X)) * state["memory"]    # (b, l, 768)
```

- Risk: the catalog's pair table lost, and a trigram table is a larger version of the same statistic. Expect a tie, and a
  loss if the injection site matters less than the table. Boundary tokens (CLS, EOS, PAD) are ordinary vocabulary rows,
  so a protein's first and last flanks read boundary rows; check that they do not leak across the document mask.

## 3. Token dropping of unmasked context in middle blocks (MLM mixture-of-depths)

- Source: Hou et al., "Token Dropping for Efficient BERT Pretraining", arXiv 2203.13240 (ACL 2022). Abstract read: tokens
  the model finds unimportant are dropped from intermediate layers, the importance comes from the MLM loss the model
  already computes, and the dropped tokens are restored at the last layer; BERT's pretraining cost falls by 25 percent
  with similar fine-tuning performance (no table read). Follow-up: Revisiting Token Dropping Strategy, arXiv 2305.15273
  (2023): its ScTD variant saves up to 57 percent of pretraining time and gains up to 1.56 percent average over vanilla
  dropping (search summary, not read in full). Raposo et al., "Mixture-of-Depths", arXiv 2404.02258 (2024): a learned
  top-k router with a fixed capacity per block keeps the graph static; the abstract reports the same FLOPs and the same
  wall-clock training as the baseline, and up to 50 percent faster post-training sampling. Its own training-speed number
  was not read.
- Evidence: all three are causal or BERT-scale pretraining, not a 10-minute protein run. The catalog's 008 and 009
  (more steps at a changed window) did not improve the loss, so more steps alone is not a guaranteed gain.
- What changes here: in blocks 6 to 29, only a fixed share (60 percent) of tokens per window pass through the attention
  and MLP. The masked residues are always kept, because they are the scored outputs; the dropped unmasked context
  bypasses those blocks and rejoins at block 30. The rule is our adaptation: Hou et al. keep the tokens the MLM loss
  finds hard, and here the hard tokens are the scored ones. Positions keep their true indices (ALiBi and RoPE read
  distance), so the gathered subset cannot re-index them.
- Expected cost: 3 to 4 hours, including a second flex block mask for the gathered length. Step: -20 to -25 percent
  (estimate: two-thirds of the blocks are in the middle, 40 percent of their tokens are saved, less gather and scatter).
  Compile: +20 to 50 s (the catalog's new-mask cost). The 117 s record compile plus that is 137 to 167 s, and the
  proposers' 15 s margin rule puts the line at 165 s, so the top of the range does not ride. Params: 0.
- Sketch (a NEW hook `route` in the block loop, because the base `Strategy` has none; the gathered blocks get the true
  positions and document ids):

```python
class TokenDrop(Strategy):
    BLOCKS = range(6, 30)                                # middle blocks; the first 6 and last 6 see every token
    KEEP = 0.6                                           # fixed share per window, so the shape is static
    def route(self, index, X, masked, positions):        # NEW hook; returns the kept indices or None
        if index not in self.BLOCKS: return None
        k = int(self.KEEP * X.shape[1])                  # static k
        score = torch.rand(X.shape[:2], device=X.device) + 2.0 * masked.float()   # masked always in the top k
        return score.topk(k, dim=1).indices.sort(dim=1).values                    # (b, k)
    # the block runs on X.gather(1, keep), with positions and document ids gathered the same way;
    # the dropped rows keep their block-6 value and rejoin at block 30 (Hou et al.; Raposo et al.)
```

- Risk: the step saving is an estimate, and the gain may not follow from it (008 and 009 show that steps alone can
  lose). The masked-count check (masked residues per window below k) must run once on the CPU before the compiled step,
  or the build fails loudly. The value residual from block 0 must reach every block's values, so the gather runs after
  the value residual is formed.

## 4. Product-key memory side path on one block

- Source: Berges et al., "Memory Layers at Scale", arXiv 2412.09764 (Meta FAIR, December 2024; ICML 2025). Abstract read:
  scaling laws with up to 128B memory parameters, pretraining to 1 trillion tokens, comparison with models of up to 8B
  parameters; memory models beat dense models with more than twice the compute and beat MoE models matched for compute
  and parameters; gains are largest on factual tasks. Lample et al., "Large Memory Layers with Product Keys", arXiv
  1907.05242 (2019): a 12-layer memory model beats a 24-layer baseline and runs about twice as fast (search summary;
  not read in full). The memory layers paper says dense layers are FLOP-bound and sparse memory layers memory-bandwidth
  bound (search summary of the paper's text).
- Evidence: the gains are at 1 trillion tokens and 8B parameters, against our 0.1 to 0.2 billion tokens of training. At
  this scale a knowledge store is the mechanism the paper credits, and protein MLM has no factual-QA analogue. Expect a
  tie.
- What changes here: one block's MLP output gets an additive path that reads 32 of 65,536 slots (256 by 256 product keys)
  per residue and returns their weighted sum. The side path sits outside the 36 x 768 x 2,048 blocks, and the rules
  call extra parameters free. At 50.3M it is a 25 percent addition to the record's 198M, and that should be reported
  beside the record's count.
- Expected cost: 4 to 6 hours (a gather kernel and its backward). Step: +1 to 2 percent (estimate: 3.2 GB of slot reads per
  step at bf16, about 1 ms at GH200 bandwidth, plus the scatter in backward). Compile: +10 to 20 s (estimate; no new
  block mask). Params: 50.3M in the value table, 0.2M in the keys.
- Sketch (hooks `build`, `initialize`, `sublayer_output`; the zero-initialized values make step 0 equal the record):

```python
class ProductKeyMemory(Strategy):
    BLOCK = 18                                           # one block's side path
    SIDE = 256                                           # 256 x 256 = 65,536 slots
    TOP = 32
    def build(self, model, max_tokens):
        self.keys1 = nn.Parameter(torch.randn(self.SIDE, HIDDEN // 2) * 0.02)
        self.keys2 = nn.Parameter(torch.randn(self.SIDE, HIDDEN // 2) * 0.02)
        self.values = nn.Parameter(torch.zeros(self.SIDE ** 2, HIDDEN))     # 50.3M, zero at step 0
    def sublayer_output(self, index, X, Y):              # X (b, l, 768) normalized input; Y (b, l, 768) block output
        if index != self.BLOCK: return Y
        t1, i1 = (X[..., :384] @ self.keys1.T).topk(self.TOP, dim=-1)      # (b, l, 32) each half
        t2, i2 = (X[..., 384:] @ self.keys2.T).topk(self.TOP, dim=-1)
        pair = (t1[..., :, None] + t2[..., None, :]).flatten(-2)           # (b, l, 1024) candidate scores
        slots = (i1[..., :, None] * self.SIDE + i2[..., None, :]).flatten(-2)   # (b, l, 1024) flat slot ids
        w, sel = pair.topk(self.TOP, dim=-1)                                 # (b, l, 32)
        picked = self.values[slots.gather(-1, sel)]                          # (b, l, 32, 768)
        return Y + (picked * w.softmax(-1)[..., None]).sum(-2)               # (b, l, 768)
```

- Risk: a tie is the likely result, and the gather path costs the most of any idea here if it is not fused. The keys
  give a single learned address per residue, so the record's zero-init and Muon groups need the sparse path in
  `parameter_groups`; the values are better in AdamW. Check the 1,024 candidates against the top-32 selection on the CPU
  before a GPU run.

## Considered and rejected (a number exists, but it does not carry to this model)

| Item | Source and number | Why not |
|---|---|---|
| STEM, token-indexed FFN up-projection | arXiv 2601.10639 (January 2026): accuracy up to 3 to 4 percent at 350M and 1B; a third of FFN parameters removed | A replacement, not an addition. Its rows are indexed by the token, so at a masked residue every row is the MASK row: no information at the scored position. It also removes the input-dependent unit selection the record uses |
| Gemma 3n per-layer embeddings (PLE) | no official number; secondary sources only (aman.ai primer; STEM paper, dimension 256 per layer) | Same MASK argument as STEM. The catalog's `embedding_mix` (pilot3: 2.7725 against 2.7718) already covers the every-block token residual |
| Hash embeddings | arXiv 1709.03933 (NeurIPS 2017): at least as good as regular embeddings; a third-party review quotes 40M against 200M parameters (not checked against the paper) | At 33 tokens nothing needs hashing; the exact table is smaller than any hashed one |
| Mixture of softmaxes, low-rank output | arXiv 1711.03953 (ICLR 2018): up to 3.6 points of PTB perplexity | The softmax bottleneck binds only when the vocabulary exceeds the embedding rank (d + 1 = 769); 33 classes sit far below it |
| Tied input and output embeddings | arXiv 1608.05859 (EACL 2017): perplexity reductions on PTB and translation models at half the size | The numbers were unreadable in our extraction, the head is 25k parameters, and the catalog tuned its two learning rates separately (022). A cheap check if Logan wants it, not a candidate |
| Recursive or shared-block transformers | arXiv 2410.20672 (ICLR 2025): recursive Gemma 1B beats TinyLlama 1.1B and Pythia 1B | Compares models of different size, not at equal compute; sharing cuts parameters, not step time, and the shape rule fixes the 36 blocks |
| Lightning-attention hybrid | arXiv 2501.08313 (2025): seven lightning layers to one softmax layer; 456B model | No MLM number read; built for long context, and our windows are 1,024 |
| Cross-layer K and V sharing | arXiv 2405.12981 (2024, read in the earlier file): halves the KV cache at near-equal accuracy | A memory result; no quality number read |
| Routing by the preceding layer's attention (A-MoD) | arXiv 2412.20875 (2024): no parameter cost for the router | No number read |
| Prefix-token prediction auxiliary loss | modded-nanogpt record 88 (1.256 to 1.243 minutes, 0.8 s) | Causal prediction; the catalog rejects it ("nothing to gain with 20 classes") |
| Key or query token embeddings | no number found | Token-indexed at a masked residue, as in the STEM row; the catalog's value embeddings are the open family for this |
| Record 92's hashed n-gram table, as a whole | modded-nanogpt record 92 (0.665 minutes): 84.6M-row table, 65B parameters, nine other changes | Not isolated; the table needs sparse gradient communication (record 71), and 65B parameters is far beyond one GH200 at 198M |

## Gaps and what to measure first

- The MASK-only corruption and the boundary tokens: confirm in `benchmark/data.py` and the mask code before any table
  idea. This scout read the catalog's statement, not the code.
- `strategy.py` hook signatures: the sketches assume them. `route` (idea 3) and any table-state dictionary (idea 2) are
  new and need one change to the block loop each.
- Engram's placement and gate: the full paper (arxiv.org/html/2601.07372v2) was not read, so the layers 2 and 15 and the
  gate are secondary-source claims.
- Mixture-of-depths and token dropping: the full papers were not read for a loss or a training-speed number; the 25
  percent and 57 percent figures are BERT or search-summary numbers.
- The record-62 pull request: read it to learn where the bigram vector enters. That decides whether the pair-table
  result (010) argues against idea 2 or only against its injection site.
- Cheapest measurement first: the record's step time with the middle blocks at 60 percent of tokens (a profile on the
  GH200 under the compile cap), which sets the ceiling for idea 3 before the build. Not run here.

## Sources

- Engram, arXiv 2601.07372: https://arxiv.org/abs/2601.07372 (HTML: https://arxiv.org/html/2601.07372v2)
- Mixture-of-Depths, arXiv 2404.02258: https://arxiv.org/abs/2404.02258
- Token Dropping for Efficient BERT Pretraining, arXiv 2203.13240: https://arxiv.org/abs/2203.13240
- Revisiting Token Dropping Strategy (ScTD), arXiv 2305.15273: https://arxiv.org/pdf/2305.15273
- Better and Faster LLMs via Multi-token Prediction, arXiv 2404.19737: https://arxiv.org/abs/2404.19737 (ICML: https://proceedings.mlr.press/v235/gloeckle24a.html)
- Masked Mixers, arXiv 2409.01482: https://arxiv.org/pdf/2409.01482
- NVIDIA Megatron-Bridge MTP: https://docs.nvidia.com/nemo/megatron-bridge/0.6.0/training/multi-token-prediction.html
- Memory Layers at Scale, arXiv 2412.09764: https://arxiv.org/abs/2412.09764
- Large Memory Layers with Product Keys, arXiv 1907.05242: https://arxiv.org/abs/1907.05242
- STEM, arXiv 2601.10639: https://arxiv.org/abs/2601.10639
- Gemma 3n primer: https://aman.ai/primers/ai/gemma-3n/
- Hash Embeddings, arXiv 1709.03933: https://arxiv.org/abs/1709.03933
- Using the Output Embedding to Improve Language Models, arXiv 1608.05859: https://arxiv.org/abs/1608.05859
- Breaking the Softmax Bottleneck, arXiv 1711.03953: https://arxiv.org/abs/1711.03953
- Relaxed Recursive Transformers, arXiv 2410.20672: https://arxiv.org/abs/2410.20672
- MiniMax-01 (lightning attention), arXiv 2501.08313: https://arxiv.org/abs/2501.08313
- Attention Routing for MoD (A-MoD), arXiv 2412.20875: https://arxiv.org/abs/2412.20875
- modded-nanogpt records and README: https://github.com/KellerJordan/modded-nanogpt
- Catalog: `C:\ws\projects\representation\speedrunning_plms\ideas\README.md` (families `neighbor` 010, `value_embeddings`
  011, `embedding_mix` pilot3, `dense` 022 family, `latent`, `batch` 008 and 009; the profile; the 15 s margin rule)
- Earlier scout: `C:\ws\projects\representation\speedrunning_plms\ideas\2026-10-30_long_protein_attention.md` (CLA
  arXiv 2405.12981, read there)
