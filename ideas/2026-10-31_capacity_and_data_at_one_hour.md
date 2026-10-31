# Capacity, memory, routing and data at one hour: literature scout

Date: 2026-10-31. A literature scout for SpeedrunningPLMs (36 blocks, 768 stream, 2,048 MLP, 15 percent MASK, one GH200).
The question is which capacity, memory, routing and data ideas pay off once a run is long enough to use them (3,600 s,
the 1-hour track started 2026-10-31 from `genomes/000_baseline.json`), under a fixed block shape. I read `program.md`,
`ideas/README.md` (families, rows 010 to 028, the profile), `search.py` SWITCHES and SPACE, `strategies/curriculum.py`,
`benchmark/data.py` (`corrupt`), and the 2026-10-30 capacity and noise files in full. The data-order file I read through
its headline numbers only. Nothing here is measured on this benchmark and nothing was run. Each gain is a source's
number or our estimate, and the text says which.

Shorthand: MLM is masked language modeling; MoE is mixture of experts; MoD is mixture-of-depths; MoR is
mixture-of-recursions; PKM is product-key memory; EMA is an exponential moving average; TPP is tokens per parameter; a
"pass" is one full read of the 393M-token stream (1.75M proteins); a "switch" is a key of `search.SWITCHES`; "covered"
means an existing switch or catalog row already tests the idea; one seed resolves about 0.0015 nats (README). Gains are
in nats of val_select CE, negative is better.

## Outcome

- Top three: (1) a multi-layer, shared-pool product-key memory, which extends the built `memory.product_keys` switch;
  (2) intermediate-block MLM heads with a decaying weight (deep supervision), which no switch covers; (3) a second pass
  over the stream with fresh masks, the one data question that only a 1-hour run can ask.
- The data ideas depend on a number that conflicts with the brief. At the record's profiled step (0.276 s, 65,536
  tokens a step, README), 3,600 s is about 13,000 steps, or about 850M token slots: about 2.2 passes over the 393M
  tokens. The first pass ends near 1,650 s; the second runs to about 3,300 s; the rest is a third pass. The brief says
  the stream is not seen once at 1 hour. The first 1-hour rung's step count decides which is right. If the 1-hour step
  is much slower than 0.276 s, the single-pass reading holds and idea 3 drops out.
- The MASK-only corruption decides which ideas are real. At a masked position the input is MASK, so any table or
  router indexed by the token's own identity is blind there. The ideas below use the neighbors, the model's own state,
  or position. `benchmark/data.py` `corrupt` confirms: MASK replaces each standard residue with probability 0.15 and
  nothing else.
- Expected gains are small. Every estimate here is under 0.01 nats, which is at most about seven seed bands. Expect ties
  on most rows; the 1-hour run is the first place the data ideas could separate.
- Two facts to reconcile before building. The `neighbor` row in `ideas/README.md` says the trigram table is hashed to
  2^16 rows (50M parameters); `2026-10-30_capacity_at_fixed_shape.md` says it is exact over 33^3 codes (35,937 rows, 27.6M
  parameters). With 33 tokens the exact form is the one the arithmetic supports. The paper summary of the memory
  layers does not state the optimizer for memory values; the switch uses AdamW, which is our choice, not a source's.

## Ranked table

Rank is by expected gain per implementation hour at 3,600 s. "Covered" says what the catalog already tests.

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats; our estimate) | Step / compile / params | Covered by |
|---|---|---|---|---|---|---|---|
| 1 | Shared-pool product-key memory with three reads (blocks 9, 18, 27) | Berges et al., arXiv 2412.09764 (Dec 2024; ICML 2025) | abstract; HTML v2 through a fetch summary (sections 1, 3.1.3, 3.1.4, 5.1 to 5.4) | 2 | 0 to -0.010 | +2 to 3 percent; compile +10 s; +0.8M per extra read | partial: `memory.product_keys` (one read, block 18) |
| 2 | Intermediate-block MLM heads (deep supervision, decaying weight; early-exit style) | Al-Rfou et al., arXiv 1808.04444 (2018); Elhoushi et al., LayerSkip, arXiv 2404.16710 (Apr 2024, v4 Oct 2024); Ren et al., DeepMIM (WACV 2025) | search summaries of 1808.04444 and DeepMIM; LayerSkip abstract and review | 1.5 to 2 | -0.000 to -0.006 | +0 to 1 percent; compile +0 to 5 s; 0 params | not covered: `latent.next_layer` predicts the detached final state, not identities |
| 3 | Second pass over the stream with fresh masks and a new permutation | Muennighoff et al., arXiv 2305.16264 (May 2023, NeurIPS 2023); Liu et al., RoBERTa, arXiv 1907.11692 (Jul 2019); protein caveat from arXiv 2411.02142 (Nov 2024) | abstracts and search summaries; RoBERTa Table 1 | 1 | -0.005 to +0.005 (sign open) | 0; compile 0 | not covered: `curriculum` has `none` and the tempered orders, all with no repeats |
| 4 | Input-side n-gram over-encoding (bigram and trigram embeddings summed at `embed`) | Huang et al., Over-Tokenized Transformer, arXiv 2501.16975 (Jan 2025); Liu et al., arXiv 2601.21204 (Jan 2026) | abstracts and summaries; no numbers used | 1.5 | 0 to -0.006 | +1 percent; compile +0 to 10 s; 28.4M params | partial: `neighbor.bigram_table` and `trigram_table` inject at the output, not the input |
| 5 | Learned router for the middle-block token drop (expert-choice scores, output gate) | Raposo et al., MoD, arXiv 2404.02258 (Apr 2024); Zhou et al., expert choice, arXiv 2202.09368 (Feb 2022); Bae et al., MoR, arXiv 2507.10524 (Jul 2025; NeurIPS 2025) | abstracts; MoR through summaries | 2 on top of the 3 to 4 h of capacity idea 3 | 0 to -0.005 | as capacity idea 3 (estimate -20 to -25 percent step if it rides); +18k params | partial: the capacity file's idea 3 and catalog B9 (random score, deferred) |
| 6 | Learned registers (unscored tokens per window), with a diagnostic first | Darcet et al., Vision Transformers Need Registers, arXiv 2309.16588 (Sep 2023; ICLR 2024); Goyal et al., pause tokens, arXiv 2310.02226 (Oct 2023; ICLR 2024) | search summaries; no benchmark number read | 3 to 4 | 0 to -0.004 | +0.4 to 0.8 percent residues lost per step; compile +10 s; <0.1M params | not covered |
| 7 | Expert-choice side-path MoE on one block (dense shape kept) | Zhou et al., arXiv 2202.09368; Puigcerver et al., soft MoE, arXiv 2308.00951 (Aug 2023; ICLR 2024); Roller et al., hash layers, arXiv 2106.04426 (Jun 2021; NeurIPS 2021); He, PEER, arXiv 2407.04153 (Jul 2024) | abstracts; summaries | 4 to 6 | 0 to -0.010; tie likely | +3 to 5 percent (one MLP-equivalent); compile +10 to 20 s; +3.1M per block | not covered |

Ideas 1, 3 and 5 are the three where a 1-hour test is the sensible first screen. Idea 2 is the cheapest new mechanism.

## 1. Shared-pool product-key memory with three reads (extends `memory.product_keys`)

- Source: Berges et al., "Memory Layers at Scale", arXiv 2412.09764 (Meta FAIR, December 2024; ICML 2025). Read: the
  abstract and the HTML version through a fetch summary, which gave section numbers. Quotes below are the summary's
  reading, not verified against the paper's text.
- Claim and evidence: the vanilla memory replaces one FFN in the middle of the network. The multi-layer version,
  Memory+, places memory layers with a stride of 4 (134M models) or 8 (larger), and "a shared pool of memory parameters
  across all memory layers" (section 3.1.3). Ablations find a sweet spot around three memory layers (section 5.4). The
  key dimension is half the model dimension (section 5.4). The 8B comparison (Table 2, section 5.3) reads: at 200B
  tokens, MMLU 41.35 for dense against 50.14 for Memory+, TQA 51.74 against 57.64; at 1T tokens, MMLU 59.68 against
  63.04, TQA 63.62 against 68.15, NQ 25.24 against 27.06. The paper says the gains are "more pronounced earlier in
  training (200B tokens)", and that the TQA and MMLU gaps shrink from 200B to 1T while NQ widens. The bandwidth claim is
  in section 1; section 3.1.4 hedges it and rests on a kernel benchmark, not an end-to-end profile.
- Why it might pay at 3,600 s: the paper's gains are largest at the early end of its own curve, and our 1-hour run is
  about 0.85B token slots, far below its 200B. That argues against an advantage at one hour, so this is a test, not a
  forecast.
- What changes here: `memory.layers` in {1, 3}. With three reads the reads sit at blocks 9, 18 and 27 (stride 9), each
  with its own 768 by 768 query and its own key halves, all reading one 65,536-row value pool. The pool stays 50.3M
  parameters; each extra read adds about 0.8M (estimate from the layout in the registry). The switch's `memory.block`
  range becomes the first read. Set the pool's learning rate and optimizer by the switch's `embed_lr` group until a
  source states otherwise.
- Expected cost: 2 hours on the built module (a loop over read blocks, one shared Parameter, per-read queries). Step:
  +2 to 3 percent (estimate: the registry's 6.4 GB of fp32 rows per read, about 2 ms at GH200 bandwidth, plus the
  backward scatter). Compile: +10 s (estimate; top-k and the bag fall back to ATen, as the switch's row says).
- Sketch (hooks as in the registry; the shared pool is the change):

```python
class SharedMemoryReads(Strategy):
    READS = (9, 18, 27)                                   # three reads, one value pool
    def build(self, model, max_tokens):
        self.values = nn.Parameter(torch.zeros(SIDE ** 2, HIDDEN))          # 50.3M, shared by all reads, zero at step 0
        self.queries = nn.ModuleDict({str(b): nn.Linear(HIDDEN, HIDDEN, bias=False) for b in self.READS})
        self.keys1 = nn.Parameter(torch.randn(len(self.READS), SIDE, HIDDEN // 2) * 0.02)
        self.keys2 = nn.Parameter(torch.randn(len(self.READS), SIDE, HIDDEN // 2) * 0.02)
    def sublayer_output(self, index, X, Y):               # the read is the existing product-key lookup, per block
        if index not in self.READS: return Y
        return Y + product_key_read(self, index, X)       # same top-32 gather and softmax as the switch
```

- Risk: the paper's own curve argues against a one-hour advantage (above). Three reads triple the memory traffic
  that the switch measured at one. The keys give one learned address per residue, so the sparse path needs the same
  `parameter_groups` care as the switch. Check the candidate set against the top-32 on the CPU before a GPU run.
- Covered: one read at block 18 is the switch, untested in the ledger (README: "open"). This idea is the extension, not
  a new family. The UltraMem line (arXiv 2411.12364, Nov 2024; UltraMemV2 arXiv 2508.18756, Aug 2025) is the same
  family with tiled values; its abstract gives no masked-LM number and its speed claim was not verified, so it is not a
  separate idea.

## 2. Intermediate-block MLM heads (deep supervision with a decaying weight)

- Sources: Al-Rfou et al., "Character-Level Language Modeling with Deeper Self-Attention", arXiv 1808.04444 (2018):
  auxiliary losses on intermediate layers with decaying weights, and only the final layer used at evaluation (search
  summary, not re-read). Elhoushi et al., "LayerSkip", arXiv 2404.16710 (Meta; first posted April 2024, v4 October
  2024): a training recipe of layer dropout plus an early-exit loss on every layer; a curriculum on the exit loss is
  needed because the loss on all layers from the start can slow training or hurt the last layer (review summary; the
  paper's own tables not read). Ren et al., "DeepMIM: Deep Supervision for Masked Image Modeling" (WACV 2025,
  openaccess.thecvf.com): auxiliary decoders on intermediate ViT blocks; its ablations say placement matters (paper
  summary). Deeply-Supervised Nets (Lee et al., 2015) is the origin of the method; not read here.
- Evidence: no source trains an MLM head on intermediate blocks of a bidirectional encoder for pretraining. The
  closest signals are the decaying schedule in the 2018 character model and the early-exit curriculum in LayerSkip.
  The 2026-10-30 capacity file's neighbor-prediction heads (idea 1 there) are the same kind of auxiliary head, and its
  evidence is also weak.
- Coverage: not covered. `latent.next_layer` (README) predicts the detached normalized last state at masked residues
  from block `source_block`; `latent.neighbors` predicts neighbor identities from the last state. Neither predicts the
  masked residue's own identity from an intermediate block through the output vocabulary. The `regularization` family's
  `early_dropout` drops sublayer outputs, not whole blocks; LayerSkip's whole-block layer dropout would need a graph
  change for its compute saving and is out of scope at one hour.
- What changes here: `deep_supervision` with one or two intermediate reads at blocks 18 and 27 (range 12 to 30). Each
  read applies the record's final norm and output projection to the stream at every position, with zero extra
  parameters, and scores the masked positions with the summed cross entropy, weighted by `weight` (0.1 start, range
  0.02 to 0.3). The weight falls linearly to 0 at `until`, a share of the clock (0.3 to 0.8). The final head is the only
  head used at evaluation.
- Expected cost: 1.5 to 2 hours. Step: +0 to 1 percent (estimate: about 3 GFLOP per read, negligible against 276 ms).
  Compile: +0 to 5 s. Params: 0.
- Sketch (the weight is a 0-d buffer the step callback refills, as in `early_dropout`, so the schedule costs no
  recompile; the loss uses weights, not boolean indexing):

```python
class DeepSupervision(Strategy):
    READS = (18, 27)
    def build(self, model, max_tokens):
        self.scale = torch.zeros((), device=...)           # refilled each step from the schedule
    def auxiliary_loss(self, model, states, targets, masked):    # states: {block: (b, l, 768)}
        terms = []
        for block in self.READS:
            logits = model.head(states[block])               # the record's head, shared; (b, l, 33)
            ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(masked)
            terms.append((ce * masked).sum() / masked.sum().clamp_min(1.0))
        return self.scale * torch.stack(terms).sum()
```

- Risk: the intermediate heads pull middle blocks toward token identity, which is the MASK-blind target at a masked
  position (the context must carry it). The decay is the guard, as LayerSkip's curriculum says. If the record's
  head is the two-layer head (switch `head_two_layer`), the intermediate reads use its final projection, not its
  dense layer; state that choice in the record.

## 3. A second pass over the stream with fresh masks (1-hour data reuse)

- Sources: Muennighoff et al., "Scaling Data-Constrained Language Models", arXiv 2305.16264 (May 2023; NeurIPS 2023):
  up to about 4 epochs of repeated text gives loss close to unique data at matched compute; returns diminish after
  that (abstract and search summary; the text is text, not proteins). Protein caveat: "Training
  Compute-Optimal Protein Language Models", arXiv 2411.02142 (November 2024): repeating UniRef50 gives diminishing
  returns for causal models and overfitting for masked models (search summary of the paper; its table and wording not
  read). Liu et al., "RoBERTa", arXiv 1907.11692 (July 2019), Table 1: dynamic masking against static masking on
  BERT-base is within noise (SQuAD 2.0 78.7 against 78.3; MNLI-m 84.0 against 84.3; SST-2 92.9 against 92.5). Wettig et
  al., "Should You Mask 15% in Masked Language Modeling?", arXiv 2202.08005 (February 2022; EACL 2023): 40 percent
  masking beats 15 percent for BERT-large on GLUE and SQuAD, and larger models favor higher rates. The rate is fixed by
  the rules here, so the paper's main result is not a lever.
- Our situation: `curriculum.py` draws one weighted permutation with no repeats ("no protein repeats, none is
  dropped"). A 180-second rung reads about 11 percent of the stream (652 steps, 43M tokens), so the tempered orders'
  pilot ties (family 2.7703, length 2.7706 against 2.7715) say nothing about a full pass. At 3,600 s the run reads
  about 2.2 passes, and the second pass starts near 1,650 s. The one-permutation design is a choice in our code, not a
  benchmark rule; if Logan reads it as a rule, that is his call.
- Protein evidence on repeats in the 10-minute regime: `ideas/README.md` row 013 (one protein per family per pass)
  gave the same curve as the unordered run, so family limits did not matter at 10 minutes. Nothing tests repeats at one
  hour.
- What changes here: `curriculum.repeat` in {`none`, `reshuffle`}. Under `reshuffle`, after the first permutation is
  exhausted, a new permutation with the same weights is drawn and reading continues. Masks are redrawn on every visit,
  because `corrupt` draws from a generator per call; confirm in the training loop that it is called per batch (read
  `corrupt`, not the loop).
- Expected gain: -0.005 to +0.005 (sign open). Muennighoff's result says a second pass costs little; the protein
  overfitting claim says it may cost something. The held-out `val_select` will show it either way.
- Expected cost: 1 hour (CPU; a second permutation, the loop change, a CPU check of the reading order). Step: 0.
  Compile: 0.
- Sketch (extends `tempered_order`; passes is ceil of the clock's tokens over the stream):

```python
def repeated_order(weights: np.ndarray, rng: np.random.Generator, passes: int) -> np.ndarray:
    return np.concatenate([tempered_order(weights, rng) for _ in range(passes)])   # same weights each pass
```

- Risk: MLM overfitting on repeated proteins; `val_select` is the guard. This idea changes what a run means relative to
  the 10-minute comparison, so the record must state the repeat setting; the comparison key already holds the budget.

## 4. Input-side n-gram over-encoding (bigram and trigram embeddings summed at `embed`)

- Sources: Huang et al., "Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling", arXiv 2501.16975 (January
  2025): the input vocabulary is enlarged with multi-gram tokens ("Over-Encoding"); the training loss falls log-linearly
  with input vocabulary size at every model size; the output side stays small (abstract and summaries; the input sum of
  1-, 2- and 3-gram embeddings is a third-party reading, not confirmed). Liu et al., "Scaling Embeddings Outperforms
  Scaling Experts in Language Models", arXiv 2601.21204 (January 2026, v2 February 2026): N-gram embedding scaling beats
  expert scaling at high sparsity, with diminishing returns in deeper models (secondary summaries and a model card; no
  number used here).
- Evidence: LLM-scale results; the MASK-only setting gives the context-only signal that these tables need. The local
  prior is against the output-side table: `pair_table` (010) was level at 270 s and behind after (2.7583 against
  2.7559). The open question is the injection site, not the table.
- Coverage: partial. `neighbor.bigram_table` and `trigram_table` inject at the output of block 35 (README), and the
  capacity file's idea 2 places them at blocks 2 and 15 (unverified reading). Over-encoding adds the same flank
  statistic to the input, so every block sees it.
- What changes here: a `neighbor` variant `input_ngram`: `embed` returns the token embedding plus a bigram row for
  (left, self) and (self, right) plus a trigram row for (left, self, right), exact indices (33^2 = 1,089 and 33^3 =
  35,937 rows), all zero-initialized, so step 0 is the record. At a masked position the centre is MASK, so the trigram
  row is the flank-conditioned statistic and nothing about the masked residue enters the input.
- Expected cost: 1.5 hours (the table code exists in `neighbor.py`; the lookup moves to `embed`). Step: +1 percent
  (estimate: three gathers of 768-wide rows per token). Compile: +0 to 10 s. Params: 28.4M (27.6M trigram, 0.84M
  bigram), outside the blocks; report beside the record's count.
- Expected gain: 0 to -0.006 (estimate). Risk: a tie or a loss, as the output-side table showed; the input-side variant
  is the test of whether injection site was the problem.

## 5. Learned router for the middle-block token drop (extends capacity idea 3)

- Sources: Raposo et al., "Mixture-of-Depths", arXiv 2404.02258 (April 2024): a learned top-k router with a fixed
  capacity per block keeps the graph static; the abstract reports the same FLOPs and matching wall-clock training
  (abstract read; the training-speed numbers not read). Zhou et al., "Mixture-of-Experts with Expert Choice Routing",
  arXiv 2202.09368 (February 2022; revised October 2022): experts choose their top tokens, so load is balanced without
  an auxiliary loss; a Hugging Face summary reports more than 2x faster convergence against Switch top-1 and GShard
  top-2 at matched compute (summary; abstract snippet cut off). Bae et al., "Mixture-of-Recursions", arXiv 2507.10524
  (July 2025; NeurIPS 2025): expert-choice routing beat token-choice at fixed compute in its comparison (secondary
  summaries; the "25 percent fewer FLOPs" figure appears in only one of two summaries that disagree, so it is not used).
- Coverage: the capacity file's idea 3 (token dropping, random scores with masked tokens forced in, 60 percent of
  tokens in blocks 6 to 29) and catalog B9 (deferred: per-token positions in the RoPE and slope score functions need a
  new block mask). This idea changes only the score: a learned router in place of `torch.rand`.
- Why it could matter: a random drop removes context at random. The context is the only information about a masked
  residue, so which context is kept may matter more than how much. Untested here.
- What changes here: `route.variant` in {`random`, `learned`}. Under `learned`, each middle block has one 768-to-1
  linear router (18k parameters over 24 blocks). The masked tokens are forced into the top set, and the block's output
  for each routed token is multiplied by its router score, so the router gets gradient, as in MoD.
- Expected cost: 2 hours on top of capacity idea 3's 3 to 4 hours. Step: as capacity idea 3 (its estimate is -20 to -25
  percent if it rides; the router adds one matmul per block). Compile: as idea 3.
- Expected gain: 0 to -0.005 (estimate), and only if capacity idea 3 shows the step saving. Risk: router collapse onto
  the same tokens; the masked-forcing rule must be explicit; a multiplicative gate changes the block's output scale,
  so check the record's residual scalars in the same run.

## 6. Learned registers, with a diagnostic first

- Sources: Darcet et al., "Vision Transformers Need Registers", arXiv 2309.16588 (September 2023; ICLR 2024): high-norm
  artifact tokens appear in ViT feature maps, mostly in low-information background regions; extra learnable tokens
  absorb that computation and are discarded at the output (arXiv listing and search summary; no benchmark number read).
  Their mechanism is credited to Memory Transformers (Burtsev et al., 2020). Goyal et al., "Think before you speak:
  Training Language Models with Pause Tokens", arXiv 2310.02226 (October 2023; ICLR 2024): pause tokens inserted at
  random pretraining positions and left out of the loss; pretraining-only pauses gave limited benefit, and the gains
  needed pauses in finetuning too (search summary; a public issue disputes the headline SQuAD percentage, so no number
  is used).
- Evidence: none for protein or MLM models. Two searches on protein or BERT register tokens returned nothing usable.
- Coverage: not covered.
- What changes here: `registers.variant` in {`none`, `learned`} with R = 4 or 8 per window. The window stays 1,024
  positions, so R of them are registers and the residues per step fall by 0.4 to 0.8 percent. Registers attend and are
  attended by every token of their window, carry no RoPE rotation and no ALiBi bias (so protein positions do not
  shift), and are excluded from the loss and from the head. The document mask must treat them as global.
- Diagnostic first (about a minute on the box, no training): measure the record's hidden-state norms at a middle block
  on `val_select`. If there are no high-norm tokens in the MLM, the artifact mechanism is absent, and the idea is cut.
- Expected cost: 3 to 4 hours (mask change, index handling, packing with R reserved slots). Step: the residue loss above.
  Compile: +10 s (estimate: a new mask with the same shape).
- Expected gain: 0 to -0.004 (estimate). Risk: the registers may simply absorb capacity the record does not need; the
  ViT evidence is for classification and dense features, not masked prediction.

## 7. Expert-choice side-path MoE on one block (dense shape kept)

- Sources: Zhou et al., arXiv 2202.09368 (above). Puigcerver et al., "From Sparse to Soft Mixtures of Experts", arXiv
  2308.00951 (August 2023; ICLR 2024): soft assignments, fully differentiable; the abstract's results are vision (ViT
  and contrastive setups); no masked-LM result found. Roller et al., "Hash Layers For Large Sparse Models", arXiv
  2106.04426 (June 2021; NeurIPS 2021): hashed routing with no learned router matches Switch and BASE; balanced and
  random hashes on the most local features worked best (abstract; table values from a fragmentary extraction, not
  used). He, "Mixture of A Million Experts", arXiv 2407.04153 (July 2024): product-key routing to single-neuron experts
  (PEER) beats dense FFN and PKM layers on performance per compute in language modeling (search summaries of the paper;
  no number used).
- Coverage: not covered. The record's `memory` family is the only sparse side path, and it is a value lookup, not an
  expert MLP.
- What changes here: a `moe_side` family: one side branch on block 18, added to the block output. E experts (8 to 32)
  of width 256 at E = 8 (so E times width equals the record's 2,048, one MLP-equivalent of work at capacity factor 1),
  an expert-choice router (768 to E) where each expert takes its top C = tokens / E tokens per window, and outputs
  weighted by the router's softmax. Expert choice balances load without an auxiliary loss, so no load term.
- The hash-routing caution: hashing on token identity would send every masked residue to one expert, because the
  input is MASK at each of them. If hashing is tried, hash on position or on the neighbors.
- Expected cost: 4 to 6 hours (gather, top-k per expert, scatter in backward). Step: +3 to 5 percent (estimate: one
  MLP-equivalent of 36 blocks, plus gather and scatter). Compile: +10 to 20 s (estimate). Params: +3.1M per block
  (outside the blocks; the shape rule leaves side paths free).
- Expected gain: 0 to -0.010 (estimate); a tie is likely at 2,048 MLP width and about 0.85B token slots. Risk: the
  added step cost trades against steps at fixed wall time, which the 1-hour budget makes concrete.

## Covered, checked, or rejected

| Item | Source and number | Status |
|---|---|---|
| Token dropping and fixed-capacity routing (MoD, BERT token dropping, MoR) | arXiv 2404.02258; arXiv 2203.13240 (March 2022; ACL 2022); arXiv 2507.10524 | covered: capacity idea 3 and catalog B9 (deferred); idea 5 here is the router change |
| Engram-style and bigram or trigram tables at the output | arXiv 2601.07372 (January 2026); `neighbor.bigram_table`, `trigram_table` | covered: capacity idea 2; `pair_table` harmful (010) |
| Hashed n-gram value embeddings (noise Track 3 #4 and #5) | noise README, fetched 2026-10-31 | covered by the neighbor and value-embedding families; no separate idea |
| Noise leaderboard, later tracks | noise README (fetched 2026-10-31): Track 1 top 3.300 at 43 min ("two-hop credit through next-token twins"), against 3.338 in the 2026-10-30 read; Tracks 2 and 3 unchanged (3.498 in 22.9 s; 3.211) | no capacity or data idea transfers: the Dust and noise methods are estimator-side, as the 2026-10-30 noise file says; the README gave no merge date for the 3.300 entry, which is not checked against the PR list |
| Length-bucket curriculum (Dataset Decomposition) | arXiv 2405.13226 (May 2024; NeurIPS 2024); Hugging Face summary: 3x faster to target on long-context benchmarks (summary; the 6x figure is unverified) | covered: `curriculum.length_tempered` (pilot tie at 180 s); re-test only with idea 3's repeats |
| ESM-2 uniform sampling over UniRef50 clusters | recAMPLIFY, arXiv 2610.26747 (October 2026), summary of ESM-2 | covered: `curriculum.family_tempered`; ESM-2's exact data pipeline not read |
| UltraMem and UltraMemV2 | arXiv 2411.12364 (November 2024); arXiv 2508.18756 (August 2025) | same family as idea 1; the 6x speed claim is a secondary summary, not used |
| PEER single-neuron experts | arXiv 2407.04153 (July 2024) | inside idea 7's family |
| Soft MoE | arXiv 2308.00951 (August 2023) | idea 7's alternative; vision-only evidence, not built |
| Masking rate | arXiv 2202.08005 (February 2022) | fixed at 15 percent by the rules; not a lever |
| Dynamic masking | arXiv 1907.11692 (July 2019) | the corruption already redraws per call; no switch needed; differences within noise |
| LayerSkip layer dropout | arXiv 2404.16710 | whole-block skip is outside the current `regularization` family and needs a graph change; out of scope at one hour |
| Mixture of softmaxes, prefix-token loss and similar | 2026-10-30 capacity and MLM files | not repeated here |

## Gaps and what to measure first

- The pass count. Read the first 1-hour rung's step count (`search_1h`) and the stream token count it implies. This
  decides whether idea 3 is a real question (about 2.2 passes) or not (one pass).
- The memory optimizer. The paper summary states no optimizer for the memory values; read the paper's section 3.1 or
  its code before choosing a learning rate for idea 1's pool.
- The trigram table: hashed to 2^16 rows (README) or exact over 33^3 codes (capacity file). Reconcile before idea 4.
- The loop's `corrupt` call. Confirm it runs per batch, so each visit gets fresh masks (idea 3).
- Protein evidence for repeats and deep supervision. Two searches each found none beyond the 2411.02142 snippet. Its
  table is the first thing to read for idea 3.
- Registers: the norm diagnostic in idea 6 is the cheapest test, and it can kill the idea before any build.
- Sources read only through abstracts, summaries or fetch summaries: 2412.09764 (fetch summary of the HTML), 1808.04444,
  2404.16710, DeepMIM, 2309.16588, 2310.02226, 2501.16975, 2601.21204, 2202.09368, 2308.00951, 2106.04426, 2407.04153,
  2507.10524, 2405.13226, 2610.26747, 2411.02142, 2411.12364, 2508.18756. Their numbers are used only where marked, and
  none enters a gain estimate.

## Sources

- Memory Layers at Scale, arXiv 2412.09764 (December 2024): https://arxiv.org/abs/2412.09764 ; HTML v2:
  https://arxiv.org/html/2412.09764v2
- Scaling Data-Constrained Language Models, arXiv 2305.16264 (May 2023): https://arxiv.org/abs/2305.16264
- Training Compute-Optimal Protein Language Models, arXiv 2411.02142 (November 2024): https://arxiv.org/html/2411.02142v1
- RoBERTa, arXiv 1907.11692 (July 2019): https://arxiv.org/abs/1907.11692
- Should You Mask 15% in Masked Language Modeling?, arXiv 2202.08005 (February 2022; EACL 2023):
  https://arxiv.org/abs/2202.08005
- Character-Level Language Modeling with Deeper Self-Attention, arXiv 1808.04444 (2018): https://arxiv.org/pdf/1808.04444
- LayerSkip, arXiv 2404.16710 (April 2024; v4 October 2024): https://arxiv.org/abs/2404.16710
- DeepMIM: Deep Supervision for Masked Image Modeling (WACV 2025):
  https://openaccess.thecvf.com/content/WACV2025/papers/Ren_DeepMIM_Deep_Supervision_for_Masked_Image_Modeling_WACV_2025_paper.pdf
- Vision Transformers Need Registers, arXiv 2309.16588 (September 2023; ICLR 2024): https://arxiv.org/abs/2309.16588
- Think before you speak: Training Language Models With Pause Tokens, arXiv 2310.02226 (October 2023; ICLR 2024):
  https://arxiv.org/abs/2310.02226
- Over-Tokenized Transformer, arXiv 2501.16975 (January 2025): https://arxiv.org/abs/2501.16975
- Scaling Embeddings Outperforms Scaling Experts in Language Models, arXiv 2601.21204 (January 2026):
  https://arxiv.org/abs/2601.21204
- Mixture-of-Experts with Expert Choice Routing, arXiv 2202.09368 (February 2022): https://arxiv.org/abs/2202.09368
- From Sparse to Soft Mixtures of Experts, arXiv 2308.00951 (August 2023; ICLR 2024): https://arxiv.org/abs/2308.00951
- Hash Layers For Large Sparse Models, arXiv 2106.04426 (June 2021; NeurIPS 2021): https://arxiv.org/abs/2106.04426
- Mixture of A Million Experts, arXiv 2407.04153 (July 2024): https://arxiv.org/abs/2407.04153
- Mixture-of-Depths, arXiv 2404.02258 (April 2024): https://arxiv.org/abs/2404.02258
- Mixture-of-Recursions, arXiv 2507.10524 (July 2025; NeurIPS 2025): https://arxiv.org/abs/2507.10524
- Token Dropping for Efficient BERT Pretraining, arXiv 2203.13240 (March 2022; ACL 2022): https://arxiv.org/abs/2203.13240
- Dataset Decomposition, arXiv 2405.13226 (May 2024; NeurIPS 2024): https://arxiv.org/abs/2405.13226
- recAMPLIFY, arXiv 2610.26747 (October 2026): https://arxiv.org/html/2610.26747
- Ultra-Sparse Memory Network, arXiv 2411.12364 (November 2024): https://arxiv.org/abs/2411.12364 ; UltraMemV2, arXiv
  2508.18756 (August 2025): https://arxiv.org/pdf/2508.18756
- noise repository and README (fetched 2026-10-31): https://github.com/qlabs-eng/noise
- Repository files: `program.md`, `ideas/README.md`, `search.py` (SWITCHES, SPACE), `src/speedrunning_plms/strategies/curriculum.py`,
  `src/speedrunning_plms/benchmark/data.py` (`corrupt`), `ideas/2026-10-30_capacity_at_fixed_shape.md`,
  `ideas/2026-10-30_noise_speedrun_transfer.md`, `ideas/2026-10-30_data_order_and_sampling.md` (headline numbers)
