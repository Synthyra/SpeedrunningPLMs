# Data order, selection and weighting: literature scout (2026-10-30)

Scope: how the stream is read under a fixed token budget when the data is larger than the budget. Read-only: no code was run and no existing file was edited. Catalog read first: `ideas/README.md` (2026-10-28 version). Nothing below repeats a catalogued result: short-first curricula (027, pilot2), family-ordered streams (013) and the shape of the batch family (008, 009) are the context for the ideas, not new ideas.

## What the budget implies

- The stream holds 393M tokens (1.75M proteins, mean about 225 tokens each once the two special tokens are counted). A run consumes about 2,100 steps of 65,536 tokens, so about 138M tokens: roughly 25 to 36 percent of the stream, each token seen once. Selection is therefore a live choice. Repetition is not: no token is ever repeated, so repetition-based data-constrained results (below) do not apply.
- The current reader (`train.py` line 172, the `batch` family docstring) takes contiguous 1,024-token windows of the shuffled stream and cuts proteins at window edges. The document mask only separates proteins that share a window. A protein cut at an edge loses its left context in the next window. This is the one defect in the reader that the literature addresses directly (idea 1).
- Conflict to check: the brief says the record gains least on long proteins. The catalog's row 028 says the value residual's gain grows with length (0.025 nats over 500 residues) and that most of it is on long proteins. I used the catalog's reading; please confirm which is right before weighting any length idea.

Evidence grades: **A** a primary paper with a number on a language model of comparable kind; **B** a number on another domain, a secondary source, or a protein sampling claim seen in one excerpt; **C** prior only, nothing measured for masked protein models.

## Ranked ideas

Ranked by expected loss gain per implementation hour under the 10-minute clock. Every precomputation below runs once on the CPU from `train_offsets.i64`, the family ids and the lengths; none adds per-step compute unless stated.

### 1. Best-fit packing of whole proteins into the 1,024-token windows

- **Source:** Ding et al., "Fewer Truncations Improve Language Modeling," ICML 2024, arXiv 2404.10830 (read in full).
- **Evidence (A, language models at 2k and 8k):** packing adds 0.0024 percent more sequences than concatenation at 2k on RefinedWeb (Table 2). Reading-comprehension average rises from 46.71 to 48.92 at 2k (Table 4). Perplexity on natural language at 2k is 9.64 for concatenation and 9.53 for packing (Table 10). Undefined-name hallucination falls by up to 58.3 percent (code). The paper's ablation shows a gain from keeping documents whole, not from a bigger window.
- **What it changes here:** the `batch` family gains a `bestfit` variant. Only proteins longer than 1,024 are cut, and those cuts are the only truncations kept. The 64 by 1,024 shape of the record is unchanged. Bins are filled by best-fit decreasing from a buffer of, say, 4,096 proteins drawn in the seed-0 shuffled order, so the stream's randomness is kept.
- **Cost:** CPU once, O(N log L) with L = 1,024. Estimate: seconds to a minute in Python for 1.75M proteins. No per-step cost. The document mask already exists.
- **Sketch:**
  ```
  lengths = np.diff(offsets)                       # includes the 2 specials
  chunks = split proteins longer than 1024 into pieces of at most 1024
  for each buffer of 4096 chunks in shuffled order:
      best-fit-decreasing into bins of capacity 1024 (segment tree or sorted list)
      emit (bin_id, protein_id, start, length) rows
  save bin index (int32, about 400k rows of segments)
  reader: step s takes bins [64s, 64s+64); gather tokens; pad unused slots; doc ids for the mask
  ```
- **Risks:** (a) Padding is not free: at a 98 to 99 percent fill, about 1 to 2 percent of the tokens a step sees are padding, which cuts real tokens per step by that much. (b) Check whether RoPE positions and the ALiBi-style local bias restart at each protein or run across the window; if they run across, a packed protein sees different positions than the record did. Confirm in `train.py` before the run. (c) Proteins longer than 1,024 are still cut (009's finding that long proteins lose stays true). (d) Expected gain is grade B at best; it is the cheapest test of whether context at window edges matters.

### 2. Family-tempered reading order

- **Sources:** ESM-2, Lin et al., Science 2023 (one excerpt seen via search, grade B): sequences sampled with even weight over about 43 million UniRef50 clusters. ProtBFN, Nature Communications 2025, doi 10.1038/s41467-025-58250-2 (search snippet, grade B): clusters sampled with probability proportional to the square root of their size, against inverse-size weighting in one model; ProtGPT2 and EvoDiff use equal weight per cluster. Bias preprint, bioRxiv 10.1101/2024.03.07.584001 (2024, preprint): two-stage UniRef sampling and large clusters dominate; ProGen2 and ESM2 show species bias from this.
- **Evidence:** no loss number for masked protein models at small scale (grade C). The cited sources argue debiasing, not a loss gain. Among them the only protein precedent for cluster-balanced reading is ESM-2's uniform-over-clusters sampling.
- **What it changes here:** the reader draws proteins with weight count(family)^(-alpha) instead of uniformly. alpha = 0 is the current behavior in distribution; alpha = 0.5 is ProtBFN's choice; alpha = 1 is equal weight per family. With a 25 to 36 percent budget, alpha decides which families appear at all.
- **Cost:** CPU once, about a second for 1.75M proteins.
- **Sketch:**
  ```
  counts = np.bincount(family, minlength=F)
  w = counts[family].astype(np.float64) ** (-alpha)
  keys = rng.random(n) ** (1.0 / w)            # Efraimidis-Spirakis weighted order
  order = np.argsort(-keys)                    # protein order fed to the window reader
  ```
- **Risks:** The stream carries family ids and lengths only (`benchmark/data.py`), so taxonomy cannot be reweighted directly; family balance is a proxy, and the bias preprint says unevenness is taxonomic. Small families are often orphans or low quality. Test-side: the evaluation is family-disjoint, so coverage of many families may help, but one seed cannot separate that from noise. Report val_select by family-size bin and length bin before reading the total.

### 3. Length-tempered reading order (long proteins up)

- **Sources:** Dataset Decomposition, Pouransari et al., NeurIPS 2024, arXiv 2405.13226 (read in full; the length-mixture result is grade A on language models). Length warmup and mixed-length curricula, Li et al. 2022 (arXiv 2108.06084, from search only).
- **Evidence:** in Dataset Decomposition, a mixture of lengths reaches the baseline's accuracy in about a third of the time at 8k context, and the authors report the gain is significant only when the target length is long. For proteins, the catalog's 028 result points the same way (long proteins gain most from context). No protein length-mixture result was found.
- **What it changes here:** the same reader as idea 2 with weight length^beta; beta = +0.5 favors long proteins, beta = -0.5 favors short ones. The retired short_first curriculum was about warmup (compile), not the sign of the length effect, so beta < 0 is not ruled out by it.
- **Cost:** CPU once, as idea 2. No per-step cost.
- **Risks:** the test metric weights tokens, so long proteins already carry most of the metric; oversampling them shifts the mix toward the test metric's own weighting, which is correct only if long-protein loss is where the record is weakest. Measure the length-bin table, not only the total. Keep the mix fixed in time (no curriculum), so no new compile graph.

### 4. Per-token selection with a cheap reference (Rho-1 adapted)

- **Source:** Lin et al., "Rho-1: Not All Tokens Are What You Need," arXiv 2404.07965, 2024 (read in full, abstract and sections 2 to 3).
- **Evidence:** Selective Language Modeling keeps the top k percent of tokens by excess loss (model loss minus reference loss). k is 60 percent for the 1.1B model and 70 percent for the 7B (Section 3.1); the 60 percent is a heuristic, not tuned per model (Section 3.5). Reported gains are on causal models in continual pretraining: up to 30 percent absolute few-shot gain on 9 math tasks at 15B tokens, and 6.8 percent average over 15 tasks on an 80B general run. The reference was a separate model trained on about 0.5B curated math tokens. No scoring overhead is quantified. No masked-model result was found (grade A for causal, C for masked).
- **What it changes here:** the masked-token loss is restricted to the top k percent of masked positions by excess loss. The reference should be cheap: a neighbor table, counts of the residue given its left and right neighbors, built once on the CPU from the stream (20 by 20 by 20 cells plus specials). Excess = the model's masked loss minus the table's loss for that position. Keep this a loss-mask change; the batch and the step count stay the same.
- **Cost:** CPU table, about a minute. Per step: one quantile over 65,536 values and a masked mean, negligible against a 0.276 s step.
- **Sketch:**
  ```
  table[l, r, y] = counts over the stream (CPU, bincount on a flattened index)
  ref_nll = -log( table[l, r, y] / table[l, r, :].sum() )        # per masked position
  excess = nll_model.detach() - ref_nll
  keep = excess >= torch.quantile(excess, 1 - k)                  # k = 0.6
  loss = (nll_model * keep).sum() / keep.sum()
  ```
- **Risks:** the evaluation is unweighted masked cross entropy, so dropping 40 percent of positions optimizes a different objective; the Rho-1 gains were on downstream few-shot, not on the pretraining loss. The neighbor table ignores long-range context, so "excess" will flag exactly the positions that need context; that could help (the catalog's open gap) or amplify noise. Use k = 0.8 as the first rung. Check the per-length-bin table.

### 5. Low-complexity and entropy screen (keep the high-complexity proteins)

- **Sources:** no protein language-model ablation found for low-complexity filtering (searched). The argument against redundancy cuts: Fournier et al., "Protein Language Models: Is Scaling Necessary?", bioRxiv 10.1101/2024.09.23.614603 (v2; grade B, search snippet): reducing redundancy can harm performance. The argument for hard examples when data is abundant: Sorscher et al., "Beyond neural scaling laws," NeurIPS 2022, arXiv 2206.14486 (theory and ImageNet, grade B: with abundant data keep hard examples; the theory is a perceptron, not a language model). Data removal in a protein model: Maiwald et al., arXiv 2610.26651 (2026): filtering viral proteins from ESM-2 degrades viral masked-token loss and grows with model size, and the authors say filtering is not a robust safeguard (grade B, abstract read).
- **Evidence:** none at this scale (grade C). The sign is open: Sorscher's rule favors hard (high-complexity) examples at abundant data, which is this regime (393M tokens against a 138M budget). Hard is defined here by a proxy, not measured.
- **What it changes here:** the reader draws from proteins ranked by Shannon entropy over residues, or by a windowed SEG-like complexity score, keeping the top 25 to 36 percent.
- **Cost:** CPU, a few minutes for 1.75M proteins with a vectorized count (about 37M cells).
- **Sketch:** `H_p = -sum(p log p)` over the residue counts per protein via `np.add.at`; `order = argsort(-H)`, then the same reader as idea 2.
- **Risks:** low-complexity proteins (collagens, coiled coils, disordered regions) are real biology, so dropping them is a composition bias, and it is the kind of selection the brief rules out unless audited. Run it only as an ablation against a random subset of the same size, with the length and family-size tables.

### 6. Perplexity-based pruning with a small reference, one shot

- **Sources:** Ankner et al., "Perplexed by Perplexity," arXiv 2405.20541 (2024; ICLR 2025): a 125M reference prunes data for a 3B model; the ICLR 2025 version reports up to 2.04 points average downstream and 1.45 times fewer steps to reach baseline (the 2024 workshop version reports 1.35 points). Marion et al., "When Less is More," arXiv 2309.04564 (2023): perplexity-ranked pruning beats EL2N and memorization; the best variant keeps the middle of the perplexity ranking with a 52B reference, not the hardest examples; the pruned models are worse on the unpruned test split and better downstream. Both are language models; abstracts and snippets read, not full text (grade B).
- **What it changes here:** a reference scores every protein once; the reader keeps the middle 25 to 36 percent.
- **Cost:** not free under the rules. A 125M reference trained for the same clock, then one forward pass over 393M tokens. Estimate (mine, not measured): the forward alone is about 1e17 FLOPs, a few minutes at realistic GH200 throughput, and the reference training about 10 minutes. This is expensive compute and needs Logan's word before anything runs.
- **Risks:** a pretrained protein model as the scorer would be a leakage path: ESMC and similar models saw UniRef, which may hold test families. Use only a reference trained on the training stream, never a published model. The middle-subset finding means dropping both the most and the least predictable proteins; check the bins.

### 7. Online high-loss batch selection (High-PPL, OPUS)

- **Sources:** OPUS, arXiv 2602.05400 (2026; read in part): per-step selection from a candidate buffer, with an additional compute overhead of 4.7 percent (abstract). The same paper's comparison, GPT-2 XL with Muon, 30B update tokens on FineWeb (Table 3): random 40.29, High-PPL (keep the highest-loss sequences) 39.82, GREATS 39.23, OPUS 41.75 average downstream accuracy. OPUS reaches random's validation loss at 60B tokens with 17B. Mindermann et al., RHO-LOSS, ICML 2022, arXiv 2206.07137: 18 times fewer steps on Clothing-1M, an image set with noisy labels (grade B, not a language model).
- **Evidence:** on language models, the naive high-loss rule (High-PPL) is below random in several settings; only OPUS's gradient-based rule beats random, and it is not cheap (grade B for language, C for masked protein models).
- **What it changes here:** a buffer of twice the windows is scored by the current model before each step.
- **Cost:** a forward pass costs about a third of a training step, so scoring a buffer of twice the windows adds about two-thirds of a step per step, roughly 1.67 times the step time (my arithmetic). That kills the gain unless scoring is far cheaper than a forward pass. The OPUS 4.7 percent figure uses ghost gradients and a sketch; implementing that is days, not hours. It also adds a new graph shape for the buffer, which is the warmup risk seen in 027.
- **Risks:** high-loss selection favors outliers and noise (the RHO-LOSS paper's own point). Rank low.

### 8. Length-bucketed power-of-two decomposition

- **Source:** Dataset Decomposition, Pouransari et al., NeurIPS 2024, arXiv 2405.13226 (read in full). Step time 244 ms against 304 ms for the 8k baseline at OpenLM-1B (Table 1); target accuracy reached about 3 times faster at 8k; more than 2 times data efficiency (Figure 1a). The authors state the speedup is significant only when the target length is long.
- **Evidence:** grade A for language models at 8k; no protein result.
- **What it changes here:** each protein is split into power-of-two pieces and sampled per bucket; every step keeps the same token count. This is the fixed-shape reader's opposite: several window shapes.
- **Cost:** CPU for the split; but several shapes means several compiled graphs. 027 measured two static graphs at a 249 s warmup against the 180 s cap, so this is blocked by the cap unless the shape count is one. The split also cuts long proteins into fragments, which removes the context 028 learned to use.
- **Verdict:** rank last. Keep only as a reference point for the length effect.

## Considered and dropped

- **Repetition and data-constrained scaling:** Muennighoff et al., "Scaling Data-Constrained Language Models," NeurIPS 2023, arXiv 2305.16264 (search snippet): up to about 4 epochs of repeated data is close to fresh data. Cheng et al., "Training Compute-Optimal Protein Language Models," arXiv 2411.02142 (NeurIPS 2024; abstract read): MLM overfits at 3B when UniRef50 is repeated, and adding metagenomic sequences removed the plateau. Neither applies, since no token is repeated here. Note only: a repetition regime would need both.
- **Within-family redundancy pruning:** Fournier et al. (above) argue it can hurt. Dropped.
- **DoReMi:** Xie et al., NeurIPS 2023, arXiv 2305.10429 (read in search snippet): a 280M proxy sets domain weights for an 8B model, 2.6 times fewer steps. The proxy run exceeds the clock; the domains it would need are the family and length bins already covered by idea 2 and 3.
- **ProGen3 data mixture:** Profluent Protein Atlas v1 (3.4B proteins, 1.1T tokens), bioRxiv 10.1101/2025.04.15.649055 (2025). Only the abstract and the cluster-size CDF at 50 percent identity (Figure 1c) were seen; the sampling procedure was not retrieved. Unverified.
- **AMPLIFY sampling:** Fournier et al. (bioRxiv 10.1101/2024.09.23.614603) compare UniRef50, UniRef90 and UniRef100 training; the sampling claim (UniRef90 members within UniRef50 clusters) comes from a secondary blog and is unverified. Re-read the full text before quoting it.
- **Data saturation:** Spinner et al., arXiv 2507.22210 (2025; abstract read): no saturation was found on yearly UniRef100 snapshots from 2011 to 2024. The data quantity is not the limit; selection is a separate question.
- **Per-protein loss normalization:** weighting each protein equally would mismatch the token-weighted test metric. Dropped.

## Top five, if one is run first

1. Best-fit whole-protein packing (idea 1): CPU only, 1 to 2 hours, no per-step cost, fixes a known defect of the reader. First check RoPE and local-bias position resets.
2. Family-tempered reading order (idea 2): CPU only, about 1 hour; the sign of the effect is open, so run alpha = 0.5 and alpha = 1 against 0.
3. Length-tempered reading order (idea 3): CPU only, same reader, beta = +0.5 first; conflicts with the brief's statement on long proteins, so check that first.
4. Cheap-reference token selection (idea 4): about 3 to 4 hours, negligible per-step cost; the objective changes, so the k = 0.8 rung comes first.
5. Low-complexity screen (idea 5): CPU only, low evidence; only as an ablation against a random subset of equal size.

Not run anything. Rule check: the perplexity-pruning reference (idea 6) and the online selection (idea 7) need Logan's word on compute before anything runs.
