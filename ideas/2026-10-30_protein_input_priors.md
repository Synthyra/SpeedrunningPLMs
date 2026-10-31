# Protein input priors and inductive biases for the 10-minute protein MLM

Date: 2026-10-30. A literature review for SpeedrunningPLMs (36 blocks, 768 wide, 2,048 MLP, 15 percent MASK, one GH200,
600 s, about 2,150 steps of 0.28 s, about 140M residues seen). It read `ideas/README.md`, `ideas/2026-10-30_protein_pretraining.md`
and `ideas/protein_pretraining.md`, and checked the other 2026-10-30 idea files where they touch input priors
(`mlm_pretraining`, `attention_and_output_head`, `capacity_at_fixed_shape`, `data_order_and_sampling`,
`regularization_and_averaging`), so it does not repeat them. Nothing here is measured on this benchmark. Each gain is an
estimate and says so; a source's number is quoted only where one exists. Shorthand: PLM is protein language model; MLM is
masked language modeling; CE is cross entropy; RoPE is rotary position embedding; BLOSUM is the BLOcks SUbstitution
Matrix (BLOSUM62 is the version used); PMI is pointwise mutual information; SS is secondary structure; TPP is tokens per
parameter; "nats" is validation CE; one seed resolves about 0.0015 nats (README).

## Outcome

- No candidate has a measured gain here. The expected gains are estimates from about -0.003 to +0.010 nats, all small
  against the 0.0015 nat seed band.
- Yardstick (rough): one unit of log(steps) is worth about 0.026 nats. This is the catalog's own slope (001: 1.53 times the
  steps gave about 0.011 nats). So 1 percent more steps is about 0.00026 nats, the seed band is about 6 percent of steps,
  100 saved steps of 2,150 are about 0.0012 nats, and losing a third of the steps costs about 0.01 nats.
- Most input-side priors are functions of the token alone. The 33-row embedding table is free and trained with AdamW, so a
  fixed per-token feature is already in the table's span. Such a prior can only change the start of training, and a
  start-up gain is bounded by the yardstick (about 0.0012 nats per 100 steps saved, at most).
- Priors that add a new function class are the ones that depend on position in the document, on the document's length,
  or on a pair of positions. Three candidates are of that kind: a fixed periodic distance bias (section 1), protein length
  and terminus tokens (section 2), and an attention pair prior (section 5).
- Best candidate: a fixed periodic distance bias on the attention scores (helix period 3.6 residues, strand period 2). By our
  arithmetic, RoPE's highest rotary frequency cannot express either period at any base (section 1). Expected -0.001 to
  -0.003 nats (about 4 to 12 percent of steps). It is a constant term beside the existing local bias, so compile cost is
  near zero.
- Negative evidence on the table-style ideas: the catalog's (left, right) neighbor pair table (010) lost 0.0024 nats, and
  Ieremie et al. (2024) found that models on reduced amino-acid alphabets did not beat the full alphabet at 76 million
  sequences. Both argue against the coarse-context ideas (section 4).
- How fast substitution statistics are learned: no source found that times this in a PLM (section "How fast the
  statistics are learned"). In the run the statistics are data-rich (about 21 million masked targets), so the open
  quantity is steps, not data.
- Literature check: the 2023 to 2026 PLM recipes in `protein_pretraining.md` (sections 1.1 to 1.15) contain no substitution,
  physicochemical, periodicity or pair-prior input. A grep of that file for these terms finds no hits. None of the
  candidates below has a 2023 to 2026 protein MLM result with a number.
- Record level: the yardstick uses the 2.7146 nat record given in the benchmark statement. The catalog's own record row
  (028) reads 2.7319 on val_select. This file does not reconcile the two. The slope does not depend on the level.

## Ranked table

Gain is the expected change in validation CE (estimate; negative is better). "Read" says how much of the source was read.
Compile is the added seconds of warmup, against a record of 106 to 122 s and a 180 s cap (headroom 58 to 74 s). Step
cost is against the 0.276 s profile in the catalog. Hours are implementation estimates.

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats) | Compile | Step cost | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | Fixed periodic distance bias on the attention scores (helix 3.6, strand 2) | arXiv 2012.03084 (2020); arXiv 2602.23179 (2026) | abstract and snippets; RoPE range is our arithmetic | 1.5 | -0.001 to -0.003 | +0 to 5 s | +0 to 3 percent | a head near period 4 in layers 3 to 9 (snippet); no period prior tested; local bias 012 gave 0.0028 |
| 2 | Protein length and terminus tokens (document-level, not spanned by the table) | no source tested it (catalog: CLS and EOS are in the stream) | none | 2.5 | -0.000 to -0.003 | +0 to 5 s | about 0 | none |
| 3 | Output-side amino-acid grouping as an auxiliary loss | no source for the output loss; Ieremie et al. (2024) for the reduced-alphabet input | full text (Ieremie); none for the auxiliary | 2 | -0.000 to -0.002 | +0 to 5 s | +0 to 1 percent | reduced alphabets lost to the full alphabet |
| 4 | Reduced-alphabet flank table (6 by 6 classes, additive to the embedding) | Ieremie et al. (2024); catalog 010 | full text (Ieremie); catalog | 2 | -0.000 to -0.001 | +0 to 5 s | +0 to 1 percent | 010 (full pair table) lost 0.0024 |
| 5 | Fixed attention pair prior from sequence co-occurrence at separations 1 to 4 | Zhang et al. (2024); Vig et al. (2020) | snippets | 5 | 0 to +0.002 | +20 to 40 s (estimate; captured buffer) | +5 to 10 percent (estimate) | ESM-2 stores separation-dependent pair statistics (snippet); captured-tensor slow path (catalog 021) |
| 6 | Per-token embedding initialization (BLOSUM rows, Atchley factors, PMI vectors) | immuneML docs; arXiv 2007.06225 (2020); arXiv 2004.03497 (2020) | snippets | 1 | 0 to -0.001 | 0 s | 0 | embeddings learn biophysics without a prior (snippets) |
| 7 | BLOSUM-weighted soft targets (a change of the loss target) | arXiv 1906.02629 (2019); ICML 2024 (EvoRank); arXiv 2304.08299 (2023) | abstract; snippets | 1.5 | +0.000 to +0.010 (a loss) | 0 s | 0 | none for MLM; the hard-label metric penalizes the blur |

The top five for a run are ranks 1 to 5. Rank 1 is the only candidate with a new function class and a near-zero compile
cost; rank 2 needs packer metadata; rank 5 is the costliest on compile and step time.

## 1. Fixed periodic distance bias on the attention scores

- Source. Filipavicius et al., "Pre-training Protein Language Models with Label-Agnostic Binding Pairs" (arXiv 2012.03084,
  2020). Read: the abstract (fetched; the PDF could not be read through the fetch tool). The claim that some heads follow
  about every fourth residue in layers 3 to 9, linked to helix hydrogen bonding, comes from a search snippet and is not
  verified against the full text. Induction Meets Biology (arXiv 2602.23179, 2026; ICML 2026 listing): read as a search
  summary. Early-layer heads attend to fixed positional offsets; ESM-3 engages neurons sensitive to SS for repeat
  detection, and ESM-C does not.
- What the sources did there. Neither tested a periodicity prior. There is no number to carry over. The head observation
  is descriptive.
- Expressibility (our arithmetic, not a measured result). A rotary pair i rotates by theta_i times the distance, with
  theta_i = base^(-2i/d_head). So theta_0 = 1 radian per residue at any base, and the shortest period a rotary pair can
  express is 2 pi, about 6.3 residues. A 3.6-residue period needs 1.75 radians per residue and a 2-residue period needs pi.
  The catalog's `rope_fraction` (partial RoPE) only removes rotary pairs and cannot add a period. The local bias (012) is
  linear in distance, so it is monotone. A fixed periodic term is a function that none of the current positional terms can
  express.
- The change. This is a kernel edit beside 012, not a strategy module: the hooks do not reach the score. In the flex
  `score_mod` that carries the local bias, add a fixed term for each head: `amp_h * cos(2 pi (q_idx - kv_idx) / period_h)`,
  with period_h in {3.6, 2.0} and amp_h in {0.25, 0.5} as constants, not learned. Learned slopes inside `score_mod` were
  invalid (021, compile 264 s); fixed constants are what 012 already uses. The document mask is applied by the block mask,
  so cross-protein pairs stay masked. Family: `positional`, a new `periodic` variant with `amp` and the head subset as
  search numbers.

```python
# inside the flex score_mod that carries the local bias (012); amp and period are fixed per-head constants
def score_mod(score, b, h, q_idx, kv_idx):
    d = (q_idx - kv_idx).to(score.dtype)
    score = score + local_bias(h, d)                                   # the existing 012 term
    return score + PERIODIC_AMP[h] * torch.cos(2 * math.pi * d / PERIODIC_PERIOD[h])
```

- Which heads. 014 showed that removing the local prior from half the heads loses most of 012's gain, so the periodic term
  goes on every head at small amplitude first. A helix-only subset is the second arm.
- Expected gain: -0.001 to -0.003 nats (estimate; about 4 to 12 percent of steps at the yardstick). The only precedent is
  012's 0.0028. Sign confidence is low, because the local signal may already be used and Hart et al. (2026, abstract and
  summary only) report that protein models lean more on semantic than positional information than text models do, which
  would shrink any positional prior.
- Compile and step cost: constants inside the existing kernel, so +0 to 5 s of compile (estimate). Step +0 to 3 percent
  (one cosine per score element, fused in the kernel; estimate).
- Risk: a period-2 term is a parity bias and adds little beyond the local slope. Run the amplitude screen before any head
  split. Implementation: 1.5 hours.

## 2. Protein length and terminus tokens

- Source. None tested in an MLM. ProGen (arXiv 2004.03497, 2020; snippet) reports per-position perplexity falling along a
  protein, so context helps more later in a protein. It gives no terminus prior. The catalog's neighbor note says the
  stream carries CLS and EOS as standard visible tokens, so the N-terminus is already visible to attention at distance 1.
- The argument. Attention reaches CLS and EOS, so the terminus is partly visible. The protein's total length is not: it
  would take a count across the whole protein, and the packed 1,024-token window carries no length. A length token
  gives every residue the protein's scale. The catalog reports the test loss flat across length bins (2.746 to 2.762),
  while the reference models gain most on long proteins. A length table does not directly add context, but it is a
  document-level function that no per-token table spans, so it is not redundant with the embedding.
- The change. A strategy module with an `embed` term added per token. Learned tables: `LEN[bin(doc_len)]` over 16 log-spaced
  bins, and `NTERM[min(i, 15)] + CTERM[min(L - 1 - i, 15)]` for the distance from each terminus, counted over standard
  residues only. Initialize all of them to zero, so the run starts at the record. If the `embed` hook receives only input
  ids, the packer's per-token document index and length must be passed in: a small `train.py` change, about 1 hour of the
  2.5 hour estimate.
- Expected gain: -0.000 to -0.003 nats (estimate). Sign is open: a length table can also overfit 16 bins.
- Compile and step cost: a fixed-shape additive term, so +0 to 5 s and about 0 step time.
- Risk: length correlates with family, and the held-out families are split at 30 percent identity. Check the length
  distribution of the validation and test splits against the train split before reading any gain as real.

```python
# embed hook (sketch): per-token terms from the packer's document index and length
x = tok_emb(input_ids) + LEN[len_bin] + NTERM[nt_idx] + CTERM[ct_idx]   # all three tables zero at init
```

## 3. Output-side amino-acid grouping as an auxiliary loss

- Source for the output loss. None found for MLM. The nearest result is on the input side. Ieremie, Ewing and Niranjan
  (Bioinformatics 40(2), 2024; full text read) pretrained three-layer BiLSTMs (about 14M parameters, 76,215,872 UniRef90
  sequences, 10 percent masking, 240k steps over about 6 days) on eight reduced alphabets, where the reduced alphabet was
  the model's token set, not an auxiliary target. Their Table 2 accuracies (full alphabet UNIPROT20, then the best reduced
  alphabet, HSDM17): REACT 26.3 against 25.4, Fold 43.3 against 42.1, Superfamily 90.7 against 87.9, Family 81.8 against
  80.6. The authors conclude that reduced alphabets are "unlikely to improve how language models learn." A 5 or 7 class
  alphabet lost more on every column. Their models are BiLSTMs, not transformers.
- The argument. A group cross entropy computed from the same 20-way logits is a marginal of the distribution the main loss
  already fits, so at the optimum it adds nothing. Any gain is early reweighting: a coarse target is easier, so the
  auxiliary term acts as a curriculum on the output side. The evidence for that reweighting is only the reduced-alphabet
  result, which points the other way.
- The change. A strategy module with an `auxiliary_loss` hook. Six fixed groups over the 20 standard residues: aliphatic
  (AVILM), aromatic (FWY), positive (KRH), negative (DE), polar (STNQ), and special (GPC). The group logit is the logsumexp of
  its members' main logits at masked positions. The auxiliary CE is added with weight lambda in {0.05, 0.1, 0.2}. No new
  parameters.

```python
# auxiliary_loss hook (sketch; names illustrative). GROUP_MASK is a fixed (6, 20) bool, one row per group
z = state.logits[keep][:, STANDARD_IDS]                                   # (n, 20), the main 20-way logits
grouped = torch.stack([torch.logsumexp(z.masked_fill(~GROUP_MASK[g], float("-inf")), -1)
                       for g in range(6)], dim=-1)                        # (n, 6)
return self.weight * F.cross_entropy(grouped, GROUP_OF[labels[keep]])
```

- Expected gain: -0.000 to -0.002 nats (estimate). Larger lambda trades the 20-way fit against the group fit, so screen
  lambda at 0.05 first.
- Compile and step cost: the auxiliary graph adds a logsumexp over six slices of 20 columns at about 15 percent of
  positions, so +0 to 5 s of compile and +0 to 1 percent step (estimate).
- Implementation: 2 hours. Test on a fixed mask set so the group term cannot change the masking.

## 4. Reduced-alphabet flank table

- Sources. Ieremie et al. (2024; section 3) for the alphabet result. The catalog's 010: a (left, right) pair table of 1,089
  by 768 added to the embedding, level at 270 s and behind after (2.7583 against 2.7559, 0.0024 nats).
- The argument. A masked position's center is MASK, so the only context a pair table can read is its two flanks. A 6 by 6
  table on the section 3 classes has 36 rows, with about 583,000 masked targets per row (21 million over 36), against about
  19,000 per row for 010's 1,089 rows. Data per row was never the limit for 010, so the coarse version is a smaller copy
  of the statistic that already lost. The reasoning predicts a repeat of 010, not a win.
- The change. An `embed` term `FLANK[class(left), class(right)]`, a learned 36 by 768 table, zero at init, AdamW at the
  embedding rate. Each position reads the ids of its two neighbors, which are in the stream. At unmasked positions the same
  table is added; the center's own embedding keeps the unmasked residue's identity.
- Expected gain: -0.000 to -0.001 nats (estimate), with 010 as a negative anchor.
- Compile and step cost: +0 to 5 s of compile (a gather), +0 to 1 percent step.
- Implementation: 2 hours.

## 5. Attention pair prior from sequence co-occurrence

- Sources. Zhang, Wayment-Steele, Brixi, Wang, Kern and Ovchinnikov (PNAS, 2024; bioRxiv 10.1101/2024.01.30.577970). Read: a
  search summary. ESM-2 learns pairwise dependencies conditioned on sequence motifs and on the separation between segments,
  and stores coevolution statistics (categorical Jacobian analysis; no number read). Vig et al. (arXiv 2006.15222, 2020).
  The PDF could not be read through the fetch tool, so the BLOSUM agreement is from search snippets only: a Pearson
  correlation of 0.73 between attention-derived similarity and BLOSUM in the paper, and 0.80 in a slide deck of the same work.
  The two figures disagree, so neither is carried into this file as a result.
- What the sources did there. Nothing was used as a prior. The statistics appear in trained models, and no source says
  when in training they appear.
- The argument. A fixed pair table on the attention logits gives the early layers a head start on residue-pair
  statistics that the model would otherwise learn itself. The table has more freedom than 010 (it acts on scores, per
  head and per separation), but it is the same statistic. The sequence-only statistic is available: log-odds of residue
  pairs at separations 1 to 4 from the training split (PMI). Contact statistics would be the better prior, but the benchmark
  ships 3Di strings and not CA coordinates, so contact maps need a new data input (`protein_pretraining.md`, idea 4).
- The change. A kernel edit: a fixed table `P[h, tok_q, tok_kv, sep_class(d)]` (33 by 33 by 4 separation classes per head,
  or head-shared), added in `score_mod`. The table gathers token ids from a captured buffer. The catalog's 021 measured a
  captured learned parameter at 28 s a step. A captured constant buffer has not been measured here, so the compile risk is
  real: the proposers' cap rule (skip a genome predicted within 15 s of the cap) applies.
- Expected gain: 0 to +0.002 nats (estimate). The closest anchor is 010's loss.
- Compile and step cost: +20 to 40 s of compile (estimate, unmeasured); step +5 to 10 percent (estimate, one gather per
  score element).
- Implementation: 5 hours. Run only after rank 1, and only if the compile sum fits the cap.

## 6. Per-token embedding initialization

- Sources. Atchley factors are five numbers per residue, from a multivariate analysis of 494 amino-acid attributes (immuneML
  encoding documentation; a snippet). ProtTrans (arXiv 2007.06225, 2020): an analysis of the token embedding layer of
  pretrained models found charge, polarity, size and hydrophobicity captured without any prior (snippet; a trained
  model, not an initialization test). ProGen (arXiv 2004.03497, 2020): randomly initialized embeddings learned to cluster
  residues by physicochemical property (snippet). No source tested BLOSUM rows or PMI vectors as an initialization.
- The argument. For a per-token feature f(a), the table E[a] is free, so E[a] = f(a) W is inside its span for any W. A
  fixed feature changes only the start point and the optimizer's path. The yardstick bounds the gain: a start that saves
  100 steps is worth about 0.0012 nats at most. The table itself is cheap to fit: each standard residue is a masked target
  about 1 million times in the run, so the table's biophysical structure appears in the first few hundred steps (estimate).
- The change. An `initialize` hook. Standard-residue rows are set to a fixed random projection of the 20 by 20 BLOSUM62
  rows (or of the Atchley rows), scaled to the record's embedding standard deviation. MASK, CLS, EOS and PAD rows stay random.
  The table keeps AdamW at the embedding rate and no parameter group changes.

```python
# initialize hook (sketch): fixed projection of BLOSUM62 rows into the 20 standard residue rows
P = torch.randn(20, D, generator=gen) / math.sqrt(20)                     # fixed, seeded
init = BLOSUM62_20.float() @ P                                            # (20, D), half-bit scores
E.data[STANDARD_IDS] = init * (E.data.std() / init.std())
```

- Expected gain: 0 to -0.001 nats (estimate; 0 to 4 percent of steps).
- Compile and step cost: 0 s and 0 percent. Implementation: 1 hour.
- The same argument covers the other per-token priors: classical SS propensity scales, class-frequency vectors, and
  training-set PMI vectors. Each is a choice of start point for a table that the run learns anyway.

## 7. BLOSUM-weighted soft targets

- Sources. Muller, Kornblith and Hinton, "When Does Label Smoothing Help?" (arXiv 1906.02629, 2019). Read: abstract. The
  reported effects are better calibration and a weaker distillation teacher, not the hard-label CE. Gong et al., "Evolution-Inspired
  Loss Functions for Protein Representation Learning" (ICML 2024; EvoRank). Read: a listing and snippets. It replaces the one-hot
  wild-type label with a distribution from the counts in a multiple sequence alignment (MSA), and its numbers were not read.
  arXiv 2304.08299 (2023): a structure-based mutation-effect network that uses BLOSUM62 to shape label smoothing (snippet;
  the identity of that paper with the snippet was not confirmed). DPLM-Evo (arXiv 2605.00182, 2026): a substitution kernel
  from a masked model (snippet; no ablation).
- The argument. Let q_y(a) be the soft target for true residue y and P(y | c) the true posterior at a masked position with
  context c. The population minimizer of the soft CE is p*(a | c) = sum over y of P(y | c) q_y(a), not P(a | c). The hard-label
  CE of p* exceeds that of P by a term that grows with the smoothing weight and with the kernel's blur. BLOSUM62 is a
  half-bit log-odds score, so q_y(a) proportional to p(a) times 2^(B[y, a] / 2) puts mass on similar residues (I, L, V, M
  and their neighbors). That blur falls on the confusions the model makes most, so the cost is largest where it matters.
  The only mechanism for a gain is lower gradient variance in the early phase, and nothing measured here supports it. The
  MLM rules allow a change of the loss target and forbid a change of the corruption, so this is in scope.
- The change. Replace the hard CE on masked positions with `(1 - lambda) CE(hard) + lambda CE(q_y)` in an `auxiliary_loss`
  hook (or the loss), with q from a fixed 20 by 20 matrix. lambda in {0.02, 0.05, 0.1}.
- Expected gain: +0.000 to +0.010 nats, a loss, with the high end at lambda 0.1 (estimate). Run lambda 0.02 only as a control.
- Compile and step cost: 0 s and 0 percent (a fixed gather and a second CE on masked positions). Implementation: 1.5 hours.

## How fast the statistics are learned

- Searches found no study that times when a protein model's substitution or contact statistics appear across pretraining
  checkpoints. ESM-2 analyses (Zhang et al., 2024) read the final model. Cheng et al. (arXiv 2411.02142, 2024; abstract
  read) report that MLM overfits when UniRef is repeated, and give no timing. Hart et al. (arXiv 2602.20449, 2026; summary)
  compare how protein and text models store information across layers, not across steps.
- Our own arithmetic. The run sees about 140M residues, so about 21 million masked targets. Spread over a 20 by 20
  substitution table, that is about 52,000 targets per cell. The statistic is data-rich in the run, so the limit is the number
  of steps, not the data. A prior that encodes only a 20 by 20 table can help the first steps, and the yardstick caps that
  near 0.0012 nats per 100 steps. More than that needs a change in function class (sections 1, 2 and 5).
- What would settle it, cheaply: a probe of pair statistics on any kept checkpoints of the existing record runs at about
  50, 200 and 500 steps, if such checkpoints exist. This is a recommendation, not run here.

## Checked and rejected

- Unigram or class-frequency head bias. Per the benchmark notes this was tried and tied. The catalog's
  `attention_and_output_head` file says a 33-value output bias is a unigram prior the head can already approximate, expected
  0. The start point is 2.996 nats (uniform over 20) against 2.893 (unigram floor, README), a 0.10 nat difference that the
  head closes in tens of steps. The yardstick gives about 0.0002 nats for 20 steps.
- Hashed k-mer and n-gram tables. Covered by `capacity_at_fixed_shape.md` (idea 2: exact bigram and trigram context with a
  gate, -0.000 to -0.004) and by the catalog's 010. Hashing addresses collisions that a 33-token vocabulary does not have.
- Domain BPE or k-mer tokens. The 20-way label space and the MASK corruption are fixed. PETA (arXiv 2310.17415, 2023)
  finds that subword vocabularies of about 50 to 200 units work best in its setting and that sizes above 800 hurt. A NeurIPS
  2026 listing (neurips.cc/virtual/2026/160495) claims domain BPE helps scaling; read as an abstract listing only, unverified.
- Bucket masking (Walton et al., arXiv 2605.16581, 2026). Masks residues that are close in 3D, so it changes the corruption,
  which is fixed. Its up to 14 percent gain is on 17 proteins in downstream fitness tasks, not an MLM loss.
- Contact-statistics pair priors. They need CA coordinates, which the benchmark does not ship. 3Di strings give no contact map.
- SS propensities computed from structure at test time. Test proteins may not carry structure, and an SS target from DSSP
  would be a new label. The per-token version is section 6.
- Signal peptide handling. No source with a number. The N-terminal segment is already in context through CLS and attention.
- Low-complexity handling. Covered by `data_order_and_sampling.md`, idea 5.
- Hierarchical softmax as the output parametrization. The 33-way head is small, and the catalog's
  `attention_and_output_head` notes the softmax bottleneck does not bind (33 is below the 769 hidden width). Only the
  auxiliary form (section 3) is kept.
- ProteinBERT (Brandes et al., Bioinformatics 2022): a Gene Ontology auxiliary loss. It needs GO labels, which are not in
  the stream. No small-budget ablation of the auxiliary loss was found.
- Learned periodic or pair biases inside `score_mod`. Captured learned parameters ran at 28 s a step (catalog 021). Only
  fixed constants are kept.
- Physicochemical features concatenated to the stream (rather than added at the embedding). It is the same per-token
  argument as section 6: no gain beyond the start point.
- Attention-based amino-acid similarity as a substitute for BLOSUM. Attention already tracks substitution similarity
  (Vig et al., snippet), which is one more reason the prior would be redundant after the first steps.

## Sources

- Ieremie, Ewing, Niranjan, "Protein language models meet reduced amino acid alphabets," Bioinformatics 40(2), 2024.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10872054/ (full text read; Tables 2 and 3)
- Filipavicius, Manica, Cadow, Rodriguez Martinez, "Pre-training Protein Language Models with Label-Agnostic Binding
  Pairs Enhances Performance in Downstream Tasks," arXiv 2012.03084, 2020. https://arxiv.org/abs/2012.03084 (abstract read;
  periodicity claim from a snippet)
- Vig et al., "BERTology Meets Biology: Interpreting Attention in Protein Language Models," arXiv 2006.15222, 2020.
  https://arxiv.org/abs/2006.15222 (PDF not readable through the fetch tool; snippets only; the 0.73 and 0.80 figures disagree)
- Zhang, Wayment-Steele, Brixi, Wang, Kern, Ovchinnikov, "Protein language models learn evolutionary statistics of
  interacting sequence motifs," PNAS 2024. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11551344/ (search summary)
- "Induction Meets Biology: Mechanisms of Repeat Detection in Protein Language Models," arXiv 2602.23179, 2026.
  https://arxiv.org/abs/2602.23179 (search summary; ICML 2026 listing)
- Hart et al., "Protein Language Models Diverge from Natural Language: Comparative Analysis and Improved Inference," arXiv
  2602.20449, 2026. https://arxiv.org/abs/2602.20449 (summary only)
- Cheng et al., "Training Compute-Optimal Protein Language Models," arXiv 2411.02142, 2024. https://arxiv.org/abs/2411.02142
  (abstract read)
- Muller, Kornblith, Hinton, "When Does Label Smoothing Help?," arXiv 1906.02629, 2019. https://arxiv.org/abs/1906.02629
  (abstract read; also in the catalog's `attention_and_output_head`)
- Gong et al., "Evolution-Inspired Loss Functions for Protein Representation Learning," ICML 2024.
  https://mlanthology.org/icml/2024/gong2024icml-evolutioninspired (listing and snippets; numbers not read)
- "Accurate and Definite Mutational Effect Prediction with Lightweight Equivariant Graph Neural Networks," arXiv 2304.08299,
  2023. https://arxiv.org/abs/2304.08299 (identity with the BLOSUM smoothing snippet not confirmed)
- DPLM-Evo, arXiv 2605.00182, 2026. https://arxiv.org/pdf/2605.00182 (snippet; no ablation)
- ProtTrans, arXiv 2007.06225, 2020. https://arxiv.org/abs/2007.06225 (snippet)
- ProGen, arXiv 2004.03497, 2020. https://arxiv.org/abs/2004.03497 (snippet)
- immuneML encoding documentation, Atchley factors. https://docs.immuneml.uio.no/immuneML.encodings.atchley_kmer_encoding.html
  (snippet)
- Walton, Goel, Aghazadeh, "Structure-Aware Masking for Protein Representation Learning," arXiv 2605.16581, 2026.
  https://arxiv.org/abs/2605.16581 (abstract snippet)
- PETA, "Evaluating the Impact of Protein Transfer Learning with Sub-word Tokenization on Downstream Applications," arXiv
  2310.17415, 2023. https://arxiv.org/abs/2310.17415 (snippet)
- Brandes, Ofer, Peleg, Rappoport, Linial, "ProteinBERT: a universal deep-learning model of protein sequence and function,"
  Bioinformatics 38(8), 2022. https://academic.oup.com/bioinformatics/article/38/8/2102/6502274 (snippet)
- NeurIPS 2026 listing on domain BPE tokenization. https://neurips.cc/virtual/2026/160495 (listing; unverified)

## Caveats

- Most sources were read from abstracts, listings or search snippets. Two PDFs (Vig et al. and Filipavicius et al.) could
  not be read through the fetch tool, so their claims stand as snippets. The Vig figures disagree (0.73 and 0.80).
- Ieremie et al. used BiLSTMs at about 14M parameters, not transformers at 36 blocks. Its result is evidence about the
  reduced-alphabet input, not a measurement of the auxiliary loss.
- Every gain above is an estimate, and most are within one seed's band. Only the periodic bias (rank 1) and the length
  tokens (rank 2) add a function class; the rest adjust the start point or the target.
- The yardstick is rough: its slope comes from the catalog's 001 step change, not from a measurement of these priors.
- The arXiv 2304.08299 link is a candidate source for a snippet, not a confirmed one. The Frontiers 2022 AAindex encoding
  paper from the search results has no URL retained and is not cited.
