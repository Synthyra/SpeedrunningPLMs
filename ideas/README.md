# The ideas queue

What to try on the 10-minute AFDB v6 speedrun, ranked by expected gain per run. Each row names the change to
`train.py`, where the evidence lives (the four literature files beside this one, read on 2026-10-28), and the risk.
Rows are crossed into the campaign ledger (`experiments/representation/speedrunning_plms/2026-10-28_afdb_v6_speedrun/runs.jsonl`
in the workspace) as they are tried; a kept change stays in `train.py`, a reverted one is noted here with its number.

The baseline already holds: 36 pre-norm blocks, 12 heads of 64 with QK-norm and RoPE, ReLU-squared MLP, zero-init
output projections, untied head, bf16 autocast, `torch.compile`, flex attention with a document mask over packed
1,024-token windows (64 a step, 65,536 tokens), Muon with Polar Express on the block matrices, AdamW on the
embedding and head, 2 percent warmup and 40 percent cooldown by clock fraction, Muon momentum warmup, the
collapse-free cross entropy, an eval on `val_select` each minute.

Expected gains are guesses for this benchmark unless a source number is quoted; the modded-nanogpt percentages are
record-to-record changes on a different benchmark and transfer only as a prior. Costs are my hours. The test is
always the same: `val_select.loss` of the selected weights against the incumbent, seed 0, same comparison key.

## Tried

| Run | Change | Verdict | val_select | Test |
|---|---|---|---|---|
| 000 | the baseline above | incumbent | 2.7680 | 2.7671 |
| 001 | Muon banked by parameter shape: one batched Polar Express per shape instead of 216 per-matrix calls; a profile had put the per-matrix loop at 76 percent of host time with the GPU idle a third of each step | kept, 1,412 to 2,157 steps | 2.7570 | 2.7554 |
| 002 | A1: Muon lr 0.04 | kept by the rule, within noise | 2.7564 | 2.7546 |
| 003 | A2: tanh logit softcap at 30 | kept by the rule, within noise | 2.7559 | 2.7539 |
| 004 | the incumbent (003) at seed 1, for the noise floor | reference: one seed resolves about 0.0015 nats | 2.7574 | 2.7551 |
| 005 | A4: U-net skips with learned scalars | reverted: not lower, 12 fewer steps | 2.7568 | 2.7545 |
| 006 | A12: weight EMA (decay 0.995) over the cooldown, scored and selected | reverted: led mid-cooldown, lagged at the end; a shorter memory is untried | 2.7578 | 2.7560 |
| 007 | A10: learnable exclusive self-attention | reverted: behind at every eval, 4 percent fewer steps | 2.7610 | 2.7593 |
| 008 | A1: 32 windows a step | reverted: 1.8x the steps, ahead at 90 s, behind from 180 s | 2.7579 | 2.7557 |
| 009 | A11 without the curriculum: 512-token windows, 128 a step | reverted: 6 percent more steps, long proteins lose | 2.7600 | 2.7580 |
| 010 | A7 for MLM: a (left neighbor, right neighbor) pair table, 1,089 x 768, added to the embedding | reverted: level to 270 s, behind after | 2.7583 | 2.7563 |
| 011 | A5: zero-initialized value embeddings on blocks 0 to 2 and 33 to 35 | reverted: behind at every eval from 90 s, 10 fewer steps | 2.7594 | 2.7573 |
| 012 | A15: fixed ALiBi-style per-head local bias on the scores beside RoPE | kept: ahead at every eval from 90 s, 4 percent fewer steps; the first gain beyond noise since 001 | 2.7531 | 2.7501 |
| 013 | A14: one protein per family per pass, packed contiguously | reverted: the same curve as 012 within noise; the budget is not family-limited | 2.7532 | 2.7502 |
| 014 | A16: the local bias on six heads, six free | reverted: loses most of 012's gain; the prior is wanted on every head | 2.7548 | 2.7515 |
| 015 | B1: NorMuon on 012 | kept: ahead from 180 s; the second gain beyond noise | 2.7495 | 2.7457 |
| 016 | A3: cautious weight decay 0.1 on the Muon banks, on 012 | reverted: behind from 180 s | 2.7564 | 2.7535 |
| 017 | A1: embedding lr 3e-3, head lr 1e-3, on 012 | reverted: inside the noise band of 012; retried on 015 | 2.7527 | 2.7495 |
| 018 | A1: Muon lr 0.06, on 012 | reverted: behind from 270 s; past the optimum | 2.7551 | 2.7521 |
| 019 | A1: cooldown over the last 60 percent, on 012 | reverted: the same within noise; ahead mid-run, level at the end | 2.7534 | 2.7498 |
| 015 s1, s2 | the record (015) at seeds 1 and 2 | reference: three-seed test mean 2.7447, span 0.002; the noise band holds | 2.7482, 2.7480 | 2.7447, 2.7437 |
| 020 | A6: mimetic V and O initialization, W_O W_V = 0.25 I at the start, on 015 | reverted: the same curve; what a block starts as is forgotten in 2,000 steps | 2.7496 | 2.7453 |
| 021 | A16 (second form): the local-bias slopes as per-head parameters inside the flex attention score function, Adam at the embedding rate, on 015 | invalid: compile warmup 264 s against the 180 s cap, then 28 s a step for 21 steps; a captured parameter in `score_mod` takes a far slower path. Learned slopes need a mechanism outside the kernel (a per-head scale on the bias after the fact, or a fixed grid) | | |
| 022 | A1: embedding lr 3e-3, head lr 1e-3, on 015 | kept by the rule: lower by 0.0005, as 017 was on 012; inside one seed's band | 2.7490 | 2.7455 |
| pilot | B7: the MLP's matmuls in FP8 (e4m3, per-tensor dynamic scales, `torch._scaled_mm`) | not run: 0.284 s a step against 0.276 in bf16 in a 60 s pilot; the quantization overhead exceeds the saving at 768 x 2,048 | | |

A second profile of the incumbent (after 001) puts the step at 0.276 s, all of it GPU time: matmuls 109 ms at about
715 TFLOP/s (near the bf16 peak of the GH200), flex attention 80 ms (its backward 55), the banked Muon 26 ms, RMS
norms 17 ms, the ReLU-squared 8 ms. More steps now come from attention (shorter windows, A11) or from FP8 matmuls
(Tier B), not from the host.

The reference rows say where the ten-minute model stands: its test loss is flat across length bins (2.746 to 2.762)
and 0.14 nats under the unigram floor (2.893), while ESM2 8M (2.625) and the larger models gain most on long
proteins. After about 142M tokens the blocks have learned residue statistics and little use of context, so the
next candidates make context cheap to use from step 0 (A15, A5) and spend the token budget on more families (A14),
before the optimizer refinements (A3, B1).

## Tier A: cheap, likely, first

| # | Change | Evidence | Expected | Risk |
|---|---|---|---|---|
| A1 | Learning-rate and batch sanity: Muon lr 0.01 / 0.04, embed and head lr x3 and /3, 32 and 128 windows a step, cooldown 0.45 and 0.6, momentum 0.95 flat | every speedrun starts here; records 26, 43 | up to several percent, the largest early gain | none |
| A2 | Logit softcap, tanh at 30, then the asymmetric rescale 23 sigmoid((z + 5) / 7.5) | records 9, 18, 54 (2.4 percent) | about 1 percent | low |
| A3 | Cautious weight decay on Muon and Adam, decay scheduled with the lr | records 43, 50 (1.25 and 0.84 percent) | 1 to 3 percent | low |
| A4 | U-net skips: the second half of the blocks adds a learned scalar times the matching first-half output | record 11 (7.7 percent); `architecture_optimization.md` 5 | 1 to 4 percent | low |
| A5 | Value embeddings on a few blocks (grouped parameter, gated) and a value residual to the first block's values | records 14, 63, 65, 70; ResFormer | about 1 percent | low |
| A6 | Mimetic V and O init in place of zeros | record 64 (1.2 percent) | under 1 percent | low |
| A7 | Dense exact bigram table over the 33-token alphabet (left and right neighbor channels, since attention is bidirectional), added to the embedding | record 62 (5.3 percent on text); `modded_nanogpt_2026.md` 62, 68 | unknown for proteins, cheap to learn | low; the table must see only training tokens, which the stream guarantees |
| A8 | Smear in both directions after the embedding | record 34 (0.7 percent) | small | low |
| A9 | Per-head sigmoid attention gates from a slice of the residual | record 28 | small | low |
| A10 | Learnable exclusive self-attention (XSA): subtract a learned share of the token's own value | records 82, 86 | small; the masked token's own value is the MASK embedding, so the gain may be larger for MLM | low |
| A11 | Length curriculum: 512-token windows for the first share of the clock, 1,024 then 2,048 at the end; raise the lr floor | records 46, 72; ESM C's two-stage context | throughput early; closes the train (1,024) versus test (2,048) length gap | medium: the clock measures steps, so the gain shows as more steps |
| A12 | Tail averaging: an EMA of the last 40 percent of weights as the selected candidate (selection still on `val_select`) | record 92c | small, free | low |
| A13 | `torch.compile` modes: `max-autotune`, CUDA graphs on the fixed-shape training step | records 92h, 66 | step time only, a few percent | low, compile time counts against the 3-minute cap |
| A14 | Sampling: one protein per family per pass (cluster-balanced stream order) instead of the shuffled stream | `protein_pretraining.md` 4.3 | unknown; likely small in 10 minutes | low |
| A15 | A fixed ALiBi-style local bias on the scores beside RoPE: per-head slopes 2^(-8 i / h) times the residue distance, symmetric, so every head is local from step 0 at twelve scales | ALiBi (Press et al. 2022); the flat length bins above | unknown; the model learns neighbor use in the first minute either way | low; a flex `score_mod` with a captured buffer |
| A16 | Follow-ups to A15: the bias on half the heads (six local scales, six free for content), steeper or wider slope ranges, a learned per-head slope (a captured parameter in the flex `score_mod`), a bias that decays with the square root of the distance | 012; ALiBi's MLM variants bias only part of the heads | small each; A15 gained 0.0038 nats | low |

## Tier B: a few hours each

| # | Change | Evidence | Expected | Risk |
|---|---|---|---|---|
| B1 | NorMuon (neuron-wise normalized update), then AdEMAMix or MARS momentum | record 41; `architecture_optimization.md` 6 | 1 to 5 percent | medium |
| B2 | Differential attention V2: paired heads with a shared KV group, attn1 minus sigmoid(lambda) attn2 | `architecture_optimization.md` 2 | unknown for MLM | medium |
| B3 | Dual Triangle Attention (the workspace's own: 2.466 against 2.474 joint on OMG-Prot50), composed with B2 in one flex call | `architecture_optimization.md` 3 | small at training length, more at 2,048 | medium |
| B4 | RoPE context: base raised for the 2,048 evals, or YaRN with a window warmup | `architecture_optimization.md` 4 | closes the length gap the loss-by-length table will show | medium |
| B5 | MUDD skips or static hyperconnections (n = 2) | records 73, 77, 81, 85; `architecture_optimization.md` 5 | up to 3 percent; throughput risk 10 to 16 percent | medium to high |
| B6 | Paired head attention, mixed-width QK (64 head, QK at 48) | records 58, 90 | step time | medium |
| B7 | FP8 MLP through torchao (if aarch64 wheels exist for torch 2.11) | records 84, 89 | step time at width 768 is uncertain | medium |
| B8 | Residual-scalar ES (two-sided, 72 scalars) beside backprop | `backprop_free.md` 1.7, 5.4 | small | low, cheap forwards |
| B9 | Unit selection: forward-only probes rank MLP units, backprop updates the top k, random-mask control | `backprop_free.md` recipe 3 | unknown, sign uncertain | low |

## Tier C: later, after a pilot

| # | Change | Evidence | Why later |
|---|---|---|---|
| C1 | nGPT normalized blocks, then relaxing the normalization | `architecture_optimization.md` 1 | 60 to 80 percent per-step overhead unfused; no MLM evidence; a different stack, not an add-on |
| C2 | Boost-style warm start, estimated credit on deep MLPs | `backprop_free.md` recipes 1, 2 | the backprop-free phase trails backprop in the one published comparison; depth cost unmeasured |
| C3 | Fused kernels (MLP, ReLU-squared, XXT) | records 59, 69, 87 | days of work for a few percent of step time; only once the recipe is stable |

## Outside the rules, or waiting on Logan's word

- A 3Di auxiliary target during training (`protein_pretraining.md` 3.2): the objective is fixed at MASK-only MLM,
  so this is not a candidate unless Logan opens the objective. Reported as the probe instead.
- Masking rate other than 15 percent, span or multi-rate masking: fixed by the rules.
- Sampled softmax, FP8 LM head, cross-entropy kernels, prefix-token prediction: built for a 50,000-token vocabulary
  or causal prediction; nothing to gain with 20 classes.
- Depth reduction: 36 blocks are fixed.
- Full Dust, the Cut chain, weight-space ES: out of a 10-minute budget on one GPU by the agents' own cost arithmetic.
- The 2.3x interpretability-inspired speedup post: its text was unreachable (HTTP 402); nothing to act on.

## Source files

- `modded_nanogpt_2026.md`: records 57 to 92 of modded-nanogpt, mechanism, code location and transfer verdict each.
- `architecture_optimization.md`: nGPT, differential attention, DTA, context extension, residual designs,
  optimizers, the softmax fix, auxiliary losses, precision and kernels.
- `backprop_free.md`: Dust, the noise tracks, Boost to Cut, Already Clean, and three hybrid recipes with costs.
- `protein_pretraining.md`: ESM C, ESM3, AMPLIFY, ProGen3, DPLM, SaProt, ProstT5, E1, masking studies, structure
  tokens, sampling, packing, evaluation.

Caveats the agents reported: several sources were read from search summaries or abstracts only; the speedrun
percentages come from the README's record table; record 91's date and record 85's timing conflict between the README
and the PRs; the Dust page and appendix disagree on one 10M-token comparison; ANVIL has no paper. Check a number
against its link before quoting it anywhere but here.
