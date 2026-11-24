# Round 9: new knobs for the genome

Date: 2026-11-03. A literature pass for SpeedrunningPLMs after the tabular surrogate replay (Logan, 2026-11-03: the more modular knobs the genomes
have, the better the system should leverage them together and the better they should predict which genome wins). Nothing here is
measured on this benchmark: every gain below is a source's number, on language unless it says otherwise, mostly causal, and mostly at tokens per
parameter far above this benchmark's under 1. One seed resolves about 0.0015 nats. Catalog rows: [`README.md`](README.md).

## Outcome

Round 9 grows the space from 99 to 136 dimensions and from 136 to 208 categorical choices, built as modules with tests and off by default, so 022's arithmetic is unchanged. Each compile-affecting knob still needs its 180 s pilot on
the GH200 before the proposers trust its `search.COMPILE_SECONDS` estimate. Count hints, the largest open lead, have their own file
([`2026-11-03_count_hints.md`](2026-11-03_count_hints.md)) and wait on a ruling.

**Knobs alone will not add predictive power.** Across the 103 adaptive runs, 32 of the 99 dimensions then in the space never moved, 31 of 136 categorical choices were never in an
adaptive run, and numbers hold at most 9 distinct values, because proposals descend from a few parents. A knob that never varies predicts nothing for any
surrogate. So new knobs need a directed screen before a surrogate can use them (the plan is below), and an isotropic GP gets a longer one-hot vector for
each knob, which is a further reason to test a tabular model on them.

**Range edges were hiding the far side.** Top runs pile up at `embed_lr`'s old ceiling of 1e-2 and near `muon_lr`'s 0.08, and the record sits at `head_lr`'s 3e-3.
`SPACE` now reaches the configs' bounds (3e-2, 0.12, 1e-2). `heads` gains 4 and 8: 6 is 95 percent of the top fifth, against 12, with neither side of it tried.

## Built: the update rule

No compile cost: all of these run in the eager optimizer.

| Knob | Mechanism | Evidence (lower is better unless noted) | Cost | Read |
|---|---|---|---|---|
| `optimizer.variant=muown` | row magnitudes by Adam, row directions by Muon | arXiv 2605.10797: 1B 11.90 against 12.20, 2.7B 9.57 against 9.82 | 0.8 GB, a few ms | Algorithm 1 |
| `optimizer.radial_out`, `radial_in` | scale an update's radial part per matrix, keep the norm that leaves; 0 and 0 is Hyperball, an outward 0 is AdamP | modded-nanogpt PR 294: 2,990 steps at 0.5 and 1 over 11 seeds, against a 3,225 public record at the wave's start; Hyperball, arXiv 2606.16899, 20 to 30 percent token-equivalent at 1.2B but a tie on track 3 (MuonH 3,325, tuned MuonW 3,325) | a norm per matrix | PR text; paper section; the code not shown |
| `optimizer.contra_start`, `contra_end` | (1 + c) polar(M) - c M / s1: c above 0 overweights small singular directions, below 0 underweights them | modded-nanogpt PRs 275, 291: a contra-to-soft schedule at 3,030 steps; coefficients unpublished | two matvecs a matrix | post, no code |
| `optimizer.variance_scale` | divide the direction elementwise by its running spread before orthogonalizing | Muon-VS, arXiv 2601.14603: 720M 2.7858 to 2.7798, 130M 3.2179 to 3.1982, batch-size sensitive | 0.8 GB | Eq. 13, Algorithm 1 |
| `optimizer.orthogonalizer=mud1`, `mud2` | Gauss-Seidel sweeps in place of the polar factor | MUD, arXiv 2603.17970, on ESM-2 150M masked language modeling: Muon 16.840 in 997 s, MUD1 16.855 in 636 s, MUD2 16.795 in 710 s; the only protein number here | saves up to about 7 percent of a step (banked Muon is 26 ms of 276) | Sections 2 and 3 |
| `optimizer.polar_steps` | the first 3 or 4 of Polar Express's 5 steps | a cheap stand-in for HTMuon (arXiv 2603.10067) and Soft-Muon, which underweight small singular values; no source tests these prefixes | saves about 2 ms a step dropped | none: ours |
| `optimizer.qk_lr_scale`, `vo_lr_scale`, `mlp_lr_scale` | Muon's rate per role | Prime Intellect's track 3 stack (2,930 steps) reports per-role rates, no number per change | none | blog; no recipe found |
| `optimizer.mona_beta` | half the gradient from a slow EMA of earlier gradients (one buffer, by the closed form of the paper's recursion) | MONA, arXiv 2605.26842: 68B benchmark mean 0.4478 to 0.4557, higher is better, at 700B tokens; ANVIL's slow rail lost its pilot here by 0.014, so a low prior | 0.8 GB | Algorithm 1 |
| `optimizer.normuon_beta`, `nesterov` | two constants of the record's update, opened | none | none | ours |
| `lookahead.variant=periodic` | every k steps, slow weights take a share of the live weights and the live weights return to them | Prime Intellect's stack, no number; Lookahead, arXiv 1907.08610, on vision | 0.8 GB, a lerp every k steps | blog |
| `schedule.cooldown_power` | the cooldown's multiplier raised to a power | Prime Intellect's stack (1.2), shape not stated | none | blog |

## Built: the block

Compile costs are estimates by kind until a pilot measures them.

| Knob | Mechanism | Evidence | Compile estimate | Read |
|---|---|---|---|---|
| `residual.variant=momentum` | a velocity stream per sublayer: X + mu V read, V <- beta V + gamma Y, X <- X + nu V | TMMFormer, arXiv 2605.24425, 12 blocks of 768 under Muon, 30k steps: OpenWebText 3.0078 to 2.9342, TinyStories 1.1569 to 1.1284; a parameter-matched control closes about 40 percent of the TinyStories gap; under AdamW the OpenWebText gap is 0.041 | 25 s | equations; starts for mu and beta not found |
| `value_residual.anchor_norm`, `variant=normalized`, `per_head`, `qk_anchor` | the record's largest gain, opened: a normalized anchor, IMU-1's normalized mix, head grain, the mix on Q and K | ExoFormer and NuResFormer, arXiv 2601.08131, 450M: base 14.79, ResFormer 14.32, NuResFormer 14.15, ExoFormer 14.13. IMU-1, arXiv 2602.02522, at 70M: -0.013 training loss for its mix, on a table whose baseline is inconsistent | 2, 15, 5, 10 s | HTML; IMU-1 ablation table |
| `attention.qk_share` | Q and K narrower than V | modded-nanogpt record 90: 128 to 96, inside a stack with fused kernels; a share of the attention's and the QK matmuls' cost goes | 10 s | README, PR notes |
| `attention.tie_kv` | the raw keys are the values | GQA-S2, arXiv 2602.01845; one of a block's six matrices goes, about 5 percent of a step by the record's profile | none | HTML summary |
| `attention.exclusive` | Y - tanh(a) <Y, v> v per head, a from zero | djdumpling's modded-nanogpt notes; 007's ungated form lost here | 10 s | blog |
| `attention.block_size` | flex's mask granularity | `2026-10-30_step_throughput.md` 4: 4 to 12 ms of flex's 80, estimated | 5 s | docs |
| `canon.variant=neighbors` | a learned per-channel mix of the neighbors inside a protein, from zero | Canon layers, arXiv 2512.17351: lifts NoPE to RoPE's level at 1.3B and helps reasoning depth 2 to 4 times on synthetic tasks; no RoPE gain in perplexity found | 25 s, 15 more at reach 2 | sections |
| `protein_context.variant` | composition at the embedding, and a protein-mean path every k blocks, from zero | the count measurement: visible composition alone is 2.8526 against the unigram's 2.8915; ProteinBERT's global track for the mean | 5, 15, 20 s | ours; no transfer number |
| `head.bias` | the logits start at the training stream's log shares | the usual final-layer trick; no number | 1 s | ours |
| `loss.substitution_share`, `substitution_until` | a fading share of the target on BLOSUM62's substitutes | no source on pretraining; the reasoning is variance in the masked label | none | ours |

## Not built, and why

| Idea | Evidence | Why not now |
|---|---|---|
| Multi-Gate Residuals, Comp n = 8 (arXiv 2605.23259) | 12, 24 and 36 blocks under Muon at 10B tokens: -0.042, -0.041, -0.036 against pre-norm, about -0.005 against AttnRes, no seeds | the best block-level number at 36 blocks. Eight (b, l, d) streams hold about 29 GB for the backward at 65,536 tokens, and AttnRes's own compile broke the cap here; a stacked-tensor form at n of 2 to 4 is the build to try, after regional compile |
| Regional compile (`2026-10-30_step_throughput.md` 8) | synthetic: 9.66 s to 0.35 s | the enabler for more modules a genome, but `block.compile` also compiles the evaluator's forward, where the length buckets would recompile; it needs a separate compiled wrapper for training, and a pilot |
| GQA with fewer K and V heads (arXiv 2602.01845) | Proust's 16:2 | every hook that shapes K or V by head count needs review; `tie_kv` takes the cheap half first |
| HTMuon's exact map (arXiv 2603.10067) | singular values to a power near 0.1 | an SVD or eigendecomposition per bank; `polar_steps` and a negative `contra_end` are the cheap neighbors |
| SOAP on the MLP and V (Prime Intellect) | track 3 stack | eigendecompositions on a schedule, and no recipe found |
| TaperNorm (arXiv 2602.10408) | up to 1.14 times decode throughput | norms are 17 ms of a step, and its one number is decoding |
| Depth-staggered cooldown (FreezeOut) | a compute saving elsewhere | a frozen block still costs its compute here, so the source's gain is absent |
| Distance-Aware Muon (arXiv 2605.18999) | 3.7166 to 3.6714 on one language task, worse on ViT | low prior against the rest |

djdumpling's failures are a prior on what is already here: AdaMuon, schedule-free training, LAWA, SSMax and xIELU each lost or slowed there, and a learned attention temperature
returned to 1. Late gradient accumulation and a token ramp need Logan's word (`2026-10-30_short_run_schedules.md` 8).

## The plan once a GH200 is the speedrun's

The new knobs and the never-run ones need variation before any surrogate can read them. All three batches run on the 10-minute record, `genomes/auto01_bayes_02.json`. The first six pilots ran before the
[head-to-head](../../../../experiments/representation/speedrunning_plms/2026-11-03_surrogate_head_to_head/README.md); Logan's deadline for it stopped the pilot at 22:01 UTC on 2026-11-03, so the other seven pilots, both screens and the two batch schedules wait behind it. Their knobs got two steps each on the GPU, so the arms can pick them without a crash for a simple reason. The readings, alone on the record at 180 s against its 2.7630 and 646 steps, are in [the catalog](README.md): every one is behind, the momentum stream by 0.0024 with an invalid warmup, `composition` by 0.0033 with 644 steps, the rest by 0.008 to 0.010.

The head-to-head's picks gave round 9's first full-budget readings, each inside a genome of several changes on seed 0, so these say what ran and not what each knob is worth ([result](../../../../results/representation/speedrunning_plms/2026-11-03_surrogate_head_to_head/README.md)). In the best genomes so far, 2.7067 and 2.7069 at 600 s against the record's 2.7083 on that machine: `loss.substitution_share` 0.05, `head.bias` 1 and `optimizer.plus_axis` `row_col`. In the worst: `optimizer.depth_taper` 0.29 with `optimizer.radial_out` 0 (2.7524), `optimizer.orthogonalizer` `mud2` (2.7283), and `batch.micro_divisor` 4 (2.7189, 6,495 steps). `batch.accumulate_late` 2 reached 2.7075 at 1,547 steps, where 4 reached 2.7134 at 1,257. `attention.block_size` 64 cannot compile.

1. **Pilot, alone at 180 s, about 1.3 hours.** One run each for `exclusive`, `canon`, `momentum_stream`, `composition`, `protein_mean`, `qk_anchor`, `value_normalized`, `value_per_head`, `qk_75`, `heads_4`, `muown`,
   `hyperball`, `mud2`. Each reads a warmup into `COMPILE_SECONDS`, a step time, and the first loss.
2. **Screen A, the update rule: a 12-run Plackett-Burman design, at 180 s with the top half at 600 s.** `mud1`, `radial_brake`, `contra_to_soft`, `variance_scale`, `mona`, `lookahead`, `mlp_lr_half`, `cooldown_power`,
   `substitution_targets`, `head_bias`, `tie_kv`.
3. **Screen B, throughput and the never-run: the same design.** `heads_8`, `polar_3`, `qk_75`, `kv_share_2`, `anchor_norm`, `value_per_head`, `adamuon`, `depth_taper`, `muon_late_2`, `sam_late`, `early_dropout`.

Between the pilot and the screens, `batch_ramp` (32 windows, 64 from a quarter of the clock) and `batch_double_late` (64 windows, 128 from 70 percent) run alone on the record at 600 s: `batch.py` holds the mechanism and this benchmark's evidence, 008's 32 windows ahead at 90 s and behind from 180 s.

`polar_3` is inert under `mud1`, and `tie_kv` under `qk_75`, so each pair is split across the screens. One effect's standard error is about 0.0009 nats at 12 runs, so effects of 0.002 and up show. Together about 6.5 hours, 15 USD at 2.29 USD an hour.

## Sources

Read this round, at the depth each table says: [ExoFormer](https://arxiv.org/html/2601.08131v3), [IMU-1](https://arxiv.org/html/2602.02522), [Momentum Streams](https://arxiv.org/html/2605.24425),
[MGR](https://arxiv.org/html/2605.23259v1), [Muown](https://arxiv.org/html/2605.10797), [MUD](https://arxiv.org/abs/2603.17970), [Muon-VS](https://arxiv.org/html/2601.14603v1), [HTMuon](https://arxiv.org/pdf/2603.10067),
[Turbo-Muon](https://arxiv.org/pdf/2512.04632), [Hyperball](https://arxiv.org/html/2606.16899v2), [MONA](https://arxiv.org/html/2605.26842), [Canon](https://arxiv.org/html/2512.17351), [GQA-S2](https://arxiv.org/html/2602.01845v1),
[Distance-Aware Muon](https://arxiv.org/html/2605.18999), [TaperNorm](https://arxiv.org/abs/2602.10408), [Contra and Soft](https://nilin.github.io/contra-muon-and-soft-muon/), [djdumpling's notes](https://djdumpling.github.io/2026/05/27/modded-nanoGPT-WR.html),
[Prime Intellect](https://www.primeintellect.ai/auto-nanogpt) and [its experiments](https://github.com/PrimeIntellect-ai/experiments-autonomous-speedrunning), modded-nanogpt PRs [267](https://github.com/KellerJordan/modded-nanogpt/pull/267),
[273](https://github.com/KellerJordan/modded-nanogpt/pull/273), [275](https://github.com/KellerJordan/modded-nanogpt/pull/275), [291](https://github.com/KellerJordan/modded-nanogpt/pull/291), [294](https://github.com/KellerJordan/modded-nanogpt/pull/294).
