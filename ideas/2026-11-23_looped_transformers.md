# Looped transformers and module routing, for the speedrun (2026-11-23)

What the `loop`, `routing` and `sglr` families take from the 2025 and 2026 literature, and the numbers they use. LOOM
and recAMPLIFY were read in their arXiv HTML on 2026-11-23; the rest from abstracts or secondary summaries, marked so.

## Read

- **LOOM, "Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts"** (He et al., arXiv 2610.01153).
  Re-injection at the start of loop t >= 2: h = (1 - g_t) h + g_t x, g_t = lambda / (t sqrt(M)), lambda 0.5, M the
  physical layers, x the input embedding (Eq. 2). Residual scale gamma = lambda / (H sqrt(M)) on both sublayer outputs,
  with an RMS norm on the MoE output (Eq. 1). Per-loop sigmoid routers in float32, z-loss 1e-3, capacity 1.5, a
  balancing bias at rate 2e-3. Segmented backpropagation, segments of 3 loops. Ablation at 350M and 9 loops, validation
  perplexity: full 18.62; without the residual scale 24.63; without re-injection 26.05; without the MoE norm 26.50;
  without the Looping Residual 19.25; with a router shared across loops 19.83. Near iso-FLOP at 700M, 5 loops 16.54
  against 18.36 for the plain model. Full backpropagation through 9 loops diverged (81.04) where segments of 3 reached
  14.80.
- **recAMPLIFY, "Large-Scale Pretraining of Recursive Protein Language Models"** (Jha et al., arXiv 2610.26747). A
  weight-tied masked PLM, prelude, core and coda of (2, 8, 2) over 6 cycles (52 applications) in the ablations, the
  input re-injected every cycle, a frozen shared initial state; 155M unique parameters on AMPLIFY's 2T tokens beat
  AMPLIFY-350M on CASP contacts and three of four supervised tasks. Variable depth samples the cycles uniformly from 1
  to 6 per minibatch and degrades less past the trained depth than a fixed count. Gated Mixture Recurrence decodes from
  a halting-weighted mixture of the visited states with a residual gate and cuts the expected depth by 32 to 40
  percent at small task costs; its halt head reads stop-gradient states.

## From abstracts or summaries

- **Lee et al., arXiv 2605.09165**: looped MoE models scale and dense looped models do not, because routers pick
  different experts on each pass.
- **Popescu et al., arXiv 2607.20519**: jointly trained halting gates do not reliably learn adaptive depth, one exit
  distribution both picking the stop and weighting the per-depth losses; fixed-prior depth supervision and post-hoc
  confidence readouts match or beat them.
- **Huginn** (Geiping et al., arXiv 2502.05171): prelude, recurrent core and coda, random depth in training, gradient
  through the last passes only.
- **Relaxed Recursive Transformers** (arXiv 2410.20672): tied layers with small per-position parameters left untied.
- **Production, autumn 2026** (vendor pages and press): Dory1.0-4B (8-20-8 blocks, 1 to 4 passes sampled per step,
  looping added late in pretraining), Inkling 2 Mini (its middle 12 of 24 layers three times, a gated cross-pass
  message that starts closed), GLM-5.4 (28 layers twice, by continued training; Z.ai says from-scratch loops trail
  plain models at equal pretraining compute). All loop 2 to 4 times; none halts per token.
- **Ekbote et al., arXiv 2610.40325**: gains from looping saturate, reverse or become unstable with more loops.

## Transfer to the speedrun

The benchmark fixes 36 blocks and the clock, so a loop here keeps 36 applications and shares parameters: the regime
the literature finds weakest (dense, equal compute), except that the 10-minute model sees 135M tokens for 198M
parameters, far from compute-optimal, and a tied model's Muon step orthogonalizes fewer matrices. LOOM's recipe is
used as published where the speedrun allows: the re-injection exactly (from the prelude's output, since this model has
a prelude), the residual scale as a number the search can move (its MoE norm cannot act on a zero-initialized output
projection), per-pass routers within the MLP's 2,048 units (the shape rule forbids more experts), Huginn's truncation
in place of LOOM's segments. Sampled depth is per window, since one compiled graph cannot change its pass count within
the warmup cap. SGLR (Logan's Self-Gated Latent Reasoners, MNIST) supplies the heterogeneous pool and the transition
router; no 2026 paper was found that routes tokens through heterogeneous modules in a looped language model.
