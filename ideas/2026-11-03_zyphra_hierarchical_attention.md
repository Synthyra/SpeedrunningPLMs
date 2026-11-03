# Zyphra Hierarchical Attention (ZHA), not yet a module

Date: 2026-11-03. Source: Lufkin, Dawes, Speicher, Krishnamurthy, Figliolia and Millidge, "Subquadratic Attention with
a Hierarchical Tree-Structured Indexer" ([arXiv 2610.42112](https://arxiv.org/abs/2610.42112), v2, 2026-11-02). Read:
the HTML through a summarizing fetch, with the claims below checked against quoted passages. Nothing here is measured
on this benchmark. Catalog row: B10 in [`README.md`](README.md).

## Outcome

ZHA is worth keeping as an idea and is unlikely to pay as a module at these lengths. It targets contexts far longer
than a protein, and its conversion recipe assumes a pretrained model. The cheap next step is a measurement, not a kernel.

## What it does

- **Tree.** Keys are grouped into contiguous chunks of size C, and each level summarizes chunks of the level below, so
  a node at level l spans C^l keys. Each node stores one summary per learned reference query, weighted by attention.
- **Routing.** Each query head descends from the top, keeping the smallest set of nodes whose estimated attention mass
  reaches p. Standard attention then runs over the selected leaf chunks, the first key and the current partial chunk.
- **Conversion.** The backbone is frozen and only the reference queries train, by KL divergence to dense attention over
  tree nodes, for about 1B tokens at 32K context.

## What the source reports

- On two models near 35B parameters, evaluated at 64K to 256K tokens, the deep router attends to 1 to 6 percent of the
  keys. RULER-13 scores 93.3 percent against 96.0 percent for dense attention.
- Its latency model predicts ZHA faster than dense attention from about 48K tokens. These speeds are modeled, and the
  authors release no optimized kernels.
- Models under 1B parameters often struggled on retrieval tasks; conversion worked much better from about 4B.
- The source reports no training from scratch. Fine-tuning the base model is listed as future work.

## Why it may not transfer

| Gap | This benchmark | The source |
|---|---|---|
| Length | 1,024-token training windows, at most 2,048 tokens at evaluation | modeled break-even near 48K tokens |
| Training | from scratch on a 600 s or 3,600 s clock | frozen pretrained backbone, dense attention as the KL target |
| Model size | about 198M parameters | poor conversion under 1B parameters |
| Direction | masked prediction over a whole window | chunks built in order, over causal visibility; bidirectional use is not evaluated |
| Kernel | one compiled graph under a 180 s warmup cap | per-head selection, which is not a fixed attention mask, and no released kernel |

Our reading, not the source's: routing would add cost to every step at these lengths, so a gain would have to come from
quality rather than step time. [Row 9 of the long-protein note](2026-10-30_long_protein_attention.md), native sparse
attention, carries the same kernel risk.

## The test, if it is wanted

Measure before building. Run the record's weights over held-out proteins of 1,000 to 2,046 residues and record, per
head and block, the share of chunks needed to reach p = 0.9 of dense attention mass, using the source's C = 16. Attention
that is not concentrated rules out routing here. Concentrated attention would justify a module test on the 1-hour
track, where longer training gives a sparse route more time to pay.

ZHA belongs on a separate track if the benchmark ever opens to inputs past 48K tokens, which no protein here reaches.
