# Count hints from the training stream, not yet a module

Date: 2026-11-03. Source: Herman Brunborg, "NanoGPT Speedrun in 6.7s" ([post](https://brunborg.com/n-gram-hinted-lm)), read in full. Measurement: the
[result](../../../../results/representation/speedrunning_plms/2026-11-03_kmer_hint_measure/README.md), no model trained. It waits on Logan's ruling below. Catalog row: `README.md`, "Outside the rules,
or waiting on Logan's word". This corrects `2026-10-30_speedrun_optimizers.md`, which set the idea aside as external data: the post's small tables are built from its own training tokens, and so
are these.

## Outcome

A table of counts over the 392.6M-token training stream reaches 2.7895 nats on `val_select` at the split's real masks, with no training, against the unigram's 2.8915, the 180 s rung's best of about
2.77 and the 10-minute record's 2.7097. A 10-minute run reads about a third of the stream, so these counts hold data the run never sees. What this leaves open is the part that matters: how much a table adds to a
model that has learned some of the same things. A blend on the record's weights answers that in minutes on the GH200, before anything is built.

## What the source did

N-gram hints took the NanoGPT speedrun from 39.9 s to 6.7 s. Each position's model input gains hints from 6-, 10- and 20-gram tables, with a learned trust for 126 kinds of hint (by count and purity), injected at three
points: the embedding, a middle block and the head. On its benchmark: no hints 3.7454, hints from 75M tokens 3.7219, hints from 37B tokens 3.2739. Its small corpus is the training stream, which is the form this benchmark's rules allow; its large one is web text, which they do not.

## What transfers, measured

- **Protein context is weaker and narrower than web text.** A context of 5 residues on each side matches 1.43 percent of positions at 74 percent accuracy. Eight-residue seeds together match 19 percent, at 2.409 nats against 2.895 there.
- **Seeds must survive masking.** At 15 percent MASK a symmetric context of 8 residues is fully visible at 28 percent of positions, which holds the symmetric chain to 2.837; one-sided and gapped seeds reach 2.7895.
- **Eight offsets are enough.** Seven such seeds carry 0.0664 of the 0.0697 nats the 13 seeds add over the low-order chain.
- **It is not low complexity.** Only 0.6 percent of matches sit in a window under 2 bits.
- **Training positions flatter it.** With their own occurrence removed, training positions at order 6 match 2.6 percent at 95 percent accuracy, against 0.67 percent at 87 percent on held-out families, so a model that learns its trust on the plain table over-trusts it.

## The design a pilot would test

1. **First, the blend, about 5 minutes.** Score the record's weights on `val_select`, combine its log probabilities with the chain's in a product of experts fit on half the proteins, and score the other half. Under about 0.002 nats, stop here.
2. **A family cross-fit.** Split the train families in two (`train_families.json`). A training position reads the other side's table, so its trust is learned on what a held-out family would see. Evaluation reads both, the same shape as Brunborg's small corpus in training and large at validation.
3. **Trust from zero.** For each of the seven seeds, a learned scalar per count and purity bucket multiplies the seed's log odds at the head, so step 0 is the record's. Then the same table's shares as a zero-initialized input at the embedding and a middle block, as the source does, if the head alone pays.
4. **Tables.** Sorted (key, residue) pairs, about 3.1 GB a seed, so about 22 GB of the H100's 96 GB for seven, or fewer through a count floor. Lookups only at masked positions: about 10,000 a step, by binary search.

## Rule question for Logan

The written rules leave tables free if built only from training tokens, and these are. Three things make it his call:

- it changes what the benchmark measures, from what a model learns in the time to what a model plus a lookup knows;
- it adds a non-parametric lookup of several GB beside a 200M model, unlike any record so far; and
- it does not transfer to ESMC, the program's point: a base model carries no such table to its downstream tools.

If allowed, the build time needs a rule too. A sort of 389M keys is about 20 s a seed on a CPU, and the reading order's precompute today sits outside every clock, so the fair reading charges it to the 180 s warmup cap.
