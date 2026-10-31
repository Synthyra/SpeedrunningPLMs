# Noise leaderboard after 2026-10-28, and what transfers to a backprop-trained 36x768 protein MLM

Date: 2026-10-30. A literature scout for SpeedrunningPLMs (36 blocks, 768 wide, 2,048 MLP, 15 percent MASK, one GH200,
600 s, about 2,150 steps, compile warmup capped at 180 s; the record is 2.7146 nats on val_select). It read
`ideas/backprop_free.md` and `ideas/2026-10-30_beyond_backprop.md` first, checked the other 2026-10-30 files in
`ideas/` for overlap, then read the noise repository, its pull requests (PRs), and the record repositories. Nothing here
is measured on this benchmark. Each gain is a source's number or our estimate, and the text says which.

Shorthand: MLM is masked language modeling; CE is cross entropy in nats; PR is a pull request on the noise repository
(the number is its PR number); EMA is an exponential moving average of the weights; LAWA is latest-weight averaging;
TPP is tokens per parameter (ours is 0.3 to 0.7); RMS is root mean square; "yardstick" is one unit of log steps, worth
about 0.026 nats (natural log, rough), and one seed resolves about 0.0015 nats. Dates are UTC from the merge or close
timestamps; the noise README gives no record dates.

## Outcome

- Leaderboard: nothing has merged since PR 52 (merged 2026-10-28 22:42 UTC). The tops are unchanged: Track 1 3.338,
  Track 2 3.498 in 22.9 s, Track 3 3.211. The newest entry, PR 53 (3.4994 in 22.29 s, 2026-10-29), was closed unmerged
  by a maintainer, who wrote that a per-tensor merge table fit beforehand to one run's weights "is outside what can be
  counted as a hyperparameter."
- Other new items: a context-length curriculum (PR 10, open since 2026-10-17, B300 hardware, no record claimed); a partial
  raw-key offset pilot (PR 50, a draft closed unmerged on 2026-10-27, not a submission); a two-point "drift leap" move at
  the end of a run (PR 27, closed unmerged, opened 2026-10-20).
- Search found no 2026 EGGROLL follow-up on language-model pretraining (only the ICML 2026 version of EGGROLL), no hybrid
  backprop and perturbation speedrun, and no new "noise speedrun" write-up beyond the repository. Two 2026 perturbation
  papers are about extrapolation, not speed, and their abstracts give no training method or numbers.
- Almost none of the leaderboard transfers as a module. The records replace backprop's credit with estimators, and
  backprop already has the exact quantity. The transferable pieces are small: a key-input shift, a move at the end of a
  run, a length schedule, and an end average.
- Expected gains, all estimates: partial key offset 0 to -0.005 nats; drift leap 0 to -0.002; coarse end average 0 to
  -0.003; context curriculum 0 to -0.01 at matched wall time (its paired-seed gains are larger at matched updates, but
  its per-update cost rose 28 percent and the PR does not explain it). Each is at or under the record's one-seed band
  unless a run shows otherwise.
- In progress, one line each (not re-scored here): the Boost readout with a progress switch; Cut routes with a route
  count; the identity-path branch ramp (`strategies/residual.py`, `branch_ramp`).

## Ranked table

Gain is the expected change in val_select CE on this benchmark (estimate; negative is better). "Read" says how much of
the source was read. Step cost is our estimate from the catalog profile (276 ms a step) unless the source gives a
number. The 180 s warmup cap applies to every compile.

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats) | Compile | Step cost | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | Partial raw-key offset: half of each head's raw key from the previous token, before key RMS normalization | noise PR 50 (2026); modded-nanogpt PR 169 (2025) | PR 50 body; PR 169 by fetched summary | 3 | 0 to -0.005 | none (slice and concat in the existing graph) | +1 percent (estimate) | 4-block pilot: official CE 5.0515 to 4.8513 with partial RoPE; record 49: 1.1 percent time |
| 2 | Drift leap: stop at 95 percent of the schedule, then move 20 percent back toward the weights from about 25 percent of the run earlier | noise PR 27 (2026) | PR body and results table (leap-run validation only) | 1 | 0 to -0.002 | none (evaluation side) | 0 | time saved 5.6 to 8.5 percent on four run pairs, 0.0013 to 0.0023 nats at equal loss |
| 3 | Coarse end average: two decays, one validation-chosen blend | noise PR 52 (2026); PR 53 (closed) | PR 52 and 53 to their merge sections | 1 | 0 to -0.003 | none | 0 | per-tensor refinement adds 0.0024 to 0.0029 (rejected as overfit); the coarse merge's own gain is not stated |
| 4 | Equal-token context segmentation, short to long | noise PR 10 (2026, open); Li, Zhang and He (2021) | PR body to the reproduction section; Li by search summary | 2 to 3 | 0 to -0.01 at matched wall time | one dynamic-shape graph, as `curriculum.py` already has | +28 percent per update in the PR (unexplained) | seeds 1 and 2: -0.056 and -0.003 nats; seed 0 -0.045, confounded |
| 5 | Noise pretraining before the data (feedback-alignment setting) | Cheon, Lee and Paik (2024), arXiv 2405.16731 | NeurIPS page summary only | 3 | 0 (no evidence) | none if the same graph | the noise phase's steps | qualitative only; no transformer or language-model numbers |

## 1. The noise leaderboard, all three tracks, with merge dates

The README (read today) matches the prior read: the same records, the same scripts at the same commits, and the same
backprop references (Adam with cooldown 3.419; ANVIL II on block matrices with Adam elsewhere 3.340). The PR list adds
the merge dates. A "10M-era" entry was built for the 10M training set, which the README archived on 2026-10-22.

Track 1, validation loss on 30M tokens:

| # | Val loss | Method (README wording, shortened) | Time | Merged (PR, UTC) |
|---|---|---|---|---|
| 1 | 3.767 | Dust with Adam, 24k population, learning-rate cooldown | 5 h 17 min | baseline; script at b6405a1 (2026-10-22) |
| 2 | 3.409 | 10M record #14: Boost to Cut with own queries, exact values, future-token probes, end merge | 1 h 43 min | built from 10M-era PRs 11, 19, 23, 31 and 40 (by their titles), merged 2026-10-18 to 2026-10-22 |
| 3 | 3.338 | Forward readout of cross-token credit, fixed Boost, lower ANVIL II weight decay | 56 min | PR 46, opened 2026-10-25, merged 2026-10-26 15:58 |

Track 2, time to validation loss 3.5 on 30M tokens:

| # | Val loss | Method (shortened) | Time | Merged (PR, UTC) |
|---|---|---|---|---|
| 1 | 3.409 | Same as Track 1 #2 | 1 h 43 min | as above |
| 2 | 3.499 | Identity path first, then Cut-lite at 8 rows per update | 8 min 20 s | PR 44, opened 2026-10-23, merged 2026-10-23 23:56 |
| 3 | 3.494 | Shorter identity phase, Cut with paired MLP roots, difficulty-balanced row order | 2 min 3 s | PR 51, opened 2026-10-27, merged 2026-10-28 22:42 |
| 4 | 3.498 | Track 2 #3's identity phase, Cut with two single roots (blocks 4 and 6), fused Gluon chain kernels, overlapped bf16 collectives | 22.9 s | PR 52, opened 2026-10-28, merged 2026-10-28 22:42 |

Track 3, open architecture on 30M tokens:

| # | Val loss | Method (shortened) | Time | Merged (PR, UTC) |
|---|---|---|---|---|
| 1 | 3.767 | Same as Track 1 #1 | 5 h 17 min | as above |
| 2 | 3.409 | Same as Track 1 #2 | 1 h 43 min | as above |
| 3 | 3.378 | Quarter-turn previous-token tap in the embedding; exact future-token probes | 3 h 50 min | PR 45, opened 2026-10-24, merged 2026-10-26 15:58 |
| 4 | 3.358 | 6 blocks with hashed n-gram tables; future and input-key probes by moments | 27 min | PR 47, opened 2026-10-25, merged 2026-10-26 15:58 |
| 5 | 3.211 | 12 x 1024 GPT with hashed n-gram value embeddings; Cut-lite with input-key probes | 2 h 57 min | PR 48, opened 2026-10-25, merged 2026-10-26 15:58 |

Changes since the 2026-10-28 read:

- No record merged after PR 52. PRs 44 to 48 were merged on 2026-10-23 and 2026-10-26; PRs 51 and 52 were merged on
  2026-10-28, the day of the prior read.
- PR 35 (Track 1, 3.736 in 12 h on 8 H200, merged 2026-10-22 18:38, before the 30M switch) is in the merged list but not in
  the README table. It is a 10M-era credit change.
- Closed and open PRs since the prior read: PR 53 closed unmerged (2026-10-29); PR 50 closed unmerged (2026-10-27); PR 10
  still open (opened 2026-10-17); PR 27 closed unmerged (opened 2026-10-20; read in full now).
- The Dust repository (last commit 2026-10-06) and the boost-cut repository (last commit 2026-10-18) have no newer code.

## 2. Partial raw-key offset before key normalization (idea 1)

- Source: noise PR 50 (opened 2026-10-27, closed as a draft the same day; a small-model pilot on one GH200, not a
  submission). It follows modded-nanogpt PR 169 (opened 2025-12-14, merged 2025-12-16 per the fetched page), recorded
  as record 49 in `ideas/modded_nanogpt_2026.md`. Read: PR 50 body, most of it; PR 169 by summary only.
- What it is: half of each head's raw key, after the key projection and before RMS normalization, is taken from the
  previous token; the other half is the token's own. PR 50 says the modded-nanogpt shift runs after normalization and
  that the change departs from it on purpose. It is combined with partial RoPE in the pilot; the PR keeps a partial-RoPE-only
  control, which was negative, and it has no full-table partial-RoPE-only ablation.
- Evidence (PR 50, two seeds, four blocks, width 128, two heads, 512-wide ReLU-squared MLP, 512 updates on 512 distinct
  sequences, three hashed n-gram tables of 1,048,576 by 128): official validation CE 5.051540 for the base and 4.851346
  with partial RoPE plus the key offset, a difference of -0.200. Training seconds 128.16 and 129.28. The fetched PR 169
  page reports a record time of 128.9 s against 131.2 s and a 50-step cut. The catalog row for record 49 says 1.1 percent.
- What changes here: our model is bidirectional, so "previous token" means the previous input position. A masked
  neighbor's input is MASK, which carries no label, so the shift cannot leak a target, as long as it reads inputs only.
  Packed windows need care: a document's first token must keep its own key. PR 50 does not say how it handles boundaries.
  The -0.200 comes from a four-block, 512-update proxy with two changes at once; it is not a transferable number.
- Expected cost: 3 h. A `keys` hook (the registry has `queries` and `values`, not `keys`), a four-block pilot, then one
  36-block run. Compile: none; a slice and a concat inside the existing graph. Step cost: +1 percent (estimate).
- Gain estimate: 0 to -0.005 nats (estimate). The record's 1.1 percent time gain is worth about 0.0003 nats on the
  yardstick, so the transfer rests on the pilot's direction, not on the size.
- Sketch (not run; needs the `keys` hook):

```python
class PartialKeyOffset(Strategy):
    def keys(self, index, k, doc_start):             # k: (b, h, l, d_h), raw keys after projection, before RMS norm
        half = k.shape[-1] // 2                       # PR 50: half of each raw key from the previous position
        prev = torch.roll(k[..., :half], shifts=1, dims=2)
        prev = torch.where(doc_start[:, None, :, None], k[..., :half], prev)   # a document's first token keeps its own key
        return torch.cat([prev, k[..., half:]], dim=-1)
```

- Risk: the transfer depends on how the document mask interacts with the shift. The partial RoPE part of the pilot had
  a negative control, so the gain may sit in the RoPE change rather than the key shift.

## 3. Drift leap: stop early, then one move back at the end (idea 2)

- Source: noise PR 27 (opened 2026-10-20, closed unmerged; the PR author's thread has one maintainer comment asking for
  PyTorch submissions, and the closing reason is not in the text read). Read: PR body and its results table. The table
  gives leap-run validation only (for example 4.1884 for #25), not base-run validation at the same settings. The PR 25
  title's 4.180 is a different run.
- What it is: stop after about 95 percent of the schedule (the learning-rate schedule stays that of the full run), then
  one move after the last update: w <- w - 0.2 (w - w_old), where w_old is the weights about 25 percent of the run
  earlier. A forward move (coefficient above zero) was worse at every point the PR checked. A backward move works once,
  at the end. Mid-run, the gain is gone 50 updates later, because the optimizer's moving averages still point along the
  old path.
- Evidence (PR 27 table): time saved 5.6 and 5.7 percent on #25 (15.50 s to 14.63 s, two node pairs), 8.5 percent on
  #24 (70.99 s to 64.96 s), and 8.1 percent on #7 (82.54 s to 75.84 s). The constants come from one #25 run and are fixed
  in the script, with leaps scored on validation. The PR also reports that on one run the backward move sits 0.05 to 0.07
  below the current weights at each point from update 60 to 450; that is a mid-run statement, not a final loss.
- What changes here: our `ema.py` already keeps a snapshot ring (`latest_k`, not yet run). For a 600 s, 2,150-step run,
  the look-back is about 500 steps. The move is an evaluation transform, applied after a stop at 95 percent, and it
  touches no gradients, so it is outside the rule's reach. Select the coefficient and look-back on val_select, not from
  the noise run's constants.
- Converted gain: a 5 to 9 percent time saving at equal loss is worth 0.0013 to 0.0023 nats on the yardstick (ln(1/0.95)
  and ln(1/0.915), times 0.026). That is one seed band. The PR does not show a loss gain at the end of its runs.
- Expected cost: 1 h. Compile: none. Step cost: none; one clone of the weights per validation pass.
- Gain estimate: 0 to -0.002 nats. A tie is the likely result on our benchmark.
- Sketch (not run; `weights(model)` is the helper the other sketches use):

```python
class DriftLeap(Strategy):
    def build(self, model, max_tokens):
        self.stop, self.c, self.lookback = 0.95, -0.2, 0.25   # PR 27: stop at 95 percent; c < 0 leaps back
        self.old = None                                        # the weights about 25 percent of the clock earlier
    def after_step(self, state: StepState):
        if self.old is None and state.step == round(self.lookback * total_steps()):
            self.old = [w.detach().clone() for w in weights(state.model)]
    @contextlib.contextmanager
    def evaluation_weights(self, model):                       # score w + c (w - w_old) after the stop, then restore
        saved = [w.detach().clone() for w in weights(model)]
        for w, old in zip(weights(model), self.old): w.data.copy_(w + self.c * (w - old))
        yield
        for w, s in zip(weights(model), saved): w.data.copy_(s)
```

- Risk: the stop at 95 percent is a schedule change (`schedule.py`), so the time saved is only worth something if the
  clock is spent on steps elsewhere. Measured at equal wall time, the gain is the 5 to 9 percent figure above.

## 4. Coarse end average: two decays, one blend chosen on validation (idea 3)

- Source: noise PR 52 (merged 2026-10-28 22:42 UTC; its end merge takes 0.14 s) and PR 53 (closed unmerged 2026-10-29)
  for the refinement. Read: PR 52 and PR 53 bodies to their merge sections. The text read does not give the merge's own
  gain over no merge.
- What it is: the end merge keeps two weight averages (decays 0.03 and 0.02, from 30 percent of the rows) and moves each
  tensor 0.75 or 1.0 of the way toward one of them, with the choice made on 64 held-out rows. On seed 0 it picks 0.75
  toward the 0.02 average for all 60 tensors. PR 53 fits one coefficient per tensor (60 in all) on one run and reports
  0.0024 to 0.0029 below PR 52's merge on the full-length runs; the maintainer closed it as an overfit fit.
- What changes here: our `ema.py` `tail` variant (decay 0.98 to 0.998) lagged the record at the end (006: val_select
  2.7578 against 2.7559). The testable part is the two-decay form with one coarse blend, which is two parameters chosen on
  val_select. The per-tensor fit is rejected (see the checked list). The sibling file `regularization_and_averaging.md`
  covers tail averages with blends; drop this item if that file already tests the blend.
- Expected cost: 1 h. Compile: none. Step cost: none.
- Gain estimate: 0 to -0.003 nats (estimate; the coarse merge's own gain is not in the text read).

## 5. Context segmentation at equal tokens per update (idea 4)

- Source: noise PR 10 (opened 2026-10-17, still open; on the 16k-population Dust recipe, not a backprop model; eight
  B300 GPUs). Read: PR body through its paired-seed and reproduction sections. Literature: Li, Zhang and He,
  arXiv 2108.06084 ("The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models",
  first posted in 2021 as a curriculum-learning paper), read as a search summary only. The "about 1.5x" figure
  comes from the MosaicML docs page and was not checked against the paper.
- What it is: each update takes the same eight rows; each row's 2,048 input-target pairs are cut into shorter contiguous
  segments that keep their own final targets, and all segments run in the same update. The fast schedule (update index
  : length) is 1:64, 33:128, 65:256, 97:512, 129:1024, 161:2048, over 610 updates. Our windows top out at 1,024.
- Evidence: seed 0, against a baseline that predates the compile fix: 3.9804 against 4.0256 (-0.045). Paired seeds with
  the repaired code: seed 1, 4.0160 to 3.9598 (-0.056, 1.40 percent); seed 2, 4.0188 to 4.0156 (-0.003, 0.08 percent).
  Wall time per run: the fast schedule took 4,486 s against 3,494 s for the constant baseline, about 28 percent more; the
  PR does not explain it. The PR says the two paired seeds do not establish an effect size.
- What changes here: the catalog's `short_first` (027) was retired at the warmup cap. Our curriculum family already
  runs shapes through one dynamic-shape graph (`strategies/curriculum.py`), so compile is not the new risk. The new risk
  is the 28 percent per-update cost. At matched wall time, a 28 percent slower step gives back about 0.006 nats on the
  yardstick (ln 1.28 times 0.026), so the paired gains shrink.
- Expected cost: 2 to 3 h, starting with a 60 s pilot that measures per-update cost. Compile: the dynamic graph we
  already have. Step cost: +28 percent in the PR; we would need the cause before assuming it.
- Gain estimate: 0 to -0.01 nats at matched wall time (estimate; two repaired seeds, one confounded seed).

## 6. Noise pretraining before the data (idea 5)

- Source: Cheon, Lee and Paik, "Pretraining with Random Noise for Fast and Robust Learning without Weight Transport"
  (NeurIPS 2024; arXiv 2405.16731). Read: the NeurIPS poster page summary and a search listing; not the full text.
- What it is: a network is pretrained on random noise before task data, which the authors model on spontaneous activity
  before birth. They argue it aligns forward weights with feedback weights, which feedback alignment needs. The page gives
  qualitative results only: faster learning, higher accuracy ("even comparable to the backpropagation algorithm"), lower
  effective dimension, and better out-of-distribution accuracy. It names no dataset and no figure, and it does not
  mention transformers or language models.
- What changes here: with backprop, the feedback-alignment mechanism is not needed. Only the warm-start idea remains,
  and it costs clock steps in a 600 s run. The cheapest test is a short noise phase on the embeddings and blocks, then
  normal training, compared at equal wall time.
- Expected cost: 3 h. Compile: none if the noise phase shares the training graph; a different loss needs its own graph.
  Step cost: the steps the noise phase uses.
- Gain estimate: 0 (no transferable evidence). The risk is positive because the clock is fixed.

## Checked and rejected

- Per-tensor end-merge coefficients (noise PR 53): closed by a maintainer as a fit to one run. Sixty coefficients chosen on
  one run carry the same overfit risk here.
- First hop with a live remainder (noise PR 35, 10M era): a credit estimator for the Cut chain. Backprop already computes
  these terms exactly, so there is nothing to add.
- Relay and backward induction over depth (noise PRs 17, 20, 21, 22; closed unmerged; the reasons are not in the text
  read): chained credit estimators. They are not modules for a backprop model.
- Dust with Muon (PR 2, closed unmerged, 4.147 on the 10M track), orthogonal Dust (PR 5, closed), and associative credit
  memory (PR 6, merged 2026-10-14, 10M era): estimator-side work. Momentum already exists in our optimizer.
- Probes (roots, future-token, input-key, moment-form, gate and query draws): gradient estimators. Backprop computes the
  exact gradient. They stay in the prior file's cosine checks (rank 4), and nothing else.
- Noise scales as a regularizer: no source in the set tests activation noise as a regularizer for a backprop model. The
  Dust scales are estimator scales. At 0.3 to 0.7 TPP we are far from the regime where regularization pays (see the
  beyond-backprop file's note on TPP). Expected gain 0.
- Group learning-rate multipliers (300x for embeddings, 30x for lambdas): `value_embeddings.py` already has an
  `embed_lr` group. The multipliers were set for Dust-style estimates and were not compared here.
- Learning-rate decay shape (Dust: linear over the last 30 percent; Boost: constant for 70 percent, then linear): our
  40 percent cooldown is the record's and the campaign already tuned it. No new evidence.
- Residual lambdas, x0 lambdas, gates, value embeddings, value residual and the identity-path ramp: present in
  `residual.py`, `embedding_mix.py`, `value_embeddings.py` and `value_residual.py`. The noise versions add nothing new.
- Hashed n-gram value embeddings (Track 3 #4 and #5): covered by `2026-10-30_capacity_at_fixed_shape.md`, idea 2. Our
  bigram pair table (010) was level at 270 s and behind after (2.7583 against 2.7559). At our 33-token vocabulary, exact
  tables are small: 1,089 bigram rows and 35,937 trigram rows. Hashing to 2^20 rows buys nothing at this vocabulary.
- Gluon chain kernels and overlapped bf16 collectives: the first is specific to the backprop-free chain; the second needs
  eight GPUs. Neither applies on one GPU.
- arXiv 2605.04344 ("Perturbation is All You Need for Extrapolating Language Models", submitted 2026-05-05) and
  arXiv 2605.13284 ("Learning Perturbations to Extrapolate Your LLM", submitted 2026-05-13): both are about predicting
  outside the training support. The abstracts state no training method, no model, no dataset names and no numbers. Not a
  speed result; abstract only.
- Secondary coverage of the noise records (a blog reporting 1.6 s and a 4,000x ratio; a 2026-10-22 explainx post saying
  the fast entries are not Dust as published): the numbers agree with the repository where checked, but neither
  is a primary source. The repository was used for every number above.
- The 2.3x interpretability post: still unreachable; unchanged from the prior file.

## Hooks the registry does not have

- `keys(index, k, doc_start)`: a transform of the raw key before RMS normalization, with the document-start mask (idea 1).
- `evaluation_weights` already exists and carries the drift leap (idea 2); a stop at 95 percent needs a schedule change in
  `schedule.py`.
- Nothing else new. The end average (idea 3) and the curriculum (idea 4) use existing EMA and data-path code.

## Gaps and read status

- Read to the results or the merge section: PR 27 (body and table), PR 50 (body), PR 52 (body to the merge section),
  PR 53 (body to the table, and the maintainer comment), PR 10 (body through the reproduction section).
- Part only: PR 35 (body to the rules section); the record scripts were not reread this time, so the prior file's reads
  of them stand.
- Summary or search snippet only: modded-nanogpt PR 169; Cheon, Lee and Paik (NeurIPS 2024); Li, Zhang and He (arXiv
  2108.06084); the two 2026 perturbation abstracts; the two secondary blogs. Their numbers are not used for any gain.
- No numbers on our benchmark for any idea. Every conversion uses the yardstick, and the slope is rough.

## Sources

- noise repository and README: https://github.com/qlabs-eng/noise
- noise pull requests (list): https://github.com/qlabs-eng/noise/pulls
- noise PRs read: https://github.com/qlabs-eng/noise/pull/10 ; https://github.com/qlabs-eng/noise/pull/27 ;
  https://github.com/qlabs-eng/noise/pull/35 ; https://github.com/qlabs-eng/noise/pull/50 ;
  https://github.com/qlabs-eng/noise/pull/52 ; https://github.com/qlabs-eng/noise/pull/53
- noise merged records by PR: https://github.com/qlabs-eng/noise/pull/44 ; https://github.com/qlabs-eng/noise/pull/45 ;
  https://github.com/qlabs-eng/noise/pull/46 ; https://github.com/qlabs-eng/noise/pull/47 ;
  https://github.com/qlabs-eng/noise/pull/48 ; https://github.com/qlabs-eng/noise/pull/51
- Benchmark landing page: https://qlabs.sh/noise
- Dust repository: https://github.com/qlabs-eng/dust
- boost-cut repository: https://github.com/spicylemonade/boost-cut
- modded-nanogpt PR 169 (partial key offset): https://github.com/KellerJordan/modded-nanogpt/pull/169
- Cheon, Lee and Paik (2024): https://arxiv.org/abs/2405.16731 ; NeurIPS poster https://neurips.cc/virtual/2024/poster/96093
- Li, Zhang and He (2021): https://arxiv.org/abs/2108.06084
- EGGROLL: https://arxiv.org/abs/2511.16652
- 2026 perturbation papers: https://arxiv.org/abs/2605.04344 ; https://arxiv.org/abs/2605.13284
- Secondary, not used for numbers: https://fourweekmba.com/ai-backprop-free-speedrun-cuts-a-tiny-models-training-to-1-6s/ ;
  https://explainx.ai/blog/q-labs-dust-noise-pretrain-transformers-without-backpropagation-4000x-claim-verified-2026
