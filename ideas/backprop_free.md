# Training without backpropagation: Dust, noise, Boost and Cut, and what they offer a 36x768 protein MLM

Dust is a 2026 method from Q Labs that trains a small GPT without autograd. It jitters activations independently at every token, scores each token's loss response, and turns those scores into an output-error estimate for every weight matrix ([paper page](https://qlabs.sh/research/dust), [appendix](https://qlabs.sh/research/dust/appendix.html), [code](https://github.com/qlabs-eng/dust)). Q Labs also runs noise, a public speedrun for backprop-free pretraining on a fixed 37.7M-parameter GPT and a fixed FineWeb training set, with three tracks and community records ([repo](https://github.com/qlabs-eng/noise), [landing page](https://qlabs.sh/noise)). Boost and Boost to Cut, by spicylemonade, are record methods inside that benchmark. They replace the Dust estimator with per-block readouts and a chained credit pass ([repo](https://github.com/spicylemonade/boost-cut)). Weight-space evolution strategies (EGGROLL) are the main comparison point ([arXiv:2511.16652](https://arxiv.org/abs/2511.16652)). On the current noise README, the best Track 1 entry (lowest validation loss, 30M tokens, one 8xH100 node) is 3.338, against backprop references of 3.340 (ANVIL II on block matrices, Adam elsewhere) and 3.419 (Adam with learning-rate cooldown). The best Track 2 entry reaches validation loss 3.5 in 22.9 s on 8 GPUs, and no backprop time to that threshold is given. The README gives no record dates. It says the 10M-token tracks were archived when the training set changed on 22 October 2026. On the Dust paper's 37.7M model with population 16,384, Dust reaches test loss 5.916 at 1M tokens against backprop's 5.959. It trails backprop at 10M (5.049 vs 4.989 in the appendix table) and at 20M (4.802 vs 4.633). Two sources disagree on the 10M comparison; section 1.14 gives both.

Shorthand used below: GPT is a decoder-only transformer, MLM is masked language modeling, ES is evolution strategies, ANVIL II is the matrix optimizer used in noise (its specification is not public, see section 2.4), and "draw" is one random jitter applied to every token of the selected layers in one forward evaluation.

---

## 1. Dust (activation perturbation, token-level rewards, outer-product weight estimate)

Source: [Dust paper page](https://qlabs.sh/research/dust), [appendix](https://qlabs.sh/research/dust/appendix.html), [dust code at noise commit b6405a1](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py), [dust repository](https://github.com/qlabs-eng/dust).

### 1.1 Setting in the paper

The base model is an 8-layer, width-512 GPT with 37.7M parameters and a 4096-token BPE tokenizer, trained on FineWeb ([paper page](https://qlabs.sh/research/dust)). The batch is 8 sequences of 2048 tokens (16,384 tokens per update) and the run is one epoch. The main experiments use SGD with momentum and a constant learning rate. Adam is tested separately in section 3.3 of the paper ([paper page](https://qlabs.sh/research/dust)).

### 1.2 The perturbation

For a linear layer with output $y_t = W x_t$, where $x_t \in \mathbb{R}^{d_{in}}$ and $y_t \in \mathbb{R}^{d_{out}}$, one draw $i$ gives

$$\tilde y_t^{(i)} = y_t + \sigma\, a_t^{(i)}, \qquad a_t^{(i)} \sim \mathcal{N}(0, I_{d_{out}}),$$

with the noise drawn independently for every token $t$ of every sequence $b$. Tensor shapes: $x \in \mathbb{R}^{B \times T \times d_{in}}$, $a \in \mathbb{R}^{B \times T \times d_{out}}$. Each token therefore acts as its own population member, and one forward pass tests many perturbations at once. The population $K$ is the number of draws per update. The paper says a draw is slightly cheaper than a forward pass of the batch ([paper page](https://qlabs.sh/research/dust)).

### 1.3 Token rewards and discounted future credit

Let $\ell_s^{(i)}$ be the loss at token $s$ under draw $i$, and $\bar\ell_s$ the mean of $\ell_s^{(i)}$ over the draws in the same chunk. The per-token reward is

$$c_s^{(i)} = \bar\ell_s - \ell_s^{(i)},$$

so a draw that lowers the loss at $s$ gets a positive reward. The discounted credit that reaches token $t$ from its future is

$$r_t^{(i)} = \sum_{s \ge t} \gamma^{\,s-t}\, c_s^{(i)}.$$

Settings in the noise code ([dust.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py)):

- Most direct draws use $\gamma = 0$, so a draw is rewarded only by its own token.
- Block 0 writers, hidden layer and hubs use `first_gamma = 0.3`, applied as a reward-to-go kernel with 12 taps (truncated where the decay falls below $10^{-6}$).
- Attention-internal scores for keys, values, gates and value embeddings use $\gamma = 0.98$, summed over later outputs inside the attention window.
- Queries are scored at their own token only.

### 1.4 Output-error estimate and weight gradient

The output error at token $t$ is

$$\hat g_t = -\frac{1}{K\sigma} \sum_{i=1}^{K} r_t^{(i)}\, a_t^{(i)} \in \mathbb{R}^{d_{out}}.$$

Because $r_t$ carries the sign of the loss change, $-r_t$ is the forward difference $\ell^{(i)} - \bar\ell$, so $\hat g_t$ estimates $\partial L / \partial y_t$. The weight estimate is the outer product summed over all tokens:

$$\widehat G_W = \sum_{b,t} \hat g_{b,t}\, x_{b,t}^\top \in \mathbb{R}^{d_{out} \times d_{in}},$$

written as `einsum('btd,bti->di')` in the code, with the result matching $W$'s shape ([dust.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py)). For the embedding, $x_t$ is one-hot, so the update is a scatter-add of $\hat g_t$ into the row of the token id, implemented with `index_add` into a `(vocab, n_embd)` tensor ([dust.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py)).

### 1.5 Attention internals (no derivatives)

Only the block's attention is recomputed under each jitter, from cached clean activations. Each draw is scored per head against the averaged attention-output error:

$$c_s = -\langle \hat g_s,\ \Delta o_s \rangle,$$

where $\Delta o_s$ is the change in the attention output at token $s$ ([paper page](https://qlabs.sh/research/dust), [dust.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py)). Queries use this score at their own token. Keys, values, gates and value embeddings sum it over later outputs with weight $\gamma^{s-t}$ inside the window. Gate outputs have shape `(b, t, n_head)`, and q, k, v and value-embedding outputs have shape `(b, t, n_embd)`.

### 1.6 LM head

The head is jittered on cached logits, one slab of 256 vocabulary columns at a time. The softmax partition is recomputed from cached sums over the other slabs, and the error (shape `(b, t, V)`) is multiplied by the cached final hidden state. The head noise is 0.1 on pre-soft-cap logits in the noise code and 0.05 in the paper's Table 3 at K = 16,384 with SGD ([dust.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py), [appendix](https://qlabs.sh/research/dust/appendix.html)). With a 20-token vocabulary this step is cheap.

### 1.7 Residual mixing scalars by ES

The residual scalars are trained by ordinary two-sided weight-space ES. The noise code has 16 scalars (`resid_lambdas` and `x0_lambdas`, two per block for 8 blocks), perturbed with $\sigma = 0.03$. Each direction is scored by the difference of summed valid-token losses divided by $2\sigma$ ([dust.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py)). The paper uses 8 such directions ([appendix](https://qlabs.sh/research/dust/appendix.html)). A 36-block model has $2L = 72$ scalars.

### 1.8 Interference handling and reruns

Each layer type is jittered in a separate pass with its own noise scale, and each block gets its own passes. A draw for block $l$ reruns only blocks $l$ onward from the cached clean forward ([paper page](https://qlabs.sh/research/dust)). This rerun rule is the main cost driver at depth (section 5.3).

### 1.9 Passes and draws per update (noise code, 8 blocks, per GPU)

From [dust.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py):

- One clean forward with capture.
- Direct draws, each rerunning from the block's input: writers (attention and MLP `c_proj`) 3 batched passes per block, hidden (`mlp.c_fc`) 6 per block, hubs 6 per block, embedding 24 full passes.
- Head: 576 cheap slab passes on cached logits.
- Attention internals: attention-only recomputes in chunks of 12 for q, v and gate, 13 for k, and 16 for value embeddings.
- Residual scalars: 2 full forwards (both signs).

Draw totals in the noise code are 384 writer draws per block, 768 hidden, 768 hubs, 576 q and 576 v, 192 gate, 2,496 k, 6,912 value embedding per block, 73,728 for the head and 3,072 for the embedding. The paper's Table 3 at K = 16,384 uses 7,040 writer-pair draws per block, 7,040 hidden, 1,664 attention output, 640 embedding and 65,536 head ([appendix](https://qlabs.sh/research/dust/appendix.html)). The two sets of totals belong to different configurations, so do not mix them.

### 1.10 Noise scales

| Site | Noise code (noise commit b6405a1) | Paper, K = 16,384, SGD (appendix Table 3) |
|---|---|---|
| Writers | 0.4 | 0.2 |
| Hidden (`mlp.c_fc`) | 0.4 | 0.2 |
| Hubs (attention output) | 0.4 (layers 0-3), 0.8 (layers 4-7) | 0.2 (0.4 in deeper half) |
| Embedding | 0.8 | 0.2 (0.4 with Adam) |
| Query, value, gate, key, value embedding | 0.1 (attention internals) | 0.05 |
| Head | 0.1 | 0.05 |
| Residual scalars | 0.03 | 0.03 |

Sources: [dust.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py), [appendix](https://qlabs.sh/research/dust/appendix.html).

### 1.11 Tuning

The paper tunes per-layer-type noise scale, attention-internal credit decay and each layer's share of the population. Two methods are used: a short training run per grid point, keeping the lowest loss, and a grid search that maximizes cosine similarity between the Dust estimate and the backprop gradient on one batch. The cosine picks candidates and training makes the final choice ([paper page](https://qlabs.sh/research/dust)). The paper reports a fitted cosine law $\cos(K) = c_{max} / \sqrt{1 + c/K}$ with per-layer RMSE below 0.06 ([paper page](https://qlabs.sh/research/dust)).

### 1.12 Optimizer in the noise code

AdamW with weight decay 0 (plain Adam), betas (0.7, 0.95), eps $10^{-8}$. Learning rate 0.0015 for matrices and the LM head, 0.45 for `wte` and value embeddings (a 300x group multiplier), 0.045 for the lambdas (30x). There is no warmup. The rate decays linearly to zero over the last 30 percent of updates ([dust.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py)). The paper's main runs use SGD with momentum, and its Adam variant is in section 3.3 ([paper page](https://qlabs.sh/research/dust)).

### 1.13 Hardware and timing

The default run uses 8 GPUs with `torchrun`, and the backprop baseline runs on one GPU ([dust repository](https://github.com/qlabs-eng/dust)). The paper appendix gives no wall-clock times or GPU counts for its experiments ([appendix](https://qlabs.sh/research/dust/appendix.html)). The noise code times each run after an untimed warmup on random tokens, synchronizes at each timed update, excludes validation pauses, and logs cumulative `train_seconds` ([dust.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py)). The recipe docstring says that cutting the embedding draws saves about 30 percent of time.

### 1.14 Results versus backprop

Test loss from the paper's Table 1 ([appendix](https://qlabs.sh/research/dust/appendix.html)):

| Method | Population | 1M tokens | 10M tokens | 20M tokens |
|---|---|---|---|---|
| Dust | 64 | 6.290 | 5.397 | 5.265 |
| Dust | 1k | 5.934 | 5.111 | 4.960 |
| Dust | 16k | 5.916 | 5.049 | 4.802 |
| Dust (fitted limit) | infinite | 5.892 | 5.013 | 4.431 |
| EGGROLL (weight-space ES) | 16k | 6.708 | 6.033 | 5.865 |
| Backprop | none | 5.959 | 4.989 | 4.633 |

Other reported numbers from the [paper page](https://qlabs.sh/research/dust):

- At 1M tokens with SGD, Dust ends below backprop from about 1,000 draws on.
- At 1M tokens with Adam (section 3.3), Dust's fitted limit is 5.248 against backprop's 5.361. The 95 percent bootstrap interval for Dust's limit is 5.17 to 5.30.
- At 20M tokens, the fitted limit 4.431 (95 percent interval 3.89 to 4.58) is an extrapolation, and the authors call it loosely constrained, not measured.
- The paper's Section 4.1 table gives 5.036 for Dust at 16k draws on the 38M model at 10M tokens, against backprop's 5.015. The appendix Table 1 gives 5.049 and 4.989 for the same comparison. The two sources disagree and I did not resolve which configuration each refers to. Use the appendix for the table and treat the 10M gap as 0.02 to 0.06 in favor of backprop.
- EGGROLL at 16k draws does not reach Dust at 64 draws. The authors estimate that matching Dust at 64 draws needs about 3,300 to 23,000 times more EGGROLL population, depending on the fit ([appendix](https://qlabs.sh/research/dust/appendix.html)).

Hardware for the paper's large runs is not stated on the pages read. The noise and repository code are written for 8 GPUs.

### 1.15 Links

- Paper page: https://qlabs.sh/research/dust (no arXiv ID found on the page)
- Appendix: https://qlabs.sh/research/dust/appendix.html
- Code: https://github.com/qlabs-eng/dust (MIT license; the README says the code keeps the paper's estimator but omits "execution optimizations used in the full experiments")
- Noise version at commit b6405a1: https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py

---

## 2. The noise track rules and records

Source: [noise README](https://github.com/qlabs-eng/noise), [landing page](https://qlabs.sh/noise), and the record scripts linked below. The README was read at its current state and the record scripts at the commits it links.

### 2.1 Rules and timing

From the README ([repo](https://github.com/qlabs-eng/noise)):

- **No backpropagation.** No autograd and no hand-written backward pass. Error signals come from forward evaluation, with at most one step through a nonlinear part of the model. Steps through linear maps do not count, and a signal that has already stepped cannot step again.
- **Tracks.** Track 1 is lowest validation loss after one epoch on one 8xH100 node, with no time limit. Track 2 is the shortest time on one 8xH100 node to reach validation loss 3.5. Track 3 is Track 1 with the architecture free.
- **Fixed model** (Tracks 1 and 2): final weights must load into `utils/model.py` with a strict state-dict match. Track 3 uses `train/arch/model.py`.
- **Fixed data.** Only `data/train.pt`, 30M tokens and 14,648 sequences of 2,048 predictions, for at most one epoch.
- **Score.** Mean next-token cross-entropy over 1,114,112 validation predictions. No checkpoint selection on validation loss.
- **Timing.** From the start of the first update to the end of the last, excluding validation. Seed 0. An untimed warmup on random tokens may compile kernels, provided training restarts from a fresh state.

The earlier 10M-token benchmark, described on the [landing page](https://qlabs.sh/noise), differs. It used the same 37.7M model with 10M training tokens and validation loss on the FineWeb validation set. It listed a Track 1 record of 4.167 in 1.76 h (Dust with AdamW), later updated to 3.676. It listed backprop with Adam at 3.769, and a Track 2 speedrun of 1.76 h later updated to 1.6 s. The page reports no test losses. Its record holders are not named in the table. The README says the 10M tracks are archived in `archive/10M/`. Do not compare its numbers with the 30M numbers below.

### 2.2 Records

From the [noise README](https://github.com/qlabs-eng/noise). Commit hashes are abbreviated as in the README. No record dates are given. Times are as reported.

**Track 1: validation loss, 30M tokens**

| # | Val loss | Method (README wording, paraphrased) | Time | Script at commit | Contributors |
|---|---|---|---|---|---|
| 1 | 3.767 | Dust with Adam, population 24k, learning-rate cooldown | 5 h 17 min | [train/val/dust.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py) (b6405a1) | none listed |
| 2 | 3.409 | 10M record #14: Boost to Cut with own queries, exact values, future-token probes and an end merge | 1 h 43 min | [train/val/run.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/run.py) (b6405a1) | JaydenTeoh, spicylemonade, hoo0681, Lazarus-931, CharlesDDDD, NotDesigned |
| 3 | 3.338 | Forward readout of cross-token credit along each token's path, fixed Boost, lower ANVIL II weight decay | 56 min | train/val/run.py (6f0c450) | CharlesDDDD |

**Track 2: speedrun to validation loss 3.5, 30M tokens**

| # | Val loss | Method | Time | Script at commit | Contributors |
|---|---|---|---|---|---|
| 1 | 3.409 | Track 1 #2 recipe | 1 h 43 min | train/speedrun/run.py (b6405a1) | as Track 1 #2 |
| 2 | 3.499 | Identity path first, then Cut-lite at 8 rows per update, faster chain reruns | 8 min 20 s | train/speedrun/run.py (4586b02) | RobertTLange |
| 3 | 3.494 | Shorter identity phase, then Cut with paired MLP roots and a difficulty-balanced row order | 2 min 3 s | [train/speedrun/run.py](https://github.com/qlabs-eng/noise/blob/4f04654/train/speedrun/run.py) (8be7921) | conglu1997 |
| 4 | 3.498 | Track 2 #3's identity phase, then Cut with two single roots, fused Gluon chain kernels and overlapped bf16 collectives | 22.9 s | [train/speedrun/run.py](https://github.com/qlabs-eng/noise/blob/4f04654/train/speedrun/run.py) (4f04654) | RobertTLange |

**Track 3: open architecture, 30M tokens**

| # | Val loss | Method | Time | Script at commit | Contributors |
|---|---|---|---|---|---|
| 1 | 3.767 | Dust with Adam (as Track 1 #1) | 5 h 17 min | train/val/dust.py (b6405a1) | none listed |
| 2 | 3.409 | Track 1 #2 recipe | 1 h 43 min | train/arch/run.py (b6405a1) | as Track 1 #2 |
| 3 | 3.378 | Quarter-turn previous-token tap in the embedding; exact future-token probes | 3 h 50 min | train/arch/run.py (d45f830) | dylantirandaz |
| 4 | 3.358 | 6 blocks with hashed n-gram tables; future and input-key probes by moments | 27 min | train/arch/run.py (e0e1ff2) | Euraba |
| 5 | 3.211 | 12 x 1024 GPT with hashed n-gram value embeddings; Cut-lite with input-key probes | 2 h 57 min | [train/arch/run.py](https://github.com/qlabs-eng/noise/blob/561c1f1/train/arch/run.py) (561c1f1) | RobertTLange |

Backprop references from the README, same model, data and epoch, seed 0: Adam with learning-rate cooldown 3.419, and ANVIL II on block matrices with Adam elsewhere 3.340.

Track 3 entries use a different architecture from the Track 1 and 2 model, so the backprop references do not apply to them.

### 2.3 What the records do, by method

The README names the methods but does not define them. The definitions below come from the scripts. The script text was partly truncated in the reads, so the notes say where the mechanism is not visible.

- **Boost and Cut.** Boost is a per-layer readout that learns only from a shared output head reading the stream right after each layer. Cut holds attention fixed and measures end-to-end credit through each token's routed MLP units. Both are described in section 3. Record #14 (Track 1 #2) is the Boost-to-Cut recipe.
- **Cut-lite.** Cut without a Boost phase (`switch = 0.0`). In the arch record (`train/arch/run.py` at 561c1f1) it uses 8 rows per update, one sequence per GPU, full Hadamard cycles for block-stream credits with `chain_sigma = 0.01` and `chain_tokens = 512`, and `route_k = 192` MLP units per token. Its head uses per-logit coordinate differences with $\sigma = 0.1$. Future-token probes are off ([arch/run.py](https://github.com/qlabs-eng/noise/blob/561c1f1/train/arch/run.py)).
- **Identity path.** The first phase of several speedrun records. In the Track 2 #4 script, the identity phase runs 355 updates of 8 rows out of a 1,831-update identity plan, and the handoff happens inside an ANVIL momentum ramp. Shaping mechanisms run for the first 195 updates: the `wte` rows keep 1/256 of their energy in 32 gate channels, value-embedding common motion gets gain 4.0, and partner-layer motion gets gain 0.5. Each `attn.c_proj` starts as a norm-matched transpose of its `c_v`. The identity-path credit formula itself is not visible in the read text ([speedrun/run.py at 4f04654](https://github.com/qlabs-eng/noise/blob/4f04654/train/speedrun/run.py)).
- **Roots.** The sites where a perturbation enters the residual stream. Boost and Cut use the embedding plus 8 mid-stream sites, nine roots in all, each with a full 512-direction orthogonal cycle ([boost-cut run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)). The Track 2 #4 script uses two single roots, at blocks 4 and 6 (`chain_depth_roots = (4, 6)`), with each root's mid stream moved along a 512-direction signed Hadamard frame whose signs are random per token ([speedrun/run.py at 4f04654](https://github.com/qlabs-eng/noise/blob/4f04654/train/speedrun/run.py)).
- **Paired MLP roots.** Earlier Track 2 variants paired roots (0-1, 2-4, 6-7). The exact pairing rule is not visible in the read text beyond the root indices.
- **Future-token probes.** Sources are the embedding and the first four layers' `mlp.c_fc` and `attn.c_q` outputs. Each source token gets 128 central pairs with $\sigma = 0.01$, in chunks of 512. Each probe is scored at later raw attention hubs ([boost-cut run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)). The arch records use exact future-token probes.
- **Input-key probes.** Directions on the attention key input, with `key_path_sigma = 0.01` and `key_path_tokens = 64` in the arch record. The count is 1,024 directions (one full cycle) in the Cut-lite estimator ([arch/run.py](https://github.com/qlabs-eng/noise/blob/561c1f1/train/arch/run.py)) and 512 in the Boost-to-Cut script.
- **Probes by moments.** The speedrun computes query and key credit from a degree-2 moment model of the softmax under finite coordinate steps, through the norm and RoPE. The arch record #4 uses "future and input-key probes by moments" ([speedrun/run.py at 4f04654](https://github.com/qlabs-eng/noise/blob/4f04654/train/speedrun/run.py); README).
- **Quarter-turn previous-token tap.** Name only. The mechanism is in the arch script, which was truncated in the read. Not found in the readable text.
- **Hashed n-gram tables.** Three tables of $2^{20}$ rows by 128, indexed by (previous token, token), (two previous tokens, token) and (token two back, token). The three lookups are summed, and each block has its own 128-to-1024 projection that feeds its value embedding. The hash function is not in the read text ([arch/run.py](https://github.com/qlabs-eng/noise/blob/561c1f1/train/arch/run.py)).
- **Forward readout of cross-token credit.** A move at source token $t$ changes query $s$'s attention weight and output. The record measures this exactly, without a derivative: each query's weight is updated by $a \cdot \mathrm{expm1}(\Delta s)$, and the value credit is the sum over queries of $A[s,t]$ times the target. This is the Track 1 #3 change ([boost-cut run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py) for the functions `joint_scores` and `key_scores`; README for the record).
- **Fused Gluon chain kernels.** The script names `fused_root_readout_fast`, `fused_root_visit_change` (Gluon), `fused_query_replay_factored_gluon` (Gluon) and `fused_resp_visit_readout_fast`. The kernel bodies are not in the read text ([speedrun/run.py at 4f04654](https://github.com/qlabs-eng/noise/blob/4f04654/train/speedrun/run.py)).
- **Overlapped bf16 collectives.** ANVIL gradients are reduce-scattered in bf16 asynchronously. Each rank updates 1/8 of every ANVIL shape batch, runs the polar step on its shard, and all-gathers the result. Other gradients are all-reduced in bf16 asynchronously, and each collective is waited on where its result is first read ([speedrun/run.py at 4f04654](https://github.com/qlabs-eng/noise/blob/4f04654/train/speedrun/run.py)).

### 2.4 ANVIL II

ANVIL II is named in the README and used in the backprop baseline (3.340) and in several records. The noise code shows its settings and parts of its structure, but not its specification. The visible constants are a fast EMA with `fast_beta = 0.85`, a slow EMA with `slow_beta = 0.98`, `lane_beta = 0.9`, a six-map quintic polar iteration with lane normalization, a step of `lr * max(1, rows/cols)^0.5`, and `anvil_wd = 2.25` ([boost-cut run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py), [speedrun/run.py at 4f04654](https://github.com/qlabs-eng/noise/blob/4f04654/train/speedrun/run.py)). A public paper or specification for ANVIL II was not found. A search for the name returned only general Muon-family optimizer work.

### 2.5 Links

- Repository: https://github.com/qlabs-eng/noise
- Landing page: https://qlabs.sh/noise
- Boost-Cut release with the two entries' checkpoints: https://github.com/spicylemonade/boost-cut/releases/tag/entries-v1

---

## 3. Boost and Boost to Cut

Source: [boost-cut README](https://github.com/spicylemonade/boost-cut), [boost-cut noise/train/boostcut/run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py). The NOTES.md and ledger files could not be read: the raw NOTES.md returned HTTP 404 and the GitHub page returned HTTP 402 to the fetch tool. The README names `ledger/verification.md`, `ledger/runs.json` and `ledger/compute.md`, none of which were read.

### 3.1 Configuration

The default preset is 38m: 8 layers, width 512, MLP hidden 2048, 8 heads of dimension 64, vocabulary 4096 (the noise tokenizer), sequence length 2048, logit soft cap 15 ([run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)). Presets run from 2m (2 layers, width 128) to 243m (16 layers, width 1024). The Adam base rate is 0.003 with betas (0.7, 0.95) and a final decay fraction of 0.3 ([run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)). The noise-run copy at commit b6405a1 uses lr 0.0021 and betas (0.5, 0.98), so the two copies are different versions.

### 3.2 The shared readout

The readout for a stream $x \in \mathbb{R}^{T \times 512}$ is

$$z = \mathrm{RMSNorm}_{\text{no gain}}(x)\, W_{head}^\top, \qquad \hat\ell = \mathrm{CE}\big(15\tanh(z/15),\ y\big),$$

where $W_{head}$ is the first $V$ rows of the shared `lm_head` and targets below zero are ignored ([run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)). Each attention and MLP layer learns only from this shared head reading the stream immediately after it. The stream after attention (`xmid`) and the stream after the MLP (`xout`) are both read. Intermediate readouts also train the head ([README](https://github.com/spicylemonade/boost-cut); [run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)).

### 3.3 Per-token credit from orthogonal nudges

Boost measures each token's credit along a full set of orthogonal directions. The direction matrix is $H = \mathrm{hadamard}(512) \cdot \mathrm{diag}(\text{signs})$, with random column signs drawn per call. Each row has norm $\sqrt{512}$. For a stream $x_t$, the step is $s = \sigma \cdot \mathrm{RMS}(x_t)$ with $\sigma = 0.005$. The forward difference along each direction is

$$d_{t,j} = \frac{\ell(x_t + s H_j) - \ell(x_t)}{s},$$

and the credit reconstructs the first-order gradient:

$$g_t = \frac{1}{512} \sum_{j=1}^{512} d_{t,j} H_j.$$

Because the readout depends only on token $t$, the measurement contains no noise from other tokens ([README](https://github.com/spicylemonade/boost-cut); [run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)). For the head itself, the code uses closed-form coordinate differences of the soft-capped cross entropy along each logit.

### 3.4 Writers and hidden units

Each block's weight gradient is an outer product summed over tokens. For `attn.c_proj`, the gradient is $g_{mid}^\top\, a_{in}$, with shapes $(T \times 512)^\top (T \times 512) \to 512 \times 512$. For `mlp.c_proj`, it is $g_{out}^\top\, \mathrm{relu}^2(c_{fc})$, with shapes $512 \times 2048$. The hidden step for `mlp.c_fc` takes the top 256 units by signed pre-activation $u$, moves them by $0.05\,\mathrm{RMS}(u)$ along a $256 \times 256$ random Hadamard, uses central differences of $\mathrm{relu}^2$, and multiplies by the output-column stream credit ([run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)). The Cut phase uses 128 units per token, ranked by $\mathrm{relu}^2(\mathrm{pre}) \times \|c_{proj}\text{ column}\|$.

### 3.5 Attention internals

- **Keys.** Exact single-key experiments in closed form. Each key dimension gets a step of $0.003\,\mathrm{RMS}(k)$.
- **Queries.** 64 draws per head, raw query moved by $\pm 0.1$ Hadamard codes, attention rerun, output change scored per head.
- **Values.** 512 draws of raw $\pm 1$ codes with no sigma, scored and carried back with the matched filter $K = A / \max_t A$.
- **Gates.** 64 draws, finite difference through the sigmoid slope with eps 0.1.
- **Value embeddings.** The value gradient times $2\,\mathrm{sigmoid}(\text{gate})$.

The Boost phase uses the attention-output targets from the writers above ([run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)).

### 3.6 The embedding and mixing scalars

The embedding gets a full cycle on the raw embedding along its own path through layer 0's attention, with other positions held clean, a step of $0.005\,\mathrm{RMS}(e)$, and 16 directions per chunk. Mixing scalars use central differences with step 0.01 ([run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)).

### 3.7 The phase switch

The Boost phase covers the first 30 percent of the epoch when run with `--switch 0.3`. It runs $n_1 = \mathrm{round}(0.3\, \text{steps}_1)$ Boost updates of 4 sequences each, rounded down to an even count. The first Cut update index is $j_0 = n_1 \cdot 4 / 8$, so Boost sees about the first 30 percent of sequences. At handoff the weights, optimizer state and data order carry over, and the ANVIL counter is set to $t = j_0$ ([run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)). The noise-run copy of the Boost-to-Cut script at commit b6405a1 uses `switch = 0.2`, which is a different version ([train/val/run.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/run.py)).

### 3.8 Cut phase (attention held clean)

Cut holds each block's mid stream clean, so a move changes only the residual stream of its own token. Each token is then its own experiment. The estimand includes credit through later MLPs, the identity path and the $x_0$ injections, but not through later attention ([run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)).

- **Routes.** Each token's top 128 MLP units per block, ranked by $\mathrm{relu}^2(\mathrm{pre}) \times \|c_{proj}\text{ column}\|$. Other units stay at their clean outputs.
- **Chain.** Each site gets a full cycle of 512 moves per token with step $0.01\,\mathrm{RMS}$. Each move propagates through later blocks: $\delta \leftarrow \lambda_r \delta + \lambda_0 \delta x_0$ at residual junctions, then MLP changes on routed units only. The final loss uses one-sided forward differences on the head.
- **Mixing scalars.** Four extra moves per block, combined as central differences over tokens.
- **Block outputs.** $d_{out} = \lambda_r[i+1]\, \mathrm{credit}_{mid}[i+1]$, or the head-only credit for the last block.

### 3.9 Optimizer, schedule and batch

Matrices (2-D, `transformer.h.*`, excluding `ve_gate`) use ANVIL II. Everything else uses AdamW with weight decay 0, eps $10^{-8}$ and betas (0.7, 0.95), with group multipliers of 300 for `wte` and `value_embeds`, 30 for `lambdas`, and 1 otherwise. The schedule is constant for 70 percent of each phase and then decays linearly. Boost uses 4 sequences per update and Cut uses 8. With 8 GPUs, Boost uses 2 GPUs per sequence and Cut uses 1 per GPU. An untimed step compiles both phases, then training restarts from a fresh model ([run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)).

### 3.10 Passes, caching and cost per update

The code does not state a count of full forward passes per update. Each update runs one clean capture, then batched finite differences. The capture hooks every `Linear` and `Embedding` to store its float32 input and output, plus each block's input. It also caches orthogonal-noise frames per site (cleared each step), Hadamard matrices and bf16 attention maps. The kernels use `torch.compile` ([run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)). Per block, Boost needs a full 512-direction cycle at each measured stream, 64 query draws per head, 512 value draws and a hidden-unit step. The count of local evaluations is therefore large per block, but each evaluation is a readout or one sublayer. Section 5.3 gives the depth scaling.

### 3.11 Reported numbers

From the [boost-cut README](https://github.com/spicylemonade/boost-cut), at 10M tokens on the 38M model, 8 x H100:

| Entry | Val loss (seed 0) | Train time | Notes |
|---|---|---|---|
| Boost to Cut | 3.945 | 456 s | Test loss 3.935 +/- 0.012 over 3 seeds |
| Boost | 4.123 | 314 s | Boost alone stalls at test loss 4.062 |
| Backprop, same model and optimizer | n/a | n/a | Test loss 3.706 +/- 0.009 over 3 seeds |
| Cut without Boost head start | n/a | n/a | 3.984 (3 seeds) |

- The record recipe reproduces at val 3.992 in 4,292 s on the authors' 8 x H100 nodes. The speedrun script runs in 1,146 s. The ratio from those two times is 3.7 by my arithmetic, while the README states 3.6x. I use the times as stated.
- Restricting Cut to 128 of 2,048 MLP units makes the phase 2.4x cheaper, at a cost of 0.006 loss (README).
- The README says the `llm-slop-review/` folder holds a paper draft written by an LLM, still under review, with some runs in progress. Treat the draft as unreviewed.
- The noise Track 1 #2 record lists the same lineage at 3.409 on the 30M track. That is a different data budget from the 10M numbers above, so the two are not comparable.

### 3.12 Links

- Repository: https://github.com/spicylemonade/boost-cut
- Per-entry script: https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py
- Entries release: https://github.com/spicylemonade/boost-cut/releases/tag/entries-v1
- Noise benchmark: https://github.com/qlabs-eng/noise

---

## 4. The interpretability-inspired 2.3x speedup and the Already Clean ideas

### 4.1 What is public about the 2.3x post

Known: the post at https://x.com/spicey_lemonade/status/2111636408540799141 is cited as a source. The fetch tool returned HTTP 402 for it. A Nitter mirror of the URL returned an unrelated Nitter project notice, with no post text. The xcancel mirror did not resolve. A web search for the post's wording and for "2.3x" with the handle returned no copy of the text. The method and the 2.3x figure are therefore **not found**, and I have not verified the claim.

Known from public sources, and not linked to the post by any source: the README reports Boost is about 3.7x faster than the record recipe on the same hardware, computed from 4,292 s and 1,146 s, though the README states 3.6x. Restricting Cut to 128 units makes the Cut phase 2.4x cheaper at 0.006 loss. The noise Track 2 runs reach validation loss 3.5 in 8 min 20 s, 2 min 3 s and 22.9 s on 8 GPUs ([README](https://github.com/qlabs-eng/noise)). None of these is the 2.3x claim, and I do not present them as its source.

### 4.2 The Already Clean essay

Source: [Already Clean, by Nick Cammarata (Goodfire)](https://already-clean.goodfire.com/) and its [full text](https://already-clean.goodfire.com/essay.md). The date of October 15, 2026 comes from a secondary report, [Cryptobriefing](https://cryptobriefing.com/nick-cammarata-neural-networks-interpretable/), and I did not verify it on the essay page. The essay does not describe a training method, an optimizer or a speedup. Everything in this section that goes beyond the essay's own text is inference, and is labeled as such.

**Clean units.** The essay argues that a network's own neurons and circuits are the units to study directly. It reframes neurons as gates that shape the flow of activations: "Now I think of them more as control flow" ([essay](https://already-clean.goodfire.com/essay.md)).

**Recursive generalization.** A neuron generalizes along several axes and branches outward, "a neuron branching out, and its branches branching again", and watching training, "the branches keep growing" ([essay](https://already-clean.goodfire.com/essay.md)). The essay says branch growth follows "a particular order" over training ([essay](https://already-clean.goodfire.com/essay.md)).

**Concentrated credit.** The essay's circuit-racing section says "the pretrained twin takes almost all the credit for the shared objective", and the broad circuit wins because "it gets more of the learning signal" ([essay](https://already-clean.goodfire.com/essay.md)). The essay does not use the phrase concentrated credit.

**Small hubs.** Removing a small set of neurons changes a prediction sharply, for example "Remove the eight neurons with a Santa negative side", and "The diagnosis hub is just four neurons" ([essay](https://already-clean.goodfire.com/essay.md)).

**Negative side of a gate.** "The negative side never gets gradient on its own" ([essay](https://already-clean.goodfire.com/essay.md)).

**Mapping to training procedures (inferred).**

1. **Credit concentration and routing.** Cut's routing of 128 of 2,048 MLP units per token by $\mathrm{relu}^2(\mathrm{pre}) \times \|c_{proj}\|$ concentrates credit on a few units. This is the same bet as the essay's circuit-racing picture. No source tests the link.
2. **Staged growth.** The identity phase followed by a Cut phase, and the shaping mechanisms in the first 195 updates of a speedrun record, are staged schedules. The essay's order of growth suggests staging, but neither source ties the two together.
3. **Unit-level selection.** The essay's small-hub observations support choosing a few units to update. This is the basis of recipe 3 in section 5.4.
4. **A blind spot in relu-squared ranking (inferred).** Cut ranks units by $\mathrm{relu}^2(\mathrm{pre})$. That value is zero whenever the pre-activation is negative. If the essay's negative-side point holds, this rule can never select a unit that is active on its negative side, and such units may carry credit. A test would compare the rule with a signed or absolute-value rule on a held-out batch.

### 4.3 Links

- Essay: https://already-clean.goodfire.com/ (full text: https://already-clean.goodfire.com/essay.md)
- Secondary report on the essay, unverified date: https://cryptobriefing.com/nick-cammarata-neural-networks-interpretable/

---

## 5. Transfer to bidirectional MLM on proteins with 10 minutes on one GPU

### 5.1 What changes from the noise setting

- **Causal versus bidirectional.** The Dust and Boost credit windows assume that later tokens are the only ones that depend on earlier ones. Under bidirectional attention, a move at any token changes every query's attention weights. Cut's frozen-attention assumption (each token is its own experiment) fails. The forward readout of cross-token credit (section 2.3) sums over all queries and so survives, at extra cost.
- **Masked positions only carry loss.** At 15 percent masking, only about 15 percent of positions have a target. The readouts already ignore targets below zero ([run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)), so reward and probe work for unmasked targets is wasted. Unmasked tokens still act as sources through attention, so input-key and value probes must keep them.
- **20-token vocabulary.** The head is small. Dust's head jitter and Boost's closed-form head differences are cheap at this size. The exact vocabulary and special-token ids depend on the tokenizer and must be checked (the repository's tokenizer, not assumed here).
- **36 deep blocks.** Dust's rerun rule and Cut's chain both scale with depth. Boost's readouts scale with the number of blocks, with a large per-block constant (section 3.10).
- **Width 768 is not a power of two.** Boost's directions are a Sylvester Hadamard matrix of order 512, and the code uses `hadamard` for power-of-two orders ([run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)). A 768-direction basis needs a different construction, such as a random orthogonal matrix from QR or a Kronecker product of a 3 x 3 orthogonal and a 256 Hadamard. The choice is my inference and must be checked before use.
- **Hardware and time.** The noise records use one 8 x H100 node and runs of seconds to hours. Our budget is 600 s on one H100-class GPU. Noise's Track 2 #4 reaches 3.5 in 22.9 s on 8 GPUs, which is about 183 GPU-seconds, and that is a different objective and model. Do not carry its timing over.
- **Selection rule.** The noise score forbids checkpoint selection on validation loss. The repository benchmark selects on validation loss and reports test loss, so the Dust and noise recipes must be adapted to that rule.
- **Data.** Probe batches must come from the training split, and the train/validation/test split must respect homology clusters. Neither source addresses this for proteins.

### 5.2 Method by method

**Dust.** The estimator works on bidirectional attention if the future-credit kernel becomes a symmetric or learned kernel over all tokens. This is an inference. Dust's residual-scalar ES transfers directly, since it touches only $2L$ scalars. The full-weight Dust estimator is the expensive part (section 5.3).

**Boost readouts.** The per-block shared head transfers without change, apart from the masked-target rule and the 768-width basis. Attention probes need the all-query form.

**Cut chain.** The chain transfers only with the bidirectional changes above and a much smaller root set. Its cost grows with depth.

**Future-token probes.** In bidirectional attention, "future" has no meaning. The probes must be redefined as probes to any unmasked source, which is an inference.

**Hashed n-gram tables.** These record gains in noise come from a causal, language setting. For proteins, residue n-grams may carry signal, but the claim is untested here. The tables must be built from the training split only, or they leak across splits.

**EGGROLL and weight-space ES.** The Dust paper compares ES with its estimator and finds ES far behind at equal population ([appendix](https://qlabs.sh/research/dust/appendix.html)). The EGGROLL abstract reports speed and throughput claims for large recurrent and integer-valued models, and no transformer MLM numbers ([arXiv:2511.16652](https://arxiv.org/abs/2511.16652)). A separate line uses ES for LLM fine-tuning without numbers in the read text ([arXiv:2509.24372](https://arxiv.org/abs/2509.24372)).

**Split forward gradient.** The Split-FG paper estimates the trunk gradient with a Jacobian-vector product and computes the head gradient exactly. Its WikiText-103 results are validation perplexity 387 for Split-FG against 150 for backprop, on a 16M GPT-2-style model ([arXiv:2607.16612](https://arxiv.org/abs/2607.16612)). This does not support a 10-minute MLM use.

**Activation-perturbation forward gradients.** Ren et al. report that perturbing activations rather than weights reduces forward-gradient variance, and they add local losses ([arXiv:2210.03310](https://arxiv.org/abs/2210.03310)). Baydin et al. introduce the forward gradient as an unbiased single-forward estimate ([arXiv:2202.08587](https://arxiv.org/abs/2202.08587)). Both support the idea that activation-space noise is the lower-variance choice. Neither gives transformer numbers in the read text.

### 5.3 Cost per update relative to a backprop step

A backprop step costs about one forward plus a backward pass. For a model with $L$ blocks, I take a backward pass as about two forwards, so the step is about $3L$ block-forward units. For $L = 36$ that is about 108 units. This is a convention, and a backward pass is often more expensive in practice.

**Dust.** Under the rerun rule, a draw for block $l$ costs $L - l + 1$ block forwards. Summed over blocks, one batched pass per block costs $\sum_{l=1}^{36} (37 - l) = 666$ units. The 8-block recipe uses 3 writer passes, 6 hidden passes and 6 hub passes per block, so $15 \times 666 \approx 9{,}990$ units per update, before the embedding's 24 full passes and the attention-internal chunks. That is roughly 90 times a backprop step. This uses the 8-block pass counts unchanged and assumes each batched pass costs one block forward per rerun block. It is an estimate from the code's counts, not a measurement, and it is the reason the full estimator does not fit the budget.

**Boost.** Each block's readouts are local, so there are no reruns through later blocks. The cost grows linearly with depth. The constant per block is large, because each block takes full direction cycles and attention draws. The code does not count it, and it must be piloted.

**Cut.** Each root reruns every later block for a full direction cycle. The cost grows roughly quadratically with depth, multiplied by the cycle length. The noise records use nine roots over eight blocks, and a 36-block model would need far more roots to match.

**Residual-scalar ES.** Two-sided ES on $2L = 72$ scalars costs a small number of full forwards per update. This is far below one backprop step.

### 5.4 Three hybrid recipes under the repository's rules

These recipes combine backprop with the backprop-free pieces. Each one uses the validation split for selection and reports test loss. The hour estimates are my engineering estimates, not measured values. The expected gains come from the cited evidence, and no recipe has been tested in this setting.

**Recipe 1: Boost-style per-block readout warm start, then backprop.** For the first $T_w$ seconds, train each block with its shared-head readout (section 3.2) and the Boost updates, then hand the weights and optimizer state to AdamW or ANVIL II and continue with backprop. Choose $T_w$ on a pilot, within the 600 s budget.

- Evidence: in the boost-cut setting, Boost alone stalls at 4.062 test loss, and Boost-to-Cut reaches 3.935 test loss over 3 seeds against backprop's 3.706 at equal model and optimizer, though the README gives no backprop wall time to compare ([README](https://github.com/spicylemonade/boost-cut)). The head start helped Cut: 3.945 val with the head start against 3.984 without it, the latter over 3 seeds ([README](https://github.com/spicylemonade/boost-cut)).
- Expected gain: uncertain. It is positive only if the warm-start weights beat a random initialization at the handoff, measured in wall-clock seconds on our GPU. The public evidence does not show this at matched time.
- Cost: 8 to 12 h. The handoff, optimizer-state transfer and per-block readouts for 36 blocks are the main work.
- Risk: high. The backprop-free phase is far below backprop in the one comparison published, and the depth cost of probes is not known.

**Recipe 2: backprop everywhere, with Boost-style estimated credit only for deep MLP hidden units.** Use backprop for embeddings, attention and early blocks. For the deep MLP hidden layers (for example blocks 25 to 36), replace the gradient with a Boost-style estimate computed from per-block readouts, restricted to the top 128 of 2,048 units per token. Check the cosine between the estimate and the backprop gradient on one batch before using it, following the Dust paper's cosine tuning ([paper page](https://qlabs.sh/research/dust)).

- Evidence: the Boost-to-Cut README reports that route restriction to 128 units costs 0.006 loss and makes the Cut phase 2.4x cheaper ([README](https://github.com/spicylemonade/boost-cut)). The cosine-tuning method is in the Dust paper ([paper page](https://qlabs.sh/research/dust)).
- Expected gain: unknown. It depends on whether the estimated gradient helps the deep blocks at all, and the cosine check decides that before any run.
- Cost: 10 to 15 h.
- Risk: moderate to high. Mixing two gradient sources can destabilize training, and the estimate may be biased for bidirectional attention.

**Recipe 3: backprop-free probes choose the units, backprop updates only the chosen units.** Every $N$ steps, run forward-only probes on one batch. Use Boost's hidden-unit step (section 3.4) to score each MLP unit. Keep the top $k$ units per block by credit magnitude, and update only those rows of `c_fc` and columns of `c_proj` with backprop. Freeze the rest. Compare with dense backprop and with a random-unit mask of the same size, so the selection rule is tested.

- Evidence: Boost's hidden step and Cut's route rule rank units this way ([run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py)). The Already Clean essay's small-hub observations support the intuition, though they are not a training result (section 4.2).
- Expected gain: unknown. Sparse updates lower the effective update size, so the gain could be negative. The random-mask control is the test.
- Cost: 3 to 5 h, forward-only probes plus a mask.
- Risk: low. The fallback is dense backprop.

A fourth option is cheap and optional: the residual-scalar ES of section 1.7 on the $2L$ mixing scalars, run beside backprop on the rest. It costs a few forwards per update and has the least risk.

---

## Ranked table

Evidence strength is my reading of the sources: strong means a published comparison on the benchmark, moderate means code or records without a clean ablation, weak means a single number, a name or an unverified claim. Cost in hours is an engineering estimate. Order is the suggested sequence for testing in this repository, subject to a pilot.

| Rank | Idea | Source | Evidence strength | Applies to our setting | Cost (h, estimate) | Risk | Suggested order |
|---|---|---|---|---|---|---|---|
| 1 | ANVIL II (or a similar polar matrix optimizer) for block matrices under backprop, Adam elsewhere | [noise README](https://github.com/qlabs-eng/noise) backprop baseline 3.340 vs 3.419 | Weak: one number on the 30M track, specification not public | Yes (backprop optimizer swap) | 2 to 4 | Low | 1 |
| 2 | Recipe 3: backprop-free hidden-unit probes select units; backprop on selected units only, with random-mask control | [boost-cut run.py](https://github.com/spicylemonade/boost-cut/blob/main/noise/train/boostcut/run.py) | Weak: code and route-rule ablation at 10M only | Adapted | 3 to 5 | Low | 2 |
| 3 | Residual-scalar two-sided ES (2L scalars) beside backprop | [Dust paper](https://qlabs.sh/research/dust), [dust.py](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py) | Moderate for mechanism, small payoff expected | Adapted | 2 to 3 | Low | 3 |
| 4 | Cosine check of Boost-style estimate against backprop on one batch (gate for recipe 2) | [Dust paper](https://qlabs.sh/research/dust) cosine tuning | Moderate: method in the paper, no bidirectional results | Adapted | 1 to 2 | Low | 4 |
| 5 | Forward readout of cross-token credit (exact, no derivative) | [noise README](https://github.com/qlabs-eng/noise) Track 1 #3 (3.338) vs #2 (3.409) | Moderate: record comparison, same track | Adapted: sum over all queries | 6 to 10 | Moderate | 5 |
| 6 | Recipe 2: backprop with Boost-style estimated credit on deep MLP units | [boost-cut README](https://github.com/spicylemonade/boost-cut) route rule | Weak: no bidirectional or backprop-mixed result | Adapted | 10 to 15 | Moderate to high | 6 |
| 7 | Recipe 1: Boost-style warm start, then backprop | [boost-cut README](https://github.com/spicylemonade/boost-cut) (Boost alone 4.062 test; Boost-to-Cut 3.935 vs backprop 3.706) | Weak: one comparison, backprop-free phase below backprop | Adapted | 8 to 12 | High | 7 |
| 8 | Hashed n-gram value embeddings | [noise README](https://github.com/qlabs-eng/noise) Track 3 #4 (3.358, 6 blocks) and #5 (3.211, 12 x 1024) | Weak: records on a different architecture and objective | Adapted (split-leak check needed) | 3 to 5 | Moderate | 8 |
| 9 | Cut-lite restricted routes (top-k MLP units) | [noise README](https://github.com/qlabs-eng/noise) Track 2 #2 and Track 3 #5 | Moderate: records, no clean ablation | Adapted | 10 to 15 | Moderate | 9 |
| 10 | Identity-path first phase and shaping mechanisms | [noise README](https://github.com/qlabs-eng/noise) Track 2 #2 to #4 | Moderate in records, mechanism partly not visible | Adapted | unknown | Moderate | 10 |
| 11 | Concentrated credit and small hubs as a prior for unit selection | [Already Clean essay](https://already-clean.goodfire.com/essay.md) | Qualitative only | Inferred hypothesis | 1 to 2 (analysis) | Low | 0 (informs 2) |
| 12 | Full Dust estimator on all weights | [Dust paper](https://qlabs.sh/research/dust) | Strong at 1M tokens (SGD); trails backprop at 10M and 20M | No for the full 36-block model in 10 minutes | 20 to 40 | High | Not in this budget |
| 13 | Cut chain credit through all blocks | [noise README](https://github.com/qlabs-eng/noise) records | Strong in its own setting (causal) | No without bidirectional rework | 30 or more | High | Not in this budget |
| 14 | Split forward gradient (forward-mode trunk, exact head) | [arXiv:2607.16612](https://arxiv.org/abs/2607.16612) | Weak: WikiText-103 perplexity 387 vs 150 | No for a 10-minute run | 10 or more | High | Not |
| 15 | Weight-space ES on all weights (EGGROLL) | [arXiv:2511.16652](https://arxiv.org/abs/2511.16652); [Dust paper](https://qlabs.sh/research/dust) comparison | Strong that it trails Dust at equal population in the Dust paper | No | not applicable | High | No |
| 16 | 2.3x speedup post (interpretability-inspired) | [X post](https://x.com/spicey_lemonade/status/2111636408540799141) | Not found: text unreachable | Unknown | unknown | unknown | Do not act until the text is read |

Verification status: the table restates results from the sources above. Items 1, 2 and 5 are the ones I would test first. None of the items has been run in this repository.
