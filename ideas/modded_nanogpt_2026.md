# modded-nanogpt records 57 to 92: what each technique is and what it implies for a 36x768 protein MLM

The modded-nanogpt speedrun trains a 124M-parameter GPT on FineWeb with causal next-token prediction, timed on 8 H100 GPUs ([README](https://github.com/KellerJordan/modded-nanogpt); [PR 360](https://github.com/KellerJordan/modded-nanogpt/pull/360) gives the same setting as "GPT-2 (124M-class)" on FineWeb10B). Records 57 to 92 ran from 2026-01-04 to 2026-08-30 and took the record time from 1.894 minutes (record 56) to 0.665 minutes (record 92) ([README record table](https://github.com/KellerJordan/modded-nanogpt)). Our benchmark differs in two ways that matter for transfer. The objective is bidirectional masked language modeling on protein sequences (15 percent of positions, MASK-only corruption, no causal mask, a 20-letter amino-acid vocabulary), so every causal structure in the records (causal masks, next-token targets, windows over past tokens) needs an explicit decision. And the run is one H100-class GPU for 10 minutes with a 36-block, 768-wide model (MLP width 2048), so the 8-GPU communication work, the sharded tables and the record step counts do not transfer directly. The record gains below are percentages to re-measure, not minutes to bank. The trainer read here has 11 layers and a 50,257-token vocabulary ([root train_gpt.py](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py)), so record layer indices do not map one to one onto 36 blocks.

Conventions in this file. "Gain" is the record's own change from the README table (before to after, in minutes), its percentage of the before time, and the same change in seconds on the 8-GPU basis (minutes times 60, derived here). Percentages and seconds are arithmetic on the table, not new measurements. "Cost" and "risk" are estimates written for this setting and are not from the sources. Where a PR page did not show the file list or an equation, the section says so. Sources were read through the PR pages, the record folders and the current track_1_short package; the package may be later than record 92, which the file notes where it matters.

## 57. Bfloat16 attention and MLP weights, mixed-precision interleaved optimizer

- Record 57, 2026-01-04. Time 1.894 to 1.878 min (0.016 min, 0.84%, about 1.0 s).
- What it is: attention and MLP weights are stored in bf16 instead of fp32. A bf16 parameter cannot hold small optimizer updates, so the optimizer keeps the low 16 bits of an fp32 master in a separate uint16 "mantissa" buffer. The PR reports that the bf16 update hurt loss without that buffer. Muon and Adam steps are interleaved so communication overlaps compute ([PR 190](https://github.com/KellerJordan/modded-nanogpt/pull/190); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-01-04_MixedPrecisionInterweavedOptimizer)).
- Where: PR 190. The same high-bits-in-parameter, low-bits-in-`mantissa` pattern appears in the current ANVIL code ([track_1_short/optim/anvil.py](https://github.com/KellerJordan/modded-nanogpt/blob/master/track_1_short/optim/anvil.py)).
- Transfer: yes, adapted. Keep bf16 weights with an fp32 master or mantissa buffer. The interleaving is a multi-GPU communication trick and does nothing on one GPU, so only the storage part applies. Check against an fp32-weight run on validation loss, because the failure is silent.
- Cost about 2 to 4 h. Risk medium.

## 58. Paired head attention

- Record 58, 2026-01-07. Time 1.878 to 1.820 min (0.058 min, 3.09%, about 3.5 s).
- What it is: heads are grouped in pairs. For each pair, the keys, queries and values of the two heads are interleaved along the sequence axis, so one attention call runs over length 2T with H/2 pair-heads. The PR gives the shapes: q and k viewed as (B, T, H/2, 2D), then after RoPE as (B, 2T, H/2, D); v reshaped to (B, 2T, H/2, D). The second head of each pair gets a staggered RoPE offset, which the author tested only in limited runs. The window size is unchanged, so a 128-token window covers only about 64 original positions ([PR 191](https://github.com/KellerJordan/modded-nanogpt/pull/191); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-01-07_PairedHeadAttention)).
- Where: the PR describes a custom `rotary()` for this module and a q/k reshape delayed until after rotary(). The file list is not shown on the PR page.
- Transfer: adapted. The PR's causal example (a query in one head sees keys in the other head only up to earlier positions) does not apply in bidirectional attention, where all pairs see all positions. Full attention cost changes: from the stated shapes, the score tensor per layer has (H/2)(2T)^2 = 2 H T^2 entries, twice the baseline. At protein lengths this may or may not be the bottleneck, so measure. The window logic does not apply.
- Cost about 4 to 6 h including a control run. Risk medium, because the attention doubling can cost more than the gain at short lengths.

## 59. Fused linear, ReLU and square MLP kernel

- Record 59, 2026-01-10. Time 1.820 to 1.781 min (0.039 min, 2.14%, about 2.3 s).
- What it is: the up-projection, ReLU and square run in one Triton kernel, `linear_relu_square_kernel`. The forward writes the two GEMM outputs (before and after the ReLU square) because the backward needs them. The down projection stays a separate GEMM ([PR 197](https://github.com/KellerJordan/modded-nanogpt/pull/197); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-01-10_FusedLinearReLUSquare)).
- Where: `FusedLinearReLUSquareFunction`, called from the MLP block in the trainer (per the PR's commit list). The PR's backward kernel is described only in general terms.
- Transfer: yes, only if the MLP uses ReLU squared. The record's MLP is wider (its MLP bank is padded to 3072 per [PR 200](https://github.com/KellerJordan/modded-nanogpt/pull/200)); ours is 2048, so the bandwidth saving is smaller. Profile with `torch.compile` first, since it may already fuse these ops.
- Cost about 4 to 8 h. Risk medium.

## 60. Fused softcapped multi-token cross entropy

- Record 60, 2026-01-16. Time 1.781 to 1.765 min (0.016 min, 0.90%, about 1.0 s).
- What it is: the logit softcap and the weighted multi-token cross entropy run inside one Triton kernel for forward and backward, avoiding intermediate tensors including a replicated target-logits tensor. The softcap is 23 times sigmoid((z + 5) / 7.5) in the current code. The kernel sums over future-offset targets with per-offset weights, skipping zero-weight or out-of-range offsets ([PR 199](https://github.com/KellerJordan/modded-nanogpt/pull/199); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-01-16_FusedSoftcappedEntropy)). The PR does not give the number of offsets or the weights, and validation uses the unfused path.
- Where: `FusedSoftcappedCrossEntropy` (autograd function), with the kernels in `train_gpt.py` at the time.
- Transfer: the bounded softcap is a reasonable option to test on the output logits. The kernel is not worth it: our output has about 25 classes, against 50,304 rows in the record, and the loss is needed only at the roughly 15 percent masked positions. The multi-token part has no next-token analogue here; see record 53 in the pre-period list.
- Cost 0 for the softcap option, the kernel is not worth building. Risk low.

## 61. Parameter banks, one optimizer, and a transposed LM head

- Record 61, 2026-01-18. Time 1.765 to 1.748 min (0.017 min, 0.96%, about 1.0 s).
- What it is: the attention QKVO matrices of all attention modules are stacked into one bank of shape (40, 768, 768), and the MLP matrices into a padded (12, 2, 3072) bank. A single `NorMuonAndAdam` optimizer reads a `param_table` with 13 entries for per-parameter settings. The value embeddings stay separate. The LM head is stored transposed as (768, V) to match the gradient layout. The input embedding keeps its shape, so the pair is untied in storage and re-tied by combining gradients and copying updates into both matrices. At the untie point (about two-thirds of training), the embedding's Adam momentum is synchronized with an all-reduce ([PR 200](https://github.com/KellerJordan/modded-nanogpt/pull/200); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-01-18_UnifiedOptimizers)).
- Where: the optimizer and bank code in `track_1_short/optim/` and `model/` in the current package; the PR's own files were `train_gpt.py` and `triton_kernels.py`.
- Transfer: banks, yes. Stacking the 36 layers' attention and MLP matrices lets Muon orthogonalize in batches with fewer kernel launches, which matters on one GPU. The transposed head does not apply to a 25-class head. Tie handling applies only if the benchmark ties embedding and output weights.
- Cost about 6 to 10 h. Risk medium.

## 62. Bigram hash embedding

- Record 62, 2026-01-19. Time 1.748 to 1.655 min (0.093 min, 5.32%, about 5.6 s). This is the largest single item in the period.
- What it is: a table of learned rows indexed by a hash of the (previous, current) token pair. The row is added to the residual stream at every layer with a learned per-layer scalar (initialized to 0.1), and the table is zero-initialized, so training starts from the baseline. The PR's hash is h_t = (36313 * x_t XOR 27191 * x_(t-1)) mod (R - 1) with R = 5 times the vocabulary, and position 0 maps to the reserved row R - 1. The forward is x = x + x0_lambda[i] * x0 + bigram_lambda[i] * x0_bigram before each block. The record's gain includes a partial-key-offset fix worth about 0.5 s ([PR 201](https://github.com/KellerJordan/modded-nanogpt/pull/201); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-01-19_BigramHashEmbedding)).
- Where: `bigram_embed` and the hash function `get_bigram_hash` in the trainer. Moved to the GPU in record 68.
- Transfer: yes, and more exact than the record. With S symbols (20 residues plus specials, about 25 in total), the exact pair table has S^2 rows, 625 for S = 25, about 0.48 million parameters at width 768. A dense table has no collisions, so no hash and no sign trick (record 83) are needed. The trigram table is S^3 = 15,625 rows, still dense. Under MASK-only corruption, a masked position's input is MASK, so its bigram carries only the unmasked neighbor. The record uses the left neighbor only; for bidirectional attention, add a right-neighbor channel (x_t, x_(t+1)) as a second table or a second lookup. No target leaks through this, because the input at a masked position is always MASK.
- Cost about 2 to 3 h. Risk low.

## 63. Untied value embeddings

- Record 63, 2026-01-26. Time 1.655 to 1.650 min (0.005 min, 0.30%, about 0.3 s).
- What it is: value embeddings are token-indexed tables mixed into attention values. The PR changes the layer-to-table pattern from one shared pattern to distinct tables, written in the PR as VE[.12...012] changed to VE[.01...345]. The page does not define the notation. The author found no loss change when value embeddings were applied to all layers from the first ([PR 209](https://github.com/KellerJordan/modded-nanogpt/pull/209); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-01-26-UntieValueEmbeddings)). Value embeddings themselves date to record 14 (2024) in the README table.
- Where: `value_embeds` in the trainer.
- Transfer: adapted. Each table is about 25 rows times 768 columns, so the cost is negligible. The layer pattern is tied to the 11-layer record and must be searched again for 36 blocks. The gain here is small.
- Cost 1 to 2 h. Risk low.

## 64. Nonzero initialization of attention V and O

- Record 64, 2026-01-30. Time 1.650 to 1.630 min (0.020 min, 1.21%, about 1.2 s).
- What it is: per head, V is initialized with +sqrt(beta) I on the head's diagonal block and O with -sqrt(beta) I, with beta = 0.05, plus Gaussian noise (std 0.01) for symmetry breaking. The product V O^T is then about -beta I per head. The PR names this as a mimetic initialization. A beta of 0.10 or more destabilized training in the author's runs ([PR 214](https://github.com/KellerJordan/modded-nanogpt/pull/214); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-01-30_MimeticValueOutput)). The PR's title reports 20 fewer steps and 1.16 s; a maintainer note gives 97.8 s at merge.
- Where: the attention weight initialization in the trainer; the PR lists commit 09214665 and no file paths.
- Transfer: yes, adapted. The record has 11 layers. With 36 residual blocks the sum of block outputs is larger, so test beta in a small grid (for example 0.02 and 0.05) and check stability in the first few hundred steps.
- Cost 1 to 2 h. Risk low to medium.

## 65. Value embeddings grouped into one parameter

- Record 65, 2026-01-30. Time 1.630 to 1.613 min (0.017 min, 1.04%, about 1.0 s).
- What it is: five `nn.Embedding` modules become one parameter of shape (5 times V, 768). The weight is viewed as (5, V, 768), and one indexed gather returns all five value-embedding lookups, replacing five separate lookups ([PR 215](https://github.com/KellerJordan/modded-nanogpt/pull/215); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-01-30_VeFused)). A maintainer's microbenchmark (in the PR thread) measured about 3.0 ms against 2.3 ms per forward and backward on H100, about 1 s over 1,500 steps.
- Where: `value_embeds` in the trainer.
- Transfer: yes. Trivial to apply; on one GPU the saving is in lookups and optimizer parameter count.
- Cost 1 h. Risk low.

## 66. PyTorch 2.10

- Record 66, 2026-01-31. Time 1.613 to 1.595 min (0.018 min, 1.12%, about 1.1 s). The README lists no folder or PR for this record. Details: not found.
- What it is: a framework upgrade, not a technique. The README table gives the record's gain.
- Transfer: not a technique. Use the benchmark's PyTorch 2.14 and re-measure the stack; the records' own Dockerfile and pins are the reference for the records' environment ([README](https://github.com/KellerJordan/modded-nanogpt)).
- Cost 0. Risk low.

## 67. Tuned softcap kernels and FP8 quantization of the LM head

- Record 67, 2026-01-31 (PR merged 2026-02-09). Time 1.595 to 1.540 min (0.055 min, 3.45%, about 3.3 s).
- What it is: the softcap kernels are tuned (num_warps 2), and the LM head is run with FP8 quantization fused into the backward ([PR 207](https://github.com/KellerJordan/modded-nanogpt/pull/207); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-01-24_ImprovedLMHead)). The author found that removing the softcap made training diverge, and a bf16 head avoided divergence but left validation loss above 3.28 after 100 extra steps. The FP8 scale format is not stated on the page.
- Where: LM head kernels in the trainer.
- Transfer: no for the FP8 head. With about 25 output classes, and loss on masked positions only, the head is a negligible part of compute. Keep the softcap as an option (see record 54 in the pre-period list), not the FP8 path.
- Cost not applicable. Risk not applicable.

## 68. Bigram index computed on the GPU

- Record 68, 2026-01-31. Time 1.540 to 1.535 min (0.005 min, 0.32%, about 0.3 s).
- What it is: the bigram hash moves from the CPU data loader to the GPU, removing one host-to-device copy per batch and one tensor from the asynchronous loader pipeline ([PR 216](https://github.com/KellerJordan/modded-nanogpt/pull/216); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-01-31-BigramHashH2D)).
- Transfer: yes. With a dense pair table (record 62), the index is a device gather, so no CPU hash is needed.
- Cost 0.5 h. Risk low.

## 69. Kernel tuning for the Muon XXT kernel and the softcapped entropy

- Record 69, 2026-02-02. Time 1.535 to 1.528 min (0.007 min, 0.46%, about 0.4 s).
- What it is: the XXT and `ba_plus_cAA` kernels switch to coalesced loads with an in-register transpose (`tl.trans`), `num_warps` for those configs goes from 4 to 8, and `softcapped_entropy` replaces inner-loop divisions with precomputed multiplications ([PR 217](https://github.com/KellerJordan/modded-nanogpt/pull/217); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-02-02_KernelTuning)). The PR title claims 1.6 s; the maintainer's merge measurement was about 0.4 s relative to the then-current trainer.
- Transfer: adapted for the XXT (Gram-matrix) kernel used in any Muon-style orthogonalization. The entropy kernel does not apply.
- Cost 2 to 4 h with profiling. Risk low.

## 70. Value-embedding layout and ve_gates

- Record 70, 2026-02-03. Time 1.528 to 1.521 min (0.007 min, 0.46%, about 0.4 s).
- What it is: the first value embedding moves to a later block, on the author's analysis that the first one is ill-conditioned. All value embeddings are initialized from Normal(0, 0.01). The value-embedding gate changes from g(x[:12]) to g(x[:6] + ve[:6]). Pulling the last value embedding back instead was worse (loss 3.2986 against 3.2793) ([PR 218](https://github.com/KellerJordan/modded-nanogpt/pull/218); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-02-03_VeTuned)).
- Transfer: adapted. The block indices are tied to the 11-layer model. Re-sweep which blocks get value embeddings for 36 blocks; a small sweep is cheap.
- Cost 2 h. Risk low.

## 71. Sparse bigram gradient communication

- Record 71, 2026-02-06. Time 1.521 to 1.516 min (0.005 min, 0.33%, about 0.3 s).
- What it is: bigram-embedding gradients are communicated sparsely across GPUs, sending only touched rows and reconstituting the rank-local gradient with extra compute that overlaps other communication ([PR 221](https://github.com/KellerJordan/modded-nanogpt/pull/221); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-02-06_SparseBigramGradient)). The PR points to PR 219 for the algorithm details, which were not read.
- Transfer: no on one GPU. The principle reappears in record 92 (updating only touched rows of the hashed table), and only matters for large tables.
- Cost not applicable. Risk not applicable.

## 72. Max sequence length schedule and a higher minimum learning rate

- Record 72, 2026-02-10. Time 1.516 to 1.496 min (0.020 min, 1.32%, about 1.2 s).
- What it is: the sequence length starts at 896 for the early steps and moves to 2048 later, and the minimum learning rate rises from 0.1 to 0.15, with the late decay flattening at 0.15 ([PR 224](https://github.com/KellerJordan/modded-nanogpt/pull/224); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-02-10_ShortWindow)). The maintainer reverted one short-window stage and cut the step count from 1547 to 1530, and credited about 1.2 s to the first two changes.
- Where: the training schedule (`schedule.py` in the current package).
- Transfer: adapted. For proteins, use a length curriculum over buckets (early steps on crops of a fixed length, later on full sequences). A crop changes what the model sees, so document it and never change the test set's lengths. Test the 0.10 and 0.15 floors at the benchmark's step count.
- Cost 2 to 3 h. Risk low to medium.

## 73. Partitioned hyperconnections

- Record 73, 2026-02-12. Time 1.496 to 1.485 min (0.011 min, 0.74%, about 0.7 s).
- What it is: the residual stream is split into two lanes, lane0 and lane1, used from block 7 onward because extra lanes add memory traffic. Each sublayer has scalars rather than full mixing matrices: resid_lambda (initialized to sqrt(1.1)) and post weights, stored as post_lambdas with shape [num_layers, 2, 2] indexed by layer, attention or MLP, and lane. Attention reads norm(lane0), the MLP reads norm(lane1), and each sublayer output is added to both lanes: lane = resid_lambda * lane + post_weight * layer_out ([PR 230](https://github.com/KellerJordan/modded-nanogpt/pull/230); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-02-12_ParallelResiduals)).
- Where: the GPT forward in the trainer. Replaced in record 77.
- Transfer: adapted. Where the lanes start is a depth choice to tune for 36 blocks; the record starts at block 7 of 11. Record 77 replaced this design, so prefer record 77 unless it fails.
- Cost 4 to 6 h. Risk medium.

## 74. Flattened GPT forward

- Record 74, 2026-02-16. Time 1.485 to 1.468 min (0.017 min, 1.14%, about 1.0 s).
- What it is: the Block and MLP modules and the module list are removed, and GPT holds its state directly, which the parameter banks (record 61) allow. Post-attention lambdas on single-lane layers are removed, and the x0 and bigram injection moves before attention. Two transpose kernels (transpose copy and transpose add) replace element-wise copies in the optimizer for the LM head and embedding, freeing SMs for communication overlap ([PR 233](https://github.com/KellerJordan/modded-nanogpt/pull/233); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-02-16_FlattenForward)). The PR's initial comparison was 89.80 s against 88.83 s over four runs; the refined version was 88.70 s against 89.73 s.
- Transfer: yes, in code structure. A flat loop over banks is easier to compile and to capture with CUDA graphs (record 92). The transpose kernels are specific to the record's head layout.
- Cost 4 to 8 h. Risk low.

## 75. Cross-entropy kernel optimizations

- Record 75, 2026-02-23. Time 1.468 to 1.453 min (0.015 min, 1.02%, about 0.9 s).
- What it is: kernel-level changes to `fused_softcapped_entropy`. The fetched PR page does not describe the code changes; the author kept only the kernel work after reverting earlier learning-rate and sequence-length tweaks ([PR 235](https://github.com/KellerJordan/modded-nanogpt/pull/235); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-02-23_CrossEntropyKernel)).
- Transfer: no. The cross-entropy kernel is not the bottleneck at about 25 classes.
- Cost not applicable. Risk not applicable.

## 76. Backward transpose-copy kernel

- Record 76, 2026-02-28. Time 1.453 to 1.446 min (0.007 min, 0.48%, about 0.4 s).
- What it is: a Triton tiled transpose kernel, `transpose_copy`, replaces `.T.contiguous()` in the FP8 weight-gradient matmul of the LM head backward. The tile moved from 32x32 to 64x128 with num_warps 8. Offsets are cast to int64 because the largest matrices (49,152 by 50,304) overflow int32 addressing ([PR 240](https://github.com/KellerJordan/modded-nanogpt/pull/240); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-02-28_TransposeCopyBackward)).
- Transfer: no, unless the benchmark uses an FP8 output head. The int64 indexing rule applies to any tensor with more than 2^31 elements.
- Cost 0 at our scale. Risk low.

## 77. A saved activation replaces partitioned hyperconnections

- Record 77, 2026-03-06. Time 1.446 to 1.435 min (0.011 min, 0.76%, about 0.7 s).
- What it is: the author found that the last attention layers largely ignored the contributions of the last MLPs. The residual stream at block index num_layers - 4 is saved as `x_backout`. The last three attention layers take it as input, and after the loop the trainer subtracts a scaled copy of it, x = x - backout_lambda * x_backout, before the head. The code comment reads that the last three attention layers query the saved context ([PR 241](https://github.com/KellerJordan/modded-nanogpt/pull/241); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-03-06_SimplifyHC)). The description calls the saved layer 7 for the 11-layer model. The PR title claims about 0.5 s; the merge comment gives -0.7 s at 86.1 s.
- Where: the block loop and the pre-head step in the GPT forward.
- Transfer: adapted. The saved context is a whole-sequence representation, so bidirectional attention poses no problem. The save index for 36 blocks is a tuning choice. This replaces record 73 with fewer parameters and simpler code.
- Cost about 2 h. Risk low to medium.

## 78. Tighter bounds on document count for variable-length attention

- Record 78, 2026-03-22. Time 1.435 to 1.426 min (0.009 min, 0.63%, about 0.5 s).
- What it is: the fixed size of the cumulative document-offset tensor (`cu_seqlens`) is set from measured maxima per batch-size stage. The PR counts documents in the first ten FineWeb shards for each stage: observed maxima 56, 90 and 121, with bounds 64, 96 and 128. Each batch stage has its own compiled graph, so each can use its own size. The same PR lowers Muon's second beta from 0.95 to 0.9, which the author reports slightly improved loss ([PR 246](https://github.com/KellerJordan/modded-nanogpt/pull/246); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-03-22_VarlenMaxDocs)). The record has a round-up fallback for unlisted batch sizes.
- Transfer: adapted, and only if sequences are packed several per row. Then the offset tensor needs a bound measured on the training split, with an assertion when the bound is exceeded rather than a silent truncation. Flex attention document masks are the analogue of the variable-length call here; this is not measured.
- Cost 2 to 3 h. Risk low, provided overflow asserts.

## 79. Fused cross entropy forward and backward

- Record 79, 2026-04-04. Time 1.426 to 1.411 min (0.015 min, 1.05%, about 0.9 s).
- What it is: forward and backward of the cross entropy with softcap are fused into one kernel, avoiding recomputation of the softcap sigmoid in the backward ([PR 251](https://github.com/KellerJordan/modded-nanogpt/pull/251); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-04-04_FuseCEFwdAndBwd)). The PR page does not give the logsumexp or gradient computation.
- Transfer: no, for the same reason as record 75.
- Cost not applicable. Risk not applicable.

## 80. Muon orthogonalizes Q and K in head pairs

- Record 80, 2026-04-08. Time 1.411 to 1.406 min (0.005 min, 0.35%, about 0.3 s).
- What it is: Muon's polar step is applied to per-head-pair groups instead of the whole (hdim, hdim) matrix, so each pair gets its own orthogonalization. K is split into per-pair groups. The PR body and README say K and V; a maintainer comment says the code applies the grouping to K and Q. The author's ablation found pairs 0 and 3, 1 and 4, 2 and 5 slightly better than adjacent pairs, and the paired-head motivation was dropped ([PR 253](https://github.com/KellerJordan/modded-nanogpt/pull/253); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-04-08_PairedHeadMuon)). The PR's timing comparisons use different step counts (1480 to 1490), so treat the gain as small and unconfirmed.
- Transfer: adapted. Needs an even head count and a choice of pairing. Test both adjacent and strided pairings.
- Cost 2 to 3 h. Risk low.

## 81. MUDD skip connections

- Record 81, 2026-04-22. Time 1.406 to 1.363 min (0.043 min, 3.06%, about 2.6 s).
- What it is: MUDD is a dynamic dense connection. For each target layer a small shared MLP reads the current hidden state (layer 9's post-MLP output in the PR) through `dense_w1` (projection to width 64), and `dense_w2` plus a bias `dense_bs` produce per-token coefficients of shape (B, T, C, L). C = 2 streams: one added to attention values (`v_mudd`) and one to the residual (`residual_mudd`). L is the number of source hidden states (the embedding x0, earlier layer outputs, value embeddings and the skip connection). The coefficients are zero-initialized with biases set to reproduce the earlier static scalars, so training starts from the baseline. `v_mudd` is a weighted sum of shape (B, T, D) reshaped to (B, T, H, D_head) and added after the QKV matmul, with no extra projection ([PR 259](https://github.com/KellerJordan/modded-nanogpt/pull/259); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-04-22_MuddFormer)). The record has v2 and v3 variants in subfolders.
- Where: `_init_mudd` and `CausalSelfAttention.forward` in the trainer.
- Transfer: adapted. Nothing in MUDD is causal: the coefficients depend on the current position's own hidden state, not on future or past tokens. The number of sources L grows with depth, so for 36 blocks the coefficient tensor per layer is (B, T, 2, L). Cheap, but the source set must be chosen.
- Cost 6 to 10 h. Risk medium to high.

## 82. Learnable exclusive self-attention

- Record 82, 2026-04-29. Time 1.363 to 1.353 min (0.010 min, 0.73%, about 0.6 s).
- What it is: XSA removes from each attention output y the part along the token's own value v. With v-hat = v / ||v||, the update is y <- y - tanh(alpha) * <y, v-hat> * v-hat, so the residual receives only information from other tokens. alpha is a learned scalar per (layer, head), zero-initialized, so the model matches the baseline exactly at step 0. The record applies it only to non-paired attention layers {1, 3, 4, 7, 8, 10}, because the paired layers' projections do not align with the value reshaping. Overhead is about 0.6 ms per step ([PR 264](https://github.com/KellerJordan/modded-nanogpt/pull/264); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-04-29_XSAGatedLayers)). A maintainer noted there are 6 heads per layer, not the 12 the PR text states.
- Transfer: yes. In bidirectional attention a token's output includes its own value, and XSA removes that self-term. At a masked position the self-value comes from the MASK embedding, so the removed term is a fixed function of the MASK embedding and the layer; this is a hypothesis to test, not a result.
- Cost 2 to 3 h. Risk low. Use the algebraic form in record 86.

## 83. Sign trick on the bigram embedding

- Record 83, 2026-05-20. Time 1.353 to 1.328 min (0.025 min, 1.85%, about 1.5 s).
- What it is: a fixed random +/-1 pattern s_j is applied to the bigram embedding g_j in the forward (g_j * s_j), and the same pattern recovers it on read. The aim is to let colliding bigrams share hashed rows with less interference. The table changed from (5V, 768) to (15V, 192), and the author attributes about 0.5 s to the sign and about 0.5 s to the smaller width ([PR 299](https://github.com/KellerJordan/modded-nanogpt/pull/299); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-05-20_BigramsSignTrick)). A reviewer questioned whether a shared row can recover the optimal embedding, since it has only 768 degrees of freedom.
- Transfer: no with an exact dense pair table (record 62), because it addresses hash collisions and an exact table has none. Use only if the benchmark moves to hashed higher-order n-grams. The smaller width is a memory choice, separate from the sign.
- Cost not applicable. Risk not applicable.

## 84. FP8 for the MLP up-projection forward

- Record 84, 2026-05-21. Time 1.328 to 1.320 min (0.008 min, 0.60%, about 0.5 s).
- What it is: the MLP up-projection forward runs in FP8 with per-tensor scaling. Weights are quantized once per optimizer step and activations once per forward. Per-row scaling was tried and regressed. The PR reports a final-loss regression of 0.00082, offset by five extra steps. The FP8 format and whether scaling is dynamic or delayed are not stated ([PR 306](https://github.com/KellerJordan/modded-nanogpt/pull/306); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-05-19_FP8MLPUpProj)).
- Transfer: adapted. The target is H100-class (Hopper), which has FP8 matrix units. The record gain is small, and the loss cost is of the same order, so measure on validation loss at the benchmark's step count.
- Cost 6 to 10 h including checks. Risk medium to high.

## 85. MUDD gates and lightweight dynamically composable attention

- Record 85, 2026-05-27. Time 1.320 to 1.271 min (0.049 min, 3.71%, about 2.9 s).
- What it is: MUDD-generated gates control the XSA gates, the attention gates, the x0 and bigram injections, and a layer-3 skip path. In the PR's v2 layout, a "pre" gate covers early XSA, attention and injection coefficients, and a "post" gate generated at layer 4 covers later XSA, attention and injection coefficients and the layer-10 composition. Layer 10 has a post-only dynamic composition variant of DCMHA (arXiv 2405.08553, as cited in the PR): FlashAttention computes the base attention and a custom Triton correction then mixes per-token, per-head tensors emitted by the post gate. Version history: v1 used standalone banks, v2 removed them, and v3 rebased onto the newer base with a learnable gate scale initialized at 0.1 because of training instability ([PR 315](https://github.com/KellerJordan/modded-nanogpt/pull/315); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-05-27-MuddGatedAndDC)). The PR text gives -3.51 s in the title and the merge comment gives -2.9 s; the fetched page does not give the final step count.
- Transfer: adapted in principle. The gate layout is record-specific and fragile. Build after record 81 works, and treat the gate schedule as a sweep.
- Cost 2 to 3 days. Risk high.

## 86. Algebraic rewrite of learnable XSA

- Record 86, 2026-05-27. Time 1.271 to 1.266 min (0.005 min, 0.39%, about 0.3 s). The README lists no folder; the PR is [PR 317](https://github.com/KellerJordan/modded-nanogpt/pull/317).
- What it is: the old code normalized v and then computed the projection. The new code computes dot = <y, v> and denom = ||v||^2 (clamped at 1e-8), and sets y <- y - alpha * (dot / denom) * v. This is the same operation, since v-hat = v / ||v||, and it removes a normalize and a multiply, which helps the compiler lower it. The epsilon handling differs (the normalize used eps 1e-4 on the norm, the rewrite clamps the squared norm at 1e-8), so near-zero vectors can differ, and rounding can differ slightly ([PR 317](https://github.com/KellerJordan/modded-nanogpt/pull/317)). A later comment says only this rewrite was merged into master.
- Transfer: yes. Use this form directly with record 82.
- Cost 0.5 h. Risk low.

## 87. Faster ReLU-squared kernel

- Record 87, 2026-06-11. Time 1.266 to 1.256 min (0.010 min, 0.79%, about 0.6 s).
- What it is: a Triton ReLU-squared MLP kernel that stores post = relu(pre)^2 and reconstructs the derivative in backward: the derivative of relu(pre)^2 is 2 relu(pre) = 2 sqrt(post), so the pre-activation need not be stored ([PR 322](https://github.com/KellerJordan/modded-nanogpt/pull/322); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-06-11_RecursiveFromBest)). The maintainer reports about 0.6 s; the PR's own benchmark (80.61 s to 77.34 s) combines five changes.
- Transfer: yes, if the MLP is ReLU squared. The choice of activation is free in our spec.
- Cost 3 to 5 h. Risk low to medium.

## 88. Prefix token prediction auxiliary loss

- Record 88, 2026-07-13. Time 1.256 to 1.243 min (0.013 min, 1.04%, about 0.8 s).
- What it is: an auxiliary target added to the cross entropy. For each vocabulary token, the target is its longest byte-prefix token, given as a fixed table of length V, built by sorting the vocabulary and keeping a stack of ancestor prefixes. The end-of-text token maps to -1 (no target). The PR says it stacks with multi-token prediction and that replacing multi-token prediction with it was worse. No extra head or learned parameter is mentioned, and no loss weight is given ([PR 337](https://github.com/KellerJordan/modded-nanogpt/pull/337); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-07-13_PrefixTokenPrediction)).
- Transfer: no as stated. The target is defined on a byte-level BPE vocabulary, and single-letter residue codes have no prefix structure. A proposed analogue, not from the record and untested: an auxiliary head at masked positions that predicts a residue class (for example hydrophobic, polar, charged or special). Label it as a proposal in any plan.
- Cost 3 to 4 h if tried. Risk medium.

## 89. FP8 MLP down-projection with delayed scaling

- Record 89, 2026-07-17. Time 1.243 to 1.230 min (0.013 min, 1.05%, about 0.8 s).
- What it is: the down-projection forward runs through `torch._scaled_mm` in FP8, and the backward stays in bf16. Scaling is delayed: the fused up-projection and ReLU-squared kernel emits FP8 activations and partial amax values, which are reduced to set the activation scale. Down-projection weights are quantized and transposed to FP8 with delayed scales. The format and the amax history length are not stated ([PR 342](https://github.com/KellerJordan/modded-nanogpt/pull/342); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-07-17_FP8DownProjection)).
- Transfer: adapted, as in record 84, with the delayed-scale bookkeeping added.
- Cost 6 to 10 h. Risk medium.

## 90. Reduced-width QK, packed FP8 attention, and a fused QK kernel

- Record 90, 2026-08-03. Time 1.230 to 1.130 min (0.100 min, 8.13%, about 6.0 s). This is the largest item after record 92.
- What it is: the Q and K head width drops from 128 to 96, while V and the residual stream stay at 128. Q, K and V are projected in one packed FP8 operation. One Triton kernel does QK normalization, RoPE, the key offset, FlashAttention padding and layout conversion. Row-major and transposed FP8 weight copies are cached once per optimizer update from a single weight read. All four MLP GEMMs (forward and backward) run in FP8, and relu(pre) is rebuilt from the stored square. The schedule adds 45 final-learning-rate iterations ([PR 344](https://github.com/KellerJordan/modded-nanogpt/pull/344); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-08-03_FP8MLPBackwardPackedQKV)). On one node run back to back, the PR measured 73.897 s for the baseline and 70.094 s for the candidate; a maintainer note gives 68.0 s at merge after the step count was cut to 1,295.
- Where: `train_gpt.py`, `triton_kernels.py`, and the record's README. The PR page lists these files but not the line-level diff.
- Transfer: partly. Reduced QK width is transferable: attention cost falls with the QK dimension, and the quality check is the benchmark's validation loss. Test 96 against 128 at 36 blocks, since our head layout is free. The FP8 parts follow records 84 and 89. The fused QK kernel depends on the attention path: flex attention is available, so a custom kernel is only worth it if profiling shows the QK step dominates.
- Cost 1 to 2 weeks for the full package, less for the QK width alone. Risk high.

## 91. Logit masking for infeasible continuations during validation

- Record 91. The README lists 2026-08-06 and the folder `2026-08-06-CanonicalMasking`; the PR page shows a merge date of 2026-09-18. Time 1.130 to 1.126 min (0.004 min, 0.35%, about 0.2 s). [PR 350](https://github.com/KellerJordan/modded-nanogpt/pull/350); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-08-06-CanonicalMasking).
- What it is: non-canonical output tokens are masked during validation only. The table is built before training (about 3.15 s of CPU work). The training objective is unchanged. The author's motivation is that the multi-token loss pushes probability mass toward non-canonical tokens. Validation loss falls by about 0.0017 averaged over 20 runs on one H100 ([PR 350](https://github.com/KellerJordan/modded-nanogpt/pull/350)). The PR does not define canonical tokens beyond that motivation.
- Transfer: adapted as an evaluation rule. At masked positions, restrict the softmax to the 20 standard residues, and decide whether the benchmark's definition excludes X, B, Z, U and O. This changes the metric, so report the unrestricted test loss as well, and do not present the restricted number as the benchmark loss without saying so.
- Cost 1 h. Risk low for training, high for comparability if not labeled.

## 92. Summary of the 0.665-minute record, and what it costs

- Record 92, 2026-08-30. Time 1.126 to 0.665 min (0.461 min, 40.9%, about 27.7 s). [PR 360](https://github.com/KellerJordan/modded-nanogpt/pull/360); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-08-30_ANVIL2).
- What it changed, per the README bullets and the PR's ablation list: sampled softmax, full FP8 MLP forward and backward with an FP8 LM-head cache, a hashed bigram and trigram table of 84,602,880 rows, the ANVIL optimizer, depth reduction, mixed-width QK attention, full CUDA-graph capture, and tail weight averaging with a raised learning-rate floor ([README](https://github.com/KellerJordan/modded-nanogpt)). The PR's ablation lists the costs of removing each piece: sampled softmax +8.40 s, the n-gram table +7.05 s, full FP8 +4.96 s, mixed-width attention +3.82 s, ANVIL +5.08 s in total, CUDA-graph training step +2.99 s, optimizer-update graph +0.92 s, sparse gradient communication and data loader +1.20 s, and a MUDD residual mix +0.48 s. The total is +33.98 s ([folder README](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-08-30_ANVIL2)).
- The 0.665 figure: the PR reports 39.914 s (plus or minus 0.120 s over 18 runs) for the training section only, against 73.889 s for record 89 measured on the same machine. That is a 46.0 percent difference on that machine. The README's 1.126 reference corresponds to about 67.6 s on the official basis, so the 40.9 percent in the table is not a like-for-like official comparison. The PR's measured delta is the more reliable comparison.
- What it costs: the PR changes 9 files with +5,068 and -2,465 lines by its own count, removes one Triton file, and ships a 26-run log set in the record folder. The n-gram table is 84,602,880 rows (the README gives 6.50e10 parameters); if the rows are 768 wide, that is about 65 billion entries, the figure commenters cite. The trainer shards it across 8 GPUs, about 16 GB per rank. The stack is pinned to torch 2.10.0 with CUDA 12.8, a driver at least 580, and 8 H100-80GB GPUs, and it uses a patched FlashAttention 3 kernel with no license information in the README.
- Transfer to our setting: only a subset. The table, the 8-GPU sharding and the 65 billion entries do not transfer. A dense exact table over the amino-acid vocabulary (see record 62) is the right replacement. The rest is covered in the sections below: sampled softmax (no, a 25-class head needs no sampling), ANVIL (see below), full FP8 MLP (see records 84 and 89), mixed-width QK (see below), CUDA-graph capture (see below), tail averaging (see below), and depth reduction (the benchmark fixes 36 blocks, so it does not apply).

### 92a. ANVIL optimizer

- What it is, from the code: per bank, the state holds two momentum rails in fp32 (a fast rail and a slow rail), a per-lane energy EMA, and a uint16 mantissa buffer. The fast rail uses beta 0.85 from step 514 and the slow rail beta 0.98. A blend weight of 0.4385 combines them into B = w v_fast + (1 - w) v_slow. The gradient is lookahead-blended, g = G + m (B - G). Then the update is orthogonalized by a cascade of six quintic maps after normalization: d = 1.05 sqrt(trace(A)) + 1e-6 with A = X^T X for tall matrices and X X^T for wide ones; X <- X / d; for each map k with coefficients (a, b, c), B_k = b A + c A^2 and X <- a X + X B_k (tall) or a X + B_k X (wide). A per-lane energy equalizer rescales the output so each lane's energy matches a running estimate. The decay is cautious: decay applies only where the slow rail and the parameter agree in sign, P <- P - P * aligned * eff_wd * eff_lr - U * eff_lr, with aligned = (slow_rail * P >= 0). Sources: [track_1_short/optim/anvil.py](https://github.com/KellerJordan/modded-nanogpt/blob/master/track_1_short/optim/anvil.py).
- Not found: the quintic coefficients, the momentum default, the learning-rate and weight-decay defaults, and the pre-step-514 fast-beta schedule (`get_rail_beta` is referenced but not in the read content). The file's own docstrings disagree in two places: the momentum field is described as the scheduled rail beta but filled with the momentum default, and the equalizer's reduction axis is described as the longer dimension while the code reduces over the shorter one for tall matrices. Use the code, not the docstrings, and confirm the code against the record's own trainer before porting.
- Record claim: about 21 millinats better than record 89's optimizer at roughly equal wall clock ([folder README](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-08-30_ANVIL2)). The ablation is +5.08 s, of which +4.16 s is the update rule and anneal and +0.92 s the optimizer-update graph capture.
- Transfer: the rails, the cautious decay and the energy equalizer are generic. The cascade replaces the Newton-Schulz or Polar Express step in Muon. Nothing in the optimizer depends on causality. Port it as a drop-in for the matrix parameters and keep Adam for embeddings and scalars.
- Cost 1 to 2 days to port faithfully and check against a Muon baseline on validation loss. Risk high (many interacting constants, and the record's own variant may be a later master).

### 92b. Sampled softmax during training

- What it is, from the code ([sampled_softmax.py](https://github.com/KellerJordan/modded-nanogpt/blob/master/track_1_short/sampled_softmax.py)): each step has a candidate set shared by every token in the rank's microbatch. It contains every target class in the microbatch (about 11,000 distinct targets at batch 24, per the file) plus negatives. Negatives come from a deterministic permutation of the vocabulary, sweep = (20011 * arange(V)) mod V, with 20011 coprime to V, and each rank starts at (V * rank) // world size. No random number generator is used. Candidate counts P are 10,240 in stages 0 and 1, and 14,336, 14,336 and 24,576 in thirds of stage 2. Full softmax is used in the stage after those, and in validation. The training loss is a cross entropy over the P candidate logits; the normalizer omits the non-candidate mass, so the loss is biased, and the lm_head gradient is dense with zeros outside the candidate set. Prefix targets not in the candidate set are dropped for that token.
- Not found: a log-Q correction and any softcap inside the sampled loss.
- Record claim: ablation +8.40 s and -2.4 millinats validation; the README says the sampled loss biases the training gradient.
- Transfer: no. The record's vocabulary is 50,257 tokens; ours is about 25. The full softmax costs little at that size, so the bias buys nothing.
- Cost not applicable. Risk not applicable.

### 92c. Tail weight averaging

- What it is, from the code ([tail_average.py](https://github.com/KellerJordan/modded-nanogpt/blob/master/track_1_short/tail_average.py), whose comments cite record 360): four fp32 accumulators follow the final few hundred steps (windows of 298, 250, 250 and 298 steps), each sharded across ranks. Each accumulator is updated with an exponential-style lerp rather than a uniform average. For example, the embedding and lm_head average uses a per-odd-step rate of 2/299. At the end, weights are shipped with blends: the embedding and lm_head are blended 0.65 toward their average, some banks are replaced, and the query-key bank is blended 0.55 toward its average. Norms of the shipped matrices are then rescaled to their pre-ship values. Validation runs on the shipped weights.
- Not found: the final checkpoint rule (what is saved), and any interaction with a learning-rate floor in this file. The PR's ablation list does not include tail averaging, so the record that introduced it is not confirmed here. The README lists "tail weight averaging (EMA) with a raised learning-rate floor" with no link. The learning-rate floor change itself is record 72.
- Transfer: adapted, for the checkpoint decision. The benchmark selects weights by validation loss and reports test loss, so compare the raw final weights against the averaged weights on validation, pick one, and report its test loss. Do not ship the averaged weights without that comparison. The gain on the record is not separable from the other changes.
- Cost 4 to 6 h. Risk medium.

### 92d. Hashed bigram and trigram table

- What it is, from the README and the PR's comments: 84,602,880 rows across a bigram channel and a trigram channel, expanded from 377,280 rows in record 89. The bigram hash is the record 89 hash widened (the PR states the same XOR-and-multiply form as record 62). Each pair of channels draws its sign pattern from a separate 8,192-row table on a different hash. The gate adds the bigram output into the first bigram_dim residual dimensions before layer 0. Rows are zero-initialized, trained by sparse Adam on touched rows only, and sharded across GPUs. The ablation is +7.05 s and +35.8 millinats ([README](https://github.com/KellerJordan/modded-nanogpt); [PR 360](https://github.com/KellerJordan/modded-nanogpt/pull/360)). The README and the PR disagree on the bigram row count (42.3 million in one comment); the figure above is the README's.
- Transfer: replace with an exact dense table. For S symbols: S^2 rows for bigrams (625 at S = 25) and S^3 rows for trigrams (15,625). S^4 is 390,625 rows at S = 25, which is still small enough for a dense table at width 768 if higher-order context helps. Hash only beyond that.
- Cost 1 h on top of record 62. Risk low.

### 92e. Full FP8 MLP forward and backward

- What it is: the README's general bullet list includes FP8 for the MLP forward and backward passes (the list is not tied to a record number), and record 92 adds an FP8 LM-head cache. The PR's ablation is +4.96 s and -0.4 millinats. A branch commit from PR 344 describes all four MLP backward GEMMs in FP8, and the merge commit kept the PR 360 trainer over PR 344, so the merged code may differ. Scaling details are not stated in the PR text ([PR 360](https://github.com/KellerJordan/modded-nanogpt/pull/360)).
- Transfer: as for records 84 and 89, adapted. The LM-head cache is not needed for a 25-class head.
- Cost 1 week with checks. Risk high.

### 92f. Mixed-width attention

- What it is: the README describes "64-dim QK on most layers, half-width V on two layers." The fetched PR summary gives five sublayers with a 64-dim QK at attention scale 0.13 (close to 1 over the square root of 64, which is 0.125), layers 1 and 8 with half-width V and O, and layers 3 and 10 at full width. It needs a patched FlashAttention 3 kernel. The ablation is +3.82 s ([PR 360](https://github.com/KellerJordan/modded-nanogpt/pull/360); [README](https://github.com/KellerJordan/modded-nanogpt)).
- Transfer: adapted. Per-layer QK and V widths are free in our spec. The attention-scale constant should be re-tuned rather than copied.
- Cost 1 day, because flex attention must accept the head dimensions. Risk medium.

### 92g. Depth reduction

- What it is: the README and the PR summary describe removing layer 7 and dropping the attention sublayers of layers 4 and 9, leaving seven attention sublayers at {0, 1, 2, 3, 5, 8, 10} in an 11-layer model.
- Transfer: no. The benchmark fixes 36 blocks. The result is evidence that the record's model does not need every attention sublayer, which is a question to test if the benchmark ever allows fewer attention sublayers.
- Cost not applicable. Risk not applicable.

### 92h. CUDA-graph capture of the training step

- What it is: the forward and backward of every step are captured as CUDA graphs, with nine graph keys covering steps 0 to 1187. The warmup grows to 37 untimed steps, against 14 in record 89. The ablation is +2.99 s for the training step, and the optimizer-update graph adds +0.92 s ([README](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-08-30_ANVIL2); [PR 360](https://github.com/KellerJordan/modded-nanogpt/pull/360)).
- Transfer: yes on one GPU. Pad proteins to fixed bucket lengths and capture one graph per bucket. Decide whether capture and warmup count toward the 10 minutes, since the record leaves warmup untimed.
- Cost 1 to 2 days. Risk medium (memory and shape bookkeeping).

## Techniques named for this period that predate record 57

The benchmark's list includes several techniques that were introduced before record 57. They are listed here with the record that introduced them, so that no one reads them as part of this period. The README table is the source for the record numbers and dates ([README](https://github.com/KellerJordan/modded-nanogpt)).

- Sparse attention gate (record 28, 2025-08-23, 2.817 to 2.812 min, 0.18%; [PR 117](https://github.com/KellerJordan/modded-nanogpt/pull/117); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-08-23_SparseAttnGate)). A sigmoid gate per head from a linear map of the first 12 residual dimensions, zero-initialized so each gate starts at 0.5, multiplying each head's attention output. Transfer: adapted; per-head gates are free in our spec, and the 12-dimension slice is a choice to test. Cost 2 h. Risk low.
- Smear (record 34, 2025-09-18, 2.565 to 2.547 min, 0.70%; [PR 130](https://github.com/KellerJordan/modded-nanogpt/pull/130); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-09-18_Smear)). Position t becomes x_t + g * x_(t-1) with g = smear_lambda * sigmoid(linear(x[:12])), applied right after the token embedding. Transfer: adapted. For bidirectional attention, add the right neighbor as well, and note that a masked position receives its neighbor's embedding directly. Cost 2 h. Risk low.
- NorMuon (record 41, 2025-10-24, 2.358 to 2.345 min, 0.55%; [PR 144](https://github.com/KellerJordan/modded-nanogpt/pull/144); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-10-24_NorMuon)). Neuron-wise normalization after Muon's orthogonalized update. The PR does not state whether it is per row or per column, and the README links arXiv 2510.05491 for the method. Transfer: adapted, if the optimizer is Muon-style; confirm the axis from the arXiv paper before implementing. Cost 3 to 4 h. Risk medium.
- Cautious weight decay (record 43, 2025-11-10, 2.313 to 2.284 min, 1.25%; [PR 154](https://github.com/KellerJordan/modded-nanogpt/pull/154); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-11-10_CautiousWD)). Decay applies only where the update and the weight share a sign, mask = (update * p >= 0), with the decay scheduled by the learning rate. Extended to Adam parameters in record 50 (2025-12-18, 2.146 to 2.128 min, 0.84%; [PR 172](https://github.com/KellerJordan/modded-nanogpt/pull/172); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-12-18_CautiousWDAdam)). Transfer: yes, adapted. Cost 1 h. Risk low.
- Batch size schedule (record 46, 2025-11-29, 2.248 to 2.203 min, 2.0%; [PR 163](https://github.com/KellerJordan/modded-nanogpt/pull/163); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-11-29_BatchSizeSchedule)). The batch size ramps over three phases, matching the window-size schedule. The exact sizes are not on the page. Record 78 later set per-stage document bounds, which is the period's change to this schedule. Transfer: adapted; on one GPU with a 10-minute clock, the trade is tokens per second against steps, so test a short ramp. Cost 2 h. Risk low.
- Partial key offset (record 49, 2025-12-14, 2.170 to 2.146 min, 1.1%; [PR 169](https://github.com/KellerJordan/modded-nanogpt/pull/169); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-12-14_PartialKeyOffset)). Keys in the stationary head dimensions (32 to 64 and 96 to 128 of each 128-dim head) are shifted forward by one token, applied in the long sliding-window layers only. The author says this lets a query pick up the value of the token that followed a matching earlier token. Changes in this period: a fix worth about 0.5 s in record 62, and the offset fused into the QK kernel in record 90. Transfer: adapted; a causal induction argument does not hold for bidirectional attention, so test a shift in each direction and no shift. Cost 2 h. Risk low.
- Multi-token prediction (record 53, 2025-12-22, 2.037 to 1.988 min, 2.4%; [PR 178](https://github.com/KellerJordan/modded-nanogpt/pull/178); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-12-22_MultiTokenPrediction)). A parameter-free weighted average of cross entropy over the next k tokens, with weights decaying from [1, 0.5, 0.25] to [1, 0.5, 0] and then to next-token only. No extra heads. Transfer: no direct analogue. The masked-language analogue would predict neighbors of masked positions as an auxiliary target; that is a proposal, untested. Cost 3 h if tried. Risk medium.
- Asymmetric logit rescale (record 54, 2025-12-26, 1.988 to 1.940 min, 2.4%; [PR 181](https://github.com/KellerJordan/modded-nanogpt/pull/181); [folder](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2025-12-26_LogitRescale)). The softcap changes from 30 * sigmoid(z / 7.5) to 23 * sigmoid((z + 5) / 7.5), so logits are bounded between 0 and 23 rather than -30 and 30. Transfer: adapted; test on the 25-class head. Cost 1 h. Risk low.

Other pre-period items that appear in the setting: Muon (record 3, 2024), untied embedding and head (record 8, 2024) and value embeddings (record 14, 2024), all listed in the README table. These are the foundations the period builds on, not period techniques.

## Summary table

Gains are the record's own change in the README table. Seconds are on the 8-GPU basis, derived. "Applies to MLM" means: yes (applies as is), adapted (applies with the change described), no. Cost is an estimate in hours for this setting. Suggested order is for the 10-minute, one-GPU benchmark, cheapest and lowest-risk first.

| Technique | Record | Gain on record (min, %, s) | Applies to MLM | Implementation cost (h, estimate) | Suggested order |
|---|---|---|---|---|---|
| Dense exact bigram table, with right-neighbor channel | 62 | 0.093, 5.32%, 5.6 s | adapted | 2 to 3 | 1 |
| Bf16 weights with mantissa buffer | 57 | 0.016, 0.84%, 1.0 s | adapted | 2 to 4 | 2 |
| Value embeddings grouped into one parameter | 65 | 0.017, 1.04%, 1.0 s | yes | 1 | 3 |
| Untied value embeddings | 63 | 0.005, 0.30%, 0.3 s | adapted | 1 to 2 | 4 |
| Value-embedding layout and gates | 70 | 0.007, 0.46%, 0.4 s | adapted | 2 | 5 |
| Nonzero V and O init (mimetic) | 64 | 0.020, 1.21%, 1.2 s | adapted | 1 to 2 | 6 |
| Learnable XSA | 82 | 0.010, 0.73%, 0.6 s | yes | 2 to 3 | 7 |
| XSA algebraic rewrite | 86 | 0.005, 0.39%, 0.3 s | yes | 0.5 | 7 |
| Max sequence length curriculum and LR floor | 72 | 0.020, 1.32%, 1.2 s | adapted | 2 to 3 | 8 |
| Device-side bigram index | 68 | 0.005, 0.32%, 0.3 s | yes | 0.5 | 9 |
| Eval restriction to canonical residues | 91 | 0.004, 0.35%, 0.2 s | adapted (metric decision) | 1 | 10, after definition agreed |
| Parameter banks, batched Muon | 61 | 0.017, 0.96%, 1.0 s | yes | 6 to 10 | 11 |
| Flattened forward | 74 | 0.017, 1.14%, 1.0 s | yes | 4 to 8 | 12 |
| Paired head Q and K orthogonalization | 80 | 0.005, 0.35%, 0.3 s | adapted | 2 to 3 | 13 |
| Fused linear-ReLU-square MLP | 59 | 0.039, 2.14%, 2.3 s | adapted | 4 to 8 | 14 |
| ReLU-squared kernel (stores post) | 87 | 0.010, 0.79%, 0.6 s | yes (if ReLU squared) | 3 to 5 | 14 |
| XXT kernel tuning | 69 | 0.007, 0.46%, 0.4 s | adapted (XXT only) | 2 to 4 | 15 |
| Paired head attention | 58 | 0.058, 3.09%, 3.5 s | adapted | 4 to 6 | 16 |
| Saved activation (replaces partitioned hyperconnections) | 77 | 0.011, 0.76%, 0.7 s | adapted | 2 | 17 |
| Partitioned hyperconnections | 73 | 0.011, 0.74%, 0.7 s | adapted | 4 to 6 | skip if 77 used |
| MUDD skip connections | 81 | 0.043, 3.06%, 2.6 s | adapted | 6 to 10 | 18 |
| CUDA-graph capture (92) | 92 | part of 0.461 (+2.99 s ablation) | adapted | 1 to 2 days | 19 |
| Mixed-width QK (92) | 92 | part of 0.461 (+3.82 s ablation) | adapted | 1 day | 20 |
| Tail weight averaging (92) | 92 | not separable | adapted (checkpoint choice) | 4 to 6 | 21 |
| FP8 MLP up-projection forward | 84 | 0.008, 0.60%, 0.5 s | adapted | 6 to 10 | 22 |
| FP8 MLP down-projection, delayed scaling | 89 | 0.013, 1.05%, 0.8 s | adapted | 6 to 10 | 23 |
| Reduced QK width 96, packed FP8, fused QK kernel | 90 | 0.100, 8.13%, 6.0 s | partly | 1 to 2 weeks | 24 (QK width alone earlier) |
| MUDD gates and DC-MHA | 85 | 0.049, 3.71%, 2.9 s | adapted | 2 to 3 days | 25 |
| ANVIL optimizer (92) | 92 | part of 0.461 (+5.08 s ablation) | adapted | 1 to 2 days | 26 |
| Full FP8 MLP forward and backward (92) | 92 | part of 0.461 (+4.96 s ablation) | adapted | 1 week | 27 |
| Hashed n-gram table (92) | 92 | part of 0.461 (+7.05 s ablation) | no as built; dense exact table only | 1 (dense) | 1 with 62, if trigrams |
| Sampled softmax (92) | 92 | part of 0.461 (+8.40 s ablation) | no | not applicable | skip |
| Depth reduction (92) | 92 | part of 0.461 | no (36 blocks fixed) | not applicable | skip |
| Fused softcapped multi-token CE | 60 | 0.016, 0.90%, 1.0 s | no (kernel); softcap option only | not applicable | skip |
| Sign trick on bigram table | 83 | 0.025, 1.85%, 1.5 s | no with dense table | not applicable | skip |
| Prefix token prediction | 88 | 0.013, 1.04%, 0.8 s | no as stated; residue-class head is an untested proposal | 3 to 4 | proposal only |
| FP8 LM head | 67 | 0.055, 3.45%, 3.3 s | no | not applicable | skip |
| Cross-entropy kernels | 75, 79 | 0.015 each | no | not applicable | skip |
| Transpose-copy kernel | 76 | 0.007, 0.48%, 0.4 s | no | not applicable | skip |
| Sparse bigram gradient communication | 71 | 0.005, 0.33%, 0.3 s | no (one GPU) | not applicable | skip |
| Tight document-count bound | 78 | 0.009, 0.63%, 0.5 s | adapted, only if packing | 2 to 3 | conditional |
| Bigram index to GPU | 68 | see above | yes | see above | see above |
| PyTorch 2.10 | 66 | 0.018, 1.12%, 1.1 s | not a technique | 0 | use the benchmark's stack |

Pre-period techniques named in the benchmark list:

| Technique | Record (date) | Gain on record (min, %) | Applies to MLM | Cost (h, estimate) |
|---|---|---|---|---|
| Sparse attention gate | 28 (2025-08-23) | 0.005, 0.18% | adapted | 2 |
| Smear | 34 (2025-09-18) | 0.018, 0.70% | adapted | 2 |
| NorMuon | 41 (2025-10-24) | 0.013, 0.55% | adapted | 3 to 4 |
| Cautious weight decay | 43, 50 (2025-11-10, 2025-12-18) | 0.029 and 0.018 | yes | 1 |
| Batch size schedule | 46 (2025-11-29) | 0.045, 2.0% | adapted | 2 |
| Partial key offset | 49 (2025-12-14) | 0.024, 1.1% | adapted (test both directions) | 2 |
| Multi-token prediction | 53 (2025-12-22) | 0.049, 2.4% | no direct; proposal only | 3 |
| Asymmetric logit rescale | 54 (2025-12-26) | 0.048, 2.4% | adapted | 1 |
| Tail weight averaging with raised LR floor | not located in the table | not separable | adapted (checkpoint choice) | 4 to 6 |

## Not found, and what was not read

- ANVIL quintic coefficients, the momentum default, the learning-rate and weight-decay defaults, and the pre-step-514 fast-beta schedule are not in the content read. The source is the current master's [anvil.py](https://github.com/KellerJordan/modded-nanogpt/blob/master/track_1_short/optim/anvil.py), which may differ from the record 92 snapshot.
- Tail averaging: the record that introduced it, its final checkpoint rule and any learning-rate-floor interaction are not found. The code comments cite record 360.
- PR 219 (algorithm for sparse bigram gradient communication, referenced by record 71) was not read.
- Record 75's kernel changes, record 79's CE internals, and record 59's backward kernel detail are not on the PR pages read.
- Record 66 has no PR or folder in the README table.
- The folder `records/track_1_short/2026-05-07_XSAGatedLayers` (a verifier and methodology for the XSA record) was not read.
- The folder `records/track_1_short/2026-07-22_AttentionPacking` is not in the README table. Its README reports 1.320 to 1.297 min (77.812 s mean over four H100 seeds, no same-node baseline) and does not state a record number. It is not counted as a record here.
- Record 92's full FP8 details and the merged trainer may differ between PR 344 and PR 360; the merge commit kept the PR 360 trainer.
- Several PR summaries conflict with each other on timings, and some PR titles claim more than the merge comment does. The figures here use the README table for before and after times, and quote PR numbers for the rest.
