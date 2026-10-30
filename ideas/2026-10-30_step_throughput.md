# Step throughput at the fixed shape: more training steps in the 600 s clock

Date: 2026-10-30. A literature and engineering scout for SpeedrunningPLMs: 36 pre-norm blocks (one attention sublayer
and one ReLU-squared MLP of width 2,048 each) on a 768-wide stream; 15 percent of residues replaced by MASK; 20-way cross
entropy (CE) on masked residues; one GH200 (Grace Hopper superchip, aarch64, H100-class GPU, driver 570, torch 2.11
cu128); bf16 autocast; `torch.compile`; flex attention with a block-diagonal document mask over 64 packed 1,024-token
windows (65,536 tokens a step); Muon (banked Polar Express, NorMuon row normalization) on block matrices, AdamW elsewhere.
The record runs about 2,150 steps of 0.276 s in the 600 s clock, with a compile warmup of 106 to 122 s against a hard
180 s cap. The file reads `ideas/2026-10-30_beyond_backprop.md` for its format, `ideas/modded_nanogpt_2026.md` so it does
not repeat those records, `train.py`, and the code that sets the step.

Nothing here is measured on this benchmark. Each gain is a source's number or an estimate, and the text says which.
Shorthand: MLM is masked language modeling; FA4 is FlashAttention-4; FA3 is FlashAttention-3; HBM is high-bandwidth
memory; TF/s and PF/s are tera- and peta-FLOP per second; ms is milliseconds; ln is the natural log; "nats" is validation
cross entropy; a "bucket" is one line of the step profile. One seed resolves about 0.0015 nats (the README band).

## Outcome

- The step profile (brief) sums to 240 ms: linear layers 109 ms, flex attention 80 ms, banked Muon 26 ms, RMS norms 17 ms,
  ReLU squared 8 ms. The other 36 ms of the 276 ms step (13 percent) is in no profiled bucket. That is the first target:
  per-step launch and synchronization overhead, the per-step block-mask build, and Muon's per-matrix work.
- The linear layers are near the roof. The 36 blocks hold 198 million matrix parameters; with 65,536 tokens and 2 FLOP per
  multiply-add, forward plus backward is about 77.9 TFLOP a step (derived here). At 109 ms that is about 715 TF/s in bf16.
  Check the bf16 dense peak against the NVIDIA datasheet (not read here). If the peak is near 1 PF/s, the bucket has under
  30 percent room, and no idea below targets it.
- Clock accounting, checked in code: compile warmup, in-run validation and the data fetch between steps are all outside the
  training clock (`clock.py` closes a step at `applied()` and reopens it at `start_step()`; `train.py` fetches the next
  batch in between). Compile and eval cuts buy headroom under the 180 s and 60 s caps and zero extra steps. Loader changes
  buy zero measured steps. The block-mask build, the Muon step and the forward and backward are on the clock.
- Largest lever with a published measurement: FA4 as the flex backend (idea 1). The PyTorch blog reports 1.41 to 1.89 times
  forward and 1.48 to 2.01 times backward on H200 for document masks at 2K and longer. Our windows are 1K, so we assume 1.2 to
  1.5 times on the flex bucket (estimate). Its compile cost is not published.
- Stacked, the non-overlapping ideas (1 to 6 and 11) could cut 40 to 60 ms, 15 to 20 percent of the step, which is about
  0.004 to 0.006 nats (estimate, rough, overlaps not removed). That is two to four seed bands. No single idea clears a band
  with confidence except idea 1 at the top of its range.
- Compile is a cap question. Warmup is 106 to 122 s, so the headroom is 58 to 74 s. Ideas that add compile time (coordinate
  descent, max-autotune, a graph capture, a second kernel family) need regional compilation (idea 8) or a warm cache (idea
  12) first, or a run risks the cap and is invalid.
- Yardstick: a time saving s is a step ratio of 1 / (1 - s), so the gain is about 0.026 x -ln(1 - s) nats (0.026 per unit
  of ln steps, from the catalog's 1.53x result). Conversions: 1 percent of step time is 0.0003 nats; 2 percent 0.0005;
  5 percent 0.0013; 10 percent 0.0027; 20 percent 0.0058. The slope comes from another change, so each is rough.
- Benchmark changes: idea 7 edits `benchmark/clock.py`, which is part of the fixed benchmark, so it needs a separate
  commit that the project owner approves. Moving work into the off-clock fetch (the block mask, for example) would take its
  cost out of the measured step. That is a measurement question, not a speedup, and it is not proposed here.
- Read caveat: the fetch tool summarizes each page it reads, so numbers from fetched pages are the tool's output, not checked
  against raw HTML. Items marked "snippet" come from search results only.
- First step: a 60 s pilot timing four things on one batch: flex with `BACKEND: FLASH`; `create_block_mask`; the Muon step;
  and the per-step CPU time against GPU time from a `torch.profiler` trace. Those decide ideas 1, 2, 3, 5 and 7 before any build.

## Ranked table

Gain is the expected change in validation loss on this benchmark (estimate; negative is better). Step cost is the change to
the 276 ms step (estimate; negative saves time). "Read" is full (the fetched page, through the summarizing tool), abstract,
snippet, or code (this repository). The 180 s warmup cap applies to every compile.

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats) | Compile | Step cost | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | Flex attention on the FA4 backend (`BACKEND: FLASH`) | PyTorch blog (2026); PyTorch 2.11 release (2026) | full | 4 to 8 | -0.001 to -0.003 | CuTeDSL JIT, unmeasured; cap risk high | -13 to -27 ms of the 80 ms flex bucket | H200 document mask 1.41 to 1.89 times fwd, 1.48 to 2.01 times bwd, at 2K and up |
| 2 | CUDA graphs: `reduce-overhead` on the step; Muon step captured with tensor `lr` | torch.compile docs (2.11); modded-nanogpt record 92 (2026) | full; record from prior file | 6 to 10 | -0.0005 to -0.002 | capture adds warmup, unmeasured; cap risk medium | -5 to -20 ms of the 36 ms unexplained | record 92: +2.99 s training step and +0.92 s optimizer graph, of 39.9 s |
| 3 | Muon elementwise fusion: compile the Polar Express loop, drop fp32 stacks, batch the updates | Polar Express (2025, arXiv 2505.16932); our `optimizer.py` | full (paper); code | 3 to 5 | -0.0005 to -0.001 | +10 to 30 s, estimate; cap risk low to medium | -5 to -10 ms of the 26 ms Muon bucket | 4.0 TFLOP of polar products a step (derived), so about 20 ms of the bucket is non-matmul |
| 4 | Flex kernel tiles and mask block size | Flex attention docs (2.11) | full | 2 to 3 | -0.0004 to -0.001 | autotune shifts; measure | -4 to -12 ms of the 80 ms flex bucket | docs: default autotuning "provides good performance" |
| 5 | Block-mask build: measure, then build from document spans | our `strategies/attention.py`; flex docs | code; docs | 1 to 3 | -0.0002 to -0.0006 | none | -2 to -6 ms, unmeasured | built inside the clock every step; snippet: mask build can exceed kernel time |
| 6 | Fused RMS norm and ReLU squared in Triton | modded-nanogpt records 59 and 87 (2026); Liger Kernel (2024, arXiv 2410.10989) | full (records); abstract (Liger) | 4 to 8 | -0.0005 to -0.001 | small | -5 to -10 ms of the 25 ms norm and ReLU buckets | norm and ReLU traffic is near the HBM floor (derived) |
| 7 | Remove per-step device synchronization from the clock | our `benchmark/clock.py` | code | 1 | -0.0001 to -0.0004 | none | -1 to -4 ms | `expired()` and `applied()` each call `torch.cuda.synchronize` |
| 8 | Regional compile: one compiled block reused (enabler) | PyTorch regional compilation recipe | full | 3 to 4 | 0 direct; 0 to +0.0003 if fusion is lost | warmup could fall below 106 s | 0 to +3 ms, estimate | synthetic test: 9.66 s full against 0.35 s regional (cold) |
| 9 | Coordinate descent or `max-autotune-no-cudagraphs` | torch.compile docs (2.11); PyTorch PR 97203 (2023) | full (docs); snippet (PR) | 2 | 0 to -0.0008 | about 2 times compile per the PR; cap risk high | 0 to -8 ms, estimate | PR: 1.005 to 1.07 times on 2023 TorchBench models |
| 10 | Polar Express in 4 steps instead of 5 | Polar Express (2025, arXiv 2505.16932) | full | 2 to 3 | -0.0003 to +0.001 (net; sign open) | none | -3 to -5 ms | paper uses five steps; four-step coefficients not in the read text |
| 11 | Muon and Adam on separate CUDA streams | our `optimizer.py`, `train.py` | code | 3 | -0.0002 to -0.0004 | none | -2 to -4 ms, estimate | the Adam group is launch-heavy and independent of Muon |
| 12 | Warm Inductor cache and cache artifacts | PyTorch compile caching recipes | snippet | 1 | 0 (steps); cap headroom on repeat runs | warm only; report cold and warm | 0 | `train.py` already sets `TORCHINDUCTOR_CACHE_DIR` |

## 1. Flex attention on the FA4 backend

- Source: "FlexAttention + FlashAttention-4: Fast and Flexible," PyTorch blog (last updated 2026-03-04,
  https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/); PyTorch 2.11 release blog (last updated
  2026-03-30, https://pytorch.org/blog/pytorch-2.11-release/). Read: full, through the fetch tool.
- Evidence: `kernel_options={"BACKEND": "FLASH"}` selects a FlashAttention-4 backend on Hopper and Blackwell. Blog results on
  H200: document mask 1.41 to 1.89 times forward and 1.48 to 2.01 times backward; sliding window 1.45 to 1.65 forward; the
  gains "are present even at shorter sequences (2K)". The 2.11 release lists the backend as API-unstable. The blog names
  limits that matter here: KV loads "can stall the pipeline, especially for pointer-chasing patterns (e.g., document masking
  with per-token metadata)"; the backward is "not yet deterministic when block-sparsity is enabled"; captured buffers that need
  gradients are unsupported in the backward. Its Hopper tile is 128 by 128, which matches our default.
- What changes here: our document mask captures a `document` tensor that needs no gradient, which the blog's limits allow. The
  flex call is at `src/speedrunning_plms/strategies/attention.py` line 123. A genome field `backend` (default Triton, so the
  record is unchanged) passes `kernel_options` only when set.
- Expected cost: 4 to 8 h: a pilot, a FlashAttention CuTe install in the Docker image that matches torch 2.11 cu128 (the blog
  asks for a recent nightly; the 2.11 release does not, so check), and a correctness check against Triton. The CuTeDSL
  kernels are JIT-instantiated, and no compile time is published. If the JIT exceeds the 58 to 74 s headroom, the option is
  invalid.
- Gain estimate: flex is 80 ms. A 1.2 to 1.5 times speedup saves 13 to 27 ms (4.7 to 9.8 percent), -0.001 to -0.003 nats
  (estimate). The top of that range assumes the blog's document-mask speedup holds at 1K windows.
- Sketch (names as at the call site; the genome field is new):

```python
# strategies/attention.py, the flex call (line 123): the genome's backend; Triton by default
kwargs = {"kernel_options": {"BACKEND": "FLASH"}} if self.backend == "flash" else {}
return flex_attention(Q, K, V, score_mod=score_mod, block_mask=block_mask, enable_gqa=group > 1, **kwargs)  # (b, q, n, e)
```

- Bit-exactness: a different kernel, so the loss will not match the Triton path bit for bit. The nondeterministic backward
  means two runs of one genome can differ. Compare on a fixed batch with a tolerance, and measure the seed-to-seed spread
  before reading any gain.
- Risk: API instability; the kernel may not exist on the box's exact build; the pointer-chasing stall the blog names is the
  document-mask pattern, so the 1K speedup may fall below the blog's range.
- What to measure: flex forward and backward on one batch, Triton against FLASH, at the run's shape; warmup seconds with the
  backend on; the loss on a fixed batch.

## 2. CUDA graphs: `reduce-overhead` on the step, with the Muon step captured

- Source: torch.compile docs, 2.11 (https://docs.pytorch.org/docs/2.11/generated/torch.compile.html). Read: full, through the
  fetch tool. modded-nanogpt record 92 (PR 360; read through the prior file). Snippets: a Hugging Face model file compiles
  `create_block_mask` without `reduce-overhead`, because the mode's CUDA graph trees "run on private streams" and manual
  capture recorded empty graphs (https://huggingface.co/tiiuae/Falcon-Perception/blob/main/attention.py); a PyTorch forum
  thread reports slower steps and endless recording logs with FSDP2 and flex attention under `reduce-overhead`
  (https://discuss.pytorch.org/t/cuda-graphs-with-fsdp-2-and-flex-attention/223437).
- Evidence: the docs say `reduce-overhead` "uses CUDA graphs to cut Python overhead", but "today, we only reduce overhead for
  CUDA only graphs which do not mutate inputs", and the reduction "isn't guaranteed". Record 92's ablation is +2.99 s for the
  training step and +0.92 s for the optimizer graph, of a 39.9 s training section (7.5 and 2.3 percent, derived). Its step is
  much shorter than ours, so the fraction does not transfer directly.
- What changes here: `curriculum.compile_options()` returns `{}` for the whole-protein variant (`strategies/curriculum.py`
  line 86), so today's step uses the default mode with no graphs. The step's inputs (`corrupted`, `labels`, block masks) are
  new tensors each step but are not mutated, which fits the restriction. Two blockers: `zero_grad(set_to_none=True)` frees
  gradients every step, and the Muon step reads `lr` and `momentum` as Python floats that change each step
  (`train.py` lines 425 to 427). A captured optimizer step needs them as device tensors.
- Expected cost: 6 to 10 h: a genome field; moving `lr` and `momentum` to tensors; capturing the Muon step. Shapes are already
  fixed at 64 by 1,024. Compile: capture adds unmeasured warmup, which must still clear 180 s. Cap risk: medium.
- Gain estimate: the 36 ms unexplained bucket is the target. A 5 to 20 ms saving (2 to 7 percent) is -0.0005 to -0.002 nats
  (estimate). Idea 7 interacts with this and belongs in the same pilot.
- Sketch:

```python
compiled = torch.compile(model, mode="reduce-overhead", dynamic=False)  # fixed (64, 1024) windows
# Muon: group["lr"] and group["momentum"] become device tensors (train.py lines 425-427) before the step is captured
```

- Bit-exactness: capture replays the same kernels in the same order, so the loss should match eager exactly. Check 50 steps
  of loss against eager before trusting it.
- Risk: the forum thread and the Hugging Face note both report graph and flex interactions; the GH200 memory cost of graph
  pools is unmeasured.
- What to measure: CPU against GPU time per step with and without the mode; warmup seconds; the 50-step loss against eager;
  peak memory.

## 3. Muon elementwise fusion: compile the Polar Express loop, drop the fp32 stacks

- Source: "The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm," Amsel, Persson, Musco
  and Gower (arXiv 2505.16932; ICLR 2026 per a search snippet). Read: full HTML, through the fetch tool. Our
  `strategies/optimizer.py`: `polar_express` at line 102; `Muon.step` at lines 155 to 187.
- Evidence (paper): five iterations of a degree-5 odd polynomial, `X <- a X + (b A + c A^2) X` with `A = X X^T`, three
  products each, 15 in all. The GPT-2 124M experiments run bf16 on 4 H100 GPUs, 1B FineWeb tokens.
- Evidence (our code): `polar_express` runs in bf16 on CUDA. `Muon.step` stacks each bank's gradients into fp32 (line 165),
  lerps the momentum, orthogonalizes, applies NorMuon's row normalization (lines 175 to 177), and applies updates with
  `unbind` and `_foreach_add_` (line 187).
- Derived arithmetic: one iteration is about 2.72 GFLOP per 768 by 768 matrix (144 of them) and 5.74 GFLOP per 2,048 by 768
  matrix (72 of them): about 805 GFLOP, or 4.0 TFLOP for five iterations. At 500 to 800 TF/s that is 5 to 8 ms of products,
  so about 20 ms of the 26 ms Muon bucket is elementwise work, copies, norms and per-matrix launches.
- What changes here: compile the Polar Express loop and the row normalization per bank shape (three shapes); keep gradients in
  bf16 until the first fp32 use if the accuracy check passes; replace the per-matrix `unbind` with one batched add.
- Expected cost: 3 to 5 h. Compile: three graphs, about +10 to 30 s (estimate); cap risk low to medium.
- Gain estimate: 5 to 10 ms of the 26 ms bucket (2 to 4 percent), -0.0005 to -0.001 nats (estimate).
- Sketch:

```python
polar_express = torch.compile(polar_express, dynamic=False)  # strategies/optimizer.py line 102: (k, m, n) in, fp32 out
```

- Bit-exactness: fusion changes the rounding order of the bf16 products, so the update is not bit-identical. Check the update
  against eager on a real gradient bank to a tolerance, and the loss over 50 steps.
- Risk: the compile adds warmup; a dynamic shape would recompile, so `dynamic=False` must hold.
- What to measure: the Muon bucket before and after; warmup seconds; the update difference on one bank.

## 4. Flex kernel tiles and mask block size

- Source: "FlexAttention" docs, PyTorch 2.11 (https://docs.pytorch.org/docs/2.11/nn.attention.flex_attention.html). Read:
  full, through the fetch tool. The FlexAttention paper (arXiv 2412.05496; abstract only).
- Evidence (docs): `kernel_options` takes `BLOCK_M` and `BLOCK_N` (forward) and `BLOCK_M1`, `BLOCK_M2`, `BLOCK_N1`, `BLOCK_N2`
  (backward); `create_block_mask` takes `BLOCK_SIZE` (default 128). `PRESCALE_QK` is "slightly faster but may have more
  numerical error"; `WRITE_DQ=False` "may be faster". The paper's one sparsity number is about 15 percent for causal masks
  from full-block splitting; it gives no document-mask number.
- What changes here: the masks use the default 128 (`strategies/attention.py` lines 209, 219, 225). A genome field sets
  `BLOCK_SIZE` and the tile sizes, swept in the pilot on one batch. Leave `PRESCALE_QK` off, since it trades numerics for speed.
- Expected cost: 2 to 3 h for four to six settings. Compile: autotuning already runs at compile time, so warmup may move either way.
- Gain estimate: 4 to 12 ms of the 80 ms flex bucket, -0.0004 to -0.001 nats (estimate). Smaller mask blocks can skip more
  empty work at document edges, where short proteins waste a 128 block.
- Sketch:

```python
flex_attention(Q, K, V, block_mask=block_mask, kernel_options={"BLOCK_M": 64, "BLOCK_N": 64})  # sweep; no PRESCALE_QK
create_block_mask(same_document, b, None, l, l, device=..., BLOCK_SIZE=64, _compile=True)     # mask granularity
```

- Bit-exactness: tile changes alter accumulation order, so the loss is not bit-identical. The mask is the same per-element
  rule at any block size, so check that the masks agree element-wise.
- Risk: the default autotuner may already pick the best tile, which would make this a null result.
- What to measure: flex forward and backward per setting; warmup seconds; `sparsity()` at 64 and 128.

## 5. Block-mask build: measure first, then build from document spans

- Source: our `strategies/attention.py` (`training_masks`, lines 198 to 225; `create_block_mask(..., _compile=True)` per batch).
  Flex docs (`create_block_mask`, default block 128). Snippet: the Flashlight paper (arXiv 2511.02043) says "constructing the
  sparse block mask is much slower than the kernel itself".
- Evidence: the mask is built from each batch's document ids inside the clock, every step (`train.py` lines 412 to 413 follow
  `clock.start_step()` at line 410). Its cost is not in a named bucket, so it may be part of the 36 ms.
- What changes here: measure the build first (one batch, median of 20). If it exceeds 2 ms, build the block arrays from
  document spans: each 128-row block's first and last document id decide full, partial or empty blocks; partial blocks keep the
  element rule. The kernel's inputs are the same by construction.
- Expected cost: 1 h to measure; 2 to 3 h to build if worth it. Compile: none new; the build is shape-fixed.
- Gain estimate: 2 to 6 ms (0.7 to 2.2 percent), -0.0002 to -0.0006 nats (estimate). If the measured build is under 1 ms, drop it.
- Sketch:

```python
first, last = document.amin(-1), document.amax(-1)   # (b, l / 128) document ids per block; padding is -1, handled explicitly
```

- Bit-exactness: exact, if the block arrays match. Compare `kv_num_blocks`, `kv_indices` and the full-block arrays with
  `create_block_mask` on 100 random batches before use.
- Risk: padding and documents that start mid-block must be handled. A wrong block skip is a silent loss error, so the equality
  check is a hard gate.

## 6. Fused RMS norm and ReLU squared in Triton

- Source: modded-nanogpt record 59 (PR 197, fused linear, ReLU and square, 2026) and record 87 (PR 322, ReLU squared storing
  `post` and rebuilding the derivative from `sqrt(post)`, 2026), read through the prior file. Liger Kernel (arXiv 2410.10989,
  2024): the abstract reports "on average a 20% increase in training throughput" and "a 60% reduction in GPU memory usage"
  against Hugging Face implementations for LLMs; the kernel list is not in the abstract. Read: records, full (prior file);
  Liger, abstract.
- Evidence: the MLP activation is 65,536 by 2,048 values, about 268 MB in bf16, so one pass is about 0.08 ms at 3.3 TB/s (an
  H100-class bandwidth estimate; check the datasheet). Several such passes per block, over 36 blocks, put 8 ms and 17 ms near the
  traffic floor, so fusion can only remove extra passes.
- What changes here: a Triton ReLU-squared fused into the up-projection epilogue (record 59's pattern), and a fused RMS norm for
  the per-block norms with the residual add. Liger's 20 percent is against eager Hugging Face for LLMs and does not transfer.
- Expected cost: 4 to 8 h, with a test against the eager op. Compile: small.
- Gain estimate: 5 to 10 ms (2 to 4 percent), -0.0005 to -0.001 nats (estimate). Inductor may already fuse part of this, so the
  profile should first show which norms are separate kernels.
- Bit-exactness: not bit-exact. Record 87's `sqrt(post)` derivative rounds differently from the pre-activation; use the
  pre-activation form if exactness matters.
- Risk: a small gain for several new kernels; a fused epilogue can change the matmul's tile choice.

## 7. Remove per-step device synchronization from the clock

- Source: our `benchmark/clock.py` (`expired()` lines 65 to 75 and `applied()` lines 77 to 81 each call
  `torch.cuda.synchronize`), and `train.py` lines 415 and 433. Read: code.
- Evidence: each step synchronizes twice, so the CPU's launch work for the next step is exposed unless the GPU is already
  waiting. With fused kernels the exposed time is probably small: 1 to 4 ms a step (estimate).
- What changes here: record CUDA events at step start and end and read the elapsed time later, without a host wait; the
  discard check in `expired()` uses the previous step's event. The accounting does not change. The clock is part of the fixed
  benchmark, so this needs a separate, approved commit.
- Expected cost: 1 h, plus review of the benchmark change.
- Gain estimate: 1 to 4 ms (0.4 to 1.4 percent), -0.0001 to -0.0004 nats (estimate). Its value rises with idea 2.
- Sketch:

```python
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record(); ...step...; end.record()   # start.elapsed_time(end) once end is complete; no host wait per step
```

- Bit-exactness: none; timing only.
- Risk: the discard rule ("the caller discards the step it was about to apply") must keep its semantics with events, which is a
  benchmark decision.
- What to measure: the gap between one step's last kernel and the next step's first, from the profiler, with and without the
  synchronizations.

## 8. Regional compile: one compiled block reused (enabler)

- Source: PyTorch regional compilation recipe (https://docs.pytorch.org/tutorials/recipes/regional_compilation.html). Read: full,
  through the fetch tool.
- Evidence: on a synthetic model of 64 identical layers, full compilation took 9.66 s and regional compilation 0.35 s, both cold.
  The page warns of a trade between speedup and compile time, and "many cases with minimal speedup differences".
- What changes here: `build_candidate` (`train.py` line 181) compiles each block, so the 36 blocks share one graph; the flex
  masks are arguments to each call. Fusion across block boundaries is lost, so the step may slow a little.
- Expected cost: 3 to 4 h. This is an enabler: warmup could fall below 106 s, freeing headroom for idea 9 or the FA4 JIT.
- Gain estimate: 0 direct; a step cost of 0 to +3 ms (estimate), so 0 to +0.0003 nats.
- Sketch:

```python
for block in candidate.blocks:
    block.compile(dynamic=False)   # nn.Module.compile per block; the outer model stays eager
```

- Bit-exactness: fusion boundaries move, so rounding changes. Not bit-exact.
- Risk: the triangle variants use three masks, which may recompile per mask type; check.

## 9. Coordinate descent tuning or `max-autotune-no-cudagraphs`

- Source: torch.compile docs, 2.11 (`max-autotune-no-cudagraphs` is "similar to max-autotune, but without CUDA graphs"; the docs do
  not describe `coordinate_descent_tuning`). Read: full (docs). PyTorch PR 97203, "coordinate descent tuning upon max-autotune"
  (2023). Read: snippet from a mirror (https://app.semanticdiff.com/gh/strisunshinewentingwang/pytorch/commit/52ecc3274b1c16fcca3a3d89bd261dbc6513d6ed).
- Evidence: the PR's spot checks show 1.005 to 1.07 times on 2023 TorchBench models and "roughly doubled" compile time against
  max-autotune. The docs say max-autotune profiles matmul configurations; our matmuls are cuBLAS, so gains would sit on the
  elementwise and flex parts.
- What changes here: a genome field adds `mode="max-autotune-no-cudagraphs"` and the Inductor coordinate-descent flag (check its
  name against the 2.11 config source).
- Expected cost: 2 h to test. Compile: roughly doubled, which on a 106 to 122 s warmup exceeds 180 s unless idea 8 or 12 frees room.
- Gain estimate: 0 to 8 ms (0 to 3 percent) if it fits the cap, so 0 to -0.0008 nats (estimate).
- Bit-exactness: autotuned kernels change accumulation order; not bit-exact.
- What to measure: cold warmup and step time; keep only if warmup stays under 180 s with margin.

## 10. Polar Express in four steps instead of five

- Source: Polar Express (arXiv 2505.16932), as in idea 3. The paper uses five iterations; for each step it replaces each
  polynomial p_t(x) with p_t(x / 1.01) as a safety factor, omitted in the last step, for a lower bound of 1e-3.
- Evidence (paper): each iteration costs three products. Four-step coefficients are not in the text read. Our code uses a
  safety factor of 0.02 and epsilon 1e-6 (`strategies/optimizer.py` line 102).
- What changes here: a four-tuple coefficient table derived with the paper's method (its appendix code was not read), and a
  check of the orthogonalization error against five steps on a real gradient bank.
- Expected cost: 2 to 3 h. Compile: none.
- Gain estimate: one of five iterations is about 3 to 5 ms (1 to 2 percent), so -0.0003 to -0.0005 nats on speed alone. A less
  exact orthogonalization could cancel it, so the net is -0.0003 to +0.001 (estimate; sign open until the loss runs).
- Bit-exactness: not bit-exact, and the update itself changes; compare loss at the same seed.
- Risk: an accuracy trade, not a pure speed win. Reject if the loss does not hold within the seed band.

## 11. Muon and Adam on separate CUDA streams

- Source: our `strategies/optimizer.py` and the optimizer loop in `train.py` (lines 428 to 430). No paper. Read: code.
- Evidence: after backward, the Muon and Adam steps touch disjoint parameters. The Adam group (embedding, head, norms, scalars) is
  launch-heavy, so it is latency-bound and can overlap.
- What changes here: run the Adam step on a second stream with an event the Muon stream waits on before the next forward.
- Expected cost: 3 h, with a check that the waits are right. Compile: none. Gain: 2 to 4 ms (0.7 to 1.4 percent), -0.0002 to
  -0.0004 nats (estimate; the Adam step's time is not yet profiled).
- Sketch:

```python
adam_stream = torch.cuda.Stream()
adam_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(adam_stream):
    adam.step()
muon.step()                                       # default stream
torch.cuda.current_stream().wait_stream(adam_stream)
```

- Bit-exactness: exact, since the same operations run on the same tensors with correct waits.
- Risk: a missing wait is a silent race; check 50 steps of loss against the serial order.

## 12. Warm Inductor cache and cache artifacts

- Source: PyTorch compile caching recipes (https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html;
  https://pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html). Read: snippet.
- Evidence (snippet): `torch.compiler.save_cache_artifacts()` and `load_cache_artifacts()` restore the FX graph, Triton, AOTAutograd,
  PGO and autotuning caches. Artifacts need the same PyTorch and Triton versions and, on CUDA, the same GPU.
  `TORCHINDUCTOR_FX_GRAPH_CACHE=1` enables the local FX cache. `train.py` (line 484) already sets `TORCHINDUCTOR_CACHE_DIR`. The
  2.11 default for the FX cache is not checked here.
- What changes here: nothing to the step. Repeat runs on one box compile faster, which helps the search and the cap headroom.
  Reports must state cold and warm warmup separately.
- Expected cost: 1 h. Gain: 0 nats; its value is headroom for ideas 1, 8 and 9.
- Bit-exactness: none; the same compiled code is reused.
- Risk: a cache from another build or GPU is silently unused or invalid; check the version key.

## Checked and rejected

- FP8 attention projections. The FP8 MLP measured 0.284 s against 0.276 s in bf16 (catalog B7), the same width class, and the
  linear bucket is near the roof. Reject.
- Fewer in-run evaluations, or zero-copy best weights. `evaluate_now` runs inside `clock.paused()` (`train.py` line 379), so
  evaluation time is not in the step count; the CPU copy of the best weights is off the clock too. These buy headroom under the
  60 s cap and no steps. Reject for steps.
- Async data loading and pinned memory. The fetch is off the clock, and each step's tokens are 64 KB of uint8, so the copy is
  small. Zero measured gain. Do not move work into the off-clock fetch to shorten the measured step.
- Overlapping the Muon step with backward. Each bank stacks same-shape matrices from all 36 blocks, so it is complete only after
  the first block's backward. The overlap left is the Adam group (idea 11).
- Bf16 parameters with fp32 master copies (record 57, prior file). Muon already keeps fp32 momentum; the saving is a weight read
  per matmul, under 1 ms (estimate), with the silent-loss risk noted in the prior file. Defer.
- Liger Kernel as a dependency. It targets LLM stacks, and its 20 percent is against eager Hugging Face. Its two relevant
  kernels (norm, ReLU squared) are covered by idea 6 as our own Triton code.
- Fused linear CE, FP8 LM head, softcap kernels (records 60, 67, 75, 79). The head scores 20 residue columns of a 33-entry
  vocabulary (`benchmark/tokens.py`, `VOCAB_SIZE = 33`; `benchmark/evaluate.py`), so it is not a bucket.
- A short-window curriculum with `dynamic=True` (`short_first`). Already in the code; the catalog's 027 compiled a static graph per
  shape and reached 249 s. Not repeated.
- FlashAttention-3 called directly, without flex. The blog puts flex at roughly 60 percent of FA3 throughput on Hopper (as of its
  date). A direct FA3 path rewrites the mask and the flex call, which takes days; it is the fallback if idea 1 fails.
- XPU graphs (2.11 release). Intel GPUs only.
- Hardware note, not an idea with a number: the GH200 module power is programmable from 450 to 1,000 W across CPU, GPU and memory
  (vendor datasheet snippet, https://supermicro.com/datasheet/datasheet-NVIDIA-MGX-1U-GH200-Grace-Hopper-Systems.pdf). If the module
  is power-limited, host work during the clock can take GPU power. Measure GPU clocks with `nvidia-smi` during a run. No source
  reads a training trace of this.
- NorMuon row normalization is already in `optimizer.py` (lines 174 to 177; NorMuon, arXiv 2510.05491, read as its Algorithm 1
  through the fetch tool). Its extra pass is part of the non-matmul Muon share and is covered by idea 3.

## Sources

- PyTorch blog, FlexAttention + FlashAttention-4 (last updated 2026-03-04): https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
- PyTorch 2.11 release blog (last updated 2026-03-30): https://pytorch.org/blog/pytorch-2.11-release/
- torch.compile docs, 2.11: https://docs.pytorch.org/docs/2.11/generated/torch.compile.html
- FlexAttention docs, 2.11: https://docs.pytorch.org/docs/2.11/nn.attention.flex_attention.html
- FlexAttention paper (abstract only): https://arxiv.org/abs/2412.05496 (HTML: https://arxiv.org/html/2412.05496v1)
- Polar Express (Amsel, Persson, Musco, Gower): https://arxiv.org/abs/2505.16932 (HTML: https://arxiv.org/html/2505.16932v1)
- NorMuon (Li, Liu, Liang, Chen, Zhao): https://arxiv.org/abs/2510.05491 (HTML: https://arxiv.org/html/2510.05491v1)
- Liger Kernel: https://arxiv.org/abs/2410.10989
- Regional compilation recipe: https://docs.pytorch.org/tutorials/recipes/regional_compilation.html
- Compile caching recipes: https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html ;
  https://pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html
- PyTorch PR 97203, coordinate descent tuning upon max-autotune (2023; snippet from a mirror):
  https://app.semanticdiff.com/gh/strisunshinewentingwang/pytorch/commit/52ecc3274b1c16fcca3a3d89bd261dbc6513d6ed
- Flashlight (arXiv 2511.02043; snippet): https://arxiv.org/pdf/2511.02043
- FlashAttention-3 (arXiv 2407.08608; snippet): https://arxiv.org/abs/2407.08608
- Hugging Face Falcon-Perception attention file (snippet): https://huggingface.co/tiiuae/Falcon-Perception/blob/main/attention.py
- PyTorch forum, CUDA graphs with FSDP 2 and flex attention (snippet): https://discuss.pytorch.org/t/cuda-graphs-with-fsdp-2-and-flex-attention/223437
- NVIDIA MGX GH200 datasheet, module power range (snippet): https://supermicro.com/datasheet/datasheet-NVIDIA-MGX-1U-GH200-Grace-Hopper-Systems.pdf
- modded-nanogpt records 59, 87 and 92 (PRs 197, 322, 360): https://github.com/KellerJordan/modded-nanogpt (through `ideas/modded_nanogpt_2026.md`)
- This repository: `train.py`; `src/speedrunning_plms/strategies/attention.py`, `optimizer.py`, `curriculum.py`;
  `src/speedrunning_plms/benchmark/clock.py`, `evaluate.py`, `tokens.py`
