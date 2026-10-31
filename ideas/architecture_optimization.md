# Architecture and optimization ideas for a 10-minute 36x768 bidirectional MLM

The benchmark fixes depth 36, residual width 768, MLP intermediate width 2048, and 15 percent MASK-only masked language modeling with bidirectional attention. It runs for exactly 10 minutes of wall-clock time on one H100-class GPU (bf16, PyTorch 2.14, flex attention, torch.compile), selects weights by validation loss, and evaluates the test split at full length up to 2,048 tokens. Heads, attention variant, normalization, embeddings, optimizer, precision and schedule are free. Because the clock is wall-clock time, a method that cuts steps but adds per-step overhead can lose, so every entry lists overhead beside step gains. Most published evidence comes from causal decoder language models or from the modded-nanogpt speedrun record table, not from bidirectional MLM, and each entry says where a claim is untested for MLM. The families that look most promising are the optimizer and residual-stream changes with speedrun evidence (Polar Express, NorMuon, cautious weight decay, logit softcapping, U-net skips, value residuals), with the attention composition (Dual Triangle Attention with differential heads) worth building only after a measured flex-attention baseline; nGPT has the largest claimed gain but is untested for bidirectional MLM and carries a per-step overhead.

## How to read this file

- Each entry gives the source, the mechanism, the reported gain and setting, known failure modes, a transfer paragraph for our 36x768 MLM, an implementation cost in hours, and a risk rating.
- "Estimate" marks our own extrapolation from the cited numbers. Implementation hours are also our estimates, not figures from the literature.
- The speedrun record table in the modded-nanogpt repository lists record times in minutes and the record that introduced each change. Those times come from that repository's own benchmark, not from a 36x768 one-GPU 10-minute setting, so only relative sizes are informative. Percentages below are computed from the listed record times: (previous time minus new time) divided by previous time. Source: [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
- "Not found" means the literature search did not surface the item. It does not mean the item does not exist.
- Links marked (search snippet) were seen only in search results and not read in full.

---

## 1. nGPT: normalized transformer with representation learning on the hypersphere

**Source.** [nGPT, arXiv 2410.01131](https://arxiv.org/abs/2410.01131) (method read from the [HTML version](https://arxiv.org/html/2410.01131v2)).

**Mechanism.** Every embedding, weight row and hidden state is unit-norm. After each optimizer step the rows of the input and output embeddings and of W_q, W_k, W_v, W_o, W_u, W_nu and W_oMLP are normalized along the embedding dimension (optionally also in the forward pass). The block equations are:

- Attention: q <- Norm(q) * s_qk, k <- Norm(k) * s_qk, with the softmax temperature changed from 1/sqrt(d_k) to sqrt(d_k) as the paper states. Then h_A = Norm(ATTN(h)) and h <- Norm(h + alpha_A * (h_A - h)).
- MLP: u = (h W_u) * s_u, nu = (h W_nu) * s_nu * sqrt(d_model), SwiGLU(u, nu) = u * SiLU(nu), h_M = Norm(SwiGLU W_oMLP), and h <- Norm(h + alpha_M * (h_M - h)).
- Output: z = E h, then z <- z * s_z.
- alpha_A and alpha_M are per-dimension eigen learning rates. The paper treats them as non-negative in the main text. Appendix A.2 reports that unconstrained rates give essentially the same results.
- Each learnable vector is stored as a value initialized to its scale, and the forward pass multiplies it by init/scale, which changes the effective learning rate without changing the global one (Section 2.5 of the paper).
- Weight decay is 0 and warmup is 0 (Table 3 of the paper, versus 0.1 and 2000 steps for the GPT baseline). RMSNorm/LayerNorm layers are removed.

Defaults from Section 2.6 of the paper: s_qk init 1, scale 1/sqrt(d_model); s_u init 1, scale 1; s_nu init 1, scale 1; s_z init 1, scale 1/sqrt(d_model); alpha_A, alpha_M init 0.05, scale 1/sqrt(d_model). Appendix A.7 gives the default alpha init as 1/sqrt(d_model), which conflicts with the 0.05 in Section 2.6. The paper does not give a default for s_v in Section 2.6 (it appears in ablations only), so it is "not found".

**Reported gain and setting.** The paper reports 4x, 10x and 20x fewer tokens to reach the same validation loss at 1k, 4k and 8k context. Setting: OpenWebText, LLaMA-2 tokenizer (32k vocabulary), 0.5B and 1B parameter causal models, global batch 512 on 64 A100 GPUs. Headline example: the 1B model at 4k context reaches at 20k iterations the loss GPT reaches at 200k iterations. Per-step time is about 80 percent higher at 4k and about 60 percent higher at 8k, which the paper attributes to six normalizations per layer and normalization kernels that are not fused. Removing q/k normalization cuts step time by about 12 percent (0.657 to 0.576 s); SLERP instead of LERP adds about 10 percent.

Follow-ups. [Training nGPT, arXiv 2608.01284](https://arxiv.org/abs/2608.01284) (abstract read only) reports a recipe for nGPT on hybrid Mamba-2 and Transformer MoE models of up to 30B total parameters. Its abstract says the nGPT model reaches the same validation loss as an unnormalized AdamW model with about half the training tokens. The recipe components named are Logit Gradient Preconditioning, logarithmic learning-rate decay, GatedAdamW, angular update control and optional exploration mechanisms. No formulas or numbers were read beyond the abstract, and the paper does not say whether bidirectional models are used. A search for independent reproductions or critiques found none (the [SOTAVerified listing](https://sotaverified.org/papers/ngpt-normalized-transformer-with) marks the original as unverified). Bidirectional or MLM use of nGPT: not found.

**Known failure modes.** The step-count gain is at the cost of overhead that the authors did not fuse. The experiments are causal, at 0.5B to 1B parameters, and use context lengths of 1k to 8k, all longer than a likely 512 to 1024 training length for us. The 52B-token ablation budget is far smaller than the scale of the headline claims.

**Transfer.** The block equations do not depend on causality; only the attention mask changes, so a bidirectional flex mask can be dropped in. For d = 768, 1/sqrt(d) is about 0.036, which is the scale of s_z and of the alpha parameterization in the paper. The initial alpha value is 0.05 in Section 2.6, and the Appendix A.7 default is the 1/sqrt(d) scale; this must be resolved against the released code before use. The main risk for our benchmark is wall-clock: the paper's own overhead is 60 to 80 percent per step, and our short sequences leave less token gain to offset it. Estimate: if the overhead is removed by torch.compile fusion, the wall-clock gain could be positive; if it is not removed, the net gain may be near zero or negative at our lengths. This must be measured with a 10-minute pilot.

How one could start on the hypersphere and relax the constraint during a run (our proposal, not in the literature): keep the normalized block as it is for the first part of the run, then interpolate toward an unnormalized residual, for example h <- (1 - tau(t)) * h_unnorm + tau(t) * Norm(h_unnorm) with tau rising from 0 to 1 over the second half. The only literature support is Appendix A.2 (unconstrained eigen learning rates give the same results) and the "angular update control" named in the 2026 follow-up without formulas. Untested.

**Cost.** 8 to 16 hours (block implementation, scale parameterization, a pilot grid over alpha and s_z, and a fused-kernel check).

**Risk.** High. Large claimed gain, but untested for MLM, overhead-sensitive, and the paper's own defaults conflict.

---

## 2. Differential Transformer V1 and V2

**Source.** [Differential Transformer, arXiv 2410.05258](https://arxiv.org/abs/2410.05258) (method read from the [HTML version](https://arxiv.org/html/2410.05258v2)). [Diff-Transformer-V2 README](https://github.com/microsoft/unilm/tree/master/Diff-Transformer/Diff-Transformer-V2) (raw README read). The V2 blog post, reported in search results as a Hugging Face post from January 2026 with a Notion mirror, could not be read; the Notion page returned no article text.

**Mechanism (V1).** Project X to Q1, Q2, K1, K2 (each dimension d) and V (dimension 2d):

DiffAttn(X) = (softmax(Q1 K1^T / sqrt(d)) - lambda * softmax(Q2 K2^T / sqrt(d))) V

lambda = exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init, with four learnable vectors in R^d per layer. lambda_init depends on the layer index l: lambda_init = 0.8 - 0.6 exp(-0.3 (l - 1)). Each head's output is scaled by (1 - lambda_init) after a per-head normalization. The text says the normalization is RMSNorm, but Figure 2 and the ablation use GroupNorm, which is an inconsistency in the paper. Head count h = d_model / (2d), with d = 128 in the experiments.

**Mechanism (V2, README).** Query heads are 2h and KV heads stay h (GQA). Q has shape (N, 2h, d), K and V have shape (N, h_kv, d), and lambda has shape (N, h, 1), computed per token and per head from the input, passed through a sigmoid. Query heads are split into even and odd index groups, not contiguous halves. One call to standard flash attention produces an output for all 2h query heads. The output for head pair i is attn_even(i) - sigmoid(lambda(i)) * attn_odd(i). The two heads in each pair must share the same K and V (same GQA group). The per-head RMSNorm of V1 is removed because the authors found it caused instability late in large-scale pretraining. The README reports no benchmark numbers; it gives design claims about decoding speed and no custom kernel.

**Reported gain and setting (V1, decoder-only only).** Zero-shot average on downstream tasks: Diff-3B 60.6 at 1T tokens versus 57.5 for OpenLLaMA-3B-v2 (Table 1). Controlled 3B comparison at 350B tokens: zero-shot 56.2 versus 55.4 (Table 8). Scaling: a 6.8B Diff model matches an 11B Transformer with 62.2 percent of the parameters; a 3B Diff model at 160B tokens matches a Transformer at 251B tokens. Multi-needle retrieval at 4K (N=6, R=2): 0.85 versus 0.55. The paper reports no encoder or masked-language-modeling experiment.

**Known failure modes.** V1 needs a custom kernel and a per-head normalization that the V2 README reports as unstable at scale. V2 has no published benchmark in the README. Query-head doubling doubles the QK^T and AV work per layer relative to a baseline with the same KV heads, so the attention share of the step time sets the overhead.

**Transfer.** Use V2, not V1. In flex attention it is a full-mask attention call with 2h query heads over h KV heads, followed by an elementwise combination of the even and odd outputs. The paired heads must share K and V, which fixes the GQA grouping. Parameter cost is one doubled query projection (768 to 1536 outputs per layer). Attention FLOPs double for the score and value products, so the estimated step-time cost is about the attention share of the step, which we must measure. No gain for bidirectional MLM is reported anywhere we read; nothing in the mechanism is causal-specific, so the transfer is plausible but unmeasured.

**Cost.** 3 to 5 hours (projection change, paired combine, flex call, correctness test against a reference implementation).

**Risk.** Medium. Simple to implement, but the gain is reported for decoder LMs at 3B and above and the overhead is not measured for our width.

---

## 3. Dual Triangle Attention (DTA) and composition with differential attention

**Source.** [Dual Triangle Attention, arXiv 2604.18603](https://arxiv.org/abs/2604.18603) (method read from the [HTML version](https://arxiv.org/html/2604.18603v1)).

**Mechanism.** Each logical head with dimension d_h is split into two sub-heads of dimension d_h/2. Of the 2n sub-heads, the first n are "down" (past and self) and the last n are "up" (future and self):

mask(h, i, j) = (i >= j) if h < n (down), and (i <= j) if h >= n (up). The diagonal is included in both triangles.

The flex implementation is a single mask_mod: (is_down AND q >= k) OR (NOT is_down AND q <= k). It runs one call, create_block_mask(mask_mod, b, 2n, l, l) followed by flex_attention(Q, K, V, block_mask=M), then reshapes the output back to n heads and applies W_O. No parameters are added beyond standard MHA. RoPE is applied to the full d_h-dimensional Q and K before the split, with base 10,000. The paper states the attention costs about half the FLOPs of standard bidirectional attention.

Inconsistency to resolve before implementing: the text says each sub-head gets an independent softmax, but Algorithm 1 applies one softmax over the combined logit matrix with full-dimensional V. The flex algorithm computes per-sub-head softmax, so we should follow the flex form and test it against a reference.

**Reported gain and setting.** Masked-language-modeling runs with 12 layers, 768 hidden, 64-dimension baseline heads (the paper's DTA heads are 128 wide before splitting), trained at length 256 and evaluated at 256 and 1,024. NLP (FineWeb-Edu, 4,096 BPE vocabulary, 1B tokens, three seeds):

- Training length 256, validation loss: DTA+RoPE 1.36 (acc 0.709), bidirectional+RoPE 1.38 (acc 0.702), causal+RoPE 3.16 (acc 0.418). The paper gives p = 0.68 for DTA versus bidirectional in Results and p = 0.59 in Discussion.
- Extended length 1,024, test: DTA+RoPE loss 1.53 (acc 0.677), bidirectional+RoPE 1.81 (acc 0.636).

Protein MLM (OMG-Prot50, ESM-2 tokenizer): at training length DTA+RoPE accuracy 0.248 versus 0.252 for bidirectional+RoPE, so no gain. At extended length, test loss 2.52 versus 2.53. The NLP extended-context gain is the only clear gain in the paper.

The paper's RoPE-off variant (RoPE for the first 70 percent of tokens, none for the last 30 percent) hurt both models in this setup: NLP DTA loss rose from 1.53 to 4.05 at 1,024, and protein DTA loss rose from 2.52 to 2.64. The paper reports no formal ablation of the split or head count.

**Known failure modes.** The softmax ambiguity above. Training length 256 is short. No YaRN or PI scaling was used. The NLP gain is at 4x length only. The paper's results come from one architecture size.

**Transfer.** DTA is expressible in one flex call on our side because mask_mod receives the head index, so the down or up choice is a per-head predicate. For d_h = 64 the sub-heads are 32 wide. Matching the paper's parameter and FLOP accounting needs a choice of head count that we would have to measure. Expected gain at our length: the NLP extended-context gain suggests a benefit when evaluating at 2,048 after training at a shorter length; the protein MLM result suggests no gain at the training length. Estimate: small to moderate and uncertain for our benchmark.

**Composition with differential attention.** One flex call can express both, because mask_mod and score_mod receive the head index: a logical head becomes four sub-heads (two differential branches, each split into down and up), with the down or up choice and the differential role decided by head index. The open design question is the value dimension and the pairing. Differential subtraction combines outputs of two softmax maps that share V; DTA splits V with the sub-heads, so the pairing must put each differential partner on the same V half. This layout is not verified and must be tested against a reference using random tensors. We found no paper that combines the two.

**Cost.** 6 to 10 hours for DTA alone (mask_mod, RoPE placement, correctness test, speed test against the full-mask baseline). Add 4 to 6 hours to compose with differential attention.

**Risk.** Medium to high. The protein evidence shows no gain at training length, and the composition is untested.

---

## 4. Context extension for RoPE in bidirectional models

**Why it matters here.** Training may use lengths shorter than 2,048 while evaluation runs at full length. Each method below is tied to that split.

**YaRN.** [YaRN, arXiv 2309.00071](https://arxiv.org/abs/2309.00071) (read from the [HTML version](https://arxiv.org/html/2309.00071v2)). Wavelength of dimension d is lambda_d = 2 pi b^(2d/|D|), with ratio r(d) = L/lambda_d, where L is the original context length. A ramp gamma(r) is 0 for r < alpha, 1 for r > beta, and (r - alpha)/(beta - alpha) otherwise. The modified frequency is h(theta_d) = (1 - gamma(r(d))) theta_d / s + gamma(r(d)) theta_d, with s = L'/L. Attention is scaled by a temperature with sqrt(1/t) = 0.1 ln(s) + 1, applied as a scale on q and k. Recommended alpha = 1 and beta = 32 for LLaMA-family models. Fine-tuning used 400 steps for s = 16 and 200 more for s = 32, at learning rate 2e-5 on PG19, for Llama 2 7B and 13B. All experiments are causal; encoder or bidirectional models were not evaluated.

**Position interpolation.** [Position Interpolation, arXiv 2306.15595](https://arxiv.org/abs/2306.15595) (read from the [HTML version](https://arxiv.org/html/2306.15595v2)). The position index is scaled: f'(x, m) = f(x, mL/L'). The paper fine-tunes for 1,000 steps. All experiments are LLaMA decoder models. Direct extrapolation past 2,048 pushed perplexity above 10^3, while PI kept it low.

**ModernBERT's base and window pattern.** [ModernBERT, arXiv 2412.13663](https://arxiv.org/abs/2412.13663) (read from the [HTML version](https://arxiv.org/html/2412.13663v2)). Pretraining at length 1,024 with RoPE base 10,000 (1.719T tokens). Context extension to 8,192 at global RoPE base 160,000 with 250B tokens (phase one) and 50B tokens (phase two, 1-sqrt decay). Every third layer is global; the other layers use a 128-token sliding window with RoPE base 10,000. The text says the alternating pattern matched all-global attention at 100B tokens with "major speedups"; no figures are given in the text we read. The 300B-token extension is far beyond a 10-minute run, but the base and the window pattern are transferable.

**Dynamic YaRN in training (speedrun).** The modded-nanogpt record table lists "Dynamic YaRN in training and validation" as record #31 (September 2025), with time 2.656 minutes against 2.717 for the previous record (2.2 percent). It is paired with window-size warmup in the same repository. The RoPE base and window lengths are not stated in the table. Source: [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). Attention window warmup is record #13: 4.66 versus 5.03 minutes (7.4 percent).

**DroPE.** [Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings, arXiv 2512.12167](https://arxiv.org/abs/2512.12167) (abstract read; the paper was accepted to ICLR 2026). It pretrains with RoPE, then removes positional embeddings and recalibrates briefly at the original length. The authors report zero-shot extension with base-context performance preserved, on models up to 7B and trillions of tokens. A company page claims recalibration costs under 1 percent of pretraining; this is not verified. Our own evidence against it is DTA's RoPE-off experiment, described above, which hurt.

**NoPE.** [Kazemnejad et al., arXiv 2305.19466](https://arxiv.org/abs/2305.19466) (read from the [HTML version](https://arxiv.org/html/2305.19466v1)). Decoder-only models without positional encoding generalize better to longer inputs on ten reasoning and mathematical tasks, at about 107M parameters. The argument rests on the causal mask giving position information. The paper studies only decoder-only models. Its background says encoder-only models without positional encoding become bag-of-words models, which is a claim and not an experiment. We should not use NoPE for bidirectional MLM without a pilot.

**RoPE base tuning for bidirectional MLM.** Only ModernBERT (base 160,000 for global layers) and DTA (base 10,000) report bases for bidirectional models. A systematic base sweep for bidirectional MLM: not found.

Transfer. Train at 512 to 1,024, evaluate at 2,048. Options in order: (a) a dynamic YaRN-style frequency ramp during training, the speedrun's choice; (b) a global base increase with a sliding-window local layer every third block, which is a mask_mod in flex; (c) a DroPE-style RoPE-off phase only after a pilot. Estimate for (a): a small gain of about 1 to 3 percent in our setting, from the speedrun number, not measured for MLM. The cost is 1 to 2 hours for (a) and 2 to 4 for (b).

**Risk.** Low for (a) and (b), high for (c).

---

## 5. Residual stream and skip designs

**Hyper-connections.** [Hyper-Connections, arXiv 2409.19606](https://arxiv.org/abs/2409.19606) (read from the [HTML version](https://arxiv.org/html/2409.19606v3)). The residual is widened to n streams, H in R^(n x d). The paper's matrices are A_m (n x 1), B (1 x n) and A_r (n x n); the layer reads h0 = A_m^T H, computes the layer once on h0, and writes back H_out = B^T (layer output)^T + A_r^T H. The paper uses n = 4 by default. Static variant: A_m, B and A_r are learned directly. Dynamic variant: each is adjusted by an input-dependent term, for example B(H) = s_beta * tanh(norm(H) W_beta)^T + B, with W_beta in R^(d x n) initialized to zero and small scales initialized to 0.01. Initialization: B is a row of ones and A_r is the identity, so the block matches a Pre-Norm residual at start.

Results: OLMo-1B at 500B tokens, validation loss 2.811 for the baseline versus 2.781 for DHC x4 with tanh; downstream average 62.5 versus 63.8. OLMo-7B loss 2.581 to 2.559. OLMoE loss down about 0.028. ViT-Large on ImageNet 77.25 to 79.94 with dynamic HC. Cost: parameters +0.03 percent, FLOPs about +0.2 percent, activation memory +26.1 percent at 1B (recomputation reduces it). n = 1 is worse than baseline; gains appear at n >= 2.

Transfer: start with static HC at n = 2 or 4, using the Pre-Norm-equivalent initialization. For 36 blocks at width 768, the memory overhead is the main cost; it reduces the batch size we can run in 10 minutes, which must be measured. Estimate: loss reduction of about 0.03 at the paper's scale, uncertain at ours. Cost 4 to 6 hours. Risk medium.

**mHC (manifold-constrained hyper-connections).** [mHC, arXiv 2512.24880](https://arxiv.org/abs/2512.24880) (read from the [HTML version](https://arxiv.org/html/2512.24880v2)). The residual mixing matrix H_res is constrained to be doubly stochastic by Sinkhorn-Knopp: M0 = exp(H_res), then alternating row and column normalization for t_max = 20 iterations. H_pre = sigmoid(...) and H_post = 2 sigmoid(...). Models: DeepSeek-V3-style MoE at 3B, 9B and 27B. At 27B, mHC reduces loss by 0.021 relative to the baseline, and HC shows a loss surge around step 12k. Overhead is 6.7 percent at n = 4 with the authors' infrastructure. No replication found. Transfer: optional after HC; the Sinkhorn loop adds per-layer work. Cost 6 to 8 hours. Risk medium-high.

**MUDD (multiway dynamic dense connections).** [MUDDFormer, arXiv 2502.12170](https://arxiv.org/abs/2502.12170) (read from the [HTML version](https://arxiv.org/html/2502.12170v2)). Each layer receives a depth-wise aggregate of the embedding and all earlier block outputs: X_bar_i = sum over j <= i of A_ij * X_j, with per-position weights A_i = G(X_i) where G(X_i) = GELU(RMSNorm(X_i) W1) W2 + a_i (W1 in R^(D x (i+1)), W2 in R^((i+1) x (i+1)), a_i in R^(i+1)). Four streams (query, key, value, residual) each get their own generator. Initialization: W2 = 0 and a_ii = 1, so the model is a Transformer at start. Parameter-allocation: the FFN width grows linearly with depth at constant total parameters. Overhead: parameters 0.23 percent at 2.8B; FLOPs 0.4 percent; memory 22 percent; training throughput 84.0 percent of baseline at 2.8B in their untrained implementation, which the authors attribute to small ops and left for fusion.

Results: at 2.8B on the Pile, perplexity 6.63 (Pythia) versus 6.29 (MUDDPythia), with MUDDPythia-2.8B matching Pythia-6.9B. Ablation at 405M: a dynamic multiway generator gives 10.83 versus 11.68 for Transformer++ (Pile perplexity, lower better). Removing the value stream's dense connections costs the most.

Speedrun evidence: record #81, "MUDD skip connections" (April 2026), 1.363 minutes versus 1.406 for the previous record (3.1 percent). The table does not say which MUDD variant was used.

Transfer: a 36-deep dense aggregation is 36 weight generators at 768 width, which is small. Start from the W2 = 0 initialization; measure throughput before deciding. Cost 6 to 10 hours. Risk medium-high, because the throughput loss in the paper's implementation could cancel the step gain at our size.

**ResFormer and SVFormer (value residual).** [Value Residual Learning, arXiv 2410.17897](https://arxiv.org/abs/2410.17897) (read from the [HTML version](https://arxiv.org/html/2410.17897v5)). Eq. 5: V_n = lambda_{n,1} V_1 + lambda_{n,2} H_{n-1} W^V_n, where V_1 is the first layer's value and H_{n-1} is the previous hidden state. Learnable-ResFormer: both lambdas are scalars initialized to 0.5 and trained. Learnable-ResFormer Plus: lambda_{n,2} starts at 0.5 and lambda_{n,1} is a softmax split of a shared scale initialized to the layer count. Dense-ResFormer: V_n uses all earlier values with learned weights. SVFormer: layers n >= 2 use V_1 only, nearly halving the KV cache.

Results: 8-layer, 82M model, validation loss 2.739 (Transformer) versus 2.705 (Learnable-ResFormer) and 2.681 (Learnable-ResFormer Plus). At 82M to 468M, ResFormer reaches Transformer's loss with 16.11 percent fewer parameters; at 468M it matches the loss with 20.3 percent fewer training tokens. Memory and compute are similar to the baseline. SVFormer needs about 12.2 percent more parameters to match, but nearly halves the KV cache; its gain depends on sequence length and learning rate. Sharing keys hurts more than sharing values (Table 1).

Speedrun evidence related to this: value embeddings (record #14, 4.66 to 4.41 minutes, 5.4 percent) and U-net value embeddings (record #15, 4.41 to 3.95 minutes, 10.4 percent). These are a separate design and may overlap with the value residual, so test them one at a time.

Transfer: Learnable-ResFormer with lambda initialized at 0.5. Keep V_1 for the whole stack (one extra tensor per batch), and add one scalar-weighted mix per layer. Cost 2 to 3 hours. Estimate: a small gain, probably below 1 percent of loss at our scale, which is consistent with the paper's 82M result. Risk low.

**U-net skip connections (speedrun).** The modded-nanogpt record table lists "U-net pattern skip connections" as record #11 (November 2024), 7.8 to 7.2 minutes (7.7 percent). The table does not give the exact connection form, so we must design it: the first half of the blocks feeds the second half through learned weights. Cost 1 to 2 hours. Risk low.

**Depth-scaled or exponential-decay initialization.** Not found as a 2024 to 2026 result showing faster convergence. Nearest matches: depth-scaled initialization from 2019 ([DS-Init, arXiv 1908.11365](https://arxiv.org/abs/1908.11365), +1.1 BLEU on 12-layer models, older than the scope); [SpanNorm, arXiv 2601.22580](https://arxiv.org/abs/2601.22580) (search snippet: scales FFN and attention output variance by 1/L and reports that 24-layer training collapses without it; not read in full); and [Optimal Scaling Needs Optimal Norm, arXiv 2510.03871](https://arxiv.org/abs/2510.03871) (search snippet: compares a total-depth gain of 1/sqrt(2N) on output projections; its convergence results were not read).

**How a 36-deep network trains fast with these.** The residual-mixing designs (HC, MUDD, value residuals, U-net skips) shorten the gradient path from late layers to early ones and let each block see earlier representations directly. Our evidence for that mechanism is the speedrun records and the paper results above; no convergence-rate measurement at 36 layers was found.

---

## 6. Optimizers

**Muon.** [Muon, Keller Jordan's writeup](https://kellerjordan.github.io/posts/muon/) (read). Nesterov-style momentum, then Newton-Schulz orthogonalization of the update. Default of 5 steps with coefficients (3.4445, -4.7750, 2.0315). Input is normalized by its Frobenius norm and cast to bf16. Applied to 2D hidden weights only; embeddings, the output head and scalar or vector parameters use AdamW. The writeup reports a NanoGPT speedrun improvement of 1.35x and a 1.5B-parameter run with about 10 8xH100-hours for Muon versus 13.3 hours for AdamW. The speedrun record table lists "Muon introduced" as record #3 (October 2024): 31.4 to 24.9 minutes (21 percent).

**Polar Express.** [Polar Express, arXiv 2505.16932](https://arxiv.org/abs/2505.16932) (read from the [HTML version](https://arxiv.org/html/2505.16932v3)). Each iteration applies an odd degree-5 polynomial p_t(x) = a_t x + b_t x^3 + c_t x^5 to the singular values; in matrix form X <- a X + (b A + c A^2) X with A = X X^T. Coefficients are chosen per iteration by a minimax (Remez-style) fit on the current interval, with the bounds updated as l_{t+1} = p_t(l_t) and u_{t+1} = 2 - l_{t+1}. Algorithm 1 table (a, b, c) before scaling, first rows: (8.28721201814563, -23.595886519098837, 17.300387312530933), (4.107059111542203, -2.9478499167379106, 0.5448431082926601), (3.9486908534822946, -2.908902115962949, 0.5518191394370137), and the last row repeats (1.875, -1.25, 0.375). Each non-final polynomial is divided by a safety factor 1.01 (a/1.01, b/1.01^3, c/1.01^5). The input is normalized as M / (||M||_F + 1e-2), and the lower bound is set to 1e-3. The paper recommends degree 5 with 5 to 6 iterations. Per-iteration cost is the same as Newton-Schulz. Results: GPT-2 Large (774M parameters, 1B FineWeb tokens) final validation loss 3.340 for Muon with Polar Express versus 3.398 for Muon with Jordan's coefficients and 4.172 for AdamW; GPT-2 124M: 3.588 versus 3.639. Speedrun record #38 (September 2025) replaced Newton-Schulz: 2.483 to 2.476 minutes (0.3 percent).

Transfer: drop-in for the Newton-Schulz call with the paper's coefficient table; keep the safety factor and the lower bound. Cost 1 to 2 hours. Risk low.

**NorMuon.** [NorMuon, arXiv 2510.05491](https://arxiv.org/abs/2510.05491) (method read from the [HTML version](https://arxiv.org/html/2510.05491v1)). Per step: M_t = beta1 M_{t-1} + (1 - beta1) G_t; O_t = NS5(M_t); v_t = beta2 v_{t-1} + (1 - beta2) mean over columns of (O_t elementwise squared), a vector with one entry per row (neuron); O_hat = O_t / (sqrt(expand(v_t)) + eps), row-wise; scale eta_hat = 0.2 eta sqrt(mn) / ||O_hat||_F; W <- W - eta lambda W - eta_hat O_hat. Row normalization after orthogonalization performed better than before. Hyperparameters for the modded-nanogpt runs: Muon and NorMuon learning rate 3.6e-4 at 124M and 7.5e-4 at 350M; beta2 = 0.95 for NorMuon.

Results: on 1.1B, a 21.74 percent reduction in steps to Adam's final loss (Muon 10.43 percent); on 5.4B, 13.91 percent (Muon 6.08 percent). Modded-nanogpt: Muon needs about 6 percent more iterations than NorMuon at 124M and about 15 percent more at 350M. Memory: optimizer state about half of AdamW's (21.19 GB versus 40.56 GB at 5.4B, distributed); step time +2.9 percent versus AdamW's baseline in their distributed setup. Speedrun record #41 (October 2025): 2.358 to 2.345 minutes (0.55 percent).

Transfer: replace the Muon update on hidden 2D weights. Single-GPU implementation is straightforward (no distributed split needed). Cost 2 to 3 hours. Risk low to medium.

**Cautious optimizers.** [Cautious Optimizers, arXiv 2411.16085](https://arxiv.org/abs/2411.16085) (read from the [HTML version](https://arxiv.org/html/2411.16085v1)). The update is masked per coordinate: m = (u * g > 0), and the step is p <- p - lr * u * m / (mean(m) + eps), where u is the optimizer's update and g is the gradient (Algorithm 1). Algorithm 2 (C-AdamW) instead rescales the learning rate by d / (||phi||_0 + 1). Weight decay is applied after the masked update and not masked. Results (LLaMA on C4, 10k steps, validation perplexity): 1B AdamW 24.02 versus C-AdamW 22.00; 350M 22.58 versus 22.14; 60M 31.17 versus 30.78. The paper reports 1.47x sample efficiency for C-AdamW at 1B, calls its evaluation preliminary, and gives theory for convex cases only. Speedrun records: "Cautious weight decay (with schedule)" as record #43 (November 2025): 2.313 to 2.284 minutes (1.25 percent); "extended to Adam params" as record #50: 2.146 to 2.128 (0.84 percent).

Transfer: apply the mask to the AdamW parameters, and to Muon's update if the masking is consistent with the orthogonalized direction (untested). Cost less than 1 hour. Risk low.

**AdEMAMix.** [AdEMAMix, arXiv 2409.03137](https://arxiv.org/abs/2409.03137) (read from the [HTML version](https://arxiv.org/html/2409.03137v2)). Updates: m1 = beta1 m1 + (1 - beta1) g; m2 = beta3 m2 + (1 - beta3) g; nu = beta2 nu + (1 - beta2) g^2; theta <- theta - eta ((m1_hat + alpha m2) / (sqrt(nu_hat) + eps) + lambda theta). Warmup of the mixing coefficient alpha linearly to its target over T steps, and of beta3 with a log-space schedule from beta1 to its target (the paper defines D_t = (1 - t/T) ln(beta3) + (t/T) ln(beta1)). Defaults: beta3 = 0.9999, alpha in [4, 10] (5, 8 or 10 by model). Results: abstract only, which states that a 1.3B model trained on 101B tokens performs like AdamW trained on 197B tokens. Token-level loss numbers were not read. The paper also reports slower forgetting.

Transfer: two EMAs add one state tensor per parameter (memory). Cost 2 hours. Risk medium: the warmup of beta3 is critical and the paper's own tuned values for the LLM runs are in an appendix we did not read.

**MARS.** [MARS, arXiv 2411.10438](https://arxiv.org/abs/2411.10438) (read from the [HTML version](https://arxiv.org/html/2411.10438v4)). Corrected gradient c_t = grad f(x_t; xi_t) + gamma_t (beta1/(1 - beta1)) [grad f(x_t; xi_t) - grad f(x_{t-1}; xi_t)]. Clip c_t to norm 1. Momentum m_t = beta1 m_{t-1} + (1 - beta1) c_t. MARS-AdamW: v_t from the clipped corrected gradient; x <- x - eta (m_hat / (sqrt(v_hat) + eps) + lambda x). MARS-Lion: sign of m_t. MARS-approx reuses the previous batch's gradient for the second term, avoiding a second gradient evaluation. Results (GPT-2 on OpenWebText): GPT-2 large, AdamW needs about 50B tokens to reach validation loss 2.58, MARS about 28B. Best validation loss: MARS-AdamW 2.511, MARS-Lion 2.534, AdamW 2.568, Lion 2.565, Muon 2.606. Default gamma and learning rate: not found in the text we read (Appendix F).

Transfer: MARS-approx is the practical form, with a memory cost of one previous-gradient tensor. Cost 2 to 3 hours. Risk medium: independent reproduction not found, and the "Fantastic optimizers" study below suggests claimed gains over tuned AdamW shrink with scale.

**Schedule-Free.** [Schedule-Free, arXiv 2405.15682](https://arxiv.org/abs/2405.15682) (read from the [HTML version](https://arxiv.org/html/2405.15682v4)). Three sequences z, y, x: y = (1 - beta) z + beta x; z <- z - gamma grad f(y); x <- (1 - c) x + c z with c = 1/t in the SGD form. Schedule-Free AdamW (Algorithm 1): v_t = beta2 v + (1 - beta2) g^2; gamma_t = gamma sqrt(1 - beta2^t) min(1, t / T_warmup); z <- z - gamma_t g/(sqrt(v) + eps) - gamma_t lambda y; c = gamma_t^2 / sum gamma_i^2. Linear warmup is still required. NanoGPT needed beta = 0.98 because beta = 0.9 diverged. Evaluation covers 28 problems, with Schedule-Free AdamW winning the MLCommons 2024 AlgoPerf self-tuning track. Only one language-model experiment (NanoGPT, 124M) was reported; no formal non-convex guarantee.

Transfer: removes the need to fix total steps, which is useful under a fixed wall-clock budget where the total step count is only known after measuring throughput. Cost 1 to 2 hours. Risk medium: learning rate and weight decay need re-tuning, and the LLM evidence is one small run.

**Warmup-stable-decay and cooldown.** [WSD, arXiv 2404.06395](https://arxiv.org/abs/2404.06395) (read from the [HTML version](https://arxiv.org/html/2404.06395v3)). Three stages: warmup, stable at the maximum rate, then decay by a decreasing factor f. On 0.036B models, a decay over about 10 percent of tokens gave the best loss; 2.5 percent fell short. The MiniCPM runs used exponential annealing. The paper does not compare decay shapes (linear, sqrt, exponential) in its experiments we read. Speedrun record #26, "Cooldown fraction raised to 0.45" (July 2025): 2.896 to 2.863 minutes (1.1 percent). Record #5 in track 2 also lengthened the cooldown with no number given.

Transfer: our 10-minute clock makes the schedule a function of elapsed time, so the decay phase is a fraction of the time budget. Estimate: 10 to 45 percent of the time budget in decay, following the speedrun's 0.45 value as the upper point. Cost under 1 hour. Risk low.

**Which optimizers have a single-GPU PyTorch implementation.** Muon, NorMuon, Polar Express, cautious masks, AdEMAMix, MARS-approx and Schedule-Free are all element-wise or matrix-level updates that fit in one process; the NorMuon distributed split is not needed on one GPU. Reference code for Muon and modded-nanogpt exists in the repositories above; we did not check licences.

**Fair comparison caveat.** [Fantastic Pretraining Optimizers and Where to Find Them, arXiv 2509.02046](https://arxiv.org/abs/2509.02046) (abstract read from the [ICLR 2026 poster record](https://iclr.cc/virtual/2026/poster/10011758) and the arXiv abstract). Ten optimizers compared at 0.1B to 1.2B parameters, with tuning per optimizer. Well-tuned alternatives are faster than the 1.4 to 2x claimed, but the speedup shrinks with scale. Muon and Soap speed up from about 1.4x at 0.1B to about 1.1x at 1.2B. Our 36x768 stack has about 254M non-embedding parameters (12 L d^2). A log-linear interpolation between the two reported points gives about 1.3x (estimate, not from the paper), so expect a smaller gain than the claims. The paper also says intermediate checkpoints can mislead, so compare at the end of the run.

**ANVIL.** Not found on arXiv. A social-media post describes an optimizer by that name, but it is a self-report with no paper, so it is excluded from the ranking.

---

## 7. The softmax and cross-entropy fix, and related logit control

**Source.** [arXiv 2610.31642](https://arxiv.org/abs/2610.31642) (Barron, Gnanapragasam, Tschannen). Only the abstract was readable; the HTML version returned only the abstract text.

**Mechanism (as stated in the abstract).** Two defects in the gradients of softmax and cross-entropy. Softmax collapse: when the target logit is much larger than the others, its gradient rounds to zero, so the gradients no longer sum to zero. Softmax cancellation: with student and teacher logits close, the loss still produces non-zero gradients when they match. The fix is described as simple corrected implementations of log-softmax, softmax and cross-entropy. The formulas were not in the text we read. The abstract reports that the corrected loss speeds grokking with weight decay and prevents accuracy collapse without weight decay, with no numbers. It also states that after memorization, logit growth is bounded by an equilibrium between weight decay and Adam's epsilon. Failure modes and experiment sizes: not found in the text we read.

**Related logit control.**

- Z-loss, [PaLM, arXiv 2204.02311](https://arxiv.org/abs/2204.02311) (read from the [HTML version](https://arxiv.org/html/2204.02311v5)): an auxiliary term 1e-4 * (log Z)^2, where Z is the softmax normalizer, pushing log Z toward 0. The paper says it increases training stability; no logit-growth numbers, and no side effects are stated.
- Logit soft-capping, [Gemma 2, arXiv 2408.00118](https://arxiv.org/abs/2408.00118) (read from the [HTML version](https://arxiv.org/html/2408.00118v3)): logits <- soft_cap * tanh(logits / soft_cap), with soft_cap 50 for attention logits and 30 for final logits. The paper gives no stability ablation. Speedrun evidence: "logit softcap" introduced in record #9 (November 2024); lowering the cap from 30 to 15 in record #18 (January 2025) took the record from 3.57 to 3.40 minutes (4.8 percent).

Transfer. Softcap on the output logits, cap 15 to 30, is a one-line change with speedrun evidence; cost under 1 hour; risk low. The softmax fix matters only if the target-logit gradient rounds to zero in our bf16 cross-entropy. The test is cheap: measure the fraction of target-logit gradients that are exactly zero in bf16 during the first minutes of a pilot. Cost 1 to 2 hours for the corrected cross-entropy and the measurement; risk low for the test, unknown for any gain, since the paper's evidence is grokking, which a 10-minute run is unlikely to reach.

---

## 8. Auxiliary objectives that could transfer to MLM

**Multi-token prediction.** [Gloeckle et al., arXiv 2404.19737](https://arxiv.org/abs/2404.19737) (read from the [HTML version](https://arxiv.org/html/2404.19737v1)). A shared trunk feeds n independent heads, each predicting the token i steps ahead, sharing one unembedding matrix. Loss L_n = -sum over t of log P(x_{t+n:t+1} | x_{t:1}), factored over a shared latent. Results: 13B code models solve 12 percent more HumanEval and 17 percent more MBPP problems than next-token baselines; 7B code models at n = 4 gain 3.8 percent MBPP and 1.2 percent HumanEval; natural-language gains are small and n = 4 regresses slightly. Training time is 1.02x to 1.22x. Speedrun record #53 (December 2025) lists multi-token prediction at 2.075 to 1.988 minutes (4.2 percent). The paper covers only causal models and does not discuss bidirectional or masked objectives.

Transfer: not directly applicable to MLM. The analogue (predicting the tokens near a masked position) has no evidence in the sources we read. Cost about 2 hours. Risk medium. Expected gain low.

**Multiple masking rates and sequence completion.** [Ankh3, arXiv 2505.20052](https://arxiv.org/abs/2505.20052) (read from the [HTML version](https://arxiv.org/html/2505.20052v1)). A T5-style encoder-decoder, trained with masked-language-modeling at rates 15, 20 and 30 percent sampled uniformly per sequence, and with sequence completion (first 50 percent as input, the rest generated). Each sequence is assigned to one task. The paper does not report a separate ablation of the multiple rates, and it reports no loss formula. Ankh3-Large (1.88B) and Ankh3-XL (5.73B) trained on UniRef50 with 59M sequences. Our benchmark fixes the masking rate at 15 percent, so using 20 or 30 percent rates changes the objective. This needs a ruling against the benchmark rules before any test.

**Neighbor prediction.** Not found for protein language models in the search we ran.

**Prefix-token prediction.** Not found in the modded-nanogpt table we read.

**Denoising at multiple rates in protein models beyond Ankh3.** A search snippet describes discrete diffusion protein language models that vary the masking ratio by a schedule ([search snippet, not read](https://arxiv.org/pdf/2605.27413)). Not verified.

**Sparse-attention gate.** Speedrun record #28 (August 2025) adds a sparse attention gate: 2.817 to 2.812 minutes (0.2 percent). Negligible on that evidence.

**Contrastive objectives.** Not searched in depth; no evidence gathered.

---

## 9. Precision, kernels and single-GPU systems

**FP8 linear layers with torchao.** [torchao](https://github.com/pytorch/ao) (read from the README). The entry point is convert_to_float8_training from torchao.float8, applied to the model; the README reports results with torch.compile enabled and up to 1.5x throughput at the 405B scale on up to 512 GPUs. Blog posts in search results report rowwise training at 1.34 to 1.43x at 2k H100 scale; these are not read in full. Speedrun FP8 records: FP8 head with offset logits (record #19, January 2025) 3.4 to 3.142 minutes (7.6 percent); FP8 MLP up-projection forward (record #84, May 2026) 0.6 percent; FP8 MLP down-projection with delayed scaling (record #89) 2.1 percent; FP8 MLP backward and FP8 QKV forward and backward (record #90) 1.23 to 1.13 minutes (8.1 percent). Bf16 cross-entropy (record #37): 0.5 percent. Bf16 attention and MLP weights with mixed-precision Muon (record #57): 0.8 percent.

Transfer: our GEMMs are small (768 by 768 and 768 by 2048), so FP8 gains are likely smaller than the 405B figures. Estimate: a few percent of step time from the MLP GEMMs, uncertain. Cost 2 to 4 hours (conversion, numerics check against bf16, and throughput). Risk medium: numerics, and dependency on the PyTorch 2.14 build.

**FlashAttention 3 versus flex attention.** [FlashAttention-3, arXiv 2407.08608](https://arxiv.org/abs/2407.08608) (read from the [HTML version](https://arxiv.org/html/2407.08608v2)). FP16 up to 740 TFLOPs/s on H100 SXM5 (about 75 percent utilization); forward 1.5 to 2.0x and backward 1.5 to 1.75x faster than FlashAttention-2; FP8 forward close to 1.2 PFLOPs/s. Benchmarks cover causal and non-causal masks, head dimensions 64, 128 and 256 (forward). The paper does not describe support for arbitrary masks, so per-head or DTA-style masks are not covered by it.

[FlexAttention, arXiv 2412.05496](https://arxiv.org/abs/2412.05496) (read from the [HTML version](https://arxiv.org/html/2412.05496v1)). API: score_mod(score, batch_idx, head_idx, q_idx, kv_idx) returns a score; mask_mod(batch_idx, head_idx, q_idx, kv_idx) returns a boolean. create_block_mask builds the block-sparse representation via torch.vmap; a block is skipped when all its scores are masked, full blocks run only score_mod, and partial blocks apply mask_mod per element. Masks compose with and_mask and or_mask. Performance: causal forward 1.00 to 1.22x and backward 0.86 to 1.05x relative to FlashAttention-2 at 1k to 64k; 0.68 to 1.43x across the seven variants that FlashAttention-2 supports; variants it does not support run 5.49 to 8.00x faster than SDPA with explicit masks. Speedrun record #12 (November 2024), "FlexAttention with long context", went from 7.2 to 5.03 minutes (30 percent, bundled with other changes).

Transfer: use flex attention for the benchmark attention. It is the only option we found that expresses per-head masks (DTA), padding and document masks, and sliding-window layers in one kernel. FA3 is faster on plain masks, so measure both on the full-mask case. Cost is part of the flex integration. Risk low.

**CUDA graphs.** [PyTorch CUDA semantics, the 2.15 docs](https://docs.pytorch.org/docs/2.15/notes/cuda.html) (read). The 2.14 documentation URL redirected to the 2.15 page, so check the 2.14 page on the machine. Warmup runs eager iterations on a side stream before capture; the graph replays the same memory addresses, so shapes must be fixed and inputs copied in place before each replay. The example is torch.cuda.CUDAGraph with a torch.cuda.graph(g) context. The docs state that the optimizer step can be captured for typical optimizers, but GradScaler.step cannot, because it synchronizes with the CPU; bf16 training needs no scaler. CUDA RNG operations are permitted. The fetched section does not describe torch.compile mode "reduce-overhead". Transfer: capture forward, backward and the optimizer step for each fixed-length bucket (padding to a few lengths, since dynamic shapes are prohibited). A 36-block model at 768 width is likely launch-bound, so this may help a lot; measure. Cost 3 to 6 hours. Risk medium.

**Fused MLP and ReLU-squared.** Speedrun record #5 (October 2024) introduced ReLU-squared, QK-norm and zero-init projections (22.3 to 15.2 minutes, a large drop that the current baseline probably already includes). A fused Triton kernel for the MLP: not verified in the sources we read.

**Batch-size and sequence-length schedules.** Speedrun record #46, "batch size schedule" (November 2025): 2.248 to 2.203 minutes (2.0 percent). Record #72, "max sequence length schedule" (February 2026): 1.516 to 1.496 minutes (1.3 percent). Under a fixed 10-minute budget, a sequence-length schedule that grows the length late in the run is the form that matches our evaluation at 2,048.

**Open rule question.** It is not stated in the benchmark text we were given whether compilation and CUDA-graph capture time counts toward the 10 minutes. The answer changes how much compile-heavy changes are worth.

---

## Ranked table

Evidence strength: "strong" means a paper result plus a speedrun record, or a reproduced result; "moderate" means one paper with a same-scale comparison, or speedrun evidence alone; "weak" means abstract-only, self-report, or a decoder-only result with no MLM check. Expected gains are estimates for our benchmark unless a paper number is quoted. Costs are our implementation estimates.

| Rank | Idea | Source | Evidence | Expected gain (estimate) | Cost (h) | Risk | Suggested order |
|---|---|---|---|---|---|---|---|
| 1 | Flex attention with CUDA-graph capture and fixed length buckets | [FlexAttention](https://arxiv.org/abs/2412.05496), [CUDA docs](https://docs.pytorch.org/docs/2.15/notes/cuda.html) | moderate (kernel and graph docs, speedrun #12) | enables all others; launch-bound gain unknown | 3 to 6 | medium | 1 |
| 2 | Cautious weight decay with cooldown 0.45 | [Cautious](https://arxiv.org/abs/2411.16085), [WSD](https://arxiv.org/abs/2404.06395), speedrun #43, #50, #26 | moderate | 1 to 3 percent | under 1 | low | 2 |
| 3 | Logit softcap at 15 to 30 | [Gemma 2](https://arxiv.org/abs/2408.00118), speedrun #9, #18 | moderate (speedrun 4.8 percent) | about 1 percent | under 1 | low | 2 |
| 4 | Polar Express in place of Newton-Schulz | [Polar Express](https://arxiv.org/abs/2505.16932), speedrun #38 | strong (paper and speedrun) | under 1 percent at fixed steps, plus fewer FLOPs | 1 to 2 | low | 3 |
| 5 | NorMuon for hidden matrices | [NorMuon](https://arxiv.org/abs/2510.05491), speedrun #41 | moderate | 1 to 5 percent (paper: 6 to 15 percent fewer steps than Muon at 0.1B to 5B; not MLM) | 2 to 3 | low to medium | 3 |
| 6 | U-net skip connections | speedrun #11 | moderate (speedrun 7.7 percent) | 1 to 4 percent | 1 to 2 | low | 4 |
| 7 | Learnable value residual (ResFormer) | [ResFormer](https://arxiv.org/abs/2410.17897) | moderate (82M loss 2.739 to 2.705, decoder only) | under 1 percent loss | 2 to 3 | low | 4 |
| 8 | Dynamic YaRN with window warmup | [YaRN](https://arxiv.org/abs/2309.00071), speedrun #31, #13 | moderate (speedrun 2.2 and 7.4 percent; paper LLaMA only) | 1 to 3 percent | 1 to 2 | low to medium | 5 |
| 9 | FP8 MLP and QKV (torchao) | [torchao](https://github.com/pytorch/ao), speedrun #89, #90 | moderate (speedrun) | 2 to 8 percent of step time, uncertain at width 768 | 2 to 4 | medium | 5 |
| 10 | Schedule-Free AdamW | [Schedule-Free](https://arxiv.org/abs/2405.15682) | moderate | not estimated; removes the need to fix total steps | 1 to 2 | medium | 6 |
| 11 | AdEMAMix | [AdEMAMix](https://arxiv.org/abs/2409.03137) | weak to moderate (abstract token-count claim) | not estimated | 2 | medium | 6 |
| 12 | MARS-approx | [MARS](https://arxiv.org/abs/2411.10438) | weak to moderate (GPT-2 large, token count 50B to 28B) | not estimated | 2 to 3 | medium | 6 |
| 13 | Static hyper-connections, n = 2 or 4 | [Hyper-Connections](https://arxiv.org/abs/2409.19606) | moderate (OLMo loss -0.03 at 1B) | about 0.03 loss at paper scale; memory +26 percent | 4 to 6 | medium | 7 |
| 14 | MUDD dense connections | [MUDDFormer](https://arxiv.org/abs/2502.12170), speedrun #81 | moderate (speedrun 3.1 percent; paper 2.8B) | 3 percent at speedrun scale; throughput risk 10 to 16 percent | 6 to 10 | medium to high | 8 |
| 15 | Diff-Transformer V2 | [DIFF V2 README](https://github.com/microsoft/unilm/tree/master/Diff-Transformer/Diff-Transformer-V2), [V1](https://arxiv.org/abs/2410.05258) | weak for MLM (decoder LMs; README has no numbers) | not estimated | 3 to 5 | medium | 8 |
| 16 | Dual Triangle Attention | [DTA](https://arxiv.org/abs/2604.18603) | moderate for NLP extended context; none for protein at training length | small to moderate at 2,048 | 6 to 10 | medium to high | 9 |
| 17 | Multi-rate MLM (Ankh3 style) | [Ankh3](https://arxiv.org/abs/2505.20052) | weak (no isolated ablation); conflicts with the 15 percent rule | not estimated | 1 | high (rule conflict) | not scheduled |
| 18 | Multi-token prediction | [MTP](https://arxiv.org/abs/2404.19737), speedrun #53 | moderate for causal; not applicable to MLM | not estimated for MLM | 2 | medium | not scheduled |
| 19 | Corrected softmax and cross-entropy | [arXiv 2610.31642](https://arxiv.org/abs/2610.31642) | weak (abstract only) | unknown; test by measuring zero target-gradients in bf16 | 1 to 2 | low | 10 |
| 20 | nGPT normalized blocks | [nGPT](https://arxiv.org/abs/2410.01131), [Training nGPT](https://arxiv.org/abs/2608.01284) | moderate for causal LM at 0.5B to 1B (4 to 20x tokens); none for MLM; overhead 60 to 80 percent | net wall-clock could be negative unless overhead is removed | 8 to 16 | high | 11, after a pilot of 1 to 4 |
| 21 | NoPE or DroPE | [NoPE](https://arxiv.org/abs/2305.19466), [DroPE](https://arxiv.org/abs/2512.12167) | weak for bidirectional; DTA's own RoPE-off hurt | negative in our evidence | 2 to 4 | high | not scheduled |
| 22 | ANVIL optimizer | not found on arXiv | none | none | not estimated | not estimated | excluded |

Suggested order, in words: first the measurement base (flex attention, CUDA graphs, fixed length buckets), then the cheap optimizer and logit changes (cautious weight decay, cooldown, softcap), then the optimizer orthogonalization upgrade (Polar Express, then NorMuon), then residual changes that are cheap to test (U-net skips, value residual), then the context and precision items, then the larger architecture changes (hyper-connections, MUDD, DTA, nGPT), each after a pilot that measures wall-clock to a fixed validation loss.

## Ideas that compose

| Composition | Components | Why they could compose | What must be checked |
|---|---|---|---|
| Optimizer stack | NorMuon (or Polar Express with Muon) + cautious weight decay + cooldown fraction 0.45 | The speedrun records apply these in sequence in the same repository (records #41, #43, #26). | Whether the cautious mask applies to the Muon update, and whether the cooldown value transfers to a time-based schedule. |
| Attention kernel | DTA + differential V2 + RoPE base change, as one flex call | mask_mod and score_mod receive the head index, so per-head masks and paired heads fit in one call; shared K and V per GQA group. | The value dimension and pairing layout (sub-heads of d/2 against paired differential heads); equality with two reference calls on random tensors. |
| Context at 2,048 | Dynamic YaRN + window warmup + ModernBERT-style local and global layers (global base raised) | The speedrun pairs YaRN with window warmup; the global and local split is a mask_mod. | Whether the global base and the YaRN ramp conflict at our training length; throughput of the mixed mask. |
| Residual and skip | U-net skips + value residual + softcap | Each is a separate speedrun or paper result; all three change what reaches the output. | Overlap: value embeddings and the value residual both mix early values back in. Test each alone before stacking. |
| Dense connections | MUDD or hyper-connections + U-net skips | MUDD subsumes dense skip connections, so the two overlap. | Choose one. Measure throughput of each alone. |
| Systems | Flex attention + CUDA graph + FP8 MLP + bf16 cross-entropy | Each lowers time per step on one GPU; bf16 cross-entropy is in the speedrun. | Fixed shapes for CUDA graphs (bucketed lengths) against variable-length batches; FP8 numerics; torch version 2.14 behavior. |
| nGPT with the rest | nGPT blocks + Muon or AdamW | No evidence found. nGPT removes weight decay and warmup, while cautious weight decay and the cooldown depend on them. | Treat nGPT as an alternative stack, not an add-on. Pilot it against the composed stack. |
| Momentum variants | AdEMAMix or MARS + Polar Express | Both change the momentum or the gradient term; no paper combines them with orthogonalization. | Test with and without the orthogonalization, one change at a time. |
