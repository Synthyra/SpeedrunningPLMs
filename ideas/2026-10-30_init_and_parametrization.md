# Initialization and parametrization ideas, 2023 to 2026

Date: 2026-10-30. A literature scout for SpeedrunningPLMs (36 pre-norm blocks, 768 stream, 2,048 ReLU-squared MLP,
about 2,100 steps in 600 s on one GH200; Muon on block matrices, AdamW on embedding, head and scalars). Read-only: no
code was run and nothing here is measured on this benchmark. Each gain is a source's number or our estimate, and the
table says which. Read first: `ideas/README.md`, `ideas/2026-10-30_speedrun_optimizers.md`,
`ideas/2026-10-30_mlm_pretraining.md`; `modded_nanogpt_2026.md` and `architecture_optimization.md` by search.

Shorthand: LN is LayerNorm; CE is validation cross entropy in nats; TPP is tokens per parameter; muP is the maximal
update parametrization; CompleteP is Dey et al.'s complete-feature-learning parametrization; "record" is the
incumbent genome (022, then 028).

## Outcome

- Four ideas have a source and a number clear enough to test. The brief asked for 8 to 12. The rest of the 2023 to
  2026 window either repeats the catalog, has no loss number in the text read, or does not fit the strategy hooks.
  Those are listed under "Checked, not ideas" with the reason. I did not pad the list.
- The mechanism that matters for this model (our inference; the sources do not state it): Muon and Adam both
  normalize gradient scale, so a constant multiplier on a branch's output acts as a learning-rate multiplier on that
  branch's output matrices. This is why depth multipliers are the most likely lever in this scope, and why `muon_lr`
  must be searched again whenever one is added.
- Regime caveat: the sources train at 9 to 20 TPP or more (LLaMA-1B at 8.9B tokens; CompleteP at 20 TPP; the Muon
  GPT-2 sweep at 315M tokens). Our 65 to 131M tokens is 0.3 to 0.7 TPP. Discount every gain here for that.
- Zero start: the record's zero-initialized output projections keep every branch at zero at step 0. Each idea keeps
  step 0 unchanged unless it says otherwise.

## Ranked table

Gain is the expected change in CE on this benchmark (our estimate; negative is better). "Read" says how much of the
source was read.

| Rank | Idea | Source (year) | Read | Cost (h) | Expected gain (nats) | Step cost | Evidence strength |
|---|---|---|---|---|---|---|---|
| 1 | Residual branch multiplier (L/L_base)^-alpha on each sublayer output | arXiv 2310.02244 (2024); arXiv 2505.01618 (2025); arXiv 2610.14344 and 2606.25971 (2026) | CompleteP full text (sections and Table 2); the two 2026 papers in part | 0.5 | 0 to -0.010 | none | moderate with AdamW at 20 TPP; weak for Muon (no loss table read) |
| 2 | LayerNorm Scaling: each sublayer's normalized input times 1/sqrt(l) | arXiv 2502.05795 (2025) | full text, Eq. 12 and Table 1 | 1 | 0 to -0.010 | under 1 percent | moderate on LLaMA at 9 to 17 TPP; untested on Muon or MLM |
| 3 | Learned per-block branch gains from a nonzero start (ReZero, LayerScale) | arXiv 2003.04887 (2020); arXiv 2103.17239 (2021) | abstract via search summary; CaiT code default | 1 | 0 to -0.003 | none | weak; no MLM or Muon number |
| 4 | Depth-scaled nonzero output init instead of zero (GPT-2 style; layer-index rescaling) | GPT-2 (Radford et al., 2019, secondary sources only); arXiv 2503.17500 (2025) | abstract only (2503.17500) | 0.5 | 0 plus or minus 0.002 | none | weak; no CE number read |

## 1. Residual branch multiplier (Depth-muP, CompleteP), tested under Muon

- Sources: Yang et al., Tensor Programs VI, arXiv 2310.02244 (ICLR 2024): each residual branch scaled by a/sqrt(L).
  Dey et al., "Don't be lazy: CompleteP", arXiv 2505.01618 (NeurIPS 2025): the update is h + L^-alpha F(h), with alpha
  in [0.5, 1]; alpha = 1 is CompleteP. Muon-side: arXiv 2610.14344 (2026) uses a residual scale K_base/K (K_base = 2)
  before the residual add and reports that the optimal Muon learning rate is nearly the same across GPT-2 depths 2 to
  128 (width 256, 1,200 steps, about 315M tokens); the text read has no table and no CE numbers. arXiv 2606.25971 (2026)
  uses a fixed block-output scale of 1/L after RMSNorm with Muon as its default and reports depth transfer in figures
  only. Counter-evidence: arXiv 2604.27077 (2026, snippet) finds that adding a 1/sqrt(depth) learning-rate correction
  moves the optimum in a normalized transformer.
- Evidence (CompleteP Table 2; AdamW; 1.5B non-embedding parameters, 20 TPP; compared with muP): at the optimal shape
  (N = 1,984, L = 32) validation CE 2.302 becomes 2.286 (-0.016); at the deepest shape (N = 832, L = 179) it becomes
  2.386 to 2.330 (-0.056). The paper reports 11.8 and 34.4 percent fewer FLOPs at matched loss. alpha = 0.5 saves 19.6
  and 23.7 percent. Sizes tested: 50M, 300M and 1.5B non-embedding; depths 2 to 128 in the sweep.
- What changes here: every sublayer output is multiplied by a constant before the residual add. In the paper's
  convention the factor is (L / L_base)^-alpha with L_base = 2, so at L = 36 alpha 1 gives 1/18 and alpha 0.5 gives
  0.236. The record was tuned at factor 1, so `muon_lr` must be searched again. The record's zero start is unchanged.
- Expected cost: 0.5 hour. Compile 0 s. Step 0 (one scalar multiply, fusable with the residual add). Memory 0.
- Sketch (hooks: `build`, `sublayer_output`; the registry names are the strategy interface's, the model names to check):

```python
class DepthBranch(Strategy):  # family "depth_branch" (new module); variant "none" or "power"
    family = "depth_branch"
    class Config(StrategyConfig):
        VARIANTS = ("none", "power"); variant: str = "none"; alpha: float = 0.5; base_depth: float = 2.0
        BOUNDS = {"alpha": (0.0, 1.0), "base_depth": (1.0, 36.0)}
    def build(self, model, max_tokens):              # one constant: (L / L_base) ** -alpha; the record is alpha 0
        self.scale = (len(model.blocks) / self.cfg.base_depth) ** -self.cfg.alpha   # model.blocks: check the name
    def sublayer_output(self, index, sublayer, Y):   # Y: (b, l, d), before the residual add
        return Y * self.scale
```

- Risk: the Muon case rests on snippets and one visible sweep with no CE numbers. The paper's claim is for 20 TPP. At
  0.3 to 0.7 TPP a branch scaled by 1/6 to 1/18 learns more slowly, so the sign may be positive. Start the search at
  alpha 0.25 and 0.5; alpha 1 at 36 blocks is far outside anything the record tested. Do not stack with idea 2 in one run.

## 2. LayerNorm Scaling (1/sqrt(l) on each sublayer's normalized input)

- Source: arXiv 2502.05795 (2025), Eq. 12 and Table 1, read in full. The output of each layer's LayerNorm is scaled by
  1/sqrt(l), l = 1 to L.
- Evidence (LLaMA causal LMs, Pre-LN baseline to Pre-LN plus LNS, perplexity): 130M at 2.2B tokens, 26.73 to 25.76
  (0.037 nats); 250M at 3.9B, 21.92 to 20.35 (0.074); 350M at 6.0B, 19.58 to 18.20 (0.073); 1B at 8.9B, 17.02 to 15.71
  (0.080). The text also reports the 7B OLMo run at 20B tokens: final loss 2.69 to 2.50 (0.19 nats). The LLaMA rows are
  at about 17, 16, 17 and 9 TPP. Optimizer Adam at 1e-3 up to 350M and 5e-4 at 1B; warmup not stated. No depth ablation
  in the text read; the Table 8 placement test checks each sublayer alone.
- What changes here (our derivation): attention output is linear in the normalized input after QK-norm removes the
  score scale, so it scales by 1/sqrt(l). The MLP is ReLU-squared with no bias (as the catalog describes; check the
  code), so it is 2-homogeneous and scales by 1/l. At block 36 the attention branch is multiplied by 1/6 and the MLP by
  1/36. Under Muon these are learning-rate cuts on those branches. The record's zero start already restrains deep
  branches at step 0; LNS keeps restraining them afterwards.
- Expected cost: 1 hour. Compile 0 s. Step under 1 percent (72 multiplies). Memory 0.
- Sketch (hook: `sublayer_output`; the sublayer names must match the model's):

```python
class LayerNormScaling(Strategy):  # family "lns"; attention by l^-p, MLP by l^-2p (2-homogeneous under ReLU-squared)
    family = "lns"
    class Config(StrategyConfig):
        VARIANTS = ("none", "layer_index"); variant: str = "none"; power: float = 0.5
        BOUNDS = {"power": (0.0, 1.0)}
    def sublayer_output(self, index, sublayer, Y):   # index is 0-based; the paper's l is index + 1
        layer = index + 1
        gain = layer ** -self.cfg.power if sublayer == "attention" else layer ** (-2.0 * self.cfg.power)
        return Y * gain
```

- Risk: the value residual and value embeddings enter before the attention output, so the scaling is exact only for the
  attention path, not for the mixed value terms. The 1/36 factor on the deepest MLPs may starve them within 2,100 steps;
  search power over 0.25 and 0.5 and keep power 0 as the control. Test alone, not with idea 1 in the same run.

## 3. Learned per-block branch gains from a nonzero start (ReZero, LayerScale)

- Sources: ReZero, Bachlechner et al., arXiv 2003.04887 (2020; abstract via search summary): one learned gate per
  residual, initialized at zero; a 12-layer Transformer on enwiki8 converges 56 percent faster (the baseline and step
  count were not read). LayerScale, Touvron et al., CaiT, arXiv 2103.17239 (2021): a learnable per-channel scale
  initialized near zero; the timm default is 1e-4 (code, not the paper); the appendix reports a small nonzero start
  beats zero (search summary; no numbers read).
- What changes here: the record's W_O and W_down are already zero-initialized, which gives the same step-0 behavior as a
  zero gate. A zero gate on a zero matrix is a dead start: every gradient is zero at step 0 and the branch never begins.
  The new variant is to keep the output matrices at zero and set the gate to 0.1 (or 0.3) on the Adam scalar group, so
  the gate learns each branch's overall scale. That is 72 scalars (one per block and sublayer).
- Expected cost: 1 hour. Compile 0 s. Step 0. Memory 72 floats.
- Sketch (hooks: `build`, `parameter_groups`, `sublayer_output`):

```python
class BranchGain(Strategy):  # family "branch_gain"; one scalar per block and sublayer, on the Adam scalar group
    family = "branch_gain"
    class Config(StrategyConfig):
        VARIANTS = ("none", "gain"); variant: str = "none"; init: float = 0.1
        BOUNDS = {"init": (1e-3, 1.0)}   # never 0: with zero output projections the start would be dead
    def build(self, model, max_tokens):              # held in an nn.Module under model.extensions (interface)
        self.gain = nn.Parameter(torch.full((len(model.blocks), 2), self.cfg.init))
    def parameter_groups(self, model):               # the scalar rate is the record's; placeholder name here
        return [ParameterGroup("branch_gain", [self.gain], "adam", lr=SCALAR_LR)]
    def sublayer_output(self, index, sublayer, Y):   # SUBLAYER = {"attention": 0, "mlp": 1}
        return Y * self.gain[index, SUBLAYER[sublayer]]
```

- Risk: this reparametrizes the scale of the output matrices, and the value residual already learns per-block scalars
  (the record's 0.5 starts), so the gates may duplicate them. Under Muon the output matrices move a fixed size per step,
  so the gate rescales a moving matrix; expect a tie unless branch scale is the bottleneck.

## 4. Depth-scaled nonzero output init (GPT-2 style, layer-index rescaling)

- Sources: GPT-2 residual scaling, Radford et al. (2019), read only through secondary sources. They give 1/sqrt(N), while
  a 2026 paper and the Hugging Face GPT-2 code divide the init std by sqrt(2 * n_layer) for the two branches per block.
  The paper itself was not read, so the constant is unverified. Layer Index Rescaling, arXiv 2503.17500 (2025): an
  initialization that depends on layer index; abstract only, one 1B LLaMA, "up to 4.6 percent" on downstream benchmarks;
  baseline and loss not stated. CaiT appendix: see idea 3.
- What changes here: the record's zero output matrices become N(0, sigma) with sigma = 0.02 / sqrt(2 * 36), about
  0.0024 (the 1/sqrt(2N) reading). Step 0 changes: each branch outputs a small random signal. Under Muon the first step on
  a zero matrix is already full size, so zero init is not frozen. Our inference, not a source's claim: the effect is small
  after the first few hundred steps.
- Expected cost: 0.5 hour. Compile 0 s. Step 0. Memory 0.
- Sketch (hook: `initialize`, which runs after the record's zero init):

```python
class BranchInit(Strategy):  # family "branch_init"; variant "gpt2" replaces the record's zero output projections
    family = "branch_init"
    class Config(StrategyConfig):
        VARIANTS = ("none", "gpt2"); variant: str = "none"
    def initialize(self, model):
        std = 0.02 / math.sqrt(2 * len(model.blocks))
        for block in model.blocks:                    # the output projections; check the bank layout (Muon banks)
            nn.init.normal_(block.attn_out.weight, std=std); nn.init.normal_(block.mlp_down.weight, std=std)
```

- Risk: a nonzero start changes step 0, so check the 90 s and 180 s points against the record's curve. Expected to tie.
  Test only with idea 3 or after one of them shows a gain, not as the first run.

## Checked, not ideas

- Warmup by loss catapult (arXiv 2406.09405, NeurIPS 2024, abstract and summary): warmup mainly lets a network tolerate a
  larger learning rate, and the starting sharpness depends on parametrization and init. No Muon or MLM number read.
  `warmup_fraction` (0 to 0.1) is already a search field, so zero warmup is in the space. No new module.
- Embedding and unembedding multipliers: u-muP (arXiv 2407.17465, 2024) argues the embedding multiplier is redundant with
  residual multipliers, which idea 1 covers. CompleteP's unembedding factor (m_N^-1, Table 1) depends on width, which is
  fixed here, so it is a constant that `head_lr` and the softcap already span. No 2025 ablation with loss numbers found.
- StableEmbedding (arXiv 2110.02861, Dettmers et al., 2021): LayerNorm after the embedding with Xavier init, motivated by
  8-bit training stability. Its layer-norm ablation numbers were not read, and it has no bf16 loss number. Not proposed.
- Sigma reparameterization (arXiv 2303.06296, 2023): each linear layer becomes spectrally normalized with a learned scalar,
  to prevent attention entropy collapse. The language-modeling numbers were not read. It needs the banked weights to be
  rewritten in the forward pass, which the hooks do not expose (4 hours or more). Parked; Muon's polar step already bounds
  each update's singular values, so the overlap is likely.
- Per-depth Muon learning rate for output matrices: arXiv 2604.27077 (snippet) finds that a 1/sqrt(depth) learning-rate
  correction moves the optimum; arXiv 2610.14344 (snippet) finds the Muon optimum depth-invariant without one. Not
  proposed; idea 1's forward multiplier is the cheaper test.
- QK-norm gain and attention scale: no 2025 study of the gain's initialization with loss numbers was found. nGPT (arXiv
  2410.01131, 2024) has a learnable-scalar ablation table that was not read. Record 92 uses a scale of 0.13 at head width
  64, near 1/sqrt(64) = 0.125. Check `attention.py` for an existing learnable QK gain first; if one exists, the SSMax
  file's fixed-length form is already in the record.
- Mimetic V and O init (modded-nanogpt record 64, beta 0.05): catalog 020 tested it on 015 and it had the same curve.
- Scalar lambdas, skip-weight init at 0.5, and value-embedding init (modded-nanogpt records 11, 14, 63, 65, 218): covered by
  `unet`, `value_embeddings`, `value_residual` and `embedding_mix` in the catalog.
- DeepNet alpha and beta (arXiv 2203.00555, 2022): a post-LN residual scale. The alpha and beta values were not found in
  the search results, and the record is pre-norm. Not applicable.
- Peri-LN and sandwich norm: covered by `2026-10-30_speedrun_optimizers.md` (Peri-LN, rank 3).
- Zero-FLOP capacity (shared or grouped parameters): no source found with a loss number. Per-token tables were already
  tested (catalog 010, the pair table, reverted; value embeddings).
- Embedding learning rate: arXiv 2605.21486 (2026, "Quantifying Hyperparameter Transfer and the Importance of Embedding
  Layer Learning Rate") surfaced in search and was not read. The catalog already searches `embed_lr`.

## Gaps and limits

- The window has few parametrization results with a loss number at this regime. I found four usable ideas, not eight.
- The Muon-side depth multipliers (arXiv 2610.14344, 2606.25971) have no loss table in the text read; their numbers are in
  figures I did not read.
- The GPT-2 init constant is unverified: secondary sources disagree on 1/sqrt(N) and 1/sqrt(2N).
- No code was run; costs and step effects are estimates from the record's profile (276 ms per step).

## Sources

- Tensor Programs VI (Depth-muP): https://arxiv.org/abs/2310.02244
- CompleteP: https://arxiv.org/abs/2505.01618 ; HTML v4: https://arxiv.org/html/2505.01618v4
- Residual scale with Muon, GPT-2 depths: https://arxiv.org/abs/2610.14344
- Fixed block-output scale with Muon: https://arxiv.org/abs/2606.25971
- LR correction counter-evidence: https://arxiv.org/abs/2604.27077
- Spectral condition for muP under width-depth scaling: https://arxiv.org/abs/2603.00541
- LayerNorm Scaling: https://arxiv.org/abs/2502.05795 ; HTML v2: https://arxiv.org/html/2502.05795v2
- ReZero: https://arxiv.org/abs/2003.04887
- LayerScale (CaiT): https://arxiv.org/abs/2103.17239
- Layer Index Rescaling: https://arxiv.org/abs/2503.17500
- Warmup mechanisms: https://arxiv.org/abs/2406.09405
- u-muP: https://arxiv.org/abs/2407.17465
- StableEmbedding (8-bit optimizers): https://arxiv.org/abs/2110.02861
- Sigma reparameterization: https://arxiv.org/abs/2303.06296
- nGPT: https://arxiv.org/abs/2410.01131
- DeepNet: https://arxiv.org/abs/2203.00555
- Embedding learning rate: https://arxiv.org/abs/2605.21486
- modded-nanogpt records: https://github.com/KellerJordan/modded-nanogpt
