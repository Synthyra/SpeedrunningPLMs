---
id: foundry.layers
kind: library_module
status: active
aliases: [foundry.layers]
---
# foundry.layers

Neural network layers that several projects' models are built from, one submodule each.

| Submodule | Holds |
|---|---|
| `rotary` | Rotary position embeddings whose tables are cached by length |
| `linear` | Linear layers and free parameters with Xavier-normal weights, and layer normalization without a bias |
| `feedforward` | SwiGLU and the pre-normalized SwiGLU feed-forward block |
| `attention` | Multi-head self-attention and two attention poolers |
| `transformer` | The transformer block: attention, then the feed-forward block |
| `attention_backends` | Attention backend selection (`flex`, `sdpa`, `kernels`), masks for each, and unpadding for FlashAttention |
| `probe_attention` | Multi-head self-attention on a selectable backend, with rotary tables computed once up to `max_seq_len` |
| `probe_transformer` | The pre-normalized transformer stack of the residue-embedding probes, and head-size resolution |
| `losses` | Training losses of probe heads, by task type (cross entropy from `cross_entropy`), and binary cross entropy with ignored targets and smoothing |
| `sigreg` | SIGReg, the sketched isotropic Gaussian regularizer of LeJEPA and LeWorldModel: the Epps-Pulley statistic along random directions |
| `cross_entropy` | Log-softmax, cross entropy, `CrossEntropyLoss`, binary cross entropy on logits and `BCEWithLogitsLoss` that keep a confident target's gradient, as drop-ins for PyTorch's, and soft-target cross entropy on teacher logits |

`cross_entropy` was added on 2026-10-25 to replace PyTorch's cross entropy, which rounds the target's loss and
gradient to zero once one logit leads by 16.6 in float32 or 5.5 in bfloat16 (Softmax Collapse,
[Barron et al. 2026](https://arxiv.org/abs/2610.31642)). It matches PyTorch's arguments, values and gradients
wherever PyTorch is right; `get_loss_fct` returns its `CrossEntropyLoss`, and `ws lint` flags PyTorch's elsewhere.
Delete it when `test_pytorch_still_collapses` fails ([decision](../../docs/decisions/foundry/layers/2026-10-25_collapse_free_cross_entropy.md)).
Tests: `tests/tier1_unit/test_layers_cross_entropy.py`.

`attention_backends`, `probe_attention`, `probe_transformer`, `losses`, and `feedforward.swiglu_ln_ffn`
are the generic layers of Protify's transformer probe and of the InterpNet probe, moved here from
`interpnet` on 2026-10-14 so the public Protify carries them without InterpNet, which stays
private. Their code and state-dict keys are unchanged. Only the `kernels` backend imports FastPLMs,
which loads FlashAttention through its `kernels.lock` and raises when a supported kernel fails to
load. Tests: `tests/tier1_unit/test_layers_attention_backends.py`,
`test_layers_probe_attention.py`, and `test_layers_losses.py`.

`linear`, `feedforward`, `attention`, and `transformer` are the annotation transformer's layers. With `nn.Linear`'s default initialization,
and in the feed-forward block a biased layer normalization, the attention and the feed-forward
block are also DSM's and synth's translator's, and the feed-forward block is InterpNet's. With an
unbiased layer normalization before its projection, and spectral normalization when asked for, the
attention is also confounders'. `attention` also holds the attention pooler of DSM, confounders,
and synth's two translators, which pools at the model width.

`sigreg` was added on 2026-10-15 for [esmc_lewm](../../projects/representation/esmc_lewm/README.md), in LeWorldModel's form
(1,024 directions, 17 knots on [0, 3]). Atlas's trainer keeps its own `_sigreg_latent_loss` (64 directions, 17 knots
on [-5, 5]), the first copy; moving it here would change Atlas's numerics, so it is left for Atlas's owner. Tests:
`tests/tier1_unit/test_layers_sigreg.py`.

## `foundry.layers.rotary`

Rotary position embeddings (RoPE) whose cosine and sine tables are cached by length.

| Name | Does |
|---|---|
| `rotate_half(x, interleaved=False)` | Turns each channel pair a quarter turn. The pairs are the two halves of the last dimension or, with `interleaved`, neighboring channels |
| `apply_rotary_emb_torch(x, cos, sin, interleaved=False)` | Rotates the first `2 * cos.shape[-1]` channels of each head of `x` `(b, l, h, d_h)` by its position. The rest pass through |
| `RotaryEmbedding(dim, base=10000.0, ..., derive_frequencies=False)` | Self-attention RoPE. `forward(q, k)` rotates both, and rebuilds the tables only for a longer sequence, another device or dtype, or training after inference |
| `RotaryEmbeddingCross(dim, ..., max_seq_len=10000)` | Cross-attention RoPE. Queries and keys are each rotated by their own positions, and the tables hold at least `max_seq_len` positions |

`inv_freq` is registered with `persistent=False`, so no checkpoint holds it, and each consumer's
`_init_weights` rebuilds it after `from_pretrained`, as
[foundry.checkpoint_loading](../checkpoint_loading/README.md) explains. With
`derive_frequencies=True`, each table rebuild also recomputes it in float32 on the input's device.
That repairs a buffer a load path left unfilled without calling `_init_weights`, and keeps the
frequencies float32 after the module is cast to a lower precision, where by default they are the
cast buffer's.

The code is ESM3's `esm/layers/rotary.py` (EvolutionaryScale). `rotary.py` keeps the Apache 2.0
notice that file carries and says what changed.

Promoted on 2026-09-24. These copies now import it:

- Atlas's `models/annotation_transformer/rotary.py`, and the `annotation_transformer` model
  family, which since 2026-09-26 holds the model sequence_gradient copied.
- confounders' `model/rotary.py`.
- DSM's `models/attention/rotary.py`, which held `RotaryEmbeddingCross` too.
- embedding_translation's `models/rotary.py`, which derived its frequencies. Its
  `models/attention.py` now asks for that with `derive_frequencies=True`.
- synth's translator's `services/translator/modules.py`, whose reduced copy had no interleaving
  or xPos. The translator's `_init_weights` now rebuilds the frequencies with
  `_compute_inv_freq`, where the copy had a public `compute_inv_freq`.

Against each copy, outputs, cached tables, frequency buffers, and state-dict keys match bit for
bit. The comparison covered head widths, sequence lengths, input dtypes, modules cast to bfloat16
and float16, training and evaluation, and, for embedding_translation, a corrupted frequency
buffer. What changed:

- Interleaved mode repeats each frequency across two neighboring channels, `(d 2)`, as flash-attn
  does. The copies repeated the tables as `(2 d)` in both modes, so an interleaved pair turned by
  two different frequencies and the result was not a rotation. No consumer uses interleaved mode.
- `scale_base` (xPos) still registers the `scale` buffer, and the forward pass still fails with
  an `AssertionError`, now before it builds tables. The xPos tables nothing could use are gone.
- `RotaryEmbeddingCross` subclasses `RotaryEmbedding`, so an `isinstance(module, RotaryEmbedding)`
  check in `_init_weights` rebuilds its frequencies too. DSM's `CrossAttention` is its only user,
  and no model builds one.
- embedding_translation's `RotaryEmbeddingCross` reads the frequency buffer now, as DSM's did.
  Nothing in embedding_translation uses it.

Not here: Atlas's other rotary embedding, `models/transformer/rotary.py`, which builds its tables
on every call. Its copy in sequence_gradient waits with the rest of Atlas's runtime, per
[the plan](../../docs/decisions/workspace/2026-09-24_foundry_unification.md). FastPLMs' ESM3 keeps its own
copy, since FastPLMs vendors no foundry. InterpNet's computes its tables once, as buffers, which
makes it a different module.

## The annotation transformer's layers

`linear`, `feedforward`, `attention`, and `transformer` hold the layers of the annotation
transformer, whose checkpoints, such as `Synthyra/AT-Base`, load into them.

| Name | Does |
|---|---|
| `linear.LinearLayer(input_size, output_size, bias=False, *, spectral_norm=False, xavier_init=True)` | `nn.Linear` with a Xavier-normal weight and a zero bias, or with `xavier_init=False` its default initialization. `spectral_norm` wraps it in `torch.nn.utils.spectral_norm`, which holds the weight as `weight_orig` beside `weight_u` and `weight_v` |
| `linear.Linear` | `LinearLayer` without a bias |
| `linear.ParameterLayer(size, *, xavier_init=True, **kwargs)` | A free `nn.Parameter` of shape `size` with Xavier-normal values, or with `xavier_init=False` the standard normal values `torch.randn` draws. `kwargs` go to `torch.randn` |
| `linear.LayerNorm` | `nn.LayerNorm` without a bias |
| `feedforward.correction_fn_256(expansion_ratio, hidden_size)` | The MLP's width: `expansion_ratio * hidden_size`, rounded up to a multiple of 256 |
| `feedforward.SwiGLU` | Gates the second half of the last dimension with the SiLU of the first |
| `feedforward.swiglu_ffn(hidden_size, expansion_ratio, dropout=0.1, *, bias=False, spectral_norm=False, norm_bias=False, xavier_init=True)` | `nn.Sequential` of `LayerNorm`, a projection to twice the MLP's width, `SwiGLU`, dropout, and a projection back, keyed `0.weight`, `1.weight`, and `4.weight`. `norm_bias` adds `0.bias`, `bias` adds `1.bias` and `4.bias`, and `bias`, `spectral_norm`, and `xavier_init` go to both projections |
| `attention.MultiHeadAttention(hidden_size, n_heads, rotary=True, causal=False, *, derive_frequencies=False, xavier_init=True, norm_bias=True, spectral_norm=False)` | Pre-normalized self-attention: a layer normalization and one projection make the queries, keys, and values, the queries and keys are normalized again and rotated by position, and scaled dot-product attention combines them. A `(b, l)` mask marks the keys to attend to; causal attention ignores the mask. `derive_frequencies` goes to the rotary embedding, `xavier_init` and `spectral_norm` to both projections, and `norm_bias` to the first layer normalization |
| `attention.AttentionPooler(hidden_size, intermediate_size, n_tokens=1)` | Pools `(b, l, d)` into `(b, n_tokens, d)` by attention from learned queries, after projecting to `intermediate_size`, in heads of 64 channels |
| `transformer.TransformerBlock(config, *, derive_frequencies=False)` | `MultiHeadAttention`, then `swiglu_ffn`, each added to its input. `config` is a `TransformerBlockConfig`: anything with `hidden_size`, `n_heads`, `rotary`, `expansion_ratio`, and `dropout` |

The code is ProteinRepresentationEnhancement's `models/utils.py`, `models/blocks.py`, and
`models/attention.py`, from sequence_gradient's typed and shape-traced port of Atlas's copy.
Each module builds its layers in the order the copies did, so after the same seed it draws the
same random numbers and starts from the same weights.

Promoted on 2026-09-24. These copies now import it:

- Atlas's `models/annotation_transformer/` `attention.py`, `blocks.py`, and `utils.py`, and
  the `annotation_transformer` model family, which imports the layers directly and replaced
  sequence_gradient's copies on 2026-09-26.
- embedding_translation's `models/attention.py`, `models/blocks.py`, and `models/utils.py`.
  `MultiHeadMLPAttention` and `CampAttention` stay in `models/attention.py` and subclass
  foundry's `MultiHeadAttention`. `models/blocks.py` held a `Proj` nothing built, confounders'
  but for its location, and it was removed on 2026-09-26. This project's rotary
  always derived its frequencies, so `models/annotation_transformer.py` builds its blocks with
  `derive_frequencies=True`, and `MultiHeadMLPAttention` passes it too.
- confounders' `model/utils.py`, and `SwiGLU` and `swiglu_ffn` in `model/blocks.py`.

Against each copy, bit for bit: construction after a seed (the random numbers drawn, parameters,
buffers, state-dict keys and their order, and each module's settings), forward outputs, rotary
tables, and gradients. The comparison covered widths, head counts, sequence lengths, padding and
four-dimensional masks, a fully padded sequence, float32, bfloat16, and float16, and training
and evaluation. It also covered whole models: Atlas's and sequence_gradient's
`AnnotationTransformer`, embedding_translation's, and confounders' `PPIModel` with and without
spectral normalization, each saved by one side and loaded by the other with no missing or
unexpected keys. What changed:

- `LinearLayer` takes `spectral_norm` by keyword, after `bias`. confounders' took it third by
  position, and every call named it.
- `swiglu_ffn` takes `spectral_norm` by keyword. confounders' `TransformerBlock` passed it fourth
  by position, and now names it.
- `ParameterLayer` takes no `spectral_norm`. confounders' accepted one, but spectral
  normalization of a parameter raised `AttributeError`, and nothing passed it. The keyword now
  reaches `torch.randn`, which raises `TypeError`.
- embedding_translation's `MultiHeadAttention` and `TransformerBlock` are foundry's, so one built
  directly takes the default, `derive_frequencies=False`, and after a cast to a lower precision
  its outputs differ from what the project's copy gave. The project's own constructions pass
  `True`. Only the `__main__` smoke run in `models/attention.py` builds one directly.

## DSM's and synth's translator's layers

Two options of the layers above let DSM and synth's translator build theirs from foundry. Each
defaults to the annotation transformer's rule:

- `xavier_init=False` keeps `nn.Linear`'s default initialization, in `LinearLayer` and in both
  projections of `swiglu_ffn` and of `MultiHeadAttention`. No other random numbers are drawn, so
  a seeded model starts from the weights it did.
- `norm_bias=True` gives `swiglu_ffn`'s layer normalization a bias, `0.bias`. With
  `xavier_init=False` too, the block is ESM3's `swiglu_ln_ffn`, with dropout.

Promoted on 2026-09-24. These copies now import them:

- DSM's `models/MLP.py`. `correction_fn_256` and `SwiGLU` are foundry's, and `swiglu_ffn` is
  foundry's with `norm_bias=True` and `xavier_init=False`. `ReLU2` and `relu2_ffn` stay.
- DSM's `models/attention/self_attention.py`. `MultiHeadAttention` subclasses foundry's and passes
  `xavier_init=False`, so built directly it still starts as DSM's did.
- synth's translator's `services/translator/modules.py`. `RotaryEmbedding`, `rotate_half`,
  `apply_rotary_emb` (foundry's `apply_rotary_emb_torch`), and `SwiGLU` are foundry's,
  `swiglu_ln_ffn` is `swiglu_ffn` with `norm_bias=True` and `xavier_init=False`, and
  `MultiHeadAttention` subclasses foundry's with `xavier_init=False`. `TransformerBlock`,
  `Transformer`, and `LMHead` stay, and `AttentionPooler` is the model-width pooler below.

Against each copy, bit for bit, as for the annotation transformer's layers: construction after a
seed, forward outputs, rotary tables, and gradients, over widths, head counts, lengths, masks,
three precisions, and training and evaluation. DSM's and the translator's transformer stacks
exchanged state dicts, and whole `SeqToAnnTranslator` models were saved by each side and loaded by
both. foundry's defaults were compared with foundry before the options in the same way. What
changed, all in the translator:

- Its attention expands a `(b, l)` mask to `(b, 1, l, l)`, as foundry's does. The copy handed the
  mask to attention as it came, where it was read as (query, key): that raised unless the batch
  was 1 or the length, misread the mask when the two were equal, and raised for an integer mask.
  The translator's transformer stack expands masks before its attention, so the model never passed
  one. The attention also holds foundry's `causal`, which is False.
- Its attention's output projection reads `(d // h) * h` channels, where the copy's read `d`. The
  two differ only for a width the head count does not divide, where attention raises either way.
- Its rotary embedding takes `interleaved` as the third positional argument, where the copy took
  `scaling_factor`, and nothing passed it by position. It holds foundry's `interleaved`,
  `scale_base`, and `derive_frequencies`, and a `scale` buffer of None, which no state dict
  holds. An odd head width fails the same assertion, whose message now says why.

## InterpNet's feed-forward block

`swiglu_ffn`'s `bias` gives both projections a bias, `1.bias` and `4.bias`, which Xavier
initialization starts at zero. With `norm_bias=True` and `xavier_init=False`, the block is
InterpNet's, whose projections take a bias when its layers' `use_bias` asks for one.

Promoted on 2026-09-24. This copy now imports it:

- InterpNet's `mlp.py`, which Protify's `model_components/mlp.py` re-exports. `SwiGLU` is
  foundry's, `intermediate_correction_fn` is `correction_fn_256` under InterpNet's name, and
  `swiglu_ln_ffn` keeps its signature, with `use_bias` fourth, and builds foundry's block.

InterpNet imports foundry for the first time, so `foundry.layers` ships wherever InterpNet does.
`ws publish` vendors it beside InterpNet: Oracles' and Protify's projections now carry it, and
synth's already did. Protify's `_copy_runtime_code` ships it beside `foundry.embedding` in every
exported probe repository. Protify's export test, which runs the exported code with nothing else
on the path, failed until it did.

Against the copy, bit for bit: the MLP's width over a grid of ratios and widths, including values
that raise, SwiGLU over shapes, four precisions, values that are not finite, and gradients, and the
block after a seed (the random numbers drawn, the state dict, forward outputs in training and
evaluation, gradients, warnings, and errors) over widths, ratios, dropout, biases, precisions, and
the ways it is called. Whole InterpNet sequence and token heads were built by each side after the
same seed, and each side's checkpoint was loaded by both with `from_pretrained`. The comparison
ran on the CPU. What changed:

- `interpnet.mlp` now holds its three names alone. It also held `torch` and `F`, which nothing
  imported from it, and a star import took them too.
- `intermediate_correction_fn.__name__` is `correction_fn_256`.

## confounders' attention

Two options of `MultiHeadAttention` let confounders build its attention from foundry. Each
defaults to the annotation transformer's rule:

- `norm_bias=False` leaves the layer normalization before the projection without a bias, so a
  checkpoint holds no `layernorm_qkv.0.bias`.
- `spectral_norm=True` wraps both projections in `torch.nn.utils.spectral_norm`, as `LinearLayer`
  does, so each holds `weight_orig` beside `weight_u` and `weight_v`.

Promoted on 2026-09-25. This copy now imports them:

- confounders' `model/attention.py`. `MultiHeadAttention` subclasses foundry's, keeps its
  signature, with `spectral_norm` fifth by position, and passes `norm_bias=False`. `PAttention`,
  `MultiHeadPAttention`, and `AttentionPooler` stay as they were.

Against the copy, bit for bit: construction after a seed (the random numbers drawn, parameters,
buffers, spectral normalization's hooks and vectors, state-dict keys and their order, and each
module's settings), forward outputs, and gradients, over widths, head counts, lengths, no mask and
eight kinds of mask, a fully padded sequence among them, four precisions, training and evaluation,
and the ways it is called, those that raise included. The attention was also compared under `torch.compile`, as
confounders' trainer runs its model, and inside whole `PPIModel`s of both block types, with and
without spectral normalization, each saved by one side and loaded by both. foundry's attention
without the options was compared with foundry before them in the same way, called as each of its
consumers calls it, and inside foundry's `TransformerBlock`. The comparison ran on the CPU.
Nothing a caller sees changed. The class has a docstring now, and `model/attention.py` holds one
more name, `attention`, which is foundry's module.

## The model-width attention pooler

| Name | Does |
|---|---|
| `attention.MultiHeadAttentionPooler(hidden_size, n_tokens=1, n_heads=16, *, bias=False, spectral_norm=False, xavier_init=True)` | Pools `(b, l, d)` into `(b, n_tokens, d)` by attention from learned queries, at the model width in `n_heads` heads. A `(b, l)` mask is expanded to `(b, 1, n, l)`, and a mask of any other shape reaches attention as given. `bias`, `spectral_norm`, and `xavier_init` go to the four projections, and `xavier_init=False` also keeps the learned queries standard normal |

`AttentionPooler` projects the sequence to another width first and counts heads of 64 channels, so
its checkpoints hold `input.weight` and queries of that width. This pooler has no input projection,
and its checkpoints hold `Q`, `Wq.weight`, `Wv.weight`, `Wk.weight`, and `Wo.weight`, with a
`.bias` for each projection under `bias`. It builds its queries and projections in that order, as
the copies did.

Promoted on 2026-09-24. These copies now subclass it, and keep their names and signatures:

- DSM's `models/attention/attention_pooler.py`, with `xavier_init=False`.
- confounders' `model/attention.py`, whose `spectral_norm` stays a fourth positional argument.
- synth's translators: V1's `services/translator/modules.py` and V2's
  `services/translator/model_v2.py`, each with `xavier_init=False`. V2 passes its `use_bias` as
  `bias`.

Against each copy, bit for bit, as above: construction after a seed, forward outputs, and
gradients, over widths, head counts, query counts, lengths, integer, boolean, and float padding
masks, a fully padded sequence, three precisions, training and evaluation, spectral normalization,
and biases. Whole models too: confounders' `PPIModel` with both block types, with and without
spectral normalization, and both translators, each saved by one side and loaded by both. The
comparison ran on the CPU. What changed:

- DSM's, confounders', and V2's poolers reshaped every mask as a `(b, l)` one, so a
  `(b, 1, n, l)` mask raised. It now reaches attention as given.
- V1's pooler expands a `(b, l)` mask to `(b, 1, n, l)`, as the others did. The copy handed it to
  attention as it came, which read it as (query, key): only a boolean mask of a single sequence
  worked, a batch of `n_tokens` sequences had its mask misread, other batches and every integer
  mask raised, and a float mask was added to the attention scores. V1's `forward`, which its
  trainer calls with integer masks, raised at the first pooler, and now runs. `inference` hands
  the poolers no mask, so what V1 serves is unchanged.
- V2 expanded a `(b, l)` mask to `(b, 1, 1, l)`, which broadcasts to the same mask as
  `(b, 1, n, l)`. The outputs are the same.
- V1's and V2's poolers hold `n_tokens` and `d_head`, and a width the head count does not divide
  fails the same assertion, whose message now says why.
- DSM's pooler module and V1's `modules.py` no longer define `Linear`, a bias-free `nn.Linear`
  that nothing else imported.

Not here: embedding_translation's `models/camp.py` builds a pooler of this kind from a module
the project does not hold. That file is unfinished and nothing imports it.

## Tests

`tests/tier1_unit/test_layers.py`. They need torch and einops, so the tooling environment skips
them, and `ws test --projects foundry` runs them under the research interpreter.
