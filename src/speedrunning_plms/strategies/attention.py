"""The attention family: the kernel each block's attention sublayer runs, with its training masks.

- `standard` (the record): h heads of d_h with QK-norm, RoPE and the positional prior, one flex call.
- `differential`: Differential Transformer V2 (B2). 2h query heads over h key and value heads; the two query heads of a
  pair share one KV head, and the pair's output is attn_even - sigmoid(lambda) attn_odd, with lambda per token and
  head from a zero-initialized projection of the input (so the pair starts at attn_even - 0.5 attn_odd).
- `triangle`: Dual Triangle Attention (B3, after `projects/representation/dual_triangle_attention`). Each head splits
  into a down sub-head (keys at or before the query) and an up sub-head (at or after) of d_h / 2, each with its own
  softmax; the split keeps RoPE's rotation pairs whole, so each sub-head stays relative-position equivariant. Two
  flex calls with directional document masks, about half the attention FLOPs of one full call.
- `triangle_differential`: both, composed: each logical head has two query branches, each split down and up; the
  two branches share the down and up keys and values, and subtract as in `differential`.
- `paired`: paired-head attention (B6, modded-nanogpt record 58). Two heads interleave along the sequence, so one
  attention over 2l entries lets a query of either head read the keys of both; bidirectional, so the scores double.

Two numbers act under every kernel, both off (0) in the record:
- `key_offset` (noise PR 50, modded-nanogpt PR 169, bidirectional form): through the `keys` hook, on the raw keys
  before QK-norm and RoPE, m = round(key_offset d_h / 2) of each head's rotation pairs (both channels of a pair, so a
  pair rotates as one) come from the previous token, and the next m pairs below them from the next token. They are
  the lowest-frequency pairs, the near-stationary ones PR 169 shifted. A token whose neighbor is in another protein
  (the document mask's boundary) or past the window keeps its own key there. Keys come from inputs, so a masked
  neighbor contributes the MASK embedding's key, never its label.
- `score_cap`: cap tanh(s / cap) on the scaled score q.k / sqrt(d_h) before the distance terms (inside the flex score
  function as a constant; capping the full score would floor far tokens' penalty at -cap and erase the slopes). With
  unit-RMS q and k the scaled score lies in [-sqrt(d_h), sqrt(d_h)], 8 at 12 heads, so a cap of 3 to 8 binds. The
  dense path (CPU and evaluation) computes the capped scores explicitly instead of SDPA.

Symbols: b windows or proteins, l tokens, d = 768, h heads, d_h per-head width, q query heads, p = h / 2 pairs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.positional import Positional, Rotary
from speedrunning_plms.strategies.strategy import ForwardContext, ParameterGroup, Strategy, StrategyConfig, norm


DIFFERENTIAL = ("differential", "triangle_differential")
TRIANGLE = ("triangle", "triangle_differential")


@dataclass(frozen=True)
class AttentionConfig(StrategyConfig):
    variant: str = "standard"
    heads: int = 12
    key_offset: float = 0.0  # share of each head's key pairs from the previous token, and as many from the next
    score_cap: float = 0.0  # tanh cap on the scaled scores; 0 is off
    VARIANTS: ClassVar[tuple[str, ...]] = ("standard", "differential", "triangle", "triangle_differential", "paired")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"heads": (6, 12), "key_offset": (0.0, 0.5),  # d_h of 128 to 64
                                                        "score_cap": (0.0, 8.0)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.heads not in (6, 12):
            raise ValueError(f"heads must be 6 or 12 (d_h 128 or 64), not {self.heads}")
        if self.score_cap != 0.0 and self.score_cap < 3.0:
            raise ValueError(f"score_cap is 0 (off) or 3 to 8, not {self.score_cap}")


def split_rotation_pairs(X: Tensor) -> tuple[Tensor, Tensor]:
    """Two sub-heads of d_h / 2 that each hold whole RoPE pairs (coordinates i and i + d_h / 2): even and odd i."""
    # X: (..., d_h); returns two (..., d_h / 2).
    halves = X.unflatten(-1, (2, X.shape[-1] // 4, 2))  # (..., 2, d_h / 4, 2): half, pair index / 2, parity
    return halves[..., 0].flatten(-2), halves[..., 1].flatten(-2)  # each (..., d_h / 2)


def combine_differential(Y: Tensor, lam: Tensor) -> Tensor:
    """Each pair of query heads, attn_even - lambda attn_odd."""
    # Y: (b, 2h, l, d_h) with pairs adjacent; lam: (b, h, l, 1) in (0, 1); returns (b, h, l, d_h).
    pairs = Y.unflatten(1, (Y.shape[1] // 2, 2))  # (b, h, 2, l, d_h)
    return pairs[:, :, 0] - lam.to(Y.dtype) * pairs[:, :, 1]  # (b, h, l, d_h)


def capped_attention(Q: Tensor, K: Tensor, V: Tensor, attention_mask: Tensor, cap: float) -> Tensor:
    """Attention with cap tanh(s / cap) on the scaled scores before the additive mask, the dense twin of the flex
    score function under `score_cap`."""
    # Q: (b, q, n, e); K, V: (b, k, n, e); attention_mask: (b, q or 1, n, n) additive; returns (b, q, n, e).
    group = Q.shape[1] // K.shape[1]
    if group > 1:
        K, V = K.repeat_interleave(group, dim=1), V.repeat_interleave(group, dim=1)  # (b, q, n, e) each
    scores = (Q.float() @ K.float().transpose(-2, -1)) * Q.shape[-1] ** -0.5  # (b, q, n, n)
    scores = cap * torch.tanh(scores / cap) + attention_mask.float()  # (b, q, n, n)
    return (torch.softmax(scores, dim=-1).to(V.dtype) @ V)  # (b, q, n, e)


def shift_key_pairs(K: Tensor, input_ids: Tensor, pairs: int) -> Tensor:
    """The raw keys with their last `pairs` rotation pairs from the previous token and the `pairs` before those from
    the next, within each protein; a token at a protein's edge or the window's keeps its own."""
    # K: (b, l, h, d_h), raw keys; input_ids: (b, l); returns (b, l, h, d_h).
    half = K.shape[-1] // 2  # pair i is channels i and i + half
    document = documents(input_ids)  # (b, l)
    same_previous = torch.cat((torch.zeros_like(document[:, :1], dtype=torch.bool), document[:, 1:] == document[:, :-1]), dim=1)  # (b, l)
    same_next = torch.cat((document[:, :-1] == document[:, 1:], torch.zeros_like(document[:, :1], dtype=torch.bool)), dim=1)  # (b, l)
    previous = torch.cat((K[:, :1], K[:, :-1]), dim=1)  # (b, l, h, d_h), token t - 1's key at t
    following = torch.cat((K[:, 1:], K[:, -1:]), dim=1)  # (b, l, h, d_h), token t + 1's key at t
    pair = torch.arange(K.shape[-1], device=K.device) % half  # (d_h,), each channel's pair
    from_previous = pair >= half - pairs  # (d_h,)
    from_next = (pair >= half - 2 * pairs) & ~from_previous  # (d_h,)
    K = torch.where(from_previous & same_previous[:, :, None, None], previous, K)  # (b, l, h, d_h)
    return torch.where(from_next & same_next[:, :, None, None], following, K)  # (b, l, h, d_h)


class Attention(nn.Module):
    def __init__(self, index: int, config: AttentionConfig, positional: Positional, positional_holder: nn.Module,
                 key_hooks: list[Strategy], query_hooks: list[Strategy], value_hooks: list[Strategy],
                 output_hooks: list[Strategy], kernel_options: dict[str, str] | None) -> None:
        super().__init__()
        d = benchmark.HIDDEN
        self.index = index
        self.kernel = config.variant
        self.heads = config.heads  # h
        self.d_h = d // config.heads
        query_heads = 2 * config.heads if config.variant in DIFFERENTIAL else config.heads
        self.query = nn.Linear(d, query_heads * self.d_h, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.output = nn.Linear(d, d, bias=False)
        if config.variant in DIFFERENTIAL:
            self.differential_lambda = nn.Linear(d, config.heads, bias=False)  # zero-initialized by the strategy
        self.rotary: Rotary = positional_holder.rotary  # type: ignore[assignment]
        self.rotate = positional.rotates(index)
        if positional.variant == "alibi":
            self.register_buffer("slopes", positional_holder.slopes, persistent=False)  # (k,)
        else:
            self.slopes = None
        self.positional = positional
        self.score_cap = config.score_cap
        self.kernel_options = kernel_options
        self.key_hooks = key_hooks
        self.query_hooks = query_hooks
        self.value_hooks = value_hooks
        self.output_hooks = output_hooks

    def forward(self, X: Tensor, context: ForwardContext) -> Tensor:
        # X: (b, l, d), the normalized stream; returns (b, l, d).
        b, l, d = X.shape
        Q = norm(self.query(X).view(b, l, -1, self.d_h))  # (b, l, q, d_h)
        K = self.key(X).view(b, l, self.heads, self.d_h)  # (b, l, h, d_h), raw
        for hook in self.key_hooks:
            K = hook.keys(self.index, K, context)  # (b, l, h, d_h)
        K = norm(K)  # (b, l, h, d_h)
        V = self.value(X).view(b, l, self.heads, self.d_h)  # (b, l, h, d_h)
        if self.rotate:
            Q, K = self.rotary(Q), self.rotary(K)  # (b, l, q, d_h), (b, l, h, d_h)
        for hook in self.query_hooks:
            Q = hook.queries(self.index, Q, context)  # (b, l, q, d_h)
        for hook in self.value_hooks:
            V = hook.values(self.index, V, context)  # (b, l, h, d_h)
        if self.kernel == "paired":
            Y = self._paired(Q, K, V, context)  # (b, h, l, d_h)
        else:
            Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # (b, q, l, d_h), (b, h, l, d_h) twice
            Y = self._triangle(Q, K, V, context) if self.kernel in TRIANGLE else self._full(Q, K, V, context)  # (b, q, l, d_h)
            if self.kernel in DIFFERENTIAL:
                lam = torch.sigmoid(self.differential_lambda(X).float()).transpose(1, 2).unsqueeze(-1)  # (b, h, l, 1)
                Y = combine_differential(Y, lam)  # (b, h, l, d_h)
        for hook in self.output_hooks:
            Y = hook.attention_output(self.index, X, Y)  # (b, h, l, d_h)
        return self.output(Y.transpose(1, 2).reshape(b, l, d))  # (b, l, d)

    def _attend(self, Q: Tensor, K: Tensor, V: Tensor, block_mask: BlockMask | None, allowed: Tensor | None,
                pair_positions: bool = False) -> Tensor:
        """One attention call: flex with the block mask in training, SDPA with the boolean mask otherwise."""
        # Q: (b, q, n, e); K, V: (b, k, n, e) with q a multiple of k; allowed: (b, 1, n or 1, n) bool; returns (b, q, n, e).
        group = Q.shape[1] // K.shape[1]
        if block_mask is not None:
            score_mod = self.positional.score_mod(self.slopes, group, pair_positions, self.score_cap)
            if self.kernel_options is not None:
                return flex_attention(Q, K, V, score_mod=score_mod, block_mask=block_mask, enable_gqa=group > 1,
                                      kernel_options=self.kernel_options)  # (b, q, n, e)
            return flex_attention(Q, K, V, score_mod=score_mod, block_mask=block_mask, enable_gqa=group > 1)  # (b, q, n, e)
        assert allowed is not None
        bias = self.positional.dense_bias(self.slopes, group, Q.shape[2], pair_positions, Q.dtype, Q.device)  # (1, q or 1, n, n)
        attention_mask = bias.masked_fill(~allowed, float("-inf"))  # (b, q or 1, n, n)
        if self.score_cap != 0.0:
            return capped_attention(Q, K, V, attention_mask, self.score_cap)  # (b, q, n, e)
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=attention_mask, enable_gqa=group > 1)  # (b, q, n, e)

    def _full(self, Q: Tensor, K: Tensor, V: Tensor, context: ForwardContext) -> Tensor:
        # Q: (b, q, l, d_h); K, V: (b, h, l, d_h); returns (b, q, l, d_h).
        block_mask = context.block_masks[0] if context.block_masks is not None else None
        return self._attend(Q, K, V, block_mask, context.dense_mask)  # (b, q, l, d_h)

    def _triangle(self, Q: Tensor, K: Tensor, V: Tensor, context: ForwardContext) -> Tensor:
        # Q: (b, q, l, d_h); K, V: (b, h, l, d_h); returns (b, q, l, d_h), the down and up halves side by side.
        Q_down, Q_up = split_rotation_pairs(Q)  # each (b, q, l, d_h / 2)
        K_down, K_up = split_rotation_pairs(K)  # each (b, h, l, d_h / 2)
        V_down, V_up = V.chunk(2, dim=-1)  # each (b, h, l, d_h / 2); values carry no rotation
        if context.block_masks is not None:
            down, up = context.block_masks
            Y_down = self._attend(Q_down, K_down, V_down, down, None)  # (b, q, l, d_h / 2)
            Y_up = self._attend(Q_up, K_up, V_up, up, None)  # (b, q, l, d_h / 2)
        else:
            assert context.dense_mask is not None
            l = Q.shape[2]
            positions = torch.arange(l, device=Q.device)  # (l,)
            lower = positions[:, None] >= positions[None, :]  # (l, l)
            diagonal = positions[:, None] == positions[None, :]  # (l, l); every row keeps its own entry, padding too
            Y_down = self._attend(Q_down, K_down, V_down, None, (context.dense_mask & lower) | diagonal)  # (b, q, l, d_h / 2)
            Y_up = self._attend(Q_up, K_up, V_up, None, (context.dense_mask & ~lower) | diagonal)  # (b, q, l, d_h / 2)
        return torch.cat((Y_down, Y_up), dim=-1)  # (b, q, l, d_h)

    def _paired(self, Q: Tensor, K: Tensor, V: Tensor, context: ForwardContext) -> Tensor:
        # Q, K, V: (b, l, h, d_h); returns (b, h, l, d_h).
        b, l, h, d_h = Q.shape

        def interleave(X: Tensor) -> Tensor:
            # X: (b, l, h, d_h); the two heads of a pair alternate along the sequence.
            return X.view(b, l, h // 2, 2, d_h).permute(0, 2, 1, 3, 4).reshape(b, h // 2, 2 * l, d_h)  # (b, p, 2l, d_h)

        block_mask = context.block_masks[0] if context.block_masks is not None else None
        allowed = None
        if block_mask is None:
            assert context.dense_mask is not None
            allowed = context.dense_mask.repeat_interleave(2, dim=-1)  # (b, 1, l or 1, 2l)
            if allowed.shape[2] > 1:
                allowed = allowed.repeat_interleave(2, dim=2)  # (b, 1, 2l, 2l)
        Y = self._attend(interleave(Q), interleave(K), interleave(V), block_mask, allowed, pair_positions=True)  # (b, p, 2l, d_h)
        return Y.view(b, h // 2, l, 2, d_h).permute(0, 1, 3, 2, 4).reshape(b, h, l, d_h)  # (b, h, l, d_h)


class AttentionKernel(Strategy):
    family = "attention"
    Config = AttentionConfig
    config: AttentionConfig

    def slope_heads(self) -> int:
        """How many attention heads carry a slope: the logical heads, or the pairs of paired-head attention."""
        return self.config.heads // 2 if self.variant == "paired" else self.config.heads

    def attention(self, model: nn.Module, index: int) -> Attention:
        """Block `index`'s attention sublayer, with the hooks of the strategies that act inside it."""
        strategies = model.strategies  # type: ignore[union-attr]
        return Attention(index, self.config, strategies.positional, model.extensions["positional"],  # type: ignore[index]
                         strategies.hooked("keys"), strategies.hooked("queries"), strategies.hooked("values"),
                         strategies.hooked("attention_output"), strategies.kernel.kernel_options())

    def overrides(self, hook: str) -> bool:
        return hook == "keys" and self.config.key_offset > 0.0

    def keys(self, index: int, K: Tensor, context: ForwardContext) -> Tensor:
        # K: (b, l, h, d_h), raw keys.
        pairs = round(self.config.key_offset * K.shape[-1] / 2)
        return shift_key_pairs(K, context.input_ids, pairs) if pairs else K  # (b, l, h, d_h)

    def initialize(self, model: nn.Module) -> None:
        if self.variant in DIFFERENTIAL:
            for block in model.blocks:  # type: ignore[union-attr]
                nn.init.zeros_(block.attention.differential_lambda.weight)  # (h, d): every pair starts at lambda 0.5

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if self.variant not in DIFFERENTIAL:
            return []
        weights = [block.attention.differential_lambda.weight for block in model.blocks]  # type: ignore[union-attr]
        return [ParameterGroup("differential_lambda", weights, "adam", model.strategies.optimizer.config.scalar_lr)]  # type: ignore[union-attr]

    def training_masks(self, input_ids: Tensor) -> tuple[BlockMask, ...]:
        """The flex block masks of one training batch: tokens attend within their own protein, and the kernel's own
        rule on top (directional halves for the triangle, interleaved positions for paired heads)."""
        # input_ids: (b, l) of packed windows.
        document = documents(input_ids)  # (b, l)
        b, l = input_ids.shape
        if self.variant == "paired":
            def same_document_paired(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
                # batch, head, query, key: () each, one attention entry; returns ().
                return document[batch, query // 2] == document[batch, key // 2]  # ()

            return (create_block_mask(same_document_paired, b, None, 2 * l, 2 * l, device=input_ids.device, _compile=True),)
        if self.variant in TRIANGLE:
            def down(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
                # batch, head, query, key: () each, one attention entry; returns ().
                return (document[batch, query] == document[batch, key]) & (query >= key)  # ()

            def up(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
                # batch, head, query, key: () each, one attention entry; returns ().
                return (document[batch, query] == document[batch, key]) & (query <= key)  # ()

            return tuple(create_block_mask(rule, b, None, l, l, device=input_ids.device, _compile=True) for rule in (down, up))

        def same_document(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
            # batch, head, query, key: () each, one attention entry; returns ().
            return document[batch, query] == document[batch, key]  # ()

        return (create_block_mask(same_document, b, None, l, l, device=input_ids.device, _compile=True),)


def documents(input_ids: Tensor) -> Tensor:
    """Each position's document in a packed window: the count of CLS tokens up to it, and -1 for padding, so a
    protein never attends to the padding the `whole` batch leaves at a window's end (the record's windows have none)."""
    # input_ids: (b, l).
    document = (input_ids == benchmark.CLS_ID).cumsum(dim=1)  # (b, l)
    return document.masked_fill(input_ids == benchmark.PAD_ID, -1)  # (b, l)


def document_dense_mask(input_ids: Tensor) -> Tensor:
    """The document rule as a dense bool mask (b, 1, l, l), for devices without flex attention."""
    # input_ids: (b, l).
    document = documents(input_ids)  # (b, l)
    return (document[:, :, None] == document[:, None, :])[:, None]  # (b, 1, l, l)
