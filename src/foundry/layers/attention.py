"""Self-attention and attention pooling, as the annotation transformer builds them.

`MultiHeadAttention` is pre-normalized self-attention whose queries and keys are layer-normalized
and rotated by their positions. `AttentionPooler` pools a sequence into a fixed number of states
by attention from learned queries. `MultiHeadAttentionPooler` pools the same way at the model
width, as DSM, confounders, and synth's translators build it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from functools import partial
from einops import rearrange
from torch import Tensor, nn

from .linear import Linear, LinearLayer, ParameterLayer
from .rotary import RotaryEmbedding


__all__ = ["AttentionPooler", "MultiHeadAttention", "MultiHeadAttentionPooler"]


class MultiHeadAttention(nn.Module):
    """Pre-normalized multi-head self-attention with normalized queries and keys.

    A layer normalization and one projection without a bias make the queries, keys, and values.
    The queries and keys are layer-normalized again, without a bias, and with `rotary` turned by
    their positions (`foundry.layers.rotary`, where `derive_frequencies` is explained). PyTorch's
    scaled dot-product attention combines them, and a last projection without a bias returns to
    the model width. `norm_bias=False` leaves the first layer normalization without a bias.
    `spectral_norm` normalizes both projections (`foundry.layers.linear.LinearLayer`), and
    `xavier_init=False` keeps `nn.Linear`'s default initialization of both.
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        rotary: bool = True,
        causal: bool = False,
        *,
        derive_frequencies: bool = False,
        xavier_init: bool = True,
        norm_bias: bool = True,
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size  # d
        self.n_heads = n_heads  # h
        self.d_head = self.hidden_size // self.n_heads  # d_h
        projection = partial(
            LinearLayer, bias=False, spectral_norm=spectral_norm, xavier_init=xavier_init
        )
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(hidden_size, bias=norm_bias),
            projection(hidden_size, hidden_size * 3),
        )
        self.out_proj = projection((hidden_size // n_heads) * n_heads, hidden_size)
        self.q_ln = nn.LayerNorm(hidden_size, bias=False)
        self.k_ln = nn.LayerNorm(hidden_size, bias=False)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = (
            RotaryEmbedding(hidden_size // n_heads, derive_frequencies=derive_frequencies)
            if rotary
            else None
        )
        self.causal = causal

    def _apply_rotary(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate the queries and keys of each head by their positions."""
        # q, k: (b, l, d)
        assert self.rotary is not None
        q = q.unflatten(-1, (self.n_heads, self.d_head))  # (b, l, h, d_h)
        k = k.unflatten(-1, (self.n_heads, self.d_head))  # (b, l, h, d_h)
        q, k = self.rotary(q, k)  # each (b, l, h, d_h)
        q = q.flatten(-2, -1)  # (b, l, d)
        k = k.flatten(-2, -1)  # (b, l, d)
        return q, k  # each (b, l, d)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Attend over the sequence, to the keys that `attention_mask` marks.

        A `(b, l)` mask is expanded to `(b, 1, l, l)`, and a mask of any other shape reaches
        scaled dot-product attention as given. Causal attention ignores the mask.
        """
        # x: (b, l, d); attention_mask: (b, l), or broadcastable to (b, h, l, l)
        b, length, _ = x.shape  # b, l, d
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :].expand(  # (b, 1, l, l)
                b,
                1,
                length,
                length,
            ).bool()

        QKV = self.layernorm_qkv(x)  # (b, l, 3 * d)
        Q, K, V = torch.chunk(QKV, 3, dim=-1)  # each (b, l, d)
        Q, K = self.q_ln(Q).to(Q.dtype), self.k_ln(K).to(Q.dtype)  # each (b, l, d)

        if self.rotary is not None:
            Q, K = self._apply_rotary(Q, K)  # each (b, l, d)

        Q, K, V = map(self.reshaper, (Q, K, V))  # each (b, h, l, d_h)
        attention_output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attention_mask if not self.causal else None,
            is_causal=self.causal,
        )  # (b, h, l, d_h)
        attention_output = rearrange(attention_output, "b h s d -> b s (h d)")  # (b, l, d)
        output = self.out_proj(attention_output)  # (b, l, d)
        return output  # (b, l, d)


class AttentionPooler(nn.Module):
    """Pool a sequence into `n_tokens` states by attention from learned queries.

    The sequence is projected to `intermediate_size` first, and attention runs in
    `intermediate_size // 64` heads.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, n_tokens: int = 1) -> None:
        super().__init__()
        self.n_tokens = n_tokens  # n
        self.n_heads = intermediate_size // 64  # h
        self.d_head = 64  # d_h
        self.input = Linear(hidden_size, intermediate_size)
        self.Q = ParameterLayer((1, n_tokens, intermediate_size))  # (1, n, d_i)
        self.Wq = Linear(intermediate_size, intermediate_size)
        self.Wv = Linear(intermediate_size, intermediate_size)
        self.Wk = Linear(intermediate_size, intermediate_size)
        self.Wo = Linear(intermediate_size, hidden_size)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Pool the sequence, attending to the positions that `attention_mask` marks."""
        # x: (b, l, d); attention_mask: (b, l); d_i is the pooler's attention width
        b, length, _ = x.size()  # b, l, d
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(  # (b, 1, n, l)
                b,
                1,
                self.n_tokens,
                length,
            ).bool()

        x = self.input(x)  # (b, l, d_i)
        Q = self.Wq(self.Q).expand(b, -1, -1)  # (b, n, d_i)
        V = self.Wv(x)  # (b, l, d_i)
        K = self.Wk(x)  # (b, l, d_i)
        Q, K, V = map(self.reshaper, (Q, K, V))  # Q: (b, h, n, d_h); K, V: (b, h, l, d_h)
        pooled = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attention_mask,
            is_causal=False,
        )  # (b, h, n, d_h)
        pooled = rearrange(pooled, "b h s d -> b s (h d)")  # (b, n, d_i)
        output = self.Wo(pooled)  # (b, n, d)
        return output  # (b, n, d)


class MultiHeadAttentionPooler(nn.Module):
    """Pool a sequence into `n_tokens` states by attention from learned queries, at the model width.

    Unlike `AttentionPooler`, the sequence is not projected to another width first: the learned
    queries and the projections of queries, keys, values, and output are `hidden_size` wide, and
    attention runs in `n_heads` heads. `bias` gives the projections a bias, and `spectral_norm`
    normalizes them (`foundry.layers.linear.LinearLayer`). `xavier_init=False` keeps
    `nn.Linear`'s default initialization of the projections and the learned queries' standard
    normal values.
    """

    def __init__(
        self,
        hidden_size: int,
        n_tokens: int = 1,
        n_heads: int = 16,
        *,
        bias: bool = False,
        spectral_norm: bool = False,
        xavier_init: bool = True,
    ) -> None:
        super().__init__()
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        self.n_tokens = n_tokens  # n
        self.d_head = hidden_size // n_heads  # d_h
        projection = partial(
            LinearLayer, bias=bias, spectral_norm=spectral_norm, xavier_init=xavier_init
        )
        self.Q = ParameterLayer((1, n_tokens, hidden_size), xavier_init=xavier_init)  # (1, n, d)
        self.Wq = projection(hidden_size, hidden_size)
        self.Wv = projection(hidden_size, hidden_size)
        self.Wk = projection(hidden_size, hidden_size)
        self.Wo = projection(hidden_size, hidden_size)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Pool the sequence, attending to the positions that `attention_mask` marks.

        A `(b, l)` mask is expanded to `(b, 1, n, l)`, and a mask of any other shape reaches
        scaled dot-product attention as given.
        """
        # x: (b, l, d); attention_mask: (b, l), or broadcastable to (b, h, n, l)
        b, length, _ = x.size()  # b, l, d
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :].expand(  # (b, 1, n, l)
                b,
                1,
                self.n_tokens,
                length,
            ).bool()

        Q = self.Wq(self.Q).expand(b, -1, -1)  # (b, n, d)
        V = self.Wv(x)  # (b, l, d)
        K = self.Wk(x)  # (b, l, d)
        Q, K, V = map(self.reshaper, (Q, K, V))  # Q: (b, h, n, d_h); K, V: (b, h, l, d_h)
        pooled = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attention_mask,
            is_causal=False,
        )  # (b, h, n, d_h)
        pooled = rearrange(pooled, "b h s d -> b s (h d)")  # (b, n, d)
        output = self.Wo(pooled)  # (b, n, d)
        return output  # (b, n, d)
