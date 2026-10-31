"""Multi-head self-attention on a selectable backend, with fixed rotary tables.

The attention of the probes that read protein language model residue embeddings: Protify's
transformer probe and the InterpNet probe. Unlike `foundry.layers.rotary`, whose tables are cached
by length, `RotaryEmbedding` here computes its tables once up to `max_seq_len`, as those probes'
checkpoints were trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from transformers import initialization as init

from .attention_backends import (
    AttentionBackend,
    BlockMask,
    flex_attention_func,
    kernels_attention_func,
    resolve_attention_backend,
    sdpa_attention_func,
)


Linear = nn.Linear
LayerNorm = nn.LayerNorm


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    """Rotate each feature pair by 90 degrees, as rotary position embeddings require."""
    # x: (..., d_r), where d_r is the even rotary width
    if not interleaved:
        first, second = x.chunk(2, dim=-1)  # each: (..., d_r / 2)
        return torch.cat((-second, first), dim=-1)  # (..., d_r)

    first, second = x[..., ::2], x[..., 1::2]  # each: (..., d_r / 2)
    return rearrange(
        torch.stack((-second, first), dim=-1),  # (..., d_r / 2, 2)
        "... d two -> ... (d two)",
        two=2,
    )  # (..., d_r)


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    """Rotate the leading d_r features of each head by position; pass the rest through."""
    # x: (b, l, h, d_h); cos, sin: (l_t, d_r / 2), tables for l_t >= l positions
    rotary_dim = cos.shape[-1] * 2  # d_r
    assert rotary_dim <= x.shape[-1], "Rotary dimension cannot exceed head dimension."
    seq_len = x.size(1)  # l
    # Interleaved pairs are neighboring channels, so each frequency repeats per pair, as (d 2).
    pattern = "s d -> 1 s 1 (d 2)" if interleaved else "s d -> 1 s 1 (2 d)"
    cos = repeat(cos[:seq_len], pattern)  # (1, l, 1, d_r)
    sin = repeat(sin[:seq_len], pattern)  # (1, l, 1, d_r)
    rotated = rotate_half(x[..., :rotary_dim], interleaved)  # (b, l, h, d_r)
    return torch.cat(
        (
            x[..., :rotary_dim] * cos + rotated * sin,  # (b, l, h, d_r)
            x[..., rotary_dim:],  # (b, l, h, d_h - d_r)
        ),
        dim=-1,
    )  # (b, l, h, d_h)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding tables for one head width, computed once up to `max_seq_len`."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scaling_factor: float = 1.0,
        max_seq_len: int = 2048,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim  # d_h
        self.base = float(base)
        self.interleaved = interleaved
        self.scaling_factor = scaling_factor
        self.max_seq_len = max_seq_len  # l_max
        inv_freq, cos_k, sin_k = self.compute_tables(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # (d_h / 2,)
        self.register_buffer("_cos_k", cos_k, persistent=False)  # (l_max, d_h / 2)
        self.register_buffer("_sin_k", sin_k, persistent=False)  # (l_max, d_h / 2)

    def compute_tables(
        self,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the inverse frequencies and the cosine and sine tables, in float32."""
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )  # (d_h / 2,)
        positions = (
            torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
            / self.scaling_factor
        )  # (l_max,)
        freqs = torch.outer(positions, inv_freq)  # (l_max, d_h / 2)
        return inv_freq, torch.cos(freqs), torch.sin(freqs)  # (d_h / 2,), (l_max, d_h / 2), (l_max, d_h / 2)

    @torch.no_grad()
    def rebuild_tables(self) -> None:
        """Refill the tables from the formula the constructor uses.

        The tables are registered with `persistent=False`, so no checkpoint holds them. Transformers
        5 builds a model on the meta device and allocates every tensor a checkpoint did not supply
        as uninitialized memory, so the `_init_weights` of every pretrained model that holds one
        calls this after `from_pretrained`. `init.copy_` leaves a tensor Transformers has
        already marked initialized as it is.
        """
        inv_freq, cos_k, sin_k = self.compute_tables(self.inv_freq.device)  # (d_h / 2,), (l_max, d_h / 2) twice
        init.copy_(self.inv_freq, inv_freq)
        init.copy_(self._cos_k, cos_k)
        init.copy_(self._sin_k, sin_k)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate queries and keys by their positions."""
        # query_states, key_states: (b, l, h, d_h)
        seq_len = query_states.shape[1]  # l
        assert seq_len <= self.max_seq_len, f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
        cos = self._cos_k[:seq_len]  # (l, d_h / 2)
        sin = self._sin_k[:seq_len]  # (l, d_h / 2)
        return (
            apply_rotary_emb_torch(query_states, cos, sin, self.interleaved),  # (b, l, h, d_h)
            apply_rotary_emb_torch(key_states, cos, sin, self.interleaved),  # (b, l, h, d_h)
        )


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention that runs on the configured backend."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        rotary: bool = True,
        attention_backend: str = "flex",
        use_bias: bool = False,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size  # d
        self.n_heads = n_heads  # h
        self.d_head = hidden_size // n_heads  # d_h
        assert self.d_head * self.n_heads == self.hidden_size, "hidden_size must be divisible by n_heads."
        self.scale = 1.0 / (self.d_head ** 0.5)
        self.q_proj = Linear(hidden_size, hidden_size, bias=use_bias)
        self.k_proj = Linear(hidden_size, hidden_size, bias=use_bias)
        self.v_proj = Linear(hidden_size, hidden_size, bias=use_bias)
        self.out_proj = Linear(hidden_size, hidden_size, bias=use_bias)
        self.rotary = RotaryEmbedding(self.d_head, max_seq_len=max_seq_len) if rotary else None
        self.attention_backend = resolve_attention_backend(attention_backend)

    def set_attention_backend(self, attention_backend: str) -> None:
        """Select the attention implementation that later forward passes run."""
        self.attention_backend = resolve_attention_backend(attention_backend)

    def prepare_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to per-head queries, keys, and values, rotating queries and keys."""
        # hidden_states: (b, l, d)
        batch_size, seq_len, _ = hidden_states.shape  # b, l, d
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.n_heads, self.d_head
        )  # (b, l, h, d_h)
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.n_heads, self.d_head
        )  # (b, l, h, d_h)
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.n_heads, self.d_head
        )  # (b, l, h, d_h)
        if self.rotary is not None:
            query_states, key_states = self.rotary(query_states, key_states)  # each: (b, l, h, d_h)

        return query_states, key_states, value_states  # each: (b, l, h, d_h)

    @torch.no_grad()
    def _compute_s_max(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Bound the largest scaled query-key dot product of each head over the batch."""
        # query_states, key_states: (b, l, h, d_h)
        query_bhld = query_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        key_bhld = key_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        query_norm = torch.linalg.vector_norm(query_bhld, dim=-1)  # (b, h, l)
        key_norm = torch.linalg.vector_norm(key_bhld, dim=-1)  # (b, h, l)
        s_max_bound = (
            query_norm.max(dim=-1).values  # (b, h)
            * key_norm.max(dim=-1).values  # (b, h)
        ).max(dim=0).values * self.scale  # (h,)
        return [s_max_bound[head_idx] for head_idx in range(self.n_heads)]  # h scalar tensors: ()

    def _manual_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask_4d: torch.Tensor | None,
        output_s_max: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        """Compute attention eagerly, the one path that returns the attention weights."""
        # query_states, key_states, value_states: (b, l, h, d_h)
        # attention_mask_4d: (b, 1, 1, l), (b, 1, l, l), or None
        query_bhld = query_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        key_bhld = key_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        value_bhld = value_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        attention_logits = torch.matmul(query_bhld, key_bhld.transpose(-2, -1)) * self.scale  # (b, h, l, l)
        if attention_mask_4d is not None:
            attention_logits = attention_logits.masked_fill(
                attention_mask_4d.logical_not(), float("-inf")
            )  # (b, h, l, l)

        attention_weights = F.softmax(attention_logits, dim=-1)  # (b, h, l, l)
        attn_output = torch.matmul(attention_weights, value_bhld)  # (b, h, l, d_h)
        s_max = self._compute_s_max(query_states, key_states) if output_s_max else None
        return attn_output, attention_weights, s_max  # (b, h, l, d_h), (b, h, l, l), optional h scalars

    def _dispatch_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask_2d: torch.Tensor | None,
        attention_mask_4d: torch.Tensor | None,
        flex_block_mask: BlockMask | None,
        output_attentions: bool,
        output_s_max: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Run the configured backend, or eager attention when the weights are requested."""
        # query_states, key_states, value_states: (b, l, h, d_h); attention_mask_2d: (b, l) or None
        # attention_mask_4d: (b, 1, 1, l), (b, 1, l, l), or None
        if output_attentions:
            return self._manual_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask_4d=attention_mask_4d,
                output_s_max=output_s_max,
            )  # (b, h, l, d_h), (b, h, l, l), optional h scalars

        query_bhld = query_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        key_bhld = key_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        value_bhld = value_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)

        if self.attention_backend == AttentionBackend.KERNELS:
            attn_output = kernels_attention_func(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask_2d=attention_mask_2d,
            ).transpose(1, 2).contiguous()  # (b, h, l, d_h)
        elif self.attention_backend == AttentionBackend.FLEX:
            attn_output = flex_attention_func(
                query_states=query_bhld,
                key_states=key_bhld,
                value_states=value_bhld,
                flex_block_mask=flex_block_mask,
            )  # (b, h, l, d_h)
        elif self.attention_backend == AttentionBackend.SDPA:
            attn_output = sdpa_attention_func(
                query_states=query_bhld,
                key_states=key_bhld,
                value_states=value_bhld,
                attention_mask_4d=attention_mask_4d,
            )  # (b, h, l, d_h)
        else:
            raise AssertionError(f"Unsupported attention backend: {self.attention_backend}.")

        s_max = self._compute_s_max(query_states, key_states) if output_s_max else None
        return attn_output, None, s_max  # (b, h, l, d_h), None, optional h scalars

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: BlockMask | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Attend over a padded batch; optionally return the weights and the s_max bounds."""
        # hidden_states: (b, l, d); attention_mask_2d: (b, l) or None
        # attention_mask_4d: (b, 1, 1, l), (b, 1, l, l), or None
        query_states, key_states, value_states = self.prepare_qkv(hidden_states)  # each: (b, l, h, d_h)
        attn_output, attention_weights, s_max = self._dispatch_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )  # (b, h, l, d_h), optional (b, h, l, l), optional h scalars
        attn_output = attn_output.transpose(1, 2).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.hidden_size,
        )  # (b, l, d)
        output = self.out_proj(attn_output)  # (b, l, d)
        return output, attention_weights, s_max  # (b, l, d), optional (b, h, l, l), optional h scalars
