# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Changed from EvolutionaryScale's esm/layers/rotary.py, which carries the notice above: typed
# and shape-traced, frequencies optionally derived at each table rebuild, a cross-attention
# variant, interleaved tables repeated per channel pair, and the unused xPos tables removed.
"""Rotary position embeddings whose cosine and sine tables are cached by length.

`RotaryEmbedding` rotates the queries and keys of self-attention, and `RotaryEmbeddingCross`
those of cross-attention, where the two may differ in length. The code is ESM3's rotary
embedding, `esm/layers/rotary.py`.
"""

from __future__ import annotations

import torch

from einops import rearrange, repeat
from torch import Tensor, nn


__all__ = ["RotaryEmbedding", "RotaryEmbeddingCross", "apply_rotary_emb_torch", "rotate_half"]


def rotate_half(x: Tensor, interleaved: bool = False) -> Tensor:
    """Turn each channel pair a quarter turn, so that `(x1, x2)` becomes `(-x2, x1)`.

    The pairs are the two halves of the last dimension or, with `interleaved`, neighboring
    channels.
    """
    # x: (..., d_r), where d_r is the even rotary width
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)  # each (..., d_r / 2)
        return torch.cat((-x2, x1), dim=-1)  # (..., d_r)
    x1, x2 = x[..., ::2], x[..., 1::2]  # each (..., d_r / 2)
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)  # (..., d_r)


def apply_rotary_emb_torch(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    interleaved: bool = False,
    _inplace: bool = False,
) -> Tensor:
    """Rotate the first `2 * cos.shape[-1]` channels of each head by the head's position.

    `cos` and `sin` may hold more positions than `x`, and the first `l` are used. Channels past
    the rotary width pass through. `_inplace` keeps flash-attn's signature and is ignored: the
    result is always a new tensor.
    """
    # x: (b, l, h, d_h); cos, sin: (l_c, d_r / 2), where l <= l_c and d_r <= d_h
    ro_dim = cos.shape[-1] * 2  # d_r
    assert ro_dim <= x.shape[-1], f"rotary width {ro_dim} exceeds head width {x.shape[-1]}"
    seqlen = x.size(1)  # l
    # Each frequency covers one channel pair: the matching halves, or two neighboring channels.
    pattern = "s d -> s 1 (d 2)" if interleaved else "s d -> s 1 (2 d)"
    cos = repeat(cos[:seqlen], pattern)  # (l, 1, d_r)
    sin = repeat(sin[:seqlen], pattern)  # (l, 1, d_r)
    rotated = x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin  # (b, l, h, d_r)
    return torch.cat([rotated, x[..., ro_dim:]], dim=-1)  # (b, l, h, d_h)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding for self-attention, with tables cached by length.

    The cosine and sine tables are rebuilt when a sequence is longer than they are, when the
    input's device or dtype differs, or when training meets tables built in inference mode, and
    sliced otherwise.

    `inv_freq` is registered with `persistent=False`, so no checkpoint holds it, and a model that
    loads checkpoints rebuilds it in `_init_weights`. With `derive_frequencies`, each table rebuild
    also recomputes it, in float32 on the input's device, which repairs a buffer that a load path
    skipping `_init_weights` left unfilled. The frequencies then stay float32 after the module is
    cast to a lower precision, where by default they are the cast buffer's.

    `scale_base` registers the xPos `scale` buffer, but the forward pass does not implement it.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: float | None = None,
        scaling_factor: float = 1.0,
        pos_idx_in_fp32: bool = True,
        device: torch.device | str | None = None,
        *,
        derive_frequencies: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim  # d_r
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        self.device = device
        self.derive_frequencies = derive_frequencies

        self._seq_len_cached = 0  # l_c
        self._cos_cached: Tensor | None = None  # (l_c, d_r / 2)
        self._sin_cached: Tensor | None = None  # (l_c, d_r / 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Register the inverse frequencies and, with `scale_base`, the xPos scale."""
        inv_freq = self._compute_inv_freq(self.device)  # (d_r / 2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        scale = None
        if self.scale_base is not None:
            arange = torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32)  # (d_r / 2,)
            scale = (arange + 0.4 * self.dim) / (1.4 * self.dim)  # (d_r / 2,)
        self.register_buffer("scale", scale)

    def _compute_inv_freq(self, device: torch.device | str | None = None) -> Tensor:
        """Return the geometric frequencies `base ** (-2i / dim)`."""
        channels = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)  # (d_r / 2,)
        return 1 / (self.base ** (channels / self.dim))  # (d_r / 2,)

    def _cache_length(self, seqlen: int) -> int:
        """Return how many positions the tables hold to serve a sequence of `seqlen`."""
        return seqlen

    def _update_cos_sin_cache(
        self,
        seqlen: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Rebuild the tables if they cannot serve `seqlen` positions on `device` in `dtype`."""
        cache_length = self._cache_length(seqlen)  # l_c
        if not (
            cache_length > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            return
        self._seq_len_cached = cache_length
        if self.derive_frequencies:
            self.inv_freq = self._compute_inv_freq(device)  # (d_r / 2,)
        if self.pos_idx_in_fp32:
            positions = torch.arange(cache_length, device=device, dtype=torch.float32)  # (l_c,)
            inv_freq = self.inv_freq.to(torch.float32)  # (d_r / 2,)
        else:
            positions = torch.arange(cache_length, device=device, dtype=self.inv_freq.dtype)  # (l_c,)
            inv_freq = self.inv_freq  # (d_r / 2,)
        positions /= self.scaling_factor  # (l_c,)
        freqs = torch.outer(positions, inv_freq)  # (l_c, d_r / 2)
        self._cos_cached = torch.cos(freqs).to(dtype)  # (l_c, d_r / 2)
        self._sin_cached = torch.sin(freqs).to(dtype)  # (l_c, d_r / 2)

    def _rotate(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate queries and keys, each by its own positions, with the cached tables."""
        # q: (b, l_q, h, d_h); k: (b, l_k, h, d_h)
        assert self._cos_cached is not None and self._sin_cached is not None
        return (
            apply_rotary_emb_torch(q, self._cos_cached, self._sin_cached, self.interleaved),  # (b, l_q, h, d_h)
            apply_rotary_emb_torch(k, self._cos_cached, self._sin_cached, self.interleaved),  # (b, l_k, h, d_h)
        )

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate queries and keys by their positions."""
        # q, k: (b, l, h, d_h)
        assert self.scale is None, "scaled rotary embeddings (scale_base) are not implemented"
        self._update_cos_sin_cache(q.shape[1], device=q.device, dtype=q.dtype)
        return self._rotate(q, k)  # (b, l, h, d_h), (b, l, h, d_h)


class RotaryEmbeddingCross(RotaryEmbedding):
    """Rotary position embedding for cross-attention, where queries and keys differ in length.

    Queries and keys are each rotated by their own positions. The tables hold at least
    `max_seq_len` positions, so shorter sequences never rebuild them.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: float | None = None,
        scaling_factor: float = 1.0,
        pos_idx_in_fp32: bool = True,
        device: torch.device | str | None = None,
        max_seq_len: int = 10000,
        *,
        derive_frequencies: bool = False,
    ) -> None:
        super().__init__(
            dim,
            base,
            interleaved,
            scale_base,
            scaling_factor,
            pos_idx_in_fp32,
            device,
            derive_frequencies=derive_frequencies,
        )
        self.max_seq_len = max_seq_len

    def _cache_length(self, seqlen: int) -> int:
        return max(seqlen, self.max_seq_len)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate queries and keys, each by its own positions."""
        # q: (b, l_q, h, d_h); k: (b, l_k, h, d_h)
        assert self.scale is None, "scaled rotary embeddings (scale_base) are not implemented"
        self._update_cos_sin_cache(max(q.shape[1], k.shape[1]), device=q.device, dtype=q.dtype)
        return self._rotate(q, k)  # (b, l_q, h, d_h), (b, l_k, h, d_h)
