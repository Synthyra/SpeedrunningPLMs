"""The annotation transformer's feed-forward block: layer normalization, then a SwiGLU MLP.

`correction_fn_256` sets the MLP's width, rounded up to a multiple of 256 channels. With a biased
layer normalization and `nn.Linear`'s default initialization, the block is the one DSM, synth's
translator, and InterpNet hold, ESM3's `swiglu_ln_ffn` with dropout. InterpNet's projections also
take a bias when its layers do.
"""

from __future__ import annotations

import torch.nn.functional as F

from torch import Tensor, nn

from .linear import LinearLayer


__all__ = ["SwiGLU", "correction_fn_256", "swiglu_ffn", "swiglu_ln_ffn"]


def correction_fn_256(expansion_ratio: float, hidden_size: int) -> int:
    """Return `expansion_ratio * hidden_size` rounded up to a multiple of 256."""
    return int(((expansion_ratio * hidden_size) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """Gate the second half of the last dimension with the SiLU of the first half."""

    def forward(self, x: Tensor) -> Tensor:
        """Reduce `(..., 2 * d_i)` to `(..., d_i)`."""
        # x: (..., 2 * d_i), where d_i is the MLP's width
        x1, x2 = x.chunk(2, dim=-1)  # each (..., d_i)
        gated = F.silu(x1) * x2  # (..., d_i)
        return gated  # (..., d_i)


def swiglu_ffn(
    hidden_size: int,
    expansion_ratio: float,
    dropout: float = 0.1,
    *,
    bias: bool = False,
    spectral_norm: bool = False,
    norm_bias: bool = False,
    xavier_init: bool = True,
) -> nn.Sequential:
    """Build the pre-normalized SwiGLU feed-forward block, which maps `(..., d)` to `(..., d)`.

    It is a layer normalization without a bias, a projection to twice the MLP's width, SwiGLU,
    dropout, and a projection back, with state-dict keys `0.weight`, `1.weight`, and `4.weight`.
    The width is `correction_fn_256(expansion_ratio, hidden_size)`. `bias` gives both projections
    a bias, `1.bias` and `4.bias`, which Xavier initialization starts at zero. `spectral_norm`
    applies spectral normalization to both projections. `norm_bias` gives the layer normalization
    a bias, `0.bias`, and `xavier_init=False` keeps `nn.Linear`'s default initialization of the
    projections.
    """
    intermediate_size = correction_fn_256(expansion_ratio, hidden_size)  # d_i
    return nn.Sequential(
        nn.LayerNorm(hidden_size, bias=norm_bias),  # (..., d)
        LinearLayer(  # (..., 2 * d_i)
            hidden_size,
            intermediate_size * 2,
            bias=bias,
            spectral_norm=spectral_norm,
            xavier_init=xavier_init,
        ),
        SwiGLU(),  # (..., d_i)
        nn.Dropout(dropout),  # (..., d_i)
        LinearLayer(  # (..., d)
            intermediate_size,
            hidden_size,
            bias=bias,
            spectral_norm=spectral_norm,
            xavier_init=xavier_init,
        ),
    )


def swiglu_ln_ffn(hidden_size: int, expansion_ratio: float, dropout: float = 0.1, use_bias: bool = False) -> nn.Sequential:
    """Build the feed-forward block of the residue-embedding probes, mapping `(..., d)` to `(..., d)`.

    `swiglu_ffn` with a biased layer normalization and `nn.Linear`'s default initialization, as
    Protify's transformer probe and InterpNet were trained; `use_bias` gives both projections a bias.
    """
    return swiglu_ffn(hidden_size, expansion_ratio, dropout, bias=use_bias, norm_bias=True, xavier_init=False)
