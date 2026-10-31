"""The annotation transformer's block: self-attention, then a SwiGLU feed-forward block."""

from __future__ import annotations

from typing import Protocol
from torch import Tensor, nn

from .attention import MultiHeadAttention
from .feedforward import swiglu_ffn


__all__ = ["TransformerBlock", "TransformerBlockConfig"]


class TransformerBlockConfig(Protocol):
    """The settings a `TransformerBlock` reads, which each annotation transformer config holds."""

    hidden_size: int
    n_heads: int
    rotary: bool
    expansion_ratio: float
    dropout: float


class TransformerBlock(nn.Module):
    """Self-attention, then a SwiGLU feed-forward block, each pre-normalized and residual.

    `derive_frequencies` goes to the attention's rotary embedding.
    """

    def __init__(self, config: TransformerBlockConfig, *, derive_frequencies: bool = False) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            rotary=config.rotary,
            causal=False,
            derive_frequencies=derive_frequencies,
        )
        self.ffn = swiglu_ffn(config.hidden_size, config.expansion_ratio, config.dropout)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Add the attention's output to `x`, then the feed-forward block's."""
        # x: (b, l, d); attention_mask: (b, l)
        x = self.attention(x, attention_mask) + x  # (b, l, d)
        x = self.ffn(x) + x  # (b, l, d)
        return x  # (b, l, d)
