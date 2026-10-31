"""The pre-normalized transformer stack of the probes that read residue embeddings.

Protify's transformer probe and the InterpNet probe build on it, each with its own head.
"""

import torch

from dataclasses import dataclass
from torch import nn
from transformers.modeling_outputs import ModelOutput

from .attention_backends import BlockMask, build_attention_masks, resolve_attention_backend
from .feedforward import swiglu_ln_ffn
from .probe_attention import LayerNorm, MultiHeadAttention


# A `head_size` the caller left out, told apart from every value a caller could pass.
HEAD_SIZE_UNSET = object()


def resolve_head_size(
    hidden_size: int,
    head_size: int | object,
    n_heads_legacy: int | None,
    default_head_size: int,
) -> int:
    """Return the head width, deriving it from a legacy `n_heads` when a config gives only that."""
    if head_size is HEAD_SIZE_UNSET and n_heads_legacy is None:
        head_size = default_head_size
    elif head_size is HEAD_SIZE_UNSET and n_heads_legacy is not None:
        assert hidden_size % n_heads_legacy == 0, (
            f"hidden_size {hidden_size} not divisible by legacy n_heads {n_heads_legacy}"
        )
        head_size = hidden_size // n_heads_legacy
    elif head_size is not HEAD_SIZE_UNSET and n_heads_legacy is not None:
        assert hidden_size % n_heads_legacy == 0, (
            f"hidden_size {hidden_size} not divisible by legacy n_heads {n_heads_legacy}"
        )
        derived = hidden_size // n_heads_legacy
        assert derived == head_size, (
            f"Conflicting head_size={head_size} and legacy n_heads={n_heads_legacy} "
            f"(derived head_size={derived})"
        )

    assert hidden_size % head_size == 0, (
        f"hidden_size {hidden_size} not divisible by head_size {head_size}"
    )
    return head_size


class TransformerBlock(nn.Module):
    """One pre-normalized block: self-attention, then a SwiGLU feed-forward, each residual."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        expansion_ratio: float = 8 / 3,
        dropout: float = 0.1,
        rotary: bool = True,
        use_bias: bool = False,
        attention_backend: str = "flex",
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        # hidden_size: d; n_heads: h; per-head width is d_h = d / h
        self.attn_norm = LayerNorm(hidden_size, bias=use_bias)
        self.attn = MultiHeadAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            rotary=rotary,
            attention_backend=attention_backend,
            use_bias=use_bias,
            max_seq_len=max_seq_len,
        )
        self.ffn = swiglu_ln_ffn(hidden_size, expansion_ratio, dropout, use_bias)

    def set_attention_backend(self, attention_backend: str) -> None:
        """Select the attention implementation this block runs."""
        self.attn.set_attention_backend(attention_backend)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: BlockMask | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Transform residue states; optionally return the attention weights and s_max bounds."""
        # hidden_states: (b, l, d); attention_mask_2d: (b, l) or None
        # attention_mask_4d: (b, 1, 1, l), (b, 1, l, l), or None
        residual = hidden_states  # (b, l, d)
        attn_output, attention_weights, s_max = self.attn(
            hidden_states=self.attn_norm(hidden_states),  # (b, l, d)
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )  # (b, l, d), optional (b, h, l, l), optional h scalars
        hidden_states = residual + attn_output  # (b, l, d)
        hidden_states = hidden_states + self.ffn(hidden_states)  # (b, l, d)
        return hidden_states, attention_weights, s_max  # (b, l, d), optional (b, h, l, l), optional h scalars


@dataclass
class TransformerOutput(ModelOutput):
    """The last residue states of a transformer stack and, when requested, each layer's."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None  # (b, l, d)
    hidden_states: tuple[torch.Tensor, ...] | None = None  # each: (b, l, d)
    attentions: tuple[torch.Tensor | None, ...] | None = None  # each: (b, h, l, l)
    s_max: tuple[list[torch.Tensor] | None, ...] | None = None  # each: h scalar tensors


class Transformer(nn.Module):
    """A stack of pre-normalized transformer blocks sharing one attention backend."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        n_layers: int,
        expansion_ratio: float = 8 / 3,
        dropout: float = 0.1,
        rotary: bool = True,
        use_bias: bool = False,
        attention_backend: str = "flex",
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        # hidden_size: d; n_heads: h; n_layers: r
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    n_heads=n_heads,
                    expansion_ratio=expansion_ratio,
                    dropout=dropout,
                    rotary=rotary,
                    use_bias=use_bias,
                    attention_backend=attention_backend,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )
        self.attention_backend = resolve_attention_backend(attention_backend)

    def set_attention_backend(self, attention_backend: str) -> None:
        """Select one attention implementation for the stack and each of its layers."""
        self.attention_backend = resolve_attention_backend(attention_backend)
        for layer in self.layers:
            layer.set_attention_backend(attention_backend)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_mask_2d: torch.Tensor | None = None,
        attention_mask_4d: torch.Tensor | None = None,
        flex_block_mask: BlockMask | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
    ) -> TransformerOutput:
        """Run every layer over residue states, masking padding with whichever mask is given."""
        # hidden_states: (b, l, d); attention_mask: (b, l), a 4D mask, or None
        # attention_mask_2d: (b, l) or None; attention_mask_4d: (b, 1, 1, l), (b, 1, l, l), or None
        batch_size, seq_len, _ = hidden_states.shape  # b, l, d
        attention_mask_2d, attention_mask_4d, flex_block_mask = build_attention_masks(
            attention_backend=self.attention_backend,
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
            attention_mask=attention_mask,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
        )  # optional (b, l), optional (b, 1, 1, l) or (b, 1, l, l), optional block mask

        hidden_state_history = () if output_hidden_states else None  # optional tuple[(b, l, d), ...]
        attention_history = () if output_attentions else None  # optional tuple[(b, h, l, l), ...]
        s_max_history = () if output_s_max else None  # optional tuple[h scalar tensors, ...]

        for layer in self.layers:
            hidden_states, attention_weights, s_max = layer(
                hidden_states=hidden_states,
                attention_mask_2d=attention_mask_2d,
                attention_mask_4d=attention_mask_4d,
                flex_block_mask=flex_block_mask,
                output_attentions=output_attentions,
                output_s_max=output_s_max,
            )  # (b, l, d), optional (b, h, l, l), optional h scalars
            if output_hidden_states:
                hidden_state_history += (hidden_states,)  # tuple[(b, l, d), ...]

            if output_attentions:
                attention_history += (attention_weights,)  # tuple[(b, h, l, l), ...]

            if output_s_max:
                s_max_history += (s_max,)  # tuple[h scalar tensors, ...]

        return TransformerOutput(
            last_hidden_state=hidden_states,
            hidden_states=hidden_state_history,
            attentions=attention_history,
            s_max=s_max_history,
        )  # last_hidden_state: (b, l, d)
