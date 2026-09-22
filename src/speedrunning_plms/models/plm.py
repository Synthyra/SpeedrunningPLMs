"""Protein MLM architectures and Hugging Face serialization.

Shape notation: b=batch size, l=sequence length, d=hidden width,
h=head count, c=vocabulary size, n_docs=document count. A leading ellipsis
means either legacy (l,) or batched (b, l) token dimensions. BlockMask
comments describe token-level coverage, not its rounded block storage.
"""

import math
import torch
import torch.nn as nn

from copy import copy
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from transformers import EsmTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput

from .attention import SelfAttention
from .layers import BottleneckMLP, Linear, MLP, correction_fn, norm


REMOTE_CODE_AUTO_MAP = {
    "AutoConfig": "plm.PLMConfig",
    "AutoModelForMaskedLM": "plm.PLM",
}


@dataclass
class PLMConfig(PretrainedConfig):
    model_type = "speedrunning_plm"

    def __init__(
        self,
        hidden_size: int = 512,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 12,
        num_unet_layers: int = 0,
        num_extra_layers: int = 0,
        max_sequence_length: int = 1024,
        vocab_size: int = 33,
        expansion_ratio: float = 2.0,
        soft_logit_cap: float = 16.0,
        sliding_window_size: int = 2048,
        tie_embeddings: Optional[bool] = None,
        unet: bool = False,
        patch_unet: bool = False,
        mlm: bool = False,
        masked_diffusion: bool = False,
        token_dropout: bool = True,
        compile_flex_attention: bool = True,
        tokenizer_name: Optional[str] = "facebook/esm2_t6_8M_UR50D",
        cls_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        standard_tie_embeddings = kwargs.pop("tie_word_embeddings", None)
        if tie_embeddings is None:
            tie_embeddings = (
                bool(standard_tie_embeddings)
                if standard_tie_embeddings is not None
                else False
            )
        super().__init__(tie_word_embeddings=bool(tie_embeddings), **kwargs)
        self.hidden_size = hidden_size  # d
        self.num_attention_heads = num_attention_heads  # h
        self.num_hidden_layers = num_hidden_layers
        self.num_unet_layers = num_unet_layers
        self.num_extra_layers = num_extra_layers
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size  # c
        self.expansion_ratio = expansion_ratio
        self.soft_logit_cap = soft_logit_cap
        self.sliding_window_size = sliding_window_size
        self.tie_embeddings = bool(tie_embeddings)
        self.unet = unet
        self.patch_unet = patch_unet
        self.mlm = mlm
        self.masked_diffusion = masked_diffusion
        self.token_dropout = token_dropout
        self.compile_flex_attention = compile_flex_attention
        self.tokenizer_name = tokenizer_name
        self.cls_token_id = cls_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        # Keep the checkpoint self-contained for AutoClass loading with
        # trust_remote_code=True. Transformers expects module.Class, not
        # repo--Class, for code stored in the same model repository.
        existing_auto_map = dict(getattr(self, "auto_map", {}))
        existing_auto_map.pop("AutoModel", None)
        self.auto_map = {**existing_auto_map, **REMOTE_CODE_AUTO_MAP}


# Backwards-compatible public alias. PLM.forward now returns the standard
# Transformers masked-language-model output type.
ESMOutput = MaskedLMOutput


def get_hidden_sizes(hidden_size: int, num_encoder_layers: int, num_attention_heads: int = 1, max_head_dim: int = 128) -> list[int]:
    """Scale encoder widths, aligned to 64 and the head count.

    Cap each width at num_attention_heads * max_head_dim.
    """
    alignment = (64 * num_attention_heads) // gcd(64, num_attention_heads)
    max_hidden = num_attention_heads * max_head_dim
    max_hidden = (max_hidden // alignment) * alignment

    sizes = []
    for i in range(num_encoder_layers):
        # Linear interpolation from 1.0 to 2.0
        scale = 1.0 + (i / max(num_encoder_layers - 1, 1))
        raw_size = hidden_size * scale
        rounded = int(((raw_size + alignment - 1) // alignment) * alignment)
        rounded = min(rounded, max_hidden)
        sizes.append(rounded)
    return sizes


class PatchMerge(nn.Module):
    """Project adjacent token pairs from (b, l, d_in) to (b, l // 2, d_out)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.projection = Linear(2 * in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, l, d_in); projection output width is d_out.
        batch_size, seq_len, hidden_size = x.shape
        assert seq_len % 2 == 0, f"Sequence length {seq_len} must be even for PatchMerge"
        x = x.view(batch_size, seq_len // 2, 2 * hidden_size)  # (b, l // 2, 2 * d_in)
        return self.projection(x)  # (b, l // 2, d_out)


class PatchExpand(nn.Module):
    """Project (b, l_half, d_in) to (b, 2 * l_half, d_out)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.projection = Linear(in_dim, 2 * out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, l_half, d_in); output length is 2 * l_half.
        batch_size, half_length, hidden_size = x.shape
        x = self.projection(x)  # (b, l_half, 2 * d_out)
        return x.view(batch_size, half_length * 2, self.out_dim)  # (b, 2 * l_half, d_out)


class ValueEmbedding(nn.Module):
    def __init__(self, config: PLMConfig) -> None:
        super().__init__()
        self.embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.hidden_size)
            for _ in range(config.num_hidden_layers // 2)
        ])

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        # inputs: (l,) or (b, l); each embedding appends hidden width d.
        ve = [emb(inputs) for emb in self.embed]  # List of (..., d) tensors; mirrored for decoder layers.
        ve += reversed(ve)  # List of (..., d) tensors; mirrored for decoder layers.
        return ve  # List of (..., d) tensors.


class LMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, soft_logit_cap: float = 30.0) -> None:
        super().__init__()
        self.dense = Linear(hidden_size, hidden_size)
        self.decoder = Linear(hidden_size, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))  # (c,)
        self.soft_logit_cap = soft_logit_cap
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d); c is the vocabulary size.
        x = self.dense(norm(x))  # (..., d)
        x = self.act(x)  # (..., d)
        x = self.decoder(x) + self.bias  # (..., c)
        return self.soft_logit_cap * torch.tanh(x / self.soft_logit_cap)  # (..., c)


class TransformerBlock(nn.Module):
    def __init__(self, config: PLMConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)
        self.unet = config.unet
        if config.unet:
            self.lambdas = nn.Parameter(torch.tensor([1., 0.]))  # (2,)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[BlockMask] = None,
            vi: Optional[torch.Tensor] = None,
            x0: Optional[torch.Tensor] = None,
            last_eos: Optional[int] = None,
            **kwargs: Any,
        ) -> torch.Tensor:
        # x, vi, x0: (..., d); attention_mask covers (b, h, l, l).
        if self.unet:
            x = self.lambdas[0] * x + self.lambdas[1] * x0  # (..., d)
            x = x + self.attn(
                x=norm(x),
                attention_mask=attention_mask,
                vi=vi,
                last_eos=last_eos,
                **kwargs,
            )  # (..., d)
        else:
            x = x + self.attn(
                x=norm(x),
                attention_mask=attention_mask,
                last_eos=last_eos,
                **kwargs,
            )  # (..., d)
        x = x + self.mlp(norm(x))  # (..., d)
        return x  # (..., d)


class Transformer(nn.Module):
    def __init__(self, config: PLMConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[BlockMask] = None,
            **kwargs: Any,
        ) -> torch.Tensor:
        # x: (..., d); attention_mask covers (b, h, l, l).
        for layer in self.layers:
            x = layer(
                x=x,
                attention_mask=attention_mask,
                **kwargs,
            )  # (..., d)
        return x  # (..., d)


class UnetTransformer(nn.Module):
    def __init__(self, config: PLMConfig) -> None:
        super().__init__()
        assert config.num_hidden_layers % 2 == 0
        self.num_encoder_layers = config.num_hidden_layers // 2
        self.num_decoder_layers = config.num_hidden_layers // 2  # n_decoder_layers

        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))  # (n_decoder_layers,)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            x: torch.Tensor,
            ve: list[torch.Tensor],
            attention_mask: Optional[BlockMask] = None,
            **kwargs: Any,
        ) -> torch.Tensor:
        # x and each ve entry: (..., d); attention_mask covers (b, h, l, l).
        x0 = x  # (..., d)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]  # Each entry: (..., d).
        skip_connections: list[torch.Tensor] = []  # One hidden-state tensor per encoder layer.
        for i in range(self.num_encoder_layers):
            x = self.layers[i](
                x=x,
                attention_mask=attention_mask,
                vi=ve_enc[i],
                x0=x0,
                **kwargs,
            )  # (..., d)
            skip_connections.append(x)  # (..., d)

        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()  # (..., d)
            x = self.layers[self.num_encoder_layers + i](
                x=x,
                attention_mask=attention_mask,
                vi=ve_dec[i],
                x0=x0,
                **kwargs,
            )  # (..., d)
        return x  # (..., d)


class BatchedTransformerBlock(nn.Module):
    """Mix input and value embeddings at one UNet resolution."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        expansion_ratio: float,
        base_hidden_size: Optional[int] = None,
        compile_flex_attention: bool = True,
    ) -> None:
        super().__init__()
        config = SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            unet=True,
            compile_flex_attention=compile_flex_attention,
        )
        self.attn = SelfAttention(config)

        corrected_dim = correction_fn(expansion_ratio, hidden_size)  # d_mlp
        self.mlp_up = Linear(hidden_size, corrected_dim)
        self.mlp_down = Linear(corrected_dim, hidden_size)
        self.mlp_down.weight.data.zero_()  # (d, d_mlp); initialize the residual projection to zero.
        self.mlp_relu = nn.ReLU()

        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))  # (2,)

        if base_hidden_size is not None and base_hidden_size != hidden_size:
            self.x0_projection = Linear(base_hidden_size, hidden_size)
        else:
            self.x0_projection = None

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[BlockMask] = None,
            vi: Optional[torch.Tensor] = None,
            x0: Optional[torch.Tensor] = None,
            **kwargs: Any,
        ) -> torch.Tensor:
        # x, vi: (b, l, d); x0: (b, l, d_base) before projection.
        if x0 is not None:
            if self.x0_projection is not None:
                x0 = self.x0_projection(x0)  # (b, l, d)
            x = self.lambdas[0] * x + self.lambdas[1] * x0  # (b, l, d)

        x = x + self.attn(x=norm(x), attention_mask=attention_mask, vi=vi, **kwargs)  # (b, l, d)
        mlp_out = self.mlp_down(self.mlp_relu(self.mlp_up(norm(x))).square())  # (b, l, d)
        x = x + mlp_out  # (b, l, d)
        return x  # (b, l, d)


class BatchedValueEmbedding(nn.Module):
    """Embed each path at full resolution using its layer-specific widths."""

    def __init__(self, vocab_size: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        num_encoder_layers = len(hidden_sizes)
        self.encoder_embed = nn.ModuleList([
            nn.Embedding(vocab_size, hidden_sizes[i])
            for i in range(num_encoder_layers)
        ])
        self.decoder_embed = nn.ModuleList([
            nn.Embedding(vocab_size, hidden_sizes[num_encoder_layers - 1 - i])
            for i in range(num_encoder_layers)
        ])

    def forward(self, input_ids: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return encoder and decoder value embeddings in layer order."""
        # input_ids: (b, l); encoder/decoder entries have their own width d_i.
        encoder_ve = [emb(input_ids) for emb in self.encoder_embed]  # Entry i: (b, l, d_i) in this path's layer order.
        decoder_ve = [emb(input_ids) for emb in self.decoder_embed]  # Entry i: (b, l, d_i) in this path's layer order.
        return encoder_ve, decoder_ve  # Two lists of (b, l, d_i) tensors.


@torch.compiler.disable
def precompute_multiresolution_masks(
    input_ids: torch.Tensor,
    cls_token_id: int,
    pad_token_id: int,
    num_levels: int,
    sliding_window_size: int,
    n_heads: int,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
) -> list[Optional[BlockMask]]:
    """Build one attention BlockMask per UNet resolution.

    input_ids and optional attention_mask have shape (b, l). CLS marks document
    starts; nonzero attention_mask entries mark valid tokens. Each level covers
    (b, h, current_length, current_length), with None at sequence length one.
    Build masks eagerly so captured tensors remain available to FlexAttention
    backward outside the compiled model graph.
    """
    batch_size, seq_len = input_ids.shape

    doc_ids = (input_ids == cls_token_id).cumsum(dim=1)  # (b, l)

    if attention_mask is None:
        valid_tokens = input_ids != pad_token_id  # (b, l)
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids; "
                f"got {attention_mask.shape} and {input_ids.shape}."
            )
        valid_tokens = attention_mask.to(device=device, dtype=torch.bool)  # (b, l)

    masks: list[Optional[BlockMask]] = []
    current_doc_ids = doc_ids  # (b, l)
    current_valid_tokens = valid_tokens  # (b, l)
    current_length = seq_len

    for level in range(num_levels):
        if current_length <= 1:
            masks.append(None)
            continue

        # Bind each resolution in a separate closure.
        def make_mask_mod(
            doc_ids_l: torch.Tensor,
            valid_tokens_l: torch.Tensor,
            sw_l: int,
        ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
            # doc_ids_l, valid_tokens_l: (b, current_length).
            def mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
                # Indices and returned masks are scalar tensors () before vmap.
                doc_mask = doc_ids_l[b, q_idx] == doc_ids_l[b, kv_idx]  # ()
                sw_mask = torch.abs(q_idx - kv_idx) < sw_l  # ()
                pad_mask = valid_tokens_l[b, q_idx] & valid_tokens_l[b, kv_idx]  # ()
                return doc_mask & sw_mask & pad_mask  # ()
            return mask_mod

        mask_mod = make_mask_mod(current_doc_ids, current_valid_tokens, sliding_window_size)

        block_mask = create_block_mask(
            mask_mod=mask_mod,
            B=batch_size,
            H=n_heads,
            Q_LEN=current_length,
            KV_LEN=current_length,
            device=device,
        )  # BlockMask covering (b, h, current_length, current_length).
        masks.append(block_mask)

        # A merged token remains valid if either source token is valid.
        if current_length > 1:
            current_doc_ids = current_doc_ids.view(batch_size, current_length // 2, 2).max(dim=-1).values  # (b, current_length // 2)
            current_valid_tokens = current_valid_tokens.view(batch_size, current_length // 2, 2).any(dim=-1)  # (b, current_length // 2)
            current_length = current_length // 2

    return masks


class BatchedUnetTransformer(nn.Module):
    """Batched UNet Transformer with Swin-style patch merging/expanding.

    Operates on (b, l, d) tensors with pre-computed multi-resolution block masks.
    Uses PatchMerge for downsampling and PatchExpand for upsampling.
    Skip connections link encoder and decoder at matching resolutions.

    Architecture:
    - Encoder: TransformerBlock -> PatchMerge -> TransformerBlock -> PatchMerge -> ...
    - BottleneckMLP at vector depth (when seq_len=1)
    - Decoder: PatchExpand -> TransformerBlock + skip -> PatchExpand -> ...
    """
    def __init__(self, config: PLMConfig) -> None:
        super().__init__()
        assert config.num_unet_layers % 2 == 0, "num_unet_layers must be even"
        assert config.max_sequence_length > 0 and (config.max_sequence_length & (config.max_sequence_length - 1)) == 0, \
            f"max_sequence_length must be a power of 2 for PatchMerge, got {config.max_sequence_length}"

        self.num_encoder_layers = config.num_unet_layers // 2
        self.num_decoder_layers = config.num_unet_layers // 2  # n_decoder_layers
        self.base_hidden_size = config.hidden_size  # d_base
        self.max_sequence_length = config.max_sequence_length

        # Vector depth: after this many downsamplings, seq_len=1
        self.vector_depth = int(math.log2(config.max_sequence_length))

        # Hidden sizes for each encoder layer depth
        self.hidden_sizes = get_hidden_sizes(config.hidden_size, self.num_encoder_layers, config.num_attention_heads)

        # Number of resolution levels (for mask pre-computation)
        self.num_resolution_levels = min(self.num_encoder_layers, self.vector_depth + 1)

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(self.num_encoder_layers):
            layer_hidden_size = self.hidden_sizes[min(i, self.vector_depth)]

            if i >= self.vector_depth:
                self.encoder_blocks.append(
                    BottleneckMLP(layer_hidden_size, config.expansion_ratio, self.base_hidden_size)
                )
            else:
                self.encoder_blocks.append(
                    BatchedTransformerBlock(
                        hidden_size=layer_hidden_size,
                        num_attention_heads=config.num_attention_heads,
                        expansion_ratio=config.expansion_ratio,
                        base_hidden_size=self.base_hidden_size,
                        compile_flex_attention=config.compile_flex_attention,
                    )
                )

            # PatchMerge between layers (not after last encoder, not past vector depth)
            if i < self.num_encoder_layers - 1 and i < self.vector_depth:
                next_hidden = self.hidden_sizes[min(i + 1, self.vector_depth)]
                self.downsamples.append(PatchMerge(layer_hidden_size, next_hidden))

        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in range(self.num_decoder_layers):
            enc_idx = self.num_encoder_layers - 1 - i
            effective_depth = enc_idx
            decoder_hidden_size = self.hidden_sizes[min(enc_idx, self.vector_depth)]

            # PatchExpand before each decoder layer (except first/bottleneck)
            prev_depth = self.num_encoder_layers - i
            if i > 0 and prev_depth <= self.vector_depth:
                prev_hidden = self.hidden_sizes[min(prev_depth, self.vector_depth)]
                self.upsamples.append(PatchExpand(prev_hidden, decoder_hidden_size))

            if effective_depth >= self.vector_depth:
                self.decoder_blocks.append(
                    BottleneckMLP(decoder_hidden_size, config.expansion_ratio, self.base_hidden_size)
                )
            else:
                self.decoder_blocks.append(
                    BatchedTransformerBlock(
                        hidden_size=decoder_hidden_size,
                        num_attention_heads=config.num_attention_heads,
                        expansion_ratio=config.expansion_ratio,
                        base_hidden_size=self.base_hidden_size,
                        compile_flex_attention=config.compile_flex_attention,
                    )
                )

        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))  # (n_decoder_layers,)

        # Input/output projections if base hidden size differs from first layer
        if self.hidden_sizes[0] != config.hidden_size:
            self.input_projection = Linear(config.hidden_size, self.hidden_sizes[0])
            self.output_projection = Linear(self.hidden_sizes[0], config.hidden_size)
        else:
            self.input_projection = None
            self.output_projection = None

    def _downsample_to_resolution(self, x: torch.Tensor, target_L: int) -> torch.Tensor:
        """Average-pool pairs to spatially downsample x to target sequence length."""
        # x: (b, l, d); target_L is the requested sequence length.
        batch_size, seq_len, hidden_size = x.shape
        while seq_len > target_L:
            assert seq_len % 2 == 0, f"Cannot halve sequence length {seq_len}"
            x = x.view(batch_size, seq_len // 2, 2, hidden_size).mean(dim=2)  # (b, seq_len // 2, d); seq_len decreases each iteration.
            seq_len = seq_len // 2
        return x  # (b, target_L, d)

    def forward(
            self,
            x: torch.Tensor,
            encoder_ve: list[torch.Tensor],
            decoder_ve: list[torch.Tensor],
            attention_masks: list[Optional[BlockMask]],
            x0_full: torch.Tensor,
            **kwargs: Any,
        ) -> torch.Tensor:
        """
        Forward pass for batched UNet.

        Args:
            x: (b, l, d) input embeddings
            encoder_ve: List of value embeddings at full resolution per encoder layer
            decoder_ve: List of value embeddings at full resolution per decoder layer
            attention_masks: Pre-computed BlockMask per resolution level
            x0_full: (b, l, d_base) original input for lambda mixing
        """
        # x: (b, l, d_base); each value embedding: (b, l, d_i).
        # d_i denotes the hidden width at the current encoder/decoder layer.
        if self.input_projection is not None:
            x = self.input_projection(x)  # (b, l, d_0)

        skip_connections: list[torch.Tensor] = []  # One hidden-state tensor per encoder layer.
        mask_idx = 0
        downsample_idx = 0
        current_length = x.shape[1]

        for i in range(self.num_encoder_layers):
            # Attention mask for this resolution
            attn_mask = attention_masks[mask_idx] if mask_idx < len(attention_masks) else None  # Covers (b, h, current_length, current_length).

            # Downsample value embedding to current resolution
            vi = None
            if i < len(encoder_ve):
                vi = self._downsample_to_resolution(encoder_ve[i], current_length)  # (b, current_length, d_i)

            # Downsample x0 to current resolution (x0 stays at base_hidden_size,
            # each block's x0_projection handles dim change)
            x0_current = self._downsample_to_resolution(x0_full, current_length)  # (b, current_length, d_base)

            x = self.encoder_blocks[i](
                x=x,
                attention_mask=attn_mask,
                vi=vi,
                x0=x0_current,
                **kwargs,
            )  # (b, current_length, d_i) at this layer.
            skip_connections.append(x)  # (b, current_length, d_i)

            if i < self.num_encoder_layers - 1 and i < self.vector_depth:
                x = self.downsamples[downsample_idx](x)  # (b, current_length // 2, d_next)
                downsample_idx += 1
                mask_idx += 1
                current_length = x.shape[1]

        upsample_idx = 0
        for i in range(self.num_decoder_layers):
            skip = skip_connections.pop()  # (b, skip_length, d_skip)

            effective_depth = self.num_encoder_layers - 1 - i
            prev_depth = self.num_encoder_layers - i

            # Upsample x to match skip resolution
            if i > 0 and prev_depth <= self.vector_depth:
                x = self.upsamples[upsample_idx](x)  # (b, 2 * current_length, d_skip)
                upsample_idx += 1
                current_length = x.shape[1]

            x = x + self.skip_weights[i] * skip  # (b, current_length, d_i) at this layer.

            # Attention mask for decoder at this resolution
            dec_mask_idx = min(effective_depth, len(attention_masks) - 1)
            attn_mask = attention_masks[dec_mask_idx] if attention_masks else None  # Covers (b, h, current_length, current_length).

            # Downsample value embedding to current resolution
            vi = None
            if i < len(decoder_ve):
                vi = self._downsample_to_resolution(decoder_ve[i], current_length)  # (b, current_length, d_i)

            # Downsample x0 to current resolution
            x0_current = self._downsample_to_resolution(x0_full, current_length)  # (b, current_length, d_base)

            x = self.decoder_blocks[i](
                x=x,
                attention_mask=attn_mask,
                vi=vi,
                x0=x0_current,
                **kwargs,
            )  # (b, current_length, d_i) at this layer.

        # Project output back to base hidden size if needed
        if self.output_projection is not None:
            x = self.output_projection(x)  # (b, l, d_base)

        return x  # (b, l, d_base)


class PLM(PreTrainedModel):
    config_class = PLMConfig
    _tied_weights_keys = ["lm_head.decoder.weight"]

    def __init__(self, config: PLMConfig) -> None:
        super().__init__(config)
        self.config = config
        explicit_token_ids = (
            config.cls_token_id,
            config.eos_token_id,
            config.pad_token_id,
            config.mask_token_id,
        )
        if all(token_id is not None for token_id in explicit_token_ids):
            self.tokenizer = None
            self.cls_token_id = int(config.cls_token_id)
            self.eos_token_id = int(config.eos_token_id)
            self.pad_token_id = int(config.pad_token_id)
            self.mask_token_id = int(config.mask_token_id)
        else:
            if config.tokenizer_name is None:
                raise ValueError("tokenizer_name is required unless all token IDs are provided in PLMConfig.")
            self.tokenizer = EsmTokenizer.from_pretrained(config.tokenizer_name)
            self.cls_token_id = self.tokenizer.cls_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id
            self.mask_token_id = self.tokenizer.mask_token_id
        # Persist resolved IDs so published checkpoints can reload without
        # fetching an external tokenizer merely to construct the model.
        self.config.cls_token_id = self.cls_token_id
        self.config.eos_token_id = self.eos_token_id
        self.config.pad_token_id = self.pad_token_id
        self.config.mask_token_id = self.mask_token_id
        self.mlm = config.mlm
        self.masked_diffusion = config.masked_diffusion
        self.token_dropout = config.token_dropout

        self.vocab_size = config.vocab_size  # c
        self.n_heads = config.num_attention_heads  # h
        self.sliding_window_size = config.sliding_window_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.unet = config.unet
        self.patch_unet = config.patch_unet

        if config.patch_unet:
            # Batched UNet with Swin-style patch merge/expand
            assert config.num_unet_layers > 0, "num_unet_layers must be > 0 for patch_unet"
            self.transformer = BatchedUnetTransformer(config)
            hidden_sizes = self.transformer.hidden_sizes
            self.value_embeds = BatchedValueEmbedding(config.vocab_size, hidden_sizes)
        elif config.unet:
            # Original UNet (skip connections only, no downsampling)
            self.transformer = UnetTransformer(config)
            self.value_embeds = ValueEmbedding(config)
        else:
            # Standard transformer
            self.transformer = Transformer(config)

        # Extra sequential transformer layers after U-Net (at full resolution)
        self.num_extra_layers = config.num_extra_layers
        if config.num_extra_layers > 0:
            # Create a config for extra layers without unet skip connections
            extra_config = copy(config)
            extra_config.unet = False
            self.extra_layers = nn.ModuleList([
                TransformerBlock(extra_config)
                for _ in range(config.num_extra_layers)
            ])
        else:
            self.extra_layers = None

        self.lm_head = LMHead(config.hidden_size, config.vocab_size, config.soft_logit_cap)
        if config.tie_embeddings:
            self.lm_head.decoder.weight = self.embedding.weight  # (c, d); shared with input embeddings.

        self.ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embedding = value

    def get_output_embeddings(self) -> Linear:
        return self.lm_head.decoder

    def set_output_embeddings(self, value: Linear) -> None:
        self.lm_head.decoder = value

    def _validated_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # input_ids and optional attention_mask: (l,) or (b, l).
        if attention_mask is None:
            return input_ids != self.pad_token_id  # (l,) or (b, l), matching input_ids.
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids; "
                f"got {attention_mask.shape} and {input_ids.shape}."
            )
        return attention_mask.to(device=input_ids.device, dtype=torch.bool)  # (l,) or (b, l), matching input_ids.

    def _get_standard_hidden_state(
        self,
        input_ids: torch.Tensor,
        sliding_window_size: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # input_ids, attention_mask: (l,) or (b, l); internal tensors are batched.
        squeeze_output = input_ids.dim() == 1
        valid_tokens = self._validated_attention_mask(input_ids, attention_mask)  # (l,) or (b, l) before batching.
        if squeeze_output:
            input_ids = input_ids.unsqueeze(0)  # (1, l)
            valid_tokens = valid_tokens.unsqueeze(0)  # (1, l)

        batch_size, seq_len = input_ids.shape
        docs = (input_ids == self.cls_token_id).cumsum(dim=1)  # (b, l)

        def doc_mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
            # Indices and returned masks are scalar tensors () before vmap.
            sliding_mask = torch.abs(q_idx - kv_idx) < sliding_window_size  # ()
            doc_mask = docs[b, q_idx] == docs[b, kv_idx]  # ()
            valid_mask = valid_tokens[b, q_idx] & valid_tokens[b, kv_idx]  # ()
            return sliding_mask & doc_mask & valid_mask  # ()

        block_mask = create_block_mask(
            mask_mod=doc_mask_mod,
            B=batch_size,
            H=self.n_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=input_ids.device,
        )  # BlockMask covering (b, h, l, l).

        x = self.embedding(input_ids)  # (b, l, d)
        if self.token_dropout:
            masked_tokens = (input_ids == self.mask_token_id) & valid_tokens  # (b, l)
            x = x.masked_fill(masked_tokens.unsqueeze(-1), 0.0)  # (b, l, d)
            real_token_count = valid_tokens.sum(dim=1, keepdim=True).float().clamp(min=1)  # (b, 1)
            mask_count = masked_tokens.sum(dim=1, keepdim=True).float()  # (b, 1)
            mask_ratio_observed = mask_count / real_token_count  # (b, 1)
            x = (x * (1 - mask_ratio_observed.unsqueeze(-1))).to(x.dtype)  # (b, l, d)

        x = norm(x)  # (b, l, d)
        if self.unet:
            ve = self.value_embeds(input_ids)  # List of (b, l, d) tensors.
            x = self.transformer(x=x, ve=ve, attention_mask=block_mask)  # (b, l, d)
        else:
            x = self.transformer(x=x, attention_mask=block_mask)  # (b, l, d)

        if self.extra_layers is not None:
            for layer in self.extra_layers:
                x = layer(x=x, attention_mask=block_mask)  # (b, l, d)
        return x.squeeze(0) if squeeze_output else x  # (l, d) or (b, l, d), matching input rank.

    def get_last_hidden_state(
        self,
        input_ids: torch.Tensor,
        sliding_window_size: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return hidden states for legacy 1D or standard batched token input."""
        # input_ids, attention_mask: (l,) or (b, l); patch UNet requires (b, l).
        if input_ids.dim() not in (1, 2):
            raise ValueError(
                "input_ids must have shape (sequence_length,) or "
                f"(batch_size, sequence_length); got {input_ids.shape}."
            )

        if self.patch_unet:
            if input_ids.dim() != 2:
                raise ValueError(
                    f"patch_unet expects batched (B, L) input, got {input_ids.shape}."
                )
            valid_tokens = self._validated_attention_mask(input_ids, attention_mask)  # (b, l)

            attention_masks = precompute_multiresolution_masks(
                input_ids=input_ids,
                cls_token_id=self.cls_token_id,
                pad_token_id=self.pad_token_id,
                num_levels=self.transformer.num_resolution_levels,
                sliding_window_size=sliding_window_size,
                n_heads=self.n_heads,
                device=input_ids.device,
                attention_mask=valid_tokens,
            )
            full_res_mask = attention_masks[0]  # Optional BlockMask covering (b, h, l, l).
            x = self.embedding(input_ids)  # (b, l, d)

            if self.token_dropout:
                masked_tokens = (input_ids == self.mask_token_id) & valid_tokens  # (b, l)
                x = x.masked_fill(masked_tokens.unsqueeze(-1), 0.0)  # (b, l, d)
                real_token_count = valid_tokens.sum(dim=1, keepdim=True).float().clamp(min=1)  # (b, 1)
                mask_count = masked_tokens.sum(dim=1, keepdim=True).float()  # (b, 1)
                mask_ratio_observed = mask_count / real_token_count  # (b, 1)
                x = (x * (1 - mask_ratio_observed.unsqueeze(-1))).to(x.dtype)  # (b, l, d)

            x = norm(x)  # (b, l, d)
            encoder_ve, decoder_ve = self.value_embeds(input_ids)  # Each path entry i: (b, l, d_i).
            x = self.transformer(
                x=x,
                encoder_ve=encoder_ve,
                decoder_ve=decoder_ve,
                attention_masks=attention_masks,
                x0_full=x.clone(),
            )  # (b, l, d)

            if self.extra_layers is not None:
                for layer in self.extra_layers:
                    x = layer(x=x, attention_mask=full_res_mask)  # (b, l, d)
            return x  # (b, l, d)

        return self._get_standard_hidden_state(
            input_ids,
            sliding_window_size,
            attention_mask,
        )  # (l, d) or (b, l, d), matching input rank.

    def get_vector_embeddings(self, input_ids: torch.Tensor, sliding_window_size: Optional[int] = None) -> torch.Tensor:
        """Pool each CLS-delimited document into one embedding.

        input_ids: (l,) or (b, l). Returns (n_docs, d).
        Batched pooling excludes padding; legacy 1D pooling includes it.
        """
        if sliding_window_size is None:
            sliding_window_size = self.sliding_window_size
        x = self.get_last_hidden_state(input_ids, sliding_window_size)  # (l, d) or (b, l, d)

        if input_ids.dim() == 2:
            # Batched: x is (b, l, d), input_ids is (b, l)
            batch_size, seq_len, hidden_size = x.shape
            doc_ids = (input_ids == self.cls_token_id).cumsum(dim=1)  # (b, l)
            # Flatten batch into single sequence for mean pooling
            x_flat = x.reshape(-1, hidden_size)  # (b * l, d)
            # Offset doc_ids per batch element so each batch has unique doc IDs
            max_docs_per_batch = doc_ids.max(dim=1).values  # (b,)
            offsets = torch.zeros(batch_size, dtype=doc_ids.dtype, device=doc_ids.device)  # (b,)
            offsets[1:] = max_docs_per_batch[:-1].cumsum(0)  # (b - 1,)
            doc_ids = doc_ids + offsets.unsqueeze(1)  # (b, l)
            doc_ids_flat = doc_ids.reshape(-1)  # (b * l,)
            pad_mask = (input_ids.reshape(-1) != self.pad_token_id)  # (b * l,)
            num_docs = doc_ids_flat.max().item()
            doc_ids_0based = doc_ids_flat - 1  # (b * l,)
            doc_embeds: list[torch.Tensor] = []  # Each pooled tensor: (d,).
            for doc_idx in range(num_docs):
                mask = (doc_ids_0based == doc_idx) & pad_mask  # (b * l,)
                if mask.any():
                    doc_embeds.append(x_flat[mask].mean(dim=0))  # Append a (d,) mean over the selected document tokens.
            return torch.stack(doc_embeds, dim=0)  # (n_docs, d)
        else:
            # Legacy 1D path
            docs = (input_ids == self.cls_token_id).cumsum(0)  # (l,)
            x = x.view(-1, self.config.hidden_size)  # (l, d)
            num_docs = docs.max().item()
            doc_ids = docs - 1  # (l,)
            doc_embeds: list[torch.Tensor] = []  # Each pooled tensor: (d,).
            for doc_idx in range(num_docs):
                mask = (doc_ids == doc_idx)  # (l,)
                doc_embeds.append(x[mask].mean(dim=0))  # Append a (d,) mean over the selected document tokens.
            return torch.stack(doc_embeds, dim=0)  # (n_docs, d)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mask_rate: Optional[torch.Tensor | float] = None,
        sliding_window_size: Optional[int] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> MaskedLMOutput | tuple[torch.Tensor | tuple[torch.Tensor, ...], ...]:
        """Run masked-language-model inference or training.

        The public contract follows ``AutoModelForMaskedLM``: batched
        ``input_ids`` and ``attention_mask`` are accepted, ``labels`` are
        optional, and outputs expose ``loss`` and ``logits`` through a standard
        ``MaskedLMOutput``. One-dimensional packed input remains supported for
        the repository's legacy training pipeline.
        """
        # input_ids, attention_mask, labels: (l,) or (b, l); mask_rate: scalar or reduced to one.
        if sliding_window_size is None:
            sliding_window_size = self.sliding_window_size
        if return_dict is None:
            return_dict = self.config.use_return_dict
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        last_hidden_state = self.get_last_hidden_state(
            input_ids,
            sliding_window_size,
            attention_mask=attention_mask,
        )  # (..., d)
        lm_logits = self.lm_head(norm(last_hidden_state))  # (..., c)

        loss = None  # () when labels are present; otherwise None.
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    "labels must have the same shape as input_ids; "
                    f"got {labels.shape} and {input_ids.shape}."
                )
            loss = self.ce(
                lm_logits.reshape(-1, self.vocab_size),
                labels.reshape(-1).long(),
            )  # ()
            if self.training and self.masked_diffusion and not self.mlm:
                if mask_rate is None:
                    valid_tokens = self._validated_attention_mask(input_ids, attention_mask)  # (l,) or (b, l)
                    predicted_tokens = (labels != -100) & valid_tokens  # (l,) or (b, l)
                    mask_rate = (
                        predicted_tokens.sum().float()
                        / valid_tokens.sum().float().clamp(min=1)
                    )  # ()
                rate = torch.as_tensor(
                    mask_rate,
                    device=loss.device,
                    dtype=loss.dtype,
                ).mean().clamp(min=torch.finfo(loss.dtype).eps)  # ()
                loss = loss / rate  # ()

        hidden_states = (last_hidden_state,) if output_hidden_states else None  # One (..., d) tensor when requested; otherwise None.
        if not return_dict:
            output = (lm_logits,)  # Tuple beginning with (..., c) logits, then optional hidden states.
            if hidden_states is not None:
                output += (hidden_states,)  # Tuple beginning with (..., c) logits, then optional hidden states.
            return ((loss,) + output) if loss is not None else output  # Optional () loss, (..., c) logits, optional hidden states.

        return MaskedLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=hidden_states,
        )  # loss: (); logits: (..., c); optional hidden states: ((..., d),).

    @torch.no_grad()
    def get_logits(
        self,
        input_ids: torch.Tensor,
        sliding_window_size: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return logits with the input token dimensions followed by vocabulary width."""
        # input_ids, attention_mask: (l,) or (b, l); logits append width c.
        if sliding_window_size is None:
            sliding_window_size = self.sliding_window_size
        hidden = self.get_last_hidden_state(
            input_ids,
            sliding_window_size,
            attention_mask=attention_mask,
        )  # (l, d) or (b, l, d)
        return self.lm_head(norm(hidden))  # (l, c) or (b, l, c)

    @torch.no_grad()
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        sliding_window_size: Optional[int] = None,
        pooling: str = 'mean',
    ) -> torch.Tensor:
        """Return CLS embeddings or mean-pooled embeddings.

        Patch UNet pools each batch row, excluding padding. Other architectures
        pool each CLS-delimited document; legacy 1D mean pooling includes padding.
        """
        # input_ids: (l,) or (b, l); hidden states append width d.
        if sliding_window_size is None:
            sliding_window_size = self.sliding_window_size
        hidden = self.get_last_hidden_state(input_ids, sliding_window_size)  # (l, d) or (b, l, d)

        if self.patch_unet:
            # Batched: hidden is (b, l, d), input_ids is (b, l)
            assert input_ids.dim() == 2
            batch_size, seq_len, hidden_size = hidden.shape
            if pooling == 'cls':
                # CLS is the first token of each chunk
                return hidden[:, 0, :]  # (b, d)
            else:
                # Mean pool over non-pad tokens per batch element
                mask = (input_ids != self.pad_token_id).unsqueeze(-1).float()  # (b, l, 1)
                return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (b, d)
        else:
            # Standard and skip-only UNet support either input rank.
            if pooling == 'cls':
                # Return embedding at each CLS position
                cls_mask = (input_ids == self.cls_token_id)  # (l,) or (b, l)
                return hidden[cls_mask]  # (n_docs, d)
            else:
                return self.get_vector_embeddings(input_ids, sliding_window_size)  # (n_docs, d)

    def save_weights_local(self, save_dir: str, step: int) -> None:
        """Save model weights and configuration in a step-specific directory."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.save_pretrained(save_path / f"step_{step:06d}")


# Tell Transformers to copy these source files and emit canonical AutoClass
# mappings whenever config/model artifacts are saved for local or Hub use.
PLMConfig.register_for_auto_class()
PLM.register_for_auto_class("AutoModelForMaskedLM")


if __name__ == "__main__":
    # py -m model.model
    import io
    import sys


    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 80)
    print("Testing Original UNet Transformer")
    print("=" * 80)
    config = PLMConfig(
        hidden_size=768,
        num_attention_heads=6,
        num_hidden_layers=24,
        expansion_ratio=8/3,
        unet=True,
        max_sequence_length=1024,
    )
    model = PLM(config).cuda()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create test input with proper structure (CLS + sequence + EOS) - 1D for legacy path
    seq_len = 128
    input_ids = torch.randint(4, 33, (seq_len,)).cuda()  # (l,)
    input_ids[0] = 0  # () element of (l,) input.
    input_ids[-1] = 2  # () element of (l,) input.
    labels = input_ids.clone()  # (l,)
    labels[labels != 32] = -100  # Selected entries of (l,) labels.
    mask_rate = torch.tensor(0.15).cuda()  # ()

    loss = model(input_ids=input_ids, labels=labels, mask_rate=mask_rate).loss  # ()
    print(f"Original UNet loss: {loss.item():.4f}")

    print("\n" + "=" * 80)
    print("Testing Batched UNet Transformer (patch_unet)")
    print("=" * 80)
    max_length = 128  # Power of 2 for patch merging
    patch_config = PLMConfig(
        hidden_size=384,
        num_attention_heads=6,
        num_unet_layers=8,  # 4 encoder + 4 decoder
        num_extra_layers=2,
        max_sequence_length=max_length,
        expansion_ratio=8/3,
        patch_unet=True,
    )
    patch_model = PLM(patch_config).cuda()
    print(f"Model parameters: {sum(p.numel() for p in patch_model.parameters()):,}")

    # Create batched test input (batch_size, max_length) with packed documents per element
    batch_size = 4
    batched_ids = torch.randint(4, 33, (batch_size, max_length)).cuda()  # (b, max_length)
    for b in range(batch_size):
        # Insert CLS at start and EOS at end of each chunk
        batched_ids[b, 0] = 0  # () element of (b, max_length) input.
        batched_ids[b, max_length - 1] = 2  # () element of (b, max_length) input.
        # Add a second document boundary in the middle
        mid = max_length // 2
        batched_ids[b, mid - 1] = 2  # () element of (b, max_length) input.
        batched_ids[b, mid] = 0  # () element of (b, max_length) input.
    batched_labels = batched_ids.clone()  # (b, max_length)
    batched_labels[batched_labels != 32] = -100  # Selected entries of (b, max_length) labels.

    loss = patch_model(
        input_ids=batched_ids,
        labels=batched_labels,
        mask_rate=mask_rate,
    ).loss  # ()
    print(f"Batched UNet loss: {loss.item():.4f}")

    print(f"\nHidden sizes: {patch_model.transformer.hidden_sizes}")
    print(f"Vector depth (log2(max_length)): {patch_model.transformer.vector_depth}")
    print(f"Num encoder layers: {patch_model.transformer.num_encoder_layers}")
    print(f"Num decoder layers: {patch_model.transformer.num_decoder_layers}")

    print("\n" + "=" * 80)
    print("Testing Batched UNet with deep layers (MLP at vector depth)")
    print("=" * 80)
    deep_config = PLMConfig(
        hidden_size=384,
        num_attention_heads=6,
        num_unet_layers=20,  # 10 encoder + 10 decoder (some will be MLPs)
        num_extra_layers=1,
        max_sequence_length=128,  # log2(128)=7, so layers 7+ become MLPs
        expansion_ratio=8/3,
        patch_unet=True,
    )
    deep_model = PLM(deep_config).cuda()

    # Count transformer vs MLP blocks
    n_transformer = sum(1 for b in deep_model.transformer.encoder_blocks if isinstance(b, BatchedTransformerBlock))
    n_mlp = sum(1 for b in deep_model.transformer.encoder_blocks if isinstance(b, BottleneckMLP))
    print(f"Encoder: {n_transformer} transformer blocks, {n_mlp} MLP blocks")

    n_transformer_dec = sum(1 for b in deep_model.transformer.decoder_blocks if isinstance(b, BatchedTransformerBlock))
    n_mlp_dec = sum(1 for b in deep_model.transformer.decoder_blocks if isinstance(b, BottleneckMLP))
    print(f"Decoder: {n_transformer_dec} transformer blocks, {n_mlp_dec} MLP blocks")

    loss = deep_model(
        input_ids=batched_ids,
        labels=batched_labels,
        mask_rate=mask_rate,
    ).loss  # ()
    print(f"Deep Batched UNet loss: {loss.item():.4f}")

    print("\n" + "=" * 80)
    print("Testing Multi-Resolution Mask Pre-computation")
    print("=" * 80)

    # Verify mask shapes at each resolution level
    masks = precompute_multiresolution_masks(
        input_ids=batched_ids,
        cls_token_id=0,
        pad_token_id=1,
        num_levels=patch_model.transformer.num_resolution_levels,
        sliding_window_size=128,
        n_heads=6,
        device=batched_ids.device,
    )
    for i, m in enumerate(masks):
        if m is not None:
            print(f"Level {i}: mask shape Q_LEN={m.shape[-2]}, KV_LEN={m.shape[-1]}")
        else:
            print(f"Level {i}: None (vector depth)")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
