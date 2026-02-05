import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass
from torch.nn.attention.flex_attention import create_block_mask
from transformers import EsmTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from model.attention import SelfAttention
from model.utils import norm, MLP, Linear, BottleneckMLP


@dataclass
class PLMConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int = 512,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 12,
        num_unet_layers: int = 0,
        num_extra_layers: int = 0,
        max_length: int = 1024,
        vocab_size: int = 33,
        expansion_ratio: float = 2.0,
        soft_logit_cap: float = 16.0,
        sliding_window_size: int = 2048,
        tie_embeddings: bool = False,
        unet: bool = False,
        conv_unet: bool = False,
        mlm: bool = False,
        masked_diffusion: bool = False,
        token_dropout: bool = True,
        compile_flex_attention: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_unet_layers = num_unet_layers
        self.num_extra_layers = num_extra_layers
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.expansion_ratio = expansion_ratio
        self.soft_logit_cap = soft_logit_cap
        self.sliding_window_size = sliding_window_size
        self.tie_embeddings = tie_embeddings
        self.unet = unet
        self.conv_unet = conv_unet
        self.mlm = mlm
        self.masked_diffusion = masked_diffusion
        self.token_dropout = token_dropout
        self.compile_flex_attention = compile_flex_attention
        # HuggingFace AutoModel mapping for trust_remote_code
        self.auto_map = {
            "AutoModel": "model--PLM",
            "AutoModelForMaskedLM": "model--PLM",
        }


@dataclass
class ESMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None


def get_hidden_sizes(hidden_size: int, num_encoder_layers: int, num_attention_heads: int = 1, max_head_dim: int = 128) -> List[int]:
    """Returns hidden size for each encoder layer, rounded to multiples of 64 and num_attention_heads.
    Scales from hidden_size toward hidden_size * 2 at the bottleneck, capped so that
    head_dim (hidden / num_heads) never exceeds max_head_dim.

    This cap prevents Triton shared memory overflow in flex_attention kernels.
    For more hidden dimension growth, increase num_attention_heads (Swin Transformer style).

    Args:
        hidden_size: Base hidden size
        num_encoder_layers: Number of encoder layers
        num_attention_heads: Number of attention heads (hidden size must be divisible by this)
        max_head_dim: Maximum per-head dimension (default 128, safe for Triton SRAM)
    """
    from math import gcd
    # Find LCM of 64 and num_attention_heads for GPU efficiency and head divisibility
    alignment = (64 * num_attention_heads) // gcd(64, num_attention_heads)
    # Maximum hidden size enforced by head_dim constraint
    max_hidden = num_attention_heads * max_head_dim
    # Round max_hidden down to alignment
    max_hidden = (max_hidden // alignment) * alignment

    sizes = []
    for i in range(num_encoder_layers):
        # Linear interpolation from 1.0 to 2.0
        scale = 1.0 + (i / max(num_encoder_layers - 1, 1))
        raw_size = hidden_size * scale
        # Round up to nearest alignment
        rounded = int(((raw_size + alignment - 1) // alignment) * alignment)
        # Clamp to max_hidden to prevent head_dim overflow
        rounded = min(rounded, max_hidden)
        sizes.append(rounded)
    return sizes


class PatchMerge(nn.Module):
    """Downsample sequence by 2x via Swin-style patch merging.
    Concatenates adjacent token pairs and projects to new dimension.
    (B, L, D_in) -> (B, L//2, D_out)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.projection = Linear(2 * in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        assert L % 2 == 0, f"Sequence length {L} must be even for PatchMerge"
        x = x.view(B, L // 2, 2 * D)
        return self.projection(x)


class PatchExpand(nn.Module):
    """Upsample sequence by 2x via linear projection and reshape.
    (B, L//2, D_in) -> (B, L, D_out)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.projection = Linear(in_dim, 2 * out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L_half, D = x.shape
        x = self.projection(x)  # (B, L_half, 2 * out_dim)
        return x.view(B, L_half * 2, self.out_dim)


class ValueEmbedding(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.hidden_size)
            for _ in range(config.num_hidden_layers // 2)
        ])

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        ve = [emb(inputs) for emb in self.embed]
        ve += reversed(ve)
        return ve


class LMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, soft_logit_cap: float = 30.0):
        super().__init__()
        self.dense = Linear(hidden_size, hidden_size)
        self.decoder = Linear(hidden_size, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.soft_logit_cap = soft_logit_cap
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(norm(x))
        x = self.act(x)
        x = self.decoder(x) + self.bias
        return self.soft_logit_cap * torch.tanh(x / self.soft_logit_cap)


class TransformerBlock(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.config = config
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)
        self.unet = config.unet
        if config.unet:
            self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
    
    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            vi: Optional[torch.Tensor] = None,
            x0: Optional[torch.Tensor] = None,
            last_eos: Optional[int] = None,
            **kwargs,
        ) -> torch.Tensor:
        if self.unet:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
            x = x + self.attn(
                x=norm(x),
                attention_mask=attention_mask,
                vi=vi,
                last_eos=last_eos,
                **kwargs,
            )
        else:
            x = x + self.attn(
                x=norm(x),
                attention_mask=attention_mask,
                last_eos=last_eos,
                **kwargs,
            )
        x = x + self.mlp(norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x=x,
                attention_mask=attention_mask,
                **kwargs,
            )
        return x
    

class UnetTransformer(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        assert config.num_hidden_layers % 2 == 0
        self.num_encoder_layers = config.num_hidden_layers // 2
        self.num_decoder_layers = config.num_hidden_layers // 2

        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            x: torch.Tensor,
            ve: List[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
        x0 = x
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.layers[i](
                x=x,
                attention_mask=attention_mask,
                vi=ve_enc[i],
                x0=x0,
                **kwargs,
            )
            skip_connections.append(x)
        
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.layers[self.num_encoder_layers + i](
                x=x,
                attention_mask=attention_mask,
                vi=ve_dec[i],
                x0=x0,
                **kwargs,
            )
        return x


class BatchedTransformerBlock(nn.Module):
    """TransformerBlock for batched (B, L, D) input with variable hidden sizes per layer.
    Supports x0 lambda mixing and value embedding mixing in attention.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        expansion_ratio: float,
        base_hidden_size: int = None,
        compile_flex_attention: bool = True,
    ):
        super().__init__()
        from types import SimpleNamespace
        config = SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            unet=True,
            compile_flex_attention=compile_flex_attention,
        )
        self.attn = SelfAttention(config)

        from model.utils import correction_fn
        corrected_dim = correction_fn(expansion_ratio, hidden_size)
        self.mlp_up = Linear(hidden_size, corrected_dim)
        self.mlp_down = Linear(corrected_dim, hidden_size)
        self.mlp_down.weight.data.zero_()
        self.mlp_relu = nn.ReLU()

        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

        if base_hidden_size is not None and base_hidden_size != hidden_size:
            self.x0_projection = Linear(base_hidden_size, hidden_size)
        else:
            self.x0_projection = None

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            vi: Optional[torch.Tensor] = None,
            x0: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
        if x0 is not None:
            if self.x0_projection is not None:
                x0 = self.x0_projection(x0)
            x = self.lambdas[0] * x + self.lambdas[1] * x0

        x = x + self.attn(x=norm(x), attention_mask=attention_mask, vi=vi, **kwargs)
        mlp_out = self.mlp_down(self.mlp_relu(self.mlp_up(norm(x))).square())
        x = x + mlp_out
        return x


class BatchedValueEmbedding(nn.Module):
    """Value embeddings for batched UNet with variable hidden sizes per layer.
    Embeddings are computed at full resolution from input_ids (B, L).
    Spatial downsampling to match each layer's resolution is handled by the transformer.
    """
    def __init__(self, vocab_size: int, hidden_sizes: List[int]):
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

    def forward(self, input_ids: torch.Tensor) -> tuple:
        """
        input_ids: (B, L)
        Returns (encoder_ve, decoder_ve) lists of value embeddings at full resolution.
        encoder_ve[i] has shape (B, L, hidden_sizes[i]).
        """
        encoder_ve = [emb(input_ids) for emb in self.encoder_embed]
        decoder_ve = [emb(input_ids) for emb in self.decoder_embed]
        return encoder_ve, decoder_ve


@torch.compiler.disable
def precompute_multiresolution_masks(
    input_ids: torch.Tensor,
    cls_token_id: int,
    pad_token_id: int,
    num_levels: int,
    sliding_window_size: int,
    n_heads: int,
    device: torch.device,
) -> List[Optional[object]]:
    """Pre-compute flex attention block masks at each UNet resolution level.

    This function is excluded from torch.compile via @torch.compiler.disable because
    create_block_mask is designed to run outside compiled regions, and tensors captured
    by mask_mod closures must be real (eager) tensors -- not Inductor ComputedBuffers
    with FlexibleLayout, which cause LoweringException in flex_attention_backward.

    Args:
        input_ids: (B, L) token IDs
        cls_token_id: CLS/BOS token ID marking document starts
        pad_token_id: PAD token ID
        num_levels: Number of resolution levels (including full resolution)
        sliding_window_size: Sliding window size for attention
        n_heads: Number of attention heads
        device: Device for mask computation

    Returns:
        List of BlockMask objects, one per resolution level. None for levels where L<=1.
    """
    B, L = input_ids.shape

    # Compute document IDs from CLS token positions (CLS marks start of each document)
    doc_ids = (input_ids == cls_token_id).cumsum(dim=1)  # (B, L)

    # Find last real (non-pad) token position per batch element
    is_real = (input_ids != pad_token_id)
    positions = torch.arange(L, device=device).expand(B, L)
    last_real = torch.where(is_real, positions, torch.zeros_like(positions)).max(dim=1).values  # (B,)

    masks = []
    current_doc_ids = doc_ids
    current_last_real = last_real
    current_L = L

    for level in range(num_levels):
        if current_L <= 1:
            masks.append(None)
            continue

        # Capture loop variables in closure via default args
        def make_mask_mod(doc_ids_l, last_real_l, sw_l):
            def mask_mod(b, h, q_idx, kv_idx):
                doc_mask = doc_ids_l[b, q_idx] == doc_ids_l[b, kv_idx]
                sw_mask = torch.abs(q_idx - kv_idx) < sw_l
                pad_mask = (q_idx <= last_real_l[b]) & (kv_idx <= last_real_l[b])
                return doc_mask & sw_mask & pad_mask
            return mask_mod

        mask_mod = make_mask_mod(current_doc_ids, current_last_real, sliding_window_size)

        block_mask = create_block_mask(
            mask_mod=mask_mod,
            B=B,
            H=n_heads,
            Q_LEN=current_L,
            KV_LEN=current_L,
            device=device,
        )
        masks.append(block_mask)

        # Downsample doc_ids and last_real for next level
        if current_L > 1:
            current_doc_ids = current_doc_ids.view(B, current_L // 2, 2).max(dim=-1).values
            current_last_real = current_last_real // 2
            current_L = current_L // 2

    return masks


class BatchedUnetTransformer(nn.Module):
    """Batched UNet Transformer with Swin-style patch merging/expanding.

    Operates on (B, L, D) tensors with pre-computed multi-resolution block masks.
    Uses PatchMerge for downsampling and PatchExpand for upsampling.
    Skip connections link encoder and decoder at matching resolutions.

    Architecture:
    - Encoder: TransformerBlock -> PatchMerge -> TransformerBlock -> PatchMerge -> ...
    - BottleneckMLP at vector depth (when L=1)
    - Decoder: PatchExpand -> TransformerBlock + skip -> PatchExpand -> ...
    """
    def __init__(self, config: PLMConfig):
        super().__init__()
        assert config.num_unet_layers % 2 == 0, "num_unet_layers must be even"
        assert config.max_length > 0 and (config.max_length & (config.max_length - 1)) == 0, \
            f"max_length must be a power of 2 for PatchMerge, got {config.max_length}"

        self.num_encoder_layers = config.num_unet_layers // 2
        self.num_decoder_layers = config.num_unet_layers // 2
        self.base_hidden_size = config.hidden_size
        self.max_length = config.max_length

        # Vector depth: after this many downsamplings, seq_len=1
        self.vector_depth = int(math.log2(config.max_length))

        # Hidden sizes for each encoder layer depth
        self.hidden_sizes = get_hidden_sizes(config.hidden_size, self.num_encoder_layers, config.num_attention_heads)

        # Number of resolution levels (for mask pre-computation)
        self.num_resolution_levels = min(self.num_encoder_layers, self.vector_depth + 1)

        # Encoder blocks
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

        # Decoder blocks
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

        # Skip connection weights
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        # Input/output projections if base hidden size differs from first layer
        if self.hidden_sizes[0] != config.hidden_size:
            self.input_projection = Linear(config.hidden_size, self.hidden_sizes[0])
            self.output_projection = Linear(self.hidden_sizes[0], config.hidden_size)
        else:
            self.input_projection = None
            self.output_projection = None

    def _downsample_to_resolution(self, x: torch.Tensor, target_L: int) -> torch.Tensor:
        """Average-pool pairs to spatially downsample x to target sequence length."""
        B, L, D = x.shape
        while L > target_L:
            assert L % 2 == 0, f"Cannot halve sequence length {L}"
            x = x.view(B, L // 2, 2, D).mean(dim=2)
            L = L // 2
        return x

    def forward(
            self,
            x: torch.Tensor,
            encoder_ve: List[torch.Tensor],
            decoder_ve: List[torch.Tensor],
            attention_masks: List[Optional[object]],
            x0_full: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
        """
        Forward pass for batched UNet.

        Args:
            x: (B, L, D) input embeddings
            encoder_ve: List of value embeddings at full resolution per encoder layer
            decoder_ve: List of value embeddings at full resolution per decoder layer
            attention_masks: Pre-computed BlockMask per resolution level
            x0_full: (B, L, D_base) original input for lambda mixing
        """
        # Project input to first layer hidden size if needed
        if self.input_projection is not None:
            x = self.input_projection(x)

        # Encoder path
        skip_connections = []
        mask_idx = 0
        downsample_idx = 0
        current_L = x.shape[1]

        for i in range(self.num_encoder_layers):
            # Attention mask for this resolution
            attn_mask = attention_masks[mask_idx] if mask_idx < len(attention_masks) else None

            # Downsample value embedding to current resolution
            vi = None
            if i < len(encoder_ve):
                vi = self._downsample_to_resolution(encoder_ve[i], current_L)

            # Downsample x0 to current resolution (x0 stays at base_hidden_size,
            # each block's x0_projection handles dim change)
            x0_current = self._downsample_to_resolution(x0_full, current_L)

            # Apply block
            x = self.encoder_blocks[i](
                x=x,
                attention_mask=attn_mask,
                vi=vi,
                x0=x0_current,
                **kwargs,
            )
            skip_connections.append(x)

            # Downsample for next layer
            if i < self.num_encoder_layers - 1 and i < self.vector_depth:
                x = self.downsamples[downsample_idx](x)
                downsample_idx += 1
                mask_idx += 1
                current_L = x.shape[1]

        # Decoder path
        upsample_idx = 0
        for i in range(self.num_decoder_layers):
            skip = skip_connections.pop()

            effective_depth = self.num_encoder_layers - 1 - i
            prev_depth = self.num_encoder_layers - i

            # Upsample x to match skip resolution
            if i > 0 and prev_depth <= self.vector_depth:
                x = self.upsamples[upsample_idx](x)
                upsample_idx += 1
                current_L = x.shape[1]

            # Add skip connection
            x = x + self.skip_weights[i] * skip

            # Attention mask for decoder at this resolution
            dec_mask_idx = min(effective_depth, len(attention_masks) - 1)
            attn_mask = attention_masks[dec_mask_idx] if attention_masks else None

            # Downsample value embedding to current resolution
            vi = None
            if i < len(decoder_ve):
                vi = self._downsample_to_resolution(decoder_ve[i], current_L)

            # Downsample x0 to current resolution
            x0_current = self._downsample_to_resolution(x0_full, current_L)

            # Apply block
            x = self.decoder_blocks[i](
                x=x,
                attention_mask=attn_mask,
                vi=vi,
                x0=x0_current,
                **kwargs,
            )

        # Project output back to base hidden size if needed
        if self.output_projection is not None:
            x = self.output_projection(x)

        return x


class PLM(PreTrainedModel):
    config_class = PLMConfig
    def __init__(self, config: PLMConfig):
        super().__init__(config)
        self.config = config
        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.cls_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.mlm = config.mlm
        self.masked_diffusion = config.masked_diffusion
        self.token_dropout = config.token_dropout

        self.vocab_size = config.vocab_size
        self.n_heads = config.num_attention_heads
        self.sliding_window_size = config.sliding_window_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.unet = config.unet
        self.conv_unet = config.conv_unet
        
        if config.conv_unet:
            # Batched UNet with Swin-style patch merge/expand
            assert config.num_unet_layers > 0, "num_unet_layers must be > 0 for conv_unet"
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
            from copy import copy
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
            self.lm_head.decoder.weight = self.embedding.weight
        
        self.ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

    def get_last_hidden_state(self, input_ids: torch.Tensor, sliding_window_size: int) -> torch.Tensor:
        if self.conv_unet:
            # Batched UNet path: input_ids is (B, L)
            assert input_ids.dim() == 2, f"conv_unet expects (B, L) input, got shape {input_ids.shape}"
            B, L = input_ids.shape

            # Pre-compute multi-resolution block masks
            attention_masks = precompute_multiresolution_masks(
                input_ids=input_ids,
                cls_token_id=self.cls_token_id,
                pad_token_id=self.pad_token_id,
                num_levels=self.transformer.num_resolution_levels,
                sliding_window_size=sliding_window_size,
                n_heads=self.n_heads,
                device=input_ids.device,
            )

            # Full resolution mask for extra layers
            full_res_mask = attention_masks[0]

            x = self.embedding(input_ids)  # (B, L, D)

            if self.token_dropout:
                x = x.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
                real_token_count = (input_ids != self.pad_token_id).sum(dim=1, keepdim=True).float().clamp(min=1)
                mask_count = (input_ids == self.mask_token_id).sum(dim=1, keepdim=True).float()
                mask_ratio_observed = mask_count / real_token_count
                x = (x * (1 - mask_ratio_observed.unsqueeze(-1))).to(x.dtype)

            x = norm(x)

            encoder_ve, decoder_ve = self.value_embeds(input_ids)

            x = self.transformer(
                x=x,
                encoder_ve=encoder_ve,
                decoder_ve=decoder_ve,
                attention_masks=attention_masks,
                x0_full=x.clone(),
            )

            # Apply extra layers at full resolution
            if self.extra_layers is not None:
                for layer in self.extra_layers:
                    x = layer(x=x, attention_mask=full_res_mask)

            return x

        # Standard / UNet path: input_ids is 1D (total_len,)
        docs = (input_ids == self.cls_token_id).cumsum(0)
        eos_positions = (input_ids == self.eos_token_id).nonzero()
        if eos_positions.numel() > 0:
            last_eos = eos_positions[-1].squeeze()
        else:
            last_eos = len(input_ids) - 1
        seq_len = len(input_ids)

        def doc_mask_mod(b, h, q_idx, kv_idx):
            bidirectional_sliding_window_mask = torch.abs(q_idx - kv_idx) < sliding_window_size
            doc_mask = docs[q_idx] == docs[kv_idx]
            pad_mask = (q_idx <= last_eos) & (kv_idx <= last_eos)
            return bidirectional_sliding_window_mask & doc_mask & pad_mask

        attention_mask = create_block_mask(
            mask_mod=doc_mask_mod,
            B=1,
            H=self.n_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=input_ids.device,
        )

        x = self.embedding(input_ids)

        if self.token_dropout:
            x = x.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            real_token_count = len(input_ids[:last_eos])
            mask_ratio_observed = (input_ids == self.mask_token_id).sum().float() / real_token_count
            x = (x * (1 - mask_ratio_observed)).to(x.dtype)

        x = norm(x)

        if self.unet:
            ve = self.value_embeds(input_ids)
            x = self.transformer(x=x, ve=ve, attention_mask=attention_mask, last_eos=last_eos)
        else:
            x = self.transformer(x=x, attention_mask=attention_mask, last_eos=last_eos)

        if self.extra_layers is not None:
            for layer in self.extra_layers:
                x = layer(x=x, attention_mask=attention_mask, last_eos=last_eos)

        return x

    def get_vector_embeddings(self, input_ids: torch.Tensor, sliding_window_size: Optional[int] = None) -> torch.Tensor:
        """Mean-pool hidden states per document to get per-document embeddings.
        
        Args:
            input_ids: (B, L) for conv_unet or (total_len,) for standard/unet
            sliding_window_size: Override sliding window size
        
        Returns:
            For conv_unet (B, L): flattened (total_docs, hidden_size) across all batch elements
            For standard (total_len,): (num_docs, hidden_size)
        """
        if sliding_window_size is None:
            sliding_window_size = self.sliding_window_size
        x = self.get_last_hidden_state(input_ids, sliding_window_size)

        if self.conv_unet:
            # Batched: x is (B, L, D), input_ids is (B, L)
            B, L, D = x.shape
            doc_ids = (input_ids == self.cls_token_id).cumsum(dim=1)  # (B, L)
            # Flatten batch into single sequence for mean pooling
            x_flat = x.reshape(-1, D)  # (B*L, D)
            # Offset doc_ids per batch element so each batch has unique doc IDs
            max_docs_per_batch = doc_ids.max(dim=1).values  # (B,)
            offsets = torch.zeros(B, dtype=doc_ids.dtype, device=doc_ids.device)
            offsets[1:] = max_docs_per_batch[:-1].cumsum(0)
            doc_ids = doc_ids + offsets.unsqueeze(1)
            doc_ids_flat = doc_ids.reshape(-1)  # (B*L,)
            # Exclude padding positions
            pad_mask = (input_ids.reshape(-1) != self.pad_token_id)
            num_docs = doc_ids_flat.max().item()
            doc_ids_0based = doc_ids_flat - 1
            doc_embeds = []
            for doc_idx in range(num_docs):
                mask = (doc_ids_0based == doc_idx) & pad_mask
                if mask.any():
                    doc_embeds.append(x_flat[mask].mean(dim=0))
            return torch.stack(doc_embeds, dim=0)
        else:
            # Legacy 1D path
            docs = (input_ids == self.cls_token_id).cumsum(0)
            x = x.view(-1, self.config.hidden_size)
            num_docs = docs.max().item()
            doc_ids = docs - 1
            doc_embeds = []
            for doc_idx in range(num_docs):
                mask = (doc_ids == doc_idx)
                doc_embeds.append(x[mask].mean(dim=0))
            return torch.stack(doc_embeds, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        mask_rate: torch.Tensor,
        sliding_window_size: Optional[int] = None,
        return_logits: bool = False,
        ) -> torch.Tensor:
        if sliding_window_size is None:
            sliding_window_size = self.sliding_window_size

        last_hidden_state = self.get_last_hidden_state(input_ids, sliding_window_size)

        lm_logits = self.lm_head(norm(last_hidden_state)) # (l, v)

        loss = self.ce(
            lm_logits.view(-1, self.vocab_size),
            labels.view(-1).long()
        )
        if self.training and self.masked_diffusion and not self.mlm:
            loss = loss / mask_rate

        if return_logits:
            return loss, lm_logits
        return loss

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor, sliding_window_size: Optional[int] = None) -> torch.Tensor:
        """Get LM logits without computing loss.

        Args:
            input_ids: (B, L) for conv_unet or (total_len,) for standard/unet
            sliding_window_size: Override sliding window size

        Returns:
            Logits tensor with shape matching input + vocab dim
        """
        if sliding_window_size is None:
            sliding_window_size = self.sliding_window_size
        hidden = self.get_last_hidden_state(input_ids, sliding_window_size)
        return self.lm_head(norm(hidden))

    @torch.no_grad()
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        sliding_window_size: Optional[int] = None,
        pooling: str = 'mean',
    ) -> torch.Tensor:
        """Get per-sequence pooled embeddings.

        Args:
            input_ids: (B, L) for conv_unet or (total_len,) for standard/unet
            sliding_window_size: Override sliding window size
            pooling: 'mean' for mean pooling over non-pad tokens, 'cls' for CLS token embedding

        Returns:
            (num_sequences, hidden_size) embeddings
        """
        if sliding_window_size is None:
            sliding_window_size = self.sliding_window_size
        hidden = self.get_last_hidden_state(input_ids, sliding_window_size)

        if self.conv_unet:
            # Batched: hidden is (B, L, D), input_ids is (B, L)
            assert input_ids.dim() == 2
            B, L, D = hidden.shape
            if pooling == 'cls':
                # CLS is the first token of each chunk
                return hidden[:, 0, :]  # (B, D)
            else:
                # Mean pool over non-pad tokens per batch element
                mask = (input_ids != self.pad_token_id).unsqueeze(-1).float()  # (B, L, 1)
                return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)
        else:
            # Legacy 1D: hidden is (total_len, D)
            if pooling == 'cls':
                # Return embedding at each CLS position
                cls_mask = (input_ids == self.cls_token_id)
                return hidden[cls_mask]  # (num_docs, D)
            else:
                # Mean pool per document
                return self.get_vector_embeddings(input_ids, sliding_window_size)

    def push_code_and_config_to_hub(self, repo_id: str):
        """Push source code and model config to HuggingFace Hub (no weights).

        Call once at the start of training so the repo is ready for
        trust_remote_code=True loading as soon as weights are uploaded later.
        """
        import shutil
        import tempfile
        from pathlib import Path
        from huggingface_hub import HfApi

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save only the config (this also writes config.json)
            self.config.save_pretrained(tmpdir)

            # Copy source files needed for trust_remote_code
            model_dir = Path(__file__).parent
            for src_file in ['model.py', 'attention.py', 'utils.py']:
                src_path = model_dir / src_file
                if src_path.exists():
                    shutil.copy2(src_path, Path(tmpdir) / src_file)

            api = HfApi()
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                repo_type="model",
            )

    def save_weights_local(self, save_dir: str, step: int):
        """Save model weights and optimizer-resumable checkpoint locally."""
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.save_pretrained(save_path / f"step_{step:06d}")

    def push_weights_to_hub(self, repo_id: str):
        """Push model weights to HuggingFace Hub (code + config already there)."""
        import tempfile
        from huggingface_hub import HfApi

        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir)

            api = HfApi()
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                repo_type="model",
            )


if __name__ == "__main__":
    # py -m model.model
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    from torchinfo import summary

    print("=" * 80)
    print("Testing Original UNet Transformer")
    print("=" * 80)
    config = PLMConfig(
        hidden_size=768,
        num_attention_heads=6,
        num_hidden_layers=24,
        expansion_ratio=8/3,
        unet=True,
    )
    model = PLM(config).cuda()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create test input with proper structure (CLS + sequence + EOS) - 1D for legacy path
    seq_len = 128
    input_ids = torch.randint(4, 33, (seq_len,)).cuda()
    input_ids[0] = 0   # CLS token
    input_ids[-1] = 2   # EOS token
    labels = input_ids.clone()
    labels[labels != 32] = -100
    mask_rate = torch.tensor(0.15).cuda()

    loss = model(input_ids, labels, mask_rate)
    print(f"Original UNet loss: {loss.item():.4f}")

    print("\n" + "=" * 80)
    print("Testing Batched UNet Transformer (conv_unet)")
    print("=" * 80)
    max_length = 128  # Power of 2 for patch merging
    conv_config = PLMConfig(
        hidden_size=384,
        num_attention_heads=6,
        num_unet_layers=8,  # 4 encoder + 4 decoder
        num_extra_layers=2,
        max_length=max_length,
        expansion_ratio=8/3,
        conv_unet=True,
    )
    conv_model = PLM(conv_config).cuda()
    print(f"Model parameters: {sum(p.numel() for p in conv_model.parameters()):,}")

    # Create batched test input (B, max_length) with packed documents per element
    B = 4
    batched_ids = torch.randint(4, 33, (B, max_length)).cuda()
    for b in range(B):
        # Insert CLS at start and EOS at end of each chunk
        batched_ids[b, 0] = 0
        batched_ids[b, max_length - 1] = 2
        # Add a second document boundary in the middle
        mid = max_length // 2
        batched_ids[b, mid - 1] = 2  # EOS for doc 1
        batched_ids[b, mid] = 0      # CLS for doc 2
    batched_labels = batched_ids.clone()
    batched_labels[batched_labels != 32] = -100

    loss = conv_model(batched_ids, batched_labels, mask_rate)
    print(f"Batched UNet loss: {loss.item():.4f}")

    print(f"\nHidden sizes: {conv_model.transformer.hidden_sizes}")
    print(f"Vector depth (log2(max_length)): {conv_model.transformer.vector_depth}")
    print(f"Num encoder layers: {conv_model.transformer.num_encoder_layers}")
    print(f"Num decoder layers: {conv_model.transformer.num_decoder_layers}")

    print("\n" + "=" * 80)
    print("Testing Batched UNet with deep layers (MLP at vector depth)")
    print("=" * 80)
    deep_config = PLMConfig(
        hidden_size=384,
        num_attention_heads=6,
        num_unet_layers=20,  # 10 encoder + 10 decoder (some will be MLPs)
        num_extra_layers=1,
        max_length=128,  # log2(128)=7, so layers 7+ become MLPs
        expansion_ratio=8/3,
        conv_unet=True,
    )
    deep_model = PLM(deep_config).cuda()

    # Count transformer vs MLP blocks
    n_transformer = sum(1 for b in deep_model.transformer.encoder_blocks if isinstance(b, BatchedTransformerBlock))
    n_mlp = sum(1 for b in deep_model.transformer.encoder_blocks if isinstance(b, BottleneckMLP))
    print(f"Encoder: {n_transformer} transformer blocks, {n_mlp} MLP blocks")

    n_transformer_dec = sum(1 for b in deep_model.transformer.decoder_blocks if isinstance(b, BatchedTransformerBlock))
    n_mlp_dec = sum(1 for b in deep_model.transformer.decoder_blocks if isinstance(b, BottleneckMLP))
    print(f"Decoder: {n_transformer_dec} transformer blocks, {n_mlp_dec} MLP blocks")

    loss = deep_model(batched_ids, batched_labels, mask_rate)
    print(f"Deep Batched UNet loss: {loss.item():.4f}")

    print("\n" + "=" * 80)
    print("Testing Multi-Resolution Mask Pre-computation")
    print("=" * 80)

    # Verify mask shapes at each resolution level
    from model.model import precompute_multiresolution_masks
    masks = precompute_multiresolution_masks(
        input_ids=batched_ids,
        cls_token_id=0,
        pad_token_id=1,
        num_levels=conv_model.transformer.num_resolution_levels,
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