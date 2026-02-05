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


@dataclass
class ESMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None


def get_hidden_sizes(hidden_size: int, num_encoder_layers: int, num_attention_heads: int = 1) -> List[int]:
    """Returns hidden size for each encoder layer, rounded to multiples of 64 and num_attention_heads.
    Scales from hidden_size to hidden_size * 2 at the bottleneck.
    
    Args:
        hidden_size: Base hidden size
        num_encoder_layers: Number of encoder layers
        num_attention_heads: Number of attention heads (hidden size must be divisible by this)
    """
    # Find LCM of 64 and num_attention_heads for GPU efficiency and head divisibility
    from math import gcd
    alignment = (64 * num_attention_heads) // gcd(64, num_attention_heads)
    
    sizes = []
    for i in range(num_encoder_layers):
        # Linear interpolation from 1.0 to 2.0
        scale = 1.0 + (i / max(num_encoder_layers - 1, 1))
        raw_size = hidden_size * scale
        # Round to nearest multiple of alignment for GPU efficiency and head divisibility
        rounded = int(((raw_size + alignment - 1) // alignment) * alignment)
        sizes.append(rounded)
    return sizes


class DownsampleConv(nn.Module):
    """Conv1D that halves sequence length and adjusts hidden dim.
    Input: (seq_len, in_dim) -> Output: (seq_len // 2, out_dim)
    
    Handles dynamic padding for short documents to ensure no document
    becomes length < 2 after downsampling.
    """
    def __init__(self, in_dim: int, out_dim: int, base_hidden_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=2, stride=2)
        self.in_dim = in_dim
        # Project base pad embedding to this layer's input dimension
        self.pad_projection = Linear(base_hidden_size, in_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        pad_embedding: torch.Tensor,
        docs: torch.Tensor,
    ) -> tuple:
        """
        Args:
            x: Hidden states (seq_len, hidden_dim)
            pad_embedding: Learned padding vector (base_hidden_size,)
            docs: Document IDs per position (seq_len,)
        
        Returns:
            x_out: Downsampled hidden states
            docs_out: Downsampled document IDs
            padding_mask: Mask indicating inserted padding positions (pre-downsample)
        """
        # Project pad embedding to this layer's dimension
        pad_vector = self.pad_projection(pad_embedding)
        
        # Insert padding for short documents
        x_padded, docs_padded, padding_mask = insert_padding_for_short_docs(
            x, docs, pad_vector, min_post_downsample_length=2
        )
        
        # Pad to even length if needed (so Conv1D doesn't lose information)
        seq_len = len(x_padded)
        if seq_len % 2 == 1:
            # Pad with zeros and extend docs with last doc ID
            x_padded = F.pad(x_padded, (0, 0, 0, 1))  # Pad sequence dimension
            docs_padded = F.pad(docs_padded, (0, 1), value=docs_padded[-1].item())
            # Extend padding_mask to indicate this position (will be removed on upsample)
            padding_mask = F.pad(padding_mask, (0, 1), value=True)
        
        # Apply convolution: x is (seq_len, hidden_dim), need (batch, channels, seq_len)
        x_conv = x_padded.unsqueeze(0).transpose(1, 2)  # (1, hidden_dim, seq_len)
        x_conv = self.conv(x_conv)  # (1, out_dim, seq_len // 2)
        x_out = x_conv.transpose(1, 2).squeeze(0)  # (seq_len // 2, out_dim)
        
        # Downsample docs to match conv output length exactly
        target_len = len(x_out)
        docs_out = downsample_docs(docs_padded, target_len)
        
        return x_out, docs_out, padding_mask


class UpsampleConv(nn.Module):
    """ConvTranspose1D that doubles sequence length and adjusts hidden dim.
    Input: (seq_len, in_dim) -> Output: (seq_len * 2, out_dim)
    
    Handles removal of padding tokens that were inserted during downsampling.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_dim, out_dim, kernel_size=2, stride=2)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Hidden states (seq_len, hidden_dim)
            padding_mask: Boolean mask indicating padding positions from corresponding downsample
        
        Returns:
            x_out: Upsampled hidden states with padding removed
        """
        # Apply transposed convolution: x is (seq_len, hidden_dim), need (batch, channels, seq_len)
        x = x.unsqueeze(0).transpose(1, 2)  # (1, hidden_dim, seq_len)
        x = self.conv(x)  # (1, out_dim, seq_len * 2)
        x = x.transpose(1, 2).squeeze(0)  # (seq_len * 2, out_dim)
        
        # Remove padding tokens if mask is provided
        if padding_mask is not None and padding_mask.any():
            x = remove_padding(x, padding_mask)
        
        return x


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


def downsample_docs(docs: torch.Tensor, target_len: int = None) -> torch.Tensor:
    """Downsample document IDs to match Conv1D downsampling behavior.
    
    Uses max pooling to preserve document boundaries. For odd lengths,
    truncates to match Conv1D's floor division behavior.
    
    Args:
        docs: Document IDs per position (seq_len,)
        target_len: Optional target length to match Conv1D output
    """
    if len(docs) <= 1:
        return docs
    
    # Conv1D with kernel=2, stride=2 produces floor(input_len / 2) outputs
    conv_output_len = len(docs) // 2
    
    # For odd sequences, truncate last element to match conv behavior
    if len(docs) % 2 == 1:
        docs = docs[:-1]  # Truncate to even length
    
    # Reshape and take max of pairs to preserve document boundaries
    docs = docs.view(-1, 2)
    result = docs.max(dim=1).values
    
    # Truncate to target length if specified
    if target_len is not None and len(result) > target_len:
        result = result[:target_len]
    
    return result


def get_doc_boundaries(docs: torch.Tensor) -> tuple:
    """Get document boundaries and lengths from docs tensor.
    
    Args:
        docs: Tensor of document IDs per position (seq_len,)
    
    Returns:
        boundaries: Tensor of start positions for each document
        lengths: Tensor of lengths for each document
        num_docs: Number of documents
    """
    seq_len = len(docs)
    if seq_len == 0:
        return torch.tensor([0], device=docs.device), torch.tensor([0], device=docs.device), 0
    
    # Find where document ID changes
    changes = torch.where(docs[:-1] != docs[1:])[0] + 1
    # Prepend 0 and append seq_len for boundaries
    boundaries = torch.cat([
        torch.tensor([0], device=docs.device),
        changes,
        torch.tensor([seq_len], device=docs.device)
    ])
    lengths = boundaries[1:] - boundaries[:-1]
    num_docs = len(lengths)
    return boundaries, lengths, num_docs


def insert_padding_for_short_docs(
    x: torch.Tensor,
    docs: torch.Tensor,
    pad_vector: torch.Tensor,
    min_post_downsample_length: int = 2,
) -> tuple:
    """Insert padding tokens for documents that would be too short after downsampling.
    
    Args:
        x: Hidden states (seq_len, hidden_dim)
        docs: Document IDs per position (seq_len,)
        pad_vector: Learned padding vector (hidden_dim,)
        min_post_downsample_length: Minimum length after stride-2 downsample
    
    Returns:
        x_padded: Padded hidden states
        docs_padded: Padded document IDs
        padding_mask: Boolean mask indicating which positions are padding (for removal later)
    """
    boundaries, lengths, num_docs = get_doc_boundaries(docs)
    
    # Calculate post-downsample lengths: ceil(length / 2)
    post_lengths = (lengths + 1) // 2
    
    # Find which docs need padding (would become < min_post_downsample_length)
    needs_padding = post_lengths < min_post_downsample_length
    
    if not needs_padding.any():
        # No padding needed
        return x, docs, torch.zeros(len(x), dtype=torch.bool, device=x.device)
    
    # Calculate how many pad tokens each doc needs
    # To have min_post_downsample_length after halving, we need 2*min_post_downsample_length before
    min_pre_length = 2 * min_post_downsample_length
    pad_counts = torch.where(
        needs_padding,
        torch.clamp(min_pre_length - lengths, min=0),
        torch.zeros_like(lengths)
    )
    total_pad = pad_counts.sum().item()
    
    if total_pad == 0:
        return x, docs, torch.zeros(len(x), dtype=torch.bool, device=x.device)
    
    # Build new padded sequence
    new_seq_len = len(x) + total_pad
    x_padded = torch.zeros(new_seq_len, x.shape[1], dtype=x.dtype, device=x.device)
    docs_padded = torch.zeros(new_seq_len, dtype=docs.dtype, device=docs.device)
    padding_mask = torch.zeros(new_seq_len, dtype=torch.bool, device=x.device)
    
    # Copy data with padding insertions
    src_pos = 0
    dst_pos = 0
    for doc_idx in range(num_docs):
        doc_start = boundaries[doc_idx].item()
        doc_end = boundaries[doc_idx + 1].item()
        doc_len = doc_end - doc_start
        doc_id = docs[doc_start].item()
        
        # Copy original document tokens
        x_padded[dst_pos:dst_pos + doc_len] = x[doc_start:doc_end]
        docs_padded[dst_pos:dst_pos + doc_len] = doc_id
        dst_pos += doc_len
        
        # Insert padding if needed
        n_pad = pad_counts[doc_idx].item()
        if n_pad > 0:
            x_padded[dst_pos:dst_pos + n_pad] = pad_vector
            docs_padded[dst_pos:dst_pos + n_pad] = doc_id  # Same doc ID for attention
            padding_mask[dst_pos:dst_pos + n_pad] = True
            dst_pos += n_pad
    
    return x_padded, docs_padded, padding_mask


def remove_padding(
    x: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Remove padding tokens that were inserted during downsampling.
    
    After upsampling via ConvTranspose1D, the sequence is restored to the
    pre-downsample padded length. We simply remove the padding positions.
    
    Args:
        x: Hidden states after upsample (restored to pre-downsample length)
        padding_mask: Boolean mask from insert_padding (same length as padded sequence)
    
    Returns:
        x_unpadded: Hidden states with padding removed
    """
    # Handle length mismatch (upsample might not perfectly restore length)
    if len(padding_mask) > len(x):
        # Truncate mask to match x
        padding_mask = padding_mask[:len(x)]
    elif len(padding_mask) < len(x):
        # Extend mask with False (extra positions are not padding)
        padding_mask = F.pad(padding_mask, (0, len(x) - len(padding_mask)), value=False)
    
    # Remove padding positions
    return x[~padding_mask]


def downsample_last_eos(last_eos: int) -> int:
    """Downsample last_eos position by halving."""
    return last_eos // 2


class FlexTransformerBlock(nn.Module):
    """TransformerBlock that accepts hidden_size as a parameter for variable sizes per layer.
    Includes x0 projection to handle variable hidden dims in Conv1D UNet.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        expansion_ratio: float, 
        base_hidden_size: int = None,
    ):
        super().__init__()
        # Create a minimal config-like object for SelfAttention
        from types import SimpleNamespace
        config = SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            unet=True,  # Enable value embedding mixing
        )
        self.attn = SelfAttention(config)
        
        # Create MLP with explicit hidden_size
        from model.utils import correction_fn
        corrected_dim = correction_fn(expansion_ratio, hidden_size)
        self.mlp_up = Linear(hidden_size, corrected_dim)
        self.mlp_down = Linear(corrected_dim, hidden_size)
        self.mlp_down.weight.data.zero_()
        self.mlp_relu = nn.ReLU()
        
        # Lambda mixing weights for x0
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        
        # Projection layer for x0 if hidden sizes differ
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
        # Apply lambda mixing with x0 (project x0 if needed)
        if x0 is not None:
            if self.x0_projection is not None:
                x0 = self.x0_projection(x0)
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        
        # Attention
        x = x + self.attn(
            x=norm(x),
            attention_mask=attention_mask,
            vi=vi,
            **kwargs,
        )
        
        # MLP
        mlp_out = self.mlp_down(self.mlp_relu(self.mlp_up(norm(x))).square())
        x = x + mlp_out
        return x


class ConvValueEmbedding(nn.Module):
    """Value embeddings for Conv1D UNet with variable hidden sizes per layer."""
    def __init__(self, vocab_size: int, hidden_sizes: List[int]):
        super().__init__()
        num_encoder_layers = len(hidden_sizes)
        # Create embeddings for encoder layers
        self.encoder_embed = nn.ModuleList([
            nn.Embedding(vocab_size, hidden_sizes[i])
            for i in range(num_encoder_layers)
        ])
        # Decoder uses reversed hidden sizes
        self.decoder_embed = nn.ModuleList([
            nn.Embedding(vocab_size, hidden_sizes[num_encoder_layers - 1 - i])
            for i in range(num_encoder_layers)
        ])
    
    def forward(self, input_ids: torch.Tensor) -> tuple:
        """Returns (encoder_ve, decoder_ve) lists of value embeddings."""
        encoder_ve = [emb(input_ids) for emb in self.encoder_embed]
        decoder_ve = [emb(input_ids) for emb in self.decoder_embed]
        return encoder_ve, decoder_ve


class ConvUnetTransformer(nn.Module):
    """Conv1D UNet Transformer with downsampling along sequence length.
    
    Architecture:
    - First encoder block: full resolution
    - Encoder blocks 1 to N-1: downsampled by 2x each
    - Decoder blocks: upsampled symmetrically
    - Last decoder block: full resolution
    - MLP blocks replace transformer blocks when sequence length = 1
    """
    def __init__(self, config: PLMConfig):
        super().__init__()
        assert config.num_unet_layers % 2 == 0, "num_unet_layers must be even"
        
        self.num_encoder_layers = config.num_unet_layers // 2
        self.num_decoder_layers = config.num_unet_layers // 2
        
        # Calculate at what depth sequence becomes a vector
        # With max_length and halving each layer after the first
        self.vector_depth = int(math.log2(config.max_length))
        
        # Get hidden sizes for each encoder layer
        self.hidden_sizes = get_hidden_sizes(config.hidden_size, self.num_encoder_layers, config.num_attention_heads)
        self.base_hidden_size = config.hidden_size
        
        # Learned padding embedding for dynamic document padding
        self.pad_embedding = nn.Parameter(torch.randn(config.hidden_size))
        
        # Track which layers are at vector depth (need MLP instead of transformer)
        # Layer 0 is full resolution, layer 1 halves, etc.
        # After layer i, seq_len = max_length / 2^i
        # Sequence becomes 1 when i >= log2(max_length)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i in range(self.num_encoder_layers):
            # Depth i means sequence has been halved i times (after downsampling)
            # Layer 0 operates at full resolution before any downsampling
            effective_depth = i  # After this layer's downsample, depth will be i
            
            # For layers at or past vector_depth, use the hidden size at vector_depth
            # (since there are no more downsamples to change dimensions)
            layer_hidden_size = self.hidden_sizes[min(i, self.vector_depth)]
            
            if effective_depth >= self.vector_depth:
                # At vector depth, use MLP
                self.encoder_blocks.append(
                    BottleneckMLP(layer_hidden_size, config.expansion_ratio, self.base_hidden_size)
                )
            else:
                # Use transformer block with x0 projection
                self.encoder_blocks.append(
                    FlexTransformerBlock(
                        hidden_size=layer_hidden_size,
                        num_attention_heads=config.num_attention_heads,
                        expansion_ratio=config.expansion_ratio,
                        base_hidden_size=self.base_hidden_size,
                    )
                )
            
            # Add downsample between layers (except after last encoder layer or at vector depth)
            # Don't downsample if we're already at or past vector depth (sequence is 1)
            if i < self.num_encoder_layers - 1 and effective_depth < self.vector_depth:
                self.downsamples.append(
                    DownsampleConv(self.hidden_sizes[i], self.hidden_sizes[i + 1], self.base_hidden_size)
                )
        
        # Decoder blocks (reversed order of hidden sizes)
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(self.num_decoder_layers):
            # Decoder layer i corresponds to encoder layer (num_encoder - 1 - i)
            enc_idx = self.num_encoder_layers - 1 - i
            
            # Effective depth for decoder (counting from bottleneck back up)
            effective_depth = self.num_encoder_layers - 1 - i
            
            # For layers at or past vector_depth, use the hidden size at vector_depth
            decoder_hidden_size = self.hidden_sizes[min(enc_idx, self.vector_depth)]
            
            # Add upsample before each decoder layer (except first which is at bottleneck)
            # Upsample if the corresponding encoder layer was reached via downsample
            prev_effective_depth = self.num_encoder_layers - i  # Previous layer's depth
            if i > 0 and prev_effective_depth <= self.vector_depth:
                prev_enc_idx = self.num_encoder_layers - i
                prev_hidden_size = self.hidden_sizes[min(prev_enc_idx, self.vector_depth)]
                self.upsamples.append(
                    UpsampleConv(prev_hidden_size, decoder_hidden_size)
                )
            
            if effective_depth >= self.vector_depth:
                # At vector depth, use MLP
                self.decoder_blocks.append(
                    BottleneckMLP(decoder_hidden_size, config.expansion_ratio, self.base_hidden_size)
                )
            else:
                # Use transformer block with x0 projection
                self.decoder_blocks.append(
                    FlexTransformerBlock(
                        hidden_size=decoder_hidden_size,
                        num_attention_heads=config.num_attention_heads,
                        expansion_ratio=config.expansion_ratio,
                        base_hidden_size=self.base_hidden_size,
                    )
                )
        
        # Skip connection weights for decoder
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        
        # Projection layers for skip connections with dimension mismatch
        self.skip_projections = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            enc_idx = self.num_encoder_layers - 1 - i
            # Skip comes from encoder layer enc_idx
            if i == 0:
                # First decoder layer: no upsample yet, skip from last encoder
                self.skip_projections.append(None)  # Same dimensions
            else:
                # Skip projection if needed (dimensions should match after upsample)
                self.skip_projections.append(None)
        
        # Initial projection from base hidden size to first encoder hidden size
        if self.hidden_sizes[0] != config.hidden_size:
            self.input_projection = Linear(config.hidden_size, self.hidden_sizes[0])
        else:
            self.input_projection = None
        
        # Final projection back to base hidden size
        if self.hidden_sizes[0] != config.hidden_size:
            self.output_projection = Linear(self.hidden_sizes[0], config.hidden_size)
        else:
            self.output_projection = None
    
    def downsample_value_embedding(self, ve: torch.Tensor, target_len: int = None) -> torch.Tensor:
        """Downsample value embedding to match conv1d downsampling behavior.
        
        Args:
            ve: Value embedding (seq_len, hidden_dim)
            target_len: Optional target length to match (for precise alignment with conv output)
        """
        if len(ve) <= 1:
            return ve
        
        # Use floor division to match Conv1D behavior (kernel=2, stride=2)
        # Conv1D output: floor(input_len / 2)
        new_len = len(ve) // 2
        
        # Truncate to target length if specified and needed
        if target_len is not None:
            new_len = min(new_len, target_len)
        
        # Take every other position, potentially truncating
        result = ve[::2]
        if len(result) > new_len:
            result = result[:new_len]
        
        return result
    
    def create_mask_for_resolution(
        self,
        docs: torch.Tensor,
        last_eos: int,
        sliding_window_size: int,
        n_heads: int,
        device: torch.device,
    ):
        """Create attention mask for a given resolution (docs tensor at that resolution)."""
        seq_len = len(docs)
        
        if seq_len <= 1:
            return None
        
        # Capture current values in closure
        _docs = docs
        _last_eos = min(last_eos, seq_len - 1)  # Clamp to valid range
        _sliding_window = sliding_window_size
        
        def doc_mask_mod(b, h, q_idx, kv_idx, docs=_docs, last_eos=_last_eos, sw=_sliding_window):
            bidirectional_sliding_window_mask = torch.abs(q_idx - kv_idx) < sw
            doc_mask = docs[q_idx] == docs[kv_idx]
            pad_mask = (q_idx <= last_eos) & (kv_idx <= last_eos)
            return bidirectional_sliding_window_mask & doc_mask & pad_mask
        
        return create_block_mask(
            mask_mod=doc_mask_mod,
            B=1,
            H=n_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )
    
    def forward(
            self,
            x: torch.Tensor,
            encoder_ve: List[torch.Tensor],
            decoder_ve: List[torch.Tensor],
            docs: torch.Tensor,
            last_eos: int,
            sliding_window_size: int,
            n_heads: int,
            x0_full: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
        """
        Forward pass with dynamic document padding.
        
        Args:
            x: Input embeddings (seq_len, hidden_dim)
            encoder_ve: Value embeddings for encoder layers
            decoder_ve: Value embeddings for decoder layers
            docs: Document IDs per position (seq_len,)
            last_eos: Position of last EOS token
            sliding_window_size: Sliding window size for attention
            n_heads: Number of attention heads
            x0_full: Original input for skip connections
        """
        device = x.device
        
        # Project input if needed
        if self.input_projection is not None:
            x = self.input_projection(x)
            x0_full = self.input_projection(x0_full)
        
        # Track x0 at each resolution for lambda mixing
        x0 = x0_full
        current_docs = docs
        current_last_eos = last_eos
        
        # Track attention masks and padding masks at each encoder resolution
        encoder_attention_masks = []
        encoder_docs = []  # docs at each encoder resolution
        encoder_last_eos = []  # last_eos at each encoder resolution
        padding_masks = []  # Stack of padding masks for upsample
        
        # Track whether padding has been applied - if so, skip value embeddings
        # (since vi positions won't align after padding insertion)
        padding_applied = False
        
        # Encoder path
        skip_connections = []
        downsample_idx = 0
        for i in range(self.num_encoder_layers):
            effective_depth = i
            
            # Store docs and last_eos at this resolution (before any padding for this layer's downsample)
            encoder_docs.append(current_docs.clone())
            encoder_last_eos.append(current_last_eos)
            
            # Create attention mask for current resolution
            attn_mask = self.create_mask_for_resolution(
                current_docs, current_last_eos, sliding_window_size, n_heads, device
            )
            encoder_attention_masks.append(attn_mask)
            
            # Get value embedding (skip if padding has been applied)
            vi = None
            if not padding_applied and i < len(encoder_ve):
                vi = encoder_ve[i]
                # Downsample vi to match current sequence length exactly
                target_len = len(x)
                while len(vi) > target_len:
                    vi = self.downsample_value_embedding(vi, target_len)
            
            # Downsample x0 to match current resolution (skip if padding has been applied)
            x0_current = None
            if not padding_applied:
                x0_current = x0
                target_len = len(x)
                while len(x0_current) > target_len:
                    # Use floor division to match conv behavior
                    x0_current = x0_current[:len(x0_current) // 2 * 2][::2]
                # Ensure exact length match
                if len(x0_current) > target_len:
                    x0_current = x0_current[:target_len]
            
            # Apply block
            x = self.encoder_blocks[i](
                x=x,
                attention_mask=attn_mask,
                vi=vi,
                x0=x0_current,
                **kwargs,
            )
            skip_connections.append(x)
            
            # Downsample for next layer (only if not at vector depth)
            if i < self.num_encoder_layers - 1 and effective_depth < self.vector_depth:
                # Downsample with dynamic padding
                x, current_docs, padding_mask = self.downsamples[downsample_idx](
                    x, self.pad_embedding, current_docs
                )
                current_last_eos = downsample_last_eos(current_last_eos)
                padding_masks.append(padding_mask)
                
                # Track if any padding was inserted
                if padding_mask.any():
                    padding_applied = True
                    
                downsample_idx += 1
        
        # Decoder path
        upsample_idx = 0
        for i in range(self.num_decoder_layers):
            # Get skip connection
            skip = skip_connections.pop()
            
            # Effective depth for this decoder layer
            effective_depth = self.num_encoder_layers - 1 - i
            prev_effective_depth = self.num_encoder_layers - i  # Previous layer's depth
            
            # Upsample x to match skip resolution (except first decoder layer)
            # Upsample if the corresponding encoder layer was reached via downsample
            if i > 0 and prev_effective_depth <= self.vector_depth:
                # Get corresponding padding mask (reverse order)
                padding_mask = padding_masks.pop() if padding_masks else None
                x = self.upsamples[upsample_idx](x, padding_mask)
                upsample_idx += 1
            
            # Ensure dimensions match for skip connection
            if x.shape != skip.shape:
                # Adjust sequence length if needed
                if x.shape[0] > skip.shape[0]:
                    x = x[:skip.shape[0]]
                elif x.shape[0] < skip.shape[0]:
                    skip = skip[:x.shape[0]]
            
            # Add skip connection
            x = x + self.skip_weights[i] * skip
            
            # Get value embedding for this decoder layer (skip if any padding was applied during encoding)
            vi = None
            if not padding_applied and i < len(decoder_ve):
                vi = decoder_ve[i]
                target_len = len(x)
                while len(vi) > target_len:
                    vi = self.downsample_value_embedding(vi, target_len)
            
            # Get x0 at current resolution (skip if any padding was applied during encoding)
            x0_current = None
            if not padding_applied:
                x0_current = x0_full
                target_len = len(x)
                while len(x0_current) > target_len:
                    # Use floor division to match conv behavior
                    x0_current = x0_current[:len(x0_current) // 2 * 2][::2]
                # Ensure exact length match
                if len(x0_current) > target_len:
                    x0_current = x0_current[:target_len]
            
            # Create attention mask dynamically for decoder based on actual x length
            # Use the encoder docs and last_eos at this resolution, adjusted for length differences
            enc_idx = self.num_encoder_layers - 1 - i
            dec_docs = encoder_docs[enc_idx]
            dec_last_eos = encoder_last_eos[enc_idx]
            
            # Adjust docs length to match x if needed
            if len(dec_docs) != len(x):
                if len(dec_docs) > len(x):
                    dec_docs = dec_docs[:len(x)]
                    dec_last_eos = min(dec_last_eos, len(x) - 1)
                else:
                    # Extend with last doc ID
                    dec_docs = F.pad(dec_docs, (0, len(x) - len(dec_docs)), value=dec_docs[-1].item())
            
            # Create mask for actual sequence length
            attn_mask = self.create_mask_for_resolution(
                dec_docs, dec_last_eos, sliding_window_size, n_heads, device
            )
            
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
            # Conv1D UNet with downsampling
            assert config.num_unet_layers > 0, "num_unet_layers must be > 0 for conv_unet"
            self.transformer = ConvUnetTransformer(config)
            # Get hidden sizes from the transformer for value embeddings
            hidden_sizes = self.transformer.hidden_sizes
            self.value_embeds = ConvValueEmbedding(config.vocab_size, hidden_sizes)
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
        docs = (input_ids == self.cls_token_id).cumsum(0)
        eos_positions = (input_ids == self.eos_token_id).nonzero()
        if eos_positions.numel() > 0:
            last_eos = eos_positions[-1].squeeze()
        else:
            # If no EOS token found, use the last position of the sequence
            last_eos = len(input_ids) - 1
        seq_len = len(input_ids)

        # Create full-resolution attention mask
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
        
        if self.conv_unet:
            # Conv1D UNet path with dynamic document padding
            encoder_ve, decoder_ve = self.value_embeds(input_ids)
            
            # Pass docs and mask params to forward for dynamic mask creation
            x = self.transformer(
                x=x,
                encoder_ve=encoder_ve,
                decoder_ve=decoder_ve,
                docs=docs,
                last_eos=last_eos.item() if isinstance(last_eos, torch.Tensor) else last_eos,
                sliding_window_size=sliding_window_size,
                n_heads=self.n_heads,
                x0_full=x.clone(),
            )
        elif self.unet:
            # Original UNet path
            ve = self.value_embeds(input_ids)
            x = self.transformer(
                x=x,
                ve=ve,
                attention_mask=attention_mask,
                last_eos=last_eos,
            )
        else:
            # Standard transformer path
            x = self.transformer(
                x=x,
                attention_mask=attention_mask,
                last_eos=last_eos,
            )
        
        # Apply extra layers after U-Net (at full resolution)
        if self.extra_layers is not None:
            for layer in self.extra_layers:
                x = layer(
                    x=x,
                    attention_mask=attention_mask,
                    last_eos=last_eos,
                )
        
        return x

    def get_vector_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        docs = (input_ids == self.cls_token_id).cumsum(0)
        x = self.get_last_hidden_state(input_ids)
        x = x.view(-1, self.config.hidden_size) # (S, hidden_size)
        # At this point, x is shape [S, hidden_size]
        # We want to mean-pool across each document index.
        # Convert docs to 0-based so we can do nice indexing
        num_docs = docs.max().item()
        doc_ids = docs - 1  # Now documents are labeled [0, 1, 2, ...]
        # Mean-pool across tokens belonging to each doc
        doc_embeds = []
        for doc_idx in range(num_docs):
            mask = (doc_ids == doc_idx)
            # Collect all token embeddings for this doc and average
            doc_embeds.append(x[mask].mean(dim=0))
        # Stack into [num_documents, hidden_size]
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

    # Create test input with proper structure (CLS + sequence + EOS)
    seq_len = 128
    input_ids = torch.randint(4, 33, (seq_len,)).cuda()  # Random amino acids
    input_ids[0] = 0  # CLS token
    input_ids[-1] = 2  # EOS token
    labels = input_ids.clone()
    labels[labels != 32] = -100  # Only compute loss on masked tokens
    mask_rate = torch.tensor(0.15).cuda()
    
    loss = model(input_ids, labels, mask_rate)
    print(f"Original UNet loss: {loss.item():.4f}")
    
    print("\n" + "=" * 80)
    print("Testing Conv1D UNet Transformer")
    print("=" * 80)
    conv_config = PLMConfig(
        hidden_size=768,
        num_attention_heads=6,
        num_unet_layers=12,  # 6 encoder + 6 decoder layers
        num_extra_layers=2,  # 2 extra layers after U-Net
        max_length=1024,
        expansion_ratio=8/3,
        conv_unet=True,
    )
    conv_model = PLM(conv_config).cuda()
    print(f"Model parameters: {sum(p.numel() for p in conv_model.parameters()):,}")
    
    # Test forward pass
    loss = conv_model(input_ids, labels, mask_rate)
    print(f"Conv1D UNet loss: {loss.item():.4f}")
    
    # Print hidden sizes for Conv UNet
    print(f"\nConv UNet hidden sizes: {conv_model.transformer.hidden_sizes}")
    print(f"Vector depth (log2(max_length)): {conv_model.transformer.vector_depth}")
    print(f"Num encoder layers: {conv_model.transformer.num_encoder_layers}")
    print(f"Num decoder layers: {conv_model.transformer.num_decoder_layers}")
    
    print("\n" + "=" * 80)
    print("Testing Conv1D UNet with deep layers (MLP at vector depth)")
    print("=" * 80)
    deep_config = PLMConfig(
        hidden_size=768,
        num_attention_heads=6,
        num_unet_layers=36,  # 12 encoder + 12 decoder layers (some will be MLPs)
        num_extra_layers=1,
        max_length=1024,  # log2(1024) = 10, so layers 10+ will be MLPs
        expansion_ratio=8/3,
        conv_unet=True,
    )
    deep_model = PLM(deep_config).cuda()
    
    # Count transformer vs MLP blocks
    n_transformer = sum(1 for b in deep_model.transformer.encoder_blocks if isinstance(b, FlexTransformerBlock))
    n_mlp = sum(1 for b in deep_model.transformer.encoder_blocks if isinstance(b, BottleneckMLP))
    print(f"Encoder: {n_transformer} transformer blocks, {n_mlp} MLP blocks")
    
    n_transformer_dec = sum(1 for b in deep_model.transformer.decoder_blocks if isinstance(b, FlexTransformerBlock))
    n_mlp_dec = sum(1 for b in deep_model.transformer.decoder_blocks if isinstance(b, BottleneckMLP))
    print(f"Decoder: {n_transformer_dec} transformer blocks, {n_mlp_dec} MLP blocks")
    
    # Test forward pass
    loss = deep_model(input_ids, labels, mask_rate)
    print(f"Deep Conv1D UNet loss: {loss.item():.4f}")
    
    print("\n" + "=" * 80)
    print("Testing Variable-Length Documents (Dynamic Padding)")
    print("=" * 80)
    
    # Create a sequence with multiple variable-length documents
    # Including some very short documents that will need padding
    # Format: CLS + doc1_tokens + EOS + CLS + doc2_tokens + EOS + ...
    
    doc_lengths = [64, 3, 128, 2, 32, 4, 16]  # Mix of long and short docs
    print(f"Document lengths: {doc_lengths}")
    
    # Build the concatenated sequence
    all_tokens = []
    for doc_len in doc_lengths:
        all_tokens.append(0)  # CLS token
        all_tokens.extend([torch.randint(4, 33, ()).item() for _ in range(doc_len)])
        all_tokens.append(2)  # EOS token
    
    input_ids_var = torch.tensor(all_tokens).cuda()
    total_len = len(input_ids_var)
    print(f"Total sequence length: {total_len}")
    
    labels_var = input_ids_var.clone()
    labels_var[labels_var != 32] = -100
    
    # Create a smaller model that can handle various document lengths
    var_config = PLMConfig(
        hidden_size=384,
        num_attention_heads=6,
        num_unet_layers=8,  # 4 encoder + 4 decoder
        num_extra_layers=1,
        max_length=512,  # Smaller for testing
        expansion_ratio=8/3,
        conv_unet=True,
    )
    var_model = PLM(var_config).cuda()
    
    # Test the padding logic helper functions
    from model.model import get_doc_boundaries, insert_padding_for_short_docs
    
    docs = (input_ids_var == 0).cumsum(0)  # Document IDs
    boundaries, lengths, num_docs = get_doc_boundaries(docs)
    print(f"\nDocument boundaries: {boundaries.tolist()}")
    print(f"Document lengths (from tensor): {lengths.tolist()}")
    print(f"Number of documents: {num_docs}")
    
    # Test padding insertion
    test_x = torch.randn(total_len, 384).cuda()
    pad_emb = torch.randn(384).cuda()
    x_padded, docs_padded, padding_mask = insert_padding_for_short_docs(test_x, docs, pad_emb)
    
    print(f"\nOriginal sequence length: {len(test_x)}")
    print(f"Padded sequence length: {len(x_padded)}")
    print(f"Padding tokens inserted: {padding_mask.sum().item()}")
    
    # Test forward pass with variable-length documents
    loss = var_model(input_ids_var, labels_var, mask_rate)
    print(f"Variable-length docs loss: {loss.item():.4f}")
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)