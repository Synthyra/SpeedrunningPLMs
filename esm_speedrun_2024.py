"""The 2024 speedrun's model, as `Synthyra/esm150_speedrun_20000_steps` was trained: its inference path only.

The repository's first trainer (December 2024, before the AFDB v6 benchmark) trained a 24-block, 768-wide U-Net
transformer with value embeddings on OMG prot50 for 20,000 steps of 524,288 tokens, 9.00 hours on one GPU, and pushed
it to the Hub. Its run log (`logs_to_keep/e631bf18-f202-492b-a3b8-fbae2cb7484a.txt` in the repository's history before
commit 651248c) embeds the `model.py` it ran; this module is that file's inference path, copied verbatim: the
configuration, norm, casted linear, rotary embedding, attention's `forward_sdpa`, the ReLU-squared MLP, a block's
`sdpa_forward`, the value embeddings, and the model's `sdpa_forward` and logits (a tanh soft cap at 30). Left out are
the masker, the tokenizer download, FlexAttention and the training losses. Training's attention window grew to 2,048
tokens within each protein, so full attention over one protein of at most 1,024 residues is the model it became.

    model = load_esm_speedrun_2024("Synthyra/esm150_speedrun_20000_steps", revision)
    logits = model.sdpa_forward(input_ids)  # input_ids: (L + 2,), ESM2's 33 tokens, <cls> residues <eos>
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple
from transformers import PretrainedConfig, PreTrainedModel


class ModelConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=33,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        expansion_ratio=8/3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.expansion_ratio = expansion_ratio


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq)
            self.seq_len_cached = seq_len
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # apply_rotary_emb(x, cos, sin)
        x1, x2 = x.chunk(2, dim=3)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_attention_heads):
        super().__init__()
        assert dim % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.qkv = CastedLinear(dim, 3 * dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim // num_attention_heads) # dim // num_attention_heads = head_dim
        self.o_proj = CastedLinear(dim, dim)
        self.o_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward_sdpa(self, x: torch.Tensor, vi: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = x.size(0), x.size(1) # batch size, sequence length
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_attention_heads, -1)
        k = k.view(B, T, self.num_attention_heads, -1)
        v = v.view(B, T, self.num_attention_heads, -1)
        v = self.lambdas[0] * v + self.lambdas[1] * vi.view_as(v) # @KoszarskyB & @Grad62304977
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True
        )
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.o_proj(y)
        return y


def correction_fn(expansion_ratio: float, d_model: int) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class MLP(nn.Module):
    def __init__(self, dim, expansion_ratio):
        super().__init__()
        self.up   = CastedLinear(dim, correction_fn(expansion_ratio, dim))
        self.down = CastedLinear(correction_fn(expansion_ratio, dim), dim)
        self.down.weight.data.zero_() # zero init suggested by @Grad62304977
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # https://arxiv.org/abs/2109.08668v2
        # ReLU squared ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        return self.down(self.relu(self.up(x)).square())


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config.hidden_size, config.num_attention_heads)
        self.mlp = MLP(config.hidden_size, config.expansion_ratio)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def sdpa_forward(self, x: torch.Tensor, vi: torch.Tensor, x0: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn.forward_sdpa(norm(x), vi, attention_mask)
        x = x + self.mlp(norm(x))
        return x


class ValueEmbedding(nn.Module):
    def __init__(self, config: "ModelConfig"):
        super().__init__()
        self.embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.hidden_size)
            for _ in range(config.num_hidden_layers // 2)
        ])

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        ve = [emb(inputs) for emb in self.embed]
        ve += reversed(ve)
        return ve


class ESM(PreTrainedModel):
    config_class = ModelConfig
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size  # the log read EsmTokenizer's 33, which the configuration also holds
        self.num_hidden_layers = config.num_hidden_layers

        # U-net design by @brendanh0gan
        assert config.num_hidden_layers % 2 == 0, "Number of layers should be even for U-net design"
        self.num_encoder_layers = config.num_hidden_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.num_hidden_layers - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.embed = nn.Embedding(self.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
        # U-net structure on token value embeddings by @leloykun
        self.value_embeds = ValueEmbedding(config)
        self.lm_head = CastedLinear(config.hidden_size, self.vocab_size)
        self.lm_head.weight.data.zero_() # @Grad62304977

    def embed_forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        x = self.embed(input_ids[None])
        x = norm(x) # @Grad62304977
        x0 = x
        ve = self.value_embeds(input_ids)
        return x, x0, ve

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        return logits

    def sdpa_forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].bool()

        x, x0, ve = self.embed_forward(input_ids)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i].sdpa_forward(x, ve_enc[i], x0, attention_mask)
            skip_connections.append(x)

        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i].sdpa_forward(x, ve_dec[i], x0, attention_mask)

        return self.get_logits(x)


def load_esm_speedrun_2024(repo: str, revision: str, device: torch.device | str = "cpu") -> ESM:
    """The checkpoint at `revision` in its training dtypes, as the trainer cast them (`model.bfloat16()`, then every
    `CastedLinear` back to float32), each tensor checked: no key missing, none left over, each dtype as saved."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config = ModelConfig.from_pretrained(repo, revision=revision)
    model = ESM(config).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    state = load_file(hf_hub_download(repo, "model.safetensors", revision=revision))
    model.load_state_dict(state, strict=True)
    for name, tensor in model.state_dict().items():
        assert tensor.dtype == state[name].dtype, f"{name}: loaded {tensor.dtype}, saved {state[name].dtype}"
    return model.to(device).eval()
