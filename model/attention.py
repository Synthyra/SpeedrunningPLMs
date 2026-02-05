import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from torch.nn.attention.flex_attention import flex_attention

from model.utils import norm, Linear


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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.d_head = self.hidden_size // self.n_heads

        assert self.hidden_size % self.n_heads == 0
        self.Wq = Linear(self.hidden_size, self.hidden_size)
        self.Wk = Linear(self.hidden_size, self.hidden_size)
        self.Wv = Linear(self.hidden_size, self.hidden_size)
        self.rotary = Rotary(self.d_head) # dim // num_attention_heads = head_dim
        self.Wo = Linear(self.hidden_size, self.hidden_size)
        self.Wo.weight.data.zero_() # zero init suggested by @Grad6230497
        
        if config.unet:
            self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))

        self.unet = config.unet
        self.flex_attention = flex_attention
        if config.compile_flex_attention:
            self.flex_attention = torch.compile(flex_attention)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            vi: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
        l, d = x.size() # batch size must be 1 for FlexAttention
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)

        q = q.view(1, l, self.n_heads, self.d_head)
        k = k.view(1, l, self.n_heads, self.d_head)
        v = v.view(1, l, self.n_heads, self.d_head)

        if self.unet and vi is not None:
            # Reshape vi from (l, d) to (1, l, n_heads, d_head) to match v's shape
            v = self.lambdas[0] * v + self.lambdas[1] * vi.view_as(v)
        
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        if attention_mask is None:
            assert l <= 1, "attention_mask is required for seq_len > 1 to avoid dense attention"
        
        y = self.flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            score_mod=None,
            block_mask=attention_mask,
            enable_gqa=True,
        )
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.Wo(y)
        return y
