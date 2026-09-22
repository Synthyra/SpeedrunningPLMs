import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Protocol
from torch.nn.attention.flex_attention import BlockMask, create_mask, flex_attention

from .layers import Linear, norm


class AttentionConfig(Protocol):
    hidden_size: int
    num_attention_heads: int
    unet: bool
    compile_flex_attention: bool


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000) -> None:
        super().__init__()
        self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))  # (d_h / 2,)
        self.seq_len_cached: Optional[int] = None
        self.cos_cached: Optional[torch.Tensor] = None  # (l, d_h / 2)
        self.sin_cached: Optional[torch.Tensor] = None  # (l, d_h / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, l, h, d_h); d_h is the even per-head width.
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)  # (l,)
            freqs = torch.outer(t, self.inv_freq)  # (l, d_h / 2)
            self.seq_len_cached = seq_len
            self.cos_cached = freqs.cos()  # (l, d_h / 2)
            self.sin_cached = freqs.sin()  # (l, d_h / 2)
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]  # each (1, l, 1, d_h / 2)
        first_half, second_half = x.chunk(2, dim=3)  # each (b, l, h, d_h / 2)
        rotated_first = first_half * cos + second_half * sin  # (b, l, h, d_h / 2)
        rotated_second = first_half * (-sin) + second_half * cos  # (b, l, h, d_h / 2)
        return torch.cat((rotated_first, rotated_second), 3).type_as(x)  # (b, l, h, d_h)


class SelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # d
        self.n_heads = config.num_attention_heads  # h
        self.d_head = self.hidden_size // self.n_heads  # d_h

        assert self.hidden_size % self.n_heads == 0
        self.Wq = Linear(self.hidden_size, self.hidden_size)
        self.Wk = Linear(self.hidden_size, self.hidden_size)
        self.Wv = Linear(self.hidden_size, self.hidden_size)
        self.rotary = Rotary(self.d_head)
        self.Wo = Linear(self.hidden_size, self.hidden_size)
        self.Wo.weight.data.zero_()  # (d, d); start with a zero attention residual.
        
        if config.unet:
            self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))  # (2,)

        self.unet = config.unet
        self.flex_attention = flex_attention
        if config.compile_flex_attention:
            self.flex_attention = torch.compile(flex_attention)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[BlockMask] = None,
            vi: Optional[torch.Tensor] = None,
            **kwargs: object,
        ) -> torch.Tensor:
        # x, vi: (l, d) or (b, l, d); attention_mask encodes (b, h, l, l).
        squeeze_out = False
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, l, d)
            squeeze_out = True
            if vi is not None:
                vi = vi.unsqueeze(0)  # (1, l, d)

        batch_size, seq_len, hidden_size = x.size()
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)  # each (b, l, d)

        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head)  # (b, l, h, d_h)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head)  # (b, l, h, d_h)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head)  # (b, l, h, d_h)

        if self.unet and vi is not None:
            V = self.lambdas[0] * V + self.lambdas[1] * vi.view_as(V)  # (b, l, h, d_h)
        
        Q, K = norm(Q), norm(K)  # each (b, l, h, d_h)
        Q, K = self.rotary(Q), self.rotary(K)  # each (b, l, h, d_h)
        if attention_mask is None:
            assert seq_len <= 1, "attention_mask is required for seq_len > 1 to avoid dense attention"
        
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # each (b, h, l, d_h)
        if Q.device.type == "cpu":
            # FlexAttention does not support CPU backward. Build the exact
            # token-level mask from the BlockMask closure and use PyTorch's
            # differentiable dense attention fallback for CPU use.
            dense_mask = None  # Optional (b, h, l, l).
            if attention_mask is not None:
                dense_mask = create_mask(
                    attention_mask.mask_mod,
                    B=batch_size,
                    H=self.n_heads,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=Q.device,
                )  # (b, h, l, l)
            output = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=dense_mask,
            )  # (b, h, l, d_h)
        else:
            output = self.flex_attention(
                Q,
                K,
                V,
                score_mod=None,
                block_mask=attention_mask,
                enable_gqa=True,
            )  # (b, h, l, d_h)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)  # (b, l, d)
        output = self.Wo(output)  # (b, l, d)

        if squeeze_out:
            output = output.squeeze(0)  # (l, d)
        return output  # (l, d) or (b, l, d), matching x on entry.
