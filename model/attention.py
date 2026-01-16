import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from model.flex_mods import generate_tanh_softcap
from model.utils import norm, Linear

try:
    _compiled_flex_attention = torch.compile(flex_attention)
except Exception:
    _compiled_flex_attention = flex_attention


def _apply_key_offset(k: torch.Tensor, key_offset: bool, shift: int, d_head: int) -> torch.Tensor:
    if shift <= 0:
        return k
    shifted = k.clone()
    quarter = d_head // 4
    half = d_head // 2
    three_quarters = 3 * d_head // 4
    shifted[:, shift:, :, quarter:half] = k[:, :-shift, :, quarter:half]
    shifted[:, shift:, :, three_quarters:] = k[:, :-shift, :, three_quarters:]
    offset_flag = torch.as_tensor(key_offset, device=k.device, dtype=torch.bool)
    return torch.where(offset_flag, shifted, k)


@dataclass
class AttentionContext:
    attention_mask: Optional[torch.Tensor]
    window_size: int
    valid_len: int


class Yarn(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=self.head_dim // 4, dtype=torch.float32
        )
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim // 4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
        self.sin = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        self.angular_freq = angular_freq
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32):
        rotations = old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

    def rotary(self, x: torch.Tensor) -> torch.Tensor:
        cos = self.cos[: x.size(-3)][None, :, None, :]
        sin = self.sin[: x.size(-3)][None, :, None, :]
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
        self.gate_dim = min(config.attn_gate_dim, self.hidden_size)
        self.value_gate_dim = min(config.value_embed_gate_dim, self.hidden_size)

        assert self.hidden_size % self.n_heads == 0
        self.Wq = Linear(self.hidden_size, self.hidden_size)
        self.Wk = Linear(self.hidden_size, self.hidden_size)
        self.Wv = Linear(self.hidden_size, self.hidden_size)
        self.Wo = Linear(self.hidden_size, self.hidden_size)
        self.Wo.weight.data.zero_()

        self.yarn = Yarn(self.d_head, config.max_seq_len)
        self.attn_gate = Linear(self.gate_dim, self.n_heads)
        self.value_embed_gate = Linear(self.value_gate_dim, self.n_heads)

        if config.unet:
            self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))

        if config.attention_soft_cap and config.add_att_soft_cap:
            self.soft_cap_mod = generate_tanh_softcap(config.attention_soft_cap, approx=True)
        else:
            self.soft_cap_mod = None
        self.unet = config.unet

    def apply_yarn(self, old_window: int, new_window: int):
        self.yarn.apply(old_window, new_window)

    def forward(
        self,
        x: torch.Tensor,
        attention_ctx: AttentionContext,
        vi: Optional[torch.Tensor] = None,
        key_offset: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        l, _ = x.size()
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)

        q = q.view(1, l, self.n_heads, self.d_head)
        k = k.view(1, l, self.n_heads, self.d_head)
        v = v.view(1, l, self.n_heads, self.d_head)

        if self.unet and vi is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * vi.view_as(v)

        q, k = norm(q), norm(k)
        q, k = self.yarn.rotary(q), self.yarn.rotary(k)

        k = _apply_key_offset(k, key_offset, shift=1, d_head=self.d_head)

        if vi is not None:
            gate_in = x[..., : self.value_embed_gate.in_features]
            ve_gate_out = 2 * torch.sigmoid(self.value_embed_gate(gate_in)).view(1, l, self.n_heads, 1)
            v = v + ve_gate_out * vi.view_as(v)

        y = _compiled_flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            score_mod=self.soft_cap_mod,
            block_mask=attention_ctx.attention_mask,
            enable_gqa=True,
        )
        y = y.transpose(1, 2)

        gate_in = x[..., : self.attn_gate.in_features]
        gate = torch.sigmoid(self.attn_gate(gate_in)).view(1, l, self.n_heads, 1)
        y = y * gate

        y = y.contiguous().view_as(x)
        y = self.Wo(y)
        return y


