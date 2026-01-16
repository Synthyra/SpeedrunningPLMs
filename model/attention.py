import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from model.flex_mods import generate_tanh_softcap
from model.utils import norm, Linear


@dataclass
class AttentionContext:
    attention_mask: Optional[torch.Tensor]
    window_size: int
    valid_len: int
    is_paired: bool


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


class YarnPairedHead(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=self.head_dim // 4, dtype=torch.float32
        )
        angular_freq = angular_freq.repeat_interleave(2)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim // 2)])
        t = torch.arange(2 * self.max_seq_len, dtype=torch.float32)
        t_even = 2 * t
        t_odd = 2 * t + 1
        theta1 = torch.outer(t_even, angular_freq)
        theta2 = torch.outer(t_odd, angular_freq)
        self.factor1 = nn.Buffer(
            torch.cat((theta1.cos(), theta2.cos()), dim=-1).to(torch.bfloat16),
            persistent=False,
        )
        self.factor2 = nn.Buffer(
            torch.cat((theta1.sin(), theta2.sin()), dim=-1).to(torch.bfloat16),
            persistent=False,
        )
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32):
        rotations = old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(2 * self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        t_even = 2 * t
        t_odd = 2 * t + 1
        theta1 = torch.outer(t_even, self.angular_freq)
        theta2 = torch.outer(t_odd, self.angular_freq)
        self.factor1.copy_(torch.cat((theta1.cos(), theta2.cos()), dim=-1))
        self.factor2.copy_(torch.cat((theta1.sin(), theta2.sin()), dim=-1))
        self.factor2[..., 1::2] *= -1
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

    def rotary(self, x: torch.Tensor) -> torch.Tensor:
        factor1 = self.factor1[None, : x.size(-3), None, :]
        factor2 = self.factor2[None, : x.size(-3), None, :]
        x_flip = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).flip(-1).view(x.shape)
        return factor1 * x + factor2 * x_flip


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

        if key_offset:
            k[:, 1:, :, self.d_head // 4 : self.d_head // 2] = k[:, :-1, :, self.d_head // 4 : self.d_head // 2]
            k[:, 1:, :, 3 * self.d_head // 4 :] = k[:, :-1, :, 3 * self.d_head // 4 :]

        if vi is not None:
            gate_in = x[..., : self.value_embed_gate.in_features]
            ve_gate_out = 2 * torch.sigmoid(self.value_embed_gate(gate_in)).view(1, l, self.n_heads, 1)
            v = v + ve_gate_out * vi.view_as(v)

        y = flex_attention(
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


class PairedHeadSelfAttention(nn.Module):
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

        self.yarn = YarnPairedHead(self.d_head, config.max_seq_len)
        self.attn_gate = Linear(self.gate_dim, self.n_heads)
        self.value_embed_gate = Linear(self.value_gate_dim, self.n_heads)

        if config.attention_soft_cap and config.add_att_soft_cap:
            self.soft_cap_mod = generate_tanh_softcap(config.attention_soft_cap, approx=True)
        else:
            self.soft_cap_mod = None

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

        q, k = norm(q), norm(k)

        q = q.view(1, l, self.n_heads // 2, self.d_head * 2)
        k = k.view(1, l, self.n_heads // 2, self.d_head * 2)
        v = v.reshape(1, l * 2, self.n_heads // 2, self.d_head)

        q, k = self.yarn.rotary(q), self.yarn.rotary(k)

        q = q.view(1, l * 2, self.n_heads // 2, self.d_head)
        k = k.view(1, l * 2, self.n_heads // 2, self.d_head)

        if key_offset:
            k[:, 2:, :, self.d_head // 4 : self.d_head // 2] = k[:, :-2, :, self.d_head // 4 : self.d_head // 2]
            k[:, 2:, :, 3 * self.d_head // 4 :] = k[:, :-2, :, 3 * self.d_head // 4 :]

        if vi is not None:
            gate_in = x[..., : self.value_embed_gate.in_features]
            ve_gate_out = 2 * torch.sigmoid(self.value_embed_gate(gate_in)).view(
                1, l * 2, self.n_heads // 2, 1
            )
            vi = vi.view(1, l, self.n_heads, self.d_head)
            vi = vi.view(1, l * 2, self.n_heads // 2, self.d_head)
            v = v + ve_gate_out * vi

        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            score_mod=self.soft_cap_mod,
            block_mask=attention_ctx.attention_mask,
            enable_gqa=True,
        )
        y = y.transpose(1, 2)

        y = y.view(1, l, self.n_heads, self.d_head)
        gate_in = x[..., : self.attn_gate.in_features]
        gate = torch.sigmoid(self.attn_gate(gate_in)).view(1, l, self.n_heads, 1)
        y = y * gate
        y = y.contiguous().view_as(x)
        y = self.Wo(y)
        return y
