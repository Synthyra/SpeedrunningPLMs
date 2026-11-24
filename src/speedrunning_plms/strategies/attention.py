"""The attention family: the kernel each block's attention sublayer runs, with its training masks.

- `standard` (the record): h heads of d_h with QK-norm, RoPE and the positional prior, one flex call.
- `differential`: Differential Transformer V2 (B2). 2h query heads over h key and value heads; the two query heads of a
  pair share one KV head, and the pair's output is attn_even - sigmoid(lambda) attn_odd, with lambda per token and
  head from a zero-initialized projection of the input (so the pair starts at attn_even - 0.5 attn_odd).
- `triangle`: Dual Triangle Attention (B3, after `projects/representation/dual_triangle_attention`). Each head splits
  into a down sub-head (keys at or before the query) and an up sub-head (at or after) of d_h / 2, each with its own
  softmax; the split keeps RoPE's rotation pairs whole, so each sub-head stays relative-position equivariant. Two
  flex calls with directional document masks, about half the attention FLOPs of one full call.
- `triangle_differential`: both, composed: each logical head has two query branches, each split down and up; the
  two branches share the down and up keys and values, and subtract as in `differential`.
- `paired`: paired-head attention (B6, modded-nanogpt record 58). Two heads interleave along the sequence, so one
  attention over 2l entries lets a query of either head read the keys of both; bidirectional, so the scores double.

Two numbers act under every kernel, both off (0) in the record:
- `key_offset` (noise PR 50, modded-nanogpt PR 169, bidirectional form): through the `keys` hook, on the raw keys
  before QK-norm and RoPE, m = round(key_offset d_h / 2) of each head's rotation pairs (both channels of a pair, so a
  pair rotates as one) come from the previous token, and the next m pairs below them from the next token. They are
  the lowest-frequency pairs, the near-stationary ones PR 169 shifted. A token whose neighbor is in another protein
  (the document mask's boundary) or past the window keeps its own key there. Keys come from inputs, so a masked
  neighbor contributes the MASK embedding's key, never its label.
- `score_cap`: cap tanh(s / cap) on the scaled score q.k / sqrt(d_h) before the distance terms (inside the flex score
  function as a constant; capping the full score would floor far tokens' penalty at -cap and erase the slopes). With
  unit-RMS q and k the scaled score lies in [-sqrt(d_h), sqrt(d_h)], 8 at 12 heads, so a cap of 3 to 8 binds. The
  dense path (CPU and evaluation) computes the capped scores explicitly instead of SDPA.
- `local_every` (ModernBERT, arXiv 2412.13663; Gemma 3; `ideas/2026-10-30_long_protein_attention.md` idea 4): in
  blocks whose index modulo `local_every` is not 0, a query sees only keys of its own protein within LOCAL_REACH
  positions on either side (a 511-token window); the other blocks stay global. `training_masks` builds the local
  masks once per step beside the document masks and appends them, so a local block takes the second half of the tuple;
  the dense path ands the same band into its mask, so evaluation sees what training saw. Only proteins longer than
  LOCAL_REACH change.
- `kv_share` (cross-layer attention, arXiv 2405.12981; `ideas/2026-10-30_long_protein_attention.md` idea 5): blocks
  whose index modulo `kv_share` is not 0 have no key or value projection and reuse the K and V of their group's first
  block, taken after its key hooks and QK-norm and before RoPE and the value hooks, which each block then applies by
  its own rule (so `nope_every`, the value residual and the value tables compose). Queries stay per block, so flex
  runs once per block as before. A throughput change: at 2, half the blocks skip 2 of their 6 matrices (12 of 54
  units of 768^2 x tokens, about 10 percent of the matmul work), at the risk of the capacity CLA reports as cheap in
  decoders; it removes 1.18M parameters a borrowing block.

Round 9 (`ideas/2026-11-03_round9_strategies.md`, no runs yet), each off at its default:
- `heads` widens to 4 and 8: screen1's 6 heads of 128 was the screen's largest effect (-0.0129, 9 percent more steps), and 6 is 95 percent
  of the search's top fifth, against 12, with nothing on either side of it tried.
- `qk_share` (modded-nanogpt record 90: QK width 128 to 96, V and the stream unchanged): Q and K narrower than V, a multiple of 4 near
  this share of d_h, so a rotation pair and the triangle's halves stay whole. Flex takes a value width of its own. The query and key matmuls and
  the attention's scores shrink by the same share, and the head's temperature follows the narrower width.
- `tie_kv` (GQA-S2, arXiv 2602.01845): the raw keys are the values, one projection fewer of the block's six. Inert under a `qk_share`
  under 1, where the widths differ.
- `exclusive` (gated XSA, djdumpling's modded-nanogpt notes): Y <- Y - tanh(a) <Y, v> v per head, v the unit own value at the token, a
  learned per block and head from zero, so step 0 is the record's; the attention can then drop what the token's own value already says, or
  learn not to. 007's ungated form lost, which is why the gate. The gates train with AdamW at the scalar rate.
- `block_size`: flex's mask granularity (`create_block_mask`, default 128; `ideas/2026-10-30_step_throughput.md` idea 4), where a smaller
  block can skip more empty work at protein edges, or cost more metadata. The rule is the same per element, so only the step time can differ.
  The kernels need a multiple of their `BLOCK_M` and `BLOCK_N` of 128, so 64 fails at compile and only larger blocks can run.

Symbols: b windows or proteins, l tokens, d = 768, h heads, d_h per-head width, e the QK width, q query heads, p = h / 2 pairs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.positional import Positional, Rotary
from speedrunning_plms.strategies.strategy import CandidateModel, ForwardContext, ParameterGroup, Strategy, StrategyConfig, norm


if TYPE_CHECKING:  # the registry imports this module
    from speedrunning_plms.strategies import Strategies


DIFFERENTIAL = ("differential", "triangle_differential")
TRIANGLE = ("triangle", "triangle_differential")
LOCAL_REACH = 256  # a local block's query sees keys fewer than this many positions away
MaskMod = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]  # flex's (batch, head, query, key) -> allowed
SHARED_KEYS, SHARED_VALUES = "shared_keys", "shared_values"  # `kv_share`'s entries in the forward's stored tensors
HEADS = (4, 6, 8, 12)  # even, so paired heads pair, and d_h holds the triangle's whole rotation pairs
DEFAULT_BLOCK_SIZE = 128  # `create_block_mask`'s own
HEAD_WIDTH_MULTIPLE = 16  # d_h the flex and SDPA kernels tile
MAX_HEAD_WIDTH = 256  # the widest d_h flex attention compiles


@dataclass(frozen=True)
class AttentionConfig(StrategyConfig):
    variant: str = "standard"
    heads: int = 12
    key_offset: float = 0.0  # share of each head's key pairs from the previous token, and as many from the next
    score_cap: float = 0.0  # tanh cap on the scaled scores; 0 is off
    local_every: int = 0  # blocks whose index modulo this is not 0 attend locally; 0 is all global
    kv_share: int = 0  # blocks whose index modulo this is not 0 reuse the group's first K and V; 0 shares none
    qk_share: float = 1.0  # the QK width over the value width
    tie_kv: int = 0  # 1 reads the values from the key projection
    exclusive: int = 0  # 1 adds the gated removal of each token's own value from the attention's output
    block_size: int = DEFAULT_BLOCK_SIZE  # flex's mask granularity
    VARIANTS: ClassVar[tuple[str, ...]] = ("standard", "differential", "triangle", "triangle_differential", "paired")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"heads": (4, 12), "key_offset": (0.0, 0.5),  # d_h of 192 to 64
                                                        "score_cap": (0.0, 8.0), "local_every": (0, 3), "kv_share": (0, 3),
                                                        "qk_share": (0.5, 1.0), "tie_kv": (0, 1), "exclusive": (0, 1),
                                                        "block_size": (32, 512)}

    def qk_width(self, hidden: int) -> int:
        """Q's and K's per-head width e on a `hidden`-wide stream: a multiple of 4 near `qk_share` of d_h, so d_h itself
        at the record's 1."""
        d_h = hidden // self.heads
        return d_h if self.qk_share == 1.0 else 4 * round(self.qk_share * d_h / 4)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.heads not in HEADS:
            raise ValueError(f"heads must be one of {HEADS} (d_h 192 to 64), not {self.heads}")
        if self.tie_kv not in (0, 1) or self.exclusive not in (0, 1):
            raise ValueError(f"tie_kv and exclusive are 0 or 1, not {self.tie_kv!r}, {self.exclusive!r}")
        if self.score_cap != 0.0 and self.score_cap < 3.0:
            raise ValueError(f"score_cap is 0 (off) or 3 to 8, not {self.score_cap}")
        if self.local_every not in (0, 2, 3):
            raise ValueError(f"local_every is 0 (off), 2 or 3, not {self.local_every!r}")
        if self.kv_share not in (0, 2, 3):
            raise ValueError(f"kv_share is 0 (off), 2 or 3, not {self.kv_share!r}")

    def local(self, index: int) -> bool:
        """True for a block that attends within LOCAL_REACH only."""
        return self.local_every > 0 and index % self.local_every != 0

    def borrows_kv(self, index: int) -> bool:
        """True for a block that reuses its group's first K and V and has no key or value projection."""
        return self.kv_share > 0 and index % self.kv_share != 0


def split_rotation_pairs(X: Tensor) -> tuple[Tensor, Tensor]:
    """Two sub-heads of d_h / 2 that each hold whole RoPE pairs (coordinates i and i + d_h / 2): even and odd i."""
    # X: (..., d_h); returns two (..., d_h / 2).
    halves = X.unflatten(-1, (2, X.shape[-1] // 4, 2))  # (..., 2, d_h / 4, 2): half, pair index / 2, parity
    return halves[..., 0].flatten(-2), halves[..., 1].flatten(-2)  # each (..., d_h / 2)


def combine_differential(Y: Tensor, lam: Tensor) -> Tensor:
    """Each pair of query heads, attn_even - lambda attn_odd."""
    # Y: (b, 2h, l, d_h) with pairs adjacent; lam: (b, h, l, 1) in (0, 1); returns (b, h, l, d_h).
    pairs = Y.unflatten(1, (Y.shape[1] // 2, 2))  # (b, h, 2, l, d_h)
    return pairs[:, :, 0] - lam.to(Y.dtype) * pairs[:, :, 1]  # (b, h, l, d_h)


def capped_attention(Q: Tensor, K: Tensor, V: Tensor, attention_mask: Tensor, cap: float) -> Tensor:
    """Attention with cap tanh(s / cap) on the scaled scores before the additive mask, the dense twin of the flex
    score function under `score_cap`."""
    # Q: (b, q, n, e); K, V: (b, k, n, e); attention_mask: (b, q or 1, n, n) additive; returns (b, q, n, e).
    group = Q.shape[1] // K.shape[1]
    if group > 1:
        K, V = K.repeat_interleave(group, dim=1), V.repeat_interleave(group, dim=1)  # (b, q, n, e) each
    scores = (Q.float() @ K.float().transpose(-2, -1)) * Q.shape[-1] ** -0.5  # (b, q, n, n)
    scores = cap * torch.tanh(scores / cap) + attention_mask.float()  # (b, q, n, n)
    return (torch.softmax(scores, dim=-1).to(V.dtype) @ V)  # (b, q, n, e)


def without_own_value(Y: Tensor, V: Tensor, gate: Tensor) -> Tensor:
    """Gated exclusive attention: each output less a learned share of its projection on the token's own value."""
    # Y, V: (b, h, l, d_h); gate: (h,) unbounded, a share through tanh; returns (b, h, l, d_h).
    unit = V / V.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (b, h, l, d_h)
    along = (Y * unit).sum(dim=-1, keepdim=True)  # (b, h, l, 1)
    return Y - torch.tanh(gate).to(Y.dtype).view(1, -1, 1, 1) * along * unit  # (b, h, l, d_h)


def shift_key_pairs(K: Tensor, input_ids: Tensor, pairs: int) -> Tensor:
    """The raw keys with their last `pairs` rotation pairs from the previous token and the `pairs` before those from
    the next, within each protein; a token at a protein's edge or the window's keeps its own."""
    # K: (b, l, h, d_h), raw keys; input_ids: (b, l); returns (b, l, h, d_h).
    half = K.shape[-1] // 2  # pair i is channels i and i + half
    document = documents(input_ids)  # (b, l)
    same_previous = torch.cat((torch.zeros_like(document[:, :1], dtype=torch.bool), document[:, 1:] == document[:, :-1]), dim=1)  # (b, l)
    same_next = torch.cat((document[:, :-1] == document[:, 1:], torch.zeros_like(document[:, :1], dtype=torch.bool)), dim=1)  # (b, l)
    previous = torch.cat((K[:, :1], K[:, :-1]), dim=1)  # (b, l, h, d_h), token t - 1's key at t
    following = torch.cat((K[:, 1:], K[:, -1:]), dim=1)  # (b, l, h, d_h), token t + 1's key at t
    pair = torch.arange(K.shape[-1], device=K.device) % half  # (d_h,), each channel's pair
    from_previous = pair >= half - pairs  # (d_h,)
    from_next = (pair >= half - 2 * pairs) & ~from_previous  # (d_h,)
    K = torch.where(from_previous & same_previous[:, :, None, None], previous, K)  # (b, l, h, d_h)
    return torch.where(from_next & same_next[:, :, None, None], following, K)  # (b, l, h, d_h)


class Attention(nn.Module):
    def __init__(self, index: int, d: int, config: AttentionConfig, positional: Positional, positional_holder: nn.Module,
                 key_hooks: list[Strategy], query_hooks: list[Strategy], value_hooks: list[Strategy],
                 output_hooks: list[Strategy], kernel_options: dict[str, str] | None) -> None:
        super().__init__()
        self.index = index
        self.kernel = config.variant
        self.heads = config.heads  # h
        self.d_h = d // config.heads
        self.qk_width = config.qk_width(d)  # e, d_h in the record
        query_heads = 2 * config.heads if config.variant in DIFFERENTIAL else config.heads
        self.query = nn.Linear(d, query_heads * self.qk_width, bias=False)
        self.borrows_kv = config.borrows_kv(index)  # no key or value projection: the group's first block lends them
        self.lends_kv = config.kv_share > 0 and not self.borrows_kv
        self.ties_kv = bool(config.tie_kv) and self.qk_width == self.d_h  # the values are the raw keys; inert when the widths differ
        self.key: nn.Linear | None = None if self.borrows_kv else nn.Linear(d, config.heads * self.qk_width, bias=False)
        self.value: nn.Linear | None = None if self.borrows_kv or self.ties_kv else nn.Linear(d, d, bias=False)
        self.output = nn.Linear(d, d, bias=False)
        if config.variant in DIFFERENTIAL:
            self.differential_lambda = nn.Linear(d, config.heads, bias=False)  # zero-initialized by the strategy
        self.exclusive = bool(config.exclusive)
        if self.exclusive:
            self.exclusive_gate = nn.Parameter(torch.zeros(config.heads))  # (h,), so step 0's output is the record's
        self.rotary: Rotary = positional_holder.rotary  # type: ignore[assignment]
        self.rotate = positional.rotates(index)
        if positional.variant == "alibi":
            self.register_buffer("slopes", positional_holder.slopes, persistent=False)  # (k,)
        else:
            self.slopes = None
        self.positional = positional
        self.score_cap = config.score_cap
        self.split_masks = config.local_every > 0  # the training masks hold global then local halves
        self.local = config.local(index)
        self.kernel_options = kernel_options
        self.key_hooks = key_hooks
        self.query_hooks = query_hooks
        self.value_hooks = value_hooks
        self.output_hooks = output_hooks

    def forward(self, X: Tensor, context: ForwardContext) -> Tensor:
        # X: (b, l, d), the normalized stream; returns (b, l, d).
        b, l, d = X.shape
        Q = norm(self.query(X).view(b, l, -1, self.qk_width))  # (b, l, q, e)
        if self.borrows_kv:
            K, V = context.stored[SHARED_KEYS], context.stored[SHARED_VALUES]  # (b, l, h, e), (b, l, h, d_h)
        else:
            assert self.key is not None
            raw_keys = self.key(X).view(b, l, self.heads, self.qk_width)  # (b, l, h, e), raw
            K = raw_keys  # (b, l, h, e)
            for hook in self.key_hooks:
                K = hook.keys(self.index, K, context)  # (b, l, h, e)
            K = norm(K)  # (b, l, h, e)
            V = raw_keys if self.value is None else self.value(X).view(b, l, self.heads, self.d_h)  # (b, l, h, d_h)
            if self.lends_kv:  # before RoPE and the value hooks, which each block applies by its own rule
                context.stored[SHARED_KEYS], context.stored[SHARED_VALUES] = K, V
        if self.rotate:
            Q, K = self.rotary(Q), self.rotary(K)  # (b, l, q, e), (b, l, h, e)
        for hook in self.query_hooks:
            Q = hook.queries(self.index, Q, context)  # (b, l, q, e)
        for hook in self.value_hooks:
            V = hook.values(self.index, V, context)  # (b, l, h, d_h)
        if self.kernel == "paired":
            Y = self._paired(Q, K, V, context)  # (b, h, l, d_h)
            V = V.transpose(1, 2)  # (b, h, l, d_h)
        else:
            Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # (b, q, l, e), (b, h, l, e), (b, h, l, d_h)
            Y = self._triangle(Q, K, V, context) if self.kernel in TRIANGLE else self._full(Q, K, V, context)  # (b, q, l, d_h)
            if self.kernel in DIFFERENTIAL:
                lam = torch.sigmoid(self.differential_lambda(X).float()).transpose(1, 2).unsqueeze(-1)  # (b, h, l, 1)
                Y = combine_differential(Y, lam)  # (b, h, l, d_h)
        if self.exclusive:
            Y = without_own_value(Y, V, self.exclusive_gate)  # (b, h, l, d_h)
        for hook in self.output_hooks:
            Y = hook.attention_output(self.index, X, Y)  # (b, h, l, d_h)
        return self.output(Y.transpose(1, 2).reshape(b, l, d))  # (b, l, d)

    def _attend(self, Q: Tensor, K: Tensor, V: Tensor, block_mask: BlockMask | None, allowed: Tensor | None,
                pair_positions: bool = False) -> Tensor:
        """One attention call: flex with the block mask in training, SDPA with the boolean mask otherwise."""
        # Q: (b, q, n, e); K: (b, k, n, e); V: (b, k, n, v) with q a multiple of k; allowed: (b, 1, n or 1, n) bool; returns (b, q, n, v).
        group = Q.shape[1] // K.shape[1]
        if block_mask is not None:
            score_mod = self.positional.score_mod(self.slopes, group, pair_positions, self.score_cap)
            if self.kernel_options is not None:
                return flex_attention(Q, K, V, score_mod=score_mod, block_mask=block_mask, enable_gqa=group > 1,
                                      kernel_options=self.kernel_options)  # (b, q, n, e)
            return flex_attention(Q, K, V, score_mod=score_mod, block_mask=block_mask, enable_gqa=group > 1)  # (b, q, n, e)
        assert allowed is not None
        bias = self.positional.dense_bias(self.slopes, group, Q.shape[2], pair_positions, Q.dtype, Q.device)  # (1, q or 1, n, n)
        attention_mask = bias.masked_fill(~allowed, float("-inf"))  # (b, q or 1, n, n)
        if self.score_cap != 0.0:
            return capped_attention(Q, K, V, attention_mask, self.score_cap)  # (b, q, n, e)
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=attention_mask, enable_gqa=group > 1)  # (b, q, n, e)

    def _full(self, Q: Tensor, K: Tensor, V: Tensor, context: ForwardContext) -> Tensor:
        # Q: (b, q, l, d_h); K, V: (b, h, l, d_h); returns (b, q, l, d_h).
        masks = self._block_masks(context)
        return self._attend(Q, K, V, masks[0] if masks is not None else None, self._dense_mask(context, Q.shape[2]))  # (b, q, l, d_h)

    def _block_masks(self, context: ForwardContext) -> tuple[BlockMask, ...] | None:
        """This block's training masks: the global half, or the local half in a local block."""
        masks = context.block_masks
        if masks is None or not self.split_masks:
            return masks
        half = len(masks) // 2
        return masks[half:] if self.local else masks[:half]

    def _dense_mask(self, context: ForwardContext, length: int) -> Tensor | None:
        """The dense path's mask, with the local band anded in for a local block."""
        # Returns (b, 1, l or 1, l) bool, (b, 1, l, l) in a local block, or None in training on flex.
        if context.dense_mask is None or not self.local:
            return context.dense_mask  # (b, 1, l or 1, l), or None
        positions = torch.arange(length, device=context.dense_mask.device)  # (l,)
        band = (positions[:, None] - positions[None, :]).abs() < LOCAL_REACH  # (l, l)
        return context.dense_mask & band  # (b, 1, l, l)

    def _triangle(self, Q: Tensor, K: Tensor, V: Tensor, context: ForwardContext) -> Tensor:
        # Q: (b, q, l, d_h); K, V: (b, h, l, d_h); returns (b, q, l, d_h), the down and up halves side by side.
        Q_down, Q_up = split_rotation_pairs(Q)  # each (b, q, l, d_h / 2)
        K_down, K_up = split_rotation_pairs(K)  # each (b, h, l, d_h / 2)
        V_down, V_up = V.chunk(2, dim=-1)  # each (b, h, l, d_h / 2); values carry no rotation
        masks = self._block_masks(context)
        if masks is not None:
            down, up = masks
            Y_down = self._attend(Q_down, K_down, V_down, down, None)  # (b, q, l, d_h / 2)
            Y_up = self._attend(Q_up, K_up, V_up, up, None)  # (b, q, l, d_h / 2)
        else:
            l = Q.shape[2]
            dense = self._dense_mask(context, l)  # (b, 1, l or 1, l)
            assert dense is not None
            positions = torch.arange(l, device=Q.device)  # (l,)
            lower = positions[:, None] >= positions[None, :]  # (l, l)
            diagonal = positions[:, None] == positions[None, :]  # (l, l); every row keeps its own entry, padding too
            Y_down = self._attend(Q_down, K_down, V_down, None, (dense & lower) | diagonal)  # (b, q, l, d_h / 2)
            Y_up = self._attend(Q_up, K_up, V_up, None, (dense & ~lower) | diagonal)  # (b, q, l, d_h / 2)
        return torch.cat((Y_down, Y_up), dim=-1)  # (b, q, l, d_h)

    def _paired(self, Q: Tensor, K: Tensor, V: Tensor, context: ForwardContext) -> Tensor:
        # Q, K: (b, l, h, e); V: (b, l, h, d_h); returns (b, h, l, d_h).
        b, l, h, _ = Q.shape
        d_h = V.shape[-1]

        def interleave(X: Tensor) -> Tensor:
            # X: (b, l, h, w); the two heads of a pair alternate along the sequence.
            return X.view(b, l, h // 2, 2, X.shape[-1]).permute(0, 2, 1, 3, 4).reshape(b, h // 2, 2 * l, X.shape[-1])  # (b, p, 2l, w)

        masks = self._block_masks(context)
        block_mask = masks[0] if masks is not None else None
        allowed = None
        if block_mask is None:
            dense = self._dense_mask(context, l)  # (b, 1, l or 1, l)
            assert dense is not None
            allowed = dense.repeat_interleave(2, dim=-1)  # (b, 1, l or 1, 2l)
            if allowed.shape[2] > 1:
                allowed = allowed.repeat_interleave(2, dim=2)  # (b, 1, 2l, 2l)
        Y = self._attend(interleave(Q), interleave(K), interleave(V), block_mask, allowed, pair_positions=True)  # (b, p, 2l, d_h)
        return Y.view(b, h // 2, l, 2, d_h).permute(0, 1, 3, 2, 4).reshape(b, h, l, d_h)  # (b, h, l, d_h)


class AttentionKernel(Strategy):
    family = "attention"
    Config = AttentionConfig
    config: AttentionConfig

    def conflicts(self, strategies: Strategies) -> list[str]:
        hidden = strategies.shape.config.hidden
        d_h = hidden // self.config.heads
        if hidden % self.config.heads or d_h % HEAD_WIDTH_MULTIPLE or d_h > MAX_HEAD_WIDTH:
            return [f"{self.config.heads} heads on a {hidden}-wide stream give a per-head width of {hidden / self.config.heads:g}, "
                    f"not a multiple of {HEAD_WIDTH_MULTIPLE} up to {MAX_HEAD_WIDTH}"]
        return []

    def slope_heads(self) -> int:
        """How many attention heads carry a slope: the logical heads, or the pairs of paired-head attention."""
        return self.config.heads // 2 if self.variant == "paired" else self.config.heads

    def attention(self, model: CandidateModel, index: int) -> Attention:
        """Block `index`'s attention sublayer, with the hooks of the strategies that act inside it."""
        strategies = model.strategies
        return Attention(index, strategies.shape.config.hidden, self.config, strategies.positional, model.extensions["positional"],
                         strategies.hooked("keys"), strategies.hooked("queries"), strategies.hooked("values"),
                         strategies.hooked("attention_output"), strategies.kernel.kernel_options())

    def overrides(self, hook: str) -> bool:
        return hook == "keys" and self.config.key_offset > 0.0

    def keys(self, index: int, K: Tensor, context: ForwardContext) -> Tensor:
        # K: (b, l, h, d_h), raw keys.
        pairs = round(self.config.key_offset * K.shape[-1] / 2)
        return shift_key_pairs(K, context.input_ids, pairs) if pairs else K  # (b, l, h, d_h)

    def initialize(self, model: CandidateModel) -> None:
        if self.variant in DIFFERENTIAL:
            for block in model.blocks:
                nn.init.zeros_(block.attention.differential_lambda.weight)  # (h, d): every pair starts at lambda 0.5

    def parameter_groups(self, model: CandidateModel) -> list[ParameterGroup]:
        scalar_lr = model.strategies.optimizer.config.scalar_lr
        groups = []
        if self.variant in DIFFERENTIAL:
            weights = [block.attention.differential_lambda.weight for block in model.blocks]  # type: ignore[union-attr]
            groups.append(ParameterGroup("differential_lambda", weights, "adam", scalar_lr))
        if self.config.exclusive:
            gates = [block.attention.exclusive_gate for block in model.blocks]  # type: ignore[union-attr]
            groups.append(ParameterGroup("exclusive_gates", gates, "adam", scalar_lr))
        return groups

    def training_masks(self, input_ids: Tensor) -> tuple[BlockMask, ...]:
        """The flex block masks of one training batch: tokens attend within their own protein, and the kernel's own
        rule on top (directional halves for the triangle, interleaved positions for paired heads). With `local_every`,
        the same rules within LOCAL_REACH follow, one more mask per rule, built once per step."""
        # input_ids: (b, l) of packed windows.
        document = documents(input_ids)  # (b, l)
        b, l = input_ids.shape
        paired = self.variant == "paired"
        if paired:
            def same_document_paired(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
                # batch, head, query, key: () each, one attention entry; returns ().
                return document[batch, query // 2] == document[batch, key // 2]  # ()

            rules: tuple[MaskMod, ...] = (same_document_paired,)
        elif self.variant in TRIANGLE:
            def down(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
                # batch, head, query, key: () each, one attention entry; returns ().
                return (document[batch, query] == document[batch, key]) & (query >= key)  # ()

            def up(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
                # batch, head, query, key: () each, one attention entry; returns ().
                return (document[batch, query] == document[batch, key]) & (query <= key)  # ()

            rules = (down, up)
        else:
            def same_document(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
                # batch, head, query, key: () each, one attention entry; returns ().
                return document[batch, query] == document[batch, key]  # ()

            rules = (same_document,)
        if self.config.local_every:
            rules = (*rules, *(within_reach(rule, paired) for rule in rules))
        size = 2 * l if paired else l
        options = {} if self.config.block_size == DEFAULT_BLOCK_SIZE else {"BLOCK_SIZE": self.config.block_size}
        return tuple(create_block_mask(rule, b, None, size, size, device=input_ids.device, _compile=True, **options) for rule in rules)


def within_reach(rule: MaskMod, paired: bool) -> MaskMod:
    """A mask rule restricted to keys fewer than LOCAL_REACH protein positions from the query."""

    def local(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        # batch, head, query, key: () each, one attention entry; returns ().
        distance = (query // 2 - key // 2).abs() if paired else (query - key).abs()  # ()
        return rule(batch, head, query, key) & (distance < LOCAL_REACH)  # ()

    return local


def documents(input_ids: Tensor) -> Tensor:
    """Each position's document in a packed window: the count of CLS tokens up to it, and -1 for padding, so a
    protein never attends to the padding the `whole` batch leaves at a window's end (the record's windows have none)."""
    # input_ids: (b, l).
    document = (input_ids == benchmark.CLS_ID).cumsum(dim=1)  # (b, l)
    return document.masked_fill(input_ids == benchmark.PAD_ID, -1)  # (b, l)


def document_dense_mask(input_ids: Tensor) -> Tensor:
    """The document rule as a dense bool mask (b, 1, l, l), for devices without flex attention."""
    # input_ids: (b, l).
    document = documents(input_ids)  # (b, l)
    return (document[:, :, None] == document[:, None, :])[:, None]  # (b, 1, l, l)
