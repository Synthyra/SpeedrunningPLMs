"""The positional family: rotary positions on queries and keys, and a fixed per-head distance prior on the scores.

`alibi` (the record, 012) subtracts slope_i |q - k| from every score, with ALiBi's geometric slopes 2^(-8 i / h)
times `slope_scale` (025 tried 2). `none` keeps RoPE alone (000 to 011). A learned slope inside the flex score
function made a step a hundred times slower (021), so the slopes stay fixed buffers.

Two numbers shape RoPE under either variant. `rope_fraction` (partial RoPE, arXiv 2603.11611) rotates only the first
k = round(fraction d_h / 2) of the d_h / 2 rotation pairs, with GPT-NeoX's `rotary_pct` spectrum: the k pairs take
frequencies base^(-2i / 2k), so they span the whole range, and the other pairs get frequency 0, which leaves them
unrotated (cos 1, sin 0) and the compiled graph unchanged; at 1.0 the table is the record's. `nope_every` (RNoPE,
arXiv 2501.18795; Gemma 3) skips RoPE in every k-th block (blocks k - 1, 2k - 1, ...); those blocks keep the slopes.
Both RoPE and the slopes depend on q - k only, so a protein's scores do not depend on where the window places it.

`periodic_scale` (`ideas/2026-10-30_protein_input_priors.md` 1) adds periodic_scale (cos(2 pi d / 3.6) + cos(pi d)) / 2
to every score, d = |q - k|: the alpha helix's 3.6-residue turn and the strand's 2-residue alternation, which no RoPE
pair can express (its shortest period is 2 pi residues) and the monotone slopes cannot either. The scale is a Python
constant inside the score function, as the slopes' rule is, never a captured parameter (021). The attention family's
`score_cap` enters the same function: cap tanh(s / cap) on the scaled q.k before the distance terms, so far tokens
keep their full penalty. At periodic_scale 0 and no cap the function is the record's.

`length_bins` (`ideas/2026-10-30_protein_input_priors.md` 2; 0 is off, else 8 or 16) adds to every token's embedding a
learned vector for its protein's log-length bin, through `embed`: the protein's non-padding tokens in the window,
binned over log2 from 32 to 2,048 tokens. A protein the window cuts counts its visible part, which is what attention
can reach; the evaluator's proteins are whole. The table starts at zero and is built from zeros, so step 0 and the
record's initialization are unchanged; AdamW at the embedding rate. The idea's terminus-distance tables are left out.

`terminus` (the same idea's terminus tokens; 0 is off, 1 on): a learned N-terminus vector added to the embedding of a
protein's first residue (the token after CLS) and a C-terminus vector to its last (the token before EOS), whatever
the token there is, MASK included; a protein the window cuts gets a marker only at the end the window holds. Both
vectors are built from zeros, AdamW at the embedding rate. Attention already reaches CLS and EOS at distance 1, so the
prior is cheap and may be redundant.

Symbols: b windows or proteins, l tokens, h heads, d_h per-head width, k attention heads a kernel runs.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ParameterGroup, Strategy, StrategyConfig, extension


ScoreMod = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
LENGTH_OCTAVES = (5.0, 11.0)  # the length bins span log2 of 32 to 2,048 tokens
HELIX_FREQUENCY = 2 * math.pi / 3.6  # radians per residue: the alpha helix's 3.6 residues a turn
STRAND_FREQUENCY = math.pi  # the beta strand's side-chain alternation, every 2 residues


@dataclass(frozen=True)
class PositionalConfig(StrategyConfig):
    variant: str = "alibi"
    rope_base: float = 10_000.0
    slope_scale: float = 1.0  # multiplies ALiBi's geometric slopes
    rope_fraction: float = 1.0  # share of each head's rotation pairs RoPE turns
    nope_every: int = 0  # every k-th block skips RoPE; 0 rotates every block
    periodic_scale: float = 0.0  # the helix and strand term's weight on the scores; 0 is off
    length_bins: int = 0  # log-length bins of the protein added to its embeddings; 0 is off
    terminus: int = 0  # 1 adds learned N- and C-terminus vectors at a protein's first and last residues
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "alibi")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"rope_base": (1_000.0, 1_000_000.0), "slope_scale": (0.25, 4.0),
                                                        "rope_fraction": (0.1, 1.0), "nope_every": (0, 6),
                                                        "periodic_scale": (0.0, 1.0), "length_bins": (0, 16), "terminus": (0, 1)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.nope_every, int):
            raise ValueError(f"nope_every must be an integer, not {self.nope_every!r}")
        if self.length_bins not in (0, 8, 16):
            raise ValueError(f"length_bins is 0 (off), 8 or 16, not {self.length_bins!r}")
        if self.terminus not in (0, 1):
            raise ValueError(f"terminus is 0 (off) or 1, not {self.terminus!r}")


def terminus_positions(input_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Each protein's first residue (the token after CLS) and last residue (the token before EOS)."""
    # input_ids: (b, l); returns two (b, l) bool.
    special = (input_ids == benchmark.CLS_ID) | (input_ids == benchmark.EOS_ID) | (input_ids == benchmark.PAD_ID)  # (b, l)
    after_cls = F.pad(input_ids[:, :-1] == benchmark.CLS_ID, (1, 0), value=False)  # (b, l)
    before_eos = F.pad(input_ids[:, 1:] == benchmark.EOS_ID, (0, 1), value=False)  # (b, l)
    return after_cls & ~special, before_eos & ~special  # (b, l) each


def length_bins(input_ids: Tensor, bins: int) -> Tensor:
    """Each token's protein log-length bin: the protein's non-padding tokens in the window, over LENGTH_OCTAVES."""
    # input_ids: (b, l); returns (b, l) in [0, bins).
    b, l = input_ids.shape
    document = (input_ids == benchmark.CLS_ID).cumsum(dim=1)  # (b, l) in [0, l]
    real = (input_ids != benchmark.PAD_ID).float()  # (b, l)
    counts = torch.zeros(b, l + 1, device=input_ids.device).scatter_add_(1, document, real)  # (b, l + 1)
    tokens = counts.gather(1, document).clamp(min=1)  # (b, l)
    low, high = LENGTH_OCTAVES
    return ((tokens.log2() - low) * (bins / (high - low))).floor().clamp(0, bins - 1).long()  # (b, l)


def rotated_pairs(d_h: int, fraction: float) -> int:
    """How many of a head's d_h / 2 rotation pairs RoPE turns: at least one."""
    return max(1, round(fraction * d_h / 2))


class Rotary(nn.Module):
    def __init__(self, d_h: int, base: float, max_tokens: int, fraction: float = 1.0) -> None:
        super().__init__()
        k = rotated_pairs(d_h, fraction)
        rotated = base ** (-torch.arange(0, 2 * k, 2).float() / (2 * k))  # (k,); the record's spectrum at k = d_h / 2
        inverse_frequency = torch.cat((rotated, torch.zeros(d_h // 2 - k)))  # (d_h / 2,); frequency 0 never rotates
        angles = torch.outer(torch.arange(max_tokens).float(), inverse_frequency)  # (max_tokens, d_h / 2)
        self.register_buffer("cos", angles.cos(), persistent=False)  # (max_tokens, d_h / 2)
        self.register_buffer("sin", angles.sin(), persistent=False)  # (max_tokens, d_h / 2)

    def forward(self, X: Tensor) -> Tensor:
        # X: (b, l, h, d_h); rotate the two halves by position.
        l = X.shape[1]
        cos, sin = self.cos[:l, None, :], self.sin[:l, None, :]  # each (l, 1, d_h / 2)
        first, second = X.float().chunk(2, dim=-1)  # each (b, l, h, d_h / 2)
        rotated = torch.cat((first * cos + second * sin, second * cos - first * sin), dim=-1)  # (b, l, h, d_h)
        return rotated.type_as(X)  # (b, l, h, d_h)


def geometric_slopes(heads: int, scale: float) -> Tensor:
    """ALiBi's slopes 2^(-8 i / h) for heads i = 1..h, times `scale`."""
    slopes = 2.0 ** (-8.0 * torch.arange(1, heads + 1) / heads)  # (h,)
    return slopes if scale == 1.0 else scale * slopes  # (h,); the record's arithmetic exactly at scale 1


class Positional(Strategy):
    family = "positional"
    Config = PositionalConfig
    config: PositionalConfig

    def build(self, model: nn.Module, max_tokens: int) -> None:
        attention = model.strategies.attention  # type: ignore[union-attr]
        holder = nn.Module()
        holder.rotary = Rotary(benchmark.HIDDEN // attention.config.heads, self.config.rope_base, max_tokens,
                               self.config.rope_fraction)
        if self.variant == "alibi":
            holder.register_buffer("slopes", geometric_slopes(attention.slope_heads(), self.config.slope_scale),
                                   persistent=False)  # (k,), one slope per attention head the kernel runs
        bins = self.config.length_bins
        if bins:
            holder.length_table = nn.Embedding(bins, benchmark.HIDDEN, _weight=torch.zeros(bins, benchmark.HIDDEN))  # (bins, d)
        if self.config.terminus:
            holder.terminus = nn.Parameter(torch.zeros(2, benchmark.HIDDEN))  # (2, d): the N- and C-terminus vectors
        model.extensions[self.family] = holder  # type: ignore[index]

    def overrides(self, hook: str) -> bool:
        return hook == "embed" and (self.config.length_bins > 0 or self.config.terminus > 0)

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        holder = extension(model, self.family)
        tables = ([holder.length_table.weight] if self.config.length_bins else []) + ([holder.terminus] if self.config.terminus else [])
        if not tables:
            return []
        return [ParameterGroup("input_priors", tables, "adam", model.strategies.optimizer.config.embed_lr)]  # type: ignore[union-attr]

    def embed(self, model: nn.Module, input_ids: Tensor, X: Tensor) -> Tensor:
        # input_ids: (b, l); X: (b, l, d).
        holder = extension(model, self.family)
        if self.config.length_bins:
            X = X + holder.length_table(length_bins(input_ids, self.config.length_bins))  # (b, l, d)
        if self.config.terminus:
            first, last = terminus_positions(input_ids)  # (b, l) each
            N, C = holder.terminus.to(X.dtype)  # (d,) each
            X = X + first[..., None].to(X.dtype) * N + last[..., None].to(X.dtype) * C  # (b, l, d)
        return X  # (b, l, d)

    def rotary(self, model: nn.Module) -> Rotary:
        return extension(model, self.family).rotary  # type: ignore[return-value]

    def rotates(self, index: int) -> bool:
        """False for the blocks `nope_every` leaves without RoPE."""
        return self.config.nope_every == 0 or (index + 1) % self.config.nope_every != 0

    def slopes(self, model: nn.Module) -> Tensor | None:
        """The fixed slopes, one per attention head the kernel runs, or None without a distance prior."""
        return extension(model, self.family).slopes if self.variant == "alibi" else None  # (k,) or None

    def score_mod(self, slopes: Tensor | None, group: int, pair_positions: bool, cap: float = 0.0) -> ScoreMod | None:
        """The flex score modifier: head i's slope is `slopes[i // group]` (query heads sharing a slope), and
        `pair_positions` halves both indices for paired-head attention, whose sequence interleaves two heads. `cap`
        (0 is off) soft-caps the scaled q.k before the distance terms."""
        # slopes: (k,) or None.
        periodic = self.config.periodic_scale
        if slopes is None and periodic == 0.0 and cap == 0.0:
            return None

        if slopes is not None and group == 1 and not pair_positions and periodic == 0.0 and cap == 0.0:
            def local_bias(score: Tensor, batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
                # score, batch, head, query, key: () each, one attention entry; returns ().
                return score - slopes[head] * (query - key).abs()  # ()

            return local_bias

        def grouped_bias(score: Tensor, batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
            # score, batch, head, query, key: () each; returns ().
            distance = (query // 2 - key // 2).abs() if pair_positions else (query - key).abs()  # ()
            if cap != 0.0:
                score = cap * torch.tanh(score / cap)  # ()
            if slopes is not None:
                score = score - slopes[head // group] * distance  # ()
            if periodic != 0.0:
                score = score + periodic * periodic_term(distance)  # ()
            return score  # ()

        return grouped_bias

    def dense_bias(self, slopes: Tensor | None, group: int, length: int, pair_positions: bool, dtype: torch.dtype,
                   device: torch.device) -> Tensor:
        """The same prior as an additive bias for SDPA: (1, k * group, length, length), or (1, 1, length, length) of
        zeros without a prior."""
        # slopes: (k,) or None.
        positions = torch.arange(length, device=device)  # (length,)
        if pair_positions:
            positions = positions // 2  # (length,), the protein position of each interleaved entry
        residues = (positions[:, None] - positions[None, :]).abs()  # (length, length), integer
        distance = residues.to(dtype)  # (length, length)
        periodic = self.config.periodic_scale
        if slopes is None and periodic == 0.0:
            return torch.zeros(1, 1, length, length, dtype=dtype, device=device)  # (1, 1, length, length)
        if slopes is None:
            return (periodic * periodic_term(residues)).to(dtype)[None, None]  # (1, 1, length, length)
        per_head = slopes.to(dtype).repeat_interleave(group) if group > 1 else slopes.to(dtype)  # (k * group,)
        bias = -per_head.view(1, -1, 1, 1) * distance  # (1, k * group, length, length)
        if periodic != 0.0:
            bias = bias + (periodic * periodic_term(residues)).to(dtype)  # (1, k * group, length, length), the cosine in fp32
        return bias  # (1, k * group, length, length)


def periodic_term(distance: Tensor) -> Tensor:
    """(cos(2 pi d / 3.6) + cos(pi d)) / 2: one at the helix's 3.6-residue turn and at even spacings, in [-1, 1]."""
    # distance: (...), |q - k| in residues as integers; returns (...) in fp32, so long distances keep their phase.
    d = distance.float()  # (...)
    return (torch.cos(d * HELIX_FREQUENCY) + torch.cos(d * STRAND_FREQUENCY)) / 2  # (...)
