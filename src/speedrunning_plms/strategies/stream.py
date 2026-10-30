"""The stream family: what each block reads from the residual stream and how its update is written back.

- `single` (the record): one stream; a block reads it and its output replaces it.
- `hyper`: static hyperconnections at the block level (B5; arXiv 2409.19606) with n streams. Block i reads
  sum_s A_m[i, s] H_s, and writes H_s <- sum_t A_r[i, s, t] H_t + B[i, s] (output - input). A_m starts one-hot on
  stream i mod n, A_r at the identity and B at ones, so with every stream starting at the embedding the network
  starts as the record's. The final state is the mean of the streams.
- `dense`: dense depth connections (MUDD, B5; arXiv 2502.12170, modded-nanogpt record 81) on the residual stream
  only. Block i reads sum_j w_ij(X_i) X_j over the embedding and the `span` latest states, with w = a static row plus,
  when `dynamic`, a per-token term GELU(norm(X_i) W1_i) W2_i whose W2 starts at zero. The static row starts one-hot on
  the latest state, so the network starts as the record's. A last row reads the final state for the head.

Symbols: b windows or proteins, l tokens, d = 768, n streams, k states a dense block reads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ParameterGroup, Strategy, StrategyConfig, norm


Stream = tuple[Tensor, ...]


@dataclass(frozen=True)
class StreamConfig(StrategyConfig):
    variant: str = "single"
    streams: int = 2  # hyper: n
    span: int = 8  # dense: the latest states each block reads, beside the embedding
    dynamic: int = 1  # dense: 1 adds MUDD's per-token weights to the static ones
    VARIANTS: ClassVar[tuple[str, ...]] = ("single", "hyper", "dense")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"streams": (2, 4), "span": (1, 36), "dynamic": (0, 1)}


class StreamMixer(Strategy):
    family = "stream"
    Config = StreamConfig
    config: StreamConfig

    @property
    def active(self) -> bool:
        return False  # a role family: the skeleton calls it directly, never as a hook

    def build(self, model: nn.Module, max_tokens: int) -> None:
        blocks, d = benchmark.BLOCKS, benchmark.HIDDEN
        if self.variant == "hyper":
            n = int(self.config.streams)
            holder = nn.Module()
            holder.read = nn.Parameter(F.one_hot(torch.arange(blocks) % n, n).float())  # (36, n), A_m
            holder.mix = nn.Parameter(torch.eye(n).repeat(blocks, 1, 1))  # (36, n, n), A_r
            holder.write = nn.Parameter(torch.ones(blocks, n))  # (36, n), B
            model.extensions[self.family] = holder  # type: ignore[index]
            self.holder = holder
        elif self.variant == "dense":
            k = min(int(self.config.span), blocks)
            holder = nn.Module()
            static = torch.zeros(blocks + 1, k + 1)  # (37, k + 1): column 0 the embedding, column k the latest state
            static[:, k] = 1.0
            holder.static = nn.Parameter(static)  # (37, k + 1)
            if self.config.dynamic:
                holder.down = nn.ModuleList(nn.Linear(d, k + 1, bias=False) for _ in range(blocks + 1))  # each (k + 1, d)
                holder.up = nn.ModuleList(nn.Linear(k + 1, k + 1, bias=False) for _ in range(blocks + 1))  # each (k + 1, k + 1)
            model.extensions[self.family] = holder  # type: ignore[index]
            self.holder = holder
            self.span = k

    def initialize(self, model: nn.Module) -> None:
        if self.variant == "dense" and self.config.dynamic:
            for up in self.holder.up:
                nn.init.zeros_(up.weight)  # (k + 1, k + 1): the per-token term starts at zero

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if self.variant == "single":
            return []
        scalar_lr = model.strategies.optimizer.config.scalar_lr  # type: ignore[union-attr]
        return [ParameterGroup(f"stream_{self.variant}", list(self.holder.parameters()), "adam", scalar_lr)]

    def begin(self, X: Tensor) -> Stream:
        # X: (b, l, d), the embedding.
        if self.variant == "hyper":
            return (X,) * int(self.config.streams)  # n of (b, l, d)
        return (X,)  # one (b, l, d), or the dense history so far

    def read(self, stream: Stream, index: int) -> Tensor:
        # Returns block `index`'s input, (b, l, d).
        if self.variant == "single":
            return stream[0]  # (b, l, d)
        if self.variant == "hyper":
            weights = self.holder.read[index]  # (n,)
            return sum(weights[s].to(H.dtype) * H for s, H in enumerate(stream))  # (b, l, d)
        return self._dense(stream, index)  # (b, l, d)

    def write(self, stream: Stream, index: int, X_in: Tensor, X_out: Tensor) -> Stream:
        # X_in: (b, l, d), what block `index` read; X_out: (b, l, d), its output.
        if self.variant == "single":
            return (X_out,)
        if self.variant == "dense":
            return (*stream, X_out)  # 2 + index of (b, l, d)
        delta = X_out - X_in  # (b, l, d)
        mix, write = self.holder.mix[index], self.holder.write[index]  # (n, n), (n,)
        return tuple(sum(mix[s, t].to(H.dtype) * H for t, H in enumerate(stream)) + write[s].to(delta.dtype) * delta
                     for s in range(len(stream)))  # n of (b, l, d)

    def end(self, stream: Stream) -> Tensor:
        # Returns the state the head reads, (b, l, d).
        if self.variant == "dense":
            return self._dense(stream, benchmark.BLOCKS)  # (b, l, d)
        return self.view(stream)  # (b, l, d)

    def view(self, stream: Stream) -> Tensor:
        """The one state a probe or a skip reads after a block: the stream, the streams' mean, or the latest state."""
        if self.variant == "hyper":
            return sum(stream) / len(stream)  # (b, l, d)
        return stream[-1]  # (b, l, d)

    def _dense(self, history: Stream, row: int) -> Tensor:
        # history: the embedding and every block output so far, each (b, l, d); returns (b, l, d).
        k = self.span
        recent = history[max(1, len(history) - k):]  # up to k of (b, l, d)
        sources = (history[0], *recent) if len(history) > 1 else history  # 1 + up to k of (b, l, d)
        columns = [0, *range(k + 1 - len(recent), k + 1)] if len(history) > 1 else [k]  # the static row's columns
        weights = self.holder.static[row, columns]  # (s,), s sources
        X_latest = history[-1]  # (b, l, d)
        if self.config.dynamic:
            hidden = F.gelu(self.holder.down[row](norm(X_latest)))  # (b, l, k + 1)
            per_token = self.holder.up[row](hidden)[..., columns]  # (b, l, s)
            return sum((weights[j] + per_token[..., j : j + 1]).to(X.dtype) * X for j, X in enumerate(sources))  # (b, l, d)
        return sum(weights[j].to(X.dtype) * X for j, X in enumerate(sources))  # (b, l, d)
