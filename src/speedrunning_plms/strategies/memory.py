"""The memory family: a product-key memory side path on the stream (Lample et al., "Large Memory Layers with Product
Keys," arXiv 1907.05242; Berges et al., "Memory Layers at Scale," arXiv 2412.09764;
`ideas/2026-10-30_capacity_at_fixed_shape.md` idea 4).

`product_keys`: at the input of block `block`, the stream gains the memory's read of its normalized state, X + m(norm X),
so the read joins the residual sum the previous block's MLP update ended, and the block reads it. A query projection
(768 to 768) splits into two halves of 384; each half scores SIDE sub-keys and keeps its top TOP; the TOP x TOP sums
of a pair's scores address the SIDE^2 = 65,536 value rows; the best TOP of those, softmax-weighted, are summed by one
`embedding_bag`, so the (tokens, TOP, 768) gather is never materialized. The values start at zero, so step 0 is the
record's forward; the sub-keys and the query start from normal draws taken after the record's initialization (built
empty, so the record's weights draw what they draw without it). The query is a plain parameter, since an `nn.Linear`
draws its default initialization when built. Everything trains with AdamW at the embedding rate:
50.3M value parameters, 0.6M in the query and 0.2M in the keys, outside the 36 blocks, which the shape rule leaves free
(tables). Per step at 65,536 tokens: 2.1M value rows read, 6.4 GB in fp32, about 2 ms at GH200 bandwidth, and the
backward's scatter into a dense 200 MB gradient.

`shared_pool` (Memory+ in Berges et al., sections 3.1.3 and 5.4 by a fetch summary: memory layers sharing one pool,
about three of them best; `ideas/2026-10-31_capacity_and_data_at_one_hour.md` idea 1): the same read at the inputs of
blocks SHARED_READS (9, 18, 27), each with its own query and sub-keys, all reading one 65,536-row value pool (`block`
is inert). Each extra read adds 0.8M parameters and the same row traffic again, three times the single read's.

Symbols: b windows, l tokens, d = 768, n = b l tokens, s = SIDE sub-keys a half, t = TOP.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ParameterGroup, Strategy, StrategyConfig, norm


SIDE = 256  # sub-keys a half; SIDE^2 value rows
TOP = 32  # sub-keys kept a half, and rows read a token
KEY_INIT_STD = 0.02
SHARED_READS = (9, 18, 27)  # shared_pool: the blocks whose inputs read the one pool


@dataclass(frozen=True)
class MemoryConfig(StrategyConfig):
    variant: str = "none"
    block: int = 18  # product_keys: the block whose input gains the read
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "product_keys", "shared_pool")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"block": (1, 35)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.block, int):
            raise ValueError(f"block must be an integer, not {self.block!r}")


class ProductKeyMemory(nn.Module):
    def __init__(self, reads: int = 1) -> None:
        # reads: 1 keeps the single read's (d, d) query and (2, s, d / 2) keys; more stacks one of each per read.
        super().__init__()
        d = benchmark.HIDDEN
        lead = () if reads == 1 else (reads,)
        self.query = nn.Parameter(torch.empty(*lead, d, d))  # (d, d) or (r, d, d), drawn in `initialize`
        self.keys = nn.Parameter(torch.empty(*lead, 2, SIDE, d // 2))  # (2, s, d / 2) or (r, 2, s, d / 2), likewise
        self.values = nn.Parameter(torch.zeros(SIDE * SIDE, d))  # (s^2, d), the one pool

    def forward(self, X: Tensor, read: int = 0) -> Tensor:
        # X: (b, l, d), the normalized stream; read: which query and keys; returns (b, l, d) in fp32.
        b, l, d = X.shape
        query, keys = (self.query, self.keys) if self.query.ndim == 2 else (self.query[read], self.keys[read])  # (d, d), (2, s, d / 2)
        Q = F.linear(X, query).view(b * l, 2, d // 2)  # (n, 2, d / 2)
        scores = torch.einsum("nhe,hse->nhs", Q, keys.to(Q.dtype)).float()  # (n, 2, s)
        top, index = scores.topk(TOP, dim=-1)  # (n, 2, t) each
        pair = (top[:, 0, :, None] + top[:, 1, None, :]).flatten(1)  # (n, t^2)
        rows = (index[:, 0, :, None] * SIDE + index[:, 1, None, :]).flatten(1)  # (n, t^2)
        best, chosen = pair.topk(TOP, dim=-1)  # (n, t) each
        weights = torch.softmax(best, dim=-1)  # (n, t), fp32
        read = F.embedding_bag(rows.gather(1, chosen), self.values, per_sample_weights=weights.to(self.values.dtype),
                               mode="sum")  # (n, d)
        return read.view(b, l, d)  # (b, l, d)


class Memory(Strategy):
    family = "memory"
    Config = MemoryConfig
    config: MemoryConfig

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.active:
            self.memory = ProductKeyMemory(len(SHARED_READS) if self.variant == "shared_pool" else 1)
            model.extensions[self.family] = self.memory  # type: ignore[index]

    def initialize(self, model: nn.Module) -> None:
        if self.active:
            nn.init.normal_(self.memory.query, std=KEY_INIT_STD)  # (d, d) or (r, d, d), the record's linear init
            nn.init.normal_(self.memory.keys, std=KEY_INIT_STD)  # (2, s, d / 2) or (r, 2, s, d / 2)
            nn.init.zeros_(self.memory.values)  # (s^2, d): the read starts at zero

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if not self.active:
            return []
        embed_lr = model.strategies.optimizer.config.embed_lr  # type: ignore[union-attr]
        return [ParameterGroup("product_key_memory", list(self.memory.parameters()), "adam", embed_lr)]

    def block_input(self, model: nn.Module, index: int, X: Tensor, states: list[Tensor]) -> Tensor:
        # X: (b, l, d), the stream entering block `index`.
        if self.variant == "shared_pool":
            if index not in SHARED_READS:
                return X  # (b, l, d)
            return X + self.memory(norm(X), SHARED_READS.index(index)).to(X.dtype)  # (b, l, d)
        if index != self.config.block:
            return X  # (b, l, d)
        return X + self.memory(norm(X)).to(X.dtype)  # (b, l, d)
