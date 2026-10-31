"""The neighbor family: zero-initialized tables over a position's neighboring tokens (attention is bidirectional, so a
context table needs both sides; 010, A7).

`pair_table`: a table over the (left, right) token pair added to the embedding. 010 on the 003 incumbent was level with
it to 270 s and behind after (val_select 2.7583 against 2.7559).
`bigram_table` and `trigram_table` (`ideas/2026-10-30_capacity_at_fixed_shape.md`, idea 2; arXiv 2601.07372 and
modded-nanogpt records 62 and 83): the table enters the last block's output, the state the head reads, scaled by a
learned sigmoid gate, at every position (the scored ones are where it matters). The bigram is exact over the (left,
right) pair, 1,089 rows; the trigram is over (left2, left, right, right2), whose 33^4 codes are hashed to 2^16 rows by
Knuth's multiplicative hash (the top 16 of the code times 2654435761's low 32 bits). The table starts at zero and the
gate's logit at 0, so step 0 is the record's and the gate starts at 0.5. The table trains with AdamW at the embedding
rate, the gate at the scalar rate.

Symbols: b windows or proteins, l tokens, d = 768, v = 33 tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor, nn

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import ForwardContext, ParameterGroup, Strategy, StrategyConfig


TRIGRAM_ROWS = 2**16
KNUTH_MULTIPLIER = 2654435761  # floor(2^32 / golden ratio), odd
TABLE_ROWS = {"pair_table": benchmark.VOCAB_SIZE**2, "bigram_table": benchmark.VOCAB_SIZE**2, "trigram_table": TRIGRAM_ROWS}


@dataclass(frozen=True)
class NeighborConfig(StrategyConfig):
    variant: str = "none"
    VARIANTS: ClassVar[tuple[str, ...]] = ("none", "pair_table", "bigram_table", "trigram_table")


def shifted(input_ids: Tensor, offset: int) -> Tensor:
    """The token `offset` positions away (negative to the left), PAD past the window's ends."""
    # input_ids: (b, l); returns (b, l).
    if offset < 0:
        return F.pad(input_ids[:, :offset], (-offset, 0), value=benchmark.PAD_ID)  # (b, l)
    return F.pad(input_ids[:, offset:], (0, offset), value=benchmark.PAD_ID)  # (b, l)


def pair_index(input_ids: Tensor) -> Tensor:
    """The exact (left, right) pair's row."""
    # input_ids: (b, l); returns (b, l) in [0, v^2).
    return shifted(input_ids, -1) * benchmark.VOCAB_SIZE + shifted(input_ids, 1)  # (b, l)


def trigram_index(input_ids: Tensor) -> Tensor:
    """The (left2, left, right, right2) code's hashed row."""
    # input_ids: (b, l); returns (b, l) in [0, 2^16).
    code = torch.zeros_like(input_ids)  # (b, l)
    for offset in (-2, -1, 1, 2):
        code = code * benchmark.VOCAB_SIZE + shifted(input_ids, offset)  # (b, l), below 33^4
    return ((code * KNUTH_MULTIPLIER) & 0xFFFFFFFF) >> 16  # (b, l)


class GatedTable(nn.Module):
    """A context table and the logit of the sigmoid gate it enters through. The table is built from zeros, so it draws
    nothing from the generator and the record's weights initialize as they do without it."""

    def __init__(self, rows: int) -> None:
        super().__init__()
        self.table = nn.Embedding(rows, benchmark.HIDDEN, _weight=torch.zeros(rows, benchmark.HIDDEN))  # (rows, d)
        self.gate = nn.Parameter(torch.zeros(()))  # ()


class Neighbor(Strategy):
    family = "neighbor"
    Config = NeighborConfig

    @property
    def gated(self) -> bool:
        return self.variant in ("bigram_table", "trigram_table")

    def overrides(self, hook: str) -> bool:
        if hook == "embed":
            return self.variant == "pair_table"
        return hook == "updated_stream" and self.gated

    def build(self, model: nn.Module, max_tokens: int) -> None:
        if self.variant == "pair_table":
            self.table = nn.Embedding(TABLE_ROWS[self.variant], benchmark.HIDDEN)  # (v * v, d)
            model.extensions[self.family] = self.table  # type: ignore[index]
        elif self.gated:
            self.gated_table = GatedTable(TABLE_ROWS[self.variant])
            self.table = self.gated_table.table
            model.extensions[self.family] = self.gated_table  # type: ignore[index]

    def initialize(self, model: nn.Module) -> None:
        if self.active:
            nn.init.zeros_(self.table.weight)  # (rows, d): the table starts silent
        if self.gated:
            nn.init.zeros_(self.gated_table.gate)  # (): sigmoid 0.5

    def parameter_groups(self, model: nn.Module) -> list[ParameterGroup]:
        if not self.active:
            return []
        config = model.strategies.optimizer.config  # type: ignore[union-attr]
        groups = [ParameterGroup("neighbor_table", [self.table.weight], "adam", config.embed_lr)]
        if self.gated:
            groups.append(ParameterGroup("neighbor_gate", [self.gated_table.gate], "adam", config.scalar_lr))
        return groups

    def embed(self, model: nn.Module, input_ids: Tensor, X: Tensor) -> Tensor:
        # input_ids: (b, l); X: (b, l, d).
        return X + self.table(pair_index(input_ids))  # (b, l, d)

    def updated_stream(self, index: int, sublayer: str, X: Tensor, context: ForwardContext) -> Tensor:
        # X: (b, l, d), the stream after block `index`'s sublayer; the table enters after the last block's MLP.
        if index != benchmark.BLOCKS - 1 or sublayer != "mlp":
            return X  # (b, l, d)
        rows = pair_index(context.input_ids) if self.variant == "bigram_table" else trigram_index(context.input_ids)  # (b, l)
        return X + torch.sigmoid(self.gated_table.gate) * self.table(rows).to(X.dtype)  # (b, l, d)
