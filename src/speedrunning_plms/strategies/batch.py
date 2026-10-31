"""The batch family: windows of the training stream, how long each is, how many make a step, and how proteins fill them.

The record trains on 64 windows of 1,024 tokens, 65,536 tokens a step. 008 tried 32 windows (1.8x the steps, behind
from 180 s) and 009 512-token windows at 128 a step (6 percent more steps, long proteins lose). A genome that leaves
`windows_per_step` at 0 keeps 65,536 tokens a step at any window (32 windows of 2,048); the search moves only the
window, and a hand-written genome may set the count.

- `packed` (the record): consecutive windows of the stream, so a protein that crosses a window's edge is cut there
  and its two parts train as separate documents. At 1,024 or 2,048 tokens the cut is the same; a longer window cuts
  fewer proteins (the longest protein is 2,048 tokens with CLS and EOS, so at 2,048 only the edges cut).
- `whole`: best-fit decreasing packing (Ding et al. 2024, arXiv 2404.10830). The proteins, in reading order, go in
  buffers of 4,096; each buffer is packed into windows by best fit over its pieces sorted longest first, and the
  windows of a buffer are shuffled. Only a protein longer than the window is cut, into window-long pieces; a piece
  after the first has no CLS, so it opens a window of its own and stays a separate document. The rest of a window is
  PAD, which the document mask keeps apart (`attention.documents`) and the corruption never labels, so a step holds
  fewer real tokens than `packed` by the padding share. The packing runs once on the CPU before the warmup.

A curriculum's protein order (`curriculum.protein_order`) changes which proteins come first under either variant.

Symbols: n proteins, t stream tokens, p pieces, s = window x windows_per_step tokens a step.
"""

from __future__ import annotations

import bisect
import numpy as np

from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

from speedrunning_plms import benchmark
from speedrunning_plms.strategies.strategy import Strategy, StrategyConfig


PACKING_BUFFER = 4096  # proteins a best-fit pass sees at once: the reading order holds beyond it
STEP_TOKENS = 65_536  # the record's tokens a step, which the window's count keeps unless a genome sets it


@dataclass(frozen=True)
class BatchConfig(StrategyConfig):
    variant: str = "packed"
    window: int = 1024  # tokens a window
    windows_per_step: int = 0  # 0 ties the count to the window, STEP_TOKENS // window: 64 at 1,024, 32 at 2,048
    VARIANTS: ClassVar[tuple[str, ...]] = ("packed", "whole")
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"window": (32, 2048), "windows_per_step": (0, 256)}

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.window & (self.window - 1) or self.windows_per_step & (self.windows_per_step - 1):
            raise ValueError(f"window {self.window} and windows_per_step {self.windows_per_step} must be powers of two")

    @property
    def count(self) -> int:
        """Windows a step: the genome's, or STEP_TOKENS // window when it leaves the count tied."""
        return self.windows_per_step or STEP_TOKENS // self.window


@dataclass(frozen=True)
class Packing:
    """Whole proteins in windows: each piece's place in the stream and in its window, grouped by window."""

    sources: np.ndarray  # (p,) int64, a piece's first token in the stream
    lengths: np.ndarray  # (p,) int64, its tokens
    placements: np.ndarray  # (p,) int64, its first position in its window
    window_starts: np.ndarray  # (w + 1,) int64, w windows; window j holds pieces window_starts[j] to window_starts[j + 1]
    window: int

    @property
    def windows(self) -> int:
        return len(self.window_starts) - 1

    @property
    def padding_share(self) -> float:
        """The share of the windows' tokens that are padding."""
        return 1.0 - float(self.lengths.sum()) / (self.windows * self.window)


def best_fit(offsets: np.ndarray, order: np.ndarray, window: int, rng: np.random.Generator) -> Packing:
    """Best-fit decreasing over buffers of PACKING_BUFFER proteins in `order`; each buffer's windows are shuffled."""
    # offsets: (n + 1,) the proteins' stream offsets; order: (n,) the reading order.
    sizes = np.diff(offsets)  # (n,) tokens of each protein, CLS and EOS included
    sources: list[int] = []
    lengths: list[int] = []
    placements: list[int] = []
    window_starts = [0]
    for first in range(0, len(order), PACKING_BUFFER):
        items: list[tuple[int, int, bool]] = []  # (length, source, opens a window)
        for protein in order[first : first + PACKING_BUFFER].tolist():
            start, size = int(offsets[protein]), int(sizes[protein])
            for cut in range(0, size, window):
                items.append((min(window, size - cut), start + cut, cut > 0))
        items.sort(key=lambda item: -item[0])
        contents: list[list[tuple[int, int, int]]] = []  # each window's (source, length, placement)
        fill: list[int] = []
        open_by_room: dict[int, list[int]] = {}  # windows by their free tokens
        rooms: list[int] = []  # the free-token counts some open window has, sorted
        for length, source, opens in items:
            position = len(rooms) if opens else bisect.bisect_left(rooms, length)
            if position == len(rooms):
                chosen = len(contents)
                contents.append([])
                fill.append(0)
            else:
                room = rooms[position]
                chosen = open_by_room[room].pop()
                if not open_by_room[room]:
                    del open_by_room[room]
                    rooms.pop(position)
            contents[chosen].append((source, length, fill[chosen]))
            fill[chosen] += length
            room = window - fill[chosen]
            if room > 0:
                if room not in open_by_room:
                    open_by_room[room] = []
                    bisect.insort(rooms, room)
                open_by_room[room].append(chosen)
        for chosen in rng.permutation(len(contents)).tolist():
            for source, length, placement in contents[chosen]:
                sources.append(source)
                lengths.append(length)
                placements.append(placement)
            window_starts.append(len(sources))
    return Packing(np.asarray(sources, dtype=np.int64), np.asarray(lengths, dtype=np.int64),
                   np.asarray(placements, dtype=np.int64), np.asarray(window_starts, dtype=np.int64), window)


def contiguous_steps(tokens: np.ndarray, step_tokens: int, seed: int) -> Iterator[np.ndarray]:
    """The record's reader: consecutive blocks of the stream from a seed's offset, wrapping at the end."""
    # tokens: (t,) uint8; each yield (s,) uint8.
    total = (len(tokens) // step_tokens) * step_tokens
    start = (seed * step_tokens) % total
    while True:
        chunk = np.asarray(tokens[start : start + step_tokens])  # (s,)
        yield chunk.copy()  # (s,)
        start = (start + step_tokens) % total


def ordered_steps(tokens: np.ndarray, offsets: np.ndarray, order: np.ndarray, step_tokens: int) -> Iterator[np.ndarray]:
    """The stream's proteins concatenated in `order`, read as consecutive blocks, wrapping at the end."""
    # tokens: (t,) uint8; offsets: (n + 1,); order: (n,); each yield (s,) uint8.
    sizes = np.diff(offsets)[order]  # (n,) in reading order
    ends = np.cumsum(sizes)  # (n,) each protein's end in the reordered stream
    total = (int(ends[-1]) // step_tokens) * step_tokens
    start = 0
    while True:
        positions = np.arange(start, start + step_tokens)  # (s,) in the reordered stream
        protein = np.searchsorted(ends, positions, side="right")  # (s,) index into `order`
        sources = offsets[order[protein]] + positions - (ends[protein] - sizes[protein])  # (s,) in the stream
        yield np.asarray(tokens[sources], dtype=np.uint8)  # (s,)
        start = (start + step_tokens) % total


def packed_steps(tokens: np.ndarray, packing: Packing, windows_per_step: int, seed: int) -> Iterator[np.ndarray]:
    """Consecutive groups of packed windows from a seed's offset, PAD after each window's pieces, wrapping at the end."""
    # tokens: (t,) uint8; each yield (s,) uint8.
    steps = packing.windows // windows_per_step
    step = seed % steps
    while True:
        first_window = step * windows_per_step
        bounds = packing.window_starts[first_window : first_window + windows_per_step + 1]  # (windows_per_step + 1,)
        pieces = slice(int(bounds[0]), int(bounds[-1]))
        lengths = packing.lengths[pieces]  # (m,), m pieces this step
        slot = np.repeat(np.arange(windows_per_step), np.diff(bounds))  # (m,) each piece's window in the step
        destinations = slot * packing.window + packing.placements[pieces]  # (m,)
        within = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)  # (r,) r real tokens
        chunk = np.full(windows_per_step * packing.window, benchmark.PAD_ID, dtype=np.uint8)  # (s,)
        chunk[np.repeat(destinations, lengths) + within] = tokens[np.repeat(packing.sources[pieces], lengths) + within]
        yield chunk  # (s,)
        step = (step + 1) % steps


class Batch(Strategy):
    family = "batch"
    Config = BatchConfig
    config: BatchConfig

    @property
    def step_tokens(self) -> int:
        return self.config.window * self.config.count

    def packing(self, offsets: np.ndarray, order: np.ndarray | None, seed: int) -> Packing:
        """The `whole` variant's windows, in the curriculum's order or the stream's."""
        # offsets: (n + 1,); order: (n,) or None.
        reading = np.arange(len(offsets) - 1) if order is None else order  # (n,)
        return best_fit(offsets, reading, self.config.window, np.random.default_rng(seed))

    def steps(self, tokens: np.ndarray, offsets: np.ndarray, order: np.ndarray | None, packing: Packing | None,
              seed: int) -> Iterator[np.ndarray]:
        """Each step's tokens as one flat block, (s,) uint8, in windows of `window` tokens."""
        # tokens: (t,) uint8; offsets: (n + 1,); order: (n,) or None.
        if self.variant == "whole":
            assert packing is not None and packing.window == self.config.window
            return packed_steps(tokens, packing, self.config.count, seed)
        if order is None:
            return contiguous_steps(tokens, self.step_tokens, seed)
        return ordered_steps(tokens, offsets, order, self.step_tokens)
