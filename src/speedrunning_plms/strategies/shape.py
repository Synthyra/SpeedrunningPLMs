"""The shape family: the model's depth, stream width and MLP width, free since 2026-11-23.

The benchmark fixed 36 blocks of a 768-wide stream with an MLP of 2,048 until Logan asked for a compute-matched
speedrun (`docs/decisions/representation/speedrunning_plms/2026-11-23_compute_matched_free_architecture.md`): the
GH200's clock prices every shape, so the search may trade depth for width, a smaller model for more steps, or stored
blocks for loop passes. Every other family sizes its per-block parameters by `blocks` and its vectors by `hidden`, and
states what it needs of the shape in its `conflicts` (a layout that fits the depth, heads that divide the width).

Widths are multiples of 64 so the matmuls tile; the attention family keeps its own head count and checks that the
per-head width is a multiple of 16 and at most 256, what the flex kernels take.

Symbols: d = `hidden`, f = `intermediate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from speedrunning_plms.strategies.strategy import Strategy, StrategyConfig


WIDTH_MULTIPLE = 64  # d and f tile the GH200's matmuls


@dataclass(frozen=True)
class ShapeConfig(StrategyConfig):
    variant: str = "dense"
    blocks: int = 36
    hidden: int = 768  # d
    intermediate: int = 2048  # f
    VARIANTS: ClassVar[tuple[str, ...]] = ("dense",)
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {"blocks": (2, 96), "hidden": (256, 2048), "intermediate": (256, 8192)}
    INTEGERS: ClassVar[tuple[str, ...]] = ("blocks", "hidden", "intermediate")

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("hidden", "intermediate"):
            if getattr(self, name) % WIDTH_MULTIPLE:
                raise ValueError(f"{name} {getattr(self, name)} must be a multiple of {WIDTH_MULTIPLE}")


class Shape(Strategy):
    family = "shape"
    Config = ShapeConfig
    config: ShapeConfig
