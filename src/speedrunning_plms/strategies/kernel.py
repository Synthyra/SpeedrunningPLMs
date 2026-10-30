"""The kernel family: which backend runs flex attention in training. No loss change is intended.

- `default` (the record): flex attention's own choice (`BACKEND` "AUTO", the Triton kernel at this shape).
- `flash`: `kernel_options={"BACKEND": "FLASH"}`, the FlashAttention-4 backend (PyTorch blog, 2026-03-04; the
  option is in PyTorch 2.11's `FlexKernelOptions`, documented there as experimental and needing the flash-attn cute-dsl
  kernels installed; flex raises if the backend cannot be used, so a missing install fails the run loudly). The blog
  reports 1.4 to 2.0 times the Triton kernel under document masks on H200 and names two limits that apply here:
  per-token mask metadata can stall the pipeline, and the backward is not deterministic with block sparsity, so two runs
  of one genome can differ. The dense path (CPU, evaluation) never uses flex, so this changes training steps only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from speedrunning_plms.strategies.strategy import Strategy, StrategyConfig


@dataclass(frozen=True)
class KernelConfig(StrategyConfig):
    variant: str = "default"
    VARIANTS: ClassVar[tuple[str, ...]] = ("default", "flash")


class Kernel(Strategy):
    family = "kernel"
    Config = KernelConfig
    config: KernelConfig

    @property
    def active(self) -> bool:
        return False  # read by the attention sublayers at construction, never as a hook

    def kernel_options(self) -> dict[str, str] | None:
        """The flex call's kernel options, or None to call it as the record does."""
        return {"BACKEND": "FLASH"} if self.variant == "flash" else None
