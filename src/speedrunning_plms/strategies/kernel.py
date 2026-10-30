"""The kernel family: which backend runs flex attention in training. No loss change is intended.

- `default` (the record): flex attention's own choice (`BACKEND` "AUTO", the Triton kernel at this shape).
- `flash`: `kernel_options={"BACKEND": "FLASH"}`, the FlashAttention-4 backend (PyTorch blog, 2026-03-04; the
  option is in PyTorch 2.11's `FlexKernelOptions`, documented there as experimental and needing the flash-attn cute-dsl
  kernels installed; flex raises if the backend cannot be used, so a missing install fails the run loudly). The blog
  reports 1.4 to 2.0 times the Triton kernel under document masks on H200 and names two limits that apply here:
  per-token mask metadata can stall the pipeline, and the backward is not deterministic with block sparsity, so two runs
  of one genome can differ. The dense path (CPU, evaluation) never uses flex, so this changes training steps only.
- `cudagraphs` (`ideas/2026-10-30_step_throughput.md` idea 2; modded-nanogpt record 92): the training model compiled
  with `mode="reduce-overhead"`, so CUDA graph trees replay each compiled graph (forward, backward, the no-grad
  forward) instead of launching its kernels from Python; the loop marks each step's start. The optimizers stay outside
  the graphs and read the learning rate and momentum as before, so no per-step float enters a graph: the values the
  graphs read that change during a run (branch ramps, readout switches, dropout rates) are 0-d buffers already, and a
  new window shape records a new graph once. The kernels and their order are the default's, so the loss should be
  the same. On the CPU the model never compiles and this is the record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from speedrunning_plms.strategies.strategy import Strategy, StrategyConfig


@dataclass(frozen=True)
class KernelConfig(StrategyConfig):
    variant: str = "default"
    VARIANTS: ClassVar[tuple[str, ...]] = ("default", "flash", "cudagraphs")


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

    def compile_options(self) -> dict[str, Any]:
        """The training model's `torch.compile` options beside the curriculum's: the record's default, or CUDA graphs."""
        return {"mode": "reduce-overhead"} if self.variant == "cudagraphs" else {}

    @property
    def marks_steps(self) -> bool:
        """True when the loop must tell the graph trees where each training step starts."""
        return self.variant == "cudagraphs"
