"""The shape rule: 36 blocks, each one attention sublayer and one MLP of intermediate 2,048 on a 768-wide stream.

A candidate exposes `model.blocks`, a sequence of 36 modules each holding `attention` and `mlp` submodules. The check
reads every `torch.nn.Linear` inside them: the MLP must hold an up projection 768 -> 2,048 (or 768 -> 4,096 for a
gated pair) and a down projection 2,048 -> 768; the attention must read a 768-wide input. Everything else (heads,
extra projections, tables, gates, scalars, normalization) is free. The receipt lists every shape so a reader can
audit a run without the code.
"""

from __future__ import annotations

import torch

from typing import Any


BLOCKS = 36
HIDDEN = 768  # d
INTERMEDIATE = 2048  # f


def _linear_shapes(module: torch.nn.Module) -> list[tuple[int, int]]:
    return [(layer.in_features, layer.out_features) for layer in module.modules() if isinstance(layer, torch.nn.Linear)]


def check_architecture(model: torch.nn.Module) -> dict[str, Any]:
    """Raise when the model breaks the shape rule; return the receipt otherwise."""
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) != BLOCKS:
        raise ValueError(f"the model must expose `blocks` with exactly {BLOCKS} entries")
    receipt: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        attention = getattr(block, "attention", None)
        mlp = getattr(block, "mlp", None)
        if attention is None or mlp is None:
            raise ValueError(f"block {index} must hold `attention` and `mlp` submodules")
        attention_shapes = _linear_shapes(attention)
        mlp_shapes = _linear_shapes(mlp)
        up = [shape for shape in mlp_shapes if shape[0] == HIDDEN and shape[1] in (INTERMEDIATE, 2 * INTERMEDIATE)]
        down = [shape for shape in mlp_shapes if shape == (INTERMEDIATE, HIDDEN)]
        wider = [shape for shape in mlp_shapes if shape[1] > 2 * INTERMEDIATE or shape[0] > 2 * INTERMEDIATE]
        if not up or not down or wider:
            raise ValueError(f"block {index}: the MLP must be {HIDDEN} -> {INTERMEDIATE} -> {HIDDEN}; found {mlp_shapes}")
        if not any(shape[0] == HIDDEN for shape in attention_shapes):
            raise ValueError(f"block {index}: attention must read a {HIDDEN}-wide stream; found {attention_shapes}")
        receipt.append({"attention": attention_shapes, "mlp": mlp_shapes})
    return {
        "blocks": BLOCKS, "hidden": HIDDEN, "intermediate": INTERMEDIATE,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "block_shapes": receipt,
    }
