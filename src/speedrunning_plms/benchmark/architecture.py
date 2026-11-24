"""The architecture receipt: what a run's model is, block by block, so a reader can audit it without the code.

Until 2026-11-23 this was the shape rule (36 blocks, a 768-wide stream, an MLP of 2,048). The speedrun is now matched by
its clock alone and the shape is the genome's (`strategies/shape.py`;
`docs/decisions/representation/speedrunning_plms/2026-11-23_compute_matched_free_architecture.md`), so the check keeps
only the skeleton's structure: `model.blocks`, a non-empty sequence of modules each holding `attention` and `mlp`
submodules, and an `embedding`. The receipt lists every block's linear shapes and the stored parameters, a matrix shared
between blocks counted once.
"""

from __future__ import annotations

import torch

from typing import Any


# The shape every run before 2026-11-23 had. Their `train.py` and `strategies/` snapshots read these names, so probe.py,
# dial.py and perplexity.py can rebuild them; working code reads the genome's shape (`tests/test_candidate.py` checks).
BLOCKS = 36
HIDDEN = 768  # d
INTERMEDIATE = 2048  # f


def _linear_shapes(module: torch.nn.Module) -> list[tuple[int, int]]:
    return [(layer.in_features, layer.out_features) for layer in module.modules() if isinstance(layer, torch.nn.Linear)]


def check_architecture(model: torch.nn.Module) -> dict[str, Any]:
    """Raise when the model lacks the skeleton's structure; return the receipt otherwise."""
    blocks = getattr(model, "blocks", None)
    embedding = getattr(model, "embedding", None)
    if not blocks or not isinstance(embedding, torch.nn.Embedding):
        raise ValueError("the model must expose an `embedding` table and a non-empty `blocks`")
    receipt: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        attention = getattr(block, "attention", None)
        mlp = getattr(block, "mlp", None)
        if attention is None or mlp is None:
            raise ValueError(f"block {index} must hold `attention` and `mlp` submodules")
        receipt.append({"attention": _linear_shapes(attention), "mlp": _linear_shapes(mlp)})
    return {
        "blocks": len(blocks), "hidden": embedding.embedding_dim,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "block_shapes": receipt,
    }
