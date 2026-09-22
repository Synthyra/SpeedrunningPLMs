"""Inspect FlexAttention modifiers as dense score or mask matrices.

Adapted from pytorch-labs/attention-gym, attn_gym/mods/softcapping.py.
"""

import math
import numpy as np
import torch
from contextlib import nullcontext
from pathlib import Path
from torch.nn.attention.flex_attention import (
    _score_mod_signature,
    _mask_mod_signature,
    _vmap_for_bhqkv,
    _ModificationType,
)

try:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
except ImportError:
    from torch._higher_order_ops.flex_attention import TransformGetItemToIndex


def create_score_mod(
    query: torch.Tensor,
    key: torch.Tensor,
    score_mod: _score_mod_signature | None,
    mask_mod: _mask_mod_signature | None,
    device: str = "cuda",
    _compile: bool = False,
    scale: float | None = None,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    # query: (m, d_h); key: (n, d_h), for one selected batch and head.
    m = query.shape[0]  # query count
    n = key.shape[0]  # key count

    batch_indices = torch.arange(0, 1, device=device) + batch_idx  # (1,)
    head_indices = torch.arange(0, 1, device=device) + head_idx  # (1,)
    query_indices = torch.arange(0, m, device=device)  # (m,)
    key_indices = torch.arange(0, n, device=device)  # (n,)

    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    modification_type = _ModificationType.SCORE_MOD if score_mod is not None else _ModificationType.MASK_MOD
    if _compile:
        ctx = nullcontext()
    else:
        ctx = TransformGetItemToIndex()

    with ctx:
        mod_fn = score_mod if modification_type == _ModificationType.SCORE_MOD else mask_mod
        prefix = (0,) if modification_type == _ModificationType.SCORE_MOD else ()
        mod = _vmap_for_bhqkv(mod_fn, prefix=prefix)
        scores = query @ key.transpose(-2, -1)  # (m, n)
        scores *= scale_factor  # (m, n)
        scores = scores.view(1, 1, m, n)  # (1, 1, m, n)
        if modification_type == _ModificationType.SCORE_MOD:
            out = mod(scores, batch_indices, head_indices, query_indices, key_indices)  # (1, 1, m, n)
        else:
            out = mod(batch_indices, head_indices, query_indices, key_indices)  # (1, 1, m, n)

    return out  # (1, 1, m, n)


def generate_dilated_sliding_window(window_size: int, dilation: int) -> _mask_mod_signature:
    """Allow distances at most window_size that are divisible by dilation."""

    def dilated_sliding_window(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        # FlexAttention supplies scalar indices (); direct calls may broadcast them.
        diff = torch.abs(q_idx - kv_idx)  # broadcast(q_idx.shape, kv_idx.shape)
        in_window = diff <= window_size  # same broadcast shape
        is_dilated = (diff % dilation) == 0  # same broadcast shape
        return in_window & is_dilated  # same broadcast shape

    dilated_sliding_window.__name__ = f"dilated_sliding_window_{window_size}_dilation_{dilation}"
    return dilated_sliding_window


def _name_to_title(name: str) -> str:
    title = name.replace("_", " ")
    title = " ".join(word.capitalize() for word in title.split())
    return title


def visualize_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    score_mod: _score_mod_signature | None = None,
    mask_mod: _mask_mod_signature | None = None,
    device: str = "cuda",
    name: str = "attention_scores",
    path: Path | None = None,
    batch_idx: int = 0,
    head_idx: int = 0,
    scale: float | None = None,
) -> None:
    """Save one batch/head's scores or mask as a 300 dpi PNG.

    Inputs have shape (b, h, m, d_h) and (b, h, n, d_h). If both modifiers
    are supplied, apply the score modifier and mask excluded scores with -inf.
    By default, use 1 / sqrt(d_h) scaling and save to name.png in the current directory.
    """
    import matplotlib.pyplot as plt

    assert score_mod is not None or mask_mod is not None, (
        "Must provide either score_mod or mask_mod"
    )
    query = query[batch_idx, head_idx, :, :]  # (m, d_h)
    key = key[batch_idx, head_idx, :, :]  # (n, d_h)
    scores_viz = create_score_mod(
        query,
        key,
        score_mod=score_mod,
        mask_mod=mask_mod,
        scale=scale,
        device=device,
        batch_idx=batch_idx,
        head_idx=head_idx,
    )  # (1, 1, m, n)
    if score_mod is not None and mask_mod is not None:
        mask_viz = create_score_mod(
            query,
            key,
            score_mod=None,
            mask_mod=mask_mod,
            scale=scale,
            device=device,
            batch_idx=batch_idx,
            head_idx=head_idx,
        )  # (1, 1, m, n)
        scores_viz = torch.where(mask_viz == 0, float("-inf"), scores_viz)  # (1, 1, m, n)

    suffix_title = f"Batch {batch_idx}, Head {head_idx}" if batch_idx != 0 or head_idx != 0 else ""

    fig, ax = plt.subplots(figsize=(12, 10))
    color = "viridis" if score_mod is not None else "cividis"
    if score_mod is not None and mask_mod is not None:
        color = "plasma"
    scores_image = scores_viz.cpu().detach()[0, 0, :, :]  # (m, n)
    im = ax.imshow(scores_image, aspect="auto", cmap=color)
    fig.colorbar(im)

    title = _name_to_title(name)
    file_path = Path(name).with_suffix(".png") if path is None else path.with_suffix(".png")
    ax.set_title(f"{title}\n{suffix_title}", fontsize=20)

    ax.set_xlabel("Key Tokens", fontsize=18)
    ax.set_ylabel("Query Tokens", fontsize=18)

    # Place key-token labels above the image.
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    # Add tick labels if the number of tokens is manageable
    num_query_tokens, num_kv_tokens = scores_viz.shape[-2:]
    if num_query_tokens <= 32 and num_kv_tokens <= 32:
        ax.set_xticks(range(num_kv_tokens))
        rotation = 45 if num_kv_tokens > 12 else 0
        ax.set_xticklabels(
            [f"KV{i}" for i in range(num_kv_tokens)], fontsize=16, rotation=rotation
        )
        ax.set_yticks(range(num_query_tokens))
        ax.set_yticklabels([f"Q{i}" for i in range(num_query_tokens)], fontsize=16)
        # Align grid with pixel boundaries
        ax.set_xticks(np.arange(-0.5, num_kv_tokens, 1), minor=True)  # boundaries: (n + 1,)
        ax.set_yticks(np.arange(-0.5, num_query_tokens, 1), minor=True)  # boundaries: (m + 1,)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualization saved as {file_path}")


def main(device: str = "cpu") -> None:
    """Visualize a dilated sliding window mask."""
    b, h, l, d_h = 1, 1, 24, 8  # batch, heads, sequence length, head width
    query = torch.ones(b, h, l, d_h, device=device)  # (b, h, l, d_h)
    key = torch.ones(b, h, l, d_h, device=device)  # (b, h, l, d_h)

    dilated_sliding_window_mask = generate_dilated_sliding_window(window_size=8, dilation=4)
    visualize_attention_scores(
        query,
        key,
        mask_mod=dilated_sliding_window_mask,
        device=device,
        name="dilated_sliding_window_mask",
    )


if __name__ == "__main__":
    main()
