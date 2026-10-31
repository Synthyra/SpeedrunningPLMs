"""Attention backend selection, masking, and sequence unpadding for `probe_attention`.

A layer runs Flex attention, PyTorch's scaled dot-product attention, or a FlashAttention kernel
that FastPLMs loads through its `kernels.lock`. The kernels backend keeps the kernel it loaded in
this module's `FLASH_KERNEL`, `FLASH_KERNEL_VARIANT`, and `FLASH_KERNEL_DTYPES`; read or reset them
here, because a name imported elsewhere keeps its value from import time.
"""

from __future__ import annotations

import logging
import warnings
import torch
import torch.nn.functional as F

from enum import Enum
from typing import TYPE_CHECKING, Any
from einops import rearrange, repeat
from torch.autograd.function import FunctionCtx


try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
except ImportError:
    create_block_mask = None
    flex_attention = None

    if TYPE_CHECKING:
        from torch.nn.attention.flex_attention import BlockMask
    else:
        BlockMask = None


logger = logging.getLogger(__name__)
# The kernels backend prefers FlashAttention 3. FastPLMs' kernels.lock pins the build and
# digest of each, so no revision is recorded here.
FLASH_KERNEL_PREFERENCE = ("flash_attention_3", "flash_attention_2")


class AttentionBackend(Enum):
    """The attention implementations a `probe_attention` layer can run."""

    KERNELS = "kernels"
    FLEX = "flex"
    SDPA = "sdpa"


VALID_ATTENTION_BACKENDS = tuple(backend.value for backend in AttentionBackend)


def _load_kernels_flash() -> tuple[Any, str, tuple[torch.dtype, ...]]:
    """Load the most preferred FlashAttention kernel that this platform and GPU can run.

    A kernel is passed over only when FastPLMs reports that no locked build of it runs here.
    Any other failure, such as a download error or a digest mismatch, raises with its cause.
    Returns the kernel, its variant, and the Q, K, and V dtypes its FastPLMs manifest entry lists.
    """
    # Only this backend needs FastPLMs, so every other backend runs without it installed.
    from fastplms import attention as fastplms_attention
    from fastplms.registry import get_model_registry

    unsupported: list[str] = []
    for implementation in FLASH_KERNEL_PREFERENCE:
        reason = fastplms_attention.flash_kernel_unsupported_reason(implementation)
        if reason is not None:
            unsupported.append(f"{implementation}: {reason}")
            continue

        flash_kernel, flash_variant = fastplms_attention._ensure_flash_kernels_loaded(implementation)
        kernel_spec = get_model_registry().attention_kernels[implementation]
        # The manifest names dtypes as torch spells them, such as "bfloat16".
        flash_dtypes = tuple(getattr(torch, dtype_name) for dtype_name in kernel_spec.dtypes)
        if unsupported:
            logger.warning(
                "The kernels attention backend uses %s. Passed over %s",
                implementation,
                "; ".join(unsupported),
            )
        else:
            logger.info("The kernels attention backend uses %s, its first preference.", implementation)

        return flash_kernel, flash_variant, flash_dtypes

    raise RuntimeError(
        "The kernels attention backend has no FlashAttention kernel that this platform and GPU "
        f"can run. {'; '.join(unsupported)}"
    )


# The kernel FastPLMs loads from the `kernels` hub at run time, which has no static type.
FLASH_KERNEL: Any = None
FLASH_KERNEL_VARIANT: str | None = None
# The Q, K, and V dtypes that the loaded kernel's FastPLMs manifest entry lists.
FLASH_KERNEL_DTYPES: tuple[torch.dtype, ...] = ()


def _ensure_flash_kernels_loaded() -> None:
    """Load the FlashAttention kernel into the module globals once per process."""
    global FLASH_KERNEL, FLASH_KERNEL_VARIANT, FLASH_KERNEL_DTYPES
    if FLASH_KERNEL is None:
        FLASH_KERNEL, FLASH_KERNEL_VARIANT, FLASH_KERNEL_DTYPES = _load_kernels_flash()


def resolve_attention_backend(requested_backend: str) -> AttentionBackend:
    """Validate a backend name and load the FlashAttention kernel when it is selected."""
    normalized_backend = requested_backend.lower()
    if normalized_backend == "kernels_flash":
        normalized_backend = AttentionBackend.KERNELS.value

    assert normalized_backend in VALID_ATTENTION_BACKENDS, (
        f"Unsupported attention backend: {requested_backend}. "
        f"Expected one of {VALID_ATTENTION_BACKENDS}."
    )

    if normalized_backend == AttentionBackend.KERNELS.value:
        _ensure_flash_kernels_loaded()
        return AttentionBackend.KERNELS

    if normalized_backend == AttentionBackend.FLEX.value:
        if flex_attention is None or create_block_mask is None:
            warnings.warn(
                "Flex attention is unavailable in this environment; falling back to sdpa.",
                stacklevel=2,
            )
            return AttentionBackend.SDPA

        return AttentionBackend.FLEX

    return AttentionBackend.SDPA


def sdpa_attention_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask_4d: torch.Tensor | None,
) -> torch.Tensor:
    """Run PyTorch scaled dot-product attention."""
    # query_states, key_states, value_states: (b, h, l, d_h)
    # attention_mask_4d: (b, 1, 1, l), (b, 1, l, l), or None
    return F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask_4d,
        is_causal=False,
    )  # (b, h, l, d_h)


def flex_attention_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    flex_block_mask: BlockMask | None,
) -> torch.Tensor:
    """Run PyTorch Flex attention with an optional block mask."""
    # query_states, key_states, value_states: (b, h, l, d_h)
    assert flex_attention is not None, "Flex attention is unavailable in this environment."
    return flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=flex_block_mask,
    )  # (b, h, l, d_h)


class IndexFirstAxis(torch.autograd.Function):
    """Gather rows of the first axis, with a backward pass that scatters the gradient back.

    n counts the rows of a flattened batch, b * l, and n_valid the unpadded rows among them.
    """

    @staticmethod
    def forward(ctx: FunctionCtx, input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather `indices` from the first axis of `input_tensor`."""
        # input_tensor: (n, ...); indices: (n_valid,)
        ctx.save_for_backward(indices)
        assert input_tensor.ndim >= 2
        ctx.first_axis_dim = input_tensor.shape[0]  # n
        other_shape = input_tensor.shape[1:]  # ...
        flattened_dim = other_shape.numel()  # d_flat, the product of the trailing dimensions
        return torch.gather(
            rearrange(input_tensor, "b ... -> b (...)"),  # (n, d_flat)
            0,
            repeat(indices, "z -> z d", d=flattened_dim),  # (n_valid, d_flat)
        ).reshape(-1, *other_shape)  # (n_valid, ...)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Scatter the gathered rows' gradient back into a zero-filled first axis."""
        # grad_output: (n_valid, ...)
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")  # (n_valid, d_flat)
        grad_input = torch.zeros(
            (ctx.first_axis_dim, grad_output.shape[1]),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )  # (n, d_flat)
        scatter_indices = repeat(indices, "z -> z d", d=grad_output.shape[1])  # (n_valid, d_flat)
        grad_input.scatter_(0, scatter_indices, grad_output)  # (n, d_flat)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None  # (n, ...)


class IndexPutFirstAxis(torch.autograd.Function):
    """Scatter rows into a zero-filled first axis, with a backward pass that gathers them."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        values: torch.Tensor,
        indices: torch.Tensor,
        first_axis_dim: int,
    ) -> torch.Tensor:
        """Place `values` at `indices` of a zero-filled first axis of length `first_axis_dim`."""
        # values: (n_valid, ...); indices: (n_valid,); first_axis_dim: n
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim,
            *values.shape[1:],
            device=values.device,
            dtype=values.dtype,
        )  # (n, ...)
        output[indices] = values  # (n, ...)
        return output  # (n, ...)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Gather the gradient of the rows the forward pass placed."""
        # grad_output: (n, ...)
        (indices,) = ctx.saved_tensors
        return grad_output[indices], None, None  # (n_valid, ...)


index_first_axis = IndexFirstAxis.apply
index_put_first_axis = IndexPutFirstAxis.apply


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Return unpadded token rows to their positions in the padded batch."""
    # hidden_states: (n_valid, ...); indices: (n_valid,)
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)  # (b * l, ...)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)  # (b, l, ...)


def _kernels_flash_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """Run the loaded FlashAttention kernel over a batch without padding."""
    # query_states: (b, l, h, d_h); key_states, value_states: (b, l, h_kv, d_h)
    assert FLASH_KERNEL is not None, "resolve_attention_backend('kernels') loads the kernel first."

    if FLASH_KERNEL_VARIANT == "flash_attn2":
        return FLASH_KERNEL.fwd(
            q=query_states, k=key_states, v=value_states, is_causal=False
        )[0]  # (b, l, h, d_h)

    if FLASH_KERNEL_VARIANT == "flash_attn3":
        output = FLASH_KERNEL.flash_attn_func(
            q=query_states,
            k=key_states,
            v=value_states,
            causal=False,
        )  # (b, l, h, d_h), or a tuple that starts with it
        if isinstance(output, tuple):
            return output[0]  # (b, l, h, d_h)

        return output  # (b, l, h, d_h)

    raise AssertionError(f"Unsupported kernels flash attention variant: {FLASH_KERNEL_VARIANT}")


def _kernels_flash_varlen_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> torch.Tensor:
    """Run the loaded FlashAttention kernel over unpadded tokens."""
    # query_states: (n_valid, h, d_h); key_states, value_states: (n_valid, h_kv, d_h)
    # cu_seqlens_q, cu_seqlens_k: (b + 1,), the offset at which each sequence starts
    assert FLASH_KERNEL is not None, "resolve_attention_backend('kernels') loads the kernel first."

    if FLASH_KERNEL_VARIANT == "flash_attn2":
        return FLASH_KERNEL.varlen_fwd(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_causal=False,
        )[0]  # (n_valid, h, d_h)

    if FLASH_KERNEL_VARIANT == "flash_attn3":
        output = FLASH_KERNEL.flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=False,
        )  # (n_valid, h, d_h), or a tuple that starts with it
        if isinstance(output, tuple):
            return output[0]  # (n_valid, h, d_h)

        return output  # (n_valid, h, d_h)

    raise AssertionError(f"Unsupported kernels flash attention variant: {FLASH_KERNEL_VARIANT}")


def _unpad_input(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask_2d: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    tuple[int, int],
]:
    """Drop padded positions and compute the sequence offsets the varlen kernel reads."""
    # query_states: (b, l, h, d_h); key_states, value_states: (b, l, h_kv, d_h)
    # attention_mask_2d: (b, l)
    batch_size, seq_len, num_heads, head_dim = query_states.shape  # b, l, h, d_h
    _, _, num_key_value_heads, _ = key_states.shape  # h_kv
    assert key_states.shape[1] == seq_len, "Query and key sequence lengths must match for kernels attention."
    assert value_states.shape[1] == seq_len, "Query and value sequence lengths must match for kernels attention."

    valid_lengths = attention_mask_2d.sum(dim=1).int()  # (b,)
    cu_seqlens = F.pad(valid_lengths.cumsum(0, dtype=torch.int32), (1, 0))  # (b + 1,)
    max_seq_len = int(valid_lengths.max().item())  # the longest row's real length
    indices = attention_mask_2d.flatten().nonzero(as_tuple=False).flatten()  # (n_valid,)

    query_states = index_first_axis(
        query_states.reshape(batch_size * seq_len, num_heads, head_dim), indices
    )  # (n_valid, h, d_h)
    key_states = index_first_axis(
        key_states.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices
    )  # (n_valid, h_kv, d_h)
    value_states = index_first_axis(
        value_states.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices
    )  # (n_valid, h_kv, d_h)
    return (
        query_states,  # (n_valid, h, d_h)
        key_states,  # (n_valid, h_kv, d_h)
        value_states,  # (n_valid, h_kv, d_h)
        indices,  # (n_valid,)
        (cu_seqlens, cu_seqlens),  # (b + 1,) each, the same offsets for queries and keys
        (max_seq_len, max_seq_len),
    )


def kernels_attention_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask_2d: torch.Tensor | None,
) -> torch.Tensor:
    """Run the loaded FlashAttention kernel, unpadding the batch when a padding mask is given."""
    # query_states: (b, l, h, d_h); key_states, value_states: (b, l, h_kv, d_h)
    # attention_mask_2d: (b, l) or None
    assert FLASH_KERNEL is not None, "resolve_attention_backend('kernels') loads the kernel first."
    # Q, K, and V share one dtype. Raise here, not inside the kernel, on one its manifest omits.
    if query_states.dtype not in FLASH_KERNEL_DTYPES:
        listed = ", ".join(str(kernel_dtype) for kernel_dtype in FLASH_KERNEL_DTYPES)
        raise RuntimeError(
            f"kernels attention cannot run {query_states.dtype} inputs: {FLASH_KERNEL_VARIANT} "
            f"takes {listed}, the dtypes its FastPLMs manifest entry lists."
        )

    if attention_mask_2d is None:
        return _kernels_flash_forward(query_states, key_states, value_states)  # (b, l, h, d_h)

    batch_size, seq_len = query_states.shape[:2]  # b, l
    (
        query_states,  # (n_valid, h, d_h)
        key_states,  # (n_valid, h_kv, d_h)
        value_states,  # (n_valid, h_kv, d_h)
        indices,  # (n_valid,)
        (cu_seqlens_q, cu_seqlens_k),  # (b + 1,) each
        (max_seqlen_q, max_seqlen_k),
    ) = _unpad_input(query_states, key_states, value_states, attention_mask_2d)

    attn_output = _kernels_flash_varlen_forward(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )  # (n_valid, h, d_h)
    return pad_input(attn_output, indices, batch_size, seq_len)  # (b, l, h, d_h)


@torch.compiler.disable
def build_attention_masks(
    attention_backend: AttentionBackend,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
    attention_mask_2d: torch.Tensor | None = None,
    attention_mask_4d: torch.Tensor | None = None,
    flex_block_mask: BlockMask | None = None,
    output_attentions: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, BlockMask | None]:
    """Build the masks the selected backend reads from whichever mask the caller passed.

    Returns the 2D padding mask, the 4D mask that SDPA and eager attention read, and the Flex
    block mask. Each is None when the backend does not read it or when no mask was passed.
    """
    # attention_mask: (b, l), a 4D mask, or None; attention_mask_2d: (b, l) or None
    # attention_mask_4d: (b, 1, 1, l), (b, 1, l, l), or None
    if attention_mask is not None:
        if attention_mask.ndim == 2:
            attention_mask_2d = attention_mask.bool()  # (b, l)
        elif attention_mask.ndim == 4:
            attention_mask_4d = attention_mask.bool()  # (b, 1, 1, l) or (b, 1, l, l)
        else:
            raise AssertionError(
                f"attention_mask must be 2D or 4D, but received shape {tuple(attention_mask.shape)}."
            )

    if attention_mask_2d is None and attention_mask_4d is None and flex_block_mask is None:
        return None, None, None  # no padding: every (b, l) position is real

    if attention_mask_2d is None and attention_mask_4d is not None:
        attention_mask_2d = attention_mask_4d[:, 0, 0, :]  # (b, l)

    # Returned attention weights come from eager attention, which needs the 4D mask built below.
    if attention_backend == AttentionBackend.KERNELS and not output_attentions:
        return attention_mask_2d, None, None  # (b, l), None, None

    if attention_backend == AttentionBackend.FLEX and flex_block_mask is None and not output_attentions:
        assert attention_mask_2d is not None, (
            "Flex attention requires a 2D padding mask or an explicit block mask. "
            "Arbitrary 4D masks should be passed as flex_block_mask."
        )
        assert create_block_mask is not None, "Flex attention is unavailable in this environment."
        valid_lengths = attention_mask_2d.sum(dim=-1)  # (b,)

        # The block mask assumes right padding: a position is real when its index is below the
        # row's count of real positions.
        def mask_mod(
            batch_idx: torch.Tensor,
            head_idx: torch.Tensor,
            query_idx: torch.Tensor,
            key_idx: torch.Tensor,
        ) -> torch.Tensor:
            # batch_idx, head_idx, query_idx, key_idx: (), one index each, as Flex attention calls it
            return (query_idx < valid_lengths[batch_idx]) & (key_idx < valid_lengths[batch_idx])  # ()

        flex_block_mask = create_block_mask(mask_mod, batch_size, 1, seq_len, seq_len, device=device)
        return attention_mask_2d, None, flex_block_mask  # (b, l), None, a block mask over (b, 1, l, l)

    # SDPA and eager attention mask only the keys, so a padded query still attends to the real
    # keys and its softmax row is not NaN, as a row of only -inf would make it.
    if attention_mask_4d is None and attention_mask_2d is not None:
        attention_mask_4d = attention_mask_2d[:, None, None, :]  # (b, 1, 1, l)

    return (
        attention_mask_2d,  # (b, l) or None
        attention_mask_4d,  # (b, 1, 1, l), (b, 1, l, l), or None
        flex_block_mask,  # a caller's block mask, or None
    )
