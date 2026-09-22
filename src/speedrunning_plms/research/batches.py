"""Deterministic corruption with optional batch preparation in one worker."""

from __future__ import annotations

import torch

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from torch import Tensor

from speedrunning_plms.research import benchmark


@contextmanager
def prepared_batches(
    batches: Iterator[Tensor],
    generator: torch.Generator,
    device: torch.device,
    *,
    prefetch: bool = False,
) -> Iterator[Iterator[tuple[Tensor, Tensor]]]:
    """Yield corrupted batches, with at most one batch prepared ahead.

    Each input and output tensor has shape (b, l). Preparation starts on the
    first next(), so the caller can check its training deadline first. The
    worker exclusively owns the source iterator and corruption generator until
    context exit; early exit may consume one unused batch and its random draws.
    """
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plm-batches") if prefetch else None
    copy_stream = torch.cuda.Stream(device=device) if prefetch and device.type == "cuda" else None

    def prepare() -> tuple[Tensor, Tensor] | None:
        tokens = next(batches, None)  # (b, l), or exhausted.
        if tokens is None:
            return None
        inputs, labels = benchmark.corrupt_tokens(tokens, generator=generator)  # each (b, l)
        if copy_stream is None:
            return inputs.to(device), labels.to(device)  # each (b, l)

        inputs, labels = inputs.pin_memory(), labels.pin_memory()  # each (b, l)
        # PyTorch's pinned allocator retains host storage until these copies finish.
        with torch.cuda.stream(copy_stream):
            return inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)  # each (b, l)

    def consume() -> Iterator[tuple[Tensor, Tensor]]:
        pending = executor.submit(prepare) if executor is not None else None
        while True:
            batch = pending.result() if pending is not None else prepare()  # two (b, l) tensors, or exhausted.
            if batch is None:
                return
            if copy_stream is not None:
                consumer_stream = torch.cuda.current_stream(device)
                consumer_stream.wait_stream(copy_stream)
                for tensor in batch:  # (b, l)
                    tensor.record_stream(consumer_stream)
            # Enqueue only after the consumer's wait, so it does not wait for
            # the next batch's transfer as well as this batch's transfer.
            if executor is not None:
                pending = executor.submit(prepare)
            yield batch  # two (b, l) tensors.

    iterator = consume()
    try:
        yield iterator
    finally:
        iterator.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if copy_stream is not None:
            copy_stream.synchronize()
