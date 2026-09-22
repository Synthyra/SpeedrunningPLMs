"""Corruption order, bounded prefetch, and worker cleanup checks."""

import threading
import pytest
import torch

from collections.abc import Iterator
from torch import Tensor

from speedrunning_plms.research import benchmark
from speedrunning_plms.research.batches import prepared_batches


def token_batches(count: int = 5) -> Iterator[Tensor]:
    for offset in range(count):
        yield torch.arange(64).reshape(4, 16).add(offset).remainder(33)  # (4, 16)


@pytest.mark.parametrize("prefetch", [False, True])
def test_corruption_matches_serial_order(prefetch: bool) -> None:
    expected_generator = torch.Generator().manual_seed(19)
    expected = [benchmark.corrupt_tokens(tokens, generator=expected_generator) for tokens in token_batches()]
    actual_generator = torch.Generator().manual_seed(19)
    with prepared_batches(token_batches(), actual_generator, torch.device("cpu"), prefetch=prefetch) as batches:
        actual = list(batches)

    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected):
        for actual_tensor, expected_tensor in zip(actual_batch, expected_batch):  # each (4, 16)
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert torch.equal(actual_generator.get_state(), expected_generator.get_state())


@pytest.mark.parametrize("prefetch", [False, True])
def test_preparation_starts_only_on_first_next(prefetch: bool) -> None:
    def forbidden_source() -> Iterator[Tensor]:
        raise AssertionError("The deadline check must precede batch preparation")
        yield  # pragma: no cover

    with prepared_batches(
        forbidden_source(), torch.Generator(), torch.device("cpu"), prefetch=prefetch,
    ):
        pass


def test_prefetch_is_bounded_and_joins_worker() -> None:
    calls: list[int] = []
    workers: list[threading.Thread] = []
    second_started = threading.Event()

    def tracked_source() -> Iterator[Tensor]:
        for index, tokens in enumerate(token_batches()):  # tokens: (4, 16)
            calls.append(index)
            workers.append(threading.current_thread())
            if index == 1:
                second_started.set()
            yield tokens  # (4, 16)

    with prepared_batches(tracked_source(), torch.Generator(), torch.device("cpu"), prefetch=True) as batches:
        next(batches)
        assert second_started.wait(timeout=5)
    assert calls == [0, 1]
    assert len(set(workers)) == 1
    assert workers[0] is not threading.current_thread()
    assert not workers[0].is_alive()
    with pytest.raises(StopIteration):
        next(batches)


@pytest.mark.parametrize("prefetch", [False, True])
def test_source_errors_propagate(prefetch: bool) -> None:
    def failing_source() -> Iterator[Tensor]:
        yield next(token_batches())  # (4, 16)
        raise ValueError("source failed")

    with prepared_batches(
        failing_source(), torch.Generator(), torch.device("cpu"), prefetch=prefetch,
    ) as batches:
        next(batches)
        with pytest.raises(ValueError, match="source failed"):
            next(batches)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_prefetch_matches_cpu_corruption() -> None:
    expected_generator = torch.Generator().manual_seed(19)
    expected = [benchmark.corrupt_tokens(tokens, generator=expected_generator) for tokens in token_batches()]
    with prepared_batches(
        token_batches(), torch.Generator().manual_seed(19), torch.device("cuda"), prefetch=True,
    ) as batches:
        for actual_batch, expected_batch in zip(batches, expected):
            for actual_tensor, expected_tensor in zip(actual_batch, expected_batch):  # each (4, 16)
                assert actual_tensor.device.type == "cuda"
                torch.testing.assert_close(actual_tensor.cpu(), expected_tensor, rtol=0, atol=0)
