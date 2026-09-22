"""Exercise binary validation, document boundaries, and masking on tiny CPU inputs."""

import numpy as np
import pytest
import torch

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from speedrunning_plms.data.bin_format import HEADER_SIZE, read_shard_num_tokens, read_shard_tokens, write_shard
from speedrunning_plms.data import tokenize as tokenization
from speedrunning_plms.data.loaders import AsyncBatchPipeline, EvalLoader, TrainLoader, apply_masking_gpu
from speedrunning_plms.data.packers import ChunkPacker, LegacyFlatPacker
from speedrunning_plms.data.tokens import TokenIds


TOKEN_IDS = TokenIds(cls_token_id=0, eos_token_id=2, pad_token_id=1, mask_token_id=32)


@pytest.mark.parametrize("field,value,message", [(0, 0, "magic number"), (1, 99, "unsupported version")])
def test_binary_reader_rejects_invalid_header(tmp_path: Path, field: int, value: int, message: str) -> None:
    path = tmp_path / "invalid.bin"
    write_shard(path, np.array([0, 5, 2], dtype=np.uint8))  # tokens: (3,)
    payload = bytearray(path.read_bytes())
    payload[field * 4:(field + 1) * 4] = np.int32(value).tobytes()
    path.write_bytes(payload)

    with pytest.raises(AssertionError, match=message):
        read_shard_num_tokens(path)
    with pytest.raises(AssertionError, match=message):
        read_shard_tokens(path)


def test_binary_reader_rejects_truncated_header(tmp_path: Path) -> None:
    path = tmp_path / "truncated_header.bin"
    path.write_bytes(bytes(HEADER_SIZE * 4 - 1))

    with pytest.raises(RuntimeError, match="size"):
        read_shard_num_tokens(path)


def test_binary_reader_rejects_truncated_payload(tmp_path: Path) -> None:
    path = tmp_path / "truncated_payload.bin"
    write_shard(path, np.array([0, 5, 2], dtype=np.uint8))  # tokens: (3,)
    path.write_bytes(path.read_bytes()[:-1])

    assert read_shard_num_tokens(path) == 3
    with pytest.raises(AssertionError, match="number of tokens read"):
        read_shard_tokens(path)


def test_empty_binary_shard_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    write_shard(path, np.empty(0, dtype=np.uint8))  # tokens: (0,)

    tokens = read_shard_tokens(path)  # (0,)
    assert read_shard_num_tokens(path) == 0
    assert tokens.shape == (0,)
    assert tokens.dtype == torch.uint8


@pytest.mark.parametrize(
    "tokens,expected",
    [
        pytest.param([], [], id="empty"),
        pytest.param([0, 5, 6], [], id="no-complete-document"),
        pytest.param([0, 5, 6, 2], [[0, 5, 6, 2]], id="exact-boundary"),
        pytest.param([0, 2, 0, 2], [[0, 2, 0, 2]], id="combine-documents"),
        pytest.param([0, 5, 2, 0, 6, 2], [[0, 5, 2, 1], [0, 6, 2, 1]], id="preserve-boundaries"),
        pytest.param([0, 5, 2, 0, 6], [[0, 5, 2, 1]], id="ignore-incomplete-tail"),
        pytest.param([0, 5, 6, 7, 8, 2], [[0, 5, 6, 7]], id="truncate-oversized-document"),
        pytest.param(
            [0, 2, 0, 5, 6, 7, 8, 2, 0, 9, 2],
            [[0, 2, 1, 1], [0, 5, 6, 7], [0, 9, 2, 1]],
            id="flush-before-truncation-and-resume",
        ),
    ],
)
def test_chunk_packer_document_boundaries(tokens: list[int], expected: list[list[int]]) -> None:
    raw_tokens = torch.tensor(tokens, dtype=torch.uint8)  # (len(tokens),)
    original = raw_tokens.clone()  # (len(tokens),)
    chunks = list(ChunkPacker(max_length=4, eos_token_id=2, pad_token_id=1).pack(raw_tokens))

    assert [chunk.tolist() for chunk in chunks] == expected
    assert all(chunk.shape == (4,) and chunk.dtype == torch.uint8 for chunk in chunks)
    torch.testing.assert_close(raw_tokens, original)


@pytest.mark.parametrize(
    "tokens,expected",
    [
        pytest.param([], [], id="empty"),
        pytest.param([0, 5, 2], [[0, 5, 2, 1]], id="pad-short-sample"),
        pytest.param([0, 5, 6, 2], [[0, 5, 6, 2]], id="exact-boundary"),
        pytest.param([0, 5, 6, 7, 8, 2], [[0, 5, 6, 7], [8, 2, 1, 1]], id="retain-oversized-tail"),
        pytest.param([0, 5, 6, 7, 8, 9, 10, 2], [[0, 5, 6, 7], [8, 9, 10, 2]], id="two-full-chunks"),
    ],
)
def test_legacy_packer_retains_sample_tokens(tokens: list[int], expected: list[list[int]]) -> None:
    sample = torch.tensor(tokens, dtype=torch.uint8)  # (len(tokens),)
    chunks = list(LegacyFlatPacker(seq_len=4, eos_token_id=2, pad_token_id=1).split_oversized(sample))

    assert [chunk.tolist() for chunk in chunks] == expected
    assert all(chunk.shape == (4,) and chunk.dtype == torch.uint8 for chunk in chunks)
    assert sample.tolist() == tokens


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("mask_rate", [0.0, 1.0])
def test_masking_extremes_preserve_special_tokens_and_targets(batched: bool, mask_rate: float) -> None:
    tokens = torch.tensor([0, 5, 6, 2, 1], dtype=torch.int32)  # (5,)
    if batched:
        tokens = tokens.unsqueeze(0).repeat(2, 1)  # (2, 5)
    original = tokens.clone()  # (5,) or (2, 5)
    special_tokens = torch.tensor([0, 2, 1], dtype=torch.int32)  # (3,)

    noisy, labels, rate = apply_masking_gpu(tokens, special_tokens, 32, mask_rate, mlm=True)
    # noisy, labels: tokens.shape; rate: ()
    expected_noisy = original.clone()  # (5,) or (2, 5)
    expected_labels = torch.full_like(original, -100)  # (5,) or (2, 5)
    if mask_rate == 1.0:
        expected_noisy[..., 1:3] = 32  # selected slice: (..., 2)
        expected_labels[..., 1:3] = original[..., 1:3]  # selected slice: (..., 2)

    torch.testing.assert_close(noisy, expected_noisy)
    torch.testing.assert_close(labels, expected_labels)
    torch.testing.assert_close(tokens, original)
    assert noisy.device.type == labels.device.type == rate.device.type == "cpu"
    assert rate.item() == mask_rate


def test_eval_masking_never_masks_cls_eos_or_padding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_shard("tiny.bin", np.array([0, 5, 2], dtype=np.uint8))  # tokens: (3,)
    loader = EvalLoader("*.bin", seq_len=5, process_rank=0, num_processes=1, tokenizer=TOKEN_IDS)
    original = torch.tensor([0, 5, 6, 2, 1], dtype=torch.uint8)  # (5,)

    # Force every position into the candidate mask to test special-token exclusion.
    with patch("speedrunning_plms.data.loaders.torch.rand", return_value=torch.zeros(5)):
        noisy, labels, rate = loader._apply_masking(original)  # noisy, labels: (5,); rate: (1,)

    assert noisy.tolist() == [0, 32, 32, 2, 1]
    assert labels.tolist() == [-100, 5, 6, -100, -100]
    assert original.tolist() == [0, 5, 6, 2, 1]
    assert noisy.dtype == labels.dtype == torch.int32
    assert rate.item() == pytest.approx(0.15)


@pytest.mark.parametrize("max_epochs", [1, 2])
def test_train_loader_preserves_documents_across_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_epochs: int,
) -> None:
    monkeypatch.chdir(tmp_path)
    write_shard("tiny_0.bin", np.array([0, 5, 2], dtype=np.uint8))  # (3,)
    write_shard("tiny_1.bin", np.array([0, 6, 2], dtype=np.uint8))  # (3,)
    loader = TrainLoader(
        "tiny_*.bin", seq_len=6, process_rank=0, num_processes=1,
        max_epochs=max_epochs, tokenizer=TOKEN_IDS, mlm=True, mask_rate=0.0,
    )
    batches = list(loader)  # each input/labels: (6,); mask_rate: (1,)

    assert len(batches) == max_epochs
    assert batches[0][0].tolist() == [0, 5, 2, 0, 6, 2]
    for inputs, labels, rate in batches:
        assert sorted(inputs.reshape(2, 3).tolist()) == [[0, 5, 2], [0, 6, 2]]
        assert labels.tolist() == [-100] * 6
        assert rate.item() == 0


@pytest.mark.parametrize("cpu_count,workers", [(None, 1), (1, 1), (8, 6)])
def test_tokenization_handles_missing_cpu_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_count: int | None, workers: int,
) -> None:
    monkeypatch.setattr(tokenization.os, "cpu_count", lambda: cpu_count)
    monkeypatch.setattr(tokenization.EsmTokenizer, "from_pretrained", Mock())
    pool = MagicMock()
    pool.__enter__.return_value.imap.return_value = [np.array([0, 5, 2], dtype=np.uint8)]  # each (3,)
    pool_factory = Mock(return_value=pool)
    monkeypatch.setattr(tokenization.mp, "Pool", pool_factory)
    tokenization.tokenize_fw([], data_name="tiny", max_length=4, shard_size=8, data_cache_dir=tmp_path)

    pool_factory.assert_called_once_with(workers)
    tokens = read_shard_tokens(tmp_path / "tiny_train_000000.bin")  # (3,)
    assert tokens.tolist() == [0, 5, 2]


def test_async_batch_records_consumer_stream_before_prefetch(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    pipeline = AsyncBatchPipeline.__new__(AsyncBatchPipeline)
    pipeline.transfer_stream = object()
    consumer = Mock()
    consumer.wait_stream.side_effect = lambda stream: events.append(("wait", stream))
    batch = Mock()
    batch.record_stream.side_effect = lambda stream: events.append(("record", stream))
    pipeline._next_batch = batch
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: consumer)
    monkeypatch.setattr(pipeline, "_prefetch", lambda: events.append(("prefetch", None)))

    assert pipeline.next_batch() is batch
    assert events == [("wait", pipeline.transfer_stream), ("record", consumer), ("prefetch", None)]
