"""Check local binary shards and CPU loader contracts."""

import numpy as np
import pytest
import torch

from pathlib import Path

from speedrunning_plms.data import TokenIds, read_shard_num_tokens, read_shard_tokens, write_shard
from speedrunning_plms.data.loaders import ChunkedTrainDataset, EvalLoader


TOKEN_IDS = TokenIds(cls_token_id=0, eos_token_id=2, pad_token_id=1, mask_token_id=32)


def test_shard_round_trip_preserves_header_contract(tmp_path: Path) -> None:
    path = tmp_path / "tiny.bin"
    tokens = np.array([0, 5, 2, 0, 6, 2], dtype=np.uint8)  # (6,)
    write_shard(path, tokens)

    assert read_shard_num_tokens(path) == len(tokens)
    actual = read_shard_tokens(path)  # (6,)
    expected = torch.tensor(tokens, dtype=torch.uint8)  # (6,)
    torch.testing.assert_close(actual, expected)


def test_eval_loader_accepts_token_ids_and_yields_cpu_masked_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    tokens = np.array([0, 5, 2, 0, 6, 2], dtype=np.uint8)  # (6,)
    write_shard("tiny_valid_000000.bin", tokens)
    torch.manual_seed(0)
    dataset = EvalLoader(
        filename_pattern="tiny_valid_*.bin",
        seq_len=6,
        process_rank=0,
        num_processes=1,
        tokenizer=TOKEN_IDS,
    )
    input_ids, labels, mask_rate = next(iter(dataset))  # (6,), (6,), (1,)

    assert input_ids.shape == labels.shape == (6,)
    assert mask_rate.shape == (1,)
    assert input_ids.device.type == labels.device.type == mask_rate.device.type == "cpu"
    assert torch.all(labels[input_ids == TOKEN_IDS.cls_token_id] == -100)


def test_chunked_train_dataset_preserves_chunk_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    tokens = np.array([0, 5, 2, 0, 6, 2, 0, 7, 2, 0, 8, 2], dtype=np.uint8)  # (12,)
    write_shard("tiny_train_000000.bin", tokens)
    dataset = ChunkedTrainDataset(
        filename_pattern="tiny_train_*.bin",
        max_length=4,
        batch_size=2,
        process_rank=0,
        num_processes=1,
        max_epochs=1,
        tokenizer=TOKEN_IDS,
        num_workers=1,
    )
    batch = next(iter(dataset))  # (2, 4)

    assert batch.shape == (2, 4)
    assert batch.dtype == torch.int32
