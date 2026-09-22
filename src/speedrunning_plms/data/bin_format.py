import numpy as np
import torch

from pathlib import Path


MAGIC = 20240520
VERSION = 1
HEADER_SIZE = 256


def read_shard_num_tokens(path: str | Path) -> int:
    header = torch.from_file(str(path), False, HEADER_SIZE, dtype=torch.int32)  # (HEADER_SIZE,)
    assert header[0] == MAGIC, "magic number mismatch in the data .bin file"
    assert header[1] == VERSION, "unsupported version"
    return int(header[2])


def read_shard_tokens(path: str | Path) -> torch.Tensor:
    path = Path(path)
    num_tokens = read_shard_num_tokens(path)
    with path.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint8)  # (num_tokens,)
        f.seek(HEADER_SIZE * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == num_tokens, "number of tokens read does not match header?"
    return tokens  # (num_tokens,)


def write_shard(path: str | Path, tokens: np.ndarray) -> None:
    # tokens: (n,), uint8
    assert len(tokens) < 2**31, "token count too large"
    header = np.zeros(HEADER_SIZE, dtype=np.int32)  # (HEADER_SIZE,)
    header[0] = MAGIC  # scalar slot
    header[1] = VERSION  # scalar slot
    header[2] = len(tokens)  # scalar slot
    with Path(path).open("wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())
