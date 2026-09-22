import torch

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkPacker:
    max_length: int
    eos_token_id: int
    pad_token_id: int

    def pack(self, raw_tokens: torch.Tensor) -> Iterator[torch.Tensor]:
        """Pack complete documents, truncating documents longer than max_length."""
        # raw_tokens: (n,); d is the number of complete documents.
        eos_positions = (raw_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]  # (d,)
        if len(eos_positions) == 0:
            return

        chunk_parts: list[torch.Tensor] = []  # each part: (document_length,)
        chunk_len = 0

        prev_start = 0
        for i in range(len(eos_positions)):
            curr_eos = eos_positions[i].item()
            doc = raw_tokens[prev_start:curr_eos + 1]  # (doc_len,)
            prev_start = curr_eos + 1
            doc_len = len(doc)

            if doc_len > self.max_length:
                if chunk_len > 0:
                    padding = torch.full((self.max_length - chunk_len,), self.pad_token_id, dtype=torch.uint8)  # (max_length - chunk_len,)
                    yield torch.cat(chunk_parts + [padding])  # (max_length,)
                    chunk_parts = []
                    chunk_len = 0
                yield doc[:self.max_length].clone()  # (max_length,)
                continue

            if doc_len + chunk_len > self.max_length:
                padding = torch.full((self.max_length - chunk_len,), self.pad_token_id, dtype=torch.uint8)  # (max_length - chunk_len,)
                yield torch.cat(chunk_parts + [padding])  # (max_length,)
                chunk_parts = []
                chunk_len = 0

            chunk_parts.append(doc)  # append (doc_len,)
            chunk_len += doc_len

            if chunk_len == self.max_length:
                yield torch.cat(chunk_parts)  # (max_length,)
                chunk_parts = []
                chunk_len = 0

        if chunk_len > 0:
            padding = torch.full((self.max_length - chunk_len,), self.pad_token_id, dtype=torch.uint8)  # (max_length - chunk_len,)
            yield torch.cat(chunk_parts + [padding])  # (max_length,)


@dataclass(frozen=True)
class LegacyFlatPacker:
    seq_len: int
    eos_token_id: int
    pad_token_id: int

    def split_oversized(self, sample: torch.Tensor) -> Iterator[torch.Tensor]:
        # sample: (n,); unlike ChunkPacker, retain every token of oversized documents.
        for j in range(0, len(sample), self.seq_len):
            chunk = sample[j:j + self.seq_len]  # (min(seq_len, n - j),)
            if len(chunk) < self.seq_len:
                padding = torch.full((self.seq_len - len(chunk),), self.pad_token_id, dtype=torch.uint8)  # (seq_len - len(chunk),)
                chunk = torch.cat([chunk, padding])  # (seq_len,)
            yield chunk  # (seq_len,)
