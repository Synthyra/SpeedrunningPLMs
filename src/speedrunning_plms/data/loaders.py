"""Packed token loaders. Sequence lengths and batch sizes follow each loader configuration."""

import random

import torch
import torch.utils.data as data

from collections.abc import Iterator
from pathlib import Path
from torch.utils.data import DataLoader, IterableDataset
from transformers import EsmTokenizer

from speedrunning_plms.data.bin_format import read_shard_tokens
from speedrunning_plms.data.packers import ChunkPacker
from speedrunning_plms.data.tokens import TokenIds


def _coerce_token_ids(tokenizer: EsmTokenizer | TokenIds) -> TokenIds:
    if isinstance(tokenizer, TokenIds):
        return tokenizer
    return TokenIds.from_tokenizer(tokenizer)


def _load_data_shard(file: Path) -> torch.Tensor:
    return read_shard_tokens(file)  # (num_tokens,)


class EvalLoader(IterableDataset):
    """Distribute masked evaluation batches across ranks."""

    def __init__(
        self,
        filename_pattern: str,
        seq_len: int,
        process_rank: int,
        num_processes: int,
        tokenizer: EsmTokenizer | TokenIds,
    ) -> None:
        self.filename_pattern = filename_pattern
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        token_ids = _coerce_token_ids(tokenizer)
        self.cls_token_id = token_ids.cls_token_id
        self.eos_token_id = token_ids.eos_token_id
        self.pad_token_id = token_ids.pad_token_id
        self.mask_token_id = token_ids.mask_token_id
        self.special_tokens = [self.cls_token_id, self.eos_token_id, self.pad_token_id]

        # Rank assignment happens after packing so each rank sees the same batch order.
        self.all_files = sorted(Path.cwd().glob(filename_pattern))
        if not self.all_files:
            raise ValueError(f"No files found matching pattern: {filename_pattern}")

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate batches, with each process taking every num_processes-th batch."""
        batch_count = 0

        for file in self.all_files:
            raw_tokens = _load_data_shard(file)  # (num_tokens,)

            eos_positions = (raw_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]  # (num_documents,)

            if len(eos_positions) == 0:
                continue

            batch_tokens = []
            curr_batch_len = 0

            for i in range(len(eos_positions)):
                curr_eos = eos_positions[i]  # () tensor index
                prev_eos_plus_one = 0 if i == 0 else eos_positions[i-1] + 1  # scalar index
                sample = raw_tokens[prev_eos_plus_one:curr_eos+1]  # (sample_length,)

                if len(sample) > self.seq_len:
                    for j in range(0, len(sample), self.seq_len):
                        chunk = sample[j:j+self.seq_len]  # (min(seq_len, sample_length - j),)
                        if len(chunk) < self.seq_len:
                            padding = torch.full((self.seq_len - len(chunk),), self.pad_token_id, dtype=torch.uint8)  # (seq_len - len(chunk),)
                            chunk = torch.cat([chunk, padding])  # (seq_len,)

                        if batch_count % self.num_processes == self.process_rank:
                            input_ids, labels, mask_rate = self._apply_masking(chunk)  # (seq_len,), (seq_len,), (1,)
                            yield input_ids, labels, mask_rate  # (seq_len,), (seq_len,), (1,)
                        batch_count += 1
                    continue

                if len(sample) + curr_batch_len > self.seq_len:
                    if curr_batch_len > 0:
                        padding = torch.full((self.seq_len - curr_batch_len,), self.pad_token_id, dtype=torch.uint8)  # (seq_len - curr_batch_len,)
                        batch_tokens.append(padding)  # append (seq_len - curr_batch_len,)
                        batch = torch.cat(batch_tokens)  # (seq_len,)

                        if batch_count % self.num_processes == self.process_rank:
                            input_ids, labels, mask_rate = self._apply_masking(batch)  # (seq_len,), (seq_len,), (1,)
                            yield input_ids, labels, mask_rate  # (seq_len,), (seq_len,), (1,)
                        batch_count += 1

                    batch_tokens = [sample]  # one (sample_length,) tensor
                    curr_batch_len = len(sample)
                else:
                    batch_tokens.append(sample)  # append (sample_length,)
                    curr_batch_len += len(sample)

                if curr_batch_len == self.seq_len:
                    batch = torch.cat(batch_tokens)  # (seq_len,)

                    if batch_count % self.num_processes == self.process_rank:
                        input_ids, labels, mask_rate = self._apply_masking(batch)  # (seq_len,), (seq_len,), (1,)
                        yield input_ids, labels, mask_rate  # (seq_len,), (seq_len,), (1,)
                    batch_count += 1
                    batch_tokens = []
                    curr_batch_len = 0

            # Yield final incomplete batch if it exists
            if curr_batch_len > 0:
                padding = torch.full((self.seq_len - curr_batch_len,), self.pad_token_id, dtype=torch.uint8)  # (seq_len - curr_batch_len,)
                batch_tokens.append(padding)  # append (seq_len - curr_batch_len,)
                batch = torch.cat(batch_tokens)  # (seq_len,)

                if batch_count % self.num_processes == self.process_rank:
                    input_ids, labels, mask_rate = self._apply_masking(batch)  # (seq_len,), (seq_len,), (1,)
                    yield input_ids, labels, mask_rate  # (seq_len,), (seq_len,), (1,)
                batch_count += 1

    def _apply_masking(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask a CPU sequence of shape (sequence_length,)."""
        sequence = sequence.to(dtype=torch.int32)  # (sequence_length,)

        # Use fixed mask rate for evaluation
        mask_rate = torch.full((1,), 0.15)  # (1,)

        p_mask = mask_rate.repeat(len(sequence))  # (sequence_length,)
        mask_indices = torch.rand(len(sequence)) < p_mask  # (sequence_length,)

        # Don't mask special tokens
        special_mask = torch.isin(sequence, torch.tensor(self.special_tokens, dtype=torch.int32))  # (sequence_length,)
        mask_indices = mask_indices & ~special_mask  # (sequence_length,)

        noisy_batch = torch.where(mask_indices, self.mask_token_id, sequence)  # (sequence_length,)
        labels = sequence.clone()  # (sequence_length,)
        labels[~mask_indices] = -100  # labels shape unchanged

        return noisy_batch, labels, mask_rate  # (sequence_length,), (sequence_length,), (1,)


class OptimizedEvalLoader:
    """Transfer masked evaluation batches to CUDA."""

    def __init__(
        self,
        filename_pattern: str,
        seq_len: int,
        process_rank: int,
        num_processes: int,
        tokenizer: EsmTokenizer | TokenIds,
    ) -> None:
        self.filename_pattern = filename_pattern
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes

        self._dataset = EvalLoader(
            filename_pattern=filename_pattern,
            seq_len=seq_len,
            process_rank=process_rank,
            num_processes=num_processes,
            tokenizer=tokenizer,
        )

        # Store file list for compatibility - all processes see all files
        self.files = self._dataset.all_files

        # Create the dataloader (single worker for evaluation to ensure deterministic order)
        self.dataloader = DataLoader(
            self._dataset,
            batch_size=None,  # Dataset returns complete batches
            num_workers=0,    # Single worker for deterministic eval order
            pin_memory=True,  # Pin memory for faster GPU transfer
        )

        self._iterator = None
        self._exhausted = False

    def reset(self) -> None:
        """Reset the dataloader iterator."""
        self._iterator = iter(self.dataloader)
        self._exhausted = False

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the next batch, ensuring GPU transfer happens here."""
        if self._iterator is None:
            self.reset()

        try:
            input_ids, labels, mask_rate = next(self._iterator)  # (seq_len,), (seq_len,), (1,)
            input_ids = input_ids.cuda(non_blocking=True)  # (seq_len,)
            labels = labels.cuda(non_blocking=True)  # (seq_len,)
            mask_rate = mask_rate.cuda(non_blocking=True)  # (1,)
            return input_ids, labels, mask_rate  # (seq_len,), (seq_len,), (1,)
        except StopIteration:
            self._exhausted = True
            # Return empty tensors to signal end of data
            return torch.empty(0, device='cuda'), torch.empty(0, device='cuda'), torch.empty(0, device='cuda')  # three (0,) tensors


class TrainLoader(IterableDataset):
    """An IterableDataset that handles distributed padded data loading with masking."""

    def __init__(
        self,
        filename_pattern: str,
        seq_len: int,
        process_rank: int,
        num_processes: int,
        max_epochs: int,
        tokenizer: EsmTokenizer | TokenIds,
        num_workers: int = 1,
        mlm: bool = False,
        mask_rate: float = 0.15,
    ) -> None:
        self.filename_pattern = filename_pattern
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.max_epochs = max_epochs
        self.num_workers = num_workers
        self.mask_rate = mask_rate
        token_ids = _coerce_token_ids(tokenizer)
        self.cls_token_id = token_ids.cls_token_id
        self.eos_token_id = token_ids.eos_token_id
        self.pad_token_id = token_ids.pad_token_id
        self.mask_token_id = token_ids.mask_token_id
        self.special_tokens = [self.cls_token_id, self.eos_token_id, self.pad_token_id]
        self.mlm = mlm
        # Get all files and distribute across processes (GPUs)
        all_files = sorted(Path.cwd().glob(filename_pattern))
        if not all_files:
            raise ValueError(f"No files found matching pattern: {filename_pattern}")

        # First distribute files across processes (GPUs)
        files_per_process = len(all_files) // self.num_processes
        extra_files = len(all_files) % self.num_processes

        start_idx = self.process_rank * files_per_process + min(self.process_rank, extra_files)
        end_idx = start_idx + files_per_process + (1 if self.process_rank < extra_files else 0)

        self.process_files = all_files[start_idx:end_idx]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Then distribute this process's files across workers
        files_per_worker = len(self.process_files) // num_workers
        extra_files = len(self.process_files) % num_workers

        start_idx = worker_id * files_per_worker + min(worker_id, extra_files)
        end_idx = start_idx + files_per_worker + (1 if worker_id < extra_files else 0)

        worker_files = self.process_files[start_idx:end_idx]

        # Process files cyclically for multiple epochs
        epoch = 0
        file_idx = 0
        leftover_tokens = torch.empty(0, dtype=torch.uint8)  # (0,)

        while epoch < self.max_epochs:
            # Shuffle files at the start of each epoch
            if file_idx == 0 and epoch > 0:
                # Include process rank for proper distributed shuffling
                random.seed(epoch + self.process_rank * 10000 + worker_id * 1000)
                random.shuffle(worker_files)

            if file_idx < len(worker_files):
                raw_tokens = _load_data_shard(worker_files[file_idx])  # (num_tokens,)
                raw_tokens = torch.cat([leftover_tokens, raw_tokens], dim=0)  # (pending_tokens + num_tokens,)
                file_idx += 1
            else:
                if leftover_tokens.numel() == 0:
                    epoch += 1
                    file_idx = 0
                    continue
                raw_tokens = leftover_tokens  # (pending_tokens,)
                leftover_tokens = torch.empty(0, dtype=torch.uint8)  # (0,)

            eos_positions = (raw_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]  # (num_documents,)

            if len(eos_positions) == 0:
                leftover_tokens = raw_tokens  # (remaining_tokens,)
                if file_idx >= len(worker_files):
                    epoch += 1
                    file_idx = 0
                continue

            batch_tokens = []
            curr_batch_len = 0

            for i in range(len(eos_positions)):
                curr_eos = eos_positions[i]  # () tensor index
                prev_eos_plus_one = 0 if i == 0 else eos_positions[i-1] + 1  # scalar index
                sample = raw_tokens[prev_eos_plus_one:curr_eos+1]  # (sample_length,)

                if len(sample) > self.seq_len:
                    for j in range(0, len(sample), self.seq_len):
                        chunk = sample[j:j+self.seq_len]  # (min(seq_len, sample_length - j),)
                        if len(chunk) < self.seq_len:
                            padding = torch.full((self.seq_len - len(chunk),), self.pad_token_id, dtype=torch.uint8)  # (seq_len - len(chunk),)
                            chunk = torch.cat([chunk, padding])  # (seq_len,)

                        input_ids, labels, mask_rate = self._apply_masking(chunk)  # (seq_len,), (seq_len,), (1,)
                        yield input_ids, labels, mask_rate  # (seq_len,), (seq_len,), (1,)
                    continue

                if len(sample) + curr_batch_len > self.seq_len:
                    if curr_batch_len > 0:
                        padding = torch.full((self.seq_len - curr_batch_len,), self.pad_token_id, dtype=torch.uint8)  # (seq_len - curr_batch_len,)
                        batch_tokens.append(padding)  # append (seq_len - curr_batch_len,)
                        batch = torch.cat(batch_tokens)  # (seq_len,)

                        input_ids, labels, mask_rate = self._apply_masking(batch)  # (seq_len,), (seq_len,), (1,)
                        yield input_ids, labels, mask_rate  # (seq_len,), (seq_len,), (1,)

                    batch_tokens = [sample]  # one (sample_length,) tensor
                    curr_batch_len = len(sample)
                else:
                    batch_tokens.append(sample)  # append (sample_length,)
                    curr_batch_len += len(sample)

                if curr_batch_len == self.seq_len:
                    batch = torch.cat(batch_tokens)  # (seq_len,)
                    input_ids, labels, mask_rate = self._apply_masking(batch)  # (seq_len,), (seq_len,), (1,)
                    yield input_ids, labels, mask_rate  # (seq_len,), (seq_len,), (1,)
                    batch_tokens = []
                    curr_batch_len = 0

            # Save leftover tokens for next file
            if len(eos_positions) > 0:
                leftover_tokens = raw_tokens[eos_positions[-1]+1:]  # (remaining_tokens,)

            # Carry complete documents that did not fill a batch into the next shard.
            if file_idx < len(worker_files) and curr_batch_len > 0:
                leftover_tokens = torch.cat(batch_tokens + [leftover_tokens])  # (pending_tokens,)

            # Yield final incomplete batch if at end of epoch
            if file_idx >= len(worker_files) and curr_batch_len > 0:
                padding = torch.full((self.seq_len - curr_batch_len,), self.pad_token_id, dtype=torch.uint8)  # (seq_len - curr_batch_len,)
                batch_tokens.append(padding)  # append (seq_len - curr_batch_len,)
                batch = torch.cat(batch_tokens)  # (seq_len,)
                input_ids, labels, mask_rate = self._apply_masking(batch)  # (seq_len,), (seq_len,), (1,)
                yield input_ids, labels, mask_rate  # (seq_len,), (seq_len,), (1,)

                epoch += 1
                file_idx = 0

    def _apply_masking(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask a CPU sequence of shape (sequence_length,)."""
        sequence = sequence.to(dtype=torch.int32)  # (sequence_length,)

        if self.mlm:
            mask_rate = torch.full((1,), self.mask_rate)  # (1,)
        else:
            eps = 1e-3
            mask_rate = torch.rand(1)  # (1,)
            mask_rate = (1 - eps) * mask_rate + eps  # (1,)

        p_mask = mask_rate.repeat(len(sequence))  # (sequence_length,)
        mask_indices = torch.rand(len(sequence)) < p_mask  # (sequence_length,)

        # Don't mask special tokens
        special_mask = torch.isin(sequence, torch.tensor(self.special_tokens, dtype=torch.int32))  # (sequence_length,)
        mask_indices = mask_indices & ~special_mask  # (sequence_length,)

        noisy_batch = torch.where(mask_indices, self.mask_token_id, sequence)  # (sequence_length,)
        labels = sequence.clone()  # (sequence_length,)
        labels[~mask_indices] = -100  # labels shape unchanged

        return noisy_batch, labels, mask_rate  # (sequence_length,), (sequence_length,), (1,)


class OptimizedTrainLoader:
    """Load masked training batches with workers and transfer them to CUDA."""

    def __init__(
        self,
        filename_pattern: str,
        seq_len: int,
        process_rank: int,
        num_processes: int,
        max_epochs: int,
        tokenizer: EsmTokenizer | TokenIds,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        mlm: bool = False,
        mask_rate: float = 0.15,
    ) -> None:
        self.filename_pattern = filename_pattern
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.mlm = mlm
        self.mask_rate = mask_rate

        self._dataset = TrainLoader(
            filename_pattern=filename_pattern,
            seq_len=seq_len,
            process_rank=process_rank,
            num_processes=num_processes,
            max_epochs=max_epochs,
            tokenizer=tokenizer,
            num_workers=num_workers,
            mlm=mlm,
            mask_rate=mask_rate,
        )

        # Store file list for compatibility - only this process's files
        self.files = self._dataset.process_files

        self.dataloader = DataLoader(
            self._dataset,
            batch_size=None,  # Dataset returns complete batches
            num_workers=num_workers,
            pin_memory=True,  # Pin memory for faster GPU transfer
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        )

        self._iterator = None
        self._exhausted = False

    def set_mask_rate(self, mask_rate: float) -> None:
        """Set the mask rate for the next batch(es)."""
        self.mask_rate = mask_rate
        self._dataset.mask_rate = mask_rate

    def set_mlm(self, mlm: bool) -> None:
        """Set whether to use MLM masking in the dataset."""
        self.mlm = mlm
        self._dataset.mlm = mlm

    def reset(self) -> None:
        """Reset the dataloader iterator."""
        self._iterator = iter(self.dataloader)
        self._exhausted = False

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the next batch, ensuring GPU transfer happens here."""
        if self._iterator is None:
            self.reset()

        try:
            input_ids, labels, mask_rate = next(self._iterator)  # (seq_len,), (seq_len,), (1,)
            input_ids = input_ids.cuda(non_blocking=True)  # (seq_len,)
            labels = labels.cuda(non_blocking=True)  # (seq_len,)
            mask_rate = mask_rate.cuda(non_blocking=True)  # (1,)
            return input_ids, labels, mask_rate  # (seq_len,), (seq_len,), (1,)
        except StopIteration:
            self._exhausted = True
            # Return empty tensors to signal end of data
            return torch.empty(0, device='cuda'), torch.empty(0, device='cuda'), torch.empty(0, device='cuda')  # three (0,) tensors


class ChunkedTrainDataset(IterableDataset):
    """Chunk-aligned IterableDataset that packs documents into fixed-length chunks.

    Each chunk is exactly max_length tokens with documents packed end-to-end.
    No document spans a chunk boundary. If a document doesn't fit in the current
    chunk, the remainder is padded and a new chunk starts. Documents exceeding
    max_length are truncated to their own chunk.

    Yields batches of (batch_size, max_length) int32 tensors containing raw input_ids
    (masking runs on the GPU in the training loop).
    """

    def __init__(
        self,
        filename_pattern: str,
        max_length: int,
        batch_size: int,
        process_rank: int,
        num_processes: int,
        max_epochs: int,
        tokenizer: EsmTokenizer | TokenIds,
        num_workers: int = 1,
    ) -> None:
        self.filename_pattern = filename_pattern
        self.max_length = max_length
        self.batch_size = batch_size
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.max_epochs = max_epochs
        self.num_workers = num_workers
        token_ids = _coerce_token_ids(tokenizer)
        self.cls_token_id = token_ids.cls_token_id
        self.eos_token_id = token_ids.eos_token_id
        self.pad_token_id = token_ids.pad_token_id

        all_files = sorted(Path.cwd().glob(filename_pattern))
        assert len(all_files) > 0, f"No files found matching pattern: {filename_pattern}"

        # Distribute files across processes (GPUs)
        files_per_process = len(all_files) // num_processes
        extra = len(all_files) % num_processes
        start = process_rank * files_per_process + min(process_rank, extra)
        end = start + files_per_process + (1 if process_rank < extra else 0)
        self.process_files = all_files[start:end]

    def _pack_chunks(self, raw_tokens: torch.Tensor) -> Iterator[torch.Tensor]:
        """Pack raw tokens into max_length-aligned chunks.

        Documents are delineated by EOS tokens. Each chunk contains one or more
        complete documents, padded at the end if needed. Oversized documents are truncated.

        Yields individual (max_length,) uint8 chunks.
        """
        # raw_tokens: (num_tokens,); each yielded chunk: (max_length,)
        yield from ChunkPacker(
            max_length=self.max_length,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        ).pack(raw_tokens)  # each chunk: (max_length,)

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Distribute this process's files across workers
        files_per_worker = len(self.process_files) // num_workers
        extra = len(self.process_files) % num_workers
        start = worker_id * files_per_worker + min(worker_id, extra)
        end = start + files_per_worker + (1 if worker_id < extra else 0)
        worker_files = list(self.process_files[start:end])

        epoch = 0
        leftover_tokens = torch.empty(0, dtype=torch.uint8)  # (0,)
        batch_chunks: list[torch.Tensor] = []  # each chunk: (max_length,)

        while epoch < self.max_epochs:
            file_idx = 0

            if epoch > 0:
                random.seed(epoch + self.process_rank * 10000 + worker_id * 1000)
                random.shuffle(worker_files)

            while file_idx < len(worker_files):
                raw_tokens = _load_data_shard(worker_files[file_idx])  # (num_tokens,)
                raw_tokens = torch.cat([leftover_tokens, raw_tokens])  # (pending_tokens + num_tokens,)
                file_idx += 1

                # Find last complete document
                eos_positions = (raw_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]  # (num_documents,)
                if len(eos_positions) == 0:
                    leftover_tokens = raw_tokens  # (remaining_tokens,)
                    continue

                last_eos_pos = eos_positions[-1].item()
                leftover_tokens = raw_tokens[last_eos_pos + 1:]  # (remaining_tokens,)
                complete_tokens = raw_tokens[:last_eos_pos + 1]  # (last_eos_pos + 1,)

                for chunk in self._pack_chunks(complete_tokens):  # chunk: (max_length,)
                    batch_chunks.append(chunk.to(torch.int32))  # append (max_length,)
                    if len(batch_chunks) == self.batch_size:
                        yield torch.stack(batch_chunks)  # (batch_size, max_length)
                        batch_chunks = []

            leftover_tokens = torch.empty(0, dtype=torch.uint8)  # (0,)
            batch_chunks = []
            epoch += 1


class ChunkedTrainLoader:
    """Chunk-aligned training data loader.

    Yields (batch_size, max_length) int32 tensors of raw input_ids on CPU (pinned memory).
    Masking runs on the GPU in the training loop.
    """

    def __init__(
        self,
        filename_pattern: str,
        max_length: int,
        micro_batch_tokens: int,
        process_rank: int,
        num_processes: int,
        max_epochs: int,
        tokenizer: EsmTokenizer | TokenIds,
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ) -> None:
        self.max_length = max_length
        batch_size = micro_batch_tokens // max_length
        assert batch_size >= 1, f"micro_batch_tokens ({micro_batch_tokens}) must be >= max_length ({max_length})"

        self._dataset = ChunkedTrainDataset(
            filename_pattern=filename_pattern,
            max_length=max_length,
            batch_size=batch_size,
            process_rank=process_rank,
            num_processes=num_processes,
            max_epochs=max_epochs,
            tokenizer=tokenizer,
            num_workers=num_workers,
        )
        self.files = self._dataset.process_files

        self.dataloader = DataLoader(
            self._dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
        self._iterator = None
        self._exhausted = False

    def reset(self) -> None:
        """Reset the dataloader iterator."""
        self._iterator = iter(self.dataloader)
        self._exhausted = False

    def next_batch(self) -> torch.Tensor:
        """Get next batch of raw input_ids (batch_size, max_length) on CPU (pinned memory)."""
        if self._iterator is None:
            self.reset()

        try:
            return next(self._iterator)  # (batch_size, max_length)
        except StopIteration:
            self._exhausted = True
            return torch.empty(0, dtype=torch.int32)  # (0,)


class ChunkedEvalDataset(IterableDataset):
    """Chunk-aligned evaluation dataset. Same packing as training but:
    - All processes see all files (distributes by sequence, not file)
    - Single epoch only
    - Yields (batch_size, max_length) int32 raw input_ids
    """

    def __init__(
        self,
        filename_pattern: str,
        max_length: int,
        batch_size: int,
        process_rank: int,
        num_processes: int,
        tokenizer: EsmTokenizer | TokenIds,
    ) -> None:
        self.filename_pattern = filename_pattern
        self.max_length = max_length
        self.batch_size = batch_size
        self.process_rank = process_rank
        self.num_processes = num_processes
        token_ids = _coerce_token_ids(tokenizer)
        self.eos_token_id = token_ids.eos_token_id
        self.pad_token_id = token_ids.pad_token_id

        self.all_files = sorted(Path.cwd().glob(filename_pattern))
        assert len(self.all_files) > 0, f"No files found matching pattern: {filename_pattern}"

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Generate batches, with each process taking every num_processes-th batch."""
        batch_count = 0
        batch_chunks: list[torch.Tensor] = []  # each chunk: (max_length,)

        packer = ChunkPacker(self.max_length, self.eos_token_id, self.pad_token_id)
        for file in self.all_files:
            raw_tokens = _load_data_shard(file)  # (num_tokens,)
            for chunk in packer.pack(raw_tokens):  # chunk: (max_length,)
                batch_chunks.append(chunk.to(torch.int32))  # append (max_length,)
                if len(batch_chunks) == self.batch_size:
                    if batch_count % self.num_processes == self.process_rank:
                        yield torch.stack(batch_chunks)  # (batch_size, max_length)
                    batch_count += 1
                    batch_chunks = []

        # Drop partial batches to preserve the fixed batch shape.


class ChunkedEvalLoader:
    """Chunk-aligned evaluation loader.

    Yields (batch_size, max_length) int32 tensors of raw input_ids on CPU.
    Distributes data by sequence across processes.
    """

    def __init__(
        self,
        filename_pattern: str,
        max_length: int,
        micro_batch_tokens: int,
        process_rank: int,
        num_processes: int,
        tokenizer: EsmTokenizer | TokenIds,
    ) -> None:
        self.max_length = max_length
        batch_size = micro_batch_tokens // max_length
        assert batch_size >= 1, f"micro_batch_tokens ({micro_batch_tokens}) must be >= max_length ({max_length})"

        self._dataset = ChunkedEvalDataset(
            filename_pattern=filename_pattern,
            max_length=max_length,
            batch_size=batch_size,
            process_rank=process_rank,
            num_processes=num_processes,
            tokenizer=tokenizer,
        )
        self.files = self._dataset.all_files

        self.dataloader = DataLoader(
            self._dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=True,
        )
        self._iterator = None
        self._exhausted = False

    def reset(self) -> None:
        """Reset the dataloader iterator."""
        self._iterator = iter(self.dataloader)
        self._exhausted = False

    def next_batch(self) -> torch.Tensor:
        """Get next batch of raw input_ids (batch_size, max_length) on CPU."""
        if self._iterator is None:
            self.reset()

        try:
            return next(self._iterator)  # (batch_size, max_length)
        except StopIteration:
            self._exhausted = True
            return torch.empty(0, dtype=torch.int32)  # (0,)


def apply_masking_gpu(
    input_ids: torch.Tensor,
    special_tokens: torch.Tensor,
    mask_token_id: int,
    mask_rate: float,
    mlm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask nonspecial tokens on the input device.

    Args:
        input_ids: (batch_size, sequence_length) or (sequence_length,) token IDs.
        special_tokens: (num_special_tokens,) IDs to never mask (CLS, EOS, PAD).
        mask_token_id: Token ID to replace masked positions with
        mask_rate: Fixed MLM rate; diffusion samples a rate independently of this value.
        mlm: If True, use fixed mask_rate. If False, sample uniform rate (masked diffusion).

    Returns:
        noisy: input_ids with masked positions replaced by mask_token_id
        labels: original token IDs at masked positions, -100 elsewhere
        rate: Actual mask rate, shape () for MLM or (1,) for diffusion.
    """
    if mlm:
        rate = torch.tensor(mask_rate, device=input_ids.device, dtype=torch.float32)  # ()
    else:
        eps = 1e-3
        rate = torch.rand(1, device=input_ids.device) * (1 - eps) + eps  # (1,)

    mask_probs = torch.rand_like(input_ids, dtype=torch.float32)  # input_ids.shape
    mask_indices = mask_probs < rate  # input_ids.shape

    # Don't mask special tokens
    special_mask = torch.isin(input_ids, special_tokens)  # input_ids.shape
    mask_indices = mask_indices & ~special_mask  # input_ids.shape

    labels = input_ids.clone()  # input_ids.shape
    labels[~mask_indices] = -100  # labels shape unchanged
    noisy = torch.where(mask_indices, mask_token_id, input_ids)  # input_ids.shape
    return noisy, labels, rate  # input_ids.shape, input_ids.shape, () or (1,)


class AsyncBatchPipeline:
    """Double-buffered CUDA stream pipeline for overlapping H2D transfer with compute.

    Wraps a data loader that yields CPU tensors. Uses a background CUDA stream
    to transfer the next batch while the current batch is being processed on
    the default stream.
    """

    def __init__(self, loader: ChunkedTrainLoader | ChunkedEvalLoader) -> None:
        """
        Args:
            loader: A data loader with .next_batch() returning CPU tensors
                    and ._exhausted attribute.
        """
        self.loader = loader
        self.files = loader.files
        self.transfer_stream = torch.cuda.Stream()
        self._next_batch = None
        self._exhausted = False

    def reset(self) -> None:
        """Reset the underlying loader and pre-fetch the first batch."""
        self.loader.reset()
        self._exhausted = False
        self._next_batch = None
        self._prefetch()

    def _prefetch(self) -> None:
        """Transfer the next batch to GPU on the background stream."""
        raw = self.loader.next_batch()  # (batch_size, max_length) or (0,)
        if raw.numel() == 0:
            self._exhausted = True
            self._next_batch = None
            return
        with torch.cuda.stream(self.transfer_stream):
            self._next_batch = raw.cuda(non_blocking=True)  # (batch_size, max_length)

    def next_batch(self) -> torch.Tensor:
        """Return the pre-staged GPU batch and start transferring the next one.

        Returns:
            input_ids on GPU (batch_size, max_length) int32, or empty tensor if exhausted.
        """
        if self._next_batch is None:
            if self._exhausted:
                return torch.empty(0, dtype=torch.int32, device='cuda')  # (0,)
            self._prefetch()
            if self._next_batch is None:
                return torch.empty(0, dtype=torch.int32, device='cuda')  # (0,)

        consumer_stream = torch.cuda.current_stream()
        consumer_stream.wait_stream(self.transfer_stream)
        batch = self._next_batch  # (batch_size, max_length)
        # Keep transfer-stream storage alive until the consumer finishes using it.
        batch.record_stream(consumer_stream)  # (batch_size, max_length)

        self._prefetch()

        return batch  # (batch_size, max_length)
