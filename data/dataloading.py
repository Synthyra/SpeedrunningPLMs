import torch
import random
import torch.utils.data as data
from pathlib import Path
from transformers import EsmTokenizer
from typing import Tuple, Optional, List
from torch.utils.data import DataLoader, IterableDataset


def _load_data_shard(file: Path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)
    assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
    assert header[1] == 1, 'unsupported version'
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open('rb', buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint8)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == num_tokens, 'number of tokens read does not match header?'
    return tokens


class EvalLoader(IterableDataset):
    """An IterableDataset specifically for evaluation that distributes data by sequences, not files."""
    
    def __init__(
        self,
        filename_pattern: str,
        seq_len: int,
        process_rank: int,
        num_processes: int,
        tokenizer: EsmTokenizer,
    ):
        self.filename_pattern = filename_pattern
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        # Tokenizer IDs
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.special_tokens = [self.cls_token_id, self.eos_token_id, self.pad_token_id]
        
        # All processes load all files (since we're distributing by sequences, not files)
        self.all_files = sorted(Path.cwd().glob(filename_pattern))
        if not self.all_files:
            raise ValueError(f"No files found matching pattern: {filename_pattern}")
    
    def __iter__(self):
        """Generate batches, with each process taking every num_processes-th batch."""
        batch_count = 0
        
        for file in self.all_files:
            raw_tokens = _load_data_shard(file)
            
            # Process the tokens into batches
            eos_positions = (raw_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]
            
            if len(eos_positions) == 0:
                continue
            
            # Process samples and create batches
            batch_tokens = []
            curr_batch_len = 0
            
            for i in range(len(eos_positions)):
                curr_eos = eos_positions[i]
                prev_eos_plus_one = 0 if i == 0 else eos_positions[i-1] + 1
                sample = raw_tokens[prev_eos_plus_one:curr_eos+1]
                
                # Handle samples that exceed batch size
                if len(sample) > self.seq_len:
                    # Split large samples into multiple batches
                    for j in range(0, len(sample), self.seq_len):
                        chunk = sample[j:j+self.seq_len]
                        if len(chunk) < self.seq_len:
                            # Pad the last chunk
                            padding = torch.full((self.seq_len - len(chunk),), self.pad_token_id, dtype=torch.uint8)
                            chunk = torch.cat([chunk, padding])
                        
                        # Check if this batch should be yielded by this process
                        if batch_count % self.num_processes == self.process_rank:
                            # Apply masking and yield batch
                            input_ids, labels, mask_rate = self._apply_masking(chunk)
                            yield input_ids, labels, mask_rate
                        batch_count += 1
                    continue
                
                # Check if adding this sample would exceed batch size
                if len(sample) + curr_batch_len > self.seq_len:
                    # Pad current batch and yield
                    if curr_batch_len > 0:
                        padding = torch.full((self.seq_len - curr_batch_len,), self.pad_token_id, dtype=torch.uint8)
                        batch_tokens.append(padding)
                        batch = torch.cat(batch_tokens)
                        
                        # Check if this batch should be yielded by this process
                        if batch_count % self.num_processes == self.process_rank:
                            # Apply masking and yield
                            input_ids, labels, mask_rate = self._apply_masking(batch)
                            yield input_ids, labels, mask_rate
                        batch_count += 1
                    
                    # Start new batch
                    batch_tokens = [sample]
                    curr_batch_len = len(sample)
                else:
                    # Add to current batch
                    batch_tokens.append(sample)
                    curr_batch_len += len(sample)
                
                # Yield complete batch
                if curr_batch_len == self.seq_len:
                    batch = torch.cat(batch_tokens)
                    
                    # Check if this batch should be yielded by this process
                    if batch_count % self.num_processes == self.process_rank:
                        input_ids, labels, mask_rate = self._apply_masking(batch)
                        yield input_ids, labels, mask_rate
                    batch_count += 1
                    batch_tokens = []
                    curr_batch_len = 0
            
            # Yield final incomplete batch if it exists
            if curr_batch_len > 0:
                padding = torch.full((self.seq_len - curr_batch_len,), self.pad_token_id, dtype=torch.uint8)
                batch_tokens.append(padding)
                batch = torch.cat(batch_tokens)
                
                # Check if this batch should be yielded by this process
                if batch_count % self.num_processes == self.process_rank:
                    input_ids, labels, mask_rate = self._apply_masking(batch)
                    yield input_ids, labels, mask_rate
                batch_count += 1
    
    def _apply_masking(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply masking to a sequence (on CPU)."""
        # Convert to int32
        sequence = sequence.to(dtype=torch.int32)
        
        # Use fixed mask rate for evaluation
        mask_rate = torch.full((1,), 0.15)
        
        # Create mask
        p_mask = mask_rate.repeat(len(sequence))
        mask_indices = torch.rand(len(sequence)) < p_mask
        
        # Don't mask special tokens
        special_mask = torch.isin(sequence, torch.tensor(self.special_tokens, dtype=torch.int32))
        mask_indices = mask_indices & ~special_mask
        
        # Create noisy batch and labels
        noisy_batch = torch.where(mask_indices, self.mask_token_id, sequence)
        labels = sequence.clone()
        labels[~mask_indices] = -100
        
        return noisy_batch, labels, mask_rate


class OptimizedEvalLoader:
    """Drop-in replacement for evaluation that distributes data by sequences rather than files."""
    
    def __init__(
        self,
        filename_pattern: str,
        seq_len: int,
        process_rank: int,
        num_processes: int,
        tokenizer: EsmTokenizer,
    ):
        self.filename_pattern = filename_pattern
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        # Create the dataset
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
        
        # Create iterator
        self._iterator = None
        self._exhausted = False
    
    def reset(self):
        """Reset the dataloader iterator."""
        self._iterator = iter(self.dataloader)
        self._exhausted = False
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the next batch, ensuring GPU transfer happens here."""
        if self._iterator is None:
            self.reset()
        
        try:
            input_ids, labels, mask_rate = next(self._iterator)
            # Transfer to GPU with non-blocking
            input_ids = input_ids.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            mask_rate = mask_rate.cuda(non_blocking=True)
            return input_ids, labels, mask_rate
        except StopIteration:
            self._exhausted = True
            # Return empty tensors to signal end of data
            return torch.empty(0, device='cuda'), torch.empty(0, device='cuda'), torch.empty(0, device='cuda')


class TrainLoader(IterableDataset):
    """An IterableDataset that handles distributed padded data loading with masking."""
    
    def __init__(
        self,
        filename_pattern: str,
        seq_len: int,
        process_rank: int,
        num_processes: int,
        max_epochs: int,
        tokenizer: EsmTokenizer,
        num_workers: int = 1,
        mlm: bool = False,
        mask_rate: float = 0.15,
    ):
        self.filename_pattern = filename_pattern
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.max_epochs = max_epochs
        self.num_workers = num_workers
        self.mask_rate = mask_rate
        # Tokenizer IDs
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
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

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:
            # Single worker mode
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
        leftover_tokens = torch.empty(0, dtype=torch.uint8)
        
        while epoch < self.max_epochs:
            # Shuffle files at the start of each epoch
            if file_idx == 0 and epoch > 0:
                # Include process rank for proper distributed shuffling
                random.seed(epoch + self.process_rank * 10000 + worker_id * 1000)
                random.shuffle(worker_files)
            
            # Load current file
            if file_idx < len(worker_files):
                raw_tokens = _load_data_shard(worker_files[file_idx])
                raw_tokens = torch.cat([leftover_tokens, raw_tokens], dim=0)
                file_idx += 1
            else:
                # End of epoch
                if leftover_tokens.numel() == 0:
                    epoch += 1
                    file_idx = 0
                    continue
                raw_tokens = leftover_tokens
                leftover_tokens = torch.empty(0, dtype=torch.uint8)
            
            # Process the tokens into batches
            eos_positions = (raw_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]
            
            if len(eos_positions) == 0:
                leftover_tokens = raw_tokens
                if file_idx >= len(worker_files):
                    epoch += 1
                    file_idx = 0
                continue
            
            # Process samples and create batches
            batch_tokens = []
            curr_batch_len = 0
            
            for i in range(len(eos_positions)):
                curr_eos = eos_positions[i]
                prev_eos_plus_one = 0 if i == 0 else eos_positions[i-1] + 1
                sample = raw_tokens[prev_eos_plus_one:curr_eos+1]
                
                # Handle samples that exceed batch size
                if len(sample) > self.seq_len:
                    # Split large samples into multiple batches
                    for j in range(0, len(sample), self.seq_len):
                        chunk = sample[j:j+self.seq_len]
                        if len(chunk) < self.seq_len:
                            # Pad the last chunk
                            padding = torch.full((self.seq_len - len(chunk),), self.pad_token_id, dtype=torch.uint8)
                            chunk = torch.cat([chunk, padding])
                        
                        # Apply masking and yield batch
                        input_ids, labels, mask_rate = self._apply_masking(chunk)
                        yield input_ids, labels, mask_rate
                    continue
                
                # Check if adding this sample would exceed batch size
                if len(sample) + curr_batch_len > self.seq_len:
                    # Pad current batch and yield
                    if curr_batch_len > 0:
                        padding = torch.full((self.seq_len - curr_batch_len,), self.pad_token_id, dtype=torch.uint8)
                        batch_tokens.append(padding)
                        batch = torch.cat(batch_tokens)
                        
                        # Apply masking and yield
                        input_ids, labels, mask_rate = self._apply_masking(batch)
                        yield input_ids, labels, mask_rate
                    
                    # Start new batch
                    batch_tokens = [sample]
                    curr_batch_len = len(sample)
                else:
                    # Add to current batch
                    batch_tokens.append(sample)
                    curr_batch_len += len(sample)
                
                # Yield complete batch
                if curr_batch_len == self.seq_len:
                    batch = torch.cat(batch_tokens)
                    input_ids, labels, mask_rate = self._apply_masking(batch)
                    yield input_ids, labels, mask_rate
                    batch_tokens = []
                    curr_batch_len = 0
            
            # Save leftover tokens for next file
            if len(eos_positions) > 0:
                leftover_tokens = raw_tokens[eos_positions[-1]+1:]
            
            # Yield final incomplete batch if at end of epoch
            if file_idx >= len(worker_files) and curr_batch_len > 0:
                padding = torch.full((self.seq_len - curr_batch_len,), self.pad_token_id, dtype=torch.uint8)
                batch_tokens.append(padding)
                batch = torch.cat(batch_tokens)
                input_ids, labels, mask_rate = self._apply_masking(batch)
                yield input_ids, labels, mask_rate
                
                epoch += 1
                file_idx = 0
    
    def _apply_masking(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply masking to a sequence (on CPU)."""
        # Convert to int32
        sequence = sequence.to(dtype=torch.int32)
        
        # Pick mask rate
        if self.mlm:
            mask_rate = torch.full((1,), self.mask_rate)
        else:
            eps = 1e-3
            mask_rate = torch.rand(1)
            mask_rate = (1 - eps) * mask_rate + eps
        
        # Create mask
        p_mask = mask_rate.repeat(len(sequence))
        mask_indices = torch.rand(len(sequence)) < p_mask
        
        # Don't mask special tokens
        special_mask = torch.isin(sequence, torch.tensor(self.special_tokens, dtype=torch.int32))
        mask_indices = mask_indices & ~special_mask
        
        # Create noisy batch and labels
        noisy_batch = torch.where(mask_indices, self.mask_token_id, sequence)
        labels = sequence.clone()
        labels[~mask_indices] = -100
        
        return noisy_batch, labels, mask_rate


class OptimizedTrainLoader:
    """Drop-in replacement for DistributedPaddedDataLoader using multi-worker optimization."""
    
    def __init__(
        self,
        filename_pattern: str,
        seq_len: int,
        process_rank: int,
        num_processes: int,
        max_epochs: int,
        tokenizer: EsmTokenizer,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        mlm: bool = False,
        mask_rate: float = 0.15,
    ):
        self.filename_pattern = filename_pattern
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.mlm = mlm
        self.mask_rate = mask_rate
        
        # Create the dataset to get file count
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
        
        # Create the optimized dataloader
        self.dataloader = DataLoader(
            self._dataset,
            batch_size=None,  # Dataset returns complete batches
            num_workers=num_workers,
            pin_memory=True,  # Pin memory for faster GPU transfer
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        )
        
        # Create iterator
        self._iterator = None
        self._exhausted = False

    def set_mask_rate(self, mask_rate: float):
        """Set the mask rate for the next batch(es)."""
        self.mask_rate = mask_rate
        self._dataset.mask_rate = mask_rate

    def set_mlm(self, mlm: bool):
        """Set whether to use MLM masking in the dataset."""
        self.mlm = mlm
        self._dataset.mlm = mlm

    def reset(self):
        """Reset the dataloader iterator."""
        self._iterator = iter(self.dataloader)
        self._exhausted = False
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the next batch, ensuring GPU transfer happens here."""
        if self._iterator is None:
            self.reset()
        
        try:
            input_ids, labels, mask_rate = next(self._iterator)
            # Transfer to GPU with non-blocking
            input_ids = input_ids.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            mask_rate = mask_rate.cuda(non_blocking=True)
            return input_ids, labels, mask_rate
        except StopIteration:
            self._exhausted = True
            # Return empty tensors to signal end of data
            return torch.empty(0, device='cuda'), torch.empty(0, device='cuda'), torch.empty(0, device='cuda')


# ========================================================================================
# Chunk-aligned data loaders (new for batched UNet + GPU-side masking)
# ========================================================================================


class ChunkedTrainDataset(IterableDataset):
    """Chunk-aligned IterableDataset that packs documents into fixed-length chunks.

    Each chunk is exactly max_length tokens with documents packed end-to-end.
    No document spans a chunk boundary. If a document doesn't fit in the current
    chunk, the remainder is padded and a new chunk starts. Documents exceeding
    max_length are truncated to their own chunk.

    Yields batches of (B, max_length) int32 tensors containing raw input_ids
    (no masking applied -- masking is done on GPU in the training loop).
    """

    def __init__(
        self,
        filename_pattern: str,
        max_length: int,
        batch_size: int,
        process_rank: int,
        num_processes: int,
        max_epochs: int,
        tokenizer: EsmTokenizer,
        num_workers: int = 1,
    ):
        self.filename_pattern = filename_pattern
        self.max_length = max_length
        self.batch_size = batch_size
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.max_epochs = max_epochs
        self.num_workers = num_workers
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

        all_files = sorted(Path.cwd().glob(filename_pattern))
        assert len(all_files) > 0, f"No files found matching pattern: {filename_pattern}"

        # Distribute files across processes (GPUs)
        files_per_process = len(all_files) // num_processes
        extra = len(all_files) % num_processes
        start = process_rank * files_per_process + min(process_rank, extra)
        end = start + files_per_process + (1 if process_rank < extra else 0)
        self.process_files = all_files[start:end]

    def _pack_chunks(self, raw_tokens: torch.Tensor):
        """Pack raw tokens into max_length-aligned chunks.

        Documents are delineated by EOS tokens. Each chunk contains one or more
        complete documents, padded at the end if needed.

        Yields individual (max_length,) uint8 chunks.
        """
        eos_positions = (raw_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) == 0:
            return

        chunk_parts: List[torch.Tensor] = []
        chunk_len = 0

        prev_start = 0
        for i in range(len(eos_positions)):
            curr_eos = eos_positions[i].item()
            doc = raw_tokens[prev_start:curr_eos + 1]
            prev_start = curr_eos + 1
            doc_len = len(doc)

            if doc_len > self.max_length:
                # Flush current chunk if it has data
                if chunk_len > 0:
                    padding = torch.full((self.max_length - chunk_len,), self.pad_token_id, dtype=torch.uint8)
                    yield torch.cat(chunk_parts + [padding])
                    chunk_parts = []
                    chunk_len = 0
                # Truncate oversized document to its own chunk
                yield doc[:self.max_length].clone()
                continue

            if doc_len + chunk_len > self.max_length:
                # Doc doesn't fit: pad and yield current chunk
                padding = torch.full((self.max_length - chunk_len,), self.pad_token_id, dtype=torch.uint8)
                yield torch.cat(chunk_parts + [padding])
                chunk_parts = []
                chunk_len = 0

            chunk_parts.append(doc)
            chunk_len += doc_len

            if chunk_len == self.max_length:
                yield torch.cat(chunk_parts)
                chunk_parts = []
                chunk_len = 0

        # Yield remaining chunk if any
        if chunk_len > 0:
            padding = torch.full((self.max_length - chunk_len,), self.pad_token_id, dtype=torch.uint8)
            yield torch.cat(chunk_parts + [padding])

    def __iter__(self):
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
        leftover_tokens = torch.empty(0, dtype=torch.uint8)
        batch_chunks: List[torch.Tensor] = []

        while epoch < self.max_epochs:
            file_idx = 0

            if epoch > 0:
                random.seed(epoch + self.process_rank * 10000 + worker_id * 1000)
                random.shuffle(worker_files)

            while file_idx < len(worker_files):
                raw_tokens = _load_data_shard(worker_files[file_idx])
                raw_tokens = torch.cat([leftover_tokens, raw_tokens])
                file_idx += 1

                # Find last complete document
                eos_positions = (raw_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) == 0:
                    leftover_tokens = raw_tokens
                    continue

                last_eos_pos = eos_positions[-1].item()
                leftover_tokens = raw_tokens[last_eos_pos + 1:]
                complete_tokens = raw_tokens[:last_eos_pos + 1]

                for chunk in self._pack_chunks(complete_tokens):
                    batch_chunks.append(chunk.to(torch.int32))
                    if len(batch_chunks) == self.batch_size:
                        yield torch.stack(batch_chunks)  # (B, max_length)
                        batch_chunks = []

            # End of epoch: drop incomplete batch, reset
            leftover_tokens = torch.empty(0, dtype=torch.uint8)
            batch_chunks = []
            epoch += 1


class ChunkedTrainLoader:
    """Chunk-aligned training data loader.

    Yields (B, max_length) int32 tensors of raw input_ids on CPU (pinned memory).
    No masking applied -- masking is handled on GPU in the training loop.
    """

    def __init__(
        self,
        filename_pattern: str,
        max_length: int,
        micro_batch_tokens: int,
        process_rank: int,
        num_processes: int,
        max_epochs: int,
        tokenizer: EsmTokenizer,
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ):
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
            persistent_workers=True if num_workers > 0 else False,
        )
        self._iterator = None
        self._exhausted = False

    def reset(self):
        """Reset the dataloader iterator."""
        self._iterator = iter(self.dataloader)
        self._exhausted = False

    def next_batch(self) -> torch.Tensor:
        """Get next batch of raw input_ids (B, max_length) on CPU (pinned memory)."""
        if self._iterator is None:
            self.reset()

        try:
            return next(self._iterator)
        except StopIteration:
            self._exhausted = True
            return torch.empty(0, dtype=torch.int32)


class ChunkedEvalDataset(IterableDataset):
    """Chunk-aligned evaluation dataset. Same packing as training but:
    - All processes see all files (distributes by sequence, not file)
    - Single epoch only
    - Yields (B, max_length) int32 raw input_ids
    """

    def __init__(
        self,
        filename_pattern: str,
        max_length: int,
        batch_size: int,
        process_rank: int,
        num_processes: int,
        tokenizer: EsmTokenizer,
    ):
        self.filename_pattern = filename_pattern
        self.max_length = max_length
        self.batch_size = batch_size
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

        self.all_files = sorted(Path.cwd().glob(filename_pattern))
        assert len(self.all_files) > 0, f"No files found matching pattern: {filename_pattern}"

    def __iter__(self):
        """Generate batches, with each process taking every num_processes-th batch."""
        batch_count = 0
        batch_chunks: List[torch.Tensor] = []

        for file in self.all_files:
            raw_tokens = _load_data_shard(file)

            eos_positions = (raw_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) == 0:
                continue

            chunk_parts: List[torch.Tensor] = []
            chunk_len = 0
            prev_start = 0

            for i in range(len(eos_positions)):
                curr_eos = eos_positions[i].item()
                doc = raw_tokens[prev_start:curr_eos + 1]
                prev_start = curr_eos + 1
                doc_len = len(doc)

                if doc_len > self.max_length:
                    if chunk_len > 0:
                        padding = torch.full((self.max_length - chunk_len,), self.pad_token_id, dtype=torch.uint8)
                        batch_chunks.append(torch.cat(chunk_parts + [padding]).to(torch.int32))
                        chunk_parts = []
                        chunk_len = 0
                        if len(batch_chunks) == self.batch_size:
                            if batch_count % self.num_processes == self.process_rank:
                                yield torch.stack(batch_chunks)
                            batch_count += 1
                            batch_chunks = []
                    batch_chunks.append(doc[:self.max_length].clone().to(torch.int32))
                    if len(batch_chunks) == self.batch_size:
                        if batch_count % self.num_processes == self.process_rank:
                            yield torch.stack(batch_chunks)
                        batch_count += 1
                        batch_chunks = []
                    continue

                if doc_len + chunk_len > self.max_length:
                    padding = torch.full((self.max_length - chunk_len,), self.pad_token_id, dtype=torch.uint8)
                    batch_chunks.append(torch.cat(chunk_parts + [padding]).to(torch.int32))
                    chunk_parts = []
                    chunk_len = 0
                    if len(batch_chunks) == self.batch_size:
                        if batch_count % self.num_processes == self.process_rank:
                            yield torch.stack(batch_chunks)
                        batch_count += 1
                        batch_chunks = []

                chunk_parts.append(doc)
                chunk_len += doc_len

                if chunk_len == self.max_length:
                    batch_chunks.append(torch.cat(chunk_parts).to(torch.int32))
                    chunk_parts = []
                    chunk_len = 0
                    if len(batch_chunks) == self.batch_size:
                        if batch_count % self.num_processes == self.process_rank:
                            yield torch.stack(batch_chunks)
                        batch_count += 1
                        batch_chunks = []

            # Flush remaining chunk from this file
            if chunk_len > 0:
                padding = torch.full((self.max_length - chunk_len,), self.pad_token_id, dtype=torch.uint8)
                batch_chunks.append(torch.cat(chunk_parts + [padding]).to(torch.int32))
                if len(batch_chunks) == self.batch_size:
                    if batch_count % self.num_processes == self.process_rank:
                        yield torch.stack(batch_chunks)
                    batch_count += 1
                    batch_chunks = []

        # Drop partial batches to maintain fixed (B, max_length) shape


class ChunkedEvalLoader:
    """Chunk-aligned evaluation loader.

    Yields (B, max_length) int32 tensors of raw input_ids on CPU.
    Distributes data by sequence across processes.
    """

    def __init__(
        self,
        filename_pattern: str,
        max_length: int,
        micro_batch_tokens: int,
        process_rank: int,
        num_processes: int,
        tokenizer: EsmTokenizer,
    ):
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

    def reset(self):
        """Reset the dataloader iterator."""
        self._iterator = iter(self.dataloader)
        self._exhausted = False

    def next_batch(self) -> torch.Tensor:
        """Get next batch of raw input_ids (B, max_length) on CPU."""
        if self._iterator is None:
            self.reset()

        try:
            return next(self._iterator)
        except StopIteration:
            self._exhausted = True
            return torch.empty(0, dtype=torch.int32)


def apply_masking_gpu(
    input_ids: torch.Tensor,
    special_tokens: torch.Tensor,
    mask_token_id: int,
    mask_rate: float,
    mlm: bool = False,
):
    """Apply masking on GPU -- much faster than CPU, no worker sync issues.

    Args:
        input_ids: (B, L) or (L,) raw token IDs on GPU
        special_tokens: 1D tensor of token IDs to never mask (CLS, EOS, PAD)
        mask_token_id: Token ID to replace masked positions with
        mask_rate: Maximum mask rate (for MLM, used directly; for MD, sampled uniformly)
        mlm: If True, use fixed mask_rate. If False, sample uniform rate (masked diffusion).

    Returns:
        noisy: input_ids with masked positions replaced by mask_token_id
        labels: original token IDs at masked positions, -100 elsewhere
        rate: scalar tensor of the actual mask rate used
    """
    if mlm:
        rate = torch.tensor(mask_rate, device=input_ids.device, dtype=torch.float32)
    else:
        eps = 1e-3
        rate = torch.rand(1, device=input_ids.device) * (1 - eps) + eps

    mask_probs = torch.rand_like(input_ids, dtype=torch.float32)
    mask_indices = mask_probs < rate

    # Don't mask special tokens
    special_mask = torch.isin(input_ids, special_tokens)
    mask_indices = mask_indices & ~special_mask

    labels = input_ids.clone()
    labels[~mask_indices] = -100
    noisy = torch.where(mask_indices, mask_token_id, input_ids)
    return noisy, labels, rate


class AsyncBatchPipeline:
    """Double-buffered CUDA stream pipeline for overlapping H2D transfer with compute.

    Wraps a data loader that yields CPU tensors. Uses a background CUDA stream
    to transfer the next batch while the current batch is being processed on
    the default stream.
    """

    def __init__(self, loader):
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

    def reset(self):
        """Reset the underlying loader and pre-fetch the first batch."""
        self.loader.reset()
        self._exhausted = False
        self._next_batch = None
        self._prefetch()

    def _prefetch(self):
        """Transfer the next batch to GPU on the background stream."""
        raw = self.loader.next_batch()
        if raw.numel() == 0:
            self._exhausted = True
            self._next_batch = None
            return
        with torch.cuda.stream(self.transfer_stream):
            self._next_batch = raw.cuda(non_blocking=True)

    def next_batch(self) -> torch.Tensor:
        """Return the pre-staged GPU batch and start transferring the next one.

        Returns:
            input_ids on GPU (B, max_length) int32, or empty tensor if exhausted.
        """
        if self._next_batch is None:
            if self._exhausted:
                return torch.empty(0, dtype=torch.int32, device='cuda')
            self._prefetch()
            if self._next_batch is None:
                return torch.empty(0, dtype=torch.int32, device='cuda')

        # Wait for the transfer to complete
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        batch = self._next_batch

        # Start prefetching the next batch
        self._prefetch()

        return batch
