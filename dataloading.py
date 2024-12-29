import torch
from pathlib import Path


def _peek_data_shard(file: Path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    return int(header[2]) # number of tokens (claimed)


def _load_data_shard(path: Path, num_tokens):
    with path.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint8, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == num_tokens, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoaderFlex:
    def __init__(self, filename_pattern, batch_size, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.batch_size = batch_size

        # glob files that match the pattern
        self.files = sorted(Path.cwd().glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        self.files_num_tokens = [_peek_data_shard(file) for file in self.files]
        self.total_num_tokens = sum(self.files_num_tokens)

        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.batch_size
        self.tokens = _load_data_shard(self.files[self.current_shard], self.files_num_tokens[self.current_shard])

    def next_batch(self):
        batch_size = self.batch_size * self.num_processes
        input_ids = self.tokens[self.current_position:self.current_position+self.batch_size]
        # host side async is sufficient;
        # no performance improvement was observed when introducing a separate stream.
        input_ids = input_ids.to(device="cuda", dtype=torch.int32, non_blocking=True) # inputs
        # advance current position and load next shard if necessary
        self.current_position += batch_size
        if self.current_position + batch_size >= len(self.tokens):
            self.advance()
        return input_ids


class DistributedDataLoaderSDPA:
    def __init__(
        self, 
        filename_pattern: str, 
        batch_size: int, 
        process_rank: int, 
        num_processes: int, 
        cls_token_id: int,
        pad_token_id: int,
        max_length: int = 1024
    ):
        """
        Args:
            filename_pattern (str): Glob-like pattern for your .bin data shards.
            batch_size (int):      Number of sequences *per process* per batch.
            process_rank (int):    Rank of this process in [0..num_processes-1].
            num_processes (int):   Total number of processes.
            cls_token_id (int):    The token ID that marks the start of each sequence.
            max_length (int):      If provided, sequences longer than max_length
                                   will be truncated. Defaults to None (no truncation).
        """
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.cls_token_id = cls_token_id
        self.pad_token_id = pad_token_id
        self.max_length = max_length

        # Glob files that match the pattern
        self.files = sorted(Path.cwd().glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files matching: {filename_pattern}"

        # Inspect all data shards for total token count
        self.files_num_tokens = [_peek_data_shard(file) for file in self.files]
        self.total_num_tokens = sum(self.files_num_tokens)

        # Shard iteration state
        self.current_shard = -1
        self.sequences = []
        self.current_sequence_idx = 0

        self.reset()

    def reset(self):
        """Reset loader to the first shard and start from zero sequence index."""
        self.current_shard = -1
        self.advance()

    def advance(self):
        """Advance (cyclically) to the next data shard, load it, and split by CLS."""
        self.current_shard = (self.current_shard + 1) % len(self.files)

        # Load the entire shard (on CPU, pinned memory)
        tokens = _load_data_shard(
            self.files[self.current_shard],
            self.files_num_tokens[self.current_shard],
        )

        # Split into sequences by cls_token_id
        self.sequences = self._split_by_cls(tokens, self.cls_token_id)
        self.current_sequence_idx = 0

    def _split_by_cls(self, tokens: torch.Tensor, cls_token_id: int):
        """
        Given a flat 1D tensor of token IDs that contains multiple 
        sequences separated by `cls_token_id`, split and return a 
        list of 1D Tensors (uint8) for each sequence.
        """
        cls_positions = (tokens == cls_token_id).nonzero(as_tuple=True)[0].tolist()
        if not cls_positions:
            # If no CLS token found, treat entire shard as one sequence
            return [tokens]

        sequences = []
        for i in range(len(cls_positions)):
            start = cls_positions[i]
            end = cls_positions[i + 1] if (i + 1) < len(cls_positions) else len(tokens)
            seq = tokens[start:end]
            sequences.append(seq)
        return sequences

    def _pad_and_mask(self, list_of_sequences):
        """
        Given a list of variable-length 1D Tensors (dtype=uint8),
        optionally truncate to `max_length`, then pad to the same length
        and produce an attention mask.

        Returns:
          input_ids:      (batch_size, padded_length) int32
          attention_mask: (batch_size, padded_length) int32
        """
        # Truncate each sequence if needed
        list_of_sequences = [seq[:self.max_length] for seq in list_of_sequences]

        # Find the maximum length among the truncated sequences
        max_len = max(s.size(0) for s in list_of_sequences)
        batch_size = len(list_of_sequences)

        # Allocate int32 CPU Tensors for input_ids and attention_mask
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.int32)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.int32)

        # Copy data over and set mask = 1 for real tokens
        for i, seq in enumerate(list_of_sequences):
            length = seq.size(0)
            # We copy (uint8 -> int32) here
            input_ids[i, :length] = seq.to(dtype=torch.int32)
            attention_mask[i, :length] = 1

        return input_ids, attention_mask

    def next_batch(self):
        """
        Return (input_ids, attention_mask) of shape (b, seq_len) each, 
        where b = self.batch_size for THIS process. 
        """
        # If we've nearly exhausted the current shard, move to the next one
        if self.current_sequence_idx + self.batch_size > len(self.sequences):
            self.advance()

        # Fetch the next self.batch_size sequences
        start = self.current_sequence_idx
        end = start + self.batch_size
        local_sequences = self.sequences[start:end]
        self.current_sequence_idx += self.batch_size

        # Pad, build attention mask (on CPU)
        input_ids, attention_mask = self._pad_and_mask(local_sequences)

        # Move to GPU (int32), non_blocking for speed
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        attention_mask = attention_mask.to(device="cuda", non_blocking=True)

        return input_ids, attention_mask
