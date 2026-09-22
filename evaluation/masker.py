"""Standardized protein masked-language-model corruption."""

from __future__ import annotations

import torch
import torch.nn as nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class ProteinMasker(nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, mask_rate: float = 0.15) -> None:
        super().__init__()
        self.mask_token_id = tokenizer.mask_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.mask_rate = mask_rate

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked input IDs and labels with unmasked positions ignored."""
        # input_ids, attention_mask: (b, l).
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)  # (b, l)

        mask_probabilities = torch.full(
            (batch_size, seq_len),
            self.mask_rate,
            device=device,
        )  # (b, l)
        mask_indices = torch.rand(batch_size, seq_len, device=device) < mask_probabilities  # (b, l)

        cls_mask = input_ids == self.cls_token_id  # (b, l)
        eos_mask = input_ids == self.eos_token_id  # (b, l)
        mask_indices = mask_indices & ~cls_mask & ~eos_mask & attention_mask.bool()  # (b, l)

        # Avoid empty-label batches for short sequences and small batch sizes.
        for row in range(batch_size):
            if not mask_indices[row].any() and attention_mask[row].sum() > 2:
                valid_positions = (
                    ~cls_mask[row]
                    & ~eos_mask[row]
                    & attention_mask[row].bool()
                )  # (l)
                if valid_positions.any():
                    candidates = valid_positions.nonzero(as_tuple=True)[0]  # (n_candidates)
                    selected = candidates[
                        torch.randint(candidates.numel(), (1,), device=device)
                    ]  # (1)
                    mask_indices[row, selected] = True  # (b, l)

        masked_input_ids = torch.where(mask_indices, self.mask_token_id, input_ids)  # (b, l)
        labels = input_ids.clone()  # (b, l)
        labels[~mask_indices | (attention_mask == 0)] = -100  # (b, l)
        return masked_input_ids, labels  # (b, l), (b, l)
