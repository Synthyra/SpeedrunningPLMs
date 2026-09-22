from dataclasses import dataclass
from typing import Protocol


class TokenizerIds(Protocol):
    """Special-token attributes consumed by the data loaders."""

    cls_token_id: int
    eos_token_id: int
    pad_token_id: int
    mask_token_id: int


@dataclass(frozen=True)
class TokenIds:
    cls_token_id: int
    eos_token_id: int
    pad_token_id: int
    mask_token_id: int

    @classmethod
    def from_tokenizer(cls, tokenizer: TokenizerIds) -> "TokenIds":
        return cls(
            cls_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id,
        )
