"""The ESM2 alphabet the benchmark tokenizes with, the 20-residue label space every model is scored on, and the 3Di states.

Token ids follow ESM-1b and ESM-2 (github.com/facebookresearch/esm/blob/main/esm/constants.py). The 20 standard residues
are scored as classes 0 to 19 in alphabetical order of their one-letter codes; a model's logits over its own vocabulary
are gathered at its 20 residue columns before the softmax, so every model faces the same 20-way problem.
"""

from __future__ import annotations

import torch

from torch import Tensor


CLS_ID = 0
PAD_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 32
VOCAB_SIZE = 33
ESM2_TOKENS = ("<cls>", "<pad>", "<eos>", "<unk>", *"LAGVSERTIDPKQNFYMHWCXBUZO.-", "<null_1>", "<mask>")
ESM2_IDS = {token: index for index, token in enumerate(ESM2_TOKENS)}

STANDARD_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"  # the 20 classes, in this order
RESIDUE_CLASS = {residue: index for index, residue in enumerate(STANDARD_RESIDUES)}
ESM2_RESIDUE_COLUMNS = torch.tensor([ESM2_IDS[residue] for residue in STANDARD_RESIDUES], dtype=torch.long)  # (20,)
ESM2_RESIDUE_IDS = frozenset(ESM2_RESIDUE_COLUMNS.tolist())
NUM_CLASSES = len(STANDARD_RESIDUES)  # c = 20

THREE_DI_STATES = "ACDEFGHIKLMNPQRSTVWY"  # Foldseek's 3Di alphabet, 20 states
THREE_DI_CLASS = {state: index for index, state in enumerate(THREE_DI_STATES)}
NO_STATE = 255  # the 3Di value of a position that is not a residue (CLS, EOS, PAD)

IGNORE_LABEL = -100


def encode_residues(sequence: str) -> list[int]:
    """ESM2 ids of a protein's residues, without CLS or EOS; only the 20 standard residues are accepted."""
    residues = sequence.strip().upper()
    unknown = set(residues).difference(STANDARD_RESIDUES)
    if not residues or unknown:
        raise ValueError(f"sequence must be nonempty and standard residues only; found {sorted(unknown)}")
    return [ESM2_IDS[residue] for residue in residues]


def encode_three_di(states: str, length: int) -> list[int]:
    """Class ids of a Foldseek 3Di string, which must be as long as its sequence."""
    states = states.strip().upper()
    unknown = set(states).difference(THREE_DI_STATES)
    if len(states) != length or unknown:
        raise ValueError(f"3Di string must have {length} states from the 3Di alphabet; found {sorted(unknown)}")
    return [THREE_DI_CLASS[state] for state in states]


def residue_labels(input_ids: Tensor) -> Tensor:
    """The 20-way class of each ESM2 token, IGNORE_LABEL where the token is not a standard residue."""
    # input_ids: (...); the return has the same shape.
    lookup = torch.full((VOCAB_SIZE,), IGNORE_LABEL, dtype=torch.long)  # (33,)
    lookup[ESM2_RESIDUE_COLUMNS] = torch.arange(NUM_CLASSES)  # (33,)
    return lookup.to(input_ids.device)[input_ids]  # (...)


def is_residue(input_ids: Tensor) -> Tensor:
    """True where an ESM2 token is one of the 20 standard residues."""
    # input_ids: (...); the return is bool of the same shape.
    lookup = torch.zeros(VOCAB_SIZE, dtype=torch.bool)  # (33,)
    lookup[ESM2_RESIDUE_COLUMNS] = True  # (33,)
    return lookup.to(input_ids.device)[input_ids]  # (...)
