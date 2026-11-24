"""Two fixed facts about the 20 residues, in the benchmark's residue order (`benchmark.tokens.STANDARD_RESIDUES`), for the
strategies that start from what is already known.

`UNIGRAM`: each residue's share of the 389,084,153 residues in the training stream at `fb3029000b880e15be1fdc9512fa59d84d3a75bc`.
`BLOSUM62`: Henikoff and Henikoff's half-bit scores, from Biopython's `Bio/Align/substitution_matrices/data/BLOSUM62`.

Symbols: c = 20 residues.
"""

from __future__ import annotations

import torch

from torch import Tensor


UNIGRAM = (0.08760, 0.01385, 0.05824, 0.06550, 0.04036, 0.06898, 0.02237, 0.05602, 0.05197, 0.09797, 0.02259, 0.04105, 0.04635, 0.03717,
           0.05809, 0.06478, 0.05434, 0.06505, 0.01510, 0.03262)
BLOSUM62 = (
    ( 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2),  # A
    ( 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2),  # C
    (-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3),  # D
    (-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2),  # E
    (-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3),  # F
    ( 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3),  # G
    (-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2),  # H
    (-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1),  # I
    (-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2),  # K
    (-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1),  # L
    (-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1),  # M
    (-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2),  # N
    (-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3),  # P
    (-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1),  # Q
    (-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2),  # R
    ( 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2),  # S
    ( 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2),  # T
    ( 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1),  # V
    (-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2),  # W
    (-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7),  # Y
)


def unigram() -> Tensor:
    """The training stream's residue shares, normalized over their rounding."""
    shares = torch.tensor(UNIGRAM, dtype=torch.float32)  # (c,)
    return shares / shares.sum()  # (c,)


def substitution_targets() -> Tensor:
    """Row a is P(b | a) proportional to p_b 2^(S_ab / 2): what BLOSUM62's half-bit log-odds imply from the training shares."""
    scores = torch.tensor(BLOSUM62, dtype=torch.float32)  # (c, c)
    joint = unigram()[None, :] * 2.0 ** (scores / 2)  # (c, c)
    return joint / joint.sum(dim=1, keepdim=True)  # (c, c)
