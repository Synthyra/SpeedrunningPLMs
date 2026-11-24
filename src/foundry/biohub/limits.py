"""Biohub's published limits and what a `fold_all_atom` request costs against them.

folding_service measured the limits against the live API on 2026-08-17 (`docs/options.md`, "Biohub Hard
Limits"). The bill was measured again on 2026-09-30 from eight folds of 674 to 764 residues, which showed the
documented charge of 25 tokens a residue is a floor: past 500 residues the service reports
`ceil(length ** 2 / 20)` tokens, which is 29,185 for a 764 residue complex. Both clients once kept their own
limits, and sequence_gradient's estimate counted residues alone, so its token budget never bound.
"""

from __future__ import annotations


# The sum of every chain's length; one residue more returns HTTP 422.
MAX_COMPLEX_LENGTH = 768
# `fold_all_atom`, the same for esmfold2-2026-05 and esmfold2-fast-2026-05.
PUBLISHED_MAX_RPM = 40
PUBLISHED_MAX_TPM = 400_000
# The default daily allotment of one token, shared by every caller that uses it, and the tokens a credit buys.
DAILY_CREDITS = 20_000
TOKENS_PER_CREDIT = 10_000
# The server saturates near five concurrent folds; more lengthens the tail without adding throughput.
SATURATING_CONCURRENCY = 5
# The documented charge per residue, which holds up to 500 residues.
BILLED_TOKENS_PER_RESIDUE = 25
# Past 500 residues the observed charge is the square of the length over this.
QUADRATIC_TOKEN_DIVISOR = 20


def estimate_request_tokens(complex_length: int) -> int:
    """Tokens one request of this total chain length is billed, and counted against the per minute budget.

    It is the larger of the documented `length * min(25, length)` and the observed `ceil(length ** 2 / 20)`: a
    197 residue complex is 4,925 tokens and a 764 residue one is 29,185 (2.9 credits), which is what the
    service reported for every fold checked.
    """
    if complex_length < 1:
        raise ValueError("a complex has at least one residue.")

    documented = complex_length * min(BILLED_TOKENS_PER_RESIDUE, complex_length)
    quadratic = -(-complex_length * complex_length // QUADRATIC_TOKEN_DIVISOR)
    return max(documented, quadratic)


def estimate_request_credits(complex_length: int) -> float:
    """Credits one request of this total chain length costs."""
    return estimate_request_tokens(complex_length) / TOKENS_PER_CREDIT


def daily_fold_capacity(complex_length: int) -> int:
    """How many folds of this total chain length the default daily allotment pays for, all callers together."""
    return DAILY_CREDITS * TOKENS_PER_CREDIT // estimate_request_tokens(complex_length)


__all__ = [
    "BILLED_TOKENS_PER_RESIDUE",
    "DAILY_CREDITS",
    "MAX_COMPLEX_LENGTH",
    "PUBLISHED_MAX_RPM",
    "PUBLISHED_MAX_TPM",
    "QUADRATIC_TOKEN_DIVISOR",
    "SATURATING_CONCURRENCY",
    "TOKENS_PER_CREDIT",
    "daily_fold_capacity",
    "estimate_request_credits",
    "estimate_request_tokens",
]
