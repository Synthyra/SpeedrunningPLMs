"""The Biohub fold API, as Synthyra's clients call it.

folding_service's `fold.biohub_client`, sequence_gradient's `models.folding.biohub_api`, and
Atlas's Biohub case-study script each fold complexes with Biohub's `fold_all_atom` endpoint. What
the three share is here: the service's address and model names, the error a failed request
raises, the limiter that holds a batch within Biohub's rolling budgets, and the writer that renders
a returned complex as a minimal mmCIF. Each client keeps its own options, requests, metrics, and
checks of what the service returns.
"""

from __future__ import annotations

import asyncio
import math
import re
import time

from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from typing import Any


DEFAULT_BIOHUB_URL = "https://biohub.ai"
DEFAULT_ENDPOINT = "/api/v1/fold_all_atom"
BIOHUB_TOKEN_ENV_KEY = "BIOHUB_TOKEN"
# Rate limiting and server-side failures, which a later attempt may get past.
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
# The Biohub model each client's model argument selects.
MODEL_NAME_BY_ARG = {
    "esmfold2": "esmfold2-2026-05",
    "esmfold2-fast": "esmfold2-fast-2026-05",
}
# The fields of a returned complex that its minimal mmCIF is written from, in the order the clients
# check them.
MINIMAL_MMCIF_FIELDS = (
    "id",
    "sequence",
    "atom_positions",
    "atom_elements",
    "atom_names",
    "atom_hetero",
    "token_to_atoms",
    "chain_id",
    "plddt",
    "metadata",
)


class BiohubAPIError(RuntimeError):
    """A non-retryable or exhausted Biohub HTTP failure."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Biohub API error {status_code}: {message}")
        self.status_code = status_code


class RollingRateLimiter:
    """Admit requests within rolling budgets of requests and estimated tokens.

    `acquire` waits until one more request fits both `max_rpm` requests and `max_tpm` estimated
    tokens over the last `window_seconds`, then records it. `pause_for` holds back every request
    until a time, as a 429's Retry-After asks. `time_func` and `sleep_func` default to the
    monotonic clock and `asyncio.sleep`, and a test passes a fake clock.
    """

    def __init__(
        self,
        max_rpm: int,
        max_tpm: int,
        *,
        window_seconds: float = 60.0,
        time_func: Callable[[], float] = time.monotonic,
        sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        assert max_rpm > 0, "max_rpm must be > 0"
        assert max_tpm > 0, "max_tpm must be > 0"
        assert window_seconds > 0, "window_seconds must be > 0"
        self.max_rpm = int(max_rpm)
        self.max_tpm = int(max_tpm)
        self.window_seconds = float(window_seconds)
        self.time_func = time_func
        self.sleep_func = sleep_func
        self.events: deque[tuple[float, int]] = deque()
        self.lock = asyncio.Lock()
        # No request waits on a pause until `pause_for` sets one.
        self.paused_until = -math.inf

    async def acquire(self, token_count: int) -> None:
        """Wait until a request of `token_count` estimated tokens fits both budgets, and record it.

        A request larger than the whole token budget could never fit, and raises `ValueError`.
        """
        assert token_count > 0, "token_count must be > 0"
        if token_count > self.max_tpm:
            raise ValueError(
                f"Estimated request tokens ({token_count}) exceed max_tpm ({self.max_tpm})"
            )

        while True:
            async with self.lock:
                now = float(self.time_func())
                if now < self.paused_until:
                    wait_seconds = max(0.01, self.paused_until - now)
                else:
                    self._drop_expired(now)
                    used_tokens = sum(tokens for _timestamp, tokens in self.events)
                    if len(self.events) < self.max_rpm and used_tokens + token_count <= self.max_tpm:
                        self.events.append((now, token_count))
                        return
                    wait_seconds = self._next_wait_seconds(now, token_count, used_tokens)

            await self.sleep_func(wait_seconds)

    async def pause_for(self, wait_seconds: float) -> None:
        """Hold back every request for `wait_seconds`, unless a pause already runs longer."""
        assert wait_seconds > 0, "wait_seconds must be > 0"
        async with self.lock:
            now = float(self.time_func())
            self.paused_until = max(self.paused_until, now + float(wait_seconds))

    def _drop_expired(self, now: float) -> None:
        """Forget the requests that left the window."""
        while len(self.events) > 0 and now - self.events[0][0] >= self.window_seconds:
            self.events.popleft()

    def _next_wait_seconds(self, now: float, token_count: int, used_tokens: int) -> float:
        """How long until the first expiry that lets the request fit, and at least 0.01 seconds."""
        waits: list[float] = []
        if len(self.events) >= self.max_rpm:
            waits.append(self.window_seconds - (now - self.events[0][0]))

        if used_tokens + token_count > self.max_tpm:
            running_tokens = used_tokens
            for timestamp, tokens in self.events:
                running_tokens -= tokens
                if running_tokens + token_count <= self.max_tpm:
                    waits.append(self.window_seconds - (now - timestamp))
                    break

        return max(0.01, min(waits) if len(waits) > 0 else 0.01)


def cif_token(value: Any) -> str:
    """One mmCIF value: `?` when empty, bare when it holds only letters, digits, and `_.+-`, and
    quoted otherwise."""
    text = str(value)
    if text == "":
        return "?"
    if re.fullmatch(r"[A-Za-z0-9_.+-]+", text) is not None:
        return text
    return "'" + text.replace("'", "''") + "'"


def lookup_chain_label(metadata: dict[str, Any], chain_number: int) -> str:
    """The label a complex's `chain_lookup` gives chain `chain_number`, under its number as text or
    as an integer, or else the character `chain_number` places after A."""
    assert "chain_lookup" in metadata, "Biohub complex metadata missing chain_lookup"
    chain_lookup = metadata["chain_lookup"]
    assert isinstance(chain_lookup, dict), "Biohub chain_lookup must be a dict"
    key_text = str(chain_number)
    if key_text in chain_lookup:
        return str(chain_lookup[key_text])
    if chain_number in chain_lookup:
        return str(chain_lookup[chain_number])
    return chr(ord("A") + chain_number)


def token_bfactor(plddt_value: Any) -> float:
    """A token's pLDDT as a B-factor, from 0 to 100: a pLDDT of at most 1 is scaled by 100."""
    value = float(plddt_value)
    if value <= 1.0:
        return value * 100.0
    return value


def minimal_mmcif(
    complex_state: Mapping[str, Any],
    block_name: str,
    *,
    chain_label: Callable[[dict[str, Any], int], str] = lookup_chain_label,
    bfactor: Callable[[Any], float] = token_bfactor,
) -> str:
    """Write a returned complex's atoms as a minimal mmCIF: one `atom_site` table in one block.

    `complex_state` is the `complex` a fold returns: per token, its residue name in `sequence`, its
    range of atoms in `token_to_atoms`, its chain number in `chain_id`, and its `plddt`; per atom,
    `atom_positions`, `atom_elements`, `atom_names`, and `atom_hetero`. The caller has checked that
    it holds every field of `MINIMAL_MMCIF_FIELDS` and that `metadata` is a dict, and names the
    block `data_<block_name>`. `chain_label` names a chain from the metadata and its number, and
    `bfactor` turns a token's pLDDT into its atoms' B-factor. Residues are numbered from 1 within
    each chain label, and a chain's entity is its number plus 1. In the shapes, `l` counts tokens
    and `n` atoms.
    """
    sequence = complex_state["sequence"]  # (l,)
    atom_positions = complex_state["atom_positions"]  # (n, 3)
    atom_elements = complex_state["atom_elements"]  # (n,)
    atom_names = complex_state["atom_names"]  # (n,)
    atom_hetero = complex_state["atom_hetero"]  # (n,)
    token_to_atoms = complex_state["token_to_atoms"]  # (l, 2), each token's atoms as [start, end)
    chain_ids = complex_state["chain_id"]  # (l,)
    plddt = complex_state["plddt"]  # (l,)
    metadata = complex_state["metadata"]

    lines = [
        f"data_{block_name}",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    residue_counter_by_chain: dict[str, int] = {}
    atom_id = 1
    for token_index, token in enumerate(sequence):
        start, end = token_to_atoms[token_index]
        chain_number = int(chain_ids[token_index])
        label = chain_label(metadata, chain_number)
        residue_index = residue_counter_by_chain.get(label, 0) + 1
        residue_counter_by_chain[label] = residue_index
        entity_id = chain_number + 1
        token_b = bfactor(plddt[token_index])
        for atom_index in range(int(start), int(end)):
            x, y, z = atom_positions[atom_index]
            group = "HETATM" if bool(atom_hetero[atom_index]) else "ATOM"
            atom_name = atom_names[atom_index]
            element = atom_elements[atom_index]
            row = [
                group,
                str(atom_id),
                cif_token(element),
                cif_token(atom_name),
                ".",
                cif_token(token),
                cif_token(label),
                str(entity_id),
                str(residue_index),
                "?",
                f"{float(x):.3f}",
                f"{float(y):.3f}",
                f"{float(z):.3f}",
                "1.00",
                f"{token_b:.2f}",
                "?",
                str(residue_index),
                cif_token(token),
                cif_token(label),
                cif_token(atom_name),
                "1",
            ]
            lines.append(" ".join(row))
            atom_id += 1
    lines.append("#")
    return "\n".join(lines) + "\n"


__all__ = [
    "BIOHUB_TOKEN_ENV_KEY",
    "DEFAULT_BIOHUB_URL",
    "DEFAULT_ENDPOINT",
    "MINIMAL_MMCIF_FIELDS",
    "MODEL_NAME_BY_ARG",
    "RETRY_STATUS_CODES",
    "BiohubAPIError",
    "RollingRateLimiter",
    "cif_token",
    "lookup_chain_label",
    "minimal_mmcif",
    "token_bfactor",
]
