---
id: foundry.biohub
kind: library_module
status: active
aliases: [foundry.biohub]
---
# foundry.biohub

The Biohub fold API as Synthyra's clients call it: the service's address and model names, the
error a failed request raises, the limiter that holds a batch within Biohub's rolling budgets, and
the writer that renders a returned complex as a minimal mmCIF.

| Name | Is |
|---|---|
| `DEFAULT_BIOHUB_URL`, `DEFAULT_ENDPOINT` | `https://biohub.ai` and `/api/v1/fold_all_atom` |
| `BIOHUB_TOKEN_ENV_KEY` | `BIOHUB_TOKEN`, the environment variable each client reads its token from |
| `MODEL_NAME_BY_ARG` | The Biohub model each client's model argument selects: `esmfold2-2026-05` for `esmfold2` and `esmfold2-fast-2026-05` for `esmfold2-fast` |
| `RETRY_STATUS_CODES` | 429, 500, 502, 503, and 504: rate limiting and server failures, which a later attempt may get past |
| `BiohubAPIError(status_code, message)` | A `RuntimeError` for a request that failed for good, `Biohub API error <status>: <message>`, with its `status_code` |
| `RollingRateLimiter(max_rpm, max_tpm, *, window_seconds=60.0, time_func, sleep_func)` | Admits requests within `max_rpm` requests and `max_tpm` estimated tokens over a rolling window. `acquire(token_count)` waits until one more request fits both budgets, sleeping until the first expiry that makes room and at least 0.01 seconds at a time, then records it; a request larger than `max_tpm` raises `ValueError`. `pause_for(seconds)` holds back every request that long, as a 429's Retry-After asks, and never shortens a longer pause. `time_func` and `sleep_func` default to `time.monotonic` and `asyncio.sleep` |
| `MINIMAL_MMCIF_FIELDS` | The fields of a returned complex the writer reads, in the order the clients check them |
| `minimal_mmcif(complex_state, block_name, *, chain_label, bfactor)` | The complex's atoms as one `atom_site` table in block `data_<block_name>`: HETATM for a hetero atom, coordinates to three decimals, each atom's B-factor its token's, residues numbered from 1 within each chain label, and a chain's entity its number plus 1 |
| `cif_token(value)` | One mmCIF value: `?` when empty, bare when it holds only letters, digits, and `_.+-`, and single-quoted otherwise, with each quote doubled |
| `lookup_chain_label(metadata, chain_number)` | The writer's default `chain_label`: the complex's `chain_lookup` entry for the chain, under its number as text and then as an integer, or else the character the number places after A |
| `token_bfactor(plddt)` | The writer's default `bfactor`: a pLDDT of at most 1 times 100, and a larger one as it is |

The writer does not check the complex. Its caller has checked that every field of
`MINIMAL_MMCIF_FIELDS` is there and that `metadata` is a dict, and names the block.

Promoted on 2026-09-24 from the three clients below, which now import these names, so each name is
still importable from its client:

- folding_service's `fold.biohub_client`, the Biohub backend of its `fold()` API, from which
  `fold.biohub_batch` takes the limiter. It checks the complex with assertions, names the block
  with its `_safe_name`, and writes with foundry's defaults. Its ESMFold2 Modal image imports none
  of this, so the image is unchanged.
- Atlas's `exploratory/scripts/fold_case_study_structures_biohub.py`, which also asserts that the
  token arrays agree in length and that the atom arrays do, names the block with its own
  `_safe_name`, and pauses the limiter on a 429.
- sequence_gradient's `models.folding.biohub_api`, whose `BiohubFoldResponse.minimal_mmcif` raises
  `ValueError` and `TypeError` for a malformed complex and passes two of its own methods: its
  `_chain_label`, which names a chain the lookup leaves out spreadsheet-style (A to Z, then AA),
  and its `_token_bfactor`, which requires each pLDDT to be one number from 0 to 1. Its resume
  compares a stored mmCIF with the writer's output byte for byte. Its Biohub backend runs from an
  image built from its projection, which now vendors `foundry/biohub/`.

Each client was compared with its original, exactly: its module's names, and the source of every
definition the wave did not change; the constants and the error; the limiter under a fake clock,
one request at a time and concurrently, over random budgets, windows, clocks, and token counts,
with Atlas's pauses; the writer and its helpers over random and malformed complexes, byte for byte
and down to each error; and each client's fold flow against a fake Biohub, with retries, HTTP
errors, rejected complexes, and resume from the files the other side staged. Nothing differed but
what follows, and each of these was checked itself:

- sequence_gradient's limiter fails an assertion (`max_rpm must be > 0`, `max_tpm must be > 0`,
  `token_count must be > 0`) for a budget or token count that is not positive, where it raised
  `ValueError`, and its error for a request over the token budget has no closing period. Its
  options reject such budgets before a limiter is built, and its token estimate is at least 2.
- Atlas's limiter names `max_rpm` and `max_tpm` in its errors, where it named `--max-rpm` and
  `--max-tpm`. Atlas checks those arguments before it builds a limiter.
- folding_service's and sequence_gradient's limiters assert a positive window. A window of zero or
  less admitted every request before; neither client passes a window.
- folding_service's and sequence_gradient's limiters gain `pause_for` and `paused_until`, which
  neither calls.
- Before any pause, the limiter is never paused. Atlas's was paused until time 0, which held back
  only a clock reading below zero, and `time.monotonic`, the clients' clock, counts up from the
  machine's boot.
- `BiohubAPIError` is `foundry.biohub.BiohubAPIError`, the name a traceback and a pickle carry.
  folding_service's and Atlas's failed rows show it in their `error_traceback`, whose frames now
  include foundry's; their `error` column is unchanged.
- folding_service and Atlas no longer define `_cif_token`, `_chain_label`, and `_token_bfactor`,
  and sequence_gradient's `BiohubFoldResponse` no longer has `_cif_token`. Nothing imported them.
  `BiohubFoldResponse` gains `_token_bfactor`, its per-token pLDDT check.
- sequence_gradient's writer reads an atom's hetero flag before its name, as folding_service's
  did. The error differs only for a complex whose names and flags both fail at the same atom, such
  as names given as text and flags as a list, both too short.

Not here, since the clients differ in them and one copy would change a client's behavior:
`_clean_response_data`, which asserts in folding_service and Atlas and raises `TypeError` in
sequence_gradient; `_safe_name`, which keeps dots in folding_service and sequence_gradient but not
in Atlas; the model aliases, two in folding_service, six in sequence_gradient, and none in Atlas;
and each client's retry wait. The default budgets and the token estimate are shared since 2026-09-30
(`limits`, below). folding_service's and Atlas's
`_complex_to_cif` stays with each too: it uses ESM's `MolecularComplex` where that is installed,
and the client's own checked writer otherwise.

## Limits and the bill

`foundry.biohub.limits` holds what the API allows and what a request costs, measured against the live API by
folding_service (2026-08-17) and rechecked on eight folds (2026-09-30)
([the decision](../../docs/decisions/foundry/biohub/2026-09-30_biohub_limits_in_one_module.md)). sequence_gradient and
folding_service import it.

| Name | Is |
|---|---|
| `MAX_COMPLEX_LENGTH` | 768, the sum of all chain lengths, not per chain; 769 returns HTTP 422 |
| `PUBLISHED_MAX_RPM`, `PUBLISHED_MAX_TPM` | 40 requests and 400,000 tokens a minute, for `esmfold2` and `esmfold2-fast` alike |
| `DAILY_CREDITS`, `TOKENS_PER_CREDIT` | 20,000 credits a day, shared by every caller of one token, and 10,000 tokens a credit; the clients do not track the day, only the minute |
| `SATURATING_CONCURRENCY` | 5: the server saturates near five or six concurrent folds, so more lengthens the tail |
| `estimate_request_tokens(length)` | The larger of the documented `length * min(25, length)` and the observed `ceil(length ** 2 / 20)`: 4,925 for 197 residues, 29,185 (2.9 credits) for 764 |
| `estimate_request_credits(length)`, `daily_fold_capacity(length)` | That in credits, and how many folds of that length a day pays for: about 6,850 at 764 residues |

Rules that follow: a fold of a 764 residue complex costs 2.9 credits and the minute budget admits 13 of them; `fast`
buys time, not a lower price; latency was about 35 s per fold at 768 tokens in folding_service's measurement and 73 s
for four concurrent folds at 20 loops and 100 steps in sequence_gradient's; LM dropout and masking must be zero and
the API returns one unseeded structure per request. folding_service's `docs/options.md` and `docs/biohub-batch.md`
carry the measurements.

## Tests

`tests/tier1_unit/test_biohub.py` and `tests/tier1_unit/test_biohub_limits.py`, with folding_service's `tests/unit/test_biohub_client.py` and
`tests/unit/test_biohub_batch.py`, Atlas's `exploratory/tests/test_biohub_structure_folding.py`,
and sequence_gradient's `tests/test_biohub_api.py` as the guards on the names they import.
