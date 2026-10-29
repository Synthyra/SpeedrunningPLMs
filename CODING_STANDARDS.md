# Coding standards

Every Python file in this repository follows these rules, `train.py` first: it is the file a contributor edits and the
example the others copy. The rules keep the data flow visible, so a reader can audit a candidate without running it.
`candidates/` holds frozen snapshots of past record holders: their bytes identify runs in the ledger, so they are never
edited and are exempt.

## Imports

A module starts with its docstring, then `from __future__ import annotations`. Every `import x` statement comes before
every `from x import y` statement. Within each block the order is standard library, third party, then this
repository, with a blank line before the repository's own names. Two blank lines follow the import block.

```python
"""What the module is for, in one or two sentences, and the symbols its shape traces use."""

from __future__ import annotations

import json
import numpy as np
import torch

from pathlib import Path
from typing import Any
from torch import Tensor

from speedrunning_plms import benchmark
```

An import that must run later stays where it is, with a comment saying why: `matplotlib.use("Agg")` before `pyplot`,
`pandas` inside `prepare` so a training run never imports it, `transformers` inside the reference model so the unigram
floor needs none.

## Types and control flow

- Every function has precise parameter and return types: `list[Tensor]`, `dict[str, Any]`, `Callable[[], Tensor]`,
  never a bare `dict`, `list` or `object` standing in for a known type. `Any` marks a genuinely dynamic boundary (a
  JSON document, an optional logger) and nothing else.
- One function, one job. Entry points parse arguments, build the pieces, call them in a visible order and print the
  receipt; the work lives in named functions.
- Write for the inputs the benchmark produces. A guard stays only where it protects a real requirement.
- Raise an ordinary exception for bad external input (`ValueError`, `FileExistsError`); assert an internal invariant.
  No broad `except`, no silent fallback, no second code path "in case". A run that breaks a rule fails loudly and is
  recorded as such.
- Extract a helper when it names a real concept or a separately testable phase, not to shorten one caller.

## Names

- Descriptive names say what a value is: `hidden_states`, `attention_mask`, `scored_residues`, `val_select`. Not
  `data`, `result`, `info`, `tmp`, `out`.
- Short mathematical symbols follow the notation of the papers: uppercase for matrices and tensors of rank two or
  more (`X`, `Q`, `K`, `V`, `G`, `U`), lowercase for scalars, vectors and dimensions (`b`, `l`, `d`, `h`, `d_h`, `f`,
  `c`, `k`, `m`, `n`). A descriptive tensor name stays lowercase snake case whatever its rank.
- Configuration that is genuinely constant is all caps near the top of the module (`MASK_RATE`, `BUDGET_SECONDS`,
  `NORMUON_BETA`).
- Public names, serialized fields (`result.json`, `runs.jsonl`, the manifest) and framework attributes (`blocks`,
  `attention`, `mlp`, `residue_columns`, `hidden_states`) are interfaces: the evaluator, the shape rule and the probe
  read them, so they keep their names.

## Shape traces

Every line that creates, reshapes, indexes, reduces, calls a layer on or returns an array ends with its shape in a
comment, so the data flow reads without executing the code. The symbols are defined once in the module docstring
(`train.py`: b windows or proteins, l tokens, d = 768, h heads, d_h per-head width, f = 2,048, v = 33 tokens, c = 20
classes, m masked residues). `()` is a scalar; `...` stands for leading dimensions the function does not care about; a
bank of k matrices is `(k, m, n)`.

```python
def forward(self, X: Tensor, block_mask: BlockMask | None, dense_mask: Tensor | None) -> Tensor:
    # X: (b, l, d); block_mask for flex attention, or dense_mask (b, 1, l or 1, l) bool for SDPA.
    b, l, d = X.shape
    Q = norm(self.query(X).view(b, l, self.heads, self.d_h))  # (b, l, h, d_h)
    Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # each (b, h, l, d_h)
    Y = flex_attention(Q, K, V, score_mod=self.score_mod(), block_mask=block_mask)  # (b, h, l, d_h)
    return self.output(Y.transpose(1, 2).reshape(b, l, d))  # (b, l, d)
```

A trace is derived from the code, never guessed; a stale trace is a bug. Function inputs are traced in the first
comment of the body when the signature does not say them.

## Comments and docstrings

- A docstring says what a function or module is for and what it promises, in one or two sentences. A comment says
  why: an intent, an assumption, a unit, a convention (which token ids, which split), a decision and its evidence
  ("the unnormalized head diverged on a Muon-trained candidate").
- No comment narrates the syntax or repeats a name. No commented-out code. No TODO without an owner and a plan.
- Plain American English in short direct sentences. No em dashes. Words that decorate rather than say something
  (`robust`, `leverage`, `seamless`, `crucial`, `delve`, `comprehensive`, `cutting-edge`) do not appear unless they are
  the exact technical term.
- A claim about speed, memory, compatibility or behavior rests on a measurement, a test or the documentation, and
  the comment says which.

## Structure

- One module, one responsibility. `benchmark/` holds the fixed parts (data, tokens, evaluator, clock, shape rule,
  result schema, probe); `optim/` the per-matrix Muon; `train.py` everything a run may change.
- Helpers live beside the feature that owns them. There is no `utils`, `helpers`, `common` or `misc` module, and no
  second copy of something that exists: search before adding.
- The cross entropy everywhere, training included, is the collapse-free one from `foundry.layers.cross_entropy`;
  PyTorch's appears only in a test, as the reference it is checked against away from the collapse margin.

## What a change may touch

`train.py` is the only file a run edits. The modules under `src/speedrunning_plms/benchmark/`, `prepare.py`,
`baselines.py`, `probe.py`, `record.py` and `plots.py` define the benchmark; a change to them changes what every result
means and is a separate, deliberate commit with its reason. `AGENTS.md` and `program.md` say how a candidate is tried,
kept, reverted and recorded.

## Checking

```bash
pip install -e ".[test]"
python -m pytest -q
```

The tests cover every fixed part on the CPU, run the working `train.py` for one step on synthetic data, and check that
its banked optimizer matches the per-matrix one. A candidate that fails them is not shipped.
