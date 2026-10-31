"""The fixed benchmark: data, corruption, evaluator, clock, shape rule, result schema and the 3Di probe.

A candidate (`train.py`) imports from here and changes nothing here. The digest of these modules enters every run's
comparison key.
"""

from __future__ import annotations

from speedrunning_plms.benchmark.architecture import BLOCKS, HIDDEN, INTERMEDIATE, check_architecture
from speedrunning_plms.benchmark.clock import BUDGET_SECONDS, EVAL_CAP_SECONDS, WARMUP_CAP_SECONDS, Clock
from speedrunning_plms.benchmark.data import (
    HeldOutSplit, TrainStream, benchmark_id, corrupt, load_held_out, load_manifest, load_train, prepare,
)
from speedrunning_plms.benchmark.evaluate import Scorer, evaluate_split, masked_cross_entropy
from speedrunning_plms.benchmark.result import (
    candidate_sha256, comparable, comparison_key, environment, evaluator_sha256, ledger_row, write_result,
)
from speedrunning_plms.benchmark.tokens import (
    CLS_ID, EOS_ID, ESM2_RESIDUE_COLUMNS, IGNORE_LABEL, MASK_ID, NUM_CLASSES, PAD_ID, VOCAB_SIZE, residue_labels,
)


__all__ = [
    "BLOCKS", "BUDGET_SECONDS", "CLS_ID", "EOS_ID", "ESM2_RESIDUE_COLUMNS", "EVAL_CAP_SECONDS", "HIDDEN", "IGNORE_LABEL",
    "INTERMEDIATE", "MASK_ID", "NUM_CLASSES", "PAD_ID", "VOCAB_SIZE", "WARMUP_CAP_SECONDS", "Clock", "HeldOutSplit",
    "Scorer", "TrainStream", "benchmark_id", "candidate_sha256", "check_architecture", "comparable", "comparison_key",
    "corrupt", "environment", "evaluate_split", "evaluator_sha256", "ledger_row", "load_held_out", "load_manifest",
    "load_train", "masked_cross_entropy", "prepare", "residue_labels", "write_result",
]
