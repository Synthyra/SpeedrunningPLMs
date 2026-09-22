"""Keep tests offline; hide GPUs unless CUDA checks are explicitly requested."""

import os
import sys
import pytest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if os.environ.get("PLM_TEST_CUDA") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def pytest_sessionstart(session: pytest.Session) -> None:
    import torch

    # Thread-pool overhead dominates the tiny models exercised here.
    torch.set_num_threads(1)
