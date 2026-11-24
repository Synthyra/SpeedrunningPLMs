"""The tiny synthetic data directory of `tiny_data.py`, prepared once for the test session."""

from __future__ import annotations

import sys
import pytest

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))  # tiny_data.py and the other scripts sit beside the package

from tiny_data import build_tiny_data  # noqa: E402


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    build_tiny_data(root / "afdb_tiny", root / "source")
    return root / "afdb_tiny"
