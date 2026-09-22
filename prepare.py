"""Prepare pinned protein data once before running experiments."""

import sys

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from speedrunning_plms.research.benchmark import prepare_main


if __name__ == "__main__":
    prepare_main()
