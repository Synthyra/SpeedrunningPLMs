"""Launch and record an isolated local or SSH experiment."""

import sys

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from speedrunning_plms.research.runner import main


if __name__ == "__main__":
    main()
