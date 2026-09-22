import sys

from pathlib import Path


_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from speedrunning_plms.research.engine import main


if __name__ == "__main__":
    main()
