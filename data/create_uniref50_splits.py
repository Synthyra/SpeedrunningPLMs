import argparse
import sys

from pathlib import Path


_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from speedrunning_plms.data.splits import build_uniref50_splits, login_if_token, push_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()
    login_if_token(args.hf_token)
    data = build_uniref50_splits()
    push_splits(data, "Synthyra/uniref50")


if __name__ == "__main__":
    main()
