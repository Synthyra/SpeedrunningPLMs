"""Archive the campaign's runs on the Hub before the box closes.

    python publish_runs.py --runs-dir /state/runs --repo-id Synthyra/SpeedrunningPLMs-runs \\
        [--weights name ...] [--state-dir /state/search --state-dir /state/search_1h] [--public]

Every run folder under --runs-dir goes up without its weights and W&B files (`result.json`, `genome.json`, the
candidate's `train.py` and its `strategies/`); the runs named by --weights also take their `best.pt`. Each --state-dir
goes up whole but for its lock and pid files (the search state, the ledger, the loop log). The repository is a private
model repository (public only with --public, on Logan's word); uploads are idempotent, so a rerun sends only what the
Hub lacks. HF_TOKEN comes from the environment.
"""

import argparse

from pathlib import Path
from huggingface_hub import HfApi


RUN_FILES = ("result.json", "genome.json", "train.py", "strategies/**")
WEIGHTS = "best.pt"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--weights", nargs="*", default=[], help="runs whose best.pt goes up too")
    parser.add_argument("--state-dir", type=Path, action="append", default=[], help="a search state folder")
    parser.add_argument("--public", action="store_true", help="only on Logan's word; private otherwise")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    api = HfApi()
    api.create_repo(arguments.repo_id, repo_type="model", private=not arguments.public, exist_ok=True)
    runs = sorted(path for path in arguments.runs_dir.iterdir() if (path / "result.json").exists())
    missing = sorted(set(arguments.weights) - {run.name for run in runs})
    assert not missing, f"no finished run folder for {missing}"
    print(f"{len(runs)} runs with a result.json, {len(arguments.weights)} with weights")
    api.upload_folder(repo_id=arguments.repo_id, repo_type="model", folder_path=str(arguments.runs_dir),
                      path_in_repo="runs", allow_patterns=[f"*/{name}" for name in RUN_FILES],
                      commit_message="runs: results, genomes and candidates")
    for name in arguments.weights:
        api.upload_file(repo_id=arguments.repo_id, repo_type="model", path_or_fileobj=str(arguments.runs_dir / name / WEIGHTS),
                        path_in_repo=f"runs/{name}/{WEIGHTS}", commit_message=f"{name}: best weights")
        print(f"weights: {name}")
    for state_dir in arguments.state_dir:
        api.upload_folder(repo_id=arguments.repo_id, repo_type="model", folder_path=str(state_dir),
                          path_in_repo=state_dir.name, ignore_patterns=["*.lock", "*.pid", "stop", "results/*"],
                          commit_message=f"{state_dir.name}: the search state, ledger and loop log")
        print(f"state: {state_dir.name}")


if __name__ == "__main__":
    main()
