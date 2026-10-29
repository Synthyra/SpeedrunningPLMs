"""The 3Di probe on a kept candidate's weights; fixed.

    python probe.py --data-dir /data/afdb_v6 --run-dir runs/000_baseline

Rebuilds the candidate from the run's `train.py` snapshot (or the working `train.py` when the run holds none), loads
`best.pt`, trains the probe head on the fixed subset and scores it on the full validation and test splits. Writes
`probe.json` beside `result.json`; never changes the result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import torch

from pathlib import Path
from types import ModuleType

from speedrunning_plms import benchmark
from speedrunning_plms.benchmark import probe


def load_candidate_module(path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location("candidate", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules["candidate"] = module
    specification.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=None, help="the train.py that produced the run; default run-dir/train.py then ./train.py")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens-per-batch", type=int, default=32_768)
    arguments = parser.parse_args()
    output = arguments.run_dir / "probe.json"
    if output.exists():
        raise FileExistsError(f"{output} exists")
    candidate_path = arguments.candidate or next(path for path in (arguments.run_dir / "train.py", Path(__file__).parent / "train.py") if path.exists())
    run_result = json.loads((arguments.run_dir / "result.json").read_text(encoding="utf-8"))
    if benchmark.candidate_sha256(candidate_path) != run_result["candidate_sha256"]:
        raise ValueError(f"{candidate_path} is not the candidate that produced {arguments.run_dir}")

    device = torch.device(arguments.device)
    candidate = load_candidate_module(candidate_path)
    manifest = benchmark.load_manifest(arguments.data_dir)
    model = candidate.Candidate(candidate.Recipe(), max(manifest["max_tokens"].values())).to(device)
    state = torch.load(arguments.run_dir / "best.pt", map_location=device)
    model.load_state_dict({name: tensor.to(torch.float32) for name, tensor in state.items()})
    model.eval()

    stream = benchmark.load_train(arguments.data_dir)
    head = probe.train_head(model, stream, device, hidden=benchmark.HIDDEN, num_hidden_states=benchmark.BLOCKS + 1)
    metrics = {split: probe.score_head(model, head, benchmark.load_held_out(arguments.data_dir, split), device,
                                       tokens_per_batch=arguments.tokens_per_batch)
               for split in ("validation", "test")}
    report = {"name": run_result["name"], "candidate_sha256": run_result["candidate_sha256"], "benchmark_id": run_result["benchmark_id"],
              "probe": {"max_tokens": probe.PROBE_MAX_TOKENS, "subset": manifest["train"]["probe_subset"],
                        "learning_rate": probe.PROBE_LEARNING_RATE, "seed": probe.PROBE_SEED},
              "metrics": metrics}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({split: {key: values[key] for key in ("accuracy", "f1_macro", "mcc")} for split, values in metrics.items()}), flush=True)


if __name__ == "__main__":
    main()
