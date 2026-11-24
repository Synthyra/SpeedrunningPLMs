"""The loop's dial study: a looped run's selected weights scored at each pass count and exit, outside the run folder.

    python dial.py --data-dir /data/afdb_v6 --run-dir runs/lp_dial_600 --output runs/analysis/lp_dial_600.json

Rebuilds the candidate from the run's own `train.py` and `strategies/` snapshots, each checked against its digest in
`result.json` (`probe.load_run_snapshot`), with the genome in that result, and loads `best.pt`. With the `loop` family on, it scores
validation at every pass count and at each confidence and fixed-point exit threshold, each with the mean passes a real
token ran (the strategy's `pass_tokens`); picks on validation the setting with the fewest mean passes whose loss is
within `--tolerance` of the full depth's; and scores test at every pass count and at that pick, so test never chooses.
With `routing` on it adds, from the validation pass at full depth, each position's group loads and, in a tied loop,
LOOM's cross-pass similarity (the cosine of a core block's loads between two of its passes; near 1 is the collapse);
with `sglr` on, the share of real tokens each step routes to each module. The report goes to `--output` and the last
line of standard output, never into the run folder, whose `SHA256SUMS` seals it. A looped run's compute is the same
at every setting here: a stopped token is computed and discarded, so the mean passes are what a sparse kernel could
save, not what this run saves.
"""

from __future__ import annotations

import argparse
import itertools
import json
import torch

from pathlib import Path
from typing import Any

from probe import load_run_snapshot
from speedrunning_plms import benchmark
from speedrunning_plms.strategies import Strategies
from speedrunning_plms.strategies.loop import Loop
from speedrunning_plms.strategies.sglr import MODULES


CONFIDENCE_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # the top residue probability that stops a token
FIXED_POINT_THRESHOLDS = (0.01, 0.03, 0.1, 0.3)  # a pass's move over the state's norm that stops a token


def dial_setting(loop: Loop, passes: int, exit: str, threshold: float) -> dict[str, Any]:
    loop.eval_passes, loop.exit, loop.exit_threshold = passes, exit, threshold
    loop.holder.pass_tokens.zero_()
    return {"passes": passes, "exit": exit, "threshold": threshold}


def scored(model: torch.nn.Module, split: benchmark.HeldOutSplit, device: torch.device, tokens_per_batch: int,
           setting: dict[str, Any], loop: Loop, keep_probabilities: bool) -> dict[str, Any]:
    """One split at one dial setting: the evaluator's metrics and the mean passes of a real token."""
    metrics = benchmark.evaluate_split(model, split, device, tokens_per_batch=tokens_per_batch, keep_probabilities=keep_probabilities)
    counts = loop.holder.pass_tokens.tolist()  # [real tokens, then the real tokens each pass updated]
    kept = ("loss", "accuracy", "f1_macro", "mcc", "roc_auc_macro", "pr_auc_macro")
    return {**setting, "mean_passes": sum(counts[1:]) / counts[0], **{key: metrics[key] for key in kept if key in metrics}}


def loop_study(model: torch.nn.Module, splits: dict[str, benchmark.HeldOutSplit], device: torch.device,
               tokens_per_batch: int, tolerance: float) -> dict[str, Any]:
    """Validation over the whole grid, the pick on validation, test at every pass count and at the pick."""
    loop: Loop = model.strategies.loop
    full = loop.config.passes
    grid = [(passes, "none", 0.0) for passes in range(1, full + 1)]
    grid += [(full, "confidence", threshold) for threshold in CONFIDENCE_THRESHOLDS]
    grid += [(full, "fixed_point", threshold) for threshold in FIXED_POINT_THRESHOLDS]
    validation = [scored(model, splits["validation"], device, tokens_per_batch, dial_setting(loop, *setting), loop, False)
                  for setting in grid]
    for row in validation:
        print(json.dumps({"validation": row}), flush=True)
    full_loss = next(row["loss"] for row in validation if row["passes"] == full and row["exit"] == "none")
    close = [row for row in validation if row["loss"] <= full_loss + tolerance]
    pick = min(close, key=lambda row: (row["mean_passes"], row["loss"]))
    tested = sorted({(passes, "none", 0.0) for passes in range(1, full + 1)} | {(pick["passes"], pick["exit"], pick["threshold"])})
    test = [scored(model, splits["test"], device, tokens_per_batch, dial_setting(loop, *setting), loop, True) for setting in tested]
    dial_setting(loop, loop.config.eval_passes or full, loop.config.exit, loop.config.exit_threshold)  # the run's own setting again
    return {"passes": full, "tolerance": tolerance, "validation": validation, "pick": pick, "test": test}


def routing_study(strategies: Strategies) -> dict[str, Any]:
    """Each position's group loads from the last validation pass at full depth, and the cross-pass similarity in a
    tied loop: for each core block, the cosine of its normalized loads between every two of its passes."""
    loads = strategies.routing.holder.eval_load.double()  # (blocks, e)
    shares = loads / loads.sum(dim=-1, keepdim=True).clamp(min=1)  # (blocks, e)
    report: dict[str, Any] = {"shares": shares.tolist()}
    loop = strategies.loop
    if loop.variant != "tied" or loop.config.passes < 2:
        return report
    config = loop.config
    by_pass: dict[str, list[float]] = {}
    for block in range(config.core):
        positions = [config.prelude + block + run * config.core for run in range(config.passes)]
        for (first, a), (second, b) in itertools.combinations(enumerate(positions), 2):
            cosine = torch.nn.functional.cosine_similarity(shares[a], shares[b], dim=0)  # ()
            by_pass.setdefault(f"{first + 1}-{second + 1}", []).append(float(cosine))
    report["cross_pass_cosine"] = {pair: sum(values) / len(values) for pair, values in by_pass.items()}
    report["cross_pass_cosine_mean"] = sum(sum(values) for values in by_pass.values()) / sum(len(values) for values in by_pass.values())
    return report


def sglr_study(strategies: Strategies) -> dict[str, Any]:
    """The share of real tokens each step routed to each module in the last validation pass."""
    routes = strategies.sglr.holder.routes.double()  # (steps, m)
    shares = routes / routes.sum(dim=-1, keepdim=True).clamp(min=1)  # (steps, m)
    return {"modules": list(MODULES), "shares_by_step": shares.tolist(), "shares": (routes.sum(dim=0) / routes.sum()).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="the report, outside the run folder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens-per-batch", type=int, default=131_072)
    parser.add_argument("--tolerance", type=float, default=0.002, help="validation loss a pick may give up against the full depth")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"{arguments.output} exists")
    if arguments.output.resolve().is_relative_to(arguments.run_dir.resolve()):
        raise ValueError("the report must stay outside the run folder, which SHA256SUMS seals")
    run_result = json.loads((arguments.run_dir / "result.json").read_text(encoding="utf-8"))
    device = torch.device(arguments.device)
    candidate = load_run_snapshot(arguments.run_dir)  # the run's own train.py and strategies, digests checked
    manifest = benchmark.load_manifest(arguments.data_dir)
    model = candidate.build_candidate(run_result["genome"], max(manifest["max_tokens"].values())).to(device)
    state = torch.load(arguments.run_dir / "best.pt", map_location=device)
    model.load_state_dict({name: tensor.to(torch.float32) if tensor.is_floating_point() else tensor for name, tensor in state.items()})
    model.eval()
    strategies: Strategies = model.strategies
    splits = {split: benchmark.load_held_out(arguments.data_dir, split) for split in ("validation", "test")}
    report: dict[str, Any] = {"name": run_result["name"], "candidate_sha256": run_result["candidate_sha256"],
                              "benchmark_id": run_result["benchmark_id"], "genome_changes": run_result["genome_changes"]}
    if strategies.loop.active:
        report["loop"] = loop_study(model, splits, device, arguments.tokens_per_batch, arguments.tolerance)
    if strategies.routing.active or strategies.sglr.active:  # one more validation pass at full depth, counted from zero
        if strategies.routing.active:
            strategies.routing.holder.eval_load.zero_()
        if strategies.sglr.active:
            strategies.sglr.holder.routes.zero_()
        if strategies.loop.active:
            dial_setting(strategies.loop, strategies.loop.config.passes, "none", 0.0)
        report["validation_at_full_depth"] = benchmark.evaluate_split(
            model, splits["validation"], device, tokens_per_batch=arguments.tokens_per_batch, keep_probabilities=False)["loss"]
        if strategies.routing.active:
            report["routing"] = routing_study(strategies)
        if strategies.sglr.active:
            report["sglr"] = sglr_study(strategies)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {key: report[key] for key in ("name", "validation_at_full_depth") if key in report}
    if "loop" in report:
        summary["pick"] = report["loop"]["pick"]
    if "routing" in report and "cross_pass_cosine_mean" in report["routing"]:
        summary["cross_pass_cosine_mean"] = report["routing"]["cross_pass_cosine_mean"]
    if "sglr" in report:
        summary["sglr_shares"] = report["sglr"]["shares"]
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
