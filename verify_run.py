"""Check a run folder submitted for a record: prints PASS, FAIL or REVIEW, then one reason a line; exits 0 only on PASS.

    python verify_run.py <run_dir> [--data-dir D] [--rescore] [--device cuda]

The folder is what `run_gh200.sh train` leaves: `result.json`, `best.pt`, the `train.py`, `strategies/` and
`genome.json` snapshots, the evidence `train.py` writes (`machine.json`, `phases.jsonl`, `gpu_samples.csv`,
`nvidia-smi_start.txt`, `nvidia-smi_end.txt`; `speedrunning_plms.evidence`) and what the script adds around the
container (`host.json`, `console.log`, `source.patch` when the tree was dirty, `SHA256SUMS`).

FAIL, the hard checks: `SHA256SUMS` matches every file and lists every file; the caps recomputed from the clock
receipt's raw seconds, not its stored flags; the comparison key recomputed with the benchmark's own function; the
evaluator digest equal to this checkout's; the snapshot `train.py`, the strategies snapshot and the genome equal to the
candidate parts, and the candidate digest recomputed from them; exactly one GPU, a GH200, whose UUID is the one the
sampler read; the curve consistent with the selection and the clock's counts; the console's JSON lines equal to
`result.json`. With `--data-dir`, the benchmark id and the training files' digests against that data's manifest.

REVIEW, the soft checks, each a threshold below that the owner tunes: GPU utilization while training, gaps in the
sampler's readings, another process on the GPU, throttling the driver reported.

NOTE, never a verdict: a file the project's own tools add to a finished folder, after `SHA256SUMS` was written, is
reported and not checked: `probe.json` from `probe.py`, and the bytecode Python caches in `__pycache__/` when
`probe.py` loads the `train.py` snapshot. Any other unlisted file fails the sums.

`--rescore` rebuilds the model from the snapshot as `probe.py` does, loads `best.pt` and scores the splits
`result.json` holds; a GPU run needs a GPU to rescore in reasonable time. It keeps Python's bytecode outside the
folder, so it neither adds a file nor runs a cached one in place of the snapshot.

What a PASS does not prove: every file here was written by the submitter's machine. A folder fabricated whole and
consistently (a result, its curve, a console log that agrees, sampler readings of a GH200's UUID, an inventory, the
sums resealed) passes every check but `--rescore`, and `--rescore` shows only that the weights score what the result
claims, not that they were trained in the clock's seconds on that GPU. A PASS means the folder is internally
consistent, came from this benchmark's evaluator and the candidate it names, broke no cap, and shows no sign of
another GPU or a shared one. A record still rests on the submitter, the W&B run and, when it matters, a rerun of the
candidate by the maintainer. Appending the run to the ledger, with its verdict checks, stays `record.py`'s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import torch

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from foundry.provenance import SUMS_NAME, directory_digests, read_samples, read_sha256sums, summarize_samples
from foundry.provenance.nvidia_smi import TIMESTAMP_FORMAT
from speedrunning_plms import benchmark


# The owner tunes these four; each sends a run to a person, never to FAIL.
UTILIZATION_REVIEW_PERCENT = 90.0  # the median GPU utilization while training below this
SAMPLER_GAP_REVIEW_SECONDS = 5.0  # a gap between readings, or the readings' edges and the training window's, above this
THROTTLE_REVIEW_MASK = 0x08 | 0x20 | 0x40 | 0x80  # HW slowdown, SW thermal, HW thermal, HW power brake (NVML's clock event
# reasons); SW power cap (0x04) is left out, as a saturated GPU at its power limit reports it
# What the rescore tolerates: best.pt holds bf16 weights, the run scored their float32 originals.
RESCORE_LOSS_TOLERANCE = 0.01  # nats
RESCORE_ACCURACY_TOLERANCE = 0.005
GH200_NAME = "NVIDIA GH200 480GB"  # nvidia-smi's and torch's name for the benchmark's GPU (results/runs.jsonl, 2026-10)
REQUIRED = ("result.json", "best.pt", "train.py", "strategies", "machine.json", "phases.jsonl", "console.log", "host.json", SUMS_NAME)
WINDOW_FLAGS = {"--window": "window", "--windows-per-step": "windows_per_step"}  # train.py's smoke overrides of the genome
# The files the project's own tools add to a sealed folder, by the path pattern and the tool; never read by a check.
AFTER_SEALING = {r"probe\.json": "probe.py's report",
                 r"__pycache__/train\.cpython-\d+(\.opt-[12])?\.pyc": "the bytecode of the train.py snapshot, cached when probe.py loads it"}


@dataclass(frozen=True)
class Finding:
    level: str  # FAIL, REVIEW, or NOTE (a file a project tool added after sealing; never changes the verdict)
    check: str
    reason: str


def fail(check: str, reason: str) -> Finding:
    return Finding("FAIL", check, reason)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def added_after_sealing(name: str) -> str | None:
    """The project tool that adds this unlisted file to a sealed folder, or None when none does."""
    return next((tool for pattern, tool in AFTER_SEALING.items() if re.fullmatch(pattern, name)), None)


def check_sums(run_dir: Path) -> list[Finding]:
    """Every file listed with its digest, and no file unlisted but those a project tool adds after sealing (a NOTE)."""
    try:
        listed = read_sha256sums(run_dir / SUMS_NAME)
    except ValueError as error:
        return [fail("sums", str(error))]
    present = directory_digests(run_dir)
    unlisted = sorted(set(present) - set(listed))
    findings = [Finding("NOTE", "sums", f"{name} was added after {SUMS_NAME}, by {tool}; not checked")
                for name in unlisted if (tool := added_after_sealing(name)) is not None]
    findings += [fail("sums", f"{name} is not listed in {SUMS_NAME}") for name in unlisted if added_after_sealing(name) is None]
    findings += [fail("sums", f"{name} is listed in {SUMS_NAME} but absent") for name in sorted(set(listed) - set(present))]
    findings += [fail("sums", f"{name} differs from its {SUMS_NAME} digest") for name in sorted(set(listed) & set(present))
                 if listed[name] != present[name]]
    return findings


def check_host(host: dict[str, Any]) -> list[Finding]:
    if host.get("exit_status") != 0:
        return [fail("bundle", f"host.json records the container's exit status as {host.get('exit_status')!r}, not 0")]
    return []


def check_clock(run_result: dict[str, Any]) -> list[Finding]:
    """The caps from the receipt's seconds; the stored flags and status must agree with them."""
    clock = run_result["clock"]
    warmup_valid = clock["warmup_seconds"] <= clock["warmup_cap_seconds"]
    eval_valid = clock["eval_seconds"] <= clock["eval_cap_seconds"]
    train_valid = clock["train_seconds"] <= clock["budget_seconds"] * 1.01  # benchmark.Clock.valid's allowance
    findings = []
    if not warmup_valid:
        findings.append(fail("clock", f"warmup {clock['warmup_seconds']:.1f} s is over the {clock['warmup_cap_seconds']:.0f} s cap"))
    if not eval_valid:
        findings.append(fail("clock", f"in-run evals {clock['eval_seconds']:.1f} s are over the {clock['eval_cap_seconds']:.0f} s cap"))
    if not train_valid:
        findings.append(fail("clock", f"training {clock['train_seconds']:.1f} s is over the {clock['budget_seconds']:.0f} s budget"))
    stored = (clock["warmup_valid"], clock["eval_valid"], clock["valid"])
    if stored != (warmup_valid, eval_valid, warmup_valid and eval_valid and train_valid):
        findings.append(fail("clock", f"the receipt's stored flags {stored} disagree with its seconds"))
    if run_result["status"] != "completed":
        findings.append(fail("clock", f"status is {run_result['status']}"))
    return findings


def check_key(run_result: dict[str, Any], machine: dict[str, Any], data_dir: Path | None) -> list[Finding]:
    """The comparison key and the evaluator from this checkout's benchmark; the start's view equal to the result's."""
    findings = []
    if benchmark.comparison_key(run_result["benchmark_id"], run_result["evaluator_sha256"], run_result["clock"]) != run_result["comparison_key"]:
        findings.append(fail("key", "the comparison key is not the one the benchmark computes from the result"))
    if run_result["evaluator_sha256"] != benchmark.evaluator_sha256():
        findings.append(fail("key", f"evaluator {run_result['evaluator_sha256']} is not this checkout's {benchmark.evaluator_sha256()}"))
    for field in ("benchmark_id", "evaluator_sha256", "candidate_sha256", "candidate_parts"):
        if machine[field] != run_result[field]:
            findings.append(fail("key", f"machine.json's {field} at the start differs from result.json's"))
    if data_dir is not None:
        manifest = benchmark.load_manifest(data_dir)
        if benchmark.benchmark_id(manifest) != run_result["benchmark_id"]:
            findings.append(fail("key", f"benchmark id {run_result['benchmark_id']} is not that of {data_dir}"))
        for name, digest in machine["training_data"].items():
            if manifest["files"][name]["sha256"] != digest:
                findings.append(fail("key", f"the run's {name} is not the one in {data_dir}'s manifest"))
    return findings


def strategies_sha256(directory: Path) -> str:
    """`strategies.sources_sha256` over a snapshot folder: each module's name and bytes, in file-name order."""
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def genome_sha256(genome: dict[str, Any]) -> str:
    """`strategies.genome_sha256` of a resolved genome, which `result.json` holds, without this checkout's registry,
    which may lack a module a submission added."""
    return hashlib.sha256(json.dumps(genome, sort_keys=True).encode("utf-8")).hexdigest()


def check_candidate(run_dir: Path, run_result: dict[str, Any], machine: dict[str, Any]) -> list[Finding]:
    """The snapshots and the genome against the candidate parts, the digest from the parts, and genome.json and the
    command's batch overrides against the genome the run trained."""
    parts = run_result["candidate_parts"]
    findings = []
    if benchmark.candidate_sha256(run_dir / "train.py") != parts["train.py"]:
        findings.append(fail("candidate", "the train.py snapshot is not the candidate's train.py"))
    if strategies_sha256(run_dir / "strategies") != parts["strategies"]:
        findings.append(fail("candidate", "the strategies snapshot is not the candidate's strategy modules"))
    if genome_sha256(run_result["genome"]) != parts["genome"]:
        findings.append(fail("candidate", "result.json's genome is not the candidate's genome"))
    if hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest() != run_result["candidate_sha256"]:
        findings.append(fail("candidate", "the candidate digest is not the digest of its parts"))
    argv = machine["argv"]
    stated = read_json(run_dir / "genome.json")["families"] if (run_dir / "genome.json").exists() else {}
    overrides = {field: int(argv[argv.index(flag) + 1]) for flag, field in WINDOW_FLAGS.items() if flag in argv}
    stated = {**stated, "batch": {**stated.get("batch", {}), **overrides}}
    for family, fields in stated.items():
        for key, value in fields.items():
            if run_result["genome"].get(family, {}).get(key) != value:
                findings.append(fail("candidate", f"genome {family}.{key} is {run_result['genome'].get(family, {}).get(key)!r}; "
                                                  f"genome.json and the command say {value!r}"))
    return findings


def check_curve(run_result: dict[str, Any]) -> list[Finding]:
    """The selection is the curve's minimum; the curve's steps and evals agree with the clock's counts."""
    curve, clock = run_result["curve"], run_result["clock"]
    if not curve:
        return [fail("curve", "the curve is empty")]
    findings = []
    steps = [point["step"] for point in curve]
    if steps != sorted(set(steps)):
        findings.append(fail("curve", f"the curve's steps {steps} do not increase"))
    best = min(curve, key=lambda point: point["loss"])
    if (run_result["selected_step"], run_result["selected_val_select_loss"]) != (best["step"], best["loss"]):
        findings.append(fail("curve", f"selected step {run_result['selected_step']} at {run_result['selected_val_select_loss']} is not the "
                                      f"curve's minimum, step {best['step']} at {best['loss']}"))
    if steps[-1] != clock["steps"]:
        findings.append(fail("curve", f"the last eval is at step {steps[-1]}, the clock applied {clock['steps']}"))
    if len(curve) != clock["eval_count"]:
        findings.append(fail("curve", f"the curve has {len(curve)} evals, the clock paused for {clock['eval_count']}"))
    if any(point["train_seconds"] > clock["train_seconds"] for point in curve):
        findings.append(fail("curve", "an eval is stamped later on the clock than the run's training seconds"))
    return findings


def console_objects(path: Path) -> list[dict[str, Any]]:
    """The JSON object lines of the raw log, in order; the evidence header and footer and other text are skipped."""
    objects = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line of a traceback or a library's message, not train.py's
            if isinstance(parsed, dict):
                objects.append(parsed)
    return objects


def check_console(run_dir: Path, run_result: dict[str, Any]) -> list[Finding]:
    """The evals, the final metrics and the ledger row the run printed, equal to `result.json`."""
    objects = console_objects(run_dir / "console.log")
    findings = []
    if [line["eval"] for line in objects if set(line) == {"eval"}] != run_result["curve"]:
        findings.append(fail("console", "the eval lines in console.log are not result.json's curve"))
    for split, values in run_result["metrics"].items():
        if {split: {key: values[key] for key in ("loss", "accuracy", "mcc")}} not in objects:
            findings.append(fail("console", f"console.log holds no {split} line equal to result.json's metrics"))
    if not objects or objects[-1] != benchmark.ledger_row(run_result):
        findings.append(fail("console", "console.log's last JSON line is not result.json's ledger row"))
    return findings


def check_hardware(run_dir: Path, run_result: dict[str, Any], machine: dict[str, Any]) -> list[Finding]:
    """Exactly one GPU, a GH200, by inventory and by torch, with the sampler's files present."""
    gpus = machine["gpu"]
    if isinstance(gpus, str):
        return [fail("hardware", f"machine.json records the GPU as {gpus!r}")]
    findings = []
    if len(gpus) != 1:
        findings.append(fail("hardware", f"nvidia-smi lists {len(gpus)} GPUs; a record runs on a machine with exactly one"))
    findings += [fail("hardware", f"GPU {gpu['index']} is {gpu['name']!r}, not {GH200_NAME!r}") for gpu in gpus if gpu["name"] != GH200_NAME]
    if run_result["environment"]["gpu"] != GH200_NAME:
        findings.append(fail("hardware", f"torch ran on {run_result['environment']['gpu']!r}, not {GH200_NAME!r}"))
    findings += [fail("hardware", f"{name} is absent") for name in ("nvidia-smi_start.txt", "nvidia-smi_end.txt", "gpu_samples.csv")
                 if not (run_dir / name).exists()]
    return findings


def training_window(phases: list[dict[str, Any]], clock: dict[str, Any]) -> tuple[datetime, datetime]:
    """From the training mark, the clocked steps and in-run evals, recomputed from the receipt."""
    marks = [phase for phase in phases if phase["phase"] == "training"]
    if len(marks) != 1:
        raise ValueError(f"phases.jsonl holds {len(marks)} training marks, not one")
    start = datetime.strptime(marks[0]["local"], TIMESTAMP_FORMAT)
    return start, start + timedelta(seconds=clock["train_seconds"] + clock["eval_seconds"] + clock["discarded_seconds"])


def check_gpu_use(run_dir: Path, run_result: dict[str, Any], machine: dict[str, Any], host: dict[str, Any]) -> list[Finding]:
    """The sampler's readings over the training window: the UUID is the inventory's (FAIL), and utilization, gaps,
    throttling and other processes (REVIEW)."""
    phases = [json.loads(line) for line in (run_dir / "phases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    start, end = training_window(phases, run_result["clock"])
    samples, truncated = read_samples(run_dir / "gpu_samples.csv")
    summary = summarize_samples(samples, (start, end), "read by verify_run.py", [], {}, truncated)
    uuid = machine["gpu"][0]["uuid"]
    if summary["uuids"] != [uuid]:
        return [fail("hardware", f"the sampler read {summary['uuids']}, machine.json lists {uuid}")]
    if not summary["window_samples"]:
        return [fail("hardware", "the sampler has no reading while the run trained")]
    findings = []
    utilization = summary["utilization_percent_median"]
    if utilization is None or utilization < UTILIZATION_REVIEW_PERCENT:
        findings.append(Finding("REVIEW", "gpu", f"median utilization while training is {utilization} %, under {UTILIZATION_REVIEW_PERCENT:.0f} %"))
    inside = sorted(sample["time"] for sample in samples if start <= sample["time"] <= end)
    edges = [(inside[0] - start).total_seconds(), (end - inside[-1]).total_seconds()]
    gap = max([summary["largest_gap_seconds"] or 0.0, *edges])
    if gap > SAMPLER_GAP_REVIEW_SECONDS:
        findings.append(Finding("REVIEW", "gpu", f"the sampler went {gap:.1f} s without a reading while training, over {SAMPLER_GAP_REVIEW_SECONDS:.0f} s"))
    throttled = [reason for reason in summary["event_reasons_seen"] if int(reason, 16) & THROTTLE_REVIEW_MASK]
    if throttled:
        findings.append(Finding("REVIEW", "gpu", f"the driver reported throttling while training: {', '.join(throttled)}"))
    others = {"machine.json at the start": machine["compute_apps"], "host.json before the run": host.get("compute_apps_before"),
              "host.json after the run": host.get("compute_apps_after")}
    findings += [Finding("REVIEW", "gpu", f"another process was on the GPU ({where}): {apps!r}") for where, apps in others.items() if apps]
    return findings


def rescore(run_dir: Path, run_result: dict[str, Any], data_dir: Path, device: torch.device, tokens_per_batch: int) -> list[Finding]:
    """Rebuild the model from the snapshot as `probe.py` does, load `best.pt` and score the splits `result.json` holds."""
    sys.pycache_prefix = tempfile.mkdtemp(prefix="verify_run_pycache_")  # no bytecode written to or read from the folder
    import probe  # the script beside this one: its loader is how a run's model is rebuilt from its train.py

    from speedrunning_plms.strategies import sources_sha256

    parts = run_result["candidate_parts"]
    if sources_sha256() != parts["strategies"]:
        return [fail("rescore", f"this checkout's strategy modules are not the run's ({parts['strategies']}); check out the "
                                "revision that matches the run's strategies/ snapshot")]
    if not run_result["metrics"]:
        return [fail("rescore", "result.json holds no final metrics to compare")]
    manifest = benchmark.load_manifest(data_dir)
    if benchmark.benchmark_id(manifest) != run_result["benchmark_id"]:
        return [fail("rescore", f"{data_dir} is not the run's data")]
    candidate = probe.load_candidate_module(run_dir / "train.py")
    model = candidate.build_candidate(run_result["genome"], max(manifest["max_tokens"].values())).to(device)
    state = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict({name: tensor.to(torch.float32) if tensor.is_floating_point() else tensor for name, tensor in state.items()})
    model.eval()
    findings = []
    for split, stored in run_result["metrics"].items():
        held_out = benchmark.load_held_out(data_dir, split)
        scored = benchmark.evaluate_split(model, held_out, device, tokens_per_batch=tokens_per_batch, keep_probabilities=False)
        print(json.dumps({"rescore": split, "loss": scored["loss"], "stored_loss": stored["loss"], "accuracy": scored["accuracy"],
                          "stored_accuracy": stored["accuracy"]}), file=sys.stderr, flush=True)
        if abs(scored["loss"] - stored["loss"]) > RESCORE_LOSS_TOLERANCE:
            findings.append(fail("rescore", f"{split} loss rescored {scored['loss']:.4f}, result.json {stored['loss']:.4f}"))
        if abs(scored["accuracy"] - stored["accuracy"]) > RESCORE_ACCURACY_TOLERANCE:
            findings.append(fail("rescore", f"{split} accuracy rescored {scored['accuracy']:.4f}, result.json {stored['accuracy']:.4f}"))
    return findings


def verify(run_dir: Path, data_dir: Path | None = None, rescore_device: torch.device | None = None,
           tokens_per_batch: int = 32_768) -> list[Finding]:
    """Every check on the folder, in order; a missing file fails the bundle and the checks that read it are skipped."""
    missing = [name for name in REQUIRED if not (run_dir / name).exists()]
    if missing:
        return [fail("bundle", f"{name} is absent") for name in missing]
    run_result, machine, host = (read_json(run_dir / name) for name in ("result.json", "machine.json", "host.json"))
    findings = check_sums(run_dir) + check_host(host) + check_clock(run_result) + check_key(run_result, machine, data_dir)
    findings += check_candidate(run_dir, run_result, machine) + check_curve(run_result) + check_console(run_dir, run_result)
    hardware = check_hardware(run_dir, run_result, machine)
    findings += hardware or check_gpu_use(run_dir, run_result, machine, host)
    if rescore_device is not None:
        assert data_dir is not None, "a rescore reads the held-out splits"
        findings += rescore(run_dir, run_result, data_dir, rescore_device, tokens_per_batch)
    return findings


def verdict(findings: list[Finding]) -> str:
    levels = {finding.level for finding in findings} - {"NOTE"}
    return "FAIL" if "FAIL" in levels else "REVIEW" if levels else "PASS"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=None, help="the benchmark data: checks its id and, with --rescore, scores on it")
    parser.add_argument("--rescore", action="store_true", help="rebuild the model from the snapshot and score best.pt again")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="the rescore's device")
    parser.add_argument("--tokens-per-batch", type=int, default=32_768, help="the rescore's eval batch")
    arguments = parser.parse_args()
    if arguments.rescore and arguments.data_dir is None:
        parser.error("--rescore needs --data-dir")
    findings = verify(arguments.run_dir, arguments.data_dir, torch.device(arguments.device) if arguments.rescore else None,
                      arguments.tokens_per_batch)
    outcome = verdict(findings)
    print(outcome)
    for finding in findings:
        print(f"{finding.level} {finding.check}: {finding.reason}")
    if arguments.data_dir is None:
        print("note: the benchmark id was not checked against the data; pass --data-dir")
    sys.exit({"PASS": 0, "FAIL": 1, "REVIEW": 3}[outcome])


if __name__ == "__main__":
    main()
