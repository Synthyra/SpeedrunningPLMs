#!/usr/bin/env bash
# Run queued candidates one after another on the box, so the GPU never waits for the laptop.
#
#   bash queue.sh                      every candidates/queue/<name>.py in order, each with its <name>.txt hypothesis
#   bash queue.sh 015_normuon 016_cautious_wd   only these
#   An optional <name>.args holds extra train.py arguments on one line, such as `--seed 1` for a confirmation run.
#   bash queue.sh --wait --probe 015_normuon --reprobe-baselines
#       wait for the GPU, probe a kept run first, run the queue, then re-probe the reference models
#
# Each queued file is a complete train.py built on the incumbent; it becomes the shipped tree's train.py for its run
# and `run_gh200.sh train` snapshots it into the run folder. A run that already has a result is skipped. The queue
# does not decide verdicts: the driver reads every result.json afterwards, keeps the best candidate that beats the
# incumbent, and builds the next queue on it.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="${SPEEDRUN_STATE:-$HOME/speedrun}"
cd "$PROJECT"

wait_for_gpu=0
probe_runs=()
reprobe_baselines=0
while [ $# -gt 0 ]; do
  case "$1" in
    --wait) wait_for_gpu=1; shift ;;                       # start once no train, probe or baselines process runs
    --probe) probe_runs+=("$2"); shift 2 ;;                # probe this run folder first (a kept candidate)
    --reprobe-baselines) reprobe_baselines=1; shift ;;     # the reference models' probe again, after the queue
    *) break ;;
  esac
done
if [ "$wait_for_gpu" = 1 ]; then
  while pgrep -f "[t]rain.py|[p]robe.py|[b]aselines.py" >/dev/null; do sleep 30; done
  echo "queue: gpu free $(date -u +%H:%M:%S)" >&2
fi
for run in "${probe_runs[@]}"; do
  echo "queue: probe $run $(date -u +%H:%M:%S)" >&2
  bash run_gh200.sh probe --run-dir "/state/runs/$run" || echo "queue: probe $run failed" >&2  # the container path
done

if [ $# -gt 0 ]; then
  names=("$@")
else
  names=()
  for file in candidates/queue/*.py; do names+=("$(basename "$file" .py)"); done
fi

for name in "${names[@]}"; do
  if [ -f "$STATE/runs/$name/result.json" ]; then
    echo "queue: $name done" >&2
    continue
  fi
  hypothesis="$(cat "candidates/queue/$name.txt")"
  cp "candidates/queue/$name.py" train.py
  echo "queue: $name starts $(date -u +%H:%M:%S)" >&2
  extra=()
  [ -f "candidates/queue/$name.args" ] && read -r -a extra < "candidates/queue/$name.args"  # e.g. --seed 1
  bash run_gh200.sh train --name "$name" --hypothesis "$hypothesis" "${extra[@]}" || echo "queue: $name failed" >&2
done
if [ "$reprobe_baselines" = 1 ]; then
  echo "queue: reprobe baselines $(date -u +%H:%M:%S)" >&2
  bash run_gh200.sh baselines --probe-only || echo "queue: reprobe failed" >&2
fi
echo "queue: finished $(date -u +%H:%M:%S)" >&2
