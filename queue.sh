#!/usr/bin/env bash
# Run queued candidates one after another on the box, so the GPU never waits for the laptop.
#
#   bash queue.sh                      every candidates/queue/<name>.py in order, each with its <name>.txt hypothesis
#   bash queue.sh 015_normuon 016_cautious_wd   only these
#   An optional <name>.args holds extra train.py arguments on one line, such as `--seed 1` for a confirmation run.
#
# Each queued file is a complete train.py built on the incumbent; it becomes the shipped tree's train.py for its run
# and `run_gh200.sh train` snapshots it into the run folder. A run that already has a result is skipped. The queue
# does not decide verdicts: the driver reads every result.json afterwards, keeps the best candidate that beats the
# incumbent, and builds the next queue on it.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="${SPEEDRUN_STATE:-$HOME/speedrun}"
cd "$PROJECT"

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
echo "queue: finished $(date -u +%H:%M:%S)" >&2
