#!/usr/bin/env bash
# The search loop on the box: propose, run, collect, again, until the deadline or a stop file, so the GPU never idles.
#
#   bash autosearch.sh --until 2026-10-31T01:30:00Z [--count 4]
#       each cycle: `autosearch.py next` (in the image) writes the queue entries of up to --count runs, newest batch
#       first, or of a newly proposed batch; `queue.sh --wait` runs them one after another; `autosearch.py collect`
#       reads the results into the state and the ledger under $STATE/search; a cycle starts only if its runs fit
#       before --until (about 13 min a run), and never after `touch $STATE/search/stop`
#   To restart with new strategy modules, ship the tree again with the same command: the new loop records its PID,
#   waits for the running cycle to collect, and the old loop ends at its next cycle.
#
# Seed $STATE/search/search.json and runs.jsonl from the workspace's experiment folder before the first start; while
# the loop runs they are the search's authority and the workspace copies are fetched from them. The loop ships
# nothing: the tree it runs in is the one `ws remote run` shipped, so a new strategy module needs a stop and a new
# shipment. `queue.sh` holds the queue lock, so a queue shipped by hand waits for the cycle and the cycle for it.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="${SPEEDRUN_STATE:-$HOME/speedrun}"
SEARCH="$STATE/search"
MINUTES_PER_RUN=13  # 600 s of training, about 2 min of warmup and final evals, a margin

until_utc=""
count=4
while [ $# -gt 0 ]; do
  case "$1" in
    --until) until_utc="$2"; shift 2 ;;   # ISO 8601 UTC, such as 2026-10-31T01:30:00Z
    --count) count="$2"; shift 2 ;;       # runs per proposed batch
    *) echo "autosearch.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
cd "$PROJECT"
mkdir -p "$SEARCH"
[ -f "$SEARCH/search.json" ] || { echo "autosearch.sh: seed $SEARCH/search.json and runs.jsonl first" >&2; exit 2; }

log() { echo "autosearch: $* $(date -u +%FT%TZ)" | tee -a "$SEARCH/loop.log" >&2; }

echo $$ > "$SEARCH/loop.pid"  # the newest loop owns the search: an older one ends at its next cycle
exec 8>"$SEARCH/cycle.lock"
while :; do
  flock 8  # one cycle at a time across loops: a restarted loop (new tree, new modules) waits for this cycle's collect
  if [ "$(cat "$SEARCH/loop.pid")" != "$$" ]; then log "a newer loop took over, this one ends"; break; fi
  if [ -f "$SEARCH/stop" ]; then log "stop file found, loop ends"; break; fi
  if [ -n "$until_utc" ]; then
    now=$(date -u +%s); end=$(date -u -d "$until_utc" +%s); need=$(( (count * MINUTES_PER_RUN + 3) * 60 ))
    if [ $((now + need)) -gt "$end" ]; then log "a batch of $count would end after $until_utc, loop ends"; break; fi
  fi
  log "cycle starts"
  bash run_gh200.sh python autosearch.py next --state-dir /state/search --count "$count" 2>&1 | tee -a "$SEARCH/loop.log"
  read -r -a names < "$SEARCH/queue_next.txt"
  if [ -n "$until_utc" ]; then  # a pending batch can be longer than --count runs
    now=$(date -u +%s); need=$(( (${#names[@]} * MINUTES_PER_RUN + 3) * 60 ))
    if [ $((now + need)) -gt "$end" ]; then log "${#names[@]} runs would end after $until_utc, loop ends"; break; fi
  fi
  bash queue.sh --wait "${names[@]}" 2>&1 | tee -a "$SEARCH/loop.log"
  crashed=()
  for name in "${names[@]}"; do [ -f "$STATE/runs/$name/result.json" ] || crashed+=("$name"); done
  bash run_gh200.sh python autosearch.py collect --state-dir /state/search ${crashed[@]+--crashed "${crashed[@]}"} 2>&1 | tee -a "$SEARCH/loop.log"
done
