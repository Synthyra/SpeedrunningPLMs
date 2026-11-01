#!/usr/bin/env bash
# The search loop on the box: propose, run, collect, again, until the deadline or a stop file, so the GPU never idles.
#
#   bash autosearch.sh --until 2026-10-31T01:30:00Z [--count 4] [--search search] [--after 2026-10-31T02:00:00Z]
#       each cycle: `autosearch.py next` (in the image) writes the queue entries of up to --count runs, newest batch
#       first, or of a newly proposed batch; `queue.sh --wait` runs them one after another; `autosearch.py collect`
#       reads the results into the state and the ledger under $STATE/<--search>; a cycle runs the prefix of its batch
#       that ends before --until (a rung run its seconds plus three minutes, a full run the track's minutes; the rest
#       stays pending), never before --after, and never after `touch $STATE/<--search>/stop`
#   Each track has its own state folder (`search` for the 10-minute track, `search_1h` for the 1-hour one); the
#   state's `track` names the clock, and the loop reads the minutes a run from it.
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

until_utc=""
after_utc=""
count=4
search_name="search"
while [ $# -gt 0 ]; do
  case "$1" in
    --until) until_utc="$2"; shift 2 ;;   # ISO 8601 UTC, such as 2026-10-31T01:30:00Z
    --after) after_utc="$2"; shift 2 ;;   # the first cycle waits for this time (the 1-hour track starts when the 10-minute window ends)
    --count) count="$2"; shift 2 ;;       # runs per proposed batch
    --search) search_name="$2"; shift 2 ;;  # the state folder under $STATE: one per track
    *) echo "autosearch.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
SEARCH="$STATE/$search_name"
cd "$PROJECT"
mkdir -p "$SEARCH"
[ -f "$SEARCH/search.json" ] || { echo "autosearch.sh: seed $SEARCH/search.json and runs.jsonl first" >&2; exit 2; }
# The state's track names its clock (`search.TRACKS`); a full run in minutes with warmup, final evals and a margin.
case "$(python3 -c "import json; print(json.load(open('$SEARCH/search.json')).get('track', '10min'))")" in
  10min) MINUTES_PER_RUN=13 ;;
  1h) MINUTES_PER_RUN=70 ;;
  *) echo "autosearch.sh: unknown track in $SEARCH/search.json" >&2; exit 2 ;;
esac

log() { echo "autosearch: $* $(date -u +%FT%TZ)" | tee -a "$SEARCH/loop.log" >&2; }
minutes_of() {  # a run's minutes from its name: a rung run (`<name>_r<seconds>`) is its seconds plus warmup and evals
  case "$1" in
    *_r[0-9]*) echo $(( ${1##*_r} / 60 + 3 )) ;;
    *) echo "$MINUTES_PER_RUN" ;;
  esac
}

echo $$ > "$SEARCH/loop.pid"  # the newest loop owns the search: an older one ends at its next cycle
exec 8>"$SEARCH/cycle.lock"
while :; do
  flock 8  # one cycle at a time across loops: a restarted loop (new tree, new modules) waits for this cycle's collect
  if [ "$(cat "$SEARCH/loop.pid")" != "$$" ]; then log "a newer loop took over, this one ends"; break; fi
  if [ -f "$SEARCH/stop" ]; then log "stop file found, loop ends"; break; fi
  if [ -n "$after_utc" ] && [ "$(date -u +%s)" -lt "$(date -u -d "$after_utc" +%s)" ]; then
    log "waiting for $after_utc"; flock -u 8; sleep 60; continue
  fi
  log "cycle starts"
  bash run_gh200.sh python autosearch.py next --state-dir "/state/$search_name" --count "$count" 2>&1 | tee -a "$SEARCH/loop.log"
  read -r -a names < "$SEARCH/queue_next.txt"
  if [ -n "$until_utc" ]; then  # run the longest prefix of the batch that ends before --until; the rest stays pending
    now=$(date -u +%s); end=$(date -u -d "$until_utc" +%s); need=$((3 * 60)); fit=0
    for name in "${names[@]}"; do
      need=$((need + $(minutes_of "$name") * 60))
      if [ $((now + need)) -gt "$end" ]; then break; fi
      fit=$((fit + 1))
    done
    if [ "$fit" -eq 0 ]; then log "no run of ${names[*]} would end before $until_utc, loop ends"; break; fi
    if [ "$fit" -lt "${#names[@]}" ]; then log "$fit of ${#names[@]} runs end before $until_utc, the rest stay pending"; fi
    names=("${names[@]:0:$fit}")
  fi
  bash queue.sh --wait "${names[@]}" 2>&1 | tee -a "$SEARCH/loop.log"
  crashed=()
  for name in "${names[@]}"; do [ -f "$STATE/runs/$name/result.json" ] || crashed+=("$name"); done
  bash run_gh200.sh python autosearch.py collect --state-dir "/state/$search_name" ${crashed[@]+--crashed "${crashed[@]}"} 2>&1 | tee -a "$SEARCH/loop.log"
done
