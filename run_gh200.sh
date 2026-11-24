#!/usr/bin/env bash
# The authorized Docker entrypoint on the speedrun GH200: every workload runs in the image, nothing in host Python.
#
#   bash run_gh200.sh build                      build the image from the Dockerfile (aarch64, CUDA 12.8)
#   bash run_gh200.sh build-kumo                 the proposer's image, the same plus Kumo Tabular (Dockerfile.kumo)
#   bash run_gh200.sh python-kumo <script> ...   a script in that image: a tabular arm's proposals, never training
#   bash run_gh200.sh build-perplexity           the perplexity plot's image, the same plus MMseqs2 and sentencepiece
#   bash run_gh200.sh python-perplexity <args>   perplexity.py in that image: the corpus, the models' scores, the star
#   bash run_gh200.sh prepare                    build the benchmark data (HF_TOKEN from the launcher's environment)
#   bash run_gh200.sh derive                     the data runs use: that data without exact copies of the temporal benchmark
#   bash run_gh200.sh data-dir                   print where runs read their data (a path in the image)
#   bash run_gh200.sh pilot                     a 60 s clock run of the working train.py, no final evals
#   bash run_gh200.sh train --name 000_baseline [--genome g.json]  a full run; result in ~/speedrun/runs/<name>/result.json,
#                                                with the run evidence verify_run.py checks (host.json, console.log, SHA256SUMS)
#   bash run_gh200.sh probe --run-dir /state/runs/000_baseline
#   bash run_gh200.sh baselines [--models ...]   the reference models, results in ~/speedrun/runs/baselines
#   bash run_gh200.sh shell                      an interactive shell in the image
#   bash run_gh200.sh wait-gpu                   return once no other process holds the GPU (queue.sh calls it per run)
#
# `ws remote run` ships the project and the foundry import root into a fresh run directory and sets PYTHONPATH to
# the import roots; that run directory is mounted at its own path so PYTHONPATH holds inside the container. Data,
# runs and caches live under $STATE and survive every run directory.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="${SPEEDRUN_STATE:-$HOME/speedrun}"
IMAGE="${SPEEDRUN_IMAGE:-speedrun_plm:afdb_v6}"
KUMO_IMAGE="${SPEEDRUN_KUMO_IMAGE:-speedrun_plm:afdb_v6_kumo}"
PERPLEXITY_IMAGE="${SPEEDRUN_PERPLEXITY_IMAGE:-speedrun_plm:afdb_v6_perplexity}"
# Where runs read their data, in the image: since 2026-11-23 AFDB v6 without exact copies of perplexity.py's new UniProt
# proteins (`derive` builds it); SPEEDRUN_DATA=/state/data/afdb_v6 reads the data before, as a paired seed must.
DATA="${SPEEDRUN_DATA:-/state/data/afdb_v6_minus_uniprot_2026_04}"
mkdir -p "$STATE/data" "$STATE/runs" "$STATE/cache/huggingface" "$STATE/cache/torchinductor" "$STATE/cache/triton"

# The run directory (holds launch.py and tree/) when shipped, else the workspace root (holds foundry/), else a clone of
# the GitHub repository, which vendors foundry in src/foundry and is its own root.
ROOT="$PROJECT"
while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/launch.py" ] && [ ! -d "$ROOT/foundry" ]; do ROOT="$(dirname "$ROOT")"; done
if [ "$ROOT" = "/" ] && [ -d "$PROJECT/src/foundry" ]; then ROOT="$PROJECT"; fi
[ "$ROOT" != "/" ] || { echo "run_gh200.sh: no shipped run directory or workspace root holds $PROJECT, and it has no src/foundry" >&2; exit 2; }

command=${1:-}
shift || true

docker_run() {
  # --ipc=host for PyTorch; the caches persist across runs so compile and Hub downloads happen once.
  # sudo resets the environment; the credential names (never values) pass through it and into the container by name.
  sudo --preserve-env=HF_TOKEN,WANDB_API_KEY,WANDB_MODE docker run --rm --gpus all --ipc=host \
    -v "$ROOT:$ROOT" -v "$STATE:/state" -w "$PROJECT" \
    -e PYTHONPATH="$PROJECT/src:${PYTHONPATH:-$ROOT}" \
    -e HF_HOME=/state/cache/huggingface -e TORCHINDUCTOR_CACHE_DIR=/state/cache/torchinductor \
    -e TRITON_CACHE_DIR=/state/cache/triton -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e HF_TOKEN -e WANDB_API_KEY -e WANDB_MODE \
    "$@"
}

json_string() {  # one JSON string: the control characters JSON forbids dropped, backslash, quote, tab, CR and LF escaped
  local value
  value=$(printf '%s' "$1" | tr -d '\000-\010\013\014\016-\037')
  value=${value//\\/\\\\}; value=${value//\"/\\\"}; value=${value//$'\t'/\\t}; value=${value//$'\r'/\\r}; value=${value//$'\n'/\\n}
  printf '"%s"' "$value"
}

write_host_json() {  # <run folder> <exit status or null> <finished, a JSON string or null> <compute apps after>
  local after=null
  [ "$2" = null ] || after=$(json_string "$4")
  printf '{"schema_version": 1, "started_utc": %s, "finished_utc": %s, "exit_status": %s, "kernel": %s, "image": %s, "image_id": %s, "image_repo_digests": %s, "git_ok": %s, "git_head": %s, "git_status_porcelain": %s, "source_patch": %s, "compute_apps_before": %s, "compute_apps_after": %s}\n' \
    "$(json_string "$started_utc")" "$3" "$2" "$(json_string "$(uname -r)")" "$(json_string "$IMAGE")" "$(json_string "$image_id")" \
    "${image_digests:-null}" "$git_ok" "$(json_string "$git_head")" "$(json_string "$git_status")" \
    "$([ -n "$git_status" ] && echo true || echo false)" "$(json_string "$apps_before")" "$after" | sudo tee "$1/host.json" >/dev/null
}

case "$command" in
  build)
    sudo docker build -t "$IMAGE" "$PROJECT"
    sudo docker image inspect "$IMAGE" --format 'image {{.Id}} created {{.Created}}'
    ;;
  build-kumo)
    sudo docker build -t "$KUMO_IMAGE" -f "$PROJECT/Dockerfile.kumo" "$PROJECT"
    sudo docker image inspect "$KUMO_IMAGE" --format 'image {{.Id}} created {{.Created}}'
    ;;
  python-kumo)
    docker_run "$KUMO_IMAGE" python "$@"
    ;;
  build-perplexity)
    sudo docker build -t "$PERPLEXITY_IMAGE" -f "$PROJECT/Dockerfile.perplexity" "$PROJECT"
    sudo docker image inspect "$PERPLEXITY_IMAGE" --format 'image {{.Id}} created {{.Created}}'
    ;;
  python-perplexity)
    docker_run "$PERPLEXITY_IMAGE" python perplexity.py "$@"
    ;;
  prepare)
    docker_run "$IMAGE" python prepare.py --output-dir /state/data/afdb_v6 --source-dir /state/data/source "$@"
    ;;
  derive)
    docker_run "$IMAGE" python prepare.py --output-dir "$DATA" --exclude-from /state/data/afdb_v6 \
      --exclusion /state/perplexity/exclusion.json "$@"
    ;;
  data-dir)
    echo "$DATA"
    ;;
  pilot)
    sudo rm -rf "$STATE/runs/pilot"  # root-owned, the container wrote it
    docker_run "$IMAGE" python train.py --data-dir "$DATA" --output-dir /state/runs/pilot --name pilot \
      --budget 60 --pilot --no-wandb --skip-final-evals "$@"
    ;;
  train)
    [ "${1:-}" = "--name" ] && [ -n "${2:-}" ] || { echo "train: --name <name> comes first" >&2; exit 2; }
    RUN="$STATE/runs/$2"
    [ ! -e "$RUN/result.json" ] || { echo "train: $RUN already holds a result; its evidence stays as it is" >&2; exit 1; }
    sudo mkdir -p "$RUN" && sudo cp "$PROJECT/train.py" "$RUN/train.py"  # the candidate's bytes, before the run
    sudo cp -r "$PROJECT/src/speedrunning_plms/strategies" "$RUN/strategies"  # the modules a genome builds from
    for ((index = 1; index < $#; index++)); do  # and the genome itself, when there is one
      if [ "${!index}" = "--genome" ]; then
        next=$((index + 1))
        sudo cp "$PROJECT/${!next}" "$RUN/genome.json"
      fi
    done
    # Run evidence (verify_run.py checks it): host.json before and after, the console, the patch of a dirty tree, and
    # SHA256SUMS over the whole folder. stdout and stderr each reach the caller as before, and both reach console.log.
    started_utc=$(date -u +%FT%TZ)
    apps_before=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader) || {
      echo "train: nvidia-smi failed on the host; a run that cannot be confirmed does not start" >&2; exit 1; }
    git_ok=true
    git_head=$(git -C "$PROJECT" rev-parse HEAD 2>&1) || git_ok=false
    git_status=""
    if [ "$git_ok" = true ]; then
      git_status=$(git -C "$PROJECT" status --porcelain -- . 2>&1)
      [ -z "$git_status" ] || git -C "$PROJECT" diff HEAD -- . | sudo tee "$RUN/source.patch" >/dev/null
    fi
    image_id=$(sudo docker image inspect "$IMAGE" --format '{{.Id}}')
    image_digests=$(sudo docker image inspect "$IMAGE" --format '{{json .RepoDigests}}')
    write_host_json "$RUN" null null ""
    sudo rm -f "$RUN/console.log" "$RUN/SHA256SUMS"
    status=0
    { docker_run "$IMAGE" python train.py --data-dir "$DATA" --output-dir "/state/runs/$2" "$@" 2>&1 1>&3 3>&- |
        sudo tee -a "$RUN/console.log" >&2 3>&-; } 3>&1 | sudo tee -a "$RUN/console.log" || status=$?
    apps_after=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>&1) ||
      apps_after="nvidia-smi failed after the run: $apps_after"
    write_host_json "$RUN" "$status" "\"$(date -u +%FT%TZ)\"" "$apps_after"
    (cd "$RUN" && sudo find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | sudo xargs -0 -r sha256sum) |
      sudo tee "$RUN/SHA256SUMS" >/dev/null
    exit "$status"
    ;;
  probe)
    docker_run "$IMAGE" python probe.py --data-dir "$DATA" "$@"
    ;;
  baselines)
    docker_run "$IMAGE" python baselines.py --data-dir "$DATA" --output-dir /state/runs/baselines "$@"
    ;;
  python)
    docker_run "$IMAGE" python "$@"
    ;;
  shell)
    docker_run -it "$IMAGE" bash
    ;;
  wait-gpu)
    # Another session's work on a shared box comes first (Logan, 2026-11-23), and a timed run beside another process
    # is not a clean run. Return at once on an idle GPU; after a busy one, only once it has stayed idle for 3 minutes,
    # so a job that pauses between its own steps keeps the GPU. The MPS daemon holds no work and does not count.
    idle_checks=0
    seen_busy=0
    while true; do
      busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader) || {
        echo "wait-gpu: nvidia-smi failed on the host" >&2; exit 1; }
      busy=$(printf '%s\n' "$busy" | grep -v -e "nvidia-cuda-mps-server" -e '^$' || true)
      if [ -n "$busy" ]; then
        [ "$seen_busy" = 1 ] || echo "wait-gpu: busy at $(date -u +%H:%M:%S): $(printf '%s' "$busy" | tr '\n' ';')" >&2
        seen_busy=1
        idle_checks=0
      elif [ "$seen_busy" = 0 ]; then
        break
      else
        idle_checks=$((idle_checks + 1))
        [ "$idle_checks" -lt 4 ] || { echo "wait-gpu: free since $(date -u -d '-3 min' +%H:%M:%S)" >&2; break; }
      fi
      sleep 60
    done
    ;;
  *)
    echo "usage: bash run_gh200.sh {build|build-kumo|python|python-kumo|prepare|pilot|train|probe|baselines|shell} [arguments]" >&2
    exit 2
    ;;
esac
