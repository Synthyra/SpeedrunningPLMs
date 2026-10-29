#!/usr/bin/env bash
# The authorized Docker entrypoint on the speedrun GH200: every workload runs in the image, nothing in host Python.
#
#   bash run_gh200.sh build                      build the image from the Dockerfile (aarch64, CUDA 12.8)
#   bash run_gh200.sh prepare                    build the benchmark data (HF_TOKEN from the launcher's environment)
#   bash run_gh200.sh pilot                      a 60 s clock run of the working train.py, no final evals
#   bash run_gh200.sh train --name 000_baseline  a full run; result in ~/speedrun/runs/<name>/result.json
#   bash run_gh200.sh probe --run-dir /state/runs/000_baseline
#   bash run_gh200.sh baselines [--models ...]   the reference models, results in ~/speedrun/runs/baselines
#   bash run_gh200.sh shell                      an interactive shell in the image
#
# `ws remote run` ships the project and the foundry import root into a fresh run directory and sets PYTHONPATH to
# the import roots; that run directory is mounted at its own path so PYTHONPATH holds inside the container. Data,
# runs and caches live under $STATE and survive every run directory.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="${SPEEDRUN_STATE:-$HOME/speedrun}"
IMAGE="${SPEEDRUN_IMAGE:-speedrun_plm:afdb_v6}"
mkdir -p "$STATE/data" "$STATE/runs" "$STATE/cache/huggingface" "$STATE/cache/torchinductor" "$STATE/cache/triton"

# The run directory (holds launch.py and tree/) when shipped, else the workspace root (holds foundry/).
ROOT="$PROJECT"
while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/launch.py" ] && [ ! -d "$ROOT/foundry" ]; do ROOT="$(dirname "$ROOT")"; done
[ "$ROOT" != "/" ] || { echo "run_gh200.sh: neither a shipped run directory nor the workspace root holds $PROJECT" >&2; exit 2; }

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

case "$command" in
  build)
    sudo docker build -t "$IMAGE" "$PROJECT"
    sudo docker image inspect "$IMAGE" --format 'image {{.Id}} created {{.Created}}'
    ;;
  prepare)
    docker_run "$IMAGE" python prepare.py --output-dir /state/data/afdb_v6 --source-dir /state/data/source "$@"
    ;;
  pilot)
    sudo rm -rf "$STATE/runs/pilot"  # root-owned, the container wrote it
    docker_run "$IMAGE" python train.py --data-dir /state/data/afdb_v6 --output-dir /state/runs/pilot --name pilot \
      --budget 60 --pilot --no-wandb --skip-final-evals "$@"
    ;;
  train)
    [ "${1:-}" = "--name" ] && [ -n "${2:-}" ] || { echo "train: --name <name> comes first" >&2; exit 2; }
    sudo mkdir -p "$STATE/runs/$2" && sudo cp "$PROJECT/train.py" "$STATE/runs/$2/train.py"  # the candidate's bytes, before the run
    docker_run "$IMAGE" python train.py --data-dir /state/data/afdb_v6 --output-dir "/state/runs/$2" "$@"
    ;;
  probe)
    docker_run "$IMAGE" python probe.py --data-dir /state/data/afdb_v6 "$@"
    ;;
  baselines)
    docker_run "$IMAGE" python baselines.py --data-dir /state/data/afdb_v6 --output-dir /state/runs/baselines "$@"
    ;;
  python)
    docker_run "$IMAGE" python "$@"
    ;;
  shell)
    docker_run -it "$IMAGE" bash
    ;;
  *)
    echo "usage: bash run_gh200.sh {build|prepare|pilot|train|probe|baselines|shell} [arguments]" >&2
    exit 2
    ;;
esac
