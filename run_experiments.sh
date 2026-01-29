#!/usr/bin/env bash
# chmod +x run_experiments.sh
# ./run_experiments.sh
set -euo pipefail

# ─── Config ────────────────────────────────────────────────────────────────────
EXPERIMENT_DIR="./experiments"
IMAGE="speedrun_plm"
HOST_MOUNT="${PWD}:/workspace"
CONTAINER_WORKDIR="/workspace"
SHM_SIZE="128g"

# ─── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -d "$EXPERIMENT_DIR" ]; then
  echo "❌ Directory '$EXPERIMENT_DIR' not found!" >&2
  exit 1
fi

# ─── Prompt for token ──────────────────────────────────────────────────────────
read -rp "🔑 Enter your HuggingFace token: " HF_TOKEN
read -rp "🔑 Enter your wandb token: " WANDB_TOKEN

# ─── Detect GPUs ──────────────────────────────────────────────────────────────
if command -v nvidia-smi &> /dev/null; then
  NUM_GPUS=$(nvidia-smi -L | wc -l | tr -d '[:space:]')
else
  echo "⚠️  'nvidia-smi' not found—defaulting to 1 GPU"
  NUM_GPUS=1
fi
echo "🖥️  Using $NUM_GPUS GPU(s)"

# ─── Loop and launch ───────────────────────────────────────────────────────────
for yaml_file in "$EXPERIMENT_DIR"/*.yaml; do
  # if no matches, break
  [ -e "$yaml_file" ] || { echo "ℹ️  No .yaml files in $EXPERIMENT_DIR"; break; }

  echo
  echo "🚀 Running experiment: $yaml_file"
  sudo docker run --gpus all \
    --shm-size="$SHM_SIZE" \
    -v "$HOST_MOUNT" \
    -w "$CONTAINER_WORKDIR" \
    "$IMAGE" \
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
      train.py \
        --hf_token "$HF_TOKEN" \
        --wandb_token "$WANDB_TOKEN" \
        --yaml_path "$yaml_file"
done
