#!/usr/bin/env bash
# train.sh — Train a nerfstudio model (nerfacto, splatfacto, etc.).
#
# Usage:
#   ./train.sh <model> <data>                  # Run with defaults
#   ./train.sh <model> <data> <ns-train args>  # Pass extra args to ns-train
#
# Examples:
#   ./train.sh nerfacto ../data/lounge/processed
#   ./train.sh splatfacto ../data/lounge/processed --max-num-iterations 5000
#   ./train.sh nerfacto ../data/lounge/processed --load-dir path/to/checkpoint
#
# GPU memory is tracked by nerfstudio itself and logged to tensorboard/wandb
# as "GPU Memory (MB)" via torch.cuda.max_memory_allocated().
set -euo pipefail

MODEL="${1:?Usage: $0 <model> <data> [ns-train args...]}"
DATA="${2:?Usage: $0 <model> <data> [ns-train args...]}"
shift 2
TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")

# ─── Extract --output-dir from args if provided ───
LOG_DIR="$DATA"
for i in $(seq 1 $#); do
    arg="${!i}"
    if [[ "$arg" == --output-dir ]]; then
        next=$((i + 1))
        LOG_DIR="${!next}"
        break
    elif [[ "$arg" == --output-dir=* ]]; then
        LOG_DIR="${arg#--output-dir=}"
        break
    fi
done
mkdir -p "$LOG_DIR"

TRAIN_LOG="$LOG_DIR/${MODEL}_train_${TIMESTAMP}.log"

if [[ $# -gt 0 ]]; then
    uv run ns-train "$MODEL" --data "$DATA" "$@" \
        2>&1 | tee "$TRAIN_LOG"
else
    uv run ns-train "$MODEL" \
      --data "$DATA" \
      --steps-per-save 5000 \
      --save-only-latest-checkpoint False \
      --vis tensorboard \
        2>&1 | tee "$TRAIN_LOG"
fi
