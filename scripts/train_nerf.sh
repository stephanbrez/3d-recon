#!/usr/bin/env bash
# train_nerf.sh — Train a nerfacto model via nerfstudio.
#
# Usage:
#   ./train_nerf.sh <load-dir>                  # Run with defaults
#   ./train_nerf.sh <load-dir> <ns-train args>  # Pass extra args to ns-train
#
# VRAM monitoring uses monitor_gpu_mem.py (PyTorch torch.cuda) instead of
# nvidia-smi because nvidia-smi reports [N/A] on unified-memory systems
# like the NVIDIA GB10 where GPU and system RAM are shared.
set -euo pipefail
DATA="../data/lounge-matched-0.8/processed"
DENSE="$DATA/colmap/dense"

TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")

start_vram_log() {
        local log_file="$1"
        local interval="${2:-10}"
        uv run python scripts/monitor_gpu_mem.py "$log_file" "$interval" &
        SMI_PID=$!
}
stop_vram_log() {
        if [[ -n "${SMI_PID:-}" ]]; then
                kill "$SMI_PID" 2>/dev/null || true
                wait "$SMI_PID" 2>/dev/null || true
                unset SMI_PID
        fi
}
cleanup() {
        stop_vram_log
}
trap cleanup EXIT
peak_vram_mb() {
        local log_file="$1"
        awk -F',' 'NR>1 {v=$5+0; if(v>max) max=v} END{print max+0}' "$log_file"
}
LOAD_DIR="${1:?Usage: $0 <load-dir> [ns-train args...]}"
shift

start_vram_log "$DATA/vram_nerfacto_${TIMESTAMP}.csv"
if [[ $# -gt 0 ]]; then
    uv run ns-train nerfacto --load-dir "$LOAD_DIR" "$@" \
        2>&1 | tee "$DATA/nerfacto_train_${TIMESTAMP}.log"
else
    uv run ns-train nerfacto \
      --data "$DATA" \
      --load-dir "$LOAD_DIR" \
      --max-num-iterations 30000 \
      --steps-per-save 5000 \
      --save-only-latest-checkpoint False \
      --vis tensorboard \
        2>&1 | tee "$DATA/nerfacto_train_${TIMESTAMP}.log"
fi
stop_vram_log
echo
echo "VRAM logs:"
echo "  $DATA/vram_nerfacto_${TIMESTAMP}.csv"
echo
echo "Peak VRAM usage (MB):"
echo "  nerfacto:  $(peak_vram_mb "$DATA/vram_nerfacto_${TIMESTAMP}.csv")"
