#!/usr/bin/env bash
set -euo pipefail

DATA="../data/lounge-matched-0.8/processed"
DENSE="$DATA/colmap/dense"

start_vram_log() {
	local log_file="$1"
	nvidia-smi \
		--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,utilization.memory \
		--format=csv,noheader,nounits \
		-l 1 >"$log_file" &
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
	awk -F', *' 'BEGIN{max=0} {if ($4+0 > max) max=$4+0} END{print max}' "$log_file"
}

start_vram_log "$DATA/vram_splatfacto.csv"
uv run ns-train splatfacto \
  --data "$DATA" \
  --vis tensorboard \
	2>&1 | tee "$DATA"/splatfacto_train.log
stop_vram_log

echo
echo "VRAM logs:"
echo "  $DATA/vram_splatfacto.csv"
echo
echo "Peak VRAM usage (MB):"
echo "  splatfacto:  $(peak_vram_mb "$DATA/vram_splatfacto.csv")"
