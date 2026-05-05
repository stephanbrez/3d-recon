#!/usr/bin/env bash
# Runs COLMAP's Poisson mesher on a fused point cloud and logs VRAM usage.
#
# Usage: pc_to_poisson.sh <data_directory>
#
# Arguments:
#   data_directory: Path to the root data directory containing colmap/dense/fused.ply
#
# Outputs:
#   - Meshed PLY file: <data_directory>/colmap/dense/meshed-poisson.ply
#   - VRAM log: <data_directory>/colmap/dense/vram_poisson.csv
#   - Process log: <data_directory>/colmap/dense/poisson_mesher.log

set -euo pipefail

DATA="${1:?Must provide data directory path as argument}"
if [[ ! -d "$DATA" ]]; then
	echo "Error: DATA path does not exist: $DATA" >&2
	exit 1
fi
DENSE="$DATA/colmap/dense"
if [[ ! -d "$DENSE" ]]; then
	echo "Error: DENSE path does not exist: $DENSE" >&2
	exit 1
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")
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

# 1) image_undistorter
start_vram_log "$DENSE/vram_poisson_${TIMESTAMP}.csv"
colmap poisson_mesher \
	--input_path "$DENSE"/fused.ply \
	--output_path "$DENSE"/meshed-poisson.ply \
	2>&1 | tee "$DENSE"/poisson_mesher.log
stop_vram_log

echo
echo "VRAM logs:"
echo "  $DENSE/vram_poisson_${TIMESTAMP}.csv"
echo
echo "Peak VRAM usage (MB):"
echo "  poisson_mesher:  $(peak_vram_mb "$DENSE/vram_poisson_${TIMESTAMP}.csv")"
