#!/usr/bin/env bash
# Runs COLMAP dense reconstruction: image_undistorter, patch_match_stereo, and stereo_fusion.
# Logs VRAM usage for each step.
#
# Usage: colmap_to_pc.sh <data_directory>
#
# Arguments:
#   data_directory: Path to the processed data directory containing colmap/sparse/0
#
# Outputs:
#   - Dense reconstruction: <data_directory>/colmap/dense/
#   - Fused point cloud: <data_directory>/colmap/dense/fused.ply
#   - VRAM logs: <data_directory>/colmap/dense/vram_*_log.csv
#   - Process logs: <data_directory>/colmap/dense/*.log

set -euo pipefail

DATA="${1:?Must provide data directory path as argument}"
if [[ ! -d "$DATA" ]]; then
	echo "Error: DATA path does not exist: $DATA" >&2
	exit 1
fi

DENSE="$DATA/colmap/dense"

# Start clean for dense reconstruction.
rm -rf "$DENSE"
mkdir -p "$DENSE"

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
start_vram_log "$DENSE/vram_undistort_${TIMESTAMP}.csv"
colmap image_undistorter \
	--image_path "$DATA/images" \
	--input_path "$DATA/colmap/sparse/0" \
	--output_path "$DENSE" \
	--output_type COLMAP \
	stop_vram_log

# 2) patch_match_stereo
start_vram_log "$DENSE/vram_patchmatch_${TIMESTAMP}.csv"
# Pass 1: photometric, no filter
colmap patch_match_stereo \
	--workspace_path "$DENSE" \
	--workspace_format COLMAP \
	--PatchMatchStereo.geom_consistency 0 \
	--PatchMatchStereo.filter 1 \
	--PatchMatchStereo.max_image_size 2000 \
	2>&1 | tee "$DENSE"/patch_match_stereo.log

# Pass 2: geometric, with filter
colmap patch_match_stereo \
	--workspace_path "$DENSE" \
	--workspace_format COLMAP \
	--PatchMatchStereo.geom_consistency 1 \
	--PatchMatchStereo.filter 1 \
	--PatchMatchStereo.filter_min_num_consistent 1 \
	--PatchMatchStereo.filter_geom_consistency_max_cost 3 \
	--PatchMatchStereo.write_consistency_graph 1 \
	--PatchMatchStereo.max_image_size 2000 \
	2>&1 | tee -a "$DENSE"/patch_match_stereo.log
stop_vram_log

# 3) stereo_fusion
start_vram_log "$DENSE/vram_fusion_${TIMESTAMP}.csv"
colmap stereo_fusion \
	--workspace_path "$DENSE" \
	--workspace_format COLMAP \
	--input_type geometric \
	--output_path "$DENSE/fused.ply" \
	2>&1 | tee "$DENSE"/stereo_fusion.log
stop_vram_log
#  --StereoFusion.min_num_pixels 2 \
#  --StereoFusion.max_reproj_error 4 \
#  --StereoFusion.max_depth_error 0.1 \
#  --StereoFusion.max_normal_error 30 \

echo
echo "VRAM logs:"
echo "  $DENSE/vram_undistort_${TIMESTAMP}.csv"
echo "  $DENSE/vram_patchmatch_${TIMESTAMP}.csv"
echo "  $DENSE/vram_fusion_${TIMESTAMP}.csv"
echo
echo "Peak VRAM usage (MB):"
echo "  undistort:  $(peak_vram_mb "$DENSE/vram_undistort_${TIMESTAMP}.csv")"
echo "  patchmatch: $(peak_vram_mb "$DENSE/vram_patchmatch_${TIMESTAMP}.csv")"
echo "  fusion:     $(peak_vram_mb "$DENSE/vram_fusion_${TIMESTAMP}.csv")"
