#!/usr/bin/env bash
# Runs COLMAP sparse reconstruction on images in Nerfstudio layout.
#
# Usage: run_colmap_nerfstudio.sh <root_directory>
#
# Arguments:
#   root_directory: Path to the root data directory containing:
#     - images/ (source images)
#     - processed/ (output directory, will be created)
#
# Outputs:
#   Sparse reconstruction in <root_directory>/processed/colmap/

set -euo pipefail

ROOT="${1:?Must provide root directory path as argument}"
if [[ ! -d "$ROOT" ]]; then
	echo "Error: ROOT path does not exist: $ROOT" >&2
	exit 1
fi

SRC_IMAGES="$ROOT/images"
if [[ ! -d "$SRC_IMAGES" ]]; then
	echo "Error: SRC_IMAGES path does not exist: $SRC_IMAGES" >&2
	exit 1
fi

DATA="$ROOT/processed"

DB="$DATA/colmap/database.db"
IMAGES="$DATA/images"
SPARSE="$DATA/colmap/sparse"

# Start clean.
rm -rf "$DATA"
mkdir -p "$IMAGES" "$SPARSE"

# Copy source images into the Nerfstudio-style processed layout.
cp -av "$SRC_IMAGES"/. "$IMAGES"/

# Sparse COLMAP reconstruction.
colmap feature_extractor \
  --database_path "$DB" \
  --image_path "$IMAGES" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --FeatureExtraction.use_gpu 1

colmap exhaustive_matcher \
  --database_path "$DB" \
  --FeatureMatching.use_gpu 1

colmap mapper \
  --database_path "$DB" \
  --image_path "$IMAGES" \
  --output_path "$SPARSE"

# Show the result.
echo
echo "Created sparse models:"
find "$SPARSE" -maxdepth 2 -type f | sort
