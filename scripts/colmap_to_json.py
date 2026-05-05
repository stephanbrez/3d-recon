"""
Converts COLMAP sparse reconstruction to Nerfstudio transforms.json format.

Usage
-----
python colmap_to_json.py <data_directory>

Parameters
----------
data_directory : str
    Path to processed data directory containing colmap/sparse/0

Outputs
-------
transforms.json
    Camera poses and image metadata in Nerfstudio format

Examples
--------
python colmap_to_json.py data/lounge-matched-0.8/processed
python -m json.tool data/lounge-matched-0.8/processed/transforms.json | head -n 40
"""
from pathlib import Path
import sys
from nerfstudio.process_data.colmap_utils import colmap_to_json

data_path = sys.argv[1]
data = Path(data_path)
if not data.exists():
    raise ValueError("Must specify a valid path")

recon_dir = data / "colmap" / "sparse" / "0"

num = colmap_to_json(
    recon_dir=recon_dir,
    output_dir=data,
    keep_original_world_coordinate=False,  # Nerfstudio-style coordinates
    use_single_camera_mode=True,           # matches your COLMAP single-camera setup
)

print(f"Wrote {data / 'transforms.json'}")
print(f"Registered images: {num}")
