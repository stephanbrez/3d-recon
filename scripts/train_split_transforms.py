"""
Splits transforms.json into train/val/test sets with interval-based stratification.

Loads camera metadata from a Nerfstudio transforms.json file, splits frames into
train/val/test sets using interval-based sampling (every nth frame), and writes
the split back to the JSON file with train_filenames, val_filenames, and
test_filenames keys.

Usage
-----
python train_split_transforms.py <transforms.json>

Parameters
----------
transforms.json : str
    Path to Nerfstudio transforms.json file with 'frames' key

Outputs
-------
Modified transforms.json
    Updated with train_filenames, val_filenames, test_filenames lists
transforms.json.bak
    Backup of original file before modification

Configuration
--------------
EVAL_INTERVAL : int
    Every nth frame goes to validation. Default 10 (~10% for ~80 images).
    Adjust based on dataset size.

Notes
-----
- Frames are sorted by file_path before splitting (matches Nerfstudio behavior)
- Test split is identical to val for convenience
- Interval-based splitting ensures temporal/spatial coverage in both sets
"""
from pathlib import Path
import json
import shutil
import sys

EVAL_INTERVAL = 10  # every 10th frame goes to val (~10% for 78 images)

TRANSFORMS = Path(sys.argv[1])
if not TRANSFORMS.exists():
    raise ValueError(f"transforms.json not found: {TRANSFORMS}")

# Load
with TRANSFORMS.open("r", encoding="utf-8") as f:
    meta = json.load(f)

frames = meta["frames"]
if not frames:
    raise RuntimeError("No frames found in transforms.json")

# Nerfstudio sorts by filename before splitting, so mirror that behavior.
frames_sorted = sorted(frames, key=lambda fr: str(Path(fr["file_path"])))

all_files = [fr["file_path"] for fr in frames_sorted]

# Interval split: every nth image -> val, rest -> train
val_files = [fp for i, fp in enumerate(all_files) if i % EVAL_INTERVAL == 0]
train_files = [fp for i, fp in enumerate(all_files) if i % EVAL_INTERVAL != 0]

if len(val_files) == 0:
    raise RuntimeError("Validation split is empty; lower EVAL_INTERVAL.")

# Write explicit split lists.
meta["train_filenames"] = train_files
meta["val_filenames"] = val_files

# Optional: make test identical to val for convenience.
# Remove this line if you do not want a test split.
meta["test_filenames"] = val_files

# Backup first
backup = TRANSFORMS.with_suffix(".json.bak")
shutil.copy2(TRANSFORMS, backup)

with TRANSFORMS.open("w", encoding="utf-8") as f:
    json.dump(meta, f, indent=4)

print(f"Wrote split into: {TRANSFORMS}")
print(f"Backup saved to:  {backup}")
print(f"Total frames:     {len(all_files)}")
print(f"Train frames:     {len(train_files)}")
print(f"Val frames:       {len(val_files)}")
print("\nFirst few val files:")
for fp in val_files[:10]:
    print("  ", fp)
