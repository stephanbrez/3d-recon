"""Reads and visualizes COLMAP binary depth maps. For quick verification.

This script loads a COLMAP depth map in binary format, prints statistics about
the data (shape, min/max values, sparsity), and generates a PNG visualization
using the turbo colormap. Useful for inspecting stereo_fusion output.

Usage
-----
python inspect_depth_map.py <depth_map.bin>

Parameters
----------
depth_map.bin : str
    Path to COLMAP binary depth map file

Outputs
-------
depth_preview.png
    PNG visualization of the depth map using turbo colormap with colorbar.
    Saved at 150 dpi.

Processing Steps
----------------
1. Parse COLMAP binary header to read width, height, and channels
2. Load float32 depth values from file
3. Report sparsity (percentage of nonzero values)
4. Display shape, min/max depth range, and sparsity statistics
5. Generate and save visualization

Notes
-----
Binary format structure:
    - Header: width&height&channels& (ASCII, separated by '&')
    - Data: float32 array of dimensions (height, width) or (height, width, channels)
    - Missing/invalid depths are stored as 0
    - Typical sparsity indicates coverage of the stereo reconstruction

Typical output:
    - Shape, min/max depth in camera units (usually meters)
    - Percentage of pixels with valid (nonzero) depth
    - Generated depth_preview.png for visual inspection
"""
import numpy as np
import sys
import matplotlib.pyplot as plt

def read_colmap_bin(path):
    with open(path, "rb") as f:
        header = b""
        amp_count = 0
        while amp_count < 3:
            ch = f.read(1)
            header += ch
            if ch == b"&":
                amp_count += 1
        parts = header.decode("ascii").rstrip("&").split("&")
        w, h, c = int(parts[0]), int(parts[1]), int(parts[2])
        data_start = f.tell()
        print(f"w={w}, h={h}, c={c}, data_offset={data_start}")
        data = np.fromfile(f, dtype=np.float32, count=w * h * c)
        nz = np.count_nonzero(data)
        print(f"Read {data.size} floats, nonzero: {nz}/{w*h*c} ({100*nz/(w*h*c):.2f}%)")
        return data[:w*h*c].reshape((h, w, c) if c > 1 else (h, w))

if __name__ == "__main__":
    dm = read_colmap_bin(sys.argv[1])
    print(f"Shape: {dm.shape}, min: {dm.min():.4f}, max: {dm.max():.4f}, "
          f"nonzero: {np.count_nonzero(dm)}/{dm.size} ({100*np.count_nonzero(dm)/dm.size:.1f}%)")
    plt.imshow(dm, cmap="turbo")
    plt.colorbar(label="depth")
    plt.title(sys.argv[1].split("/")[-1])
    plt.savefig("depth_preview.png", dpi=150)
    print("Saved depth_preview.png")
