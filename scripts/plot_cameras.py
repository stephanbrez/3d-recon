"""
Visualizes COLMAP camera poses in 3D space.

Reads camera positions and orientations from a COLMAP binary images file,
converts quaternion + translation to world coordinates, and generates a 3D
scatter plot of camera positions labeled with image names.

Usage
-----
python plot_cameras.py <images.bin>

Parameters
----------
images.bin : str
    Path to COLMAP binary images file (from sparse reconstruction)

Outputs
-------
cameras.png
    3D scatter plot of camera positions with image name labels (150 dpi)

Processing
----------
1. Parse COLMAP binary format: quaternion (qw,qx,qy,qz) + translation (tx,ty,tz)
2. Convert quaternion to rotation matrix using scipy.spatial.transform.Rotation
3. Compute camera center C = -R^T @ t (world coordinates)
4. Plot each camera center as red point with last 8 chars of image filename
5. Label axes as X, Y, Z in world coordinate frame

Notes
-----
- Camera center C is in world coordinates, not camera coordinates
- Image names are truncated to last 8 characters for readability
- Useful for visualizing sparse reconstruction quality and coverage
"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_images_bin(path):
    cameras = []
    with open(path, "rb") as f:
        num_images = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_images):
            _image_id = struct.unpack("I", f.read(4))[0]  # unused, must read to advance cursor
            qw, qx, qy, qz = struct.unpack("dddd", f.read(32))
            tx, ty, tz = struct.unpack("ddd", f.read(24))
            _cam_id = struct.unpack("I", f.read(4))[0]  # unused, must read to advance cursor
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            num_pts = struct.unpack("Q", f.read(8))[0]
            f.read(num_pts * 24)  # skip 2D points
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            C = -R.T @ np.array([tx, ty, tz])
            cameras.append((name.decode(), C))
    return cameras

cams = read_images_bin(sys.argv[1])
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection="3d")
assert isinstance(ax, Axes3D)
for name, C in cams:
    ax.scatter(*C, c="r", s=30)
    ax.text(*C, name[-8:], fontsize=4)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
plt.savefig("cameras.png", dpi=200)
print("Saved cameras.png")
