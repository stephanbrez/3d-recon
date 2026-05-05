"""
Converts a point cloud to a surface mesh using Poisson surface reconstruction.

Usage
-----
python export_nerf_poisson.py <point_cloud.ply>

Parameters
----------
point_cloud.ply : str
    Path to input PLY point cloud file

Outputs
-------
exports/mesh/mesh_raw.ply
    Raw Poisson mesh
exports/mesh/mesh_trimmed.ply
    Mesh with low-density fringe removed

Notes
-----
Prerequisites:
    Export point cloud from Nerfstudio first:
    ns-export pointcloud --load-config path/to/config.yml --output-dir exports/pcd/ \\
        --num-points 1000000 --remove-outliers True --normal-method open3d \\
        --save-world-frame False
"""
import open3d as o3d
import numpy as np
import sys
import os

filename = sys.argv[1]
if filename.endswith(".ply") and os.path.exists(filename):
    pcd = o3d.io.read_point_cloud(filename)
else:
    raise ValueError("Unsupported file format. Only .ply files are supported.")

print("Has normals:", pcd.has_normals())
# Estimate normals
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=50
    )
)

# Make normals consistent
pcd.orient_normals_consistent_tangent_plane(50)

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd,
    depth=10,          # tune as needed
    scale=1.1,
    linear_fit=False,
    n_threads=1,      # critical workaround
)

densities = np.asarray(densities)
print("Mesh vertices:", len(mesh.vertices))
print("Mesh triangles:", len(mesh.triangles))
print("Density min/max:", densities.min(), densities.max())

# Write raw mesh
o3d.io.write_triangle_mesh("exports/mesh/mesh_raw.ply", mesh)

# Optional: remove low-density fringe
threshold = np.quantile(densities, 0.05)
vertices_to_remove = densities < threshold
mesh.remove_vertices_by_mask(vertices_to_remove)

mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh("exports/mesh/mesh_trimmed.ply", mesh)
print("Wrote exports/mesh/mesh_trimmed.ply")
