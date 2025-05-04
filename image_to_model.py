import numpy as np
import trimesh

def image_to_mesh(img, mask, depth):
    # Back-project depth to point cloud
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    points = np.stack([xs.flatten(), ys.flatten(), depth.flatten()], axis=-1)
    # Filter by mask
    mask_flat = mask.flatten().astype(bool)
    points = points[mask_flat]

    cloud = trimesh.points.PointCloud(points)
    # Surface reconstruction via ball pivoting or alpha shape
    mesh = trimesh.points.marching_cubes(points, pitch=1.0)
    mesh = trimesh.remesh.filter_smooth_laplacian(mesh)
    return 
