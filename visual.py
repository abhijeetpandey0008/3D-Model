import numpy as np
setattr(np, "infty", np.inf)


import pyrender
import trimesh
import imageio

def render_mesh(mesh):
    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

# code to  visualize the mode in image formate not in object formate
def snapshot_mesh(mesh, width=800, height=600, out_path="snapshot.png"):
    """
    Render `mesh` offscreen at the given resolution and save to `out_path`.
    """
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    r = pyrender.OffscreenRenderer(viewport_width=width,
                                   viewport_height=height)  
    color, _ = r.render(scene)
    r.delete()
    imageio.imsave(out_path, color)
    print(f"Snapshot saved to {out_path}")

