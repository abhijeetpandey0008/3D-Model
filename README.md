# 3D-Model


This repository implements a CLI tool (generate.py) that produces a 3D mesh (.obj) from either an input image—by performing background removal with OpenCV, depth estimation via MiDaS/PyTorch, and surface reconstruction with Trimesh—or a text prompt (currently stubbed with a placeholder sphere via CLIP), then visualizes it interactively or saves an offscreen snapshot with Pyrender.


generate.py
Purpose: CLI entry point.

Flow:

Parse arguments: 
either --image or --prompt, plus optional snapshot parameters.

Image path:

Load & preprocess image (preprocess.load_and_preprocess_image)

Remove background (preprocess.remove_background)

Estimate depth (preprocess.estimate_depth)

Convert to mesh (image_to_model.image_to_mesh)

Prompt path:

Convert text prompt to mesh stub (text_to_model.prompt_to_mesh)

Export mesh to OBJ, render in a window (visual.render_mesh), and optionally save a PNG snapshot (visual.snapshot_mesh).

preprocess.py

Image I/O & normalization: uses OpenCV (cv2) to read, convert BGR→RGB, resize, and normalize 
PyPI
.

Background removal: 
GrabCut algorithm to create a binary foreground mask.

Depth estimation:
Loads a MiDaS model via PyTorch Hub, transforms input, runs inference, and resizes output to original image size 
PyPI.

image_to_model.py

Point cloud construction: Back-projects 2D pixels and depth into 3D points, filters by the mask.

Surface reconstruction: Uses Trimesh’s Marching Cubes or Ball-Pivoting to form a watertight mesh, then smooths with Laplacian filtering 
PyPI
.



text_to_model.py

Placeholder:
Tokenizes and encodes the prompt with CLIP models (Hugging Face Transformers) and returns a unit sphere (trimesh.primitives.Sphere) until a bona fide text-to-3D generator is integrated.

visual.py
Interactive render: Uses Pyrender to open a live viewer.

Offscreen snapshot: Renders to an image buffer and saves via ImageIO.

utils.py
Directory helper: Ensures target directories exist before writing files.
