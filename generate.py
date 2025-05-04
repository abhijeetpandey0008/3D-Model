import argparse
from preprocess import load_and_preprocess_image, remove_background, estimate_depth
from image_to_model import image_to_mesh
from text_to_model import prompt_to_mesh
from visual import render_mesh, snapshot_mesh
from utils import ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Generate a 3D model from an image or text prompt.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to input image file.")
    group.add_argument("--prompt", help="Text prompt to generate 3D model.")
    parser.add_argument("--output", default="output.obj", help="Filename for output mesh (OBJ).")
    parser.add_argument(
        "--snapshot", action="store_true",
        help="Also save an offscreen PNG snapshot of the model."
    )
    parser.add_argument(
        "--snapshot-width", type=int, default=1024,
        help="Width of the offscreen snapshot (default: 1024)."
    )
    parser.add_argument(
        "--snapshot-height", type=int, default=768,
        help="Height of the offscreen snapshot (default: 768)."
    )
    parser.add_argument(
        "--snapshot-output", default="model_view.png",
        help="Filename for the PNG snapshot (default: model_view.png)."
    )
    args = parser.parse_args()

    try:
        # Generate mesh from image or text
        if args.image:
            img = load_and_preprocess_image(args.image)
            mask = remove_background(img)
            depth = estimate_depth(img)
            mesh = image_to_mesh(img, mask, depth)
        else:
            mesh = prompt_to_mesh(args.prompt)

        # Ensure the output directory exists and export the mesh
        ensure_dir(args.output)
        mesh.export(args.output)
        print(f"Saved 3D model to {args.output}")

        # Launch interactive viewer
        render_mesh(mesh)

        # Optionally save an offscreen snapshot
        if args.snapshot:
            ensure_dir(args.snapshot_output)
            snapshot_mesh(
                mesh,
                width=args.snapshot_width,
                height=args.snapshot_height,
                out_path=args.snapshot_output
            )
            print(f"Saved offscreen render to {args.snapshot_output}")

    except Exception as e:
        print(f"Error during generation: {e}")
        exit(1)

if __name__ == "__main__":
    main()
