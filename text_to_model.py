import torch
from transformers import CLIPTokenizer, CLIPTextModel
import trimesh

# Optional imports for text-to-3D backends
try:
    from clip_forge import CLIPForgeGenerator
except ImportError:
    CLIPForgeGenerator = None

try:
    from stable_dreamfusion import DreamFusionGenerator
except ImportError:
    DreamFusionGenerator = None

def prompt_to_mesh(
    prompt: str,
    out_size: int = 256,
    steps: int = 50,
    backend: str = "clip_forge"
) 
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    text_model.to(device).eval()


    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = text_model(**inputs).last_hidden_state

    mesh = None


    if backend == "clip_forge" and CLIPForgeGenerator is not None:
        # Initialize generator with desired settings
        gen = CLIPForgeGenerator(device=device, out_size=out_size, steps=steps)
        mesh = gen.generate(text_features)

    # Backend
    elif backend in ("dreamfusion", "stable_dreamfusion") and DreamFusionGenerator is not None:
        gen = DreamFusionGenerator(device=device, out_size=out_size, steps=steps)
        mesh = gen.generate(prompt)
    
    if mesh is None:
        mesh = trimesh.primitives.Sphere(radius=1.0)

    return mesh


