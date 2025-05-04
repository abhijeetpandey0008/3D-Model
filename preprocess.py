import cv2
import numpy as np
import torch

def load_and_preprocess_image(path, size=(256, 256)):
    """
    Load an image, convert to RGB, resize, and normalize to [0,1].
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

def remove_background(img):
    """
    Generate a foreground mask using OpenCV's GrabCut.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (10, 10, w - 20, h - 20)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut((img * 255).astype('uint8'), mask, rect,
                bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    # Keep only sure foreground and probable foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask2

def estimate_depth(img, model_type="DPT_Large"):
    """
    Estimate a depth map from an RGB image using MiDaS via PyTorch Hub.

    Args:
        img (np.ndarray): RGB image array, shape (H,W,3), float32 in [0,1].
        model_type (str): One of "DPT_Large", "DPT_Hybrid", or "MiDaS_small".

    Returns:
        depth_map (np.ndarray): 2D depth map, same HxW as input.
    """
    # Load MiDaS model from PyTorch Hub
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas = torch.hub.load("intel-isl/MiDaS", model_type)      
    midas.to(device).eval()

    # Load transforms for the chosen model type
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = midas_transforms.dpt_transform               
    else:
        transform = midas_transforms.small_transform

    # Prepare input batch
    input_batch = transform(img).to(device)

    # Inference and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

    depth_map = prediction.cpu().numpy()
    return depth_map

