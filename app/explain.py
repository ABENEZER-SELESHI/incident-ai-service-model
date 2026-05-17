# INCIDENT-AI-SERVICE/app/explain.py
"""
Grad-CAM explanation utilities for incident classifier.
"""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_explanation(model, image_path: str, target_class: int = None) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for the given image.

    Args:
        model: The loaded PyTorch model.
        image_path: Path to the input image.
        target_class: Class index to explain. Defaults to predicted class.

    Returns:
        RGB numpy array with the CAM overlay.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    raw = Image.open(image_path).convert("RGB")
    input_tensor = transform(raw).unsqueeze(0)

    # Target the last conv layer of EfficientNet-B0
    target_layers = [model.conv_head]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=None  # uses predicted class when None
        )[0]

    rgb_img = np.array(raw.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization
