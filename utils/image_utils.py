"""Image utility functions for downloading, loading, and preprocessing images."""

import logging
import requests
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# ImageNet normalization constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def download_image(url: str, save_path: str) -> bool:
    """Download an image from a URL and save it to disk.

    Args:
        url: The image URL to download.
        save_path: Local file path to save the image.

    Returns:
        True on success, False on failure.
    """
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.ConnectionError:
        logger.warning("Connection error (domain blocked or down): %s", url)
        return False
    except Exception as e:
        logger.warning("Failed to download image from %s: %s", url, e)
        return False


def load_image(path: str) -> Image.Image:
    """Load an image from disk as a PIL Image in RGB mode.

    Args:
        path: Local file path to the image.

    Returns:
        PIL Image in RGB mode.
    """
    return Image.open(path).convert("RGB")


def preprocess_for_cnn(image: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image for CNN inference.

    Resizes to 224x224, converts to tensor, and applies ImageNet normalization.

    Args:
        image: PIL Image (RGB).

    Returns:
        Preprocessed tensor of shape (1, 3, 224, 224).
    """
    tensor = _preprocess_transform(image)
    return tensor.unsqueeze(0)  # add batch dimension
