"""Feature fusion module — combines CNN, color, and texture embeddings."""

import logging

import numpy as np
from PIL import Image

from features.cnn_features import CNNFeatureExtractor
from features.color_features import extract_color_features
from features.texture_features import extract_texture_features
from utils.image_utils import load_image

logger = logging.getLogger(__name__)


def fuse_features(
    cnn: np.ndarray,
    color: np.ndarray,
    texture: np.ndarray,
) -> np.ndarray:
    """Concatenate CNN (2048), color (512), and texture (256) vectors and L2-normalize.

    Args:
        cnn: L2-normalized CNN embedding, shape (2048,).
        color: L2-normalized color histogram, shape (512,).
        texture: L2-normalized texture histogram, shape (256,).

    Returns:
        np.ndarray of shape (2816,), L2-normalized.
    """
    fused = np.concatenate([cnn, color, texture])  # (2816,)
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused


def extract_fused_embedding(
    image_path: str,
    extractor: CNNFeatureExtractor,
) -> np.ndarray:
    """Load an image and return a normalized 2816-dim fused embedding.

    Args:
        image_path: Path to the image file.
        extractor: A pre-initialized CNNFeatureExtractor instance.

    Returns:
        np.ndarray of shape (2816,), L2-normalized.
    """
    image: Image.Image = load_image(image_path)
    cnn_vec = extractor.extract(image)
    color_vec = extract_color_features(image)
    texture_vec = extract_texture_features(image)
    return fuse_features(cnn_vec, color_vec, texture_vec)
