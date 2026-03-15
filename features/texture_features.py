"""LBP texture feature extractor."""

import logging

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

logger = logging.getLogger(__name__)

_LBP_POINTS = 24
_LBP_RADIUS = 3
_LBP_METHOD = "uniform"
_HIST_BINS = 256


def extract_texture_features(image: Image.Image) -> np.ndarray:
    """Extract an L2-normalized 256-dim LBP texture histogram from a PIL image.

    Args:
        image: PIL Image.

    Returns:
        np.ndarray of shape (256,), L2-normalized. Returns zero vector on error.
    """
    try:
        gray = np.array(image.convert("L"), dtype=np.uint8)

        lbp = local_binary_pattern(gray, P=_LBP_POINTS, R=_LBP_RADIUS, method=_LBP_METHOD)

        hist, _ = np.histogram(lbp.ravel(), bins=_HIST_BINS, range=(0, _HIST_BINS))
        vec = hist.astype(np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    except Exception as exc:
        logger.error("extract_texture_features failed: %s", exc)
        return np.zeros(_HIST_BINS, dtype=np.float32)
