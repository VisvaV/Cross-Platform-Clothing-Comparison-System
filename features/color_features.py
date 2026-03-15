"""HSV color histogram feature extractor."""

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def extract_color_features(image: Image.Image) -> np.ndarray:
    """Extract an L2-normalized 512-dim HSV color histogram from a PIL image.

    Args:
        image: PIL Image (RGB).

    Returns:
        np.ndarray of shape (512,), L2-normalized. Returns zero vector on error.
    """
    try:
        # PIL (RGB) → numpy BGR → HSV
        rgb_array = np.array(image.convert("RGB"))
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        hsv_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)

        # 3D histogram: 8 bins per channel → 8×8×8 = 512 dims
        hist = cv2.calcHist(
            [hsv_array],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256],
        )
        vec = hist.flatten().astype(np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    except Exception as exc:
        logger.error("extract_color_features failed: %s", exc)
        return np.zeros(512, dtype=np.float32)
