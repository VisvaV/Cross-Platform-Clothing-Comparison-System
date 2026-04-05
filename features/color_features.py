"""HSV color feature utilities (histogram + dominant color labels).

Uses PIL + NumPy only (no OpenCV runtime dependency), so this module remains
portable in minimal Linux environments where `libGL.so.1` may be unavailable.
"""

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_COLOR_KEYWORDS = {
    "black": {"black"},
    "white": {"white", "off-white", "ivory", "cream"},
    "gray": {"gray", "grey", "charcoal", "silver"},
    "red": {"red", "maroon", "burgundy", "crimson"},
    "orange": {"orange", "rust", "coral", "peach"},
    "yellow": {"yellow", "mustard", "gold"},
    "green": {"green", "olive", "mint", "emerald", "teal"},
    "blue": {"blue", "navy", "indigo", "cyan", "turquoise"},
    "purple": {"purple", "violet", "lavender", "lilac"},
    "pink": {"pink", "fuchsia", "magenta", "rose"},
    "brown": {"brown", "beige", "tan", "khaki", "camel"},
}


def _dominant_color_from_hsv_array(hsv_array: np.ndarray) -> tuple[str, float]:
    """Estimate dominant coarse color from HSV image.

    Returns:
        (color_name, confidence) where confidence is ratio of pixels assigned
        to the dominant coarse color in [0, 1].
    """
    h = hsv_array[..., 0].astype(np.float32)  # [0, 179]
    s = hsv_array[..., 1].astype(np.float32)  # [0, 255]
    v = hsv_array[..., 2].astype(np.float32)  # [0, 255]

    labels = np.full(h.shape, "unknown", dtype=object)

    # Low saturation / value bucket first (grayscale shades)
    low_sat = s < 30
    labels[np.logical_and(low_sat, v < 50)] = "black"
    labels[np.logical_and(low_sat, v > 200)] = "white"
    labels[np.logical_and(low_sat, (v >= 50) & (v <= 200))] = "gray"

    # Chromatic buckets by hue
    chromatic = ~low_sat
    labels[np.logical_and(chromatic, np.logical_or(h < 8, h >= 170))] = "red"
    labels[np.logical_and(chromatic, (h >= 8) & (h < 20))] = "orange"
    labels[np.logical_and(chromatic, (h >= 20) & (h < 35))] = "yellow"
    labels[np.logical_and(chromatic, (h >= 35) & (h < 85))] = "green"
    labels[np.logical_and(chromatic, (h >= 85) & (h < 130))] = "blue"
    labels[np.logical_and(chromatic, (h >= 130) & (h < 150))] = "purple"
    labels[np.logical_and(chromatic, (h >= 150) & (h < 170))] = "pink"

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) == 0:
        return "unknown", 0.0

    idx = int(np.argmax(counts))
    dominant = str(unique_labels[idx])
    confidence = float(counts[idx] / counts.sum()) if counts.sum() else 0.0
    return dominant, confidence


def extract_dominant_color(image: Image.Image) -> tuple[str, float]:
    """Extract dominant coarse color label and confidence from a PIL image."""
    try:
        # PIL HSV ranges: H,S,V in [0, 255]. Convert H to OpenCV-style [0, 179]
        hsv_pil = np.array(image.convert("HSV"), dtype=np.uint8)
        hsv_array = hsv_pil.astype(np.float32)
        hsv_array[..., 0] = hsv_array[..., 0] * (179.0 / 255.0)
        return _dominant_color_from_hsv_array(hsv_array)
    except Exception as exc:
        logger.error("extract_dominant_color failed: %s", exc)
        return "unknown", 0.0


def dominant_color_from_text(text: str) -> str:
    """Infer coarse color label from product text/title using keyword map."""
    lowered = (text or "").lower()
    for canonical, variants in _COLOR_KEYWORDS.items():
        if any(token in lowered for token in variants):
            return canonical
    return "unknown"


def extract_color_features(image: Image.Image) -> np.ndarray:
    """Extract an L2-normalized 512-dim HSV color histogram from a PIL image.

    Args:
        image: PIL Image (RGB).

    Returns:
        np.ndarray of shape (512,), L2-normalized. Returns zero vector on error.
    """
    try:
        # PIL HSV ranges all channels to [0, 255]
        hsv_array = np.array(image.convert("HSV"), dtype=np.float32)
        h = hsv_array[..., 0] * (180.0 / 256.0)  # map H to [0, 180)
        s = hsv_array[..., 1]
        v = hsv_array[..., 2]

        # 3D histogram: 8 bins per channel → 8×8×8 = 512 dims
        hist, _ = np.histogramdd(
            sample=np.column_stack((h.ravel(), s.ravel(), v.ravel())),
            bins=(8, 8, 8),
            range=((0, 180), (0, 256), (0, 256)),
        )
        vec = hist.flatten().astype(np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    except Exception as exc:
        logger.error("extract_color_features failed: %s", exc)
        return np.zeros(512, dtype=np.float32)
