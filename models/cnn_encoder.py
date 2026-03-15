"""CNN encoder module — ResNet50 backbone for feature extraction and fine-tuning."""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights

from utils.image_utils import preprocess_for_cnn

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS_PATH = os.path.join("models", "resnet50_finetuned.pth")


class CNNEncoder(nn.Module):
    """ResNet50 backbone with the FC layer replaced by nn.Identity().

    Produces 2048-dim feature vectors. Used as the shared encoder inside
    TripletNet during training, and directly for inference via CNNFeatureExtractor.

    Args:
        pretrained: If True, initialise with ImageNet weights (default: True).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        base.fc = nn.Identity()
        self.backbone = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Image tensor of shape (N, 3, 224, 224).

        Returns:
            Embedding tensor of shape (N, 2048).
        """
        return self.backbone(x)


class CNNFeatureExtractor:
    """Inference wrapper around CNNEncoder.

    Loads optional fine-tuned weights, runs in eval mode, and returns
    L2-normalised 2048-dim numpy vectors.

    Args:
        weights_path: Path to fine-tuned state-dict (.pth). Falls back to
                      ImageNet weights when the file is absent.
    """

    def __init__(self, weights_path: str = _DEFAULT_WEIGHTS_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CNNEncoder(pretrained=True).to(self.device)

        if weights_path and os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.backbone.load_state_dict(state)
            logger.info("Loaded fine-tuned weights from %s", weights_path)
        else:
            logger.info(
                "Fine-tuned weights not found at %s; using ImageNet weights.",
                weights_path,
            )

        self.model.eval()

    def extract(self, image: Image.Image) -> np.ndarray:
        """Extract an L2-normalised 2048-dim embedding from a PIL image.

        Args:
            image: PIL Image (RGB).

        Returns:
            np.ndarray of shape (2048,), L2-normalised.
        """
        tensor = preprocess_for_cnn(image).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)          # (1, 2048)
        vec = embedding.squeeze(0).cpu().numpy()    # (2048,)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
