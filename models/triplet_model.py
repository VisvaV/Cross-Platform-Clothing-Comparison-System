"""Triplet network for metric learning on fashion images."""

import torch
import torch.nn as nn

from models.cnn_encoder import CNNEncoder


class TripletNet(nn.Module):
    """Shared-weight triplet network wrapping CNNEncoder as the backbone.

    All three branches (anchor, positive, negative) share the same encoder
    weights. The encoder is ResNet50 with the FC layer replaced by
    nn.Identity(), producing 2048-dim embeddings.
    """

    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder(pretrained=True)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run all three branches through the shared encoder.

        Args:
            anchor:   Batch of anchor images,   shape (N, 3, 224, 224).
            positive: Batch of positive images, shape (N, 3, 224, 224).
            negative: Batch of negative images, shape (N, 3, 224, 224).

        Returns:
            Tuple of (anchor_emb, positive_emb, negative_emb), each (N, 2048).
        """
        return (
            self.forward_once(anchor),
            self.forward_once(positive),
            self.forward_once(negative),
        )
