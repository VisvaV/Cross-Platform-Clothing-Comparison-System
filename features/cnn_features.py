"""CNN feature extractor — re-exports from models.cnn_encoder for backward compatibility."""

from models.cnn_encoder import CNNEncoder, CNNFeatureExtractor  # noqa: F401

__all__ = ["CNNEncoder", "CNNFeatureExtractor"]
