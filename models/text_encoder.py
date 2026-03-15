"""TF-IDF based text encoder for product title embeddings."""

import logging
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.text_utils import clean_text

logger = logging.getLogger(__name__)


class TextEncoder:
    """Wraps a TF-IDF vectorizer for fitting and transforming product titles."""

    def __init__(self):
        self._vectorizer = TfidfVectorizer(max_features=10000)
        self._fitted = False

    def fit(self, texts: list) -> None:
        """Train the TF-IDF vectorizer on a list of product titles.

        Args:
            texts: List of raw product title strings.
        """
        cleaned = [clean_text(t) for t in texts]
        self._vectorizer.fit(cleaned)
        self._fitted = True
        logger.info("TextEncoder fitted on %d texts.", len(texts))

    def transform(self, text: str) -> np.ndarray:
        """Transform a single text string into a dense TF-IDF vector.

        Args:
            text: Raw product title or query string.

        Returns:
            Dense numpy array of shape (max_features,).
        """
        if not self._fitted:
            raise RuntimeError("TextEncoder must be fitted before calling transform().")
        cleaned = clean_text(text)
        sparse = self._vectorizer.transform([cleaned])
        return sparse.toarray()[0]

    def save(self, path: str) -> None:
        """Persist the fitted vectorizer to disk.

        Args:
            path: File path to save the vectorizer (e.g. 'data/tfidf_vectorizer.pkl').
        """
        joblib.dump(self._vectorizer, path)
        logger.info("TextEncoder saved to %s.", path)

    def load(self, path: str) -> None:
        """Load a previously fitted vectorizer from disk.

        Args:
            path: File path of the saved vectorizer.
        """
        self._vectorizer = joblib.load(path)
        self._fitted = True
        logger.info("TextEncoder loaded from %s.", path)
