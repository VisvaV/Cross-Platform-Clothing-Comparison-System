"""High-level similarity search interface.

Supports image-only, text-only, and hybrid (image + text) search modes.
Requires the offline pipeline to have been run first to produce:
  - data/faiss_index.bin
  - data/faiss_id_map.pkl
  - data/tfidf_vectorizer.pkl
  - data/text_embeddings.npy
  - data/products.db
"""

import logging
import sqlite3
from typing import Optional

import numpy as np

from features.cnn_features import CNNFeatureExtractor
from features.feature_fusion import extract_fused_embedding
from models.text_encoder import TextEncoder
from utils.db_utils import init_db, get_product_by_id
from vector_search.faiss_index import FaissIndex

logger = logging.getLogger(__name__)

# Default artifact paths (relative to project root)
DEFAULT_DB_PATH = "data/products.db"
DEFAULT_FAISS_INDEX = "data/faiss_index.bin"
DEFAULT_FAISS_MAP = "data/faiss_id_map.pkl"
DEFAULT_TFIDF_PATH = "data/tfidf_vectorizer.pkl"
DEFAULT_TEXT_EMBEDDINGS = "data/text_embeddings.npy"


class SimilaritySearch:
    """Loads pre-built artifacts and serves image, text, and hybrid search queries."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        faiss_index_path: str = DEFAULT_FAISS_INDEX,
        faiss_map_path: str = DEFAULT_FAISS_MAP,
        tfidf_path: str = DEFAULT_TFIDF_PATH,
        text_embeddings_path: str = DEFAULT_TEXT_EMBEDDINGS,
    ):
        # FAISS index
        self._faiss = FaissIndex()
        self._faiss.load(faiss_index_path, faiss_map_path)

        # TF-IDF encoder
        self._text_encoder = TextEncoder()
        self._text_encoder.load(tfidf_path)

        # Pre-computed text embedding matrix — shape (N, vocab_size)
        self._text_embeddings: np.ndarray = np.load(text_embeddings_path)

        # SQLite connection
        self._conn: sqlite3.Connection = init_db(db_path)

        # CNN extractor (lazy — only instantiated when image search is used)
        self._cnn_extractor: Optional[CNNFeatureExtractor] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cnn_extractor(self) -> CNNFeatureExtractor:
        if self._cnn_extractor is None:
            self._cnn_extractor = CNNFeatureExtractor()
        return self._cnn_extractor

    def _lookup_products(self, product_ids: list[int]) -> list[dict]:
        """Fetch product metadata from SQLite for a list of product IDs."""
        results = []
        for pid in product_ids:
            record = get_product_by_id(self._conn, pid)
            if record:
                results.append(record)
        return results

    # ------------------------------------------------------------------
    # Public search methods
    # ------------------------------------------------------------------

    def search_by_image(self, image_path: str, k: int = 20) -> list[dict]:
        """Search by image using fused CNN+color+texture embedding via FAISS.

        Args:
            image_path: Path to the query image file.
            k: Number of results to return.

        Returns:
            List of product dicts, each with an added 'similarity_score' key,
            sorted by similarity descending.
        """
        extractor = self._get_cnn_extractor()
        query_vec = extract_fused_embedding(image_path, extractor)

        faiss_results = self._faiss.search(query_vec, k)  # [(product_id, score), ...]

        results = []
        for product_id, score in faiss_results:
            record = get_product_by_id(self._conn, product_id)
            if record:
                record["similarity_score"] = float(score)
                results.append(record)

        results.sort(key=lambda r: r["similarity_score"], reverse=True)
        return results

    def search_by_text(self, query: str, k: int = 20) -> list[dict]:
        """Search by text description using TF-IDF cosine similarity.

        Args:
            query: Free-text clothing description.
            k: Number of results to return.

        Returns:
            List of product dicts with 'similarity_score', sorted descending.
        """
        query_vec = self._text_encoder.transform(query).astype(np.float32)

        # Cosine similarity: both query and stored embeddings may not be unit-norm,
        # so compute dot product after normalizing.
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            logger.warning("Query text produced a zero TF-IDF vector.")
            return []

        query_unit = query_vec / query_norm

        # Normalize stored text embeddings row-wise
        norms = np.linalg.norm(self._text_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        unit_embeddings = self._text_embeddings / norms

        scores = unit_embeddings.dot(query_unit)  # shape (N,)

        # Top-K indices (unsorted first, then sort)
        k_actual = min(k, len(scores))
        top_indices = np.argpartition(scores, -k_actual)[-k_actual:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        # id_map aligns with text_embeddings rows — use FAISS id_map for product IDs
        results = []
        for idx in top_indices:
            product_id = self._faiss.id_map[idx]
            record = get_product_by_id(self._conn, product_id)
            if record:
                record["similarity_score"] = float(scores[idx])
                results.append(record)

        return results

    def search_hybrid(
        self, image_path: str, query: str, k: int = 20
    ) -> list[dict]:
        """Search using both image and text, combined via weighted hybrid score.

        Hybrid score = 0.65 * image_sim + 0.25 * text_sim + 0.10 * (1 / (norm_price + 1e-6))

        Args:
            image_path: Path to the query image file.
            query: Free-text clothing description.
            k: Number of results to return.

        Returns:
            List of product dicts with 'similarity_score' and 'hybrid_score',
            sorted by hybrid_score descending.
        """
        # Retrieve a larger candidate pool to merge results from both modalities
        pool = k * 3

        image_results = self.search_by_image(image_path, k=pool)
        text_results = self.search_by_text(query, k=pool)

        # Build score maps keyed by product_id
        image_scores: dict[int, float] = {
            r["id"]: r["similarity_score"] for r in image_results
        }
        text_scores: dict[int, float] = {
            r["id"]: r["similarity_score"] for r in text_results
        }

        # Union of candidate product IDs
        all_ids = set(image_scores) | set(text_scores)

        # Collect full records (prefer image_results for metadata)
        record_map: dict[int, dict] = {}
        for r in image_results + text_results:
            record_map.setdefault(r["id"], r)

        # Normalize prices across the candidate set
        prices = [record_map[pid].get("price") or 0.0 for pid in all_ids]
        min_p, max_p = min(prices), max(prices)
        price_range = max_p - min_p if max_p != min_p else 1.0

        candidates = []
        for pid in all_ids:
            record = dict(record_map[pid])
            img_sim = image_scores.get(pid, 0.0)
            txt_sim = text_scores.get(pid, 0.0)
            raw_price = record.get("price") or 0.0
            norm_price = (raw_price - min_p) / price_range

            # price_score: 1.0 for cheapest, 0.0 for most expensive (no blow-up)
            price_score = 1.0 - norm_price

            hybrid = (
                0.65 * img_sim
                + 0.25 * txt_sim
                + 0.10 * price_score
            )

            record["similarity_score"] = img_sim
            record["hybrid_score"] = float(hybrid)
            candidates.append(record)

        candidates.sort(key=lambda r: r["hybrid_score"], reverse=True)
        return candidates[:k]


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Test SimilaritySearch")
    parser.add_argument("--image", help="Path to query image")
    parser.add_argument("--text", help="Text query string")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    searcher = SimilaritySearch()

    if args.image and args.text:
        results = searcher.search_hybrid(args.image, args.text, k=args.k)
        label = "Hybrid"
    elif args.image:
        results = searcher.search_by_image(args.image, k=args.k)
        label = "Image"
    elif args.text:
        results = searcher.search_by_text(args.text, k=args.k)
        label = "Text"
    else:
        parser.error("Provide --image, --text, or both.")

    print(f"\n{label} search — top {args.k} results:")
    for i, r in enumerate(results, 1):
        score = r.get("hybrid_score") or r.get("similarity_score", 0)
        print(f"  {i}. [{r['platform']}] {r['product_title']} — ₹{r['price']} (score: {score:.4f})")
