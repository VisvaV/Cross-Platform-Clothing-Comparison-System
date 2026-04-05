"""High-level similarity search interface.

Supports image-only, text-only, and hybrid (image + text) search modes.
Requires the offline pipeline to have been run first to produce:
  - data/faiss_index.bin
  - data/faiss_id_map.pkl
  - data/tfidf_vectorizer.pkl
  - data/text_embeddings.npy        (aligned with text_id_map.pkl)
  - data/text_id_map.pkl            (NEW — product IDs aligned with text_embeddings rows)
  - data/products.db
"""

import logging
import pickle
import sqlite3
from typing import Optional

import numpy as np

from features.cnn_features import CNNFeatureExtractor
from features.color_features import dominant_color_from_text, extract_dominant_color
from features.feature_fusion import extract_fused_embedding
from models.text_encoder import TextEncoder
from utils.db_utils import init_db, get_product_by_id
from utils.image_utils import load_image
from vector_search.faiss_index import FaissIndex

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH          = "data/products.db"
DEFAULT_FAISS_INDEX      = "data/faiss_index.bin"
DEFAULT_FAISS_MAP        = "data/faiss_id_map.pkl"
DEFAULT_TFIDF_PATH       = "data/tfidf_vectorizer.pkl"
DEFAULT_TEXT_EMBEDDINGS  = "data/text_embeddings.npy"
DEFAULT_TEXT_ID_MAP      = "data/text_id_map.pkl"   # NEW


class SimilaritySearch:
    """Loads pre-built artifacts and serves image, text, and hybrid search queries."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        faiss_index_path: str = DEFAULT_FAISS_INDEX,
        faiss_map_path: str = DEFAULT_FAISS_MAP,
        tfidf_path: str = DEFAULT_TFIDF_PATH,
        text_embeddings_path: str = DEFAULT_TEXT_EMBEDDINGS,
        text_id_map_path: str = DEFAULT_TEXT_ID_MAP,
    ):
        # FAISS index (image embeddings)
        self._faiss = FaissIndex()
        self._faiss.load(faiss_index_path, faiss_map_path)

        # TF-IDF encoder
        self._text_encoder = TextEncoder()
        self._text_encoder.load(tfidf_path)

        # Pre-computed text embedding matrix — shape (N_text, vocab_size)
        self._text_embeddings: np.ndarray = np.load(text_embeddings_path)

        # ── FIX: text_embeddings rows are aligned with ALL products, not just
        # those with images.  We need a separate id_map for the text matrix.
        # generate_embeddings.py now saves this; fall back gracefully if missing
        # (e.g. old pipeline run) by reading DB order.
        if text_id_map_path and __import__("os").path.exists(text_id_map_path):
            with open(text_id_map_path, "rb") as f:
                self._text_id_map: list[int] = pickle.load(f)
        else:
            logger.warning(
                "text_id_map.pkl not found — falling back to DB insertion order. "
                "Re-run generate_embeddings.py to fix text search accuracy."
            )
            conn_tmp = init_db(db_path)
            from utils.db_utils import get_all_products
            self._text_id_map = [p["id"] for p in get_all_products(conn_tmp)]
            conn_tmp.close()

        # SQLite connection
        self._conn: sqlite3.Connection = init_db(db_path)

        # CNN extractor (lazy)
        self._cnn_extractor: Optional[CNNFeatureExtractor] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cnn_extractor(self) -> CNNFeatureExtractor:
        if self._cnn_extractor is None:
            self._cnn_extractor = CNNFeatureExtractor()
        return self._cnn_extractor

    def _lookup_products(self, product_ids: list[int]) -> list[dict]:
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
        """Search by image using fused CNN+color+texture embedding via FAISS."""
        extractor = self._get_cnn_extractor()
        query_vec = extract_fused_embedding(image_path, extractor)

        # ── FIX: load_image was called but image_path is already a local temp
        # file written by save_upload(); just pass it directly.
        query_image = load_image(image_path)
        query_color, color_conf = extract_dominant_color(query_image)

        pool = max(k * 5, 60)
        faiss_results = self._faiss.search(query_vec, pool)

        results = []
        for product_id, score in faiss_results:
            record = get_product_by_id(self._conn, product_id)
            if record:
                title = record.get("product_title") or ""
                brand = record.get("brand") or ""
                text_color = dominant_color_from_text(f"{title} {brand}")

                boost = 0.0
                if query_color != "unknown" and color_conf >= 0.35:
                    if text_color == query_color:
                        boost = 0.08
                    elif text_color != "unknown":
                        boost = -0.04

                record["similarity_score"] = float(score + boost)
                record["base_similarity_score"] = float(score)
                record["query_dominant_color"] = query_color
                results.append(record)

        results.sort(key=lambda r: r["similarity_score"], reverse=True)
        return results[:max(0, k)]

    def search_by_text(self, query: str, k: int = 20) -> list[dict]:
        """Search by text description using TF-IDF cosine similarity.

        BUG FIXED: previously used self._faiss.id_map[idx] to map text embedding
        row indices to product IDs.  The FAISS id_map only contains IDs of
        products that had images, so for any product without a downloaded image
        (very common) the mapping was completely wrong — wrong product returned,
        or IndexError.  Now uses self._text_id_map which is built from ALL
        products in DB insertion order, matching text_embeddings.npy row order.
        """
        query_vec = self._text_encoder.transform(query).astype(np.float32)

        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            logger.warning("Query '%s' produced a zero TF-IDF vector.", query)
            return []

        query_unit = query_vec / query_norm

        norms = np.linalg.norm(self._text_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        unit_embeddings = self._text_embeddings / norms

        scores = unit_embeddings.dot(query_unit)  # shape (N,)

        k_actual = min(k, len(scores))
        top_indices = np.argpartition(scores, -k_actual)[-k_actual:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            # ── FIX: use text_id_map, not faiss.id_map ──
            if idx >= len(self._text_id_map):
                logger.warning("text index %d out of range for text_id_map (len=%d)", idx, len(self._text_id_map))
                continue
            product_id = self._text_id_map[idx]
            record = get_product_by_id(self._conn, product_id)
            if record:
                record["similarity_score"] = float(scores[idx])
                results.append(record)

        return results

    def search_hybrid(self, image_path: str, query: str, k: int = 20) -> list[dict]:
        """Search using both image and text, combined via weighted hybrid score."""
        pool = k * 3

        image_results = self.search_by_image(image_path, k=pool)
        text_results  = self.search_by_text(query, k=pool)

        image_scores: dict[int, float] = {r["id"]: r["similarity_score"] for r in image_results}
        text_scores:  dict[int, float] = {r["id"]: r["similarity_score"] for r in text_results}

        all_ids = set(image_scores) | set(text_scores)

        record_map: dict[int, dict] = {}
        for r in image_results + text_results:
            record_map.setdefault(r["id"], r)

        # ── FIX: guard against None prices before min/max ──
        valid_prices = [
            record_map[pid].get("price")
            for pid in all_ids
            if record_map[pid].get("price") is not None
        ]
        min_p = min(valid_prices) if valid_prices else 0.0
        max_p = max(valid_prices) if valid_prices else 0.0
        price_range = (max_p - min_p) if max_p != min_p else 1.0

        candidates = []
        for pid in all_ids:
            record = dict(record_map[pid])
            img_sim   = image_scores.get(pid, 0.0)
            txt_sim   = text_scores.get(pid, 0.0)
            raw_price = record.get("price") or 0.0
            norm_price = (raw_price - min_p) / price_range
            price_score = 1.0 - norm_price

            hybrid = 0.65 * img_sim + 0.25 * txt_sim + 0.10 * price_score

            record["similarity_score"] = img_sim
            record["hybrid_score"]     = float(hybrid)
            candidates.append(record)

        candidates.sort(key=lambda r: r["hybrid_score"], reverse=True)
        return candidates[:k]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Test SimilaritySearch")
    parser.add_argument("--image", help="Path to query image")
    parser.add_argument("--text",  help="Text query string")
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