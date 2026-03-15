import pickle
import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 2816


class FaissIndex:
    def __init__(self):
        self.index = None
        self.id_map: list[int] = []

    def build(self, embeddings: np.ndarray, id_map: list[int]) -> None:
        """Build IndexFlatIP from normalized embeddings and store product id mapping."""
        embeddings = embeddings.astype(np.float32)
        cpu_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        cpu_index.add(embeddings)

        # Try GPU, fall back to CPU silently
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            logger.info("FAISS index moved to GPU.")
        except Exception:
            self.index = cpu_index
            logger.info("FAISS index running on CPU.")

        self.id_map = list(id_map)

    def save(self, index_path: str, map_path: str) -> None:
        """Persist FAISS index binary and id_map pickle to disk."""
        # If index is on GPU, move back to CPU before writing
        try:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        except Exception:
            cpu_index = self.index

        faiss.write_index(cpu_index, index_path)
        with open(map_path, "wb") as f:
            pickle.dump(self.id_map, f)
        logger.info(f"FAISS index saved to {index_path}, id_map saved to {map_path}.")

    def load(self, index_path: str, map_path: str) -> None:
        """Load FAISS index and id_map from disk."""
        cpu_index = faiss.read_index(index_path)

        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            logger.info("Loaded FAISS index onto GPU.")
        except Exception:
            self.index = cpu_index
            logger.info("Loaded FAISS index on CPU.")

        with open(map_path, "rb") as f:
            self.id_map = pickle.load(f)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return top-k (product_id, score) pairs for a query embedding."""
        query = query.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.id_map[idx], float(score)))
        return results
