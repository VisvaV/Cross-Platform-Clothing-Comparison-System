# Design Document

## Overview

The Cross-Platform Clothing Comparison System is a Python-based MVP that combines web scraping, deep learning feature extraction, and vector similarity search to let users find visually similar clothing products across multiple e-commerce platforms. The system is split into two phases:

1. **Offline Pipeline** — run once to scrape products, generate embeddings, and build the FAISS index.
2. **Online App** — a Streamlit UI that loads the pre-built index and serves search queries in real time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OFFLINE PIPELINE                         │
│                                                                 │
│  Scrapers (Playwright) → SQLite (products.db)                   │
│       ↓                                                         │
│  Image Downloader → data/images/                                │
│       ↓                                                         │
│  Feature Extraction:                                            │
│    cnn_features (ResNet50, 2048)                                 │
│    color_features (HSV histogram, 512)                          │
│    texture_features (LBP, 256)                                  │
│    feature_fusion → 2816-dim normalized vector                  │
│       ↓                                                         │
│  TF-IDF Vectorizer (fit on titles) → tfidf_vectorizer.pkl       │
│       ↓                                                         │
│  FAISS IndexFlatIP → faiss_index.bin                            │
│  Product ID mapping → faiss_id_map.pkl                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        ONLINE APP (Streamlit)                   │
│                                                                 │
│  User Input (image / text / both)                               │
│       ↓                                                         │
│  similarity_search.py                                           │
│    → image path  → feature_fusion → FAISS query                 │
│    → text query  → TF-IDF transform → cosine similarity         │
│    → hybrid      → weighted score combination                   │
│       ↓                                                         │
│  ranking_engine.py → sort by score or price                     │
│       ↓                                                         │
│  SQLite metadata lookup → product cards in Streamlit UI         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
clothing-comparison-system/
├── datasets/
│   └── DeepFashion/          # Already downloaded locally
│       ├── Anno/
│       ├── Eval/
│       └── img/
├── data/
│   ├── images/               # Downloaded product images
│   ├── products.db           # SQLite metadata
│   ├── faiss_index.bin       # Persisted FAISS index
│   ├── faiss_id_map.pkl      # FAISS position → product_id mapping
│   └── tfidf_vectorizer.pkl  # Fitted TF-IDF vectorizer
├── features/
│   ├── cnn_features.py
│   ├── color_features.py
│   ├── texture_features.py
│   └── feature_fusion.py
├── scraper/
│   ├── myntra_scraper.py
│   ├── ajio_scraper.py
│   ├── hm_scraper.py
│   ├── zara_scraper.py
│   ├── amazon_scraper.py
│   └── flipkart_scraper.py
├── models/
│   ├── cnn_encoder.py
│   ├── text_encoder.py
│   └── triplet_model.py
├── training/
│   ├── train_triplet_network.py
│   └── dataset_loader.py
├── embeddings/
│   └── generate_embeddings.py
├── vector_search/
│   ├── faiss_index.py
│   └── similarity_search.py
├── ranking/
│   └── ranking_engine.py
├── app/
│   └── streamlit_app.py
├── utils/
│   ├── image_utils.py
│   └── text_utils.py
├── requirements.txt
└── README.md
```

---

## Components and Interfaces

### `features/cnn_features.py`

Wraps ResNet50 as a feature extractor.

```python
class CNNFeatureExtractor:
    def __init__(self, weights_path: str = None)
    def extract(self, image: PIL.Image) -> np.ndarray  # shape (2048,), L2-normalized
```

- Loads ResNet50 pretrained on ImageNet via `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)`
- Replaces `model.fc` with `nn.Identity()` to expose the 2048-dim GAP output
- Loads fine-tuned weights from `models/resnet50_finetuned.pth` if the file exists
- Moves model to `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Runs `model.eval()` — no gradient computation during inference

### `features/color_features.py`

Extracts HSV color histogram.

```python
def extract_color_features(image: PIL.Image) -> np.ndarray  # shape (512,), L2-normalized
```

- Converts PIL image to BGR numpy array, then to HSV via `cv2.cvtColor`
- Computes 3D histogram: `cv2.calcHist` with bins [8, 8, 8] over H, S, V channels
- Flattens to 512-dim vector, normalizes to unit length
- Returns zero vector on error

### `features/texture_features.py`

Extracts LBP texture histogram.

```python
def extract_texture_features(image: PIL.Image) -> np.ndarray  # shape (256,), L2-normalized
```

- Converts PIL image to grayscale numpy array
- Computes LBP via `skimage.feature.local_binary_pattern(gray, P=24, R=3, method='uniform')`
- Builds histogram with 256 bins, normalizes to unit length
- Returns zero vector on error

### `features/feature_fusion.py`

Fuses all three feature vectors.

```python
def fuse_features(
    cnn: np.ndarray,      # (2048,)
    color: np.ndarray,    # (512,)
    texture: np.ndarray   # (256,)
) -> np.ndarray           # (2816,), L2-normalized

def extract_fused_embedding(image_path: str, extractor: CNNFeatureExtractor) -> np.ndarray
```

- Concatenates the three vectors: `np.concatenate([cnn, color, texture])`
- Normalizes the result to unit length
- `extract_fused_embedding` is the single entry point used by the pipeline and the app

### `models/cnn_encoder.py`

Thin wrapper around `CNNFeatureExtractor` — re-exports it for use by other modules.

### `models/text_encoder.py`

TF-IDF vectorizer wrapper.

```python
class TextEncoder:
    def fit(self, texts: list[str]) -> None
    def transform(self, text: str) -> np.ndarray   # sparse → dense
    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

- Uses `sklearn.feature_extraction.text.TfidfVectorizer(max_features=10000)`
- Persists via `joblib.dump` / `joblib.load`

### `models/triplet_model.py`

Triplet network wrapper.

```python
class TripletNet(nn.Module):
    def __init__(self, base_encoder: nn.Module)
    def forward(self, anchor, positive, negative) -> tuple[Tensor, Tensor, Tensor]
```

- Shares weights across all three branches (same `base_encoder` instance)

### `training/dataset_loader.py`

DeepFashion triplet dataset.

```python
class DeepFashionTripletDataset(Dataset):
    def __init__(self, dataset_root: str, partition_file: str, split: str = "train")
    def __len__(self) -> int
    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor]  # anchor, positive, negative
```

- Parses `DeepFashion/Eval/list_eval_partition.txt` to get image paths grouped by item_id and split
- For each sample: anchor and positive are two different images of the same item_id; negative is a random image from a different item_id
- Applies standard ImageNet preprocessing transforms

### `training/train_triplet_network.py`

Training script.

- Instantiates `DeepFashionTripletDataset`, `TripletNet`, `TripletMarginLoss(margin=0.3)`
- Trains for configurable epochs with Adam optimizer
- Saves best weights to `models/resnet50_finetuned.pth`
- Logs loss per epoch to stdout

### `scraper/` — Platform Scrapers

Each scraper follows the same contract:

```python
async def scrape_products(category: str, limit: int) -> list[dict]
```

Each returned dict:
```python
{
    "product_title": str,
    "brand": str,
    "price": float,
    "product_url": str,
    "image_url": str,
    "category": str,
    "platform": str
}
```

Common scraper behavior:
- `async_playwright()` with Chromium, `headless=True`
- User-agent: `"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"`
- `await page.wait_for_load_state("networkidle")`
- Auto-scroll loop: scroll by viewport height, check if `scrollHeight` increased, stop when stable
- Randomized `asyncio.sleep(random.uniform(1, 3))` between requests
- Full `try/except` — log error and return `[]` on failure
- Multiple CSS selector fallbacks per field

### `embeddings/generate_embeddings.py`

Offline pipeline orchestrator.

```python
def run_pipeline(categories: list[str], limit_per_platform: int = 50)
```

Steps:
1. For each category × platform: call `scrape_products`, insert into SQLite
2. Download images to `data/images/{platform}_{product_id}.jpg`
3. For each product without an embedding: call `extract_fused_embedding`, store as numpy array
4. Fit `TextEncoder` on all product titles, save to `data/tfidf_vectorizer.pkl`
5. Build FAISS index from all fused embeddings, save to `data/faiss_index.bin`
6. Save FAISS position → product_id map to `data/faiss_id_map.pkl`

### `vector_search/faiss_index.py`

FAISS index management.

```python
class FaissIndex:
    def build(self, embeddings: np.ndarray, id_map: list[int]) -> None
    def save(self, index_path: str, map_path: str) -> None
    def load(self, index_path: str, map_path: str) -> None
    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]
    # returns list of (product_id, similarity_score)
```

- Uses `faiss.IndexFlatIP` (inner product = cosine similarity on normalized vectors)
- Wraps with `faiss.index_cpu_to_gpu` when GPU resource is available

### `vector_search/similarity_search.py`

High-level search interface.

```python
class SimilaritySearch:
    def search_by_image(self, image_path: str, k: int = 20) -> list[dict]
    def search_by_text(self, query: str, k: int = 20) -> list[dict]
    def search_hybrid(self, image_path: str, query: str, k: int = 20) -> list[dict]
```

- `search_by_image`: extract fused embedding → FAISS search → SQLite lookup
- `search_by_text`: TF-IDF transform → cosine similarity against stored text embeddings → top-K
- `search_hybrid`: compute both scores, normalize prices, apply weighted formula, sort

### `ranking/ranking_engine.py`

```python
def rank_by_score(results: list[dict]) -> list[dict]
def rank_by_price(results: list[dict]) -> list[dict]
def compute_hybrid_score(
    image_sim: float, text_sim: float, normalized_price: float
) -> float
```

- `compute_hybrid_score`: `0.65 * image_sim + 0.25 * text_sim + 0.10 * (1 / (normalized_price + 1e-6))`

### `utils/image_utils.py`

```python
def download_image(url: str, save_path: str) -> bool
def load_image(path: str) -> PIL.Image
def preprocess_for_cnn(image: PIL.Image) -> torch.Tensor
```

### `utils/text_utils.py`

```python
def clean_text(text: str) -> str          # lowercase, strip punctuation
def extract_price(raw: str) -> float      # parse "₹1,299" → 1299.0
```

### `app/streamlit_app.py`

Layout:
- Sidebar or top panel: file uploader + text input + Search button + sort radio
- Main area: grid of product cards (3 columns)
- Each card: `st.image`, title, platform badge, price, similarity score, `st.link_button`
- Loading spinner via `st.spinner`
- Empty state message when index not found

---

## Data Models

### SQLite `products` table

| Column        | Type    | Notes                          |
|---------------|---------|--------------------------------|
| id            | INTEGER | Primary key, autoincrement     |
| product_title | TEXT    | Product name                   |
| brand         | TEXT    | Brand name                     |
| price         | REAL    | Numeric price                  |
| platform      | TEXT    | e.g. "Myntra", "H&M"           |
| category      | TEXT    | e.g. "hoodies", "dresses"      |
| product_url   | TEXT    | UNIQUE — deduplication key     |
| image_url     | TEXT    | Original URL                   |
| image_path    | TEXT    | Local path under data/images/  |

### In-memory result dict (returned by search)

```python
{
    "product_id": int,
    "product_title": str,
    "brand": str,
    "price": float,
    "platform": str,
    "category": str,
    "product_url": str,
    "image_path": str,
    "similarity_score": float,
    "hybrid_score": float   # only in hybrid search
}
```

### FAISS artifacts on disk

| File                      | Format  | Contents                              |
|---------------------------|---------|---------------------------------------|
| `data/faiss_index.bin`    | binary  | FAISS IndexFlatIP, dim=2816           |
| `data/faiss_id_map.pkl`   | pickle  | `list[int]` — position → product_id  |
| `data/tfidf_vectorizer.pkl` | joblib | Fitted TfidfVectorizer                |
| `data/text_embeddings.npy` | numpy  | Dense text embedding matrix           |

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Scraper blocked by platform | Log warning, return `[]`, continue pipeline |
| Image download fails | Log warning, skip product embedding |
| FAISS index not found on app start | Show "Run pipeline first" message in UI |
| CUDA not available | Fall back to CPU silently |
| Fine-tuned weights not found | Fall back to ImageNet pretrained weights |
| Image file corrupt or missing | Return zero vector for that feature |
| Price parsing fails | Default price to 0.0 |

---

## Testing Strategy

- `training/train_triplet_network.py` can be run standalone to verify GPU training works
- `embeddings/generate_embeddings.py` accepts a `--dry-run` flag to test pipeline flow without scraping
- Each scraper module can be run directly (`python scraper/hm_scraper.py`) to test independently
- `similarity_search.py` includes a `__main__` block for quick CLI testing
- The Streamlit app gracefully handles missing index files with user-facing error messages

---

## Key Design Decisions

1. **IndexFlatIP over IndexIVFFlat**: For an MVP with up to a few thousand products, exact search is fast enough and avoids the complexity of IVF training. Can be upgraded later.

2. **Fused embedding stored once**: The 2816-dim fused vector is computed offline and stored in FAISS. At query time only the query image needs feature extraction — no per-product recomputation.

3. **TF-IDF over dense text embeddings**: Avoids external model dependencies (no sentence-transformers, no API calls). Sufficient for keyword-level clothing description matching.

4. **Scraper-per-platform isolation**: Each scraper is fully independent. A failure in one does not affect others. Fallback platforms are tried automatically.

5. **DeepFashion triplet construction**: Uses item_id grouping from `list_eval_partition.txt`. Multiple images per item_id naturally provide anchor/positive pairs. Negatives are sampled randomly from other item_ids — simple but effective for metric learning.

6. **Fine-tuned weights are optional**: The system works with ImageNet pretrained weights out of the box. Fine-tuning is an enhancement that improves fashion-specific retrieval quality.
