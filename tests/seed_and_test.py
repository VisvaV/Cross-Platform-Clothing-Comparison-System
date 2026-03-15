"""
End-to-end pipeline test with synthetic data.
Seeds the DB with mock products, generates real embeddings,
builds the FAISS index, then tests text, image, and hybrid search.
"""

import logging
import os
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

# ── ensure project root is on path ──────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
DB_PATH          = "data/test_products.db"
IMAGES_DIR       = "data/test_images"
FAISS_INDEX_PATH = "data/test_faiss_index.bin"
FAISS_MAP_PATH   = "data/test_faiss_id_map.pkl"
TFIDF_PATH       = "data/test_tfidf_vectorizer.pkl"
TEXT_EMB_PATH    = "data/test_text_embeddings.npy"

os.makedirs("data", exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ── synthetic product catalogue ──────────────────────────────────────────────
MOCK_PRODUCTS = [
    {"product_title": "Red Floral Summer Dress",      "brand": "Zara",     "price": 2499, "platform": "Zara",     "category": "dresses",  "product_url": "https://zara.com/1",     "image_url": "", "color": (220, 60,  60)},
    {"product_title": "Blue Denim Jacket",            "brand": "H&M",      "price": 1999, "platform": "H&M",      "category": "jackets",  "product_url": "https://hm.com/2",       "image_url": "", "color": (60,  100, 200)},
    {"product_title": "White Cotton T-Shirt",         "brand": "Uniqlo",   "price":  799, "platform": "Uniqlo",   "category": "t-shirts", "product_url": "https://uniqlo.com/3",   "image_url": "", "color": (240, 240, 240)},
    {"product_title": "Black Slim Fit Jeans",         "brand": "Myntra",   "price": 1499, "platform": "Myntra",   "category": "jeans",    "product_url": "https://myntra.com/4",   "image_url": "", "color": (30,  30,  30)},
    {"product_title": "Green Floral Maxi Dress",      "brand": "ASOS",     "price": 3299, "platform": "ASOS",     "category": "dresses",  "product_url": "https://asos.com/5",     "image_url": "", "color": (60,  180, 80)},
    {"product_title": "Yellow Striped Hoodie",        "brand": "Flipkart", "price": 1299, "platform": "Flipkart", "category": "hoodies",  "product_url": "https://flipkart.com/6", "image_url": "", "color": (240, 220, 50)},
    {"product_title": "Pink Casual Kurta",            "brand": "Ajio",     "price":  999, "platform": "Ajio",     "category": "kurtas",   "product_url": "https://ajio.com/7",     "image_url": "", "color": (240, 130, 180)},
    {"product_title": "Navy Blue Formal Shirt",       "brand": "Amazon",   "price": 1199, "platform": "Amazon",   "category": "shirts",   "product_url": "https://amazon.com/8",   "image_url": "", "color": (20,  40,  120)},
    {"product_title": "Orange Printed Kurti",         "brand": "Myntra",   "price":  849, "platform": "Myntra",   "category": "kurtas",   "product_url": "https://myntra.com/9",   "image_url": "", "color": (240, 130, 40)},
    {"product_title": "Purple Velvet Evening Gown",   "brand": "Zara",     "price": 5999, "platform": "Zara",     "category": "dresses",  "product_url": "https://zara.com/10",    "image_url": "", "color": (130, 50,  200)},
    {"product_title": "Grey Oversized Sweatshirt",    "brand": "H&M",      "price": 1599, "platform": "H&M",      "category": "hoodies",  "product_url": "https://hm.com/11",      "image_url": "", "color": (160, 160, 160)},
    {"product_title": "Beige Linen Trousers",         "brand": "Uniqlo",   "price": 1899, "platform": "Uniqlo",   "category": "trousers", "product_url": "https://uniqlo.com/12",  "image_url": "", "color": (210, 190, 160)},
    {"product_title": "Teal Wrap Midi Dress",         "brand": "ASOS",     "price": 2799, "platform": "ASOS",     "category": "dresses",  "product_url": "https://asos.com/13",    "image_url": "", "color": (40,  180, 170)},
    {"product_title": "Maroon Woolen Sweater",        "brand": "Flipkart", "price": 1699, "platform": "Flipkart", "category": "sweaters", "product_url": "https://flipkart.com/14","image_url": "", "color": (140, 30,  50)},
    {"product_title": "Cream Embroidered Blouse",     "brand": "Ajio",     "price": 1099, "platform": "Ajio",     "category": "tops",     "product_url": "https://ajio.com/15",    "image_url": "", "color": (245, 235, 210)},
]


def make_solid_image(color: tuple, size: int = 224) -> Image.Image:
    """Create a solid-colour PIL image with a small shape overlay for texture."""
    img = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(img)
    # add a contrasting rectangle so texture features are non-trivial
    c2 = tuple(max(0, c - 80) for c in color)
    draw.rectangle([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=c2)
    return img


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — seed DB
# ════════════════════════════════════════════════════════════════════════════
logger.info("=== STEP 1: Seeding database ===")
from utils.db_utils import init_db, insert_product, get_all_products

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = init_db(DB_PATH)
for p in MOCK_PRODUCTS:
    record = {k: v for k, v in p.items() if k != "color"}
    insert_product(conn, record)

products = get_all_products(conn)
logger.info("Inserted %d products into DB.", len(products))
assert len(products) == len(MOCK_PRODUCTS), "DB seed count mismatch"


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — generate synthetic images and update image_path
# ════════════════════════════════════════════════════════════════════════════
logger.info("=== STEP 2: Generating synthetic images ===")
for product, mock in zip(products, MOCK_PRODUCTS):
    img = make_solid_image(mock["color"])
    path = os.path.join(IMAGES_DIR, f"product_{product['id']}.jpg")
    img.save(path)
    conn.execute("UPDATE products SET image_path = ? WHERE id = ?", (path, product["id"]))
conn.commit()
logger.info("Generated %d synthetic images.", len(products))


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — generate fused image embeddings
# ════════════════════════════════════════════════════════════════════════════
logger.info("=== STEP 3: Generating fused image embeddings ===")
from features.feature_fusion import extract_fused_embedding
from models.cnn_encoder import CNNFeatureExtractor

extractor = CNNFeatureExtractor()
products = get_all_products(conn)

embeddings, id_map = [], []
for p in products:
    vec = extract_fused_embedding(p["image_path"], extractor)
    embeddings.append(vec)
    id_map.append(p["id"])

embeddings = np.array(embeddings, dtype=np.float32)
logger.info("Embeddings shape: %s", embeddings.shape)
assert embeddings.shape == (len(MOCK_PRODUCTS), 2816), f"Unexpected shape: {embeddings.shape}"


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — fit TF-IDF and save text embeddings
# ════════════════════════════════════════════════════════════════════════════
logger.info("=== STEP 4: Fitting TF-IDF text encoder ===")
from models.text_encoder import TextEncoder

titles = [p["product_title"] for p in products]
encoder = TextEncoder()
encoder.fit(titles)
encoder.save(TFIDF_PATH)

text_matrix = np.array([encoder.transform(t) for t in titles], dtype=np.float32)
np.save(TEXT_EMB_PATH, text_matrix)
logger.info("Text embeddings shape: %s", text_matrix.shape)
assert text_matrix.shape[0] == len(MOCK_PRODUCTS)


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — build FAISS index
# ════════════════════════════════════════════════════════════════════════════
logger.info("=== STEP 5: Building FAISS index ===")
from vector_search.faiss_index import FaissIndex

faiss_idx = FaissIndex()
faiss_idx.build(embeddings, id_map)
faiss_idx.save(FAISS_INDEX_PATH, FAISS_MAP_PATH)
logger.info("FAISS index built with %d vectors.", embeddings.shape[0])


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — load SimilaritySearch and run all three search modes
# ════════════════════════════════════════════════════════════════════════════
logger.info("=== STEP 6: Loading SimilaritySearch ===")
from vector_search.similarity_search import SimilaritySearch

searcher = SimilaritySearch(
    db_path=DB_PATH,
    faiss_index_path=FAISS_INDEX_PATH,
    faiss_map_path=FAISS_MAP_PATH,
    tfidf_path=TFIDF_PATH,
    text_embeddings_path=TEXT_EMB_PATH,
)

# ── TEXT SEARCH ──────────────────────────────────────────────────────────────
logger.info("--- TEXT SEARCH: 'red floral dress' ---")
text_results = searcher.search_by_text("red floral dress", k=5)
assert len(text_results) > 0, "Text search returned no results"
print("\n[TEXT SEARCH] Query: 'red floral dress'")
for i, r in enumerate(text_results, 1):
    print(f"  {i}. [{r['platform']}] {r['product_title']} — ₹{r['price']} (score: {r['similarity_score']:.4f})")
assert text_results[0]["similarity_score"] > 0, "Top text result has zero score"

# ── IMAGE SEARCH ─────────────────────────────────────────────────────────────
logger.info("--- IMAGE SEARCH: red-ish solid image ---")
query_img = make_solid_image((210, 55, 55))  # red-ish, should match "Red Floral Summer Dress"
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    query_img.save(tmp.name)
    tmp_path = tmp.name

image_results = searcher.search_by_image(tmp_path, k=5)
os.unlink(tmp_path)
assert len(image_results) > 0, "Image search returned no results"
print("\n[IMAGE SEARCH] Query: red-ish image")
for i, r in enumerate(image_results, 1):
    print(f"  {i}. [{r['platform']}] {r['product_title']} — ₹{r['price']} (score: {r['similarity_score']:.4f})")
assert image_results[0]["similarity_score"] > 0, "Top image result has zero score"

# ── HYBRID SEARCH ────────────────────────────────────────────────────────────
logger.info("--- HYBRID SEARCH: blue image + 'denim jacket' text ---")
query_img2 = make_solid_image((55, 90, 200))  # blue-ish
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    query_img2.save(tmp.name)
    tmp_path2 = tmp.name

hybrid_results = searcher.search_hybrid(tmp_path2, "blue denim jacket", k=5)
os.unlink(tmp_path2)
assert len(hybrid_results) > 0, "Hybrid search returned no results"
print("\n[HYBRID SEARCH] Query: blue image + 'blue denim jacket'")
for i, r in enumerate(hybrid_results, 1):
    score = r.get("hybrid_score", 0)
    print(f"  {i}. [{r['platform']}] {r['product_title']} — ₹{r['price']} (hybrid: {score:.4f})")
assert hybrid_results[0].get("hybrid_score", 0) > 0, "Top hybrid result has zero score"

# ── RANKING ──────────────────────────────────────────────────────────────────
logger.info("--- RANKING: rank_by_price ---")
from ranking.ranking_engine import rank_by_price, rank_by_score

price_ranked = rank_by_price(text_results)
prices = [r["price"] for r in price_ranked]
assert prices == sorted(prices), "rank_by_price not sorted ascending"
print("\n[RANK BY PRICE]")
for r in price_ranked:
    print(f"  ₹{r['price']}  {r['product_title']}")

score_ranked = rank_by_score(text_results)
scores = [r["similarity_score"] for r in score_ranked]
assert scores == sorted(scores, reverse=True), "rank_by_score not sorted descending"

# ════════════════════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ALL TESTS PASSED — pipeline is fully functional.")
print("="*60)
conn.close()
