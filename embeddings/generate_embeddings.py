"""Offline embedding generation pipeline.

Run once to scrape products, download images, generate fused embeddings,
fit the TF-IDF encoder, and build the FAISS index.

Usage:
    python -m embeddings.generate_embeddings --categories hoodies dresses --limit 50
"""

import argparse
import asyncio
import logging
import os
import pickle
import sqlite3

import numpy as np

from features.cnn_features import CNNFeatureExtractor
from features.feature_fusion import extract_fused_embedding
from models.text_encoder import TextEncoder
from utils.db_utils import init_db, insert_product, get_all_products
from utils.image_utils import download_image
from vector_search.faiss_index import FaissIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH              = "data/products.db"
IMAGES_DIR           = "data/images"
FAISS_INDEX_PATH     = "data/faiss_index.bin"
FAISS_MAP_PATH       = "data/faiss_id_map.pkl"
TFIDF_PATH           = "data/tfidf_vectorizer.pkl"
TEXT_EMBEDDINGS_PATH = "data/text_embeddings.npy"
TEXT_ID_MAP_PATH     = "data/text_id_map.pkl"   # NEW — keeps text rows aligned with product IDs

# ---------------------------------------------------------------------------
# Platform scraper registry
# ---------------------------------------------------------------------------
from scraper.hm_scraper       import scrape_products as scrape_hm
from scraper.zara_scraper      import scrape_products as scrape_zara
from scraper.myntra_scraper    import scrape_products as scrape_myntra
from scraper.ajio_scraper      import scrape_products as scrape_ajio
from scraper.amazon_scraper    import scrape_products as scrape_amazon
from scraper.flipkart_scraper  import scrape_products as scrape_flipkart
from scraper.asos_scraper      import scrape_products as scrape_asos
from scraper.uniqlo_scraper    import scrape_products as scrape_uniqlo

PRIMARY_SCRAPERS = [
    ("H&M",      scrape_hm),
    ("Zara",     scrape_zara),
    ("Myntra",   scrape_myntra),
    ("Ajio",     scrape_ajio),
    ("Amazon",   scrape_amazon),
    ("Flipkart", scrape_flipkart),
]

FALLBACK_SCRAPERS = [
    ("ASOS",   scrape_asos),
    ("Uniqlo", scrape_uniqlo),
]


# ---------------------------------------------------------------------------
# Step 1 — Scrape and store products
# ---------------------------------------------------------------------------

async def _scrape_platform(name: str, scrape_fn, category: str, limit: int) -> list:
    try:
        products = await scrape_fn(category, limit)
        logger.info("[%s] scraped %d products for category '%s'.", name, len(products), category)
        return products
    except Exception as exc:
        logger.warning("[%s] scraper failed for category '%s': %s", name, category, exc)
        return []


async def scrape_and_store(
    conn: sqlite3.Connection,
    categories: list,
    limit_per_platform: int,
) -> None:
    scrapers = PRIMARY_SCRAPERS[:]
    any_primary_succeeded = False

    for category in categories:
        for name, scrape_fn in scrapers:
            products = await _scrape_platform(name, scrape_fn, category, limit_per_platform)
            if products:
                any_primary_succeeded = True
            for product in products:
                product.setdefault("image_path", None)
                insert_product(conn, product)

    if not any_primary_succeeded:
        logger.warning("All primary scrapers returned no results — trying fallbacks.")
        for category in categories:
            for name, scrape_fn in FALLBACK_SCRAPERS:
                products = await _scrape_platform(name, scrape_fn, category, limit_per_platform)
                for product in products:
                    product.setdefault("image_path", None)
                    insert_product(conn, product)


# ---------------------------------------------------------------------------
# Step 2 — Download images
# ---------------------------------------------------------------------------

def download_images(conn: sqlite3.Connection) -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    products = get_all_products(conn)

    for product in products:
        if product.get("image_path") and os.path.exists(product["image_path"]):
            continue

        image_url = product.get("image_url")
        if not image_url:
            continue

        platform   = (product.get("platform") or "unknown").replace(" ", "_").lower()
        product_id = product["id"]
        save_path  = os.path.join(IMAGES_DIR, f"{platform}_{product_id}.jpg")

        success = download_image(image_url, save_path)
        if success:
            conn.execute(
                "UPDATE products SET image_path = ? WHERE id = ?",
                (save_path, product_id),
            )
            conn.commit()
        else:
            logger.warning("Skipping image download for product_id=%d.", product_id)


# ---------------------------------------------------------------------------
# Step 3 — Generate fused image embeddings
# ---------------------------------------------------------------------------

def generate_image_embeddings(
    conn: sqlite3.Connection,
    extractor: CNNFeatureExtractor,
) -> tuple:
    """Returns (embeddings_matrix, id_map) for products that have a local image."""
    products   = get_all_products(conn)
    embeddings = []
    id_map     = []

    for product in products:
        image_path = product.get("image_path")
        if not image_path or not os.path.exists(image_path):
            logger.debug("No image for product_id=%d — skipping image embedding.", product["id"])
            continue

        try:
            vec = extract_fused_embedding(image_path, extractor)
            embeddings.append(vec)
            id_map.append(product["id"])
        except Exception as exc:
            logger.warning("Failed to extract embedding for product_id=%d: %s", product["id"], exc)

    if not embeddings:
        return np.empty((0, 2816), dtype=np.float32), []

    return np.array(embeddings, dtype=np.float32), id_map


# ---------------------------------------------------------------------------
# Step 4 — Fit TF-IDF and save text embeddings
#
# FIX: previously, text_embeddings.npy rows were ordered by DB insertion order
# (ALL products), but similarity_search.py was using faiss.id_map (which only
# covers products WITH images) to map row indices to product IDs — completely
# wrong for any query involving non-hoodie categories where many image downloads
# fail.
#
# Fix: save a text_id_map.pkl alongside text_embeddings.npy that explicitly
# records which product ID each row corresponds to.  similarity_search.py
# loads this map instead of reusing faiss.id_map.
# ---------------------------------------------------------------------------

def generate_text_embeddings(conn: sqlite3.Connection) -> None:
    """Fit TextEncoder on all product titles and save embeddings + id map."""
    products = get_all_products(conn)
    # Combine title, brand, and category into a single rich text string for TF-IDF
    titles   = [f"{p.get('product_title') or ''} {p.get('brand') or ''} {p.get('category') or ''}".strip() for p in products]
    ids      = [p["id"] for p in products]

    if not titles:
        logger.warning("No product titles found — skipping text encoder fitting.")
        return

    encoder = TextEncoder()
    encoder.fit(titles)
    encoder.save(TFIDF_PATH)

    matrix = np.array([encoder.transform(t) for t in titles], dtype=np.float32)
    np.save(TEXT_EMBEDDINGS_PATH, matrix)

    # ── NEW: save the aligned id map so similarity_search can look up products
    with open(TEXT_ID_MAP_PATH, "wb") as f:
        pickle.dump(ids, f)

    logger.info(
        "Text embeddings saved: %s  shape=%s  id_map: %d entries",
        TEXT_EMBEDDINGS_PATH, matrix.shape, len(ids),
    )


# ---------------------------------------------------------------------------
# Step 5 — Build and save FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray, id_map: list) -> None:
    if embeddings.shape[0] == 0:
        logger.warning("No image embeddings to index — FAISS index will not be created.")
        return

    index = FaissIndex()
    index.build(embeddings, id_map)
    index.save(FAISS_INDEX_PATH, FAISS_MAP_PATH)
    logger.info("FAISS index saved: %s  (%d vectors)", FAISS_INDEX_PATH, embeddings.shape[0])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_pipeline_async(categories: list, limit_per_platform: int = 50) -> None:
    os.makedirs("data", exist_ok=True)
    conn = init_db(DB_PATH)

    logger.info("=== Step 1: Scraping products ===")
    await scrape_and_store(conn, categories, limit_per_platform)

    logger.info("=== Step 2: Downloading images ===")
    download_images(conn)

    logger.info("=== Step 3: Generating fused image embeddings ===")
    extractor = CNNFeatureExtractor()
    embeddings, id_map = generate_image_embeddings(conn, extractor)
    logger.info("Accumulated %d image embeddings.", len(id_map))

    logger.info("=== Step 4: Fitting TF-IDF and saving text embeddings + id map ===")
    generate_text_embeddings(conn)

    logger.info("=== Step 5: Building FAISS index ===")
    build_faiss_index(embeddings, id_map)

    total = len(get_all_products(conn))
    logger.info("=== Pipeline complete: %d total products indexed. ===", total)
    conn.close()


def run_pipeline(categories: list, limit_per_platform: int = 50) -> None:
    asyncio.run(run_pipeline_async(categories, limit_per_platform))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the offline clothing embedding pipeline.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["hoodies", "dresses", "t-shirts", "jeans"],
    )
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    run_pipeline(args.categories, args.limit)