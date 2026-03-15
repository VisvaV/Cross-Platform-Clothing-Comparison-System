import sqlite3
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def init_db(db_path: str) -> sqlite3.Connection:
    """Create the products table if it doesn't exist and return a connection."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            product_title TEXT,
            brand         TEXT,
            price         REAL,
            platform      TEXT,
            category      TEXT,
            product_url   TEXT UNIQUE,
            image_url     TEXT,
            image_path    TEXT
        )
    """)
    conn.commit()
    return conn


def insert_product(conn: sqlite3.Connection, record: dict) -> None:
    """Insert a product record, silently skipping duplicates on product_url."""
    conn.execute(
        """
        INSERT OR IGNORE INTO products
            (product_title, brand, price, platform, category, product_url, image_url, image_path)
        VALUES
            (:product_title, :brand, :price, :platform, :category, :product_url, :image_url, :image_path)
        """,
        {
            "product_title": record.get("product_title"),
            "brand":         record.get("brand"),
            "price":         record.get("price"),
            "platform":      record.get("platform"),
            "category":      record.get("category"),
            "product_url":   record.get("product_url"),
            "image_url":     record.get("image_url"),
            "image_path":    record.get("image_path"),
        },
    )
    conn.commit()


def get_product_by_id(conn: sqlite3.Connection, product_id: int) -> Optional[dict]:
    """Return a single product record as a dict, or None if not found."""
    row = conn.execute(
        "SELECT * FROM products WHERE id = ?", (product_id,)
    ).fetchone()
    return dict(row) if row else None


def get_all_products(conn: sqlite3.Connection) -> list:
    """Return all product records as a list of dicts."""
    rows = conn.execute("SELECT * FROM products").fetchall()
    return [dict(row) for row in rows]
