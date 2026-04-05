"""Text utility functions for cleaning product text and parsing prices."""

import re
import logging

logger = logging.getLogger(__name__)

_CURRENCY_PATTERN = re.compile(r"[₹$£€¥\s,]")
_PRICE_PATTERN    = re.compile(r"[\d]+(?:\.\d+)?")


def clean_text(text: str) -> str:
    """Lowercase and strip punctuation from text."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_price(raw_str: str) -> float | None:
    """Parse a price string with currency symbols into a float.

    Returns:
        Numeric price as float, or None if parsing fails.

    FIX: previously returned 0.0 on failure, which caused ranking_engine and
    similarity_search hybrid scoring to treat products with unknown prices as
    "free" (cheapest possible), skewing price-based ranking heavily.  Returning
    None lets callers distinguish "price unknown" from "price is zero".
    """
    if not raw_str:
        return None
    try:
        cleaned = _CURRENCY_PATTERN.sub("", str(raw_str))
        match   = _PRICE_PATTERN.search(cleaned)
        if match:
            return float(match.group())
    except Exception as e:
        logger.warning("Failed to parse price from '%s': %s", raw_str, e)
    return None