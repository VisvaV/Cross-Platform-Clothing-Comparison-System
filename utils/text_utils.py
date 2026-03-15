"""Text utility functions for cleaning product text and parsing prices."""

import re
import logging

logger = logging.getLogger(__name__)

# Currency symbols and their common string representations
_CURRENCY_PATTERN = re.compile(r"[₹$£€¥\s,]")
_PRICE_PATTERN = re.compile(r"[\d]+(?:\.\d+)?")


def clean_text(text: str) -> str:
    """Lowercase and strip punctuation from text.

    Args:
        text: Raw product title or description.

    Returns:
        Cleaned lowercase string with punctuation removed.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_price(raw_str: str) -> float:
    """Parse a price string with currency symbols into a float.

    Handles ₹, $, £, €, ¥ and comma-separated numbers.

    Args:
        raw_str: Raw price string, e.g. "₹1,299", "$49.99", "£35".

    Returns:
        Numeric price as float, or 0.0 if parsing fails.
    """
    if not raw_str:
        return 0.0
    try:
        # Remove currency symbols, spaces, and commas
        cleaned = _CURRENCY_PATTERN.sub("", str(raw_str))
        match = _PRICE_PATTERN.search(cleaned)
        if match:
            return float(match.group())
    except Exception as e:
        logger.warning("Failed to parse price from '%s': %s", raw_str, e)
    return 0.0
