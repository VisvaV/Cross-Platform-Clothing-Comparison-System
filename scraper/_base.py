"""Shared utilities for all Playwright-based scrapers."""

import asyncio
import logging
import random

logger = logging.getLogger(__name__)

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


async def random_delay(min_s: float = 1.0, max_s: float = 3.0) -> None:
    """Sleep for a random duration to reduce bot detection."""
    await asyncio.sleep(random.uniform(min_s, max_s))


async def auto_scroll(page) -> None:
    """Scroll the page until no new content loads."""
    prev_height = 0
    while True:
        curr_height = await page.evaluate("document.body.scrollHeight")
        if curr_height == prev_height:
            break
        prev_height = curr_height
        await page.evaluate("window.scrollBy(0, window.innerHeight)")
        await asyncio.sleep(0.8)


def first_text(element, selectors: list[str]) -> str:
    """Try each CSS selector and return the first non-empty text found."""
    for sel in selectors:
        try:
            el = element.query_selector(sel)
            if el:
                text = el.inner_text().strip()
                if text:
                    return text
        except Exception:
            pass
    return ""


def first_attr(element, selectors: list[str], attr: str) -> str:
    """Try each CSS selector and return the first non-empty attribute value."""
    for sel in selectors:
        try:
            el = element.query_selector(sel)
            if el:
                val = el.get_attribute(attr) or ""
                if val.strip():
                    return val.strip()
        except Exception:
            pass
    return ""
