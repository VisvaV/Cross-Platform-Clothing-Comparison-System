"""Shared utilities for all camoufox-based scrapers."""

import asyncio
import logging
import random

logger = logging.getLogger(__name__)

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


async def random_delay(min_s: float = 1.0, max_s: float = 3.0) -> None:
    """Sleep for a random duration to reduce bot detection."""
    await asyncio.sleep(random.uniform(min_s, max_s))


async def auto_scroll(page, pause: float = 1.0, max_scrolls: int = 15) -> None:
    """Scroll the page incrementally until no new content loads or max_scrolls hit.

    The original version started prev_height=0 and had no scroll cap, risking
    an infinite loop on pages that always report a growing scrollHeight.
    """
    prev_height: int = await page.evaluate("document.body.scrollHeight")
    for _ in range(max_scrolls):
        await page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
        await asyncio.sleep(pause)
        curr_height: int = await page.evaluate("document.body.scrollHeight")
        if curr_height == prev_height:
            break
        prev_height = curr_height


async def first_text(element, selectors: list[str]) -> str:
    """Try each CSS selector and return the first non-empty text found."""
    for sel in selectors:
        try:
            el = await element.query_selector(sel)
            if el:
                text = (await el.inner_text()).strip()
                if text:
                    return text
        except Exception:
            pass
    return ""


async def first_attr(element, selectors: list[str], attr: str) -> str:
    """Try each CSS selector and return the first non-empty attribute value."""
    for sel in selectors:
        try:
            el = await element.query_selector(sel)
            if el:
                val = await el.get_attribute(attr) or ""
                if val.strip():
                    return val.strip()
        except Exception:
            pass
    return ""


async def dismiss_popups(page) -> None:
    """Handle common cookie/location/login popups across platforms."""
    selectors = [
        "button[data-id='continue']",
        "button.accept",
        "#onetrust-accept-btn-handler",
        "button[aria-label='Close']",
        "button._2KpZ6l._2doB4z",   # Flipkart login dismiss
    ]
    for selector in selectors:
        try:
            btn = await page.query_selector(selector)
            if btn:
                await btn.click()
                await asyncio.sleep(1)
        except Exception:
            pass