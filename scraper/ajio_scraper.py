"""Ajio product scraper using camoufox (bypasses Cloudflare bot detection)."""

import asyncio
import logging
import sys
from urllib.parse import quote_plus

from scraper._base import random_delay, auto_scroll, dismiss_popups, first_text, first_attr
from utils.text_utils import extract_price

logger = logging.getLogger(__name__)

PLATFORM = "Ajio"
# quote_plus handles "green hoodie" → "green+hoodie" correctly
BASE_URL = "https://www.ajio.com/search/?text={category}"

CARD_SELECTOR  = "div.item"
BRAND_SELECTOR = "div.brand"
NAME_SELECTOR  = "div.nameCls"
PRICE_SELECTOR = "span.price strong"
LINK_SELECTOR  = "a"
IMG_SELECTOR   = "img"


async def scrape_products(category: str, limit: int = 50) -> list[dict]:
    """Scrape Ajio products for a given category.

    Args:
        category: Clothing category to search (e.g. "hoodies", "green hoodie").
        limit: Maximum number of products to return.

    Returns:
        List of product record dicts, or [] on failure.
    """
    results: list[dict] = []
    try:
        from camoufox.async_api import AsyncCamoufox

        async with AsyncCamoufox(
            headless=True,
            geoip=True,
            humanize=True,
            os="windows",
        ) as browser:
            page = await browser.new_page()

            # URL-encode so "green hoodie" → "green+hoodie", not a broken URL
            encoded = quote_plus(category)
            url = BASE_URL.format(category=encoded)
            logger.info("Navigating to %s", url)

            await page.goto(url, timeout=60_000, wait_until="domcontentloaded")
            await asyncio.sleep(3)
            await dismiss_popups(page)

            try:
                await page.wait_for_selector(CARD_SELECTOR, timeout=25_000)
            except Exception:
                html = await page.content()
                with open("ajio_debug.html", "w", encoding="utf-8") as f:
                    f.write(html)
                logger.error(
                    "Ajio: no cards found for '%s'. Saved debug HTML to ajio_debug.html.",
                    category,
                )
                return []

            await auto_scroll(page)

            cards = await page.query_selector_all(CARD_SELECTOR)
            logger.info("Ajio: found %d raw cards", len(cards))

            for card in cards:
                if len(results) >= limit:
                    break
                try:
                    brand_el = await card.query_selector(BRAND_SELECTOR)
                    name_el  = await card.query_selector(NAME_SELECTOR)
                    price_el = await card.query_selector(PRICE_SELECTOR)
                    link_el  = await card.query_selector(LINK_SELECTOR)
                    img_el   = await card.query_selector(IMG_SELECTOR)

                    brand     = (await brand_el.text_content() or "").strip() if brand_el else "Unknown"
                    name      = (await name_el.text_content()  or "").strip() if name_el  else None
                    price_raw = (await price_el.text_content() or "").strip() if price_el else None
                    href      = await link_el.get_attribute("href") if link_el else None
                    # Ajio lazy-loads images — check both src and data-src
                    src = None
                    if img_el:
                        src = await img_el.get_attribute("src") or await img_el.get_attribute("data-src")

                    if not name or not href:
                        continue

                    if href.startswith("/"):
                        href = "https://www.ajio.com" + href
                    if src and src.startswith("//"):
                        src = "https:" + src

                    results.append({
                        "product_title": f"{brand} {name}".strip() if brand != "Unknown" else name,
                        "brand":         brand,
                        "price":         extract_price(price_raw),
                        "product_url":   href,
                        "image_url":     src or "",
                        "category":      category,
                        "platform":      PLATFORM,
                    })
                except Exception as exc:
                    logger.debug("Ajio card parse error: %s", exc)

            await random_delay()

    except Exception as exc:
        logger.warning("Ajio scraper failed for category '%s': %s", category, exc)
        return []

    logger.info("Ajio: scraped %d / %d for '%s'", len(results), limit, category)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cat = sys.argv[1] if len(sys.argv) > 1 else "hoodies"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    products = asyncio.run(scrape_products(cat, limit=lim))
    for p in products:
        print(p)