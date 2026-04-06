"""Myntra product scraper using camoufox."""

import asyncio
import logging
import sys
from urllib.parse import quote_plus

from utils.text_utils import extract_price
from scraper._base import random_delay, auto_scroll, dismiss_popups, first_text, first_attr

logger = logging.getLogger(__name__)

PLATFORM = "Myntra"

# Myntra uses specific URL path slugs that don't always match the plain category
# name.  Multi-word or unrecognised categories fall back to the search endpoint.
_SLUG_MAP = {
    "hoodies":   "hoodies-sweatshirts",
    "t-shirts":  "tshirts",
    "tshirts":   "tshirts",
    "jeans":     "jeans",
    "dresses":   "dresses",
    "kurtas":    "kurtas",
    "shirts":    "shirts",
    "trousers":  "trousers",
    "tops":      "tops",
    "jackets":   "jackets",
    "sweaters":  "sweaters",
    "skirts":    "skirts",
    "shorts":    "shorts",
    "sarees":    "sarees",
    "leggings":  "leggings",
}

BASE_URL_PATH   = "https://www.myntra.com/{slug}"
BASE_URL_SEARCH = "https://www.myntra.com/search?rawQuery={query}"


def _build_url(category: str) -> str:
    """
    Choose the right Myntra URL:
    - Known single-word category → path style using slug map
    - Multi-word query           → search style
    - Unknown single word        → try path style directly
    """
    key = category.strip().lower()
    if " " in key:
        return BASE_URL_SEARCH.format(query=quote_plus(category))
    slug = _SLUG_MAP.get(key, key)   # fall back to the raw word if not in map
    return BASE_URL_PATH.format(slug=slug)


async def scrape_products(category: str, limit: int = 50) -> list[dict]:
    """Scrape Myntra products for a given category.

    Args:
        category: Clothing category to search (e.g. "hoodies", "green hoodie").
        limit: Maximum number of products to return.

    Returns:
        List of product record dicts, or [] on failure.
    """
    results = []
    try:
        from camoufox.async_api import AsyncCamoufox

        async with AsyncCamoufox(
            headless=False,
            geoip=True,
            humanize=True,
            os="windows",
        ) as browser:
            page = await browser.new_page()

            url = _build_url(category)
            logger.info("Navigating to %s", url)

            await page.goto(url, timeout=60_000, wait_until="domcontentloaded")
            await asyncio.sleep(3)
            await dismiss_popups(page)

            try:
                await page.wait_for_selector(
                    "li.product-base, li[class*='product-base']", timeout=20_000
                )
            except Exception:
                logger.error("Myntra: no cards found for '%s'.", category)
                return []

            await auto_scroll(page)

            cards = await page.query_selector_all(
                "li.product-base, li[class*='product-base'], div[class*='product-base']"
            )

            for card in cards:
                if len(results) >= limit:
                    break
                try:
                    brand = await first_text(card, [
                        "h3.product-brand", "[class*='product-brand']",
                        "h3", "[class*='brand']",
                    ])
                    title = await first_text(card, [
                        "h4.product-product", "[class*='product-product']",
                        "h4", "[class*='product-name']",
                    ])
                    price_raw = await first_text(card, [
                        "span.product-discountedPrice",
                        "[class*='discountedPrice']",
                        "[class*='price']",
                        "span[class*='Price']",
                    ])
                    product_url = await first_attr(card, ["a"], "href")
                    image_url   = await first_attr(card, [
                        "img[class*='img']", "img",
                    ], "src")

                    if not title or not product_url:
                        continue

                    if product_url.startswith("/"):
                        product_url = "https://www.myntra.com" + product_url
                    if image_url.startswith("//"):
                        image_url = "https:" + image_url

                    results.append({
                        "product_title": f"{brand} {title}".strip() if brand else title,
                        "brand":         brand or "Unknown",
                        "price":         extract_price(price_raw),
                        "product_url":   product_url,
                        "image_url":     image_url,
                        "category":      category,
                        "platform":      PLATFORM,
                    })
                except Exception as exc:
                    logger.debug("Myntra card parse error: %s", exc)

        await random_delay()

    except Exception as exc:
        logger.warning("Myntra scraper failed for category '%s': %s", category, exc)
        return []

    logger.info("Myntra: scraped %d products for '%s'", len(results), category)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cat = sys.argv[1] if len(sys.argv) > 1 else "hoodies"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    products = asyncio.run(scrape_products(cat, limit=lim))
    for p in products:
        print(p)