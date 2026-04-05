"""Flipkart Fashion product scraper using camoufox."""

import asyncio
import logging
import sys
from urllib.parse import quote_plus

from utils.text_utils import extract_price
from scraper._base import random_delay, auto_scroll, dismiss_popups, first_text, first_attr

logger = logging.getLogger(__name__)

PLATFORM = "Flipkart"
BASE_URL = "https://www.flipkart.com/search?q={category}+clothing&otracker=search"


async def scrape_products(category: str, limit: int = 50) -> list[dict]:
    """Scrape Flipkart Fashion products for a given category.

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
            headless=True,
            geoip=True,
            humanize=True,
            os="windows",
        ) as browser:
            page = await browser.new_page()

            encoded = quote_plus(category)
            url = BASE_URL.format(category=encoded)
            logger.info("Navigating to %s", url)

            await page.goto(url, timeout=60_000, wait_until="domcontentloaded")
            await asyncio.sleep(3)

            # Dismiss Flipkart login popup
            await dismiss_popups(page)
            try:
                close_btn = page.locator("button._2KpZ6l._2doB4z")
                if await close_btn.count() > 0:
                    await close_btn.first.click()
                    await asyncio.sleep(1)
            except Exception:
                pass

            try:
                await page.wait_for_selector(
                    "div._1AtVbE, div._13oc-S, div[data-id]", timeout=20_000
                )
            except Exception:
                logger.error("Flipkart: no cards found for '%s'.", category)
                return []

            await auto_scroll(page)

            cards = await page.query_selector_all(
                "div._1AtVbE, div[class*='_1AtVbE'], div._13oc-S"
            )

            for card in cards:
                if len(results) >= limit:
                    break
                try:
                    title = await first_text(card, [
                        "div._4rR01T", "a[class*='IRpwTa']",
                        "[class*='product-title']",
                        "a[title]", "div[class*='_4rR01T']",
                    ])
                    price_raw = await first_text(card, [
                        "div._30jeq3", "[class*='_30jeq3']",
                        "[class*='price']",
                    ])
                    brand = await first_text(card, [
                        "div._2WkVRV", "[class*='_2WkVRV']",
                        "[class*='brand']",
                    ])
                    product_url = await first_attr(card, [
                        "a[class*='IRpwTa']", "a[href*='/p/']", "a",
                    ], "href")
                    image_url = await first_attr(card, [
                        "img._396cs4", "img[class*='_396cs4']", "img",
                    ], "src")

                    if not title or not product_url:
                        continue

                    if product_url.startswith("/"):
                        product_url = "https://www.flipkart.com" + product_url
                    if image_url.startswith("//"):
                        image_url = "https:" + image_url

                    results.append({
                        "product_title": title,
                        "brand":         brand or "Unknown",
                        "price":         extract_price(price_raw),
                        "product_url":   product_url,
                        "image_url":     image_url,
                        "category":      category,
                        "platform":      PLATFORM,
                    })
                except Exception as exc:
                    logger.debug("Flipkart card parse error: %s", exc)

        await random_delay()

    except Exception as exc:
        logger.warning("Flipkart scraper failed for category '%s': %s", category, exc)
        return []

    logger.info("Flipkart: scraped %d products for '%s'", len(results), category)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cat = sys.argv[1] if len(sys.argv) > 1 else "hoodies"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    products = asyncio.run(scrape_products(cat, limit=lim))
    for p in products:
        print(p)