"""Amazon Fashion product scraper using camoufox."""

import asyncio
import logging
from urllib.parse import quote_plus

from utils.text_utils import extract_price
from scraper._base import CHROME_UA, random_delay, auto_scroll, first_text, first_attr

logger = logging.getLogger(__name__)

PLATFORM = "Amazon"
BASE_URL = "https://www.amazon.in/s?k={category}+clothing&i=fashion"


async def scrape_products(category: str, limit: int = 50) -> list[dict]:
    """Scrape Amazon Fashion products for a given category.

    Args:
        category: Clothing category to search (e.g. "hoodies", "dresses").
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

            encoded = quote_plus(category)
            url = BASE_URL.format(category=encoded)
            logger.info("Navigating to %s", url)

            await page.goto(url, timeout=60000, wait_until="domcontentloaded")
            import asyncio
            await asyncio.sleep(3)

            from scraper._base import dismiss_popups
            await dismiss_popups(page)
            
            try:
                await page.wait_for_selector("div, a, li, img", timeout=25000)
            except Exception:
                logger.error("Cards not found on %s. Adjust selectors or proxies.", PLATFORM)
                return []

            await auto_scroll(page)

            cards = await page.query_selector_all(
                "div[data-component-type='s-search-result']"
            )

            for card in cards:
                if len(results) >= limit:
                    break
                try:
                    title = await first_text(card, [
                        "h2 a span", "h2 span",
                        "[class*='product-title']",
                        "span[class*='a-text-normal']",
                    ])
                    price_raw = await first_text(card, [
                        "span.a-price span.a-offscreen",
                        "span.a-price-whole",
                        "[class*='a-price']",
                    ])
                    brand = await first_text(card, [
                        "span.a-size-base-plus",
                        "[class*='brand']",
                        "h5",
                    ])
                    product_url = await first_attr(card, [
                        "h2 a", "a[class*='a-link-normal']",
                    ], "href")
                    image_url = await first_attr(card, [
                        "img.s-image", "img",
                    ], "src")

                    if not title or not product_url:
                        continue

                    if product_url.startswith("/"):
                        product_url = "https://www.amazon.in" + product_url
                    if image_url.startswith("//"):
                        image_url = "https:" + image_url

                    results.append({
                        "product_title": title,
                        "brand": brand or "Unknown",
                        "price": extract_price(price_raw),
                        "product_url": product_url,
                        "image_url": image_url,
                        "category": category,
                        "platform": PLATFORM,
                    })
                except Exception as e:
                    logger.debug("Error parsing Amazon card: %s", e)

            await random_delay()
            await browser.close()
    except Exception as e:
        logger.warning("Amazon scraper failed for category '%s': %s", category, e)
        return []

    logger.info("Amazon: scraped %d products for '%s'", len(results), category)
    return results


if __name__ == "__main__":
    import asyncio
    import sys
    cat = sys.argv[1] if len(sys.argv) > 1 else "hoodies"
    products = asyncio.run(scrape_products(cat, limit=5))
    for p in products:
        print(p)