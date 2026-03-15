"""Amazon Fashion product scraper using Playwright."""

import logging
from playwright.async_api import async_playwright
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
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=CHROME_UA)
            page = await context.new_page()

            url = BASE_URL.format(category=category)
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=15000)
            await auto_scroll(page)

            cards = await page.query_selector_all(
                "div[data-component-type='s-search-result']"
            )

            for card in cards:
                if len(results) >= limit:
                    break
                try:
                    title = first_text(card, [
                        "h2 a span", "h2 span",
                        "[class*='product-title']",
                        "span[class*='a-text-normal']",
                    ])
                    price_raw = first_text(card, [
                        "span.a-price span.a-offscreen",
                        "span.a-price-whole",
                        "[class*='a-price']",
                    ])
                    brand = first_text(card, [
                        "span.a-size-base-plus",
                        "[class*='brand']",
                        "h5",
                    ])
                    product_url = first_attr(card, [
                        "h2 a", "a[class*='a-link-normal']",
                    ], "href")
                    image_url = first_attr(card, [
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
