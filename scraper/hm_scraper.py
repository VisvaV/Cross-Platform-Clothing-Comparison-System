"""H&M product scraper using Playwright."""

import logging
from playwright.async_api import async_playwright
from utils.text_utils import extract_price
from scraper._base import CHROME_UA, random_delay, auto_scroll, first_text, first_attr

logger = logging.getLogger(__name__)

PLATFORM = "H&M"
BASE_URL = "https://www2.hm.com/en_in/search-results.html?q={category}"


async def scrape_products(category: str, limit: int = 50) -> list[dict]:
    """Scrape H&M products for a given category.

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
                "li.product-item, article.product-item, li[class*='product']"
            )

            for card in cards:
                if len(results) >= limit:
                    break
                try:
                    title = first_text(card, [
                        "h3.item-heading a", "h2.item-heading", ".item-heading",
                        "[class*='product-name']", "[class*='title']",
                    ])
                    price_raw = first_text(card, [
                        ".item-price .price-value", ".price-value",
                        "[class*='price']", "span[class*='Price']",
                    ])
                    brand = "H&M"
                    product_url = first_attr(card, [
                        "h3.item-heading a", "a[class*='product']", "a",
                    ], "href")
                    image_url = first_attr(card, [
                        "img[class*='product']", "img",
                    ], "src")

                    if not title or not product_url:
                        continue

                    if product_url.startswith("/"):
                        product_url = "https://www2.hm.com" + product_url
                    if image_url.startswith("//"):
                        image_url = "https:" + image_url

                    results.append({
                        "product_title": title,
                        "brand": brand,
                        "price": extract_price(price_raw),
                        "product_url": product_url,
                        "image_url": image_url,
                        "category": category,
                        "platform": PLATFORM,
                    })
                except Exception as e:
                    logger.debug("Error parsing H&M card: %s", e)

            await random_delay()
            await browser.close()
    except Exception as e:
        logger.warning("H&M scraper failed for category '%s': %s", category, e)
        return []

    logger.info("H&M: scraped %d products for '%s'", len(results), category)
    return results


if __name__ == "__main__":
    import asyncio
    import sys
    cat = sys.argv[1] if len(sys.argv) > 1 else "hoodies"
    products = asyncio.run(scrape_products(cat, limit=5))
    for p in products:
        print(p)
