"""Flipkart Fashion product scraper using Playwright."""

import logging
from playwright.async_api import async_playwright
from utils.text_utils import extract_price
from scraper._base import CHROME_UA, random_delay, auto_scroll, first_text, first_attr

logger = logging.getLogger(__name__)

PLATFORM = "Flipkart"
BASE_URL = "https://www.flipkart.com/search?q={category}+clothing&otracker=search"


async def scrape_products(category: str, limit: int = 50) -> list[dict]:
    """Scrape Flipkart Fashion products for a given category.

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

            # Dismiss login popup if present
            try:
                await page.goto(BASE_URL.format(category=category), timeout=30000)
                await page.wait_for_load_state("networkidle", timeout=15000)
                close_btn = page.locator("button._2KpZ6l._2doB4z")
                if await close_btn.count() > 0:
                    await close_btn.first.click()
            except Exception:
                pass

            await auto_scroll(page)

            cards = await page.query_selector_all(
                "div._1AtVbE, div[class*='_1AtVbE'], div._13oc-S"
            )

            for card in cards:
                if len(results) >= limit:
                    break
                try:
                    title = first_text(card, [
                        "div._4rR01T", "a[class*='IRpwTa']",
                        "[class*='product-title']",
                        "a[title]", "div[class*='_4rR01T']",
                    ])
                    price_raw = first_text(card, [
                        "div._30jeq3", "[class*='_30jeq3']",
                        "[class*='price']",
                    ])
                    brand = first_text(card, [
                        "div._2WkVRV", "[class*='_2WkVRV']",
                        "[class*='brand']",
                    ])
                    product_url = first_attr(card, [
                        "a[class*='IRpwTa']", "a[href*='/p/']", "a",
                    ], "href")
                    image_url = first_attr(card, [
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
                        "brand": brand or "Unknown",
                        "price": extract_price(price_raw),
                        "product_url": product_url,
                        "image_url": image_url,
                        "category": category,
                        "platform": PLATFORM,
                    })
                except Exception as e:
                    logger.debug("Error parsing Flipkart card: %s", e)

            await random_delay()
            await browser.close()
    except Exception as e:
        logger.warning("Flipkart scraper failed for category '%s': %s", category, e)
        return []

    logger.info("Flipkart: scraped %d products for '%s'", len(results), category)
    return results


if __name__ == "__main__":
    import asyncio
    import sys
    cat = sys.argv[1] if len(sys.argv) > 1 else "hoodies"
    products = asyncio.run(scrape_products(cat, limit=5))
    for p in products:
        print(p)
