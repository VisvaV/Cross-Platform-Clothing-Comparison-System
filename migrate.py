import os
import glob
import re

scraper_dir = r"d:\PSG\Semester 6\Deep Learning\Package\scraper"

# 1. Update _base.py
base_path = os.path.join(scraper_dir, "_base.py")
with open(base_path, "r", encoding="utf-8") as f:
    base_code = f.read()

dismiss_func = """

async def dismiss_popups(page) -> None:
    \"\"\"Handle cookie/location popups if they appear.\"\"\"
    for selector in ["button[data-id='continue']", "button.accept", "#onetrust-accept-btn-handler"]:
        try:
            btn = await page.query_selector(selector)
            if btn:
                await btn.click()
                await asyncio.sleep(1)
        except Exception:
            pass
"""

if "dismiss_popups" not in base_code:
    base_code += dismiss_func
    with open(base_path, "w", encoding="utf-8") as f:
        f.write(base_code)


# 2. Update scrapers
playwright_block = r'''        async with async_playwright\(\) as pw:
            browser = await pw\.chromium\.launch\(headless=True\)
            context = await browser\.new_context\(user_agent=CHROME_UA\)
            page = await context\.new_page\(\)

            url = BASE_URL\.format\(category=category\)
            await page\.goto\(url, timeout=30000\)
            await page\.wait_for_load_state\("networkidle", timeout=15000\)
            await auto_scroll\(page\)'''

camoufox_block = '''        from camoufox.async_api import AsyncCamoufox
        async with AsyncCamoufox(
            headless=False,
            geoip=True,
            humanize=True,
            os="windows",
        ) as browser:
            page = await browser.new_page()

            url = BASE_URL.format(category=category)
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

            await auto_scroll(page)'''

scrapers = glob.glob(os.path.join(scraper_dir, "*_scraper.py"))

for scraper in scrapers:
    if "ajio_scraper.py" in scraper:
        continue
    
    with open(scraper, "r", encoding="utf-8") as f:
        code = f.read()
        
    if "from playwright.async_api import async_playwright" in code:
        code = code.replace("from playwright.async_api import async_playwright", "")
        
    code = re.sub(playwright_block, camoufox_block, code)
    
    with open(scraper, "w", encoding="utf-8") as f:
        f.write(code)

print("Migration completed successfully!")
