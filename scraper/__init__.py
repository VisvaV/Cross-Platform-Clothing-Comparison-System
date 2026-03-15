"""Platform scrapers package."""

from scraper.hm_scraper import scrape_products as scrape_hm
from scraper.zara_scraper import scrape_products as scrape_zara
from scraper.myntra_scraper import scrape_products as scrape_myntra
from scraper.ajio_scraper import scrape_products as scrape_ajio
from scraper.amazon_scraper import scrape_products as scrape_amazon
from scraper.flipkart_scraper import scrape_products as scrape_flipkart
from scraper.asos_scraper import scrape_products as scrape_asos
from scraper.uniqlo_scraper import scrape_products as scrape_uniqlo

__all__ = [
    "scrape_hm",
    "scrape_zara",
    "scrape_myntra",
    "scrape_ajio",
    "scrape_amazon",
    "scrape_flipkart",
    "scrape_asos",
    "scrape_uniqlo",
]
