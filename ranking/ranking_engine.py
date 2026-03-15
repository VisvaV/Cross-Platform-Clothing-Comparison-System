"""
Ranking engine for clothing comparison search results.
"""


def rank_by_score(results: list[dict]) -> list[dict]:
    """Sort results by similarity_score descending."""
    return sorted(results, key=lambda r: r.get("similarity_score", 0.0), reverse=True)


def rank_by_price(results: list[dict]) -> list[dict]:
    """Sort results by price ascending."""
    return sorted(results, key=lambda r: r.get("price", 0.0))


def compute_hybrid_score(
    image_sim: float, text_sim: float, normalized_price: float
) -> float:
    """
    Compute weighted hybrid score combining image similarity, text similarity,
    and inverse price score.

    Formula: 0.65 * image_sim + 0.25 * text_sim + 0.10 * (1 - normalized_price)
    price_score is 1.0 for cheapest, 0.0 for most expensive.
    """
    price_score = 1.0 - normalized_price
    return 0.65 * image_sim + 0.25 * text_sim + 0.10 * price_score


def normalize_prices(results: list[dict]) -> list[dict]:
    """
    Min-max normalize the price field across the result set.
    Adds a normalized_price key to each result dict.
    """
    prices = [r.get("price", 0.0) for r in results]
    min_price = min(prices, default=0.0)
    max_price = max(prices, default=0.0)
    price_range = max_price - min_price

    for r in results:
        if price_range > 0:
            r["normalized_price"] = (r.get("price", 0.0) - min_price) / price_range
        else:
            r["normalized_price"] = 0.0

    return results
