"""
Ranking engine for clothing comparison search results.
"""


def rank_by_score(results: list[dict]) -> list[dict]:
    """Sort results by similarity_score descending."""
    return sorted(results, key=lambda r: r.get("similarity_score", 0.0), reverse=True)


def rank_by_price(results: list[dict]) -> list[dict]:
    """Sort results by price ascending. Products with no price go to the end."""
    return sorted(
        results,
        key=lambda r: r.get("price") if r.get("price") is not None else float("inf"),
    )


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

    Handles None prices safely — products with no price get normalized_price=0.5
    (neutral, neither cheapest nor most expensive).
    """
    if not results:
        return results

    # Only consider products that actually have a price
    valid_prices = [
        r["price"] for r in results
        if r.get("price") is not None
    ]

    if not valid_prices:
        # No prices at all — assign neutral score to everything
        for r in results:
            r["normalized_price"] = 0.5
        return results

    min_price = min(valid_prices)
    max_price = max(valid_prices)
    price_range = max_price - min_price

    for r in results:
        price = r.get("price")
        if price is None:
            # Missing price → neutral position in ranking
            r["normalized_price"] = 0.5
        elif price_range > 0:
            r["normalized_price"] = (price - min_price) / price_range
        else:
            # All prices are identical
            r["normalized_price"] = 0.0

    return results