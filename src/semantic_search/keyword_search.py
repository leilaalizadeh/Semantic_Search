"""Traditional keyword search: score = number of query words found in the product."""

import re

from semantic_search.data import Product

STOPWORDS = {"a", "an", "the", "to", "for", "of", "on", "in", "my", "and", "or", "with", "i"}


def tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def keyword_search(query: str, products: list[Product], top_k: int = 3) -> list[tuple[float, Product]]:
    query_words = tokenize(query)
    scored = []
    for p in products:
        score = len(query_words & tokenize(p.text))
        if score > 0:
            scored.append((float(score), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]
