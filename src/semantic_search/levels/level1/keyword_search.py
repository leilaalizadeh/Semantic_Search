"""Traditional keyword search: score = number of query words found in the product."""

# re = regular expressions, used to find the words in a text
import re

# the Product class from data.py
from semantic_search.levels.level1.data import Product

STOPWORDS = {"a", "an", "the", "to", "for", "of", "on", "in", "my", "and", "or", "with", "i"}


def tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}
    # The same line written as a normal loop:
    # words = set()
    # for w in re.findall(r"[a-z0-9]+", text.lower()):
    #    if w not in STOPWORDS:
    #      words.add(w)
    # return words


# Gives each product a score = how many words it shares with the query, then keeps the best top_k.
def keyword_search(query: str, products: list[Product], top_k: int = 3) -> list[tuple[float, Product]]:
    query_words = tokenize(query)

    # Example:
    # "gift for a 6 year old"  ->  tokenize  ->  {"gift", "6", "year", "old"}
    # "JBL Flip 6 Bluetooth..."  ->  tokenize  ->  {"jbl", "flip", "6", "bluetooth", ...}
    # score = number of shared words = 1 ("6")
    scored = []
    for p in products:
        # & = set intersection: the words that are in both sets
        score = len(query_words & tokenize(p.text))
        if score > 0:
            scored.append((float(score), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]
