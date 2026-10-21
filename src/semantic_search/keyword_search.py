"""Traditional keyword search: score = number of query words found in the product."""

import re
#regular expression to find words in a string, ignoring case and punctuation

from semantic_search.data import Product
#import class product from data.py

STOPWORDS = {"a", "an", "the", "to", "for", "of", "on", "in", "my", "and", "or", "with", "i"}


def tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}
    # words = set()
    # for w in re.findall(r"[a-z0-9]+", text.lower()):
    #    if w not in STOPWORDS:
    #      words.add(w)
    # return words

def keyword_search(query: str, products: list[Product], top_k: int = 3) -> list[tuple[float, Product]]:
    query_words = tokenize(query)
    scored = []
    for p in products:
        # & is set intersection
        score = len(query_words & tokenize(p.text))
        if score > 0:
            scored.append((float(score), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]
