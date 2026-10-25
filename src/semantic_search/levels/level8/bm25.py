"""BM25: the standard keyword-ranking formula (used by Elasticsearch, Lucene, ...).

Better than counting matching words (keyword_search.py) in three ways:
  1. IDF: rare words ("camping") count much more than common words ("this").
  2. k1:  repeating a word helps, but less and less.
  3. b:   long products don't win just because they contain more words.

score(query, product) = sum over query words of
    IDF(word) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_length))

    tf         = how many times the word appears in the product
    length     = number of words in the product
    IDF(word)  = log(1 + (N - df + 0.5) / (df + 0.5))
    N          = number of products, df = number of products containing the word
"""

import math
import re
from collections import Counter, defaultdict

from semantic_search.levels.level8.data import Product
from semantic_search.levels.level8.keyword_search import STOPWORDS


# BM25 needs to know that "tent" appears 3 times in a product, so we tokenize into a list of words, not a set.
# (It also needs the length of each product, the average length, and how many products contain
# each word for IDF. Those are computed in BM25Search.__init__ below.)
def tokenize(text: str) -> list[str]:
    """Like keyword_search.tokenize, but returns a list: BM25 needs to count repeats."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class BM25Search:
    def __init__(self, products: list[Product], k1: float = 1.5, b: float = 0.75):
        self.products = products
        self.k1 = k1
        self.b = b

        # Inverted index: word -> list of (product position, how many times the word appears).
        # Built once, so a search only looks at products that contain the query words.
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for i, p in enumerate(products):
            words = tokenize(p.text)
            self.lengths.append(len(words))
            for word, tf in Counter(words).items():
                self.postings[word].append((i, tf))
        self.avg_length = sum(self.lengths) / max(len(products), 1)

        n = len(products)
        # Rare words count more than common words. The "1 +" inside the log keeps IDF above 0,
        # even for a word that is in almost every product.
        self.idf = {
            word: math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            for word, docs in self.postings.items()
        }

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, Product]]:
        scores: dict[int, float] = defaultdict(float)
        for word in set(tokenize(query)):
            idf = self.idf.get(word)
            if idf is None:  # no product contains this word
                continue
            for i, tf in self.postings[word]:
                length_norm = 1 - self.b + self.b * self.lengths[i] / self.avg_length
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)

        best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(score, self.products[i]) for i, score in best]
