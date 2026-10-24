"""Semantic search: embed products once, embed each query, rank by cosine similarity."""

import numpy as np
# A library that loads pretrained models that turn text into embeddings
from sentence_transformers import SentenceTransformer

from semantic_search.levels.level1.data import Product

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # small, free, runs locally on CPU


class SemanticSearch:
    def __init__(self, products: list[Product], model_name: str = DEFAULT_MODEL):
        self.products = products
        self.model = SentenceTransformer(model_name)
        # Runs once, at startup: the 30 products become 30 vectors of 384 numbers.
        # Normalized vectors (length 1) -> dot product equals cosine similarity.
        self.embeddings = self.model.encode([p.text for p in products], normalize_embeddings=True)

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, Product]]:
        # Runs for every search: the query becomes one vector, then it is compared to the 30 product vectors.
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        # @ is matrix multiplication: the dot product of the query vector with each product vector,
        # so we get 30 similarity scores in one line.
        scores = self.embeddings @ query_vec
        # argsort sorts from low to high, so -scores puts the highest score first
        best = np.argsort(-scores)[:top_k]
        # the top_k products with their scores, as a list of (score, product) pairs
        return [(float(scores[i]), self.products[i]) for i in best]
