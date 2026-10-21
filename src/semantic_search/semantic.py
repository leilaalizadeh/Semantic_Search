"""Semantic search: embed products once, embed each query, rank by cosine similarity."""

import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_search.data import Product

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # small, free, runs locally on CPU


class SemanticSearch:
    def __init__(self, products: list[Product], model_name: str = DEFAULT_MODEL):
        self.products = products
        self.model = SentenceTransformer(model_name)
        # Normalized vectors -> dot product equals cosine similarity.
        self.embeddings = self.model.encode([p.text for p in products], normalize_embeddings=True)

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, Product]]:
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec
        best = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self.products[i]) for i in best]
