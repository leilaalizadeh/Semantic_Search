"""Semantic search: embed products once, embed each query, rank by cosine similarity."""

import numpy as np
# A library that loads pretrained models that turn text into embeddings
from sentence_transformers import SentenceTransformer

from semantic_search.levels.level2.data import Product

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # small, free, runs locally on CPU


class SemanticSearch:
    def __init__(self, products: list[Product], model_name: str = DEFAULT_MODEL):
        self.products = products
        self.model = SentenceTransformer(model_name)
        # Embed all products once, at the start. Then each search only embeds the query
        # and compares it to these saved product vectors.
        # normalize_embeddings=True: every vector gets length 1, so dot product = cosine similarity.
        self.embeddings = self.model.encode(
            [p.text for p in products],
            normalize_embeddings=True,
            batch_size=64,  # embed 64 products at a time (faster than one by one)
            show_progress_bar=len(products) > 100,
        )
        # The result is a 2D array of shape (num_products, 384). Each row is the vector of one product.
        # 5000 products x 384 numbers x 4 bytes = about 7.7 MB of RAM, which is fine on a laptop.

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, Product]]:
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        # @ is matrix multiplication: one dot product per product.
        # The dot product of two normalized vectors is their cosine similarity, which is what we rank by.
        scores = self.embeddings @ query_vec
        best = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self.products[i]) for i in best]
