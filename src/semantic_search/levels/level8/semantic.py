"""Semantic search: embed products once, embed each query, rank by cosine similarity."""

import hashlib
from pathlib import Path

import faiss  # library for fast vector search (by Meta)
import numpy as np
# A library that loads pretrained models that turn text into embeddings
from sentence_transformers import SentenceTransformer

from semantic_search.levels.level8.data import DATA_DIR, Product

# Level 6 result: gte-small beat BGE-small on 11 of 18 labeled queries (nDCG@10 0.734 vs 0.674),
# with the same size and speed. It needs no query instruction.
DEFAULT_MODEL = "thenlper/gte-small"
# Instructions some models were trained with. They go in front of queries only, not products.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
CACHE_DIR = DATA_DIR / "cache"


def cache_path(texts: list[str], model_name: str, cache_dir: Path = CACHE_DIR) -> Path:
    """File name = fingerprint (hash) of the model + all product texts.

    If the products or the model change, the fingerprint changes,
    so we never load embeddings that don't match the data.
    """
    fingerprint = hashlib.sha256((model_name + "\n" + "\n".join(texts)).encode()).hexdigest()[:16]
    return cache_dir / f"embeddings_{fingerprint}.npy"


def build_index(embeddings: np.ndarray, index_type: str = "flat", ef_search: int = 64) -> faiss.Index:
    """Put the vectors in a FAISS index.

    flat: compare the query with every vector -> exact, but slow for millions of vectors.
    hnsw: graph of neighbours, jump toward the query -> approximate, but very fast.
    """
    dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    if index_type == "flat":
        # "Flat" = no shortcuts, "IP" = inner (dot) product.
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "hnsw":
        # 32 = how many neighbours each vector is linked to in the graph.
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = ef_search  # how many candidates to explore: higher = more accurate, slower
    else:
        raise ValueError(f"Unknown index type: {index_type!r} (use 'flat' or 'hnsw')")
    index.add(embeddings)
    return index


class SemanticSearch:
    def __init__(
        self,
        products: list[Product],
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path = CACHE_DIR,
        index_type: str = "flat",
        query_prefix: str | None = None,  # None = use the model's known instruction, if any
        batch_size: int = 64,  # smaller = slower, but needs less memory (big models)
    ):
        self.products = products
        self.model = SentenceTransformer(model_name)
        # Some models (like BGE) were trained with an instruction in front of each query.
        # Only queries get it, not products.
        self.query_prefix = QUERY_PREFIXES.get(model_name, "") if query_prefix is None else query_prefix

        texts = [p.text for p in products]
        path = cache_path(texts, model_name, cache_dir)
        self.loaded_from_cache = path.exists()

        if self.loaded_from_cache:
            self.embeddings = np.load(path)  # fast: just read the numbers from disk
        else:
            # Normalized vectors -> dot product equals cosine similarity.
            self.embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=batch_size,  # embed 64 products at a time (faster than one by one)
                show_progress_bar=len(products) > 100,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.embeddings)  # save for next time

        self.index = build_index(self.embeddings, index_type)

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, Product]]:
        query_vecs = self.model.encode([self.query_prefix + query], normalize_embeddings=True)  # shape (1, 384)
        # FAISS searches many queries at once, so it returns one row per query:
        # scores[0] = top_k similarity scores, ids[0] = top_k product positions.
        scores, ids = self.index.search(query_vecs, top_k)
        # ids is -1 when there are fewer than top_k products.
        return [(float(s), self.products[i]) for s, i in zip(scores[0], ids[0]) if i != -1]

    # Before FAISS (Step 1), search was done by hand with numpy:
    #   scores = self.embeddings @ query_vec      # @ is matrix multiplication
    #   best = np.argsort(-scores)[:top_k]
