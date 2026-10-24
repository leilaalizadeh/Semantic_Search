"""Semantic search: embed products once, embed each query, rank by cosine similarity."""

import hashlib
from pathlib import Path

# FAISS (from Meta) is a library made for one job: given a vector, find the most similar vectors, fast.
# Vector databases (Chroma, Qdrant, Pinecone) work in the same way.
import faiss
import numpy as np
# A library that loads pretrained models that turn text into embeddings
from sentence_transformers import SentenceTransformer

from semantic_search.levels.level3.data import DATA_DIR, Product

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # small, free, runs locally on CPU
CACHE_DIR = DATA_DIR / "cache"


def cache_path(texts: list[str], model_name: str, cache_dir: Path = CACHE_DIR) -> Path:
    """File name = fingerprint (hash) of the model + all product texts.

    If the products or the model change, the fingerprint changes,
    so we never load embeddings that don't match the data.
    """
    # Hashing is fast, and the result is a short string that can be used as a file name.
    fingerprint = hashlib.sha256((model_name + "\n" + "\n".join(texts)).encode()).hexdigest()[:16]
    return cache_dir / f"embeddings_{fingerprint}.npy"


# Flat gives the same results as the numpy search of Level 2, just with a standard tool.
# HNSW is the one that makes search much faster on large datasets.
def build_index(embeddings: np.ndarray, index_type: str = "flat", ef_search: int = 64) -> faiss.Index:
    """Put the vectors in a FAISS index.

    flat: compare the query with every vector -> exact, but slow for millions of vectors.
    hnsw: graph of neighbours, jump toward the query -> approximate, but very fast.
    """
    dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    if index_type == "flat":
        # "Flat" = no shortcuts, "IP" = inner (dot) product.
        # Creates an empty index for 384 numbers per vector. The vectors are added below.
        # The index is kept in RAM and FAISS is written in C++, so search is fast.
        # IP: the dot product of two normalized vectors is their cosine similarity, which is what we rank by.
        index = faiss.IndexFlatIP(dimension)
        # When searching, FAISS returns 2 arrays: scores and ids. An id is the position of a vector
        # in the index, which is also the position of the product in self.products.
        # FAISS can also search many queries at once (batching) for even more speed.
    elif index_type == "hnsw":
        # HNSW = Hierarchical Navigable Small World graph. Each vector is linked to its nearest
        # neighbours, in several layers, like highway -> main roads -> small streets.
        # A search jumps from neighbour to neighbour toward the query and checks only a small part of the products.
        # Approximate, but very fast.
        # 32 = how many neighbours each vector is linked to. More neighbours = more accurate,
        # but the index is slower to build and uses more RAM.
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = ef_search  # how many candidates to explore: higher = more accurate, slower
        # ef_search = 64 is a good balance between speed and accuracy for 5000 products.
        # For millions of products, you may need 128 or 256 to keep the same accuracy.
    else:
        raise ValueError(f"Unknown index type: {index_type!r} (use 'flat' or 'hnsw')")
    index.add(embeddings)  # add all product vectors (5000 for the Amazon data), then the index is ready
    return index


class SemanticSearch:
    def __init__(
        self,
        products: list[Product],
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path = CACHE_DIR,
        index_type: str = "flat",
    ):
        self.products = products
        self.model = SentenceTransformer(model_name)

        texts = [p.text for p in products]
        path = cache_path(texts, model_name, cache_dir)
        self.loaded_from_cache = path.exists()

        if self.loaded_from_cache:
            # Fast: just read the numbers from disk, no need to compute the embeddings again.
            # 5000 products x 384 numbers x 4 bytes = about 7.7 MB, so it's fine to keep in memory.
            self.embeddings = np.load(path)
        else:
            # Normalized vectors -> dot product equals cosine similarity.
            self.embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=64,  # embed 64 products at a time (faster than one by one)
                show_progress_bar=len(products) > 100,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.embeddings)  # save for next time

        self.index = build_index(self.embeddings, index_type)

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, Product]]:
        query_vecs = self.model.encode([query], normalize_embeddings=True)  # shape (1, 384)
        # FAISS searches many queries at once, so it returns one row per query:
        # scores[0] = top_k similarity scores, ids[0] = top_k product positions.
        scores, ids = self.index.search(query_vecs, top_k)
        # ids is -1 when there are fewer than top_k products.
        return [(float(s), self.products[i]) for s, i in zip(scores[0], ids[0]) if i != -1]

    # Before FAISS (Step 1), search was done by hand with numpy:
    #   scores = self.embeddings @ query_vec      # @ is matrix multiplication
    #   best = np.argsort(-scores)[:top_k]
