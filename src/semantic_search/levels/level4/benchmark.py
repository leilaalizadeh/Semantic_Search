"""Benchmark: exact search (Flat) vs. approximate search (HNSW).

5,000 products is too small to see a speed difference, so we scale up:
each extra vector is a real product vector + a little random noise.
This keeps the "shape" of real data (similar products stay close together).

For each index we measure:
  - build time   : how long it takes to create the index
  - search time  : average time for one query
  - recall@10    : % of the true top 10 (from Flat) that the index also found
"""

import argparse
import time

import faiss
import numpy as np

from semantic_search.levels.level4.data import AMAZON_PRODUCTS_PATH, load_products
from semantic_search.levels.level4.semantic import SemanticSearch, build_index

QUERIES = [
    "going camping this weekend", "keep me warm when it snows", "my car is dirty",
    "I can't sleep at night", "my back hurts when I sit all day", "I want to learn music",
    "my kitchen is too messy", "gift for a 6 year old", "something to fix a leaking sink",
    "running shoes", "phone charger", "dog toy", "birthday gift for mom", "office chair",
    "garden tools", "rain jacket", "coffee mug", "led light", "yoga mat", "car phone holder",
]
K = 10


def scale_up(base: np.ndarray, size: int, noise: float, seed: int = 0) -> np.ndarray:
    """Make `size` vectors: random real vectors + small noise, normalized again."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(base), size)
    vectors = base[picks] + rng.normal(0, noise, (size, base.shape[1])).astype(np.float32)
    faiss.normalize_L2(vectors)  # length 1 again, so dot product = cosine similarity
    return np.vstack([base, vectors])


def time_search(index: faiss.Index, queries: np.ndarray, repeats: int = 5) -> tuple[float, np.ndarray]:
    """Average milliseconds per query (one query at a time, like a real user)."""
    start = time.perf_counter()
    for _ in range(repeats):
        for q in queries:
            index.search(q.reshape(1, -1), K)
    ms = (time.perf_counter() - start) / (repeats * len(queries)) * 1000
    _, ids = index.search(queries, K)
    return ms, ids


def recall(found: np.ndarray, truth: np.ndarray) -> float:
    hits = sum(len(set(f) & set(t)) for f, t in zip(found, truth))
    return hits / truth.size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=200_000, help="Total number of vectors")
    parser.add_argument("--noise", type=float, default=0.03, help="Noise added to the copied vectors")
    args = parser.parse_args()

    engine = SemanticSearch(load_products(AMAZON_PRODUCTS_PATH))  # real product vectors (from cache)
    vectors = scale_up(engine.embeddings, args.size - len(engine.embeddings), args.noise)
    queries = engine.model.encode(QUERIES, normalize_embeddings=True)
    print(f"\n{len(vectors):,} vectors x {vectors.shape[1]} dims, {len(queries)} queries, top {K}\n")

    print(f"{'index':<22}{'build':>10}{'search/query':>15}{'recall@10':>12}")
    print("-" * 59)

    start = time.perf_counter()
    flat = build_index(vectors, "flat")
    build_s = time.perf_counter() - start
    flat_ms, truth = time_search(flat, queries)
    print(f"{'Flat (exact)':<22}{build_s:>9.1f}s{flat_ms:>13.2f}ms{100:>11.0f}%")

    start = time.perf_counter()
    hnsw = build_index(vectors, "hnsw")
    build_s = time.perf_counter() - start
    for ef in [8, 16, 32, 64, 128, 256]:
        hnsw.hnsw.efSearch = ef  # same graph, just explore more or fewer candidates
        ms, ids = time_search(hnsw, queries)
        label = f"HNSW efSearch={ef}"
        build = f"{build_s:>9.1f}s" if ef == 8 else " " * 10
        print(f"{label:<22}{build}{ms:>13.2f}ms{recall(ids, truth) * 100:>11.0f}%   {flat_ms / ms:>4.0f}x faster")


if __name__ == "__main__":
    main()
