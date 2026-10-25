"""Experiment 2: compare embedding models on our labeled queries.

Run:  uv run python -m semantic_search.levels.level7.compare_models

For each model: embed all products (cached after the first run), then measure
semantic search alone and hybrid search (Level 4 setting and the tuned setting).
"unlabeled" = results nobody labeled. They count as not relevant, which is unfair
to a new model that finds good products the old methods missed (pool bias).
"""

import gc
import time

import torch

from semantic_search.levels.level7.bm25 import BM25Search
from semantic_search.levels.level7.data import AMAZON_PRODUCTS_PATH, load_products
from semantic_search.levels.level7.evaluate import K, METRICS
from semantic_search.levels.level7.hybrid import rrf
from semantic_search.levels.level7.labels import load_labels, load_pool
from semantic_search.levels.level7.semantic import QUERY_PREFIXES, SemanticSearch
from semantic_search.levels.level7.train import OUTPUT_DIR, SHOPPING_OUTPUT_DIR

BGE_PREFIX = QUERY_PREFIXES["BAAI/bge-small-en-v1.5"]
MODELS = {  # name -> (query prefix, batch size)
    "all-MiniLM-L6-v2": ("", 64),
    "BAAI/bge-small-en-v1.5": (BGE_PREFIX, 64),
    # 5x bigger: with batches of 64 long product texts it filled the Mac's memory and slowed to a crawl
    "all-mpnet-base-v2": ("", 16),
    # Level 6: BGE-small after fine-tuning (train.py). It was trained with BGE's query instruction.
    str(OUTPUT_DIR): (BGE_PREFIX, 64),
    # Level 6, Step 3: BGE-small trained on shopping-style queries (shopping_queries.py)
    str(SHOPPING_OUTPUT_DIR): (BGE_PREFIX, 64),
    # Level 6, Step 4: other search models, without training
    "thenlper/gte-small": ("", 64),                  # same size as BGE-small, no query instruction
    "BAAI/bge-base-en-v1.5": (BGE_PREFIX, 16),       # 3x bigger BGE; small batches to fit in 8 GB
}
CANDIDATES = 50


def main() -> None:
    products = load_products(AMAZON_PRODUCTS_PATH)
    bm25 = BM25Search(products)
    truth = load_labels("leila")
    pool = [q for q in load_pool() if any(truth[q["query"]].values())]  # queries with at least one relevant
    bm25_lists = {q["query"]: bm25.search(q["query"], CANDIDATES) for q in pool}

    rows = []
    for model_name, (prefix, batch_size) in MODELS.items():
        start = time.perf_counter()
        semantic = SemanticSearch(products, model_name, query_prefix=prefix, batch_size=batch_size)
        load_s = time.perf_counter() - start

        settings = {
            "semantic": lambda b, s: [str(p.id) for _, p in s[:K]],
            "hybrid k=60 w=1": lambda b, s: [str(r.product.id) for r in rrf(b, s, 60, 1.0, 1.0, K)],
            "hybrid k=5 w=2": lambda b, s: [str(r.product.id) for r in rrf(b, s, 5, 1.0, 2.0, K)],
        }
        totals = {name: {m: 0.0 for m in METRICS} | {"unlabeled": 0} for name in settings}
        query_ms = 0.0
        for q in pool:
            labels = truth[q["query"]]
            relevant = {pid for pid, label in labels.items() if label == 1}
            start = time.perf_counter()
            semantic_list = semantic.search(q["query"], CANDIDATES)
            query_ms += (time.perf_counter() - start) * 1000
            for name, rank_fn in settings.items():
                ranked = rank_fn(bm25_lists[q["query"]], semantic_list)
                totals[name]["unlabeled"] += sum(pid not in labels for pid in ranked)
                for m, f in METRICS.items():
                    totals[name][m] += f(ranked, relevant)
        for name, t in totals.items():
            rows.append((model_name, name, {m: t[m] / len(pool) for m in METRICS}, t["unlabeled"] / (K * len(pool))))
        print(f"{model_name.split('/')[-1]}: ready in {load_s:.0f}s "
              f"({'loaded from cache' if semantic.loaded_from_cache else 'embedded 5000 products'}), "
              f"{query_ms / len(pool):.0f} ms per query, vectors of {semantic.embeddings.shape[1]} numbers")
        # free the model before loading the next one
        del semantic
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"\n{'model':<24}{'method':<17}" + "".join(f"{m:>9}" for m in METRICS) + f"{'unlabeled':>11}")
    for model_name, name, r, unlabeled in rows:
        print(f"{model_name.split('/')[-1]:<24}{name:<17}" + "".join(f"{r[m]:>9.3f}" for m in METRICS) + f"{unlabeled:>10.0%}")


if __name__ == "__main__":
    main()
