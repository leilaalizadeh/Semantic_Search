"""Build the pool of products to label (pooling).

For each test query, take the top 10 from every method (keyword, BM25, semantic,
hybrid) and merge them. Only these products get labeled; a product that no method
puts in its top 10 counts as not relevant.

Run:  uv run python -m semantic_search.levels.level7.pool                 (build a new pool)
      uv run python -m semantic_search.levels.level7.pool --add-models    (add the bigger models' results)
Writes data/eval/pool.json

--add-models keeps the pool as it is and only adds products that the new models
(BGE-small, MPNet) put in their top 10. Without this, the new models are judged
unfairly: their new results were never labeled and count as not relevant (pool bias).
"""

import argparse
import json

from semantic_search.levels.level7.bm25 import BM25Search
from semantic_search.levels.level7.data import AMAZON_PRODUCTS_PATH, load_products
from semantic_search.levels.level7.hybrid import HybridSearch
from semantic_search.levels.level7.keyword_search import keyword_search
from semantic_search.levels.level7.labels import EVAL_DIR, POOL_PATH
from semantic_search.levels.level7.queries import QUERIES
from semantic_search.levels.level7.semantic import SemanticSearch

POOL_DEPTH = 10  # top 10 from each method

# Build the search methods.
# The pool was first built with this model (before Level 5 switched the default to BGE-small).
POOL_MODEL = "all-MiniLM-L6-v2"


def build_methods(products, depth: int = POOL_DEPTH, model_name: str | None = None) -> dict:
    """The 4 search methods we compare. Each one takes a query and returns (score, product) pairs.

    model_name=None uses the Level 5 default model (BGE-small).
    """
    semantic = SemanticSearch(products) if model_name is None else SemanticSearch(products, model_name)
    bm25 = BM25Search(products)
    hybrid = HybridSearch(semantic, bm25)
    return {
        "keyword": lambda q: keyword_search(q, products, depth),
        "bm25": lambda q: bm25.search(q, depth),
        "semantic": lambda q: semantic.search(q, depth),
        "hybrid": lambda q: hybrid.search(q, depth),
    }


def build_new_model_methods(products, depth: int = POOL_DEPTH) -> dict:
    """The bigger models from Experiment 2: semantic alone and both hybrid settings."""
    from semantic_search.levels.level7.compare_models import MODELS
    from semantic_search.levels.level7.hybrid import rrf

    bm25 = BM25Search(products)
    methods = {}
    for model_name, (prefix, batch_size) in MODELS.items():
        if model_name == "all-MiniLM-L6-v2":  # already in the pool
            continue
        semantic = SemanticSearch(products, model_name, query_prefix=prefix, batch_size=batch_size)
        short = model_name.split("/")[-1]
        methods[f"{short} semantic"] = lambda q, s=semantic: s.search(q, depth)
        for k, w in [(60, 1.0), (5, 2.0)]:
            methods[f"{short} hybrid k={k} w={w}"] = lambda q, s=semantic, k=k, w=w: [
                (r.score, r.product) for r in rrf(bm25.search(q, 50), s.search(q, 50), k, 1.0, w, depth)]
    return methods


def add_results(candidates: dict[str, dict], name: str, results) -> None:
    """Add a method's results to a query's candidates, remembering which method found each one."""
    for rank, (_, p) in enumerate(results, start=1):
        c = candidates.setdefault(str(p.id), {
            "id": str(p.id),
            "title": p.title,
            "category": p.category,
            "description": p.description[:300],
            "found_by": {},
        })
        c["found_by"][name] = rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--add-models", action="store_true", help="Add the bigger models' results to the existing pool")
    args = parser.parse_args()

    products = load_products(AMAZON_PRODUCTS_PATH)
    if args.add_models:
        with open(POOL_PATH) as f:
            old_pool = {q["query"]: q for q in json.load(f)}
        methods = build_new_model_methods(products)
    else:
        old_pool = {}
        methods = build_methods(products, model_name=POOL_MODEL)  # same as the original pool

    pool = []
    for item in QUERIES:
        # product id -> candidate (a dict keeps the first-seen order); start from the old pool if we add to it
        candidates = {c["id"]: c for c in old_pool.get(item["query"], {}).get("candidates", [])}
        before = len(candidates)
        # For each query, find the top products from each search method.
        for name, search in methods.items():
            add_results(candidates, name, search(item["query"]))
        pool.append({**item, "candidates": list(candidates.values())})
        print(f"{len(candidates) - before:>3} new products to label  <-  {item['query']}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(POOL_PATH, "w") as f:
        json.dump(pool, f, indent=1, ensure_ascii=False)
    total = sum(len(q["candidates"]) for q in pool)
    print(f"\nSaved {total} query-product pairs for {len(pool)} queries to {POOL_PATH}")


if __name__ == "__main__":
    main()
