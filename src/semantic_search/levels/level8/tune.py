"""Experiment 1: tune hybrid search (weights and rrf_k) and measure each setting.

Run:  uv run python -m semantic_search.levels.level8.tune

Each query is searched once (BM25 top 50 + semantic top 50); every setting is then
just a different way of fusing those two lists, so trying many settings is fast.

Careful: we tune and measure on the same 20 labeled queries, so the best setting
looks a little better than it really is. Prefer settings that are good in a whole
area of the table, not one lucky cell.
"""

from semantic_search.levels.level8.bm25 import BM25Search
from semantic_search.levels.level8.data import AMAZON_PRODUCTS_PATH, load_products
from semantic_search.levels.level8.evaluate import K, METRICS, ndcg_at
from semantic_search.levels.level8.hybrid import rrf
from semantic_search.levels.level8.labels import load_labels, load_pool
from semantic_search.levels.level8.semantic import SemanticSearch

CANDIDATES = 50
RRF_KS = [5, 20, 60, 100]
SEMANTIC_WEIGHTS = [1.0, 1.5, 2.0, 3.0, 5.0]


def main() -> None:
    products = load_products(AMAZON_PRODUCTS_PATH)
    # Experiment 1 tuned hybrid with the original model, before Level 5 switched to BGE-small
    semantic, bm25 = SemanticSearch(products, "all-MiniLM-L6-v2"), BM25Search(products)
    truth = load_labels("leila")

    queries = []  # (relevant ids, labeled ids, bm25 list, semantic list)
    for q in load_pool():
        relevant = {pid for pid, label in truth[q["query"]].items() if label == 1}
        if relevant:
            queries.append((relevant, set(truth[q["query"]]), bm25.search(q["query"], CANDIDATES),
                            semantic.search(q["query"], CANDIDATES)))

    def evaluate(rank_fn) -> dict:
        totals = {m: 0.0 for m in METRICS}
        unlabeled = 0
        for relevant, labeled, bm25_list, semantic_list in queries:
            ranked = rank_fn(bm25_list, semantic_list)
            unlabeled += sum(pid not in labeled for pid in ranked)
            for m, f in METRICS.items():
                totals[m] += f(ranked, relevant)
        result = {m: v / len(queries) for m, v in totals.items()}
        result["unlabeled"] = unlabeled / (K * len(queries))  # results we never labeled count as not relevant
        return result

    ids = lambda pairs: [str(p.id) for _, p in pairs[:K]]
    baselines = {
        "bm25 only": evaluate(lambda b, s: ids(b)),
        "semantic only": evaluate(lambda b, s: ids(s)),
    }
    grid = {}
    for k in RRF_KS:
        for w in SEMANTIC_WEIGHTS:
            grid[k, w] = evaluate(lambda b, s, k=k, w=w: [str(r.product.id) for r in rrf(b, s, k, 1.0, w, K)])

    print(f"\nnDCG@10 of hybrid ({len(queries)} queries). Rows: rrf_k, columns: semantic weight (BM25 weight = 1)")
    print(f"{'rrf_k':>8}" + "".join(f"{w:>8}" for w in SEMANTIC_WEIGHTS))
    for k in RRF_KS:
        print(f"{k:>8}" + "".join(f"{grid[k, w]['nDCG@10']:>8.3f}" for w in SEMANTIC_WEIGHTS))

    print(f"\n{'setting':<28}" + "".join(f"{m:>9}" for m in METRICS) + f"{'unlabeled':>11}")
    rows = list(baselines.items()) + [("hybrid k=60 w=1 (Level 4)", grid[60, 1.0])]
    best = max(grid, key=lambda key: grid[key]["nDCG@10"])
    rows.append((f"hybrid k={best[0]} w={best[1]} (best)", grid[best]))
    for name, r in rows:
        print(f"{name:<28}" + "".join(f"{r[m]:>9.3f}" for m in METRICS) + f"{r['unlabeled']:>10.0%}")


if __name__ == "__main__":
    main()
