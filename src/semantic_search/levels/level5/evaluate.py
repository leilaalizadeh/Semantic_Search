"""Level 5: compare two people's labels, then measure every search method with numbers.

Run:  uv run python -m semantic_search.levels.level5.evaluate
      uv run python -m semantic_search.levels.level5.evaluate --truth ai_model   (use the other labels as truth)

Metrics (all between 0 and 1, higher is better):
  P@5     precision: how many of the top 5 results are relevant
  R@10    recall: how many of all relevant products appear in the top 10
  MRR     mean reciprocal rank: 1 / position of the first relevant result (1st = 1, 2nd = 0.5, ...)
  nDCG@10 like precision, but a relevant result at position 1 counts more than one at position 10
"""

import argparse
import math
from collections import defaultdict

from semantic_search.levels.level5.data import AMAZON_PRODUCTS_PATH, load_products
from semantic_search.levels.level5.labels import load_labels, load_pool
from semantic_search.levels.level5.pool import build_methods

K = 10


# ---------- agreement between two people ----------
# Compare the labels of two people to see how much they agree.
# kappa = (observed agreement - expected by chance) / (1 - expected by chance)
def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Agreement corrected for luck. 1 = always agree, 0 = no better than chance."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n # how often we gave the same label
    pa, pb = sum(a) / n, sum(b) / n
    by_chance = pa * pb + (1 - pa) * (1 - pb)  # both say 1 by luck + both say 0 by luck - agreement expected by pure luck
    return 1.0 if by_chance == 1 else (observed - by_chance) / (1 - by_chance)


# ---------- search metrics for one query ----------
# ranked = product ids in the order a method returned them, relevant = set of relevant ids

def precision_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(pid in relevant for pid in ranked[:k]) / k


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(pid in relevant for pid in ranked[:k]) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for position, pid in enumerate(ranked, start=1):
        if pid in relevant:
            return 1 / position
    return 0.0


def ndcg_at(ranked: list[str], relevant: set[str], k: int) -> float:
    # each hit is worth 1 / log2(position + 1): position 1 -> 1.0, position 2 -> 0.63, position 10 -> 0.29
    dcg = sum(1 / math.log2(i + 2) for i, pid in enumerate(ranked[:k]) if pid in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))  # all relevant ones first
    return dcg / ideal


METRICS = {
    "P@5": lambda r, rel: precision_at(r, rel, 5),
    "R@10": lambda r, rel: recall_at(r, rel, K),
    "MRR": reciprocal_rank, # how high is the first good result?
    "nDCG@10": lambda r, rel: ndcg_at(r, rel, K), # the overall score: good results near the top count more
}


def print_agreement(pool: list[dict], truth: dict, other: dict, truth_name: str, other_name: str) -> None:
    a, b, disagreements = [], [], []
    print(f"\n=== Agreement: {truth_name} vs {other_name} ===")
    print(f"{'query':<36}{'agree':>7}{truth_name[:8]:>10}{other_name[:8]:>10}")
    for q in pool:
        qa, qb = [], []
        for c in q["candidates"]:
            x, y = truth.get(q["query"], {}).get(c["id"]), other.get(q["query"], {}).get(c["id"])
            if x is None or y is None:  # only compare pairs that both people labeled
                continue
            qa.append(x)
            qb.append(y)
            if x != y:
                disagreements.append((q["query"], c["title"], x, y))
        a += qa
        b += qb
        same = sum(x == y for x, y in zip(qa, qb)) / max(len(qa), 1)
        print(f"{q['query'][:35]:<36}{same:>7.0%}{sum(qa):>10}{sum(qb):>10}")
    same = sum(x == y for x, y in zip(a, b)) / len(a)
    print(f"{'ALL':<36}{same:>7.0%}{sum(a):>10}{sum(b):>10}")
    print(f"Cohen's kappa: {cohen_kappa(a, b):.2f}   (above 0.6 = good, above 0.8 = very good)")
    print(f"\n{len(disagreements)} disagreements (1 = relevant):")
    for query, title, x, y in disagreements:
        print(f"  [{query[:28]:<28}] {truth_name}={x} {other_name}={y}  {title[:70]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth", default="leila", help="Whose labels are the correct answers")
    parser.add_argument("--other", default="ai_model", help="Labels to compare with")
    args = parser.parse_args()

    pool = load_pool()
    truth, other = load_labels(args.truth), load_labels(args.other)
    if other:
        print_agreement(pool, truth, other, args.truth, args.other)

    methods = build_methods(load_products(AMAZON_PRODUCTS_PATH))
    scores = defaultdict(lambda: defaultdict(list))  # method -> metric -> one value per query
    by_type = defaultdict(lambda: defaultdict(list))  # method -> query type -> nDCG values
    skipped = []
    for q in pool:
        relevant = {pid for pid, label in truth[q["query"]].items() if label == 1}
        if not relevant:  # recall, MRR and nDCG need at least one relevant product
            skipped.append(q["query"])
            continue
        for name, search in methods.items():
            ranked = [str(p.id) for _, p in search(q["query"])]
            for metric, f in METRICS.items():
                scores[name][metric].append(f(ranked, relevant))
            by_type[name][q["type"]].append(ndcg_at(ranked, relevant, K))

    used = len(pool) - len(skipped)
    print(f"\n=== Search quality, truth = {args.truth}'s labels ({used} queries) ===")
    if skipped:
        print(f"skipped (no relevant product in the pool): {', '.join(skipped)}")
    mean = lambda xs: sum(xs) / len(xs)
    print(f"{'method':<10}" + "".join(f"{m:>9}" for m in METRICS))
    for name in methods:
        print(f"{name:<10}" + "".join(f"{mean(scores[name][m]):>9.3f}" for m in METRICS))

    types = sorted({t for name in by_type for t in by_type[name]})
    print(f"\nnDCG@10 by query type:")
    print(f"{'method':<10}" + "".join(f"{t:>9}" for t in types))
    for name in methods:
        print(f"{name:<10}" + "".join(f"{mean(by_type[name][t]):>9.3f}" if by_type[name][t] else f"{'-':>9}" for t in types))


if __name__ == "__main__":
    main()
