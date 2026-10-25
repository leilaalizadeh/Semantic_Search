"""Hybrid search: combine BM25 and semantic search with Reciprocal Rank Fusion (RRF).

BM25 scores (0..20+) and semantic scores (0..1) are on different scales, so we
can't just add them. RRF only uses each product's *position* in each list:

    rrf_score = sum over lists of weight / (rrf_k + rank)     (both weights are 1 by default)

Products that rank well in both lists end up on top.
rrf_k = 60 comes from the original RRF paper (Cormack et al., 2009); it stops
the #1 position from dominating everything.
"""

from dataclasses import dataclass

from semantic_search.levels.level7.bm25 import BM25Search
from semantic_search.levels.level7.data import Product
from semantic_search.levels.level7.semantic import SemanticSearch


@dataclass
class HybridResult:
    score: float
    product: Product
    bm25_rank: int | None      # position in the BM25 list (1 = best), None if not in it
    semantic_rank: int | None  # position in the semantic list


# Hybrid search: combines BM25 and semantic search with Reciprocal Rank Fusion (RRF).
# RRF only uses each product's position in each list, so the different score scales don't matter.
# Products that rank well in both lists end up on top.
def rrf(bm25_list, semantic_list, rrf_k: int = 60, bm25_weight: float = 1.0, semantic_weight: float = 1.0,
        top_k: int = 3) -> list[HybridResult]:
    """Weighted Reciprocal Rank Fusion of two result lists of (score, product) pairs.

    Each list gives a product weight / (rrf_k + rank). With both weights = 1 this is plain RRF.
    A bigger semantic_weight trusts semantic search more than BM25.
    """
    results: dict[Product, HybridResult] = {}
    for rank, (_, product) in enumerate(bm25_list, start=1):
        result = results.setdefault(product, HybridResult(0.0, product, None, None))
        result.bm25_rank = rank
        result.score += bm25_weight / (rrf_k + rank)
    for rank, (_, product) in enumerate(semantic_list, start=1):
        result = results.setdefault(product, HybridResult(0.0, product, None, None))
        result.semantic_rank = rank
        result.score += semantic_weight / (rrf_k + rank)
    return sorted(results.values(), key=lambda r: r.score, reverse=True)[:top_k]


class HybridSearch:
    def __init__(self, semantic: SemanticSearch, bm25: BM25Search, rrf_k: int = 60, candidates: int = 50,
                 bm25_weight: float = 1.0, semantic_weight: float = 1.0):
        self.semantic = semantic
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.candidates = candidates  # how many results to take from each method before fusing
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

    def fuse(self, query: str, top_k: int = 3) -> list[HybridResult]:
        bm25_list = self.bm25.search(query, self.candidates)
        semantic_list = self.semantic.search(query, self.candidates)
        return rrf(bm25_list, semantic_list, self.rrf_k, self.bm25_weight, self.semantic_weight, top_k)

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, Product]]:
        """Same shape as the other search methods: (score, product) pairs."""
        return [(r.score, r.product) for r in self.fuse(query, top_k)]
