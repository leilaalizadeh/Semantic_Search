"""Hybrid search: combine BM25 and semantic search with Reciprocal Rank Fusion (RRF).

BM25 scores (0..20+) and semantic scores (0..1) are on different scales, so we
can't just add them. RRF only uses each product's *position* in each list:

    rrf_score = sum over lists of 1 / (rrf_k + rank)

Products that rank well in both lists end up on top.
rrf_k = 60 comes from the original RRF paper (Cormack et al., 2009); it stops
the #1 position from dominating everything.
"""

from dataclasses import dataclass

from semantic_search.levels.level4.bm25 import BM25Search
from semantic_search.levels.level4.data import Product
from semantic_search.levels.level4.semantic import SemanticSearch


@dataclass
class HybridResult:
    score: float
    product: Product
    bm25_rank: int | None      # position in the BM25 list (1 = best), None if not in it
    semantic_rank: int | None  # position in the semantic list


class HybridSearch:
    def __init__(self, semantic: SemanticSearch, bm25: BM25Search, rrf_k: int = 60, candidates: int = 50):
        self.semantic = semantic
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.candidates = candidates  # how many results to take from each method before fusing

    def fuse(self, query: str, top_k: int = 3) -> list[HybridResult]:
        bm25_list = self.bm25.search(query, self.candidates)
        semantic_list = self.semantic.search(query, self.candidates)

        results: dict[Product, HybridResult] = {}
        for rank, (_, product) in enumerate(bm25_list, start=1):
            result = results.setdefault(product, HybridResult(0.0, product, None, None))
            result.bm25_rank = rank
            result.score += 1 / (self.rrf_k + rank)
        for rank, (_, product) in enumerate(semantic_list, start=1):
            result = results.setdefault(product, HybridResult(0.0, product, None, None))
            result.semantic_rank = rank
            result.score += 1 / (self.rrf_k + rank)

        return sorted(results.values(), key=lambda r: r.score, reverse=True)[:top_k]

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, Product]]:
        """Same shape as the other search methods: (score, product) pairs."""
        return [(r.score, r.product) for r in self.fuse(query, top_k)]
