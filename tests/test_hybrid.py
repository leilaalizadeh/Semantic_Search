import pytest

from semantic_search.levels.level4.bm25 import BM25Search
from semantic_search.levels.level4.data import load_products
from semantic_search.levels.level4.hybrid import HybridSearch
from semantic_search.levels.level4.semantic import SemanticSearch


@pytest.fixture(scope="module")
def hybrid():
    products = load_products()
    return HybridSearch(SemanticSearch(products), BM25Search(products))


def test_rrf_score_is_sum_of_reciprocal_ranks(hybrid):
    for r in hybrid.fuse("wireless headphones", top_k=5):
        expected = sum(1 / (hybrid.rrf_k + rank) for rank in (r.bm25_rank, r.semantic_rank) if rank)
        assert r.score == pytest.approx(expected)


def test_product_found_by_both_methods_wins(hybrid):
    top = hybrid.fuse("noise cancelling headphones", top_k=1)[0]
    assert top.bm25_rank is not None and top.semantic_rank is not None
    assert "Headphones" in top.product.title or "AirPods" in top.product.title


def test_semantic_only_results_still_appear(hybrid):
    # No product contains these words, so BM25 finds nothing; hybrid falls back to semantic.
    results = hybrid.fuse("gift for a 6 year old", top_k=3)
    assert all(r.bm25_rank is None or r.semantic_rank is not None for r in results)
    assert any("LEGO" in r.product.title for r in results)


def test_weighted_rrf_trusts_the_heavier_list():
    from semantic_search.levels.level5.data import Product
    from semantic_search.levels.level5.hybrid import rrf

    a, b = Product(id=1, title="A", description=""), Product(id=2, title="B", description="")
    bm25_list, semantic_list = [(9.0, a), (8.0, b)], [(0.9, b), (0.8, a)]  # the two lists disagree
    assert rrf(bm25_list, semantic_list)[0].score == rrf(bm25_list, semantic_list)[1].score  # equal weights: a tie
    assert rrf(bm25_list, semantic_list, semantic_weight=2.0)[0].product == b   # semantic counts more
    assert rrf(bm25_list, semantic_list, bm25_weight=2.0)[0].product == a       # BM25 counts more
