import numpy as np
import pytest

from semantic_search.levels.level3 import SemanticSearch, keyword_search, load_products
from semantic_search.levels.level3.semantic import build_index


@pytest.fixture(scope="module")
def products():
    return load_products()


@pytest.fixture(scope="module")
def engine(products):
    return SemanticSearch(products)


def test_load_products(products):
    assert len(products) == 30
    assert products[0].title


def test_keyword_search_exact_word(products):
    results = keyword_search("headphones", products)
    assert results[0][1].title.startswith("Sony")


def test_keyword_search_misses_intent(products):
    # Only the "6" in "JBL Flip 6" matches -> keyword search returns a speaker, not a toy.
    top_title = keyword_search("gift for a 6 year old", products)[0][1].title
    assert "LEGO" not in top_title


@pytest.mark.parametrize(
    "query, expected_word",
    [
        ("gift for a 6 year old", "LEGO"),
        ("charge my phone on a trip", "Power Bank"),
        ("block out noise on the airplane", "Headphones"),
    ],
)
def test_semantic_search_understands_intent(engine, query, expected_word):
    top_titles = [p.title for _, p in engine.search(query, top_k=2)]
    assert any(expected_word in t for t in top_titles)


def test_embeddings_are_cached(products, tmp_path):
    first = SemanticSearch(products, cache_dir=tmp_path)
    assert not first.loaded_from_cache          # first time: computed
    assert len(list(tmp_path.glob("*.npy"))) == 1

    second = SemanticSearch(products, cache_dir=tmp_path)
    assert second.loaded_from_cache             # second time: read from disk
    assert (first.embeddings == second.embeddings).all()


def test_cache_changes_when_products_change(products, tmp_path):
    SemanticSearch(products, cache_dir=tmp_path)
    engine = SemanticSearch(products[:10], cache_dir=tmp_path)  # different data
    assert not engine.loaded_from_cache


def test_faiss_matches_numpy(engine):
    # IndexFlatIP is exact, so it must give the same ranking as plain numpy.
    query_vec = engine.model.encode(["warm winter clothes"], normalize_embeddings=True)[0]
    numpy_best = list(np.argsort(-(engine.embeddings @ query_vec))[:5])
    _, faiss_ids = engine.index.search(query_vec.reshape(1, -1), 5)
    assert list(faiss_ids[0]) == numpy_best


def test_hnsw_finds_same_best_product_as_flat(engine):
    # On small data HNSW should find the same top result as exact search.
    hnsw = build_index(engine.embeddings, "hnsw")
    query_vecs = engine.model.encode(["warm winter clothes", "make coffee"], normalize_embeddings=True)
    _, flat_ids = engine.index.search(query_vecs, 1)
    _, hnsw_ids = hnsw.search(query_vecs, 1)
    assert (flat_ids == hnsw_ids).all()


def test_unknown_index_type_fails(engine):
    with pytest.raises(ValueError):
        build_index(engine.embeddings, "magic")
