import pytest

from semantic_search import SemanticSearch, keyword_search, load_products


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
