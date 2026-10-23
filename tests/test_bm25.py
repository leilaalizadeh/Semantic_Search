from semantic_search.levels.level4.bm25 import BM25Search, tokenize
from semantic_search.levels.level4.data import Product


def product(i, title, description=""):
    return Product(id=i, title=title, description=description)


PRODUCTS = [
    product(1, "Camping Tent", "Tent for camping trips with family."),
    product(2, "Kitchen Knife", "This knife is great. This is sharp. This is steel."),
    product(3, "Garden Hose", "This hose is long. This is green."),
    product(4, "Coffee Mug", "This mug keeps coffee hot."),
]


def test_tokenize_keeps_repeats():
    assert tokenize("tent tent camping") == ["tent", "tent", "camping"]


def test_camping_query_finds_the_tent():
    results = BM25Search(PRODUCTS).search("going camping this weekend", top_k=3)
    assert results[0][1].title == "Camping Tent"


def test_common_words_have_low_idf():
    # "blue" is in 4 of 8 products, "camping" in only 1 -> "camping" is worth more.
    blue_things = [product(10 + i, "Blue thing", "blue") for i in range(4)]
    bm25 = BM25Search(blue_things + PRODUCTS)
    assert bm25.idf["blue"] < bm25.idf["camping"]


def test_no_match_returns_nothing():
    assert BM25Search(PRODUCTS).search("xylophone") == []
