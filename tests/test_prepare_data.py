from semantic_search.data import Product
from semantic_search.prepare_data import clean, clean_text


def make_row(**overrides):
    row = {
        "parent_asin": "B000TEST",
        "title": "Trail Running Shoes",
        "description": "Grippy shoes for running on muddy mountain trails.",
        "main_category": "Sports & Outdoors",
        "price": 59.99,
        "average_rating": 4.5,
    }
    return {**row, **overrides}


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello \n\n  world  ") == "hello world"
    assert clean_text(None) == ""


def test_clean_text_shortens_at_word_boundary():
    assert clean_text("one two three four", max_chars=10) == "one two..."


def test_clean_keeps_good_rows_as_products():
    products = clean([make_row()])
    assert len(products) == 1
    product = Product(**products[0])
    assert product.category == "Sports & Outdoors"
    assert "Category: Sports & Outdoors" in product.text


def test_clean_drops_bad_rows():
    rows = [
        make_row(),
        make_row(title="TRAIL RUNNING SHOES"),          # duplicate title
        make_row(title="x"),                            # title too short
        make_row(description=None),                     # no description
        make_row(main_category="Digital Music"),        # not a physical product
    ]
    assert len(clean(rows)) == 1
