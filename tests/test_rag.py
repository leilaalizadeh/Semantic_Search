from semantic_search.levels.level7.data import Product
from semantic_search.levels.level7.rag import NOTHING_FOUND, build_messages, cited_numbers
from semantic_search.levels.level7.rag_eval import made_up_mentions, recommended_positions
from semantic_search.levels.level7.rag_service import to_json

PRODUCTS = [
    Product(id=1, title="Coleman Camp Burst 4-Person Camping Tent", description="Pop-up tent", price=129.99),
    Product(id=2, title="CORE Sleeping Bags for Adults", description="Warm sleeping bag"),
    Product(id=3, title="Qistubay Silver Dog Statue", description="Metal dog sculpture"),
]


def test_prompt_contains_numbered_products_and_question():
    user = build_messages("going camping", PRODUCTS)[1]["content"]
    assert "[1] Coleman Camp Burst 4-Person Camping Tent" in user and "$129.99" in user
    assert "[3] Qistubay Silver Dog Statue" in user
    assert "going camping" in user


def test_cited_numbers():
    assert cited_numbers("Take the tent [1] and bags [2], or [1, 3].") == [1, 2, 3]
    assert cited_numbers("No citations here.") == []


def test_recommended_by_number_or_by_name():
    answer = "I recommend [1] and the [CORE Sleeping Bags for Adults]."
    assert recommended_positions(answer, PRODUCTS) == [1, 2]


def test_made_up_mentions():
    assert made_up_mentions("Take [7] and [Patagonia Jacket].", PRODUCTS) == ["7", "Patagonia Jacket"]
    assert made_up_mentions("Take [2] and [Coleman Camp Burst 4-Person].", PRODUCTS) == []


def test_service_numbers_only_kept_products():
    # the check removed product 2, so the answer cites the tent as [1] and the statue as [2]
    result = {"searched": PRODUCTS, "judged": [True, False, True], "products": [PRODUCTS[0], PRODUCTS[2]],
              "answer": NOTHING_FOUND, "cited": [], "search_ms": 1.0, "llm_s": 2.0}
    reply = to_json(result)
    assert [p["number"] for p in reply["products"]] == [1, None, 2]
    assert [p["kept"] for p in reply["products"]] == [True, False, True]
