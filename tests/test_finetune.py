from semantic_search.levels.level6.shopping_queries import TEST_QUERIES, clean_question, normalize, title_query


def test_clean_question_keeps_the_product_words():
    assert clean_question("what is a hose nozzle") == "hose nozzle"
    assert clean_question("how big is the baby crib?") == "baby crib"


def test_clean_question_drops_leftover_questions_and_long_queries():
    assert clean_question("what furniture does chs have") is None             # still a question
    assert clean_question("what is the best way to clean a very dirty car") is None  # too long


def test_title_query_takes_the_product_type():
    assert title_query("Ergonomic Office Chair, Desk Chair with Lumbar Support") == "office chair"
    assert title_query("Coleman Camp Burst 4-Person Camping Tent, Umbrella-Style") == "camping tent"
    assert title_query("Set of Mugs") is None  # filler word "of"


def test_test_queries_are_recognized_with_plurals():
    # used to keep our 20 test queries (and their plurals) out of the training data
    assert normalize("Coffee Mugs") in TEST_QUERIES
    assert normalize("Office Chairs") in TEST_QUERIES
    assert normalize("water bottle") not in TEST_QUERIES  # only "stainless steel water bottle" is a test query
