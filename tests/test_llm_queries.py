from semantic_search.levels.level8.generate_llm_queries import parse_queries


def test_parse_removes_numbers_labels_and_quotes():
    text = '1. Camping tent\n2. Need: going camping with kids\n3) "waterproof tent for 4"'
    assert parse_queries(text) == ["camping tent", "going camping with kids", "waterproof tent for 4"]


def test_parse_drops_too_short_or_long_lines_and_repeats():
    text = "tent\n- camping tent\n- camping tent\n- a very very long search that nobody would ever type in a shop"
    assert parse_queries(text) == ["camping tent"]
