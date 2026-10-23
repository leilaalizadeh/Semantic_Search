import pytest
from fastapi.testclient import TestClient

from semantic_search.web import app

client = TestClient(app)


def test_page_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "Semantic Search" in response.text


def test_level_1_search():
    response = client.get("/api/search", params={"q": "gift for a 6 year old", "level": 1, "k": 3})
    assert response.status_code == 200
    data = response.json()
    assert [c["name"] for c in data["columns"]] == ["Keyword", "Semantic"]
    semantic = data["columns"][1]["results"]
    assert len(semantic) == 3
    assert "LEGO" in semantic[0]["title"]


def test_empty_query_is_rejected():
    assert client.get("/api/search", params={"q": "   "}).status_code == 400


def test_unknown_level_is_rejected():
    assert client.get("/api/search", params={"q": "shoes", "level": 9}).status_code == 422


def test_level_4_columns():
    from semantic_search.levels.level4.data import AMAZON_PRODUCTS_PATH

    if not AMAZON_PRODUCTS_PATH.exists():
        pytest.skip("Amazon data not downloaded (run `uv run prepare-data`)")
    data = client.get("/api/search", params={"q": "camping", "level": 4, "k": 3}).json()
    assert [c["kind"] for c in data["columns"]] == ["bm25", "semantic", "hybrid"]
    assert "bm25_rank" in data["columns"][2]["results"][0]
