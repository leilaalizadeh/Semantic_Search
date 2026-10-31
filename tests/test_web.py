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


def test_level_3_hnsw_does_not_crash():
    # Building HNSW after PyTorch has run used to crash the server (OpenMP conflict on macOS).
    from semantic_search.levels.level3.data import AMAZON_PRODUCTS_PATH

    if not AMAZON_PRODUCTS_PATH.exists():
        pytest.skip("Amazon data not downloaded (run `uv run prepare-data`)")
    client.get("/api/search", params={"q": "Columbia", "level": 1})  # PyTorch runs first
    data = client.get("/api/search", params={"q": "Columbia", "level": 3, "k": 3}).json()
    assert [c["kind"] for c in data["columns"]] == ["semantic", "semantic"]


def test_level_5_uses_bge_small():
    from semantic_search.levels.level5.data import AMAZON_PRODUCTS_PATH

    if not AMAZON_PRODUCTS_PATH.exists():
        pytest.skip("Amazon data not downloaded (run `uv run prepare-data`)")
    data = client.get("/api/search", params={"q": "yoga mat", "level": 5, "k": 3}).json()
    assert [c["name"] for c in data["columns"]] == ["BM25", "Semantic (BGE-small)", "Hybrid (RRF)"]


def test_level_6_compares_three_versions():
    from semantic_search.levels.level6.data import AMAZON_PRODUCTS_PATH
    from semantic_search.web import LEVEL6_MODELS
    from pathlib import Path

    trained = [model for name, (model, _) in LEVEL6_MODELS.items() if "trained on" in name]
    if not AMAZON_PRODUCTS_PATH.exists() or not all(Path(m).exists() for m in trained):
        pytest.skip("Amazon data or fine-tuned models not available")
    data = client.get("/api/search", params={"q": "coffee mug", "level": 6, "k": 3}).json()
    assert [c["name"] for c in data["columns"]] == list(LEVEL6_MODELS)


def test_level_8_shows_untrained_gte_small():
    from semantic_search.levels.level8.data import AMAZON_PRODUCTS_PATH

    if not AMAZON_PRODUCTS_PATH.exists():
        pytest.skip("Amazon data not downloaded (run `uv run prepare-data`)")
    data = client.get("/api/search", params={"q": "phone charger", "level": 8, "k": 3}).json()
    assert data["columns"][0]["name"] == "gte-small, no training"
