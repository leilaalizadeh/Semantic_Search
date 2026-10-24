import json

import pytest
from fastapi.testclient import TestClient

from semantic_search import web
from semantic_search.levels.level5 import labels

POOL = [
    {"query": "yoga mat", "type": "simple", "candidates": [
        {"id": "A1", "title": "Yoga Mat", "category": "Sports", "description": "thick mat", "found_by": {"bm25": 1}},
        {"id": "B2", "title": "Coffee Mug", "category": "Home", "description": "mug", "found_by": {"semantic": 3}},
    ]},
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "pool.json").write_text(json.dumps(POOL))
    monkeypatch.setattr(web, "EVAL_DIR", tmp_path)  # don't touch the real data/eval
    return TestClient(web.app), tmp_path


def test_label_data_hides_which_method_found_it(client):
    c, _ = client
    data = c.get("/api/label-data", params={"annotator": "test"}).json()
    candidate = data["queries"][0]["candidates"][0]
    assert "found_by" not in candidate
    assert data["labels"] == {}


def test_save_and_remove_label(client):
    c, tmp = client
    body = {"annotator": "test", "query": "yoga mat", "product_id": "A1", "label": 1}
    assert c.post("/api/label", json=body).status_code == 200
    assert json.loads((tmp / "labels_test.json").read_text()) == {"yoga mat": {"A1": 1}}

    c.post("/api/label", json={**body, "label": None})
    assert json.loads((tmp / "labels_test.json").read_text()) == {"yoga mat": {}}


def test_unknown_product_or_bad_name_is_rejected(client):
    c, _ = client
    body = {"annotator": "test", "query": "yoga mat", "product_id": "ZZZ", "label": 1}
    assert c.post("/api/label", json=body).status_code == 400
    assert c.post("/api/label", json={**body, "product_id": "A1", "annotator": "../x"}).status_code == 400
    assert c.post("/api/label", json={**body, "product_id": "A1", "label": 5}).status_code == 400


def test_blind_order_is_shuffled_but_stable():
    candidates = [{"id": str(i)} for i in range(20)]
    once = [c["id"] for c in labels.blind_order("q", candidates)]
    again = [c["id"] for c in labels.blind_order("q", candidates)]
    assert once == again
    assert once != [c["id"] for c in candidates]
