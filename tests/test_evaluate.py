import math

import pytest

from semantic_search.levels.level5.evaluate import cohen_kappa, ndcg_at, precision_at, recall_at, reciprocal_rank

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "d", "x"}  # "x" is relevant but was not found


def test_precision_and_recall():
    assert precision_at(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)  # b, d out of 5 results
    assert recall_at(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)     # found 2 of the 3 relevant


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, RELEVANT) == pytest.approx(1 / 2)  # first hit "b" at position 2
    assert reciprocal_rank(RANKED, {"zzz"}) == 0.0


def test_ndcg_by_hand():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)                    # hits at positions 2 and 4
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # 3 relevant at positions 1, 2, 3
    assert ndcg_at(RANKED, RELEVANT, 5) == pytest.approx(dcg / ideal)
    assert ndcg_at(["b", "d", "x"], RELEVANT, 5) == pytest.approx(1.0)  # perfect order


def test_cohen_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)   # always agree
    # textbook example: 20 items, agree on 15 -> observed 0.75, chance 0.5 -> kappa 0.5
    a = [1] * 10 + [0] * 10
    b = [1] * 7 + [0] * 3 + [1] * 2 + [0] * 8
    assert cohen_kappa(a, b) == pytest.approx((0.75 - 0.5) / (1 - 0.5))
