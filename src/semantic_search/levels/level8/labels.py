"""Read and write relevance labels.

data/eval/labels_<annotator>.json looks like:
    {"going camping this weekend": {"B07ABC123": 1, "B08XYZ789": 0}, ...}

1 = relevant, 0 = not relevant, missing = not labeled yet.
Each person (annotator) has their own file, so their labels stay independent.
"""

import hashlib
import json
import re
from pathlib import Path

from semantic_search.levels.level8.data import DATA_DIR

EVAL_DIR = DATA_DIR / "eval"
POOL_PATH = EVAL_DIR / "pool.json"
ANNOTATOR_NAME = re.compile(r"^[a-z0-9_-]{1,30}$")  # used in a file name, so keep it simple


def labels_path(annotator: str, eval_dir: Path = EVAL_DIR) -> Path:
    if not ANNOTATOR_NAME.match(annotator):
        raise ValueError("Name must be 1-30 characters: a-z, 0-9, - or _")
    return eval_dir / f"labels_{annotator}.json"


def load_pool(eval_dir: Path = EVAL_DIR) -> list[dict]:
    with open(eval_dir / POOL_PATH.name) as f:
        return json.load(f)


def blind_order(query: str, candidates: list[dict]) -> list[dict]:
    """Shuffle the candidates in a fixed way, so no method's results come first.

    Sorting by a hash of (query, id) looks random but is the same every time.
    """
    return sorted(candidates, key=lambda c: hashlib.sha256(f"{query}|{c['id']}".encode()).hexdigest())


def load_labels(annotator: str, eval_dir: Path = EVAL_DIR) -> dict[str, dict[str, int]]:
    path = labels_path(annotator, eval_dir)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_label(annotator: str, query: str, product_id: str, label: int | None, eval_dir: Path = EVAL_DIR) -> None:
    """Set one label (1 or 0), or remove it with None."""
    labels = load_labels(annotator, eval_dir)
    if label is None:
        labels.get(query, {}).pop(product_id, None)
    else:
        labels.setdefault(query, {})[product_id] = label

    path = labels_path(annotator, eval_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file first, then replace: if something goes wrong, the old file is still complete.
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(labels, f, indent=1, ensure_ascii=False, sort_keys=True)
    tmp.replace(path)
