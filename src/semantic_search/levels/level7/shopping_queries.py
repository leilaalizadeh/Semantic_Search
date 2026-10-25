"""Level 6, Step 3: training queries in shopping style.

Step 2 trained on T5 questions ("what is a hose nozzle") where only ONE product was right.
Our test queries look different ("coffee mug") and many products are right. So we build
new training data from two sources:

  1. T5 questions without the question part:  "what is a hose nozzle" -> "hose nozzle"
  2. short queries from product titles:        "Ergonomic Office Chair, Desk Chair ..." -> "office chair"
     Many office chairs get the same query, so the model learns that many products can match.

Training queries that are (almost) one of our 20 test queries are removed: never train on the exam.

Run:  uv run python -m semantic_search.levels.level7.shopping_queries
Writes data/train/shopping_queries.jsonl
"""

import json
import re
from collections import Counter

from semantic_search.levels.level7.data import AMAZON_PRODUCTS_PATH, load_products
from semantic_search.levels.level7.generate_queries import TRAIN_PATH
from semantic_search.levels.level7.queries import QUERIES

SHOPPING_PATH = TRAIN_PATH.parent / "shopping_queries.jsonl"
MAX_WORDS = 5  # longer queries are still questions or descriptions, not searches

# The question part at the start of a T5 query, e.g. "what is a", "how big is the", "what kind of"
QUESTION_START = re.compile(
    r"^(what|which|who|where|how|when|why|is|are|does|do|can)\b"
    r"(\s+(is|are|was|does|do|can|kind|type|size|sizes|brand|color|big|long|much|many|tall|wide|fast|heavy|to|of|makes?|you|i))*"
    r"(\s+(an?|the))?\s+"
)
# Leftovers of a question ("furniture does chs have") or filler words from titles ("set of", "x inch")
QUESTION_WORDS = {"does", "do", "did", "have", "has", "made", "make", "use", "used", "mean", "means",
                  "fit", "work", "works", "cost", "costs", "look", "like", "called", "need"}
FILLER_WORDS = {"of", "to", "x", "and", "or", "the", "a", "an", "in", "on", "inch", "inches", "set", "pack",
                "pcs", "piece", "pieces", "compatible", "size", "new", "upgraded", "large", "small"}
# Where the product type usually ends in a title: before ", with / for / | - (" and so on
TITLE_END = re.compile(r",| with | for | - | – |\||\(|/| by ", re.IGNORECASE)


def normalize(text: str) -> str:
    """Lowercase, letters and digits only, and no plural "s", so "Coffee Mugs" == "coffee mug"."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words)


TEST_QUERIES = {normalize(q["query"]) for q in QUERIES}


def clean_question(query: str) -> str | None:
    q = QUESTION_START.sub("", query.lower().strip()).strip(" ?.")
    words = q.split()
    if not words or words[0] in {"what", "which", "who", "where", "how", "when", "why"}:
        return None
    if QUESTION_WORDS & set(words):  # still a question, just without its first words
        return None
    return q if len(words) <= MAX_WORDS else None


def title_query(title: str) -> str | None:
    """The last 2 words of the title's first part, without words that have digits."""
    first_part = TITLE_END.split(title, maxsplit=1)[0]
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", first_part) if not any(ch.isdigit() for ch in w)]
    last_two = [w.lower() for w in words[-2:]]
    if len(last_two) < 2 or FILLER_WORDS & set(last_two) or min(len(w) for w in last_two) < 3:
        return None
    return " ".join(last_two)


def main() -> None:
    products = load_products(AMAZON_PRODUCTS_PATH)
    pairs, sources = [], Counter()

    for line in open(TRAIN_PATH):
        row = json.loads(line)
        q = clean_question(row["query"])
        if q:
            pairs.append({"query": q, "product_id": row["product_id"], "source": "t5 cleaned"})
    for p in products:
        q = title_query(p.title)
        if q:
            pairs.append({"query": q, "product_id": str(p.id), "source": "title"})

    kept, seen = [], set()
    removed_test = 0
    for pair in pairs:
        key = (pair["query"], pair["product_id"])
        if key in seen:
            continue
        seen.add(key)
        if normalize(pair["query"]) in TEST_QUERIES:  # never train on the exam
            removed_test += 1
            continue
        kept.append(pair)
        sources[pair["source"]] += 1

    with open(SHOPPING_PATH, "w") as f:
        for pair in kept:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"{len(kept)} pairs: {dict(sources)}; removed {removed_test} that matched a test query")
    print(f"Saved to {SHOPPING_PATH}")


if __name__ == "__main__":
    main()
