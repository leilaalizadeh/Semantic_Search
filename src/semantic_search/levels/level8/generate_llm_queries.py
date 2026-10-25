"""Level 8, Step 1: training queries written by an LLM, in the style of real shoppers.

Level 6 used T5, which wrote questions like "what is a hose nozzle". Our test queries look
different: product types ("coffee mug") and needs ("my car is dirty"). So here an instruction
LLM writes 3 searches per product: the kind of product, a need it solves, and a feature.

Run on a GPU (Colab):  python -m semantic_search.levels.level8.generate_llm_queries
Quick look on a laptop: uv run python -m semantic_search.levels.level8.generate_llm_queries --limit 4 --model Qwen/Qwen2.5-0.5B-Instruct
Writes data/train/llm_queries.jsonl

Our 20 test queries (also as plurals) are removed: never train on the exam.
"""

import argparse
import json
import re
import time

from semantic_search.levels.level8 import llm
from semantic_search.levels.level8.data import AMAZON_PRODUCTS_PATH, load_products
from semantic_search.levels.level8.generate_queries import TRAIN_PATH
from semantic_search.levels.level8.shopping_queries import TEST_QUERIES, normalize

LLM_QUERIES_PATH = TRAIN_PATH.parent / "llm_queries.jsonl"
WRITER_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # fits on a free Colab T4 GPU

SYSTEM_PROMPT = "You write the searches that shoppers type into an online shop."
USER_PROMPT = """Product: {title}{category}
{description}

Write 3 different short searches (2 to 6 words) a shopper might type to find this product:
1. the kind of product (for example "camping tent")
2. a need or problem it solves (for example "going camping with kids")
3. a feature someone might want (for example "waterproof tent for 4")
Do not use the brand name. Write only the 3 searches, one per line."""


def messages_for(p) -> list[dict]:
    category = f" ({p.category})" if p.category else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(title=p.title[:150], category=category,
                                                       description=p.description[:250])},
    ]


def parse_queries(text: str) -> list[str]:
    """Turn the LLM's answer into clean queries: one per line, without numbers, quotes or labels."""
    queries = []
    for line in text.splitlines():
        q = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line)       # "1. ", "2) ", "- "
        q = re.sub(r"^[A-Za-z ]{3,25}:\s*", "", q)             # "Need: ", "Feature: "
        q = q.strip(" \"'“”`.").lower()
        if 2 <= len(q.split()) <= 8 and q not in queries:
            queries.append(q)
    return queries[:3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=WRITER_MODEL)
    parser.add_argument("--batch-size", type=int, default=32, help="Products per LLM call (lower it if the GPU runs out of memory)")
    parser.add_argument("--limit", type=int, help="Only the first N products, and print instead of saving")
    args = parser.parse_args()

    products = load_products(AMAZON_PRODUCTS_PATH)[:args.limit]
    start = time.perf_counter()
    pairs, removed = [], 0
    for i in range(0, len(products), args.batch_size):
        batch = products[i:i + args.batch_size]
        answers = llm.chat_batch([messages_for(p) for p in batch], max_new_tokens=60, model_name=args.model)
        for p, answer in zip(batch, answers):
            for q in parse_queries(answer):
                if normalize(q) in TEST_QUERIES:  # never train on the exam
                    removed += 1
                    continue
                pairs.append({"query": q, "product_id": str(p.id)})
        done = i + len(batch)
        if done % (args.batch_size * 10) == 0 or done == len(products):
            print(f"  {done}/{len(products)} products, {len(pairs)} queries, {time.perf_counter() - start:.0f}s", flush=True)

    print(f"{len(pairs)} queries for {len(products)} products; removed {removed} that matched a test query")
    if args.limit:
        by_id = {str(p.id): p for p in products}
        for pair in pairs:
            print(f"  {pair['query']:<40} <- {by_id[pair['product_id']].title[:60]}")
        return

    LLM_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LLM_QUERIES_PATH, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved to {LLM_QUERIES_PATH}")


if __name__ == "__main__":
    main()
