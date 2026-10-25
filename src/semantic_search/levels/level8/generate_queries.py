"""Level 6, Step 1: make training data by generating queries from product texts.

A small T5 model (doc2query) was trained to read a text and write a search query that
would find it. We give it each product and keep its queries as (query, product) pairs.
Nobody labels anything by hand: this is "synthetic" training data.

Run:  uv run python -m semantic_search.levels.level8.generate_queries --limit 20   (quick look)
      uv run python -m semantic_search.levels.level8.generate_queries               (all products)
Writes data/train/synthetic_queries.jsonl (one {"query", "product_id"} per line)

Our 20 test queries from Level 5 are removed from the training data:
never train on the exam.
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from semantic_search.levels.level8.data import AMAZON_PRODUCTS_PATH, DATA_DIR, load_products
from semantic_search.levels.level8.queries import QUERIES

MODEL = "doc2query/msmarco-t5-small-v1"
TRAIN_PATH = DATA_DIR / "train" / "synthetic_queries.jsonl"
TEST_QUERIES = {q["query"].lower() for q in QUERIES}


def generate(products, per_product: int = 3, batch_size: int = 16, seed: int = 42):
    """Yield {"query", "product_id"} dicts, a few different queries per product."""
    torch.manual_seed(seed)
    # Use MPS if available, otherwise CPU. GPU is not used because it is not available on Apple Silicon.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device).eval()

    # Generate queries in batches to avoid running out of memory
    for start in range(0, len(products), batch_size):
        batch = products[start:start + batch_size]
        # Tokenize the product texts and move them to the device
        inputs = tokenizer([p.text for p in batch], max_length=256, truncation=True,
                           padding=True, return_tensors="pt").to(device)
        # Generate queries using the model
        with torch.no_grad():  # we only use the model, we don't train it
            outputs = model.generate(
                **inputs,
                max_length=32, # limit the length of generated queries
                do_sample=True,  # sampling gives different queries each time, greedy would repeat itself
                top_p=0.95, # nucleus sampling: only sample from the top 95% of probability mass
                num_return_sequences=per_product, #
            )
        # Decode the generated queries and yield them as (query, product_id) pairs
        queries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, p in enumerate(batch):
            mine = queries[i * per_product:(i + 1) * per_product]
            for q in dict.fromkeys(q.strip() for q in mine):  # remove duplicates, keep the order
                if q and q.lower() not in TEST_QUERIES:
                    yield {"query": q, "product_id": str(p.id)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="Only use the first N products (to try it quickly)")
    parser.add_argument("--per-product", type=int, default=3, help="Queries to generate per product")
    args = parser.parse_args()

    products = load_products(AMAZON_PRODUCTS_PATH)[:args.limit]
    start = time.perf_counter()
    pairs = []
    for pair in generate(products, args.per_product):
        pairs.append(pair)
        if len(pairs) % 1500 == 0:
            print(f"  {len(pairs)} queries, {time.perf_counter() - start:.0f}s", flush=True)
    seconds = time.perf_counter() - start
    print(f"{len(pairs)} queries for {len(products)} products in {seconds:.0f}s")

    if args.limit:  # quick look: print, don't save
        by_id = {str(p.id): p for p in products}
        for pair in pairs[:30]:
            print(f"  {pair['query']:<45} <- {by_id[pair['product_id']].title[:60]}")
        return

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_PATH, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved to {TRAIN_PATH}")


if __name__ == "__main__":
    main()
