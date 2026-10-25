"""Dataset: https://huggingface.co/datasets/milistu/AMAZON-Products-2023 (~117k products, 4 files)

Steps:
  1. Download only the columns we need (Parquet is column-based, so we skip the
     big precomputed "embeddings" column and save ~90% of the download).
  2. Clean: drop empty / duplicate / non-physical products, tidy whitespace,
     shorten very long descriptions.
  3. Randomly sample N products and save them as JSON.
"""

import argparse
import json
import random
import re

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from semantic_search.levels.level8.data import AMAZON_PRODUCTS_PATH

PARQUET_DIR = "datasets/milistu/AMAZON-Products-2023@refs%2Fconvert%2Fparquet/default/train"
COLUMNS = ["parent_asin", "title", "description", "main_category", "price", "average_rating"]

# Digital items (music, apps...) aren't things you'd search for in a shop.
SKIP_CATEGORIES = {"Digital Music", "Appstore for Android", "Movies & TV", "Books", "Kindle Store"}
MAX_DESCRIPTION_CHARS = 500  # the model only reads ~256 tokens anyway


def download(num_files: int) -> list[dict]:
    """Download the first `num_files` Parquet files (~34k products each) as a list of dicts."""
    fs = HfFileSystem()
    rows = []
    for i in range(num_files):
        path = f"{PARQUET_DIR}/{i:04d}.parquet"
        print(f"Downloading file {i + 1}/{num_files} (about 1 minute each)...")
        with fs.open(path, block_size=2**20) as f:
            rows.extend(pq.read_table(f, columns=COLUMNS).to_pylist())
    return rows


def clean_text(text: str | None, max_chars: int | None = None) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()  # newlines / double spaces -> one space
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."  # cut at a word boundary
    return text


def clean(rows: list[dict]) -> list[dict]:
    products, seen_titles = [], set()
    for row in rows:
        title = clean_text(row["title"])
        description = clean_text(row["description"], MAX_DESCRIPTION_CHARS)
        category = row["main_category"] or ""

        if len(title) < 5 or len(description) < 20 or category in SKIP_CATEGORIES:
            continue
        if title.lower() in seen_titles:  # duplicates
            continue
        seen_titles.add(title.lower())

        products.append({
            "id": row["parent_asin"],
            "title": title,
            "description": description,
            "category": category,
            "price": row["price"],
            "rating": row["average_rating"],
        })
    return products


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--num-products", type=int, default=5000, help="How many products to keep")
    parser.add_argument("--files", type=int, default=1, choices=[1, 2, 3, 4], help="How many dataset files to download")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, so the sample is reproducible")
    parser.add_argument("-o", "--output", default=AMAZON_PRODUCTS_PATH)
    args = parser.parse_args()

    rows = download(args.files)
    products = clean(rows)
    print(f"Downloaded {len(rows)} products, {len(products)} left after cleaning.")

    random.seed(args.seed)
    sample = random.sample(products, min(args.num_products, len(products)))

    with open(args.output, "w") as f:
        json.dump(sample, f, indent=1, ensure_ascii=False)
    print(f"Saved {len(sample)} products to {args.output}")


if __name__ == "__main__":
    main()
