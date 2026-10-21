# Semantic Search

A beginner project that compares **traditional keyword search** with **semantic search** on a small product catalog, similar to searching on Amazon.

| Keyword search | Semantic search |
|---|---|
| Matches the exact words in the query | Matches what the query *means* (intent, context) |
| "gift for a 6 year old" → JBL Flip **6** Speaker | → LEGO Classic Bricks, Harry Potter Box Set |

## How it works

```
products.json ──► embedding model ──► product vectors (computed once)
                                               │
user query ────► embedding model ──► query vector ──► cosine similarity ──► top-k products
```

1. Each product (`title + description`) is turned into a vector (an **embedding**) with the
   free, local model [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
2. The user query is embedded with the same model.
3. Products are ranked by **cosine similarity** between the query vector and each product vector.

## Project structure

```
├── data/
│   ├── products.json             # tiny hand-made catalog (30 products)
│   └── amazon_products.json      # real Amazon sample (created by `prepare-data`, not in git)
├── src/semantic_search/
│   ├── data.py                   # Product model + loader
│   ├── prepare_data.py           # download + clean real Amazon data
│   ├── keyword_search.py         # baseline word-matching search
│   ├── semantic.py               # embedding-based search
│   └── cli.py                    # interactive command-line app
└── tests/
```

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                           # install dependencies
uv run semantic-search                            # interactive mode
uv run semantic-search "gift for a 6 year old"    # one query
uv run pytest                                     # run tests
```

The first run downloads the embedding model (~90 MB).

Example:

```
Search> gift for a 6 year old
  Keyword:
     1.000  JBL Flip 6 Bluetooth Speaker
  Semantic:
     0.340  LEGO Classic Creative Bricks
     0.251  Harry Potter Box Set
     0.250  Dog Chew Toy
```

## Level 2: real Amazon data

```bash
uv run prepare-data                        # download, clean, keep 5,000 random products (~1 min)
uv run semantic-search --amazon            # search them
uv run prepare-data -n 20000 --files 2     # more products (slower to embed)
```

Data: [milistu/AMAZON-Products-2023](https://huggingface.co/datasets/milistu/AMAZON-Products-2023)
(117k products from the Amazon Reviews 2023 dataset). Only the needed Parquet columns are
downloaded; rows with empty or duplicate titles, very short descriptions, or digital-only
categories are dropped.

```
Search> keep me warm when it snows
  Keyword:
     4.000  Red White Blue Patriotic Dangle Earrings for Women Girls 4th of July ...
  Semantic:
     0.496  Tough Headwear Full Face Mask for Cold Weather for Men - Work Balaclava ...
     0.445  Columbia Men's Basin Trail Fleece Full Zip
```

## Next steps

- Store embeddings in a vector database (FAISS, Chroma, Qdrant)
- Hybrid search: combine keyword (BM25) and semantic scores
- Add a web UI (FastAPI / Streamlit)
