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
├── data/products.json            # sample catalog (30 products)
├── src/semantic_search/
│   ├── data.py                   # Product model + loader
│   ├── keyword_search.py         # baseline word-matching search
│   ├── semantic.py               # embedding-based search
│   └── cli.py                    # interactive command-line app
└── tests/test_search.py
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

## Next steps

- Use a real dataset (e.g. the Amazon product datasets on Kaggle / Hugging Face)
- Store embeddings in a vector database (FAISS, Chroma, Qdrant)
- Hybrid search: combine keyword (BM25) and semantic scores
- Add a web UI (FastAPI / Streamlit)
