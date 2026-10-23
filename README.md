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
│   ├── web.py                    # web app (FastAPI), one tab per level
│   ├── static/index.html         # search page
│   └── levels/                   # each level's full code (from the git tags)
│       ├── level1/               # data, keyword_search, semantic, cli
│       ├── level2/               # + prepare_data (download Amazon data)
│       ├── level3/               # + cache, FAISS / HNSW, benchmark
│       └── level4/               # + BM25 and hybrid search (latest)
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

## Level 4: BM25 and hybrid search

- **BM25** (`levels/level4/bm25.py`): keyword ranking where rare words count more (IDF),
  repeated words count less and less (k1), and long texts are normalized (b). An inverted
  index makes it about 1000x faster than the old keyword search (0.07 ms vs 75 ms).
- **Hybrid** (`levels/level4/hybrid.py`): combines BM25 and semantic rankings with
  Reciprocal Rank Fusion, `1/(60 + rank)` summed over both lists.

| Query | BM25 | Semantic | Hybrid |
|---|---|---|---|
| i have cold keep me warm | winter coat, mittens | face mask, fleece | mittens, face mask, coat |
| something to fix a leaking sink | "I Fix Stuff" hat ❌ | kitchen mat | sink caddy (BM25 #2, semantic #2) |
| going camping this weekend | sweater dress ❌ (rare word "weekend") | tent, sleeping bag | sleeping pad, sleeping bag (tent drops out) |

Hybrid removes the worst mistakes of each method, but it is not better on every query.
Measuring that properly is the next step (evaluation).

## Web app

```bash
uv run web          # then open http://127.0.0.1:8000
```

One tab per level: Level 1 (30 products), Level 2 (Amazon data), Level 3 (Flat vs HNSW with
search times and an efSearch slider), Level 4 (BM25 vs semantic vs hybrid). Each tab runs that level's own code from
`src/semantic_search/levels/`, copied from the git tags. The `semantic-search`, `prepare-data` and
`benchmark` commands run the latest level. Only the import paths and the data
folder depth were changed, and `tests/test_levels.py` checks that against git.
The first search on the Level 2 tab takes about 25 seconds, because Level 2 had no embedding cache.

You can also run an old level's command-line app:

```bash
uv run python -m semantic_search.levels.level1.cli
uv run python -m semantic_search.levels.level2.cli --amazon
``` A link like `http://127.0.0.1:8000/?q=running+shoes#level2`
opens a search directly.

## Next steps

- Store embeddings in a vector database (FAISS, Chroma, Qdrant)
- Hybrid search: combine keyword (BM25) and semantic scores
- Add a web UI (FastAPI / Streamlit)
