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
│       ├── level4/               # + BM25 and hybrid search
│       └── level5/               # + evaluation: labels, metrics, experiments (latest)
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

## Level 5: evaluation

Measure every method on labeled queries instead of looking at a few examples.

1. **Test queries** (`levels/level5/queries.py`): 20 queries of 4 types: intent, simple, exact, tricky.
2. **Pooling** (`pool.py`): the top 10 of every method are merged into `data/eval/pool.json`
   (684 query-product pairs). Only these are labeled.
3. **Labeling** (`/label` page in the web app): each pair is marked relevant or not, without seeing
   which method found it. Two labelers labeled everything separately and agreed on 91% of the pairs
   (Cohen's kappa 0.80).
4. **Metrics** (`evaluate.py`): P@5, R@10, MRR and nDCG@10.
5. **Experiments**: tuning hybrid (`tune.py`) and comparing embedding models (`compare_models.py`).

| Method | Model | nDCG@10 |
|---|---|---|
| Keyword | - | 0.360 |
| BM25 | - | 0.570 |
| Semantic | all-MiniLM-L6-v2 (Levels 1-4) | 0.664 |
| Semantic | all-mpnet-base-v2 (5x bigger) | 0.646 |
| Semantic | BAAI/bge-small-en-v1.5 | 0.684 |
| Hybrid | MiniLM, tuned (k=5, semantic weight 2) | 0.695 |
| Hybrid | **BGE-small (k=60, equal weights)** | **0.697** |

Level 5 uses BGE-small with plain hybrid by default. The bigger model was not better, and the
first model comparison looked wrong until the new models' results were labeled too (pool bias).
With 18 queries, the top results are too close to call a clear winner.

```bash
uv run python -m semantic_search.levels.level5.evaluate          # agreement + metrics
uv run python -m semantic_search.levels.level5.tune              # Experiment 1: hybrid settings
uv run python -m semantic_search.levels.level5.compare_models    # Experiment 2: embedding models
uv run python -m semantic_search.levels.level5.pool --add-models # add new models' results to the pool
```

## Web app

```bash
uv run web          # then open http://127.0.0.1:8000
```

One tab per level: Level 1 (30 products), Level 2 (Amazon data), Level 3 (Flat vs HNSW with
search times and an efSearch slider), Level 4 (BM25 vs semantic vs hybrid), Level 5 (results table, and
BM25 vs semantic vs hybrid with BGE-small). Each tab runs that level's own code from
`src/semantic_search/levels/`, copied from the git tags. The `semantic-search`, `prepare-data` and
`benchmark` commands run the latest level. Only the import paths and the data
folder depth were changed, and `tests/test_levels.py` checks that against git.
The first search on the Level 2 tab takes about 25 seconds, because Level 2 had no embedding cache.

You can also run an old level's command-line app:

```bash
uv run python -m semantic_search.levels.level1.cli
uv run python -m semantic_search.levels.level2.cli --amazon
```

A link like `http://127.0.0.1:8000/?q=running+shoes#level2` opens a search directly.

## Next steps

- Fine-tune the embedding model. Train on synthetic queries made by an LLM,
  and test on the hand-labeled queries from Level 5 (never train on them).
- Add more test queries, so close methods can be told apart.
- RAG: let an LLM answer questions like "what should I buy for camping?" using the search results.
