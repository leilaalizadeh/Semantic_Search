"""Web app to try the search of each level in the browser.

Run:  uv run web   then open http://127.0.0.1:8000

  GET /                 -> the HTML page (static/index.html)
  GET /api/search?...   -> search results as JSON, used by the page
"""

import argparse
import time
from functools import cache
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer

from semantic_search.data import AMAZON_PRODUCTS_PATH, DEFAULT_PRODUCTS_PATH, Product, load_products
from semantic_search.keyword_search import keyword_search
from semantic_search.semantic import DEFAULT_MODEL, SemanticSearch, build_index

PAGE = Path(__file__).parent / "static" / "index.html"

app = FastAPI(title="Semantic Search")


# @cache: build each thing the first time it is needed, then reuse it.
@cache
def get_model() -> SentenceTransformer:
    return SentenceTransformer(DEFAULT_MODEL)


@cache
def get_engine(dataset: str) -> SemanticSearch:
    if dataset == "small":
        path = DEFAULT_PRODUCTS_PATH
    elif AMAZON_PRODUCTS_PATH.exists():
        path = AMAZON_PRODUCTS_PATH
    else:
        raise HTTPException(404, "Amazon data not found. Run `uv run prepare-data` first.")
    return SemanticSearch(load_products(path), model=get_model())


@cache
def get_hnsw_index():
    return build_index(get_engine("amazon").embeddings, "hnsw")


def to_dict(score: float, p: Product) -> dict:
    return {
        "score": round(float(score), 3),
        "id": str(p.id),
        "title": p.title,
        "description": p.description,
        "category": p.category,
        "price": p.price,
        "rating": p.rating,
    }


def ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def index_search(engine: SemanticSearch, index, query_vecs, k: int) -> dict:
    start = time.perf_counter()
    scores, ids = index.search(query_vecs, k)
    ms = ms_since(start)
    results = [to_dict(s, engine.products[i]) for s, i in zip(scores[0], ids[0]) if i != -1]
    return {"ms": ms, "results": results}


@app.get("/api/search")
def search(
    q: str,
    level: int = Query(1, ge=1, le=3),
    k: int = Query(5, ge=1, le=50),
    ef_search: int = Query(64, ge=1, le=1024),
):
    query = q.strip()
    if not query:
        raise HTTPException(400, "Empty query")

    engine = get_engine("small" if level == 1 else "amazon")

    start = time.perf_counter()
    query_vecs = engine.model.encode([query], normalize_embeddings=True)
    embed_ms = ms_since(start)

    if level in (1, 2):
        start = time.perf_counter()
        keyword = [to_dict(s, p) for s, p in keyword_search(query, engine.products, k)]
        columns = [
            {"name": "Keyword", "ms": ms_since(start), "results": keyword},
            {"name": "Semantic", **index_search(engine, engine.index, query_vecs, k)},
        ]
    else:
        hnsw = get_hnsw_index()
        hnsw.hnsw.efSearch = ef_search
        columns = [
            {"name": "Flat (exact)", **index_search(engine, engine.index, query_vecs, k)},
            {"name": f"HNSW (efSearch={ef_search})", **index_search(engine, hnsw, query_vecs, k)},
        ]

    return {
        "level": level,
        "query": query,
        "num_products": len(engine.products),
        "embed_ms": embed_ms,
        "columns": columns,
    }


@app.get("/")
def page():
    return FileResponse(PAGE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the search web app")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Open http://127.0.0.1:{args.port} in your browser (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
