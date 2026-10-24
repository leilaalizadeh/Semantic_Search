"""Web app to try the search of each level in the browser

Run:  uv run web   then open http://127.0.0.1:8000

  GET /                 -> the HTML page (static/index.html)
  GET /api/search?...   -> search results as JSON, used by the page
"""

import argparse
import time
from functools import cache
from pathlib import Path

# Import FAISS before anything that loads PyTorch (level1 does, through sentence-transformers).
# Both bring their own copy of OpenMP (multi-core math). On macOS, if PyTorch loads it first,
# building the HNSW index (Level 3) crashes the whole server with a segmentation fault.
import faiss  # noqa: F401  (imported only for its load order)

# uvicorn: the server program that listens on a port (8000) and passes the requests to FastAPI
import uvicorn
# FastAPI: a web framework that turns Python functions into URLs and handles requests and responses
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from semantic_search.levels import level1, level2, level3, level4
from semantic_search.levels.level2.data import AMAZON_PRODUCTS_PATH
from semantic_search.levels.level3.semantic import build_index
from semantic_search.levels.level1.keyword_search import tokenize as level1_tokenize
from semantic_search.levels.level2.keyword_search import tokenize as level2_tokenize
from semantic_search.levels.level4.bm25 import BM25Search
from semantic_search.levels.level4.bm25 import tokenize as bm25_tokenize
from semantic_search.levels.level4.hybrid import HybridSearch

PAGE = Path(__file__).parent / "static" / "index.html"

# Creates the web app. The @app.get(...) lines below connect each URL to a function.
app = FastAPI(title="Semantic Search")


def require_amazon_data() -> None:
    if not AMAZON_PRODUCTS_PATH.exists():
        raise HTTPException(404, "Amazon data not found. Run `uv run prepare-data` first.")


# @cache: build each engine the first time its tab is used, then reuse it.
# (In the mini example the engine was built once at startup. Here that would load
# all levels at every start, and Level 2 alone takes about 25 seconds.)
@cache
def level1_engine() -> level1.SemanticSearch:
    return level1.SemanticSearch(level1.load_products())


@cache
def level2_engine() -> level2.SemanticSearch:
    # Level 2 had no cache, so this embeds all 5,000 products (~25 s) the first time.
    return level2.SemanticSearch(level2.load_products(AMAZON_PRODUCTS_PATH))


@cache
def level3_engine() -> level3.SemanticSearch:
    return level3.SemanticSearch(level3.load_products(AMAZON_PRODUCTS_PATH))


@cache
def level3_hnsw_index():
    return build_index(level3_engine().embeddings, "hnsw")


@cache
def level4_engine() -> level4.SemanticSearch:
    return level4.SemanticSearch(level4.load_products(AMAZON_PRODUCTS_PATH))


@cache
def level4_bm25() -> BM25Search:
    return BM25Search(level4_engine().products)


@cache
def level4_hybrid() -> HybridSearch:
    return HybridSearch(level4_engine(), level4_bm25())


def matched_words(query: str, p, tokenize) -> list[str]:
    """The words the query and the product share, using the level's own tokenize()."""
    return sorted(set(tokenize(query)) & set(tokenize(p.text)))


def to_dict(score: float, p, matched: list[str] | None = None) -> dict:
    # Level 1 products have no category / price / rating, so use getattr with a default.
    return {
        "matched": matched,  # words to highlight on the page (keyword-based columns only)
        "score": round(float(score), 4),
        "id": str(p.id),
        "title": p.title,
        "description": p.description,
        "category": getattr(p, "category", ""),
        "price": getattr(p, "price", None),
        "rating": getattr(p, "rating", None),
    }


def ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def keyword_vs_semantic(level, engine, query: str, k: int, tokenize) -> list[dict]:
    """Levels 1 and 2: run the level's own keyword_search and SemanticSearch.search."""
    start = time.perf_counter()
    keyword = level.keyword_search(query, engine.products, k)
    keyword_ms = ms_since(start)

    start = time.perf_counter()
    semantic = engine.search(query, k)  # embeds the query + numpy search
    semantic_ms = ms_since(start)

    return [
        {"name": "Keyword", "kind": "keyword", "ms": keyword_ms, "results": [to_dict(s, p, matched_words(query, p, tokenize)) for s, p in keyword]},
        {"name": "Semantic", "kind": "semantic", "ms": semantic_ms, "note": "includes embedding the query",
         "results": [to_dict(s, p) for s, p in semantic]},
    ]


def index_search(engine, index, query_vecs, k: int, repeats: int = 5) -> dict:
    # One search takes well under a millisecond, so a single timing is noisy.
    # Run it a few times and keep the fastest.
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        scores, ids = index.search(query_vecs, k)
        times.append(ms_since(start))
    results = [to_dict(s, engine.products[i]) for s, i in zip(scores[0], ids[0]) if i != -1]
    return {"kind": "semantic", "ms": min(times), "results": results}


@app.get("/api/search")  # URL: /api/search?q=...&level=1&k=5&ef_search=64
def search(
    q: str,
    level: int = Query(1, ge=1, le=4),
    k: int = Query(5, ge=1, le=50),
    ef_search: int = Query(64, ge=1, le=1024),
):
    query = q.strip()
    if not query:
        raise HTTPException(400, "Empty query")

    embed_ms = None
    if level == 1:
        engine = level1_engine()
        columns = keyword_vs_semantic(level1, engine, query, k, level1_tokenize)
    elif level == 2:
        require_amazon_data()
        engine = level2_engine()
        columns = keyword_vs_semantic(level2, engine, query, k, level2_tokenize)
    elif level == 4:
        require_amazon_data()
        engine, bm25, hybrid = level4_engine(), level4_bm25(), level4_hybrid()  # build before timing

        start = time.perf_counter()
        bm25_results = bm25.search(query, k)
        bm25_ms = ms_since(start)

        start = time.perf_counter()
        semantic_results = engine.search(query, k)
        semantic_ms = ms_since(start)

        start = time.perf_counter()
        hybrid_results = hybrid.fuse(query, k)
        hybrid_ms = ms_since(start)

        columns = [
            {"name": "BM25", "kind": "bm25", "ms": bm25_ms,
             "results": [to_dict(s, p, matched_words(query, p, bm25_tokenize)) for s, p in bm25_results]},
            {"name": "Semantic", "kind": "semantic", "ms": semantic_ms, "note": "includes embedding the query",
             "results": [to_dict(s, p) for s, p in semantic_results]},
            {"name": "Hybrid (RRF)", "kind": "hybrid", "ms": hybrid_ms, "note": "runs both + fuses",
             "results": [{**to_dict(r.score, r.product, matched_words(query, r.product, bm25_tokenize)), "bm25_rank": r.bm25_rank, "semantic_rank": r.semantic_rank}
                         for r in hybrid_results]},
        ]
    else:
        require_amazon_data()
        engine = level3_engine()
        start = time.perf_counter()
        query_vecs = engine.model.encode([query], normalize_embeddings=True)
        embed_ms = ms_since(start)
        hnsw = level3_hnsw_index()
        hnsw.hnsw.efSearch = ef_search
        columns = [
            {"name": "Flat (exact)", **index_search(engine, engine.index, query_vecs, k)},
            {"name": f"HNSW (efSearch={ef_search})", **index_search(engine, hnsw, query_vecs, k)},
        ]
    # FastAPI sends this dict to the browser as JSON. The page's JavaScript shows it as columns of results.
    return {
        "level": level,
        "query": query,
        "num_products": len(engine.products),
        "embed_ms": embed_ms,
        "columns": columns,
    }


@app.get("/")  # URL: / returns the page (static/index.html) with the search box and the result columns
def page():
    return FileResponse(PAGE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the search web app")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Restart automatically when the code changes")
    args = parser.parse_args()
    print(f"Open http://127.0.0.1:{args.port} in your browser (Ctrl+C to stop)")
    # reload needs the app as an "import string" so uvicorn can re-import it after a change.
    # reload_dirs: only watch our own code. Otherwise uvicorn also watches the ~12,000
    # library files in .venv/, which uses a lot of CPU and can leave the server stuck.
    uvicorn.run(
        "semantic_search.web:app",
        host="127.0.0.1",
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(Path(__file__).parent)] if args.reload else None,
        reload_includes=["*.py", "*.html"] if args.reload else None,
    )


if __name__ == "__main__":
    main()
