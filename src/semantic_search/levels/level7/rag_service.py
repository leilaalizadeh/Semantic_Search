"""Level 7, Step 4: RAG as a small service in its own process, for the web app.

The web server loads FAISS first (the Level 3 HNSW tab crashes otherwise), but the LLM must be
loaded before FAISS (Step 1). So the web server runs RAG in a separate Python process and talks
to it through stdin/stdout, one JSON line per question and per answer:

    web server  --{"question": "..."}-->  this program  --RESULT {"answer": ...}-->  web server

Run by hand to try it:  uv run python -m semantic_search.levels.level7.rag_service
then type:  {"question": "I need a toy for my dog"}
"""

import json
import sys

from semantic_search.levels.level7.rag import RAG  # loads the LLM before FAISS

READY = "READY"      # printed once the models are loaded
RESULT = "RESULT "   # every answer line starts with this, so other output can't be mistaken for it


def to_json(result: dict) -> dict:
    kept = iter(range(1, len(result["products"]) + 1))  # the answer cites kept products as [1], [2], ...
    products = []
    for p, good in zip(result["searched"], result["judged"]):
        products.append({
            "id": str(p.id), "title": p.title, "description": p.description, "category": p.category,
            "price": p.price, "rating": p.rating,
            "kept": good, "number": next(kept) if good else None,
        })
    return {
        "answer": result["answer"],
        "products": products,
        "cited": result["cited"],
        "search_ms": round(result["search_ms"], 1),
        "llm_s": round(result["llm_s"], 1),
    }


def main() -> None:
    rag = RAG(check=True)  # the 1.5B model with the product check (the best setup that fits on the laptop)
    print(READY, flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            question = json.loads(line)["question"]
            reply = to_json(rag.ask(question))
        except Exception as e:  # keep the service alive and tell the web server what went wrong
            reply = {"error": f"{type(e).__name__}: {e}"}
        print(RESULT + json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
