"""Interactive CLI comparing keyword search and semantic search side by side."""

import argparse
from pathlib import Path

from semantic_search.levels.level2.data import AMAZON_PRODUCTS_PATH, DEFAULT_PRODUCTS_PATH, load_products
from semantic_search.levels.level2.keyword_search import keyword_search
from semantic_search.levels.level2.semantic import DEFAULT_MODEL, SemanticSearch


# Cuts long titles so they fit in the terminal. It only changes what is printed, not the search results.
def shorten(text: str, width: int = 80) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def print_results(name: str, results) -> None:
    print(f"  {name}:")
    if not results:
        print("    (no results)")
    for score, p in results:
        extra = [p.category] if p.category else []
        if p.price is not None:
            extra.append(f"${p.price:.2f}")
        suffix = f"  [{' | '.join(extra)}]" if extra else ""
        print(f"    {score:6.3f}  {shorten(p.title)}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Run one query and exit (omit for interactive mode)")
    parser.add_argument("--data", default=DEFAULT_PRODUCTS_PATH, help="Path to products JSON")
    parser.add_argument("--amazon", action="store_true", help="Use the real Amazon data (run `prepare-data` first)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    parser.add_argument("-k", "--top-k", type=int, default=3)
    args = parser.parse_args()

    data_path = AMAZON_PRODUCTS_PATH if args.amazon else args.data
    if not Path(data_path).exists():
        parser.error(f"{data_path} not found. Run `uv run prepare-data` first.")
    products = load_products(data_path)
    print(f"Embedding {len(products)} products with {args.model}...")
    engine = SemanticSearch(products, args.model)

    def run(query: str) -> None:
        print_results("Keyword", keyword_search(query, products, args.top_k))
        print_results("Semantic", engine.search(query, args.top_k))
        print()

    if args.query:
        run(args.query)
        return

    print("Ready! Type a query (empty line to quit).\n")
    while query := input("Search> ").strip():
        run(query)


if __name__ == "__main__":
    main()
