"""Interactive CLI comparing keyword search and semantic search side by side."""

import argparse

from semantic_search.levels.level1.data import DEFAULT_PRODUCTS_PATH, load_products
from semantic_search.levels.level1.keyword_search import keyword_search
from semantic_search.levels.level1.semantic import DEFAULT_MODEL, SemanticSearch


def print_results(name: str, results) -> None:
    print(f"  {name}:")
    if not results:
        print("    (no results)")
    for score, p in results:
        print(f"    {score:6.3f}  {p.title}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Run one query and exit (omit for interactive mode)")
    parser.add_argument("--data", default=DEFAULT_PRODUCTS_PATH, help="Path to products JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    parser.add_argument("-k", "--top-k", type=int, default=3)
    args = parser.parse_args()

    products = load_products(args.data)
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
