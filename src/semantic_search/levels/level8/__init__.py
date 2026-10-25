from semantic_search.levels.level8.data import Product, load_products
from semantic_search.levels.level8.keyword_search import keyword_search

# SemanticSearch is NOT imported here (unlike Levels 1-5): it loads FAISS, and the T5 model in
# generate_queries.py crashes when FAISS is loaded in the same program.
# Import it from the module instead: from semantic_search.levels.level8.semantic import SemanticSearch

__all__ = ["Product", "load_products", "keyword_search"]
