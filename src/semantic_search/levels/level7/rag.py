"""Level 7, Step 2: RAG = Retrieval-Augmented Generation.

  1. Retrieval:  our search (gte-small) finds the top products for the question
  2. Augmented:  the products are added to the LLM's prompt, numbered [1], [2], ...
  3. Generation: the LLM answers using ONLY those products, citing them by number

Run:  uv run python -m semantic_search.levels.level7.rag "I'm going camping with my kids, what should I buy?"
"""

import argparse
import re
import time

# The LLM must be loaded before FAISS (used by the search), or the program crashes (Step 1).
from semantic_search.levels.level7 import llm
from semantic_search.levels.level7.data import AMAZON_PRODUCTS_PATH, Product, load_products

SYSTEM_PROMPT = """You are a helpful shopping assistant for an online shop.
You get a customer question and a numbered list of products from the shop.
Write a short answer (2 to 4 sentences) in your own words:
- recommend the 2 or 3 products that fit the question best, and cite each one by its number, like [2]
- say in a few words why each one is useful for the customer
- never copy the product list, and never mention a product that is not in the list
- if no product fits, say that the shop has nothing suitable"""


def product_line(number: int, p: Product, description_chars: int = 150) -> str:
    price = f" | ${p.price:.2f}" if p.price is not None else ""
    description = p.description[:description_chars].rsplit(" ", 1)[0]
    category = f" | {p.category}" if p.category else ""
    return f"[{number}] {p.title[:120]}{category}{price}\n    {description}"


def build_messages(question: str, products: list[Product]) -> list[dict]:
    product_list = "\n".join(product_line(i, p) for i, p in enumerate(products, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Products:\n{product_list}\n\nCustomer question: {question}\n\n"
                                    "Your short answer (2 to 4 sentences, cite products like [2]):"},
    ]


# First version: "would a customer who typed this search be happy to see this product?"
# The 3B model read that too strictly: a sleeping bag alone doesn't "complete" a camping trip,
# so it said no to 40 of 59 relevant products. This version asks if the product is useful.
CHECK_PROMPT = """You check search results for an online shop.
You get a customer's search or question and one product. Answer only "yes" or "no".
Say "yes" if the product could be useful for what the customer wants, even if it is only
one of several things they need. Say "no" only if the product does not fit what they want."""

NOTHING_FOUND = "Sorry, the shop doesn't have anything that fits your question."


def check_messages(question: str, p: Product) -> list[dict]:
    category = f" ({p.category})" if p.category else ""
    return [
        {"role": "system", "content": CHECK_PROMPT},
        {"role": "user", "content": f"Search: {question}\nProduct: {p.title[:150]}{category}\n"
                                    f"{p.description[:200]}\n\nCould this product be useful for what the customer wants? Answer yes or no."},
    ]


def cited_numbers(answer: str) -> list[int]:
    """The product numbers the answer cites, like [2] or [1, 3], in order, without repeats."""
    numbers = [int(n) for group in re.findall(r"\[([\d,\s]+)\]", answer) for n in re.findall(r"\d+", group)]
    return list(dict.fromkeys(numbers))


class RAG:
    def __init__(self, top_k: int = 5, llm_model: str = llm.LLM_MODEL, check: bool = False):
        self.llm_model = llm_model
        self.check = check  # ask the LLM "yes or no" for every product before answering
        llm.load(llm_model)  # first the LLM ...
        from semantic_search.levels.level7.semantic import SemanticSearch  # ... then the search, which loads FAISS

        self.search = SemanticSearch(load_products(AMAZON_PRODUCTS_PATH))  # gte-small, the Level 6 winner
        self.top_k = top_k

    def is_good_result(self, question: str, p: Product) -> bool:
        reply = llm.chat(check_messages(question, p), max_new_tokens=3, model_name=self.llm_model)
        return reply.strip().lower().startswith("yes")

    def ask(self, question: str) -> dict:
        start = time.perf_counter()
        searched = [p for _, p in self.search.search(question, self.top_k)]
        search_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        judged = [self.is_good_result(question, p) for p in searched] if self.check else [True] * len(searched)
        products = [p for p, good in zip(searched, judged) if good]  # only these go into the answer
        if products:
            answer = llm.chat(build_messages(question, products), model_name=self.llm_model)
        else:
            answer = NOTHING_FOUND  # every product was rejected: don't let the LLM force an answer
        llm_s = time.perf_counter() - start

        cited = cited_numbers(answer)
        return {
            "question": question,
            "searched": searched,  # what the search found
            "judged": judged,      # the LLM's yes (True) / no (False) for each searched product
            "products": products,  # what the answer was allowed to use
            "answer": answer,
            "cited": [n for n in cited if 1 <= n <= len(products)],
            "invalid_citations": [n for n in cited if not 1 <= n <= len(products)],  # numbers not in the list
            "search_ms": search_ms,
            "llm_s": llm_s,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="How many products to give the LLM")
    parser.add_argument("--llm", default=llm.LLM_MODEL, help="LLM to use, e.g. Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--check", action="store_true", help="Ask the LLM yes/no for each product before answering")
    args = parser.parse_args()

    result = RAG(args.top_k, args.llm, args.check).ask(args.question)
    print(f"\nSearch found ({result['search_ms']:.0f} ms):")
    for p, good in zip(result["searched"], result["judged"]):
        print(f"  {'kept   ' if good else 'removed'}  {p.title[:90]}")
    print("Products given to the answer:")
    for i, p in enumerate(result["products"], start=1):
        print(f"  [{i}] {p.title[:90]}")
    print(f"\nAnswer ({result['llm_s']:.0f} s):\n{result['answer']}\n")
    print(f"cited products: {result['cited'] or 'none'}"
          + (f" | INVALID numbers: {result['invalid_citations']}" if result["invalid_citations"] else ""))


if __name__ == "__main__":
    main()
