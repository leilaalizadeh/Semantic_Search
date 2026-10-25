"""Level 7, Step 3: measure RAG on our 20 labeled test queries.

For each query:
  - relevant in prompt: how many of the products given to the LLM are relevant (search quality)
  - recommended:        the products the answer picks, cited by number [2] or by name
  - answer precision:   how many recommended products are relevant (LLM quality)
  - made-up:            bracketed numbers or names that are not in the product list
  - refusal:            when no relevant product reached the prompt, did the answer say so?

Run:  uv run python -m semantic_search.levels.level7.rag_eval
      uv run python -m semantic_search.levels.level7.rag_eval --check                    (LLM checks each product)
      uv run python -m semantic_search.levels.level7.rag_eval --check --llm Qwen/Qwen2.5-3B-Instruct   (on a GPU)
Saves every answer to data/eval/rag_results*.json
"""

import argparse
import json
import re
import time

from semantic_search.levels.level7 import llm
from semantic_search.levels.level7.rag import RAG, cited_numbers  # loads the LLM before FAISS
from semantic_search.levels.level7.labels import EVAL_DIR, load_labels
from semantic_search.levels.level7.queries import QUERIES

RESULTS_PATH = EVAL_DIR / "rag_results.json"
REFUSAL_WORDS = ["nothing suitable", "no suitable", "none of", "not have", "doesn't have", "does not have",
                 "don't have", "do not have", "no product", "unfortunately", "not find", "isn't any", "aren't any"]


def words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def recommended_positions(answer: str, products) -> list[int]:
    """Positions (1 = first) of products the answer recommends: cited as [2], or named.

    A product counts as named when the first 3 words of its title appear in the answer.
    """
    found = [n for n in cited_numbers(answer) if 1 <= n <= len(products)]
    answer_words = f" {words(answer)} "
    for i, p in enumerate(products, start=1):
        title_start = " ".join(words(p.title).split()[:3])
        if i not in found and title_start and f" {title_start} " in answer_words:
            found.append(i)
    return sorted(found)


def made_up_mentions(answer: str, products) -> list[str]:
    """Bracketed text like [7] or [Some Product] that matches nothing in the list."""
    titles = [words(p.title) for p in products]
    made_up = []
    for inside in re.findall(r"\[([^\]]+)\]", answer):
        numbers = re.findall(r"\d+", inside)
        if re.fullmatch(r"[\d,\s]+", inside):
            made_up += [n for n in numbers if not 1 <= int(n) <= len(products)]
        elif not any(t.startswith(words(inside)[:25]) for t in titles):
            made_up.append(inside)
    return made_up


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", default=llm.LLM_MODEL)
    parser.add_argument("--check", action="store_true", help="LLM says yes/no for each product before answering")
    args = parser.parse_args()

    rag = RAG(llm_model=args.llm, check=args.check)
    truth = load_labels("leila")
    judge = {"yes_relevant": 0, "yes_not_relevant": 0, "no_relevant": 0, "no_not_relevant": 0}
    rows = []
    for item in QUERIES:
        start = time.perf_counter()
        result = rag.ask(item["query"])
        labels = truth[item["query"]]
        for p, good in zip(result["searched"], result["judged"]):  # the yes/no decisions vs our labels
            judge[f"{'yes' if good else 'no'}_{'relevant' if labels.get(str(p.id)) == 1 else 'not_relevant'}"] += 1
        relevant = [labels.get(str(p.id)) == 1 for p in result["products"]]
        picked = recommended_positions(result["answer"], result["products"])
        rows.append({
            "query": item["query"],
            "type": item["type"],
            "relevant_in_prompt": sum(relevant),
            "recommended": picked,
            "recommended_relevant": sum(relevant[i - 1] for i in picked),
            "made_up": made_up_mentions(result["answer"], result["products"]),
            "refused": any(w in result["answer"].lower() for w in REFUSAL_WORDS),
            "answer": result["answer"],
            "products": [{"title": p.title, "relevant": r} for p, r in zip(result["products"], relevant)],
        })
        r = rows[-1]
        print(f"{item['query'][:32]:<33} relevant in prompt {r['relevant_in_prompt']}/5 | "
              f"recommended {len(picked)} ({r['recommended_relevant']} relevant)"
              f"{' | REFUSED' if r['refused'] else ''}{' | made up: ' + str(r['made_up']) if r['made_up'] else ''}"
              f"  [{time.perf_counter() - start:.0f}s]", flush=True)

    short = args.llm.split("/")[-1]
    path = RESULTS_PATH if not args.check and args.llm == llm.LLM_MODEL else \
        RESULTS_PATH.with_name(f"rag_results_{short}{'_check' if args.check else ''}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)

    answered = [r for r in rows if r["relevant_in_prompt"] > 0]
    nothing = [r for r in rows if r["relevant_in_prompt"] == 0]
    recommended = sum(len(r["recommended"]) for r in rows)
    print(f"\n=== RAG on {len(rows)} queries, LLM {short}{', with product check' if args.check else ''} ===")
    if args.check:
        print(f"product check: said yes to {judge['yes_relevant']} relevant and {judge['yes_not_relevant']} not relevant, "
              f"no to {judge['no_relevant']} relevant and {judge['no_not_relevant']} not relevant products")
    print(f"search: relevant products in the prompt, average {sum(r['relevant_in_prompt'] for r in rows) / len(rows):.1f} of 5")
    print(f"answers: {recommended} products recommended, "
          f"{sum(r['recommended_relevant'] for r in rows)} of them relevant "
          f"({sum(r['recommended_relevant'] for r in rows) / max(recommended, 1):.0%} answer precision)")
    print(f"queries with no recommendation recognized: {sum(not r['recommended'] for r in answered)} of {len(answered)}")
    print(f"made-up mentions: {sum(len(r['made_up']) for r in rows)}")
    print(f"no relevant product in the prompt: {len(nothing)} queries; the answer refused in "
          f"{sum(r['refused'] for r in nothing)} and still recommended something in "
          f"{sum(bool(r['recommended']) for r in nothing)}")
    print(f"Saved all answers to {path}")


if __name__ == "__main__":
    main()
