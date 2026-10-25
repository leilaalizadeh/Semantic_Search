"""Level 7, Step 1: a small LLM that writes text, running on the laptop.

An LLM (large language model) continues a conversation: we give it messages
("system" = instructions, "user" = the question) and it writes the "assistant" answer,
one word piece (token) at a time.

Run:  uv run python -m semantic_search.levels.level8.llm "What should I take on a camping trip?"

The model is one setting (LLM_MODEL). The 1.5B model fits on an 8 GB laptop; on a GPU
(for example Colab) a bigger one like "Qwen/Qwen2.5-3B-Instruct" writes even better answers.
"""

import argparse
import time
from functools import cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1.5 billion parameters, ~3 GB. It fits on an 8 GB laptop and follows instructions much better
# than the 0.5B model, which only copied the product list. About 15-20 seconds per answer.
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@cache
def load(model_name: str = LLM_MODEL):
    # NVIDIA GPU (for example on Colab), else the Mac's GPU (MPS), else the CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # float16 = half the memory of float32; GPUs handle it well, CPUs don't
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32 if device == "cpu" else torch.float16)
    return tokenizer, model.to(device).eval()


def chat(messages: list[dict], max_new_tokens: int = 300, model_name: str = LLM_MODEL) -> str:
    """messages = [{"role": "system" or "user", "content": "..."}]. Returns the assistant's answer."""
    tokenizer, model = load(model_name)
    # The chat template turns the messages into the exact text format the model was trained on
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # stop after this many tokens (about 0.75 words each)
            do_sample=False,                # always pick the most likely next token: same answer every time
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]  # only the answer, not our prompt
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def chat_batch(conversations: list[list[dict]], max_new_tokens: int = 60, model_name: str = LLM_MODEL) -> list[str]:
    """Like chat(), but for many conversations at once: much faster on a GPU."""
    tokenizer, model = load(model_name)
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in conversations]
    # Prompts have different lengths, so the shorter ones get padding. It must be on the LEFT:
    # the model writes after the last token, and that has to be the end of the prompt, not padding.
    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output[:, inputs["input_ids"].shape[1]:]
    return [t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--model", default=LLM_MODEL)
    args = parser.parse_args()

    start = time.perf_counter()
    load(args.model)
    print(f"model loaded in {time.perf_counter() - start:.0f}s")
    start = time.perf_counter()
    answer = chat([{"role": "user", "content": args.question}], model_name=args.model)
    print(f"answer in {time.perf_counter() - start:.0f}s:\n\n{answer}")


if __name__ == "__main__":
    main()
