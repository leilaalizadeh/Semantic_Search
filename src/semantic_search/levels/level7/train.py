"""Level 6, Step 2: fine-tune the embedding model on the generated (query, product) pairs.

Run:  uv run python -m semantic_search.levels.level7.train --max-steps 20                 (quick test)
      uv run python -m semantic_search.levels.level7.train --batch-size 16              (Step 2: T5 questions)
      uv run python -m semantic_search.levels.level7.train --batch-size 16 --data shopping  (Step 3)
Saves the trained model to models/bge-small-finetuned or models/bge-small-shopping

How it learns (MultipleNegativesRankingLoss):
  a batch has 32 pairs (query 1, product 1) ... (query 32, product 32).
  For query 1, product 1 is the right answer and products 2..32 are wrong answers.
  The loss is low when each query is closest to its own product, so training pulls
  right pairs together and pushes wrong pairs apart.
"""

import argparse
import json
import random

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from semantic_search.levels.level7.data import AMAZON_PRODUCTS_PATH, DATA_DIR, load_products
from semantic_search.levels.level7.generate_queries import TRAIN_PATH

BASE_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # BGE's query instruction
MODELS_DIR = DATA_DIR.parent / "models"
OUTPUT_DIR = MODELS_DIR / "bge-small-finetuned"           # Step 2: trained on T5 questions
SHOPPING_OUTPUT_DIR = MODELS_DIR / "bge-small-shopping"    # Step 3: trained on shopping-style queries
DATASETS = {  # --data name -> (training file, where to save the model)
    "t5": (TRAIN_PATH, OUTPUT_DIR),
    "shopping": (TRAIN_PATH.parent / "shopping_queries.jsonl", SHOPPING_OUTPUT_DIR),
}


def load_pairs(path=TRAIN_PATH, validation_share: float = 0.05, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Training and validation data as (anchor = query, positive = product text).

    The split is by product: all queries of a product go to the same side, so the
    validation check uses products the model never saw during training.
    """
    products = {str(p.id): p for p in load_products(AMAZON_PRODUCTS_PATH)}
    pairs = [json.loads(line) for line in open(path)]

    ids = sorted(products)
    random.Random(seed).shuffle(ids)
    validation_ids = set(ids[:int(len(ids) * validation_share)])

    def to_dataset(rows):
        return Dataset.from_dict({
            "anchor": [QUERY_PREFIX + r["query"] for r in rows],  # same instruction as when searching
            "positive": [products[r["product_id"]].text for r in rows],
        })

    train = [r for r in pairs if r["product_id"] not in validation_ids]
    validation = [r for r in pairs if r["product_id"] in validation_ids]
    return to_dataset(train), to_dataset(validation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=float, default=1, help="How many times to go through all training pairs")
    parser.add_argument("--batch-size", type=int, default=32, help="Pairs per batch (more = more wrong answers per query, more memory)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after this many batches (for a quick test)")
    parser.add_argument("--max-length", type=int, default=128, help="Cut texts after this many tokens (less = less memory)")
    parser.add_argument("--cpu", action="store_true", help="Train on the CPU instead of the Mac's GPU (MPS)")
    parser.add_argument("--data", choices=DATASETS, default="t5", help="Which training queries to use")
    args = parser.parse_args()

    data_path, output_dir = DATASETS[args.data]
    train, validation = load_pairs(data_path)
    print(f"training pairs: {len(train)}, validation pairs: {len(validation)}")

    model = SentenceTransformer(BASE_MODEL)
    # Product texts are cut after this many tokens. Training needs much more memory than searching
    # (it also stores gradients), and 256 tokens filled an 8 GB Mac.
    model.max_seq_length = args.max_length
    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(MODELS_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,  # small steps: we only adjust a model that already works
        warmup_ratio=0.1,    # start with even smaller steps for the first 10% of training
        # A product with 3 queries could land 2 times in one batch; then its own other query
        # would count as a "wrong answer". This sampler keeps each product once per batch.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=25,
        save_strategy="no",  # we save the final model ourselves
        report_to="none",
        use_cpu=args.cpu,
        seed=42,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        loss=loss,
    )
    trainer.train()

    if args.max_steps == -1:
        model.save(str(output_dir))
        print(f"Saved the fine-tuned model to {output_dir}")


if __name__ == "__main__":
    main()
