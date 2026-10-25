"""The test queries for evaluation.

type tells us what each query tests:
  intent  - the query describes a need, not a product ("my car is dirty")
  simple  - a normal product search ("yoga mat")
  exact   - brand names or exact words matter ("Columbia jacket")
  tricky  - negation, which embedding models often get wrong ("without laces")
"""

QUERIES = [
    {"query": "going camping this weekend", "type": "intent"},
    {"query": "i have cold keep me warm", "type": "intent"},
    {"query": "something to fix a leaking sink", "type": "intent"},
    {"query": "my car is dirty", "type": "intent"},
    {"query": "my back hurts when I sit all day", "type": "intent"},
    {"query": "gift for a 6 year old", "type": "intent"},
    {"query": "I want to learn music", "type": "intent"},
    {"query": "my kitchen is too messy", "type": "intent"},
    {"query": "running shoes", "type": "simple"},
    {"query": "rain jacket", "type": "simple"},
    {"query": "yoga mat", "type": "simple"},
    {"query": "office chair", "type": "simple"},
    {"query": "dog toy", "type": "simple"},
    {"query": "coffee mug", "type": "simple"},
    {"query": "garden tools", "type": "simple"},
    {"query": "Columbia jacket", "type": "exact"},
    {"query": "stainless steel water bottle", "type": "exact"},
    {"query": "LED light", "type": "exact"},
    {"query": "phone charger", "type": "exact"},
    {"query": "shoes without laces", "type": "tricky"},
]
