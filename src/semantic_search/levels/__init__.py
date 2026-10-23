"""Frozen copies of the code of each level, taken from the git tags.

  levels/level1/  <- tag Level1  (30 products, numpy search)
  levels/level2/  <- tag level2  (Amazon data, numpy search)
  levels/level3/  <- tag level3  (cache, FAISS Flat / HNSW)

Only two things differ from the tagged files:
  - imports point to the level's own folder (semantic_search.levels.levelN...)
  - parents[2] -> parents[4], because the files are two folders deeper
tests/test_levels.py checks this against git.

Don't edit finished levels. A new level starts as a copy of the previous one
(e.g. levels/level4/ copied from level3/) and the commands in pyproject.toml
are pointed at it.
"""
