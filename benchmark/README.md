# Frozen benchmark (Day 7, CP2)

- `dev.json`: 17 cases, 15 answerable + 2 unanswerable.
- `test.json`: 7 new held-out cases, 5 answerable + 2 unanswerable. Not used for model selection or dev tuning.
- `split_lock.json`: canonical checksums and split policy.
- `scripts/cp2_validate.py`: read-only checking and agent-safe input projection.

Do not pass gold, expected answers, evidence, absence reasons or the lock file to any LLM/tool. See `docs/CP2_BENCHMARK.md`.
