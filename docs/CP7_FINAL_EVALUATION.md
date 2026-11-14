# CP7 — Final held-out test evaluation (completed)

**Date:** 14 November 2026. **All required CP7 test executions completed** with the previously frozen model `gemini-3.5-flash-lite`, using the new local API key. The results below are grounded in actual saved tool/model traces. The CP6 frozen configuration, source code, and dev/test split were verified before evaluating test and were not changed in response to test results.

## Setup and fairness

- Original document: Allen B. Downey, *Think Python 2nd Edition*, CC BY-NC 3.0. 35 indexed content pages (chapters 1–4), 160 paragraph-B chunks.
- Same 7 held-out CP2 test questions for both models: **5 answerable and 2 unanswerable**. These are new Day7 test questions, not the previously inspected Day6 test set.
- Dense baseline: `sentence-transformers/all-MiniLM-L6-v2` normalized 384-dimensional embeddings and Day6 PostgreSQL/pgvector cosine search (strategy B).
- Frozen Hybrid: same Dense plus BM25 (k1=1.5, b=0.75), balanced RRF 1.0/1.0, k=60, 12 candidate chunks per retriever.
- Both produce top-3 chunks, context budget 3800 characters. Agent uses `gemini-3.5-flash-lite`, original CP3 system prompt, temperature=0.1, max 1200 output tokens, maximum three search calls per question. Prompt and configuration were chosen from dev and frozen in `artifacts/cp6/final_config_lock.json` **before test**.
- No gold/expected answers were provided to the Agent; evaluator read them afterward to score source/page/quote alignment and responses.

## Retrieval results — complete 7-case test

| Metric (only 5 answerable questions count) | Dense baseline | Selected Hybrid RRF |
|---|---:|---:|
| Hit@1 | 3/5 (60%) | 3/5 (60%) |
| Evidence Hit@3 | **5/5 (100%)** | **5/5 (100%)** |
| Context Evidence Rate | **5/5 (100%)** | **5/5 (100%)** |
| Context grade 2 (source-assisted draft) | 5/5 | 5/5 |

A retrieval hit requires the correct source ID, physical PDF page and literal gold evidence span in the returned chunk. Context Evidence Rate requires the full gold span in the packed context. Both retrieval systems supplied the required evidence on this small test; **Hybrid did not beat Dense on test**, even though Hybrid had higher Evidence Hit@3 on dev (15/15 versus Dense 13/15). Selection remained unchanged. Context grade 2 means the reference source span was present; these grades are AI-assisted and require human confirmation if the course requires an independent reviewer. Evidence: `artifacts/cp7/retrieval_{dense,hybrid}_{summary,traces}.json`, `context_{dense,hybrid}_review_assisted.json`.

## Gemini Agent test — complete 7/7 for both strategies

| Indicator | Dense | Selected Hybrid |
|---|---:|---:|
| Final test cases processed | **7/7** | **7/7** |
| Answered | 5/5 answerable | 5/5 answerable |
| Correct abstention | 2/2 unanswerable | 2/2 unanswerable |
| False abstention | 0/5 | 0/5 |
| Retrieval tool use | 7/7 | 7/7 |
| Search calls | 10 | 10 |
| Model calls | 17 | 17 |
| Provider errors **in final accepted traces** | **0** | **0** |
| Grounded answer grade 2 — source-assisted draft | 5/5 | 5/5 |
| Answer citation coverage | 5/5 | 5/5 |
| Mean end-to-end latency over final seven traces | 53.14 seconds | 20.27 seconds |
| Provider-reported tokens (final seven traces) | 15,528 | 15,705 |

All five answerable test outputs in both configurations provide specific chunk citations and answer the relevant questions on execution flow, local scope, stack frames, circle interface, and encapsulation/generalization; both abstain from syntax not covered in the indexed corpus (`async def`, `yield from`). Citation ID and page metadata are validated programmatically, and supporting passages were compared against expected answers in an **AI-assisted review draft**, recorded at `artifacts/cp7/agent_{dense,hybrid}_reviews_assisted.json`. These grades have `human_confirmed=false`: do not misrepresent them as independently human-verified.

The generated JSON summaries and complete model/tool traces are at `artifacts/cp7/agent_{dense,hybrid}_{summary,traces}.json`. All seven current cases per method are valid. The latency figures are observational, non-randomized and include different cold/warm conditions; they are not proof that Hybrid is intrinsically faster.

## Provider history and recovery

Some initial Gemini 3.5 requests hit HTTP 503/timeouts when the earlier API key was exhausted. **The user supplied a new key**, and only the three previously incomplete test cases were rerun with the same frozen model and configuration, under a request-level limit of **2 RPM**. All three reruns succeeded: Hybrid D7-T03, Dense D7-T02 and Dense D7-T04. The old provider failures are archived locally as `starter/results/cp7/agent/{strategy}/*_previous_error_*.json` (Git-ignored). They are preserved for audit, but there are **no unresolved errors in the final 7/7 test runs**. This is recovery from provider errors, not post-test hyperparameter tuning. Earlier Gemini 3.8 probes failed and are kept separately under `artifacts/cp7_supplemental_gemini38/`; they **do not contribute to any reported CP7 score**.

The actual historical dev and test model is `gemini-3.5-flash-lite`, matching the now-current local `.env`, `.env.example`, CP6 lock and all current CP7 test traces. API keys, database passwords and `.env` are not part of the submission.

## API cost, time and limitations

Provider-reported token counts and latency are in each trace. A clearly labeled **hypothetical paid-tier estimate** for the recorded CP5 + CP7 tokens is computed by `scripts/cp7_estimated_costs.py` and saved in `artifacts/cp7/estimated_api_costs.json` (approximately **US$0.0531** based on assumed rates). This is **not a verified bill or actual spending**: failed requests lacking usage, free-tier credits, CPU resources, cache rates and other projects are excluded; actual cost remains unknown (`api_cost_usd=null` in trace summaries).

The test contains only five answerable questions and two intentionally unanswerable questions, so apparent perfect evidence/answer metrics are statistically fragile. Gemini answer and citation scores are AI-assisted rather than independently human confirmed. The advanced bonus requiring a different 100+ page, multimodal document was **not attempted**; this is the complete basic Day7 lab.

## Reproduce without accidentally changing the test

```bash
# In the Lab7 repository root, with Day6 PostgreSQL running:
py scripts/cp2_validate.py
py scripts/cp6_freeze.py

# Offline: regenerate summaries from the already stored real test traces.
py scripts/cp7_offline_summary.py
py scripts/cp7_context_review.py
py scripts/cp7_mixed_models.py
py scripts/cp7_estimated_costs.py

# Verify code and frozen test integrity.
PYTHONUTF8=1 py -m unittest discover -s tests -v
```

The actual Agent tester is `scripts/cp7_test.py`; **re-running is unnecessary** because its existing seven-case traces are complete, and the script skips completed cases. Should new official evaluation be necessary, it enforces per-request pacing with `scripts/cp7_pacing.py`, stores each result immediately, never silently retries errors, and refuses to run a different model than CP6's lock. The old Gemini 3.8 supplementary scripts are diagnostic only and do not affect the frozen test.

## Grading files

- `artifacts/cp7/retrieval_dense_summary.json`, `retrieval_hybrid_summary.json` and corresponding `_traces.json`.
- `artifacts/cp7/agent_dense_summary.json`, `agent_hybrid_summary.json` and corresponding `_traces.json` and `_reviews_assisted.json`.
- `artifacts/cp7/context_dense_review_assisted.json`, `context_hybrid_review_assisted.json` and `context_review_comparison.json`.
- `artifacts/cp7/estimated_api_costs.json`.
- Primary full report: `report/individual/2A202603551_NguyenVanMinh.md`.
