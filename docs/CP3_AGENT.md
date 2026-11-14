# CP3 — Real Gemini tool-calling agent with Day 6 pgvector

**Implemented on 2026-11-14.** Uses the real Gemini GenerateContent REST API and the existing read-only Day 6 vector index. This is a working dense-agent prototype; lexical/hybrid experiments and full benchmarking belong to CP4+.

## Architecture

1. `scripts/cp3_run.py` selects a single ad-hoc or CP2 **dev** question using the no-gold projection (`scripts/cp2_validate.py::agent_inputs`). The CLI deliberately has **no test mode**.
2. `src/gemini_client.py` calls the real Gemini `models/{model}:generateContent` endpoint with function declaration `search_documents(query)`. First turn `ANY` forces retrieval, later turns `AUTO` can request additional searches; after hitting the limit, the final response has tools disabled. All tool calling is handled by the application; no pretend tool responses are generated.
3. `src/agent.py` validates the tool arguments, invokes `Day6DenseRetriever` (`src/day6_adapter.py`) using matching normalized 384-d `all-MiniLM-L6-v2` embeddings, then sends ranked/deduplicated source blocks back to Gemini. All SQL is read-only and preserves source URL, physical PDF page, chunk ID, chapter, score, and passage text.
4. Each response is classified `answered`, `abstain`, or `error`; citations must be from the selected context (supports separate `[id]` and grouped `[id1, id2]` syntax). One no-citation repair turn may be attempted. **Syntactically valid citation IDs do not prove factual grounding**: CP5+ must independently check evidence support.
5. Each query writes a JSON trace including model/tool events, retrieved candidates, final context, draft answer(s), citation metadata, number of calls, timing, token usage and safe error category. Credentials/DB connection strings are not logged. API dollar cost remains `null`: obtaining tokens is not a verified price calculation.

## Configuration / Windows Git Bash

Use the same running Docker service and Python virtualenv established in CP1. Store secrets in **Lab 7 `.env` locally**, never commit it. Copy `.env.example` then set `GEMINI_API_KEY` in the ignored `.env`; `GEMINI_MODEL` defaults to `gemini-3.5-flash-lite`. Google API access and quotas may vary between accounts. `DATABASE_URL` is inherited from the existing Day 6 `starter/.env` if not explicitly set in the environment.

```bash
cd /e/code/Vin/lab6/K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion/starter
docker compose -p day6_student_starter up -d --wait
cd /e/code/Vin/lab7/K5-L3A-Day07-NguyenVanMinh-2A202603551-RetrievalContext

# One non-benchmark question
/e/code/Vin/lab6/K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion/starter/.venv/Scripts/python.exe scripts/cp3_run.py --question "What is a variable in Python?"

# One dev answerable, one dev unanswerable
/e/code/Vin/lab6/K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion/starter/.venv/Scripts/python.exe scripts/cp3_run.py --dev-id D7-D06
/e/code/Vin/lab6/K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion/starter/.venv/Scripts/python.exe scripts/cp3_run.py --dev-id D7-D16

# Optional: all dev (multiple real LLM calls, subject to quota)
# .../.venv/Scripts/python.exe scripts/cp3_run.py --dev-all

# Unit tests (mocked provider; not proof of production quality)
PYTHONUTF8=1 py -m unittest discover -s tests -v
```

Trace files are timestamped in `starter/results/cp3/`, ignored by Git. Runtime trace and examples must be explicitly included in a shareable submission artifact when permitted (no secrets). Do not label mock test results as real agent benchmarks.

## Live dev smoke runs and observations

| Dev case | Expectation | Observed status | Search calls | Model calls | Duration | Observation |
|---|---|---|---:|---:|---:|---|
| D7-D06 | Answerable, definition of variable | `answered` | 1 | 2 | 15.88 seconds | Correct definition; cited valid context IDs at PDF pages 31 and 36 |
| D7-D16 | No evidence for list comprehensions | `abstain` | 2 | 3 | 17.08 seconds | Returned explicit lack-of-evidence answer, no fabricated citation |

**Limitations:** These are two exploratory dev smoke cases, **not** quality metrics across the complete 17-dev or 7-test benchmark. Cold model loading, network time and provider quotas affect latency. Model-generated content may still be unsupported even with valid citation IDs. The initial 400 response was caused by setting `allowedFunctionNames` in `AUTO` mode; fixed by including it only with `ANY`. Initial grouped `[id1, id2]` citations needed parser support. Previous failed runs are retained locally with `status=error` or `abstain` rather than erased or counted as successful. No test queries or gold were supplied to the model.

## Files

- `src/gemini_client.py` — native Gemini tool declaration, request/response client with error handling.
- `src/agent.py` — bounded tool-calling loop, selection of context, source/citation validation and traces.
- `scripts/cp3_run.py` — CLI for one question or dev split (never test).
- `tests/test_cp3_agent.py` — deterministic agent protocol tests (mock provider/retriever).
- `.env.example` — configuration template without real keys.
- `starter/results/cp3/*.json` — actual run traces generated locally, ignored by Git.

## Next phase

CP4: add lexical BM25/TF-IDF and hybrid RRF, implement an explicit strategy selector for dense/lexical/hybrid. CP5: benchmark all three on the **same dev** with trace/evidence measurements, manual grounding/citation review, then CP6 improvement selection; final test only after freezing the chosen configuration.

## Traces for grading

Two verified live dev runs have also been copied into the tracked `artifacts/cp3_smoke/` folder: `D7-D06.json` (answer with citations) and `D7-D16.json` (abstention). The export script `scripts/cp3_export_smoke.py` selects only successful live statuses and checks for obvious secret patterns. The complete original timestamped traces remain in ignored `starter/results/cp3/`. Evidence must still be manually reviewed when grading; exporting trace files does not turn them into scored benchmarks.
