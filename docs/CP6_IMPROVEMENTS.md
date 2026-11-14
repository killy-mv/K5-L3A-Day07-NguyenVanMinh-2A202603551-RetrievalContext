# CP6 — Dev experiments, error analysis and frozen selection

**Completed 14 November 2026**, using only the 17 locked dev questions (15 answerable, 2 unanswerable). The frozen test benchmark has not been evaluated. The initial experiment registration and test SHA are in `artifacts/cp6/experiment_plan.json`.

## Experiments and measurements

Two independently motivated experiments were registered *before* running: E1 changes retrieval rank fusion weights; E2 changes the Gemini synthesis prompt while keeping retrieval unchanged. When Gemini became unavailable, an explicitly post-hoc local retrieval experiment E2b was also run. Do **not** present E2b as a successful E2 prompt experiment.

| Configuration | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Interpretation |
|---|---:|---:|---:|---|
| Dense baseline, top3/context3800 (CP5) | 10/15 | 13/15 | 13/15 | Reference for E2b |
| E2b Dense top5/context5000 | 10/15 | 13/15 | **14/15** | Hit@5=14/15; context expansion helps one case but costs more context |
| **Balanced Hybrid RRF 1:1 (CP5)** | **12/15** | **15/15** | **15/15** | Best measured dev evidence coverage |
| E1 Weighted RRF Dense 0.7/BM25 1.3 | 11/15 | 14/15 | 14/15 | Reject: worse than balanced Hybrid |

E1 was a real execution against the same Day6 vector index and BM25 corpus, with RRF k=60, 12 candidates per component, output top3 and 3800 context characters unchanged. Giving lexical more weight pushed D7-D01's correctly sourced definition of *program* outside top3. This **disproves E1's improvement hypothesis** on dev. Per-question changes and traces: `artifacts/cp6/e1_weighted_rrf_{summary,traces}.json`.

E2 (the registered generic *complete answer* instruction in `src/agent.py::SYSTEM_COMPLETE`) was executed with the same balanced Hybrid and Gemini `gemini-3.5-flash-lite`, but four dev cases produced only provider HTTP 503 or timeout, including an attempted retry. No valid model answer was generated, so **no answer-grade improvement can be claimed**. The error audit is `artifacts/cp6/e2_prompt_provider_audit.json`; original failure traces remain local under `starter/results/cp6/e2_agent/` (Git-ignored). Model failure is neither abstention nor hallucination.

E2b is an additional post-blocker experiment, not preregistered E2: Dense top3/context3800 became Dense top5/context5000. This included the gold chunk for D7-D07 in the final context, improving coverage 13/15 to 14/15. D7-D12 was still missed. **Hit@3 stayed 13/15**, so we do not compare Dense Hit@5 against Hybrid Hit@3 as the same metric. No Gemini answer generation was claimed for E2b. Evidence: `artifacts/cp6/e2b_dense_context_expansion_{summary,traces}.json`.

## Final configuration selected and locked BEFORE test

Use the original, measured **balanced Hybrid RRF** from CP5: MiniLM-L6-v2 normalized 384-d embeddings + pgvector Day6 strategy B, BM25 (k1=1.5, b=0.75), RRF k=60, component weights 1.0/1.0, candidate_k=12 per method, top_k=3, context budget=3800 characters; Gemini `gemini-3.5-flash-lite`, original CP3 system prompt (NOT the unvalidated E2 prompt), temperature=0.1, max output tokens=1200, max_searches=3.

Reasoning: balanced Hybrid has the best measured Evidence Hit@3 and context evidence on dev (15/15), 17/17 completed CP5 Gemini dev runs, and correct dev abstention status (2/2). E1 harmed retrieval. E2 could not be evaluated due to provider failure. E2b improved Dense but remained inferior to Hybrid and would increase LLM context tokens. The CP5 grounded/citation grades are **AI-assisted draft reviews (`human_confirmed=false`)**, not independent human scores; none of these outcomes predicts test accuracy.

`artifacts/cp6/final_config_lock.json` contains the exact selected settings, canonical CP2 corpus/dev/test SHA256s, checksums of core source code, and decision evidence. The script `py scripts/cp6_freeze.py` only creates the lock if missing and otherwise **verifies without overwriting it**. If source or benchmark changes, it fails. **Test cases used to select = 0.** CP7 must verify this lock before evaluating test once, and must not tune on test.

## Reproduction (Git Bash, repo root)

```bash
DAY6PY=/e/code/Vin/lab6/K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion/starter/.venv/Scripts/python.exe
$DAY6PY scripts/cp6_e1_weighted_rrf.py
# E2 prompt was blocked by Gemini HTTP 503/timeout; audit errors, do not invent scores.
py scripts/cp6_provider_audit.py
$DAY6PY scripts/cp6_e2_context_expansion.py
py scripts/cp6_freeze.py
PYTHONUTF8=1 py -m unittest discover -s tests -v
```

The original CP6 plan, actual variants, failed API traces, negative results, alternative measurements, and locked selection are retained separately for reproducibility. Remaining work: CP7 held-out test, independent label confirmation where required, complete report/submission.
