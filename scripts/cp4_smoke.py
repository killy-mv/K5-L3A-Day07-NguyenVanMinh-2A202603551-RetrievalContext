"""CP4: real retrieval-only smoke comparison; no LLM or held-out test data.

Only two non-benchmark queries are used. This is not the CP5 evaluation.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.day6_adapter import Day6DenseRetriever, database_url, default_day6_root
from src.retrieval import BM25Retriever, HybridRRFRetriever, load_corpus

QUERIES = [
    {'id': 'SMOKE-01',
     'question': 'In Python, what happens when you evaluate a numerical expression?',
     'filters': None},
    {'id': 'SMOKE-02',
     'question': 'How can a turtle move and turn repeatedly to draw a geometric shape?',
     'filters': {'chapter': 'chapter_4'}},
]


def main():
    corpus = load_corpus(ROOT / 'starter/data/chunks.json')
    url = database_url(default_day6_root())
    lexical = BM25Retriever(corpus)
    dense = Day6DenseRetriever(url)
    hybrid = HybridRRFRetriever(dense, lexical, candidate_k=12, rrf_k=60)
    methods = {'dense': dense, 'lexical': lexical, 'hybrid': hybrid}
    cases = []
    for case in QUERIES:
        row = {'id': case['id'], 'question': case['question'],
               'filters': case['filters'], 'retrievals': {}}
        for label, method in methods.items():
            start = time.perf_counter()
            hits = method.search(case['question'], filters=case['filters'], top_k=3)
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            if not hits:
                raise RuntimeError(f'{case["id"]} has no {label} candidates')
            for h in hits:
                if h['id'] not in {c['id'] for c in corpus}:
                    raise ValueError('Unknown chunk ID in retrieval')
                if case['filters'] and h['metadata'].get('chapter') != case['filters']['chapter']:
                    raise ValueError('Trusted filter violated')
            row['retrievals'][label] = {
                'elapsed_ms': elapsed, 'top3': [
                    {'id': h['id'], 'source_id': h['source_id'],
                     'source': h['source'], 'page': h['page'],
                     'chapter': h['metadata'].get('chapter'),
                     'score': round(h['score'], 6),
                     'score_type': h.get('score_type', 'cosine_similarity'),
                     'raw_score': round(h['raw_score'], 6) if 'raw_score' in h else None,
                     'ranks': h.get('ranks'),
                     'component_scores': h.get('component_scores'),
                    } for h in hits],
            }
            print(f"{case['id']} {label}: {[(h['id'], h['page'], round(h['score'], 4)) for h in hits]}", flush=True)
        cases.append(row)
    result = {
        'stage': 'CP4', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'non-benchmark smoke only; no dev/test answers, no gold',
        'corpus': 'Think Python chapters 1-4, 160 strategy-B chunks from Day 6',
        'methods': ['dense', 'lexical', 'hybrid'],
        'dense': {'model': 'all-MiniLM-L6-v2', 'dimensions': 384, 'store': 'pgvector cosine'},
        'lexical': {'model': 'BM25', 'k1': lexical.k1, 'b': lexical.b,
                    'tokenizer': 'NFC unicode word tokens, casefold'},
        'hybrid': {'fusion': 'RRF', 'rrf_k': hybrid.rrf_k,
                   'candidate_k_per_retriever': hybrid.candidate_k},
        'benchmark_test_evaluated': False,
        'cases': cases,
    }
    output = ROOT / 'artifacts/cp4_smoke.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Wrote', output.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
