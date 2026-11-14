"""Prepare Lab 7 corpus export and verify the existing Day 6 vector store.

Run from the Lab 7 root with the Day 6 virtual environment.
Does not load benchmark gold or run test-set evaluation.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.day6_adapter import (
    Day6DenseRetriever, audit_pgvector, database_url,
    default_day6_root, load_day6_chunks, validate_source,
)


def main():
    parser = argparse.ArgumentParser(description='Lab 7 CP1: verify and export Day 6 data')
    parser.add_argument('--day6-root', type=Path, default=default_day6_root())
    parser.add_argument('--output', type=Path, default=ROOT / 'starter' / 'data' / 'chunks.json')
    parser.add_argument('--smoke', action='store_true', help='Use real query embeddings and pgvector')
    args = parser.parse_args()

    day6 = args.day6_root.resolve()
    provenance = validate_source(day6)
    chunks = load_day6_chunks(day6)
    url = database_url(day6)
    audit = audit_pgvector(url, chunks)
    stats = {
        'stage': 'CP1', 'utc': datetime.now(timezone.utc).isoformat(),
        'source': provenance, 'index': audit,
        'exported_chunks': len(chunks),
        'unique_pages': len(set(c['page'] for c in chunks)),
        'chapters': sorted(set(c['metadata']['chapter'] for c in chunks)),
        'database_write_operations': 0,
        'benchmark_gold_used': False,
    }
    if args.smoke:
        retriever = Day6DenseRetriever(url)
        questions = [
            ('What is a function in Python?', None),
            ('How can turtle graphics draw a square?', {'chapter': 'chapter_4'}),
        ]
        checks = []
        for question, filters in questions:
            t0 = time.perf_counter()
            matches = retriever.search(question, filters=filters, top_k=3)
            elapsed = time.perf_counter() - t0
            if len(matches) != 3:
                raise AssertionError('Expected top-3 from real vector store')
            if filters and any(c['metadata'].get('chapter') != filters['chapter'] for c in matches):
                raise AssertionError('Chapter filter not respected')
            checks.append({
                'query': question, 'filters': filters,
                'elapsed_seconds': round(elapsed, 4),
                'top3': [{
                    'id': c['id'], 'page': c['page'], 'score': round(c['score'], 6),
                    'chapter': c['metadata']['chapter'], 'source': c['source'],
                } for c in matches],
            })
        stats['smoke_queries'] = checks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    results = ROOT / 'starter' / 'results'
    results.mkdir(parents=True, exist_ok=True)
    (results / 'cp1_summary.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
