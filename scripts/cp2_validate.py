"""Validate frozen CP2 benchmark without updating it or running test retrieval."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cp2_build import canonical_sha, validate_benchmark


def validate_files(root=ROOT):
    chunks = json.loads((root / 'starter/data/chunks.json').read_text(encoding='utf-8'))
    dev = json.loads((root / 'benchmark/dev.json').read_text(encoding='utf-8'))
    test = json.loads((root / 'benchmark/test.json').read_text(encoding='utf-8'))
    lock = json.loads((root / 'benchmark/split_lock.json').read_text(encoding='utf-8'))
    actual = {'corpus': canonical_sha(chunks), 'dev': canonical_sha(dev), 'test': canonical_sha(test)}
    if actual != lock.get('sha256_canonical'):
        raise ValueError('CP2 corpus, questions or split have changed since freeze')
    counts = validate_benchmark(chunks, dev, test)
    if counts != lock['counts']:
        raise ValueError('CP2 split count mismatch')
    return counts


def agent_inputs(split='dev', root=ROOT):
    """Agent-only questions. No gold, model answers or absence rationales."""
    if split not in ('dev', 'test'):
        raise ValueError('split must be dev/test')
    questions = json.loads((root / f'benchmark/{split}.json').read_text(encoding='utf-8'))
    return [{key: q[key] for key in ('id', 'question', 'filter') if key in q}
            for q in questions]


if __name__ == '__main__':
    print('Validated frozen CP2:', validate_files())
