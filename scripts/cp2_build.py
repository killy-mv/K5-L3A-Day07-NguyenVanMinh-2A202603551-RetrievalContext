"""Build and freeze Lab 7 benchmark. Only CP2 calls this to inspect gold.

Existing Day 6 *dev* questions are carried over; the Day 6 test questions
are excluded because their outcomes have already been inspected. The new
held-out test must not be used for development or retriever selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAY6 = ROOT.parent.parent / 'lab6' / 'K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion'
OUT = ROOT / 'benchmark'
SOURCE_ID = 'thinkpython2_pdf'

# These are intentionally held out; source pages are stable physical PDF pages.
# Anchors are short, literal text fragments in the *existing* Day 6 chunk export.
NEW_TEST = [
    dict(id='D7-T01', question='When a function is called, how does the flow of execution return to where it left off?',
         expected_answer='Execution jumps into the function body, executes it, and then returns to the caller.',
         page=43, anchor='A function call is like a detour in the flow of execution.', question_type='explanation'),
    dict(id='D7-T02', question='What happens to a local variable created inside a function after that function finishes?',
         expected_answer='It exists only within the function and is destroyed when the function terminates.',
         page=44, anchor='Whencat_twiceterminates, the variablecatis destroyed.', question_type='scope'),
    dict(id='D7-T03', question='In a stack diagram, what does each frame represent?',
         expected_answer='A function, with its name and the parameters and variables belonging to it.',
         page=45, anchor='Each function is represented by aframe.', question_type='definition'),
    dict(id='D7-T04', question='When designing circle(t, r), why should the radius be part of the interface but not necessarily the segment count n?',
         expected_answer='Radius specifies the circle requested by the caller, whereas n is an internal rendering detail.',
         page=55, anchor='nis\nless appropriate because it pertains to the details ofhowthe circle should be rendered.', question_type='comparison'),
    dict(id='D7-T05', question='What are the initial three steps of the encapsulation-and-generalization development plan?',
         expected_answer='Start with a small program without functions; encapsulate a working piece in a named function; generalize it by adding suitable parameters.',
         page=57, anchor='1. Start by writing a small program with no function definitions.', question_type='sequence'),
]

NEGATIVE = [
    dict(id='D7-D16', split='dev', question='According to the indexed chapters, how does Python evaluate a list comprehension with an if clause?',
         absence_reason='Chapters 1–4 in the indexed corpus do not teach list-comprehension syntax or its conditional evaluation.',
         check_terms=['list comprehension', 'comprehensions']),
    dict(id='D7-D17', split='dev', question='What syntax does the book give for declaring a Python class with an inherited base class?',
         absence_reason='The indexed chapters discuss basic values, functions, and turtle graphics, not class definition or inheritance syntax.',
         check_terms=['class definition', 'inheritance', 'subclass']),
    dict(id='D7-T06', split='test', question='According to the indexed chapters, what is the syntax for declaring an asynchronous coroutine with async def?',
         absence_reason='Asynchronous function declarations and async/await are outside the indexed chapters 1–4.',
         check_terms=['async def', 'await ']),
    dict(id='D7-T07', split='test', question='How does the yield from statement delegate iteration to a subgenerator according to the indexed book chapters?',
         absence_reason='Generators and the yield from statement are not covered by the indexed first four chapters.',
         check_terms=['yield from', 'subgenerator']),
]


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def flatten(s):
    """Normalize whitespace only; preserve exact case/characters from the corpus."""
    return ''.join(s.split())


def extract_exact(chunk_text, old_evidence):
    """Resolve a Day 6 whitespace-normalized quote to exact chunk substring.

    This is a source alignment, not an answer generation or a retrieval score.
    """
    lookup = flatten(old_evidence)
    compact = []
    positions = []
    for i, char in enumerate(chunk_text):
        if not char.isspace():
            compact.append(char)
            positions.append(i)
    joined = ''.join(compact)
    offset = joined.find(lookup)
    if offset >= 0:
        return chunk_text[positions[offset]:positions[offset+len(lookup)-1]+1]
    return None


def evidence_for_old(q, chunks):
    rows = [c for c in chunks if c['source_id'] == q['gold_source_id'] and c['page'] == q['gold_page']]
    for c in rows:
        match = extract_exact(c['text'], q['evidence_quote'])
        if match is not None:
            return c, match, None
    if q['id'] == 'Q10':
        # The Day 6 gold quote spans two B chunks. The second chunk contains
        # the most direct statement supporting the answer (why > what).
        for c in rows:
            if 'it is more useful to' in c['text']:
                match = extract_exact(c['text'], 'it is more useful to explainwhy.')
                if match is not None:
                    return c, match, 'Original Day 6 quote crosses chunk boundary; gold shortened to supporting source text.'
    raise ValueError(f"Cannot align Day 6 dev evidence for {q['id']} at physical page {q['gold_page']}")


def get_test_anchor(q, chunks):
    rows = [c for c in chunks if c['page'] == q['page'] and c['source_id'] == SOURCE_ID]
    matches = [(c, q['anchor']) for c in rows if q['anchor'] in c['text']]
    if len(matches) != 1:
        raise ValueError(f"Test anchor must resolve in exactly one chunk: {q['id']} found={len(matches)}")
    return matches[0]


def assemble(day6_questions, chunks):
    dev = []
    for old in day6_questions:
        if old['split'] != 'dev':
            continue
        chunk, quote, note = evidence_for_old(old, chunks)
        item = {
            'id': 'D7-D' + old['id'][1:], 'split': 'dev',
            'question_type': old.get('question_type', 'direct'),
            'question': old['question'], 'answerable': True,
            'expected_behavior': 'search_and_answer',
            'expected_answer': old['expected_answer'],
            'gold': [{'source_id': chunk['source_id'], 'page': chunk['page'],
                      'evidence': quote, 'chunk_id': chunk['id']}],
            'provenance': {'origin': 'Day 6 dev ' + old['id'],
                           'source_url': chunk['source']},
        }
        if note:
            item['evidence_alignment_note'] = note
        if old.get('filter'):
            item['filter'] = old['filter']
            item['filter_origin'] = old.get('filter_origin')
        dev.append(item)

    test = []
    for q in NEW_TEST:
        chunk, evidence = get_test_anchor(q, chunks)
        # Prefer an evidence span whose complete information supports the
        # answer, not merely a heading or unique search keyword.
        if q['id'] == 'D7-T01':
            evidence = chunk['text']
        elif q['id'] == 'D7-T02':
            evidence = chunk['text']
        elif q['id'] == 'D7-T03':
            evidence = chunk['text']
        elif q['id'] == 'D7-T04':
            evidence = chunk['text']
        elif q['id'] == 'D7-T05':
            evidence = chunk['text']
        test.append({
            'id': q['id'], 'split': 'test', 'question_type': q['question_type'],
            'question': q['question'], 'answerable': True,
            'expected_behavior': 'search_and_answer', 'expected_answer': q['expected_answer'],
            'gold': [{'source_id': SOURCE_ID, 'page': q['page'],
                      'evidence': evidence, 'chunk_id': chunk['id']}],
            'provenance': {'origin': 'new Lab 7 held-out test', 'source_url': chunk['source']},
        })
    for item in NEGATIVE:
        rec = {k: v for k, v in item.items() if k != 'check_terms'}
        rec.update(answerable=False, expected_behavior='search_and_abstain',
                   absence_check_scope='Day 6 indexed corpus: 160 chunks, four chapters',
                   absence_check_terms=item['check_terms'])
        (dev if item['split'] == 'dev' else test).append(rec)
    return dev, test


def validate_benchmark(chunks, dev, test):
    if len(chunks) != 160:
        raise ValueError('Unexpected corpus version; CP2 is locked to 160 B chunks')
    ids = [x['id'] for x in dev + test]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate query IDs')
    if (len(dev), len(test)) != (17, 7):
        raise ValueError('Required split counts: 17 dev, 7 test')
    if [sum(bool(q['answerable']) == v for q in part) for part in (dev, test) for v in (True, False)] != [15, 2, 5, 2]:
        raise ValueError('Invalid answerable/abstention counts')
    byid = {c['id']: c for c in chunks}
    for q in dev + test:
        if q['split'] not in ('dev', 'test') or not q['question'].strip():
            raise ValueError('Invalid question/split')
        if q['answerable']:
            if not q.get('expected_answer') or not q.get('gold'):
                raise ValueError(f"Missing gold for {q['id']}")
            for g in q['gold']:
                c = byid[g['chunk_id']]
                if not (c['source_id'] == g['source_id'] and c['page'] == g['page'] and g['evidence'] in c['text']):
                    raise ValueError(f"Evidence not in original source chunk: {q['id']}")
        else:
            if 'gold' in q or 'expected_answer' in q:
                raise ValueError('Negative example cannot contain answer gold')
            corpus_text = '\n'.join(c['text'] for c in chunks).casefold()
            for term in q['absence_check_terms']:
                if term.casefold() in corpus_text:
                    raise ValueError(f"Absence claim requires review; term {term!r} exists ({q['id']})")
    return {'dev': {'answerable': 15, 'unanswerable': 2, 'total': 17},
            'test': {'answerable': 5, 'unanswerable': 2, 'total': 7}, 'total': 24}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate-only', action='store_true', help='Check existing frozen benchmark without rewriting it')
    args = ap.parse_args()
    chunks = json.loads((ROOT / 'starter/data/chunks.json').read_text(encoding='utf-8'))
    original = json.loads((DAY6 / 'benchmark/questions.json').read_text(encoding='utf-8'))
    dev, test = assemble(original, chunks)
    counts = validate_benchmark(chunks, dev, test)
    paths = [OUT / 'dev.json', OUT / 'test.json', OUT / 'split_lock.json']
    hashes = {'corpus': canonical_sha(chunks), 'dev': canonical_sha(dev), 'test': canonical_sha(test)}
    lock = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'policy': 'Tune only on dev; never expose test gold to retrieval/agent. Day 6 test excluded.',
        'counts': counts, 'sha256_canonical': hashes,
        'source_corpus': 'Day 6 B: 160 chunks, Think Python chapters 1–4',
        'test_evaluated': False,
    }
    if args.validate_only:
        for path, key in zip(paths[:2], ('dev', 'test')):
            existing = json.loads(path.read_text(encoding='utf-8'))
            if canonical_sha(existing) != hashes[key]:
                raise ValueError(f'Frozen {key} benchmark differs from expected content')
        existing_lock = json.loads(paths[2].read_text(encoding='utf-8'))
        if existing_lock['sha256_canonical'] != hashes:
            raise ValueError('Split lock does not match benchmark and corpus')
        print('CP2 locked files valid:', json.dumps(counts, ensure_ascii=False))
        return
    if any(p.exists() for p in paths):
        raise FileExistsError('Frozen CP2 files already exist. Use --validate-only; do not overwrite a held-out benchmark.')
    write_json(paths[0], dev)
    write_json(paths[1], test)
    write_json(paths[2], lock)
    print('CP2 frozen:', json.dumps(counts, ensure_ascii=False))
    print('Canonical SHA256:', json.dumps(hashes, ensure_ascii=False))


if __name__ == '__main__':
    main()
