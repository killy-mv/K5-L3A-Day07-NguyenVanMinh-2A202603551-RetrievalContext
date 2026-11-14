"""CP4 retrieval correctness: BM25/RRF, filters, ranking and agent interface."""
from __future__ import annotations

import math
import unittest
from pathlib import Path

from src.retrieval import BM25Retriever, HybridRRFRetriever, load_corpus, make_retriever, tokenize


def doc(cid, text, chapter='chapter_1', page=23):
    return {'id': cid, 'source_id': 'thinkpython2_pdf', 'source': 'https://example.test/book.pdf',
            'text': text, 'page': page, 'metadata': {'chapter': chapter}}


class FakeDense:
    def __init__(self, docs):
        self.docs = list(docs)
        self.calls = []

    def search(self, query, *, filters, top_k):
        self.calls.append({'query': query, 'filters': filters, 'top_k': top_k})
        return [{**docrow, 'score': 0.9-idx*.1} for idx, docrow in
                enumerate(self.docs) if not filters or docrow['metadata']['chapter']==filters['chapter']][:top_k]


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.docs = [doc('a', 'a variable is a name that refers to a value', 'chapter_1'),
                     doc('b', 'a turtle draws a square using four sides', 'chapter_4', 52),
                     doc('c', 'variable assignment assigns a value', 'chapter_2', 31)]
        self.bm25 = BM25Retriever(self.docs)

    def test_tokenizer_normalizes_without_gold(self):
        self.assertEqual(tokenize('PYTHON variable! Function_1'), ['python', 'variable', 'function_1'])

    def test_bm25_prioritizes_matching_text_and_metadata(self):
        rows = self.bm25.search('turtle square', top_k=3)
        self.assertEqual([r['id'] for r in rows], ['b'])
        self.assertEqual(rows[0]['page'], 52)
        self.assertEqual(rows[0]['strategy'], 'lexical')
        self.assertGreater(rows[0]['raw_score'], 0)
        self.assertTrue(0 < rows[0]['score'] < 1)
        rows = self.bm25.search('variable', filters={'chapter': 'chapter_2'})
        self.assertEqual([r['id'] for r in rows], ['c'])

    def test_bm25_empty_candidates_and_invalid_filter(self):
        self.assertEqual(self.bm25.search('utterly_unmatched_token'), [])
        with self.assertRaises(ValueError):
            self.bm25.search('variable', filters={'gold_page': '23'})
        with self.assertRaises(ValueError):
            self.bm25.search('variable', top_k=0)

    def test_bm25_tie_order_is_deterministic(self):
        index = BM25Retriever([doc('z', 'same'), doc('a', 'same')])
        self.assertEqual([r['id'] for r in index.search('same')], ['a', 'z'])

    def test_no_leakage_corpus_schema(self):
        path = Path(__file__).resolve().parents[1] / 'starter/data/chunks.json'
        data = load_corpus(path)
        self.assertEqual(len(data), 160)
        self.assertTrue(all(not {'gold', 'expected_answer', 'split'} & set(c) for c in data))
        self.assertEqual(len(set(c['id'] for c in data)), 160)


class RRFTests(unittest.TestCase):
    def setUp(self):
        self.docs = [doc('a', 'variable name and Python'), doc('b', 'function call Python'),
                     doc('c', 'variable assignment expression')]

    def test_rrf_rank_fusion_and_scores_without_raw_score_sum(self):
        dense = FakeDense([self.docs[0], self.docs[1]])
        lexical = FakeDense([self.docs[1], self.docs[0]])
        rrf = HybridRRFRetriever(dense, lexical, rrf_k=60, candidate_k=3)
        rows = rrf.search('variable', top_k=3)
        self.assertEqual([r['id'] for r in rows], ['a', 'b'])  # tie -> ID order
        expected = (1/61 + 1/62) * 61 / 2
        self.assertTrue(math.isclose(rows[0]['score'], expected))
        self.assertEqual(rows[0]['ranks'], {'dense': 1, 'lexical': 2})
        self.assertLess(rows[0]['score'], 1)
        self.assertEqual(dense.calls[0]['top_k'], 3)

    def test_rrf_no_double_count_same_component(self):
        duplicated = FakeDense([self.docs[0], self.docs[0]])
        rrf = HybridRRFRetriever(duplicated, FakeDense([]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            rrf.search('variable')

    def test_rrf_applies_chapter_filter_to_both_sources(self):
        data = [doc('a', 'variable', 'chapter_1'), doc('b', 'variable', 'chapter_2')]
        dense, lexical = FakeDense(data), BM25Retriever(data)
        rows = HybridRRFRetriever(dense, lexical).search('variable', filters={'chapter': 'chapter_2'})
        self.assertEqual([r['id'] for r in rows], ['b'])
        self.assertEqual(dense.calls[0]['filters'], {'chapter': 'chapter_2'})

    def test_rrf_rejects_mismatched_source(self):
        mismatch = {**self.docs[0], 'text': 'changed version'}
        rrf = HybridRRFRetriever(FakeDense([self.docs[0]]), FakeDense([mismatch]))
        with self.assertRaisesRegex(ValueError, 'Mismatched'):
            rrf.search('variable')

    def test_rrf_rejects_filter_violation(self):
        class BrokenRetriever:
            def search(self, query, *, filters, top_k):
                return [{**doc('a', 'wrong source chapter', 'chapter_1'), 'score': .5}]
        rrf = HybridRRFRetriever(BrokenRetriever(), FakeDense([]))
        with self.assertRaisesRegex(ValueError, 'filter'):
            rrf.search('variable', filters={'chapter': 'chapter_2'})

    def test_factory_and_unknown_strategy(self):
        corpus = Path(__file__).resolve().parents[1] / 'starter/data/chunks.json'
        self.assertIsInstance(make_retriever('lexical', corpus_path=corpus), BM25Retriever)
        with self.assertRaises(ValueError):
            make_retriever('unknown')
        with self.assertRaises(ValueError):
            make_retriever('dense', url=None)
        with self.assertRaises(ValueError):
            HybridRRFRetriever(FakeDense([]), FakeDense([]), rrf_k=0)


if __name__ == '__main__':
    unittest.main()
