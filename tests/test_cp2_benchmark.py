"""Read-only checks of frozen CP2 split and no-gold API."""
import json
import unittest
from pathlib import Path

from scripts.cp2_validate import agent_inputs, validate_files
from scripts.cp2_build import canonical_sha, extract_exact

ROOT = Path(__file__).resolve().parents[1]


class CP2BenchmarkTests(unittest.TestCase):
    def test_frozen_split_and_evidence_resolve(self):
        counts = validate_files(ROOT)
        self.assertEqual(counts['total'], 24)
        self.assertEqual(counts['dev']['answerable'], 15)
        self.assertEqual(counts['test']['unanswerable'], 2)

    def test_agent_inputs_exclude_gold_and_expected_answers(self):
        for split, count in [('dev', 17), ('test', 7)]:
            with self.subTest(split=split):
                queries = agent_inputs(split, ROOT)
                self.assertEqual(len(queries), count)
                self.assertTrue(all(set(q) <= {'id', 'question', 'filter'} for q in queries))

    def test_dev_and_test_question_ids_disjoint(self):
        dev = json.loads((ROOT / 'benchmark/dev.json').read_text(encoding='utf-8'))
        test = json.loads((ROOT / 'benchmark/test.json').read_text(encoding='utf-8'))
        self.assertFalse(set(q['id'] for q in dev) & set(q['id'] for q in test))
        self.assertTrue(all(q['provenance']['origin'] == 'new Lab 7 held-out test'
                            for q in test if q['answerable']))

    def test_whitespace_normalization_exact_alignment(self):
        self.assertEqual(extract_exact('hello\nworld', 'helloworld'), 'hello\nworld')
        self.assertIsNone(extract_exact('hello world', 'different'))


if __name__ == '__main__':
    unittest.main()
