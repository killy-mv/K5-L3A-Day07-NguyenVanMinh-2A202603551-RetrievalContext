"""Regression checks for honest mixed-model aggregation and context scoring."""
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(rel):
    return json.loads((ROOT/rel).read_text(encoding='utf8'))


class TestCP7MixedContext(unittest.TestCase):
    def test_five_context_grades_both_methods(self):
        for method in ('dense','hybrid'):
            with self.subTest(method=method):
                result=read(f'artifacts/cp7/context_{method}_review_assisted.json')
                self.assertEqual(result['graded_answerable'],5)
                self.assertEqual(result['grade_counts'],{'0':0,'1':0,'2':5})
                self.assertFalse(result['human_confirmed'])

    def test_mixed_errors_not_reclassified(self):
        for method,total in [('dense',7),('hybrid',7)]:
            with self.subTest(method=method):
                result=read(f'artifacts/cp7/mixed_models_{method}_summary.json')
                self.assertEqual(result['valid_responses'],total)
                self.assertEqual(result['valid_from_supplemental_model'],0)
                self.assertEqual(result['errors'],7-total)
                self.assertFalse(result['human_confirmed'])

    def test_supplement_38_preserved_as_provider_error(self):
        row=read('artifacts/cp7_supplemental_gemini38/hybrid_D7-T03.json')
        self.assertEqual(row['status'],'error')
        self.assertEqual(row['model'],'gemini-3.8-flash')
        self.assertEqual(row['request_count_this_case'],1)
        self.assertEqual(row['request_rpm_limit'],2.0)


if __name__=='__main__':unittest.main()
