"""Read-only audit of frozen CP5 dev artifacts and evaluation semantics."""
import json
import unittest
from pathlib import Path

from scripts.cp5_retrieval import gold_hit, summarize
from scripts.cp5_summarize import agent_context_hit, PARTIAL, NON_SUPPORTING
from scripts.cp2_build import canonical_sha

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/cp5'


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


class CP5AuditTests(unittest.TestCase):
    def test_dev_lock_unchanged_and_test_not_evaluated(self):
        dev=load(ROOT/'benchmark/dev.json')
        lock=load(ROOT/'benchmark/split_lock.json')
        self.assertEqual(canonical_sha(dev),lock['sha256_canonical']['dev'])
        self.assertEqual(len(dev),17)
        self.assertTrue(all(q['id'].startswith('D7-D') and q['split']=='dev' for q in dev))
        self.assertEqual(sum(q['answerable'] for q in dev),15)

    def test_retrieval_real_traces_complete(self):
        for method in ('dense','lexical','hybrid'):
            with self.subTest(method=method):
                data=load(ART/f'retrieval_{method}_traces.json')
                self.assertEqual(len(data),17)
                self.assertEqual(sum(x['answerable'] for x in data),15)
                self.assertEqual(set(x['query_id'] for x in data),set(q['id'] for q in load(ROOT/'benchmark/dev.json')))
                self.assertTrue(all(x['query_type']=='original_user_question' and x['split']=='dev' for x in data))
                self.assertTrue(all(x['hit1'] is None and x['hit3'] is None for x in data if not x['answerable']))
                self.assertEqual(summarize(data,method)['evidence_hit3']['N'],15)

    def test_grounding_requires_real_citation_and_review_status(self):
        for method in ('dense','lexical','hybrid'):
            with self.subTest(method=method):
                cases=load(ART/f'agent_{method}_traces.json')
                reviews=load(ART/f'agent_{method}_reviews_assisted.json')
                summary=load(ART/f'agent_{method}_summary.json')
                self.assertEqual(len(cases),17)
                self.assertEqual(len(reviews),17)
                self.assertEqual(summary['total_cases'],17)
                self.assertFalse(summary['human_confirmed'])
                self.assertTrue(all(not r['human_confirmed'] for r in reviews))
                self.assertTrue(all(x['status'] in ('answered','abstain') for x in cases))
                self.assertTrue(all(x['split']=='dev' and x['retrieval_strategy']==method for x in cases))
                self.assertTrue(all(not {'gold','expected_answer'} & set(x) for x in cases))
                self.assertTrue(all(set(x['citations']) <= set(x['context_ids']) for x in cases))
                self.assertEqual(summary['agent_metrics']['tool_use'],{'n':17,'N':17})

    def test_gold_hit_requires_source_page_and_exact_text(self):
        row={'source_id':'doc','page':4,'text':'exact evidence here'}
        q={'gold':[{'source_id':'doc','page':4,'evidence':'exact evidence'}]}
        self.assertTrue(gold_hit(row,q))
        self.assertFalse(gold_hit({**row,'page':5},q))
        self.assertFalse(gold_hit({**row,'source_id':'other'},q))
        self.assertFalse(gold_hit({**row,'text':'different text'},q))

    def test_context_evidence_rate_uses_exact_gold_and_chunk_id(self):
        q={'answerable':True,'gold':[{'chunk_id':'a','evidence':'correct passage'}]}
        self.assertTrue(agent_context_hit(q,{'context_ids':['a'],'context':'correct passage'}))
        self.assertFalse(agent_context_hit(q,{'context_ids':['b'],'context':'correct passage'}))
        self.assertFalse(agent_context_hit(q,{'context_ids':['a'],'context':'unrelated'}))

    def test_assisted_review_flags_real_missing_details(self):
        self.assertIn('D7-D03',PARTIAL)
        self.assertIn('D7-D14',PARTIAL)
        self.assertIn(('dense','D7-D04','paragraph-61a6e79256b1aafefa75'),NON_SUPPORTING)
        report=load(ART/'provider_error_audit.json')
        self.assertTrue(report['all_recovered'])
        self.assertTrue(all(x['historical_provider_http_429'] for x in report['historical_errors']))


if __name__=='__main__':
    unittest.main()
