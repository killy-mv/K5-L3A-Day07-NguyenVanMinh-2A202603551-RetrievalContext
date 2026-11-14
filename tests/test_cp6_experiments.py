"""CP6 dev-only experiment checks and pre-test frozen config integrity."""
import json
import unittest
from pathlib import Path
from scripts.cp2_build import canonical_sha
from scripts.cp6_freeze import freeze
from src.agent import SYSTEM, SYSTEM_COMPLETE
from src.retrieval import HybridRRFRetriever

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/cp6'


def load(name):
    return json.loads((OUT/name).read_text(encoding='utf8'))


class CP6Tests(unittest.TestCase):
    def test_pre_registered_hypotheses_exist(self):
        d=load('experiment_plan.json')
        self.assertTrue(d['registered_before_experiments'])
        self.assertEqual([x['id'] for x in d['experiments']],
                         ['E1_weighted_rrf','E2_complete_answer_prompt'])
        self.assertFalse(d['test_evaluated'])

    def test_e1_fusion_weights_change_ranking_and_not_normalize_above_one(self):
        def item(i):
            return {'id':i,'text':'text '+i,'source':'source',
                    'source_id':'doc','page':1,'metadata':{'chapter':'chapter_1'},'score':0.8}
        class Static:
            def __init__(self,order):self.order=order
            def search(self,query,*,filters,top_k):return [item(i) for i in self.order[:top_k]]
        balanced=HybridRRFRetriever(Static(['a','b']),Static(['b','a']),rrf_k=60,candidate_k=2)
        weighted=HybridRRFRetriever(Static(['a','b']),Static(['b','a']),rrf_k=60,candidate_k=2,
                                   dense_weight=.7,lexical_weight=1.3)
        self.assertEqual(balanced.search('test')[0]['id'],'a')
        ranked=weighted.search('test')
        self.assertEqual(ranked[0]['id'],'b')
        self.assertTrue(all(0<=r['score']<=1 for r in ranked))

    def test_e1_and_e2b_are_dev_only_and_honest(self):
        e1=load('e1_weighted_rrf_summary.json')
        e2b=load('e2b_dense_context_expansion_summary.json')
        self.assertEqual(e1['evidence_hit3'],{'n':14,'N':15,'rate':14/15})
        self.assertEqual(e1['baseline_hybrid']['evidence_hit3']['n'],15)
        self.assertEqual(e2b['baseline']['context_evidence_rate']['n'],13)
        self.assertEqual(e2b['changed']['context_evidence_rate']['n'],14)
        self.assertEqual(e2b['changed']['hit3']['n'],13)
        self.assertEqual(e2b['changed']['hit5']['n'],14)
        self.assertFalse(e1['test_evaluated'])
        self.assertFalse(e2b['test_evaluated'])
        self.assertEqual(len(load('e1_weighted_rrf_traces.json')),17)
        self.assertEqual(len(load('e2b_dense_context_expansion_traces.json')),17)

    def test_e2_prompt_is_generic_without_gold_answer(self):
        self.assertTrue(SYSTEM_COMPLETE.startswith(SYSTEM))
        self.assertIn('question wording',SYSTEM_COMPLETE)
        self.assertNotIn('D7-D03',SYSTEM_COMPLETE)
        self.assertNotIn('D7-D14',SYSTEM_COMPLETE)
        self.assertNotIn('expected_answer',SYSTEM_COMPLETE)

    def test_locked_test_untouched_and_final_selection_verified(self):
        lock=load('final_config_lock.json')
        cp2=json.loads((ROOT/'benchmark/split_lock.json').read_text(encoding='utf8'))
        self.assertEqual(lock['benchmark_hashes'],cp2['sha256_canonical'])
        self.assertFalse(lock['test_evaluated'])
        self.assertEqual(lock['test_cases_used_to_select'],0)
        self.assertEqual(lock['selected']['strategy'],'hybrid')
        self.assertEqual(lock['selected']['dense_weight'],1.0)
        self.assertEqual(lock['selected']['lexical_weight'],1.0)
        self.assertEqual(lock['selected']['top_k'],3)
        self.assertEqual(freeze(),lock)


if __name__=='__main__':unittest.main()
