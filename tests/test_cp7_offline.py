"""Offline CP7 integrity checks (never send network/API requests)."""
import json
import tempfile
import unittest
from pathlib import Path
from scripts.cp7_pacing import PacedGemini
from scripts.cp7_test import check_lock, load_test

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/cp7'


class MockProvider:
    model='test-model'
    def __init__(self): self.calls=0
    def generate(self,contents,*,instructions,force_search,allow_search):
        self.calls+=1
        return {'candidates':[{'content':{'role':'model','parts':[{'text':'test'}]}}]}


class CP7OfflineTests(unittest.TestCase):
    def test_frozen_and_locked_test(self):
        lock=check_lock()
        q=load_test(lock)
        self.assertEqual(len(q),7)
        self.assertEqual(sum(x['answerable'] for x in q),5)
        self.assertEqual(lock['selected']['strategy'],'hybrid')

    def test_retrieval_test_complete(self):
        for method in ('dense','hybrid'):
            with self.subTest(method=method):
                d=json.loads((ART/f'retrieval_{method}_summary.json').read_text(encoding='utf8'))
                self.assertEqual(d['hit1']['N'],5)
                self.assertEqual(d['evidence_hit3']['n'],5)
                self.assertEqual(d['context_evidence_rate']['n'],5)

    def test_agent_errors_are_not_counted_as_abstention(self):
        for method, valid, error in [('dense',7,0),('hybrid',7,0)]:
            with self.subTest(method=method):
                d=json.loads((ART/f'agent_{method}_summary.json').read_text(encoding='utf8'))
                self.assertEqual(d['valid_cases'],valid)
                self.assertEqual(d['provider_or_retriever_errors'],error)
                self.assertEqual(d['metrics_valid_only']['abstention'],{'n':2,'N':2})
                self.assertFalse(d['human_confirmed'])

    def test_request_level_pacing_across_instances(self):
        clock=[1000.0]
        sleeps=[]
        def now():return clock[0]
        def sleep(n):sleeps.append(n);clock[0]+=n
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'pace.json'
            a=MockProvider()
            p=PacedGemini(a,rpm=2,state_file=file,now=now,sleep=sleep)
            p.generate([{'role':'user'}],instructions='x')
            self.assertEqual(a.calls,1)
            self.assertEqual(sleeps,[])
            another=MockProvider()
            p2=PacedGemini(another,rpm=2,state_file=file,now=now,sleep=sleep)
            p2.generate([{'role':'user'}],instructions='x')
            self.assertEqual(sleeps,[30.0])
            self.assertEqual(another.calls,1)

    def test_invalid_rpm(self):
        with self.assertRaises(ValueError):
            PacedGemini(MockProvider(),rpm=0,state_file=Path('noop'))

if __name__=='__main__':unittest.main()
