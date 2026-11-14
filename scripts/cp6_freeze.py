"""Freeze the best fully tested DEV-only config before any CP7 TEST evaluation.

Create-only. Refuse to overwrite; verify read-only on subsequent calls.
The test hash comes from CP2 lock, not opening or evaluating test questions.
"""
from __future__ import annotations
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.cp2_build import canonical_sha

OUT=ROOT/'artifacts/cp6/final_config_lock.json'


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    lock=json.loads((ROOT/'benchmark/split_lock.json').read_text(encoding='utf8'))
    dev=json.loads((ROOT/'benchmark/dev.json').read_text(encoding='utf8'))
    corpus=json.loads((ROOT/'starter/data/chunks.json').read_text(encoding='utf8'))
    if canonical_sha(dev)!=lock['sha256_canonical']['dev'] or canonical_sha(corpus)!=lock['sha256_canonical']['corpus']:
        raise ValueError('CP2 benchmark/corpus changed')
    baseline=json.loads((ROOT/'artifacts/cp5/retrieval_hybrid_summary.json').read_text(encoding='utf8'))
    e1=json.loads((ROOT/'artifacts/cp6/e1_weighted_rrf_summary.json').read_text(encoding='utf8'))
    e2b=json.loads((ROOT/'artifacts/cp6/e2b_dense_context_expansion_summary.json').read_text(encoding='utf8'))
    if (baseline['evidence_hit3']['n'],baseline['evidence_hit3']['N'])!=(15,15):
        raise ValueError('Selected hybrid baseline differs from measured CP5')
    if e1['evidence_hit3']['n']>=15 or e2b['changed']['context_evidence_rate']['n']>15:
        raise ValueError('Experiment summary inconsistent with selection rationale')
    settings={
       'strategy':'hybrid', 'dense_weight':1.0, 'lexical_weight':1.0,
       'rrf_k':60,'candidate_k':12,'top_k':3,'context_budget_chars':3800,
       'max_searches':3,'gemini_model':'gemini-3.5-flash-lite',
       'system_prompt':'CP3 SYSTEM v1',
       'embedding_model':'sentence-transformers/all-MiniLM-L6-v2',
       'embedding_revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
       'embedding_dimensions':384,'vector_strategy':'B','metric':'cosine',
       'lexical_model':'BM25','bm25_k1':1.5,'bm25_b':0.75,
       'agent_temperature':0.1,'agent_max_output_tokens':1200,
    }
    bundle={
       'stage':'CP6','status':'frozen_before_cp7',
       'selected_dev_only':True,
       'selection':'Keep balanced Hybrid RRF from CP5: full 15/15 dev Evidence Hit@3 and context coverage with valid 17/17 dev agent runs. E1 harmed hit quality, E2 prompt was provider-blocked, E2b context expansion improved Dense but remained below hybrid.',
       'selected':settings,
       'selection_evidence':{
          'cp5_hybrid_hit1':baseline['hit1'],
          'cp5_hybrid_evidence_hit3':baseline['evidence_hit3'],
          'cp5_hybrid_context_evidence_rate':baseline['context_evidence_rate'],
          'e1_weighted_evidence_hit3':e1['evidence_hit3'],
          'e2b_dense_context_evidence_rate':e2b['changed']['context_evidence_rate']},
       'source_code_sha256':{str(path.relative_to(ROOT)):sha(path) for path in [
          ROOT/'src/agent.py',ROOT/'src/retrieval.py',ROOT/'src/day6_adapter.py',ROOT/'src/gemini_client.py']},
       'cp2_lock_sha256':sha(ROOT/'benchmark/split_lock.json'),
       'benchmark_hashes':lock['sha256_canonical'],
       'test_evaluated':False,'test_cases_used_to_select':0,
       'manual_label_status':'AI-assisted CP5 provisional, human_confirmed=false',
    }
    if OUT.exists():
        current=json.loads(OUT.read_text(encoding='utf8'))
        for key in ('selected','source_code_sha256','cp2_lock_sha256','benchmark_hashes','selection_evidence'):
            if current[key]!=bundle[key]:
                raise ValueError('Final CP6 config is already frozen and no longer matches source/benchmark: '+key)
        print('CP6 frozen config verified unchanged:',OUT)
        return current
    bundle['frozen_at_utc']=datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('CP6 FINAL CONFIG FROZEN:',OUT)
    print('Hybrid RRF 1:1; top_k=3; context=3800; Gemini baseline CP3 prompt')
    return bundle

if __name__=='__main__':freeze()
