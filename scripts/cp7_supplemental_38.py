"""Supplementary non-comparable Gemini 3.8 runs for three failed CP7 cases.

Does NOT overwrite or change CP6 model 3.5 freeze or original CP7 scores.
Every model request is paced at 2 RPM below user's stated 3 RPM quota.
Run one case per invocation. No retries, no running other concurrent agents.
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.cp7_test import check_lock,load_test
from scripts.cp7_pacing import PacedGemini
from scripts.cp5_retrieval import write_json
from src.agent import AgentConfig,SYSTEM,run_agent
from src.gemini_client import GeminiClient
from src.day6_adapter import Day6DenseRetriever,database_url,default_day6_root
from src.retrieval import make_retriever

CASES={'dense':('D7-T02','D7-T04'),'hybrid':('D7-T03',)}
OUT=ROOT/'artifacts/cp7_supplemental_gemini38'
STATE=ROOT/'starter/results/cp7/gemini38_request_pacing.json'


def run(method,case_id,rpm):
    lock=check_lock()
    test=load_test(lock)
    if case_id not in CASES.get(method,()):
        raise ValueError('Only the three documented Gemini 3.5 provider-failed cases are eligible')
    prior=ROOT/'starter/results/cp7/agent'/method/f'{case_id}.json'
    original=json.loads(prior.read_text(encoding='utf8'))
    if original['status']!='error' or original['model']!=lock['selected']['gemini_model']:
        raise ValueError('Original frozen-model test does not have an unresolved provider error')
    path=OUT/f'{method}_{case_id}.json'
    if path.exists():
        record=json.loads(path.read_text(encoding='utf8'))
        print('Already attempted supplemental 3.8, no new requests:',path.name,record['status'])
        return
    model=GeminiClient.from_env()
    if model.model!='gemini-3.8-flash':
        raise ValueError('Supplement must explicitly use configured gemini-3.8-flash model')
    paced=PacedGemini(model,rpm=rpm,state_file=STATE)
    dense=Day6DenseRetriever(database_url(default_day6_root()))
    retriever=dense if method=='dense' else make_retriever('hybrid',dense=dense)
    question=next(q for q in test if q['id']==case_id)
    record=run_agent(question['question'],paced,retriever,filters=question.get('filter'),
                     config=AgentConfig(top_k=3,max_searches=3,context_budget_chars=3800),
                     instructions=SYSTEM)
    record.update({'query_id':case_id,'split':'test','retrieval_strategy':method,
                   'model_comparability':'SUPPLEMENTAL_UNFROZEN_MODEL_3_8_DO_NOT_MERGE_WITH_PRIMARY_CP7',
                   'primary_cp6_frozen_model':lock['selected']['gemini_model'],
                   'source_error_trace':str(prior.relative_to(ROOT)),
                   'request_rpm_limit':rpm,'request_count_this_case':paced.request_count,
                   'evaluation_utc':datetime.now(timezone.utc).isoformat()})
    write_json(path,record)
    print('SUPPLEMENTAL ONLY:',method,case_id,record['status'],
          'requests=',paced.request_count,'ms=',record['elapsed_ms'], 'trace=',path.relative_to(ROOT),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--strategy',choices=('dense','hybrid'),required=True)
    ap.add_argument('--case-id',required=True)
    ap.add_argument('--rpm',type=float,default=2.0)
    args=ap.parse_args()
    if not 0<args.rpm<=2.0:
        ap.error('RPM maximum 2.0 for supplemental 3.8 to stay below 3 RPM')
    run(args.strategy,args.case_id,args.rpm)
