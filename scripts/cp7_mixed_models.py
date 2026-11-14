"""Combine valid Gemini 3.5 primary and 3.8 supplemental test traces transparently.

Never overwrite original test scores. Use only actual valid supplemental
responses, never convert 503/429 errors into answered/abstained.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'starter'))
from agent_metrics import aggregate
from scripts.cp7_test import check_lock,load_test
from scripts.cp5_retrieval import write_json

SUPPLEMENT=ROOT/'artifacts/cp7_supplemental_gemini38'
OUT=ROOT/'artifacts/cp7'


def compile():
    lock=check_lock()
    questions=load_test(lock)
    summaries=[]
    for method in ('dense','hybrid'):
        original=json.loads((OUT/f'agent_{method}_traces.json').read_text(encoding='utf8'))
        original_reviews=json.loads((OUT/f'agent_{method}_reviews_assisted.json').read_text(encoding='utf8'))
        original_reviews={r['query_id']:r for r in original_reviews}
        byid={r['query_id']:r for r in original}
        merged=[]; model_by_case=[]; provenance=[]
        for q in questions:
            primary=byid[q['id']]
            supplemental_file=SUPPLEMENT/f'{method}_{q["id"]}.json'
            chosen=primary
            source='frozen_gemini_3_5_primary'
            if primary['status']=='error' and supplemental_file.exists():
                extra=json.loads(supplemental_file.read_text(encoding='utf8'))
                if (extra.get('status') in ('answered','abstain') and
                    extra.get('model')=='gemini-3.8-flash' and
                    extra.get('query_id')==q['id'] and extra.get('retrieval_strategy')==method and
                    extra.get('question')==q['question'] and
                    extra.get('split')=='test'):
                    chosen=extra
                    source='supplemental_gemini_3_8_valid'
            merged.append(chosen)
            model_by_case.append({'id':q['id'],'status':chosen['status'],
                                  'model':chosen['model'],'source':source})
            provenance.append({'id':q['id'], 'original_status':primary['status'],
                               'supplement_attempted':supplemental_file.exists(),
                               'supplement_status':(json.loads(supplemental_file.read_text(encoding='utf8'))['status']
                                    if supplemental_file.exists() else None),
                               'chosen':source})
        # Scores only from cases with previously reviewed valid source, or
        # supplemental cases if an explicit independent review is later supplied.
        reviews=[]
        for row in merged:
            review=original_reviews.get(row['query_id'])
            if review and row['model']==lock['selected']['gemini_model']:
                reviews.append(review)
        eligible=[row for row in merged if row['status']!='error']
        if len(reviews)==len(eligible):
            metrics=aggregate(questions,merged,reviews)
        else:
            metrics=None
        summary={'split':'test','strategy':method,
                 'evaluation_type':'MIXED_MODELS_NOT_SINGLE_FROZEN_MODEL',
                 'primary_frozen_model':'gemini-3.5-flash-lite',
                 'supplement_model':'gemini-3.8-flash',
                 'valid_from_supplemental_model':sum(x['source']=='supplemental_gemini_3_8_valid' for x in model_by_case),
                 'valid_responses':len(eligible),'test_cases':len(questions),
                 'errors':len(questions)-len(eligible),
                 'grades_ai_assisted':metrics,
                 'human_confirmed':False,
                 'model_by_case':model_by_case,'provenance':provenance,
                 'warning':'Not comparable to official frozen single-model CP7; never overwrite source traces.'}
        write_json(OUT/f'mixed_models_{method}_summary.json',summary)
        summaries.append(summary)
        print('MIXED',method,'valid',len(eligible),'/7','supplemental recovered',summary['valid_from_supplemental_model'])
    write_json(OUT/'mixed_models_comparison.json',{'methods':summaries,
               'frozen_test_config_preserved':True,'human_confirmed':False,
               'api_calls':0})

if __name__=='__main__':compile()
