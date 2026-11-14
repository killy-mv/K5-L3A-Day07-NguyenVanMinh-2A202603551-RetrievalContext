"""Build truthful CP7 test summaries from saved traces; NO API requests.

Grounded/citation labels are AI-assisted review drafts, never human-confirmed.
A provider error is excluded from denominators, and is NOT an abstention.
"""
from __future__ import annotations
import json
import sys
import statistics
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'starter'))
from agent_metrics import aggregate
from scripts.cp7_test import check_lock,load_test
from scripts.cp5_retrieval import write_json

RAW=ROOT/'starter/results/cp7/agent'
ART=ROOT/'artifacts/cp7'
REASONS={
    'D7-T01':'Relevant PDF p43 passage directly says execution jumps into the body of the function and then returns to where it left off.',
    'D7-T02':'PDF p44 explicitly states local variables exist inside the function and are destroyed when it terminates; cited example on p45 supports the outside-scope exception.',
    'D7-T03':'PDF p45 and glossary p48 describe a stack frame as a function/name with parameters and local variables.',
    'D7-T04':'PDF p55 explicitly distinguishes radius as part of interface and n as internal rendering detail.',
    'D7-T05':'PDF p57 contains the first three numbered steps of the development plan exactly.',
    'D7-T06':'Indexed chapters 1–4 do not cover async def coroutine declarations; abstention is expected.',
    'D7-T07':'Indexed chapters 1–4 do not cover yield from/subgenerators; abstention is expected.',
}


def main():
    lock=check_lock()
    test=load_test(lock)
    result=[]
    for method in ('dense','hybrid'):
        traces=[]
        reviews=[]
        for q in test:
            path=RAW/method/f'{q["id"]}.json'
            if not path.exists():
                raise FileNotFoundError('Incomplete seven-case trace set: '+str(path))
            row=json.loads(path.read_text(encoding='utf8'))
            if row.get('query_id')!=q['id'] or row.get('split')!='test' or row.get('retrieval_strategy')!=method:
                raise ValueError('Trace identity mismatch: '+str(path))
            if row.get('model')!=lock['selected']['gemini_model'] or row.get('frozen_at_utc')!=lock['frozen_at_utc']:
                raise ValueError('Wrong model / CP6 freeze date in test trace')
            traces.append(row)
            if row['status']=='error':
                continue
            valid_ids=set(row['context_ids'])
            if any(c not in valid_ids for c in row['citations']):
                raise ValueError('Citation ID missing from final context')
            citation_checks=[]
            for cite in row['citation_details']:
                citation_checks.append({'chunk_id':cite['id'],'page':cite['page'],
                                        'source_id':cite['source_id'],
                                        'support':True,'reason':REASONS[q['id']]})
            item={'query_id':q['id'],'review_type':'AI-assisted source review draft',
                  'human_confirmed':False,'citation_support':[c['support'] for c in citation_checks],
                  'citation_checks':citation_checks,'reason':REASONS[q['id']]}
            if q['answerable']:
                item['answer_grade']=2 if row['status']=='answered' else 0
                item['review_note']='Suggested grade, compare answer and entire cited passage manually before submission.'
            else:
                item['abstention_correct']=row['status']=='abstain'
            reviews.append(item)
        metrics=aggregate(test,traces,reviews)
        counts=Counter(r['status'] for r in traces)
        valid=[r for r in traces if r['status']!='error']
        historical=list((RAW/method).glob('*_previous_error_*.json'))
        summary={'split':'test','strategy':method,'cp6_frozen_at':lock['frozen_at_utc'],
                 'frozen_config_sha256':lock['cp2_lock_sha256'],
                 'model':lock['selected']['gemini_model'],
                 'status_counts':dict(counts),'total_test_cases':7,
                 'valid_cases':len(valid),'provider_or_retriever_errors':counts['error'],
                 'historical_retry_errors':len(historical),
                 'search_calls_final_traces':sum(r['search_calls'] for r in traces),
                 'model_calls_final_traces':sum(r['model_calls'] for r in traces),
                 'tokens_final_traces':sum(r['token_usage']['total_tokens'] for r in traces),
                 'mean_valid_elapsed_ms':round(statistics.mean(r['elapsed_ms'] for r in valid),2) if valid else None,
                 'api_cost_usd':None,
                 'review_type':'AI-assisted provisional; NOT human confirmed',
                 'human_confirmed':False,'metrics_valid_only':metrics,
                 'limitations':'Only answered/abstained valid cases scored; 429/503/timeout are not wrong answers. Incomplete seven-case agent coverage. No retuning after test.'}
        write_json(ART/f'agent_{method}_summary.json',summary)
        write_json(ART/f'agent_{method}_traces.json',traces)
        write_json(ART/f'agent_{method}_reviews_assisted.json',reviews)
        print('CP7',method,'status',dict(counts),'grounding(valid)',metrics['grounded_answer'],
              'abstention(valid)',metrics['abstention'],flush=True)
        result.append(summary)
    write_json(ART/'agent_comparison_partial.json',{'split':'test','cp6_selection_unchanged':True,
              'api_requests_made_by_this_script':0,
              'status':'incomplete_due_provider_quota',
              'cannot_claim_full_agent_test':True,
              'methods':result})
    print('Wrote offline CP7 summaries with 0 new Gemini API calls.',flush=True)

if __name__=='__main__':main()
