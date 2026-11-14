"""Summarize historical quota errors without exposing provider content or keys."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'starter/results/cp5/agent'
ART=ROOT/'artifacts/cp5'


def main():
    recovered=[]
    by_method=Counter()
    for method in ('dense','lexical','hybrid'):
        folder=SOURCE/method
        for file in folder.glob('*_previous_error_*.json'):
            old=json.loads(file.read_text(encoding='utf8'))
            if old.get('status')!='error':
                raise ValueError('Historic failure trace unexpectedly successful')
            case=old['query_id']
            current=json.loads((folder/f'{case}.json').read_text(encoding='utf8'))
            recovered.append({'strategy':method,'query_id':case,
                 'historical_status':old['status'],'historical_error_kind':old.get('error'),
                 'historical_provider_http_429':any('HTTP 429' in (e.get('safe_detail') or '')
                     for e in old.get('events',[]) if e.get('event')=='error'),
                 'final_status':current['status'],
                 'note':'Original error preserved locally; final run reran the question independently.'})
            by_method[method]+=1
    report={'stage':'CP5','split':'dev','historical_error_count':len(recovered),
         'historical_errors_by_method':dict(by_method),
         'all_recovered':all(x['final_status'] in ('answered','abstain') for x in recovered),
         'historical_errors':recovered,'test_queries_evaluated':0,
         'notes':'Not counted as correct, wrong, or abstain; provider failures are separate. No dollar cost is inferred.'}
    ART.mkdir(exist_ok=True)
    (ART/'provider_error_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('CP5 historical provider errors:',len(recovered),'recovered:',report['all_recovered'],dict(by_method))

if __name__=='__main__':
    main()
