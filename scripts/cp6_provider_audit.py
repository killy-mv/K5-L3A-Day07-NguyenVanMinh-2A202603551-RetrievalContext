"""Document E2 prompt experiment provider failure without converting to scores."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'starter/results/cp6/e2_agent'
OUT=ROOT/'artifacts/cp6/e2_prompt_provider_audit.json'


def run():
    records=[]
    for file in sorted(DIR.glob('D7-*.json')):
        d=json.loads(file.read_text(encoding='utf8'))
        records.append({'query_id':d['query_id'],'attempt_file':file.name,
            'status':d['status'],'model_calls_completed':d['model_calls'],
            'retrieval_calls_completed':d['search_calls'],
            'elapsed_ms':d['elapsed_ms'],
            'error':d.get('error'),'safe_diagnostic':[
                e.get('safe_detail') for e in d['events'] if e.get('event')=='error']})
    summary={'stage':'CP6','experiment':'E2_complete_answer_prompt',
        'evaluation_status':'blocked_by_provider',
        'dev_cases_attempted':len({r['query_id'] for r in records}),
        'saved_attempts':len(records),
        'eligible_answer_reviews':sum(r['status'] in ('answered','abstain') for r in records),
        'test_evaluated':False,
        'note':'HTTP 503 and timeout on Gemini. No valid model response for attempted cases; do not invent grounded-answer improvements. Historical error attempts kept in ignored local trace folder.',
        'attempts':records}
    OUT.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('E2 provider audit:',summary['dev_cases_attempted'],'dev cases attempted;',summary['eligible_answer_reviews'],'valid answers')

if __name__=='__main__':run()
