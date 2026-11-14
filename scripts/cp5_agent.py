"""CP5: real Gemini agent runs on dev ONLY; resumable per method/case.

No gold or expected answers are loaded by the agent runner. Passes only
question/id/explicit filters. Every completed real run is saved immediately;
errors remain errors, not abstentions, and may be rerun separately.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.agent import AgentConfig, run_agent
from src.day6_adapter import database_url, default_day6_root
from src.gemini_client import GeminiClient
from src.retrieval import STRATEGIES, make_retriever
from scripts.cp2_validate import agent_inputs, validate_files

OUT=ROOT/'starter/results/cp5/agent'
CONFIG=AgentConfig(top_k=3,max_searches=3,context_budget_chars=3800)


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    ap=argparse.ArgumentParser(description='Real agent evaluation on dev only (resumable)')
    ap.add_argument('--strategy',choices=STRATEGIES,required=True)
    ap.add_argument('--start',type=int,default=0,help='0-based inclusive offset in 17 dev questions')
    ap.add_argument('--limit',type=int,default=17,help='Number of dev questions from start')
    ap.add_argument('--retry-errors',action='store_true')
    ap.add_argument('--pace-seconds',type=float,default=7.0)
    ap.add_argument('--max-quota-retries',type=int,default=3)
    args=ap.parse_args()
    if not 0<=args.start<17 or not 1<=args.limit<=17 or not 0<=args.pace_seconds<=120 or not 0<=args.max_quota_retries<=5:
        ap.error('Invalid start, limit, or pacing')
    validate_files(ROOT)  # verify dev/test freeze; never send test to agent
    items=agent_inputs('dev',ROOT)
    if len(items)!=17:
        raise ValueError('Expected exactly 17 locked dev question inputs')
    model=GeminiClient.from_env()
    url=database_url(default_day6_root()) if args.strategy!='lexical' else None
    retriever=make_retriever(args.strategy,url=url)
    folder=OUT/args.strategy
    for ordinal,q in enumerate(items[args.start:args.start+args.limit],start=args.start+1):
        path=folder/f'{q["id"]}.json'
        if path.exists():
            prior=json.loads(path.read_text(encoding='utf8'))
            if prior.get('question')!=q['question'] or prior.get('retrieval_strategy')!=args.strategy:
                raise ValueError(f'Existing run mismatch: {path}')
            if prior.get('status')!='error' or not args.retry_errors:
                print(f'{args.strategy} {ordinal}/17 {q["id"]}: EXISTING {prior["status"]}',flush=True)
                continue
            backup=folder/f'{q["id"]}_previous_error_{datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")}.json'
            write_json(backup,prior)
        record=None
        for attempt in range(args.max_quota_retries+1):
            record=run_agent(q['question'],model,retriever,
                             filters=q.get('filter'),config=CONFIG)
            is_quota=(record['status']=='error' and any(
                event.get('event')=='error' and 'HTTP 429' in (event.get('safe_detail') or '')
                for event in record.get('events',[])))
            if not is_quota or attempt>=args.max_quota_retries:
                break
            pause=20*(attempt+1)
            previous=folder/f'{q["id"]}_quota_retry{attempt+1}.json'
            record.update({'query_id':q['id'],'split':'dev','retrieval_strategy':args.strategy})
            write_json(previous,record)
            print(f'{args.strategy} {q["id"]} HTTP 429: cooling down {pause}s for retry {attempt+1}',flush=True)
            time.sleep(pause)
        record.update({'query_id':q['id'],'split':'dev',
                       'retrieval_strategy':args.strategy,
                       'benchmark_set':'CP2 locked dev, question-only input',
                       'run_utc':datetime.now(timezone.utc).isoformat()})
        if args.strategy=='hybrid':
            record.update({'rrf_k':retriever.rrf_k,'rrf_candidate_k':retriever.candidate_k})
        write_json(path,record)
        print(f'{args.strategy} {ordinal}/17 {q["id"]}: {record["status"]}, '
              f'searches={record["search_calls"]} model={record["model_calls"]} '
              f'ms={record["elapsed_ms"]} error={record.get("error")}',flush=True)
        if args.pace_seconds:
            time.sleep(args.pace_seconds)
    status=Counter()
    for q in items:
        p=folder/f'{q["id"]}.json'
        if p.exists():
            status[json.loads(p.read_text(encoding='utf8'))['status']]+=1
        else:
            status['pending']+=1
    print('PROGRESS',args.strategy,dict(status),flush=True)

if __name__=='__main__':
    main()
