"""CP6 E2: dev-only, real Gemini with preregistered completeness prompt.

Baseline is the frozen CP5 balanced hybrid RRF. Retrieval and all other
settings are unchanged. Test and gold labels are never passed to the model.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.agent import AgentConfig,SYSTEM_COMPLETE,run_agent
from src.day6_adapter import database_url,default_day6_root
from src.gemini_client import GeminiClient
from src.retrieval import make_retriever
from scripts.cp2_validate import agent_inputs,validate_files

OUT=ROOT/'starter/results/cp6/e2_agent'
CONFIG=AgentConfig(top_k=3,max_searches=3,context_budget_chars=3800)
PROMPT_SHA=hashlib.sha256(SYSTEM_COMPLETE.encode('utf8')).hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    ap=argparse.ArgumentParser(description='CP6 E2 real Gemini, 17 dev only, resumable')
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--limit',type=int,default=17)
    ap.add_argument('--pace-seconds',type=float,default=7)
    ap.add_argument('--retry-errors',action='store_true')
    args=ap.parse_args()
    if not 0<=args.start<17 or not 1<=args.limit<=17 or not 0<=args.pace_seconds<=120:
        ap.error('Invalid start/limit/pace')
    validate_files(ROOT)
    questions=agent_inputs('dev',ROOT)
    model=GeminiClient.from_env()
    retriever=make_retriever('hybrid',url=database_url(default_day6_root()),candidate_k=12)
    for i,q in enumerate(questions[args.start:args.start+args.limit],args.start+1):
        output=OUT/f'{q["id"]}.json'
        if output.exists():
            previous=json.loads(output.read_text(encoding='utf8'))
            if (previous.get('question')!=q['question'] or
                    previous.get('prompt_sha256')!=PROMPT_SHA):
                raise ValueError('Existing E2 trace mismatch: '+str(output))
            if previous.get('status')!='error' or not args.retry_errors:
                print(f'E2 {i}/17 {q["id"]} EXISTING {previous["status"]}',flush=True)
                continue
            archived=OUT/f'{q["id"]}_previous_error_{datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")}.json'
            write_json(archived,previous)
        result=run_agent(q['question'],model,retriever,filters=q.get('filter'),
                         config=CONFIG,instructions=SYSTEM_COMPLETE)
        result.update({'query_id':q['id'],'split':'dev','retrieval_strategy':'hybrid',
                       'experiment':'E2_complete_answer_prompt','prompt_sha256':PROMPT_SHA,
                       'rrf_k':60,'rrf_candidate_k':12,'rrf_weights':{'dense':1.0,'lexical':1.0},
                       'run_utc':datetime.now(timezone.utc).isoformat()})
        write_json(output,result)
        print(f'E2 {i}/17 {q["id"]} {result["status"]} '
              f'search={result["search_calls"]} model={result["model_calls"]} '
              f'elapsed_ms={result["elapsed_ms"]}',flush=True)
        if args.pace_seconds:
            time.sleep(args.pace_seconds)
    from collections import Counter
    counts=Counter()
    for q in questions:
        file=OUT/f'{q["id"]}.json'
        counts[json.loads(file.read_text(encoding='utf8'))['status'] if file.exists() else 'pending']+=1
    print('E2 PROGRESS:',dict(counts),flush=True)

if __name__=='__main__':main()
