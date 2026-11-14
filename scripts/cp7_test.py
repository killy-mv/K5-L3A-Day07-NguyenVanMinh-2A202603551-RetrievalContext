"""CP7 one-time held-out test: frozen Dense baseline vs frozen Hybrid RRF.

The lock is verified BEFORE opening the test; retrieval is deterministic and
results persist per case. Agent receives question/filter ONLY. No tuning.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.cp6_freeze import freeze
from scripts.cp2_build import canonical_sha
from scripts.cp5_retrieval import one_run,write_json,summarize
from scripts.cp2_validate import agent_inputs
from src.day6_adapter import Day6DenseRetriever,database_url,default_day6_root
from src.retrieval import make_retriever,load_corpus
from src.agent import AgentConfig,run_agent,SYSTEM
from src.gemini_client import GeminiClient
from scripts.cp7_pacing import PacedGemini

OUT=ROOT/'artifacts/cp7'
LOCAL=ROOT/'starter/results/cp7'
METHODS=('dense','hybrid')


def check_lock():
    config=freeze()
    cfg=config['selected']
    if (cfg['strategy'],cfg['dense_weight'],cfg['lexical_weight'],cfg['rrf_k'],
        cfg['candidate_k'],cfg['top_k'],cfg['context_budget_chars'],cfg['max_searches'],
        cfg['system_prompt'])!=('hybrid',1.0,1.0,60,12,3,3800,3,'CP3 SYSTEM v1'):
        raise ValueError('Unexpected frozen CP6 configuration')
    return config


def load_test(lock):
    # Only read held-out test AFTER validating CP6 lock.
    questions=json.loads((ROOT/'benchmark/test.json').read_text(encoding='utf8'))
    if canonical_sha(questions)!=lock['benchmark_hashes']['test']:
        raise ValueError('Test checksum mismatch')
    if len(questions)!=7 or sum(bool(q['answerable']) for q in questions)!=5:
        raise ValueError('Unexpected held-out test size')
    return questions


def retrieval_run(test,lock):
    dense=Day6DenseRetriever(database_url(default_day6_root()))
    hybrid=make_retriever('hybrid',dense=dense)
    for name,engine in (('dense',dense),('hybrid',hybrid)):
        rows=[]
        for q in test:
            file=LOCAL/'retrieval'/name/f'{q["id"]}.json'
            if file.is_file():
                row=json.loads(file.read_text(encoding='utf8'))
                if row.get('query_id')!=q['id'] or row.get('question')!=q['question']:
                    raise ValueError('Stored retrieval trace does not match locked test')
            else:
                row=one_run(q,engine,name)
                row['evaluation']='held_out_test_after_CP6_freeze'
                write_json(file,row)
            rows.append(row)
            print(f"TEST RETRIEVAL {name} {q['id']} H1={row['hit1']} H3={row['hit3']} context={row['context_hit']}",flush=True)
        summary=summarize(rows,name)
        summary.update({'test_sha256':lock['benchmark_hashes']['test'],
                        'frozen_at_utc':lock['frozen_at_utc'],
                        'first_test_evaluation':True})
        write_json(OUT/f'retrieval_{name}_traces.json',rows)
        write_json(OUT/f'retrieval_{name}_summary.json',summary)
        print(name,'RESULT',summary['hit1'],summary['evidence_hit3'],summary['context_evidence_rate'],flush=True)


def agent_run(test,lock,method,pace,retry_errors=False,rpm=1.0,max_questions=1):
    import time
    model=GeminiClient.from_env()
    if model.model!=lock['selected']['gemini_model']:
        raise ValueError('Model differs from frozen CP6 configuration')
    # Request-level pacing applies to EVERY model.generate call, including
    # follow-up tool-calling turns. Shared persistent state survives restarts.
    model=PacedGemini(model,rpm=rpm,state_file=LOCAL/'gemini_request_pacing.json')
    dense=Day6DenseRetriever(database_url(default_day6_root()))
    retriever=(dense if method=='dense' else make_retriever('hybrid',dense=dense))
    # Explicit no-gold projection, even though evaluator has test in memory.
    safe=agent_inputs('test',ROOT)
    if len(safe)!=len(test) or any(s['id']!=q['id'] or s['question']!=q['question'] for s,q in zip(safe,test)):
        raise ValueError('Test question projection mismatch')
    config=AgentConfig(top_k=3,max_searches=3,context_budget_chars=3800)
    attempted=0
    for q in safe:
        if attempted >= max_questions:
            print(f'CP7 safe stop: reached --max-questions={max_questions}',flush=True)
            break
        file=LOCAL/'agent'/method/f'{q["id"]}.json'
        if file.exists():
            row=json.loads(file.read_text(encoding='utf8'))
            if row['question']!=q['question'] or row['retrieval_strategy']!=method:
                raise ValueError('Existing frozen test trace does not match')
            if row['status'] != 'error' or not retry_errors:
                print(f'TEST AGENT {method} {q["id"]}: EXISTING {row["status"]}',flush=True)
                continue
            archive=file.with_name(q['id']+'_previous_error_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')
            write_json(archive,row)
            print(f'TEST AGENT {method} {q["id"]}: archived provider error and retrying once',flush=True)
        attempted+=1
        row=run_agent(q['question'],model,retriever,filters=q.get('filter'),
                      config=config,instructions=SYSTEM)
        row.update({'query_id':q['id'],'split':'test',
                    'retrieval_strategy':method,'test_frozen_before_run':True,
                    'frozen_at_utc':lock['frozen_at_utc'],
                    'evaluation_utc':datetime.now(timezone.utc).isoformat(),
                    'request_rpm_limit':rpm,
                    'request_count_this_case':model.request_count})
        write_json(file,row)
        print(f'TEST AGENT {method} {q["id"]}: {row["status"]} '
              f'searches={row["search_calls"]} calls={row["model_calls"]} '
              f'ms={row["elapsed_ms"]} error={row.get("error")}',flush=True)
        if row['status']=='error':
            # Quota/provider failures should not trigger new calls for later
            # questions in the same batch. No silent retry.
            print('CP7 stopped after provider/retriever error; existing traces preserved.',flush=True)
            break
        if pace:
            time.sleep(pace)


def main():
    ap=argparse.ArgumentParser(description='Held-out CP7 evaluation after fixed CP6 lock')
    ap.add_argument('--mode',choices=('retrieval','agent'),required=True)
    ap.add_argument('--strategy',choices=METHODS,help='Required for --mode agent')
    ap.add_argument('--pace-seconds',type=float,default=8)
    ap.add_argument('--retry-errors',action='store_true',help='Explicit retry of saved errors with archival')
    ap.add_argument('--rpm',type=float,default=1.0,
                    help='Max model requests/minute across cases and CLI restarts (default 1)')
    ap.add_argument('--max-questions',type=int,default=1,
                    help='Maximum NEW questions processed per invocation (default 1)')
    args=ap.parse_args()
    if args.mode=='agent' and not args.strategy:
        ap.error('--strategy dense|hybrid required in agent mode')
    if args.pace_seconds<0 or args.pace_seconds>120:
        ap.error('Invalid pace')
    if not 0 < args.rpm <= 60 or not 1 <= args.max_questions <= 7:
        ap.error('Invalid RPM or max-questions; see --help')
    lock=check_lock()
    test=load_test(lock)
    if args.mode=='retrieval':retrieval_run(test,lock)
    else:agent_run(test,lock,args.strategy,args.pace_seconds,args.retry_errors,args.rpm,args.max_questions)

if __name__=='__main__':main()
