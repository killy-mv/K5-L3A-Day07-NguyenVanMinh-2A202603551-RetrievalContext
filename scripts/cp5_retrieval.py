"""CP5 development-only original-query retrieval/context benchmark, no test access.

Three real methods, identical 15 answerable + 2 unanswerable dev questions.
Run original question, not agent-written search query. Save each case promptly.
"""
from __future__ import annotations
import json
import statistics
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.agent import build_context
from src.day6_adapter import Day6DenseRetriever, database_url, default_day6_root
from src.retrieval import BM25Retriever, HybridRRFRetriever, load_corpus
from scripts.cp2_build import canonical_sha

OUT=ROOT/'starter/results/cp5/retrieval'
ART=ROOT/'artifacts/cp5'
METHODS=('dense','lexical','hybrid')
TOP_K=3
CONTEXT_CHARS=3800


def gold_hit(row, q):
    return any(row['source_id']==g['source_id'] and row['page']==g['page']
               and g['evidence'] in row['text'] for g in q['gold'])


def one_run(q, retriever, label):
    start=time.perf_counter()
    hits=retriever.search(q['question'],filters=q.get('filter'),top_k=TOP_K)
    elapsed=round((time.perf_counter()-start)*1000,3)
    if len(hits)>TOP_K or len({c['id'] for c in hits})!=len(hits):
        raise ValueError(f'Invalid top-3 from {label}: {q["id"]}')
    context, context_ids=build_context({h['id']:h for h in hits},CONTEXT_CHARS)
    answerable=q['answerable']
    matches=[gold_hit(h,q) for h in hits] if answerable else []
    context_hit=(any(g['chunk_id'] in context_ids and g['evidence'] in context
                     for g in q['gold']) if answerable else None)
    return {'query_id':q['id'],'split':'dev','answerable':answerable,
            'question':q['question'],'filters':q.get('filter'),
            'strategy':label,'query_type':'original_user_question',
            'top_k':TOP_K,'context_budget_chars':CONTEXT_CHARS,
            'elapsed_ms':elapsed,'results':hits,'context':context,
            'context_ids':context_ids,'context_chars':len(context),
            'hit1':bool(matches and matches[0]) if answerable else None,
            'hit3':any(matches) if answerable else None,
            'context_hit':context_hit,'gold_comparison':matches if answerable else None,
            'cost_usd':0, 'llm_called':False}


def summarize(rows,label):
    answerable=[r for r in rows if r['answerable']]
    n=len(answerable)
    def metric(k):
        good=sum(r[k] is True for r in answerable)
        return {'n':good,'N':n,'rate':good/n if n else None}
    return {'strategy':label,'split':'dev','query_type':'original_user_question',
            'processed':len(rows),'answerable':n,
            'unanswerable':len(rows)-n,
            'hit1':metric('hit1'),'evidence_hit3':metric('hit3'),
            'context_evidence_rate':metric('context_hit'),
            'top_k':TOP_K,'context_budget_chars':CONTEXT_CHARS,
            'mean_retrieval_ms':round(statistics.mean(r['elapsed_ms'] for r in rows),3),
            'median_retrieval_ms':round(statistics.median(r['elapsed_ms'] for r in rows),3),
            'mean_context_chars':round(statistics.mean(r['context_chars'] for r in rows),1),
            'retrieval_service_cost_usd':0,
            'note':'Machine exact gold substring/source/page matching only; independent semantic grading pending.'}


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def run():
    dev=json.loads((ROOT/'benchmark/dev.json').read_text(encoding='utf8'))
    corpus=load_corpus(ROOT/'starter/data/chunks.json')
    if len(dev)!=17 or sum(q['answerable'] for q in dev)!=15:
        raise ValueError('Unexpected dev benchmark dimensions')
    lock=json.loads((ROOT/'benchmark/split_lock.json').read_text(encoding='utf8'))
    if (canonical_sha(dev)!=lock['sha256_canonical']['dev'] or
            canonical_sha(corpus)!=lock['sha256_canonical']['corpus']):
        raise ValueError('CP2 dev/corpus lock mismatch; do not benchmark modified inputs')
    lexical=BM25Retriever(corpus)
    dense=Day6DenseRetriever(database_url(default_day6_root()))
    hybrid=HybridRRFRetriever(dense,lexical,rrf_k=60,candidate_k=12)
    engines={'dense':dense,'lexical':lexical,'hybrid':hybrid}
    hashes={'dev':canonical_sha(dev),'corpus':canonical_sha(corpus)}
    for label,engine in engines.items():
        rows=[]
        for n,q in enumerate(dev,1):
            path=OUT/label/f'{q["id"]}.json'
            if path.exists():
                row=json.loads(path.read_text(encoding='utf8'))
                if row.get('question')!=q['question'] or row.get('strategy')!=label:
                    raise ValueError('Existing trace mismatch: '+str(path))
            else:
                row=one_run(q,engine,label)
                write_json(path,row)
            rows.append(row)
            print(f'{label} {n}/17 {q["id"]} H1={row["hit1"]} H3={row["hit3"]} context={row["context_hit"]}',flush=True)
        summary=summarize(rows,label)
        summary['hashes']=hashes
        write_json(ART/f'retrieval_{label}_summary.json',summary)
        write_json(ART/f'retrieval_{label}_traces.json',rows)
        print('FINAL',label,{k:summary[k] for k in ('hit1','evidence_hit3','context_evidence_rate')},flush=True)
    write_json(ART/'retrieval_comparison.json',{'split':'dev','methods':list(METHODS),
        'hashes':hashes,'rows':[summarize([json.loads((OUT/label/f'{q["id"]}.json').read_text(encoding='utf8')) for q in dev],label)
                 for label in METHODS]})
    print('CP5 retrieval fully complete, dev-only; 51 real calls.',flush=True)

if __name__=='__main__':
    run()
