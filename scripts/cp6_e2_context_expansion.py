"""CP6 fallback E2b: dev-only context expansion after provider E2 prompt failures.

Unlike E2_prompt (not evaluable due provider HTTP503), this is a separate,
explicitly documented local retrieval experiment: dense top_k 5, context 5000.
Compare equivalent Hit@1 and Evidence Hit@3 against original Dense top3,
plus evidence coverage in the full context. No LLM is called and no test used.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.day6_adapter import Day6DenseRetriever,database_url,default_day6_root
from src.agent import build_context
from scripts.cp2_build import canonical_sha
from scripts.cp5_retrieval import gold_hit,write_json

OUT=ROOT/'artifacts/cp6'
TOP_K=5
BUDGET=5000


def run():
    dev=json.loads((ROOT/'benchmark/dev.json').read_text(encoding='utf8'))
    lock=json.loads((ROOT/'benchmark/split_lock.json').read_text(encoding='utf8'))
    if canonical_sha(dev)!=lock['sha256_canonical']['dev']:
        raise ValueError('DEV lock modified')
    retriever=Day6DenseRetriever(database_url(default_day6_root()))
    rows=[]
    for q in dev:
        hits=retriever.search(q['question'],filters=q.get('filter'),top_k=TOP_K)
        context,ids=build_context({h['id']:h for h in hits},BUDGET)
        scores=[gold_hit(h,q) for h in hits] if q['answerable'] else []
        covered=(any(g['chunk_id'] in ids and g['evidence'] in context for g in q['gold']) if q['answerable'] else None)
        row={'query_id':q['id'],'split':'dev','answerable':q['answerable'],
             'query':q['question'],'filters':q.get('filter'),'strategy':'dense_top5_context5000',
             'top_k':TOP_K,'context_budget_chars':BUDGET,
             'results':hits,'context':context,'context_ids':ids,
             'hit1':bool(scores and scores[0]) if q['answerable'] else None,
             'hit3':any(scores[:3]) if q['answerable'] else None,
             'hit5':any(scores) if q['answerable'] else None,
             'context_hit':covered,'candidate_count':len(hits),
             'context_chars':len(context)}
        rows.append(row)
        print(q['id'],'Hit1',row['hit1'],'Hit3',row['hit3'],'Hit5',row['hit5'],'context',covered,flush=True)
    originals=json.loads((ROOT/'artifacts/cp5/retrieval_dense_traces.json').read_text(encoding='utf8'))
    def metric(key,rs):
        eligible=[r for r in rs if r['answerable']]
        n=sum(r[key] is True for r in eligible)
        return {'n':n,'N':len(eligible),'rate':n/len(eligible)}
    summary={'experiment':'E2b_dense_context_expansion','split':'dev',
        'status':'measured after E2 prompt became provider-blocked',
        'baseline':{'dense_top_k':3,'context_budget_chars':3800,
                    'hit1':metric('hit1',originals),
                    'hit3':metric('hit3',originals),
                    'context_evidence_rate':metric('context_hit',originals)},
        'changed':{'dense_top_k':5,'context_budget_chars':BUDGET,
                   'hit1':metric('hit1',rows),'hit3':metric('hit3',rows),
                   'hit5':metric('hit5',rows),
                   'context_evidence_rate':metric('context_hit',rows)},
        'per_case':[{'id':r['query_id'],'baseline_context_hit':b['context_hit'],
                      'expanded_context_hit':r['context_hit'],
                      'baseline_hit3':b['hit3'],'expanded_hit3':r['hit3'],
                      'context_chars':r['context_chars']}
                     for r,b in zip(rows,originals)],
        'test_evaluated':False,
        'warning':'This is a dev-only substitute experiment, not a result of the preregistered E2 Gemini prompt; it changes two linked context-size parameters and requires more tokens. Hit@3 remains same-cutoff.'}
    write_json(OUT/'e2b_dense_context_expansion_traces.json',rows)
    write_json(OUT/'e2b_dense_context_expansion_summary.json',summary)
    print('E2b COMPARISON',summary['baseline'],summary['changed'],flush=True)

if __name__=='__main__':run()
