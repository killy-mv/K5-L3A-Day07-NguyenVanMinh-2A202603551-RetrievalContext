"""CP6 E1 preregistered weighted RRF (0.7 dense / 1.3 lexical): dev only."""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.day6_adapter import Day6DenseRetriever,database_url,default_day6_root
from src.retrieval import BM25Retriever,HybridRRFRetriever,load_corpus
from scripts.cp2_build import canonical_sha
from scripts.cp5_retrieval import one_run,summarize,write_json

OUT=ROOT/'artifacts/cp6'
WEIGHTS={'dense':0.7,'lexical':1.3}


def main():
    dev=json.loads((ROOT/'benchmark/dev.json').read_text(encoding='utf8'))
    corpus=load_corpus(ROOT/'starter/data/chunks.json')
    lock=json.loads((ROOT/'benchmark/split_lock.json').read_text(encoding='utf8'))
    if canonical_sha(dev)!=lock['sha256_canonical']['dev'] or canonical_sha(corpus)!=lock['sha256_canonical']['corpus']:
        raise ValueError('Frozen dev/corpus changed')
    dense=Day6DenseRetriever(database_url(default_day6_root()))
    lexical=BM25Retriever(corpus)
    hybrid=HybridRRFRetriever(dense,lexical,rrf_k=60,candidate_k=12,
                              dense_weight=WEIGHTS['dense'],lexical_weight=WEIGHTS['lexical'])
    rows=[]
    for q in dev:
        r=one_run(q,hybrid,'weighted_rrf')
        rows.append(r)
        print(q['id'],'hit1=',r['hit1'],'hit3=',r['hit3'],flush=True)
    old=json.loads((ROOT/'artifacts/cp5/retrieval_hybrid_traces.json').read_text(encoding='utf8'))
    base={r['query_id']:r for r in old}
    change=[]
    for row in rows:
        prev=base[row['query_id']]
        change.append({'id':row['query_id'],'answerable':row['answerable'],
            'baseline_top3':[h['id'] for h in prev['results']],
            'weighted_top3':[h['id'] for h in row['results']],
            'baseline_hit1':prev['hit1'],'weighted_hit1':row['hit1'],
            'baseline_hit3':prev['hit3'],'weighted_hit3':row['hit3']})
    summary=summarize(rows,'weighted_rrf')
    summary['experiment']='E1_weighted_rrf'
    summary['rrf_parameters']={'rrf_k':60,'candidate_k':12,'dense_weight':0.7,'lexical_weight':1.3}
    summary['baseline_hybrid']=summarize(old,'hybrid')
    summary['per_case_changes']=change
    summary['dev_hash']=canonical_sha(dev)
    summary['test_evaluated']=False
    write_json(OUT/'e1_weighted_rrf_traces.json',rows)
    write_json(OUT/'e1_weighted_rrf_summary.json',summary)
    print('E1 SUMMARY',json.dumps({k:summary[k] for k in ('hit1','evidence_hit3','context_evidence_rate')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
