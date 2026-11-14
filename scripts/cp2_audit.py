"""Complete CP2 source, gold, split and absence audit (no retrieval of test).

Do not reveal test questions/answers in the audit output; validator verifies
hashes and source alignment, audit commentary is explicitly scoped to dev.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cp2_validate import validate_files
from scripts.cp2_build import canonical_sha


def main():
    counts=validate_files(ROOT)
    chunks=json.loads((ROOT/'starter/data/chunks.json').read_text(encoding='utf8'))
    dev=json.loads((ROOT/'benchmark/dev.json').read_text(encoding='utf8'))
    lock=json.loads((ROOT/'benchmark/split_lock.json').read_text(encoding='utf8'))
    index={c['id']:c for c in chunks}
    rows=[]
    for q in dev:
        item={'id':q['id'],'answerable':q['answerable'],'source_scope':'indexed Think Python chapters 1-4'}
        if q['answerable']:
            refs=[]
            for g in q['gold']:
                c=index[g['chunk_id']]
                refs.append({'chunk_id':c['id'],'source_id':c['source_id'],'pdf_page':c['page'],
                    'evidence_exact_in_chunk':g['evidence'] in c['text'],
                    'evidence_chars':len(g['evidence'])})
            item['evidence']=refs
            item['semantic_review']='source-aligned; answer-support must be checked from passage when grading'
            if q['id'] in ('D7-D03','D7-D05','D7-D10','D7-D13'):
                item['semantic_review']='source-aligned; potential multi-claim/chunk-boundary evidence. Review full cited context before awarding grade 2.'
        else:
            item['absence_reason']=q['absence_reason']
            item['absence_review']='machine term scan + scoped chapter review, not formal semantic proof'
        rows.append(item)
    report={'stage':'CP2_AUDIT','benchmark_counts':counts,
        'corpus':{'chunks':len(chunks),'pdf_pages':len({c['page'] for c in chunks}),
                  'source_ids':sorted({c['source_id'] for c in chunks}),
                  'chapters':sorted({c['metadata']['chapter'] for c in chunks})},
        'split_lock_verified': True,'sha256_canonical':lock['sha256_canonical'],
        'dev_gold_exact_alignment':sum(q['answerable'] and all(g['evidence_exact_in_chunk'] for g in row['evidence']) for q,row in zip(dev,rows)),
        'dev_unanswerable_count':sum(not q['answerable'] for q in dev),
        'test_queries_evaluated':0,
        'dev_case_audit':rows,
        'limitation':'Exact evidence alignment is machine-verified, not an independent human semantic review. Four absence claims require review against source scope.'}
    path=ROOT/'artifacts/cp2_audit.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('CP2 AUDIT:',counts,'dev exact gold:',report['dev_gold_exact_alignment'],'test queries evaluated:',0)
    print('Wrote:',path)

if __name__=='__main__':
    main()
