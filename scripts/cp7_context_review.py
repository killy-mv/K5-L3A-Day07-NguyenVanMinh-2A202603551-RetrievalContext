"""Context evidence scoring on held-out test, offline only.

0=no relevant evidence; 1=partial; 2=all reference evidence. Labels here are
source-alignment-assisted drafts, not independent human verified grades.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.cp7_test import check_lock,load_test
from scripts.cp5_retrieval import write_json

REASONS={
 'D7-T01':'Source p43 directly explains the detour into the function body and return to the caller.',
 'D7-T02':'Source p44 directly states a local variable is destroyed when its function terminates.',
 'D7-T03':'Source p45 explains that a stack frame represents a function and holds its parameters and local variables.',
 'D7-T04':'Source p55 directly contrasts radius as an interface parameter with n as a rendering detail.',
 'D7-T05':'Source p57 enumerates the first three encapsulation/generalization development steps.',
}


def review():
    lock=check_lock()
    test=load_test(lock)
    output=[]
    for method in ('dense','hybrid'):
        traces=json.loads((ROOT/f'artifacts/cp7/retrieval_{method}_traces.json').read_text(encoding='utf8'))
        byid={r['query_id']:r for r in traces}
        rows=[]
        for q in test:
            if not q['answerable']:
                continue
            r=byid[q['id']]
            found=[]
            for g in q['gold']:
                matches=[c for c in r['results'] if c['id']==g['chunk_id'] and c['source_id']==g['source_id']
                         and c['page']==g['page'] and g['evidence'] in c['text']]
                in_context=(g['chunk_id'] in r['context_ids'] and g['evidence'] in r['context'])
                found.append({'chunk_id':g['chunk_id'],'source_id':g['source_id'],
                              'page':g['page'],'top3_match':bool(matches),'in_context':in_context})
            count=sum(g['in_context'] for g in found)
            grade=2 if count==len(found) else 1 if count else 0
            rows.append({'query_id':q['id'],'strategy':method,'context_grade':grade,
                         'human_confirmed':False,
                         'source_alignment':found,'context_chars':r['context_chars'],
                         'review_reason':REASONS[q['id']] if grade==2 else
                           'At least one reference evidence span absent from packed context; requires semantic review.',
                         'review_type':'AI-assisted reference evidence alignment; not independent semantic confirmation'})
        summary={'strategy':method,'split':'test','graded_answerable':len(rows),
                 'grade_counts':{str(g):sum(row['context_grade']==g for row in rows) for g in (0,1,2)},
                 'mean_context_grade':sum(row['context_grade'] for row in rows)/len(rows),
                 'human_confirmed':False,'test_sha256':lock['benchmark_hashes']['test'],
                 'rows':rows}
        write_json(ROOT/f'artifacts/cp7/context_{method}_review_assisted.json',summary)
        output.append(summary)
        print(method,'context grades',summary['grade_counts'])
    write_json(ROOT/'artifacts/cp7/context_review_comparison.json',
               {'split':'test','methods':output,'human_confirmed':False,'api_calls':0})

if __name__=='__main__':review()
