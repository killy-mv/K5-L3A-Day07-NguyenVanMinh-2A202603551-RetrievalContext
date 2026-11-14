"""CP5 dev-only summaries and preliminary assisted source-review labels.

The labels below are explicitly AI-assisted review drafts, NOT independent
human verification. They were made after reading all 15 dev answers and the
cited source chunks, including each alternate citation and omitted detail.
Do not alter the frozen benchmark or use test cases at this stage.
"""
from __future__ import annotations
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'starter'))
from agent_metrics import aggregate
from scripts.cp2_build import canonical_sha

SRC=ROOT/'starter/results/cp5/agent'
OUT=ROOT/'artifacts/cp5'
METHODS=('dense','lexical','hybrid')

# Source-reading findings. D03 and D14 contain the essential answer but omit
# one detail explicitly present in expected_answer. Use conservative 1/2.
PARTIAL={
    'D7-D03': 'Correctly identifies ** for exponentiation, but omits that ^ is XOR in Python; the expected contrast is incomplete.',
    'D7-D14': 'Correctly identifies bk/lt/rt, but does not mention that lt/rt angles are expressed in degrees; expected detail omitted.',
}
CASE_EVIDENCE_REASONS = {'D7-D01': 'Source defines a program as instructions specifying computation; examples of computation are also present.', 'D7-D02': 'Author recommends running Python in a browser, with PythonAnywhere as a specific example.', 'D7-D03': 'Source distinguishes exponentiation ** from bitwise XOR ^; generated answer states only **.', 'D7-D04': 'Source uses type(2) -> int and type(42.0) -> float; auxiliary glossary is general, not direct evidence for examples.', 'D7-D05': 'Source explicitly distinguishes syntax tokens and structural combinations of tokens.', 'D7-D06': 'Source defines variable as a name referring to a value, including a chapter glossary.', 'D7-D07': 'Source lists class among Python keywords and explains that keywords cannot be variable names.', 'D7-D08': 'Source says standalone expression is evaluated but not printed in script mode and recommends print.', 'D7-D09': 'Source defines string concatenation as linking strings end-to-end with +.', 'D7-D10': 'Source contrasts why versus what in comments and illustrates redundant versus useful comments.', 'D7-D11': 'Source explicitly states int() chops off fractional part rather than rounding.', 'D7-D12': 'Source defines module dot notation and shows module.function examples.', 'D7-D13': 'Source states a function definition creates an object without executing its body until called.', 'D7-D14': 'Source names bk, lt, rt and states turn angle is in degrees; generated answer omits angle detail.', 'D7-D15': 'Source shows fixed size square generalized into square(t,length) that supports any length.'}

# Strict, claim-specific citation check: p29 glossary describes categories but
# does not independently identify the example values 2 and 42.0.
NON_SUPPORTING={
    ('dense','D7-D04','paragraph-61a6e79256b1aafefa75'):
        'General type glossary on PDF p29 does not independently substantiate the specific examples 2 and 42.0.',
}


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def audit_one(q,row,method,chunks):
    if row.get('split')!='dev' or row.get('query_id')!=q['id']:
        raise ValueError('Unexpected trace/split: '+q['id'])
    if row.get('status') not in ('answered','abstain','error'):
        raise ValueError('Unknown trace status')
    if row['status']=='error':
        return None
    citations=row.get('citations',[])
    context_ids=set(row.get('context_ids',[]))
    citation_checks=[]
    for cid in citations:
        chunk=chunks.get(cid)
        if chunk is None:
            raise ValueError(f'Cited unknown chunk {cid}')
        if cid not in context_ids:
            raise ValueError('Citation not in model context')
        if not any(d['id']==cid and d['source_id']==chunk['source_id'] and d['page']==chunk['page']
                   for d in row.get('citation_details',[])):
            raise ValueError('Citation page/source mismatch: '+cid)
        not_support=NON_SUPPORTING.get((method,q['id'],cid))
        citation_checks.append({'chunk_id':cid,'source_id':chunk['source_id'],
             'page':chunk['page'],'support':not not_support,
             'reason':not_support or CASE_EVIDENCE_REASONS[q['id']]})
    result={'query_id':q['id'],'review_type':'AI-assisted source-reading draft',
            'human_confirmed':False,
            'citation_support':[c['support'] for c in citation_checks],
            'citation_checks':citation_checks}
    if q['answerable']:
        if row['status']=='abstain':
            grade=0
            reason='Answered question was refused, so quality grade must be 0.'
        elif q['id'] in PARTIAL:
            grade=1
            reason=PARTIAL[q['id']]
        else:
            grade=2
            reason=CASE_EVIDENCE_REASONS[q['id']]
        result.update({'answer_grade':grade,'answer_reason':reason})
    else:
        result.update({'abstention_correct':row['status']=='abstain',
                       'answer_reason':'Out-of-scope topic for the indexed four chapters; correct behavior is to search and abstain.'})
    return result


def agent_context_hit(q,row):
    if not q['answerable']:
        return None
    ids=set(row.get('context_ids',[]))
    context=row.get('context','')
    return any(g['chunk_id'] in ids and g['evidence'] in context for g in q['gold'])


def main():
    dev=load(ROOT/'benchmark/dev.json')
    lock=load(ROOT/'benchmark/split_lock.json')
    if canonical_sha(dev)!=lock['sha256_canonical']['dev']:
        raise ValueError('CP2 dev frozen hash mismatch')
    if len(dev)!=17 or sum(bool(q['answerable']) for q in dev)!=15:
        raise ValueError('Unexpected dev split')
    chunks={c['id']:c for c in load(ROOT/'starter/data/chunks.json')}
    result=[]
    for method in METHODS:
        traces=[]
        reviews=[]
        for q in dev:
            path=SRC/method/f'{q["id"]}.json'
            if not path.exists():
                raise FileNotFoundError('Incomplete CP5 agent batch: '+str(path))
            row=load(path)
            if row.get('retrieval_strategy')!=method:
                raise ValueError('Trace method mismatch')
            row['agent_exact_gold_in_context']=agent_context_hit(q,row)
            traces.append(row)
            reviewed=audit_one(q,row,method,chunks)
            if reviewed:
                reviews.append(reviewed)
        scored=aggregate(dev,traces,reviews)
        scores=[r for r in traces if r['status']!='error']
        positive=[r for q,r in zip(dev,traces) if q['answerable'] and r['status']!='error']
        context_hits=sum(r['agent_exact_gold_in_context'] is True for r in positive)
        counts=Counter(r['status'] for r in traces)
        summary={'strategy':method,'split':'dev','model':traces[0]['model'],
                 'review_type':'AI-assisted source-reading draft; not independent human-verified',
                 'human_confirmed':False,'total_cases':len(traces),'status_counts':dict(counts),
                 'agent_exact_gold_context':{'n':context_hits,'N':len(positive),
                       'rate':context_hits/len(positive) if positive else None},
                 'agent_metrics':scored,
                 'search_calls_total':sum(r['search_calls'] for r in traces),
                 'model_calls_total':sum(r['model_calls'] for r in traces),
                 'tool_call_distribution':dict(Counter(str(r['search_calls']) for r in traces)),
                 'total_reported_input_tokens':sum(r['token_usage']['input_tokens'] for r in traces),
                 'total_reported_output_tokens':sum(r['token_usage']['output_tokens'] for r in traces),
                 'total_reported_tokens':sum(r['token_usage']['total_tokens'] for r in traces),
                 'mean_run_elapsed_ms':round(statistics.mean(r['elapsed_ms'] for r in traces),2),
                 'median_run_elapsed_ms':round(statistics.median(r['elapsed_ms'] for r in traces),2),
                 'api_cost_usd':None,
                 'historical_provider_errors':'Transient HTTP 429 responses were recorded separately; final resumed runs are scored here.',
                 'benchmark_dev_sha256':canonical_sha(dev)}
        save(OUT/f'agent_{method}_summary.json',summary)
        save(OUT/f'agent_{method}_traces.json',traces)
        save(OUT/f'agent_{method}_reviews_assisted.json',reviews)
        result.append(summary)
        print(method,dict(counts),'grounding',scored['grounded_answer'],'citations',scored['citation_validity'],
              'agent_context_exact',context_hits,'/',len(positive),flush=True)
    save(OUT/'agent_comparison.json',{'split':'dev','human_confirmed':False,
                                    'review_type':'AI-assisted draft', 'strategies':result})
    print('CP5 dev agent: all 51 final traces and assisted source reviews exported.',flush=True)

if __name__=='__main__':
    main()
