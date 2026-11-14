"""Compute reproducible hypothetical paid-tier API cost from usage tokens.

Not a billing claim: omits failed requests with unreported usage, cached
pricing, other projects and any free-tier allocations.
Source: https://ai.google.dev/gemini-api/docs/pricing (2026-11).
"""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RATE={'gemini-3.5-flash-lite':{'input_per_million_usd':0.30,'output_per_million_usd':2.50},
      'gemini-3.8-flash':{'input_per_million_usd':0.75,'output_per_million_usd':3.75}}
OUT=ROOT/'artifacts/cp7/estimated_api_costs.json'


def cost(model,usage):
    r=RATE[model]
    return (usage['input_tokens']*r['input_per_million_usd']+
            usage['output_tokens']*r['output_per_million_usd'])/1_000_000


def main():
    rows=[]
    for split in ('cp5','cp7'):
        for method in (('dense','lexical','hybrid') if split=='cp5' else ('dense','hybrid')):
            traces=json.loads((ROOT/f'artifacts/{split}/agent_{method}_traces.json').read_text(encoding='utf8'))
            bymodel={}
            for trace in traces:
                model=trace['model']
                record=bymodel.setdefault(model,{'input_tokens':0,'output_tokens':0,'total_tokens':0,
                                                  'valid_cases':0,'provider_error_cases':0})
                for token in ('input_tokens','output_tokens','total_tokens'):
                    record[token]+=int(trace['token_usage'][token])
                if trace['status']=='error':record['provider_error_cases']+=1
                else:record['valid_cases']+=1
            for model,metrics in bymodel.items():
                rows.append({'split':split,'strategy':method,'model':model,'reported_usage':metrics,
                             'paid_tier_estimated_usd':round(cost(model,metrics),8)})
    output={'price_source':'https://ai.google.dev/gemini-api/docs/pricing',
            'price_date':'2026-11-14', 'paid_tier_rates_usd_per_1m':RATE,
            'rows':rows,
            'estimated_total_recorded_paid_tier_usd':round(sum(r['paid_tier_estimated_usd'] for r in rows),8),
            'billing_confirmed':False,
            'limits':'Estimated only for tokens with provider-reported usage. Excludes unsuccessful/abandoned requests with no usage metadata, cache charges, embeddings CPU cost, free-tier adjustments, and external apps. Do not present as invoice or zero actual cost.'}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Paid-tier estimated cost of recorded CP5/CP7 token usage:',output['estimated_total_recorded_paid_tier_usd'],'USD')
    for r in rows:print(r['split'],r['strategy'],r['model'],r['paid_tier_estimated_usd'])

if __name__=='__main__':main()
