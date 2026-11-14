"""One request to check configured 3.8 endpoint independently of function calls.

Persistent 2 RPM pacing shared with supplemental CP7 calls. No credentials
or raw provider response included in output. Never pretends this is an Agent
result, never alters frozen benchmark or core model code.
"""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
from src.gemini_client import GeminiClient,ProviderError
from scripts.cp7_pacing import PacedGemini
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/cp7_supplemental_gemini38/provider_probe.json'
STATE=ROOT/'starter/results/cp7/gemini38_request_pacing.json'


def main():
    if OUT.exists():
        print('Already probed. Existing result:',OUT)
        return
    model=GeminiClient.from_env()
    if model.model!='gemini-3.8-flash':
        raise ValueError('Expected configured gemini-3.8-flash')
    client=PacedGemini(model,rpm=2,state_file=STATE)
    status='unavailable';error=None;tokens=None
    try:
        result=client.generate([{'role':'user','parts':[{'text':'Reply with one word: OK.'}]}],
                               instructions='Reply briefly.',allow_search=False)
        status='available';tokens=result.get('usageMetadata',{})
    except ProviderError as exc:
        error=str(exc)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
         'model':model.model,'diagnostic_type':'nonagent_plain_text_single_request',
         'status':status,'safe_error':error,'usage':tokens,
         'requests':client.request_count,'rpm_limit':2,
         'note':'Not a benchmark question or evaluated Agent answer.'},indent=2)+'\n',encoding='utf8')
    print('Configured Gemini 3.8 plain-text endpoint status:',status,'safe diagnostic:',error)

if __name__=='__main__':main()
