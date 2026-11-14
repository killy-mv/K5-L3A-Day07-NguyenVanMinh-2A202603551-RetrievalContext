"""Copy two real CP4 dev agent smoke traces for submission (without gold)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'starter/results/cp4'
out = ROOT / 'artifacts/cp4_agent_smoke'
out.mkdir(parents=True, exist_ok=True)
for strategy in ('lexical', 'hybrid'):
    found = sorted(source.glob(f'*_{strategy}_D7-D06.json'))
    validated = []
    for file in found:
        record = json.loads(file.read_text(encoding='utf-8'))
        if (record.get('status') == 'answered' and record.get('split') == 'dev'
                and record.get('retrieval_strategy') == strategy and record.get('search_calls', 0) > 0):
            validated.append(record)
    if not validated:
        raise RuntimeError(f'No successful live trace for {strategy}')
    text = json.dumps(validated[-1], ensure_ascii=False, indent=2) + '\n'
    if re.search(r'(?i)(GEMINI_API_KEY\s*=|GOOGLE_API_KEY\s*=|postgresql://|x-goog-api-key)', text):
        raise ValueError(f'Possible secret material in {strategy} trace')
    path = out / f'{strategy}_D7-D06.json'
    path.write_text(text, encoding='utf-8')
    print(strategy, '->', path.relative_to(ROOT))
