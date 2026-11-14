"""Export two selected CP3 live dev traces as reviewed submission artifacts.

No credentials, connection strings, or gold labels are included in agent traces.
"""
from pathlib import Path
import json
import re

root = Path(__file__).resolve().parents[1]
source = root / 'starter' / 'results' / 'cp3'
target = root / 'artifacts' / 'cp3_smoke'
target.mkdir(parents=True, exist_ok=True)
for case, status in [('D7-D06', 'answered'), ('D7-D16', 'abstain')]:
    candidates = sorted(source.glob(f'*_{case}.json'))
    valid = []
    for path in candidates:
        row = json.loads(path.read_text(encoding='utf-8'))
        if row.get('status') == status and row.get('query_id') == case:
            valid.append(row)
    if not valid:
        raise RuntimeError(f'Missing real live dev trace: {case}/{status}')
    row = valid[-1]
    if row.get('split') != 'dev' or row.get('model') == 'fake-in-tests-only':
        raise ValueError('Mock or non-dev trace not eligible')
    rendered = json.dumps(row, ensure_ascii=False, indent=2) + '\n'
    if re.search(r'(?i)(GEMINI_API_KEY\s*=|GOOGLE_API_KEY\s*=|postgresql://|x-goog-api-key)', rendered):
        raise ValueError(f'Potential credential information in {case}')
    destination = target / f'{case}.json'
    destination.write_text(rendered, encoding='utf-8')
    print(case, status, '->', destination.relative_to(root))
