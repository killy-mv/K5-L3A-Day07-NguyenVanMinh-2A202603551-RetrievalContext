"""Request-level Gemini rate limiter for CP7, persistent across CLI restarts.

Limits every generateContent request, not just the delay between questions.
No automatic retry on 429/503, and does not change the frozen model/prompt.
Do not run concurrent CP7 evaluation processes against the same pace file.
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
from typing import Callable


class PacedGemini:
    def __init__(self, provider, *, rpm: float = 1.0, state_file: Path,
                 now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        if not math.isfinite(rpm) or not 0 < rpm <= 60:
            raise ValueError('RPM must be in (0, 60]')
        self.provider = provider
        self.model = provider.model
        self.rpm = rpm
        self.interval_seconds = 60.0 / rpm
        self.state_file = Path(state_file)
        self.now = now
        self.sleep = sleep
        self.request_count = 0

    def generate(self, contents, *, instructions, force_search=False, allow_search=True):
        last = None
        if self.state_file.exists():
            value = json.loads(self.state_file.read_text(encoding='utf-8'))
            if not isinstance(value.get('last_request_epoch'), (int, float)):
                raise ValueError('Invalid persistent Gemini pacing state')
            last = float(value['last_request_epoch'])
        current = self.now()
        if last is not None:
            pause = max(0.0, last + self.interval_seconds - current)
            if pause:
                print(f'Gemini pacing: sleeping {pause:.1f}s before next request', flush=True)
                self.sleep(pause)
        current = self.now()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix('.tmp')
        temporary.write_text(json.dumps({'last_request_epoch':current,
                                         'rpm_limit':self.rpm})+'\n',encoding='utf-8')
        temporary.replace(self.state_file)
        self.request_count += 1
        return self.provider.generate(contents, instructions=instructions,
                                      force_search=force_search, allow_search=allow_search)
