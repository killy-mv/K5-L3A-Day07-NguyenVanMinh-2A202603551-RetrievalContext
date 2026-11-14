"""Gemini GenerateContent REST client, with native function calling.

Uses the official generateContent v1beta endpoint, not synthetic replies. Key
is sent in a header and never stored in logs or trace JSON. No SDK required.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MODEL = 'gemini-3.5-flash-lite'
ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/'
SEARCH_DECLARATION = {
    'name': 'search_documents',
    'description': ('Search the indexed Think Python chapters 1-4 for factual evidence. '
                    'Call before answering; use precise English search terms. '
                    'Return matching source chunks with their citation IDs.'),
    'parameters': {
        'type': 'OBJECT',
        'properties': {'query': {'type': 'STRING', 'description': 'Natural-language retrieval query'}},
        'required': ['query'],
    },
}


class ProviderError(RuntimeError):
    """A real model/transport failure; cannot be counted as abstention."""


@dataclass
class GeminiClient:
    api_key: str
    model: str = DEFAULT_MODEL
    timeout_s: int = 75

    def __post_init__(self):
        if not self.api_key or not self.api_key.strip():
            raise ValueError('GEMINI_API_KEY is missing. Set .env or process environment.')
        if not re.fullmatch(r'[A-Za-z0-9._-]{3,120}', self.model):
            raise ValueError('Invalid Gemini model name')
        if not 5 <= self.timeout_s <= 300:
            raise ValueError('Timeout must be between 5 and 300 seconds')

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        load_dotenv(root / '.env', override=False)
        return cls(
            api_key=os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or '',
            model=os.getenv('GEMINI_MODEL', DEFAULT_MODEL),
            timeout_s=int(os.getenv('GEMINI_TIMEOUT_SECONDS', '75')),
        )

    def generate(self, contents: list[dict], *, instructions: str,
                 force_search: bool = False, allow_search: bool = True) -> dict:
        if not contents:
            raise ValueError('Conversation content is required')
        payload = {
            'systemInstruction': {'parts': [{'text': instructions}]},
            'contents': contents,
            'generationConfig': {'temperature': 0.1, 'maxOutputTokens': 1200},
        }
        if allow_search:
            payload['tools'] = [{'functionDeclarations': [SEARCH_DECLARATION]}]
            # allowedFunctionNames is valid in ANY mode only (Gemini REST).
            function_cfg = {'mode': 'ANY' if force_search else 'AUTO'}
            if force_search:
                function_cfg['allowedFunctionNames'] = ['search_documents']
            payload['toolConfig'] = {'functionCallingConfig': function_cfg}
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        request = Request(
            ENDPOINT + self.model + ':generateContent', data=body,
            headers={'Content-Type': 'application/json', 'x-goog-api-key': self.api_key},
            method='POST',
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as resp:
                result = json.loads(resp.read().decode('utf-8'))
        except HTTPError as exc:
            # Provider codes only: raw errors may contain request content or credentials.
            try:
                detail = json.loads(exc.read().decode('utf-8')).get('error', {}).get('message', '')
            except (ValueError, UnicodeError):
                detail = ''
            diagnostic = ''
            for flag in ('thought_signature', 'thought signature', 'function response', 'functionResponse', 'INVALID_ARGUMENT', 'not found', 'quota'):
                if flag.lower() in detail.lower():
                    diagnostic = ' (' + flag.replace(' ', '_') + ')'
                    break
            # Never include server error body or user context in persistent traces.
            raise ProviderError(f'Gemini HTTP {exc.code}{diagnostic} (request/schema/quota)') from None
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f'Gemini network error ({type(exc).__name__})') from None
        except (ValueError, UnicodeError):
            raise ProviderError('Gemini returned invalid JSON') from None
        if not isinstance(result, dict) or not result.get('candidates'):
            raise ProviderError('Gemini returned no candidate (check content filters and API status)')
        first = result['candidates'][0]
        if not first.get('content', {}).get('parts'):
            raise ProviderError('Gemini candidate has no content parts')
        return result
