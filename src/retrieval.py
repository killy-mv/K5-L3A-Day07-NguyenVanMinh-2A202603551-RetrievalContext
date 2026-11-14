"""CP4: three comparable retrieval methods on the same Day 6 strategy-B corpus.

Dense: normalized MiniLM embeddings from PostgreSQL/pgvector.
Lexical: in-memory BM25 over the exact 160 CP1 chunk texts.
Hybrid: reciprocal-rank fusion (RRF) of dense and BM25 ranked candidates.

No benchmark labels or expected answers are consulted by these methods.
All interfaces implement search(query, filters=..., top_k=...) for src.agent.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .day6_adapter import Day6DenseRetriever, validate_retrieval_params

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
DEFAULT_CANDIDATES = 12
STRATEGIES = ('dense', 'lexical', 'hybrid')


def tokenize(text: str) -> list[str]:
    """Deterministic Unicode word tokens; no gold-dependent preprocessing."""
    if not isinstance(text, str):
        raise TypeError('Expected text string')
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Load only the published CP1 chunk schema; never benchmark labels."""
    if not path.is_file():
        raise FileNotFoundError(f'CP1 corpus missing: {path}; run scripts/cp1_prepare.py')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list) or not data:
        raise ValueError('CP1 chunks must be a nonempty JSON array')
    seen = set()
    for doc in data:
        if not isinstance(doc, dict):
            raise ValueError('Invalid CP1 chunk row')
        for field in ('id', 'text', 'source_id', 'source'):
            if not isinstance(doc.get(field), str) or not doc[field].strip():
                raise ValueError(f'CP1 chunk missing {field}')
        if not isinstance(doc.get('page'), int) or doc['page'] < 1:
            raise ValueError('CP1 chunk has invalid PDF page')
        if not isinstance(doc.get('metadata'), dict) or not doc['metadata'].get('chapter'):
            raise ValueError('CP1 chunk has invalid metadata')
        if doc['id'] in seen:
            raise ValueError(f'Duplicate CP1 chunk ID {doc["id"]}')
        seen.add(doc['id'])
    return data


class BM25Retriever:
    """A standard Robertson BM25 index computed from the CP1 chunk corpus.

    Output score is a monotone transform raw / (1 + raw) in [0,1), so it
    meets src.agent's normalized score contract. Raw BM25 remains in traces.
    """
    strategy = 'lexical'

    def __init__(self, chunks: list[dict[str, Any]], *,
                 k1: float = BM25_K1, b: float = BM25_B) -> None:
        if not chunks or not (math.isfinite(k1) and k1 > 0) or not (0 <= b <= 1):
            raise ValueError('Invalid BM25 corpus or hyperparameters')
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(tokenize(doc['text'])) for doc in chunks]
        self.lengths = [sum(freq.values()) for freq in self.term_freqs]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        self.idf: dict[str, float] = {}
        doc_freq = Counter(term for freq in self.term_freqs for term in freq)
        size = len(chunks)
        for term, df in doc_freq.items():
            self.idf[term] = math.log(1 + (size - df + .5) / (df + .5))

    def search(self, query: str, *, filters: dict[str, str] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        validate_retrieval_params(query, filters, top_k)
        terms = set(tokenize(query))
        matches: list[dict] = []
        for chunk, freq, length in zip(self.chunks, self.term_freqs, self.lengths):
            if filters and chunk['metadata'].get('chapter') != filters['chapter']:
                continue
            raw = 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if tf:
                    divisor = tf + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
                    raw += self.idf[term] * (tf * (self.k1 + 1)) / divisor
            if raw > 0:
                matches.append({**chunk, 'score': raw / (1 + raw),
                                'raw_score': raw, 'score_type': 'bm25_normalized',
                                'strategy': self.strategy})
        matches.sort(key=lambda row: (-row['score'], row['id']))
        return matches[:top_k]


class HybridRRFRetriever:
    """Merge independent dense+lexical rankings without adding raw scores.

    RRF score = sum(1/(RRF_K + rank)). Normalize by max possible for
    2 rank-one hits so all scored chunks are in (0,1]. Rank is 1-based.
    """
    strategy = 'hybrid'

    def __init__(self, dense: Any, lexical: Any, *,
                 rrf_k: int = RRF_K, candidate_k: int = DEFAULT_CANDIDATES,
                 dense_weight: float = 1.0, lexical_weight: float = 1.0):
        if type(rrf_k) is not int or rrf_k < 1:
            raise ValueError('rrf_k must be a positive integer')
        if type(candidate_k) is not int or not 1 <= candidate_k <= 100:
            raise ValueError('candidate_k must be 1..100')
        self.dense, self.lexical = dense, lexical
        if (not math.isfinite(dense_weight) or not math.isfinite(lexical_weight) or
                dense_weight <= 0 or lexical_weight <= 0):
            raise ValueError('RRF weights must be positive finite numbers')
        self.rrf_k, self.candidate_k = rrf_k, candidate_k
        self.dense_weight, self.lexical_weight = dense_weight, lexical_weight

    def search(self, query: str, *, filters: dict[str, str] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        validate_retrieval_params(query, filters, top_k)
        requested = max(self.candidate_k, top_k)
        dense = self.dense.search(query, filters=filters, top_k=requested)
        lexical = self.lexical.search(query, filters=filters, top_k=requested)
        rows: dict[str, dict] = {}
        for label, ranked in (('dense', dense), ('lexical', lexical)):
            seen = set()
            for rank, chunk in enumerate(ranked, 1):
                cid = chunk['id']
                if cid in seen:
                    raise ValueError(f'{label} returned duplicate chunk ID')
                seen.add(cid)
                if filters and chunk['metadata'].get('chapter') != filters['chapter']:
                    raise ValueError(f'{label} violated trusted chapter filter')
                if cid not in rows:
                    rows[cid] = {**{k: chunk[k] for k in
                                     ('id', 'source_id', 'source', 'page', 'text', 'metadata')},
                                 'score': 0.0, 'score_type': 'rrf_normalized',
                                 'strategy': self.strategy, 'rrf_score': 0.0,
                                 'ranks': {}, 'component_scores': {}}
                else:
                    prev = rows[cid]
                    # Two indexes MUST refer to the same immutable source chunk.
                    for field in ('source_id', 'source', 'page', 'text', 'metadata'):
                        if prev[field] != chunk[field]:
                            raise ValueError(f'Mismatched dense/lexical content for {cid}')
                rows[cid]['ranks'][label] = rank
                rows[cid]['component_scores'][label] = chunk['score']
                weight = self.dense_weight if label == 'dense' else self.lexical_weight
                rows[cid]['rrf_score'] += weight / (self.rrf_k + rank)
        for row in rows.values():
            row['score'] = row['rrf_score'] * (self.rrf_k + 1) / (self.dense_weight + self.lexical_weight)
        ordered = sorted(rows.values(), key=lambda row: (-row['score'], row['id']))
        return ordered[:top_k]


def make_retriever(strategy: str, *, url: str | None = None,
                   corpus_path: Path | None = None,
                   dense: Any = None, lexical: Any = None,
                   candidate_k: int = DEFAULT_CANDIDATES,
                   dense_weight: float = 1.0, lexical_weight: float = 1.0) -> Any:
    """Build only components needed by the selected strategy.

    Tests may inject custom retrievers; CLI always uses real CP1 corpus and
    Day 6 pgvector. Providing dense=None with strategy=hybrid needs a DB URL.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f'Unknown retrieval strategy: {strategy}')
    if strategy in ('lexical', 'hybrid') and lexical is None:
        path = corpus_path or Path(__file__).resolve().parents[1] / 'starter/data/chunks.json'
        lexical = BM25Retriever(load_corpus(path))
    if strategy in ('dense', 'hybrid') and dense is None:
        if not url:
            raise ValueError('DATABASE_URL required for dense or hybrid retrieval')
        dense = Day6DenseRetriever(url)
    if strategy == 'dense':
        return dense
    if strategy == 'lexical':
        return lexical
    return HybridRRFRetriever(dense, lexical, candidate_k=candidate_k,
                              dense_weight=dense_weight, lexical_weight=lexical_weight)
