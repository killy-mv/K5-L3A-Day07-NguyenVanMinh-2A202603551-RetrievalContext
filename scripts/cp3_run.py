"""CP3/CP4 CLI: real Gemini Agent and selectable retrieval; dev-only."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(ROOT))
from src.agent import AgentConfig, run_agent
from src.day6_adapter import database_url, default_day6_root
from src.retrieval import STRATEGIES, make_retriever
from src.gemini_client import GeminiClient
from scripts.cp2_validate import agent_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description='Real Gemini RAG: dense / BM25 / hybrid-RRF (CP3/CP4)')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--question', help='A new, non-benchmark question')
    selection.add_argument('--dev-id', help='Run exactly one frozen dev question, e.g. D7-D06')
    selection.add_argument('--dev-all', action='store_true', help='Run all 17 dev questions (API quota!)')
    parser.add_argument('--strategy', choices=STRATEGIES, default='dense')
    parser.add_argument('--candidate-k', type=int, default=12,
                        help='Candidates per retriever in hybrid RRF (>= top-k, default 12)')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--max-searches', type=int, default=3)
    parser.add_argument('--context-chars', type=int, default=3800)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    config = AgentConfig(args.top_k, args.max_searches, args.context_chars)
    if not 1 <= args.candidate_k <= 100:
        parser.error('--candidate-k must be 1..100')
    if args.output_dir is None:
        args.output_dir = ROOT / ('starter/results/cp3' if args.strategy == 'dense'
                                  else 'starter/results/cp4')
    if args.question:
        inputs = [{'id': 'ad-hoc', 'question': args.question}]
    else:
        questions = agent_inputs('dev', ROOT)  # only dev, no gold
        if args.dev_all:
            inputs = questions
        else:
            inputs = [q for q in questions if q['id'] == args.dev_id]
            if not inputs:
                parser.error('Unknown dev ID; only CP2 dev questions are available')

    try:
        model = GeminiClient.from_env()
        url = database_url(default_day6_root())
    except (ValueError, RuntimeError, OSError) as exc:
        # Configuration error, never log secret values.
        print(f'Configuration error: {type(exc).__name__}: set GEMINI_API_KEY in Lab 7 .env; ensure Day 6 DB is running.', file=sys.stderr)
        return 2

    retriever = make_retriever(args.strategy, url=url, candidate_k=args.candidate_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall_ok = True
    for q in inputs:
        record = run_agent(q['question'], model, retriever,
                           filters=q.get('filter'), config=config)
        record['retrieval_strategy'] = args.strategy
        if args.strategy == 'hybrid':
            record['rrf_k'] = retriever.rrf_k
            record['rrf_candidate_k'] = retriever.candidate_k
        record['query_id'] = q['id']
        record['split'] = 'ad-hoc' if args.question else 'dev'
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        destination = args.output_dir / f"{stamp}_{args.strategy}_{q['id']}.json"
        destination.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(f"\n[{q['id']}] status={record['status']} model={record['model']} "
              f"searches={record['search_calls']} calls={record['model_calls']} "
              f"elapsed_ms={record['elapsed_ms']}")
        if record['status'] == 'error':
            print('ERROR:', record.get('error'))
            overall_ok = False
        else:
            print(record['answer'])
            for citation in record['citation_details']:
                print(f"  SOURCE [{citation['id']}] PDF page {citation['page']} — {citation['source']}")
        print('Trace:', destination)
    return 0 if overall_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
