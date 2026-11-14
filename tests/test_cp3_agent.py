"""CP3 deterministic protocol tests. These do NOT prove real model quality."""
import unittest

from src.agent import ABSTENTION, AgentConfig, build_context, run_agent
from src.gemini_client import GeminiClient


def hit(chunk_id='paragraph-test', score=.83):
    return {'id': chunk_id, 'text': 'A variable is a name that refers to a value.',
            'source_id': 'thinkpython2_pdf', 'source': 'https://example.org/book.pdf',
            'page': 31, 'metadata': {'chapter': 'chapter_2'}, 'score': score}


def tool_call(query='python variable', call_id='call1'):
    return {'candidates': [{'content': {'role': 'model', 'parts': [
        {'functionCall': {'name': 'search_documents', 'args': {'query': query}, 'id': call_id}}
    ]}, 'finishReason': 'STOP'}], 'usageMetadata': {'promptTokenCount': 20, 'candidatesTokenCount': 8, 'totalTokenCount': 28}}


def text_response(text):
    return {'candidates': [{'content': {'role': 'model', 'parts': [{'text': text}]}, 'finishReason': 'STOP'}],
            'usageMetadata': {'promptTokenCount': 70, 'candidatesTokenCount': 30, 'totalTokenCount': 100}}


class StubModel:
    model = 'fake-in-tests-only'
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, contents, *, instructions, force_search=False, allow_search=True):
        self.calls.append((contents, force_search, allow_search))
        if not self.responses:
            raise RuntimeError('No test model response left')
        return self.responses.pop(0)


class StubRetriever:
    def __init__(self, rows=None, fail=False):
        self.rows = rows if rows is not None else [hit()]
        self.fail = fail
        self.calls = []

    def search(self, query, *, filters, top_k):
        self.calls.append((query, filters, top_k))
        if self.fail:
            raise RuntimeError('mock retriever unavailable')
        return self.rows[:top_k]


class AgentTests(unittest.TestCase):
    def test_search_before_answer_and_real_citation_id(self):
        model = StubModel([tool_call(), text_response('A variable names a value [paragraph-test].')])
        retriever = StubRetriever()
        output = run_agent('What is a variable?', model, retriever,
                           filters={'chapter': 'chapter_2'})
        self.assertEqual(output['status'], 'answered')
        self.assertEqual(output['citations'], ['paragraph-test'])
        self.assertEqual(output['citation_details'][0]['page'], 31)
        self.assertEqual(output['token_usage']['total_tokens'], 128)
        self.assertEqual(output['search_calls'], 1)
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(retriever.calls[0][1], {'chapter': 'chapter_2'})
        self.assertTrue(model.calls[0][1])  # forced tool on first call
        reply = model.calls[1][0][-1]['parts'][0]['functionResponse']
        self.assertEqual(reply['name'], 'search_documents')
        self.assertEqual(reply['id'], 'call1')
        self.assertIn('paragraph-test', reply['response']['context'])

    def test_grouped_citations_are_supported_without_inventing_ids(self):
        model = StubModel([tool_call(), text_response(
            'A variable names a value [paragraph-test, paragraph-second].')])
        retriever = StubRetriever([hit(), hit('paragraph-second', .7)])
        output = run_agent('What is a variable?', model, retriever)
        self.assertEqual(output['status'], 'answered')
        self.assertEqual(output['citations'], ['paragraph-test', 'paragraph-second'])

    def test_unknown_citation_not_accepted(self):
        model = StubModel([tool_call(), text_response('Invented [paragraph-fabricated].')])
        data = run_agent('variable?', model, StubRetriever())
        self.assertEqual(data['status'], 'abstain')
        self.assertEqual(data['answer'], ABSTENTION)
        self.assertEqual(data['citations'], [])
        self.assertTrue(any(e['event'] == 'invalid_citation' for e in data['events']))

    def test_missing_evidence_abstains_without_fake_citations(self):
        model = StubModel([tool_call('nonexistent syntax'), text_response('Not enough evidence.')])
        data = run_agent('Describe async syntax', model, StubRetriever([]))
        self.assertEqual(data['status'], 'abstain')
        self.assertEqual(data['search_calls'], 1)

    def test_multiple_searches_and_forced_final(self):
        model = StubModel([tool_call('variable', 'a'), tool_call('variable again', 'b'),
                           text_response('A variable names a value [paragraph-test].')])
        output = run_agent('What is a variable?', model, StubRetriever(),
                           config=AgentConfig(max_searches=2))
        self.assertEqual(output['status'], 'answered')
        self.assertEqual(output['search_calls'], 2)
        self.assertFalse(model.calls[2][2])  # tools disabled after search limit
        self.assertEqual(output['context_ids'], ['paragraph-test'])  # dedup

    def test_tool_failure_is_error_not_abstention(self):
        output = run_agent('What is a variable?', StubModel([tool_call()]),
                           StubRetriever(fail=True))
        self.assertEqual(output['status'], 'error')
        self.assertIn('RuntimeError', output['error'])
        self.assertFalse(output['citations'])

    def test_no_retrieval_cannot_count_as_answer(self):
        output = run_agent('What is a variable?', StubModel([text_response('I know.')]), StubRetriever())
        self.assertEqual(output['status'], 'error')
        self.assertEqual(output['error'], 'model_skipped_required_retrieval')

    def test_context_budget_excludes_oversized_and_keeps_source(self):
        context, ids = build_context({'a': hit('a'), 'b': hit('b', .7)}, 600)
        self.assertIn('page=31', context)
        self.assertTrue(ids)
        self.assertLessEqual(len(context), 600)

    def test_reject_untrusted_tool_filter_arguments(self):
        bad = tool_call()
        bad['candidates'][0]['content']['parts'][0]['functionCall']['args']['chapter'] = 'chapter_1'
        output = run_agent('variable?', StubModel([bad]), StubRetriever())
        self.assertEqual(output['status'], 'error')

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            AgentConfig(max_searches=0)
        with self.assertRaises(ValueError):
            GeminiClient(api_key='', model='gemini-3.5-flash-lite')


if __name__ == '__main__':
    unittest.main()
