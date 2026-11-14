"""CP1 tests for the real Day 6-to-Day 7 adapter contract."""
import unittest
from pathlib import Path
from unittest.mock import patch

from src.day6_adapter import (
    DIMENSIONS, Day6DenseRetriever, _to_day7_chunk, default_day6_root,
    load_day6_chunks, validate_retrieval_params,
)


class Day6AdapterTests(unittest.TestCase):
    def test_default_day6_root_sibling(self):
        self.assertEqual(default_day6_root().name, 'K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion')

    def test_day6_chunk_preserves_source_page_and_metadata(self):
        item = _to_day7_chunk({
            'chunk_id': 'one', 'document_id': 'doc', 'source': 'https://example.org/book.pdf',
            'page': 33, 'text': 'Python function', 'metadata': {'chapter': 'chapter_2'},
        })
        self.assertEqual(item['id'], 'one')
        self.assertEqual(item['source_id'], 'doc')
        self.assertEqual(item['page'], 33)
        self.assertEqual(item['metadata']['chapter'], 'chapter_2')

    def test_reject_invalid_chunk_metadata(self):
        with self.assertRaises(ValueError):
            _to_day7_chunk({
                'chunk_id': 'a', 'document_id': 'b', 'source': 'x',
                'page': 1, 'text': 'x', 'metadata': {},
            })

    def test_query_and_filter_validation(self):
        for query, filt, count in [('', None, 3), ('ok', {'gold': 'x'}, 3),
                                   ('ok', {'chapter': ''}, 3), ('ok', None, 0)]:
            with self.subTest(query=query, filters=filt, top_k=count):
                with self.assertRaises(ValueError):
                    validate_retrieval_params(query, filt, count)
        validate_retrieval_params('ok', {'chapter': 'chapter_4'}, 3)

    def test_retriever_rejects_bad_query_embedding(self):
        class BrokenEmbedder:
            def embed_query(self, query):
                return [0.1] * (DIMENSIONS - 1)
        retriever = Day6DenseRetriever('postgres://placeholder', BrokenEmbedder())
        with self.assertRaisesRegex(ValueError, 'dimension'):
            retriever.search('query')


if __name__ == '__main__':
    unittest.main()
