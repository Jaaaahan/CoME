"""QA-only bundles must not block evidence selection or pruning."""
import unittest
from scripts.evaluate_atm_retrieval import metrics_for

class OptionalRetrievalMetricsTest(unittest.TestCase):
    def test_unannotated_bundle_is_explicitly_unscored(self):
        q = {'question_id': 'q', 'answer': {'text': 'reference'}}
        actual = metrics_for([q], {'q': {'retrieved_memory_ids': ['m']}}, 20, allow_empty=True)
        self.assertEqual(actual['status'], 'unavailable')
        self.assertEqual(actual['questions'], 0)
        self.assertIsNone(actual['R@20'])
        with self.assertRaisesRegex(ValueError, 'no questions with gold evidence'):
            metrics_for([q], {}, 20)

    def test_annotated_bundle_keeps_metrics(self):
        q = {'question_id': 'q', 'evidence': [{'memory_id': 'm'}]}
        actual = metrics_for([q], {'q': {'retrieved_memory_ids': ['m']}}, 20, allow_empty=True)
        self.assertEqual(actual['R@20'], 1.0)
        self.assertEqual(actual['questions'], 1)

if __name__ == '__main__':
    unittest.main()
