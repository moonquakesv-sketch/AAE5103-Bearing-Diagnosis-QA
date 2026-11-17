"""New52-unit scope fixture with the actual unchanged pinned runtime, offline only."""
from pathlib import Path
import json
import sys
import unittest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import test_rating_runtime as T


class Ref4Tests(unittest.TestCase):
    def test_full_52_units_156_slots_no_implicit_limit13(self):
        p, t, report, directory = T.run_case(n=52)
        self.assertEqual(t.calls, 156)
        self.assertEqual(report['valid_final_ratings'], 52)
        self.assertEqual(report['attempt_counts'], {'judge_output': 104, 'adjudicate_output': 52})
        self.assertEqual(report['reserved_USD'], '4.6006272')
        self.assertFalse(report['batch_stopped'])
        self.assertEqual(len(list(directory.glob('*.result.json'))), 52)
    def test_guard_stops_all52_after_first_transport(self):
        p, t, report, directory = T.run_case('alias', n=52)
        self.assertEqual(t.calls, 1)
        self.assertTrue(report['batch_stopped'])
        self.assertEqual(report['planned_questions'], 52)
        self.assertEqual(len(list(directory.glob('*.result.json'))), 52)
    def test_known_content_failure_consumes_without_retry_and_continues156(self):
        p, t, report, directory = T.run_case('model_bad_json', n=52)
        self.assertEqual(t.calls, 156)
        self.assertFalse(report['batch_stopped'])
        self.assertEqual(report['valid_final_ratings'], 52)
        first = T.read_json(directory / 'ratings/SYN-Q00_L_REF047_A.original.json')
        self.assertIsNone(first['rating'])
    def test_all_groups_keep_identical_reference_and_business_context(self):
        sources = T.read_json(ROOT / 'inputs/saved_answer_bindings.json')['sources']
        self.assertEqual(len(sources), 52)
        grouped = {}
        for source in sources:
            packet = T.read_json(ROOT / source['packet'])
            projection = {k: packet[k] for k in ['business_request', 'common_context', 'reference_points', 'reference_material']}
            qid = source['original_question_id']
            if qid in grouped:
                self.assertEqual(grouped[qid], projection)
            grouped[qid] = projection
            self.assertEqual(packet['quote_source_sha256'], source['observed_text_sha256'])
            self.assertNotIn('group', packet)
            self.assertNotIn('actual_complete_cards', packet)
            self.assertNotIn('draft', packet)
        self.assertEqual(len(grouped), 13)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Ref4Tests))
    print(json.dumps({'success': result.wasSuccessful(), 'tests': result.testsRun,
                      'failures': len(result.failures), 'errors': len(result.errors),
                      'real_model_calls': 0, 'transport': 'SYNTHETIC injected only'}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
