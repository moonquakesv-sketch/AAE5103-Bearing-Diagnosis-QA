"""Synthetic arithmetic checks only; no retrieval or answer generation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import check_e01_metrics as metrics

EXAMPLES_PATH = ROOT / 'runs/e01/20261024_193035_937d9e18/synthetic_metrics.json'


class TestSyntheticMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = metrics.load_strict_json(EXAMPLES_PATH.read_text(encoding='utf-8'))
        cls.examples = {item['id']: item for item in cls.bundle['examples']}

    def test_01_counts_and_value_preserved(self):
        self.assertEqual(metrics.ratio(1, 2), {
            'numerator': 1, 'denominator': 2, 'value': 0.5, 'status': 'measured'})

    def test_02_zero_denominator_is_not_applicable(self):
        result = metrics.ratio(0, 0)
        self.assertIsNone(result['value'])
        self.assertEqual(result['status'], 'not_applicable')

    def test_03_negative_denominator_rejected(self):
        with self.assertRaises(metrics.MetricsError):
            metrics.ratio(0, -1)

    def test_04_boolean_counts_rejected(self):
        for first, second in ((True, 2), (0, False)):
            with self.subTest(values=(first, second)):
                with self.assertRaises(metrics.MetricsError):
                    metrics.ratio(first, second)

    def test_05_bounded_count_and_zero_denominator_consistency(self):
        for first, second in ((2, 1), (1, 0)):
            with self.subTest(values=(first, second)):
                with self.assertRaises(metrics.MetricsError):
                    metrics.ratio(first, second)

    def test_06_scores_use_declared_half_credit(self):
        self.assertEqual(metrics.score_ratio([1, 0.5, 0])['value'], 0.5)

    def test_07_illegal_or_nonfinite_scores_rejected(self):
        for value in (1.5, -0.5, float('nan'), float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(metrics.MetricsError):
                    metrics.score_ratio([value])

    def test_08_boolean_score_rejected(self):
        with self.assertRaises(metrics.MetricsError):
            metrics.score_ratio([True])

    def test_09_incomplete_and_does_not_get_partial_by_default(self):
        atom = {'relevant': ['A', 'B'], 'support_any_of': [['A', 'B']],
                'partial_any_of': []}
        self.assertEqual(metrics.support_score(atom, ['A']), 0.0)
        self.assertEqual(metrics.support_score(atom, ['A', 'B']), 1.0)

    def test_10_or_alternative_does_not_require_all_cards(self):
        atom = self.examples['S1']['atoms'][0]
        self.assertEqual(metrics.support_score(atom, ['C']), 1.0)

    def test_11_only_explicit_partial_combination_gets_half(self):
        atom = self.examples['S1']['atoms'][1]
        self.assertEqual(metrics.support_score(atom, ['E']), 0.5)
        self.assertEqual(metrics.support_score(atom, []), 0.0)

    def test_12_error_id_duplicates_do_not_inflate_counts(self):
        actual = metrics.error_change_metrics(['e1', 'e1', 'e2', 'e3'],
                                              ['e2', 'e2', 'e4'],
                                              ['e1', 'e1', 'e2', 'x'])
        self.assertAlmostEqual(actual['correction_rate']['value'], 2 / 3)
        self.assertEqual(actual['false_alarm_rate']['denominator'], 3)
        self.assertEqual(actual['new_error_count']['value'], 1)

    def test_13_service_failure_is_not_reasonable_abstention(self):
        actual = metrics.compute_example(self.examples['S3_failure'])
        self.assertEqual(actual['insufficiency_handling']['value'], 0)
        self.assertEqual(actual['failure_count']['value'], 1)
        self.assertEqual(actual['unsupported_claim_rate']['status'],
                         'missing_due_to_failure')

    def test_14_no_citations_does_not_mean_perfect_support(self):
        actual = metrics.compute_example(self.examples['S3'])
        self.assertIsNone(actual['citation_id_error_rate']['value'])
        self.assertEqual(actual['citation_id_error_rate']['denominator'], 0)
        self.assertEqual(actual['insufficiency_handling']['value'], 1)

    def test_15_all_synthetic_examples_match_and_mismatch_is_detected(self):
        self.assertTrue(metrics.check_bundle(self.bundle)['all_passed'])
        changed = copy.deepcopy(self.bundle)
        changed['examples'][0]['expected']['support_coverage'] = 1.0
        self.assertFalse(metrics.check_bundle(changed)['all_passed'])

    def test_16_strict_synthetic_json_and_schema(self):
        for data in ('{"synthetic":true,"synthetic":true}', '{"n":NaN}'):
            with self.subTest(data=data):
                with self.assertRaises(metrics.MetricsError):
                    metrics.load_strict_json(data)
        changed = copy.deepcopy(self.bundle)
        changed['synthetic'] = False
        with self.assertRaises(metrics.MetricsError):
            metrics.check_bundle(changed)

    def test_17_cli_roundtrip_and_exclusive_output(self):
        base = Path(os.environ.get('E01_TEST_RUN', str(EXAMPLES_PATH.parent)))
        self.assertTrue(base.is_dir(), 'E01_TEST_RUN must be an existing new run directory.')
        with tempfile.TemporaryDirectory(prefix='metrics_synthetic_', dir=base) as temporary:
            output = Path(temporary) / 'result.json'
            command = [sys.executable, str(ROOT / 'scripts/check_e01_metrics.py'),
                       '--examples', str(EXAMPLES_PATH), '--output', str(output)]
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            first = subprocess.run(command, env=environment, capture_output=True,
                                   text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = output.read_bytes()
            self.assertTrue(json.loads(original)['all_passed'])
            second = subprocess.run(command, env=environment, capture_output=True,
                                    text=True, check=False)
            self.assertEqual(second.returncode, 2)
            self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
