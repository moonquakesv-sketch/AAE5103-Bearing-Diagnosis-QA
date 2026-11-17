"""Regression checks for D01 review R1-R3; all mutations remain in memory/tempdirs."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import validate_d01 as validator
from data_contract import ContractError, validate_config, resample_signal, resample_parameters, processed_length


class MetadataTests(unittest.TestCase):
    def setUp(self):
        with (ROOT / 'data/records.csv').open() as f:
            self.rows = list(csv.DictReader(f))
        self.cfg = yaml.safe_load((ROOT / 'configs/data.yaml').read_text())
        self.audit = json.loads((ROOT / 'runs/d01/source_audit.json').read_text())

    def check(self):
        return validator.validate_record_metadata(self.rows, self.cfg, self.audit)

    def test_current_metadata(self):
        self.assertEqual(len(self.check()), 16)

    def test_swapped_labels_rejected(self):
        rows = {r['record_id']: r for r in self.rows}
        rows['cwru_105']['true_label'], rows['cwru_118']['true_label'] = 'rolling_element', 'inner_race'
        with self.assertRaisesRegex(ContractError, 'true_label'):
            self.check()

    def test_wrong_record_identity_rejected(self):
        original = copy.deepcopy(self.rows)
        for field, value in [('source_url', 'wrong'), ('official_name', 'IR007_3'),
                             ('load_hp', '3'), ('channel', 'X098_DE_time'),
                             ('fault_size', '0.021'), ('fault_position', '3:00')]:
            with self.subTest(field=field):
                self.rows = copy.deepcopy(original)
                next(r for r in self.rows if r['record_id'] == 'cwru_130')[field] = value
                with self.assertRaises(ContractError):
                    self.check()

    def test_noninteger_rate_ratio_rejected(self):
        next(r for r in self.rows if r['true_label'] == 'normal')['original_fs_hz'] = '48001'
        with self.assertRaisesRegex(ContractError, 'original_fs_hz'):
            self.check()

    def test_wrong_resampling_ratio_rejected(self):
        self.cfg['sampling']['resample']['normal_down'] = 2
        with self.assertRaisesRegex(ContractError, 'resampling ratio'):
            self.check()

    def test_invalid_resampling_factors_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                self.cfg['sampling']['resample']['normal_up'] = value
                with self.assertRaises(ContractError):
                    validate_config(self.cfg)

    def test_inconsistent_window_and_frequency_rules_rejected(self):
        original = copy.deepcopy(self.cfg)
        for section, field, value in [('cwt', 'sampling_period_seconds', 1 / 24000),
                                      ('segmentation', 'first_start_sample', 0),
                                      ('segmentation', 'stride_samples', 1024)]:
            with self.subTest(field=field):
                self.cfg = copy.deepcopy(original)
                self.cfg[section][field] = value
                with self.assertRaises(ContractError):
                    validate_config(self.cfg)

    def test_shared_resampling_uses_configuration(self):
        x = np.sin(np.arange(1001, dtype=float))
        expected = resample_signal(x, self.cfg, 48000, 'normal')
        self.cfg['sampling']['resample'].update(normal_up=2, normal_down=8)
        actual = resample_signal(x, self.cfg, 48000, 'normal')
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(processed_length(len(x), resample_parameters(self.cfg, 48000, 'normal')), len(actual))
        self.assertEqual(len(actual), 251)
        self.cfg['sampling']['resample']['window'] = ['kaiser', 1.0]
        self.assertFalse(np.array_equal(actual, resample_signal(x, self.cfg, 48000, 'normal')))


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'protocol.md').write_text('frozen')
        (self.root / 'baseline.sha256').write_text(validator.digest(b'frozen') + '  protocol.md\n')

    def test_snapshot_match(self):
        self.assertEqual(validator.verify_snapshot(self.root, 'baseline.sha256')['status'], 'PASS')

    def test_changed_or_missing_file_rejected(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                path = self.root / 'protocol.md'
                path.unlink() if missing else path.write_text('changed')
                result = validator.verify_snapshot(self.root, 'baseline.sha256')
                self.assertEqual(result['status'], 'FAIL')
                self.assertEqual(result['differences'][0]['path'], 'protocol.md')

    def test_snapshot_failure_skips_content_and_preserves_prior_report(self):
        (self.root / 'protocol.md').write_text('changed')
        with patch.object(validator, 'validate_content') as content:
            first, report = validator.execute(self.root, 'baseline.sha256')
            saved = first.read_bytes()
            second, _ = validator.execute(self.root, 'baseline.sha256')
            content.assert_not_called()
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['content_check']['status'], 'NOT_RUN')
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), saved)

    def test_content_failure_reported_separately(self):
        with patch.object(validator, 'validate_content', side_effect=ContractError('wrong label')):
            _, report = validator.execute(self.root, 'baseline.sha256')
        self.assertEqual(report['snapshot_check']['status'], 'PASS')
        self.assertEqual(report['content_check']['status'], 'FAIL')
        self.assertEqual(report['status'], 'FAIL')

    def test_historical_directory_protected_including_alias(self):
        historical = self.root / 'runs/d01'
        historical.mkdir(parents=True)
        (self.root / 'alias').symlink_to(historical, target_is_directory=True)
        for name in ('runs/d01/validation.json', 'alias/new.json'):
            with self.subTest(name=name), self.assertRaises(ContractError):
                validator.output_path(self.root, name)

    def test_existing_report_never_overwritten(self):
        path = self.root / 'report.json'
        path.write_text('original')
        with self.assertRaises(ContractError):
            validator.output_path(self.root, 'report.json')
        with self.assertRaises(FileExistsError):
            validator.write_report(path, {'status': 'PASS'})
        self.assertEqual(path.read_text(), 'original')

    def test_cli_returns_failure_on_mismatch(self):
        (self.root / 'protocol.md').write_text('changed')
        with patch.object(validator, 'ROOT', self.root), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(validator.main(['--baseline', 'baseline.sha256']), 1)


if __name__ == '__main__':
    unittest.main()
