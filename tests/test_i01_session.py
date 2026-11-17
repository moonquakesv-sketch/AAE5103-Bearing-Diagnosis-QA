"""I01 new-batch controls; real durable ledgers with synthetic static release.

Only the static release lookup is replaced. No provider, classifier, old run or
evaluation data is accessed. Temporary evidence stays inside the active run.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import i01_session as session
import q01_flow as flow

ACTIVE = Path(os.environ.get('I01_ACTIVE_RUN', ROOT / 'runs/i01/20261026_031418_385e0a04'))


class BatchControls(unittest.TestCase):
    def setUp(self):
        (ACTIVE / 'tmp').mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='batch_controls_', dir=ACTIVE / 'tmp'))
        self.run = self.root / 'runs/i01/synthetic'
        self.run.mkdir(parents=True)
        session.put(self.run / 'batch_release.json', {'kind': 'SYNTHETIC static release'})
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(session, 'ROOT', self.root))
        self.stack.enter_context(patch.object(session, '_batch_release', return_value={}))
        self.stack.enter_context(patch.object(socket, 'socket', side_effect=AssertionError('No network in offline controls')))
        self.budgets = []

    def tearDown(self):
        for budget in self.budgets:
            budget.close()
        self.stack.close()

    def budget(self):
        item = session.BatchBudget(self.run)
        self.budgets.append(item)
        return item

    def test_twelve_nonrefundable_credits_and_no_thirteenth(self):
        (self.run / 'cases').mkdir()  # An empty pre-created output folder is safe.
        budget = self.budget()
        for number in range(12):
            case = budget.claim_flow()
            self.assertEqual(case.name, f'{number + 1:03d}')
            budget.finish_flow(case, 'synthetic_failed', identify_attempts=0, qa_attempts=0)
            with self.assertRaises(session.Error):
                budget.finish_flow(case, 'duplicate', identify_attempts=0, qa_attempts=0)
        summary = budget.snapshot()
        self.assertEqual(summary['claimed_flows'], 12)
        self.assertEqual(summary['remaining_flows'], 0)
        self.assertEqual(summary['reserved_credit_usd'], '0.24')
        self.assertEqual(summary['reserved_attempt_ceiling'], 60)
        self.assertEqual(summary['actual_attempts'], 0)
        with self.assertRaises(session.Error) as rejected:
            budget.claim_flow()
        self.assertEqual(rejected.exception.code, 'batch_budget_exhausted')

    def test_restart_restores_failed_and_unfinished_credits(self):
        budget = self.budget()
        failed = budget.claim_flow()
        budget.finish_flow(failed, 'failed', identify_attempts=1, qa_attempts=2)
        unfinished = budget.claim_flow()
        budget.close()
        resumed = self.budget()
        snapshot = resumed.snapshot()
        self.assertEqual(snapshot['claimed_flows'], 2)
        self.assertEqual(snapshot['reserved_credit_usd'], '0.04')
        self.assertEqual(snapshot['reserved_attempt_ceiling'], 10)
        self.assertEqual(snapshot['actual_attempts'], 'unknown')
        self.assertEqual(snapshot['unfinished_cases'], [unfinished.name])
        self.assertEqual(resumed.claim_flow().name, '003')

    def test_second_owner_is_rejected_without_changing_first(self):
        budget = self.budget()
        before = (self.run / 'batch_ledger.jsonl').read_bytes()
        with self.assertRaises(session.Error):
            session.BatchBudget(self.run)
        self.assertEqual((self.run / 'batch_ledger.jsonl').read_bytes(), before)
        self.assertEqual(budget.snapshot()['remaining_flows'], 12)

    def test_concurrent_claims_never_exceed_twelve(self):
        budget = self.budget()
        def claim(_):
            try:
                return budget.claim_flow().name
            except session.Error as error:
                self.assertEqual(error.code, 'batch_budget_exhausted')
                return None
        with ThreadPoolExecutor(max_workers=8) as executor:
            claims = list(executor.map(claim, range(32)))
        self.assertEqual(sorted(c for c in claims if c), [f'{n:03d}' for n in range(1, 13)])
        self.assertEqual(budget.snapshot()['reserved_credit_usd'], '0.24')

    def test_modified_ledger_poison_and_restart_fail_closed(self):
        budget = self.budget()
        budget.claim_flow()
        with (self.run / 'batch_ledger.jsonl').open('ab') as stream:
            stream.write(b'{"unregistered":"append"}\n')
        with self.assertRaises(session.Error):
            budget.snapshot()
        with self.assertRaises(session.Error):
            budget.claim_flow()
        budget.close()
        with self.assertRaises(session.Error):
            self.budget()

    def test_replaced_ledger_inode_is_rejected(self):
        budget = self.budget()
        budget.claim_flow()
        path = self.run / 'batch_ledger.jsonl'
        replacement = self.run / 'replacement.jsonl'
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        with self.assertRaises(session.Error):
            budget.snapshot()
        budget.close()
        with self.assertRaises(session.Error):
            self.budget()

    def test_missing_ledger_does_not_reset_evidence(self):
        budget = self.budget()
        budget.claim_flow()
        budget.close()
        (self.run / 'batch_ledger.jsonl').unlink()
        with self.assertRaises(session.Error):
            self.budget()

    def test_incomplete_claim_retains_append_and_blocks_recovery(self):
        budget = self.budget()
        original = session.put
        def fail_claim(path, value):
            if path.name == 'flow_claim.json':
                raise OSError('SYNTHETIC one-shot claim write failure')
            return original(path, value)
        with patch.object(session, 'put', side_effect=fail_claim):
            with self.assertRaises(session.Error):
                budget.claim_flow()
        rows = [json.loads(line) for line in (self.run / 'batch_ledger.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['event'], 'claim_flow')
        self.assertEqual(rows[-1]['credit_usd'], '0.02')
        with self.assertRaises(session.Error):
            budget.claim_flow()
        budget.close()
        with self.assertRaises(session.Error):
            self.budget()

    def test_failed_checkpoint_cannot_be_reset_or_refunded(self):
        budget = self.budget()
        original = session.put
        def fail_checkpoint(path, value):
            if path.parent.name == 'batch_ledger_hashes':
                raise OSError('SYNTHETIC checkpoint failure')
            return original(path, value)
        with patch.object(session, 'put', side_effect=fail_checkpoint):
            with self.assertRaises(session.Error):
                budget.claim_flow()
        self.assertIn(b'claim_flow', (self.run / 'batch_ledger.jsonl').read_bytes())
        budget.close()
        with self.assertRaises(session.Error):
            self.budget()

    def test_finish_counts_enforce_one_plus_four_and_unknown_stays_unknown(self):
        budget = self.budget()
        case = budget.claim_flow()
        for identify, qa in [(2, 0), (0, 5), (-1, 0), (True, 0), (0, False)]:
            with self.subTest(identify=identify, qa=qa), self.assertRaises(ValueError):
                budget.finish_flow(case, 'failed', identify_attempts=identify, qa_attempts=qa)
        budget.finish_flow(case, 'failed', identify_attempts='unknown', qa_attempts='unknown')
        snapshot = budget.snapshot()
        self.assertEqual(snapshot['reserved_credit_usd'], '0.02')
        self.assertEqual(snapshot['reserved_attempt_ceiling'], 5)
        self.assertEqual(snapshot['actual_attempts'], 'unknown')


class SubmissionBinding(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='submission_binding_', dir=ACTIVE / 'tmp'))
        self.batch = self.root / 'runs/i01/synthetic'
        self.case = self.batch / 'cases/001'
        (self.case / 'inputs').mkdir(parents=True)
        session.put(self.batch / 'batch_release.json', {'kind': 'SYNTHETIC static gate release'})
        self.submission = {'question': 'What should be inspected?',
                           'explicitly_allowed_operating_context': {}, 'sample_choice': 'sample_1'}
        self.sample = {'selected_row': {'sample_id': 'synthetic_sample', 'split': 'validation', 'true_label': 'inner_race'},
                       'file_hashes': {'synthetic.png': 'a' * 64}}
        self.classifier = {'kind': 'fresh_image_inference', 'sample_id': 'synthetic_sample',
                           'predicted_class': 'outer_race', 'prediction': {'true_label': 'inner_race'}}
        self.checked = {'release': {}, 'registry': {'samples': {'sample_1': self.sample}},
                        'profile': {}, 'intent_config': {}, 'prompts': {}}
        claim = {'event': 'claim_flow', 'case_id': '001',
                 'case_path': self.case.relative_to(self.root).as_posix(),
                 'batch_release_sha256': flow.file_hash(self.batch / 'batch_release.json')}
        session.put(self.case / 'flow_claim.json', claim)
        (self.batch / 'batch_ledger.jsonl').write_text(flow.canonical(claim) + '\n')
        session.put(self.case / 'inputs/sample_registration.json', self.sample)
        session.put(self.case / 'classifier_record.json', self.classifier)
        self.persist_submission(self.submission)
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(session, 'ROOT', self.root))
        self.stack.enter_context(patch.object(session, 'check_release', return_value=self.checked))
        self.stack.enter_context(patch.object(socket, 'socket', side_effect=AssertionError('No network in binding tests')))

    def tearDown(self):
        self.stack.close()

    def persist_submission(self, submission):
        for name in ('submission.json', 'source_registration.json', 'four_fields.json'):
            path = self.case / 'inputs' / name
            if path.exists():
                path.unlink()  # Only synthetic test fixtures are rewritten.
        raw = session.put(self.case / 'inputs/submission.json', submission)
        session.put(self.case / 'inputs/source_registration.json', {
            'path': (self.case / 'inputs/submission.json').relative_to(self.root).as_posix(),
            'purpose': 'SYNTHETIC page submission before parsing',
            'actual_sha256': hashlib.sha256(raw).hexdigest()})
        session.put(self.case / 'inputs/four_fields.json', {
            'predicted_class': 'outer_race', 'scenario_id': 'SC-K01-OR-001',
            'question': submission['question'],
            'explicitly_allowed_operating_context': submission['explicitly_allowed_operating_context']})

    def gate(self, classifier=None):
        return session.LiveGate(self.case, SimpleNamespace(identity={'synthetic': True}),
                                classifier or self.classifier, batch_run=self.batch,
                                actual_submission=self.submission)

    def test_actual_prediction_controls_route_despite_metadata_label(self):
        gate = self.gate()
        self.assertEqual(gate.projection['predicted_class'], 'outer_race')
        self.assertEqual(gate.projection['scenario_id'], 'SC-K01-OR-001')
        self.assertNotIn('true_label', gate.projection)
        self.assertNotIn('sample_choice', gate.projection)
        self.assertNotIn('task_types', gate.projection)

    def test_consistent_disk_substitution_cannot_replace_actual_submission(self):
        replacement = dict(self.submission, question='What should be prepared?')
        self.persist_submission(replacement)
        with self.assertRaises(session.Error) as rejected:
            self.gate()
        self.assertEqual(rejected.exception.code, 'request_identity_mismatch')

    def test_persisted_prediction_cannot_replace_memory_held_inference(self):
        actual = dict(self.classifier, predicted_class='normal')
        with self.assertRaises(session.Error) as rejected:
            self.gate(actual)
        self.assertEqual(rejected.exception.code, 'classifier_binding_invalid')

    def test_mutating_source_after_gate_is_rejected_before_send(self):
        gate = self.gate()
        self.persist_submission(dict(self.submission, question='Changed after gate?'))
        with self.assertRaises(session.Error):
            gate.check()


if __name__ == '__main__':
    unittest.main()
