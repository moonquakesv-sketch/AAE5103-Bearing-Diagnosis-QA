"""CTRL-P2-001 acceptance tests; all inference/provider results are synthetic.

The production finalizer and durable claim are exercised with real I01 budget
constructors. Classifier, tokenizer/release, gate and provider boundaries are
explicit synthetic replacements; this suite tests finalization, not those gates.
No old test fixtures are imported. All new files remain under rev024/tmp.

I01_PAGE_SOURCE can select the preserved before source for the same positive
acceptance oracle. It never changes an existing source or historical run.
"""
import copy
from contextlib import ExitStack
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'runs/i01_fix/rev024'
PAGE_SOURCE = Path(os.environ.get('I01_PAGE_SOURCE', ROOT / 'scripts/i01_page.py')).absolute()
sys.path.insert(0, str(ROOT / 'scripts'))
import i01_http as wire
import q01_flow as flow


def load_page(name):
    spec = importlib.util.spec_from_file_location(name, PAGE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


page = load_page('i01_finalization_under_test')
SECRET = 'SYNTHETIC_SECRET_MUST_NEVER_REACH_FAILURE_RECORD'
ANSWER = 'SYNTHETIC ACCEPTED ANSWER'
COUNTER_IDENTITY = {'kind': 'synthetic_finalization_counter'}
INTENT_CAPS = {'max_input_tokens': 2048, 'max_output_tokens': 256}
QA_CAPS = {'generate': (8192, 1024), 'verify': (8192, 512), 'revise': (8192, 1024)}
REGISTERED_BUDGET = {
    'max_attempts': 5, 'currency': 'USD', 'max_cost': '0.02',
    'allowed_groups': ['T', 'GV'],
    'model_prices': {'deepseek-flash': {
        'input_per_million': '0.30', 'output_per_million': '1.20',
        'documented_served_identity': 'DeepSeek-V4.1-Flash-1017',
        'source_url': 'https://api-docs.deepseek.com/quick_start/pricing'}},
    'approved_counter_sha256': None}


class SyntheticCase:
    """Isolated allowed I01 namespace, retaining actual no-refund ledgers."""
    def __init__(self, test, *, ready=True, fail_writes=(), close_error=False):
        self.test, self.ready = test, ready
        self.fail_writes, self.close_error = set(fail_writes), close_error
        self.root = Path(tempfile.mkdtemp(prefix='finalization_', dir=FIX / 'tmp'))
        self.run = self.root / 'runs/i01/synthetic_case'
        (self.run / 'inputs').mkdir(parents=True)
        (self.root / 'configs').mkdir()
        (self.root / 'knowledge/k01').mkdir(parents=True)
        self.write_original = flow.write_record
        self.write_original(self.root / 'configs/q01_deepseek.json', {'kind': 'synthetic_profile'})
        self.write_original(self.root / 'configs/q01_provider_prompts.json', {'kind': 'synthetic_prompts'})
        self.write_original(self.root / 'configs/q01_deepseek_budget.json', REGISTERED_BUDGET)
        (self.root / 'knowledge/k01/scenarios.csv').write_text(
            'scenario_id,assumption_text\nSC-K01-OR-001,SYNTHETIC scenario assumption\n', encoding='utf-8')
        self.write_original(self.run / 'live_release.json', {'scope': 'SYNTHETIC finalization only'})
        self.approved = {'question': 'SYNTHETIC fixed question?', 'explicitly_allowed_operating_context': {}}
        self.write_original(self.run / 'inputs/approved_content.json', self.approved)
        self.request = dict(self.approved, predicted_class='outer_race',
                            scenario_id='SC-K01-OR-001', task_types=['inspection'])
        self.prediction = {'kind': 'synthetic_classifier_boundary', 'sample_id': 'SYNTHETIC',
                           'predicted_class': 'outer_race'}
        self.qa_config = copy.deepcopy(REGISTERED_BUDGET)
        self.qa_config.update(max_attempts=4, max_cost='0.018',
                              approved_counter_sha256=flow.digest(COUNTER_IDENTITY))
        self.record = {
            'status': 'completed_verified_model_report', 'service_message': None,
            'publishable': True, 'final_answer': ANSWER,
            'retrieval': {'information_separation': {'demo_scenario': {
                'assumption_text': 'SYNTHETIC scenario assumption'}}},
            'returned_cards': [{'evidence_id': 'SYN-EV-001',
                'source': {'title': 'SYNTHETIC source', 'organization_author': 'SYNTHETIC author'},
                'original_text': 'SYNTHETIC original evidence remains inspectable.',
                'source_url': 'https://example.invalid/synthetic-evidence',
                'source_version': 'synthetic-v1', 'page_or_section': 'synthetic section',
                'conditions': 'SYNTHETIC conditions', 'limitations': 'SYNTHETIC limitations'}]}
        self.events, self.failed_writes = [], []
        self.budget = self.slot = self.qa = self.transport = self.router = None
        self.close_calls = 0
        self.original_close = None
        self.counter = SimpleNamespace(identity=COUNTER_IDENTITY)
        self.classifier = Mock(return_value=copy.deepcopy(self.prediction))
        self.no_socket = Mock(side_effect=AssertionError('Socket use forbidden in synthetic tests'))
        self.stack = ExitStack()

    def __enter__(self):
        case = self
        real_budget = wire.ChainQABudget

        class Gate:
            def __init__(self, *args):
                self.projection = {k: v for k, v in case.request.items() if k != 'task_types'}
                self.intent_config, self.prompts = INTENT_CAPS, {}
                self.release = {'qa_budget': case.qa_config}

            def freeze(self):
                case.events.append('gate_freeze')

            def bind_derived(self, *args):
                case.events.append('derived_bound')

        class Transport:
            def __init__(self, profile, config, prompts, slot, gate, **kwargs):
                self.slot = case.slot = slot
                self.httpx_post_invocations = 0
                case.transport = self

            def request(self, *args):
                raise AssertionError('Actual provider request forbidden')

            def snapshot(self):
                return {'kind': 'synthetic_no_http',
                        'actual_official_http_requests': self.httpx_post_invocations}

        class Router:
            def __init__(self, *args, **kwargs):
                self.attempts = []
                case.router = self

            def prepare_request(self, projection):
                # These are synthetic counters; actual transport.request is never invoked.
                case.slot.reserve(INTENT_CAPS)
                case.transport.httpx_post_invocations = 1
                self.attempts.append({'token_usage': {'prompt_tokens': 100, 'completion_tokens': 10}})
                if not case.ready:
                    return {'status': 'unsupported', 'request': None,
                            'service_message': 'SYNTHETIC unsupported task'}
                return {'status': 'ready', 'request': copy.deepcopy(case.request),
                        'service_message': None}

            def snapshot(self):
                return {'kind': 'synthetic_router', 'attempts': copy.deepcopy(self.attempts)}

        class QA:
            def __init__(self, profile, budget, request_sha, **kwargs):
                self.budget, self.attempts = budget, []
                self.identity = {'kind': 'synthetic_provider_no_http'}
                budget.begin_flow('GV', request_sha)
                case.qa = self

            def call(self, action, payload):
                input_cap, output_cap = QA_CAPS[action]
                reservation = self.budget.reserve('GV', action, input_cap, output_cap)
                self.attempts.append({'action': action, 'token_usage': {
                    'prompt_tokens': 1000, 'completion_tokens': 20}})
                self.budget.record_result(reservation, {'kind': 'synthetic_result_no_http'})
                return {'kind': 'synthetic_' + action}

            def generate(self, payload): return self.call('generate', payload)
            def verify(self, payload): return self.call('verify', payload)
            def revise(self, payload): return self.call('revise', payload)

            def snapshot(self):
                return {'kind': 'synthetic_provider_no_http',
                        'actual_attempt_count': len(self.attempts),
                        'attempts': copy.deepcopy(self.attempts)}

        def budget_factory(*args):
            case.budget = real_budget(*args)
            case.original_close = case.budget.close

            def close():
                case.close_calls += 1
                case.events.append('budget_close')
                case.original_close()
                if case.close_error:
                    raise OSError(SECRET)

            self.stack.enter_context(patch.object(case.budget, 'close', side_effect=close))
            return case.budget

        def synthetic_flow(request, config, saved_provider, **kwargs):
            for action in ('generate', 'verify', 'revise', 'verify'):
                getattr(saved_provider, action)({'kind': 'synthetic_payload'})
            return copy.deepcopy(case.record)

        def write(path, value):
            name = Path(path).name
            case.events.append(name)
            if name in case.fail_writes and name not in case.failed_writes:
                case.failed_writes.append(name)
                raise OSError(SECRET)
            return case.write_original(path, value)

        for target, name, value in (
                (page, 'ROOT', self.root), (wire, 'ROOT', self.root),
                (page, 'counter_for', Mock(return_value=self.counter)),
                (wire, 'check_release', Mock(return_value={})),
                (page, 'classify', self.classifier), (wire, 'LiveGate', Gate),
                (wire, 'IntentHTTP', Transport), (page.intent, 'ModelTaskRouter', Router),
                (wire, 'ChainQABudget', budget_factory), (wire, 'DerivedQAProvider', QA),
                (flow, 'run_flow', synthetic_flow), (flow, 'write_record', write),
                (socket, 'socket', self.no_socket), (socket, 'create_connection', self.no_socket)):
            self.stack.enter_context(patch.object(target, name, value))
        return self

    def __exit__(self, *args):
        try:
            self.no_socket.assert_not_called()
        finally:
            self.stack.close()
            if self.original_close is not None:
                self.original_close()

    def read(self, name):
        return wire.read(self.run / name)


class FinalizationContracts(unittest.TestCase):
    def assert_recorded_failure(self, case, result, stages):
        self.assertEqual(result['status'], 'input_or_setup_failed')
        self.assertEqual(result['failure'], {'error_type': 'OSError',
                                            'error_code': 'final_recording_failed'})
        self.assertEqual(result['finalization_errors'], [
            {'stage': stage, 'error_type': 'OSError'} for stage in stages])
        self.assertEqual(case.read('page_record.json'), result)
        self.assertTrue((case.run / 'live_batch_reserved.json').is_file())
        self.assertNotIn(SECRET, json.dumps(result, ensure_ascii=False))
        self.assertNotIn(SECRET, (case.run / 'page_record.json').read_text())

    def test_one_recoverable_identify_snapshot_error_saves_readable_failure(self):
        with SyntheticCase(self, ready=False, fail_writes=['final_identify_http.json']) as case:
            result = page.execute_once(case.run)
            self.assert_recorded_failure(case, result, ['final_identify_http.json'])
            self.assertEqual(case.failed_writes, ['final_identify_http.json'])
            self.assertEqual(result['routing']['status'], 'unsupported')
            self.assertEqual(result['summary']['identify_posts'], 1)
            self.assertEqual(result['summary']['qa_posts'], 0)
            self.assertEqual(result['summary']['total_posts'], 1)
            self.assertEqual(Decimal(result['summary']['reserved_cost_usd']), Decimal('.0009216'))
            case.write_original(case.run / 'storage_still_writable.json', {'ok': True})
            self.assertEqual(case.read('storage_still_writable.json'), {'ok': True})

    def test_known_counts_and_fullcap_reservations_survive_without_refund(self):
        with SyntheticCase(self, fail_writes=['final_identify_http.json']) as case:
            result = page.execute_once(case.run)
            self.assert_recorded_failure(case, result, ['final_identify_http.json'])
            self.assertEqual({k: result['summary'][k] for k in (
                'identify_posts', 'qa_posts', 'total_posts')},
                {'identify_posts': 1, 'qa_posts': 4, 'total_posts': 5})
            self.assertEqual(Decimal(result['summary']['reserved_cost_usd']), Decimal('.0144384'))
            self.assertEqual(Decimal(result['summary']['estimated_cost_usd']), Decimal('.001338'))
            self.assertEqual(result['summary']['actual_bill'], 'unknown')
            self.assertEqual(Decimal(wire.read(case.run / 'intent_slot/reservation.json')['reserved_usd']),
                             Decimal('.0009216'))
            rows = [json.loads(line) for line in (case.run / 'qa_session/session_ledger.jsonl').read_text().splitlines()]
            reservations = [row for row in rows if row['event'] == 'reserve']
            outcomes = [row for row in rows if row['event'] == 'result']
            self.assertEqual(len(reservations), 4)
            self.assertEqual(Decimal(reservations[-1]['cumulative_reserved_cost']), Decimal('.0135168'))
            self.assertEqual(len(outcomes), 4)
            self.assertTrue(all(row['reservation_refunded'] is False for row in outcomes))
            with self.assertRaises(wire.Error):
                case.slot.reserve(INTENT_CAPS)
            self.assertEqual(case.budget.snapshot()['attempt_count'], 4)
            display = page.render(result)
            self.assertEqual(display[4], '本次没有可发布答案。')
            self.assertNotIn(ANSWER, '\n'.join(display))
            self.assertIn('SYNTHETIC original evidence remains inspectable.', display[5])
            self.assertEqual(result['flow_record']['final_answer'], ANSWER)

    def test_budget_snapshot_write_error_still_closes_and_writes_other_snapshots(self):
        with SyntheticCase(self, fail_writes=['qa_budget_after.json']) as case:
            result = page.execute_once(case.run)
            self.assert_recorded_failure(case, result, ['qa_budget_after.json'])
            self.assertEqual(case.close_calls, 1)
            self.assertTrue(case.budget.snapshot()['closed'])
            self.assertEqual(case.read('final_identify_http.json')['actual_official_http_requests'], 1)
            self.assertEqual(case.read('final_qa_provider.json')['actual_attempt_count'], 4)
            self.assertEqual(Decimal(result['summary']['reserved_cost_usd']), Decimal('.0144384'))

    def test_identify_snapshot_error_does_not_skip_final_qa_snapshot(self):
        with SyntheticCase(self, fail_writes=['final_identify_http.json']) as case:
            result = page.execute_once(case.run)
            self.assert_recorded_failure(case, result, ['final_identify_http.json'])
            self.assertEqual(case.read('final_qa_provider.json')['actual_attempt_count'], 4)
            self.assertEqual(case.read('qa_budget_after.json')['attempt_count'], 4)
            self.assertEqual(case.events.count('final_identify_http.json'), 1)
            self.assertEqual(case.events.count('final_qa_provider.json'), 1)
            self.assertEqual(case.close_calls, 1)

    def test_multiple_final_errors_accumulate_and_cleanup_failure_is_redacted(self):
        with SyntheticCase(self, fail_writes=['qa_budget_after.json', 'final_identify_http.json'],
                           close_error=True) as case:
            result = page.execute_once(case.run)
            self.assert_recorded_failure(case, result, [
                'qa_budget_after.json', 'budget_close', 'final_identify_http.json'])
            self.assertTrue(case.budget.snapshot()['closed'])
            self.assertEqual(case.close_calls, 1)
            self.assertEqual(case.read('final_qa_provider.json')['actual_attempt_count'], 4)
            self.assertEqual(result['summary']['total_posts'], 5)

    def test_normal_finalization_preserves_result_render_summary_and_snapshot_timing(self):
        with SyntheticCase(self) as case:
            result = page.execute_once(case.run)
            self.assertEqual(result['status'], 'completed_verified_model_report')
            self.assertNotIn('failure', result)
            self.assertFalse(result.get('finalization_errors'))
            self.assertEqual(case.read('page_record.json'), result)
            self.assertEqual(result['classifier'], case.prediction)
            self.assertEqual(result['routing']['request'], case.request)
            self.assertEqual(result['flow_record']['final_answer'], ANSWER)
            self.assertEqual(result['summary']['total_posts'], 5)
            self.assertEqual(Decimal(result['summary']['reserved_cost_usd']), Decimal('.0144384'))
            self.assertEqual(Decimal(result['summary']['estimated_cost_usd']), Decimal('.001338'))
            self.assertEqual(result['summary']['actual_bill'], 'unknown')
            self.assertFalse(case.read('qa_budget_after.json')['closed'])
            self.assertTrue(case.budget.snapshot()['closed'])
            self.assertLess(case.events.index('qa_budget_after.json'), case.events.index('budget_close'))
            self.assertEqual(case.read('final_qa_provider.json')['actual_attempt_count'], 4)
            self.assertEqual(page.render(result)[4], ANSWER)
            self.assertIn('SYNTHETIC original evidence remains inspectable.', page.render(result)[5])
            self.assertEqual(case.classifier.call_count, 1)

    def test_saved_failure_repeats_and_new_module_replays_without_dispatch(self):
        with SyntheticCase(self, fail_writes=['final_identify_http.json']) as case:
            first = page.execute_once(case.run)
            self.assert_recorded_failure(case, first, ['final_identify_http.json'])
            ledger_before = (case.run / 'qa_session/session_ledger.jsonl').read_bytes()
            claim_before = (case.run / 'live_batch_reserved.json').read_bytes()
            with patch.object(page, 'run_case', side_effect=AssertionError('Repeat dispatch forbidden')):
                self.assertEqual(page.execute_once(case.run), first)
            restarted = load_page('i01_finalization_restarted')
            with patch.object(restarted, 'ROOT', case.root), \
                 patch.object(restarted, 'run_case', side_effect=AssertionError('Restart dispatch forbidden')):
                self.assertEqual(restarted.execute_once(case.run), first)
            self.assertEqual((case.run / 'qa_session/session_ledger.jsonl').read_bytes(), ledger_before)
            self.assertEqual((case.run / 'live_batch_reserved.json').read_bytes(), claim_before)
            self.assertEqual(case.classifier.call_count, 1)
            self.assertEqual(case.transport.httpx_post_invocations, 1)
            self.assertEqual(len(case.qa.attempts), 4)

    def test_outer_unexpected_exception_saves_safe_failure_with_unknown_counts(self):
        class UnexpectedFailure(RuntimeError):
            code = SECRET

        with SyntheticCase(self) as case:
            with patch.object(page, 'run_case', side_effect=UnexpectedFailure(SECRET)) as dispatch:
                result = page.execute_once(case.run)
                self.assertEqual(page.execute_once(case.run), result)
            self.assertEqual(dispatch.call_count, 1)
            self.assertEqual(result['status'], 'input_or_setup_failed')
            self.assertEqual(result['failure'], {'error_type': 'UnexpectedFailure', 'error_code': 'case_failed'})
            self.assertEqual(result['summary'], dict.fromkeys((
                'identify_posts', 'qa_posts', 'total_posts', 'reserved_cost_usd',
                'estimated_cost_usd', 'actual_bill'), 'unknown'))
            self.assertEqual(case.read('page_record.json'), result)
            self.assertNotIn(SECRET, json.dumps(result, ensure_ascii=False))
            self.assertTrue((case.run / 'live_batch_reserved.json').is_file())
            self.assertEqual(page.render(result)[4], '本次没有可发布答案。')
            case.classifier.assert_not_called()


if __name__ == '__main__':
    unittest.main()
