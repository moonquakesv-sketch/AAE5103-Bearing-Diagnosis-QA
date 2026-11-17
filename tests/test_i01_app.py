"""Targeted synthetic page display, deduplication and finalization acceptance.

No provider or classifier is run. Static releases are synthetic; the global
credit ledger and production submit/render/finalizer functions are exercised.
"""
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import i01_app as app
import i01_session as wire
import q01_flow as flow

ACTIVE = Path(os.environ.get('I01_ACTIVE_RUN', ROOT / 'runs/i01/20261026_031418_385e0a04'))
ANSWER = 'SYNTHETIC verified constrained answer.'
EVIDENCE = 'SYNTHETIC full original evidence; keep this qualification.'
SECRET = 'SYNTHETIC_AUTH_VALUE_NEVER_LOG'


def record(status='completed_verified_model_report', *, publishable=True):
    return {'status': status, 'service_message': None, 'publishable': publishable,
            'final_answer': ANSWER, 'identity': {},
            'returned_cards': [{'evidence_id': 'SYN-EV-001', 'original_text': EVIDENCE,
                'source': {'title': 'SYNTHETIC source', 'organization_author': 'SYNTHETIC author'},
                'source_url': 'https://example.invalid/synthetic', 'source_version': 'v1',
                'page_or_section': 'section one', 'conditions': 'SYNTHETIC condition',
                'limitations': 'SYNTHETIC limitation'}]}


def result(predicted='inner_race', status='completed_verified_model_report'):
    return {'classifier': {'predicted_class': predicted},
            'scenario': {'assumption_text': 'SYNTHETIC independent demo assumption',
                         'applicability': 'SYNTHETIC generic education only'},
            'routing': {'request': {'task_types': ['inspection']}},
            'flow_record': record(status), 'status': status, 'service_message': None,
            'summary': {'identify_posts': 1, 'qa_posts': 2, 'total_posts': 3,
                        'reserved_cost_usd': '.0076800', 'estimated_cost_usd': '.001', 'actual_bill': 'unknown'}}


class Display(unittest.TestCase):
    def test_four_class_display_uses_actual_prediction_and_independent_assumption(self):
        for predicted in ('normal', 'inner_race', 'outer_race', 'rolling_element'):
            with self.subTest(predicted=predicted):
                shown = app.render(result(predicted))
                self.assertEqual(shown[0], predicted)
                self.assertIn('independent demo assumption', shown[1])
                self.assertEqual(shown[2], 'inspection')
                self.assertEqual(shown[4], ANSWER)
                self.assertIn(EVIDENCE, shown[5])
                self.assertIn('https://example.invalid/synthetic', shown[5])
                self.assertNotIn('单情景接线', shown[3])
                if predicted == 'normal':
                    self.assertIn('不证明设备无损伤', shown[1])
                    self.assertIn('此分支只给一般监测知识', shown[1])

    def test_partial_keeps_evidence_insufficient_label(self):
        shown = app.render(result(status='evidence_insufficient'))
        self.assertIn('证据不足', shown[3])
        self.assertEqual(shown[4], ANSWER)

    def test_failed_verification_and_outer_failure_hide_draft(self):
        failed = result(status='verification_failed')
        failed['flow_record']['publishable'] = False
        self.assertNotIn(ANSWER, app.render(failed)[4])
        outer = result()
        outer['failure'] = {'error_code': 'final_recording_failed'}
        self.assertNotIn(ANSWER, app.render(outer)[4])
        self.assertIn(EVIDENCE, app.render(outer)[5])

    def test_recognition_block_and_unknown_knowledge_are_explicit(self):
        for status, label in [('needs_clarification', '需要澄清'), ('unsupported', '不支持'),
                              ('recognition_failed', '任务识别失败'), ('knowledge_not_covered', '当前知识未覆盖')]:
            shown = app.render({'status': status, 'flow_record': None})
            self.assertIn(label, shown[3])
            self.assertEqual(shown[4], '本次没有可发布答案。')


class Submission(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='app_submit_', dir=ACTIVE / 'tmp'))
        self.batch = self.root / 'runs/i01/synthetic'
        self.batch.mkdir(parents=True)
        wire.put(self.batch / 'batch_release.json', {'kind': 'SYNTHETIC app release'})
        self.checked = {'registry': {'samples': {'sample_1': {
            'selected_row': {'sample_id': 'SYNTHETIC', 'split': 'validation'}, 'file_hashes': {}}}},
            'prompts': {'instructions': 'SYNTHETIC task identification'},
            'intent_config': {'max_input_tokens': 2048}}
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(app, 'ROOT', self.root))
        self.stack.enter_context(patch.object(wire, 'ROOT', self.root))
        self.stack.enter_context(patch.object(wire, '_batch_release', return_value=self.checked))
        self.stack.enter_context(patch.object(socket, 'socket', side_effect=AssertionError('No network in synthetic page tests')))
        self.run_case = self.stack.enter_context(patch.object(app, 'run_case', return_value=result()))
        self.budgets = []

    def tearDown(self):
        for budget in self.budgets:
            budget.close()
        self.stack.close()

    def workflow(self):
        workflow = app.Workflow.__new__(app.Workflow)
        workflow.batch, workflow.checked = self.batch, self.checked
        workflow.counter = SimpleNamespace(identity={'kind': 'SYNTHETIC'},
                                           count_messages=lambda *_: {'input_tokens': 100})
        workflow.retriever = SimpleNamespace()
        workflow.budget = wire.BatchBudget(self.batch)
        workflow.lock = threading.Lock()
        self.budgets.append(workflow.budget)
        return workflow

    def test_identical_submission_replays_without_new_credit_including_restart(self):
        workflow = self.workflow()
        first = workflow.submit('sample_1', 'What should be inspected?')
        self.assertEqual(self.run_case.call_count, 1)
        self.assertEqual(workflow.budget.snapshot()['claimed_flows'], 1)
        self.assertEqual(workflow.submit('sample_1', 'What should be inspected?'), first)
        self.assertEqual(self.run_case.call_count, 1)
        workflow.budget.close()
        restarted = self.workflow()
        self.assertEqual(restarted.submit('sample_1', 'What should be inspected?'), first)
        self.assertEqual(self.run_case.call_count, 1)
        self.assertEqual(restarted.budget.snapshot()['claimed_flows'], 1)
        self.assertEqual(wire.read(self.batch / 'cases/001/display_record.json'), list(app.render(first)))

    def test_auxiliary_display_write_failure_keeps_committed_result_and_replay_identical(self):
        workflow = self.workflow()
        original_put = wire.put

        def failed_display(path, value):
            if Path(path).name == 'display_record.json':
                raise OSError(SECRET)
            return original_put(path, value)

        with patch.object(wire, 'put', side_effect=failed_display):
            first = workflow.submit('sample_1', 'What should be inspected?')
        saved = wire.read(self.batch / 'cases/001/page_record.json')
        self.assertEqual(first, saved)
        self.assertEqual(workflow.submit('sample_1', 'What should be inspected?'), saved)
        self.assertEqual(first['status'], 'completed_verified_model_report')
        diagnostic = wire.read(self.batch / 'cases/001/display_record_failure.json')
        self.assertEqual(diagnostic, {'error_type': 'OSError', 'canonical_result_unchanged': True})
        self.assertNotIn(SECRET, flow.canonical(diagnostic))
        self.assertEqual(self.run_case.call_count, 1)
        self.assertEqual(workflow.budget.snapshot()['claimed_flows'], 1)

    def test_incomplete_identical_case_and_busy_click_never_send(self):
        workflow = self.workflow()
        case = workflow.budget.claim_flow()
        wire.put(case / 'inputs/submission.json', {'sample_choice': 'sample_1',
                 'question': 'What should be inspected?', 'explicitly_allowed_operating_context': {}})
        self.assertEqual(workflow.submit('sample_1', 'What should be inspected?')['status'], 'batch_consumed')
        workflow.lock.acquire()
        try:
            self.assertEqual(workflow.submit('sample_1', 'A different question?')['status'], 'batch_consumed')
        finally:
            workflow.lock.release()
        self.assertEqual(self.run_case.call_count, 0)
        self.assertEqual(workflow.budget.snapshot()['claimed_flows'], 1)

    def test_invalid_input_costs_no_credit(self):
        workflow = self.workflow()
        for choice, question in [('unregistered', 'What should be inspected?'), ('sample_1', ''),
                                  ('sample_1', 'x' * 2001)]:
            with self.subTest(choice=choice, length=len(question)):
                stopped = workflow.submit(choice, question)
                self.assertEqual(stopped['status'], 'input_or_setup_failed')
        self.assertEqual(workflow.budget.snapshot()['claimed_flows'], 0)
        self.run_case.assert_not_called()

    def test_unexpected_case_exception_is_saved_and_consumed_with_unknown_counts(self):
        workflow = self.workflow()
        self.run_case.side_effect = RuntimeError(SECRET)
        stopped = workflow.submit('sample_1', 'What should be inspected?')
        self.assertEqual(stopped['failure']['error_code'], 'case_failed')
        self.assertNotIn(SECRET, flow.canonical(stopped))
        self.assertEqual(stopped['summary']['total_posts'], 'unknown')
        self.assertEqual(wire.read(self.batch / 'cases/001/page_record.json'), stopped)
        self.assertEqual(workflow.budget.snapshot()['actual_attempts'], 'unknown')
        workflow.submit('sample_1', 'What should be inspected?')
        self.assertEqual(self.run_case.call_count, 1)


class Finalization(unittest.TestCase):
    def test_one_auxiliary_write_failure_still_closes_and_preserves_known_consumption(self):
        root = Path(tempfile.mkdtemp(prefix='app_finalize_', dir=ACTIVE / 'tmp'))
        case = root / 'case'
        (case / 'inputs').mkdir(parents=True)
        counter = SimpleNamespace(identity={'kind': 'SYNTHETIC'})
        submission = {'question': 'What should be inspected?', 'sample_choice': 'sample_1',
                      'explicitly_allowed_operating_context': {}}
        four = {'question': submission['question'], 'predicted_class': 'inner_race',
                'scenario_id': 'SC-I01-IR-001', 'explicitly_allowed_operating_context': {}}
        request = dict(four, task_types=['inspection'])
        route = {'status': 'ready', 'request': request, 'service_message': None}
        events = []
        class Gate:
            def __init__(self, *args, **kwargs):
                self.projection, self.intent_config, self.prompts = four, {}, {}
                self.release = {'qa_budget': {}}
            def freeze(self): events.append('freeze')
            def bind_derived(self, *args): events.append('derived')
        transport = SimpleNamespace(httpx_post_invocations=1, request=lambda *_: None,
                                    snapshot=lambda: {'kind': 'SYNTHETIC HTTP snapshot'})
        router = SimpleNamespace(attempts=[{'token_usage': {'prompt_tokens': 100, 'completion_tokens': 10}}],
                                 prepare_request=lambda *_: route, snapshot=lambda: {'kind': 'SYNTHETIC router'})
        budget = SimpleNamespace(snapshot=lambda: {'reserved_cost': '.0067584'},
                                 close=lambda: events.append('close'))
        qa = SimpleNamespace(attempts=[{'token_usage': {'prompt_tokens': 100, 'completion_tokens': 10}}] * 2,
                             snapshot=lambda: {'actual_attempt_count': 2}, identity={'kind': 'SYNTHETIC'})
        scenario = {'assumption_text': 'SYNTHETIC independent assumption', 'applicability': 'education only'}
        retriever = SimpleNamespace(scenarios={'SC-I01-IR-001': scenario}, knowledge_hashes={}, identity={})
        original_write = flow.write_record
        def write(path, value):
            if path.name == 'final_identify_http.json':
                raise OSError(SECRET)
            return original_write(path, value)
        with ExitStack() as stack:
            stack.enter_context(patch.object(socket, 'socket', side_effect=AssertionError('No network in finalization')))
            stack.enter_context(patch.object(wire, 'check_release', return_value={'profile': {}}))
            stack.enter_context(patch.object(app.single, 'classify', return_value={
                'kind': 'fresh_image_inference', 'predicted_class': 'inner_race', 'sample_id': 'SYNTHETIC'}))
            stack.enter_context(patch.object(wire, 'LiveGate', Gate))
            stack.enter_context(patch.object(wire, 'IntentSlot', return_value=SimpleNamespace(reserved='.0009216')))
            stack.enter_context(patch.object(wire, 'IntentHTTP', return_value=transport))
            stack.enter_context(patch.object(app.intent, 'ModelTaskRouter', return_value=router))
            stack.enter_context(patch.object(wire, 'ChainQABudget', return_value=budget))
            stack.enter_context(patch.object(wire, 'DerivedQAProvider', return_value=qa))
            stack.enter_context(patch.object(flow, 'run_flow', return_value=record()))
            stack.enter_context(patch.object(flow, 'write_record', side_effect=write))
            stopped = app.run_case(case, submission, counter, retriever, root)
        self.assertEqual(events, ['freeze', 'derived', 'close'])
        self.assertEqual(stopped['status'], 'input_or_setup_failed')
        self.assertEqual(stopped['failure']['error_code'], 'final_recording_failed')
        self.assertEqual(stopped['summary']['total_posts'], 3)
        self.assertEqual(stopped['summary']['reserved_cost_usd'], '0.0076800')
        self.assertNotIn(SECRET, flow.canonical(stopped))
        self.assertNotIn(ANSWER, app.render(stopped)[4])
        self.assertTrue((case / 'final_qa_provider.json').is_file())
        self.assertTrue((case / 'qa_budget_after.json').is_file())


if __name__ == '__main__':
    unittest.main()
