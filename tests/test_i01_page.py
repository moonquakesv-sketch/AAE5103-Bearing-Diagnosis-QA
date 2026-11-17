"""Small I01 adapter/page regressions. All responses are synthetic; no live calls."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import i01_page as page
import i01_http as wire
import q01_flow as flow
import q01_intent as intent
import q01_intent_http as prior
import test_q01_flow as fixtures
import test_q01_intent as intent_fixtures
import httpx


class PageContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = Path(os.environ['I01_RUN']).absolute()
        if not cls.stage.is_relative_to(ROOT / 'runs/i01'):
            raise ValueError('New I01 run required.')
        cls.observations = []
        cls.observation_path = Path(tempfile.mkdtemp(prefix='i01_observations_', dir=cls.stage / 'tmp')) / 'observations.json'

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix='i01_contract_', dir=self.stage / 'tmp'))
        (self.folder / 'inputs').mkdir()
        flow.write_record(self.folder / 'live_release.json', {'synthetic': True})

    @classmethod
    def tearDownClass(cls):
        flow.write_record(cls.observation_path,
                          {'kind': 'synthetic_offline_contracts', 'actual_http_requests': 0,
                           'observations': cls.observations})

    def save(self, name, value):
        self.observations.append({'case': name, 'value': flow.json_snapshot(value)})

    def test_claim_survives_new_module_and_incomplete_batch(self):
        result = {'status': 'SYNTHETIC_DONE', 'classifier': {'predicted_class': 'normal'}}
        with patch.object(page, 'run_case', return_value=result) as dispatch:
            self.assertEqual(page.execute_once(self.folder), result)
            self.assertEqual(page.execute_once(self.folder), result)
            spec = importlib.util.spec_from_file_location('i01_restart_fixture', ROOT / 'scripts/i01_page.py')
            restarted = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(restarted)
            with patch.object(restarted, 'run_case', side_effect=AssertionError('Repeat dispatch')):
                self.assertEqual(restarted.execute_once(self.folder), result)
            self.assertEqual(dispatch.call_count, 1)
        incomplete = self.folder / 'incomplete'
        incomplete.mkdir()
        flow.write_record(incomplete / 'live_release.json', {'synthetic': True})
        wire.put(incomplete / 'live_batch_reserved.json', {'synthetic_interrupted': True})
        with patch.object(page, 'run_case', side_effect=AssertionError('Interrupted batch redispatched')):
            self.assertEqual(page.execute_once(incomplete)['status'], 'batch_consumed')
        self.save('durable_claim_once', {'dispatch_count': 1, 'restart_replayed': True,
                                        'interrupted_batch_blocked': True})

    def test_failure_is_saved_readable_and_secret_text_is_absent(self):
        synthetic_secret = 'SYNTHETIC_AUTH_VALUE_NOT_A_REAL_KEY'
        with patch.object(page, 'counter_for', return_value=Mock()), \
             patch.object(wire, 'check_release', return_value=None), \
             patch.object(page, 'classify', side_effect=RuntimeError(synthetic_secret)):
            result = page.execute_once(self.folder)
        self.assertEqual(result['status'], 'input_or_setup_failed')
        self.assertEqual(wire.read(self.folder / 'page_record.json'), result)
        self.assertNotIn(synthetic_secret, json.dumps(result))
        with patch.object(page, 'run_case', side_effect=AssertionError('Failure retried')):
            self.assertEqual(page.execute_once(self.folder), result)
        self.save('failure_readback_without_auth', result)

    def test_uncovered_prediction_stops_before_recognition(self):
        prediction = {'predicted_class': 'inner_race', 'sample_id': 'SYN-IMAGE'}
        with patch.object(page, 'counter_for', return_value=Mock()), \
             patch.object(wire, 'check_release', return_value=None), \
             patch.object(page, 'classify', return_value=prediction), \
             patch.object(wire, 'LiveGate') as gate, \
             patch.object(intent, 'ModelTaskRouter') as router:
            result = page.run_case(self.folder)
        self.assertEqual(result['classifier'], prediction)
        self.assertEqual(result['status'], 'knowledge_not_covered')
        gate.assert_not_called()
        router.assert_not_called()
        self.assertIn('当前知识未覆盖', page.render(result)[3])
        self.assertEqual(result['summary']['total_posts'], 0)
        self.save('uncovered_stops', result)

    def test_recognition_block_retains_question_context_and_actual_class(self):
        approved = {'question': 'SYNTHETIC fixed registered question?',
                    'explicitly_allowed_operating_context': {}}
        flow.write_record(self.folder / 'inputs/approved_content.json', approved)
        prediction = {'predicted_class': 'outer_race', 'sample_id': 'SYN-IMAGE'}
        gate = Mock(intent_config=wire.read(ROOT / 'configs/q01_intent.json'),
                    prompts=wire.read(ROOT / 'configs/q01_intent_prompts.json'))
        gate.projection = dict(approved, predicted_class='outer_race', scenario_id=page.SCENARIO)
        slot = Mock(reserved='0')
        transport = Mock(httpx_post_invocations=0)
        transport.snapshot.return_value = {'kind': 'synthetic_transport_no_http'}
        router = Mock(attempts=[])
        router.prepare_request.return_value = {'status': 'unsupported', 'request': None,
                                              'service_message': 'SYNTHETIC task unsupported.'}
        router.snapshot.return_value = {'kind': 'synthetic_task_response'}
        counter = Mock(identity={'kind': 'synthetic_counter'})
        with patch.object(page, 'counter_for', return_value=counter), \
             patch.object(wire, 'check_release', return_value=None), \
             patch.object(page, 'classify', return_value=prediction), \
             patch.object(wire, 'LiveGate', return_value=gate) as gate_constructor, \
             patch.object(wire, 'IntentSlot', return_value=slot), \
             patch.object(wire, 'IntentHTTP', return_value=transport), \
             patch.object(intent, 'ModelTaskRouter', return_value=router), \
             patch.object(wire, 'ChainQABudget') as qa_budget, \
             patch.object(wire, 'DerivedQAProvider') as qa:
            result = page.run_case(self.folder)
        self.assertEqual(gate_constructor.call_args.args[2], prediction)
        self.assertEqual(wire.read(self.folder / 'inputs/four_fields.json'), gate.projection)
        router.prepare_request.assert_called_once_with(gate.projection)
        qa_budget.assert_not_called()
        qa.assert_not_called()
        self.assertEqual(result['status'], 'unsupported')
        self.assertIn('任务识别阻断', page.render(result)[3])
        self.save('recognition_block_projection', result)

    def test_model_task_source_is_response_and_question_only(self):
        profile = wire.read(ROOT / 'configs/q01_deepseek.json')
        config = wire.read(ROOT / 'configs/q01_intent.json')
        prompts = wire.read(ROOT / 'configs/q01_intent_prompts.json')
        calls = []
        report = {'status': 'identified', 'task_types': ['preparation'], 'message': None}
        def exchange(endpoint, body, settings):
            calls.append(copy.deepcopy(body))
            return {'status_code': 200, 'body': json.dumps({'id': 'SYN-ID', 'model': 'deepseek-flash',
                    'usage': {'prompt_tokens': 20, 'completion_tokens': 15, 'total_tokens': 35},
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': json.dumps(report)},
                                 'finish_reason': 'stop'}]})}
        four = {k: copy.deepcopy(v) for k, v in fixtures.REQUEST.items() if k != 'task_types'}
        router = intent.ModelTaskRouter(profile, config, prompts, exchange)
        routed = router.prepare_request(four)
        self.assertEqual(routed['request']['task_types'], ['preparation'])
        self.assertEqual(json.loads(calls[0]['messages'][1]['content']), {'question': four['question']})
        self.assertEqual({k: routed['request'][k] for k in four}, four)
        self.save('model_task_response_source', {'route': routed, 'synthetic_request': calls})

    def test_flow_failure_partial_and_evidence_rendering(self):
        cases = [('verify_exception', 'verification_failed'),
                 ('revision_still_issues', 'content_issues_remaining'),
                 ('partial_answer', 'evidence_insufficient')]
        for name, expected in cases:
            request, config, retriever, provider = fixtures.synthetic_case(name)
            if name == 'partial_answer':
                provider = flow.ScriptedProvider({'generate': [fixtures.answer(sufficiency='partial')],
                                                  'verify': [{'verdict': 'pass', 'issues': []}]})
            record = flow.run_flow(request, flow.group_config(config), provider, retriever=retriever)
            self.assertEqual(record['status'], expected)
            result = {'status': record['status'], 'classifier': {'predicted_class': 'outer_race'},
                      'routing': {'request': request}, 'flow_record': record}
            before = copy.deepcopy(result)
            display = page.render(result)
            self.assertEqual(result, before)
            if expected != 'evidence_insufficient':
                self.assertIn('未完成核验', display[3])
                self.assertNotIn(record['draft']['answer'], display[4])
                self.assertFalse(record['publishable'])
            else:
                self.assertIn('证据不足', display[3])
                self.assertEqual(display[4], record['final_answer'])
            card = record['returned_cards'][0]
            self.assertIn(card['original_text'], display[5])
            self.assertIn(card['source_url'], display[5])
            self.assertIn(card['source']['title'], display[5])
            self.assertIn(card['conditions'], display[5])
            self.assertIn(card['limitations'], display[5])
            self.assertEqual(record['provider_trace'][0]['payload']['request'], request)
            self.assertEqual(record['provider_trace'][0]['payload']['evidence_cards'], flow.evidence_payload(record['retrieval']))
            self.save('render_' + name, {'record': record, 'display': list(display)})

    def test_mock_http_auth_echo_and_exception_are_redacted_and_readable(self):
        profile = wire.read(ROOT / 'configs/q01_deepseek.json')
        config = wire.read(ROOT / 'configs/q01_intent.json')
        prompts = wire.read(ROOT / 'configs/q01_intent_prompts.json')
        four = {k: copy.deepcopy(v) for k, v in fixtures.REQUEST.items() if k != 'task_types'}
        secret = 'SYNTHETIC_I01_AUTH_VALUE'
        for name in ('echo', 'timeout'):
            folder = self.folder / name
            folder.mkdir()
            counter = intent_fixtures.Counter()
            slot = wire.IntentSlot(folder / 'slot', flow.digest(counter.identity))
            gate = intent_fixtures.FakeGate(folder, four)
            calls = []
            def respond(request):
                calls.append(True)
                if name == 'timeout':
                    raise httpx.ReadTimeout('SYNTHETIC timeout ' + secret)
                return httpx.Response(503, text='SYNTHETIC service echo ' + secret)
            transport = wire.IntentHTTP(profile, config, prompts, slot, gate, token_counter=counter,
                                        http_transport=httpx.MockTransport(respond), test_credential=secret)
            router = intent.ModelTaskRouter(profile, config, prompts, transport.request,
                                             transport_kind='http_mock', redact_values=(secret,))
            route = router.prepare_request(four)
            again = router.prepare_request(four)
            record = {'route': route, 'again': again, 'router': router.snapshot(), 'http': transport.snapshot()}
            flow.write_record(folder / 'failed_response.json', record)
            self.assertEqual(wire.read(folder / 'failed_response.json'), record)
            self.assertNotIn(secret, json.dumps(record))
            self.assertNotIn('Authorization', json.dumps(record))
            self.assertEqual(route['status'], 'recognition_failed')
            self.assertEqual(again['error_code'], 'budget_exhausted')
            self.assertEqual(len(calls), 1)
            self.assertEqual(transport.snapshot()['actual_official_http_requests'], 0)
            self.assertEqual(slot.reserved, '0.0009216')
            self.save('mock_http_redaction_' + name, record)

    def test_gate_binds_actual_classifier_and_original_four_fields(self):
        source = wire.read(self.stage / 'inputs/source_registration.json')
        sample = wire.read(self.stage / 'inputs/sample_registration.json')
        approved = wire.read(self.stage / 'inputs/approved_content.json')
        for name, value in [('source_registration', source), ('sample_registration', sample),
                            ('approved_content', approved)]:
            flow.write_record(self.folder / 'inputs' / (name + '.json'), value)
        counter = intent_fixtures.Counter()
        counter_sha = flow.digest(counter.identity)
        budget = wire.read(ROOT / 'configs/q01_deepseek_budget.json')
        budget.update(max_attempts=4, max_cost='0.018', approved_counter_sha256=counter_sha)
        release = {'release': True, 'allocation': wire.ALLOCATION, 'counter_identity_sha256': counter_sha,
                   'qa_budget': budget, 'file_hashes': {p: flow.file_hash(ROOT / p) for p in
                        wire.CODE_FILES | wire.CONFIG_FILES | set(sample['file_hashes']) |
                        {(self.folder / 'inputs/approved_content.json').relative_to(ROOT).as_posix()}},
                   'source_registration_file_sha256': flow.file_hash(self.folder / 'inputs/source_registration.json'),
                   'sample_registration_file_sha256': flow.file_hash(self.folder / 'inputs/sample_registration.json')}
        # Visibly synthetic binder fixture; no classifier or official HTTP runs here.
        classifier = {'kind': 'fresh_image_inference', 'sample_id': sample['selected_row']['sample_id'],
                      'predicted_class': 'outer_race', 'synthetic_fixture_only': True}
        four = dict(approved, predicted_class='outer_race', scenario_id=page.SCENARIO)
        (self.folder / 'live_release.json').write_text(flow.canonical(release) + '\n')
        flow.write_record(self.folder / 'classifier_record.json', classifier)
        flow.write_record(self.folder / 'inputs/four_fields.json', four)
        wire.check_release(self.folder, counter)
        with self.assertRaises(wire.Error) as counter_mismatch:
            wire.check_release(self.folder, Mock(identity={'kind': 'different_synthetic_counter'}))
        self.assertEqual(counter_mismatch.exception.code, 'release_invalid')
        gate = wire.LiveGate(self.folder, counter, classifier)
        self.assertEqual(gate.projection, four)
        changed_memory = dict(classifier, predicted_class='inner_race')
        with self.assertRaises(wire.Error) as mismatch:
            wire.LiveGate(self.folder, counter, changed_memory)
        self.assertEqual(mismatch.exception.code, 'classifier_binding_invalid')
        changed_four = dict(four, question='SYNTHETIC unapproved question')
        (self.folder / 'inputs/four_fields.json').write_text(flow.canonical(changed_four) + '\n')
        with self.assertRaises(wire.Error):
            gate.check()
        (self.folder / 'inputs/four_fields.json').write_text(flow.canonical(four) + '\n')
        (self.folder / 'classifier_record.json').write_text(flow.canonical(changed_memory) + '\n')
        with self.assertRaises(wire.Error):
            gate.check()
        self.save('classifier_projection_binding', {'synthetic_fixture_only': True,
                  'actual_memory_mismatch_rejected': True, 'changed_question_rejected': True,
                  'changed_classifier_file_rejected': True, 'projection': four})

    def test_i01_slot_and_qa_budget_caps_and_no_restart(self):
        counter_sha = '1' * 64
        slot = wire.IntentSlot(self.folder / 'slot', counter_sha)
        config = wire.read(ROOT / 'configs/q01_intent.json')
        slot.reserve(config)
        slot.result({'synthetic': 'failed_no_refund'})
        with self.assertRaises(wire.Error):
            slot.reserve(config)
        self.assertEqual(wire.read(self.folder / 'slot/reservation.json')['attempt'], 1)
        self.assertEqual(slot.reserved, '0.0009216')
        budget_config = wire.read(ROOT / 'configs/q01_deepseek_budget.json')
        budget_config.update(max_attempts=4, max_cost='0.018', approved_counter_sha256=counter_sha)
        budget = wire.ChainQABudget(self.folder / 'qa', budget_config, '2' * 64)
        try:
            budget.begin_flow('GV', '2' * 64)
            with self.assertRaises(wire.Error):
                budget.begin_flow('GV', '2' * 64)
            for action, cap in [('generate',1024), ('verify',512), ('revise',1024), ('verify',512)]:
                reservation = budget.reserve('GV', action, 8192, cap)
                budget.record_result(reservation, {'status': 'synthetic_failure_no_refund'})
            self.assertEqual(budget.snapshot()['attempt_count'], 4)
            with self.assertRaises(wire.Error):
                budget.reserve('GV', 'verify', 8192, 512)
            snap = budget.snapshot()
            self.assertEqual(snap['attempt_count'], 4)
            self.assertEqual(snap['reserved_cost'], '0.0135168')
        finally:
            budget.close()
        with self.assertRaises(ValueError):
            wire.ChainQABudget(self.folder / 'qa', budget_config, '2' * 64)
        money = wire.ChainQABudget(self.folder / 'money', budget_config, '2' * 64)
        try:
            money.begin_flow('GV', '2' * 64)
            with self.assertRaises(wire.Error) as denied:
                money.reserve('GV', 'generate', 1000000, 1024)
            self.assertEqual(denied.exception.code, 'cost_budget_exhausted')
            self.assertEqual(money.snapshot()['attempt_count'], 0)
        finally:
            money.close()
        self.assertIs(wire.IntentHTTP.request, prior.IntentHTTP.request)
        self.assertIs(wire.DerivedRequestHTTP._context, prior.DerivedRequestHTTP._context)
        self.assertIs(wire.LiveGate.bind_derived, prior.LiveGate.bind_derived)
        self.assertIs(wire.ChainQABudget.reserve, prior.ChainQABudget.reserve)
        self.save('inherited_budget_guards', {'qa': snap, 'slot_reserved': slot.reserved,
                                            'fifth_attempt_blocked': True, 'money_excess_blocked': True})


if __name__ == '__main__':
    unittest.main()
