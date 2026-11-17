"""Synthetic I01 prediction/task provenance regressions; no inference or API calls."""
import copy
from decimal import Decimal
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import i01_app as app
import i01_session as wire
import q01_flow as flow
from test_q01_flow import FakeRetriever, answer, retrieval

PROMPT = 'configs/i01_provider_prompts.json'
STATIC = 'routing_category_only_not_an_actual_prediction'


def fixture(predicted='outer_race', status='retrieved'):
    request = {'predicted_class': predicted, 'scenario_id': app.knowledge.SCENARIOS[predicted],
               'question': 'SYNTHETIC bearing inspection question.', 'task_types': ['inspection'],
               'explicitly_allowed_operating_context': {}}
    returned = retrieval(status)
    returned['request'] = copy.deepcopy(request)
    returned['information_separation']['demo_scenario'].update(
        scenario_id=request['scenario_id'], predicted_class=predicted, class_semantics=STATIC)
    classifier = {'kind': 'fresh_image_inference', 'predicted_class': predicted,
                  'sample_id': 'SYNTHETIC_PRIVATE_ID', 'true_label': 'SYNTHETIC_PRIVATE_LABEL',
                  'image_path': 'SYNTHETIC_PRIVATE_PATH', 'load': 'SYNTHETIC_PRIVATE_LOAD'}
    route = {'status': 'ready', 'request': copy.deepcopy(request),
             'model_report': {'status': 'identified', 'task_types': ['inspection'], 'message': None},
             'identity': {'kind': 'real_provider', 'transport': 'real_http', 'role': 'task_identification'}}
    return request, returned, classifier, route


class MockProvider:
    identity = {'kind': 'provider_contract_mock', 'label': 'SYNTHETIC provenance test'}

    def __init__(self, verification=None, sufficiency='sufficient'):
        self.calls = []
        self.verification = list(verification or [{'verdict': 'pass', 'issues': []}])
        self.sufficiency = sufficiency

    def call(self, action, payload):
        self.calls.append({'action': action, 'payload': copy.deepcopy(payload)})
        return self.verification.pop(0) if action == 'verify' else answer(sufficiency=self.sufficiency)

    def generate(self, payload): return self.call('generate', payload)
    def verify(self, payload): return self.call('verify', payload)
    def revise(self, payload): return self.call('revise', payload)


class Provenance(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket, 'socket', side_effect=AssertionError('No network in provenance tests'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def runtime(self):
        return {'record_kind': 'provider_contract_mock', 'prompts': wire.read(ROOT / PROMPT),
                'generation_settings': {'provider': 'SYNTHETIC mock'}}

    def test_all_four_predictions_preserve_cards_and_separate_static_metadata(self):
        for predicted in app.knowledge.SCENARIOS:
            with self.subTest(predicted=predicted):
                request, raw, classifier, route = fixture(predicted)
                before = copy.deepcopy(raw)
                wrapped = app.PredictionContext(FakeRetriever(raw), classifier, route)
                result = wrapped.retrieve(request, mode='graph', top_k=3)
                separation = result['information_separation']
                self.assertEqual(raw, before)
                self.assertEqual(result['results'], raw['results'])
                self.assertEqual(separation['static_class_semantics'], STATIC)
                self.assertEqual(separation['class_origin'], 'frozen_classifier_fresh_image_inference')
                self.assertEqual(separation['task_origin'], 'current_model_task_identification')
                self.assertIn('actual output', separation['demo_scenario']['class_semantics'])
                self.assertNotIn(STATIC, separation['demo_scenario']['class_semantics'])
                for key, value in before['information_separation']['demo_scenario'].items():
                    if key != 'class_semantics':
                        self.assertEqual(separation['demo_scenario'][key], value)

    def test_generate_verify_revise_share_context_and_do_not_leak_source_metadata(self):
        request, raw, classifier, route = fixture()
        provider = MockProvider([{'verdict': 'issues', 'issues': ['SYNTHETIC source clarification']},
                                 {'verdict': 'pass', 'issues': []}])
        record = flow.run_flow(request, flow.group_config('GV'), provider,
            retriever=app.PredictionContext(FakeRetriever(raw), classifier, route), runtime=self.runtime())
        self.assertTrue(record['publishable'])
        self.assertEqual([c['action'] for c in provider.calls], ['generate', 'verify', 'revise', 'verify'])
        for event in provider.calls:
            payload = event['payload']
            self.assertEqual(payload['request'], request)
            self.assertEqual(payload['evidence_cards'], flow.evidence_payload(raw))
            self.assertEqual(payload['scenario_context'], provider.calls[0]['payload']['scenario_context'])
            text = flow.canonical(payload)
            self.assertNotIn(STATIC, text)
            self.assertNotIn('Caller-supplied class and task types are stubs', text)
            for hidden in ('SYNTHETIC_PRIVATE_ID', 'SYNTHETIC_PRIVATE_LABEL', 'SYNTHETIC_PRIVATE_PATH',
                           'SYNTHETIC_PRIVATE_LOAD', 'static_class_semantics'):
                self.assertNotIn(hidden, text)
            self.assertIn('task_types', text)
            self.assertIn('actual output', payload['scenario_context']['class_semantics'])
        self.assertEqual(record['retrieval']['information_separation']['static_class_semantics'], STATIC)
        self.assertIn('not prove', self.runtime()['prompts']['common'])
        self.assertIn('not manual stubs', self.runtime()['prompts']['common'])
        self.assertIn('provenance', self.runtime()['prompts']['verify'])

    def test_unbound_or_mismatched_sources_fail_closed(self):
        for change in ('stub', 'class', 'route', 'tasks', 'identity'):
            request, raw, classifier, route = fixture()
            if change == 'stub': classifier['kind'] = 'caller_stub'
            if change == 'class': classifier['predicted_class'] = 'normal'
            if change == 'route': route['status'] = 'recognition_failed'
            if change == 'tasks': route['model_report']['task_types'] = ['explanation']
            if change == 'identity': route['identity']['transport'] = 'offline_fixture'
            with self.subTest(change=change), self.assertRaises(ValueError):
                app.PredictionContext(FakeRetriever(raw), classifier, route)

    def test_changed_request_or_scenario_is_rejected_before_generation(self):
        request, raw, classifier, route = fixture()
        wrapped = app.PredictionContext(FakeRetriever(raw), classifier, route)
        altered = dict(request, question='SYNTHETIC changed question')
        with self.assertRaises(ValueError): wrapped.retrieve(altered)
        raw['information_separation']['demo_scenario']['predicted_class'] = 'normal'
        with self.assertRaises(ValueError):
            app.PredictionContext(FakeRetriever(raw), classifier, route).retrieve(request)

    def test_partial_and_failed_verification_keep_original_publication_contract(self):
        for failure in (False, True):
            request, raw, classifier, route = fixture(status='partial_task_coverage')
            provider = MockProvider([{'verdict': 'issues', 'issues': ['SYNTHETIC unresolved issue']}] * 2
                                    if failure else None)
            record = flow.run_flow(request, flow.group_config('GV'), provider,
                retriever=app.PredictionContext(FakeRetriever(raw), classifier, route), runtime=self.runtime())
            self.assertEqual(record['status'], 'content_issues_remaining' if failure else 'evidence_insufficient')
            self.assertEqual(record['publishable'], not failure)
            if failure: self.assertIsNone(record['final_answer'])

    def test_new_prompt_is_a_required_release_identity(self):
        self.assertIn(PROMPT, wire.CONFIG_FILES)
        self.assertIn('configs/q01_provider_prompts.json', wire.CONFIG_FILES)

    def test_i01_bridge_uses_new_prompt_and_rejects_old_prompt_in_real_mode(self):
        provider = wire.DerivedQAProvider.__new__(wire.DerivedQAProvider)
        provider._http = SimpleNamespace(mode='real_http', events=[])
        provider._request_sha256 = '0' * 64
        provider._official_attempts = []
        request, _, _, _ = fixture()
        prompts = wire.read(ROOT / PROMPT)
        payload = {'instructions': [prompts['common'], prompts['generate']], 'request': request}
        with patch.object(wire.old.codec.DeepSeekChatProvider, '_bridge', return_value='SYNTHETIC') as bridge:
            self.assertEqual(provider._bridge('generate', payload, {}), 'SYNTHETIC')
            self.assertEqual(provider._active_context['prompt_bundle_sha256'], flow.file_hash(ROOT / PROMPT))
            bridge.assert_called_once_with(provider, 'generate', payload, {})
            old = wire.read(ROOT / 'configs/q01_provider_prompts.json')
            with self.assertRaises(wire.Error):
                provider._bridge('generate', dict(payload, instructions=[old['common'], old['generate']]), {})

    def test_i01_http_context_binds_new_prompt_and_current_business_request(self):
        root = Path(os.environ.get('I01_PROVENANCE_TMP', ROOT / 'runs/i01_fix/rev026/tests/semantic/tmp'))
        root.mkdir(parents=True, exist_ok=True)
        case = Path(tempfile.mkdtemp(prefix='context_', dir=root))
        request, _, _, route = fixture()
        for name, value in [('derived_request.json', request), ('recognition_record.json', route)]:
            wire.put(case / name, value)
        source = {k: request[k] for k in app.intent.rules.INPUT_FIELDS}
        route['source_four_fields_sha256'] = flow.digest(source)
        (case / 'recognition_record.json').write_text(flow.canonical(route) + '\n')
        budget = SimpleNamespace(config={'SYNTHETIC': True}, request_sha256=flow.file_hash(case / 'derived_request.json'),
                                 _append=lambda *_: None)
        gate = SimpleNamespace(run=case, projection=source, require_derived=lambda: None,
                               release={'qa_budget': budget.config, 'four_fields_sha256': flow.digest(source),
                                        'file_hashes': {p: flow.file_hash(ROOT / p) for p in wire.CODE_FILES | {PROMPT}}})
        sender = wire.DerivedRequestHTTP.__new__(wire.DerivedRequestHTTP)
        sender.gate, sender.budget = gate, budget
        context = {'request_sha256': budget.request_sha256, 'business_request_sha256': flow.digest(request),
                   'prompt_bundle_sha256': flow.file_hash(ROOT / PROMPT),
                   'schema_sha256': flow.digest(wire.old.SCHEMAS),
                   'code_hashes': {p.relative_to(ROOT).as_posix(): flow.file_hash(p) for p in wire.old.CODE_PATHS}}
        body = {'messages': [{'content': 'SYNTHETIC'}, {'content': flow.canonical({'request': request})}]}
        checked = sender._context(body, context)
        self.assertEqual(checked['prompt_bundle_sha256'], context['prompt_bundle_sha256'])
        self.assertTrue(set(wire.CODE_FILES) <= set(checked['code_hashes']))
        for bad in (dict(context, prompt_bundle_sha256=flow.file_hash(ROOT / 'configs/q01_provider_prompts.json')),
                    dict(context, business_request_sha256='f' * 64)):
            with self.assertRaises(wire.Error): sender._context(body, bad)

    def test_encoded_http_mock_uses_new_context_and_preserves_durable_budget(self):
        from test_q01_deepseek_http import SyntheticCounter
        root = Path(os.environ.get('I01_PROVENANCE_TMP', ROOT / 'runs/i01_fix/rev026/tests/semantic/tmp'))
        synthetic_root = Path(tempfile.mkdtemp(prefix='encoded_http_', dir=root))
        (synthetic_root / 'configs').mkdir()
        config_bytes = (ROOT / 'configs/q01_deepseek_budget.json').read_bytes()
        (synthetic_root / 'configs/q01_deepseek_budget.json').write_bytes(config_bytes)
        case = synthetic_root / 'runs/i01/case'
        case.mkdir(parents=True)
        request, raw, classifier, route = fixture()
        route['source_four_fields_sha256'] = flow.digest({k: request[k] for k in app.intent.rules.INPUT_FIELDS})
        for name, value in [('derived_request.json', request), ('recognition_record.json', route)]:
            wire.put(case / name, value)
        counter = SyntheticCounter(tokens=100)
        config = json.loads(config_bytes)
        config.update(max_attempts=4, max_cost='0.018', approved_counter_sha256=flow.digest(counter.identity))
        request_hash = flow.file_hash(case / 'derived_request.json')
        with patch.object(wire.base, 'ROOT', synthetic_root):
            budget = wire.ChainQABudget(case / 'qa_session', config, request_hash)
        self.addCleanup(budget.close)
        gate = SimpleNamespace(run=case, projection={k: request[k] for k in app.intent.rules.INPUT_FIELDS},
            require_derived=lambda: None, release={'qa_budget': config,
                'four_fields_sha256': route['source_four_fields_sha256'],
                'file_hashes': {p: flow.file_hash(ROOT / p) for p in wire.CODE_FILES | {PROMPT}}})
        actions, calls = ['generate', 'verify', 'revise', 'verify'], []
        replies = [answer(sufficiency='partial'), {'verdict': 'issues', 'issues': ['SYNTHETIC check']},
                   answer(sufficiency='partial'), {'verdict': 'pass', 'issues': []}]

        def handler(encoded):
            rows = [json.loads(line) for line in budget.ledger_path.read_text().splitlines()]
            reserves = [row for row in rows if row['event'] == 'reserve']
            self.assertEqual(len(reserves), len(calls) + 1)
            self.assertEqual(rows[-1]['event'], 'reserve')
            body = json.loads(encoded.content)
            calls.append(body)
            return httpx.Response(200, json={'id': 'SYNTHETIC-' + str(len(calls)), 'model': 'deepseek-flash',
                'choices': [{'index': 0, 'message': {'role': 'assistant',
                    'content': flow.canonical(replies[len(calls) - 1])}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}})

        provider = wire.DerivedQAProvider(wire.read(ROOT / 'configs/q01_deepseek.json'), budget,
            request_hash, gate=gate, token_counter=counter, http_transport=httpx.MockTransport(handler),
            test_credential='SYNTHETIC-I01-HTTP-TEST-KEY')
        prompts = wire.read(ROOT / PROMPT)
        result = flow.run_flow(request, flow.group_config('GV'), provider,
            retriever=app.PredictionContext(FakeRetriever(raw), classifier, route), runtime={
                'record_kind': 'provider_contract_mock', 'prompts': prompts,
                'generation_settings': {'provider': 'SYNTHETIC http_mock'}})
        self.assertEqual(result['status'], 'evidence_insufficient')
        self.assertTrue(result['publishable'])
        self.assertEqual(len(calls), 4)
        for action, body in zip(actions, calls):
            self.assertIn(prompts['common'], body['messages'][0]['content'])
            self.assertIn(prompts[action], body['messages'][0]['content'])
            business = json.loads(body['messages'][1]['content'])
            self.assertEqual(business['request'], request)
            self.assertIn('actual output', business['scenario_context']['class_semantics'])
            self.assertEqual(business['evidence_cards'], flow.evidence_payload(raw))
            self.assertNotIn(STATIC, flow.canonical(body))
        self.assertEqual(Decimal(budget.snapshot()['reserved_cost']), Decimal('.0135168'))
        self.assertEqual(budget.snapshot()['attempt_count'], 4)
        self.assertEqual(provider.snapshot()['actual_http_request_count'], 0)
        self.assertNotIn('SYNTHETIC-I01-HTTP-TEST-KEY', flow.canonical(provider.snapshot()))


if __name__ == '__main__':
    unittest.main()
