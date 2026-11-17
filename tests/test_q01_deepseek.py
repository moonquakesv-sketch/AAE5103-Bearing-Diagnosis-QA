"""Official DeepSeek protocol tests using synthetic in-process exchanges only.

Set Q01_DEEPSEEK_RUN_DIR to a fresh provider-run contract directory containing
expectations.json. No credential, HTTP client or evaluation material is loaded.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import q01_flow as flow
from q01_deepseek import DeepSeekChatProvider
from q01_provider import ProviderError
from test_q01_flow import REQUEST, FakeRetriever, answer, retrieval
from test_q01_provider import config as base_config, runtime


ACTIONS = ('generate', 'verify', 'revise')
PASS = {'verdict': 'pass', 'issues': []}
ISSUES = {'verdict': 'issues', 'issues': ['SYNTHETIC preserve the supplied condition.']}
ENDPOINT = 'https://api.deepseek.com/chat/completions'
MODEL = 'deepseek-flash'
SERVED = 'DeepSeek-V4.1-Flash-1017'


def config(max_attempts=4):
    value = base_config(max_attempts)
    value.update(provider_name='DeepSeek', endpoint=ENDPOINT,
                 models={action: MODEL for action in ACTIONS},
                 parameters={action: {'thinking': {'type': 'disabled'},
                                      'temperature': 0, 'top_p': 1}
                             for action in ACTIONS})
    value['timeout_seconds'] = 60
    value['max_input_tokens'] = {action: 8192 for action in ACTIONS}
    value['max_output_tokens'] = {'generate': 1024, 'verify': 512, 'revise': 1024}
    return value


def response(output=None, finish='stop', **metadata):
    if output is None:
        output = answer()
    body = {'id': 'SYNTHETIC-deepseek-response', 'model': SERVED,
            'choices': [{'index': 0, 'finish_reason': finish,
                         'message': {'role': 'assistant', 'content': json.dumps(output)}}]}
    body.update(metadata)
    return {'status_code': 200, 'body': json.dumps(body)}


def changed_body(reply, change):
    reply = copy.deepcopy(reply)
    body = json.loads(reply['body'])
    change(body)
    reply['body'] = json.dumps(body)
    return reply


class OfficialFixture:
    def __init__(self, outputs, mutate=False):
        self.outputs = list(outputs)
        self.calls = []
        self.mutate = mutate

    def __call__(self, endpoint, body, settings):
        self.calls.append({'endpoint': endpoint, 'body': copy.deepcopy(body),
                           'settings': copy.deepcopy(settings)})
        if self.mutate:
            body['messages'][1]['content'] = 'MUTATED'
            body['model'] = 'MUTATED'
            settings['parameters']['mutation'] = True
        result = self.outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class DeepSeekContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(os.environ['Q01_DEEPSEEK_RUN_DIR']).absolute()
        if (not base.is_relative_to(ROOT / 'runs/q01_provider') or
                not base.is_dir() or base.is_symlink()):
            raise RuntimeError('Output must be an existing new provider-run directory.')
        cls.oracle_path = base / 'expectations.json'
        cls.oracle = json.loads(cls.oracle_path.read_bytes())
        cls.oracle_hash = flow.file_hash(cls.oracle_path)
        cls.output = Path(tempfile.mkdtemp(prefix='attempt_', dir=base))
        cls.bindings = {p: flow.file_hash(ROOT / p) for p in (
            'scripts/q01_flow.py', 'scripts/q01_provider.py', 'scripts/q01_deepseek.py',
            'tests/test_q01_flow.py', 'tests/test_q01_provider.py', 'tests/test_q01_deepseek.py',
            'configs/q01_offline.json', 'configs/q01_prompts.json',
            'configs/q01_provider_prompts.json')}
        cls.results = []
        cls.failures = []

    @classmethod
    def tearDownClass(cls):
        after = {p: flow.file_hash(ROOT / p) for p in cls.bindings}
        summary = {'before': cls.bindings, 'after': after,
                   'source_unchanged_during_test': after == cls.bindings,
                   'oracle_before_sha256': cls.oracle_hash,
                   'oracle_after_sha256': flow.file_hash(cls.oracle_path),
                   'record_kind': 'provider_contract_mock', 'network_calls': 0,
                   'observations': cls.results, 'observation_count': len(cls.results),
                   'test_claim': 'Official protocol codec and control flow only; no real provider call.'}
        (cls.output / 'binding.json').write_text(json.dumps(summary, indent=2) + '\n')
        if after != cls.bindings or flow.file_hash(cls.oracle_path) != cls.oracle_hash:
            raise AssertionError('Tested source or prior expectation changed during execution.')
        print(json.dumps({'deepseek_attempt': str(cls.output), 'observations': len(cls.results)}))

    def flow_case(self, name, outputs, group='GV', result=None, mutate=False,
                  max_attempts=4, redact_values=()):
        service = OfficialFixture(outputs, mutate=mutate)
        provider = DeepSeekChatProvider(config(max_attempts), service, redact_values=redact_values)
        request_before = copy.deepcopy(REQUEST)
        rec = flow.run_flow(copy.deepcopy(REQUEST), flow.group_config(group), provider,
                            retriever=FakeRetriever(result), runtime=runtime())
        rec['provider_observation'] = provider.snapshot()
        path = self.output / (name + '.json')
        flow.write_record(path, rec)
        self.assertEqual(json.loads(path.read_bytes()), rec)
        self.assertEqual(REQUEST, request_before)
        self.assertEqual(rec['identity']['code_sha256'], self.bindings['scripts/q01_flow.py'])
        self.assertIn(self.bindings['scripts/q01_deepseek.py'], json.dumps(provider.identity))
        self.assertEqual(rec['semantic_support'], 'not_established')
        self.assertEqual(provider.snapshot()['record_kind'], 'provider_contract_mock')
        self.assertFalse(provider.identity['network_enabled'])
        self.assertEqual(len(service.calls), len(provider.attempts))
        for stage, ceiling in self.oracle['caps'].items():
            self.assertLessEqual(rec['counts'][stage], ceiling)
        self.results.append({'case': name, 'record': path.name, 'status': rec['status'],
                             'attempts': len(provider.attempts),
                             'errors': [event.get('error_code') for event in provider.attempts]})
        return rec, provider, service

    def test_normal_generation_verification_and_revision(self):
        for name, outputs, group, counts in (
                ('normal_generate', [response(answer())], 'T', (1, 0, 0)),
                ('normal_verify', [response(answer()), response(PASS)], 'GV', (1, 1, 0)),
                ('one_revision', [response(answer()), response(ISSUES),
                                  response(answer()), response(PASS)], 'GV', (1, 2, 1))):
            with self.subTest(name=name):
                rec, provider, service = self.flow_case(name, outputs, group)
                self.assertEqual(rec['status'], self.oracle[name])
                self.assertTrue(rec['publishable'])
                self.assertEqual(tuple(rec['counts'][a] for a in ACTIONS), counts)
                self.assertEqual([event['action'] for event in provider.attempts],
                                 ['generate'] if group == 'T' else
                                 ['generate', 'verify', 'revise', 'verify'] if counts[2] else
                                 ['generate', 'verify'])
                self.assertEqual(len(service.calls), sum(counts))
                self.assertEqual(provider.snapshot()['actual_bill'], 'unknown')
                self.assertEqual(provider.snapshot()['estimated_cost'], 'unknown')
                for event in provider.attempts:
                    self.assertEqual(event['requested_model'], MODEL)
                    self.assertEqual(event['returned_model'], SERVED)
                    self.assertEqual(event['token_usage'], 'unknown')
                    self.assertGreater(event['elapsed_seconds'], 0)

    def test_official_encoding_full_cards_and_data_isolation(self):
        ret = retrieval()
        secret = 'SYNTHETIC_ADJACENT_EVALUATION_NOT_FOR_PROVIDER'
        ret.update(reference_answer=secret, scoring_rubrics=secret, candidate_paths=secret)
        ret['results'][0].update(hidden_reference=secret)
        ret['results'][0]['source']['audit_notes'] = secret
        ret['information_separation']['demo_scenario']['supporting_evidence_ids'] = [secret]
        rec, _, service = self.flow_case('official_encode', [response(answer()), response(PASS)], result=ret)
        expected_cards = flow.evidence_payload(ret)
        for call, trace in zip(service.calls, rec['provider_trace']):
            body = call['body']
            self.assertEqual(call['endpoint'], self.oracle['endpoint'])
            self.assertEqual(body['model'], self.oracle['request_model'])
            self.assertIs(body['stream'], False)
            self.assertEqual(body['response_format'], {'type': 'json_object'})
            self.assertEqual(body['max_tokens'], config()['max_output_tokens'][trace['action']])
            self.assertEqual(body['temperature'], 0)
            self.assertEqual(body['top_p'], 1)
            self.assertEqual(body['thinking'], {'type': 'disabled'})
            self.assertFalse({'tools', 'tool_choice', 'functions', 'web_search', 'retrieve'} & set(body))
            self.assertNotIn(secret, json.dumps(body))
            self.assertEqual([m['role'] for m in body['messages']], ['system', 'user'])
            system = body['messages'][0]['content']
            self.assertIn('json', system.lower())
            self.assertIn('citations' if trace['action'] != 'verify' else 'verdict', system)
            user = json.loads(body['messages'][1]['content'])
            expected = {k: v for k, v in trace['payload'].items()
                        if k not in ('instructions', 'generation_settings')}
            self.assertEqual(user, expected)
            self.assertEqual(user['evidence_cards'], expected_cards)
            self.assertEqual(user['request'], REQUEST)
            self.assertFalse({'rank', 'score', 'relation_paths'} & set(user['evidence_cards'][0]))
            self.assertEqual(call['settings']['retries'], 0)
            self.assertEqual(call['settings']['concurrency'], 1)
            self.assertEqual(call['settings']['timeout_seconds'], 60)
            self.assertEqual(call['settings']['max_input_tokens'], 8192)

    def test_finish_reason_and_refusal_fail_closed(self):
        cases = {
            'length': response(answer(), finish='length'),
            'content_filter': response(answer(), finish='content_filter'),
            'refusal_message': changed_body(response(), lambda b: b['choices'][0]['message'].update(
                refusal='SYNTHETIC refusal', content=None)),
            'tool_calls': response(answer(), finish='tool_calls'),
            'aborted': response(answer(), finish='aborted'),
            'insufficient_system_resource': response(answer(), finish='insufficient_system_resource'),
            'unknown': response(answer(), finish='SYNTHETIC_UNKNOWN_FINISH'),
        }
        for name, reply in cases.items():
            with self.subTest(name=name):
                rec, provider, service = self.flow_case('finish_' + name, [reply], 'T')
                self.assertFalse(rec['publishable'])
                self.assertIsNone(rec['final_answer'])
                self.assertEqual(len(service.calls), 1)
                self.assertEqual(provider.attempts[0]['error_code'],
                                 self.oracle['finish_reason_contract'][name])

    def test_strict_outer_and_inner_json_and_exact_schemas(self):
        cases = {
            'empty_content': changed_body(response(), lambda b: b['choices'][0]['message'].update(content='')),
            'malformed_content': changed_body(response(), lambda b: b['choices'][0]['message'].update(content='{bad')),
            'duplicate_content_keys': changed_body(response(), lambda b: b['choices'][0]['message'].update(
                content='{"answer":"a","answer":"b"}')),
            'missing_candidate_field': response({'answer': 'SYNTHETIC incomplete candidate'}),
            'extra_candidate_field': response(dict(answer(), extra='SYNTHETIC forbidden')),
            'missing_choices': {'status_code': 200, 'body': '{"model":"synthetic"}'},
            'multiple_choices': changed_body(response(), lambda b: b['choices'].append(copy.deepcopy(b['choices'][0]))),
            'empty_choices': changed_body(response(), lambda b: b.update(choices=[])),
            'malformed_body': {'status_code': 200, 'body': '{bad'},
            'duplicate_body_keys': {'status_code': 200, 'body': '{"choices":[],"choices":[]}'},
            'null_content': changed_body(response(), lambda b: b['choices'][0]['message'].update(content=None)),
            'missing_content': changed_body(response(), lambda b: b['choices'][0]['message'].pop('content')),
            'invalid_role': changed_body(response(), lambda b: b['choices'][0]['message'].update(role='user')),
            'boolean_choice_index': changed_body(response(), lambda b: b['choices'][0].update(index=False)),
            'tool_payload_with_stop': changed_body(response(), lambda b: b['choices'][0]['message'].update(
                tool_calls=[{'id': 'SYNTHETIC tool'}])),
            'unrecordable_response': {'status_code': 200, 'body': {'SYNTHETIC_UNRECORDABLE_SET'}},
        }
        self.assertEqual(set(cases), set(self.oracle['output_errors']))
        for name, reply in cases.items():
            with self.subTest(name=name):
                rec, provider, service = self.flow_case(name, [reply], 'T')
                self.assertFalse(rec['publishable'])
                self.assertEqual(len(service.calls), 1)
                self.assertEqual(provider.attempts[0]['error_code'], self.oracle['output_errors'][name])

    def test_transport_errors_consume_attempt_without_retry(self):
        cases = {'timeout': TimeoutError('SYNTHETIC timeout'),
                 'rate_limit': {'status_code': 429, 'body': 'SYNTHETIC limited'},
                 'service_error': {'status_code': 503, 'body': 'SYNTHETIC unavailable'}}
        for name, reply in cases.items():
            with self.subTest(name=name):
                rec, provider, service = self.flow_case('transport_' + name, [reply], 'T')
                self.assertFalse(rec['publishable'])
                self.assertEqual(rec['counts']['generate'], 1)
                self.assertEqual(len(service.calls), 1)
                self.assertEqual(provider.attempts[0]['error_code'], self.oracle['transport_errors'][name])
                self.assertEqual(provider.snapshot()['actual_attempt_count'], 1)

    def test_unknown_and_supplied_usage_and_model_identity(self):
        reply = changed_body(response(), lambda b: (b.pop('model'), b.pop('id')))
        _, provider, _ = self.flow_case('unknown_metadata', [reply], 'T')
        event = provider.attempts[0]
        for field in ('returned_model', 'response_id', 'request_id', 'token_usage'):
            self.assertEqual(event[field], 'unknown')
        usage = {'prompt_tokens': 123, 'completion_tokens': 17, 'total_tokens': 140,
                 'prompt_cache_hit_tokens': 100, 'prompt_cache_miss_tokens': 23}
        reply = response(usage=usage)
        reply['headers'] = {'x-request-id': 'SYNTHETIC-request-id'}
        _, provider, _ = self.flow_case('supplied_usage', [reply], 'T')
        event = provider.attempts[0]
        self.assertEqual(event['token_usage'], usage)
        self.assertEqual(event['request_id'], 'SYNTHETIC-request-id')
        self.assertEqual(event['requested_model'], MODEL)
        self.assertEqual(event['returned_model'], SERVED)

    def test_attempt_budgets_and_no_second_revision(self):
        rec, provider, service = self.flow_case('total_budget', [response(answer())], max_attempts=1)
        self.assertFalse(rec['publishable'])
        self.assertEqual(rec['status'], 'verification_failed')
        self.assertEqual(len(service.calls), 1)
        self.assertEqual(provider.snapshot()['actual_attempt_count'], 1)
        self.assertEqual(provider.denials[0]['code'], 'budget_exhausted')
        rec, provider, service = self.flow_case('action_budgets', [response(answer()), response(ISSUES),
                                                                response(answer()), response(PASS)])
        self.assertTrue(rec['publishable'])
        self.assertEqual(len(service.calls), 4)
        payloads = {event['action']: event['payload'] for event in rec['provider_trace']}
        for action in ACTIONS:
            with self.subTest(action=action), self.assertRaises(ProviderError) as caught:
                getattr(provider, action)(payloads[action])
            self.assertEqual(caught.exception.code, 'budget_exhausted')
            self.assertEqual(len(service.calls), 4)
        self.assertEqual(provider.snapshot()['action_counts'], {'generate': 1, 'verify': 2, 'revise': 1})
        flow.write_record(self.output / 'action_budget_denials.json', provider.snapshot())

    def test_payload_and_snapshot_copies_preserve_candidate(self):
        original_answer = answer()
        rec, provider, service = self.flow_case('copies', [response(original_answer), response(ISSUES),
                                                         response(original_answer), response(PASS)], mutate=True)
        self.assertTrue(rec['publishable'])
        self.assertEqual(rec['draft'], original_answer)
        self.assertEqual(rec['revisions'], [original_answer])
        for call in service.calls:
            self.assertNotIn('MUTATED', json.dumps(call))
            user = json.loads(call['body']['messages'][1]['content'])
            self.assertEqual(user['request'], REQUEST)
            self.assertEqual(user['evidence_cards'], flow.evidence_payload(retrieval()))
            if 'candidate' in user:
                self.assertEqual(user['candidate'], original_answer)
        snap = provider.snapshot()
        snap['attempts'][0]['status'] = 'MUTATED'
        self.assertNotEqual(provider.snapshot()['attempts'][0]['status'], 'MUTATED')

    def test_log_redaction_for_body_headers_ids_and_exception(self):
        secret = 'SYNTHETIC-DEEPSEEK-CREDENTIAL-SENTINEL'
        reply = response(model=secret, id=secret)
        reply['headers'] = {'Authorization': 'Bearer ' + secret, 'x-request-id': secret}
        reply['request_id'] = secret
        reply = changed_body(reply, lambda b: b.update(debug={'Authorization': 'Bearer ' + secret}))
        rec, provider, _ = self.flow_case('redacted_metadata', [reply], 'T', redact_values=[secret])
        self.assertTrue(rec['publishable'])
        self.assertNotIn(secret, json.dumps(rec))
        self.assertNotIn('Authorization', json.dumps(provider.snapshot()))
        rec, _, _ = self.flow_case('redacted_exception', [RuntimeError('Authorization: Bearer ' + secret)],
                                   'T', redact_values=[secret])
        self.assertFalse(rec['publishable'])
        self.assertNotIn(secret, json.dumps(rec))
        echoed = answer()
        echoed['answer'] = secret
        rec, _, _ = self.flow_case('credential_in_candidate', [response(echoed)], 'T', redact_values=[secret])
        self.assertFalse(rec['publishable'])
        self.assertNotIn(secret, json.dumps(rec))

    def test_partial_coverage_insufficient_and_verify_failure(self):
        for name, outputs in (
                ('partial_pass', [response(answer()), response(PASS)]),
                ('partial_revision', [response(answer()), response(ISSUES), response(answer()), response(PASS)])):
            rec, _, _ = self.flow_case(name, outputs, result=retrieval('partial_task_coverage'))
            self.assertEqual(rec['status'], 'evidence_insufficient')
            self.assertTrue(rec['publishable'])
            self.assertTrue(rec['service_message'])
        rec, _, _ = self.flow_case('insufficient', [response(answer(sufficiency='insufficient')), response(PASS)])
        self.assertFalse(rec['publishable'])
        self.assertIsNone(rec['final_answer'])
        self.assertTrue(rec['service_message'])
        rec, _, _ = self.flow_case('verify_failure', [response(answer()), TimeoutError('SYNTHETIC verify timeout')])
        self.assertEqual(rec['status'], 'verification_failed')
        self.assertFalse(rec['publishable'])
        self.assertIsNone(rec['final_answer'])

    def test_configuration_parameter_and_identity_whitelists(self):
        for field, value in (('endpoint', 'https://third-party.invalid/chat/completions'),
                             ('retries', 1), ('concurrency', 2), ('max_attempts', 6),
                             ('models', {a: 'SYNTHETIC-UNSELECTED-MODEL' for a in ACTIONS})):
            bad = config()
            bad[field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, ProviderError)):
                DeepSeekChatProvider(bad, OfficialFixture([]))
        for extra in ({'tools': []}, {'stream': True}, {'max_tokens': 999999}, {'n': 2},
                      {'temperature': -1}, {'top_p': 0}):
            bad = config()
            bad['parameters']['generate'].update(extra)
            with self.subTest(extra=extra), self.assertRaises((ValueError, ProviderError)):
                DeepSeekChatProvider(bad, OfficialFixture([]))

    def test_stage_models_parameters_and_limits_are_independent(self):
        cfg = config()
        cfg['models']['verify'] = 'deepseek-v4-pro'
        cfg['parameters']['verify'] = {'thinking': {'type': 'disabled'}, 'temperature': 0.2, 'top_p': 0.8}
        cfg['max_input_tokens']['verify'] = 4096
        cfg['max_output_tokens']['verify'] = 256
        service = OfficialFixture([response(answer()), response(PASS, model='SYNTHETIC-pro-response')])
        provider = DeepSeekChatProvider(cfg, service)
        rec = flow.run_flow(copy.deepcopy(REQUEST), flow.group_config('GV'), provider,
                            retriever=FakeRetriever(), runtime=runtime())
        rec['provider_observation'] = provider.snapshot()
        path = self.output / 'independent_stage_parameters.json'
        flow.write_record(path, rec)
        self.assertTrue(rec['publishable'])
        generate, verify = service.calls
        self.assertEqual(generate['body']['model'], MODEL)
        self.assertEqual(verify['body']['model'], 'deepseek-v4-pro')
        self.assertEqual(verify['body']['temperature'], 0.2)
        self.assertEqual(verify['body']['top_p'], 0.8)
        self.assertEqual(verify['body']['max_tokens'], 256)
        self.assertEqual(verify['settings']['max_input_tokens'], 4096)
        self.assertEqual(provider.attempts[1]['requested_model'], 'deepseek-v4-pro')
        self.assertEqual(provider.attempts[1]['returned_model'], 'SYNTHETIC-pro-response')
        self.results.append({'case': 'independent_stage_parameters', 'record': path.name,
                             'status': rec['status'], 'attempts': len(provider.attempts)})


if __name__ == '__main__':
    unittest.main(verbosity=2)
