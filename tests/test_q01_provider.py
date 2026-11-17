"""Offline normalized-service contract tests; never load a network client or E01.

Set Q01_PROVIDER_RUN_DIR to this run's existing contract directory. Expectations
must be on disk before this suite is executed. The service below is synthetic.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import q01_flow as flow
from q01_provider import BoundedJSONProvider, ProviderError
from test_q01_flow import REQUEST, FakeRetriever, answer, retrieval


ACTIONS = ('generate', 'verify', 'revise')
PASS = {'verdict': 'pass', 'issues': []}
ISSUES = {'verdict': 'issues', 'issues': ['SYNTHETIC preserve the abstract condition.']}


def config(max_attempts=4):
    return {'provider_name': 'SYNTHETIC-CONTRACT', 'endpoint': 'https://mock.example.invalid/v1',
            'models': {a: 'mock-model-v1' for a in ACTIONS},
            'parameters': {a: {} for a in ACTIONS}, 'max_attempts': max_attempts,
            'timeout_seconds': 10, 'max_input_tokens': {a: 4096 for a in ACTIONS},
            'max_output_tokens': {a: 512 for a in ACTIONS}, 'retries': 0, 'concurrency': 1}


def reply(output, **metadata):
    return {'status_code': 200, 'body': json.dumps(dict(
        {'output_text': json.dumps(output), 'model': 'mock-returned-model', 'id': 'mock-response'},
        **metadata))}


def runtime():
    return {'record_kind': 'provider_contract_mock',
            'prompts': json.loads((ROOT / 'configs/q01_provider_prompts.json').read_text()),
            'generation_settings': {'provider': 'provider_contract_mock'}}


class MockService:
    def __init__(self, outputs, mutate=False):
        self.outputs = list(outputs)
        self.calls = []
        self.mutate = mutate

    def __call__(self, action, payload, settings):
        self.calls.append({'action': action, 'payload': copy.deepcopy(payload),
                           'settings': copy.deepcopy(settings)})
        if self.mutate:
            payload['request']['question'] = 'MUTATED'
            payload['evidence_cards'][0]['original_text'] = 'MUTATED'
            if 'candidate' in payload:
                payload['candidate']['answer'] = 'MUTATED'
            if 'issue_report' in payload:
                payload['issue_report']['issues'].append('MUTATED')
            settings['parameters']['mutation'] = True
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class ProviderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(os.environ['Q01_PROVIDER_RUN_DIR']).absolute()
        if (not base.is_relative_to(ROOT / 'runs/q01_provider') or
                not base.is_dir() or base.is_symlink()):
            raise RuntimeError('Output must be a new provider-run contract directory.')
        cls.expected_path = base / 'expectations.json'
        cls.expected = json.loads(cls.expected_path.read_text())
        cls.output = Path(tempfile.mkdtemp(prefix='attempt_', dir=base))
        cls.bindings = {p: flow.file_hash(ROOT / p) for p in (
            'scripts/q01_flow.py', 'scripts/q01_provider.py', 'tests/test_q01_provider.py',
            'configs/q01_offline.json', 'configs/q01_prompts.json',
            'configs/q01_provider_prompts.json', 'configs/q01_provider_pending.json')}
        cls.oracle_hash = flow.file_hash(cls.expected_path)
        cls.results = []

    @classmethod
    def tearDownClass(cls):
        after = {p: flow.file_hash(ROOT / p) for p in cls.bindings}
        (cls.output / 'binding.json').write_text(json.dumps({
            'before': cls.bindings, 'after': after, 'code_unchanged_during_test': after == cls.bindings,
            'oracle_before_sha256': cls.oracle_hash,
            'oracle_after_sha256': flow.file_hash(cls.expected_path),
            'record_kind': 'provider_contract_mock', 'network_calls': 0,
            'observations': cls.results}, indent=2) + '\n')
        if after != cls.bindings or flow.file_hash(cls.expected_path) != cls.oracle_hash:
            raise AssertionError('Source or oracle changed during contract execution.')

    def flow_case(self, name, outputs, group='GV', result=None, mutate=False, max_attempts=4,
                  redact_values=()):
        service = MockService(outputs, mutate=mutate)
        provider = BoundedJSONProvider(config(max_attempts), service, redact_values=redact_values)
        retriever = FakeRetriever(result)
        record = flow.run_flow(copy.deepcopy(REQUEST), flow.group_config(group), provider,
                               retriever=retriever, runtime=runtime())
        record['provider_observation'] = provider.snapshot()
        path = self.output / (name + '.json')
        flow.write_record(path, record)
        self.assertEqual(json.loads(path.read_text()), record)
        self.assertEqual(record['identity']['code_sha256'], self.bindings['scripts/q01_flow.py'])
        self.assertEqual(record['identity']['provider']['code_sha256'], self.bindings['scripts/q01_provider.py'])
        self.assertEqual(len(service.calls), len(provider.attempts))
        self.assertEqual(record['semantic_support'], 'not_established')
        for stage, ceiling in self.expected['caps'].items():
            self.assertLessEqual(record['counts'][stage], ceiling)
        self.results.append({'case': name, 'record': path.name, 'status': record['status'],
                             'actual_attempts': len(provider.attempts)})
        return record, provider, service

    def test_normal_and_single_revision(self):
        for name, outputs, group, counts in (
            ('normal_generate', [reply(answer())], 'T', (1, 0, 0)),
            ('normal_verify', [reply(answer()), reply(PASS)], 'GV', (1, 1, 0)),
            ('one_revision', [reply(answer()), reply(ISSUES), reply(answer()), reply(PASS)], 'GV', (1, 2, 1))):
            with self.subTest(name=name):
                rec, provider, service = self.flow_case(name, outputs, group)
                self.assertEqual(rec['status'], self.expected[name])
                self.assertTrue(rec['publishable'])
                self.assertEqual(tuple(rec['counts'][a] for a in ACTIONS), counts)
                self.assertEqual(len(service.calls), sum(counts))
                self.assertEqual(provider.snapshot()['actual_bill'], 'unknown')
                for event in provider.attempts:
                    self.assertGreater(event['elapsed_seconds'], 0)
                    self.assertEqual(event['returned_model'], 'mock-returned-model')
                    self.assertEqual(event['token_usage'], 'unknown')
                    self.assertEqual(event['settings']['retries'], 0)

    def test_failure_taxonomy_and_no_retry(self):
        cases = {
            'timeout': TimeoutError('SYNTHETIC timeout'),
            'rate_limit': {'status_code': 429, 'body': 'SYNTHETIC rate limit'},
            'service_error': {'status_code': 503, 'body': 'SYNTHETIC unavailable'},
            'refusal': {'status_code': 200, 'body': '{"refusal":true}'},
            'truncated': {'status_code': 200, 'body': '{"truncated":true}'},
            'empty_output': {'status_code': 200, 'body': ''},
            'invalid_json': {'status_code': 200, 'body': '{not json'},
            'schema_mismatch': reply({'answer': 'SYNTHETIC missing fields'}),
            'unrecordable_response': {'unsafe': {1, 2}},
            'transport_error': RuntimeError('SYNTHETIC transport failure'),
        }
        self.assertEqual(set(cases), set(self.expected['errors']))
        for name, response in cases.items():
            with self.subTest(name=name):
                rec, provider, _ = self.flow_case(name, [response], 'T')
                self.assertFalse(rec['publishable'])
                self.assertIsNone(rec['final_answer'])
                self.assertEqual(rec['counts']['generate'], 1)
                self.assertEqual(len(provider.attempts), 1)
                self.assertEqual(provider.attempts[0]['error_code'], name)

    def test_strict_json_and_exact_fields(self):
        texts = ['{}', '{"answer":"x","answer":"y"}', '{"answer":NaN}',
                 json.dumps(dict(answer(), extra='forbidden')),
                 '```json\n' + json.dumps(answer()) + '\n```']
        for index, text in enumerate(texts):
            rec, provider, _ = self.flow_case('strict_' + str(index), [
                {'status_code': 200, 'body': json.dumps({'output_text': text})}], 'T')
            self.assertFalse(rec['publishable'])
            self.assertEqual(len(provider.attempts), 1)

    def test_partial_and_insufficient_and_failed_verify(self):
        ret = retrieval('partial_task_coverage')
        for name, outputs in (
                ('partial_after_pass', [reply(answer()), reply(PASS)]),
                ('partial_after_revision', [reply(answer()), reply(ISSUES), reply(answer()), reply(PASS)])):
            rec, _, _ = self.flow_case(name, outputs, result=ret)
            self.assertEqual(rec['status'], self.expected[name])
            self.assertTrue(rec['publishable'])
            self.assertTrue(rec['service_message'])
            self.assertEqual(rec['retrieval'], dict(ret, retrieval_mode='graph'))
        for name, outputs in (
                ('candidate_insufficient', [reply(answer(sufficiency='insufficient')), reply(PASS)]),
                ('verify_failure', [reply(answer()), TimeoutError('SYNTHETIC verify timeout')])):
            rec, _, _ = self.flow_case(name, outputs)
            self.assertFalse(rec['publishable'])
            self.assertIsNone(rec['final_answer'])
            if name == 'verify_failure':
                self.assertEqual(rec['status'], self.expected[name]['status'])
            else:
                self.assertTrue(rec['service_message'])

    def test_budget_denial_and_payload_copies(self):
        rec, provider, service = self.flow_case('budget', [reply(answer())], max_attempts=1)
        self.assertEqual(rec['status'], 'verification_failed')
        self.assertEqual(len(provider.attempts), 1)
        self.assertEqual(provider.denials[0]['code'], 'budget_exhausted')
        rec, provider, service = self.flow_case('copies', [reply(answer()), reply(ISSUES),
                                                         reply(answer()), reply(PASS)], mutate=True)
        self.assertTrue(rec['publishable'])
        cards = flow.evidence_payload(retrieval())
        for call in service.calls:
            self.assertEqual(call['payload']['request'], REQUEST)
            self.assertEqual(call['payload']['evidence_cards'], cards)
            self.assertFalse({'rank', 'score', 'relation_paths'} & set(call['payload']['evidence_cards'][0]))
            self.assertNotIn('MUTATED', json.dumps(call))
        with self.assertRaises(ProviderError) as caught:
            provider.generate(service.calls[0]['payload'])
        self.assertEqual(caught.exception.code, 'budget_exhausted')
        self.assertEqual(len(service.calls), 4)

    def test_redaction_in_response_and_exception(self):
        secret = 'SYNTHETIC-CREDENTIAL-SENTINEL'
        response = reply(answer(), model='mock-' + secret, id='mock-' + secret,
                         usage={'input_tokens': 12, 'output_tokens': 3})
        response['headers'] = {'Authorization': 'Bearer ' + secret}
        response['request_id'] = secret
        rec, provider, _ = self.flow_case('redacted_metadata', [response], 'T', redact_values=[secret])
        self.assertTrue(rec['publishable'])
        self.assertNotIn(secret, json.dumps(rec))
        self.assertNotIn('Authorization', json.dumps(provider.snapshot()))
        self.assertEqual(provider.attempts[0]['token_usage'], {'input_tokens': 12, 'output_tokens': 3})
        rec, _, _ = self.flow_case('redacted_exception', [RuntimeError('Authorization: Bearer ' + secret)],
                                   'T', redact_values=[secret])
        self.assertNotIn(secret, json.dumps(rec))

    def test_runtime_and_configuration_fail_closed(self):
        provider = BoundedJSONProvider(config(), MockService([]))
        bad_runtime = runtime()
        bad_runtime['record_kind'] = 'real_provider'
        for value in (None, bad_runtime, dict(runtime(), unsafe={1, 2})):
            rec = flow.run_flow(copy.deepcopy(REQUEST), flow.group_config('GV'), provider,
                                retriever=FakeRetriever(), runtime=value)
            self.assertEqual(rec['status'], 'invalid_config')
            self.assertFalse(rec['publishable'])
            self.assertEqual(sum(rec['counts'].values()), 0)
            flow.write_record(self.output / ('bad_runtime_' + str(len(self.results)) + '.json'), rec)
            self.results.append({'case': 'runtime_rejection', 'status': rec['status']})
        for field, value in (('retries', 1), ('concurrency', 2), ('max_attempts', 6),
                             ('endpoint', 'https://example.com/api')):
            bad = config()
            bad[field] = value
            with self.assertRaises(ValueError):
                BoundedJSONProvider(bad, MockService([]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
