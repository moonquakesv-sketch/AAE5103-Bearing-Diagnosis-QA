"""HTTP and durable-session contracts using httpx.MockTransport only.

All credentials and token counters below are visibly synthetic. The real-mode
branch is exercised only with closed gates and a transport-construction trap.
"""
import ast
import copy
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import q01_flow as flow
import q01_deepseek_http as http_adapter
from q01_deepseek_http import DeepSeekHTTPProvider, SessionBudget
from q01_provider import ProviderError
from test_q01_deepseek import config, response, PASS, ISSUES, MODEL
from test_q01_flow import REQUEST, FakeRetriever, answer, retrieval
from test_q01_provider import runtime


KEY = 'SYNTHETIC-HTTP-CONTRACT-CREDENTIAL-NOT-A-REAL-KEY'
REQUEST_HASH = flow.digest(REQUEST)


def session_config(max_cost='0.02'):
    return {'max_attempts': 5, 'currency': 'USD', 'max_cost': max_cost,
            'allowed_groups': ['T', 'GV'],
            'model_prices': {'deepseek-flash': {
                'input_per_million': '0.30', 'output_per_million': '1.20',
                'documented_served_identity': 'DeepSeek-V4.1-Flash-1017',
                'source_url': 'https://api-docs.deepseek.com/quick_start/pricing'}},
            'approved_counter_sha256': None}


def payload():
    prompts = runtime()['prompts']
    return {'instructions': [prompts['common'], prompts['generate']],
            'request': copy.deepcopy(REQUEST),
            'scenario_context': {'assumption_text': 'SYNTHETIC HTTP contract assumption',
                                 'unknown_information': 'SYNTHETIC HTTP contract unknown'},
            'evidence_cards': flow.evidence_payload(retrieval()),
            'generation_settings': {'provider': 'provider_contract_mock'}}


class SyntheticCounter:
    def __init__(self, tokens=8192, mutate=False, models=None):
        self.tokens = tokens
        self.mutate = mutate
        self.calls = []
        self.identity = {'kind': 'http_mock_token_counter', 'scope': 'complete_chat_messages',
                         'verified_models': models if models is not None else [MODEL],
                         'chat_template_sha256': '0' * 64, 'artifact_sha256': '1' * 64}

    def count_messages(self, messages, model):
        before = copy.deepcopy(messages)
        self.calls.append({'model': model, 'messages': before})
        if self.mutate:
            messages[1]['content'] = 'MUTATED BY SYNTHETIC COUNTER'
        return {'input_tokens': self.tokens, 'model': model,
                'messages_sha256': flow.digest(before),
                'chat_template_sha256': self.identity['chat_template_sha256']}


class DeepSeekHTTPContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(os.environ['Q01_HTTP_RUN_DIR']).absolute()
        if (not base.is_relative_to(ROOT / 'runs/q01_provider') or
                not base.is_dir() or base.is_symlink()):
            raise RuntimeError('Use an existing isolated provider-run HTTP contract directory.')
        cls.oracle_path = base / 'expectations.json'
        cls.oracle = json.loads(cls.oracle_path.read_bytes())
        cls.oracle_hash = flow.file_hash(cls.oracle_path)
        cls.output = Path(tempfile.mkdtemp(prefix='attempt_', dir=base))
        cls.binding = {p: flow.file_hash(ROOT / p) for p in (
            'scripts/q01_flow.py', 'scripts/q01_provider.py', 'scripts/q01_deepseek.py',
            'scripts/q01_deepseek_http.py', 'tests/test_q01_deepseek_http.py',
            'tests/test_q01_deepseek.py', 'tests/test_q01_flow.py',
            'tests/test_q01_provider.py', 'configs/q01_provider_prompts.json')}
        cls.observations = []

    @classmethod
    def tearDownClass(cls):
        after = {p: flow.file_hash(ROOT / p) for p in cls.binding}
        record = {'before': cls.binding, 'after': after,
                  'source_unchanged_during_execution': cls.binding == after,
                  'oracle_before_sha256': cls.oracle_hash,
                  'oracle_after_sha256': flow.file_hash(cls.oracle_path),
                  'record_kind': 'http_mock', 'actual_official_http_requests': 0,
                  'observations': cls.observations}
        (cls.output / 'binding.json').write_text(json.dumps(record, indent=2) + '\n')
        if cls.binding != after or flow.file_hash(cls.oracle_path) != cls.oracle_hash:
            raise AssertionError('HTTP source or frozen expectation changed during test.')
        print(json.dumps({'http_contract_attempt': str(cls.output),
                          'observations': len(cls.observations)}))

    def budget(self, label, max_cost='0.02'):
        directory = self.output / label
        directory.mkdir(exist_ok=False)
        value = SessionBudget(directory, session_config(max_cost), REQUEST_HASH)
        self.addCleanup(value.close)
        return value, directory

    def provider(self, budget, group, handler, *, counter=True, key=KEY, cfg=None):
        if counter is True:
            counter = SyntheticCounter()
        return DeepSeekHTTPProvider(config(5) if cfg is None else cfg, budget, group, REQUEST_HASH,
                                    token_counter=counter,
                                    credential_env='Q01_SYNTHETIC_UNUSED_CREDENTIAL_ENV',
                                    http_transport=httpx.MockTransport(handler), test_credential=key)

    def save(self, label, provider=None, record=None, budget=None):
        value = {}
        if provider is not None:
            value['provider'] = provider.snapshot()
        if record is not None:
            value['record'] = record
        if budget is not None:
            value['session_budget'] = budget.snapshot()
        path = self.output / (label + '.json')
        self.assertFalse(KEY in json.dumps(value), 'Synthetic credential echo in provider snapshot.')
        flow.write_record(path, value)
        self.assertEqual(json.loads(path.read_bytes()), value)
        self.observations.append({'case': label, 'record': path.name})

    def run_flow(self, provider, group):
        return flow.run_flow(copy.deepcopy(REQUEST), flow.group_config(group), provider,
                             retriever=FakeRetriever(), runtime=runtime())

    def test_normal_T_GV_shared_five_attempts_and_durable_reserves(self):
        budget, directory = self.budget('normal_session')
        replies = [response(answer()), response(answer()), response(ISSUES),
                   response(answer()), response(PASS)]
        calls = []

        def handler(request):
            ledger = [json.loads(line) for line in (directory / 'session_ledger.jsonl').read_text().splitlines()]
            reserves = [row for row in ledger if row['event'] == 'reserve']
            self.assertEqual(len(reserves), len(calls) + 1, 'Reserve must be durable before post.')
            self.assertTrue((directory / 'mock_freeze.json').is_file())
            self.assertEqual(request.method, 'POST')
            self.assertEqual(str(request.url), self.oracle['endpoint'])
            self.assertTrue(request.headers['authorization'] == 'Bearer ' + KEY,
                            'Mock authentication header assertion failed.')
            body = json.loads(request.content)
            self.assertFalse(body['stream'])
            self.assertFalse({'tools', 'tool_choice'} & set(body))
            calls.append(body)
            reply = replies.pop(0)
            return httpx.Response(reply['status_code'], content=reply['body'])

        provider_t = self.provider(budget, 'T', handler)
        rec_t = self.run_flow(provider_t, 'T')
        provider_gv = self.provider(budget, 'GV', handler)
        rec_gv = self.run_flow(provider_gv, 'GV')
        self.assertTrue(rec_t['publishable'])
        self.assertTrue(rec_gv['publishable'])
        self.assertEqual(len(calls), 5)
        snap = budget.snapshot()
        self.assertEqual(snap['attempt_count'], self.oracle['aggregate_caps']['total'])
        self.assertEqual(snap['counts'], {'generate': 2, 'verify': 2, 'revise': 1})
        self.assertEqual(Decimal(snap['reserved_cost']), Decimal(self.oracle['full_T_GV_reservation']))
        self.assertEqual(provider_t.identity['kind'], 'provider_contract_mock')
        self.assertEqual(provider_t.identity['transport'], 'http_mock')
        self.assertEqual(provider_gv.snapshot()['actual_http_request_count'], 0)
        self.assertNotEqual(rec_gv['status'], 'completed_verified_model_report')
        self.save('normal_T', provider_t, rec_t, budget)
        self.save('normal_GV', provider_gv, rec_gv, budget)
        with self.assertRaises(ProviderError) as caught:
            provider_gv.verify(rec_gv['provider_trace'][1]['payload'])
        self.assertEqual(caught.exception.code, 'budget_exhausted')
        self.assertEqual(len(calls), 5)
        self.save('aggregate_denial', provider_gv, budget=budget)

    def test_http_client_controls_and_explicit_native_zero_retry(self):
        budget, _ = self.budget('client_controls')
        calls = []

        def handler(request):
            calls.append(request.extensions.get('timeout'))
            reply = response()
            return httpx.Response(200, content=reply['body'])

        provider = self.provider(budget, 'T', handler)
        real_client = httpx.Client
        with mock.patch.object(http_adapter.httpx, 'Client', wraps=real_client) as client:
            rec = self.run_flow(provider, 'T')
        self.assertTrue(rec['publishable'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(client.call_count, 1)
        kwargs = client.call_args.kwargs
        self.assertIs(kwargs['follow_redirects'], False)
        self.assertIs(kwargs['trust_env'], False)
        self.assertTrue(all(value == 60 for value in calls[0].values()))
        tree = ast.parse((ROOT / 'scripts/q01_deepseek_http.py').read_text())
        native_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == 'HTTPTransport']
        self.assertTrue(native_calls, 'The real transport must explicitly set its zero retry policy.')
        self.assertTrue(all(any(k.arg == 'retries' and isinstance(k.value, ast.Constant)
                                and k.value.value == 0 for k in node.keywords) for node in native_calls))
        self.save('client_controls', provider, rec, budget)

    def test_failures_one_post_no_retry_no_release_or_redirect(self):
        cases = {
            'timeout': httpx.ReadTimeout('SYNTHETIC mock read timeout'),
            '429': httpx.Response(429, text='SYNTHETIC rate limited'),
            '503': httpx.Response(503, text='SYNTHETIC unavailable'),
            'redirect': httpx.Response(302, headers={'location': 'https://third-party.invalid/forbidden'}),
            'refusal': httpx.Response(200, content=response(finish='content_filter')['body']),
            'truncated': httpx.Response(200, content=response(finish='length')['body']),
            'invalid_json': httpx.Response(200, text='{SYNTHETIC malformed body'),
        }
        self.assertEqual(set(cases), set(self.oracle['failure_cases']))
        for label, reply in cases.items():
            with self.subTest(label=label):
                budget, directory = self.budget('failure_' + label)
                calls = []

                def handler(request):
                    calls.append(request.method)
                    if isinstance(reply, Exception):
                        raise reply
                    return reply

                provider = self.provider(budget, 'T', handler)
                rec = self.run_flow(provider, 'T')
                self.assertFalse(rec['publishable'])
                self.assertIsNone(rec['final_answer'])
                self.assertEqual(calls, ['POST'])
                snap = budget.snapshot()
                self.assertEqual(snap['attempt_count'], 1)
                self.assertEqual(Decimal(snap['reserved_cost']), Decimal('0.0036864'))
                reserves = [json.loads(line) for line in (directory / 'session_ledger.jsonl').read_text().splitlines()
                            if json.loads(line)['event'] == 'reserve']
                self.assertEqual(len(reserves), 1)
                self.save('failure_' + label, provider, rec, budget)

    def test_budget_and_flow_input_guards_never_post(self):
        budget, _ = self.budget('money_denial', max_cost='0.003')
        calls = []
        provider = self.provider(budget, 'T', lambda request: calls.append(request))
        rec = self.run_flow(provider, 'T')
        self.assertFalse(rec['publishable'])
        self.assertEqual(calls, [])
        self.assertEqual(budget.snapshot()['attempt_count'], 0)
        self.assertEqual(Decimal(budget.snapshot()['reserved_cost']), Decimal('0'))
        self.save('money_denial', provider, rec, budget)

        budget, _ = self.budget('flow_guards')
        self.provider(budget, 'T', lambda request: calls.append(request))
        for group, request_hash in (('T', REQUEST_HASH), ('G', REQUEST_HASH), ('GV', 'f' * 64)):
            with self.subTest(group=group), self.assertRaises((ProviderError, ValueError)):
                DeepSeekHTTPProvider(config(5), budget, group, request_hash,
                    token_counter=SyntheticCounter(), http_transport=httpx.MockTransport(lambda req: calls.append(req)),
                    credential_env='Q01_SYNTHETIC_UNUSED_CREDENTIAL_ENV', test_credential=KEY)
        self.assertEqual(calls, [])
        self.assertEqual(budget.snapshot()['attempt_count'], 0)
        self.save('flow_guards', budget=budget)

    def test_exclusive_ledger_owner_and_no_reopen_reset(self):
        budget, directory = self.budget('exclusive_owner')
        before = (directory / 'session_ledger.jsonl').read_bytes()
        with self.assertRaises((ProviderError, ValueError, FileExistsError)):
            SessionBudget(directory, session_config(), REQUEST_HASH)
        self.assertEqual((directory / 'session_ledger.jsonl').read_bytes(), before)
        budget.close()
        with self.assertRaises((ProviderError, ValueError, FileExistsError)):
            SessionBudget(directory, session_config(), REQUEST_HASH)
        self.assertEqual((directory / 'session_ledger.jsonl').read_bytes(), before)
        self.save('exclusive_owner', budget=budget)

    def test_shared_session_concurrency_denied_before_second_post(self):
        budget, _ = self.budget('concurrency')
        entered, release = threading.Event(), threading.Event()
        calls, thread_result = [], []

        def handler(request):
            calls.append(request.method)
            entered.set()
            if not release.wait(5):
                raise RuntimeError('SYNTHETIC concurrency test release timeout')
            return httpx.Response(200, content=response()['body'])

        provider_t = self.provider(budget, 'T', handler)
        provider_gv = self.provider(budget, 'GV', handler)

        def first_call():
            try:
                thread_result.append({'output': provider_t.generate(payload())})
            except Exception as error:
                thread_result.append({'error': type(error).__name__, 'message': str(error)})

        worker = threading.Thread(target=first_call)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaises(ProviderError) as caught:
                provider_gv.generate(payload())
            self.assertEqual(caught.exception.code, 'concurrency_forbidden')
            self.assertEqual(calls, ['POST'])
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertIn('output', thread_result[0])
        self.assertEqual(budget.snapshot()['attempt_count'], 1)
        self.save('concurrency_T', provider_t, budget=budget)
        self.save('concurrency_GV', provider_gv, budget=budget)

    def test_counter_credential_and_token_cap_gates(self):
        cases = (('missing_counter', None, KEY), ('missing_key', SyntheticCounter(), ''),
                 ('over_input_cap', SyntheticCounter(8193), KEY),
                 ('counter_model_mismatch', SyntheticCounter(models=['SYNTHETIC-other-model']), KEY))
        for label, counter, key in cases:
            with self.subTest(label=label):
                budget, directory = self.budget(label)
                calls = []
                provider = self.provider(budget, 'T', lambda request: calls.append(request),
                                         counter=counter, key=key)
                rec = self.run_flow(provider, 'T')
                self.assertFalse(rec['publishable'])
                self.assertEqual(calls, [])
                self.assertEqual(budget.snapshot()['attempt_count'], 0)
                self.assertFalse((directory / 'first_request_freeze.json').exists())
                self.save(label, provider, rec, budget)

    def test_first_freeze_prevents_config_change_and_tampering(self):
        for label, tamper in (('config_change', False), ('freeze_tamper', True)):
            with self.subTest(label=label):
                budget, directory = self.budget(label)
                calls = []

                def handler(request):
                    calls.append(request.method)
                    return httpx.Response(200, content=response()['body'])

                first = self.provider(budget, 'T', handler)
                first_record = self.run_flow(first, 'T')
                self.assertTrue(first_record['publishable'])
                self.save(label + '_first_T', first, first_record, budget)
                changed = config(5)
                if tamper:
                    (directory / 'mock_freeze.json').write_text('{"SYNTHETIC":"tampered"}\n')
                else:
                    changed['timeout_seconds'] = 61
                second = self.provider(budget, 'GV', handler, cfg=changed)
                rec = self.run_flow(second, 'GV')
                self.assertFalse(rec['publishable'])
                self.assertEqual(calls, ['POST'])
                self.assertEqual(budget.snapshot()['attempt_count'], 1)
                self.save(label, second, rec, budget)

    def test_unknown_usage_and_bill_are_separate_from_reserve(self):
        budget, _ = self.budget('unknown_usage')
        provider = self.provider(budget, 'T', lambda request: httpx.Response(200, content=response()['body']))
        rec = self.run_flow(provider, 'T')
        self.assertTrue(rec['publishable'])
        snap = provider.snapshot()
        self.assertEqual(snap['actual_bill'], 'unknown')
        self.assertEqual(snap['attempts'][0]['token_usage'], 'unknown')
        self.assertEqual(Decimal(budget.snapshot()['reserved_cost']), Decimal('0.0036864'))
        self.save('unknown_usage', provider, rec, budget)

    def test_credentials_headers_body_and_exception_are_redacted(self):
        for label, raises in (('response_echo', False), ('exception_echo', True)):
            with self.subTest(label=label):
                budget, directory = self.budget(label)

                def handler(request):
                    self.assertTrue(request.headers['authorization'] == 'Bearer ' + KEY,
                                    'Mock authentication header assertion failed.')
                    if raises:
                        raise httpx.ReadTimeout('Authorization: Bearer ' + KEY)
                    reply = response(model=KEY, id=KEY)
                    body = json.loads(reply['body'])
                    body['debug'] = {'Authorization': 'Bearer ' + KEY}
                    return httpx.Response(200, json=body,
                                          headers={'Authorization': 'Bearer ' + KEY, 'x-request-id': KEY})

                provider = self.provider(budget, 'T', handler)
                rec = self.run_flow(provider, 'T')
                self.assertFalse(KEY in json.dumps(rec), 'Synthetic credential echo in flow record.')
                self.assertFalse(KEY in json.dumps(provider.snapshot()), 'Synthetic credential echo in provider log.')
                self.assertFalse(KEY in (directory / 'session_ledger.jsonl').read_text(),
                                 'Synthetic credential echo in ledger.')
                self.save(label, provider, rec, budget)

    def test_counter_receives_detached_full_messages(self):
        budget, _ = self.budget('counter_copy')
        counter = SyntheticCounter(mutate=True)
        calls = []

        def handler(request):
            body = json.loads(request.content)
            self.assertNotIn('MUTATED', json.dumps(body))
            user = json.loads(body['messages'][1]['content'])
            self.assertEqual(user['request'], REQUEST)
            self.assertEqual(user['evidence_cards'], flow.evidence_payload(retrieval()))
            calls.append(body)
            return httpx.Response(200, content=response()['body'])

        provider = self.provider(budget, 'T', handler, counter=counter)
        rec = self.run_flow(provider, 'T')
        self.assertTrue(rec['publishable'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(counter.calls), 1)
        self.save('counter_copy', provider, rec, budget)

    def test_real_identity_branch_stays_closed_without_verified_counter(self):
        budget, directory = self.budget('real_identity_closed')
        provider = DeepSeekHTTPProvider(config(5), budget, 'T', REQUEST_HASH,
                                       token_counter=None,
                                       credential_env='Q01_SYNTHETIC_UNUSED_CREDENTIAL_ENV')
        rt = runtime()
        rt['record_kind'] = 'real_provider'
        with mock.patch.object(http_adapter.httpx, 'HTTPTransport', side_effect=AssertionError('NO NETWORK')) as native:
            rec = flow.run_flow(copy.deepcopy(REQUEST), flow.group_config('T'), provider,
                                retriever=FakeRetriever(), runtime=rt)
        self.assertEqual(provider.identity['kind'], 'real_provider')
        self.assertFalse(rec['publishable'])
        self.assertEqual(provider.snapshot()['actual_http_request_count'], 0)
        self.assertEqual(native.call_count, 0)
        self.assertEqual(budget.snapshot()['attempt_count'], 0)
        self.assertFalse((directory / 'first_request_freeze.json').exists())
        self.save('real_identity_closed', provider, rec, budget)


if __name__ == '__main__':
    unittest.main(verbosity=2)
