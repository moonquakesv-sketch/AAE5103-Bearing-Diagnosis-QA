"""Pure read-only second-pass recomputation of stored SYNTHETIC evidence."""
from pathlib import Path
from decimal import Decimal
import ast
import base64
import datetime
import hashlib
import json
import os
import sys

R = Path(__file__).absolute().parents[1]
P = R.parents[2]
OLD = P / 'runs/e02_llm_baseline_preparation/20261105_003337_rev042_3b109719'
def no_effects(event, args):
    if event == 'open':
        mode, flags = args[1:3]
        assert not ((isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
    assert not event.startswith('socket.')
    assert event not in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.symlink', 'os.link', 'os.chmod', 'os.utime', 'subprocess.Popen', 'os.system'}
sys.addaudithook(no_effects)
def read(path):
    return json.loads(path.read_bytes())
def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

before = read(R / 'validation/before_corrected/results.json')
after = read(R / 'validation/after/results.json')
assert before['test_source_sha256'] == after['test_source_sha256'] == sha(R / 'tests/test_stop_contract.py')
assert after['passed'] == after['total'] == 15
assert before['passed'] == 4 and before['total'] == 15
table = []
for name, expected, skipped in [('generate_cap', 0, [0, 1, 2]), ('judge_cap', 1, [1, 2]), ('third_cap', 3, [])]:
    pre, post = [read(R / 'validation' / phase / name / 'observation.json') for phase in ['before_corrected', 'after']]
    cell = read(R / 'validation/after' / name / 'batch/SYNTHETIC-00.result.json')
    assert post['transport_calls'] == expected and post['ledger_broken'] is True
    assert cell['final']['rating'] is None and cell['metrics']['REF_cov']['unknown'] == 1
    assert all(cell['original_ratings'][i] == {'status': 'not_attempted_batch_stopped', 'rating': None} for i in skipped)
    raw_requests = list((R / 'validation/after' / name / 'batch/raw').glob('*.request.json'))
    assert len(raw_requests) == expected
    table.append({'case': name, 'before_transport_calls': pre['transport_calls'], 'after_transport_calls': len(raw_requests), 'after_skipped_rating_indices': skipped, 'final_rating_is_null': True})

ledger_reviews = []
for batch in sorted((R / 'validation/after').glob('*/batch')):
    ledger = [json.loads(x) for x in (batch / 'ledger.jsonl').read_text().splitlines()]
    assert [r['sequence'] for r in ledger] == list(range(1, len(ledger) + 1))
    reserves = [r for r in ledger if r['event'] == 'reserve']
    assert len({r['slot'] for r in reserves}) == len(reserves)
    total = Decimal(0)
    for entry in reserves:
        total += Decimal(entry['amount_USD'])
        assert Decimal(entry['total_reserved_USD']) == total and entry['no_refund_or_retry'] is True
        request = read(batch / 'raw' / (entry['slot'] + '.request.json'))
        assert request['transport_kind'] == 'injected_synthetic_transport'
        assert entry['body_sha256'] == digest(request['body'])
        assert request['count']['messages_sha256'] == digest(request['body']['messages'])
        packet = json.loads(request['body']['messages'][1]['content'])
        core = packet.get('packet', packet)
        assert core['business_request']['question'].startswith('SYNTHETIC:')
        http = read(batch / 'raw' / (entry['slot'] + '.http.json'))
        response = base64.b64decode(http['raw_base64'])
        assert hashlib.sha256(response).hexdigest() == http['raw_sha256']
        assert http['raw_size'] == len(response) and http['http_client_post_invoked'] is False
    assert not any(r['event'] in {'refund', 'retry', 'resume'} for r in ledger)
    observation = read(batch.parent / 'observation.json')
    assert total == Decimal(observation['reserved_USD'])
    ledger_reviews.append({'case': batch.parent.name, 'reserved_slots': len(reserves), 'reserved_USD': str(total), 'raw_requests_and_responses_matched': True})
full = next(x for x in ledger_reviews if x['case'] == 'full52')
assert full['reserved_slots'] == 52 and Decimal(full['reserved_USD']) == Decimal('1.2460032')
original_provider = (OLD / 'runtime/provider.py').read_text()
expected_provider = original_provider.replace("        try:\n            require(digest(self.config)==self._config_pin,'provider configuration mutated')", "        try:\n            self.ledger.check()\n            require(digest(self.config)==self._config_pin,'provider configuration mutated')", 1).replace("        except Exception as exc:\n            if reserved:", "        except Exception as exc:\n            if not reserved:\n                # A rejected count/binding/budget must stop later slots too.\n                self.ledger.broken = True\n            if reserved:", 1)
assert (R / 'runtime/provider.py').read_text() == expected_provider
original_workflow = (OLD / 'runtime/workflow.py').read_text()
expected_workflow = original_workflow.replace(" for suffix in ['A','B','C']:\n  role=", " for suffix in ['A','B','C']:\n  if provider.ledger.broken:\n   originals.append({'status':'not_attempted_batch_stopped','rating':None})\n   continue\n  role=", 1)
assert (R / 'runtime/workflow.py').read_text() == expected_workflow
def entry(source):
    return ast.dump(next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == 'execute_authorized'), include_attributes=False)
assert entry(expected_workflow) == entry(original_workflow)
assert (R / 'config/execution.json').read_bytes() == (OLD / 'config/execution.json').read_bytes()
assert (R / 'config/new_authorization_template.json').read_bytes() == (OLD / 'config/new_authorization_template.json').read_bytes()
print(json.dumps({'status': 'PASS', 'review_type': 'same_agent_separate_process_read_only_evidence_recomputation', 'not_independent_agent_or_expert_review': True, 'checked_at_JST': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(), 'same_test_oracle_before_after': True, 'cap_stop_comparison': table, 'ledger_reviews': ledger_reviews, 'normal_chain_calls': 52, 'normal_full_cap_reserve_USD': full['reserved_USD'], 'only_runtime_edits': ['runtime/provider.py', 'runtime/workflow.py'], 'execute_authorized_source_unchanged': True, 'business_config_and_false_template_byte_identical': True, 'real_model_API_calls': 0, 'credentials_read': False, 'filesystem_writes': 0}, ensure_ascii=False, indent=2))

