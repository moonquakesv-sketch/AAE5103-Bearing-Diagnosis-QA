"""SYNTHETIC stop-contract regressions; never writes into the selected target."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys
import traceback
from decimal import Decimal

REPAIR = Path(__file__).absolute().parents[1]
PROJECT = REPAIR.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
TARGET = args.target.absolute()
OUT = args.output.absolute()
assert OUT.is_relative_to(REPAIR / 'validation') and not OUT.exists()
assert not any(p.is_symlink() for p in [OUT, *OUT.parents])
denied = []
approved_write_paths = []

def audit(event, values):
    if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system'}:
        denied.append(event)
        raise RuntimeError('SYNTHETIC suite prohibits network and subprocesses')
    if event == 'open':
        mode, flags = values[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing:
            if isinstance(values[0], int):
                descriptor = os.fstat(values[0])
                if not any(p.exists() and (p.stat().st_dev, p.stat().st_ino) == (descriptor.st_dev, descriptor.st_ino) for p in reversed(approved_write_paths)):
                    denied.append(event)
                    raise RuntimeError('SYNTHETIC fdopen must refer to a permitted output file')
                return
            path = Path(values[0]).absolute()
            if not path.is_relative_to(OUT):
                denied.append(event)
                raise RuntimeError('SYNTHETIC output must stay inside its unique new directory')
            approved_write_paths.append(path)
    if event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.symlink', 'os.link', 'os.chmod', 'os.utime'}:
        if not Path(values[0]).absolute().is_relative_to(OUT):
            denied.append(event)
            raise RuntimeError('SYNTHETIC mutation outside output directory')
sys.addaudithook(audit)
OUT.mkdir()
sys.path.insert(0, str(TARGET / 'runtime'))
from common import canonical, digest, read_json, Rejected
import provider as runtime_provider
import workflow
from provider import Counter, Ledger, Provider
from baseline import messages, generation_packet

live_attempts = []
def forbidden_live(*_args, **_kwargs):
    live_attempts.append(True)
    raise RuntimeError('SYNTHETIC suite prohibits live authorization and HTTP')
runtime_provider.LiveHTTP.__init__ = forbidden_live
runtime_provider.LiveGate.__init__ = forbidden_live
workflow.execute_authorized = forbidden_live
COUNTER = Counter(PROJECT, read_json(TARGET / 'config/counter_assets.json'))
BASE = read_json(TARGET / 'config/execution.json')
QUESTION = {'predicted_class': 'normal', 'scenario_id': 'SYNTHETIC-SCENARIO', 'question': 'SYNTHETIC: describe the fixture boundary.', 'task_types': ['inspection'], 'explicitly_allowed_operating_context': {}}
RUBRIC = {'points': [{'point_id': 'SYNTHETIC-P1', 'content_role': 'boundary', 'half_credit_eligible': False}]}
SUFFIXES = [('generate', 'generate'), ('A', 'judge_output'), ('B', 'judge_output'), ('C', 'adjudicate_output')]

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

class SyntheticTransport:
    def __init__(self, config, mode='normal'):
        self.config, self.mode, self.calls = config, mode, []
        self.judges = 0

    def __call__(self, body):
        self.calls.append(copy.deepcopy(body))
        role = next(k for k, v in self.config['actions'].items() if v['output_cap'] == body['max_tokens'])
        packet = json.loads(body['messages'][1]['content'])
        if role == 'generate':
            answer = 'SYNTHETIC visible fixture answer.'
            if self.mode == 'literal':
                answer += ' ｜DSML｜'
            value = {'answer': answer, 'citations': [], 'evidence_limitations': []}
        else:
            if role == 'judge_output':
                self.judges += 1
            core = packet['packet'] if role == 'adjudicate_output' else packet
            text = core['observed_text']
            value = {'points': [{'point_id': 'SYNTHETIC-P1', 'score': 1, 'quotes': [{'start': 0, 'end': len(text), 'text': text}], 'reason': 'SYNTHETIC fixture quote.'}], 'claims': [], 'voluntary_citations': [], 'complete_claim_inventory': True, 'complete_citation_inventory': True}
            if self.mode == 'invalid_rating_A' and role == 'judge_output' and self.judges == 1:
                value = {}
        envelope = {'model': self.config['model'], 'id': 'SYNTHETIC', 'choices': [{'finish_reason': 'length' if self.mode == 'truncated_generation' and role == 'generate' else 'stop', 'message': {'content': canonical(value)}}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        if self.mode == 'unknown_usage':
            del envelope['usage']
        if self.mode == 'alias':
            envelope['model'] = 'SYNTHETIC_DIFFERENT_ALIAS'
        if self.mode == 'usage_cap':
            n = self.config['actions'][role]['input_cap'] + 1
            envelope['usage'] = {'prompt_tokens': n, 'completion_tokens': 1, 'total_tokens': n + 1}
        return {'http_status': 200, 'request_id': 'SYNTHETIC_ONLY', 'raw_body': canonical(envelope).encode()}

def setup(name, adjust=None, mode='normal', units=1):
    case = OUT / name
    case.mkdir()
    config = copy.deepcopy(BASE)
    config['scope']['stage'] = 'SYNTHETIC'
    slots, bindings, keys = {}, {}, []
    for i in range(units + 1):
        key = 'SYNTHETIC-%02d' % i
        keys.append(key)
        for suffix, role in SUFFIXES:
            slot = key + '_' + suffix
            slots[slot] = role
            bindings[slot] = {'business_sha256': digest(QUESTION), 'flow_key': key}
    config['slot_bindings'] = bindings
    config['allowed_slots_sha256'] = digest(slots)
    if adjust:
        adjust(config)
    ledger = Ledger(case / 'batch', config, slots, case)
    transport = SyntheticTransport(config, mode)
    subject = Provider(config, COUNTER, ledger, transport, ledger.directory / 'raw')
    return case, config, ledger, transport, subject, keys

def observed(case, ledger, transport, result):
    value = {'stage': 'SYNTHETIC', 'transport_calls': len(transport.calls), 'output_caps': [b['max_tokens'] for b in transport.calls], 'ledger_broken': ledger.broken, 'counts': ledger.counts, 'reserved_USD': str(ledger.reserved), 'original_rating_statuses': [v['status'] for v in result['original_ratings']], 'final_status': result['final']['status'], 'metrics': result['metrics']}
    write(case / 'observation.json', value)

def assert_stopped(name, expected_calls, adjust=None, mode='normal', mutate=None, skipped=()):
    case, config, ledger, transport, subject, keys = setup(name, adjust, mode)
    try:
        if mutate:
            mutate(subject)
        result = workflow.cell(subject, keys[0], 'N-SYNTHETIC', QUESTION, RUBRIC, [], ledger.directory)
        observed(case, ledger, transport, result)
        assert ledger.broken, 'stop rejection did not poison the batch ledger'
        assert len(transport.calls) == expected_calls, 'a later slot reached transport after stop'
        assert result['final']['rating'] is None, 'stop must retain unknown rating'
        for i in skipped:
            assert result['original_ratings'][i]['status'] == 'not_attempted_batch_stopped'
            assert result['original_ratings'][i]['rating'] is None
        assert result['metrics']['REF_cov']['denominator'] == 1
        assert result['metrics']['REF_cov']['unknown'] == 1
        assert result['metrics']['REF_unsupported_claim']['denominator'] is None
        reserved_before, counts_before = ledger.reserved, dict(ledger.counts)
        try:
            subject.send(keys[1] + '_generate', 'generate', messages('generate', generation_packet(QUESTION)))
        except Rejected:
            pass
        else:
            raise AssertionError('next question was allowed after batch stop')
        assert len(transport.calls) == expected_calls
        assert ledger.reserved == reserved_before and ledger.counts == counts_before
        expected_reserve = sum((Decimal(config['actions'][role]['input_cap']) * Decimal(config['price']['input_per_million']) + Decimal(config['actions'][role]['output_cap']) * Decimal(config['price']['output_per_million'])) / Decimal(1000000) * count for role, count in ledger.counts.items())
        assert ledger.reserved == expected_reserve, 'already reserved amounts were changed'
        http_records = list((ledger.directory / 'raw').glob('*.http.json'))
        assert len(http_records) == expected_calls, 'raw response evidence missing'
    finally:
        ledger.close()

def input_cap(role):
    return lambda c: c['actions'][role].update(input_cap=1)

def test_generate_cap():
    assert_stopped('generate_cap', 0, input_cap('generate'), skipped=(0, 1, 2))

def test_judge_cap():
    assert_stopped('judge_cap', 1, input_cap('judge_output'), skipped=(1, 2))

def test_third_cap():
    assert_stopped('third_cap', 3, input_cap('adjudicate_output'))

def test_context_cap():
    assert_stopped('context_cap', 0, lambda c: c.update(context_limit=1), skipped=(0, 1, 2))

def test_counter_literal():
    assert_stopped('counter_literal', 1, mode='literal', skipped=(1, 2))

def test_business_binding():
    def adjust(c):
        c['slot_bindings']['SYNTHETIC-00_A']['business_sha256'] = '0' * 64
    assert_stopped('business_binding', 1, adjust, skipped=(1, 2))

def test_provider_config_drift():
    assert_stopped('provider_config_drift', 0, mutate=lambda p: p.config.update(timeout_seconds=121), skipped=(0, 1, 2))

def test_budget_rejection():
    assert_stopped('budget_rejection', 1, lambda c: c.update(proposed_hard_budget_USD='0.0073728'), skipped=(1, 2))

def test_unknown_usage():
    assert_stopped('unknown_usage', 1, mode='unknown_usage', skipped=(0, 1, 2))

def test_returned_alias():
    assert_stopped('returned_alias', 1, mode='alias', skipped=(0, 1, 2))

def test_usage_cap():
    assert_stopped('usage_cap', 1, mode='usage_cap', skipped=(0, 1, 2))

def soft_failure(name, mode, expected_first_status):
    case, config, ledger, transport, subject, keys = setup(name, mode=mode)
    try:
        result = workflow.cell(subject, keys[0], 'N-SYNTHETIC', QUESTION, RUBRIC, [], ledger.directory)
        observed(case, ledger, transport, result)
        assert not ledger.broken and len(transport.calls) == 4
        assert result['original_ratings'][0]['status'] == expected_first_status
        assert result['final']['status'] == 'resolved_third'
        assert len(list((ledger.directory / 'raw').glob('*.http.json'))) == 4
    finally:
        ledger.close()

def test_rating_format_failure_continues():
    soft_failure('rating_format_failure', 'invalid_rating_A', 'evaluator_failure')

def test_generation_truncation_continues():
    soft_failure('generation_truncation', 'truncated_generation', 'valid')

def test_normal_52_and_single_consumption():
    case, config, ledger, transport, subject, keys = setup('full52', units=13)
    try:
        for i in range(13):
            result = workflow.cell(subject, keys[i], 'N-SYNTHETIC-%02d' % i, QUESTION, RUBRIC, [], ledger.directory)
            assert result['final']['status'] == 'resolved_third'
            assert result['metrics']['REF_cov']['value'] == 1
        observed(case, ledger, transport, result)
        assert len(transport.calls) == 52
        assert ledger.counts == {'generate': 13, 'judge_output': 26, 'adjudicate_output': 13}
        assert ledger.reserved == Decimal('1.2460032')
        assert not ledger.broken
        request = read_json(ledger.directory / 'raw/SYNTHETIC-00_generate.request.json')
        assert request['body'] == transport.calls[0]
        assert request['count']['messages_sha256'] == digest(transport.calls[0]['messages'])
        before = ledger.reserved
        try:
            subject.send(keys[0] + '_generate', 'generate', messages('generate', generation_packet(QUESTION)))
        except Rejected:
            pass
        else:
            raise AssertionError('attempt slot was reused')
        assert len(transport.calls) == 52 and ledger.reserved == before
    finally:
        ledger.close()
    try:
        Ledger(case / 'batch', config, {}, case)
    except Rejected:
        pass
    else:
        raise AssertionError('batch directory was resumed')

def test_false_authorization_templates_unchanged():
    flags = ['formal_experiment_released', 'independent_cross_audit_accepted', 'paid_execution_authorized']
    template = read_json(TARGET / 'config/new_authorization_template.json')
    assert all(BASE[k] is False and template[k] is False for k in flags)
    assert template['user_authorization_record'] is None
    assert not Path(BASE['batch_output']).exists()

cases = [v for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
results = []
for case in cases:
    try:
        case()
        results.append({'case': case.__name__, 'status': 'PASS'})
    except Exception as exc:
        results.append({'case': case.__name__, 'status': 'FAIL', 'error_type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
assert not live_attempts and not denied
summary = {'stage': 'SYNTHETIC', 'target': str(TARGET), 'output': str(OUT), 'test_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'target_source_sha256': {str(p.relative_to(TARGET)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [TARGET / 'runtime/provider.py', TARGET / 'runtime/workflow.py']}, 'passed': sum(r['status'] == 'PASS' for r in results), 'total': len(results), 'results': results, 'real_model_API_calls': 0, 'credentials_read': False, 'LiveHTTP_constructed': False, 'execute_authorized_called': False, 'sealed_author_suite_run': False, 'audit_denied_operations': denied, 'outputs_are_new_unique_directories': True}
write(OUT / 'results.json', summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))
sys.exit(0 if summary['passed'] == summary['total'] else 1)

