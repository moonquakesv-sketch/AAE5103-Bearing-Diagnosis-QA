"""Q01-S01 offline orchestration; imports never read evaluation data or keys.

Provider contract: generate(payload), verify(payload), revise(payload) -> JSON
object. The default remains offline; an explicit runtime binds separate prompts
and settings to the provider identity. This module never opens a network socket.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys
import time

import retrieve_k02 as k02
import validate_k01 as k01

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'q01.flow.v1_s02'
EMPTY = frozenset(('empty_question', 'knowledge_not_covered', 'no_valid_terms',
                   'no_vocabulary_terms', 'no_candidates', 'zero_similarity'))
SUCCESS = frozenset(('retrieved', 'partial_task_coverage'))
COUNTS = ('retrieval', 'generate', 'basic_check', 'verify', 'revise')
SCENARIO_FIELDS = ('class_semantics', 'assumed_cause', 'assumption_text',
                   'known_conditions', 'unknown_information', 'applicability', 'cannot_infer')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()


def json_snapshot(value):
    """Accept a native JSON tree, then detach it without injected deepcopy hooks."""
    active = set()

    def check(item):
        kind = type(item)
        if item is None or kind in (str, bool, int):
            return
        if kind is float:
            if not math.isfinite(item):
                raise ValueError('Non-finite JSON number.')
            return
        if kind not in (dict, list):
            raise TypeError('Expected native JSON values; got ' + kind.__name__ + '.')
        if id(item) in active:
            raise ValueError('Circular JSON value.')
        active.add(id(item))
        try:
            if kind is dict:
                if any(type(key) is not str for key in item):
                    raise TypeError('JSON object keys must be strings.')
                for child in item.values():
                    check(child)
            else:
                for child in item:
                    check(child)
        finally:
            active.remove(id(item))

    check(value)
    return json.loads(canonical(value))


def error_text(error):
    try:
        message = str.__getitem__(str(error), slice(512))
    except Exception:
        message = '<message unavailable>'
    return type(error).__name__ + ': ' + message


def unrecordable(value, error):
    try:
        representation = str.__getitem__(repr(value), slice(512))
    except Exception:
        representation = '<representation unavailable>'
    return {'unserializable_type': type(value).__name__,
            'representation': representation, 'error': error_text(error)}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path=None):
    return k02.strict_json(Path(path or ROOT / 'configs/q01_offline.json').read_bytes())


def group_config(group, config=None):
    config = load_config() if config is None else config
    return dict(copy.deepcopy(config['shared']), **copy.deepcopy(config['groups'][group]), group_id=group)


def validate_config(config):
    expected = group_config(config.get('group_id'))
    if canonical(config) != canonical(expected):
        raise ValueError('Only registered offline group configurations are accepted.')


class ScriptedProvider:
    """Fixed queues; no text scoring, task inference or hidden retry/rewrite."""
    identity = {'kind': 'scripted_offline', 'semantic_ability_demonstrated': False,
                'billing': 'not_applicable_offline'}

    def __init__(self, script):
        self.script = copy.deepcopy(script)
        self.calls = []
        self.identity = dict(self.identity, script_sha256=digest(script))

    def _call(self, action, payload):
        self.calls.append({'action': action, 'payload': copy.deepcopy(payload)})
        queue = self.script.get(action, [])
        if not queue:
            raise RuntimeError('No scripted output for ' + action)
        result = queue.pop(0)
        if type(result) is dict and set(result) == {'raise'}:
            raise RuntimeError(str(result['raise']))
        return copy.deepcopy(result)

    def generate(self, payload):
        return self._call('generate', payload)

    def verify(self, payload):
        return self._call('verify', payload)

    def revise(self, payload):
        return self._call('revise', payload)


def basic_check(candidate, evidence_ids):
    """Syntax/schema and ID membership only; no semantic support assertion."""
    result = {'ok': False, 'error': 'schema_failed', 'reason': None,
              'semantic_support': 'not_assessed'}
    fields = {'answer', 'citations', 'evidence_sufficiency'}
    if type(candidate) is not dict or set(candidate) != fields:
        result['reason'] = 'Candidate must have exactly the three answer fields.'
        return result
    answer, citations = candidate['answer'], candidate['citations']
    if (type(answer) is not str or not answer.strip() or type(citations) is not list
            or not citations or not all(type(x) is str for x in citations)
            or len(citations) != len(set(citations))
            or candidate['evidence_sufficiency'] not in ('sufficient', 'partial', 'insufficient')):
        result['reason'] = 'Invalid answer schema or empty/duplicate citation list.'
        return result
    try:
        answer.encode('utf-8')
        canonical(candidate)
    except (ValueError, TypeError, UnicodeError):
        result['reason'] = 'Candidate is not valid Unicode/JSON.'
        return result
    # Bracketed inline IDs are optional, but any present must also be declared.
    inline = re.findall(r'\[([^\[\]\n]+)\]', answer)
    if not set(citations) <= set(evidence_ids) or not set(inline) <= set(citations):
        result.update(error='citation_failed', reason='Citation absent from this returned set or undeclared inline citation.')
        return result
    result.update(ok=True, error=None, reason='Schema and returned-ID membership only.')
    return result


def valid_report(report):
    return (type(report) is dict and set(report) == {'verdict', 'issues'}
            and report['verdict'] in ('pass', 'issues') and type(report['issues']) is list
            and all(type(x) is str and x.strip() for x in report['issues'])
            and ((report['verdict'] == 'pass' and not report['issues'])
                 or (report['verdict'] == 'issues' and bool(report['issues']))))


def evidence_payload(retrieved):
    """Project only registered card fields; never copy retrieval-side metadata."""
    cards = []
    for card in retrieved['results']:
        fields = k01.FIELDS['evidence'].split()
        source_fields = k01.FIELDS['sources'].split()
        if not all(type(card.get(f)) is str for f in fields):
            raise ValueError('Incomplete evidence card.')
        if type(card.get('source')) is not dict or not all(type(card['source'].get(f)) is str for f in source_fields):
            raise ValueError('Incomplete source registration.')
        cards.append(dict({f: card[f] for f in fields},
                          source={f: card['source'][f] for f in source_fields}))
    return cards


def run_flow(request, config, provider, retriever=None, runtime=None):
    """No writes. Callers persist the diagnostic record separately from final text."""
    started = time.perf_counter()
    record = {'version': VERSION, 'request_hash': None, 'request': None,
              'config': None, 'publishable': False, 'final_answer': None,
              'service_message': None, 'final_output_kind': 'no_answer',
              'status': None, 'reason': None, 'draft': None, 'revisions': [],
              'revision_count': 0, 'base_checks': [], 'verification': [],
              'content_verification_state': 'not_performed', 'semantic_support': 'not_established',
              'retrieval': None, 'returned_cards': [], 'provider_trace': [],
              'counts': dict.fromkeys(COUNTS, 0), 'timings_seconds': {k: None for k in COUNTS},
              'metrics': {'token_usage': 'unknown', 'cost': 'not_applicable_offline',
                          'billing_note': 'No real provider bill or tokenizer was supplied.'},
              'identity': {'code_sha256': file_hash(__file__), 'k02_code_sha256': file_hash(k02.__file__),
                           'k01_validator_sha256': file_hash(k01.__file__),
                           'config_file_sha256': file_hash(ROOT / 'configs/q01_offline.json'),
                           'expected_k02_knowledge_hashes': copy.deepcopy(k02.KNOWLEDGE_HASHES),
                           'knowledge_binding_note': 'Expected pinned K02 corpus; actual retrieval identity recorded after retrieval.',
                           'provider': None}}

    config_error, identity_error = None, None
    runtime_kind = None
    try:
        record['config'] = json_snapshot(config)
    except Exception as error:
        config_error = error
        record['config'] = unrecordable(config, error)
    provider_identity = None
    try:
        try:
            provider_identity = getattr(provider, 'identity')
        except AttributeError:
            missing = object()
            if inspect.getattr_static(provider, 'identity', missing) is not missing:
                raise
            provider_identity = {'kind': 'injected_test_double'}
        record['identity']['provider'] = json_snapshot(provider_identity)
    except Exception as error:
        identity_error = error
        record['identity']['provider'] = unrecordable(provider_identity, error)

    def finish(status, reason, final=None):
        record.update(status=status, reason=reason, publishable=final is not None, final_answer=final,
                      final_output_kind='answer' if final is not None else 'no_answer',
                      elapsed_seconds=time.perf_counter() - started)
        return json_snapshot(record)

    def call(stage, function, *args, **kwargs):
        record['counts'][stage] += 1
        stamp = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - stamp
            record['timings_seconds'][stage] = (record['timings_seconds'][stage] or 0.0) + elapsed

    def provider_call(action, payload):
        event = {'action': action, 'payload': copy.deepcopy(payload), 'payload_sha256': digest(payload)}
        record['provider_trace'].append(event)
        try:
            raw_value = call(action, getattr(provider, action), copy.deepcopy(payload))
            # Keep malformed/missing-field outputs for diagnosis if JSON-compatible.
            try:
                value = json_snapshot(raw_value)
            except Exception as error:
                event['output'] = unrecordable(raw_value, error)
                if action == 'generate':
                    record['draft'] = copy.deepcopy(event['output'])
                raise
            event['output'] = copy.deepcopy(value)
            return value
        except Exception as error:
            event['error'] = error_text(error)
            raise

    try:
        record['request_hash'] = digest(request)
        k02.validate_request(request)
    except Exception as error:
        return finish('invalid_input', getattr(error, 'code', 'invalid_request'))
    record['request'] = copy.deepcopy(request)
    try:
        if config_error is not None:
            raise config_error
        validate_config(config)
        record['identity']['configuration_id'] = digest(config)
        generation_settings = copy.deepcopy(config['generation_settings'])
        if runtime is None:
            if (type(provider_identity) is dict and provider_identity.get('kind') in
                    ('real_provider', 'provider_contract_mock')):
                raise ValueError('This provider requires an explicit runtime identity and prompts.')
            prompt_path = ROOT / config['prompts']
            prompts = k02.strict_json(prompt_path.read_bytes())
            record['identity']['prompts_sha256'] = file_hash(prompt_path)
        else:
            runtime = json_snapshot(runtime)
            if (type(runtime) is not dict or set(runtime) !=
                    {'record_kind', 'prompts', 'generation_settings'}):
                raise ValueError('Runtime must contain exactly the registered provider fields.')
            runtime_kind = runtime['record_kind']
            if (runtime_kind not in ('provider_contract_mock', 'real_provider') or
                    type(provider_identity) is not dict or
                    provider_identity.get('kind') != runtime_kind):
                raise ValueError('Runtime and provider identity must agree.')
            prompts = runtime['prompts']
            if (type(prompts) is not dict or set(prompts) !=
                    {'version', 'common', 'generate', 'verify', 'revise'} or
                    not all(type(v) is str and v.strip() for v in prompts.values())):
                raise ValueError('Runtime prompts must be the complete explicit prompt contract.')
            generation_settings = runtime['generation_settings']
            if type(generation_settings) is not dict or not generation_settings:
                raise ValueError('Explicit provider generation settings are required.')
            record['runtime'] = runtime
            record['identity']['runtime_sha256'] = digest(runtime)
            record['identity']['prompts_sha256'] = digest(prompts)
            record['identity']['prompts_hash_kind'] = 'canonical_json_sha256'
            if runtime_kind == 'real_provider':
                record['metrics'] = {'token_usage': 'unknown', 'cost': 'unknown',
                                     'estimated_cost': 'unknown', 'actual_bill': 'unknown',
                                     'billing_note': 'Provider usage is telemetry; no actual bill was supplied.'}
    except Exception as error:
        return finish('invalid_config', error_text(error))
    if identity_error is not None:
        return finish('input_or_setup_failed', error_text(identity_error))
    try:
        # Constructor errors count as the one retrieval attempt, too.
        def retrieve():
            engine = retriever if retriever is not None else k02.Retriever()
            return engine.retrieve(copy.deepcopy(request), mode=config['retrieval_mode'], top_k=3)
        retrieved = call('retrieval', retrieve)
        try:
            retrieved = json_snapshot(retrieved)
        except Exception as error:
            record['retrieval'] = unrecordable(retrieved, error)
            raise
        record['retrieval'] = copy.deepcopy(retrieved)
        if type(retrieved) is not dict or retrieved.get('status') not in EMPTY | SUCCESS:
            raise ValueError('Retriever returned an unknown/failure status.')
        cards = retrieved.get('results')
        if type(cards) is not list or len(cards) > 3:
            raise ValueError('Invalid complete-card budget.')
        if (retrieved['status'] in EMPTY) != (not cards):
            raise ValueError('Retriever status and returned cards disagree.')
        ids = [c['evidence_id'] for c in cards]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate evidence IDs.')
        record['returned_cards'] = copy.deepcopy(cards)
        record['identity']['retrieval_identity'] = {k: copy.deepcopy(retrieved.get(k, 'unknown')) for k in
            ('implementation_version', 'knowledge_version', 'configuration_id', 'index_id', 'config_file_sha256')}
        if not cards:
            status = 'knowledge_not_covered' if retrieved['status'] == 'knowledge_not_covered' else 'evidence_insufficient'
            record['service_message'] = 'No answer was generated: the current returned evidence does not cover this request.'
            return finish(status, retrieved.get('reason') or retrieved['status'])
        prompt_cards = evidence_payload(retrieved)
        scenario = retrieved.get('information_separation', {}).get('demo_scenario') or {}
        context = {k: scenario[k] for k in SCENARIO_FIELDS if k in scenario}
        payload = {'instructions': [prompts['common'], prompts['generate']],
                   'request': copy.deepcopy(request), 'scenario_context': context, 'evidence_cards': prompt_cards,
                   'generation_settings': copy.deepcopy(generation_settings)}
        record['payload_size'] = {'characters': len(json.dumps(payload, ensure_ascii=False)),
                                  'utf8_bytes': len(json.dumps(payload, ensure_ascii=False).encode('utf-8')),
                                  'tokens': 'unknown', 'equal_text_length_claimed': False}
    except Exception as error:
        return finish('retrieval_failed', error_text(error))
    try:
        candidate = provider_call('generate', payload)
        record['draft'] = copy.deepcopy(candidate)
    except Exception as error:
        return finish('generation_failed', error_text(error))

    def check(value):
        result = call('basic_check', basic_check, value, ids)
        record['base_checks'].append(result)
        return result

    checked = check(candidate)
    if not checked['ok']:
        return finish(checked['error'], checked['reason'])
    if config['content_verification']:
        for iteration in range(2):
            verify_payload = dict(payload, instructions=[prompts['common'], prompts['verify']],
                                  candidate=copy.deepcopy(candidate))
            try:
                report = provider_call('verify', verify_payload)
                record['verification'].append(copy.deepcopy(report))
                if not valid_report(report):
                    raise ValueError('Invalid verification schema; provider pass is not a program decision.')
            except Exception as error:
                record['content_verification_state'] = 'failed'
                return finish('verification_failed', error_text(error))
            if report['verdict'] == 'pass':
                record['content_verification_state'] = (
                    'model_reported_no_issues' if runtime_kind == 'real_provider' else
                    'mock_service_no_issues' if runtime_kind == 'provider_contract_mock' else
                    'scripted_no_issues')
                break
            record['content_verification_state'] = (
                'model_reported_issues' if runtime_kind == 'real_provider' else
                'mock_service_issues' if runtime_kind == 'provider_contract_mock' else
                'scripted_issues')
            if iteration == 1 or record['revision_count'] >= config['max_revision']:
                return finish('content_issues_remaining', 'Provider issues remain after the single allowed revision.')
            revise_payload = dict(payload, instructions=[prompts['common'], prompts['revise']],
                                  candidate=copy.deepcopy(candidate), issue_report=copy.deepcopy(report))
            record['revision_count'] += 1
            try:
                candidate = provider_call('revise', revise_payload)
                record['revisions'].append(copy.deepcopy(candidate))
            except Exception as error:
                return finish('revision_failed', error_text(error))
            checked = check(candidate)
            if not checked['ok']:
                return finish(checked['error'], checked['reason'])
    verified_status = ('completed_verified_model_report' if runtime_kind == 'real_provider' else
                       'completed_verified_contract_mock' if runtime_kind == 'provider_contract_mock' else
                       'completed_verified_synthetic')
    status = verified_status if config['content_verification'] else 'completed_unverified'
    if candidate['evidence_sufficiency'] != 'sufficient' or retrieved['status'] == 'partial_task_coverage':
        status = 'evidence_insufficient'
    if candidate['evidence_sufficiency'] == 'insufficient':
        record['service_message'] = 'The candidate reports insufficient evidence; no answer is published.'
        return finish(status, 'Candidate reports insufficient evidence; retained only for diagnosis.')
    if retrieved['status'] == 'partial_task_coverage':
        record['service_message'] = 'Some requested tasks have no returned evidence; only a limited candidate is published.'
        return finish(status, 'Task coverage remains partial; provider sufficiency does not remove this limitation.', candidate['answer'])
    return finish(status, 'Flow accepted; semantic truth and real model quality are unproven.', candidate['answer'])


def write_record(path, record):
    blob = (json.dumps(record, ensure_ascii=True, indent=2, allow_nan=False) + '\n').encode('ascii')
    with Path(path).open('xb') as stream:
        stream.write(blob)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--group', choices=('T', 'TV', 'G', 'GV'), required=True)
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        # Reserve before reading inputs, constructing retrieval or calling provider.
        with args.output.open('xb') as stream:
            request_bytes, request = None, None
            try:
                request_bytes = args.request.read_bytes()
                request = k02.strict_json(request_bytes)
                script = k02.strict_json(args.script.read_bytes())
                record = run_flow(request, group_config(args.group), ScriptedProvider(script))
                record['request_file_sha256'] = hashlib.sha256(request_bytes).hexdigest()
            except Exception as error:
                # Produce the same diagnostic envelope even when parsing/setup fails.
                record = run_flow(None, group_config(args.group), ScriptedProvider({}))
                record.update(status='input_or_setup_failed', request_hash='unknown_unparseable_or_unreadable',
                              error_code=getattr(error, 'code', 'invalid_file_or_json'),
                              reason=type(error).__name__ + ': ' + str(error))
                if request_bytes is not None:
                    record['request_file_sha256'] = hashlib.sha256(request_bytes).hexdigest()
                else:
                    record['request_file_sha256'] = 'unknown_unreadable'
                if request is not None:
                    record['request_hash'] = digest(request)
                record['identity']['prompts_sha256'] = file_hash(ROOT / 'configs/q01_prompts.json')
            stream.write((json.dumps(record, ensure_ascii=True, indent=2, allow_nan=False) + '\n').encode('ascii'))
        print(json.dumps({'status': record['status'], 'publishable': record['publishable'], 'output': str(args.output)}, ensure_ascii=True))
        return 0 if record['publishable'] else 2
    except OSError as error:
        print(json.dumps({'status': 'output_rejected', 'reason': str(error)}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
