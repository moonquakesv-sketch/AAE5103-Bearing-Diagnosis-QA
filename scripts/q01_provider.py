"""Bounded, provider-neutral JSON service contract for Q01-S02 offline development.

This module has no HTTP client and never reads credentials. ``exchange`` is a
trusted, in-process mock callable; its normalized response is NOT an official
provider protocol. A selected provider, official codec and approved hard budget
are still required before real wiring. It must not be used as a paid-service
adapter merely by changing the identity or supplying a networking callable.
"""
import datetime
import json
import re
import threading
import time
from urllib.parse import urlsplit

import q01_flow as flow
import retrieve_k02 as k02
import validate_k01 as k01


VERSION = 'q01.provider.contract.v1'
ACTIONS = ('generate', 'verify', 'revise')
ACTION_LIMITS = {'generate': 1, 'verify': 2, 'revise': 1}
CONFIG_FIELDS = frozenset(('provider_name', 'endpoint', 'models', 'parameters',
                           'max_attempts', 'timeout_seconds', 'max_input_tokens',
                           'max_output_tokens', 'retries', 'concurrency'))
AUTH_KEYS = frozenset(('authorization', 'proxyauthorization', 'xapikey', 'apikey',
                       'secret', 'clientsecret', 'token', 'accesstoken',
                       'refreshtoken', 'credential', 'credentials', 'password'))


class ProviderError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(code + ': ' + message)


def _auth_key(key):
    return re.sub(r'[^a-z]', '', key.lower()) in AUTH_KEYS


def _has_auth_key(value):
    if type(value) is dict:
        return any(_auth_key(k) or _has_auth_key(v) for k, v in value.items())
    if type(value) is list:
        return any(_has_auth_key(v) for v in value)
    return False


def _unicode(value):
    if type(value) is str:
        value.encode('utf-8')
    elif type(value) is dict:
        for key, child in value.items():
            _unicode(key)
            _unicode(child)
    elif type(value) is list:
        for child in value:
            _unicode(child)


def _candidate(value):
    if type(value) is not dict or set(value) != {'answer', 'citations', 'evidence_sufficiency'}:
        return False
    citations = value['citations']
    return (type(value['answer']) is str and bool(value['answer'].strip())
            and type(citations) is list and bool(citations)
            and all(type(c) is str and bool(c.strip()) for c in citations)
            and len(citations) == len(set(citations))
            and value['evidence_sufficiency'] in ('sufficient', 'partial', 'insufficient'))


def validate_config(config):
    """Validate only the offline mock contract, never grant live authorization."""
    config = flow.json_snapshot(config)
    _unicode(config)
    if type(config) is not dict or set(config) != CONFIG_FIELDS or _has_auth_key(config):
        raise ValueError('Exact credential-free provider contract configuration required.')
    if type(config['provider_name']) is not str or not config['provider_name'].strip():
        raise ValueError('A nonblank mock provider name is required.')
    if type(config['endpoint']) is not str:
        raise ValueError('Endpoint must be a string.')
    endpoint = urlsplit(config['endpoint'])
    if (endpoint.scheme != 'https' or not endpoint.hostname
            or not endpoint.hostname.endswith('.invalid') or endpoint.username
            or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError('Offline contract endpoint must be credential-free HTTPS under .invalid.')
    for field in ('models', 'parameters', 'max_input_tokens', 'max_output_tokens'):
        if type(config[field]) is not dict or set(config[field]) != set(ACTIONS):
            raise ValueError(field + ' must declare exactly generate, verify and revise.')
    for action in ACTIONS:
        if type(config['models'][action]) is not str or not config['models'][action].strip():
            raise ValueError('Each action needs an explicit mock model identity.')
        if type(config['parameters'][action]) is not dict:
            raise ValueError('Each action needs a JSON parameter object.')
        for field in ('max_input_tokens', 'max_output_tokens'):
            if type(config[field][action]) is not int or config[field][action] <= 0:
                raise ValueError(field + ' must contain positive integer ceilings.')
    if type(config['max_attempts']) is not int or not 1 <= config['max_attempts'] <= 5:
        raise ValueError('Total exchange attempt ceiling must be between 1 and 5.')
    if (type(config['timeout_seconds']) not in (int, float)
            or config['timeout_seconds'] <= 0):
        raise ValueError('Timeout must be positive and finite.')
    if type(config['retries']) is not int or config['retries'] != 0:
        raise ValueError('Implicit and explicit transport retries are forbidden.')
    if type(config['concurrency']) is not int or config['concurrency'] != 1:
        raise ValueError('Only serial exchange is allowed.')
    return config


class BoundedJSONProvider:
    """One flow, at most one mock exchange per method call; no internal retry.

    ``exchange(action, detached_payload, detached_settings)`` returns exactly
    ``status_code`` (integer), ``body`` (string), and optionally ``headers`` and
    ``request_id``. Successful body fields are ``output_text`` (required unless
    refused/truncated), and optional ``model``, ``id``, ``usage``, ``refusal`` and
    ``truncated``. ``output_text`` is strict JSON, never fenced or repaired.

    Timeout/token controls are passed to the mock. Their presence does not
    establish a real tokenizer, deadline, price or provider-side hard limit.
    Instances cannot reset their limits. T and GV would require a separately
    approved aggregate session gate before real use.
    """
    def __init__(self, config, exchange, *, redact_values=()):
        self._config = validate_config(config)
        if not callable(exchange):
            raise TypeError('exchange must be a trusted mock callable.')
        if type(redact_values) not in (tuple, list) or any(type(v) is not str or not v for v in redact_values):
            raise TypeError('redact_values must contain only nonempty mock strings.')
        self._redact_values = tuple(redact_values)
        if self._redact(self._config) != self._config:
            raise ValueError('Credential material is forbidden in provider configuration.')
        self._exchange = exchange
        self._attempts = []
        self._denials = []
        self._counts = dict.fromkeys(ACTIONS, 0)
        self._lock = threading.Lock()

    def _redact_text(self, value):
        # Decode JSON containers for redaction only, including escaped/nested
        # credential echoes. Leave the exact original string when unchanged.
        try:
            decoded = flow.json_snapshot(k02.strict_json(value))
            if type(decoded) in (dict, list):
                cleaned = self._redact(decoded)
                if cleaned != decoded:
                    value = json.dumps(cleaned, ensure_ascii=True, separators=(',', ':'))
        except Exception:
            pass
        for secret in self._redact_values:
            for variant in (secret, json.dumps(secret, ensure_ascii=True)[1:-1],
                            json.dumps(secret, ensure_ascii=False)[1:-1]):
                value = value.replace(variant, '<redacted>')
        value = re.sub(r'(?i)\bBearer\s+[^\s\"\'<> ,;\}]+', 'Bearer <redacted>', value)
        value = re.sub(r'(?i)((?:authorization|proxy.authorization|x.api.key|api.key|access.token|refresh.token|password|client.secret|credentials?)\s*[\"\']?\s*[:=]\s*[\"\']?)[^\r\n\"\',;\}]+',
                       r'\1<redacted>', value)
        return value

    def _redact(self, value):
        if type(value) is str:
            return self._redact_text(value)
        if type(value) is list:
            return [self._redact(v) for v in value]
        if type(value) is dict:
            return {self._redact_text(k): self._redact(v) for k, v in value.items()
                    if not _auth_key(k) and k.lower() != 'headers'}
        return value

    @property
    def identity(self):
        return {'kind': 'provider_contract_mock', 'version': VERSION,
                'provider_name': self._config['provider_name'],
                'endpoint': self._config['endpoint'],
                'models': flow.json_snapshot(self._config['models']),
                'configuration_sha256': flow.digest(self._config),
                'code_sha256': flow.file_hash(__file__),
                'network_enabled': False, 'token_limits_enforced_for_real_service': False,
                'semantic_ability_demonstrated': False, 'billing': 'unknown'}

    @property
    def attempts(self):
        return flow.json_snapshot(self._attempts)

    @property
    def denials(self):
        return flow.json_snapshot(self._denials)

    def snapshot(self):
        return flow.json_snapshot({'identity': self.identity, 'config': self._config,
                                   'attempts': self._attempts, 'denials': self._denials,
                                   'actual_attempt_count': len(self._attempts),
                                   'action_counts': self._counts,
                                   'actual_bill': 'unknown', 'estimated_cost': 'unknown',
                                   'record_kind': 'provider_contract_mock'})

    def _payload(self, action, value):
        value = flow.json_snapshot(value)
        _unicode(value)
        required = {'instructions', 'request', 'scenario_context', 'evidence_cards', 'generation_settings'}
        if action in ('verify', 'revise'):
            required.add('candidate')
        if action == 'revise':
            required.add('issue_report')
        if type(value) is not dict or set(value) != required:
            raise ValueError('Unexpected provider payload fields.')
        k02.validate_request(value['request'])
        context = value['scenario_context']
        if (type(context) is not dict or not set(context) <= set(flow.SCENARIO_FIELDS)
                or any(type(v) is not str for v in context.values())):
            raise ValueError('Only registered scenario context fields are allowed.')
        instructions = value['instructions']
        if (type(instructions) is not list or not instructions
                or any(type(v) is not str or not v.strip() for v in instructions)
                or type(value['generation_settings']) is not dict):
            raise ValueError('Invalid instructions or generation settings.')
        cards = value['evidence_cards']
        fields = set(k01.FIELDS['evidence'].split())
        source_fields = set(k01.FIELDS['sources'].split())
        if type(cards) is not list or not 1 <= len(cards) <= 3:
            raise ValueError('One to three complete returned cards are required.')
        for card in cards:
            if (type(card) is not dict or set(card) != fields | {'source'}
                    or any(type(card[f]) is not str for f in fields)
                    or type(card['source']) is not dict or set(card['source']) != source_fields
                    or any(type(v) is not str for v in card['source'].values())):
                raise ValueError('Evidence/source fields must be complete and exact; diagnostics are excluded.')
        if len({c['evidence_id'] for c in cards}) != len(cards):
            raise ValueError('Duplicate evidence cards.')
        if action in ('verify', 'revise') and not _candidate(value['candidate']):
            raise ValueError('Invalid candidate in derived payload.')
        if action == 'revise' and not flow.valid_report(value['issue_report']):
            raise ValueError('Invalid issue report in derived payload.')
        if self._redact(value) != value:
            raise ValueError('Authentication material is forbidden in business payloads.')
        return value

    def _parse(self, response, event):
        try:
            response = flow.json_snapshot(response)
            _unicode(response)
        except Exception as error:
            event['raw_response'] = self._redact(flow.unrecordable(response, error))
            raise ProviderError('unrecordable_response', self._redact_text(flow.error_text(error))) from None
        event['raw_response'] = self._redact(response)
        if (type(response) is not dict or not {'status_code', 'body'} <= set(response)
                or not set(response) <= {'status_code', 'body', 'headers', 'request_id'}
                or type(response['status_code']) is not int or type(response['body']) is not str):
            raise ProviderError('schema_mismatch', 'Invalid normalized mock HTTP envelope.')
        event['http_status'] = response['status_code']
        if type(response.get('request_id')) is str:
            event['request_id'] = self._redact_text(response['request_id'])
        status = response['status_code']
        if status == 429:
            raise ProviderError('rate_limit', 'Mock service returned HTTP 429; no retry.')
        if status in (408, 504):
            raise ProviderError('timeout', 'Mock service returned a timeout HTTP status; no retry.')
        if not 200 <= status < 300:
            raise ProviderError('service_error', 'Mock service returned HTTP ' + str(status) + '; no retry.')
        if not response['body'].strip():
            raise ProviderError('empty_output', 'Empty response body.')
        try:
            body = flow.json_snapshot(k02.strict_json(response['body']))
            _unicode(body)
        except Exception as error:
            raise ProviderError('invalid_json', self._redact_text(flow.error_text(error))) from None
        if (type(body) is not dict or not set(body) <= {'output_text', 'model', 'id', 'usage', 'refusal', 'truncated'}
                or any(k in body and type(body[k]) is not str for k in ('model', 'id'))
                or any(k in body and type(body[k]) is not bool for k in ('refusal', 'truncated'))
                or ('usage' in body and type(body['usage']) is not dict)):
            raise ProviderError('schema_mismatch', 'Invalid normalized mock service body.')
        for key, target in (('model', 'returned_model'), ('id', 'response_id'), ('usage', 'token_usage')):
            if key in body:
                event[target] = self._redact(body[key])
        if body.get('refusal'):
            raise ProviderError('refusal', 'Mock service refused the request.')
        if body.get('truncated'):
            raise ProviderError('truncated', 'Mock service output was truncated; no repair or retry.')
        if 'output_text' not in body or type(body['output_text']) is not str:
            raise ProviderError('schema_mismatch', 'output_text must be present as a string.')
        if not body['output_text'].strip():
            raise ProviderError('empty_output', 'Empty generated output.')
        try:
            output = flow.json_snapshot(k02.strict_json(body['output_text']))
            _unicode(output)
        except Exception as error:
            raise ProviderError('invalid_json', self._redact_text(flow.error_text(error))) from None
        if self._redact(output) != output:
            raise ProviderError('credential_echo', 'Authentication material in generated output; blocked.')
        valid = flow.valid_report(output) if event['action'] == 'verify' else _candidate(output)
        if not valid:
            raise ProviderError('schema_mismatch', 'Generated JSON does not match the exact action schema.')
        return output

    def _deny(self, action, code, reason):
        self._denials.append({'action': action, 'code': code, 'reason': self._redact_text(reason),
                              'actual_attempt_count': len(self._attempts), 'exchange_invoked': False})
        raise ProviderError(code, self._redact_text(reason))

    def _call(self, action, payload):
        if not self._lock.acquire(blocking=False):
            self._deny(action, 'concurrency_forbidden', 'An exchange is already in progress.')
        try:
            try:
                payload = self._payload(action, payload)
            except Exception as error:
                self._deny(action, 'invalid_payload', flow.error_text(error))
            if len(self._attempts) >= self._config['max_attempts'] or self._counts[action] >= ACTION_LIMITS[action]:
                self._deny(action, 'budget_exhausted', 'Total or action attempt budget exhausted; no exchange.')
            settings = {'model': self._config['models'][action],
                        'parameters': flow.json_snapshot(self._config['parameters'][action]),
                        'max_input_tokens': self._config['max_input_tokens'][action],
                        'max_output_tokens': self._config['max_output_tokens'][action],
                        'timeout_seconds': self._config['timeout_seconds'], 'retries': 0, 'concurrency': 1}
            event = {'attempt': len(self._attempts) + 1, 'action': action,
                     'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     'requested_endpoint': self._config['endpoint'],
                     'requested_model': settings['model'], 'returned_model': 'unknown',
                     'response_id': 'unknown', 'request_id': 'unknown', 'token_usage': 'unknown',
                     'actual_bill': 'unknown', 'estimated_cost': 'unknown',
                     'http_status': 'unknown', 'raw_response': 'unknown',
                     'payload': flow.json_snapshot(payload), 'payload_sha256': flow.digest(payload),
                     'settings': flow.json_snapshot(settings), 'parse_diagnostic': 'not_parsed',
                     'status': 'in_progress', 'error_code': None, 'exchange_invoked': True,
                     'record_kind': 'provider_contract_mock'}
            self._attempts.append(event)
            self._counts[action] += 1
            start = time.perf_counter()
            try:
                try:
                    response = self._exchange(action, flow.json_snapshot(payload), flow.json_snapshot(settings))
                except TimeoutError as error:
                    raise ProviderError('timeout', self._redact_text(flow.error_text(error))) from None
                except Exception as error:
                    raise ProviderError('transport_error', self._redact_text(flow.error_text(error))) from None
                output = self._parse(response, event)
                event.update(status='success', parse_diagnostic='strict JSON and exact action schema accepted',
                             output=flow.json_snapshot(output))
                return flow.json_snapshot(output)
            except ProviderError as error:
                event.update(status='failed', error_code=error.code,
                             parse_diagnostic=self._redact_text(str(error)))
                raise
            finally:
                event['elapsed_seconds'] = time.perf_counter() - start
        finally:
            self._lock.release()

    def generate(self, payload):
        return self._call('generate', payload)

    def verify(self, payload):
        return self._call('verify', payload)

    def revise(self, payload):
        return self._call('revise', payload)
