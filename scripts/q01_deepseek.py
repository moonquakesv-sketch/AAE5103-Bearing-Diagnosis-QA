"""DeepSeek Chat Completions codec, exercised through offline fixtures only.

Official protocol: https://api-docs.deepseek.com/api/create-chat-completion/
Model routing: https://api-docs.deepseek.com/
No HTTP implementation, SDK, environment access or credential lookup is present.
The injected exchange is a trusted offline fixture, not a live transport. Token,
timeout, permission and cost gates must be resolved before any real wiring.
"""
import datetime
import json
import time

import q01_flow as flow
import q01_provider as contract
import retrieve_k02 as k02


VERSION = 'q01.deepseek.codec.v1'
OFFICIAL_ENDPOINT = 'https://api.deepseek.com/chat/completions'
API_MODEL = 'deepseek-flash'
SUPPORTED_MODELS = frozenset(('deepseek-flash', 'deepseek-v4-pro'))
DOCUMENTED_SERVED_IDENTITY = 'DeepSeek-V4.1-Flash-1017'
PARAMETERS = frozenset(('thinking', 'temperature', 'top_p', 'reasoning_effort'))
EFFORTS = frozenset(('none', 'low', 'high', 'max', 'minimal', 'medium', 'xhigh'))
ProviderError = contract.ProviderError


def _mock_config(config):
    config = flow.json_snapshot(config)
    if type(config) is not dict or set(config) != contract.CONFIG_FIELDS:
        raise ValueError('Exact DeepSeek codec configuration fields are required.')
    if config['endpoint'] != OFFICIAL_ENDPOINT:
        raise ValueError('Only the exact official DeepSeek Chat Completions endpoint is registered.')
    if type(config['provider_name']) is not str or config['provider_name'].casefold() != 'deepseek':
        raise ValueError('provider_name must identify DeepSeek.')
    mock = flow.json_snapshot(config)
    mock['endpoint'] = 'https://deepseek-codec.example.invalid/chat/completions'
    mock = contract.validate_config(mock)
    for action in contract.ACTIONS:
        if config['models'][action] not in SUPPORTED_MODELS:
            raise ValueError('Unsupported official DeepSeek API model identity.')
        if config['max_output_tokens'][action] > 393216:
            raise ValueError('max_output_tokens exceeds the documented max_tokens range.')
        params = config['parameters'][action]
        if not set(params) <= PARAMETERS:
            raise ValueError('Unsupported DeepSeek parameter; tools/search/fetch are forbidden.')
        if 'thinking' in params:
            value = params['thinking']
            if (type(value) is not dict or set(value) != {'type'}
                    or value['type'] not in ('enabled', 'disabled')):
                raise ValueError('thinking must contain exactly type=enabled or disabled.')
        for key, lower, upper, exclusive in (('temperature', 0, 2, False), ('top_p', 0, 1, True)):
            if key in params:
                value = params[key]
                if (type(value) not in (int, float) or value > upper
                        or value < lower or (exclusive and value == lower)):
                    raise ValueError('Invalid ' + key + ' range.')
        if 'reasoning_effort' in params:
            effort = params['reasoning_effort']
            if type(effort) is not str or effort not in EFFORTS:
                raise ValueError('Unsupported reasoning_effort value.')
            if 'thinking' in params and ((effort == 'none') != (params['thinking']['type'] == 'disabled')):
                raise ValueError('thinking and reasoning_effort specify inconsistent modes.')
    return config, mock


class DeepSeekChatProvider:
    """One Q01 flow using an injected offline official-protocol fixture.

    ``exchange(official_endpoint, detached_request_body, detached_settings)`` is
    invoked once per attempt. It returns the HTTP-like envelope used by the
    common contract: status_code/body and optional headers/request_id. This
    class does not send HTTP, read a key or approve a live budget.
    """
    def __init__(self, config, exchange, *, redact_values=()):
        self._config, mock = _mock_config(config)
        if not callable(exchange):
            raise TypeError('exchange must be a trusted offline fixture.')
        self._exchange = exchange
        self._official_attempts = []
        self._inner = contract.BoundedJSONProvider(mock, self._bridge, redact_values=redact_values)

    @property
    def identity(self):
        models = set(self._config['models'].values())
        return {'kind': 'provider_contract_mock', 'version': VERSION,
                'protocol': 'deepseek_chat_completions', 'transport': 'offline_fixture',
                'provider_name': 'DeepSeek', 'official_endpoint': OFFICIAL_ENDPOINT,
                'user_selected_model': next(iter(models)) if len(models) == 1 else 'per_action_config',
                'documented_served_identity': DOCUMENTED_SERVED_IDENTITY if models == {API_MODEL} else 'unknown',
                'actual_service_identity_verified': False,
                'models': flow.json_snapshot(self._config['models']),
                'configuration_sha256': flow.digest(self._config),
                'code_sha256': flow.file_hash(__file__),
                'common_code_sha256': flow.file_hash(contract.__file__),
                'network_enabled': False, 'actual_http_request_count': 0,
                'token_limits_enforced_for_real_service': False,
                'semantic_ability_demonstrated': False, 'billing': 'unknown'}

    @property
    def attempts(self):
        return flow.json_snapshot(self._official_attempts)

    @property
    def denials(self):
        return self._inner.denials

    def snapshot(self):
        normalized = self._inner.snapshot()
        return flow.json_snapshot({'identity': self.identity, 'config': self._config,
                                   'record_kind': 'provider_contract_mock',
                                   'transport': 'offline_fixture', 'actual_http_request_count': 0,
                                   'actual_attempt_count': len(self._official_attempts),
                                   'attempt_unit': 'offline_fixture_exchange_invocations',
                                   'action_counts': normalized['action_counts'],
                                   'attempts': self._official_attempts, 'denials': normalized['denials'],
                                   'normalized_contract': normalized,
                                   'actual_bill': 'unknown', 'estimated_cost': 'unknown'})

    def _encode(self, action, payload, settings):
        schema = ('Return JSON with exactly verdict (pass or issues) and issues '
                  '(empty list for pass, nonempty list of nonblank strings for issues).'
                  if action == 'verify' else
                  'Return JSON with exactly answer (nonblank English string), citations '
                  '(nonempty unique returned evidence-ID string list), and evidence_sufficiency '
                  '(sufficient, partial or insufficient).')
        data = {k: flow.json_snapshot(v) for k, v in payload.items()
                if k not in ('instructions', 'generation_settings')}
        body = {'model': settings['model'],
                'messages': [{'role': 'system', 'content': '\n\n'.join(payload['instructions']) + '\n\n' + schema},
                             {'role': 'user', 'content': flow.canonical(data)}],
                'stream': False, 'response_format': {'type': 'json_object'},
                'max_tokens': settings['max_output_tokens']}
        body.update(flow.json_snapshot(settings['parameters']))
        return body

    def _decode(self, response, row):
        try:
            response = flow.json_snapshot(response)
            contract._unicode(response)
        except Exception as error:
            row['raw_response'] = self._inner._redact(flow.unrecordable(response, error))
            raise ProviderError('unrecordable_response', self._inner._redact_text(flow.error_text(error))) from None
        row['raw_response'] = self._inner._redact(response)
        if (type(response) is not dict or not {'status_code', 'body'} <= set(response)
                or not set(response) <= {'status_code', 'body', 'headers', 'request_id'}
                or type(response['status_code']) is not int or type(response['body']) is not str):
            raise ProviderError('schema_mismatch', 'Invalid offline HTTP response envelope.')
        row['http_status'] = response['status_code']
        if 'request_id' in response:
            if type(response['request_id']) is not str:
                raise ProviderError('schema_mismatch', 'request_id must be a string when supplied.')
            row['request_id'] = self._inner._redact_text(response['request_id'])
        elif type(response.get('headers')) is dict:
            ids = [v for k, v in response['headers'].items() if k.casefold() == 'x-request-id']
            if len(ids) == 1 and type(ids[0]) is str:
                row['request_id'] = self._inner._redact_text(ids[0])
        if not 200 <= response['status_code'] < 300 or not response['body'].strip():
            return {k: v for k, v in response.items() if k != 'headers'}
        try:
            body = flow.json_snapshot(k02.strict_json(response['body']))
            contract._unicode(body)
        except Exception as error:
            raise ProviderError('invalid_json', self._inner._redact_text(flow.error_text(error))) from None
        if type(body) is not dict:
            raise ProviderError('schema_mismatch', 'Official response body must be an object.')
        normalized = {}
        for key, target, expected in (('id', 'response_id', str), ('model', 'returned_model', str),
                                      ('usage', 'token_usage', dict)):
            if key in body:
                if type(body[key]) is not expected:
                    raise ProviderError('schema_mismatch', 'Invalid response metadata field: ' + key)
                row[target] = self._inner._redact(body[key])
                normalized[key] = body[key]
        choices = body.get('choices')
        if type(choices) is not list or len(choices) != 1:
            raise ProviderError('schema_mismatch', 'Exactly one official response choice is required.')
        choice = choices[0]
        if (type(choice) is not dict or type(choice.get('index')) is not int or choice['index'] != 0
                or type(choice.get('message')) is not dict or choice['message'].get('role') != 'assistant'):
            raise ProviderError('schema_mismatch', 'Choice must have index0 and an assistant message.')
        message = choice['message']
        finish = choice.get('finish_reason')
        row['finish_reason'] = self._inner._redact_text(finish) if type(finish) is str else 'unknown'
        if finish not in ('stop', 'length', 'content_filter', 'tool_calls', 'insufficient_system_resource', 'aborted'):
            raise ProviderError('schema_mismatch', 'Unknown or missing finish_reason.')
        if finish in ('insufficient_system_resource', 'aborted'):
            raise ProviderError('service_error', 'Official generation interrupted: ' + finish)
        if ('tool_calls' in message and message['tool_calls'] is not None
                and type(message['tool_calls']) is not list):
            raise ProviderError('schema_mismatch', 'Invalid tool_calls field.')
        if finish == 'tool_calls' or message.get('tool_calls'):
            raise ProviderError('schema_mismatch', 'Tool calls are forbidden in this stage.')
        refusal = message.get('refusal')
        if refusal is not None and type(refusal) not in (str, bool):
            raise ProviderError('schema_mismatch', 'Invalid refusal field.')
        if finish == 'content_filter' or refusal:
            normalized['refusal'] = True
        elif finish == 'length':
            normalized['truncated'] = True
        else:
            if 'content' not in message:
                raise ProviderError('schema_mismatch', 'Assistant content field is missing.')
            if message['content'] is not None and type(message['content']) is not str:
                raise ProviderError('schema_mismatch', 'Assistant content must be a string or null.')
            normalized['output_text'] = message['content'] or ''
        result = {'status_code': response['status_code'],
                  'body': json.dumps(normalized, ensure_ascii=True, allow_nan=False)}
        if row['request_id'] != 'unknown':
            result['request_id'] = row['request_id']
        return result

    def _bridge(self, action, payload, settings):
        body = self._encode(action, payload, settings)
        row = {'attempt': len(self._official_attempts) + 1, 'action': action,
               'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'intended_official_endpoint': OFFICIAL_ENDPOINT, 'protocol_endpoint': OFFICIAL_ENDPOINT,
               'requested_model': settings['model'], 'returned_model': 'unknown',
               'request_id': 'unknown', 'response_id': 'unknown', 'token_usage': 'unknown',
               'actual_bill': 'unknown', 'estimated_cost': 'unknown', 'http_status': 'unknown',
               'raw_response': 'unknown', 'official_request': self._inner._redact(body),
               'official_request_sha256': flow.digest(body),
               'settings': flow.json_snapshot(settings), 'status': 'in_progress',
               'error_code': None, 'normalization_error_code': None,
               'parse_diagnostic': 'not_parsed', 'transport': 'offline_fixture',
               'http_request_sent': False, 'response_identity_origin': 'offline_fixture',
               'record_kind': 'provider_contract_mock'}
        self._official_attempts.append(row)
        start = time.perf_counter()
        try:
            try:
                response = self._exchange(OFFICIAL_ENDPOINT, flow.json_snapshot(body), flow.json_snapshot(settings))
            except TimeoutError as error:
                row.update(status='failed', error_code='timeout',
                           parse_diagnostic=self._inner._redact_text(flow.error_text(error)))
                raise
            except Exception as error:
                row.update(status='failed', error_code='transport_error',
                           parse_diagnostic=self._inner._redact_text(flow.error_text(error)))
                raise
            try:
                return self._decode(response, row)
            except ProviderError as error:
                row.update(status='failed', error_code=error.code, normalization_error_code=error.code,
                           parse_diagnostic=self._inner._redact_text(str(error)))
                raise
        finally:
            row['elapsed_seconds'] = time.perf_counter() - start

    def _call(self, action, payload):
        count = len(self._official_attempts)
        try:
            result = getattr(self._inner, action)(payload)
        except ProviderError as error:
            if len(self._official_attempts) > count:
                row = self._official_attempts[-1]
                row['common_boundary_error_code'] = error.code
                if row['normalization_error_code'] is not None:
                    # Preserve the unmodified common diagnostic too; do not
                    # forge an HTTP status merely to classify a codec failure.
                    raise ProviderError(row['normalization_error_code'], row['parse_diagnostic']) from None
                row.update(status='failed', error_code=error.code,
                           parse_diagnostic=self._inner._redact_text(str(error)))
            raise
        row = self._official_attempts[-1]
        row.update(status='success', parse_diagnostic='Official envelope normalized; strict JSON/action schema accepted',
                   output=flow.json_snapshot(result))
        return result

    def generate(self, payload):
        return self._call('generate', payload)

    def verify(self, payload):
        return self._call('verify', payload)

    def revise(self, payload):
        return self._call('revise', payload)
