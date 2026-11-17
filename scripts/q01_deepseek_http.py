"""Single-post DeepSeek HTTP transport with a durable shared development budget.

No work happens at import. Real use fails closed without an independently
registered, artifact-bound exact chat counter. Tests inject httpx.MockTransport
and an explicit synthetic credential; they never read the credential environment.
"""
import datetime
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import stat
import threading
import time

import httpx

import q01_deepseek as codec
import q01_flow as flow
import q01_provider as contract
import retrieve_k02 as k02


VERSION = 'q01.deepseek.http.v1'
ROOT = Path(__file__).resolve().parents[1]
ProviderError = contract.ProviderError
GROUP_LIMITS = {'T': {'generate': 1, 'verify': 0, 'revise': 0},
                'GV': {'generate': 1, 'verify': 2, 'revise': 1}}
TOTAL_LIMITS = {'generate': 2, 'verify': 2, 'revise': 1}
REQUEST_PATH = ROOT / 'runs/k02/20261024_135147_4d6ea8a8/inputs/inspection.json'
REQUEST_SHA256 = '6dabcfc59eadead7060f848eb9b0e8aa752b8dded2a3da019a80d9d4fec55854'
CODE_PATHS = (Path(flow.__file__), Path(contract.__file__), Path(codec.__file__), Path(__file__))
SCHEMAS = {'candidate': ['answer', 'citations', 'evidence_sufficiency'], 'report': ['verdict', 'issues']}


def _sha(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _decimal(value):
    if type(value) is not str:
        raise ValueError('Budget and prices must be explicit Decimal strings.')
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise ValueError('Invalid decimal budget or price.') from None
    if not result.is_finite() or result < 0:
        raise ValueError('Budget and prices must be finite and nonnegative.')
    return result


def _sensitive_registration(value):
    if type(value) is str:
        return re.search(r'(?i)\b(?:Bearer\s+\S+|(?:Authorization|Proxy[-_ ]?Authorization|X[-_ ]?Api[-_ ]?Key|Api[-_ ]?Key|Access[-_ ]?Token|Refresh[-_ ]?Token|Password|Client[-_ ]?Secret|Credentials?)\s*[\"\']?\s*[:=])', value) is not None
    if type(value) is dict:
        return contract._has_auth_key(value) or any(_sensitive_registration(k) or _sensitive_registration(v)
                                                   for k, v in value.items())
    if type(value) is list:
        return any(_sensitive_registration(v) for v in value)
    return False


def _no_links(path):
    path = Path(path).absolute()
    if '..' in path.parts:
        raise ValueError('Parent traversal is forbidden.')
    for parent in reversed((path,) + tuple(path.parents)):
        try:
            if stat.S_ISLNK(parent.lstat().st_mode):
                raise ValueError('Symbolic links are forbidden for controlled files.')
        except FileNotFoundError:
            continue
    return path


def _regular_bytes(path):
    path = _no_links(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('Expected an actual regular file.')
    return path.read_bytes()


class SessionBudget:
    """Exclusive, append-only, fsynced T/GV owner; never resume/rewrite a ledger."""
    def __init__(self, run_dir, session_config, request_sha256):
        config = flow.json_snapshot(session_config)
        required = {'max_attempts', 'currency', 'max_cost', 'allowed_groups', 'model_prices', 'approved_counter_sha256'}
        if type(config) is not dict or set(config) != required or _sensitive_registration(config):
            raise ValueError('Exact credential-free shared budget configuration required.')
        if (type(config['max_attempts']) is not int or not 1 <= config['max_attempts'] <= 5
                or config['currency'] != 'USD' or not Decimal('0') < _decimal(config['max_cost']) <= Decimal('0.02')
                or type(config['allowed_groups']) is not list or len(config['allowed_groups']) != 2
                or set(config['allowed_groups']) != {'T', 'GV'} or not _sha(request_sha256)):
            raise ValueError('Invalid approved T/GV hard budget or request identity.')
        if config['approved_counter_sha256'] is not None and not _sha(config['approved_counter_sha256']):
            raise ValueError('Approved counter identity must be an explicit SHA-256 or null.')
        prices = config['model_prices']
        if type(prices) is not dict or not prices:
            raise ValueError('An explicitly model-bound price snapshot is required.')
        for model, entry in prices.items():
            if (model not in codec.SUPPORTED_MODELS or type(entry) is not dict
                    or set(entry) != {'input_per_million', 'output_per_million', 'documented_served_identity', 'source_url'}
                    or type(entry['documented_served_identity']) is not str or not entry['documented_served_identity']
                    or type(entry['source_url']) is not str or not entry['source_url'].startswith('https://api-docs.deepseek.com/')):
                raise ValueError('Invalid registered model/price identity.')
            _decimal(entry['input_per_million'])
            _decimal(entry['output_per_million'])
        run = _no_links(run_dir)
        if not run.is_relative_to(ROOT / 'runs/q01_provider'):
            raise ValueError('Budget outputs must remain in a new project provider run.')
        if run.exists():
            if not run.is_dir() or any(run.iterdir()):
                raise ValueError('Budget directory must be new or empty; existing evidence cannot be reused.')
        else:
            run.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.run_dir, self._config, self.request_sha256 = run, config, request_sha256
        self.ledger_path = run / 'session_ledger.jsonl'
        self._fd = os.open(self.ledger_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._file_identity = (os.fstat(self._fd).st_dev, os.fstat(self._fd).st_ino)
        self._ledger_sha = hashlib.sha256(b'').hexdigest()
        self._lock, self.exchange_lock = threading.RLock(), threading.Lock()
        self._closed, self._broken, self._sequence = False, False, 0
        self._attempt_count, self._reserved_cost = 0, Decimal('0')
        self._counts, self._flows, self._group_counts, self._denials = dict.fromkeys(TOTAL_LIMITS, 0), {}, {}, []
        self._freeze_path, self._freeze_bytes, self._freeze_binding = None, None, None
        self._append('init', {'config': config, 'request_sha256': request_sha256,
                              'ledger_policy': 'exclusive owner; fsync before HTTP; no refunds or restart'})

    @property
    def config(self):
        return flow.json_snapshot(self._config)

    def _live(self):
        if self._closed or self._broken:
            raise ProviderError('session_unusable', 'Budget session is closed or broken; no request permitted.')

    def _check_ledger(self):
        current = self.ledger_path.lstat()
        owned = os.fstat(self._fd)
        if (not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != self._file_identity
                or not stat.S_ISREG(owned.st_mode) or (owned.st_dev, owned.st_ino) != self._file_identity):
            raise OSError('Ledger identity or bytes changed externally.')
        raw = _regular_bytes(self.ledger_path)
        after = self.ledger_path.lstat()
        if ((after.st_dev, after.st_ino) != self._file_identity or not stat.S_ISREG(after.st_mode)
                or hashlib.sha256(raw).hexdigest() != self._ledger_sha):
            raise OSError('Ledger identity or bytes changed externally.')
        return raw

    def _append(self, event, data):
        self._live()
        row = dict(data, seq=self._sequence + 1, event=event,
                   recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            before = self._check_ledger()
            blob = (flow.canonical(row) + '\n').encode('ascii')
            offset = 0
            while offset < len(blob):
                written = os.write(self._fd, blob[offset:])
                if written <= 0:
                    raise OSError('Short ledger write.')
                offset += written
            os.fsync(self._fd)
            current, owned = self.ledger_path.lstat(), os.fstat(self._fd)
            if (not stat.S_ISREG(current.st_mode) or not stat.S_ISREG(owned.st_mode)
                    or (current.st_dev, current.st_ino) != self._file_identity
                    or (owned.st_dev, owned.st_ino) != self._file_identity):
                raise OSError('Ledger path or descriptor identity changed during durable append.')
            after = _regular_bytes(self.ledger_path)
            current = self.ledger_path.lstat()
            if ((current.st_dev, current.st_ino) != self._file_identity
                    or not stat.S_ISREG(current.st_mode) or after != before + blob):
                raise OSError('Ledger bytes changed during durable append.')
            self._ledger_sha = hashlib.sha256(after).hexdigest()
            self._sequence += 1
        except Exception:
            self._broken = True
            raise ProviderError('ledger_write_failed', 'Durable ledger write failed; session disabled.') from None
        return row

    def _deny(self, code, reason, **details):
        row = dict(details, code=code, reason=reason, http_request_sent=False)
        self._denials.append(row)
        self._append('denial', row)
        raise ProviderError(code, reason)

    def begin_flow(self, group, request_sha256):
        with self._lock:
            self._live()
            if group not in GROUP_LIMITS or group not in self.config['allowed_groups']:
                self._deny('group_forbidden', 'Only the approved T and GV flows are permitted.')
            if request_sha256 != self.request_sha256:
                self._deny('request_identity_mismatch', 'Both flows must use the same registered request bytes.', group=group)
            if group in self._flows:
                self._deny('flow_already_started', 'Each approved flow may begin only once.', group=group)
            self._append('begin_flow', {'group': group, 'request_sha256': request_sha256})
            self._flows[group] = request_sha256
            self._group_counts[group] = dict.fromkeys(TOTAL_LIMITS, 0)

    def reserve(self, group, action, input_tokens, output_cap, *, model=codec.API_MODEL):
        with self._lock:
            self._live()
            if group not in self._flows or action not in TOTAL_LIMITS:
                self._deny('flow_not_started', 'A registered flow and action are required.')
            if type(input_tokens) is not int or input_tokens < 0 or type(output_cap) is not int or output_cap <= 0:
                self._deny('invalid_token_count', 'Reservation counts must be native nonnegative/positive integers.', group=group)
            if model not in self.config['model_prices']:
                self._deny('unregistered_model_price', 'No approved price/identity snapshot for this model.', group=group, action=action)
            if (self._attempt_count >= self.config['max_attempts'] or self._counts[action] >= TOTAL_LIMITS[action]
                    or self._group_counts[group][action] >= GROUP_LIMITS[group][action]):
                self._deny('budget_exhausted', 'Total, group or action attempt limit exhausted.', group=group, action=action)
            price = self.config['model_prices'][model]
            cost = (Decimal(input_tokens) * _decimal(price['input_per_million'])
                    + Decimal(output_cap) * _decimal(price['output_per_million'])) / Decimal('1000000')
            if self._reserved_cost + cost > _decimal(self.config['max_cost']):
                self._deny('cost_budget_exhausted', 'Worst-case reservation exceeds the approved USD hard budget.', group=group, action=action)
            reservation = {'attempt': self._attempt_count + 1, 'group': group, 'action': action,
                           'model': model, 'input_tokens': input_tokens, 'output_cap': output_cap,
                           'reserved_cost': str(cost), 'cumulative_reserved_cost': str(self._reserved_cost + cost),
                           'currency': 'USD', 'price_snapshot_sha256': flow.digest(price),
                           'reservation_policy': 'No refund on timeout, failure or unknown result.'}
            self._append('reserve', reservation)
            self._attempt_count += 1
            self._reserved_cost += cost
            self._counts[action] += 1
            self._group_counts[group][action] += 1
            return flow.json_snapshot(reservation)

    def freeze(self, mode, binding, first_request):
        with self._lock:
            self._live()
            try:
                self._check_ledger()
            except Exception:
                self._broken = True
                raise ProviderError('ledger_write_failed', 'Ledger integrity failed before freeze; session disabled.') from None
            if self._freeze_path is not None:
                try:
                    if _regular_bytes(self._freeze_path) != self._freeze_bytes or binding != self._freeze_binding:
                        self._deny('freeze_identity_mismatch', 'Frozen bytes or session/provider identity changed before HTTP.')
                except ProviderError:
                    raise
                except Exception:
                    self._broken = True
                    raise ProviderError('freeze_identity_mismatch', 'Frozen file identity changed; session disabled.') from None
                return
            name = 'mock_freeze.json' if mode == 'http_mock' else 'first_request_freeze.json'
            path = self.run_dir / name
            record = {'transport': mode, 'real_http_authorized_by_this_record': mode == 'real_http',
                      'binding': binding, 'first_request': first_request,
                      'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            blob = (json.dumps(record, ensure_ascii=True, indent=2, allow_nan=False) + '\n').encode('ascii')
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                try:
                    offset = 0
                    while offset < len(blob):
                        written = os.write(fd, blob[offset:])
                        if written <= 0:
                            raise OSError('Short freeze write.')
                        offset += written
                    os.fsync(fd)
                finally:
                    os.close(fd)
                if _regular_bytes(path) != blob:
                    raise OSError('Freeze byte readback failed.')
                self._append('freeze', {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'transport': mode})
            except Exception:
                self._broken = True
                raise ProviderError('freeze_write_failed', 'Exclusive freeze failed; no request allowed.') from None
            self._freeze_path, self._freeze_bytes, self._freeze_binding = path, blob, flow.json_snapshot(binding)

    def record_result(self, reservation, outcome):
        with self._lock:
            self._append('result', {'attempt': reservation['attempt'], 'group': reservation['group'],
                                    'action': reservation['action'], 'outcome': outcome,
                                    'reservation_refunded': False})

    def snapshot(self):
        with self._lock:
            return flow.json_snapshot({'config': self.config, 'request_sha256': self.request_sha256,
                                       'ledger_path': str(self.ledger_path), 'ledger_sha256': self._ledger_sha,
                                       'attempt_count': self._attempt_count, 'reserved_cost': str(self._reserved_cost),
                                       'counts': self._counts, 'flows': self._flows, 'group_counts': self._group_counts,
                                       'denials': self._denials, 'broken': self._broken, 'closed': self._closed,
                                       'freeze_path': str(self._freeze_path) if self._freeze_path else None})

    def close(self):
        with self._lock:
            if not self._closed:
                os.close(self._fd)
                self._closed = True


class OneShotDeepSeekHTTP:
    def __init__(self, config, budget, *, token_counter=None, credential_env='DEEPSEEK_API_KEY',
                 http_transport=None, test_credential=None):
        self._config, _ = codec._mock_config(config)
        self.budget, self.counter, self.credential_env = budget, token_counter, credential_env
        if type(credential_env) is not str or re.fullmatch(r'[A-Z][A-Z0-9_]*', credential_env) is None:
            raise ValueError('Credential configuration must be a variable name, not a value.')
        if http_transport is not None and type(http_transport) is not httpx.MockTransport:
            raise ValueError('Injected transport must be exactly httpx.MockTransport.')
        if http_transport is None and test_credential is not None:
            raise ValueError('Synthetic credentials are permitted only with MockTransport.')
        self.mode = 'http_mock' if http_transport is not None else 'real_http'
        self._transport, self._test_credential = http_transport, test_credential
        self._redactor = contract.BoundedJSONProvider(codec._mock_config(config)[1], lambda *_: None,
            redact_values=(test_credential,) if type(test_credential) is str and test_credential else ())
        self.events, self.last_error, self.httpx_post_invocations = [], None, 0

    @property
    def config(self):
        return flow.json_snapshot(self._config)

    def _counter_result(self, body):
        if self.counter is None:
            raise ProviderError('token_counter_required', 'No verified full-chat token counter was supplied.')
        try:
            identity = flow.json_snapshot(self.counter.identity)
        except Exception:
            raise ProviderError('token_counter_unverified', 'Counter identity is not recordable.') from None
        base = {'kind', 'scope', 'verified_models', 'chat_template_sha256', 'artifact_sha256'}
        if (type(identity) is not dict or not base <= set(identity)
                or contract._has_auth_key(identity) or self._redactor._redact(identity) != identity
                or identity['scope'] != 'complete_chat_messages'
                or type(identity['verified_models']) is not list or body['model'] not in identity['verified_models']
                or not _sha(identity['chat_template_sha256']) or not _sha(identity['artifact_sha256'])):
            raise ProviderError('token_counter_unverified', 'Counter does not bind this model and complete chat template.')
        if self.mode == 'http_mock':
            if identity['kind'] != 'http_mock_token_counter':
                raise ProviderError('token_counter_unverified', 'Mock HTTP requires an explicitly synthetic counter identity.')
        else:
            if (identity['kind'] != 'verified_chat_token_counter'
                    or self.budget.config['approved_counter_sha256'] != flow.digest(identity)
                    or identity.get('provider_config_sha256') != flow.digest(self.config)):
                raise ProviderError('token_counter_unverified', 'No approved real counter identity registration.')
            for prefix in ('artifact', 'chat_template', 'implementation', 'verification'):
                try:
                    path = Path(identity[prefix + '_path']).absolute()
                    if path.is_relative_to(ROOT / 'evaluation'):
                        raise ValueError('Evaluation files cannot be tokenizer artifacts.')
                    if hashlib.sha256(_regular_bytes(path)).hexdigest() != identity[prefix + '_sha256']:
                        raise ValueError('Counter artifact bytes differ from approved identity.')
                except Exception:
                    raise ProviderError('token_counter_unverified', 'Approved counter artifact/template/code/verification mismatch.') from None
            try:
                source = inspect.getsourcefile(self.counter.count_messages)
                if source is None:
                    raise ValueError('Counter implementation source is unavailable.')
                actual = _no_links(source)
                expected = _no_links(identity['implementation_path'])
                if (actual != expected or hashlib.sha256(_regular_bytes(actual)).hexdigest()
                        != identity['implementation_sha256']):
                    raise ValueError('Actual counter implementation differs from its approved artifact.')
            except Exception:
                raise ProviderError('token_counter_unverified', 'Actual counter source is not the approved implementation.') from None
        try:
            result = flow.json_snapshot(self.counter.count_messages(flow.json_snapshot(body['messages']), body['model']))
        except Exception as error:
            raise ProviderError('invalid_token_count', self._redactor._redact_text(flow.error_text(error))) from None
        if (type(result) is not dict or set(result) != {'input_tokens', 'model', 'messages_sha256', 'chat_template_sha256'}
                or type(result['input_tokens']) is not int or result['input_tokens'] < 0
                or result['model'] != body['model'] or result['messages_sha256'] != flow.digest(body['messages'])
                or result['chat_template_sha256'] != identity['chat_template_sha256']):
            raise ProviderError('invalid_token_count', 'Counter result is not an exact nonnegative model/messages/template-bound integer.')
        return identity, result

    def _context(self, body, context):
        required = {'request_sha256', 'business_request_sha256', 'prompt_bundle_sha256', 'schema_sha256', 'code_hashes'}
        context = flow.json_snapshot(context)
        if type(context) is not dict or set(context) != required:
            raise ProviderError('freeze_context_invalid', 'Complete request/prompt/schema/code identities are mandatory.')
        hashes = {str(p.relative_to(ROOT)): flow.file_hash(p) for p in CODE_PATHS}
        if (context['request_sha256'] != self.budget.request_sha256 or context['code_hashes'] != hashes
                or context['schema_sha256'] != flow.digest(SCHEMAS) or not _sha(context['prompt_bundle_sha256'])):
            raise ProviderError('freeze_context_invalid', 'Code/schema/request identity does not match final bytes.')
        try:
            business = k02.strict_json(body['messages'][1]['content'])['request']
            k02.validate_request(business)
        except Exception:
            raise ProviderError('freeze_context_invalid', 'Encoded user message lacks the exact business request.') from None
        if context['business_request_sha256'] != flow.digest(business):
            raise ProviderError('request_identity_mismatch', 'Encoded business request hash changed.')
        if self.mode == 'real_http':
            if self.budget.request_sha256 != REQUEST_SHA256:
                raise ProviderError('request_identity_mismatch', 'Only the approved pre-existing development input may be sent.')
            raw = _regular_bytes(REQUEST_PATH)
            actual_hash = hashlib.sha256(raw).hexdigest()
            if actual_hash != REQUEST_SHA256:
                raise ProviderError('request_identity_mismatch', 'Approved input bytes differ.')
            with self.budget._lock:
                self.budget._append('input_registration', {
                    'path': str(REQUEST_PATH), 'actual_sha256': actual_hash,
                    'purpose': 'Only approved pre-existing development input for controlled provider wiring; registered before parsing.'})
            if k02.strict_json(raw) != business:
                raise ProviderError('request_identity_mismatch', 'Approved input bytes or parsed business request differ.')
        return context

    def request(self, group, action, endpoint, body, settings, freeze_context):
        self.last_error = None
        event = {'group': group, 'action': action, 'transport': self.mode,
                 'httpx_post_invoked': False, 'official_http_request_attempted': False,
                 'intended_official_endpoint': codec.OFFICIAL_ENDPOINT, 'status': 'preflight',
                 'error_code': None, 'raw_response': 'unknown', 'actual_bill': 'unknown'}
        self.events.append(event)
        start, locked, reservation = time.perf_counter(), False, None
        try:
            if not self.budget.exchange_lock.acquire(blocking=False):
                raise ProviderError('concurrency_forbidden', 'Shared session already has an in-flight exchange.')
            locked = True
            self.budget._live()
            body, settings = flow.json_snapshot(body), flow.json_snapshot(settings)
            if (endpoint != codec.OFFICIAL_ENDPOINT or body.get('model') != self.config['models'].get(action)
                    or body.get('stream') is not False or body.get('response_format') != {'type': 'json_object'}
                    or body.get('max_tokens') != self.config['max_output_tokens'].get(action)
                    or set(body) != {'model', 'messages', 'stream', 'response_format', 'max_tokens'} | set(self.config['parameters'].get(action, {}))
                    or any(body.get(k) != v for k, v in self.config['parameters'].get(action, {}).items())
                    or type(body.get('messages')) is not list or len(body['messages']) != 2
                    or [m.get('role') for m in body['messages']] != ['system', 'user']
                    or any(type(m.get('content')) is not str for m in body['messages'])):
                raise ProviderError('request_configuration_mismatch', 'HTTP body differs from frozen model/parameters/schema or enables unsupported fields.')
            if (settings.get('model') != body['model'] or settings.get('parameters') != self.config['parameters'][action]
                    or settings.get('max_input_tokens') != self.config['max_input_tokens'][action]
                    or settings.get('max_output_tokens') != self.config['max_output_tokens'][action]
                    or settings.get('timeout_seconds') != self.config['timeout_seconds']
                    or settings.get('retries') != 0 or settings.get('concurrency') != 1):
                raise ProviderError('request_configuration_mismatch', 'HTTP controls differ from approved configuration.')
            context = self._context(body, freeze_context)
            counter_identity, counted = self._counter_result(body)
            event['token_count'] = counted
            if counted['input_tokens'] > self.config['max_input_tokens'][action]:
                raise ProviderError('input_token_budget_exceeded', 'Complete chat input exceeds the configured hard token cap.')
            if body['model'] not in self.budget.config['model_prices']:
                raise ProviderError('unregistered_model_price', 'No model-bound approved price snapshot; no HTTP permitted.')
            # Counter/config/context gates run before any real credential access.
            if self.mode == 'http_mock':
                credential = self._test_credential
            else:
                credential = os.environ.get(self.credential_env)
            if type(credential) is not str or not credential.strip():
                raise ProviderError('credential_missing', 'The configured credential is unavailable; no HTTP permitted.')
            self._redactor._redact_values = (credential, repr(credential.encode('utf-8'))[2:-1])
            binding = {'provider_config': self.config, 'session_config': self.budget.config,
                       'request_sha256': context['request_sha256'],
                       'business_request_sha256': context['business_request_sha256'],
                       'prompt_bundle_sha256': context['prompt_bundle_sha256'], 'schema_sha256': context['schema_sha256'],
                       'code_hashes': context['code_hashes'], 'token_counter_identity': counter_identity,
                       'credential_environment_name': self.credential_env, 'transport': self.mode,
                       'httpx_version': httpx.__version__, 'retries': 0, 'concurrency': 1,
                       'follow_redirects': False, 'trust_env': False}
            if self._redactor._redact(binding) != binding:
                raise ProviderError('sensitive_registration', 'Authentication material cannot enter a frozen registration.')
            self.budget.freeze(self.mode, binding, {'group': group, 'action': action,
                               'official_request_sha256': flow.digest(body), 'settings': settings})
            # Reserve full input cap, not a smaller actual count. Never refund.
            reservation = self.budget.reserve(group, action, self.config['max_input_tokens'][action],
                                               self.config['max_output_tokens'][action], model=body['model'])
            event.update(reservation=reservation, official_request=self._redactor._redact(body),
                         requested_model=body['model'])
            transport = self._transport if self.mode == 'http_mock' else httpx.HTTPTransport(retries=0, trust_env=False)
            try:
                with httpx.Client(transport=transport, follow_redirects=False, trust_env=False,
                                  timeout=httpx.Timeout(self.config['timeout_seconds'])) as client:
                    event['httpx_post_invoked'] = True
                    event['official_http_request_attempted'] = self.mode == 'real_http'
                    self.httpx_post_invocations += 1
                    response = client.post(endpoint, json=body, headers={'Authorization': 'Bearer ' + credential,
                                                                      'Content-Type': 'application/json'})
                    raw = response.content
                    event['response_bytes_sha256'] = hashlib.sha256(raw).hexdigest()
                    try:
                        text = raw.decode('utf-8')
                    except UnicodeDecodeError:
                        event['raw_response'] = {'body_bytes_escaped': self._redactor._redact_text(repr(raw)),
                                                 'encoding': 'invalid_utf8', 'byte_count': len(raw)}
                        raise ProviderError('invalid_json', 'Response bytes are not valid UTF-8; no replacement decoding.') from None
                    envelope = {'status_code': response.status_code, 'body': text}
                    if 'x-request-id' in response.headers:
                        envelope['request_id'] = response.headers['x-request-id']
                    event['raw_response'] = self._redactor._redact(envelope)
                    event['http_status'] = response.status_code
            except httpx.TimeoutException as error:
                raise ProviderError('timeout', self._redactor._redact_text(flow.error_text(error))) from None
            except httpx.HTTPError as error:
                raise ProviderError('transport_error', self._redactor._redact_text(flow.error_text(error))) from None
            event['status'] = 'http_response_received'
            self.budget.record_result(reservation, {'status': event['status'], 'http_status': response.status_code,
                                                   'transport': self.mode, 'response_bytes_sha256': event['response_bytes_sha256']})
            reservation = None
            return envelope
        except Exception as error:
            code = error.code if isinstance(error, ProviderError) else 'transport_error'
            message = self._redactor._redact_text(flow.error_text(error))
            self.last_error = (code, message)
            event.update(status='failed', error_code=code, diagnostic=message)
            if reservation is not None:
                try:
                    self.budget.record_result(reservation, {'status': 'failed', 'error_code': code, 'transport': self.mode})
                except Exception as ledger_error:
                    event['result_record_error'] = self._redactor._redact_text(flow.error_text(ledger_error))
            raise ProviderError(code, message) from None
        finally:
            event['elapsed_seconds'] = time.perf_counter() - start
            if locked:
                self.budget.exchange_lock.release()

    @property
    def identity(self):
        return {'kind': 'provider_contract_mock' if self.mode == 'http_mock' else 'real_provider',
                'transport': self.mode, 'code_sha256': flow.file_hash(__file__),
                'httpx_version': httpx.__version__, 'retries': 0, 'concurrency': 1,
                'actual_official_http_requests': sum(e['official_http_request_attempted'] for e in self.events)}

    def snapshot(self):
        return flow.json_snapshot({'identity': self.identity, 'events': self.events,
                                   'httpx_post_invocations': self.httpx_post_invocations,
                                   'actual_official_http_requests': sum(e['official_http_request_attempted'] for e in self.events)})


class DeepSeekHTTPProvider(codec.DeepSeekChatProvider):
    def __init__(self, config, budget, group, request_sha256, *, token_counter=None,
                 credential_env='DEEPSEEK_API_KEY', http_transport=None, test_credential=None):
        self._http = OneShotDeepSeekHTTP(config, budget, token_counter=token_counter,
            credential_env=credential_env, http_transport=http_transport, test_credential=test_credential)
        self._budget, self._group, self._request_sha256 = budget, group, request_sha256
        self._active_action, self._active_context = None, None
        super().__init__(config, self._exchange_http, redact_values=(test_credential,) if type(test_credential) is str and test_credential else ())
        budget.begin_flow(group, request_sha256)

    def _exchange_http(self, endpoint, body, settings):
        try:
            return self._http.request(self._group, self._active_action, endpoint, body, settings, self._active_context)
        finally:
            # The live key remains only in memory. Share its masking patterns
            # with the codec before it logs/parses any original response text.
            self._inner._redact_values = self._http._redactor._redact_values

    @property
    def identity(self):
        identity = super().identity
        identity.update(kind='provider_contract_mock' if self._http.mode == 'http_mock' else 'real_provider',
                        version=VERSION, transport=self._http.mode, network_enabled=self._http.mode == 'real_http',
                        code_sha256=flow.file_hash(__file__), codec_code_sha256=flow.file_hash(codec.__file__),
                        token_limits_enforced_for_real_service='conditional_on_verified_counter_gate',
                        actual_http_request_count=self._http.identity['actual_official_http_requests'])
        return identity

    def _bridge(self, action, payload, settings):
        prompt_path = ROOT / 'configs/q01_provider_prompts.json'
        prompts = k02.strict_json(_regular_bytes(prompt_path))
        if self._http.mode == 'real_http' and payload['instructions'] != [prompts['common'], prompts[action]]:
            raise ProviderError('prompt_identity_mismatch', 'Actual prompt differs from the registered complete prompt bundle.')
        self._active_action = action
        self._active_context = {'request_sha256': self._request_sha256,
            'business_request_sha256': flow.digest(payload['request']),
            'prompt_bundle_sha256': hashlib.sha256(_regular_bytes(prompt_path)).hexdigest(),
            'schema_sha256': flow.digest(SCHEMAS),
            'code_hashes': {str(p.relative_to(ROOT)): flow.file_hash(p) for p in CODE_PATHS}}
        count = len(self._official_attempts)
        try:
            return super()._bridge(action, payload, settings)
        finally:
            for row in self._official_attempts[count:]:
                event = self._http.events[-1] if self._http.events else {}
                attempted = bool(event.get('official_http_request_attempted'))
                received = attempted and type(event.get('http_status')) is int
                row.update(transport=self._http.mode, record_kind=self.identity['kind'],
                           response_identity_origin=self._http.mode,
                           http_request_attempted=attempted,
                           http_request_sent=True if received else ('unknown' if attempted else False),
                           official_service_response_received=received)

    def _call(self, action, payload):
        self._http.last_error = None
        count = len(self._official_attempts)
        try:
            return super()._call(action, payload)
        except ProviderError:
            if self._http.last_error is not None:
                code, message = self._http.last_error
                for row in self._official_attempts[count:]:
                    row.update(status='failed', error_code=code, parse_diagnostic=message)
                raise ProviderError(code, message) from None
            raise

    def snapshot(self):
        record = super().snapshot()
        record.update(identity=self.identity, transport=self._http.mode, record_kind=self.identity['kind'],
                      codec_exchange_invocation_count=record['actual_attempt_count'],
                      actual_attempt_count=self._http.httpx_post_invocations,
                      actual_http_request_count=self._http.identity['actual_official_http_requests'],
                      attempt_unit='httpx_post_invocations', http_transport=self._http.snapshot(),
                      session_budget=self._budget.snapshot())
        record['normalized_contract']['identity'] = {'kind': 'mechanical_normalization',
            'source_provider_kind': self.identity['kind'], 'transport': self._http.mode,
            'code_sha256': flow.file_hash(contract.__file__), 'semantic_ability_demonstrated': False}
        record['normalized_contract']['record_kind'] = 'mechanical_normalization'
        for row in record['normalized_contract']['attempts']:
            row['record_kind'] = 'mechanical_normalization'
            row['transport'] = self._http.mode
        return flow.json_snapshot(record)

    @property
    def attempts(self):
        return flow.json_snapshot(self._official_attempts)

    def close(self):
        """No persistent HTTP client is kept; the shared budget is caller-owned."""
        return None
