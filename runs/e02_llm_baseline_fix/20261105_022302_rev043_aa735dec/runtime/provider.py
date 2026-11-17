"""Pinned non-thinking JSON profile, complete-chat counter, durable no-retry ledger."""
from decimal import Decimal
import datetime
import fcntl
import hashlib
import importlib
import os
from pathlib import Path
import sys
import threading
import time
import base64
from common import require, Rejected, canonical, digest, snapshot, read_json, read_bytes, file_hash, safe_path, write_new


class Counter:
    def __init__(self, root, assets):
        self.root, self.assets = Path(root), snapshot(assets)
        self._asset_pin=digest(self.assets)
        self.check()
        vendor = self.root / assets['vendor_path']
        sys.path.insert(0, str(vendor))
        recipe = importlib.import_module('deepseek_recipe')
        native = importlib.import_module('deepseek_recipe._native')
        require(Path(recipe.__file__).absolute() == self.root / assets['files']['python_init']['path'], 'loaded recipe identity mismatch')
        require(Path(native.__file__).absolute() == self.root / assets['files']['native']['path'], 'native identity mismatch')
        require(recipe.__version__ == assets['recipe_version'], 'recipe version mismatch')
        self.recipe = recipe
        tokenizer = recipe.Tokenizer.from_file(str(self.root / assets['files']['tokenizer']['path']))
        self.encoding = recipe.DeepseekV41Encoding().with_tokenizer(tokenizer)
        self.options = recipe.ConversionOptions(default_thinking_mode=False).with_model_alias('deepseek-flash', recipe.ModelVersion.DeepseekV41Flash)
        self.literals = {'｜DSML｜', '<｜image｜>', '<｜deepseek_image｜>'}
        for key in ['tokenizer', 'recipe_tokenizer']:
            self.literals.update(x['content'] for x in read_json(self.root / assets['files'][key]['path'])['added_tokens'])

    def check(self):
        require(digest(self.assets)==self._asset_pin,'counter registration mutated in memory')
        for entry in self.assets['files'].values():
            require(file_hash(self.root / entry['path']) == entry['sha256'], 'counter asset hash drift')

    def count(self, messages):
        self.check()
        require(type(messages) is list and len(messages) == 2 and [m.get('role') for m in messages] == ['system', 'user'], 'system/user new-context profile only')
        for m in messages:
            require(set(m) == {'role', 'content'} and type(m['content']) is str, 'plain text messages only')
            m['content'].encode('utf-8')
            require(not any(t in m['content'] for t in self.literals), 'ambiguous special-token literal')
        body = {'model': 'deepseek-flash', 'messages': snapshot(messages), 'thinking': {'type': 'disabled'}, 'response_format': {'type': 'json_object'}}
        converted = self.recipe.ChatCompletionRequest(body).convert(self.options)
        return {'input_tokens': len(self.encoding.encode(converted.conversation)), 'messages_sha256': digest(messages),
                'counter_assets_sha256': digest(self.assets), 'billing_equivalence_proven': False}


class Ledger:
    """One process, one new directory, one append stream. Restart never resumes it."""
    def __init__(self, directory, config, allowed_slots, output_root, gate=None):
        self._approved_live=False
        if config['scope']['stage']=='E02':
            require(type(gate) is LiveGate,'E02 ledger requires a separately approved live gate')
            gate.check()
            require(digest(config)==gate.binding['configuration_sha256'],'gate/ledger configuration mismatch')
            self._approved_live=True
        self.directory = safe_path(directory, output_root)
        require(not self.directory.exists(), 'new batch directory required; no resume/reset')
        self.directory.mkdir(parents=False)
        self.path = self.directory / 'ledger.jsonl'
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        st = os.fstat(self.fd)
        self.identity = st.st_dev, st.st_ino
        self.expected_bytes = b''
        self.lock = threading.RLock()
        self.exchange = threading.Lock()
        self.config = snapshot(config)
        self._config_pin=digest(self.config)
        self.allowed = snapshot(allowed_slots)
        require(digest(self.allowed)==config['allowed_slots_sha256'],'unapproved attempt-slot registry')
        self.used, self.counts = set(), {}
        self.reserved = Decimal('0')
        self.broken = False
        self.closed = False
        self._append({'event': 'init', 'config_sha256': digest(config), 'slots_sha256': digest(allowed_slots),
                      'scope': 'fresh independent E02 ledger; no old ledger imported'})

    def check(self):
        require(not self.closed and not self.broken, 'ledger closed or poisoned')
        require(digest(self.config)==self._config_pin,'ledger configuration mutated')
        try:
            st = self.path.lstat()
            require((st.st_dev, st.st_ino) == self.identity and read_bytes(self.path) == self.expected_bytes, 'ledger replaced or altered')
        except Exception:
            self.broken = True
            raise

    def _append(self, record):
        self.check()
        raw = (canonical(dict(record, sequence=self.expected_bytes.count(b'\n') + 1)) + '\n').encode()
        try:
            written = os.write(self.fd, raw)
            require(written == len(raw), 'short durable append')
            os.fsync(self.fd)
            self.expected_bytes += raw
            self.check()
        except Exception:
            self.broken = True
            raise

    def reserve(self, slot, action, body_hash):
        with self.lock:
            self.check()
            require(slot in self.allowed and self.allowed[slot] == action and slot not in self.used, 'unapproved or already consumed attempt')
            role = self.config['actions'][action]
            n = self.counts.get(action, 0)
            require(n < role['max_attempts'], 'role attempt ceiling')
            amount = (Decimal(role['input_cap']) * Decimal(self.config['price']['input_per_million']) + Decimal(role['output_cap']) * Decimal(self.config['price']['output_per_million'])) / 1000000
            require(self.reserved + amount <= Decimal(self.config['proposed_hard_budget_USD']), 'whole batch money ceiling')
            self.used.add(slot)
            self.counts[action] = n + 1
            self.reserved += amount
            self._append({'event': 'reserve', 'slot': slot, 'action': action, 'body_sha256': body_hash, 'amount_USD': str(amount),
                          'total_reserved_USD': str(self.reserved), 'no_refund_or_retry': True})
            return str(amount)

    def result(self, slot, status, metadata):
        with self.lock:
            self._append({'event': 'result', 'slot': slot, 'status': status, 'metadata': metadata})

    def close(self):
        if not self.closed:
            os.close(self.fd)
            self.closed = True


class LiveGate:
    def __init__(self, preparation, config, authorization):
        require(authorization.get('paid_execution_authorized') is True and authorization.get('formal_experiment_released') is True, 'E02 not authorized')
        require(authorization.get('independent_cross_audit_accepted') is True, 'independent cross-audit not accepted')
        prep = safe_path(preparation)
        pin = read_json(prep / 'config/execution_binding.json')
        for p, h in pin['files'].items():
            require(file_hash(prep.parents[2] / p) == h, 'preparation artifact changed')
        import importlib.metadata,platform
        environment=read_json(prep/'config/environment.json')
        require(sys.executable==environment['executable'] and sys.version==environment['python'] and platform.platform()==environment['platform'],'runtime environment changed; rebind required')
        require(all(importlib.metadata.version(k)==v for k,v in environment['packages'].items()),'runtime package versions changed')
        require(authorization.get('preparation_binding_sha256') == file_hash(prep / 'config/execution_binding.json'), 'release binding mismatch')
        require(authorization.get('configuration_sha256') == digest(config), 'exact configuration authorization mismatch')
        require(authorization.get('scope') == config['scope'] and authorization.get('hard_budget_USD') == config['proposed_hard_budget_USD'], 'scope/budget authorization mismatch')
        require(authorization.get('max_calls') == sum(a['max_attempts'] for a in config['actions'].values()), 'attempt authorization mismatch')
        user_record=authorization.get('user_authorization_record')
        require(type(user_record) is str and bool(user_record.strip()) and authorization.get('user_authorization_sha256')==hashlib.sha256(user_record.encode()).hexdigest(), 'actual separate user authorization record/hash required')
        require(authorization.get('cross_audit_report_path') and authorization.get('cross_audit_report_sha256'), 'audit provenance required')
        root = prep.parents[2]
        require(file_hash(safe_path(root / authorization['cross_audit_report_path'], root)) == authorization['cross_audit_report_sha256'], 'cross-audit report drift')
        checked = datetime.datetime.fromisoformat(config['price']['checked_at'])
        age = datetime.datetime.now(datetime.timezone.utc) - checked.astimezone(datetime.timezone.utc)
        require(0 <= age.total_seconds() <= config['price']['maximum_age_hours'] * 3600, 'price/route snapshot needs a new official recheck')
        self.authorization = snapshot(authorization)
        self.binding = pin
        self.preparation = prep

    def check(self):
        root=self.preparation.parents[2]
        for p,h in self.binding['files'].items():require(file_hash(root/p)==h,'frozen preparation changed during run')


class LiveHTTP:
    """Not constructed in preparation. Credentials are read only after exact release checks."""
    def __init__(self, gate, config):
        require(type(gate) is LiveGate, 'live transport requires exact live gate')
        gate.check()
        import httpx
        self._credential = os.environ.get(config['credential_environment_variable'])
        require(type(self._credential) is str and self._credential, 'credential not configured')
        self.client = httpx.Client(trust_env=False, follow_redirects=False, timeout=config['timeout_seconds'], transport=httpx.HTTPTransport(retries=0))
        self.endpoint = config['endpoint']
        self.gate=gate
        self.last_post_invoked=False

    def __call__(self, body):
        self.last_post_invoked=False
        try:
            self.gate.check()
            self.last_post_invoked=True
            response = self.client.post(self.endpoint, json=body, headers={'Authorization': 'Bearer ' + self._credential})
            require(self._credential.encode('utf-8') not in response.content,'credential echo must not be persisted')
            return {'http_status': response.status_code, 'request_id': response.headers.get('x-request-id'), 'raw_body': response.content}
        except Exception as exc:
            # No exception string can accidentally serialize a credential/header.
            import httpx
            raise Rejected('transport timeout; no retry' if isinstance(exc, httpx.TimeoutException) else 'transport failed; no retry') from None

    def close(self):
        self.client.close()


class Provider:
    def __init__(self, config, counter, ledger, transport, record_root):
        self.config, self.counter, self.ledger, self.transport = snapshot(config), counter, ledger, transport
        require(type(counter) is Counter, 'only pinned complete-message Counter is permitted')
        counter.check()
        from baseline import ROOT
        require(digest(counter.assets)==digest(read_json(ROOT/'config/counter_assets.json')), 'counter registration does not match preparation')
        require(counter.root.absolute()==ROOT.parents[2], 'counter project root identity differs')
        self._config_pin=digest(self.config)
        if config['scope']['stage']=='E02':require(type(transport) is LiveHTTP and ledger._approved_live,'real E02 cannot use an injected fixture transport')
        self.transport_kind='real_http' if type(transport) is LiveHTTP else 'injected_synthetic_transport'
        self.record_root = safe_path(record_root, ledger.directory)
        self.record_root.mkdir(exist_ok=True)
        self.flow_cards={}

    def build(self, action, messages):
        role = self.config['actions'][action]
        count = self.counter.count(messages)
        require(count['input_tokens'] <= role['input_cap'], 'overlength_stop: input cap')
        require(count['input_tokens'] + role['output_cap'] <= self.config['context_limit'], 'overlength_stop: context reserve')
        return {'model': self.config['model'], 'messages': snapshot(messages), 'thinking': {'type': 'disabled'},
                'temperature': 0, 'response_format': {'type': 'json_object'}, 'stream': False, 'max_tokens': role['output_cap']}, count

    def send(self, slot, action, messages):
        require(self.ledger.exchange.acquire(blocking=False), 'concurrency not permitted')
        start = time.monotonic()
        reserved = False
        try:
            self.ledger.check()
            require(digest(self.config)==self._config_pin,'provider configuration mutated')
            from baseline import validate_messages
            binding=self.config['slot_bindings'].get(slot)
            require(binding is not None, 'unapproved slot')
            validate_messages(self.config,action,messages,binding)
            body, count = self.build(action, messages)
            amount = self.ledger.reserve(slot, action, digest(body))
            reserved = True
            # Persist the complete credential-free actual body before the one transport invocation.
            write_new(self.record_root / (slot + '.request.json'), {'slot': slot, 'action': action, 'body': body, 'count': count, 'reserve_USD': amount,
                      'transport_kind':self.transport_kind,'phase':'before_transport'}, self.ledger.directory)
            raw = self.transport(snapshot(body))
            require(type(raw) is dict, 'transport response envelope missing')
            response_bytes = raw['raw_body']
            require(type(response_bytes) is bytes, 'response type/length limit')
            write_new(self.record_root / (slot + '.http.json'), {'http_status':raw.get('http_status'),'request_id':raw.get('request_id'),
                      'raw_sha256':hashlib.sha256(response_bytes).hexdigest(),'raw_base64':base64.b64encode(response_bytes).decode('ascii'),
                       'raw_size':len(response_bytes),'response_limit_exceeded':len(response_bytes)>self.config['max_response_bytes'],
                      'transport_kind':self.transport_kind,'http_client_post_invoked':bool(getattr(self.transport,'last_post_invoked',False))}, self.ledger.directory)
            require(len(response_bytes) <= self.config['max_response_bytes'], 'response type/length limit')
            require(raw.get('http_status') == 200, 'HTTP status failure')
            parsed = read_json_response(response_bytes)
            write_new(self.record_root / (slot + '.response.json'), {'response': parsed, 'http_status': 200, 'request_id': raw.get('request_id'),
                      'raw_sha256': hashlib.sha256(response_bytes).hexdigest(), 'raw_utf8': response_bytes.decode('utf-8')}, self.ledger.directory)
            require(parsed.get('model') == self.config['model'], 'returned alias differs from frozen alias')
            usage = parsed.get('usage')
            require(type(usage) is dict and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ['prompt_tokens', 'completion_tokens', 'total_tokens']), 'usage unavailable; stop batch')
            require(usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'usage arithmetic mismatch')
            detail=usage.get('completion_tokens_details') or {}
            require(type(detail) is dict and detail.get('reasoning_tokens') in [None,0], 'unexpected reasoning-token mode; stop batch')
            role = self.config['actions'][action]
            require(usage['prompt_tokens'] <= role['input_cap'] and usage['completion_tokens'] <= role['output_cap'], 'actual service usage exceeded frozen caps; stop batch')
            choice = parsed['choices'][0]
            require(choice['message'].get('reasoning_content') in [None, ''], 'unexpected billed thinking; stop batch')
            meta = {'usage': usage, 'returned_model': parsed['model'], 'documented_route': self.config['documented_route'],
                    'transport_kind':self.transport_kind,'http_client_post_invoked':bool(getattr(self.transport,'last_post_invoked',False)),
                    'backend_identity_independently_confirmed': False, 'system_fingerprint': parsed.get('system_fingerprint'),
                    'response_id': parsed.get('id'), 'latency_seconds': time.monotonic() - start, 'timing_censored':False,
                    'local_input_tokens':count['input_tokens'],'local_messages_sha256':count.get('messages_sha256'),
                    'local_billing_equivalence_proven':False,'actual_bill': 'unknown',
                    'estimated_peak_USD': str((Decimal(usage['prompt_tokens']) * Decimal(self.config['price']['input_per_million']) + Decimal(usage['completion_tokens']) * Decimal(self.config['price']['output_per_million'])) / 1000000)}
            self.ledger.result(slot, 'received', meta)
            # Truncation/blank/JSON failures consume a normal attempt, never a hidden retry.
            require(choice.get('finish_reason') == 'stop', 'output truncated or unexpected finish')
            text = choice['message'].get('content')
            require(type(text) is str and text.strip(), 'blank JSON content')
            from common import strict_json
            value = strict_json(text)
            require(type(value) is dict, 'JSON output must be an object')
            return value, meta
        except Exception as exc:
            if not reserved:
                # A rejected count/binding/budget must stop later slots too.
                self.ledger.broken = True
            if reserved:
                try:
                    self.ledger.result(slot, 'failed', {'error_type': type(exc).__name__, 'message': str(exc) if isinstance(exc, Rejected) else 'invalid response/recording',
                     'transport_kind':self.transport_kind,'http_client_post_invoked':bool(getattr(self.transport,'last_post_invoked',False)),
                     'local_input_tokens':count['input_tokens'],'local_messages_sha256':count.get('messages_sha256'),
                     'timing_censored':isinstance(exc,Rejected) and 'transport timeout' in str(exc),
                     'latency_seconds': time.monotonic() - start})
                finally:
                    # Accounting/transport/recording uncertainty prevents further spending.
                    if not isinstance(exc, Rejected) or not any(x in str(exc) for x in ['truncated', 'blank JSON', 'JSON output must']):
                        self.ledger.broken = True
            raise
        finally:
            self.ledger.exchange.release()


def read_json_response(raw):
    from common import strict_json
    return strict_json(raw)
