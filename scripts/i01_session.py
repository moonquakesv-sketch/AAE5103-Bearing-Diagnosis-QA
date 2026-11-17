"""New I01 batch gates and durable twelve-flow credits; old transports unchanged.

Import performs no inference, credential access, writes or network calls. Each
claim consumes a whole USD .02 credit, including failures and interrupted cases.
"""
import datetime
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import threading

import i01_http as base
import q01_flow as flow
import q01_intent as intent
import q01_intent_http as wire
import q01_deepseek_http as old

ROOT = old.ROOT
Error = wire.Error
put, read = wire.put, wire.read
ALLOCATION = wire.ALLOCATION
CODE_FILES = base.CODE_FILES | {
    'scripts/i01_session.py', 'scripts/i01_app.py', 'scripts/i01_knowledge.py'}
QA_PROMPT_FILE = 'configs/i01_provider_prompts.json'
CONFIG_FILES = wire.CONFIG_FILES | {QA_PROMPT_FILE}
BATCH_ALLOCATION = {'max_flows': 12, 'max_cost_usd': '0.24',
                    'flow_credit_usd': '0.02', 'max_attempts': 60,
                    'retries': 0, 'concurrency': 1}


def _folder(run):
    run = old._no_links(run)
    if not run.is_relative_to(ROOT / 'runs/i01') or run == ROOT / 'runs/i01':
        raise ValueError('A new registered I01 batch directory is required.')
    return run


def _sync_folder(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _batch_release(run, counter=None):
    run = _folder(run)
    release = read(run / 'batch_release.json')
    authorization_path = run / 'inputs/authorization.json'
    registry_path = run / 'inputs/sample_registry.json'
    requests_path = run / 'inputs/acceptance_requests.json'
    required = CODE_FILES | CONFIG_FILES | {
        path.relative_to(ROOT).as_posix()
        for path in (authorization_path, registry_path, requests_path)}
    if (type(release) is not dict or release.get('release') is not True
            or type(release.get('file_hashes')) is not dict
            or not required <= set(release['file_hashes'])
            or release.get('allocation') != ALLOCATION
            or release.get('batch_allocation') != BATCH_ALLOCATION
            or not old._sha(release.get('counter_identity_sha256'))
            or release.get('authorization_file_sha256') != flow.file_hash(authorization_path)):
        raise Error('release_invalid', 'Complete approved batch, code and allocation release required.')
    if counter is not None and release['counter_identity_sha256'] != flow.digest(counter.identity):
        raise Error('release_invalid', 'Actual approved counter differs from the batch release.')
    for relative, expected in release['file_hashes'].items():
        if type(relative) is not str or not old._sha(expected):
            raise Error('release_invalid', 'Released files need relative paths and SHA-256 identities.')
        path = old._no_links(ROOT / relative)
        if (Path(relative).is_absolute() or not path.is_relative_to(ROOT)
                or flow.file_hash(path) != expected):
            raise Error('release_changed', 'Released code, inputs, knowledge or model changed.')
    authorization = read(authorization_path)
    approved = {'provider': 'DeepSeek', 'endpoint': 'https://api.deepseek.com/chat/completions',
                'model': 'deepseek-flash', 'thinking': {'type': 'disabled'}, 'temperature': 0,
                'max_flows': 12, 'max_attempts_per_flow': 5,
                'max_cost_per_flow_usd': '0.02', 'max_total_cost_usd': '0.24',
                'retries': 0, 'concurrency': 1}
    if any(authorization.get(key) != value for key, value in approved.items()):
        raise Error('release_invalid', 'This batch requires the exact twelve-flow USD .24 authorization.')
    expected = read(ROOT / 'configs/q01_deepseek_budget.json')
    expected.update(max_attempts=4, max_cost='0.018',
                    approved_counter_sha256=release['counter_identity_sha256'])
    if release.get('qa_budget') != expected:
        raise Error('release_invalid', 'QA allocation differs from the approved exact four/.018 budget.')
    registry = read(registry_path)
    if (type(registry) is not dict or type(registry.get('samples')) is not dict
            or len(registry['samples']) != 4):
        raise Error('release_invalid', 'The released four-sample registry is required.')
    for sample in registry['samples'].values():
        if (type(sample) is not dict or set(sample) != {'selected_row', 'file_hashes'}
                or sample['selected_row'].get('split') not in ('train', 'validation')
                or type(sample['file_hashes']) is not dict or not sample['file_hashes']
                or any(release['file_hashes'].get(path) != value
                       for path, value in sample['file_hashes'].items())):
            raise Error('release_invalid', 'Every registered training/validation sample and model must be frozen.')
    return {'release': release, 'registry': registry,
            'profile': read(ROOT / 'configs/q01_deepseek.json'),
            'intent_config': read(ROOT / 'configs/q01_intent.json'),
            'prompts': read(ROOT / 'configs/q01_intent_prompts.json')}


def check_release(run, counter):
    """Validate all static batch controls before any image inference."""
    return _batch_release(run, counter)


class LiveGate(wire.LiveGate):
    """A batch release plus one claimed submission and actual image prediction."""
    def __init__(self, run, counter, actual_classifier, *, batch_run=None, actual_submission):
        self.run = old._no_links(run)
        self.batch_run = _folder(batch_run or self.run.parent.parent)
        if self.run.parent != self.batch_run / 'cases' or self.run.name not in {
                f'{number:03d}' for number in range(1, 13)}:
            raise ValueError('A claimed numbered case in this batch is required.')
        self.counter = counter
        self.release_path = self.batch_run / 'batch_release.json'
        self._release_bytes = old._regular_bytes(self.release_path)
        self.release = flow.k02.strict_json(self._release_bytes)
        self._freeze_bytes = self._derived_binding = self._derived_hashes = None
        self._actual_classifier = flow.json_snapshot(actual_classifier)
        self._actual_submission = flow.json_snapshot(actual_submission)
        self._case_hashes = {name: flow.file_hash(self.run / name) for name in (
            'flow_claim.json', 'inputs/submission.json', 'inputs/source_registration.json',
            'inputs/sample_registration.json', 'classifier_record.json', 'inputs/four_fields.json')}
        self.check()

    def check(self):
        import i01_knowledge as knowledge
        if old._regular_bytes(self.release_path) != self._release_bytes:
            raise Error('release_changed', 'Batch release bytes changed.')
        checked = check_release(self.batch_run, self.counter)
        self.release = checked['release']
        if any(flow.file_hash(self.run / name) != expected for name, expected in self._case_hashes.items()):
            raise Error('request_identity_mismatch', 'Claim, registered source or actual prediction changed.')
        claim = read(self.run / 'flow_claim.json')
        ledger = old._regular_bytes(self.batch_run / 'batch_ledger.jsonl')
        rows = [flow.k02.strict_json(line) for line in ledger.splitlines()]
        claims = [row for row in rows if row.get('event') == 'claim_flow'
                  and row.get('case_id') == self.run.name]
        if (len(claims) != 1 or claim != claims[0]
                or claim.get('batch_release_sha256') != hashlib.sha256(self._release_bytes).hexdigest()
                or claim.get('case_path') != self.run.relative_to(ROOT).as_posix()):
            raise Error('batch_claim_invalid', 'Case must belong to one durable nonrefundable batch claim.')
        registration = read(self.run / 'inputs/source_registration.json')
        source_path = self.run / 'inputs/submission.json'
        source_raw = old._regular_bytes(source_path)
        if (type(registration) is not dict
                or set(registration) != {'path', 'purpose', 'actual_sha256'}
                or registration['path'] != source_path.relative_to(ROOT).as_posix()
                or type(registration['purpose']) is not str or not registration['purpose'].strip()
                or registration['actual_sha256'] != hashlib.sha256(source_raw).hexdigest()):
            raise Error('request_identity_mismatch', 'Submission source must be registered before parsing.')
        # No source parsing takes place until its registration and bytes agree.
        source = flow.k02.strict_json(source_raw)
        if source != self._actual_submission:
            raise Error('request_identity_mismatch', 'Registered submission differs from the actual page input.')
        if type(source) is not dict or set(source) != {
                'question', 'explicitly_allowed_operating_context', 'sample_choice'}:
            raise Error('request_identity_mismatch', 'Exactly question, allowed context and sample choice are required.')
        choice = source['sample_choice']
        if type(choice) is not str or choice not in checked['registry']['samples']:
            raise Error('sample_binding_invalid', 'Only the released sample choices are allowed.')
        sample = read(self.run / 'inputs/sample_registration.json')
        if sample != checked['registry']['samples'][choice]:
            raise Error('sample_binding_invalid', 'Case sample differs from the selected released registry row.')
        actual = self._actual_classifier
        if (read(self.run / 'classifier_record.json') != actual
                or actual.get('kind') != 'fresh_image_inference'
                or actual.get('sample_id') != sample['selected_row']['sample_id']
                or actual.get('predicted_class') not in knowledge.SCENARIO_IDS):
            raise Error('classifier_binding_invalid', 'Only the actual supported image prediction may reach QA.')
        four = {'predicted_class': actual['predicted_class'],
                'scenario_id': knowledge.SCENARIO_IDS[actual['predicted_class']],
                'question': source['question'],
                'explicitly_allowed_operating_context': source['explicitly_allowed_operating_context']}
        try:
            self.projection = intent.validate_input(read(self.run / 'inputs/four_fields.json'))
        except Exception:
            raise Error('request_identity_mismatch', 'The business projection is malformed.') from None
        if self.projection != four:
            raise Error('request_identity_mismatch', 'Business fields must come from this question and actual prediction.')
        self.release['four_fields_sha256'] = flow.digest(four)
        self.release['four_fields_file_sha256'] = self._case_hashes['inputs/four_fields.json']
        self.release['actual_classifier_record_sha256'] = self._case_hashes['classifier_record.json']
        self.profile, self.intent_config, self.prompts = checked['profile'], checked['intent_config'], checked['prompts']
        if (self._freeze_bytes is not None
                and old._regular_bytes(self.run / 'master_first_request_freeze.json') != self._freeze_bytes):
            raise Error('freeze_changed', 'First-request master freeze changed.')


class IntentHTTP(wire.IntentHTTP):
    def __init__(self, profile, config, prompts, slot, gate, *, token_counter,
                 http_transport=None, test_credential=None):
        old.OneShotDeepSeekHTTP.__init__(self, profile, slot, token_counter=token_counter,
                                        http_transport=http_transport, test_credential=test_credential)
        intent.ModelTaskRouter(profile, config, prompts, lambda *_: None)
        if self.mode == 'real_http' and type(gate) is not LiveGate:
            raise ValueError('Real batch intent requires the exact new I01 gate.')
        self.intent_config, self.prompts = flow.json_snapshot(config), flow.json_snapshot(prompts)
        self.gate, self._lock = gate, threading.Lock()


class DerivedRequestHTTP(wire.DerivedRequestHTTP):
    def __init__(self, *args, gate, **kwargs):
        old.OneShotDeepSeekHTTP.__init__(self, *args, **kwargs)
        self.gate = gate
        if self.mode == 'real_http' and type(gate) is not LiveGate:
            raise ValueError('Real batch QA requires the exact new I01 gate.')

    def _context(self, body, context):
        # The accepted Q01 guards are retained here with I01's own prompt pin.
        self.gate.require_derived()
        context = flow.json_snapshot(context)
        required = {'request_sha256', 'business_request_sha256', 'prompt_bundle_sha256',
                    'schema_sha256', 'code_hashes'}
        if type(context) is not dict or set(context) != required:
            raise Error('freeze_context_invalid', 'Complete QA context required.')
        hashes = {str(p.relative_to(ROOT)): flow.file_hash(p) for p in old.CODE_PATHS}
        prompt_hash = flow.file_hash(ROOT / QA_PROMPT_FILE)
        if (context['code_hashes'] != hashes or context['schema_sha256'] != flow.digest(old.SCHEMAS)
                or context['prompt_bundle_sha256'] != prompt_hash
                or self.gate.release['file_hashes'].get(QA_PROMPT_FILE) != prompt_hash
                or self.budget.config != self.gate.release['qa_budget']):
            raise Error('freeze_context_invalid', 'I01 QA code/prompt/schema/budget differs from release.')
        path = self.gate.run / 'derived_request.json'
        raw = old._regular_bytes(path)
        actual_hash = hashlib.sha256(raw).hexdigest()
        self.budget._append('input_registration', {
            'path': str(path), 'purpose': 'Only the request derived from this actual task-model result; hash registered before parse.',
            'actual_sha256': actual_hash})
        if context['request_sha256'] != actual_hash or self.budget.request_sha256 != actual_hash:
            raise Error('request_identity_mismatch', 'Derived request bytes changed.')
        request = flow.k02.strict_json(raw)
        flow.k02.validate_request(request)
        report = read(self.gate.run / 'recognition_record.json')
        model = intent.validate_report(report['model_report'])
        business = flow.k02.strict_json(body['messages'][1]['content'])['request']
        if (report['status'] != 'ready' or model['status'] != 'identified' or report['request'] != request
                or report['source_four_fields_sha256'] != self.gate.release['four_fields_sha256']
                or {k: request[k] for k in intent.rules.INPUT_FIELDS} != self.gate.projection
                or request['task_types'] != [t for t in flow.k02.TASKS if t in model['task_types']]
                or business != request or context['business_request_sha256'] != flow.digest(request)):
            raise Error('request_identity_mismatch', 'QA is not bound to the actual model-derived tasks and original four fields.')
        context['code_hashes'] = dict(hashes, **{p: self.gate.release['file_hashes'][p] for p in CODE_FILES})
        return context


class DerivedQAProvider(wire.DerivedQAProvider):
    def __init__(self, config, budget, request_sha256, *, gate, token_counter,
                 http_transport=None, test_credential=None):
        old.DeepSeekHTTPProvider.__init__(self, config, budget, 'GV', request_sha256,
                                         token_counter=token_counter, http_transport=http_transport,
                                         test_credential=test_credential)
        self._http = DerivedRequestHTTP(config, budget, gate=gate, token_counter=token_counter,
                                       http_transport=http_transport, test_credential=test_credential)
        self.gate = gate

    def _bridge(self, action, payload, settings):
        prompt_path = ROOT / QA_PROMPT_FILE
        prompts = read(prompt_path)
        if self._http.mode == 'real_http' and payload['instructions'] != [prompts['common'], prompts[action]]:
            raise Error('prompt_identity_mismatch', 'Actual I01 prompt differs from the registered complete prompt bundle.')
        self._active_action = action
        self._active_context = {
            'request_sha256': self._request_sha256,
            'business_request_sha256': flow.digest(payload['request']),
            'prompt_bundle_sha256': hashlib.sha256(old._regular_bytes(prompt_path)).hexdigest(),
            'schema_sha256': flow.digest(old.SCHEMAS),
            'code_hashes': {str(p.relative_to(ROOT)): flow.file_hash(p) for p in old.CODE_PATHS}}
        count = len(self._official_attempts)
        try:
            return old.codec.DeepSeekChatProvider._bridge(self, action, payload, settings)
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


IntentSlot, ChainQABudget = base.IntentSlot, base.ChainQABudget


class BatchBudget:
    """Exclusive append-only batch credits, with conservative verified recovery.

    Every ledger line has a separately exclusive/fsynced hash checkpoint. An
    interrupted append/claim is uncertain and disables recovery, never resets it.
    """
    def __init__(self, run):
        self.run = _folder(run)
        checked = _batch_release(self.run)
        self.release_sha256 = flow.file_hash(self.run / 'batch_release.json')
        self._release_bytes = old._regular_bytes(self.run / 'batch_release.json')
        self.ledger_path = self.run / 'batch_ledger.jsonl'
        self.anchors = self.run / 'batch_ledger_hashes'
        self._lock = threading.RLock()
        self._closed = self._broken = False
        self._sequence, self._ledger_sha = 0, hashlib.sha256(b'').hexdigest()
        self._claims, self._finished = {}, {}
        self._fd = self._owner_fd = None
        try:
            self._owner_fd = os.open(self.run / 'batch_owner.lock',
                                     os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            fcntl.flock(self._owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owned = os.fstat(self._owner_fd)
            self._owner_identity = (owned.st_dev, owned.st_ino)
            new = not self.ledger_path.exists()
            identity_path = self.run / 'batch_ledger_identity.json'
            if new:
                if (identity_path.exists() or self.anchors.exists()
                        or ((self.run / 'cases').exists() and any((self.run / 'cases').iterdir()))):
                    raise ValueError('Missing ledger with existing batch evidence; recovery is forbidden.')
                self._fd = os.open(self.ledger_path,
                                   os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            else:
                self._fd = os.open(self.ledger_path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW)
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owned = os.fstat(self._fd)
            self._file_identity = (owned.st_dev, owned.st_ino)
            identity = {'device': owned.st_dev, 'inode': owned.st_ino,
                        'batch_release_sha256': self.release_sha256}
            if new:
                self.anchors.mkdir(mode=0o700, exist_ok=False)
                (self.run / 'cases').mkdir(mode=0o700, exist_ok=True)
                put(identity_path, identity)
                _sync_folder(self.run)
                self._append('init', {'allocation': BATCH_ALLOCATION,
                                     'batch_release_sha256': self.release_sha256,
                                     'ledger_policy': 'No credit refund or case replay; uncertain recovery fails closed.'})
            else:
                if read(identity_path) != identity:
                    raise ValueError('Batch ledger inode/release identity changed.')
                self._restore()
        except Exception:
            self._broken = True
            for descriptor in (self._fd, self._owner_fd):
                if descriptor is not None:
                    os.close(descriptor)
            self._closed = True
            raise Error('batch_unusable', 'Batch owner, approval or durable ledger is unavailable; no reset allowed.') from None

    def _live(self):
        if self._closed or self._broken:
            raise Error('batch_unusable', 'Batch is closed or poisoned.')
        if old._regular_bytes(self.run / 'batch_release.json') != self._release_bytes:
            self._broken = True
            raise Error('release_changed', 'Batch release changed after opening its budget.')

    def _read_owned(self):
        for path, descriptor, identity in (
                (self.ledger_path, self._fd, self._file_identity),
                (self.run / 'batch_owner.lock', self._owner_fd, self._owner_identity)):
            current, owned = path.lstat(), os.fstat(descriptor)
            if (not stat.S_ISREG(current.st_mode) or not stat.S_ISREG(owned.st_mode)
                    or (current.st_dev, current.st_ino) != identity
                    or (owned.st_dev, owned.st_ino) != identity):
                raise OSError('Durable batch owner identity changed.')
        raw = old._regular_bytes(self.ledger_path)
        after = self.ledger_path.lstat()
        if (after.st_dev, after.st_ino) != self._file_identity:
            raise OSError('Durable batch ledger changed during readback.')
        return raw

    def _append(self, event, data):
        self._live()
        try:
            before = self._read_owned()
            if hashlib.sha256(before).hexdigest() != self._ledger_sha:
                raise OSError('Batch ledger bytes changed.')
            row = dict(flow.json_snapshot(data), seq=self._sequence + 1, event=event,
                       previous_sha256=self._ledger_sha,
                       recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            blob = (flow.canonical(row) + '\n').encode('ascii')
            offset = 0
            while offset < len(blob):
                written = os.write(self._fd, blob[offset:])
                if written <= 0:
                    raise OSError('Short batch ledger write.')
                offset += written
            os.fsync(self._fd)
            after = self._read_owned()
            if after != before + blob:
                raise OSError('Batch ledger full readback mismatch.')
            digest = hashlib.sha256(after).hexdigest()
            put(self.anchors / f'{row["seq"]:06d}.json', {'seq': row['seq'], 'ledger_sha256': digest})
            _sync_folder(self.anchors)
            self._sequence, self._ledger_sha = row['seq'], digest
            return row
        except Exception:
            self._broken = True
            raise Error('batch_ledger_failed', 'Batch durable append failed; credit state cannot be reset.') from None

    def _restore(self):
        raw = self._read_owned()
        if not raw or not raw.endswith(b'\n'):
            raise ValueError('Empty or partial batch ledger cannot be recovered.')
        prefix = b''
        for sequence, line in enumerate(raw.splitlines(keepends=True), 1):
            row = flow.k02.strict_json(line)
            if (row.get('seq') != sequence or row.get('previous_sha256') != hashlib.sha256(prefix).hexdigest()
                    or (flow.canonical(row) + '\n').encode('ascii') != line):
                raise ValueError('Batch sequence, hash chain or canonical bytes changed.')
            prefix += line
            if read(self.anchors / f'{sequence:06d}.json') != {
                    'seq': sequence, 'ledger_sha256': hashlib.sha256(prefix).hexdigest()}:
                raise ValueError('Missing or changed durable batch hash checkpoint.')
            event = row['event']
            if sequence == 1:
                if (event != 'init' or row.get('allocation') != BATCH_ALLOCATION
                        or row.get('batch_release_sha256') != self.release_sha256):
                    raise ValueError('Original approved batch initialization required.')
            elif event == 'claim_flow':
                expected_id = f'{len(self._claims) + 1:03d}'
                case = self.run / 'cases' / expected_id
                if (len(self._claims) >= 12 or row.get('case_id') != expected_id
                        or row.get('case_path') != case.relative_to(ROOT).as_posix()
                        or row.get('credit_usd') != '0.02'
                        or row.get('batch_release_sha256') != self.release_sha256
                        or read(case / 'flow_claim.json') != row):
                    raise ValueError('Batch claim is incomplete, duplicated or changed.')
                self._claims[expected_id] = row
            elif event == 'finish_flow':
                case_id = row.get('case_id')
                if case_id not in self._claims or case_id in self._finished:
                    raise ValueError('Finish has no unique preceding batch claim.')
                self._validate_finish(row)
                self._finished[case_id] = row
            else:
                raise ValueError('Unknown batch ledger event.')
        if set(path.name for path in self.anchors.iterdir()) != {
                f'{number:06d}.json' for number in range(1, sequence + 1)}:
            raise ValueError('Unexpected durable batch hash checkpoints.')
        if set(path.name for path in (self.run / 'cases').iterdir()) != set(self._claims):
            raise ValueError('Unregistered case directories cannot be recovered.')
        self._sequence, self._ledger_sha = sequence, hashlib.sha256(raw).hexdigest()

    def claim_flow(self):
        with self._lock:
            self._live()
            if len(self._claims) >= 12:
                raise Error('batch_budget_exhausted', 'All twelve nonrefundable flow credits are consumed.')
            case_id = f'{len(self._claims) + 1:03d}'
            case = self.run / 'cases' / case_id
            row = self._append('claim_flow', {'case_id': case_id,
                               'case_path': case.relative_to(ROOT).as_posix(),
                               'credit_usd': '0.02', 'no_refunds': True,
                               'batch_release_sha256': self.release_sha256})
            self._claims[case_id] = row
            try:
                case.mkdir(mode=0o700, exist_ok=False)
                (case / 'inputs').mkdir(mode=0o700, exist_ok=False)
                put(case / 'flow_claim.json', row)
                _sync_folder(case)
                _sync_folder(self.run / 'cases')
            except Exception:
                self._broken = True
                raise Error('batch_claim_failed', 'Flow credit was consumed; incomplete claim disables this batch.') from None
            return case

    @staticmethod
    def _validate_finish(row):
        if type(row.get('status')) is not str or not row['status'].strip():
            raise ValueError('Explicit nonblank terminal status required.')
        for key, ceiling in (('identify_attempts', 1), ('qa_attempts', 4)):
            count = row.get(key)
            if count != 'unknown' and (type(count) is not int or not 0 <= count <= ceiling):
                raise ValueError('Attempt count must fit the original one/four ceilings or be unknown.')

    def finish_flow(self, case, status, *, identify_attempts=0, qa_attempts=0):
        with self._lock:
            self._live()
            case = old._no_links(case)
            case_id = case.name
            if (case.parent != self.run / 'cases' or case_id not in self._claims
                    or case_id in self._finished):
                raise Error('batch_claim_invalid', 'Only this unique unfinished claim may finish.')
            details = {'case_id': case_id, 'status': status, 'identify_attempts': identify_attempts,
                       'qa_attempts': qa_attempts, 'credit_refunded': False}
            self._validate_finish(details)
            row = self._append('finish_flow', details)
            self._finished[case_id] = row
            return flow.json_snapshot(row)

    def snapshot(self):
        with self._lock:
            self._live()
            try:
                if hashlib.sha256(self._read_owned()).hexdigest() != self._ledger_sha:
                    raise OSError('Batch ledger changed before snapshot.')
            except Exception:
                self._broken = True
                raise Error('batch_ledger_failed', 'Batch ledger integrity failed; no reset permitted.') from None
            known = len(self._claims) == len(self._finished) and all(
                row['identify_attempts'] != 'unknown' and row['qa_attempts'] != 'unknown'
                for row in self._finished.values())
            actual = sum(row['identify_attempts'] + row['qa_attempts']
                         for row in self._finished.values()) if known else 'unknown'
            return flow.json_snapshot({'allocation': BATCH_ALLOCATION,
                'claimed_flows': len(self._claims), 'finished_flows': len(self._finished),
                'remaining_flows': 12 - len(self._claims),
                'reserved_credit_usd': f'{len(self._claims) * 2 / 100:.2f}',
                'reserved_attempt_ceiling': len(self._claims) * 5, 'actual_attempts': actual,
                'unfinished_cases': sorted(set(self._claims) - set(self._finished)),
                'ledger_path': str(self.ledger_path), 'ledger_sha256': self._ledger_sha,
                'closed': self._closed, 'broken': self._broken})

    def close(self):
        with self._lock:
            if not self._closed:
                for descriptor in (self._fd, self._owner_fd):
                    if descriptor is not None:
                        os.close(descriptor)
                self._closed = True
