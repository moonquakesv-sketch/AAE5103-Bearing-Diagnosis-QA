"""I01-only constructor adapters; the accepted Q01 send/budget methods stay inherited.

No network, credential access or records at import. The new release binds one
registered image and the original development question, never a general input.
"""
import fcntl
import hashlib
import os
import threading
from decimal import Decimal

import q01_flow as flow
import q01_intent as intent
import q01_intent_http as wire
import q01_deepseek_http as old

ROOT = old.ROOT
Error = wire.Error
put = wire.put
read = wire.read
CODE_FILES = wire.CODE_FILES | {'scripts/i01_http.py', 'scripts/i01_page.py'}
CONFIG_FILES = wire.CONFIG_FILES
ALLOCATION = wire.ALLOCATION
SCENARIO_ID = 'SC-K01-OR-001'


def check_release(run, counter):
    """Check the offline release before inference, without any classifier output."""
    run = old._no_links(run)
    if not run.is_relative_to(ROOT / 'runs/i01'):
        raise ValueError('A registered I01 run is required.')
    release = read(run / 'live_release.json')
    sample_path = run / 'inputs/sample_registration.json'
    source_path = run / 'inputs/source_registration.json'
    approved_path = run / 'inputs/approved_content.json'
    if (flow.file_hash(sample_path) != release['sample_registration_file_sha256']
            or flow.file_hash(source_path) != release['source_registration_file_sha256']):
        raise Error('release_changed', 'Registered image or question source changed.')
    sample = read(sample_path)
    required = CODE_FILES | CONFIG_FILES | set(sample['file_hashes']) | {approved_path.relative_to(ROOT).as_posix()}
    if (release.get('release') is not True or not required <= set(release['file_hashes'])
            or release['allocation'] != ALLOCATION
            or release['counter_identity_sha256'] != flow.digest(counter.identity)
            or any(release['file_hashes'].get(p) != h for p, h in sample['file_hashes'].items())):
        raise Error('release_invalid', 'Incomplete I01 code/image/counter/allocation release.')
    for relative, expected_hash in release['file_hashes'].items():
        path = old._no_links(ROOT / relative)
        if not path.is_relative_to(ROOT) or '..' in path.parts or flow.file_hash(path) != expected_hash:
            raise Error('release_changed', 'Released I01 input, code, model or prompt changed.')
    registered = read(source_path)
    raw = old._regular_bytes(old.REQUEST_PATH)
    if (registered['actual_sha256'] != old.REQUEST_SHA256
            or hashlib.sha256(raw).hexdigest() != old.REQUEST_SHA256
            or registered['path'] != old.REQUEST_PATH.relative_to(ROOT).as_posix()):
        raise Error('request_identity_mismatch', 'Only the registered existing development question is allowed.')
    # Both registration and source bytes have now been checked, before parsing.
    source = flow.k02.strict_json(raw)
    approved = read(approved_path)
    expected_content = {k: source[k] for k in ('question', 'explicitly_allowed_operating_context')}
    if approved != expected_content:
        raise Error('request_identity_mismatch', 'Approved question/context differs from the original source.')
    expected = read(ROOT / 'configs/q01_deepseek_budget.json')
    expected.update(max_attempts=4, max_cost='0.018',
                    approved_counter_sha256=release['counter_identity_sha256'])
    if expected != release['qa_budget']:
        raise Error('release_invalid', 'QA allocation/prices differ from the approved budget.')
    return {'release': release, 'sample': sample, 'approved_content': approved,
            'profile': read(ROOT / 'configs/q01_deepseek.json'),
            'intent_config': read(ROOT / 'configs/q01_intent.json'),
            'prompts': read(ROOT / 'configs/q01_intent_prompts.json')}


class LiveGate(wire.LiveGate):
    """Frozen I01 release plus this page invocation's actual image prediction."""
    def __init__(self, run, counter, actual_classifier):
        self.run = old._no_links(run)
        if not self.run.is_relative_to(ROOT / 'runs/i01'):
            raise ValueError('A new I01 run is required.')
        self.counter = counter
        self.release_path = self.run / 'live_release.json'
        self._release_bytes = old._regular_bytes(self.release_path)
        self.release = flow.k02.strict_json(self._release_bytes)
        self._freeze_bytes = self._derived_binding = self._derived_hashes = None
        self._actual_classifier = flow.json_snapshot(actual_classifier)
        classifier_path = self.run / 'classifier_record.json'
        if (read(classifier_path) != self._actual_classifier
                or self._actual_classifier.get('kind') != 'fresh_image_inference'):
            raise Error('classifier_binding_invalid', 'Classifier record is not the actual image-inference result.')
        self._classifier_hash = flow.file_hash(classifier_path)
        self._projection_hash = flow.file_hash(self.run / 'inputs/four_fields.json')
        self.check()

    def check(self):
        if old._regular_bytes(self.release_path) != self._release_bytes:
            raise Error('release_changed', 'I01 release bytes changed.')
        checked = check_release(self.run, self.counter)
        self.release = checked['release']
        sample = checked['sample']
        if (flow.file_hash(self.run / 'classifier_record.json') != self._classifier_hash
                or read(self.run / 'classifier_record.json') != self._actual_classifier
                or self._actual_classifier['sample_id'] != sample['selected_row']['sample_id']
                or self._actual_classifier['predicted_class'] != 'outer_race'):
            raise Error('classifier_binding_invalid', 'Only this actual supported image prediction may reach QA.')
        four = {'predicted_class': self._actual_classifier['predicted_class'],
                'scenario_id': SCENARIO_ID, **checked['approved_content']}
        try:
            self.projection = intent.validate_input(read(self.run / 'inputs/four_fields.json'))
        except Exception:
            raise Error('request_identity_mismatch', 'I01 business projection is malformed.') from None
        if (self.projection != four
                or flow.file_hash(self.run / 'inputs/four_fields.json') != self._projection_hash):
            raise Error('request_identity_mismatch', 'Question/context or actual prediction projection changed.')
        # Existing derived-request checks use this identity after actual classification.
        self.release['four_fields_sha256'] = flow.digest(four)
        self.release['four_fields_file_sha256'] = self._projection_hash
        self.release['actual_classifier_record_sha256'] = self._classifier_hash
        self.profile, self.intent_config, self.prompts = checked['profile'], checked['intent_config'], checked['prompts']
        if (self._freeze_bytes is not None
                and old._regular_bytes(self.run / 'master_first_request_freeze.json') != self._freeze_bytes):
            raise Error('freeze_changed', 'I01 first-request master freeze changed.')


class ChainQABudget(wire.ChainQABudget):
    """Only namespace initialization differs; durable Q01 methods are inherited."""
    def __init__(self, run_dir, session_config, request_sha256):
        config = flow.json_snapshot(session_config)
        required = {'max_attempts', 'currency', 'max_cost', 'allowed_groups',
                    'model_prices', 'approved_counter_sha256'}
        registered = read(ROOT / 'configs/q01_deepseek_budget.json')
        if (type(config) is not dict or set(config) != required or old._sensitive_registration(config)
                or type(config['max_attempts']) is not int or config['max_attempts'] != 4
                or config['currency'] != 'USD' or config['max_cost'] != '0.018'
                or config['allowed_groups'] != registered['allowed_groups']
                or config['model_prices'] != registered['model_prices'] or not old._sha(request_sha256)
                or not old._sha(config['approved_counter_sha256'])):
            raise ValueError('Exact QA 4-attempt/USD.018 allocation and approved counter required.')
        run = old._no_links(run_dir)
        if not run.is_relative_to(ROOT / 'runs/i01'):
            raise ValueError('New I01 QA directory required.')
        if run.exists():
            if not run.is_dir() or any(run.iterdir()):
                raise ValueError('New or empty QA ledger directory required.')
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
        self._counts, self._flows, self._group_counts, self._denials = dict.fromkeys(old.TOTAL_LIMITS, 0), {}, {}, []
        self._freeze_path, self._freeze_bytes, self._freeze_binding = None, None, None
        self._append('init', {'config': config, 'request_sha256': request_sha256,
                              'ledger_policy': 'I01 single GV; inherited exclusive/fsync/no refund/no restart.'})


class IntentSlot(wire.IntentSlot):
    def __init__(self, folder, counter_sha256):
        self.folder = old._no_links(folder)
        if not self.folder.is_relative_to(ROOT / 'runs/i01') or not old._sha(counter_sha256):
            raise ValueError('New I01 intent directory and approved counter required.')
        self.folder.mkdir(exist_ok=False)
        self.config = {'approved_counter_sha256': counter_sha256}
        self.used, self.raw, self.identity = False, None, None
        self.reserved, self.broken = '0', False


class IntentHTTP(wire.IntentHTTP):
    def __init__(self, profile, config, prompts, slot, gate, *, token_counter,
                 http_transport=None, test_credential=None):
        old.OneShotDeepSeekHTTP.__init__(self, profile, slot, token_counter=token_counter,
                                        http_transport=http_transport, test_credential=test_credential)
        intent.ModelTaskRouter(profile, config, prompts, lambda *_: None)
        if self.mode == 'real_http' and type(gate) is not LiveGate:
            raise ValueError('Real I01 intent requires the exact I01 gate.')
        self.intent_config, self.prompts = flow.json_snapshot(config), flow.json_snapshot(prompts)
        self.gate, self._lock = gate, threading.Lock()


class DerivedRequestHTTP(wire.DerivedRequestHTTP):
    def __init__(self, *args, gate, **kwargs):
        old.OneShotDeepSeekHTTP.__init__(self, *args, **kwargs)
        self.gate = gate
        if self.mode == 'real_http' and type(gate) is not LiveGate:
            raise ValueError('Real I01 derived QA requires the exact I01 gate.')


class DerivedQAProvider(wire.DerivedQAProvider):
    def __init__(self, config, budget, request_sha256, *, gate, token_counter,
                 http_transport=None, test_credential=None):
        old.DeepSeekHTTPProvider.__init__(self, config, budget, 'GV', request_sha256,
                                         token_counter=token_counter, http_transport=http_transport,
                                         test_credential=test_credential)
        self._http = DerivedRequestHTTP(config, budget, gate=gate, token_counter=token_counter,
                                       http_transport=http_transport, test_credential=test_credential)
        self.gate = gate
