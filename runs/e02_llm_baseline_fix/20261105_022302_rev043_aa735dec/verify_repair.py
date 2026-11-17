"""Read-only evidence checks, with exact history/self inventory; never launches runtime."""
from pathlib import Path
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import sys

R = Path(__file__).absolute().parent
P = R.parents[2]
OLD = P / 'runs/e02_llm_baseline_preparation/20261105_003337_rev042_3b109719'
CONTROLS = {'final_manifest.json', 'final_verification.json'}
EXCLUSIONS = ['runs/i01/20261104_184700_live_demo_fcb6b670/runtime', 'runs/i01/20261104_184700_live_demo_fcb6b670/executions']

def safe(path, link=False):
    assert path.is_absolute() and '..' not in path.parts and path.is_relative_to(P)
    assert not any(p.is_symlink() for p in path.parents)
    if not link:
        assert not path.is_symlink()
    return path

def record(path):
    safe(path, True)
    before = path.lstat()
    value = {'size': before.st_size, 'mode': before.st_mode, 'dev': before.st_dev, 'ino': before.st_ino, 'mtime_ns': before.st_mtime_ns}
    if stat.S_ISLNK(before.st_mode):
        value['target'] = os.readlink(path)
    else:
        assert stat.S_ISREG(before.st_mode)
        safe(path)
        h = hashlib.sha256()
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
            for chunk in iter(lambda: stream.read(1048576), b''):
                h.update(chunk)
        after = path.lstat()
        assert (before.st_size, before.st_mode, before.st_dev, before.st_ino, before.st_mtime_ns) == (after.st_size, after.st_mode, after.st_dev, after.st_ino, after.st_mtime_ns)
        value['sha256'] = h.hexdigest()
    return value

def read(path):
    return json.loads(safe(path).read_bytes())

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def scan(root, exclude=()):
    safe(root)
    files, links = {}, {}
    for base, directories, names in os.walk(root, followlinks=False):
        for name in list(directories) + names:
            path = Path(base) / name
            if path in exclude:
                if name in directories:
                    directories.remove(name)
                continue
            entry = path.lstat()
            rel = str(path.relative_to(root))
            if stat.S_ISLNK(entry.st_mode):
                links[rel] = record(path)
                if name in directories:
                    directories.remove(name)
            elif stat.S_ISREG(entry.st_mode):
                files[rel] = record(path)
            else:
                assert stat.S_ISDIR(entry.st_mode)
    return files, links

def exact(actual, expected, label):
    changes = {k: {'expected': expected[k], 'current': actual[k]} for k in set(actual) & set(expected) if actual[k] != expected[k]}
    assert actual == expected, {'scope': label, 'added': sorted(set(actual) - set(expected)), 'missing': sorted(set(expected) - set(actual)), 'changed': changes}

def history():
    baseline, compatibility = read(R / 'baseline.json'), read(R / 'compatibility.json')
    assert baseline['exact_active_demo_exclusions'] == EXCLUSIONS
    assert set(compatibility['changed_existing_files']) == {'AAE5103_项目计划与交接.md'}
    expected = dict(baseline['files'])
    for path, value in compatibility['changed_existing_files'].items():
        assert expected[path] == value['before']
        expected[path] = value['after']
    assert not set(compatibility['added_files']) & set(expected)
    expected.update(compatibility['added_files'])
    files, links = scan(P, {R} | {P / p for p in EXCLUSIONS})
    exact(files, expected, 'history ordinary files')
    exact(links, baseline['links'], 'history links')
    original_manifest = read(OLD / 'final_manifest.json')
    original_files, original_links = scan(OLD)
    assert set(original_files) == set(original_manifest['files']) | CONTROLS
    exact({k: original_files[k] for k in original_manifest['files']}, original_manifest['files'], 'original sealed ordinary files')
    exact(original_links, original_manifest['links'], 'original sealed links')
    assert read(OLD / 'final_verification.json')['manifest_record'] == record(OLD / 'final_manifest.json')
    return {'status': 'PASS', 'historical_regular_files': len(baseline['files']), 'historical_links': len(links), 'registered_existing_changes': 1, 'registered_added_docs': len(compatibility['added_files']), 'original_rev042_seal_preserved': True, 'exact_active_demo_exclusions': EXCLUSIONS}

def technical():
    binding = read(R / 'config/execution_binding.json')
    config = read(R / 'config/execution.json')
    for path, expected in binding['files'].items():
        assert record(P / path)['sha256'] == expected, path
    assert binding['configuration_sha256'] == digest(config)
    assert (R / 'config/execution.json').read_bytes() == (OLD / 'config/execution.json').read_bytes()
    assert (R / 'config/new_authorization_template.json').read_bytes() == (OLD / 'config/new_authorization_template.json').read_bytes()
    assert not Path(config['batch_output']).exists()
    for path in ['config/execution.json', 'config/new_authorization_template.json']:
        assert all(read(R / path)[k] is False for k in ['formal_experiment_released', 'independent_cross_audit_accepted', 'paid_execution_authorized'])
    changes = read(R / 'repair_changes.json')['runtime_changes']
    assert set(changes) == {'runtime/provider.py', 'runtime/workflow.py'}
    copies = read(R / 'inputs/asset_copy_manifest.json')
    for row in copies:
        assert record(P / row['source'])['sha256'] == row['sha256']
        expected = changes[row['copy']]['after_sha256'] if row['copy'] in changes else row['sha256']
        assert record(R / row['copy'])['sha256'] == expected
    after, before = read(R / 'validation/after/results.json'), read(R / 'validation/before_corrected/results.json')
    assert after['passed'] == after['total'] == 15
    assert after['test_source_sha256'] == before['test_source_sha256'] == record(R / 'tests/test_stop_contract.py')['sha256']
    assert before['passed'] == 4 and before['total'] == 15
    assert after['real_model_API_calls'] == 0 and after['credentials_read'] is False
    assert read(R / 'validation/evidence_review.json')['status'] == 'PASS'
    assert read(R / 'acceptance.json')['offline_repair_verified'] is True
    environment = read(R / 'config/environment.json')
    current = {'executable': sys.executable, 'python': sys.version, 'platform': platform.platform(), 'packages': {k: importlib.metadata.version(k) for k in environment['packages']}}
    assert all(current[k] == environment[k] for k in current), {'environment_changed': current}
    now = datetime.datetime.now(datetime.timezone.utc)
    checked = datetime.datetime.fromisoformat(config['price']['checked_at'])
    expiry = checked + datetime.timedelta(hours=config['price']['maximum_age_hours'])
    return {'status': 'PASS', 'technical_pins': len(binding['files']), 'configuration_sha256': digest(config), 'binding_sha256': record(R / 'config/execution_binding.json')['sha256'], 'business_configuration_byte_identical': True, 'original_template_byte_identical_false': True, 'asset_copies': len(copies), 'environment_exact_match': True, 'price_snapshot_within_original_window': checked <= now <= expiry, 'price_expiry_JST': expiry.isoformat(), 'same_oracle_tests': 'before4/15; after15/15', 'independent_agent_review_claimed': False, 'requires_new_explicit_user_approval': True}

def verify():
    manifest, receipt = read(R / 'final_manifest.json'), read(R / 'final_verification.json')
    files, links = scan(R)
    assert manifest['root'] == str(R) and set(manifest['control_files']) == CONTROLS
    assert set(files) == set(manifest['files']) | CONTROLS
    exact({k: files[k] for k in manifest['files']}, manifest['files'], 'repair self ordinary files')
    exact(links, manifest['links'], 'repair self links')
    assert receipt['manifest_record'] == record(R / 'final_manifest.json')
    assert receipt['status'] == 'PASS'
    return {'status': 'PASS', 'ordinary_files': len(files), 'links': len(links), 'self_exact_set_and_metadata': True, 'history': history(), 'technical': technical(), 'manifest_sha256': files['final_manifest.json']['sha256'], 'final_verification_sha256': files['final_verification.json']['sha256'], 'pure_read_only': True, 'model_API_calls': 0, 'paid_execution_authorized': False}

if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))

