"""Pure read-only admission for the newly authorized old52 REF-only batch."""
from pathlib import Path
from decimal import Decimal
from collections import Counter as Bag
import datetime
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys

R = Path(__file__).absolute().parent
S = json.loads((R / 'settings.json').read_bytes())
PROJECT_ROOT = R.parents[2]
for key in ("project", "external", "run", "output", "source_ready", "old_answers", "L_execution", "metadata_helper", "acceptance_run"):
    if key in S and not Path(S[key]).is_absolute():
        S[key] = str(PROJECT_ROOT / S[key])
P = Path(S['project'])
spec = importlib.util.spec_from_file_location('metadata_only', S['metadata_helper'])
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def modules():
    sys.path.insert(0, str(R / 'runtime'))
    paths = {}
    for name in ['common', 'baseline', 'quote_anchor', 'rating_contract', 'provider', 'workflow']:
        module = importlib.import_module(name)
        V.check(Path(module.__file__).absolute() == R / 'runtime' / (name + '.py'), 'wrong module: ' + name)
        if hasattr(module, 'ROOT'):
            V.check(Path(module.ROOT).absolute() == R, 'wrong ROOT: ' + name)
        paths[name] = str(Path(module.__file__).absolute())
    return paths


def verify(sealed=True, credential=True):
    V.check(sys.dont_write_bytecode and sys.flags.optimize == 0, 'use existing Python -B without -O')
    config = V.read(R / 'config/execution.json')
    pin = V.read(R / 'config/execution_binding.json')
    for path, sha in pin['files'].items():
        V.check(V.record(P / path)['sha256'] == sha, 'frozen binding drift: ' + path)
    V.check(V.digest(config) == pin['configuration_sha256'], 'configuration hash')
    V.check(config['scope']['groups'] == ['T', 'TV', 'G', 'GV'] and config['scope']['saved_answer_rescoring'] == 52, 'only old52')
    V.check(config['scope']['L_rescoring'] == config['scope']['new_answers'] == 0, 'no new L rating or generation')
    slots = V.read(R / 'config/slots.json')
    V.check(len(slots) == 156 and V.digest(slots) == config['allowed_slots_sha256'], 'slot identity')
    V.check(dict(Bag(slots.values())) == {'judge_output': 104, 'adjudicate_output': 52}, 'role ceiling')
    V.check(config['max_calls'] == 156 and config['proposed_hard_budget_USD'] == '5.00', 'whole batch ceilings')
    sources = V.read(R / 'inputs/saved_answer_bindings.json')['sources']
    V.check(len(sources) == 52 and dict(Bag(s['group'] for s in sources)) == {'T': 13, 'TV': 13, 'G': 13, 'GV': 13}, 'all planned source groups')
    V.check(len({s['question_id'] for s in sources}) == 52 and len({s['original_question_id'] for s in sources}) == 13, 'cell/question distinction')
    old = Path(S['source_ready'])
    provenance = V.read(R / 'review/core_provenance.json')
    prior = V.read(Path(provenance['prior_independent_runtime_verdict']))
    V.check(prior['accepted_for_runtime'] is True and provenance['prior_independent_accepted'] is True, 'prior independent core acceptance')
    V.check(V.record(Path(provenance['prior_independent_runtime_verdict']))['sha256'] == provenance['prior_independent_verdict_sha256'], 'prior verdict drift')
    for rel, sha in provenance['exact_copied_assets'].items():
        V.check(V.record(R / rel)['sha256'] == sha == V.record(old / rel)['sha256'], 'core/contract changed: ' + rel)
    for source in sources:
        V.check(V.record(P / source['source_result_path'])['sha256'] == source['source_result_sha256'], 'old saved output changed')
    env = V.read(R / 'config/environment.json')
    V.check(sys.executable == env['executable'] and sys.version == env['python'] and platform.platform() == env['platform'], 'environment drift')
    V.check(all(importlib.metadata.version(k) == v for k, v in env['packages'].items()), 'package drift')
    checked = datetime.datetime.fromisoformat(config['price']['checked_at'])
    expiry = checked + datetime.timedelta(hours=config['price']['maximum_age_hours'])
    V.check(checked <= datetime.datetime.now(datetime.timezone.utc) <= expiry, 'original official price/route window expired')
    V.check(expiry.isoformat() == '2026-11-06T19:50:22.736344+09:00', 'price window must not extend')
    reserve = sum((Decimal(x['input_cap']) * Decimal(config['price']['input_per_million']) + Decimal(x['output_cap']) * Decimal(config['price']['output_per_million'])) / 1000000 * x['max_attempts'] for x in config['actions'].values())
    V.check(reserve == Decimal('4.6006272') <= Decimal('5.00'), 'all-cap internal guard')
    output = Path(S['output'])
    V.check(not output.exists() and not output.is_symlink(), 'output exists: consumed identity never reused')
    auth = V.read(R / 'authorization/authorization.json')
    V.check(auth['user_authorization_record'].encode() == (R / 'authorization/user_directive.txt').read_bytes(), 'actual user directive')
    V.check(auth['user_authorization_sha256'] == V.record(R / 'authorization/user_directive.txt')['sha256'], 'user directive hash')
    V.check(auth['approval_context_sha256'] == V.record(R / 'authorization/context.md')['sha256'], 'authorization context')
    for flag in ['paid_execution_authorized', 'formal_experiment_released', 'independent_cross_audit_accepted']:
        V.check(auth[flag] is True, 'unreleased: ' + flag)
    V.check(auth['scope'] == config['scope'] and auth['hard_budget_USD'] == '5.00' and auth['max_calls'] == 156 and auth['batch_output'] == S['output'], 'exact new scope approval')
    if credential:
        V.check(bool(os.environ.get(config['credential_environment_variable'], '').strip()), 'credential nonempty before consumption')
    paths = modules()
    from provider import LiveGate
    LiveGate(R, config, auth).check()
    if sealed:
        own, own_links = V.scan(R)
        manifest = V.read(R / 'ready_manifest.json')
        V.check(set(own) == set(manifest['files']) | {'ready_manifest.json', 'ready_receipt.json'}, 'readiness exact own set')
        V.exact({k: own[k] for k in manifest['files']}, manifest['files'], 'readiness own bytes/identity/metadata')
        V.exact(own_links, manifest['links'], 'readiness links')
        V.check(V.read(R / 'ready_receipt.json')['manifest_record'] == V.record(R / 'ready_manifest.json'), 'readiness receipt')
        baseline = V.read(R / 'evidence/baseline.json')
        V.check(baseline['exact_active_demo_exclusions'] == V.EXCLUSIONS, 'no exclusion expansion')
        expected = dict(baseline['files'])
        for name, rec in own.items():
            rel = str((R / name).relative_to(P))
            V.check(rel not in expected, 'new package shadows history')
            expected[rel] = rec
        actual, links = V.scan(P, {P / x for x in V.EXCLUSIONS})
        V.exact(actual, expected, 'all protected history and enumerated new package')
        V.exact(links, baseline['links'], 'all protected links')
    return {'status': 'PASS', 'source_units': 52, 'groups': ['T', 'TV', 'G', 'GV'],
            'same_six_runtime_modules_and_same_REF_contract': True, 'module_paths': paths,
            'review_identity': 'prior independent unchanged core reused; new adapter/metadata deterministic separate-process review, no new independent agent',
            'price_expiry_JST': expiry.isoformat(), 'output_identity_consumed': False,
            'credential_nonempty_checked': credential, 'real_model_calls': 0, 'pure_read_only': True}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
