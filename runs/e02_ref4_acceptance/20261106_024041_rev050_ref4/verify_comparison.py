"""Read-only final verification of real common-REF execution, contrasts and protection."""
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import difflib
import hashlib
import importlib.util
import json
import sys

RUN = Path(__file__).absolute().parent
S = json.loads((RUN / 'settings.json').read_bytes())
PROJECT_ROOT = RUN.parents[2]
for key in ("project", "external", "run", "output", "source_ready", "old_answers", "L_execution", "metadata_helper", "acceptance_run"):
    if key in S and not Path(S[key]).is_absolute():
        S[key] = str(PROJECT_ROOT / S[key])
P, READY, OUTPUT = Path(S['project']), Path(S['run']), Path(S['output'])
spec = importlib.util.spec_from_file_location('metadata_only', S['metadata_helper'])
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
spec = importlib.util.spec_from_file_location('five_group_math', RUN / 'analyze_results.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
CONTROLS = {'final_manifest.json', 'final_verification.json'}


def verify(sealed=True):
    V.check(sys.dont_write_bytecode and sys.flags.optimize == 0, 'Python -B without -O')
    for name, obj in A.compute().items():
        V.check(V.read(RUN / name) == obj, 'independent arithmetic/evidence differs: ' + name)
    stats = V.read(RUN / 'reference_statistics.json')
    contributions = []
    for target, baseline_group in A.CONTRASTS:
        for metric in A.ROLES:
            for split in A.SPLITS:
                targets = [q for q in stats['questions'] if q['group'] == target and q['metric'] == metric and q['split'] == split and q['denominator']]
                base = {q['question_id']: q for q in stats['questions'] if q['group'] == baseline_group and q['metric'] == metric and q['split'] == split and q['denominator']}
                grouped = defaultdict(list)
                for q in targets:
                    grouped[q['family_id']].append(q)
                for family, qs in grouped.items():
                    weight = Fraction(1, len(grouped) * len(qs))
                    for q in qs:
                        other = base[q['question_id']]
                        low = (Fraction(q['exact_lower']) - Fraction(other['exact_upper'])) * weight * 100
                        high = (Fraction(q['exact_upper']) - Fraction(other['exact_lower'])) * weight * 100
                        contributions.append({'contrast': target + '-' + baseline_group, 'metric': metric, 'split': split,
                                              'question_id': q['question_id'], 'family_id': family, 'macro_weight': str(weight),
                                              'contribution_lower_pp': float(low), 'contribution_upper_pp': float(high)})
    V.check(contributions == V.read(RUN / 'contrast_contributions.json'), 'exact paired per-question contrast decomposition')
    for row in V.read(RUN / 'contrasts.json'):
        selected = [x for x in contributions if x['contrast'] == row['contrast'] and x['metric'] == row['metric'] and x['split'] == row['split']]
        if selected:
            V.check(abs(sum(x['contribution_lower_pp'] for x in selected) - row['lower_pp']) < 1e-10 and
                    abs(sum(x['contribution_upper_pp'] for x in selected) - row['upper_pp']) < 1e-10, 'contributions sum to full contrast')
    baseline = V.read(RUN / 'evidence/baseline.json')
    compat = V.read(RUN / 'compatibility.json')
    V.check(baseline['exact_active_demo_exclusions'] == compat['exact_active_demo_exclusions'] == V.EXCLUSIONS, 'no exclusion expansion')
    V.check(len(baseline['files']) == 34643 and len(baseline['links']) == 11, 'old baseline denominator')
    V.check(hashlib.sha256((RUN / 'evidence/baseline.json').read_bytes()).hexdigest() == compat['source_baseline_sha256'], 'baseline hash')
    for name in ['baseline.json', 'plan_before.md', 'prior_acceptance.stdout.json', 'prior_acceptance.stderr.log',
                 'verify_ready.stdout.json', 'verify_ready.stderr.log', 'launch_receipt.json',
                 'execute.stdout.log', 'execute.stderr.log', 'execution_outcome.json', 'post_execution_history.json',
                 'execution_inventory.json', 'retrieval_availability.json', 'analysis.stdout.json', 'analysis.stderr.log']:
        V.check((RUN / 'evidence' / name).read_bytes() == (Path(S['external']) / name).read_bytes(), 'exact external evidence: ' + name)
    inventory = V.read(RUN / 'evidence/execution_inventory.json')
    files, links = V.scan(OUTPUT)
    V.exact(files, inventory['files'], 'all new execution file bytes/identity/metadata')
    V.exact(links, inventory['links'], 'all new execution links')
    V.check(V.record(OUTPUT) == inventory['directory_record'], 'new consumed directory identity')
    ready_files, ready_links = V.scan(READY)
    original_ready = V.read(READY / 'ready_manifest.json')
    V.check(set(ready_files) == set(original_ready['files']) | {'ready_manifest.json', 'ready_receipt.json'}, 'ready exact self set')
    V.exact({k: ready_files[k] for k in original_ready['files']}, original_ready['files'], 'unchanged new scoring package')
    V.exact(ready_links, original_ready['links'], 'new scoring package links')
    V.check(V.read(READY / 'ready_receipt.json')['manifest_record'] == V.record(READY / 'ready_manifest.json'), 'new scoring package receipt')
    V.check(V.read(RUN / 'evidence/verify_ready.stdout.json')['status'] == 'PASS', 'current prelaunch passed')
    launch = V.read(RUN / 'evidence/launch_receipt.json')
    outcome = V.read(RUN / 'evidence/execution_outcome.json')
    V.check(launch['entry_invocations'] == 1 and not launch['output_existed_before'], 'unique new start')
    V.check(outcome['exit_code'] == 0 and outcome['entry_invocations'] == 1 and not outcome['resume_retry_reset'], 'closed once-only execution')
    post = V.read(RUN / 'evidence/post_execution_history.json')
    V.check(post['status'] == 'PASS' and post['old_files_unchanged'] and post['links_exact'], 'post-execution history protected')
    V.check(set(compat['changed_existing_files']) == {'AAE5103_项目计划与交接.md'}, 'only current plan changed')
    expected = dict(baseline['files'])
    for path, change in compat['changed_existing_files'].items():
        V.check(expected[path] == change['before'], 'before plan record')
        expected[path] = change['after']
    for group in ['added_ready_files', 'added_execution_files', 'added_docs']:
        V.check(not expected.keys() & compat[group].keys(), 'new addition shadows old file')
        expected.update(compat[group])
    own, own_links = V.scan(RUN)
    V.check(not own_links, 'no new audit links')
    if sealed:
        manifest, receipt = V.read(RUN / 'final_manifest.json'), V.read(RUN / 'final_verification.json')
        V.check(manifest['root'] == str(RUN) and set(manifest['control_files']) == CONTROLS, 'own exact root/controls')
        V.check(set(own) == set(manifest['files']) | CONTROLS, 'own exact set')
        V.exact({k: own[k] for k in manifest['files']}, manifest['files'], 'own exact bytes/identity/metadata')
        V.exact(own_links, manifest['links'], 'own exact links')
        V.check(receipt['status'] == 'PASS' and receipt['manifest_record'] == V.record(RUN / 'final_manifest.json'), 'own seal receipt')
    for name, rec in own.items():
        rel = str((RUN / name).relative_to(P))
        V.check(rel not in expected, 'audit shadows history')
        expected[rel] = rec
    actual, actual_links = V.scan(P, {P / x for x in V.EXCLUSIONS})
    V.exact(actual, expected, 'entire old tree and exact individually enumerated additions')
    V.exact(actual_links, baseline['links'], 'eleven protected old links')
    old = (RUN / 'before/AAE5103_项目计划与交接.md').read_bytes()
    latest_before = (RUN / 'evidence/latest_plan_before_sync.md').read_bytes()
    new = (P / 'AAE5103_项目计划与交接.md').read_bytes()
    V.check(new.decode().splitlines()[2] == '版本：v1.47｜状态修订号：050｜更新日期：2026-11-06（Asia/Tokyo）', 'latest plan revision')
    V.check(old == (RUN / 'evidence/plan_before.md').read_bytes(), 'complete original plan')
    V.check(old == (P / 'docs/handoffs/2026-11-06-049.md').read_bytes(), 'old049 full archive unchanged')
    V.check(new == (RUN / 'snapshots/AAE5103_项目计划与交接.md').read_bytes() == (P / S['new_handoff_archive']).read_bytes(), 'new full plan snapshots')
    expected_diff = ''.join(difflib.unified_diff(latest_before.decode().splitlines(keepends=True), new.decode().splitlines(keepends=True), fromfile='before/AAE5103_项目计划与交接.md', tofile='after/AAE5103_项目计划与交接.md'))
    V.check(expected_diff == (RUN / 'diffs/AAE5103_项目计划与交接.md.diff').read_text(), 'exact latest-plan diff')
    V.check((RUN / '结果与交接.md').read_bytes() == (P / S['new_handoff_doc']).read_bytes(), 'user-facing handoff exact')
    proof = V.read(RUN / 'execution_acceptance.json')
    return {'status': 'PASS', 'execution': proof, 'comparison': V.read(RUN / 'comparison_checks.json'),
            'history': {'old_regular_files': 34643, 'unchanged_during_execution': 34643,
                        'unchanged_after_sync': 34642, 'old_links_exact': True, 'old_links': 11,
                        'changed_existing_files': ['AAE5103_项目计划与交接.md'],
                        'added_ready_files': len(compat['added_ready_files']),
                        'added_execution_files': len(compat['added_execution_files']), 'added_docs': len(compat['added_docs']),
                        'acceptance_files': len(own), 'current_total_files': len(actual),
                        'exact_active_demo_exclusions': V.EXCLUSIONS,
                        'old_L_and_old52_raw_scores_ledger_flags_and_manifests_unchanged': True},
            'plan': S['new_plan_version'], 'self_seal_checked': sealed, 'pure_read_only': True,
            'model_calls_by_verifier': 0, 'causal_or_expert_truth_claim': False,
            'new_independent_agent_review': False,
            'readiness_verifier_recalled_after_consumption': False}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
