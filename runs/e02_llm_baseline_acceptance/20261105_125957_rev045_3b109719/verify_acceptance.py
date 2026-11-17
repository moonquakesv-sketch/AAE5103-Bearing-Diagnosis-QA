"""Pure read-only audit of the closed batch, fixed denominators and exact history."""
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import difflib
import hashlib
import json
import os
import stat
import sys

RUN = Path(__file__).absolute().parent
PROJECT = RUN.parents[2]
PREP = PROJECT / 'runs/e02_llm_baseline_fix/20261105_022302_rev043_aa735dec'
RELEASE = PROJECT / 'runs/e02_llm_baseline_release/20261105_024326_rev044_be2ae061'
OUTPUT = PROJECT / 'runs/e02_llm_baseline_execution/20261105_rev042_L13R1_3b109719'
EXCLUSIONS = ['runs/i01/20261104_184700_live_demo_fcb6b670/runtime',
              'runs/i01/20261104_184700_live_demo_fcb6b670/executions']
CONTROLS = {'final_manifest.json', 'final_verification.json'}
ROLES = {'REF_cov': None, 'REF_knowledge': 'positive_knowledge',
         'REF_boundary': 'boundary', 'REF_source_state': 'source_status'}


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    check(not path.is_symlink(), 'symlink JSON: ' + str(path))
    return json.loads(path.read_bytes())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def record(path):
    check(not any(p.is_symlink() for p in path.parents), 'symlink parent')
    s = path.lstat()
    result = {'size': s.st_size, 'mode': s.st_mode, 'dev': s.st_dev,
              'ino': s.st_ino, 'mtime_ns': s.st_mtime_ns}
    if stat.S_ISREG(s.st_mode):
        h = hashlib.sha256()
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
            for block in iter(lambda: stream.read(1048576), b''):
                h.update(block)
        result['sha256'] = h.hexdigest()
    elif stat.S_ISLNK(s.st_mode):
        result['target'] = os.readlink(path)
    else:
        check(stat.S_ISDIR(s.st_mode), 'unexpected filesystem type')
    after = path.lstat()
    check((s.st_size, s.st_mode, s.st_dev, s.st_ino, s.st_mtime_ns) == (
        after.st_size, after.st_mode, after.st_dev, after.st_ino, after.st_mtime_ns), 'file changed while reading')
    return result


def scan(root, exclusions=()):
    files, links = {}, {}
    for base, directories, names in os.walk(root, followlinks=False):
        for name in list(directories) + names:
            p = Path(base) / name
            if p in exclusions:
                if name in directories:
                    directories.remove(name)
                continue
            if p.is_symlink():
                links[str(p.relative_to(root))] = record(p)
                if name in directories:
                    directories.remove(name)
            elif p.is_file():
                files[str(p.relative_to(root))] = record(p)
            else:
                check(p.is_dir(), 'unexpected filesystem entry')
    return files, links


def exact(actual, expected, label):
    if actual != expected:
        added = sorted(actual.keys() - expected.keys())
        missing = sorted(expected.keys() - actual.keys())
        changed = {k: {'expected': expected[k], 'actual': actual[k]}
                   for k in actual.keys() & expected.keys() if actual[k] != expected[k]}
        raise ValueError({'scope': label, 'added': added, 'missing': missing, 'changed': changed})


def recompute_ratio(scores):
    if len(scores) == 0:
        return {'status': 'not_applicable', 'value': None, 'denominator': 0, 'lower': None, 'upper': None}
    known = sum(Fraction(x) for x in scores if x is not None)
    missing = sum(x is None for x in scores)
    n = len(scores)
    return {'status': 'unknown' if missing else 'known', 'value': None if missing else float(known / n),
            'numerator': float(known), 'denominator': n, 'unknown': missing,
            'lower': float(known / n), 'upper': float((known + missing) / n)}


def arithmetic():
    bindings = read(PREP / 'inputs/business_bindings.json')['bindings']
    check(len({r['family_id'] for r in bindings}) == 9, 'planned family denominator')
    rubrics = {r['question_id']: r for r in read(PREP / 'inputs/rubrics.json')['questions']}
    units = read(RUN / 'unit_statuses.json')
    point_rows = read(RUN / 'fixed_point_statuses.json')
    stats = read(RUN / 'reference_statistics.json')
    check([r['question_id'] for r in units] == [r['question_id'] for r in bindings], 'planned question identity')
    check(len(units) == 13 and len(point_rows) == 63, 'fixed denominator lost')
    check(len({(p['question_id'], p['point_id']) for p in point_rows}) == 63, 'duplicate fixed point')
    scores = {(p['question_id'], p['point_id']): p for p in point_rows}
    questions = []
    totals = Counter()
    for binding, unit in zip(bindings, units):
        qid = binding['question_id']
        check(unit['family_id'] == binding['family_id'] and unit['split'] == binding['split'], 'split/family drift')
        check(unit['acceptance_projection'] and not unit['runtime_result_file_exists'], 'fabricated runtime record')
        check(unit['generated'] is None and unit['observed_text'] is None, 'fabricated answer')
        check(unit['generation'] == {'status': 'not_attempted_batch_startup_failed', 'valid': False}, 'generation state')
        for key in ['original_A', 'original_B', 'third']:
            check(unit[key] == {'status': 'not_attempted_batch_startup_failed', 'rating': None}, 'original scoring state')
        check(unit['final'] == {'status': 'unknown_not_evaluated', 'rating': None}, 'fabricated final rating')
        for point in rubrics[qid]['points']:
            actual = scores[(qid, point['point_id'])]
            check(actual['content_role'] == point['content_role'] and actual['score'] is None,
                  'fixed content role or unknown changed')
            check(actual['family_id'] == binding['family_id'] and actual['split'] == binding['split'], 'point family/split')
            totals['REF_cov'] += 1
            for name, role in ROLES.items():
                if role == point['content_role']:
                    totals[name] += 1
        ratios = {}
        for name, role in ROLES.items():
            values = [scores[(qid, p['point_id'])]['score'] for p in rubrics[qid]['points']
                      if role is None or p['content_role'] == role]
            ratios[name] = recompute_ratio(values)
            check(unit['metrics'][name] == ratios[name], 'question ratio: ' + qid + '/' + name)
        questions.append((binding, ratios))
        for name in ['REF_unsupported_claim', 'REF_voluntary_citation_error']:
            check(unit[name] == {'status': 'unknown', 'value': None, 'denominator': None}, 'invented dynamic denominator')
    check(dict(totals) == stats['fixed_point_totals'] == {
        'REF_cov': 63, 'REF_knowledge': 39, 'REF_boundary': 17, 'REF_source_state': 7}, 'fixed totals')
    for name in ROLES:
        for split in ['debug', 'operational_reserve']:
            grouped = defaultdict(list)
            for binding, ratios in questions:
                if binding['split'] == split:
                    grouped[binding['family_id']].append(ratios[name])
            famrows, bounds = [], []
            for family, values in sorted(grouped.items()):
                applicable = [x for x in values if x['denominator'] > 0]
                n = len(applicable)
                low = sum((Fraction(str(x['lower'])) for x in applicable), Fraction()) / n if n else None
                high = sum((Fraction(str(x['upper'])) for x in applicable), Fraction()) / n if n else None
                famrows.append({'family_id': family, 'planned_questions': len(values), 'applicable_questions': n,
                                'not_applicable_questions': len(values) - n,
                                'fixed_points': sum(x['denominator'] for x in applicable),
                                'status': 'unknown' if n else 'not_applicable', 'value': None,
                                'lower': float(low) if n else None, 'upper': float(high) if n else None})
                if n:
                    bounds.append((low, high))
            check(stats['families'][name][split] == famrows, 'family macro: ' + name + '/' + split)
            n = len(bounds)
            macro = {'status': 'unknown' if n else 'not_applicable', 'value': None, 'families': n,
                     'planned_families': len(grouped), 'planned_questions': sum(len(x) for x in grouped.values()),
                     'applicable_questions': sum(x['applicable_questions'] for x in famrows),
                     'not_applicable_questions': sum(x['not_applicable_questions'] for x in famrows),
                     'fixed_points': sum(x['fixed_points'] for x in famrows),
                     'lower': float(sum(x[0] for x in bounds) / n) if n else None,
                     'upper': float(sum(x[1] for x in bounds) / n) if n else None}
            check(stats['fixed_macros'][name][split] == macro, 'split macro: ' + name + '/' + split)
    dynamic = stats['dynamic_claim_and_citation_metrics']
    check(dynamic['status'] == 'unknown' and dynamic['denominator'] is None and dynamic['overall_mean'] is None,
          'invented overall dynamic mean')
    check(read(RUN / 'old38_applicability.json') == read(PREP / 'analysis/metric_applicability_38.json'), 'old38 applicability')
    return {'status': 'PASS', 'planned_questions': 13, 'fixed_points': 63, 'planned_families': 9,
            'question_metric_rows': 52, 'split_macro_rows': 8, 'all_fixed_scores_unknown': True,
            'N_A_preserved': True, 'dynamic_denominators_unknown': True, 'semantic_effect_proven': False}


def execution():
    config = read(PREP / 'config/execution.json')
    auth = read(RELEASE / 'authorization/authorization.json')
    slots = read(PREP / 'config/slots.json')
    inventory = read(RUN / 'execution_inventory.json')
    check(record(OUTPUT) == inventory['directory_record'], 'consumed directory identity drift')
    files, links = scan(OUTPUT)
    exact(files, inventory['files'], 'execution files')
    exact(links, inventory['links'], 'execution links')
    check(set(files) == {'authorization.consumed.json', 'ledger.jsonl'}, 'unexpected execution files')
    check(read(OUTPUT / 'authorization.consumed.json') == {'authorization_sha256': digest(auth), 'no_reuse': True}, 'consumption provenance')
    ledger = [json.loads(x) for x in (OUTPUT / 'ledger.jsonl').read_text().splitlines()]
    check(ledger == [{'config_sha256': digest(config), 'event': 'init',
                      'scope': 'fresh independent E02 ledger; no old ledger imported',
                      'sequence': 1, 'slots_sha256': digest(slots)}], 'ledger not init-only')
    check(read(RUN / 'evidence/verify_release.stdout.json')['status'] == 'PASS', 'prelaunch not PASS')
    check(not (RUN / 'evidence/verify_release.stderr.log').read_bytes(), 'prelaunch stderr')
    check(not (RUN / 'evidence/execute_approved.stdout.log').read_bytes(), 'unexpected execution stdout')
    stderr = (RUN / 'evidence/execute_approved.stderr.log').read_text()
    check(stderr.endswith('common.Rejected: credential not configured\n') and
          'transport=LiveHTTP(gate,config)' in stderr, 'failure stage not witnessed')
    provider_source = (PREP / 'runtime/provider.py').read_text()
    guard = "require(type(self._credential) is str and self._credential, 'credential not configured')"
    check(provider_source.index(guard) < provider_source.index('self.client = httpx.Client('), 'HTTP before failure guard')
    commands = read(RUN / 'evidence/commands.json')
    check(commands['execute']['argv'] == ['/Users/moonquakes/miniforge3/envs/aae5103/bin/python', '-B',
          str(RELEASE / 'execute_approved.py'), '--execute-approved-L13R1'], 'alternate entry command')
    outcome = read(RUN / 'evidence/execution_outcome.json')
    check(outcome['exit_code'] == 1 and outcome['explicit_entry_invocations'] == 1 and
          outcome['real_http_attempts'] == outcome['real_http_responses'] == 0 and not outcome['resumed_or_reset'], 'outcome')
    status = read(RUN / 'execution_acceptance.json')
    for key in ['http_attempts', 'http_responses', 'new_valid_answers', 'A_valid', 'B_valid', 'third_valid', 'final_valid']:
        check(status[key] == 0, 'invalid observed result: ' + key)
    check(status['identity_consumed'] and status['status'] == 'FAILED_BATCH_CLOSED' and
          status['run_acceptance'] == 'PASS_AS_FAITHFUL_FAILED_BATCH_RECORD' and
          status['unknown_ratings'] == status['not_attempted_questions'] == 13 and
          status['not_attempted_slots'] == 52 and not status['semantic_effect_accepted'], 'failure acceptance scope')
    projected = read(RUN / 'slot_statuses.json')
    check(len(projected) == 52 and {r['slot']: r['action'] for r in projected} == slots, 'slot denominator')
    check(all(r['status'] == 'not_attempted_batch_startup_failed' and not r['reserved'] and
              not r['http_attempted'] and not r['http_response'] and r['rating'] is None for r in projected), 'slot state')
    costs = read(RUN / 'calls_and_costs.json')
    theoretical = {action: str((Decimal(a['input_cap']) * Decimal(config['price']['input_per_million']) +
                                Decimal(a['output_cap']) * Decimal(config['price']['output_per_million'])) /
                               Decimal(1000000) * a['max_attempts']) for action, a in config['actions'].items()}
    check(sum(map(Decimal, theoretical.values())) == Decimal('1.2460032') and
          costs['potential_full_cap_reserves_by_role_USD'] == theoretical, 'potential reserve')
    check(costs['actual_ledger_reserved_USD'] == costs['refunded_USD'] == '0' and
          costs['estimated_peak_cache_miss_USD_for_observed_calls'] == '0.0000000' and
          costs['actual_bill'] == 'unknown' and not costs['provider_invoice_obtained'], 'actual costs')
    usage = costs['usage']
    check(usage['response_usage_records'] == usage['observed_prompt_tokens'] ==
          usage['observed_completion_tokens'] == usage['observed_total_tokens'] == 0, 'usage records')
    check(usage['usage_coverage_over_http_attempts'] == {'status': 'not_applicable', 'numerator': 0, 'denominator': 0}, 'usage denominator')
    return {'status': 'PASS', 'entry_invocations': 1, 'entry_exit_code': 1, 'http_attempts': 0,
            'http_responses': 0, 'roles': dict(Counter(slots.values())), 'all_role_attempts': 0,
            'actual_reserved_USD': '0', 'estimated_peak_USD': '0.0000000', 'actual_bill': 'unknown',
            'identity_consumed': True, 'retry_resume_reset': False, 'execution_files': len(files)}


def verify(check_seal=True):
    check(sys.dont_write_bytecode and sys.flags.optimize == 0, 'Use existing Python -B without -O')
    baseline = read(RUN / 'evidence/prelaunch_baseline.json')
    compat = read(RUN / 'compatibility.json')
    check(baseline['exact_active_demo_exclusions'] == compat['exact_active_demo_exclusions'] == EXCLUSIONS, 'exclusion expansion')
    check(hashlib.sha256((RUN / 'evidence/prelaunch_baseline.json').read_bytes()).hexdigest() ==
          compat['source_baseline_sha256'], 'baseline provenance')
    external = Path(compat['source_external_baseline'])
    check(hashlib.sha256(external.read_bytes()).hexdigest() == compat['source_baseline_sha256'], 'external baseline differs')
    for name in ['commands.json', 'verify_release.stdout.json', 'verify_release.stderr.log',
                 'launch_receipt.json', 'execute_approved.stdout.log', 'execute_approved.stderr.log',
                 'execution_outcome.json', 'post_execution_history.json']:
        check((external.parent / name).read_bytes() == (RUN / 'evidence' / name).read_bytes(), 'external log provenance: ' + name)
    check(len(baseline['files']) == 26392 and len(baseline['links']) == 11, 'prelaunch denominator')
    check(baseline['output_identity_existed'] is False, 'identity already existed before launch')
    check(set(compat['changed_existing_files']) == {'AAE5103_项目计划与交接.md'}, 'unexpected historical edit')
    check(set(compat['added_docs']) == {'docs/E02_L13R1_运行验收与交接_rev045.md', 'docs/handoffs/2026-11-05-045.md'}, 'unexpected added docs')
    check(set(compat['added_execution_files']) == {str((OUTPUT / n).relative_to(PROJECT))
          for n in ['authorization.consumed.json', 'ledger.jsonl']}, 'execution additions')
    expected = dict(baseline['files'])
    for path, change in compat['changed_existing_files'].items():
        check(expected[path] == change['before'], 'before record mismatch')
        expected[path] = change['after']
    for group in ['added_docs', 'added_execution_files']:
        check(not expected.keys() & compat[group].keys(), 'added file shadows history')
        expected.update(compat[group])
    own, own_links = scan(RUN)
    check(not own_links, 'audit run contains unexpected links')
    if check_seal:
        manifest, receipt = read(RUN / 'final_manifest.json'), read(RUN / 'final_verification.json')
        check(manifest['root'] == str(RUN) and set(manifest['control_files']) == CONTROLS, 'seal root/controls')
        check(set(own) == set(manifest['files']) | CONTROLS, 'self exact file set')
        exact({k: own[k] for k in manifest['files']}, manifest['files'], 'self file identities/bytes/metadata')
        exact(own_links, manifest['links'], 'self links')
        check(receipt['manifest_record'] == record(RUN / 'final_manifest.json') and receipt['status'] == 'PASS', 'seal receipt')
    for name, rec in own.items():
        rel = str((RUN / name).relative_to(PROJECT))
        check(rel not in expected, 'audit run shadows history')
        expected[rel] = rec
    actual, actual_links = scan(PROJECT, {PROJECT / p for p in EXCLUSIONS})
    exact(actual, expected, 'full project exact historical files and individually enumerated additions')
    exact(actual_links, baseline['links'], 'full project historical links')
    old_plan = (RUN / 'before/AAE5103_项目计划与交接.md').read_bytes()
    new_plan = (PROJECT / 'AAE5103_项目计划与交接.md').read_bytes()
    check(hashlib.sha256(old_plan).hexdigest() == baseline['files']['AAE5103_项目计划与交接.md']['sha256'], 'old plan lost')
    check(old_plan == (PROJECT / 'docs/handoffs/2026-11-05-044.md').read_bytes(), '044 archive changed')
    check(new_plan == (PROJECT / 'docs/handoffs/2026-11-05-045.md').read_bytes() ==
          (RUN / 'snapshots/AAE5103_项目计划与交接.md').read_bytes(), 'new plan snapshot differs')
    check(new_plan.decode().splitlines()[2] == '版本：v1.42｜状态修订号：045｜更新日期：2026-11-05（Asia/Tokyo）', 'current version')
    diff = ''.join(difflib.unified_diff(old_plan.decode().splitlines(keepends=True), new_plan.decode().splitlines(keepends=True),
                                      fromfile='before/AAE5103_项目计划与交接.md', tofile='after/AAE5103_项目计划与交接.md'))
    check(diff == (RUN / 'diffs/AAE5103_项目计划与交接.md.diff').read_text(), 'plan diff not exact')
    check((PROJECT / 'docs/E02_L13R1_运行验收与交接_rev045.md').read_bytes() ==
          (RUN / '结果与交接.md').read_bytes(), 'handoff differs')
    post = read(RUN / 'evidence/post_execution_history.json')
    check(post['status'] == 'PASS' and not post['changed_existing_files'] and not post['missing_files'] and
          post['links_exact'] and post['added_execution_files'] == compat['added_execution_files'], 'post-execution pre-sync protection')
    exec_result, math_result = execution(), arithmetic()
    return {'status': 'PASS', 'scope': 'Closed failed batch evidence/arithmetic/protection, not semantic effect acceptance',
            'execution': exec_result, 'statistics': math_result,
            'history': {'status': 'PASS', 'baseline_regular_files': len(baseline['files']), 'baseline_links': len(baseline['links']),
                        'unchanged_old_regular_files': len(baseline['files']) - 1, 'changed_existing_files': ['AAE5103_项目计划与交接.md'],
                        'added_execution_files': 2, 'added_docs': 2, 'self_run_regular_files': len(own),
                        'current_total_regular_files': len(actual), 'links_exact': True,
                        'exact_active_demo_exclusions': EXCLUSIONS, 'old_release_and_false_templates_unchanged': True},
            'plan': 'v1.42/rev045', 'self_seal_checked': check_seal, 'pure_read_only': True,
            'new_model_API_calls_by_verifier': 0, 'old_prelaunch_verify_called_after_consumption': False}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
