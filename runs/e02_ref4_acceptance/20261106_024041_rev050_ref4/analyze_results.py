"""Audit actual REF4 calls and compute five-group fixed-denominator paired contrasts."""
from pathlib import Path
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
import base64
import hashlib
import json
import math
import sys

E = Path(__file__).absolute().parent
S = json.loads((E / 'settings.json').read_bytes())
PROJECT_ROOT = E.parents[2]
for key in ("project", "external", "run", "output", "source_ready", "old_answers", "L_execution", "metadata_helper", "acceptance_run"):
    if key in S and not Path(S[key]).is_absolute():
        S[key] = str(PROJECT_ROOT / S[key])
P, R, O = Path(S['project']), Path(S['run']), Path(S['output'])
sys.path.insert(0, str(R / 'runtime'))
import common as C
import quote_anchor as QA

ROLES = {'REF_cov': None, 'REF_knowledge': 'positive_knowledge',
         'REF_boundary': 'boundary', 'REF_source_state': 'source_status'}
GROUPS = ['L', 'T', 'TV', 'G', 'GV']
SPLITS = ['debug', 'operational_reserve']
MASK = {'QV3-D03-P2', 'QV3-H01-P5'}
CONTRASTS = [('G', 'L'), ('GV', 'L'), ('G', 'T'), ('GV', 'G'), ('TV', 'T')]


def read(path):
    return C.read_json(path)


def eq(a, b, reason):
    C.require(a == b, reason)


def ratio(values):
    if not values:
        return {'status': 'not_applicable', 'value': None, 'denominator': 0,
                'lower': None, 'upper': None, 'exact_lower': None, 'exact_upper': None}
    known = sum((Fraction(str(x)) for x in values if x is not None), Fraction())
    missing = sum(x is None for x in values)
    low, high = known / len(values), (known + missing) / len(values)
    return {'status': 'unknown' if missing else 'known', 'value': float(low) if not missing else None,
            'numerator': float(known), 'denominator': len(values), 'unknown': missing,
            'lower': float(low), 'upper': float(high), 'exact_lower': str(low), 'exact_upper': str(high)}


def fixed(units, mask=False):
    questions, families, macros = [], [], []
    for unit in units:
        for metric, role in ROLES.items():
            values = [None if mask and p['point_id'] in MASK else p['score'] for p in unit['points']
                      if role is None or p['content_role'] == role]
            questions.append({'group': unit['group'], 'question_id': unit['question_id'],
                              'family_id': unit['family_id'], 'split': unit['split'], 'metric': metric, **ratio(values)})
    for group in GROUPS:
        for metric in ROLES:
            for split in SPLITS:
                planned = [q for q in questions if q['group'] == group and q['metric'] == metric and q['split'] == split]
                by_family = defaultdict(list)
                for q in planned:
                    by_family[q['family_id']].append(q)
                frows = []
                for family, values in sorted(by_family.items()):
                    active = [q for q in values if q['denominator'] > 0]
                    low = sum((Fraction(q['exact_lower']) for q in active), Fraction()) / len(active) if active else None
                    high = sum((Fraction(q['exact_upper']) for q in active), Fraction()) / len(active) if active else None
                    row = {'group': group, 'metric': metric, 'split': split, 'family_id': family,
                           'planned_questions': len(values), 'applicable_questions': len(active),
                           'N_A_questions': len(values) - len(active), 'fixed_points': sum(q['denominator'] for q in values),
                           'status': ('known' if low == high else 'unknown') if active else 'not_applicable',
                           'value': float(low) if active and low == high else None,
                           'lower': float(low) if active else None, 'upper': float(high) if active else None,
                           'exact_lower': str(low) if active else None, 'exact_upper': str(high) if active else None}
                    families.append(row)
                    frows.append(row)
                active = [q for q in frows if q['applicable_questions'] > 0]
                low = sum((Fraction(q['exact_lower']) for q in active), Fraction()) / len(active) if active else None
                high = sum((Fraction(q['exact_upper']) for q in active), Fraction()) / len(active) if active else None
                macros.append({'group': group, 'metric': metric, 'split': split,
                               'planned_questions': len(planned), 'planned_families': len(by_family),
                               'applicable_families': len(active), 'N_A_questions': sum(q['N_A_questions'] for q in frows),
                               'fixed_points': sum(q['fixed_points'] for q in frows),
                               'status': ('known' if low == high else 'unknown') if active else 'not_applicable',
                               'value': float(low) if active and low == high else None,
                               'lower': float(low) if active else None, 'upper': float(high) if active else None,
                               'exact_lower': str(low) if active else None, 'exact_upper': str(high) if active else None})
    return {'questions': questions, 'families': families, 'macros': macros,
            'mask': sorted(MASK) if mask else [], 'aggregation': 'fixed point -> question -> family -> split',
            'denominator_policy': 'all planned units retained; unknown full0..1; structural N/A unchanged'}


def contrasts(stats):
    macros = {(m['group'], m['metric'], m['split']): m for m in stats['macros']}
    rows = []
    for target, baseline in CONTRASTS:
        for metric in ROLES:
            for split in SPLITS:
                a, b = macros[(target, metric, split)], macros[(baseline, metric, split)]
                eq(a['fixed_points'], b['fixed_points'], 'same fixed contrast point denominator')
                eq(a['applicable_families'], b['applicable_families'], 'same contrast family denominator')
                if a['status'] == 'not_applicable':
                    rows.append({'contrast': target + '-' + baseline, 'metric': metric, 'split': split,
                                 'status': 'not_applicable', 'lower_pp': None, 'upper_pp': None})
                    continue
                low = (Fraction(a['exact_lower']) - Fraction(b['exact_upper'])) * 100
                high = (Fraction(a['exact_upper']) - Fraction(b['exact_lower'])) * 100
                direction = 'positive_within_reported_bounds' if low > 0 else ('negative_within_reported_bounds' if high < 0 else ('equal' if low == high == 0 else 'direction_unresolved'))
                rows.append({'contrast': target + '-' + baseline, 'metric': metric, 'split': split,
                             'status': 'known' if low == high else 'unknown_interval',
                             'lower_pp': float(low), 'upper_pp': float(high),
                             'exact_lower_pp': str(low), 'exact_upper_pp': str(high),
                             'interpretation': direction, 'applicable_families': a['applicable_families'],
                             'fixed_points_per_group': a['fixed_points'],
                             'causal_or_statistical_significance_claim': False})
    return rows


def audit_execution():
    config = read(R / 'config/execution.json')
    slots = read(R / 'config/slots.json')
    auth = read(R / 'authorization/authorization.json')
    sources = read(R / 'inputs/saved_answer_bindings.json')['sources']
    ledger = [C.strict_json(x) for x in (O / 'ledger.jsonl').read_bytes().splitlines()]
    eq([x['sequence'] for x in ledger], list(range(1, len(ledger) + 1)), 'ledger sequence')
    eq(ledger[0]['config_sha256'], C.digest(config), 'new independent ledger config')
    eq(read(O / 'authorization.consumed.json'), {'authorization_sha256': C.digest(auth), 'no_reuse': True}, 'consumed new identity')
    reserves = [x for x in ledger if x['event'] == 'reserve']
    reserved = {x['slot']: x for x in reserves}
    eq(len(reserved), len(reserves), 'no repeated attempts')
    planned = [s['question_id'] + '_L_REF047_' + label for s in sources for label in ['A', 'B', 'C']]
    eq([x['slot'] for x in reserves], planned[:len(reserves)], 'no sends after batch stop')
    C.require(len(reserves) <= 156, '156 limit')
    events = defaultdict(list)
    for x in ledger:
        if x['event'] == 'result':
            events[x['slot']].append(x)
    total_reserve = Decimal('0')
    usage_total, role_attempts, role_valid, codes = Counter(), Counter(), Counter(), Counter()
    slotrows, units, failure_rows = [], [], []
    checks = Counter()
    for source in sources:
        packet = read(R / source['packet'])
        eq(C.file_hash(P / source['source_result_path']), source['source_result_sha256'], 'saved original output bytes')
        resultpath = O / (source['question_id'] + '_L_REF047.result.json')
        result = read(resultpath) if resultpath.exists() else None
        originals = result['original_ratings'] if result else [None] * 3
        for i, label in enumerate(['A', 'B', 'C']):
            slot = source['question_id'] + '_L_REF047_' + label
            original = originals[i]
            reqpath, httppath, resppath = [O / 'raw' / (slot + suffix) for suffix in ['.request.json', '.http.json', '.response.json']]
            req = read(reqpath) if reqpath.exists() else None
            http = read(httppath) if httppath.exists() else None
            resp = read(resppath) if resppath.exists() else None
            rs = reserved.get(slot)
            ev = events[slot]
            flags = [x['metadata']['http_client_post_invoked'] for x in ev]
            if http:
                flags.append(http['http_client_post_invoked'])
                eq(http['transport_kind'], 'real_http', 'real transport')
                raw = base64.b64decode(http['raw_base64'], validate=True)
                eq(hashlib.sha256(raw).hexdigest(), http['raw_sha256'], 'raw HTTP hash')
                eq(len(raw), http['raw_size'], 'raw HTTP length')
                codes[str(http['http_status'])] += 1
                if resp:
                    eq(resp['raw_utf8'].encode(), raw, 'raw text exact')
                    eq(C.strict_json(raw), resp['response'], 'parsed HTTP exact')
            attempted = True if True in flags else (False if flags or not rs else None)
            if attempted:
                role_attempts[label] += 1
            if rs:
                role = config['actions'][slots[slot]]
                amount = (Decimal(role['input_cap']) * Decimal(config['price']['input_per_million']) + Decimal(role['output_cap']) * Decimal(config['price']['output_per_million'])) / 1000000
                total_reserve += amount
                eq(Decimal(rs['amount_USD']), amount, 'full cap reserve')
                eq(Decimal(rs['total_reserved_USD']), total_reserve, 'running reserve')
            if req:
                eq(C.digest(req['body']), rs['body_sha256'], 'complete sent body binding')
                eq(C.digest(req['body']['messages']), req['count']['messages_sha256'], 'complete counted message binding')
                eq(req['count']['counter_assets_sha256'], C.digest(read(R / 'config/counter_assets.json')), 'pinned Counter asset binding')
                role = config['actions'][slots[slot]]
                C.require(req['count']['input_tokens'] <= role['input_cap'], 'full actual input cap')
                eq({k: req['body'][k] for k in ['model', 'thinking', 'temperature', 'response_format', 'stream', 'max_tokens']},
                   {'model': 'deepseek-flash', 'thinking': {'type': 'disabled'}, 'temperature': 0,
                    'response_format': {'type': 'json_object'}, 'stream': False, 'max_tokens': role['output_cap']}, 'send profile')
                if label == 'C':
                    primary = [{k: x[k] for k in ['status', 'rating', 'source_binding']} for x in originals[:2]]
                    payload = QA.third_packet(packet, *primary)
                    checks['actual_current_A_B_third_packets'] += 1
                else:
                    payload = packet
                    eq(C.digest(req['body']['messages']), C.file_hash(R / 'serialized_judge' / (source['unit_id'] + '.messages.json')), 'frozen initial full bytes')
                eq(req['body']['messages'], QA.messages(slots[slot], payload), 'whole system/user/schema/current A/B unchanged')
                checks['complete_request_count_bindings'] += 1
            received = [x for x in ev if x['status'] == 'received']
            C.require(len(received) <= 1, 'one response per slot')
            if received:
                u = resp['response']['usage']
                eq(u, received[0]['metadata']['usage'], 'raw/ledger usage')
                eq(u['prompt_tokens'] + u['completion_tokens'], u['total_tokens'], 'usage arithmetic')
                C.require(all(type(u[k]) is int and u[k] >= 0 for k in ['prompt_tokens', 'completion_tokens', 'total_tokens']), 'strict usage integers')
                eq(resp['response']['model'], 'deepseek-flash', 'returned alias')
                C.require(set(u) <= {'prompt_tokens', 'completion_tokens', 'total_tokens',
                                    'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens',
                                    'prompt_tokens_details', 'completion_tokens_details'}, 'no extra billing dimensions')
                if 'prompt_cache_hit_tokens' in u or 'prompt_cache_miss_tokens' in u:
                    C.require(all(type(u[k]) is int and u[k] >= 0 for k in ['prompt_cache_hit_tokens', 'prompt_cache_miss_tokens']), 'strict cache counters')
                    eq(u['prompt_cache_hit_tokens'] + u['prompt_cache_miss_tokens'], u['prompt_tokens'], 'cache arithmetic')
                if u.get('prompt_tokens_details') is not None:
                    detail = u['prompt_tokens_details']
                    C.require(type(detail) is dict and set(detail) <= {'cached_tokens'}, 'prompt billing detail profile')
                    if 'cached_tokens' in detail:
                        eq(detail['cached_tokens'], u['prompt_cache_hit_tokens'], 'cached details coherent')
                if u.get('completion_tokens_details') is not None:
                    detail = u['completion_tokens_details']
                    C.require(type(detail) is dict and set(detail) <= {'reasoning_tokens'}, 'completion billing detail profile')
                    reasoning = detail.get('reasoning_tokens')
                    C.require(reasoning is None or type(reasoning) is int and reasoning == 0, 'nonthinking usage')
                C.require(u['prompt_tokens'] <= role['input_cap'] and u['completion_tokens'] <= role['output_cap'], 'actual service caps')
                eq(len(resp['response']['choices']), 1, 'single choice')
                eq(resp['response']['choices'][0]['finish_reason'], 'stop', 'known text completion')
                message = resp['response']['choices'][0]['message']
                eq(message['role'], 'assistant', 'completion role')
                C.require(message.get('reasoning_content') in [None, ''], 'no returned thinking')
                C.require(message.get('tool_calls') is None or message['tool_calls'] == [], 'no returned tools')
                C.require(message.get('function_call') is None and message.get('audio') is None, 'no nontext billable output')
                for k in ['prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens']:
                    if k in u:
                        usage_total[k] += u[k]
                checks['known_usage_records'] += 1
            if original:
                path = O / 'ratings' / (slot + '.original.json')
                if path.exists():
                    eq(read(path), original, 'original A/B/C retained')
                if original['status'] == 'valid':
                    wire = C.strict_json(resp['response']['choices'][0]['message']['content'])
                    eq(wire, original['raw_rating'], 'actual wire response')
                    anchored = QA.validate_rating(wire, packet, source['binding'])
                    eq(anchored['rating'], original['rating'], 'canonical exact source mapping')
                    eq({k: v for k, v in anchored.items() if k != 'rating'}, original['anchor_metadata'], 'anchor hashes/metadata')
                    role_valid[label] += 1
                    checks['valid_wire_canonical_mappings'] += 1
                    checks['resolved_source_quotes'] += anchored['resolved_quote_count']
                else:
                    eq(original['rating'], None, 'failed rating stays null')
                    failure = None
                    if resp:
                        try:
                            wire = C.strict_json(resp['response']['choices'][0]['message']['content'])
                            QA.validate_rating(wire, packet, source['binding'])
                        except Exception as exc:
                            failure = {'type': type(exc).__name__, 'reason': str(exc)}
                    if original['status'] == 'evaluator_failure' and received:
                        C.require(failure is not None, 'coding failure must reproduce from saved raw')
                    failure_rows.append({'slot': slot, 'cell_key': source['question_id'], 'group': source['group'],
                                         'status': original['status'], 'reason': original.get('reason'),
                                         'raw_failure_reproduced': failure})
            slotrows.append({'slot': slot, 'cell_key': source['question_id'], 'group': source['group'],
                             'question_id': source['original_question_id'], 'label': label,
                             'action': slots[slot], 'reserved': rs is not None,
                             'http_attempted': attempted, 'http_status': http['http_status'] if http else None,
                             'status': original['status'] if original else 'unknown_missing_record',
                             'rating_valid': bool(original and original['status'] == 'valid'),
                             'reason': original.get('reason') if original else None,
                             'input_tokens': req['count']['input_tokens'] if req else None,
                             'usage': received[0]['metadata']['usage'] if received else None})
        if result:
            eq(result['observed_text'], packet['observed_text'], 'actual displayed original retained')
            eq(result['source_binding'], source['binding'], 'result source binding')
            eq(result['new_answer_generated'], False, 'no generation')
            if originals[2]['status'] == 'valid':
                eq(result['final'], {'status': 'resolved_third', 'rating': originals[2]['rating']}, 'full third final only')
            else:
                eq(result['final']['rating'], None, 'third failure remains unknown')
        units.append(make_unit(source, result, packet, source['group'], source['original_question_id'], source['comparison_split'], str(resultpath)))
    report = read(O / 'reference_report.json')
    eq(Decimal(report['reserved_USD']), total_reserve, 'report reserve')
    eq(report['planned_questions'], 52, 'all52 planned output units retained')
    eq(report['valid_final_ratings'], sum(u['final_valid'] for u in units), 'new final count')
    peak = (Decimal(usage_total['prompt_tokens']) * Decimal(config['price']['input_per_million']) + Decimal(usage_total['completion_tokens']) * Decimal(config['price']['output_per_million'])) / 1000000
    proof = {'status': 'PASS', 'candidate_id': S['candidate_id'], 'entry_invocations': 1,
             'http_attempts': sum(x['http_attempted'] is True for x in slotrows),
             'unknown_HTTP_attempts': sum(x['http_attempted'] is None for x in slotrows),
             'http_responses': sum(codes.values()), 'http_status_codes': dict(codes),
             'attempts_by_label': dict(role_attempts), 'valid_ratings_by_label': dict(role_valid),
             'valid_final': sum(u['final_valid'] for u in units), 'unknown_final': sum(not u['final_valid'] for u in units),
             'not_called_slots': [x['slot'] for x in slotrows if x['http_attempted'] is False],
             'batch_stopped': report['batch_stopped'], 'stop_reason': report['stop_reason'],
             'reserved_USD': str(total_reserve), 'usage_peak_estimate_USD': str(peak), 'usage': dict(usage_total),
             'actual_bill': 'unknown', 'new_answers': 0, 'retry_resume_reset': False,
             'checks': dict(checks), 'new_model_calls_by_audit': 0}
    return units, slotrows, failure_rows, proof


def make_unit(source, result, packet, group, qid, split, path):
    rubric = next(q for q in read(R / 'inputs/rubrics.json')['questions'] if q['question_id'] == qid)
    final = result['final']['rating'] if result else None
    scores = {p['point_id']: p['score'] for p in final['points']} if final else {}
    pts = [{'point_id': p['point_id'], 'content_role': p['content_role'], 'score': scores.get(p['point_id'])} for p in rubric['points']]
    unit = {'group': group, 'question_id': qid, 'cell_key': source['question_id'], 'unit_id': source['unit_id'],
            'family_id': source['family_id'], 'split': split, 'source_status': source.get('source_status', 'L_saved_answer'),
            'source_binding': source['binding'], 'observed_text_sha256': QA.source_hash(packet['observed_text']),
            'result_path': path, 'final_valid': final is not None,
            'final_status': result['final']['status'] if result else 'unknown_missing_record',
            'original_statuses': [x['status'] for x in result['original_ratings']] if result else ['unknown'] * 3,
            'points': pts, 'metrics': result['metrics'] if result else None,
            'dynamic_denominators': {'claims': len(final['claims']) if final else None,
                                     'citations': len(final['voluntary_citations']) if final else None}}
    for metric, role in ROLES.items():
        computed = ratio([p['score'] for p in pts if role is None or p['content_role'] == role])
        if result:
            actual = result['metrics'][metric]
            for field in ['lower', 'upper', 'value']:
                if computed[field] is not None:
                    C.require(math.isclose(actual[field], computed[field], rel_tol=1e-12, abs_tol=1e-12), 'independent question metric')
                else:
                    eq(actual[field], None, 'unknown/N/A retained')
            eq(actual['denominator'], computed['denominator'], 'question fixed denominator')
    return unit


def compute():
    units, slotrows, failures, proof = audit_execution()
    old_ready = Path(S['source_ready'])
    l_sources = read(old_ready / 'inputs/saved_answer_bindings.json')['sources']
    for source in l_sources:
        resultpath = Path(S['L_execution']) / (source['question_id'] + '_L_REF047.result.json')
        result = read(resultpath)
        packet = read(old_ready / source['packet'])
        units.append(make_unit(source, result, packet, 'L', source['question_id'], source['split'], str(resultpath)))
    eq(Counter(u['group'] for u in units), {g: 13 for g in GROUPS}, '65 planned units')
    for group in GROUPS:
        pts = [p for u in units if u['group'] == group for p in u['points']]
        eq(len(pts), 63, '63 fixed points each group')
        eq(Counter(p['content_role'] for p in pts), {'positive_knowledge': 39, 'boundary': 17, 'source_status': 7}, 'role denominators')
    main = fixed(units)
    sensitivity = fixed(units, mask=True)
    runtime = read(O / 'reference_report.json')['fixed_macros']
    for m in main['macros']:
        if m['group'] == 'L':
            continue
        actual = runtime[m['metric']][m['group'] + '|' + m['split']]
        for key in ['lower', 'upper']:
            if m[key] is None:
                eq(actual[key], None, 'runtime N/A macro')
            else:
                C.require(math.isclose(actual[key], m[key], abs_tol=1e-12), 'runtime independent group/family macro')
        eq(actual['families'], m['applicable_families'], 'runtime macro denominator')
    retrieval = []
    for qid in [s['question_id'] for s in l_sources]:
        group_rows = {}
        for group in ['T', 'G']:
            record = read(Path(S['old_answers']) / 'tested' / (qid + '_' + group + '_R01.json'))
            group_rows[group] = [x['evidence_id'] for x in record['actual_complete_cards']]
        retrieval.append({'question_id': qid, 'evidence_ids': group_rows,
                          'same_evidence_set': set(group_rows['T']) == set(group_rows['G'])})
    return {'execution_acceptance.json': proof, 'slot_statuses.json': slotrows, 'unit_statuses.json': units,
            'coding_failures.json': failures, 'reference_statistics.json': main,
            'contrasts.json': contrasts(main), 'sensitivity_statistics.json': sensitivity,
            'sensitivity_contrasts.json': contrasts(sensitivity), 'retrieval_diagnostic.json': retrieval,
            'comparison_checks.json': {'status': 'PASS', 'planned_units': 65, 'fixed_points': 315,
                                       'independent_question_metric_rows': 260, 'independent_group_split_macros': 40,
                                       'contrast_rows': 40, 'sensitivity_contrast_rows': 40,
                                       'all_groups_same_two_points_masked': sorted(MASK), 'canonical_scores_changed': False,
                                       'dynamic_overall_mean': None, 'model_judged': True,
                                       'independence': 'same agent, separate deterministic arithmetic/source audit; no expert truth or new independent agent claimed'}}


if __name__ == '__main__':
    artifacts = compute()
    target = E
    if sys.argv[1:] == ['--verify']:
        for name, obj in artifacts.items():
            eq(read(target / name), obj, 'saved analysis artifact: ' + name)
    else:
        eq(sys.argv[1:], [], 'unknown argument')
        target.mkdir(exist_ok=False)
        for name, obj in artifacts.items():
            with (target / name).open('x') as stream:
                stream.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS', 'execution': artifacts['execution_acceptance.json'],
                      'primary_debug': [x for x in artifacts['contrasts.json'] if x['contrast'] == 'G-L' and x['split'] == 'debug']},
                     ensure_ascii=False, indent=2))
