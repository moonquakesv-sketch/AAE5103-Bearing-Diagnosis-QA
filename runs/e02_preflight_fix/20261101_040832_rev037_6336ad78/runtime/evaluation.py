"""Full-package independent-context coding, validation and third-agent adjudication."""
import copy
from pathlib import Path
import jsonschema
from common import require, Rejected, snapshot, digest, canonical, write_new
from engine import available_components, expected_gaps


def failure_reason(exc):
    if isinstance(exc,Rejected):return str(exc)
    if isinstance(exc,jsonschema.ValidationError):return 'schema violation at '+str(list(exc.absolute_path))+' ('+str(exc.validator)+')'
    return 'evaluation failure of type '+type(exc).__name__


def validate_spans(obj, channels):
    if isinstance(obj, dict):
        if {'channel', 'start', 'end', 'quote'} <= set(obj):
            require(obj['channel'] in channels, 'quote uses hidden/unavailable channel')
            text = channels[obj['channel']] or ''
            a, b = obj['start'], obj['end']
            require(type(a) is int and type(b) is int and 0 <= a <= b <= len(text) and text[a:b] == obj['quote'], 'quote span mismatch')
        for v in obj.values():
            validate_spans(v, channels)
    elif isinstance(obj, list):
        for v in obj:
            validate_spans(v, channels)


def score_rows(rows, expected, half_ids, failure=False):
    require(len(rows) == len(expected) and {r['id'] for r in rows} == set(expected), 'fixed applicable rows missing/duplicate')
    for r in rows:
        require(r['status'] != 'not_applicable', 'fixed requested point cannot be N/A')
        require(r['half_credit_allowed'] == (r['id'] in half_ids), 'half eligibility differs from frozen rule')
        if r['score'] == .5:
            require(r['id'] in half_ids, 'unapproved half score')
        if failure or r['status'] in ['tested_failure', 'missing']:
            require(r['score'] == 0, 'tested failure/missing must retain zero coverage')
        else:
            require((r['score'] is None) == (r['status'] in ['evaluator_failure', 'machine_unresolved']), 'score/status mismatch')


def validate_output(value, schema, packet, record, rubric):
    jsonschema.Draft202012Validator(schema).validate(value)
    require(value['evaluation_status']!='evaluator_failure','evaluator declared failure')
    require(value['unit_id'] == packet['unit_id'], 'anonymous unit binding mismatch')
    require(value['tested_status'] == record['status'], 'judge changed tested-system status')
    require(value['publishable_mechanical'] == record['publishable'], 'judge changed mechanical publication status')
    points = {p['point_id']: p for p in rubric['points']}
    half = {k for k, p in points.items() if p['half_credit_eligible']}
    failure = record['status'] in ['tested_failure', 'missing']
    score_rows(value['point_rows'], points, half, failure)
    boundary = {k for k, p in points.items() if p['content_role'] == 'boundary'}
    require({r['id'] for r in value['boundary_rows']} == boundary and len(value['boundary_rows']) == len(boundary), 'fixed boundary subset mismatch')
    pointrows = {r['id']: r for r in value['point_rows']}
    for row in value['boundary_rows']:
        require(all(row[k] == pointrows[row['id']][k] for k in ['status', 'score', 'half_credit_allowed']), 'boundary mirror differs')
    claims = {c['id']: c for c in value['claims']}
    require(len(claims) == len(value['claims']) and len({c['dedup_key'] for c in value['claims']}) == len(claims), 'duplicate claim coding')
    ids = {c['evidence_id'] for c in packet['actual_complete_cards']}
    for c in claims.values():
        require(bool(c['spans']) and any(s['quote'] for s in c['spans']), 'expressed claim requires an actual nonempty quote')
        require(not c['critical_error'] or c['support'] == 'none', 'critical error cannot be partly/full supported')
        require(all(set(combo) <= ids for combo in c['support_any_of']), 'claim support includes unseen cards')
        if c['support'] == 'full' and c['requires_external']:
            require(any(combo and set(combo) <= ids for combo in c['support_any_of']), 'full external claim lacks actual full support')
    pairs = value['citation_pairs']
    require(len({(p['claim_id'], p['evidence_id']) for p in pairs}) == len(pairs), 'duplicate citation pair')
    for p in pairs:
        require(p['claim_id'] in claims and p['id_valid'] == (p['evidence_id'] in ids), 'citation pair binding mismatch')
        require(p['id_valid'] or p['support'] == 'none', 'invalid citation cannot have semantic support')
        actual_text=(packet['final_answer'] or '')+'\n'+(packet['service_message'] or '')
        require(p['evidence_id'] in actual_text,'citation pair ID is not visible in service text')
    cond = value['conditions']
    require(len({(c['id'], c['claim_id']) for c in cond}) == len(cond), 'condition unit duplicate')
    for c in cond:
        require(c['claim_id'] in claims, 'orphan condition')
        # v3 has no enumerated noncritical condition-half branch. Extra claims cannot invent one.
        require(not c['half_credit_allowed'] and c['score'] != .5, 'no predeclared condition half branch')
        require((c['score'] is None) == (c['status'] != 'scored'), 'condition state mismatch')
    for c in claims.values():
        require(len(c['necessary_condition_ids']) == len(set(c['necessary_condition_ids'])) and
                set(c['necessary_condition_ids']) == {x['id'] for x in cond if x['claim_id'] == c['id']}, 'necessary conditions dangling')
        own=[x for x in cond if x['claim_id']==c['id']]
        if any(x['critical'] and x['score']==0 for x in own):
            require(c['support']=='none' and c['critical_error'],'critical omitted condition must invalidate complete claim')
    insuff = value['insufficiency']
    applicable = any(c['applicable'] for c in rubric['insufficiency_criteria'])
    require(insuff['applicable'] == applicable, 'insufficiency applicability changed')
    expected_available = available_components(rubric, packet['actual_complete_cards'])
    require(len(insuff['available_point_ids']) == len(set(insuff['available_point_ids'])) and set(insuff['available_point_ids']) == set(expected_available), 'actual available subset mismatch')
    gaps, decisions = expected_gaps(rubric, packet['actual_complete_cards'])
    require(len(insuff['actual_gap_ids']) == len(set(insuff['actual_gap_ids'])) and set(insuff['actual_gap_ids']) == set(gaps), 'actual gap set mismatch')
    require(len(insuff['gap_support_decisions']) == len(decisions), 'missing support decisions')
    got = {x['gap_id']: x for x in insuff['gap_support_decisions']}
    for expected in decisions:
        require(expected['gap_id'] in got and all(got[expected['gap_id']][k] == v for k, v in expected.items()), 'support gap status mismatch')
    iids = ['I01', 'I02', 'I03'] if applicable else []
    ihalf = {c['criterion_id'] for c in rubric['insufficiency_criteria'] if c['half_credit_eligible']}
    score_rows(insuff['rows'], iids, ihalf, failure)
    ibyid={r['id']:r for r in insuff['rows']}
    if applicable:
        if ibyid['I03']['score']==.5:require(len(expected_available)>=2,'I03 half requires at least two actually supported components')
        if ibyid['I02']['score']==.5:require(len(gaps)>=2,'I02 half requires at least two separate applicable gaps')
        if value['full_refusal'] and expected_available:require(ibyid['I01']['score']==0,'blanket refusal of available content cannot earn I01 credit')
    if value['release_adherence'] == 1:
        require(record['publishable'] and not failure and all(r['score'] == 1 for r in value['point_rows']), 'release success contradicts fixed coverage')
        require(all(c['support'] == 'full' and not c['critical_error'] for c in claims.values()) and all(c['score'] == 1 for c in cond), 'release success contradicts errors')
    for error in value['errors']:
        require(error['kind']=='omission' or any(s['quote'] and s['end']>s['start'] for s in error['spans']),
                'asserted output error needs a nonempty quote')
    validate_spans(value, {'final_answer': packet['final_answer'], 'service_message': packet['service_message']})
    return value


def validate_pair(value, schema, packet):
    jsonschema.Draft202012Validator(schema).validate(value)
    require(value['evaluation_status']!='evaluator_failure','pair evaluator declared failure')
    require(value['pair_id'] == packet['pair_id'], 'pair binding mismatch')
    require(value['final_observable'] == (packet['B']['final_answer'] is not None), 'final-body observability changed')
    require(value['report_observable'] == packet['verification_report_observable'], 'report observability changed')
    d = {e['error_id'] for e in value['draft_errors']}
    f = {e['error_id'] for e in value['final_errors']}
    require(len(d) == len(value['draft_errors']) and len(f) == len(value['final_errors']), 'duplicate stable error ID')
    issues = {i['issue_id'] for i in packet['verification_issues_subprojection']}
    require(set(value['reported_issue_ids']) == issues and len(value['reported_issue_ids']) == len(issues), 'verifier issue list mismatch')
    mapping = value['report_error_mapping']
    require(len(mapping) == len(issues) and {m['issue_id'] for m in mapping} == issues, 'issue-error mapping incomplete')
    require(all(m['error_id'] is None or m['error_id'] in d for m in mapping), 'mapping to unknown draft error')
    detected = {m['error_id'] for m in mapping if m['error_id'] is not None}
    require(set(value['detected_error_ids']) == detected and len(value['detected_error_ids']) == len(detected), 'detected error ID mismatch')
    require(value['report_observable'] or (not issues and not mapping and not detected), 'missing report cannot imply detection')
    if packet['A']['draft'] is None:
        require(value['evaluation_status'] == 'machine_unresolved', 'missing draft cannot be coded as no draft errors')
        require(not value['draft_errors'],'cannot invent unobservable draft errors')
    if packet['B']['final_answer'] is None:
        # Body observability controls correction credit. Visible service text is still codable.
        for e in value['final_errors']:
            require(bool(packet['B']['service_message']) and any(s['quote'] and s['end']>s['start'] for s in e['spans']) and
                    all(s['channel']=='service_message' for s in e['spans']), 'no body: only quoted visible service errors permitted')
    for e in value['draft_errors']:
        require(all(s['channel']=='draft' for s in e['spans']),'draft error uses final text')
        require(e['kind']=='omission' or any(s['quote'] and s['end']>s['start'] for s in e['spans']),'asserted draft error needs a nonempty quote')
    for e in value['final_errors']:
        require(all(s['channel'] in ['final_answer','service_message'] for s in e['spans']),'final error borrows hidden draft')
        require(e['kind']=='omission' or any(s['quote'] and s['end']>s['start'] for s in e['spans']),'asserted final error needs a nonempty quote')
    channels = {'draft': packet['A']['draft'], 'final_answer': packet['B']['final_answer'], 'service_message': packet['B']['service_message'],
                'verification_issues': canonical(packet['verification_issues_subprojection'])}
    validate_spans(value, channels)
    return value


def semantic_view(value):
    if isinstance(value, dict):
        return {k: semantic_view(v) for k, v in value.items() if k not in ['reason', 'spans', 'judge_identity', 'failure_reason']}
    if isinstance(value, list):
        return [semantic_view(v) for v in value]
    return value


def high_risks(value, packet, kind):
    risks = set(value.get('high_risks', []))
    if kind == 'pair':
        risks.add('draft_final_change')
    if kind == 'output':
        if packet['final_answer'] is None: risks.add('no_body')
        if value['insufficiency']['applicable']: risks.add('insufficiency')
        if any(c['critical_error'] for c in value['claims']): risks.add('critical_condition')
        if any(r['score'] in [.5, None] for r in value['point_rows']): risks.add('half_or_unresolved')
    return sorted(risks)


class AutomaticEvaluation:
    def __init__(self, runtime, directory):
        self.runtime = runtime
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)

    def evaluate(self, unit, kind, packet, record, rubric):
        role = 'judge_output' if kind == 'output' else 'judge_pair'
        prompt_role = 'judge' if kind == 'output' else 'pair_judge'
        schema = self.runtime.schemas[kind]
        validator = (lambda v: validate_output(v, schema, packet, record, rubric)) if kind == 'output' else (lambda v: validate_pair(v, schema, packet))
        originals = []
        original_hashes = []
        for who in ['A', 'B']:
            messages = self.runtime.messages(prompt_role, packet)
            slot = unit + '_' + kind + '_' + who
            raw = None
            try:
                raw, metadata = self.runtime.provider.send(slot, role, messages)
                validator(raw)
                out = {'status': 'valid', 'raw_rating': snapshot(raw), 'rating': snapshot(raw), 'metadata': metadata}
                # Identity comes from orchestration/actual provider metadata, never the model's self-assertion.
                out['identity'] = {'configured_model': self.runtime.config['model'], 'returned_model': metadata['returned_model'], 'hidden_backend': None,
                                   'context_id': digest({'slot': slot, 'messages': messages}), 'prompt_sha256': digest(messages[0]['content'])}
            except Exception as exc:
                out = {'status': 'evaluator_failure', 'rating': None, 'raw_rating': snapshot(raw) if raw is not None else None,
                       'failure_type': type(exc).__name__,'failure_reason':failure_reason(exc),
                       'failure_origin':'batch_stopped' if self.runtime.provider.ledger.broken else 'coding_or_evaluator_failure'}
            h=write_new(self.directory / (slot + '.original.json'), out, self.runtime.provider.ledger.directory)
            original_hashes.append(h)
            originals.append(out)
        disputed = any(o['status'] != 'valid' for o in originals)
        if not disputed:
            disputed = semantic_view(originals[0]['rating']) != semantic_view(originals[1]['rating'])
        risks = sorted(set(r for o in originals if o['status'] == 'valid' for r in high_risks(o['rating'], packet, kind)))
        triggers = (['primary_disagreement_or_failure'] if disputed else []) + risks
        final = {'status': 'machine_unresolved', 'rating': None, 'original_hashes': original_hashes, 'triggers': triggers,
                 'review': {'assigned': bool(triggers), 'status': 'not_assigned' if not triggers else 'evaluator_failure'}}
        if not triggers:
            final.update(status='resolved_primary_agreement', rating=originals[0]['rating'])
        else:
            anonymous_originals=[{'status':o['status'],'rating':o['rating'],**({'failure_type':o['failure_type']} if o['status']!='valid' else {})} for o in originals]
            adjudication_packet = {'unit_id': unit, 'kind': kind, 'packet': packet, 'originals': anonymous_originals, 'original_hashes':original_hashes, 'triggers': triggers,
                                  'instruction': 'Return exactly one root decision at path empty string. selected_value must be the COMPLETE final coding object under the original output/pair schema, or null with machine_unresolved. Preserve two raw originals.'}
            result = None
            metadata = None
            review_record = None
            try:
                result, metadata = self.runtime.provider.send(unit + '_' + kind + '_C', 'adjudicate_' + kind, self.runtime.messages('adjudicator', adjudication_packet))
                jsonschema.Draft202012Validator(self.runtime.schemas['adjudication']).validate(result)
                require(result['unit_id'] == unit and result['original_rating_a_sha256'] == original_hashes[0] and result['original_rating_b_sha256'] == original_hashes[1], 'adjudication original hashes mismatch')
                require(len(result['decisions']) == 1 and result['decisions'][0]['path'] == '', 'single complete coding root decision required')
                decision = result['decisions'][0]
                if decision['status'] == 'resolved':
                    validator(decision['selected_value'])
                else:
                    require(decision['selected_value'] is None, 'unresolved decision cannot hide a forced rating')
                quote_channels = {'final_answer': packet['final_answer'], 'service_message': packet['service_message']} if kind == 'output' else {
                    'draft': packet['A']['draft'], 'final_answer': packet['B']['final_answer'], 'service_message': packet['B']['service_message'],
                    'verification_issues': canonical(packet['verification_issues_subprojection'])}
                validate_spans(result, quote_channels)
                # Commit only after the entire response, including outer quotes, is valid.
                if decision['status'] == 'resolved':
                    final.update(status='resolved_third_agent', rating=snapshot(decision['selected_value']))
                else:
                    final.update(status=decision['status'], rating=None)
                final['adjudication'] = result
                final['adjudicator_metadata'] = metadata
                final['review']['status'] = 'evaluator_failure' if decision['status']=='evaluator_failure' else 'valid'
                review_record = {'status': final['review']['status'], 'raw_adjudication': snapshot(result), 'metadata': metadata}
            except Exception as exc:
                final.update(status='evaluator_failure', rating=None, failure_type=type(exc).__name__,failure_reason=failure_reason(exc))
                review_record = {'status': 'evaluator_failure', 'raw_adjudication': snapshot(result) if result is not None else None,
                                 'metadata': metadata, 'failure_type': type(exc).__name__, 'failure_reason': failure_reason(exc)}
            final['review']['record_sha256'] = write_new(self.directory / (unit + '_' + kind + '_C.original.json'), review_record, self.runtime.provider.ledger.directory)
        write_new(self.directory / (unit + '_' + kind + '.canonical.json'), final, self.runtime.provider.ledger.directory)
        return {'originals': originals, 'canonical': final}
