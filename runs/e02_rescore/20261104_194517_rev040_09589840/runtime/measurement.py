"""rev040 semantic wire: exact quotes and deterministic contract reconstruction.

No scoring heuristic, fuzzy matching or repair of rejected historical ratings.
Canonical encodings still pass the byte-copied rev037 validators and schemas.
"""
import copy
import jsonschema
from common import require, canonical, digest, snapshot
from engine import available_components, expected_gaps
from evaluation import validate_output, validate_pair

LOCKED = ['schema_version', 'method', 'judge_identity']


def remove(schema, fields):
    for name in fields:
        schema['properties'].pop(name, None)
        if name in schema.get('required', []):
            schema['required'].remove(name)


def wire_schema(kind, canonical_schema):
    schema = snapshot(canonical_schema)
    remove(schema, LOCKED)
    if kind == 'output':
        remove(schema, ['unit_id', 'tested_status', 'publishable_mechanical', 'boundary_rows', 'failure_reason'])
        remove(schema['properties']['point_rows']['items'], ['status', 'half_credit_allowed'])
        remove(schema['properties']['conditions']['items'], ['status', 'half_credit_allowed'])
        remove(schema['properties']['claims']['items'], ['necessary_condition_ids'])
        remove(schema['properties']['citation_pairs']['items'], ['id_valid'])
        insuff = schema['properties']['insufficiency']
        remove(insuff, ['applicable', 'available_point_ids', 'actual_gap_ids', 'gap_support_decisions'])
        remove(insuff['properties']['rows']['items'], ['status', 'half_credit_allowed'])
    else:
        remove(schema, ['pair_id', 'detected_error_ids', 'reported_issue_ids', 'final_observable', 'report_observable'])
    def quotes(node):
        if isinstance(node, dict):
            properties = node.get('properties', {})
            if {'channel', 'start', 'end', 'quote'} <= set(properties):
                remove(node, ['start', 'end'])
                properties['quote']['minLength'] = 1
                properties['occurrence'] = {'type': 'integer', 'minimum': 0}
                node['description'] = 'Exact visible substring; occurrence is zero-based and required only when repeated. No offsets.'
            for value in node.values():
                quotes(value)
        elif isinstance(node, list):
            for value in node:
                quotes(value)
    quotes(schema)
    return schema


def channels(packet, kind):
    return ({'final_answer': packet['final_answer'], 'service_message': packet['service_message']} if kind == 'output'
            else {'draft': packet['A']['draft'], 'final_answer': packet['B']['final_answer'],
                  'service_message': packet['B']['service_message'],
                  'verification_issues': canonical(packet['verification_issues_subprojection'])})


def exact_spans(value, texts):
    if isinstance(value, dict):
        if {'channel', 'quote'} <= set(value):
            require(set(value) <= {'channel', 'quote', 'occurrence'}, 'quote contains unexpected fields')
            channel, quote = value['channel'], value['quote']
            require(channel in texts and type(quote) is str and bool(quote), 'quote must be nonempty and visible')
            text = texts[channel] or ''
            positions, cursor = [], 0
            while True:
                position = text.find(quote, cursor)
                if position < 0:
                    break
                positions.append(position)
                cursor = position + 1
            require(bool(positions), 'quote is not an exact visible substring')
            if 'occurrence' not in value:
                require(len(positions) == 1, 'ambiguous quote requires occurrence')
            occurrence = value.get('occurrence', 0)
            require(type(occurrence) is int and 0 <= occurrence < len(positions), 'quote occurrence out of range')
            start = positions[occurrence]
            return {'channel': channel, 'start': start, 'end': start + len(quote), 'quote': quote}
        return {key: exact_spans(item, texts) for key, item in value.items()}
    if isinstance(value, list):
        return [exact_spans(item, texts) for item in value]
    return value


def facts(packet, kind):
    rubric = packet['rubric']
    if kind == 'pair':
        return {'final_body_observable': packet['B']['final_answer'] is not None,
                'draft_observable': packet['A']['draft'] is not None,
                'verification_report_observable': packet['verification_report_observable'],
                'reported_issue_ids': [issue['issue_id'] for issue in packet['verification_issues_subprojection']]}
    gaps, decisions = expected_gaps(rubric, packet['actual_complete_cards'])
    return {'tested_status': packet['service_observation']['tested_status'],
            'publishable_mechanical': packet['service_observation']['publishable_mechanical'],
            'point_ids': [point['point_id'] for point in rubric['points']],
            'boundary_point_ids': [point['point_id'] for point in rubric['points'] if point['content_role'] == 'boundary'],
            'point_half_eligible': [p['point_id'] for p in rubric['points'] if p['half_credit_eligible']],
            'I_half_eligible': [p['criterion_id'] for p in rubric['insufficiency_criteria'] if p['half_credit_eligible']],
            'insufficiency_applicable': any(c['applicable'] for c in rubric['insufficiency_criteria']),
            'available_point_ids': available_components(rubric, packet['actual_complete_cards']),
            'actual_gap_ids': gaps, 'gap_support_decisions': decisions,
            'condition_half_credit_allowed': False}


def decode(value, kind, schema, packet, record, rubric, identity):
    jsonschema.Draft202012Validator(wire_schema(kind, schema)).validate(value)
    result = exact_spans(value, channels(packet, kind))
    result.update(schema_version=schema['properties']['schema_version']['enum'][0],
                  method='model_judged', judge_identity=snapshot(identity))
    if kind == 'output':
        result.update(unit_id=packet['unit_id'], tested_status=record['status'],
                      publishable_mechanical=record['publishable'], failure_reason=None)
        failed = record['status'] in ['tested_failure', 'missing']
        halves = {p['point_id'] for p in rubric['points'] if p['half_credit_eligible']}
        ihalves = {p['criterion_id'] for p in rubric['insufficiency_criteria'] if p['half_credit_eligible']}
        def row_metadata(row, half_ids):
            # A value is never supplied or changed here: all semantic scores come from this model response.
            row.update(status=(record['status'] if failed else 'scored') if row['score'] is not None else 'machine_unresolved',
                       half_credit_allowed=row['id'] in half_ids)
        for row in result['point_rows']:
            row_metadata(row, halves)
        boundary_ids = {p['point_id'] for p in rubric['points'] if p['content_role'] == 'boundary'}
        result['boundary_rows'] = [snapshot(row) for row in result['point_rows'] if row['id'] in boundary_ids]
        for condition in result['conditions']:
            condition.update(status='scored' if condition['score'] is not None else 'machine_unresolved', half_credit_allowed=False)
        for claim in result['claims']:
            claim['necessary_condition_ids'] = [c['id'] for c in result['conditions'] if c['claim_id'] == claim['id']]
        visible = {card['evidence_id'] for card in packet['actual_complete_cards']}
        for pair in result['citation_pairs']:
            pair['id_valid'] = pair['evidence_id'] in visible
        frozen = facts(packet, kind)
        insuff = result['insufficiency']
        insuff.update(applicable=frozen['insufficiency_applicable'], available_point_ids=frozen['available_point_ids'],
                      actual_gap_ids=frozen['actual_gap_ids'], gap_support_decisions=[dict(d, reason='Deterministic frozen support-set membership, not a semantic rating.') for d in frozen['gap_support_decisions']])
        for row in insuff['rows']:
            row_metadata(row, ihalves)
        validate_output(result, schema, packet, record, rubric)
    else:
        issues = [issue['issue_id'] for issue in packet['verification_issues_subprojection']]
        result.update(pair_id=packet['pair_id'], reported_issue_ids=issues,
                      detected_error_ids=sorted({m['error_id'] for m in result['report_error_mapping'] if m['error_id'] is not None}),
                      final_observable=packet['B']['final_answer'] is not None,
                      report_observable=packet['verification_report_observable'])
        validate_pair(result, schema, packet)
    return result


def encode_fixture(canonical_value, kind):
    """Mechanical inverse for SYNTHETIC test fixtures only, never for old model scores."""
    result = snapshot(canonical_value)
    for key in LOCKED + (['unit_id', 'tested_status', 'publishable_mechanical', 'boundary_rows', 'failure_reason'] if kind == 'output' else
                         ['pair_id', 'detected_error_ids', 'reported_issue_ids', 'final_observable', 'report_observable']):
        result.pop(key, None)
    if kind == 'output':
        for row in result['point_rows'] + result['conditions'] + result['insufficiency']['rows']:
            row.pop('status', None)
            row.pop('half_credit_allowed', None)
        for claim in result['claims']:
            claim.pop('necessary_condition_ids')
        for row in result['citation_pairs']:
            row.pop('id_valid')
        for key in ['applicable', 'available_point_ids', 'actual_gap_ids', 'gap_support_decisions']:
            result['insufficiency'].pop(key)
    def strip(value):
        if isinstance(value, dict):
            if {'channel', 'quote', 'start', 'end'} <= set(value):
                return {'channel': value['channel'], 'quote': value['quote']}
            return {k: strip(v) for k, v in value.items()}
        return [strip(v) for v in value] if isinstance(value, list) else value
    return strip(result)
