"""Validate E01 preparation files without importing or running a retriever.

Reads an explicit file list; exporting selects a bound request line, never a
rubric or a directory scan. Structural validation cannot discover all answer
leakage or semantic errors in free text. Human source/intent review is separate.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
VERSION = 'e01.validator.v1_rev014'
TASKS = frozenset(('explanation', 'inspection', 'preparation'))
CLASSES = frozenset(('normal', 'inner_race', 'outer_race', 'rolling_element'))
REQUEST_FIELDS = frozenset(('predicted_class', 'scenario_id', 'question',
                            'task_types', 'explicitly_allowed_operating_context'))
CONTEXT = {'operating_state': ('unknown', 'standstill'),
           'external_vibration': ('unknown', 'assumed_for_demo')}
SPLITS = ('debug', 'heldout')
BASIS_PATHS = frozenset(['knowledge/k01/' + name for name in
                        ('scenarios.csv', 'gaps.md', 'README.md', 'sources.csv',
                         'evidence.csv', 'interface.json')] + ['docs/E01_运行协议.md'])
DATA_FILES = ('design_contract.json', 'debug_requests.jsonl',
              'heldout_requests.jsonl', 'question_bindings.json', 'rubrics.json',
              'boundary_cases.json', 'contract_cases.json')
BINDING_FIELDS = frozenset(('question_id', 'split', 'line_number',
                            'request_sha256', 'family_id'))
RUBRIC_FIELDS = frozenset(('question_id', 'split', 'family_id', 'tasks',
                          'answerability', 'points', 'relevant_evidence_ids',
                          'forbidden_claims', 'required_unknowns',
                          'source_limitations', 'manual_confirmation',
                          'exposure_note', 'insufficiency_criteria'))
POINT_FIELDS = frozenset(('point_id', 'task_type', 'claim', 'conditions',
                         'support_any_of', 'basis_kind', 'basis_refs',
                         'citations', 'full_credit', 'partial_credit',
                         'unsupported_credit'))


class E01Error(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def require(condition, code, message):
    if not condition:
        raise E01Error(code, message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def load_strict_json(data):
    """Parse JSON text, rejecting duplicate keys and NaN/Infinity constants."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'invalid_json', 'Duplicate JSON key: ' + key)
            result[key] = value
        return result

    def constant(value):
        raise E01Error('invalid_json', 'Non-finite JSON number: ' + value)

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise E01Error('invalid_json', str(error)) from error


def string(value, where, nonempty=True):
    require(type(value) is str, 'type', where + ' must be a string.')
    require(not nonempty or bool(value.strip()), 'empty', where + ' must not be empty.')
    try:
        value.encode('utf-8')
    except UnicodeEncodeError as error:
        raise E01Error('invalid_unicode', where + ' has a non-scalar Unicode value.') from error


def string_list(value, where, nonempty=False, unique=False):
    require(type(value) is list, 'type', where + ' must be a list.')
    require(not nonempty or bool(value), 'empty', where + ' must not be empty.')
    for item in value:
        string(item, where + ' item')
    require(not unique or len(value) == len(set(value)), 'duplicate', where + ' has duplicates.')


def object_fields(value, expected, where):
    require(type(value) is dict, 'type', where + ' must be an object.')
    require(set(value) == expected, 'fields', where + ' fields differ: ' +
            str(sorted(set(value) ^ expected, key=str)))


def stable_id(value, where):
    string(value, where)
    require(re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', value) is not None,
            'id_format', where + ' is not a stable ASCII identifier.')


def validate_business_request(request):
    """Mirror the pinned five-field input contract; no retrieval takes place."""
    object_fields(request, REQUEST_FIELDS, 'business request')
    for field in ('predicted_class', 'scenario_id', 'question'):
        string(request[field], field, nonempty=field != 'question')
    require(bool(request['question'].strip()), 'empty', 'question must not be empty in E01.')
    require(request['predicted_class'] in CLASSES, 'class', 'Unknown predicted_class.')
    tasks = request['task_types']
    string_list(tasks, 'task_types', nonempty=True, unique=True)
    require(set(tasks) <= TASKS, 'tasks', 'Unknown task type.')
    context = request['explicitly_allowed_operating_context']
    require(type(context) is dict and set(context) <= set(CONTEXT),
            'context_fields', 'Unapproved context fields or type.')
    for key, value in context.items():
        require(type(value) is str and value in CONTEXT[key],
                'context_value', 'Unapproved context value: ' + key)
    return request


def read_bytes(path):
    path = Path(path)
    require(not path.is_symlink(), 'symlink', 'Refuse symbolic link: ' + str(path))
    require(path.is_file(), 'file', 'Missing regular file: ' + str(path))
    return path.read_bytes()


def read_json(path):
    return load_strict_json(read_bytes(path))


def read_requests(path):
    text = read_bytes(path).decode('utf-8')
    lines = text.splitlines()
    require(bool(lines) and all(line.strip() for line in lines),
            'jsonl_lines', 'Request JSONL must contain nonblank lines.')
    requests = [load_strict_json(line) for line in lines]
    for request in requests:
        validate_business_request(request)
    return requests


def load_basis(root):
    blobs = {name: read_bytes(Path(root) / name) for name in BASIS_PATHS}
    tables = {}
    for path, blob in blobs.items():
        if path.endswith('.csv'):
            rows = list(csv.DictReader(io.StringIO(blob.decode('utf-8'), newline='')))
            id_key = {'scenarios.csv': 'scenario_id', 'sources.csv': 'source_id',
                      'evidence.csv': 'evidence_id'}[Path(path).name]
            require(len({row[id_key] for row in rows}) == len(rows),
                    'knowledge_id', 'Duplicate ID in ' + path)
            tables[path] = {row[id_key]: row for row in rows}
    return blobs, tables


def validate_basis_ref(ref, blobs, tables):
    object_fields(ref, frozenset(('path', 'locator', 'exact_value')), 'basis ref')
    path, locator, expected = (ref[key] for key in ('path', 'locator', 'exact_value'))
    string(path, 'basis path')
    string(locator, 'basis locator')
    require(path in BASIS_PATHS, 'basis_path', 'Basis path is outside explicit whitelist.')
    if path.endswith('.csv'):
        require(':' in locator, 'basis_locator', 'CSV locator must be row ID:field.')
        row_id, field = locator.split(':', 1)
        row = tables[path].get(row_id)
        require(row is not None and field in row, 'basis_locator', 'Unknown CSV row or field.')
        require(type(expected) is str and expected == row[field],
                'basis_value', 'CSV basis value differs at ' + locator)
    elif path.endswith('.json'):
        actual = load_strict_json(blobs[path])
        for key in locator.split('.'):
            require(type(actual) is dict and key in actual, 'basis_locator', 'Unknown JSON path.')
            actual = actual[key]
        require(type(actual) is type(expected) and actual == expected,
                'basis_value', 'JSON basis value differs at ' + locator)
    else:
        require(locator == 'verbatim_substring', 'basis_locator', 'Text locator must be verbatim_substring.')
        string(expected, 'basis exact_value')
        require(expected in blobs[path].decode('utf-8'), 'basis_value', 'Text basis substring is absent.')


def load_bound_requests(dataset_dir):
    folder = Path(dataset_dir)
    bindings = read_json(folder / 'question_bindings.json')
    require(type(bindings) is list and bool(bindings), 'bindings', 'Bindings must be a nonempty list.')
    requests = {split: read_requests(folder / (split + '_requests.jsonl')) for split in SPLITS}
    seen_ids, seen_slots = set(), set()
    by_id = {}
    for binding in bindings:
        object_fields(binding, BINDING_FIELDS, 'question binding')
        qid, split, line = (binding[key] for key in ('question_id', 'split', 'line_number'))
        stable_id(qid, 'question_id')
        stable_id(binding['family_id'], 'family_id')
        require(qid not in seen_ids, 'duplicate_question', 'Duplicate question ID.')
        require(split in SPLITS, 'split', 'Unknown split.')
        require(type(line) is int and 1 <= line <= len(requests[split]),
                'line_number', 'Binding line is outside request file.')
        require((split, line) not in seen_slots, 'duplicate_binding', 'Request line bound twice.')
        request = requests[split][line - 1]
        require(binding['request_sha256'] == digest(request), 'request_hash', 'Request hash differs: ' + qid)
        seen_ids.add(qid)
        seen_slots.add((split, line))
        by_id[qid] = (binding, request)
    all_slots = {(split, line) for split in SPLITS for line in range(1, len(requests[split]) + 1)}
    require(seen_slots == all_slots, 'unbound_request', 'Every request line must be bound exactly once.')
    return bindings, requests, by_id


def export_request(dataset_dir, question_id):
    """Export only a validated bound request; ignore every adjacent file."""
    _, _, by_id = load_bound_requests(dataset_dir)
    require(question_id in by_id, 'question_id', 'Unknown question ID for export.')
    request = by_id[question_id][1]
    # Explicit field construction remains safe if the storage layout expands.
    return {key: request[key] for key in sorted(REQUEST_FIELDS)}


def validate_dataset(dataset_dir, root=ROOT):
    folder = Path(dataset_dir)
    design = read_json(folder / 'design_contract.json')
    require(type(design) is dict and design.get('fixed_before_question_authoring') is True,
            'design', 'Question design must have been fixed before authoring.')
    bindings, requests, by_id = load_bound_requests(folder)
    require(type(design.get('question_count')) is int and len(bindings) == design['question_count'],
            'count', 'Question count differs from design.')
    require(type(design.get('split_counts')) is dict and set(design['split_counts']) == set(SPLITS),
            'design', 'Design split counts are missing.')
    require(type(design.get('family_allocation')) is dict and set(design['family_allocation']) == set(SPLITS),
            'design', 'Design family allocation is missing.')
    families = {}
    for split in SPLITS:
        expected = design['split_counts'][split]
        require(type(expected) is int and len(requests[split]) == expected, 'count', 'Split count differs: ' + split)
        actual = {binding['family_id'] for binding in bindings if binding['split'] == split}
        string_list(design['family_allocation'][split], 'family allocation', nonempty=True, unique=True)
        require(actual == set(design['family_allocation'][split]), 'family_design', 'Family allocation differs: ' + split)
        families[split] = actual
    require(not families['debug'] & families['heldout'], 'family_leakage', 'A family occurs in both splits.')
    question_texts = [request['question'].strip().casefold() for split in SPLITS for request in requests[split]]
    require(len(question_texts) == len(set(question_texts)), 'duplicate_text', 'Duplicate question text.')
    blobs, tables = load_basis(root)
    evidence = tables['knowledge/k01/evidence.csv']
    rubrics = read_json(folder / 'rubrics.json')
    require(type(rubrics) is list and len(rubrics) == len(bindings), 'rubrics', 'Rubric count differs.')
    seen_rubrics, seen_points, all_relevant = set(), set(), set()
    point_count = basis_ref_count = citation_count = support_set_count = 0
    for rubric in rubrics:
        object_fields(rubric, RUBRIC_FIELDS, 'rubric')
        qid = rubric['question_id']
        require(type(qid) is str and qid in by_id and qid not in seen_rubrics,
                'rubric_binding', 'Unknown or duplicate rubric question.')
        binding, request = by_id[qid]
        require(rubric['split'] == binding['split'] and rubric['family_id'] == binding['family_id'],
                'rubric_binding', 'Rubric split/family differs from binding.')
        require(request['predicted_class'] == 'outer_race' and request['scenario_id'] == 'SC-K01-OR-001',
                'single_scenario', 'Normal E01 set is limited to the registered outer-race scenario.')
        string_list(rubric['tasks'], 'rubric tasks', nonempty=True, unique=True)
        require(set(rubric['tasks']) == set(request['task_types']), 'tasks', 'Rubric tasks differ from business tasks.')
        for field in ('answerability', 'exposure_note'):
            string(rubric[field], field)
        for field in ('forbidden_claims', 'required_unknowns', 'source_limitations', 'manual_confirmation'):
            string_list(rubric[field], field, nonempty=True)
        insufficiency = rubric['insufficiency_criteria']
        require(type(insufficiency) is list and len(insufficiency) == 3,
                'insufficiency_criteria', 'Each rubric needs exactly three insufficiency criteria.')
        criterion_ids = []
        for criterion in insufficiency:
            object_fields(criterion, frozenset(('criterion_id', 'applicable', 'full_credit',
                                              'partial_credit', 'zero_credit')), 'insufficiency criterion')
            require(type(criterion['criterion_id']) is str and criterion['criterion_id'] in ('i1', 'i2', 'i3'),
                    'insufficiency_criteria', 'Insufficiency criterion ID must be i1, i2 or i3.')
            require(type(criterion['applicable']) is bool, 'insufficiency_criteria',
                    'Insufficiency applicability must be boolean.')
            for field in ('full_credit', 'partial_credit', 'zero_credit'):
                string(criterion[field], 'insufficiency ' + field)
            criterion_ids.append(criterion['criterion_id'])
        require(set(criterion_ids) == {'i1', 'i2', 'i3'}, 'insufficiency_criteria',
                'Insufficiency criterion IDs must occur exactly once each.')
        relevant = rubric['relevant_evidence_ids']
        string_list(relevant, 'relevant evidence', unique=True)
        require(set(relevant) <= set(evidence), 'evidence_id', 'Unknown relevant evidence.')
        all_relevant.update(relevant)
        require(type(rubric['points']) is list and bool(rubric['points']), 'points', 'Each rubric needs atomic points.')
        point_tasks = set()
        for point in rubric['points']:
            object_fields(point, POINT_FIELDS, 'point')
            stable_id(point['point_id'], 'point_id')
            require(point['point_id'] not in seen_points, 'duplicate_point', 'Duplicate point ID.')
            seen_points.add(point['point_id'])
            require(point['task_type'] in rubric['tasks'], 'point_task', 'Point task is outside request tasks.')
            point_tasks.add(point['task_type'])
            for field in ('claim', 'full_credit', 'partial_credit', 'unsupported_credit'):
                string(point[field], field)
            string_list(point['conditions'], 'conditions')
            require(point['basis_kind'] in ('registered_knowledge', 'project_boundary', 'explicitly_unsupported'),
                    'basis_kind', 'Unknown point basis kind.')
            require(type(point['basis_refs']) is list, 'basis_refs', 'basis_refs must be a list.')
            for ref in point['basis_refs']:
                validate_basis_ref(ref, blobs, tables)
                basis_ref_count += 1
            combos = point['support_any_of']
            require(type(combos) is list, 'support', 'support_any_of must be a list.')
            supporting = set()
            combo_keys = []
            for combo in combos:
                string_list(combo, 'support combination', nonempty=True, unique=True)
                require(set(combo) <= set(evidence), 'evidence_id', 'Unknown support evidence.')
                supporting.update(combo)
                combo_keys.append(tuple(sorted(combo)))
            require(len(combo_keys) == len(set(combo_keys)), 'support', 'Duplicate support alternatives.')
            if point['basis_kind'] == 'registered_knowledge':
                require(bool(combos), 'unsupported_point', 'Registered-knowledge point has no support.')
            else:
                require(bool(point['basis_refs']), 'unsupported_point', 'Boundary/unsupported point needs an explicit basis.')
            if point['basis_kind'] == 'explicitly_unsupported':
                require(not combos, 'unsupported_point', 'Explicitly unsupported claim must not have support alternatives.')
            require(type(point['citations']) is list, 'citations', 'Point citations must be a list.')
            cited = set()
            for citation in point['citations']:
                require(type(citation) is dict and type(citation.get('evidence_id')) is str,
                        'citations', 'Citation needs an evidence ID.')
                eid = citation['evidence_id']
                require(eid in evidence and citation == evidence[eid],
                        'citation_copy', 'Citation must exactly equal complete registered row: ' + eid)
                require(eid not in cited, 'citations', 'Duplicate citation.')
                cited.add(eid)
            require(supporting <= cited and cited <= set(relevant),
                    'citation_coverage', 'Support IDs need complete citations and relevant-evidence membership.')
            require(bool(combos) or bool(point['basis_refs']), 'unsupported_point', 'Point has neither evidence nor explicit basis.')
            citation_count += len(cited)
            support_set_count += len(combos)
            point_count += 1
        require(point_tasks == set(rubric['tasks']), 'task_coverage', 'Each requested task needs at least one scoring point.')
        seen_rubrics.add(qid)
    require(seen_rubrics == set(by_id), 'rubric_binding', 'Questions and rubrics are not one-to-one.')
    boundary_cases = read_json(folder / 'boundary_cases.json')
    contract_cases = read_json(folder / 'contract_cases.json')
    seen_cases = set(by_id) | seen_points
    require(type(boundary_cases) is list and type(contract_cases) is list, 'cases', 'Separate cases must be lists.')
    for case in boundary_cases:
        object_fields(case, frozenset(('case_id', 'request', 'expected_scope_status', 'scoring_scope', 'reason')), 'boundary case')
        stable_id(case['case_id'], 'case_id')
        require(case['case_id'] not in seen_cases, 'duplicate_case', 'Duplicate case ID.')
        seen_cases.add(case['case_id'])
        validate_business_request(case['request'])
        require(case['request']['scenario_id'] != 'SC-K01-OR-001' or case['request']['predicted_class'] != 'outer_race',
                'boundary_case', 'Boundary example must be outside current covered class/scenario.')
        require(case['scoring_scope'] == 'excluded_from_QA_quality', 'boundary_case', 'Boundary case must be excluded from quality denominator.')
        string(case['expected_scope_status'], 'expected_scope_status')
        string(case['reason'], 'boundary reason')
    for case in contract_cases:
        object_fields(case, frozenset(('case_id', 'payload_json', 'expected', 'reason')), 'contract case')
        stable_id(case['case_id'], 'case_id')
        require(case['case_id'] not in seen_cases, 'duplicate_case', 'Duplicate case ID.')
        seen_cases.add(case['case_id'])
        string(case['payload_json'], 'payload_json')
        string(case['reason'], 'contract reason')
        require(case['expected'] == 'reject', 'contract_case', 'Contract case must expect rejection.')
        rejected = False
        try:
            validate_business_request(load_strict_json(case['payload_json']))
        except E01Error:
            rejected = True
        require(rejected, 'contract_case', 'Contract bad case was unexpectedly accepted: ' + case['case_id'])
    separate = design.get('separate_sets')
    require(type(separate) is dict and separate.get('boundary_cases') == len(boundary_cases)
            and separate.get('contract_bad_cases') == len(contract_cases)
            and separate.get('include_in_QA_quality_denominator') is False,
            'case_counts', 'Separate-set counts or exclusion differ from design.')
    return {'status': 'PASS', 'validator_version': VERSION,
            'scope': 'E01 current single-scenario preparation; no retrieval or generated answers',
            'question_count': len(bindings), 'split_counts': {s: len(requests[s]) for s in SPLITS},
            'family_counts': {s: len(families[s]) for s in SPLITS},
            'point_count': point_count, 'basis_ref_count': basis_ref_count,
            'registered_relevant_evidence_ids': sorted(all_relevant),
            'complete_citation_copy_count': citation_count, 'support_alternative_count': support_set_count,
            'boundary_case_count': len(boundary_cases), 'contract_bad_case_count': len(contract_cases),
            'checked_files': {name: hashlib.sha256(read_bytes(folder / name)).hexdigest() for name in DATA_FILES},
            'knowledge_basis_hashes': {name: hashlib.sha256(blob).hexdigest() for name, blob in sorted(blobs.items())},
            'limitations': ['Checks exact registered row/value copies, not manufacturer semantic truth.',
                            'Field whitelist cannot identify all free-text leakage or false premises.',
                            'Natural English, intent separation and heldout exposure require documented human/agent review.',
                            'Support alternatives are OR sets; evidence within one alternative is an AND set.']}


def write_new_json(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'output_exists', 'Refuse existing output: ' + str(path))
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New validation report path.')
    parser.add_argument('--export-id', help='Optional explicit request ID; E01 runs use synthetic export only.')
    parser.add_argument('--export-output', type=Path)
    args = parser.parse_args(argv)
    require(bool(args.export_id) == bool(args.export_output), 'arguments', 'Both export arguments are required together.')
    # Check destinations before validation so reruns cannot replace evidence.
    for path in (args.output, args.export_output):
        if path is not None:
            require(not path.exists() and not path.is_symlink(), 'output_exists', 'Refuse existing output: ' + str(path))
    if args.export_output is not None:
        require(args.export_output.resolve() != args.output.resolve(), 'arguments', 'Report and export paths must differ.')
    try:
        report = validate_dataset(args.dataset)
        exported = export_request(args.dataset, args.export_id) if args.export_id else None
    except (E01Error, OSError, UnicodeError) as error:
        report = {'status': 'FAIL', 'validator_version': VERSION,
                  'error_code': getattr(error, 'code', type(error).__name__), 'message': str(error)}
        write_new_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 1
    if exported is not None:
        write_new_json(args.export_output, exported)
    write_new_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except E01Error as error:
        print(json.dumps({'status': 'FAIL', 'error_code': error.code, 'message': str(error)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
