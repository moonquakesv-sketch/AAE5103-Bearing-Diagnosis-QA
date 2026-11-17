"""Bounded Q01 repair regressions using abstract, independently authored doubles.

Set Q01_REPAIR_RUN_DIR to a fresh repair-run directory. Prepare expectations
before importing Q01 with: python tests/test_q01_repairs.py --prepare-oracle
Then run this file through unittest discovery. All artifacts remain in that run.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import unittest


ROOT = Path(__file__).resolve().parents[1]
STAGES = ('retrieval', 'generate', 'basic_check', 'verify', 'revise')
GROUPS = ('T', 'G', 'TV', 'GV')
EVIDENCE_FIELDS = ('evidence_id source_id knowledge_id task_type source_url source_version '
                   'pdf_file_page printed_page page_or_section original_text chinese_paraphrase '
                   'claim_en conditions limitations modality applicability reviewed_by').split()
SOURCE_FIELDS = ('source_id title organization_author version_or_date original_url access_date '
                 'timezone format page_or_section applicability full_text_status local_path sha256 '
                 'local_copy_status access_limitations rights_use reviewed_by').split()
REQUEST = {'predicted_class': 'outer_race', 'scenario_id': 'SC-K01-OR-001',
           'question': 'SYNTHETIC REPAIR: discuss three abstract task tokens.',
           'task_types': ['explanation', 'inspection', 'preparation'],
           'explicitly_allowed_operating_context': {}}
TEXT = 'SYNTHETIC REPAIR placeholder using only the supplied abstract card [REPAIR-EV-1].'
JSON_DIAGNOSTIC = {'label': 'SYNTHETIC JSON diagnostic', 'nested': {'list': [0, 1.25, True, None, '']}}


def run_directory():
    value = os.environ.get('Q01_REPAIR_RUN_DIR')
    if not value:
        raise RuntimeError('Q01_REPAIR_RUN_DIR is required; no default output location.')
    run = Path(value).absolute()
    if not run.is_relative_to(ROOT / 'runs' / 'q01_fix') or not run.is_dir():
        raise RuntimeError('Output must be an existing project runs/q01_fix directory.')
    if run.is_symlink():
        raise RuntimeError('Repair output directory must not be a symlink.')
    return run


def expected_cases():
    rows = []
    for group in GROUPS:
        verified = int(group in ('TV', 'GV'))
        for family in ('partial_retrieval_sufficient', 'retrieved_partial', 'retrieved_insufficient',
                       'json_diagnostics_preserved'):
            status = 'evidence_insufficient'
            publish = family != 'retrieved_insufficient'
            if family == 'json_diagnostics_preserved':
                status = 'completed_verified_synthetic' if verified else 'completed_unverified'
            rows.append({'case': family + '_' + group, 'family': family, 'group': group,
                         'status': status, 'publishable': publish,
                         'final_answer': TEXT if publish else None,
                         'service_message_required': not publish,
                         'counts': dict(zip(STAGES, (1, 1, 1, verified, 0))),
                         'revisions': 0, 'verification': verified})
    for group in ('TV', 'GV'):
        rows.append({'case': 'partial_after_revision_' + group,
                     'family': 'partial_after_revision', 'group': group,
                     'status': 'evidence_insufficient', 'publishable': True,
                     'final_answer': TEXT, 'service_message_required': False,
                     'counts': dict(zip(STAGES, (1, 1, 2, 2, 1))),
                     'revisions': 1, 'verification': 2})
    for origin, status in (('config', 'invalid_config'), ('identity', 'input_or_setup_failed'),
                           ('retrieval', 'retrieval_failed')):
        for kind in ('set', 'nan', 'path', 'cycle'):
            rows.append({'case': origin + '_' + kind, 'family': 'unsafe_' + origin,
                         'bad_value_kind': kind, 'group': 'TV', 'status': status,
                         'publishable': False, 'final_answer': None,
                         'service_message_required': False,
                         'counts': dict(zip(STAGES, (int(origin == 'retrieval'), 0, 0, 0, 0))),
                         'revisions': 0, 'verification': 0})
    return rows


def prepare_oracle():
    base = run_directory() / 'regression'
    base.mkdir(exist_ok=True)
    target = base / 'expected_cases.json'
    payload = {'schema': 'q01.repair.independent_oracle.v1',
               'independence': 'Newly authored synthetic doubles; no author test or sealed review probe imported/copied.',
               'semantics': 'Partial retrieval preserves a known task-coverage limitation; this is not a semantic entailment assertion.',
               'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'reviewed_code_before_sha256': hashlib.sha256((ROOT / 'scripts/q01_flow.py').read_bytes()).hexdigest(),
               'request': REQUEST, 'normal_json_diagnostic': JSON_DIAGNOSTIC,
               'cases': expected_cases()}
    with target.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=True, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({'prepared': str(target), 'cases': len(payload['cases']),
                      'expected_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                      'code_before_sha256': payload['reviewed_code_before_sha256']}))


def card(identifier):
    value = {field: 'SYNTHETIC REPAIR ' + field for field in EVIDENCE_FIELDS}
    value.update(evidence_id=identifier, source_id='REPAIR-SOURCE', task_type='inspection')
    value['source'] = {field: 'SYNTHETIC REPAIR SOURCE ' + field for field in SOURCE_FIELDS}
    value['source']['source_id'] = 'REPAIR-SOURCE'
    value.update(rank=int(identifier.rsplit('-', 1)[1]), score=0.25, relation_paths=[])
    return value


def retrieved(partial=False):
    cards = [card('REPAIR-EV-' + str(i)) for i in (1, 2, 3)]
    if not partial:
        for task, evidence in zip(REQUEST['task_types'], cards):
            evidence['task_type'] = task
    result = {'status': 'partial_task_coverage' if partial else 'retrieved',
              'reason': 'SYNTHETIC REPAIR retrieval outcome', 'request': copy.deepcopy(REQUEST),
              'results': cards, 'returned_count': 3,
              'information_separation': {'demo_scenario': {
                  'assumption_text': 'SYNTHETIC independent assumption',
                  'unknown_information': 'SYNTHETIC unknown',
                  'cannot_infer': 'SYNTHETIC scope limit'}},
              'per_task': {task: {'status': 'retrieved', 'returned_ids': [cards[index]['evidence_id']],
                                  'candidate_count': 1, 'positive_count': 1,
                                  'reason': 'SYNTHETIC abstract task coverage'}
                           for index, task in enumerate(REQUEST['task_types'])}}
    if partial:
        result['per_task'] = {
            'explanation': {'status': 'budget_exhausted', 'returned_ids': [], 'candidate_count': 1,
                            'positive_count': 1, 'reason': 'SYNTHETIC global 3-card budget used'},
            'inspection': {'status': 'retrieved', 'returned_ids': ['REPAIR-EV-1', 'REPAIR-EV-2', 'REPAIR-EV-3'],
                           'candidate_count': 3, 'positive_count': 3, 'reason': 'SYNTHETIC three cards'},
            'preparation': {'status': 'zero_similarity', 'returned_ids': [], 'candidate_count': 1,
                            'positive_count': 0, 'reason': 'SYNTHETIC no positive card'}}
    return result


def bad_value(kind):
    if kind == 'set':
        return {'SYNTHETIC_UNRECORDABLE_SET'}
    if kind == 'nan':
        return float('nan')
    if kind == 'path':
        return run_directory() / 'SYNTHETIC_NO_FILE_READ'
    value = {'label': 'SYNTHETIC circular diagnostic'}
    value['cycle'] = value
    return value


class SyntheticRetriever:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def retrieve(self, request, mode, top_k):
        self.calls.append({'request': copy.deepcopy(request), 'mode': mode, 'top_k': top_k})
        return copy.deepcopy(self.result)


class SyntheticProvider:
    def __init__(self, sufficiency='sufficient', revise=False):
        self.identity = {'kind': 'synthetic_repair_double'}
        self.calls = []
        self.sufficiency = sufficiency
        self.revision_requested = revise
        self.verify_count = 0

    def generate(self, payload):
        self.calls.append({'action': 'generate', 'payload': copy.deepcopy(payload)})
        return {'answer': TEXT, 'citations': ['REPAIR-EV-1'], 'evidence_sufficiency': self.sufficiency}

    def verify(self, payload):
        self.calls.append({'action': 'verify', 'payload': copy.deepcopy(payload)})
        self.verify_count += 1
        if self.revision_requested and self.verify_count == 1:
            return {'verdict': 'issues', 'issues': ['SYNTHETIC one allowed revision required']}
        return {'verdict': 'pass', 'issues': []}

    def revise(self, payload):
        self.calls.append({'action': 'revise', 'payload': copy.deepcopy(payload)})
        return {'answer': TEXT, 'citations': ['REPAIR-EV-1'], 'evidence_sufficiency': 'sufficient'}


class Q01RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = run_directory() / 'regression'
        expected_bytes = (cls.base / 'expected_cases.json').read_bytes()
        cls.oracle = json.loads(expected_bytes)
        actual_test_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if cls.oracle['test_sha256'] != actual_test_hash:
            raise RuntimeError('Test source changed after expectation sealing; preserve and re-review expectations.')
        cls.attempt = Path(tempfile.mkdtemp(prefix='attempt_', dir=cls.base))
        cls.observations = []
        cls.binding = {'expected_sha256': hashlib.sha256(expected_bytes).hexdigest(),
                       'test_sha256': actual_test_hash,
                       'code_sha256': hashlib.sha256((ROOT / 'scripts/q01_flow.py').read_bytes()).hexdigest(),
                       'argv': sys.argv, 'cwd': str(Path.cwd()),
                       'environment': {k: os.environ.get(k) for k in
                                       ('PYTHONDONTWRITEBYTECODE', 'TMPDIR', 'Q01_REPAIR_RUN_DIR')}}
        (cls.attempt / 'binding.json').write_text(json.dumps(cls.binding, indent=2) + '\n')
        sys.path.insert(0, str(ROOT / 'scripts'))
        import q01_flow
        cls.q01 = q01_flow

    @classmethod
    def tearDownClass(cls):
        summary = {'binding': cls.binding, 'case_count': len(cls.observations),
                   'passed': sum(row['pass'] for row in cls.observations), 'cases': cls.observations,
                   'code_after_sha256': hashlib.sha256((ROOT / 'scripts/q01_flow.py').read_bytes()).hexdigest()}
        (cls.attempt / 'comparison.json').write_text(json.dumps(summary, ensure_ascii=True, allow_nan=False, indent=2) + '\n')
        print(json.dumps({'repair_attempt': str(cls.attempt), 'cases': summary['case_count'],
                          'passed': summary['passed'], 'failed': summary['case_count'] - summary['passed']}))

    def execute_case(self, expected):
        family = expected['family']
        partial = family in ('partial_retrieval_sufficient', 'partial_after_revision')
        config = self.q01.group_config(expected['group'])
        result = retrieved(partial)
        sufficiency = {'retrieved_partial': 'partial', 'retrieved_insufficient': 'insufficient'}.get(family, 'sufficient')
        provider = SyntheticProvider(sufficiency, revise=family == 'partial_after_revision')
        if family == 'unsafe_config':
            config['SYNTHETIC_bad_metadata'] = bad_value(expected['bad_value_kind'])
        elif family == 'unsafe_identity':
            provider.identity['metadata'] = bad_value(expected['bad_value_kind'])
        elif family == 'unsafe_retrieval':
            result['metadata'] = bad_value(expected['bad_value_kind'])
        elif family == 'json_diagnostics_preserved':
            provider.identity['metadata'] = copy.deepcopy(JSON_DIAGNOSTIC)
            result['metadata'] = copy.deepcopy(JSON_DIAGNOSTIC)
            result['results'][0]['metadata'] = copy.deepcopy(JSON_DIAGNOSTIC)
        retriever = SyntheticRetriever(result)
        record = None
        errors = []
        exception = None
        serialized = False
        try:
            record = self.q01.run_flow(copy.deepcopy(REQUEST), config, provider, retriever=retriever)
        except Exception:
            exception = traceback.format_exc()
            errors.append('run_flow raised instead of returning a diagnostic record')
        if record is not None:
            try:
                record_path = self.attempt / (expected['case'] + '.record.json')
                self.q01.write_record(record_path, record)
                decoded = json.loads(record_path.read_bytes())
                if decoded != record:
                    errors.append('JSON record readback differs from returned record')
                json.dumps(decoded, allow_nan=False)
                serialized = True
            except Exception:
                exception = traceback.format_exc()
                errors.append('record cannot be written/read back as strict JSON')
                (self.attempt / (expected['case'] + '.record_repr.txt')).write_text(repr(record), encoding='utf-8')
            for key in ('status', 'publishable', 'final_answer', 'counts'):
                if record.get(key) != expected[key]:
                    errors.append(key + ': expected ' + repr(expected[key]) + ', actual ' + repr(record.get(key)))
            if expected['service_message_required'] and not record.get('service_message'):
                errors.append('insufficient candidate lacks separate service_message')
            for field, key in (('revisions', 'revisions'), ('verification', 'verification')):
                if len(record[field]) != expected[key]:
                    errors.append('unexpected diagnostic count: ' + field)
            if record.get('semantic_support') != 'not_established':
                errors.append('structural checks promoted semantic support')
            if len(retriever.calls) != expected['counts']['retrieval']:
                errors.append('actual retriever invocations exceed expected budget')
            expected_actions = sum(expected['counts'][k] for k in ('generate', 'verify', 'revise'))
            if len(provider.calls) != expected_actions:
                errors.append('actual provider invocations exceed expected budget')
            for stage, count in record['counts'].items():
                elapsed = record['timings_seconds'][stage]
                if count == 0 and elapsed is not None:
                    errors.append('skipped stage has invented timing: ' + stage)
                elif count and not (type(elapsed) in (int, float) and elapsed > 0):
                    errors.append('executed stage lacks measured timing: ' + stage)
            if family.startswith('unsafe_'):
                if not record.get('reason'):
                    errors.append('rejected value lacks an error reason')
                if serialized:
                    diagnostic_text = json.dumps(record, ensure_ascii=True).lower()
                    tokens = {'set': ('set',), 'nan': ('nan', 'float', 'finite'),
                              'path': ('path',), 'cycle': ('circular', 'cycle')}[expected['bad_value_kind']]
                    if not any(token in diagnostic_text for token in tokens):
                        errors.append('safe diagnostic lost the rejected value type/error identity')
            else:
                if record['retrieval'] != result or record['returned_cards'] != result['results']:
                    errors.append('JSON-compatible retrieval/card diagnostics changed')
                for call in provider.calls:
                    if call['payload']['request'] != REQUEST or len(call['payload']['evidence_cards']) != 3:
                        errors.append('shared request or complete-card budget changed')
                if family == 'json_diagnostics_preserved' and record['identity']['provider'] != provider.identity:
                    errors.append('JSON-compatible provider identity was not preserved')
        row = {'case': expected['case'], 'expected': expected, 'pass': not errors,
               'differences': errors, 'exception': exception, 'strict_json_readback': serialized,
               'provider_actions': [call['action'] for call in provider.calls],
               'retriever_calls': retriever.calls}
        if record is not None:
            row['actual'] = {key: record.get(key) for key in
                             ('status', 'publishable', 'final_answer', 'counts', 'reason')}
        self.observations.append(row)
        self.assertFalse(errors, '\n'.join(errors))

    def test_coverage_and_candidate_sufficiency(self):
        for case in self.oracle['cases']:
            if not case['family'].startswith('unsafe_'):
                with self.subTest(case=case['case']):
                    self.execute_case(case)

    def test_unrecordable_config_identity_and_retrieval(self):
        for case in self.oracle['cases']:
            if case['family'].startswith('unsafe_'):
                with self.subTest(case=case['case']):
                    self.execute_case(case)


if __name__ == '__main__':
    if sys.argv[1:] == ['--prepare-oracle']:
        prepare_oracle()
    else:
        unittest.main(verbosity=2)
