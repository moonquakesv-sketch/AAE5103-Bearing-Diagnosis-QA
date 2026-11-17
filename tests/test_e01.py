"""Synthetic E01 guard/export checks; never invokes K02 or real heldout requests.

E01_TEST_RUN must identify this run's new directory. All fixture writes stay
there. These checks are interface/fixture tests, not QA performance samples.
"""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('validate_e01', ROOT / 'scripts/validate_e01.py')
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def business(question='This synthetic fixture asks about a possible mechanism.'):
    return {'predicted_class': 'outer_race', 'scenario_id': 'SC-K01-OR-001',
            'question': question, 'task_types': ['explanation'],
            'explicitly_allowed_operating_context': {}}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def fixture(folder):
    with (ROOT / 'knowledge/k01/evidence.csv').open(encoding='utf-8', newline='') as stream:
        evidence = next(csv.DictReader(stream))
    with (ROOT / 'knowledge/k01/scenarios.csv').open(encoding='utf-8', newline='') as stream:
        scenario = next(csv.DictReader(stream))
    design = {'fixed_before_question_authoring': True, 'question_count': 2,
              'split_counts': {'debug': 1, 'heldout': 1},
              'family_allocation': {'debug': ['synthetic_debug'], 'heldout': ['synthetic_reserved']},
              'separate_sets': {'boundary_cases': 1, 'contract_bad_cases': 1,
                                'include_in_QA_quality_denominator': False}}
    bindings, rubrics = [], []
    for split, qid, family in [('debug', 'SYN-D01', 'synthetic_debug'),
                               ('heldout', 'SYN-H01', 'synthetic_reserved')]:
        request = business('This synthetic ' + split + ' fixture asks about a possible mechanism.')
        (folder / (split + '_requests.jsonl')).write_text(v.canonical(request) + '\n', encoding='utf-8')
        bindings.append({'question_id': qid, 'split': split, 'line_number': 1,
                         'request_sha256': v.digest(request), 'family_id': family})
        point = {'point_id': qid + '-P01', 'task_type': 'explanation',
                 'claim': 'Synthetic rubric statement; this is not a generated answer.',
                 'conditions': ['Retain the stationary external-vibration condition.'],
                 'support_any_of': [[evidence['evidence_id']]],
                 'basis_kind': 'registered_knowledge',
                 'basis_refs': [{'path': 'knowledge/k01/scenarios.csv',
                                 'locator': scenario['scenario_id'] + ':cannot_infer',
                                 'exact_value': scenario['cannot_infer']}],
                 'citations': [copy.deepcopy(evidence)],
                 'full_credit': 'Synthetic: supported claim with conditions.',
                 'partial_credit': 'Synthetic: only part of the claim.',
                 'unsupported_credit': 'Synthetic: zero for invented certainty.'}
        rubrics.append({'question_id': qid, 'split': split, 'family_id': family,
                        'tasks': ['explanation'], 'answerability': 'answerable_with_conditions',
                        'points': [point], 'relevant_evidence_ids': [evidence['evidence_id']],
                        'forbidden_claims': ['Synthetic forbidden claim.'],
                        'required_unknowns': ['Synthetic unknown.'],
                        'source_limitations': ['Registered sources do not establish semantic truth.'],
                        'manual_confirmation': ['Synthetic review marker; no human scoring occurred.'],
                        'insufficiency_criteria': [
                            {'criterion_id': criterion_id, 'applicable': False,
                             'full_credit': 'Synthetic full uncertainty credit.',
                             'partial_credit': 'Synthetic partial uncertainty credit.',
                             'zero_credit': 'Synthetic zero uncertainty credit.'}
                            for criterion_id in ('i1', 'i2', 'i3')],
                        'exposure_note': 'Synthetic fixture, not an E01 question.'})
    boundary = business('This synthetic fixture asks an uncovered question.')
    boundary['predicted_class'] = 'inner_race'
    boundary['scenario_id'] = 'SC-SYNTHETIC-UNCOVERED'
    boundaries = [{'case_id': 'SYN-B01', 'request': boundary,
                   'expected_scope_status': 'uncovered', 'scoring_scope': 'excluded_from_QA_quality',
                   'reason': 'Synthetic boundary case.'}]
    bad = business()
    bad['rubric'] = 'SYNTHETIC_SECRET'
    contracts = [{'case_id': 'SYN-C01', 'payload_json': json.dumps(bad),
                  'expected': 'reject', 'reason': 'Synthetic forbidden extra field.'}]
    for name, value in [('design_contract.json', design), ('question_bindings.json', bindings),
                        ('rubrics.json', rubrics), ('boundary_cases.json', boundaries),
                        ('contract_cases.json', contracts)]:
        write_json(folder / name, value)
    return design, bindings, rubrics


class E01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir = Path(os.environ['E01_TEST_RUN']).resolve()
        if not cls.run_dir.is_dir() or not cls.run_dir.is_relative_to(ROOT / 'runs/e01'):
            raise RuntimeError('E01_TEST_RUN must be an existing new runs/e01 directory.')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_validator_', dir=self.run_dir)
        self.folder = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.design, self.bindings, self.rubrics = fixture(self.folder)

    def reject_dataset(self, code):
        with self.assertRaises(v.E01Error) as caught:
            v.validate_dataset(self.folder)
        self.assertEqual(caught.exception.code, code)

    def test_synthetic_valid_dataset_and_report_units(self):
        report = v.validate_dataset(self.folder)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['question_count'], 2)
        self.assertEqual(report['point_count'], 2)
        self.assertEqual(report['complete_citation_copy_count'], 2)
        self.assertEqual(report['contract_bad_case_count'], 1)

    def test_strict_json_rejects_duplicate_nonfinite_and_malformed(self):
        for data in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{']:
            with self.subTest(data=data), self.assertRaises(v.E01Error):
                v.load_strict_json(data)

    def test_business_whitelist_types_context_and_unicode(self):
        cases = []
        for extra in ('question_id', 'split', 'rubric', 'retrieval_mode', 'top_k'):
            request = business()
            request[extra] = 'SYNTHETIC_SECRET'
            cases.append(request)
        request = business()
        request['task_types'] = 'explanation'
        cases.append(request)
        request = business()
        request['task_types'] = ['explanation', 'explanation']
        cases.append(request)
        request = business()
        request['explicitly_allowed_operating_context'] = {'operating_state': {'answer': 'secret'}}
        cases.append(request)
        request = business()
        request['explicitly_allowed_operating_context'] = {'answer': 'secret'}
        cases.append(request)
        request = business()
        request['question'] = '\ud800'
        cases.append(request)
        for request in cases:
            with self.subTest(keys=list(request)), self.assertRaises(v.E01Error):
                v.validate_business_request(request)

    def test_family_isolation_rejects_same_intent_family_across_splits(self):
        self.bindings[1]['family_id'] = 'synthetic_debug'
        self.design['family_allocation']['heldout'] = ['synthetic_debug']
        self.rubrics[1]['family_id'] = 'synthetic_debug'
        write_json(self.folder / 'question_bindings.json', self.bindings)
        write_json(self.folder / 'design_contract.json', self.design)
        write_json(self.folder / 'rubrics.json', self.rubrics)
        self.reject_dataset('family_leakage')

    def test_duplicate_question_and_unbound_lines_rejected(self):
        self.bindings[1]['question_id'] = self.bindings[0]['question_id']
        write_json(self.folder / 'question_bindings.json', self.bindings)
        self.reject_dataset('duplicate_question')
        self.bindings.pop()
        write_json(self.folder / 'question_bindings.json', self.bindings)
        self.reject_dataset('unbound_request')

    def test_request_hash_prevents_unregistered_line_change(self):
        request = business('This synthetic line was changed after binding.')
        (self.folder / 'debug_requests.jsonl').write_text(v.canonical(request) + '\n', encoding='utf-8')
        self.reject_dataset('request_hash')

    def test_complete_citation_copy_rejects_condition_or_locator_change(self):
        baseline = copy.deepcopy(self.rubrics)
        for field in ('conditions', 'limitations', 'knowledge_id', 'source_id', 'page_or_section'):
            self.rubrics = copy.deepcopy(baseline)
            self.rubrics[0]['points'][0]['citations'][0][field] = 'SYNTHETIC_CHANGED'
            write_json(self.folder / 'rubrics.json', self.rubrics)
            with self.subTest(field=field):
                self.reject_dataset('citation_copy')

    def test_basis_ref_must_match_its_exact_row_field(self):
        ref = self.rubrics[0]['points'][0]['basis_refs'][0]
        ref['exact_value'] += ' SYNTHETIC_CHANGED'
        write_json(self.folder / 'rubrics.json', self.rubrics)
        self.reject_dataset('basis_value')

    def test_basis_ref_path_whitelist_and_locator(self):
        ref = self.rubrics[0]['points'][0]['basis_refs'][0]
        ref['path'] = 'knowledge/k01/../k01/scenarios.csv'
        write_json(self.folder / 'rubrics.json', self.rubrics)
        self.reject_dataset('basis_path')
        ref['path'] = 'knowledge/k01/scenarios.csv'
        ref['locator'] = 'SC-K01-OR-001:not_a_field'
        write_json(self.folder / 'rubrics.json', self.rubrics)
        self.reject_dataset('basis_locator')

    def test_explicitly_unsupported_requires_basis_and_no_support(self):
        point = self.rubrics[0]['points'][0]
        point['basis_kind'] = 'explicitly_unsupported'
        point['support_any_of'] = []
        point['citations'] = []
        write_json(self.folder / 'rubrics.json', self.rubrics)
        self.assertEqual(v.validate_dataset(self.folder)['status'], 'PASS')
        point['basis_refs'] = []
        write_json(self.folder / 'rubrics.json', self.rubrics)
        self.reject_dataset('unsupported_point')

    def test_support_and_point_task_membership(self):
        point = self.rubrics[0]['points'][0]
        point['support_any_of'] = [['EV-SYNTHETIC-MISSING']]
        write_json(self.folder / 'rubrics.json', self.rubrics)
        self.reject_dataset('evidence_id')
        point['support_any_of'] = [['EV-K01-001']]
        point['task_type'] = 'preparation'
        write_json(self.folder / 'rubrics.json', self.rubrics)
        self.reject_dataset('point_task')

    def test_export_uses_explicit_binding_and_never_serializes_rubric(self):
        # Adjacent evaluation/hostile files do not enter the explicit exporter.
        (self.folder / 'evil_scoring_secret.json').write_text('{"answer":"SYNTHETIC_SECRET"}', encoding='utf-8')
        self.rubrics[0]['forbidden_claims'].append('SYNTHETIC_SECRET')
        write_json(self.folder / 'rubrics.json', self.rubrics)
        exported = v.export_request(self.folder, 'SYN-D01')
        self.assertEqual(set(exported), v.REQUEST_FIELDS)
        self.assertNotIn('SYNTHETIC_SECRET', v.canonical(exported))
        self.assertEqual(exported['question'], 'This synthetic debug fixture asks about a possible mechanism.')
        with self.assertRaises(v.E01Error):
            v.export_request(self.folder, 'SYN-MISSING')

    def test_export_still_rejects_extra_field_inside_bound_request(self):
        request = business()
        request['rubric'] = 'SYNTHETIC_SECRET'
        (self.folder / 'debug_requests.jsonl').write_text(v.canonical(request) + '\n', encoding='utf-8')
        with self.assertRaises(v.E01Error):
            v.export_request(self.folder, 'SYN-D01')

    def test_contract_case_must_really_be_rejected(self):
        contracts = [{'case_id': 'SYN-C01', 'payload_json': json.dumps(business()),
                      'expected': 'reject', 'reason': 'Intentionally false synthetic bad case.'}]
        write_json(self.folder / 'contract_cases.json', contracts)
        self.reject_dataset('contract_case')

    def test_new_outputs_cannot_replace_existing_file(self):
        target = self.folder / 'synthetic_report.json'
        v.write_new_json(target, {'marker': 'first'})
        before = target.read_bytes()
        with self.assertRaises(v.E01Error):
            v.write_new_json(target, {'marker': 'second'})
        self.assertEqual(target.read_bytes(), before)

    def test_insufficiency_criteria_require_three_ids_and_boolean_applicability(self):
        baseline = copy.deepcopy(self.rubrics)
        for mutation in ('missing', 'duplicate', 'nonboolean'):
            self.rubrics = copy.deepcopy(baseline)
            criteria = self.rubrics[0]['insufficiency_criteria']
            if mutation == 'missing':
                criteria.pop()
            elif mutation == 'duplicate':
                criteria[1]['criterion_id'] = 'i1'
            else:
                criteria[0]['applicable'] = 'false'
            write_json(self.folder / 'rubrics.json', self.rubrics)
            with self.subTest(mutation=mutation):
                self.reject_dataset('insufficiency_criteria')


if __name__ == '__main__':
    unittest.main()
