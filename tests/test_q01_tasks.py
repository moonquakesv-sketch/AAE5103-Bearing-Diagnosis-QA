"""Synthetic task-grammar contracts; all outputs in an explicit new task run."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import q01_tasks as tasks
import q01_flow as flow
import test_q01_flow as fixtures


def input_for(question='Inspect the demo bearing.'):
    return {'predicted_class': 'outer_race', 'scenario_id': 'SC-K01-OR-001',
            'question': question, 'explicitly_allowed_operating_context': {}}


class Untouched:
    def __getattribute__(self, name):
        raise AssertionError('Blocked route touched an engine object: ' + name)


class TaskContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ['Q01_TASK_RUN_DIR']).absolute()
        if '..' in path.parts or not path.is_relative_to(ROOT / 'runs/q01_task'):
            raise ValueError('New task output root required.')
        for parent in (path,) + tuple(path.parents):
            if parent.is_symlink():
                raise ValueError('No linked output paths.')
        cls.contract = json.loads((path / 'verification/contract.json').read_text())
        cls.output = Path(tempfile.mkdtemp(prefix='contract_', dir=path / 'outputs'))
        cls.observations = []

    @classmethod
    def tearDownClass(cls):
        flow.write_record(cls.output / 'observations.json', {
            'source_sha256': flow.file_hash(tasks.__file__),
            'flow_source_sha256': flow.file_hash(flow.__file__),
            'case_count': len(cls.observations), 'cases': cls.observations,
            'actual_api_requests': 0, 'actual_K02_retrievals': 0,
            'fixtures': 'All grammar/example/flow statements are synthetic, not E01.'})

    def save(self, name, value):
        self.observations.append({'case': name, 'record': value})
        flow.write_record(self.output / ('case_' + name + '.json'), value)

    def test_frozen_grammar_cases(self):
        for case in self.contract['expected_cases']:
            with self.subTest(case=case['case']):
                result = tasks.prepare_request(input_for(case['question']))
                self.save(case['case'], result)
                self.assertEqual(result['status'], case['status'])
                self.assertEqual(result['task_types'], case['task_types'])
                if case['status'] == 'ready':
                    self.assertEqual(set(result['request']), flow.k02.INPUT_FIELDS)
                    self.assertEqual(result['request']['question'], case['question'])
                    flow.k02.validate_request(result['request'])
                else:
                    self.assertIsNone(result['request'])

    def test_blocked_never_touches_any_engine(self):
        for i, question in enumerate(['Help with the bearing.', 'Inspect the bearing and calculate a price.', 'Do not inspect the bearing.']):
            value = tasks.route_and_run(input_for(question), Untouched(), Untouched(), retriever=Untouched(), runtime=Untouched())
            self.save('blocked_' + str(i), value)
            self.assertIsNone(value['flow_record'])

    def test_strict_input_and_existing_explicit_tags(self):
        values = []
        for key in ['task_types', 'reference_answer', 'question_id', 'notes']:
            value = input_for(); value[key] = ['inspection'] if key == 'task_types' else 'SYNTHETIC'
            values.append((key, value))
        for key, value in [('predicted_class', 'unknown'), ('scenario_id', ''), ('explicitly_allowed_operating_context', {'notes': 'SYNTHETIC'})]:
            item = input_for(); item[key] = value; values.append((key, item))
        for name, payload in values:
            result = tasks.prepare_request(payload); self.save('invalid_' + name, result)
            self.assertEqual(result['status'], 'invalid_input'); self.assertIsNone(result['request'])

    def test_safe_json_failures(self):
        circular = {}; circular['cycle'] = circular
        class Hostile:
            def __repr__(self):
                raise AssertionError('repr invoked')
            def __deepcopy__(self, memo):
                raise AssertionError('deepcopy invoked')
        for i, payload in enumerate([circular, {'x': float('nan')}, {'x': Hostile()}, {1: 'SYNTHETIC'}]):
            result = tasks.prepare_request(payload); self.save('unsafe_' + str(i), result)
            self.assertEqual(result['status'], 'invalid_input'); json.dumps(result, allow_nan=False)

    def test_copy_and_context_preservation(self):
        original = input_for(); original['explicitly_allowed_operating_context'] = {'operating_state': 'unknown'}
        before = copy.deepcopy(original); result = tasks.prepare_request(original)
        self.assertEqual(result['request']['explicitly_allowed_operating_context'], before['explicitly_allowed_operating_context'])
        result['request']['explicitly_allowed_operating_context']['operating_state'] = 'standstill'
        result['task_types'].append('explanation')
        self.assertEqual(original, before)
        self.assertEqual(tasks.prepare_request(original)['task_types'], ['inspection'])

    def test_T_GV_and_partial_flow_wiring(self):
        for group in ['T', 'GV']:
            payload = input_for('Explain the meaning of the demo bearing label and inspect the demo bearing.')
            req = tasks.prepare_request(payload)['request']
            ret = fixtures.retrieval('partial_task_coverage'); ret['request'] = copy.deepcopy(req)
            provider = flow.ScriptedProvider({'generate': [fixtures.answer()], 'verify': [{'verdict': 'pass', 'issues': []}]})
            value = tasks.route_and_run(payload, flow.group_config(group), provider, retriever=fixtures.FakeRetriever(ret))
            self.save('wiring_' + group, value)
            record = value['flow_record']
            self.assertEqual(record['status'], 'evidence_insufficient'); self.assertTrue(record['publishable'])
            self.assertEqual(record['request']['task_types'], ['explanation', 'inspection'])
            self.assertEqual(record['counts']['retrieval'], 1); self.assertEqual(record['counts']['generate'], 1)
            self.assertEqual(record['counts']['verify'], 1 if group == 'GV' else 0)
            self.assertEqual(record['counts']['revise'], 0)
            self.assertEqual(record['semantic_support'], 'not_established')
            for event in record['provider_trace']:
                self.assertNotIn('routing', event['payload'])
                self.assertEqual(event['payload']['request'], req)
                self.assertNotIn('rank', event['payload']['evidence_cards'][0])
                self.assertEqual(event['payload']['evidence_cards'][0]['source'], ret['results'][0]['source'])

    def test_verification_failure_never_publishes_draft(self):
        class Provider:
            identity = {'kind': 'injected_test_double'}
            def generate(self, payload): return fixtures.answer()
            def verify(self, payload): raise TimeoutError('SYNTHETIC verification failure')
            def revise(self, payload): raise AssertionError('Unexpected revision')
        payload = input_for(); ret = fixtures.retrieval(); ret['request'] = tasks.prepare_request(payload)['request']
        result = tasks.route_and_run(payload, flow.group_config('GV'), Provider(), retriever=fixtures.FakeRetriever(ret))
        self.save('verification_failure', result)
        self.assertFalse(result['flow_record']['publishable']); self.assertIsNone(result['flow_record']['final_answer'])
        self.assertEqual(result['flow_record']['counts']['verify'], 1)
        self.assertEqual(result['flow_record']['counts']['revise'], 0)

    def test_cli_strict_json_no_overwrite_and_path_guard(self):
        source = self.output / 'cli_input.json'; source.write_text(json.dumps(input_for()))
        target = self.output / 'cli_output.json'
        argv = [sys.executable, str(ROOT / 'scripts/q01_tasks.py'), '--input', str(source), '--output', str(target)]
        def call(name, args):
            proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
            flow.write_record(self.output / (name + '_command.json'), {'argv': args, 'cwd': str(ROOT), 'controlled_environment': {k: os.environ.get(k) for k in ['PYTHONDONTWRITEBYTECODE', 'TMPDIR', 'TMP', 'TEMP']}, 'exit_code': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr})
            return proc
        self.assertEqual(call('cli_ok', argv).returncode, 0); before = target.read_bytes()
        self.assertEqual(call('cli_existing', argv).returncode, 2); self.assertEqual(before, target.read_bytes())
        bad = self.output / 'duplicate.json'; bad.write_text('{"question":"SYNTHETIC","question":"SYNTHETIC duplicate"}')
        bad_target = self.output / 'duplicate_result.json'
        self.assertEqual(call('cli_duplicate', argv[:-4] + ['--input', str(bad), '--output', str(bad_target)]).returncode, 1)
        self.assertEqual(json.loads(bad_target.read_bytes())['status'], 'invalid_input')
        forbidden_target = self.output / 'guard_should_not_create.json'
        self.assertEqual(call('cli_path_guard', argv[:-4] + ['--input', str(ROOT / 'evaluation/e01/DO_NOT_OPEN.json'), '--output', str(forbidden_target)]).returncode, 2)
        self.assertFalse(forbidden_target.exists())
        # Link only within the new synthetic run; remove it so the old three-link set stays exact.
        link = self.output / 'linked.json'; link.symlink_to(source)
        linked_target = self.output / 'linked_result.json'
        try:
            self.assertEqual(call('cli_link_guard', argv[:-4] + ['--input', str(link), '--output', str(linked_target)]).returncode, 2)
            self.assertFalse(linked_target.exists())
        finally:
            link.unlink()


if __name__ == '__main__':
    unittest.main()
