"""K02 development checks, not E01 questions or performance evaluation.

K02_TEST_RUN must point to a NEW audit run directory; no historical temp writes.
"""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import retrieve_k02 as k


def request(tasks=None, question='What may explain false brinelling during standstill vibration?'):
    return {'predicted_class': 'outer_race', 'scenario_id': 'SC-K01-OR-001',
            'question': question, 'task_types': tasks or ['explanation'],
            'explicitly_allowed_operating_context': {}}


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir = Path(os.environ['K02_TEST_RUN']).resolve()
        assert cls.run_dir.is_dir() and cls.run_dir.is_relative_to(k.ROOT / 'runs/k02')
        cls.engine = k.Retriever()

    def test_all_tasks_and_multi_task(self):
        for tasks in [['explanation'], ['inspection'], ['preparation'], list(k.TASKS)]:
            q = request(tasks, 'What vibration, texture, cleaning, drawings and lubricant information should be considered?')
            for mode in ['text', 'graph']:
                with self.subTest(tasks=tasks, mode=mode):
                    r = self.engine.retrieve(q, mode, 11)
                    self.assertEqual(set(r['per_task']), set(tasks))
                    self.assertTrue(all(t['status'] == 'retrieved' for t in r['per_task'].values()))
                    self.assertTrue(all(e['score'] > 0 for e in r['results']))

    def test_shared_corpus_query_scoring_and_independent_text_candidates(self):
        before = self.engine.index_snapshot()
        q = request(['inspection'])
        a, b = (self.engine.retrieve(q, mode) for mode in ['text', 'graph'])
        for key in ['request', 'query_text', 'shared_config_id', 'index_id', 'budget', 'information_separation']:
            self.assertEqual(a[key], b[key])
        self.assertEqual(a['candidate_ids'], self.engine.ids)
        self.assertEqual(b['candidate_count'], 3)
        scores = {v['evidence_id']: v['score'] for v in a['scored_candidates']}
        for row in b['scored_candidates']:
            self.assertEqual(scores[row['evidence_id']], row['score'])
        self.assertEqual(before, self.engine.index_snapshot())

    def test_paths_are_real_edges_and_reverse_class_route(self):
        rows = {r['relation_id']: r for r in self.engine.tables['relations']}
        for task, count in zip(k.TASKS, [3, 3, 5]):
            result = self.engine.retrieve(request([task]), 'graph', 11)
            self.assertEqual(result['candidate_count'], count)
            for eid, paths in result['graph_candidate_paths'].items():
                for path in paths:
                    self.assertEqual([s['direction'] for s in path], ['reverse', 'forward', 'forward', 'forward'])
                    self.assertEqual(path[0]['from_node'], 'PART-OUTER-RACE')
                    self.assertEqual(path[-1]['to_node'], eid)
                    for i, step in enumerate(path):
                        self.assertEqual(step['edge'], rows[step['edge']['relation_id']])
                        if i:
                            self.assertEqual(path[i - 1]['to_node'], step['from_node'])
                    self.assertEqual(path[2]['edge']['evidence_id'], eid)
                    self.assertEqual(path[3]['edge']['evidence_id'], eid)

    def test_synthetic_connection_changes_candidates_without_evidence_filter(self):
        tables = copy.deepcopy(self.engine.tables)
        args = (tables['entities'], 'outer_race', 'SC-K01-OR-001', ['explanation'])
        baseline = k.graph_paths(tables['relations'], *args)
        edge = next(e for e in tables['relations'] if e['relation'] == 'has_evidence_backed_content'
                    and e['subject_id'] == 'TASK-EXPLANATION')
        removed = edge['evidence_id']
        edge['subject_id'] = 'TASK-INSPECTION'
        changed = k.graph_paths(tables['relations'], *args)
        self.assertEqual(set(changed), set(baseline) - {removed})
        inspection = k.graph_paths(tables['relations'], tables['entities'], 'outer_race', 'SC-K01-OR-001', ['inspection'])
        self.assertIn(removed, inspection)
        # evidence.task_type remains explanation: pure traversal genuinely followed the changed edge.
        self.assertEqual(self.engine.evidence[removed]['task_type'], 'explanation')

    def test_synthetic_break_each_link_and_no_candidate_runtime(self):
        base = copy.deepcopy(self.engine.tables)
        for predicate in ['routes_from_class', 'has_task', 'has_evidence_backed_content', 'supported_by']:
            edges = [e for e in base['relations'] if e['relation'] != predicate]
            self.assertEqual(k.graph_paths(edges, base['entities'], 'outer_race', 'SC-K01-OR-001', list(k.TASKS)), {})
        # Synthetic in-memory ablation only; production loader never accepts altered K01.
        with patch.object(self.engine, 'tables', dict(base, relations=[])):
            result = self.engine.retrieve(request(), 'graph')
            self.assertEqual(result['status'], 'no_candidates')
            self.assertEqual(result['results'], [])
            self.assertEqual(self.engine.retrieve(request(), 'text')['candidate_count'], 11)

    def test_loader_rejects_actual_file_drift_version_mirror_and_dangling(self):
        mutations = [('interface.json', 'version'), ('entities.csv', 'mirror'),
                     ('relations.csv', 'dangling'), ('evidence.csv', 'bytes')]
        for name, kind in mutations:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=self.run_dir) as temp:
                folder = Path(temp)
                for allowed in k.KNOWLEDGE_HASHES:
                    shutil.copyfile(k.ROOT / 'knowledge/k01' / allowed, folder / allowed)
                p = folder / name
                if kind == 'version':
                    data = json.loads(p.read_text()); data['knowledge_version'] = '20261023_rev010'
                    p.write_text(json.dumps(data))
                elif kind == 'bytes':
                    p.write_bytes(p.read_bytes() + b'\n')
                else:
                    with p.open(newline='') as stream:
                        rows = list(csv.DictReader(stream))
                    if kind == 'mirror':
                        next(e for e in rows if e['entity_id'] == 'EV-K01-001')['content_ref'] = 'evidence.csv#EV-K01-003'
                    else:
                        rows[0]['object_id'] = 'PART-MISSING'
                    with p.open('w', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                with self.assertRaisesRegex(k.K02Error, 'expected .* actual'):
                    k.load_knowledge(folder)
                if kind != 'bytes':
                    # Separate synthetic unit layer: simulate a digest match, still require structure rejection.
                    hashes = dict(k.KNOWLEDGE_HASHES)
                    hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
                    with patch.object(k, 'KNOWLEDGE_HASHES', hashes), self.assertRaises(k.K02Error) as caught:
                        k.load_knowledge(folder)
                    self.assertEqual(caught.exception.code, 'knowledge_structure')

    def test_loader_reads_only_six_allowlisted_files(self):
        seen = []
        original = Path.read_bytes
        def recording(path):
            seen.append(path.name)
            return original(path)
        with patch.object(Path, 'read_bytes', recording):
            k.load_knowledge(k.ROOT / 'knowledge/k01')
        self.assertEqual(set(seen), set(k.KNOWLEDGE_HASHES))
        self.assertEqual(len(seen), 6)

    def test_index_fields_and_documents_are_exact(self):
        snapshot = self.engine.index_snapshot()
        self.assertEqual(snapshot['indexed_fields'], ['original_text', 'claim_en'])
        self.assertEqual([r['evidence_id'] for r in snapshot['documents']], self.engine.ids)
        for row in snapshot['documents']:
            e = self.engine.evidence[row['evidence_id']]
            self.assertEqual(row['text'], e['original_text'] + '\n' + e['claim_en'])
        self.assertFalse(snapshot['source_files_indexed'])
        self.assertFalse(snapshot['audit_material_indexed'])

    def test_forbidden_unknown_fields_and_missing_fields(self):
        for field in sorted(k.FORBIDDEN | {'retrieval_mode', 'top_k', 'unregistered'}):
            with self.subTest(field=field):
                q = request(); q[field] = 'forbidden'
                with self.assertRaises(k.K02Error): self.engine.retrieve(q)
        with self.assertRaises(k.K02Error): self.engine.retrieve({})
        with self.assertRaises(k.K02Error): self.engine.retrieve([])
        q = request(); del q['question']
        with self.assertRaises(k.K02Error): self.engine.retrieve(q)

    def test_wrong_types_tasks_values_and_config(self):
        mutations = {'predicted_class': [None, 1, 'unsupported'], 'scenario_id': [False, ''],
                     'question': [[], None, 99], 'task_types': ['inspection', [], ['bad'], [1], ['inspection', 'inspection']],
                     'explicitly_allowed_operating_context': [[], None, 'standstill']}
        for field, values in mutations.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    q = request(); q[field] = value
                    with self.assertRaises(k.K02Error): self.engine.retrieve(q)
        for value in [True, 0, 12, 2.5, '3']:
            with self.assertRaises(k.K02Error): self.engine.retrieve(request(), top_k=value)
        with self.assertRaises(k.K02Error): self.engine.retrieve(request(), mode='unknown')

    def test_nested_context_is_closed_and_demo_only(self):
        for context in [{'notes': 'answer'}, {'operating_state': {'ground_truth': 'outer_race'}},
                        {'external_vibration': True}, {'operating_state': 'rotating'},
                        {'external_vibration': 'observed'}, {'sample_path': '/sample'}]:
            q = request(); q['explicitly_allowed_operating_context'] = context
            with self.assertRaises(k.K02Error): self.engine.retrieve(q)
        q = request(); q['explicitly_allowed_operating_context'] = {'operating_state': 'standstill', 'external_vibration': 'assumed_for_demo'}
        self.assertEqual(self.engine.retrieve(q)['request'], q)

    def test_class_conflict_and_uncovered_categories_and_scenario(self):
        for cls in ['normal', 'inner_race', 'rolling_element']:
            q = request(); q['predicted_class'] = cls
            with self.assertRaises(k.K02Error) as caught: self.engine.retrieve(q)
            self.assertEqual(caught.exception.code, 'class_scenario_conflict')
            q['scenario_id'] = 'SC-UNREGISTERED'
            r = self.engine.retrieve(q)
            self.assertEqual(r['status'], 'knowledge_not_covered')
            self.assertEqual(r['results'], [])
        q = request(); q['scenario_id'] = 'SC-NOT-THERE'
        self.assertEqual(self.engine.retrieve(q)['status'], 'knowledge_not_covered')

    def test_empty_no_terms_oov_zero_similarity_never_padded(self):
        for question, status in [(' ', 'empty_question'), ('the and !!!', 'no_valid_terms'),
                                 ('qzxwv unlistedtokenzz', 'no_vocabulary_terms')]:
            for mode in ['text', 'graph']:
                r = self.engine.retrieve(request(question=question), mode)
                self.assertEqual(r['status'], status)
                self.assertEqual(r['results'], [])
        r = self.engine.retrieve(request(['preparation'], 'What does lubrication mean?'), 'graph', 11)
        self.assertEqual(r['status'], 'zero_similarity')
        self.assertEqual(r['returned_count'], 0)
        self.assertEqual(r['budget']['unused_slots'], 11)

    def test_complete_cards_sources_conditions_and_limits_preserved(self):
        q = request(list(k.TASKS), 'What vibration, texture, cleaning, drawings, lubrication, lubricant, OEM and equipment information is relevant?')
        for mode in ['text', 'graph']:
            r = self.engine.retrieve(q, mode, 11)
            self.assertEqual(r['returned_count'], 11)
            for row in r['results']:
                expected = self.engine.evidence[row['evidence_id']]
                self.assertEqual({f: row[f] for f in expected}, expected)
                self.assertEqual(row['source'], self.engine.sources[row['source_id']])
            limited = self.engine.retrieve(q, mode, 1)
            self.assertEqual(limited['results'][0], r['results'][0])
            self.assertEqual(limited['information_separation'], r['information_separation'])
            self.assertEqual(limited['notices'], r['notices'])
            self.assertIn('budget_exhausted', [t['status'] for t in limited['per_task'].values()])

    def test_deterministic_order_ties_task_order_and_no_query_fit(self):
        before = self.engine.index_snapshot()
        q = request(list(reversed(k.TASKS)))
        a = self.engine.retrieve(q, 'graph', 4)
        self.assertEqual(a, self.engine.retrieve(q, 'graph', 4))
        q['task_types'] = list(k.TASKS)
        b = self.engine.retrieve(q, 'graph', 4)
        self.assertEqual(a['results'], b['results'])
        self.assertEqual(a['query_text'], b['query_text'])
        self.assertEqual(before, self.engine.index_snapshot())
        self.assertEqual(k.rank_candidates(['b', 'a', 'z'], {'a': 0.5, 'b': 0.5, 'z': 0}), ['a', 'b'])

    def test_json_duplicate_keys_nonfinite_and_config_drift(self):
        for data in ['{"question": "one", "question": "two"}', '{"x": NaN}']:
            with self.assertRaises(k.K02Error): k.strict_json(data)
        with tempfile.TemporaryDirectory(dir=self.run_dir) as temp:
            p = Path(temp) / 'bad_config.json'
            c = copy.deepcopy(k.FIXED_CONFIG); c['tfidf']['sublinear_tf'] = True
            p.write_text(json.dumps(c))
            with self.assertRaises(k.K02Error): k.Retriever(config_path=p)

    def test_real_cli_success_rejection_and_exclusive_output(self):
        with tempfile.TemporaryDirectory(dir=self.run_dir) as temp:
            folder = Path(temp); inp = folder / 'request.json'; out = folder / 'result.json'
            inp.write_text(json.dumps(request()))
            command = [sys.executable, str(k.ROOT / 'scripts/retrieve_k02.py'), '--request', str(inp),
                       '--mode', 'graph', '--output', str(out)]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            first = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(first.returncode, 0, first.stderr)
            prior = out.read_bytes()
            again = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(again.returncode, 2)
            self.assertEqual(prior, out.read_bytes())
            q = request(); q['ground_truth'] = 'outer_race'; inp.write_text(json.dumps(q))
            command[-1] = str(folder / 'rejected.json')
            bad = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(bad.returncode, 2)
            rejected = json.loads(Path(command[-1]).read_text())
            self.assertEqual(rejected['error_code'], 'forbidden_field')
            self.assertNotIn('request', rejected)

    def test_unicode_api_rejection_and_valid_scalar_text(self):
        for field in ['predicted_class', 'scenario_id', 'question']:
            for surrogate in ['\ud800', '\udfff']:
                with self.subTest(field=field, surrogate=repr(surrogate)):
                    q = request(); q[field] += surrogate
                    with self.assertRaises(k.K02Error) as caught:
                        self.engine.retrieve(q)
                    self.assertEqual(caught.exception.code, 'invalid_unicode')
                    self.assertNotIn(q[field], str(caught.exception))
        q = request(question='Vibration \u00e9 \u4e2d\u6587 \U0001f642')
        for mode in ['text', 'graph']:
            result = self.engine.retrieve(q, mode)
            self.assertEqual(result['request'], q)
            self.assertTrue(result['results'])

    def test_real_cli_unicode_rejection_and_valid_pair(self):
        with tempfile.TemporaryDirectory(dir=self.run_dir) as temp:
            folder = Path(temp)
            for mode in ['text', 'graph']:
                for label, question, code in [('high', 'vibration\ud800', 2),
                                              ('low', 'vibration\udfff', 2),
                                              ('pair', 'vibration \U0001f642', 0)]:
                    with self.subTest(mode=mode, case=label):
                        inp = folder / f'{mode}_{label}_request.json'
                        out = folder / f'{mode}_{label}_result.json'
                        q = request(question=question)
                        inp.write_text(json.dumps(q, ensure_ascii=True))
                        command = [sys.executable, str(k.ROOT / 'scripts/retrieve_k02.py'),
                                   '--request', str(inp), '--mode', mode, '--output', str(out)]
                        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
                        actual = subprocess.run(command, capture_output=True, text=True, env=env)
                        self.assertEqual(actual.returncode, code, actual.stderr)
                        self.assertEqual(actual.stderr, '')
                        result = json.loads(out.read_text(encoding='utf-8'))
                        if code:
                            self.assertEqual(result['status'], 'rejected')
                            self.assertEqual(result['error_code'], 'invalid_unicode')
                            self.assertFalse(result['request_echoed'])
                            self.assertNotIn('request', result)
                        else:
                            self.assertEqual(result['request'], q)
                        before = out.read_bytes()
                        again = subprocess.run(command, capture_output=True, text=True, env=env)
                        self.assertEqual(again.returncode, 2)
                        self.assertEqual(out.read_bytes(), before)

    def test_cli_serialization_failure_writes_complete_rejection(self):
        # Inject a malformed internal result to exercise output handling independently
        # of the earlier input guard; no production knowledge is modified.
        with tempfile.TemporaryDirectory(dir=self.run_dir) as temp:
            folder = Path(temp); inp = folder / 'request.json'
            inp.write_text(json.dumps(request()))
            for label, value in [('unicode', '\ud800'), ('nonfinite', float('nan'))]:
                with self.subTest(case=label):
                    out = folder / (label + '.json')
                    args = ['retrieve_k02', '--request', str(inp), '--mode', 'graph', '--output', str(out)]
                    with patch.object(k.Retriever, 'retrieve', return_value={'status': 'retrieved', 'bad': value}), \
                         patch.object(sys, 'argv', args), patch.object(sys, 'stdout'):
                        self.assertEqual(k.main(), 2)
                    result = json.loads(out.read_text(encoding='utf-8'))
                    self.assertEqual(result['status'], 'rejected')
                    self.assertEqual(result['error_code'], 'invalid_file_or_json')
                    self.assertFalse(result['request_echoed'])
                    self.assertNotIn('bad', result)

    def test_real_cli_argument_errors_have_no_output(self):
        with tempfile.TemporaryDirectory(dir=self.run_dir) as temp:
            folder = Path(temp); inp = folder / 'request.json'
            inp.write_text(json.dumps(request()))
            for label, flags, structured in [('mode', ['--mode', 'alien'], False),
                                             ('integer', ['--mode', 'graph', '--top-k', '2.5'], False),
                                             ('range', ['--mode', 'graph', '--top-k', '0'], True)]:
                with self.subTest(case=label):
                    out = folder / (label + '.json')
                    command = [sys.executable, str(k.ROOT / 'scripts/retrieve_k02.py'),
                               '--request', str(inp), '--output', str(out)] + flags
                    actual = subprocess.run(command, capture_output=True, text=True,
                                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                    self.assertEqual(actual.returncode, 2)
                    self.assertEqual(out.exists(), structured)
                    if structured:
                        self.assertEqual(json.loads(out.read_text())['error_code'], 'invalid_budget')
                    else:
                        self.assertIn('usage:', actual.stderr)
                        self.assertEqual(actual.stdout, '')

    def test_real_cli_duplicate_surrogate_key_is_safe_rejection(self):
        with tempfile.TemporaryDirectory(dir=self.run_dir) as temp:
            folder = Path(temp); inp = folder / 'request.json'; out = folder / 'result.json'
            inp.write_text('{"\\ud800": 1, "\\ud800": 2}')
            actual = subprocess.run([sys.executable, str(k.ROOT / 'scripts/retrieve_k02.py'),
                                     '--request', str(inp), '--mode', 'graph', '--output', str(out)],
                                    capture_output=True, text=True,
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(actual.returncode, 2, actual.stderr)
            result = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(result['error_code'], 'invalid_json')
            self.assertFalse(result['request_echoed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
