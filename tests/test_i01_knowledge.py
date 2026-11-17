"""Pinned new-corpus integration over unchanged K02; no model/API calls."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import i01_knowledge as knowledge
import retrieve_k02 as core

ACTIVE = Path(os.environ.get('I01_ACTIVE_RUN', ROOT / 'runs/i01/20261106_200240_english_demo_rev053_287f5f2b'))
# Captured after independent readback; the loader never derives its own approval.
PIN = 'a496b98356740c6389d343f93994dc7920440424d38b61ca28dd961ae935ed1a'
QUESTIONS = {
    'explanation': 'Explain bearing damage and possible causes under the demonstration assumption.',
    'inspection': 'What inspection observations, lubricant, bearing surfaces, vibration or temperature should be checked?',
    'preparation': 'What maintenance preparation, lubricant condition or dismounting tools should be prepared?',
}


class CorpusIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ACTIVE / 'tmp').mkdir(parents=True, exist_ok=True)
        cls.retriever = knowledge.Retriever(manifest_sha256=PIN)

    def request(self, predicted, task):
        return {'predicted_class': predicted, 'scenario_id': knowledge.SCENARIOS[predicted],
                'question': QUESTIONS[task], 'task_types': [task], 'explicitly_allowed_operating_context': {}}

    def test_old_five_tables_preserved_as_identical_rows(self):
        for table in core.k01.NAMES:
            with self.subTest(table=table), (ROOT / 'knowledge/k01' / (table + '.csv')).open(newline='', encoding='utf-8') as source:
                old = list(csv.DictReader(source))
                new = self.retriever.tables[table]
                self.assertEqual(new[:len(old)], old)

    def test_all_twelve_class_task_graph_routes_and_full_cards(self):
        with patch.object(socket, 'socket', side_effect=AssertionError('No network in retrieval tests')):
            for predicted in core.CLASSES:
                for task in core.TASKS:
                    with self.subTest(predicted=predicted, task=task):
                        request = self.request(predicted, task)
                        before = copy.deepcopy(request)
                        returned = self.retriever.retrieve(request, mode='graph', top_k=3)
                        self.assertEqual(request, before)
                        self.assertEqual(returned['request'], before)
                        expected = self.retriever.validation['graph_task_candidates'][predicted][task]
                        self.assertEqual(returned['candidate_ids'], expected)
                        self.assertEqual(returned['status'], 'retrieved')
                        self.assertGreater(returned['returned_count'], 0)
                        self.assertLessEqual(returned['returned_count'], 3)
                        self.assertEqual(returned['budget']['truncation'], 'none')
                        for card in returned['results']:
                            original = self.retriever.evidence[card['evidence_id']]
                            self.assertEqual({key: card[key] for key in original}, original)
                            self.assertEqual(card['source'], self.retriever.sources[card['source_id']])
                            self.assertEqual(card['task_type'], task)
                            self.assertTrue(card['relation_paths'])
                        demo = returned['information_separation']['demo_scenario']
                        self.assertEqual(demo, self.retriever.scenarios[before['scenario_id']])

    def test_identity_names_new_corpus_and_unchanged_algorithm(self):
        identity = self.retriever.identity
        self.assertEqual(identity['manifest_sha256'], PIN)
        self.assertEqual(identity['algorithm_version'], core.VERSION)
        self.assertEqual(identity['knowledge_version'], 'i01.knowledge.v1_rev025')
        self.assertEqual(self.retriever.config['index_fields'], ['original_text', 'claim_en'])
        self.assertEqual(self.retriever.config['tfidf'], core.FIXED_CONFIG['tfidf'])
        for mode in ('text', 'graph'):
            returned = self.retriever.retrieve(self.request('normal', 'inspection'), mode=mode)
            self.assertEqual(returned['knowledge_version'], identity['knowledge_version'])
            self.assertEqual(returned['manifest_sha256'], PIN)
            self.assertEqual(returned['algorithm_version'], core.VERSION)
            self.assertNotIn('Standstill external vibration is an independent demo assumption; poor lubrication is only another possibility.', returned['notices'])

    def test_unknown_scenario_has_no_outer_fallback_and_conflict_rejected(self):
        request = self.request('normal', 'inspection')
        request['scenario_id'] = 'SC-UNKNOWN-001'
        missing = self.retriever.retrieve(request, mode='graph')
        self.assertEqual(missing['status'], 'knowledge_not_covered')
        self.assertEqual(missing['results'], [])
        request['scenario_id'] = knowledge.SCENARIOS['outer_race']
        with self.assertRaises(core.K02Error) as rejected:
            self.retriever.retrieve(request, mode='graph')
        self.assertEqual(rejected.exception.code, 'class_scenario_conflict')

    def test_external_manifest_pin_is_required_and_changed_bundle_rejected(self):
        with self.assertRaises(core.K02Error) as unpinned:
            knowledge.validate_bundle(manifest_sha256='0' * 64)
        self.assertEqual(unpinned.exception.code, 'knowledge_hash_drift')
        folder = Path(tempfile.mkdtemp(prefix='knowledge_drift_', dir=ACTIVE / 'tmp'))
        manifest = ROOT / 'knowledge/i01/v1/manifest.json'
        files = dict(self.retriever.validation['file_hashes'])
        for relative in [*files, 'knowledge/i01/v1/manifest.json']:
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
        with (folder / 'knowledge/i01/v1/evidence.csv').open('ab') as stream:
            stream.write(b'\nSYNTHETIC_UNRELEASED_CHANGE\n')
        with self.assertRaises(core.K02Error) as changed:
            knowledge.validate_bundle(folder / 'knowledge/i01/v1', manifest_sha256=PIN, project_root=folder)
        self.assertEqual(changed.exception.code, 'knowledge_hash_drift')

    def test_task_nodes_cannot_be_shared_between_scenarios(self):
        tables = copy.deepcopy(self.retriever.tables)
        interface = copy.deepcopy(self.retriever.interface)
        files = self.retriever.validation['file_hashes']
        blobs = {relative: (ROOT / relative).read_bytes() for relative in files}
        outer_task = next(edge['object_id'] for edge in tables['relations']
                          if edge['subject_id'] == knowledge.SCENARIOS['outer_race'] and edge['relation'] == 'has_task')
        inner_task = next(edge for edge in tables['relations']
                          if edge['subject_id'] == knowledge.SCENARIOS['inner_race'] and edge['relation'] == 'has_task')
        inner_task['object_id'] = outer_task
        with self.assertRaises(core.K02Error):
            knowledge._validate(tables, interface, blobs, files, excerpt_index=json.loads((ROOT / 'knowledge/i01/v1/source_excerpt_index.json').read_bytes()))

    def test_lost_relation_conditions_are_rejected(self):
        tables = copy.deepcopy(self.retriever.tables)
        interface = copy.deepcopy(self.retriever.interface)
        files = self.retriever.validation['file_hashes']
        blobs = {relative: (ROOT / relative).read_bytes() for relative in files}
        edge = next(e for e in tables['relations'] if e['assertion_basis'] == 'source_knowledge')
        edge['conditions'] = 'SYNTHETIC stripped condition'
        with self.assertRaises(core.K02Error):
            knowledge._validate(tables, interface, blobs, files, excerpt_index=json.loads((ROOT / 'knowledge/i01/v1/source_excerpt_index.json').read_bytes()))


if __name__ == '__main__':
    unittest.main()
