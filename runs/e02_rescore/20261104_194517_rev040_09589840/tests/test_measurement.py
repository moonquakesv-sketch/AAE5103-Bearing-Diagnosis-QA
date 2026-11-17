"""Explicit SYNTHETIC oracles. Never applies fixture ratings to real outputs."""
from pathlib import Path
from types import SimpleNamespace
import base64
import copy
import hashlib
import json
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT / 'runtime'), str(ROOT / 'tests')]
from common import Rejected, snapshot, read_json, digest, canonical, strict_json
from engine import final_packet, pair_packet
from evaluation import validate_output, validate_pair
from measurement import decode, encode_fixture, exact_spans, wire_schema
from rescore_evaluation import Evaluation, messages
from provider import Ledger, Provider, CodingFailure
from reporting import summarize, canonical_rating
from fixtures import REQUEST, CARDS, RUBRIC, candidate, output_rating, pair_rating, FixtureCounter

IDENTITY = {'configured_model': 'SYNTHETIC', 'returned_model': 'SYNTHETIC', 'backend_identity': None,
            'context_id': 'SYNTHETIC', 'prompt_sha256': 'SYNTHETIC'}
SCHEMAS = {kind: read_json(ROOT / 'schemas' / (kind + '.json')) for kind in ['output', 'pair']}


def record(group='GV', correct=True):
    from engine import displayed_text
    return {'cell_key': 'SYN-Q1_' + group, 'group': group, 'business_request': snapshot(REQUEST),
            'status': 'completed', 'publishable': True, 'actual_complete_cards': snapshot(CARDS),
            'final_answer': displayed_text(candidate(correct)), 'draft': candidate(False),
            'verification_observable': True, 'verification_issues': [], 'service_message': None,
            'service_error_category': None, 'tested_latency_seconds': 2.0, 'timing_censored': False,
            'retrieval': {'status': 'retrieved'}}


class SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deny = patch.object(socket.socket, 'connect', side_effect=AssertionError('SYNTHETIC suite forbids network'))
        cls.deny.start()
    @classmethod
    def tearDownClass(cls):
        cls.deny.stop()
    def base(self, kind='output', rec=None):
        rec = rec or record()
        packet = final_packet('SYN-O', rec, RUBRIC, {c['evidence_id']: c for c in CARDS}) if kind == 'output' else pair_packet('SYN-O', rec, RUBRIC, {c['evidence_id']: c for c in CARDS})
        gold = output_rating(packet) if kind == 'output' else pair_rating(packet)
        return rec, packet, encode_fixture(gold, kind)
    def decode(self, value, packet, rec, kind='output'):
        return decode(value, kind, SCHEMAS[kind], packet, rec, RUBRIC, IDENTITY)
    def test_deterministic_facts_have_independent_expected_values(self):
        rec, packet, wire = self.base()
        value = self.decode(wire, packet, rec)
        self.assertEqual([r['score'] for r in value['point_rows']], [1, 1])
        self.assertEqual(value['insufficiency']['available_point_ids'], ['SYN-P1'])
        # Independent direct canonical-string hash of the frozen toy unknown; not production expected_gaps.
        fixed = 'fixed:' + hashlib.sha256(json.dumps('actual torque not supplied', separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()[:16]
        self.assertEqual(value['insufficiency']['actual_gap_ids'], [fixed])
        self.assertEqual(value['insufficiency']['gap_support_decisions'][0]['status'], 'available')
        self.assertEqual(value['boundary_rows'], [value['point_rows'][1]])
        self.assertEqual(value['unit_id'], 'SYN-O')
    def test_no_semantic_score_is_invented_for_unknown(self):
        rec, packet, wire = self.base()
        wire['point_rows'][0]['score'] = None
        wire['release_adherence'] = 0
        value = self.decode(wire, packet, rec)
        self.assertIsNone(value['point_rows'][0]['score'])
        self.assertEqual(value['point_rows'][0]['status'], 'machine_unresolved')
    def test_unicode_repeat_requires_explicit_occurrence(self):
        texts = {'final_answer': 'SYNTHETIC α α'}
        with self.assertRaises(Rejected):
            exact_spans({'channel': 'final_answer', 'quote': 'α'}, texts)
        self.assertEqual(exact_spans({'channel': 'final_answer', 'quote': 'α', 'occurrence': 1}, texts),
                         {'channel': 'final_answer', 'start': 12, 'end': 13, 'quote': 'α'})
        for quote in ['', 'synthetic', 'α  α']:
            with self.assertRaises(Rejected):
                exact_spans({'channel': 'final_answer', 'quote': quote}, texts)
    def test_unseen_or_hidden_quote_rejected(self):
        rec, packet, wire = self.base()
        wire['claims'][0]['spans'][0] = {'channel': 'draft', 'quote': 'SYNTHETIC incorrect'}
        with self.assertRaises(Rejected):
            self.decode(wire, packet, rec)
    def test_conditions_linked_but_wrong_semantics_not_fixed(self):
        rec, packet, wire = self.base()
        wire['conditions'] = [{'id': 'SYN-C1', 'claim_id': 'c1', 'critical': False, 'score': 1,
                               'spans': [], 'reason': 'SYNTHETIC known condition'}]
        value = self.decode(wire, packet, rec)
        self.assertEqual(value['claims'][0]['necessary_condition_ids'], ['SYN-C1'])
        wire['conditions'][0].update(critical=True, score=0)
        with self.assertRaises(Rejected):
            self.decode(wire, packet, rec)  # no automatic support downgrade or other semantic repair.
    def test_empty_non_omission_error_rejected_but_omission_allowed(self):
        rec, packet, wire = self.base()
        wire['release_adherence'] = 0
        wire['errors'] = [{'error_id': 'SYN-ERR', 'kind': 'unsupported', 'proposition': 'SYNTHETIC', 'spans': [], 'reason': 'SYNTHETIC'}]
        with self.assertRaises(Rejected):
            self.decode(wire, packet, rec)
        wire['errors'][0]['kind'] = 'omission'
        self.decode(wire, packet, rec)
    def test_pair_service_is_not_final_body(self):
        rec = record();rec['final_answer'] = None;rec['publishable'] = False;rec['service_message'] = 'SYNTHETIC service says torque99.'
        rec, packet, wire = self.base('pair', rec)
        wire['final_errors'] = [{'error_id': 'SYN-SERVICE', 'kind': 'unsupported', 'proposition': 'SYNTHETIC service error',
                                 'spans': [{'channel': 'service_message', 'quote': 'torque99'}], 'reason': 'SYNTHETIC'}]
        value = self.decode(wire, packet, rec, 'pair')
        self.assertFalse(value['final_observable'])
        self.assertEqual(value['final_errors'][0]['spans'][0]['quote'], 'torque99')
        wire['final_errors'][0]['spans'][0]['channel'] = 'draft'
        with self.assertRaises(Rejected):
            self.decode(wire, packet, rec, 'pair')
    def test_tested_failure_cannot_receive_coverage_credit(self):
        rec, packet, wire = self.base()
        rec['status'] = 'tested_failure'
        packet['service_observation']['tested_status'] = 'tested_failure'
        with self.assertRaises(Rejected):
            self.decode(wire, packet, rec)
    def test_original_canonical_schema_stays_strict(self):
        rec, packet, wire = self.base()
        wire['insufficiency']['available_point_ids'] = ['SYN-P1']
        with self.assertRaises(Exception):
            self.decode(wire, packet, rec)
    def test_numeric_overflow_and_invalid_unicode_are_rejected_not_repaired(self):
        for raw in ['{"score":1e999}', '{"score":NaN}', '{"text":"\\ud800"}']:
            with self.assertRaises((Rejected, UnicodeError)):
                strict_json(raw)
        self.assertEqual(strict_json('{"score":0.5,"text":"SYNTHETIC α"}')['score'], .5)
    def test_third_full_schema_and_invalid_third_keeps_null(self):
        rec, packet, wire = self.base()
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_', dir=ROOT / 'tests') as folder:
            folder = Path(folder)
            class FakeProvider:
                ledger = SimpleNamespace(directory=folder, broken=False)
                def send(self, slot, action, body):
                    payload = json.loads(body[1]['content'])
                    if slot.endswith('_C'):
                        self.system = body[0]['content']
                        self.packet = payload
                        bad = snapshot(wire)
                        bad['claims'][0]['spans'][0]['quote'] = 'SYNTHETIC absent quote'
                        return bad, {'returned_model': 'SYNTHETIC'}
                    return {'SYNTHETIC_INVALID': True}, {'returned_model': 'SYNTHETIC'}
            provider = FakeProvider()
            evaluator = Evaluation(ROOT, {'model': 'SYNTHETIC'}, provider, folder / 'evaluation')
            result = evaluator.evaluate('SYN-O', 'output', packet, rec, RUBRIC)
            self.assertIsNone(result['canonical']['rating'])
            self.assertEqual(result['canonical']['status'], 'evaluator_failure')
            self.assertTrue(all(o['rating'] is None for o in result['originals']))
            self.assertIn('SEMANTIC WIRE SCHEMA', provider.system)
            self.assertIn('point_rows', provider.system)
            self.assertEqual([o['semantic_coding'] for o in provider.packet['current_new_originals']], [None, None])
            self.assertTrue((folder / 'evaluation/SYN-O_output_C.original.json').exists())
    def test_all38_metrics_wired_with_independent_toy_scores(self):
        records, outputs, mappings = [], {}, []
        for group in ['T', 'TV', 'G', 'GV']:
            correct = group in ['TV', 'GV']
            rec = record(group, correct)
            unit = 'SYN-O-' + group
            packet = final_packet(unit, rec, RUBRIC, {c['evidence_id']: c for c in CARDS})
            wire = encode_fixture(output_rating(packet), 'output')
            rating = decode(wire, 'output', SCHEMAS['output'], packet, rec, RUBRIC, IDENTITY)
            originals = [{'status': 'valid', 'rating': rating}] * 2
            outputs[unit] = {'originals': originals, 'canonical': {'status': 'resolved_primary_agreement', 'rating': rating, 'review': {'assigned': False}}}
            records.append(rec)
            mappings.append({'cell_key': rec['cell_key'], 'group': group, 'question_id': 'SYN-Q1', 'unit_id': unit,
                             'repeat': 1, 'split': 'debug', 'family_id': 'SYNTHETIC-family'})
        ledger = SimpleNamespace(expected_bytes=b'', counts={}, allowed={}, reserved=0)
        metric_ids = [m['metric_id'] for m in read_json(ROOT / 'inputs/metric_catalog.json')]
        report = summarize(records, outputs, {}, {'SYN-Q1': RUBRIC}, SimpleNamespace(cards={c['evidence_id']: c for c in CARDS}), mappings, ledger, metric_ids)
        self.assertEqual(len(report['rows']), 4)
        for row in report['rows']:
            self.assertEqual(len(row['metrics']), 38)
            self.assertEqual(row['metrics']['model_judged_A_cov']['value'], 1 if row['group'] in ['TV', 'GV'] else 0)


class ProviderTests(unittest.TestCase):
    def environment(self, folder, responses):
        config = read_json(ROOT / 'config/execution.json')
        config['scope']['stage'] = 'SYNTHETIC_validation'
        allowed = {slot: 'judge_output' for slot in ['SYN-O_output_A', 'SYN-O_output_B']}
        config['allowed_slots_sha256'] = digest(allowed)
        config['slot_bindings'] = {s: {'business_sha256': digest(REQUEST), 'flow_key': 'SYNTHETIC'} for s in allowed}
        config['knowledge']['card_hashes'] = {c['evidence_id']: digest(c) for c in CARDS}
        ledger = Ledger(folder / 'batch', config, allowed, folder)
        class HTTP:
            def __init__(self): self.calls = 0
            def __call__(self, body):
                response = responses[self.calls];self.calls += 1
                return {'http_status': 200, 'request_id': 'SYNTHETIC-id', 'raw_body': response}
        transport = HTTP()
        provider = Provider(config, FixtureCounter(), ledger, transport, ledger.directory / 'provider')
        body = [{'role': 'system', 'content': 'SYNTHETIC only'}, {'role': 'user', 'content': canonical({'business_request': REQUEST, 'common_context': REQUEST['explicitly_allowed_operating_context'], 'actual_complete_cards': CARDS})}]
        return ledger, provider, transport, body
    def envelope(self, text, usage=True):
        return json.dumps({'model': 'deepseek-flash', 'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30} if usage else None,
                           'choices': [{'finish_reason': 'stop', 'message': {'content': text}}]}).encode()
    def test_accounted_malformed_content_does_not_retry_or_poison_other_slots(self):
        raw = self.envelope('{"SYNTHETIC":1} trailing')
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_', dir=ROOT / 'tests') as tmp:
            ledger, provider, transport, body = self.environment(Path(tmp), [raw, self.envelope('{}')])
            try:
                with self.assertRaises(CodingFailure): provider.send('SYN-O_output_A', 'judge_output', body)
                self.assertFalse(ledger.broken)
                http = read_json(ledger.directory / 'provider/SYN-O_output_A.http.json')
                self.assertEqual(base64.b64decode(http['raw_base64']), raw)
                self.assertEqual(http['request_id'], 'SYNTHETIC-id')
                provider.send('SYN-O_output_B', 'judge_output', body)
                self.assertEqual(transport.calls, 2)
                with self.assertRaises(Rejected): provider.send('SYN-O_output_A', 'judge_output', body)
                self.assertEqual(transport.calls, 2)
            finally: ledger.close()
    def test_unknown_usage_stops_all_future_spending(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_', dir=ROOT / 'tests') as tmp:
            ledger, provider, transport, body = self.environment(Path(tmp), [self.envelope('{}', usage=False)])
            try:
                with self.assertRaises(Rejected): provider.send('SYN-O_output_A', 'judge_output', body)
                self.assertTrue(ledger.broken)
                with self.assertRaises(Rejected): provider.send('SYN-O_output_B', 'judge_output', body)
                self.assertEqual(transport.calls, 1)
            finally: ledger.close()
    def test_oversize_received_bytes_preserved_before_rejection(self):
        raw = b'SYNTHETIC:' + b'x' * (16 * 1024 * 1024 + 1 - len(b'SYNTHETIC:'))
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_', dir=ROOT / 'tests') as tmp:
            ledger, provider, transport, body = self.environment(Path(tmp), [raw])
            try:
                with self.assertRaises(Rejected): provider.send('SYN-O_output_A', 'judge_output', body)
                saved = read_json(ledger.directory / 'provider/SYN-O_output_A.http.json')
                self.assertEqual(saved['raw_size'], 16 * 1024 * 1024 + 1)
                self.assertEqual(base64.b64decode(saved['raw_base64']), raw)
                self.assertTrue(ledger.broken)
                self.assertEqual(transport.calls, 1)
            finally: ledger.close()


if __name__ == '__main__':
    unittest.main()
