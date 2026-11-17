"""SYNTHETIC protocol and rejection tests; no model calls or factual scoring."""
from pathlib import Path
import copy
import json
import random
import sys
import unittest
import jsonschema

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import quote_anchor as qa
from baseline import validate_rating as old_validate
from common import digest


def fixture(text='SYNTHETIC: Alpha is visible. Beta is qualified!\nGamma is unknown.'):
    packet = {'unit_id': 'N-SYNTHETIC', 'business_request': {'question': 'SYNTHETIC: no real evaluation'},
              'common_context': {}, 'reference_points': [{'point_id': 'SYN-P1', 'half_credit_eligible': False}],
              'reference_material': [], 'observed_text': text}
    packet, binding = qa.prepare_packet(packet)
    span = packet['quote_spans'][0]['span_id']
    raw = {'schema_version': qa.WIRE_VERSION,
           'points': [{'point_id': 'SYN-P1', 'score': 1, 'quotes': [{'span_ids': [span]}], 'reason': 'SYNTHETIC fixture'}],
           'claims': [{'claim_id': 'SYN-C1', 'quote': {'span_ids': [span]}, 'support': 'full', 'reason': 'SYNTHETIC fixture'}],
           'voluntary_citations': [], 'complete_claim_inventory': True, 'complete_citation_inventory': True}
    return packet, binding, raw


class QuoteAnchorTests(unittest.TestCase):
    def test_old_numeric_failure_and_new_resolution(self):
        packet, binding, raw = fixture()
        canonical = qa.validate_rating(raw, packet, binding)['rating']
        broken = copy.deepcopy(canonical)
        broken['points'][0]['quotes'][0]['start'] += 1
        with self.assertRaisesRegex(ValueError, 'quote offsets/verbatim mismatch'):
            old_validate(broken, packet)
        self.assertEqual(canonical['points'][0]['quotes'][0]['text'], packet['quote_spans'][0]['text'])

    def test_raw_and_packet_are_unchanged(self):
        p, b, r = fixture(); originals = copy.deepcopy((p, b, r))
        qa.validate_rating(r, p, b)
        self.assertEqual((p, b, r), originals)

    def test_scores_and_reasons_are_unchanged(self):
        p, b, r = fixture(); result = qa.validate_rating(r, p, b)['rating']
        self.assertEqual(result['points'][0]['score'], r['points'][0]['score'])
        self.assertEqual(result['points'][0]['reason'], r['points'][0]['reason'])
        self.assertEqual(result['claims'][0]['support'], r['claims'][0]['support'])

    def test_contiguous_range_includes_original_whitespace(self):
        p, b, r = fixture('SYNTHETIC: One.\r\n\r\nTwo!  Three.')
        r['points'][0]['quotes'][0]['span_ids'] = [x['span_id'] for x in p['quote_spans'][:2]]
        q = qa.validate_rating(r, p, b)['rating']['points'][0]['quotes'][0]
        self.assertEqual(q['text'], 'SYNTHETIC: One.\r\n\r\nTwo!')

    def test_repeated_text_has_distinct_explicit_ids(self):
        p, b, r = fixture('SYNTHETIC: Repeat. Repeat. Repeat.')
        spans = qa.build_spans(p['observed_text'])
        self.assertEqual(spans[1]['text'], spans[2]['text'])
        self.assertNotEqual(spans[1]['span_id'], spans[2]['span_id'])
        r['points'][0]['quotes'][0]['span_ids'] = [spans[2]['span_id']]
        self.assertEqual(qa.validate_rating(r, p, b)['rating']['points'][0]['quotes'][0]['start'], spans[2]['start'])

    def test_unicode_codepoints_crlf_combining_and_emoji(self):
        text = 'SYNTHETIC: 轴承😀 e\u0301.\r\n第二句！\tLast?'
        p, b, r = fixture(text)
        for span in qa.build_spans(text):
            q = qa.resolve_quote({'span_ids': [span['span_id']]}, text, qa.build_spans(text))
            self.assertEqual(text[q['start']:q['end']], span['text'])
        qa.validate_rating(r, p, b)

    def test_abbreviations_do_not_lose_visible_characters(self):
        text = 'SYNTHETIC: e.g. A... B.\n1. Example.'
        covered = set()
        for s in qa.build_spans(text):
            covered.update(range(s['start'], s['end']))
        self.assertTrue(all(i in covered for i, c in enumerate(text) if not c.isspace()))

    def test_deterministic_fuzz_visible_coverage(self):
        rng = random.Random(47)
        for _ in range(100):
            text = 'SYNTHETIC: ' + ''.join(rng.choice('ab.?! \t\r\n😀中e\u0301') for _ in range(80))
            spans = qa.build_spans(text)
            self.assertEqual(spans, qa.build_spans(text))
            covered = set(i for s in spans for i in range(s['start'], s['end']))
            self.assertTrue(all(i in covered for i, c in enumerate(text) if not c.isspace()))

    def test_unknown_id_rejected(self):
        p, b, r = fixture(); r['claims'][0]['quote']['span_ids'] = ['S000000000000-9999']
        with self.assertRaisesRegex(ValueError, 'unknown or wrong-source'): qa.validate_rating(r, p, b)

    def test_wrong_source_id_rejected(self):
        p, b, r = fixture(); other, _, _ = fixture('SYNTHETIC: Other source.')
        r['claims'][0]['quote']['span_ids'] = [other['quote_spans'][0]['span_id']]
        with self.assertRaisesRegex(ValueError, 'wrong-source'): qa.validate_rating(r, p, b)

    def test_reversed_ids_rejected(self):
        p, b, r = fixture(); r['points'][0]['quotes'][0]['span_ids'] = [x['span_id'] for x in p['quote_spans'][:2]][::-1]
        with self.assertRaisesRegex(ValueError, 'ordered and contiguous'): qa.validate_rating(r, p, b)

    def test_skipped_range_rejected(self):
        p, b, r = fixture(); r['points'][0]['quotes'][0]['span_ids'] = [p['quote_spans'][i]['span_id'] for i in [0, 2]]
        with self.assertRaisesRegex(ValueError, 'ordered and contiguous'): qa.validate_rating(r, p, b)

    def test_duplicate_ids_rejected(self):
        p, b, r = fixture(); r['points'][0]['quotes'][0]['span_ids'] *= 2
        with self.assertRaises(jsonschema.ValidationError): qa.validate_rating(r, p, b)

    def test_empty_ids_rejected(self):
        p, b, r = fixture(); r['claims'][0]['quote']['span_ids'] = []
        with self.assertRaises(jsonschema.ValidationError): qa.validate_rating(r, p, b)

    def test_numeric_offsets_forbidden_in_new_wire(self):
        p, b, r = fixture(); r['claims'][0]['quote']['start'] = 0
        with self.assertRaises(jsonschema.ValidationError): qa.validate_rating(r, p, b)

    def test_free_text_quote_forbidden_in_new_wire(self):
        p, b, r = fixture(); r['claims'][0]['quote']['text'] = 'Invented words'
        with self.assertRaises(jsonschema.ValidationError): qa.validate_rating(r, p, b)

    def test_wrong_source_hash_rejected(self):
        p, b, r = fixture(); b['observed_text_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'source hash'): qa.validate_rating(r, p, b)

    def test_changed_packet_is_rejected(self):
        p, b, r = fixture(); p['reference_material'] = ['tampered']
        with self.assertRaisesRegex(ValueError, 'packet changed'): qa.validate_rating(r, p, b)

    def test_tampered_span_registry_rejected_even_if_packet_rebound(self):
        p, b, r = fixture(); p['quote_spans'][0]['text'] = 'not source text'; b['packet_sha256'] = digest(p)
        with self.assertRaisesRegex(ValueError, 'span registry'): qa.validate_rating(r, p, b)

    def test_cross_unit_binding_rejected(self):
        p, b, r = fixture(); b['unit_id'] = 'N-OTHER'
        with self.assertRaisesRegex(ValueError, 'unit binding'): qa.validate_rating(r, p, b)

    def test_missing_fixed_point_rejected(self):
        p, b, r = fixture(); r['points'] = []
        with self.assertRaisesRegex(ValueError, 'complete fixed points'): qa.validate_rating(r, p, b)

    def test_duplicate_fixed_point_rejected(self):
        p, b, r = fixture(); r['points'] *= 2
        with self.assertRaisesRegex(ValueError, 'complete fixed points'): qa.validate_rating(r, p, b)

    def test_positive_without_quote_rejected(self):
        p, b, r = fixture(); r['points'][0]['quotes'] = []
        with self.assertRaisesRegex(ValueError, 'positive point'): qa.validate_rating(r, p, b)

    def test_ineligible_half_credit_rejected(self):
        p, b, r = fixture(); r['points'][0]['score'] = 0.5
        with self.assertRaisesRegex(ValueError, 'half credit'): qa.validate_rating(r, p, b)

    def test_eligible_half_credit_preserved(self):
        p, b, r = fixture(); p['reference_points'][0]['half_credit_eligible'] = True; b['packet_sha256'] = digest(p); r['points'][0]['score'] = 0.5
        self.assertEqual(qa.validate_rating(r, p, b)['rating']['points'][0]['score'], 0.5)

    def test_zero_and_null_omissions_preserved(self):
        for score in [0, None]:
            p, b, r = fixture(); r['points'][0]['score'] = score; r['points'][0]['quotes'] = []
            self.assertEqual(qa.validate_rating(r, p, b)['rating']['points'][0]['score'], score)

    def test_duplicate_claim_id_rejected(self):
        p, b, r = fixture(); r['claims'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate coding ID'): qa.validate_rating(r, p, b)

    def test_empty_reason_rejected(self):
        p, b, r = fixture(); r['points'][0]['reason'] = ' '
        with self.assertRaisesRegex(ValueError, 'reason missing'): qa.validate_rating(r, p, b)

    def test_source_citation_anchor_preserved(self):
        p, b, r = fixture(); r['voluntary_citations'] = [{'citation_id': 'SYN-V1', 'quote': {'span_ids': [p['quote_spans'][0]['span_id']]}, 'status': 'unverifiable', 'reason': 'SYNTHETIC'}]
        self.assertEqual(qa.validate_rating(r, p, b)['rating']['voluntary_citations'][0]['status'], 'unverifiable')

    def test_old_wire_version_not_accepted(self):
        p, b, r = fixture(); r.pop('schema_version')
        with self.assertRaises(jsonschema.ValidationError): qa.validate_rating(r, p, b)

    def test_whitespace_source_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no visible source spans'): fixture(' \r\n\t')

    def test_generation_messages_prohibited(self):
        p, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'generation prohibited'): qa.messages('generate', p)

    def test_complete_schema_and_fresh_context(self):
        p, _, _ = fixture(); messages = qa.messages('judge_output', p)
        self.assertEqual([m['role'] for m in messages], ['system', 'user'])
        self.assertIn('ref.rating.span_ids.v1', messages[0]['content'])
        self.assertEqual(json.loads(messages[1]['content']), p)

    def test_third_retains_full_new_primary_outcomes(self):
        p, b, r = fixture(); rating = qa.validate_rating(r, p, b)['rating']
        a = {'status': 'valid', 'rating': rating, 'source_binding': b}
        failed = {'status': 'evaluator_failure', 'rating': None, 'source_binding': b}
        third = qa.third_packet(p, a, failed)
        self.assertEqual(third['A']['rating'], rating)
        self.assertIsNone(third['B']['rating'])
        self.assertEqual(set(third), {'packet', 'A', 'B'})

    def test_cross_source_primary_rejected(self):
        p, b, r = fixture(); a = {'status': 'evaluator_failure', 'rating': None, 'source_binding': dict(b, unit_id='N-OTHER')}
        with self.assertRaisesRegex(ValueError, 'different frozen source'): qa.third_packet(p, a, a)

    def test_failed_primary_cannot_carry_rating(self):
        p, b, r = fixture(); a = {'status': 'evaluator_failure', 'rating': r, 'source_binding': b}
        with self.assertRaisesRegex(ValueError, 'null/status mismatch'): qa.third_packet(p, a, a)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QuoteAnchorTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'success': result.wasSuccessful(), 'fixtures': 'SYNTHETIC only', 'model_API_calls': 0,
               'new_scores_for_real_answers': 0}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
