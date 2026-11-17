"""English-only rendering regression checks. No model or classifier execution."""
from pathlib import Path
import copy
import csv
import hashlib
import io
import json
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
PROJECT = ROOT.parents[2]
LEGACY = PROJECT / 'runs/i01/20261104_184700_live_demo_fcb6b670'
sys.path.insert(0, str(ROOT))
import english_display as display
import live_demo


class EnglishDisplayTests(unittest.TestCase):
    def assert_english(self, values):
        for value in values:
            self.assertIsNone(display.CHINESE.search(value), value)

    def records(self):
        return sorted((LEGACY / 'executions').glob('*/record.json'))

    def test_all_saved_runs_render_in_English_without_mutating_sources(self):
        for path in self.records():
            before = path.read_bytes()
            record = json.loads(before)
            original = copy.deepcopy(record)
            record['replay'] = True
            rendered = display.render(record)
            self.assert_english(rendered)
            self.assertIn('0 new API requests', rendered[6])
            if display.CHINESE.search(original['answer']):
                self.assertIn('English translation of the saved answer', rendered[6])
            self.assertEqual(record['answer'], original['answer'])
            self.assertEqual(path.read_bytes(), before)

    def test_progress_and_legacy_stage_names_never_leak_Chinese(self):
        record = json.loads(self.records()[0].read_bytes())
        for state in ['pending', 'running', 'completed', 'failed', 'skipped', 'interrupted']:
            value = copy.deepcopy(record)
            value['status'] = 'running'
            value['answer'] = None
            value['publishable'] = False
            for stage in value['stages']:
                stage['status'] = state
            rendered = display.render(value)
            self.assert_english(rendered)
            self.assertIn('SVM Image Classification', rendered[0])

    def test_all_four_scenarios_and_16_cards_retain_English_conditions_and_limits(self):
        folder = PROJECT / 'knowledge/i01/v1'
        sources = {r['source_id']: r for r in csv.DictReader(io.StringIO((folder / 'sources.csv').read_text()))}
        cards = list(csv.DictReader(io.StringIO((folder / 'evidence.csv').read_text())))
        for card in cards:
            card['source'] = sources[card['source_id']]
        self.assertEqual(len(cards), 16)
        evidence = display.evidence_text({'results': cards})
        self.assert_english([evidence])
        self.assertEqual(evidence.count('Conditions:'), 16)
        self.assertEqual(evidence.count('Limitations:'), 16)
        for scenario in csv.DictReader(io.StringIO((folder / 'scenarios.csv').read_text())):
            text = display.scenario_text(scenario['predicted_class'], scenario)
            self.assert_english([text])
            self.assertIn('Unknown information:', text)

    def test_Chinese_live_answer_is_blocked_and_English_citations_still_work(self):
        retrieval = {'results': [{'evidence_id': 'EV-K01-001'}]}
        with self.assertRaises(live_demo.DemoError) as error:
            live_demo.validate_answer('\u4e2d\u6587 [EV-K01-001]', retrieval)
        self.assertEqual(error.exception.code, 'language_failed')
        self.assert_english([str(error.exception)])
        text = 'English guidance [EV-K01-001].'
        self.assertEqual(live_demo.validate_answer(text, retrieval), text)

    def test_Chinese_failure_and_unknown_saved_text_have_English_fallbacks(self):
        record = json.loads(self.records()[0].read_bytes())
        record.update(status='failed', publishable=False,
                      failure={'code': 'unsupported', 'message': '\u4e2d\u6587\u9519\u8bef'})
        values = display.render(record)
        self.assert_english(values)
        self.assertIn('outside the supported', values[6])
        text, translated = display.answer_text('\u672a\u77e5\u7684\u4e2d\u6587\u56de\u7b54')
        self.assert_english([text])
        self.assertFalse(translated)
        self.assertIn('not available', text)

    def test_translation_entries_match_original_answer_and_record_hashes(self):
        self.assertEqual(len(display.CATALOG['saved_answers']), 4)
        for digest, entry in display.CATALOG['saved_answers'].items():
            path = Path(entry['source_record'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry['source_record_sha256'])
            original = json.loads(path.read_bytes())['answer']
            self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), digest)
            before_ids = set(live_demo.re.findall(r'EV-[A-Za-z0-9-]+', original))
            after_ids = set(live_demo.re.findall(r'EV-[A-Za-z0-9-]+', entry['text']))
            self.assertEqual(before_ids, after_ids)
            self.assert_english([entry['text']])

    def test_November_1_backup_replays_are_English_and_preserve_failure(self):
        for key in ['demo_1', 'demo_2']:
            old = json.loads((ROOT / f'inputs/{key}_{key}_nov01_record.json').read_bytes())
            values = display.render({'historical': True, 'old_record': old, 'replay': True})
            self.assert_english(values)
            if key == 'demo_1':
                self.assertIn('No publishable answer', values[6])
                self.assertIn('No answer is available', values[5])
            else:
                self.assertIn('evidence is insufficient', values[6])
                self.assertIn('EV-K01-011', values[5])

    def test_answer_prompt_requires_English_and_keeps_complete_canonical_cards(self):
        record = json.loads(self.records()[0].read_bytes())
        messages = live_demo.answer_messages(record)
        self.assertIn('Respond entirely in English', messages[0]['content'])
        self.assertIn('Do not output Chinese', messages[0]['content'])
        payload = json.loads(messages[1]['content'])
        self.assertEqual(payload['returned_cards'], record['retrieval']['results'])
        self.assertEqual(set(payload['classifier_context']), {'origin', 'predicted_class'})

    def test_page_labels_are_English_in_missing_key_and_ready_states(self):
        import app
        engine = live_demo.Engine()
        for key in ['', 'SYNTHETIC-only-key']:
            with patch.dict('os.environ', {'DEEPSEEK_API_KEY': key}):
                page = app.build_page(engine)
                for component in page.config['components']:
                    props = component.get('props', {})
                    for name in ['label', 'value', 'info', 'placeholder']:
                        if isinstance(props.get(name), str) and not props[name].startswith('/'):
                            self.assert_english([props[name]])
        self.assertIn('changeLocale("en")', app.english_locale_js())
        self.assertIn('import("/assets/', app.english_locale_js())


if __name__ == '__main__':
    unittest.main()
