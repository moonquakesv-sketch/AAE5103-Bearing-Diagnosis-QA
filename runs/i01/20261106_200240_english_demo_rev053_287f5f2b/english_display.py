"""English presentation only; canonical evidence and historical records stay intact."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).absolute().parent
CATALOG = json.loads((ROOT / 'inputs/presentation.en.json').read_bytes())
CHINESE = re.compile(r'[\u2e80-\u2fff\u3000-\u303f\u3100-\u312f\u31a0-\u31bf'
                     r'\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]')
STAGES = ['SVM Image Classification', 'API Task Identification',
          'Knowledge Graph Retrieval', 'API Evidence-Based Answer']
CLASSES = {'normal', 'inner_race', 'outer_race', 'rolling_element'}
TASKS = {'explanation', 'inspection', 'preparation'}


def english(value, fallback):
    return value if isinstance(value, str) and value.strip() and not CHINESE.search(value) else fallback


def failure_message(failure):
    code = failure.get('code', '')
    fallback = {
        'needs_clarification': 'The question needs clarification. No downstream request was sent.',
        'unsupported': 'The question is outside the supported demonstration scope.',
        'language_failed': 'The model response was not English-only. It was saved but is not displayed.',
        'timeout': 'The model request timed out. No automatic retry was made.',
        'citation_failed': 'The answer did not pass the evidence-ID check and is not displayed.',
    }.get(code, 'This run did not complete. The record was retained; no automatic retry was made.')
    return english(failure.get('message'), fallback)


def scenario_text(predicted_class, scenario):
    if not scenario:
        return 'Classification selects a retrieval route. The scenario cause is an independent teaching assumption.'
    translated = CATALOG['scenarios'].get(predicted_class, {})
    fields = [('assumption_text', 'No teaching assumption is available in English.'),
              ('applicability', 'Generic bearing education only.'),
              ('unknown_information', 'Actual equipment conditions and root cause remain unknown.')]
    values = [english(scenario.get(key), translated.get(key, default)) for key, default in fields]
    return values[0] + '\n\n**Scope:** ' + values[1] + '\n\n**Unknown information:** ' + values[2]


def evidence_text(retrieval):
    rows = []
    for card in (retrieval or {}).get('results', []):
        eid = english(card.get('evidence_id'), 'Evidence ID unavailable')
        source = card.get('source') or {}
        translated = CATALOG['cards'].get(eid, {})
        conditions = english(card.get('conditions'), translated.get('conditions',
                             'Conditions are not available in English. Consult the original record.'))
        limitations = english(card.get('limitations'), translated.get('limitations',
                              'Limitations are not available in English. Consult the original record.'))
        title = english(source.get('title'), 'Source')
        author = english(source.get('organization_author'), 'Source organization')
        url = english(card.get('source_url'), '#')
        locator = english(card.get('page_or_section'), 'Location unavailable in English')
        original = english(card.get('original_text'), 'The original excerpt is not available in English.')
        rows.append(f'**{eid} - {title}**\n\n{original}\n\n'
                    + f'Source: [{author}]({url}); {locator}\n\n'
                    + 'Conditions: ' + conditions + '\n\nLimitations: ' + limitations)
    return '\n\n---\n\n'.join(rows) or 'No evidence has been retrieved yet.'


def answer_text(text):
    if not isinstance(text, str) or not text.strip():
        return 'No answer is available for display.', False
    if not CHINESE.search(text):
        return text, False
    digest = hashlib.sha256(text.encode()).hexdigest()
    translation = CATALOG['saved_answers'].get(digest)
    if translation:
        return english(translation['text'], 'The saved translation is unavailable.'), True
    return ('The original response was retained, but an English presentation translation '
            'is not available for this saved answer.'), False


def render(record):
    if record.get('historical'):
        old = record['old_record']
        flow = old.get('flow_record') or {}
        classifier = old.get('classifier') or {}
        predicted = classifier.get('predicted_class')
        predicted = predicted if predicted in CLASSES else 'Not classified'
        tasks = [t for t in flow.get('request', {}).get('task_types', []) if t in TASKS]
        text, translated = answer_text(flow.get('final_answer') if flow.get('publishable') else None)
        if flow.get('publishable'):
            citations = (flow.get('draft') or {}).get('citations', [])
            if citations:
                text += '\n\n**Cited evidence:** ' + ', '.join(english(eid, 'Unavailable') for eid in citations)
        state = '**Saved Replay - 2026-11-01 - 0 new API requests**'
        state += '\n\n**Limited answer: evidence is insufficient.**' if flow.get('publishable') else '\n\n**No publishable answer was produced in this historical run.**'
        if translated:
            state += '\n\nThe displayed answer is an English translation of the saved response.'
        values = ('Independent stage durations were not saved in this historical record.',
                  predicted, ', '.join(tasks) or 'Not identified',
                  scenario_text(predicted, old.get('scenario') or {'assumption_text': ''}),
                  evidence_text(flow.get('retrieval')), text, state)
    else:
        lines = ['| Stage | Status | Duration |', '| --- | --- | --- |']
        states = {'pending': 'Pending', 'running': 'Running', 'completed': 'Completed',
                  'failed': 'Failed', 'skipped': 'Skipped', 'interrupted': 'Interrupted'}
        for index, stage in enumerate(record.get('stages', [])):
            name = STAGES[index] if index < len(STAGES) else 'Additional stage'
            duration = stage.get('elapsed_seconds')
            elapsed = f'{duration:.2f} s' if isinstance(duration, (int, float)) else '-'
            lines.append(f"| {name} | {states.get(stage.get('status'), 'Unknown')} | {elapsed} |")
        classifier = record.get('classifier') or {}
        predicted = classifier.get('predicted_class')
        predicted = predicted if predicted in CLASSES else 'Not classified'
        tasks = [t for t in (record.get('task_report') or {}).get('task_types', []) if t in TASKS]
        state = 'Saved Replay' if record.get('replay') else 'Live Demonstration - New Run'
        state += ' - ' + english(record.get('started_at'), 'Time not recorded')
        if record.get('failure'):
            state += '\n\n**Run incomplete** - ' + failure_message(record['failure'])
        elif record.get('status') == 'completed':
            state += ('\n\n**Answer generated.** Evidence IDs were checked; the content was not '
                      'independently verified. Retain the stated evidence limitations.')
        count = len(record.get('calls', []))
        state += f'\n\nAPI requests in the recorded run: {count}. '
        state += 'This replay added 0 new API requests.' if record.get('replay') else 'Maximum 2 requests; no automatic retries.'
        published = record.get('publishable') and record.get('status') == 'completed'
        text, translated = answer_text(record.get('answer') if published else None)
        if translated:
            state += '\n\n**English translation of the saved answer.** The original model response is unchanged.'
        values = ('\n'.join(lines), predicted, ', '.join(tasks) or 'Not identified',
                  scenario_text(predicted, record.get('scenario')), evidence_text(record.get('retrieval')),
                  text, state)
    return tuple(english(value, 'Content unavailable in English.') for value in values)
