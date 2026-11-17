"""Offline K02 retrieval. No classifier, source crawling, answer generation or API.

The public entry points are Retriever(knowledge_dir, config_path).retrieve(request,
mode='text'|'graph', top_k=3), index_snapshot(), and the CLI below.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import validate_k01 as k01

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'k02.v1_rev013'
KNOWLEDGE_VERSION = '20261023_rev011'
TASKS = ('explanation', 'inspection', 'preparation')
CLASSES = ('normal', 'inner_race', 'outer_race', 'rolling_element')
INPUT_FIELDS = frozenset(k01.ALLOWED_INPUTS)
FORBIDDEN = k01.FORBIDDEN_INPUTS | {
    'true_label', 'sample_id', 'sample_path', 'sample_filename', 'file_name',
    'filename', 'path', 'answer', 'reference_answer', 'rubric', 'notes',
}
CONTEXT = {'operating_state': ('unknown', 'standstill'),
           'external_vibration': ('unknown', 'assumed_for_demo')}
KNOWLEDGE_HASHES = {
    'sources.csv': '8b18c3b602e85844859622e3c46a69f0ad1fb8e46feaee38adeeb9ffb6d92987',
    'scenarios.csv': '64ad3bad7a7e1581ed64515476e490afce13be21315a88978fbd1d5f7ba0bfe0',
    'evidence.csv': '87af57bd684f3abb9a29e11bde5749363ed7451834ca74dab11e0e7d2559c262',
    'entities.csv': '9ae1e2838d0226bd57f2ba286fad4507a83c476787005cd6e3cbd60fbcd6d16d',
    'relations.csv': 'cf54bd05be9aad91f1dd634bd22cdf787bfea90220663d2210804b7ca47fc181',
    'interface.json': '708b48aaa4a8b5ff2c72832f9d7c0373569d05464f612e5d648382a37ef3a607',
}
FIXED_CONFIG = {
    'implementation_version': VERSION, 'knowledge_version': KNOWLEDGE_VERSION,
    'knowledge_hashes': KNOWLEDGE_HASHES,
    'index_fields': ['original_text', 'claim_en'],
    'tfidf': {'lowercase': True, 'strip_accents': 'unicode',
              'token_pattern': r'(?u)\b[a-zA-Z][a-zA-Z]+\b', 'stop_words': 'english',
              'ngram_range': [1, 1], 'norm': 'l2', 'use_idf': True,
              'smooth_idf': True, 'sublinear_tf': False},
    'score': 'cosine_similarity', 'tie_decimals': 12,
    'tie_break': 'evidence_id_ascending', 'minimum_score_exclusive': 0,
    'budget': 'global_top_k_unique_complete_cards_no_task_quotas',
    'query_policy': 'question_class_assumed_cause_canonical_tasks_sorted_context_v1',
    'terminology_mapping': None,
}
NOTICES = [
    'Retrieval finds material; it does not establish answer sufficiency or semantic verification.',
    'Scores are text similarity, not knowledge truth, root-cause probability or maintenance confidence.',
    'The class is caller supplied; no classifier is connected in K02 development runs.',
    'Standstill external vibration is an independent demo assumption; poor lubrication is only another possibility.',
    'EV-K01-010 gives inspection directions only; it cannot alone decide reuse, replacement or release.',
    'Physical surface inspection is not a CWT image criterion.',
    'Timken cards retain registered online-text provenance; same-version local PDF/hash and visual checks remain missing.',
    'No model-specific or aviation instructions, parameters or disposition conclusions are produced.',
    'Field checks cannot detect all hidden answers, incorrect premises or non-English content inside free text.',
]


class K02Error(ValueError):
    def __init__(self, code, reason):
        self.code = code
        super().__init__(reason)


def require(condition, code, reason):
    if not condition:
        raise K02Error(code, reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'invalid_json', 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid_constant(value):
        raise K02Error('invalid_json', 'Non-finite JSON number: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid_constant)


def validate_request(request):
    require(type(request) is dict, 'invalid_input', 'Business input must be an object.')
    keys = set(request)
    require(not keys & FORBIDDEN, 'forbidden_field', 'Forbidden input fields: ' + str(sorted(keys & FORBIDDEN)))
    require(keys == INPUT_FIELDS, 'input_fields',
            f'Missing fields: {sorted(INPUT_FIELDS - keys)}; unknown fields: {sorted(keys - INPUT_FIELDS, key=str)}')
    for field in ('predicted_class', 'scenario_id', 'question'):
        require(type(request[field]) is str, 'input_type', field + ' must be a string.')
        try:
            request[field].encode('utf-8')
        except UnicodeEncodeError as error:
            raise K02Error('invalid_unicode', field + ' must contain valid Unicode scalar values.') from error
    require(request['predicted_class'] in CLASSES, 'unknown_class', 'Unknown predicted_class.')
    require(bool(request['scenario_id'].strip()), 'empty_scenario', 'scenario_id must not be empty.')
    tasks = request['task_types']
    require(type(tasks) is list and bool(tasks) and all(type(t) is str for t in tasks),
            'input_type', 'task_types must be a nonempty string list.')
    require(len(tasks) == len(set(tasks)) and set(tasks) <= set(TASKS),
            'unknown_or_duplicate_task', 'task_types contains unknown or duplicate tasks.')
    context = request['explicitly_allowed_operating_context']
    require(type(context) is dict, 'input_type', 'explicitly_allowed_operating_context must be an object.')
    require(set(context) <= set(CONTEXT), 'context_fields', 'Unapproved operating-context key; arbitrary notes/nesting are forbidden.')
    for key, value in context.items():
        require(type(value) is str and value in CONTEXT[key], 'context_value',
                key + ' must be one of ' + str(CONTEXT[key]) + '; values describe demo context only.')


def load_knowledge(folder):
    """Read exactly six pinned files, once; parse the same bytes that were hashed."""
    blobs = {}
    for name, expected in KNOWLEDGE_HASHES.items():
        path = Path(folder) / name
        require(not path.is_symlink(), 'knowledge_file', 'Knowledge file must not be a symlink: ' + name)
        blob = path.read_bytes()
        actual = hashlib.sha256(blob).hexdigest()
        require(actual == expected, 'knowledge_hash_drift',
                f'{name}: expected {expected}; actual {actual}; knowledge must be independently re-approved before a new version.')
        blobs[name] = blob.decode('utf-8')
    tables = {}
    for name in k01.NAMES:
        reader = csv.DictReader(io.StringIO(blobs[name + '.csv'], newline=''))
        require(reader.fieldnames == k01.FIELDS[name].split(), 'knowledge_schema', name + ': unexpected fields.')
        tables[name] = list(reader)
    interface = strict_json(blobs['interface.json'])
    # Structural mapping checks remain mandatory, even after provenance checks.
    try:
        k01.validate(tables, check_files=False, interface=interface)
    except k01.Invalid as error:
        raise K02Error('knowledge_structure', str(error)) from error
    return tables, interface


def graph_paths(relations, entities, predicted_class, scenario_id, tasks):
    """Pure traversal; isolated tests may supply synthetic relations, never load them as K01."""
    forward, reverse = {}, {}
    for edge in sorted(relations, key=lambda r: r['relation_id']):
        forward.setdefault((edge['subject_id'], edge['relation']), []).append(edge)
        reverse.setdefault((edge['object_id'], edge['relation']), []).append(edge)
    part_nodes = [e['entity_id'] for e in entities
                  if e['entity_type'] == 'part' and e['label'].split(' / ')[0] == predicted_class]
    task_nodes = {e['entity_id']: e['label'] for e in entities if e['entity_type'] == 'task'}
    paths = {}
    def step(edge, backwards=False):
        return {'from_node': edge['object_id'] if backwards else edge['subject_id'],
                'to_node': edge['subject_id'] if backwards else edge['object_id'],
                'direction': 'reverse' if backwards else 'forward', 'edge': copy.deepcopy(edge)}
    for part in part_nodes:
        for route in reverse.get((part, 'routes_from_class'), []):
            if route['subject_id'] != scenario_id:
                continue
            for task in forward.get((scenario_id, 'has_task'), []):
                if task_nodes.get(task['object_id']) not in tasks:
                    continue
                for knowledge in forward.get((task['object_id'], 'has_evidence_backed_content'), []):
                    for support in forward.get((knowledge['object_id'], 'supported_by'), []):
                        path = [step(route, True), step(task), step(knowledge), step(support)]
                        paths.setdefault(support['object_id'], []).append(path)
    return paths


def rank_candidates(ids, scores):
    return sorted((eid for eid in ids if scores[eid] > 0),
                  key=lambda eid: (-round(scores[eid], FIXED_CONFIG['tie_decimals']), eid))


class Retriever:
    def __init__(self, knowledge_dir=None, config_path=None):
        config_path = Path(config_path) if config_path else ROOT / 'configs/k02_retrieval.json'
        config_bytes = config_path.read_bytes()
        config = strict_json(config_bytes)
        require(canonical(config) == canonical(FIXED_CONFIG), 'config_drift',
                'This implementation accepts only its fixed registered scoring/corpus configuration.')
        self.config = copy.deepcopy(config)
        self.config_file_sha256 = hashlib.sha256(config_bytes).hexdigest()
        self.tables, self.interface = load_knowledge(knowledge_dir or ROOT / 'knowledge/k01')
        self.evidence = {e['evidence_id']: e for e in self.tables['evidence']}
        self.sources = {s['source_id']: s for s in self.tables['sources']}
        self.scenarios = {s['scenario_id']: s for s in self.tables['scenarios']}
        self.ids = sorted(self.evidence)
        self.documents = ['\n'.join(self.evidence[eid][f] for f in config['index_fields']) for eid in self.ids]
        params = dict(config['tfidf'], ngram_range=tuple(config['tfidf']['ngram_range']))
        self.vectorizer = TfidfVectorizer(**params)
        self.matrix = self.vectorizer.fit_transform(self.documents)
        self.index_id = digest(self.index_snapshot())

    def index_snapshot(self):
        return {'knowledge_version': KNOWLEDGE_VERSION, 'knowledge_hashes': KNOWLEDGE_HASHES,
                'indexed_fields': self.config['index_fields'],
                'documents': [{'evidence_id': eid, 'text': text} for eid, text in zip(self.ids, self.documents)],
                'feature_names': self.vectorizer.get_feature_names_out().tolist(),
                'idf': self.vectorizer.idf_.tolist(), 'tfidf_matrix': self.matrix.toarray().tolist(),
                'fit_population': 'fixed evidence rows only; queries never passed to fit',
                'source_files_indexed': False, 'audit_material_indexed': False}

    def retrieve(self, request, mode='text', top_k=3):
        validate_request(request)
        require(type(mode) is str and mode in ('text', 'graph'), 'invalid_mode', 'mode must be text or graph.')
        require(type(top_k) is int and 1 <= top_k <= 11, 'invalid_budget', 'top_k must be an integer in [1, 11], not bool.')
        request = copy.deepcopy(request)
        tasks = [t for t in TASKS if t in request['task_types']]
        scenario = self.scenarios.get(request['scenario_id'])
        if scenario:
            require(request['predicted_class'] == scenario['predicted_class'], 'class_scenario_conflict',
                    'The selected scenario belongs to ' + scenario['predicted_class'] + '.')
        config_id = digest({'shared_config': self.config, 'mode': mode, 'top_k': top_k})
        result = {'status': None, 'reason': None, 'request': request,
                  'implementation_version': VERSION, 'knowledge_version': KNOWLEDGE_VERSION,
                  'retrieval_mode': mode, 'configuration_id': config_id,
                  'shared_config_id': digest(self.config), 'config_file_sha256': self.config_file_sha256,
                  'index_id': self.index_id, 'canonical_task_order': tasks,
                  'score_semantics': 'TF-IDF cosine text similarity only',
                  'budget': {'top_k': top_k, 'unit': 'unique_complete_evidence_cards',
                             'allocation': self.config['budget'], 'truncation': 'none', 'unused_slots': top_k},
                  'information_separation': {
                      'caller_class': request['predicted_class'], 'class_origin': 'caller_supplied_not_verified',
                      'demo_scenario': copy.deepcopy(scenario),
                      'operating_context_semantics': 'caller-selected demo assumptions or unknown; no observed measurements',
                      'source_knowledge': 'evidence cards below retain modality and source provenance',
                      'unknowns': scenario['unknown_information'] if scenario else 'No scenario knowledge is registered.'},
                  'notices': list(NOTICES), 'claim_en_semantics': self.interface['claim_en_semantics'],
                  'candidate_count': 0, 'candidate_ids': [], 'positive_candidate_count': 0,
                  'returned_count': 0, 'scored_candidates': [], 'graph_candidate_paths': {},
                  'query_text': None, 'per_task': {}, 'results': []}

        def empty(status, reason):
            result.update(status=status, reason=reason)
            result['per_task'] = {t: {'status': status, 'reason': reason, 'returned_ids': [],
                                      'candidate_count': 0, 'positive_count': 0} for t in tasks}
            return result
        if not request['question'].strip():
            return empty('empty_question', 'The question is blank; no retrieval was performed.')
        if not scenario:
            return empty('knowledge_not_covered', 'No registered scenario for this request; no outer-race fallback.')
        analyzed = self.vectorizer.build_analyzer()(request['question'])
        if not analyzed:
            return empty('no_valid_terms', 'The question has no terms after the fixed tokenizer and stop-word removal.')
        if self.vectorizer.transform([request['question']]).nnz == 0:
            return empty('no_vocabulary_terms', 'No question terms occur in the fixed knowledge vocabulary; context cannot manufacture a match.')
        context = request['explicitly_allowed_operating_context']
        query = '\n'.join([request['question'], request['predicted_class'].replace('_', ' '),
                           scenario['assumed_cause'].replace('_', ' '), ' '.join(tasks)] +
                          [context[key].replace('_', ' ') for key in sorted(context)])
        result['query_text'] = query
        scores_array = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        scores = {eid: float(score) for eid, score in zip(self.ids, scores_array)}
        paths = graph_paths(self.tables['relations'], self.tables['entities'], request['predicted_class'],
                            request['scenario_id'], tasks) if mode == 'graph' else {}
        candidates = sorted(paths) if mode == 'graph' else list(self.ids)
        ranked = rank_candidates(candidates, scores)
        chosen = ranked[:top_k]
        result.update(candidate_count=len(candidates), candidate_ids=candidates,
                      positive_candidate_count=len(ranked), returned_count=len(chosen),
                      graph_candidate_paths=paths,
                      scored_candidates=[{'evidence_id': eid, 'score': scores[eid]} for eid in candidates])
        result['budget']['unused_slots'] = top_k - len(chosen)
        for position, eid in enumerate(chosen, 1):
            evidence = copy.deepcopy(self.evidence[eid])
            evidence.update(rank=position, score=scores[eid],
                            source=copy.deepcopy(self.sources[evidence['source_id']]),
                            relation_paths=copy.deepcopy(paths.get(eid, [])))
            result['results'].append(evidence)
        for task in tasks:
            own = [eid for eid in candidates if self.evidence[eid]['task_type'] == task]
            positive = [eid for eid in ranked if eid in own]
            returned = [eid for eid in chosen if eid in own]
            if returned:
                status, reason = 'retrieved', 'At least one card tagged for this task was returned; answer sufficiency is unverified.'
            elif not own:
                status, reason = 'no_candidates', 'No candidate card is tagged for this task.'
            elif not positive:
                status, reason = 'zero_similarity', 'All candidate cards tagged for this task have zero similarity.'
            else:
                status, reason = 'budget_exhausted', 'Positive cards for this task did not fit the shared global top_k budget.'
            result['per_task'][task] = dict(status=status, reason=reason, candidate_count=len(own),
                                            positive_count=len(positive), returned_ids=returned)
        if not candidates:
            result.update(status='no_candidates', reason='No evidence was reachable under the selected relationship constraints.')
        elif not chosen:
            result.update(status='zero_similarity', reason='All candidate scores are zero; zero-score cards are never used to fill the budget.')
        else:
            complete = all(v['status'] == 'retrieved' for v in result['per_task'].values())
            result.update(status='retrieved' if complete else 'partial_task_coverage',
                          reason='Positive-similarity cards retrieved; see per_task for card coverage, not answer correctness.')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--mode', choices=['text', 'graph'], required=True)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--knowledge', type=Path, default=ROOT / 'knowledge/k01')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/k02_retrieval.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        # Reserve before work; existing files (including dangling symlinks) are never overwritten.
        with args.output.open('xb') as stream:
            try:
                request = strict_json(args.request.read_text(encoding='utf-8'))
                result = Retriever(args.knowledge, args.config).retrieve(request, args.mode, args.top_k)
                # Finish serialization and encoding before writing any result bytes.
                payload = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
                code = 0
            except (K02Error, OSError, ValueError, UnicodeError) as error:
                result = {'status': 'rejected', 'error_code': getattr(error, 'code', 'invalid_file_or_json'),
                          'reason': str(error), 'request_echoed': False}
                # Error text can itself contain invalid Unicode, for example a duplicate JSON key.
                payload = (json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
                code = 2
            stream.write(payload)
        print(json.dumps({'status': result['status'], 'output': str(args.output)}, ensure_ascii=True))
        return code
    except OSError as error:
        print(json.dumps({'status': 'output_rejected', 'reason': str(error)}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
