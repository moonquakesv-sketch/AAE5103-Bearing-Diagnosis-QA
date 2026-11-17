"""Load frozen complete v2.1 cards; preserve K02 TF-IDF and graph ranking policy."""
import csv
import io
import sys
from pathlib import Path
import copy
from common import require, read_bytes, read_json, file_hash, digest, safe_path, validate_business, TASKS, canonical


class Corpus:
    def __init__(self, root, binding, contexts=None):
        self.root = safe_path(root)
        self.binding = copy.deepcopy(binding)
        folder = self.root / binding['directory']
        require(file_hash(folder / 'manifest.json') == binding['manifest_sha256'], 'knowledge manifest pin changed')
        self.manifest = read_json(folder / 'manifest.json')
        for p, h in binding['files'].items():
            require(file_hash(self.root / p) == h, 'frozen knowledge/source byte drift: ' + p)
        self.cards = {c['evidence_id']: c for c in read_json(folder / 'cards.json')}
        self.card_pins = {k:digest(c) for k,c in self.cards.items()}
        require(self.card_pins==binding['card_hashes'],'complete canonical card pins differ')
        require(len(self.cards) == 16, 'exactly16 frozen cards required')
        self.tables = {}
        for name in ['evidence', 'entities', 'relations', 'scenarios', 'sources']:
            self.tables[name] = list(csv.DictReader(io.StringIO(read_bytes(folder / (name + '.csv')).decode())))
        self.scenarios = {r['scenario_id']: r for r in self.tables['scenarios']}
        self.contexts = contexts
        for row in self.tables['evidence']:
            require(all(self.cards[row['evidence_id']][k] == v for k, v in row.items()), 'CSV/card mirror mismatch')
        bykn = {c['knowledge_id']: c for c in self.cards.values()}
        for e in self.tables['entities']:
            if e['entity_id'] in bykn:
                require(e['label'] == bykn[e['entity_id']]['claim_en'], 'entity/card mirror mismatch')
        for e in self.tables['relations']:
            if e['evidence_id']:
                c = self.cards[e['evidence_id']]
                require(e['conditions'] == c['conditions'] and e['limitations'] == c['limitations'], 'relationship conditions drift')
        sys.path.insert(0, str(self.root / 'scripts'))
        import retrieve_k02 as k02
        self.k02 = k02
        require(file_hash(Path(k02.__file__)) == binding['algorithm_sha256'], 'K02 algorithm changed')
        self.ids = sorted(self.cards)
        cfg = copy.deepcopy(k02.FIXED_CONFIG)
        # Source documents, rubrics, references and scores are not index inputs.
        self.documents = ['\n'.join(self.cards[eid][f] for f in cfg['index_fields']) for eid in self.ids]
        self.vectorizer = k02.TfidfVectorizer(**dict(cfg['tfidf'], ngram_range=tuple(cfg['tfidf']['ngram_range'])))
        self.matrix = self.vectorizer.fit_transform(self.documents)
        self.identity = {'knowledge_version': self.manifest['knowledge_version'], 'manifest_sha256': binding['manifest_sha256'],
                         'algorithm_sha256': binding['algorithm_sha256'], 'index_fields': cfg['index_fields'],
                         'index_sha256': digest({'documents': self.documents, 'idf': self.vectorizer.idf_.tolist()}),
                         'query_policy': 'question,class,assumption,tasks,canonical-approved-nested-context; same for text/graph'}

    def retrieve(self, request, mode, top_k=3):
        validate_business(request, self.scenarios, self.contexts)
        require(mode in ['text', 'graph'] and type(top_k) is int and top_k == 3, 'fixed text/graph global top3 only')
        tasks = [t for t in TASKS if t in request['task_types']]
        paths = self.k02.graph_paths(self.tables['relations'], self.tables['entities'], request['predicted_class'], request['scenario_id'], tasks) if mode == 'graph' else {}
        candidates = sorted(paths) if mode == 'graph' else self.ids[:]
        q = request['question']
        query = '\n'.join([q, request['predicted_class'].replace('_', ' '), self.scenarios[request['scenario_id']]['assumed_cause'].replace('_', ' '),
                           ' '.join(tasks), canonical(request['explicitly_allowed_operating_context'])])
        if self.vectorizer.transform([q]).nnz == 0:
            scores = {eid: 0.0 for eid in self.ids}
        else:
            values = self.k02.cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
            scores = dict(zip(self.ids, map(float, values)))
        ranked = self.k02.rank_candidates(candidates, scores)
        selected = ranked[:3]
        per_task = {t: [eid for eid in selected if self.cards[eid]['task_type'] == t] for t in tasks}
        status = 'zero_similarity' if not selected else ('retrieved' if all(per_task.values()) else 'partial_task_coverage')
        return {'status': status, 'cards': [copy.deepcopy(self.cards[eid]) for eid in selected],
                'ordered_ids': selected, 'ranking': [{'evidence_id': eid, 'score': scores[eid]} for eid in ranked],
                'per_task': per_task, 'graph_paths': paths, 'identity': self.identity,
                'payload_sha256': digest([self.cards[eid] for eid in selected])}

    def validate_returned(self, cards):
        ids = [c.get('evidence_id') for c in cards]
        require(len(ids) <= 3 and len(set(ids)) == len(ids), 'global unique complete-card cap')
        for c in cards:
            require(c == self.cards.get(c.get('evidence_id')), 'actual complete card must exactly match canonical; no refill/repair')
            if hasattr(self,'card_pins'):require(digest(c)==self.card_pins.get(c.get('evidence_id')),'in-memory canonical card changed')


class FixtureCorpus:
    """Explicit synthetic corpus used only for transport/state/metric tests."""
    def __init__(self, request, cards):
        require(request['question'].startswith('SYNTHETIC:'), 'fixture request must be synthetic')
        self.request = copy.deepcopy(request)
        self.cards = {c['evidence_id']: c for c in cards}
        self.scenarios = {request['scenario_id']: {'predicted_class': request['predicted_class']}}
        self.contexts = {request['scenario_id']: request['explicitly_allowed_operating_context']}
        self.calls = []

    def retrieve(self, request, mode, top_k=3):
        require(request == self.request and mode in ['text', 'graph'] and top_k == 3, 'synthetic fixture scope')
        self.calls.append(mode)
        return {'status': 'retrieved', 'cards': copy.deepcopy(list(self.cards.values())[:3]), 'ordered_ids': list(self.cards)[:3],
                'per_task': {t: list(self.cards)[:1] for t in request['task_types']}, 'ranking': [], 'graph_paths': {}, 'identity': {'synthetic': True}}

    validate_returned = Corpus.validate_returned
