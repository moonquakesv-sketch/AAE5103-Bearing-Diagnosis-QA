"""Pinned I01 corpus loader and thin adapter over the unchanged K02 algorithms.

The manifest is pinned by the caller's external release. Source files are checked
for provenance and excerpt location, but only K02's two evidence fields are fit.
"""
import copy
import csv
import hashlib
import io
from pathlib import Path
import re

import retrieve_k02 as k02

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'i01.retriever.v1_rev025'
KNOWLEDGE_VERSION = 'i01.knowledge.v1_rev025'
INTERFACE_VERSION = 'i01.knowledge.v1'
DEFAULT_KNOWLEDGE = ROOT / 'knowledge/i01/v1'
# Filled with the final independently released bundle, never computed at load.
PINNED_MANIFEST_SHA256 = None
SCENARIOS = {
    'outer_race': 'SC-K01-OR-001',
    'inner_race': 'SC-I01-IR-001',
    'rolling_element': 'SC-I01-RE-001',
    'normal': 'SC-I01-NORMAL-001',
}
SCENARIO_IDS = SCENARIOS
SHARED_EVIDENCE_IDS = frozenset({'EV-K01-009', 'EV-K01-010', 'EV-K01-011'})
Error = k02.K02Error


def require(condition, reason, code='knowledge_structure'):
    k02.require(condition, code, reason)


def _safe_file(project_root, relative):
    require(type(relative) is str and bool(relative), 'Empty registered file path.', 'knowledge_file')
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts and path.as_posix() == relative,
            'Unsafe registered file path: ' + relative, 'knowledge_file')
    path = project_root / path
    for parent in reversed((path, *path.parents)):
        require(not parent.is_symlink(), 'Registered paths cannot contain links: ' + relative, 'knowledge_file')
    require(path.is_file(), 'Registered regular file missing: ' + relative, 'knowledge_file')
    return path


def _sha(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _load_bundle(knowledge_dir=None, *, manifest_sha256=None, project_root=ROOT):
    project_root = Path(project_root).absolute()
    folder = Path(knowledge_dir or project_root / 'knowledge/i01/v1').absolute()
    require(folder.is_relative_to(project_root), 'Knowledge directory must be inside the released project.', 'knowledge_file')
    prefix = folder.relative_to(project_root).as_posix()
    expected = manifest_sha256 if manifest_sha256 is not None else PINNED_MANIFEST_SHA256
    require(_sha(expected), 'An external release must pin the manifest SHA-256.', 'manifest_binding_required')
    manifest_bytes = _safe_file(project_root, prefix + '/manifest.json').read_bytes()
    actual_manifest = hashlib.sha256(manifest_bytes).hexdigest()
    require(actual_manifest == expected, 'Manifest differs from the external release pin.', 'knowledge_hash_drift')
    manifest = k02.strict_json(manifest_bytes)
    require(type(manifest) is dict and manifest.get('knowledge_version', manifest.get('version')) == KNOWLEDGE_VERSION,
            'Unexpected knowledge manifest version.')
    require(manifest.get('interface_version') == INTERFACE_VERSION, 'Unexpected manifest interface version.')
    files = manifest.get('files')
    require(type(files) is dict and bool(files), 'Manifest files must be a nonempty hash map.')
    required = {prefix + '/' + name + '.csv' for name in k02.k01.NAMES} | {prefix + '/interface.json'}
    require(required <= set(files) and prefix + '/manifest.json' not in files, 'Missing tables/interface or self-referential manifest.')
    blobs = {}
    for relative, expected_hash in files.items():
        require(_sha(expected_hash), 'Invalid file hash: ' + str(relative))
        blob = _safe_file(project_root, relative).read_bytes()
        require(hashlib.sha256(blob).hexdigest() == expected_hash,
                'Released knowledge/source bytes changed: ' + relative, 'knowledge_hash_drift')
        blobs[relative] = blob
    tables = {}
    for name in k02.k01.NAMES:
        reader = csv.DictReader(io.StringIO(blobs[prefix + '/' + name + '.csv'].decode('utf-8'), newline=''))
        require(reader.fieldnames == k02.k01.FIELDS[name].split(), name + ': unexpected fields.', 'knowledge_schema')
        tables[name] = list(reader)
    interface = k02.strict_json(blobs[prefix + '/interface.json'])
    excerpt_index = None
    if manifest.get('distribution_profile') == 'excerpt-only-v1':
        index_path = manifest.get('source_excerpt_index')
        require(index_path in files, 'Submission source excerpt index is not pinned.')
        excerpt_index = k02.strict_json(blobs[index_path])
        require(excerpt_index.get('profile') == 'excerpt-only-v1', 'Unexpected source excerpt profile.')
    summary = _validate(tables, interface, blobs, files, excerpt_index=excerpt_index)
    summary.update(knowledge_version=KNOWLEDGE_VERSION, interface_version=INTERFACE_VERSION,
                   manifest_sha256=actual_manifest, file_hashes=copy.deepcopy(files),
                   manifest_binding='external release SHA-256; manifest does not authenticate itself')
    return tables, interface, manifest, summary


def _validate(tables, interface, blobs, files, *, excerpt_index=None):
    indexes = {}
    for name, rows in tables.items():
        require(bool(rows), name + ': empty table.')
        fields = set(k02.k01.FIELDS[name].split())
        for row in rows:
            require(set(row) == fields and all(type(v) is str for v in row.values()), name + ': malformed row.')
            require(all(v.strip() for f, v in row.items() if f not in k02.k01.OPTIONAL.get(name, set())),
                    name + ': missing required field.')
        ids = [r[k02.k01.KEYS[name]] for r in rows]
        require(len(ids) == len(set(ids)), name + ': duplicate ID.')
        indexes[name] = dict(zip(ids, rows))
    src, scenarios, evidence, entities, relations = (indexes[n] for n in k02.k01.NAMES)
    require(type(interface) is dict and interface.get('schema_version') == INTERFACE_VERSION
            and interface.get('knowledge_version') == KNOWLEDGE_VERSION, 'Interface version mismatch.')
    require(set(interface.get('tables', [])) == {n + '.csv' for n in k02.k01.NAMES}
            and set(interface.get('allowed_query_inputs', [])) == k02.INPUT_FIELDS
            and set(interface.get('forbidden_input_categories', [])) == k02.k01.FORBIDDEN_INPUTS,
            'Interface table or business-input contract mismatch.')
    require(interface.get('claim_en_semantics') == k02.k01.CLAIM_SEMANTICS
            and interface.get('assumptions_are_not_evidence') is True
            and interface.get('audit_material_is_retrieval_input') is False,
            'Interface evidence/assumption separation mismatch.')
    require(set(scenarios) == set(SCENARIOS.values()), 'Exactly the four registered scenarios are required.')
    local_text, deferred = {}, []
    for sid, source in src.items():
        require(source['original_url'].startswith('https://') and source['format'] in {'html', 'pdf'}, sid + ': source locator.')
        if source['local_path']:
            relative = source['local_path']
            require(source['format'] == 'html', sid + ': only local HTML source checking is registered.')
            if excerpt_index is None:
                require(relative in files and files[relative] == source['sha256'], sid + ': unpinned source asset.')
                parser = k02.k01.VisibleHTML()
                parser.feed(blobs[relative].decode('utf-8'))
                local_text[sid] = k02.k01.normalize(' '.join(parser.parts))
            else:
                exported = excerpt_index.get('sources', {}).get(sid, {})
                require(exported.get('snapshot_sha256') == source['sha256']
                        and exported.get('original_url') == source['original_url'],
                        sid + ': source excerpt provenance mismatch.')
                fragments = exported.get('excerpt_fragments')
                require(type(fragments) is list and fragments
                        and all(type(x) is str and x.strip() for x in fragments),
                        sid + ': missing exported excerpts.')
                local_text[sid] = k02.k01.normalize(' '.join(fragments))
        else:
            require(not source['sha256'] and source['format'] == 'pdf'
                    and source['local_copy_status'] == 'not_saved_http_403', sid + ': missing local source must be explicit.')
            deferred.append(sid)
    knowledge_ids = [e['knowledge_id'] for e in evidence.values()]
    require(len(knowledge_ids) == len(set(knowledge_ids)), 'Duplicate evidence knowledge ID.')
    knowledge_to_evidence = dict(zip(knowledge_ids, evidence))
    checked_quotes = []
    for eid, card in evidence.items():
        require(card['source_id'] in src and card['task_type'] in k02.TASKS, eid + ': missing source or invalid task.')
        source = src[card['source_id']]
        require(card['source_url'] == source['original_url'] and card['source_version'] == source['version_or_date'],
                eid + ': source metadata mismatch.')
        require(card['modality'] in k02.k01.MODALITIES and card['applicability'] == 'generic_bearing_education_only',
                eid + ': invalid modality or scope.')
        require(card['knowledge_id'] in entities and entities[card['knowledge_id']]['entity_type'] == 'knowledge'
                and entities[card['knowledge_id']]['label'] == card['claim_en'], eid + ': knowledge entity mismatch.')
        require(eid in entities and entities[eid]['entity_type'] == 'evidence', eid + ': evidence entity mismatch.')
        if source['format'] == 'pdf':
            require(card['pdf_file_page'].isdigit() and int(card['pdf_file_page']) >= 1 and card['printed_page'].isdigit(),
                    eid + ': invalid PDF locator.')
        else:
            require(not card['pdf_file_page'] and not card['printed_page'], eid + ': HTML cannot claim PDF pages.')
        if card['source_id'] in local_text:
            require(all(k02.k01.normalize(fragment) in local_text[card['source_id']]
                        for fragment in card['original_text'].split('; ')), eid + ': excerpt absent from pinned source.')
            checked_quotes.append(eid)
    types = {'part', 'scenario', 'assumption', 'mechanism', 'task', 'knowledge', 'evidence'}
    for entity_id, entity in entities.items():
        kind, ref = entity['entity_type'], entity['content_ref']
        require(kind in types, entity_id + ': unknown entity type.')
        if kind in {'knowledge', 'evidence', 'mechanism'}:
            require(ref.startswith('evidence.csv#') and ref.split('#', 1)[1] in evidence, entity_id + ': broken evidence content_ref.')
            if kind in {'knowledge', 'evidence'}:
                expected = knowledge_to_evidence.get(entity_id) if kind == 'knowledge' else entity_id
                require(ref == 'evidence.csv#' + str(expected), entity_id + ': wrong evidence mirror.')
        elif kind in {'scenario', 'assumption'}:
            require(ref.startswith('scenarios.csv#') and ref.split('#', 1)[1] in scenarios, entity_id + ': broken scenario content_ref.')
        elif kind == 'task':
            require(entity['label'] in k02.TASKS and ref == 'project_task_contract', entity_id + ': task contract mismatch.')
        else:
            require(ref == 'project_class_contract', entity_id + ': class contract mismatch.')
    task_owners, routes, task_links, support_links, edge_keys = {}, {}, {}, set(), set()
    for rid, edge in relations.items():
        a, b, relation, eid = (edge[f] for f in ('subject_id', 'object_id', 'relation', 'evidence_id'))
        require(a in entities and b in entities, rid + ': dangling entity.')
        key = (a, relation, b, eid)
        require(key not in edge_keys, rid + ': duplicate relation.')
        edge_keys.add(key)
        if relation in {'routes_from_class', 'has_task', 'assumes_for_demo'}:
            basis = 'demo_assumption' if relation == 'assumes_for_demo' else 'project_contract'
            require(a in scenarios and edge['assertion_basis'] == basis and not eid, rid + ': invalid project/assumption relation.')
            target_type = {'routes_from_class': 'part', 'has_task': 'task', 'assumes_for_demo': 'assumption'}[relation]
            require(entities[b]['entity_type'] == target_type, rid + ': wrong route target type.')
            if relation == 'routes_from_class':
                routes.setdefault(a, []).append(b)
            elif relation == 'has_task':
                require(b not in task_owners, rid + ': task nodes must belong to one scenario.')
                task_owners[b] = a
            else:
                require(entities[b]['content_ref'] == 'scenarios.csv#' + a, rid + ': cross-scenario assumption.')
            continue
        require(edge['assertion_basis'] == 'source_knowledge' and eid in evidence, rid + ': missing source evidence.')
        card = evidence[eid]
        require(edge['conditions'] == card['conditions'] and edge['limitations'] == card['limitations'], rid + ': lost card conditions/limitations.')
        if relation == 'has_evidence_backed_content':
            require(entities[a]['entity_type'] == 'task' and entities[a]['label'] == card['task_type']
                    and b == card['knowledge_id'], rid + ': task/knowledge/evidence mismatch.')
            task_links.setdefault(eid, []).append(a)
        elif relation == 'supported_by':
            require(a == card['knowledge_id'] and b == eid, rid + ': wrong support edge.')
            support_links.add(eid)
        else:
            allowed = {'has_reported_example': ('part', 'mechanism'), 'is_possible_cause_of': ('assumption', 'mechanism')}
            require(relation in allowed and (entities[a]['entity_type'], entities[b]['entity_type']) == allowed[relation],
                    rid + ': invalid source relation.')
    require(set(task_links) == support_links == set(evidence), 'Unlinked evidence or missing source support.')
    require(set(task_owners) == {i for i, e in entities.items() if e['entity_type'] == 'task'}, 'Unowned task node.')
    coverage = {}
    for cls, scenario_id in SCENARIOS.items():
        scenario = scenarios[scenario_id]
        require(scenario['predicted_class'] == cls and scenario['class_semantics'] == 'routing_category_only_not_an_actual_prediction',
                scenario_id + ': class/scenario mismatch.')
        require(scenario_id in entities and entities[scenario_id]['entity_type'] == 'scenario'
                and entities[scenario_id]['content_ref'] == 'scenarios.csv#' + scenario_id, scenario_id + ': scenario mirror mismatch.')
        require(len(routes.get(scenario_id, [])) == 1
                and entities[routes[scenario_id][0]]['label'].split(' / ')[0] == cls, scenario_id + ': class route mismatch.')
        owned_tasks = [t for t, owner in task_owners.items() if owner == scenario_id]
        require(len(owned_tasks) == 3 and {entities[t]['label'] for t in owned_tasks} == set(k02.TASKS)
                and set(scenario['task_types'].split('|')) == set(k02.TASKS), scenario_id + ': task coverage mismatch.')
        paths = k02.graph_paths(tables['relations'], tables['entities'], cls, scenario_id, k02.TASKS)
        require(set(scenario['supporting_evidence_ids'].split('|')) <= set(paths), scenario_id + ': supporting evidence is not reachable.')
        coverage[cls] = {task: sorted(e for e in paths if evidence[e]['task_type'] == task) for task in k02.TASKS}
        require(all(coverage[cls].values()), scenario_id + ': missing graph task evidence.')
    for eid, task_ids in task_links.items():
        require(all(t in task_owners for t in task_ids), eid + ': evidence linked from unowned task.')
        owners = {task_owners[t] for t in task_ids}
        require(len(owners) == 1 or eid in SHARED_EVIDENCE_IDS, eid + ': class-specific evidence crosses scenarios.')
    normal = scenarios[SCENARIOS['normal']]
    require(normal['assumed_cause'] == 'none_no_fault_cause_assumed', 'Normal must not assume a fault cause.')
    return {'status': 'PASS', 'counts': {n: len(tables[n]) for n in tables}, 'graph_task_candidates': coverage,
            'local_excerpt_presence_checks': checked_quotes if excerpt_index is None else [],
            'exported_excerpt_identity_checks': checked_quotes if excerpt_index is not None else [],
            'source_distribution_profile': 'excerpt-only-v1' if excerpt_index is not None else 'full-local-snapshots',
            'online_only_source_checks_deferred_to_manual': deferred,
            'semantics': ('Structure and pinned excerpt identity only; original webpage verification was performed '
                          'when the distribution was built, not by this offline loader.' if excerpt_index is not None
                          else 'Structure, pinned bytes and excerpt presence only; no automatic truth or safety finding.')}


def validate_bundle(knowledge_dir=None, *, manifest_sha256=None, project_root=ROOT):
    return _load_bundle(knowledge_dir, manifest_sha256=manifest_sha256, project_root=project_root)[3]


class Retriever(k02.Retriever):
    """New pinned corpus; inherited K02 retrieve, graph traversal and ranking."""
    def __init__(self, knowledge_dir=None, *, manifest_sha256=None, project_root=ROOT):
        self.tables, self.interface, self.manifest, self.validation = _load_bundle(
            knowledge_dir, manifest_sha256=manifest_sha256, project_root=project_root)
        self.knowledge_hashes = copy.deepcopy(self.validation['file_hashes'])
        self.config = copy.deepcopy(k02.FIXED_CONFIG)
        self.config.update(implementation_version=VERSION, knowledge_version=KNOWLEDGE_VERSION,
                           knowledge_hashes=self.knowledge_hashes)
        self.config_file_sha256 = None
        self.evidence = {e['evidence_id']: e for e in self.tables['evidence']}
        self.sources = {s['source_id']: s for s in self.tables['sources']}
        self.scenarios = {s['scenario_id']: s for s in self.tables['scenarios']}
        self.ids = sorted(self.evidence)
        self.documents = ['\n'.join(self.evidence[eid][f] for f in self.config['index_fields']) for eid in self.ids]
        params = dict(self.config['tfidf'], ngram_range=tuple(self.config['tfidf']['ngram_range']))
        self.vectorizer = k02.TfidfVectorizer(**params)
        self.matrix = self.vectorizer.fit_transform(self.documents)
        self.index_id = k02.digest(self.index_snapshot())

    @property
    def identity(self):
        return {'implementation_version': VERSION, 'knowledge_version': KNOWLEDGE_VERSION,
                'interface_version': INTERFACE_VERSION, 'manifest_sha256': self.validation['manifest_sha256'],
                'knowledge_hashes': copy.deepcopy(self.knowledge_hashes), 'index_id': self.index_id,
                'configuration_id': k02.digest(self.config), 'configuration_kind': 'inline fixed adapter configuration',
                'algorithm_version': k02.VERSION}

    def index_snapshot(self):
        value = super().index_snapshot()
        value.update(knowledge_version=KNOWLEDGE_VERSION, knowledge_hashes=copy.deepcopy(self.knowledge_hashes),
                     manifest_sha256=self.validation['manifest_sha256'], algorithm_version=k02.VERSION)
        return value

    def retrieve(self, request, mode='text', top_k=3):
        result = super().retrieve(request, mode=mode, top_k=top_k)
        result.update(implementation_version=VERSION, knowledge_version=KNOWLEDGE_VERSION,
                      interface_version=INTERFACE_VERSION, manifest_sha256=self.validation['manifest_sha256'],
                      algorithm_version=k02.VERSION, configuration_kind='inline fixed adapter configuration')
        result['notices'] = [
            'Retrieval and caller-supplied class do not establish source truth, cause, answer sufficiency or semantic verification.',
            'The I01 corpus uses unchanged K02 TF-IDF scoring, graph traversal, ranking and global complete-card budget.',
            'Graph candidates follow explicit scenario-specific task nodes; text mode searches the whole pinned evidence corpus.',
            'Scenario causes are independent educational assumptions; a location class cannot establish a mechanism or root cause.',
            'Normal has no assumed fault cause and does not establish absence of damage, serviceability or safety.',
            'Every returned card retains its source, modality, conditions, limitations and locators; sources are not indexed.',
            'Physical inspection guidance is not a CWT image criterion or an authorization to disassemble equipment.',
            'Timken provenance still lacks same-version local PDF/hash and visual checks; this inherited limitation remains explicit.',
            'No model-specific or aviation parameters, reuse, replacement or release decisions are produced.',
        ]
        return result
