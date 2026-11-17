"""K01 structure/reference/byte checks. This does NOT decide semantic truth.

Read only knowledge tables and explicitly registered source files. It never imports
or invokes D01-D04 code, models, feature readers or image decoders.
"""
import argparse
import copy
import csv
import hashlib
import json
import re
from pathlib import Path
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_VERSION = '20261023_rev011'
CLAIM_SEMANTICS = ('Project-agent-authored English paraphrase, not a verbatim manufacturer quotation; '
                   'no human or domain-expert sign-off is claimed.')
NAMES = ('sources', 'scenarios', 'evidence', 'entities', 'relations')
KEYS = dict(sources='source_id', scenarios='scenario_id', evidence='evidence_id',
            entities='entity_id', relations='relation_id')
FIELDS = {
 'sources':'source_id title organization_author version_or_date original_url access_date timezone format page_or_section applicability full_text_status local_path sha256 local_copy_status access_limitations rights_use reviewed_by',
 'scenarios':'scenario_id predicted_class class_semantics assumed_cause assumption_text known_conditions unknown_information applicability cannot_infer supporting_evidence_ids support_meaning task_types status',
 'evidence':'evidence_id source_id knowledge_id task_type source_url source_version pdf_file_page printed_page page_or_section original_text chinese_paraphrase claim_en conditions limitations modality applicability reviewed_by',
 'entities':'entity_id entity_type label content_ref',
 'relations':'relation_id subject_id relation object_id assertion_basis evidence_id conditions limitations semantic_note',
}
OPTIONAL = {'sources':{'local_path','sha256'},'evidence':{'pdf_file_page','printed_page'},'relations':{'evidence_id'}}
TASKS={'explanation','inspection','preparation'}
MODALITIES={'reported_example','possible_cause','alternative_possibilities','inspection_cue',
            'preparation_guidance','inspection_guidance','scope_constraint','sequence_constraint'}
ALLOWED_INPUTS={'predicted_class','scenario_id','question','task_types','explicitly_allowed_operating_context'}
FORBIDDEN_INPUTS={'ground_truth','sample_file_names','sample_paths','answer_bearing_notes','evaluation_answers','scoring_rubrics'}
MECHANISM_EVIDENCE={'MECH-K01-FB':'EV-K01-002'}

class Invalid(ValueError): pass

def require(condition, message):
    if not condition: raise Invalid(message)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def load(folder):
    tables={}
    for name in NAMES:
        with (folder/(name+'.csv')).open(encoding='utf-8',newline='') as stream:
            reader=csv.DictReader(stream)
            require(reader.fieldnames == FIELDS[name].split(), f'{name}: schema/header mismatch')
            tables[name]=list(reader)
    return tables

class VisibleHTML(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]; self.hidden=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'): self.hidden+=1
    def handle_endtag(self,tag):
        if tag in ('script','style'): self.hidden-=1
    def handle_data(self,data):
        if not self.hidden: self.parts.append(data)

def normalize(text): return ' '.join(text.split())

def validate_interface(interface):
    """Validate the declared K01 contract, not future runtime input filtering."""
    require(isinstance(interface,dict),'interface: expected object')
    expected_keys={'schema_version','knowledge_version','implementation_status','tables',
        'allowed_query_inputs','forbidden_input_categories','retrieval_corpus_policy',
        'claim_en_semantics','assumptions_are_not_evidence','audit_material_is_retrieval_input',
        'scope','separator_for_id_lists','encoding','pdf_file_page_base','null_policy'}
    require(set(interface)==expected_keys,'interface: missing or unexpected fields')
    for field,expected in [('tables',{n+'.csv' for n in NAMES}),
                           ('allowed_query_inputs',ALLOWED_INPUTS),
                           ('forbidden_input_categories',FORBIDDEN_INPUTS)]:
        values=interface[field]
        require(isinstance(values,list) and all(isinstance(v,str) for v in values),f'interface: {field} must be a string list')
        require(len(values)==len(set(values)),f'interface: duplicate {field}')
        require(set(values)==expected,f'interface: {field} contract mismatch')
    require(not set(interface['allowed_query_inputs']) & set(interface['forbidden_input_categories']),
            'interface: allowed/forbidden conflict')
    expected_values={'schema_version':'k01.v1','knowledge_version':KNOWLEDGE_VERSION,
        'implementation_status':'tables_only_no_retrieval','claim_en_semantics':CLAIM_SEMANTICS,
        'assumptions_are_not_evidence':True,'audit_material_is_retrieval_input':False,
        'scope':'generic bearing education; no model-specific or aviation validation',
        'separator_for_id_lists':'|','encoding':'UTF-8','pdf_file_page_base':1}
    for field,expected in expected_values.items():
        require(type(interface[field]) is type(expected) and interface[field]==expected,
                f'interface: {field} mismatch')
    for field in ('retrieval_corpus_policy','null_policy'):
        require(isinstance(interface[field],str) and bool(interface[field].strip()),f'interface: empty {field}')

def validate(tables, root=ROOT, check_files=True, interface=None):
    require(set(tables)==set(NAMES),'missing or unexpected tables')
    indexes={}
    for name,rows in tables.items():
        require(rows, f'{name}: empty table')
        ids=[]
        for row in rows:
            require(set(row)==set(FIELDS[name].split()),f'{name}: wrong row fields')
            for field,value in row.items():
                require(isinstance(value,str), f'{name}.{field}: not a string')
                if field not in OPTIONAL.get(name,set()):
                    require(bool(value.strip()),f'{name}.{field}: required field empty')
            ids.append(row[KEYS[name]])
        require(len(ids)==len(set(ids)), f'{name}: duplicate ID')
        indexes[name]=dict(zip(ids,rows))
    src,sc,ev,en,rel=(indexes[n] for n in NAMES)
    patterns={'sources':r'SRC-K01-\d{2}', 'scenarios':r'SC-K01-OR-\d{3}',
              'evidence':r'EV-K01-\d{3}', 'relations':r'REL-K01-\d{3}'}
    for table,pattern in patterns.items():
        require(all(re.fullmatch(pattern,i) for i in indexes[table]),f'{table}: malformed ID')
    local_text={}; source_file_checks=[]; online_only=[]
    for sid,s in src.items():
        require(s['original_url'].startswith('https://'),f'{sid}: URL required')
        require(re.fullmatch(r'\d{4}-\d{2}-\d{2}',s['access_date']),f'{sid}: access date')
        require(s['format'] in ('html','pdf'),f'{sid}: source format')
        if s['local_copy_status']=='not_saved_http_403':
            require(not s['local_path'] and not s['sha256'],f'{sid}: unavailable original cannot have fabricated path/hash')
            require(s['format']=='pdf' and s['full_text_status']=='full_pdf_available_online_selected_pages_read',f'{sid}: online source state')
            online_only.append(sid)
        else:
            require(s['local_copy_status']=='saved_public_html',f'{sid}: unknown copy state')
            p=Path(s['local_path'])
            require(not p.is_absolute() and '..' not in p.parts and p.parts[:3]==('knowledge','k01','sources'),f'{sid}: unsafe source path')
            require(re.fullmatch('[0-9a-f]{64}',s['sha256']),f'{sid}: missing hash')
            if check_files:
                require((root/p).is_file(),f'{sid}: missing original')
                require(sha(root/p)==s['sha256'],f'{sid}: source hash mismatch')
                parser=VisibleHTML();parser.feed((root/p).read_text(encoding='utf-8'))
                local_text[sid]=normalize(' '.join(parser.parts))
                source_file_checks.append(sid)
    require(len(sc)==1,'K01 must have exactly one scenario')
    for sid,s in sc.items():
        require(s['predicted_class']=='outer_race',f'{sid}: chosen category')
        require(s['class_semantics']=='routing_category_only_not_an_actual_prediction',f'{sid}: prediction semantics')
        require(set(s['task_types'].split('|'))==TASKS,f'{sid}: tasks')
        for eid in s['supporting_evidence_ids'].split('|'):
            require(eid in ev,f'{sid}: missing supporting evidence {eid}')
        require(sid in en and en[sid]['entity_type']=='scenario',f'{sid}: scenario entity')
    counts={task:0 for task in TASKS}; words={sid:0 for sid in src}; exact_quotes=[]
    for eid,e in ev.items():
        require(e['source_id'] in src,f'{eid}: missing source')
        s=src[e['source_id']]
        require(e['source_url']==s['original_url'] and e['source_version']==s['version_or_date'],f'{eid}: source metadata mismatch')
        require(e['task_type'] in TASKS,f'{eid}: unknown task')
        require(e['modality'] in MODALITIES,f'{eid}: unknown modality')
        require(e['applicability']=='generic_bearing_education_only',f'{eid}: unsupported scope enum')
        require(e['knowledge_id'] in en and en[e['knowledge_id']]['entity_type']=='knowledge',f'{eid}: missing knowledge entity')
        require(eid in en and en[eid]['entity_type']=='evidence',f'{eid}: missing evidence entity')
        require(en[e['knowledge_id']]['label']==e['claim_en'],f'{eid}: knowledge label drift')
        if s['format']=='pdf':
            require(e['pdf_file_page'].isdigit() and 1<=int(e['pdf_file_page'])<=24,f'{eid}: invalid PDF file page')
            require(e['printed_page'].isdigit(),f'{eid}: missing printed page')
        else:
            require(not e['pdf_file_page'] and not e['printed_page'],f'{eid}: HTML cannot claim PDF page')
        require(bool(re.search('[A-Za-z]',e['original_text'])),f'{eid}: no English excerpt')
        # Semicolon separates explicitly non-contiguous short quotations.
        fragments=e['original_text'].split('; ')
        words[e['source_id']]+=sum(len(f.split()) for f in fragments)
        if check_files and e['source_id'] in local_text:
            require(all(normalize(f) in local_text[e['source_id']] for f in fragments),f'{eid}: excerpt absent from local original')
            exact_quotes.append(eid)
        counts[e['task_type']]+=1
    require(all(counts.values()),'missing task coverage')
    require(all(n<=25 for n in words.values()),f'quotation budget exceeded: {words}')
    require(len({e['knowledge_id'] for e in ev.values()})==len(ev),'duplicate knowledge ID')
    knowledge_evidence={e['knowledge_id']:eid for eid,e in ev.items()}
    expected_entities={'PART-OUTER-RACE','SC-K01-OR-001','ASM-K01-001','MECH-K01-FB'} | {'TASK-'+t.upper() for t in TASKS} | set(ev) | {e['knowledge_id'] for e in ev.values()}
    require(set(en)==expected_entities,'unexpected/missing entity ID')
    for i,e in en.items():
        require(e['entity_type'] in {'part','scenario','assumption','mechanism','task','knowledge','evidence'},f'{i}: entity type')
        if e['entity_type'] in {'knowledge','evidence','mechanism'}:
            mappings={'knowledge':knowledge_evidence,'evidence':{eid:eid for eid in ev},
                      'mechanism':MECHANISM_EVIDENCE}
            expected=mappings[e['entity_type']].get(i)
            require(expected in ev and e['content_ref']=='evidence.csv#'+expected,
                    f'{i}: incorrect {e["entity_type"]} content_ref mapping')
        elif e['entity_type'] in {'scenario','assumption'}:
            require(e['content_ref']=='scenarios.csv#SC-K01-OR-001',f'{i}: scenario content_ref')
        elif e['entity_type']=='task':
            require(e['label'] in TASKS and i=='TASK-'+e['label'].upper() and e['content_ref']=='project_task_contract',f'{i}: task entity mismatch')
        else:
            require(i=='PART-OUTER-RACE' and e['content_ref']=='project_class_contract',f'{i}: part entity')
    expected_project={('SC-K01-OR-001','routes_from_class','PART-OUTER-RACE','project_contract'),
        ('SC-K01-OR-001','assumes_for_demo','ASM-K01-001','demo_assumption')} | {
        ('SC-K01-OR-001','has_task','TASK-'+t.upper(),'project_contract') for t in TASKS}
    found_project=set(); task_edges=set(); support_edges=set(); special=set(); edge_keys=[]
    for rid,r in rel.items():
        a,b=r['subject_id'],r['object_id'];pred=r['relation'];eid=r['evidence_id']
        require(a in en and b in en,f'{rid}: dangling entity')
        edge_keys.append((a,pred,b,eid))
        if r['assertion_basis']!='source_knowledge':
            key=(a,pred,b,r['assertion_basis'])
            require(key in expected_project and not eid,f'{rid}: assumption/project relation misrepresented as source fact')
            found_project.add(key);continue
        require(eid in ev,f'{rid}: missing evidence reference')
        e=ev[eid]
        require(r['conditions']==e['conditions'] and r['limitations']==e['limitations'],f'{rid}: lost or changed conditions/limitations')
        if pred=='has_evidence_backed_content':
            require(a=='TASK-'+e['task_type'].upper() and b==e['knowledge_id'],f'{rid}: task/knowledge/evidence mismatch')
            task_edges.add(eid)
        elif pred=='supported_by':
            require(a==e['knowledge_id'] and b==eid,f'{rid}: incorrect support edge')
            support_edges.add(eid)
        else:
            key=(a,pred,b,eid)
            require(key in {('PART-OUTER-RACE','has_reported_example','MECH-K01-FB','EV-K01-001'),
                ('ASM-K01-001','is_possible_cause_of','MECH-K01-FB','EV-K01-002')},f'{rid}: invalid knowledge relation')
            special.add(key)
    require(len(edge_keys)==len(set(edge_keys)),'duplicate relation')
    require(found_project==expected_project,'missing project routing/assumption edge')
    require(task_edges==support_edges==set(ev),'unlinked evidence or missing knowledge support')
    require(len(special)==2,'missing conditional mechanism relation')
    if interface is None and check_files:
        interface=json.loads((root/'knowledge/k01/interface.json').read_text(encoding='utf-8'))
    if interface is not None:
        validate_interface(interface)
    # The contract cannot detect hidden answers or unsupported claims inside free text.
    return dict(status='PASS',counts={n:len(tables[n]) for n in NAMES},task_coverage_counts=counts,
        quote_word_counts_by_source=words,local_source_hash_checks=source_file_checks,
        local_excerpt_presence_checks=exact_quotes,online_only_source_checks_deferred_to_manual=online_only,
        interface_checked=interface is not None,
        semantics='NOT_AUTOMATICALLY_VERIFIED; require manual_review.csv and relation_review.csv',
        scope='structure, referential integrity, declared enums, literal excerpt presence and registered source bytes only')

def counterexamples(original):
    def bad(t,table,row,field,value): t[table][row][field]=value
    cases=[
      ('missing_evidence_ref',lambda t:bad(t,'relations',5,'evidence_id','')),
      ('wrong_evidence_id',lambda t:bad(t,'relations',5,'evidence_id','EV-K01-999')),
      ('dangling_entity',lambda t:bad(t,'relations',5,'object_id','MECH-MISSING')),
      ('duplicate_id',lambda t:t['evidence'].append(copy.deepcopy(t['evidence'][0]))),
      ('missing_required_condition',lambda t:bad(t,'evidence',0,'conditions','')),
      ('wrong_source_id',lambda t:bad(t,'evidence',0,'source_id','SRC-K01-99')),
      ('wrong_task_evidence',lambda t:bad(t,'relations',7,'evidence_id','EV-K01-004')),
      ('assumption_as_fact',lambda t:bad(t,'relations',1,'assertion_basis','source_knowledge')),
      ('lost_condition_on_edge',lambda t:bad(t,'relations',5,'conditions','condition removed')),
      ('bad_pdf_page',lambda t:bad(t,'evidence',3,'pdf_file_page','0')),
      ('missing_task_coverage',lambda t:t['evidence'].__setitem__(slice(None),[e for e in t['evidence'] if e['task_type']!='preparation'])),
      ('fabricated_download_hash',lambda t:bad(t,'sources',1,'sha256','0'*64)),
      ('forbidden_extra_field',lambda t:bad(t,'scenarios',0,'true_label','outer_race')),
    ]
    results=[]
    for name,mutate in cases:
        fixture=copy.deepcopy(original);mutate(fixture)
        try:validate(fixture,check_files=False)
        except Invalid as error:results.append(dict(case=name,rejected=True,reason=str(error)))
        else:raise Invalid('counterexample unexpectedly accepted: '+name)
    return results

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--knowledge',type=Path,default=ROOT/'knowledge/k01')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    tables=load(args.knowledge)
    interface_path=args.knowledge/'interface.json'
    result=validate(tables,interface=json.loads(interface_path.read_text(encoding='utf-8')))
    result['isolated_counterexamples']=counterexamples(tables)
    result['counterexample_count']=len(result['isolated_counterexamples'])
    # Bind any later human-readable review to exact table bytes, not merely its PASS string.
    result['validated_table_hashes']={n+'.csv':sha(args.knowledge/(n+'.csv')) for n in NAMES}
    result['validated_interface_sha256']=sha(interface_path)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
