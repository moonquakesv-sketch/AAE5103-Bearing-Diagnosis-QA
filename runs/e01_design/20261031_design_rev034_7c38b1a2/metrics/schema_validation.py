"""Validate this limited emitted JSON Schema vocabulary, including negative synthetic cases."""
from pathlib import Path
import json
RUN=Path(__file__).resolve().parents[1]
def validate(v,s):
    if 'anyOf' in s:
        accepted=False
        for branch in s['anyOf']:
            try:validate(v,branch);accepted=True;break
            except AssertionError:pass
        assert accepted,'no anyOf branch matches'
    if 'enum' in s: assert any(type(v)==type(e) and v==e or type(v) in (int,float) and type(e) in (int,float) and v==e for e in s['enum']),('enum',v)
    if 'type' in s:
        ts=s['type'] if isinstance(s['type'],list) else [s['type']]
        kinds={'object':type(v) is dict,'array':type(v) is list,'string':type(v) is str,
               'null':v is None,'boolean':type(v) is bool,'integer':type(v) is int,'number':type(v) in (int,float)}
        assert any(kinds[t] for t in ts),('type',v,ts)
    if isinstance(v,dict) and s.get('type')=='object':
        assert set(s.get('required',[]))<=set(v),('missing_keys',set(s.get('required',[]))-set(v))
        if s.get('additionalProperties') is False: assert set(v)<=set(s['properties']),('extra_keys',set(v)-set(s['properties']))
        for k,x in v.items():
            if k in s.get('properties',{}):validate(x,s['properties'][k])
    if isinstance(v,list) and 'items' in s:
        for x in v:validate(x,s['items'])
    if isinstance(v,list) and 'maxItems' in s:assert len(v)<=s['maxItems']
    if 'minimum' in s: assert v>=s['minimum']
def semantics(o,text,rubric_ids,frozen_half_ids=(),boundary_ids=(),actual_card_ids=(),actual_available_ids=(),actual_gap_ids=(),frozen_insuff_half_ids=()):
    assert {r['id'] for r in o['point_rows']}==set(rubric_ids)
    assert len(o['point_rows'])==len(set(rubric_ids))
    for row in o['point_rows']:
        assert row['status']!='not_applicable','fixed requested point cannot be N/A'
        assert row['half_credit_allowed']==(row['id'] in frozen_half_ids)
        assert row['score']!=.5 or row['id'] in frozen_half_ids
        assert (row['score'] is None)==(row['status'] in {'evaluator_failure','machine_unresolved','not_applicable'})
        if row['status'] in {'tested_failure','missing'} or o['tested_status'] in {'tested_failure','missing'}:assert row['score']==0
    def all_spans(v):
        if isinstance(v,dict):
            if 'channel' in v and 'start' in v and 'quote' in v:
                assert v['channel'] in {'final_answer','service_message'},'hidden draft/report in final scoring'
                value=text[v['channel']] or ''
                assert type(v['start']) is int and type(v['end']) is int
                assert 0<=v['start']<=v['end']<=len(value)
                assert value[v['start']:v['end']]==v['quote']
            for x in v.values():all_spans(x)
        elif isinstance(v,list):
            for x in v:all_spans(x)
    all_spans(o)
    ids=[c['id'] for c in o['claims']];assert len(ids)==len(set(ids))
    assert all(not c['critical_error'] or c['support']=='none' for c in o['claims'])
    pairs=[(p['claim_id'],p['evidence_id']) for p in o['citation_pairs']];assert len(pairs)==len(set(pairs))
    assert all(p['claim_id'] in ids for p in o['citation_pairs'])
    assert all(p['id_valid']==(p['evidence_id'] in actual_card_ids) for p in o['citation_pairs'])
    assert len({(c['id'],c['claim_id']) for c in o['conditions']})==len(o['conditions'])
    for c in o['conditions']:
        assert c['claim_id'] in ids
        assert c['score']!=.5 or (not c['critical'] and c['half_credit_allowed'])
        assert c['status']!='scored' or c['score'] is not None
        assert c['status']=='scored' or c['score'] is None
    for c in o['claims']:
        assert len(c['necessary_condition_ids'])==len(set(c['necessary_condition_ids']))
        assert set(c['necessary_condition_ids'])=={r['id'] for r in o['conditions'] if r['claim_id']==c['id']}
    assert {b['id'] for b in o['boundary_rows']}==set(boundary_ids)
    assert len(o['boundary_rows'])==len(set(boundary_ids))
    point={r['id']:r for r in o['point_rows']}
    assert all(b['id'] in point and all(b[k]==point[b['id']][k] for k in ['score','status','half_credit_allowed']) for b in o['boundary_rows'])
    available=o['insufficiency']['available_point_ids']
    assert len(available)==len(set(available)) and set(available)<=set(rubric_ids)
    assert set(available)==set(actual_available_ids),'available IDs differ from actual complete-card support result'
    gaps=o['insufficiency']['actual_gap_ids']
    assert len(gaps)==len(set(gaps)) and set(gaps)==set(actual_gap_ids)
    decisions=o['insufficiency']['gap_support_decisions']
    assert len({d['gap_id'] for d in decisions})==len(decisions)
    assert all(d['point_id'] in rubric_ids for d in decisions)
    assert [r['id'] for r in o['insufficiency']['rows']]==(['I01','I02','I03'] if o['insufficiency']['applicable'] else [])
    for r in o['insufficiency']['rows']:
        assert r['half_credit_allowed']==(r['id'] in frozen_insuff_half_ids)
        assert r['score']!=.5 or r['id'] in frozen_insuff_half_ids
        assert (r['score'] is None)==(r['status'] in {'evaluator_failure','machine_unresolved'})
        if r['status'] in {'tested_failure','missing'} or o['tested_status'] in {'tested_failure','missing'}:assert r['score']==0
    if o['release_adherence']==1:
        assert o['publishable_mechanical'] and o['tested_status'] not in {'tested_failure','missing'}
        assert o['evaluation_status']=='complete' and all(r['score']==1 for r in o['point_rows'])
        assert all(c['support']=='full' and not c['critical_error'] for c in o['claims'])
        assert all(c['score']==1 for c in o['conditions'])
def pair_semantics(o,text):
    draft=[x['error_id'] for x in o['draft_errors']];final=[x['error_id'] for x in o['final_errors']]
    assert len(draft)==len(set(draft)) and len(final)==len(set(final))
    issues=o['reported_issue_ids'];assert len(issues)==len(set(issues))
    m=o['report_error_mapping'];assert {x['issue_id'] for x in m}==set(issues) and len(m)==len(issues)
    assert o['report_observable'] or (not issues and not m and not o['detected_error_ids'])
    assert all(x['error_id'] is None or x['error_id'] in draft for x in m)
    assert set(o['detected_error_ids'])=={x['error_id'] for x in m if x['error_id'] is not None}
    def spans(v):
        if isinstance(v,dict):
            if {'channel','start','end','quote'}<=set(v):
                t=text[v['channel']] or '';assert 0<=v['start']<=v['end']<=len(t) and t[v['start']:v['end']]==v['quote']
            for x in v.values():spans(x)
        elif isinstance(v,list):
            for x in v:spans(x)
    spans(o)
checks=[]
schema=json.loads((RUN/'metrics/output_schema.json').read_text())
o={'schema_version':'e01.design.v3.output_coding','unit_id':'SYNTHETIC',
 'judge_identity':{'configured_model':None,'returned_model':None,'backend_identity':None,'context_id':'synthetic','prompt_sha256':'synthetic'},
 'method':'model_judged','tested_status':'reasonable_insufficiency','evaluation_status':'complete',
 'point_rows':[{'id':'p1','status':'scored','score':0,'half_credit_allowed':False,'spans':[{'channel':'service_message','start':0,'end':5,'quote':'Sorry'}],'reason':'Synthetic generic message does not answer requested fact.'}],
 'claims':[],'citation_pairs':[],'conditions':[],'boundary_rows':[],
 'insufficiency':{'applicable':True,'available_point_ids':[],'actual_gap_ids':[],'gap_support_decisions':[],
 'rows':[{'id':i,'status':'scored','score':0,'half_credit_allowed':False,'spans':[],'reason':'synthetic'} for i in ['I01','I02','I03']]},
 'full_refusal':True,'errors':[],'publishable_mechanical':True,'release_adherence':0,'high_risks':['no_body'],'failure_reason':None}
def run(obj):validate(obj,schema);semantics(obj,{'service_message':'Sorry, insufficient evidence.','final_answer':''},['p1'])
run(o);checks.append({'case':'complete_output_schema_and_spans','status':'PASS'})
for label,change in [('extra_group',lambda x:x.update(group='GV')),
 ('missing_point',lambda x:x.update(point_rows=[])),
 ('invented_quote',lambda x:x['point_rows'][0]['spans'][0].update(quote='hello')),
 ('unapproved_half',lambda x:x['point_rows'][0].update(score=.5)),
 ('bool_not_score',lambda x:x['point_rows'][0].update(score=True)),
 ('evaluator_failure_not_zero',lambda x:x['point_rows'][0].update(status='evaluator_failure',score=0)),
 ('tested_failure_not_positive',lambda x:x['point_rows'][0].update(status='tested_failure',score=1)),
 ('fixed_requested_point_not_NA',lambda x:x['point_rows'][0].update(status='not_applicable',score=None)),
 ('claim_hidden_draft',lambda x:x['claims'].append({'id':'c','proposition':'synthetic','spans':[{'channel':'draft','start':0,'end':1,'quote':'x'}],'dedup_key':'c','requires_external':False,'support':'none','support_any_of':[],'necessary_condition_ids':[],'critical_error':True,'reason':'synthetic'})),
 ('orphan_critical_half',lambda x:x['conditions'].append({'id':'k','claim_id':'orphan','critical':True,'half_credit_allowed':True,'score':.5,'status':'scored','spans':[],'reason':'synthetic'})),
 ('unknown_duplicate_available',lambda x:x['insufficiency'].update(available_point_ids=['nonexistent','nonexistent'])),
 ('insuff_test_failure_positive',lambda x:x['insufficiency']['rows'][0].update(status='tested_failure',score=1)),
 ('release_contradicts_zero_coverage',lambda x:x.update(release_adherence=1))]:
    copy=json.loads(json.dumps(o));change(copy)
    try:run(copy)
    except AssertionError:checks.append({'case':label,'status':'correctly_rejected'})
    else:raise AssertionError('negative case accepted: '+label)
pair_schema=json.loads((RUN/'metrics/pair_schema.json').read_text())
pair={'schema_version':'e01.design.v3.pair_coding','pair_id':'SYNTHETIC-PAIR','method':'model_judged',
 'evaluation_status':'complete','draft_errors':[{'error_id':'e1','kind':'unsupported','proposition':'synthetic','spans':[],'reason':'synthetic'}],
 'final_errors':[],'detected_error_ids':['e1'],'reported_issue_ids':['issue1'],
 'report_error_mapping':[{'issue_id':'issue1','error_id':'e1','reason':'synthetic'}],
 'final_observable':True,'report_observable':True,'alignment_reason':'synthetic','high_risks':[],
 'judge_identity':o['judge_identity']}
validate(pair,pair_schema);pair_semantics(pair,{})
checks.append({'case':'distinct_issue_error_ids','status':'PASS'})
bad=json.loads(json.dumps(pair));bad['report_error_mapping']=[]
try:validate(bad,pair_schema);pair_semantics(bad,{})
except AssertionError:checks.append({'case':'missing_pair_mapping','status':'correctly_rejected'})
else:raise AssertionError('missing pair mapping accepted')
(RUN/'validation/schema_synthetic.json').write_text(json.dumps({'status':'PASS','checks':checks,'validator':'local subset covering emitted vocabulary; no dependency installed; not full JSON Schema meta-validator'},ensure_ascii=False,indent=2)+'\n')
print('PASS schema synthetic cases='+str(len(checks)))
