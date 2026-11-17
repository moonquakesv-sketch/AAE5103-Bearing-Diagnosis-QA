"""Offline regressions over actual new modules. Only SYNTHETIC queries and responses."""
from pathlib import Path
import copy
import importlib.util
import json
import os
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

R=Path(__file__).resolve().parents[1];ROOT=R.parents[2]
sys.path[:0]=[str(R/'runtime'),str(R/'tests')]
from common import Rejected,read_json,write_new,canonical,digest,file_hash,safe_path
from corpus import Corpus,FixtureCorpus
from provider import Counter,Ledger,Provider,LiveGate,LiveHTTP
from engine import Runtime,final_packet,pair_packet,expected_gaps,available_components
from evaluation import validate_output,validate_pair
from cli import execute_batch
from reporting import output_metrics,pair_metrics,bounded,aggregate
from fixtures import REQUEST,CARDS,RUBRIC,FixtureCounter,FixtureHTTP,output_rating,pair_rating,candidate


def config():
 c=read_json(R/'config/execution.json')
 for a,role in c['actions'].items():
  role['max_attempts']//=13;role['input_cap']=512;role['output_cap']=128
 c['scope']=dict(c['scope'],stage='synthetic_validation',purpose='SYNTHETIC fixtures only; no E02',Q=1,question_ids=['SYN-Q1'],splits={'debug':1,'operational_reserve':0})
 rows,slots=schedule()
 c['allowed_slots_sha256']=digest(slots)
 c['slot_bindings']={slot:{'business_sha256':digest(REQUEST),'flow_key':m['cell_key']} for m in rows for slot in slots
                    if slot.startswith(m['cell_key']+'_') or slot.startswith(m['unit_id']+'_')}
 c['knowledge']['card_hashes']={x['evidence_id']:digest(x) for x in CARDS}
 return c


def schedule():
 rows=[];slots={}
 for group in ['T','TV','G','GV']:
  key='SYN-Q1_'+group;unit='SYN-O-'+group
  rows.append({'question_id':'SYN-Q1','group':group,'repeat':1,'cell_key':key,'unit_id':unit,'split':'debug','family_id':'synthetic_family'})
  slots[key+'_generate']='generate'
  if group in ['TV','GV']:
   for a in ['verify_initial','revise','verify_final']:slots[key+'_'+a]=a
  for who in ['A','B']:slots[unit+'_output_'+who]='judge_output'
  slots[unit+'_output_C']='adjudicate_output'
  if group in ['TV','GV']:
   for who in ['A','B']:slots[unit+'_pair_'+who]='judge_pair'
   slots[unit+'_pair_C']='adjudicate_pair'
 return rows,slots


def environment(mode='revise',counter=None,mutate_rating=None):
 parent=Path(tempfile.mkdtemp(prefix='fixture_',dir=R/'synthetic'))
 cfg=config();rows,slots=schedule();ledger=Ledger(parent/'batch',cfg,slots,parent)
 transport=FixtureHTTP(mode,mutate_rating)
 provider=Provider(cfg,counter or FixtureCounter(),ledger,transport,ledger.directory/'provider')
 corpus=FixtureCorpus(REQUEST,CARDS)
 return Runtime(R,cfg,provider,corpus),rows,transport,ledger


class RuntimeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.network_attempts=[]
  def deny(*a,**k):cls.network_attempts.append('blocked');raise AssertionError('offline suite attempted network')
  cls.patches=[patch('socket.create_connection',deny),patch.object(socket.socket,'connect',deny)]
  for p in cls.patches:p.start()
 @classmethod
 def tearDownClass(cls):
  for p in reversed(cls.patches):p.stop()
  if cls.network_attempts:raise AssertionError('network attempted in offline suite')

 def test_full28_call_pipeline_and_all38_metrics(self):
  rt,rows,http,ledger=environment()
  try:
   ids=[m['metric_id'] for m in read_json(ROOT/'runs/e01_design/20261031_design_rev034_7c38b1a2/metrics/metric_catalog.json')]
   report=execute_batch(rt,rows,{'SYN-Q1':{'business_request':REQUEST}},{'SYN-Q1':RUBRIC},ledger.directory,ids)
   self.assertEqual(len(http.calls),28)
   self.assertEqual(ledger.counts,{'generate':4,'verify_initial':2,'revise':2,'verify_final':2,'judge_output':8,'adjudicate_output':4,'judge_pair':4,'adjudicate_pair':2})
   self.assertEqual(rt.corpus.calls,['text','graph'])
   self.assertEqual(len(report['rows']),4)
   self.assertEqual(sum(x['planned'] for x in report['fixed_point_strata'] if x['dimension']=='class'),8)
   self.assertEqual(report['class_diagnostics']['normal']['TV']['model_judged_A_cov']['split_macro']['debug']['value'],1)
   for row in report['rows']:
    self.assertEqual(len(row['metrics']),38)
    self.assertEqual(row['metrics']['model_judged_A_cov']['value'],1 if row['group'] in ['TV','GV'] else 0)
    self.assertEqual(row['metrics']['model_judged_U_claim']['value'],0 if row['group'] in ['TV','GV'] else 1)
   self.assertEqual(report['V_new_full_planned_bounds']['lower']['value'],0)
   self.assertEqual(report['V_new_full_planned_bounds']['upper']['value'],0)
   self.assertEqual(http.calls[0],http.calls[1]) # same text retrieval/generation body for T and TV
   for p in (ledger.directory/'packets').glob('*_output.json'):
    raw=p.read_text();obj=json.loads(raw)
    self.assertNotIn('group',obj);self.assertNotIn('draft',obj);self.assertNotIn('verification_pass',obj)
   for call in http.calls:
    obj=json.loads(call['messages'][1]['content'])
    if 'originals' in obj:
     self.assertTrue(all('metadata' not in x and 'identity' not in x for x in obj['originals']))
   original=list((ledger.directory/'evaluation').glob('*.original.json'))
   self.assertEqual(len(original),18) #12 primary encodings plus6 preserved third originals
   self.assertTrue(all(read_json(p)['status']=='valid' for p in original))
  finally:ledger.close()

 def test_no_revision_uses6_tested_calls_not10(self):
  rt,rows,http,ledger=environment('no_revision')
  try:
   for m in rows:rt.tested_cell(m['cell_key'],m['group'],REQUEST,rt.corpus.retrieve(REQUEST,'text' if m['group'] in ['T','TV'] else 'graph'))
   self.assertEqual(len(http.calls),6) #4 generate+2 initial verify, no revisions
   self.assertNotIn('revise',ledger.counts)
  finally:ledger.close()

 def test_slot_request_and_same_flow_card_binding(self):
  rt,rows,http,ledger=environment()
  try:
   retrieval=rt.corpus.retrieve(REQUEST,'text')
   packet={'business_request':REQUEST,'common_context':REQUEST['explicitly_allowed_operating_context'],'actual_complete_cards':CARDS}
   bad=copy.deepcopy(packet);bad['business_request']['question']='SYNTHETIC: substituted question'
   with self.assertRaises(Rejected):rt.provider.send(rows[0]['cell_key']+'_generate','generate',rt.messages('generation',bad))
   self.assertFalse(http.calls)
   rt.provider.send(rows[1]['cell_key']+'_generate','generate',rt.messages('generation',packet))
   altered=copy.deepcopy(packet);altered['actual_complete_cards']=[]
   with self.assertRaises(Rejected):rt.provider.send(rows[1]['cell_key']+'_verify_initial','verify_initial',rt.messages('verification',dict(altered,draft=candidate(False))))
   self.assertEqual(len(http.calls),1)
  finally:ledger.close()

 def test_overlength_stops_before_reservation(self):
  rt,rows,http,ledger=environment()
  try:
   class TooLong:
    def count(self,messages):return {'input_tokens':100000,'synthetic':True}
   rt.provider.counter=TooLong()
   r=rt.tested_cell(rows[0]['cell_key'],'T',REQUEST,rt.corpus.retrieve(REQUEST,'text'))
   self.assertEqual(r['service_error_category'],'overlength_stop');self.assertFalse(http.calls);self.assertEqual(ledger.reserved,0)
  finally:ledger.close()

 def test_changed_provider_config_and_money_ceiling(self):
  rt,rows,http,ledger=environment()
  try:
   rt.provider.config['model']='SYNTHETIC-unapproved'
   rt.tested_cell(rows[0]['cell_key'],'T',REQUEST,rt.corpus.retrieve(REQUEST,'text'))
   self.assertFalse(http.calls)
  finally:ledger.close()
  parent=Path(tempfile.mkdtemp(prefix='money_',dir=R/'synthetic'));cfg=config();cfg['proposed_hard_budget_USD']='0'
  ledger=Ledger(parent/'batch',cfg,schedule()[1],parent)
  try:
   with self.assertRaises(Rejected):ledger.reserve(schedule()[0][0]['cell_key']+'_generate','generate','x')
   self.assertFalse(ledger.used);self.assertEqual(ledger.reserved,0)
  finally:ledger.close()

 def test_no_silent_duplicate_or_ledger_reset(self):
  rt,rows,http,ledger=environment()
  try:
   r=rt.corpus.retrieve(REQUEST,'text');rt.tested_cell(rows[0]['cell_key'],'T',REQUEST,r)
   rt.tested_cell(rows[0]['cell_key'],'T',REQUEST,r)
   self.assertEqual(len(http.calls),1)
   with self.assertRaises(Rejected):Ledger(ledger.directory,config(),schedule()[1],ledger.directory.parent)
  finally:ledger.close()

 def test_http_failure_preserved_and_no_further_spending(self):
  rt,rows,http,ledger=environment('http500')
  try:
   r=rt.tested_cell(rows[0]['cell_key'],'T',REQUEST,rt.corpus.retrieve(REQUEST,'text'))
   self.assertFalse(r['publishable']);self.assertTrue(ledger.broken)
   self.assertEqual(len(http.calls),1);self.assertEqual(len(list((ledger.directory/'provider').glob('*.http.json'))),1)
   rt.tested_cell(rows[1]['cell_key'],'TV',REQUEST,rt.corpus.retrieve(REQUEST,'text'))
   self.assertEqual(len(http.calls),1)
  finally:ledger.close()

 def test_unknown_usage_stops_batch(self):
  rt,rows,http,ledger=environment('usage_missing')
  try:
   rt.tested_cell(rows[0]['cell_key'],'T',REQUEST,rt.corpus.retrieve(REQUEST,'text'))
   self.assertTrue(ledger.broken);self.assertEqual(len(http.calls),1)
   self.assertGreater(ledger.reserved,0)
  finally:ledger.close()

 def test_blank_and_truncated_are_not_retried(self):
  for mode in ['blank','truncated']:
   with self.subTest(mode=mode):
    rt,rows,http,ledger=environment(mode)
    try:
     r=rt.tested_cell(rows[0]['cell_key'],'T',REQUEST,rt.corpus.retrieve(REQUEST,'text'))
     self.assertIsNone(r['final_answer']);self.assertEqual(len(http.calls),1)
    finally:ledger.close()

 def test_fsync_failure_prevents_transport(self):
  rt,rows,http,ledger=environment()
  try:
   with patch('provider.os.fsync',side_effect=OSError('SYNTHETIC fsync failure')):
    r=rt.tested_cell(rows[0]['cell_key'],'T',REQUEST,rt.corpus.retrieve(REQUEST,'text'))
   self.assertFalse(http.calls);self.assertTrue(ledger.broken);self.assertFalse(r['publishable'])
  finally:ledger.close()

 def test_ledger_tamper_and_unknown_slot_rejected(self):
  rt,rows,http,ledger=environment()
  try:
   with self.assertRaises(Rejected):ledger.reserve('SYN-unregistered','generate','x')
   with ledger.path.open('ab') as f:f.write(b'changed')
   with self.assertRaises(Rejected):ledger.reserve(rows[0]['cell_key']+'_generate','generate','x')
   self.assertFalse(http.calls)
  finally:ledger.close()

 def test_global_top3_full_card_and_snapshot(self):
  rt,rows,http,ledger=environment()
  try:
   cards=copy.deepcopy(CARDS);cards[0].pop('limitations')
   with self.assertRaises(Rejected):rt.corpus.validate_returned(cards)
   with self.assertRaises(Rejected):rt.corpus.validate_returned(CARDS*4)
   out=final_packet('SYN-O',{'actual_complete_cards':CARDS,'business_request':REQUEST,'status':'completed','publishable':True,
     'final_answer':None,'service_message':'SYNTHETIC only service','service_error_category':None},RUBRIC,rt.corpus.cards)
   self.assertIsNone(out['final_answer']);self.assertNotIn('draft',out)
  finally:ledger.close()

 def test_schema_semantic_negative_cases(self):
  rt,rows,http,ledger=environment('no_revision')
  try:
   rec=rt.tested_cell(rows[0]['cell_key'],'T',REQUEST,rt.corpus.retrieve(REQUEST,'text'))
   packet=final_packet('SYN-O',rec,RUBRIC,rt.corpus.cards);good=output_rating(packet)
   validate_output(good,rt.schemas['output'],packet,rec,RUBRIC)
   cases=[lambda v:v['point_rows'].pop(),lambda v:v['point_rows'][0].update(score=.5),
    lambda v:v['point_rows'][0].update(status='not_applicable',score=None),
    lambda v:v['claims'][0]['spans'][0].update(quote='forged'),
    lambda v:v['claims'][0]['spans'][0].update(channel='draft'),
    lambda v:v['citation_pairs'][0].update(evidence_id='UNRETURNED',id_valid=True),
    lambda v:v['insufficiency'].update(available_point_ids=[]),
    lambda v:v['insufficiency'].update(actual_gap_ids=[]),
    lambda v:v['insufficiency']['rows'][1].update(score=.5,half_credit_allowed=True),
    lambda v:v['claims'][0].update(necessary_condition_ids=['orphan']),
    lambda v:v.update(group='GV')]
   for change in cases:
    bad=copy.deepcopy(good);change(bad)
    with self.assertRaises((Rejected,__import__('jsonschema').ValidationError)):validate_output(bad,rt.schemas['output'],packet,rec,RUBRIC)
   half_rubric=copy.deepcopy(RUBRIC);half_rubric['insufficiency_criteria'][2]['half_credit_eligible']=True
   bad=copy.deepcopy(good);bad['insufficiency']['rows'][2].update(score=.5,half_credit_allowed=True)
   with self.assertRaises(Rejected):validate_output(bad,rt.schemas['output'],packet,rec,half_rubric)
  finally:ledger.close()

 def test_pair_namespace_and_no_final_credit(self):
  rt,rows,http,ledger=environment()
  try:
   rec=rt.tested_cell(rows[1]['cell_key'],'TV',REQUEST,rt.corpus.retrieve(REQUEST,'text'))
   p=pair_packet('SYN-O',rec,RUBRIC,rt.corpus.cards);v=pair_rating(p);validate_pair(v,rt.schemas['pair'],p)
   self.assertNotEqual(v['reported_issue_ids'][0],v['detected_error_ids'][0])
   bad=copy.deepcopy(v);bad['report_error_mapping']=[]
   with self.assertRaises(Rejected):validate_pair(bad,rt.schemas['pair'],p)
   rec['final_answer']=None;rec['publishable']=False
   p=pair_packet('SYN-O',rec,RUBRIC,rt.corpus.cards);v=pair_rating(p);validate_pair(v,rt.schemas['pair'],p)
   m=pair_metrics(rec,{'canonical':{'status':'resolved_primary_agreement','rating':v}})
   self.assertEqual(m['model_judged_V_fix']['value'],0)
   self.assertIsNone(m['model_judged_V_new_run']['value'])
  finally:ledger.close()

 def test_zero_unknown_failure_and_family_aggregation(self):
  self.assertIsNone(bounded([])['value'])
  self.assertEqual(bounded([1,None,0])['lower'],1/3);self.assertEqual(bounded([1,None,0])['upper'],2/3)
  rows=[{'group':'T','split':'debug','family_id':'f1','question_id':'q1','metrics':{'x':bounded([1])}},
        {'group':'T','split':'debug','family_id':'f1','question_id':'q1','metrics':{'x':bounded([0])}},
        {'group':'T','split':'debug','family_id':'f1','question_id':'q2','metrics':{'x':bounded([1])}},
        {'group':'T','split':'debug','family_id':'f2','question_id':'q3','metrics':{'x':bounded([0])}}]
  self.assertEqual(aggregate(rows)['T']['x']['split_macro']['debug']['value'],.375)

 def test_live_transport_rejects_before_credential_access(self):
  class Env:
   def get(self,*a):raise AssertionError('credential environment inspected')
  with patch('provider.os.environ',Env()):
   with self.assertRaises(Rejected):LiveHTTP(object(),read_json(R/'config/execution.json'))

 def test_real_configuration_cannot_initialize_unreleased_ledger(self):
  parent=Path(tempfile.mkdtemp(prefix='unreleased_',dir=R/'synthetic'));target=parent/'batch'
  cfg=read_json(R/'config/execution.json');slots=read_json(R/'config/schedule_private.json')['allowed_slots']
  with self.assertRaises(Rejected):Ledger(target,cfg,slots,parent)
  self.assertFalse(target.exists())

 def test_path_symlink_rejected_without_following(self):
  parent=Path(tempfile.mkdtemp(prefix='path_',dir=R/'synthetic'));target=parent/'ordinary';target.write_text('SYNTHETIC')
  link=parent/'alias';link.symlink_to(target)
  try:
   with self.assertRaises(Rejected):safe_path(link,parent)
  finally:link.unlink()
  self.assertEqual(target.read_text(),'SYNTHETIC')


if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeTests)
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 report={'status':'PASS' if result.wasSuccessful() else 'FAIL','test_methods':result.testsRun,
  'failures':[(str(t),s) for t,s in result.failures],'errors':[(str(t),s) for t,s in result.errors],
  'network_attempts':RuntimeTests.network_attempts,'actual_model_API_calls':0,'real_question_retrievals':0,'new_tested_answers':0,
  'synthetic_fixture_queries_only':True,'note':'nested subcases and complete28-call mocked pipeline included; no production model correctness inference'}
 report['code_sha256']={p.name:file_hash(p) for p in sorted((R/'runtime').glob('*.py'))}
 report['configuration_sha256']=digest(read_json(R/'config/execution.json'))
 (R/'validation/runtime_tests.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 sys.exit(0 if result.wasSuccessful() else 1)
