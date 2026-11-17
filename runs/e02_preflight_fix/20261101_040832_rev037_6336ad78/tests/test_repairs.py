"""Saved synthetic repair oracles; same assertions run before and after edits."""
from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal
import sys,json,copy,tempfile,socket,unittest
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];ROOT=R.parents[2]
sys.path[:0]=[str(R/'runtime'),str(R/'tests')]
import test_runtime as legacy
import reporting,metric_math
from common import Rejected,canonical,digest,read_json
from engine import Runtime,final_packet,pair_packet
from evaluation import AutomaticEvaluation,validate_pair
from provider import Ledger,Provider
from fixtures import REQUEST,CARDS,RUBRIC,candidate,output_rating,pair_rating,FixtureCounter,FixtureHTTP
def cloned(v):return copy.deepcopy(v)
def record(body=True,service=None):
 return {'cell_key':'SYN-CELL','group':'TV','business_request':cloned(REQUEST),'actual_complete_cards':cloned(CARDS),'retrieval':{'status':'retrieved'},'status':'completed','publishable':body,'final_answer':candidate(True)['answer'] if body else None,'draft':candidate(False),'service_message':service,'service_error_category':None,'tested_latency_seconds':2,'verification_issues':[],'verification_observable':True}
def eval_fixture(rec=None):
 rec=rec or record()
 p=final_packet('SYN-UNIT',rec,RUBRIC,{c['evidence_id']:c for c in CARDS});v=output_rating(p)
 return {'originals':[{'status':'valid','rating':cloned(v)},{'status':'valid','rating':cloned(v)}],'canonical':{'status':'resolved_primary_agreement','rating':v,'review':{'assigned':False,'status':'not_assigned'}}}
def make_report(rec=None,evaluation=None,pair_eval=None,events=None):
 rec=rec or record();rec['group']='T' if pair_eval is None else 'TV'
 ev=evaluation or eval_fixture(rec)
 usage={'prompt_tokens':10,'completion_tokens':4,'total_tokens':14,'prompt_cache_hit_tokens':3,'prompt_cache_miss_tokens':7,'completion_tokens_details':{'reasoning_tokens':0}}
 slot=rec['cell_key']+'_generate'
 if events is None:events=[{'event':'reserve','slot':slot},{'event':'result','slot':slot,'status':'received','metadata':{'usage':usage,'local_input_tokens':9,'latency_seconds':2,'timing_censored':False}}]
 led=SimpleNamespace(expected_bytes=b'\n'.join(json.dumps(e).encode() for e in events),allowed={slot:'generate'},counts={'generate':1},reserved=Decimal('0.001'))
 mappings=[{'question_id':'SYN-Q','family_id':'SYN-F','split':'debug','repeat':1,'cell_key':rec['cell_key'],'unit_id':'SYN-UNIT','group':rec['group']}]
 ids=[x['metric_id'] for x in read_json(ROOT/'runs/e01_design/20261031_design_rev034_7c38b1a2/metrics/metric_catalog.json')]
 return reporting.summarize([rec],{'SYN-UNIT':ev},{'SYN-UNIT':pair_eval} if pair_eval else {},{'SYN-Q':RUBRIC},SimpleNamespace(cards={c['evidence_id']:c for c in CARDS}),mappings,led,ids)

class WrongThird(FixtureHTTP):
 def __call__(self,body):
  raw=super().__call__(body);p=json.loads(body['messages'][1]['content'])
  if 'originals' in p:
   envelope=json.loads(raw['raw_body']);v=json.loads(envelope['choices'][0]['message']['content'])
   v['decisions'][0]['quote_spans']=[{'channel':'final_answer','start':0,'end':1,'quote':'Z'}]
   return self.wrap(v)
  return raw

class Repairs(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.network=[]
  def deny(*a,**k):cls.network.append('blocked');raise AssertionError('network forbidden')
  cls.patches=[patch('socket.create_connection',deny),patch.object(socket.socket,'connect',deny)]
  for p in cls.patches:p.start()
 @classmethod
 def tearDownClass(cls):
  for p in reversed(cls.patches):p.stop()
  if cls.network:raise AssertionError('network attempted')
 def rejected_third(self):
  rt,rows,_,ledger=legacy.environment();rt.provider.transport=WrongThird()
  rec=rt.tested_cell(rows[1]['cell_key'],'TV',REQUEST,rt.corpus.retrieve(REQUEST,'text'))
  packet=final_packet(rows[1]['unit_id'],rec,RUBRIC,rt.corpus.cards)
  ev=AutomaticEvaluation(rt,ledger.directory/'evaluation').evaluate(packet['unit_id'],'output',packet,rec,RUBRIC)
  return rt,ledger,rec,packet,ev
 def test_001_rejected_third_clears_rating(self):
  rt,led,rec,p,ev=self.rejected_third()
  try:
   self.assertEqual(ev['canonical']['status'],'evaluator_failure')
   self.assertIsNone(ev['canonical']['rating'])
   m=reporting.output_metrics(rec,RUBRIC,ev,rt.corpus.cards)['model_judged_A_cov']
   self.assertIsNone(m['value']);self.assertEqual((m['lower'],m['upper']),(0,1))
  finally:led.close()
 def test_001_invalid_canonical_never_enters_point_or_pair_metrics(self):
  rec=record();ev=eval_fixture(rec);ev['canonical']['status']='evaluator_failure'
  m=reporting.output_metrics(rec,RUBRIC,ev,{c['evidence_id']:c for c in CARDS})
  self.assertIsNone(m['model_judged_A_cov']['value'])
  report=make_report(rec,ev)
  self.assertTrue(all(p['score'] is None for p in report['rows'][0]['point_scores']))
 def test_002_third_receives_complete_schema_for_both_kinds(self):
  rt,rows,_,led=legacy.environment()
  try:
   for kind in ['output','pair']:
    message=rt.messages('adjudicator',{'kind':kind,'SYNTHETIC':'both primary ratings null','originals':[{'rating':None},{'rating':None}]})
    self.assertIn(canonical(rt.schemas[kind]),message[0]['content'])
  finally:led.close()
 def test_003_empty_quote_asserted_error_rejected(self):
  rec=record();p=pair_packet('SYN-PAIR',rec,RUBRIC,{c['evidence_id']:c for c in CARDS});v=pair_rating(p)
  v['draft_errors'][0]['spans']=[{'channel':'draft','start':0,'end':0,'quote':''}]
  with self.assertRaises(Rejected):validate_pair(v,read_json(R/'schemas/pair.json'),p)
 def test_003_omission_without_quote_remains_valid(self):
  rec=record();p=pair_packet('SYN-PAIR',rec,RUBRIC,{c['evidence_id']:c for c in CARDS});v=pair_rating(p)
  v['draft_errors'][0].update(kind='omission',spans=[])
  validate_pair(v,read_json(R/'schemas/pair.json'),p)
 def test_004_oversized_raw_saved_once_reserve_kept(self):
  parent=Path(tempfile.mkdtemp(prefix='raw_',dir=R/'synthetic'));cfg=legacy.config();cfg['max_response_bytes']=8
  rows,slots=legacy.schedule();led=Ledger(parent/'batch',cfg,slots,parent)
  raw=b'SYNTHETIC-TOO-LARGE'
  class HTTP:
   calls=0
   def __call__(self,body):self.calls+=1;return {'http_status':200,'request_id':'SYN-RAW','raw_body':raw}
  http=HTTP();provider=Provider(cfg,FixtureCounter(),led,http,led.directory/'provider');rt=Runtime(R,cfg,provider,None)
  packet={'business_request':REQUEST,'common_context':REQUEST['explicitly_allowed_operating_context'],'actual_complete_cards':CARDS}
  try:
   with self.assertRaises(Rejected):provider.send(rows[0]['cell_key']+'_generate','generate',rt.messages('generation',packet))
   self.assertEqual(http.calls,1);self.assertTrue(led.broken);self.assertGreater(led.reserved,0)
   p=led.directory/'provider'/(rows[0]['cell_key']+'_generate.http.json')
   self.assertTrue(p.exists());v=read_json(p)
   import base64,hashlib
   self.assertEqual(base64.b64decode(v['raw_base64']),raw);self.assertEqual(v['raw_sha256'],hashlib.sha256(raw).hexdigest());self.assertEqual(v['request_id'],'SYN-RAW')
  finally:led.close()
 def test_005_third_failed_package_in_failure_rate(self):
  rt,led,rec,p,ev=self.rejected_third()
  try:
   m=reporting.output_metrics(rec,RUBRIC,ev,rt.corpus.cards)['evaluator_failure_rate']
   self.assertEqual((m['numerator'],m['denominator']),(1,3))
   self.assertTrue(list((led.directory/'evaluation').glob('*_C.original.json')))
  finally:led.close()
 def test_005_pair_and_review_breakdown(self):
  rec=record();out=eval_fixture(rec);p=pair_packet('SYN-UNIT',rec,RUBRIC,{c['evidence_id']:c for c in CARDS});v=pair_rating(p)
  pe={'originals':[{'status':'evaluator_failure','rating':None},{'status':'valid','rating':v}],'canonical':{'status':'resolved_third_agent','rating':v,'review':{'assigned':True,'status':'valid'}}}
  report=make_report(rec,out,pe)
  c=report['rows'][0]['evaluation_packages']
  self.assertEqual(c['pair']['initial']['failed'],1);self.assertEqual(c['pair']['initial']['assigned'],2)
  self.assertEqual(c['pair']['review']['assigned'],1)
  self.assertEqual(report['rows'][0]['metrics']['evaluator_failure_rate']['denominator'],5)
 def test_006_no_body_service_error_retained_separately(self):
  service='SYNTHETIC: no manufacturer has any safety evidence anywhere.'
  rec=record(False,service);p=pair_packet('SYN-UNIT',rec,RUBRIC,{c['evidence_id']:c for c in CARDS});v=pair_rating(p)
  v['final_errors']=[{'error_id':'SYN-NEW-SERVICE','kind':'scope','proposition':'SYNTHETIC universal absence','spans':[{'channel':'service_message','start':0,'end':len(service),'quote':service}],'reason':'explicit fixture'}]
  validate_pair(v,read_json(R/'schemas/pair.json'),p)
  pe={'originals':[{'status':'valid','rating':v}]*2,'canonical':{'status':'resolved_primary_agreement','rating':v,'review':{'assigned':False,'status':'not_assigned'}}}
  m=reporting.pair_metrics(rec,pe)
  self.assertEqual(m['model_judged_V_fix']['value'],0);self.assertIsNone(m['model_judged_V_new_run']['value'])
  report=make_report(rec,eval_fixture(rec),pe)
  self.assertEqual(report['service_pair_diagnostics'][0]['new_service_error_ids'],['SYN-NEW-SERVICE'])
 def test_007_latency_median_range_and_missing(self):
  rows=[{'group':'T','split':'debug','family_id':'SYN-F','question_id':'SYN-Q'+str(i),'metrics':{'latency':{'numerator':x,'denominator':1,'value':x,'lower':x,'upper':x,'status':'ok'}}} for i,x in enumerate([1,2,100])]
  v=reporting.aggregate(rows)['T']['latency']['observed_timing']
  self.assertEqual((v['mean'],v['median'],v['minimum'],v['maximum']),(103/3,2,1,100));self.assertEqual(v['observed'],3)
 def test_007_stage_timing_summary_present(self):
  r=make_report()
  v=r['timing']['provider_by_action']['generate']
  self.assertEqual(v['median'],2);self.assertEqual(v['observed'],1)
 def test_008_token_categories_and_unknown_preserved(self):
  report=make_report();v=report['accounting']['usage_by_action']['generate']
  self.assertEqual(v['cache_hit_tokens']['total'],3);self.assertEqual(v['cache_miss_tokens']['total'],7);self.assertEqual(v['reasoning_tokens']['total'],0)
  slots='SYN-CELL_generate';events=[{'event':'reserve','slot':slots},{'event':'result','slot':slots,'status':'received','metadata':{'usage':{'prompt_tokens':10,'completion_tokens':4,'total_tokens':14},'latency_seconds':2}}]
  unknown=make_report(events=events)['accounting']['usage_by_action']['generate']['cache_hit_tokens']
  self.assertIsNone(unknown['total']);self.assertEqual(unknown['observed'],0);self.assertEqual(unknown['missing'],1)
 def test_009_role_sums_and_zero_knowledge_na(self):
  report=make_report();roles=report['rows'][0]['coverage_by_role']
  self.assertEqual(roles['positive_knowledge']['numerator'],1);self.assertEqual(roles['boundary']['denominator'],1)
  self.assertEqual(roles['source_status']['status'],'not_applicable')
  self.assertIn('role_coverage',report)
 def test_001_valid_canonical_stays_usable(self):
  rec=record();v=reporting.output_metrics(rec,RUBRIC,eval_fixture(rec),{c['evidence_id']:c for c in CARDS})
  self.assertEqual(v['model_judged_A_cov']['value'],1)

if __name__=='__main__':
 phase=sys.argv[1];assert phase in ['before','after']
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Repairs);result=unittest.TextTestRunner(verbosity=2).run(suite)
 report={'phase':phase,'success':result.wasSuccessful(),'tests':result.testsRun,'failures':[(str(t),s) for t,s in result.failures],'errors':[(str(t),s) for t,s in result.errors],'network_attempts':Repairs.network,'project_model_API_calls':0,'credential_reads':0,'real_question_dynamic_calls':0}
 p=R/'validation'/('repairs_'+phase+'.json')
 with p.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
 sys.exit(0 if result.wasSuccessful() else 1)

