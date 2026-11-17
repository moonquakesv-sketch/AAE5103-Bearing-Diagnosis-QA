"""Literal SYNTHETIC regression oracles, run unchanged against copied rev036 and rev037."""
from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal
from collections import Counter
import sys,json,copy,unittest,tempfile,socket,os
RUN=Path(__file__).absolute().parents[1]
PHASE=sys.argv[1];assert PHASE in ['before','after']
PREP=RUN/'before' if PHASE=='before' else RUN
sys.path[:0]=[str(PREP/'runtime'),str(PREP/'tests'),str(RUN/'analysis')]
from audit_io import write,safe
from common import digest,snapshot,Rejected,canonical
from provider import Ledger,Provider
from corpus import FixtureCorpus
from engine import Runtime,final_packet
from cli import execute_batch
from evaluation import validate_output
import reporting
from fixtures import REQUEST,CARDS,RUBRIC,candidate,output_rating,FixtureCounter,FixtureHTTP
PROJECT=RUN.parents[2]
IDS=[m['metric_id'] for m in json.loads((PROJECT/'runs/e01_design/20261031_design_rev034_7c38b1a2/metrics/metric_catalog.json').read_text())]
OUT=RUN/'synthetic'/('rev037_oracles_'+PHASE);safe(OUT).mkdir(parents=True)
def config(groups):
 c=json.loads((PREP/'config/execution.json').read_text());c['scope']['stage']='SYNTHETIC_REV037_ONLY'
 c['knowledge']['card_hashes']={c['evidence_id']:digest(c) for c in CARDS}
 rows=[];slots={}
 for g in groups:
  key='SYNTHETIC-CELL-'+g;u='SYNTHETIC-UNIT-'+g
  rows.append({'question_id':'SYN-Q1','family_id':'SYNTHETIC-F','group':g,'split':'debug','repeat':1,'cell_key':key,'unit_id':u})
  slots[key+'_generate']='generate'
  if g in ['TV','GV']:
   for a in ['verify_initial','revise','verify_final']:slots[key+'_'+a]=a
  for who in ['A','B']:slots[u+'_output_'+who]='judge_output'
  slots[u+'_output_C']='adjudicate_output'
  if g in ['TV','GV']:
   for who in ['A','B']:slots[u+'_pair_'+who]='judge_pair'
   slots[u+'_pair_C']='adjudicate_pair'
 c['allowed_slots_sha256']=digest(slots)
 c['slot_bindings']={s:{'business_sha256':digest(REQUEST),'flow_key':m['cell_key']} for m in rows for s in slots if s.startswith(m['cell_key']+'_') or s.startswith(m['unit_id']+'_')}
 return c,rows,slots
def pipeline(label,groups,failmode=None,timeout=False):
 cfg,rows,slots=config(groups);parent=OUT/label;safe(parent).mkdir()
 ledger=Ledger(parent/'batch',cfg,slots,parent);transport=FixtureHTTP('no_revision')
 class SyntheticTransport:
  def __call__(self,body):
   p=json.loads(body['messages'][1]['content'])
   if timeout and ('unit_id' in p or 'originals' in p):raise Rejected('transport timeout; no retry')
   return transport(body)
 provider=Provider(cfg,FixtureCounter(),ledger,SyntheticTransport(),ledger.directory/'provider')
 corpus=FixtureCorpus(REQUEST,CARDS);calls=[];orig=corpus.retrieve
 def retrieve(req,mode,top_k=3):
  assert req['question'].startswith('SYNTHETIC:')
  calls.append(mode)
  if mode==failmode and calls.count(mode)==1:raise OSError('SYNTHETIC transient local retrieval failure')
  return orig(req,mode,top_k)
 corpus.retrieve=retrieve
 rt=Runtime(PREP,cfg,provider,corpus)
 try:
  report=execute_batch(rt,rows,{'SYN-Q1':{'business_request':REQUEST}},{'SYN-Q1':RUBRIC},ledger.directory,IDS)
  records={r['group']:json.loads((ledger.directory/'tested'/(r['cell_key']+'.json')).read_text()) for r in rows}
  return report,records,calls,ledger.directory
 finally:ledger.close()
def rec(i,group='TV',split='debug',test=5,evaluation=4):
 return {'cell_key':'SYNTHETIC-DIRECT-C'+str(i),'group':group,'business_request':snapshot(REQUEST),'actual_complete_cards':snapshot(CARDS),'retrieval':{'status':'retrieved'},
 'status':'completed','publishable':True,'final_answer':candidate(True)['answer'],'draft':candidate(False),'service_message':None,'service_error_category':None,
 'verification_issues':[],'verification_observable':True,'tested_latency_seconds':test,'evaluation_latency_seconds':evaluation,
 'workflow_active_latency_seconds':test+evaluation,'synthetic_split':split}
def report_direct(records,events=None,no_evaluation=False,rubric=None):
 rubric=rubric or RUBRIC;mappings=[];outputs={};slots={}
 for i,r in enumerate(records):
  u='SYNTHETIC-DIRECT-U'+str(i)
  mappings.append({'question_id':'SYN-Q1','family_id':'SYNTHETIC-F','group':r['group'],'split':r['synthetic_split'],'repeat':i+1,'cell_key':r['cell_key'],'unit_id':u})
  slots[u+'_output_A']='judge_output'
  if not no_evaluation:
   p=final_packet(u,r,rubric,{c['evidence_id']:c for c in CARDS});rating=output_rating(p)
   outputs[u]={'originals':[{'status':'valid','rating':rating}]*2,'canonical':{'status':'resolved_primary_agreement','rating':rating}}
 if events is None:events=[]
 ledger=SimpleNamespace(expected_bytes=b'\n'.join(json.dumps(e).encode() for e in events),allowed=slots,counts=dict(Counter(slots[e['slot']] for e in events if e['event']=='reserve')),reserved=Decimal('0'))
 return reporting.summarize(records,outputs,{}, {'SYN-Q1':rubric},SimpleNamespace(cards={c['evidence_id']:c for c in CARDS}),mappings,ledger,IDS)
class FixOracles(unittest.TestCase):
 def test_failed_text_outcome_shared_both_orders(self):
  for i,groups in enumerate([['T','TV'],['TV','T']]):
   with self.subTest(groups=groups):
    report,rs,calls,_=pipeline('text-failure-'+str(i),groups,'text')
    self.assertEqual(calls,['text'])
    self.assertTrue(all(rs[g]['status']=='tested_failure' and not rs[g]['actual_complete_cards'] for g in groups))
    self.assertEqual({rs[g]['failure_type'] for g in groups},{'OSError'})
 def test_failed_graph_outcome_shared_both_orders(self):
  for i,groups in enumerate([['G','GV'],['GV','G']]):
   with self.subTest(groups=groups):
    _,rs,calls,_=pipeline('graph-failure-'+str(i),groups,'graph')
    self.assertEqual(calls,['graph'])
    self.assertTrue(all(rs[g]['status']=='tested_failure' and not rs[g]['actual_complete_cards'] for g in groups))
 def test_real_synthetic_evaluation_timeout_propagates_and_persists(self):
  r,_,_,batch=pipeline('evaluation-timeout',['T'],timeout=True)
  for phase in ['evaluation_total','workflow_active_total']:
   self.assertEqual(r['timing'][phase]['observed'],0);self.assertIsNone(r['timing'][phase]['mean'])
   self.assertEqual(len(r['timing'][phase]['censored']),1)
  record=json.loads(next((batch/'timing').glob('*.json')).read_text())
  self.assertTrue(record['evaluation_timing_censored']);self.assertTrue(record['workflow_timing_censored'])
 def test_normal_pipeline_timing_files_match_report(self):
  r,_,calls,batch=pipeline('normal-four-groups',['T','TV','G','GV'])
  self.assertEqual(calls,['text','graph'])
  self.assertEqual(len(r['rows']),4);self.assertTrue(all(len(x['metrics'])==38 for x in r['rows']))
  saved={p.stem:json.loads(p.read_text()) for p in (batch/'timing').glob('*.json')}
  self.assertEqual(len(saved),4)
  for row in r['timing']['per_cell']:
   for key in ['evaluation_latency_seconds','workflow_active_latency_seconds','evaluation_timing_censored','workflow_timing_censored']:
    self.assertEqual(saved[row['cell_key']][key],row[key])
 def test_group_split_and_action_statistics(self):
  records=[rec(0,'TV','debug',5,4),rec(1,'TV','operational_reserve',5,40),rec(2,'GV','debug',5,8),rec(3,'GV','operational_reserve',5,80)]
  events=[]
  for i,t in enumerate([4,40,8,80]):
   s='SYNTHETIC-DIRECT-U'+str(i)+'_output_A'
   events+=[{'event':'reserve','slot':s},{'event':'result','slot':s,'status':'received','metadata':{'latency_seconds':t,'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}}]
  r=report_direct(records,events)
  write(OUT/'SYNTHETIC_four_strata_report.json',r)
  self.assertEqual(r['timing']['evaluation_total']['mean'],33)
  for g,s,v in [('TV','debug',4),('TV','operational_reserve',40),('GV','debug',8),('GV','operational_reserve',80)]:
   a=r['timing']['by_group_split'][g][s]
   self.assertEqual((a['evaluation_total']['mean'],a['evaluation_total']['median'],a['evaluation_total']['minimum'],a['evaluation_total']['maximum'],a['evaluation_total']['observed'],a['evaluation_total']['planned']),(v,v,v,v,1,1))
   self.assertEqual(a['workflow_active_total']['mean'],v+5)
   self.assertEqual(a['provider_by_action']['judge_output']['mean'],v)
  self.assertEqual(r['timing']['by_group']['TV']['evaluation_total']['mean'],22)
  self.assertEqual(r['timing']['by_split']['debug']['evaluation_total']['mean'],6)
 def test_timeout_lower_bound_100_105_not_completed(self):
  x=rec(0,'TV','debug',5,100);s='SYNTHETIC-DIRECT-U0_output_A'
  events=[{'event':'reserve','slot':s},{'event':'result','slot':s,'status':'failed','metadata':{'latency_seconds':100,'timing_censored':True}}]
  r=report_direct([x],events,no_evaluation=True)
  write(OUT/'SYNTHETIC_100_105_report.json',r)
  for phase,lower in [('evaluation_total',100),('workflow_active_total',105)]:
   v=r['timing'][phase];self.assertIsNone(v['mean']);self.assertEqual(v['observed'],0);self.assertEqual(v['planned'],1)
   self.assertEqual(v['censored'][0]['observed_lower_bound_seconds'],lower)
  self.assertEqual(r['timing']['tested_total']['mean'],5)
 def test_output_errors_quote_and_omission_branches(self):
  r=rec(0);p=final_packet('SYNTHETIC-ERRORS',r,RUBRIC,{c['evidence_id']:c for c in CARDS})
  good=output_rating(p);schema=json.loads((PREP/'schemas/output.json').read_text())
  def value(kind,spans):
   v=snapshot(good);v['errors']=[{'error_id':'SYNTHETIC-E','kind':kind,'proposition':'SYNTHETIC assertion','spans':spans,'reason':'SYNTHETIC exact quote oracle'}];return v
  for spans in [[],[{'channel':'final_answer','start':0,'end':0,'quote':''}],[{'channel':'final_answer','start':0,'end':1,'quote':'Z'}]]:
   with self.subTest(spans=spans):
    with self.assertRaises(Rejected):validate_output(value('unsupported',spans),schema,p,r,RUBRIC)
  validate_output(value('unsupported',[{'channel':'final_answer','start':0,'end':9,'quote':'SYNTHETIC'}]),schema,p,r,RUBRIC)
  validate_output(value('omission',[]),schema,p,r,RUBRIC)
 def test_absent_evaluator_preserves_frozen_I_NA(self):
  q=snapshot(RUBRIC);q['required_unknowns']=[];q['points']=[q['points'][0]]
  for c in q['insufficiency_criteria']:c['applicable']=False
  r=report_direct([rec(0)],no_evaluation=True,rubric=q)
  v=r['rows'][0]['metrics']['model_judged_A_insuff']
  self.assertEqual(v['denominator'],0);self.assertEqual(v['status'],'not_applicable')
def audit(event,args):
 if event in ['socket.connect','socket.getaddrinfo','subprocess.Popen']:raise AssertionError('SYNTHETIC suite forbids network/subprocess')
 if event=='open':
  p,mode,flags=args
  if isinstance(p,(str,bytes,os.PathLike)) and ((isinstance(mode,str) and any(x in mode for x in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):safe(Path(os.fsdecode(p)))
sys.addaudithook(audit)
res=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FixOracles))
out={'phase':PHASE,'tests':res.testsRun,'success':res.wasSuccessful(),'failures':[(str(t),s) for t,s in res.failures],'errors':[(str(t),s) for t,s in res.errors],
 'synthetic_only':True,'same_assertions_both_phases':True,'project_model_API_calls':0,'LiveHTTP_constructed':False}
write(RUN/'validation'/('rev037_oracles_'+PHASE+'.json'),out)
print(json.dumps(out,ensure_ascii=False))
sys.exit(0 if res.wasSuccessful() else 1)
