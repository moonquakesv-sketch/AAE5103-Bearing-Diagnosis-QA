"""No CLI/launch. Explicit future release required for E02; offline tests use SYNTHETIC only."""
from pathlib import Path
import time
from common import require,digest,read_json,write_new,safe_path,snapshot
from baseline import generation_packet,evaluation_packet,messages,validate_candidate,validate_rating,resolve,fixed_ratio,family_macro
from provider import LiveGate,LiveHTTP,Ledger,Provider

def metrics(rating,points):
 ids={p['point_id']:p for p in points}
 scores={p['point_id']:p['score'] for p in rating['points']} if rating else {k:None for k in ids}
 result={'REF_cov':fixed_ratio(list(scores.values()))}
 for role,name in [('positive_knowledge','REF_knowledge'),('boundary','REF_boundary'),('source_status','REF_source_state')]:result[name]=fixed_ratio([scores[k] for k,p in ids.items() if p['content_role']==role])
 result['REF_unsupported_claim']=fixed_ratio([{'full':0,'partial':0.5,'none':1,'outside_reference':None,'machine_unresolved':None}[r['support']] for r in rating['claims']]) if rating else {'status':'unknown','value':None,'denominator':None,'lower':None,'upper':None}
 result['REF_voluntary_citation_error']=fixed_ratio([{'supported':0,'contradicted':1,'fabricated_id':1,'unverifiable':None,'machine_unresolved':None}[r['status']] for r in rating['voluntary_citations']]) if rating else {'status':'unknown','value':None,'denominator':None,'lower':None,'upper':None}
 return result

def cell(provider,key,unit,request,rubric,reference_cards,output_root):
 """Fresh messages each call; retain all original provider records/coding failures."""
 started=time.monotonic(); generated=None; candidate_info=None; failure=None
 try:
  generated,meta=provider.send(key+'_generate','generate',messages('generate',generation_packet(request)))
  candidate_info=validate_candidate(generated)
 except Exception as e:failure={'error_type':type(e).__name__}
 observed=candidate_info['observed_text'] if candidate_info else 'The service could not publish an answer for this planned cell.'
 packet=evaluation_packet(unit,request,rubric,reference_cards,observed)
 originals=[]
 for suffix in ['A','B','C']:
  if provider.ledger.broken:
   originals.append({'status':'not_attempted_batch_stopped','rating':None})
   continue
  role='adjudicate_output' if suffix=='C' else 'judge_output'
  payload={'packet':packet,'A':originals[0],'B':originals[1]} if suffix=='C' else packet
  try:
   value,meta=provider.send(key+'_'+suffix,role,messages(role,payload))
   validated=validate_rating(value,packet)
   original={'status':'valid','rating':validated}
  except Exception as e:original={'status':'evaluator_failure','rating':None,'error_type':type(e).__name__}
  originals.append(original)
 final=resolve(*originals)
 data={'generated':generated,'candidate_policy':candidate_info,'generation_failure':failure,'observed_text':observed,'original_ratings':originals,'final':final,'metrics':metrics(final['rating'],rubric['points']),'active_seconds':time.monotonic()-started,'raw_provider_records_retained':True,'no_real_result_in_preparation':True}
 write_new(Path(output_root)/(key+'.result.json'),data,output_root)
 return data


def execute_authorized(preparation,config,authorization):
 """Future explicit entry only. Cannot work with the sealed false template."""
 prep=safe_path(preparation);require(config['scope']['stage']=='E02','real stage only')
 gate=LiveGate(prep,config,authorization)
 from provider import Counter
 counter=Counter(prep.parents[2],read_json(prep/'config/counter_assets.json'))
 require(authorization.get('batch_output')==config['batch_output'],'exact unique output path')
 directory=safe_path(config['batch_output'],prep.parents[2]);safe_path(directory.parent).mkdir(parents=True,exist_ok=True)
 slots=read_json(prep/'config/slots.json')
 # New exact directory and ledger O_EXCL atomically consume this sole output identity.
 ledger=Ledger(directory,config,slots,directory.parent,gate=gate)
 write_new(directory/'authorization.consumed.json',{'authorization_sha256':digest(authorization),'no_reuse':True},directory)
 transport=None
 try:
  transport=LiveHTTP(gate,config)
  provider=Provider(config,counter,ledger,transport,directory/'raw')
  import json
  qs=[json.loads(x) for x in (prep/'inputs/business_requests.jsonl').read_text().splitlines()]
  bs=read_json(prep/'inputs/business_bindings.json')['bindings'];rubrics=read_json(prep/'inputs/rubrics.json')['questions'];cards=read_json(prep/'evaluation_only/cards.json')
  all_results=[]
  for i,(b,q) in enumerate(zip(bs,qs)):
   r=next(x for x in rubrics if x['question_id']==b['question_id'])
   result=cell(provider,b['question_id']+'_L_R01','N-%03d'%i,q,r,cards,directory)
   all_results.append((b,result))
   require(not ledger.broken,'batch stopped after uncertain transport/usage/recording')
  fixed_macros={name:family_macro([{'split':b['split'],'family_id':b['family_id'],'ratio':v['metrics'][name]} for b,v in all_results]) for name in ['REF_cov','REF_knowledge','REF_boundary','REF_source_state']}
  write_new(directory/'reference_report.json',{'fixed_macros':fixed_macros,'planned_questions':13,'observed_results':len(all_results),'dynamic_claim_and_citation_metrics':'per-cell with unknown denominator retained; no aggregate when coding/applicability is unknown','old38_applicability':read_json(prep/'analysis/metric_applicability_38.json'),'attempt_counts':ledger.counts,'full_cap_reserved_USD':str(ledger.reserved),'actual_bill':'unknown','status':'executed only if future separately authorized; preparation contains no real results'},directory)
 finally:
  ledger.close()
  if transport:transport.close()
