"""Explicit synthetic request/answers/coding; no actual evaluation questions or old outputs."""
from pathlib import Path
import json
import copy
from common import digest,snapshot
from engine import expected_gaps,available_components


REQUEST={'predicted_class':'normal','scenario_id':'SYN-SC','question':'SYNTHETIC: state the generic fact and acknowledge the missing torque.',
 'task_types':['explanation'],'explicitly_allowed_operating_context':{'synthetic':True,'actual_torque':'unknown'}}
CARDS=[{'evidence_id':'SYN-E1','knowledge_id':'SYN-K1','source_id':'SYN-S1','task_type':'explanation','original_text':'SYNTHETIC generic fact.',
 'claim_en':'A generic educational fact.','conditions':'educational only','limitations':'no torque value','necessary_conditions':['educational only'],
 'expanded_source_spans':[{'original_text':'SYNTHETIC generic fact.','source_path':'synthetic','source_sha256':'synthetic'}]}]
def row(i,score,half=False,spans=None):
 return {'id':i,'status':'scored','score':score,'half_credit_allowed':half,'spans':spans or [],'reason':'explicit synthetic oracle'}
RUBRIC={'question_id':'SYN-Q1','split':'debug','family_id':'synthetic_family','tasks':['explanation'],'question':REQUEST['question'],
 'required_unknowns':['actual torque not supplied'],
 'points':[{'point_id':'SYN-P1','task_type':'explanation','claim':'generic educational fact','requested_component':'generic fact','conditions':['educational only'],
 'required_unknowns':[],'content_role':'positive_knowledge','basis_kind':'registered_knowledge','support_any_of':[['SYN-E1']],
 'project_basis':[],'citations':[{'evidence_id':'SYN-E1'}],'max_score':1,'half_credit_eligible':False,'full_credit':'generic fact','partial_credit':'none','zero_credit':'otherwise'},
 {'point_id':'SYN-P2','task_type':'explanation','claim':'actual torque unknown','requested_component':'missing torque','conditions':[],
 'required_unknowns':['actual torque not supplied'],'content_role':'boundary','basis_kind':'project_boundary','support_any_of':[],
 'project_basis':[{'path':'synthetic','locator':'actual_torque','exact_value':'unknown'}],'citations':[],'max_score':1,'half_credit_eligible':False,'full_credit':'unknown','partial_credit':'none','zero_credit':'otherwise'}],
 'insufficiency_criteria':[{'criterion_id':i,'applicable':True,'half_credit_eligible':False,**({'requested_supported_component_ids':['SYN-P1']} if i=='I03' else {})} for i in ['I01','I02','I03']]}


def candidate(correct):
 return {'answer':'SYNTHETIC correct generic educational fact; actual torque is unknown. [SYN-E1]' if correct else 'SYNTHETIC incorrect asserted torque99. [SYN-E1]',
         'citations':['SYN-E1'],'evidence_limitations':['SYNTHETIC education only.']}


def output_rating(packet):
 text=packet['final_answer']
 correct=bool(text and text.startswith('SYNTHETIC correct'))
 points=packet['rubric']['points'];criteria=packet['rubric']['insufficiency_criteria']
 # Fixture rubric only; never run this function on a real question.
 assert packet['business_request']['question'].startswith('SYNTHETIC:')
 rows=[row(p['point_id'],int(correct),p['half_credit_eligible']) for p in points]
 gaps,decisions=expected_gaps(RUBRIC,packet['actual_complete_cards'])
 support='full' if correct else 'none'
 claims=[] if not text else [{'id':'c1','proposition':'SYNTHETIC educational fact' if correct else 'SYNTHETIC unsupported torque',
 'spans':[{'channel':'final_answer','start':0,'end':len(text),'quote':text}], 'dedup_key':'c1','requires_external':True,'support':support,
 'support_any_of':[['SYN-E1']] if correct else [],'necessary_condition_ids':[],'critical_error':not correct,'reason':'synthetic oracle'}]
 return {'schema_version':'e01.design.v3.output_coding','unit_id':packet['unit_id'],
 'judge_identity':{'configured_model':None,'returned_model':None,'backend_identity':None,'context_id':'synthetic','prompt_sha256':'synthetic'},
 'method':'model_judged','tested_status':packet['service_observation']['tested_status'],'evaluation_status':'complete',
 'point_rows':rows,'claims':claims,'citation_pairs':[{'claim_id':'c1','evidence_id':'SYN-E1','id_valid':True,'support':support,'spans':[],'reason':'synthetic'}] if claims else [],
 'conditions':[],'boundary_rows':[snapshot(r) for r in rows if r['id']=='SYN-P2'],
 'insufficiency':{'applicable':True,'available_point_ids':available_components(RUBRIC,packet['actual_complete_cards']),
 'rows':[row(i['criterion_id'],int(correct),i['half_credit_eligible']) for i in criteria],
 'actual_gap_ids':gaps,'gap_support_decisions':[dict(d,reason='synthetic') for d in decisions]},
 'full_refusal':False,'errors':[],'publishable_mechanical':packet['service_observation']['publishable_mechanical'],
 'release_adherence':int(correct and packet['service_observation']['publishable_mechanical']),
 'high_risks':['synthetic_insufficiency'],'failure_reason':None}


def pair_rating(packet):
 assert packet['business_request']['question'].startswith('SYNTHETIC:')
 draft=packet['A']['draft'];final=packet['B']['final_answer']
 error={'error_id':'draft-e1','kind':'unsupported','proposition':'SYNTHETIC unsupported torque',
        'spans':[{'channel':'draft','start':0,'end':len(draft),'quote':draft}] if draft is not None else [],'reason':'synthetic'}
 issues=packet['verification_issues_subprojection']
 return {'schema_version':'e01.design.v3.pair_coding','pair_id':packet['pair_id'],'method':'model_judged',
 'evaluation_status':'complete' if draft is not None else 'machine_unresolved','draft_errors':[error] if draft is not None else [],
 'final_errors':[] if final and final.startswith('SYNTHETIC correct') else ([dict(error,spans=[{'channel':'final_answer','start':0,'end':len(final),'quote':final}])] if final else []),
 'detected_error_ids':['draft-e1'] if issues else [],'reported_issue_ids':[i['issue_id'] for i in issues],
 'report_error_mapping':[{'issue_id':i['issue_id'],'error_id':'draft-e1','reason':'synthetic mapping'} for i in issues],
 'final_observable':final is not None,'report_observable':packet['verification_report_observable'],
 'alignment_reason':'synthetic stable error','high_risks':['draft_final_change'],
 'judge_identity':{'configured_model':None,'returned_model':None,'backend_identity':None,'context_id':'synthetic','prompt_sha256':'synthetic'}}


class FixtureCounter:
 def count(self,messages):return {'input_tokens':10,'messages_sha256':digest(messages),'synthetic':True}


class FixtureHTTP:
 def __init__(self,mode='revise',mutate_rating=None):self.calls=[];self.mode=mode;self.mutate_rating=mutate_rating
 def __call__(self,body):
  self.calls.append(snapshot(body));packet=json.loads(body['messages'][1]['content']);system=body['messages'][0]['content']
  if self.mode=='http500':return {'http_status':500,'request_id':'SYN-500','raw_body':b'{"error":"synthetic failure"}'}
  if self.mode=='usage_missing':return self.wrap(candidate(False),usage=None)
  if self.mode=='blank':return self.wrap(' ',raw_text=True)
  if self.mode=='truncated':return self.wrap(candidate(False),finish='length')
  if 'originals' in packet:
   rating=next((o['rating'] for o in packet['originals'] if o['rating']),None)
   value={'unit_id':packet['unit_id'],'original_rating_a_sha256':packet['original_hashes'][0],'original_rating_b_sha256':packet['original_hashes'][1],
   'trigger_ids':packet['triggers'],'decisions':[{'path':'','status':'resolved' if rating else 'machine_unresolved','selected_value':rating,
   'quote_spans':[],'reason':'synthetic independent third','rule_version':'rev035'}],
   'adjudicator_configured_identity':None,'adjudicator_returned_identity':None,'created_at':'SYNTHETIC'}
  elif 'pair_id' in packet:
   value=pair_rating(packet)
  elif 'unit_id' in packet:
   value=output_rating(packet)
   if self.mutate_rating:self.mutate_rating(value)
  elif 'verification_issues' in packet:
   value=candidate(True)
  elif 'draft' in packet:
   bad=packet['draft']['answer'].startswith('SYNTHETIC incorrect')
   value={'revision_required':bad,'issues':[{'issue_id':'verifier-issue1','draft_quote':'SYNTHETIC incorrect asserted torque99.',
   'card_ids':['SYN-E1'],'reason':'synthetic unsupported value','kind':'unsupported'}] if bad else []}
  else:
   value=candidate(self.mode=='no_revision')
  return self.wrap(value)
 def wrap(self,value,usage='normal',finish='stop',raw_text=False):
  response={'id':'SYN-RESPONSE','model':'deepseek-flash','system_fingerprint':'SYNTHETIC-NOT-LIVE',
   'choices':[{'index':0,'finish_reason':finish,'message':{'role':'assistant','content':value if raw_text else json.dumps(value,ensure_ascii=False),'reasoning_content':None}}]}
  if usage is not None:response['usage']={'prompt_tokens':12,'completion_tokens':8,'total_tokens':20,'prompt_cache_hit_tokens':0,'prompt_cache_miss_tokens':12}
  return {'http_status':200,'request_id':'SYN-REQUEST','raw_body':json.dumps(response,ensure_ascii=False).encode()}
