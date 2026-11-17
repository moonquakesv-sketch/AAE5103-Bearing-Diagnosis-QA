"""Shared-context no-retrieval packets, reference-only coding and fixed-denominator math."""
from pathlib import Path
import re
import jsonschema
from common import canonical,digest,snapshot,require,read_json,safe_path,BUSINESS_FIELDS
ROOT=Path(__file__).absolute().parents[1]

def messages(role,packet,root=ROOT):
 name='generation' if role=='generate' else ('judge' if role=='judge_output' else 'adjudicator')
 require(role in ['generate','judge_output','adjudicate_output'],'unknown action')
 schema=read_json(root/'schemas'/('generation.json' if role=='generate' else 'reference_rating.json'))
 system=(root/'prompts'/(name+'.txt')).read_text()+'\nReturn strict JSON following this complete schema:\n'+canonical(schema)
 return [{'role':'system','content':system},{'role':'user','content':canonical(packet)}]

def generation_packet(request):
 require(type(request) is dict and set(request)==BUSINESS_FIELDS,'five business fields only')
 return {'business_request':snapshot(request),'common_context':snapshot(request['explicitly_allowed_operating_context'])}

def evaluation_packet(unit,request,rubric,reference_cards,observed_text):
 require(type(observed_text) is str,'actual observed service text required')
 # Full reference is evaluator-only; no retrieval, ranking or injection availability inference.
 points=[{k:v for k,v in p.items() if k not in ['citations']} for p in rubric['points']]
 return dict(generation_packet(request),unit_id=unit,reference_points=points,reference_material=snapshot(reference_cards),observed_text=observed_text)

def validate_messages(config,action,msgs,binding):
 from common import strict_json
 require(type(msgs) is list and len(msgs)==2,'fresh system/user only')
 packet=strict_json(msgs[1]['content']); core=packet['packet'] if action=='adjudicate_output' else packet
 request=core['business_request'];require(digest(request)==binding['business_sha256'],'slot/business identity mismatch')
 require(core['common_context']==request['explicitly_allowed_operating_context'],'context mismatch')
 if config['scope']['stage']!='E02':
  require(config['scope']['stage']=='SYNTHETIC' and request['question'].startswith('SYNTHETIC:'),'fixture must be SYNTHETIC')
 else:
  reqs=[__import__('json').loads(x) for x in (ROOT/'inputs/business_requests.jsonl').read_text().splitlines()]
  require(request in reqs,'request is not frozen')
 if action=='generate':
  require(packet==generation_packet(request),'generation evidence/metadata leakage')
  if config['scope']['stage']=='E02':
   import json
   binds=read_json(ROOT/'inputs/business_bindings.json')['bindings']
   qid=next(b['question_id'] for b in binds if b['business_line_1_based']==reqs.index(request)+1)
   require(canonical(msgs).encode('utf-8')==(ROOT/'serialized_generation'/(qid+'.messages.json')).read_bytes(),'serialized generation byte identity drift')
 else:
  require(set(core)=={'business_request','common_context','unit_id','reference_points','reference_material','observed_text'},'scorer projection keys')
  require(type(core['unit_id']) is str and core['unit_id'].startswith('N-'),'anonymous unit ID')
  if config['scope']['stage']=='E02':
   rubrics=read_json(ROOT/'inputs/rubrics.json')['questions']
   rubric=next(r for r in rubrics if r['question']==request['question'])
   expected=evaluation_packet(core['unit_id'],request,rubric,read_json(ROOT/'evaluation_only/cards.json'),core['observed_text'])
   require(core==expected,'reference/rubric injection changed')
  if action=='adjudicate_output':require(set(packet)=={'packet','A','B'},'third projection keys')
 require(msgs==messages(action,packet),'wire prompt/schema identity changed')
 return True

def validate_candidate(candidate):
 jsonschema.Draft202012Validator(read_json(ROOT/'schemas/generation.json')).validate(candidate)
 require(bool(candidate['answer'].strip()),'blank answer')
 require(len(candidate['citations'])==len(set(candidate['citations'])),'duplicate citations')
 text=candidate['answer']+'\n'+'\n'.join(candidate['evidence_limitations'])
 if candidate['citations']:text+='\nSources:\n'+'\n'.join(candidate['citations'])
 # No internal IDs were supplied. Retain raw response as a fabrication diagnostic.
 invented=bool(re.search(r'EV-[A-Za-z0-9-]+|SYN-E[0-9]+',text+'\n'+'\n'.join(candidate['citations'])))
 return {'publishable':not invented,'citation_policy_error':invented,'observed_text':text if not invented else 'The service withheld this response because it claimed unsupplied evidence IDs.','raw_diagnostic_text':text}

def validate_rating(value,packet):
 jsonschema.Draft202012Validator(read_json(ROOT/'schemas/reference_rating.json')).validate(value)
 text=packet['observed_text']; points={p['point_id']:p for p in packet['reference_points']}
 require(len(value['points'])==len(points) and {p['point_id'] for p in value['points']}==set(points),'complete fixed points required')
 def quote(q):
  require(q['start']<q['end']<=len(text) and text[q['start']:q['end']]==q['text'] and bool(q['text'].strip()),'quote offsets/verbatim mismatch')
 for row in value['points']:
  p=points[row['point_id']]
  if row['score']==0.5:require(p.get('half_credit_eligible') is True,'half credit not eligible')
  require(row['score'] in [None,0] or bool(row['quotes']),'positive point needs visible quote')
  for q in row['quotes']:quote(q)
  require(bool(row['reason'].strip()),'reason missing')
 for category,key in [('claims','claim_id'),('voluntary_citations','citation_id')]:
  require(len({r[key] for r in value[category]})==len(value[category]),'duplicate coding ID')
  for row in value[category]:quote(row['quote']);require(bool(row['reason'].strip()),'reason missing')
 return value

def coding(raw,packet):
 from common import strict_json
 try:return {'status':'valid','rating':validate_rating(strict_json(raw),packet),'raw':raw}
 except Exception as e:return {'status':'evaluator_failure','rating':None,'raw':raw,'error_type':type(e).__name__}

def resolve(a,b,third=None):
 # All cells have source/condition/unknown risks: third independently full recodes all13.
 if third and third.get('status')=='valid':return {'status':'resolved_third','rating':third['rating']}
 return {'status':'evaluator_failure','rating':None,'primary_agreement_diagnostic':a.get('status')=='valid' and b.get('status')=='valid' and a.get('rating')==b.get('rating')}

def fixed_ratio(scores):
 require(all(x in [None,0,0.5,1] and type(x) is not bool for x in scores),'invalid score')
 if not scores:return {'status':'not_applicable','value':None,'denominator':0,'lower':None,'upper':None}
 known=sum(x for x in scores if x is not None);unknown=scores.count(None);n=len(scores)
 return {'status':'unknown' if unknown else 'known','value':None if unknown else known/n,'numerator':known,'denominator':n,'unknown':unknown,'lower':known/n,'upper':(known+unknown)/n}

def family_macro(rows):
 # rows explicitly contain one planned question ratio, including unknown and N/A.
 result={}
 for split in sorted({r['split'] for r in rows}):
  fam={}
  for row in rows:
   if row['split']==split:fam.setdefault(row['family_id'],[]).append(row['ratio'])
  bounds=[]; na=0
  for values in fam.values():
   applicable=[v for v in values if v['status']!='not_applicable'];na+=len(values)-len(applicable)
   if applicable:bounds.append((sum(v['lower'] for v in applicable)/len(applicable),sum(v['upper'] for v in applicable)/len(applicable)))
  result[split]={'families':len(bounds),'planned_families':len(fam),'not_applicable_questions':na,'lower':sum(v[0] for v in bounds)/len(bounds) if bounds else None,'upper':sum(v[1] for v in bounds)/len(bounds) if bounds else None}
 return result
