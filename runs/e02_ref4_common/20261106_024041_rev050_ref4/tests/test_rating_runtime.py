"""SYNTHETIC transport integration. Never instantiate LiveHTTP or the true output."""
from pathlib import Path
import copy
import json
import subprocess
import sys
import tempfile
import unittest

R=Path(__file__).absolute().parents[1];P=R.parents[2]
sys.path.insert(0,str(R/'runtime'))
import quote_anchor as qa
import workflow
from common import digest,read_json,Rejected
from provider import Counter,Ledger,Provider


def fixture(n=13):
    config=read_json(R/'config/execution.json');config['scope']['stage']='SYNTHETIC'
    sources=[];packets={};rubrics={};slots={};binds={}
    for i in range(n):
        qid='SYN-Q%02d'%i;unit='N-SYN-%02d'%i
        points=[{'point_id':'SYN-P1','content_role':'positive_knowledge','half_credit_eligible':False}]
        packet={'unit_id':unit,'business_request':{'question':'SYNTHETIC: safe fixture %d'%i},'common_context':{},
                'reference_points':points,'reference_material':[],'observed_text':'SYNTHETIC: Evidence is visible. The scope is limited.'}
        packet,binding=qa.prepare_packet(packet);packets[unit]=packet
        sources.append({'unit_id':unit,'question_id':qid,'family_id':'SYN-F%02d'%(i%9),'split':'debug' if i<11 else 'operational_reserve',
                        'binding':binding,'source_result_sha256':digest({'SYNTHETIC':i})})
        rubrics[qid]={'points':points}
        for label,action in [('A','judge_output'),('B','judge_output'),('C','adjudicate_output')]:
            slot=qid+'_L_REF047_'+label;slots[slot]=action
            binds[slot]=dict(binding,action=action,label=label,synthetic_packet=packet)
    config['slot_bindings']=binds;config['allowed_slots_sha256']=digest(slots)
    return config,sources,packets,rubrics,slots


def wire(packet):
    quote={'span_ids':[packet['quote_spans'][0]['span_id']]}
    return {'schema_version':qa.WIRE_VERSION,'points':[{'point_id':'SYN-P1','score':1,'quotes':[quote],'reason':'SYNTHETIC'}],
            'claims':[{'claim_id':'SYN-C1','quote':quote,'support':'full','reason':'SYNTHETIC'}],
            'voluntary_citations':[],'complete_claim_inventory':True,'complete_citation_inventory':True}


class SyntheticTransport:
    def __init__(self,counter,mode='normal'):
        self.counter=counter;self.mode=mode;self.calls=0;self.last_post_invoked=False
    def __call__(self,body):
        self.calls+=1;self.last_post_invoked=True
        payload=json.loads(body['messages'][1]['content']);packet=payload.get('packet',payload)
        if self.mode=='timeout':raise Rejected('transport timeout; no retry')
        value=wire(packet)
        if self.mode=='bad_quote' and self.calls==1:value['points'][0]['quotes'][0]['span_ids']=['S000000000000-9999']
        usage={'prompt_tokens':self.counter.count(body['messages'])['input_tokens'],'completion_tokens':10}
        usage['total_tokens']=usage['prompt_tokens']+10
        response={'id':'SYNTHETIC-%03d'%self.calls,'model':'deepseek-flash','usage':usage,
                  'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(value)}}]}
        if self.mode=='alias':response['model']='SYNTHETIC-WRONG-ALIAS'
        if self.mode=='missing_usage':response.pop('usage')
        if self.mode=='usage_arithmetic':usage['total_tokens']+=1
        if self.mode=='thinking':response['choices'][0]['message']['reasoning_content']='SYNTHETIC unexpected mode'
        if self.mode=='cache_negative':usage.update({'prompt_cache_hit_tokens':-1,'prompt_cache_miss_tokens':usage['prompt_tokens']+1})
        if self.mode=='cache_sum':usage.update({'prompt_cache_hit_tokens':1,'prompt_cache_miss_tokens':usage['prompt_tokens']})
        if self.mode=='reasoning_bool':usage['completion_tokens_details']={'reasoning_tokens':False}
        if self.mode=='details_list':usage['completion_tokens_details']=[]
        if self.mode=='new_billed_dimension':usage['web_search_requests']=1
        if self.mode=='tools':response['choices'][0]['message']['tool_calls']=[{'id':'SYNTHETIC-tool','type':'function'}]
        if self.mode=='multiple_choices':response['choices']*=2
        if self.mode=='model_bad_json' and self.calls==1:response['choices'][0]['message']['content']='{'
        if self.mode=='model_duplicate_json' and self.calls==1:response['choices'][0]['message']['content']='{"x":1,"x":2}'
        if self.mode=='model_nan_json' and self.calls==1:response['choices'][0]['message']['content']='{"x":NaN}'
        if self.mode=='finish_length' and self.calls==1:response['choices'][0]['finish_reason']='length'
        if self.mode=='finish_unknown':response['choices'][0]['finish_reason']='SYNTHETIC-UNKNOWN'
        if self.mode=='finish_tool':response['choices'][0]['finish_reason']='tool_calls'
        if self.mode=='bad_http_JSON':return {'http_status':200,'request_id':'SYNTHETIC','raw_body':b'{'}
        return {'http_status':200,'request_id':'SYNTHETIC-REQUEST','raw_body':json.dumps(response).encode()}


def setup_case(mode='normal',n=13,change=None):
    config,sources,packets,rubrics,slots=fixture(n)
    counter=Counter(P,read_json(R/'config/counter_assets.json'))
    if change:change(config,counter,sources,packets)
    base=R/'validation/author_cases';base.mkdir(exist_ok=True)
    parent=Path(tempfile.mkdtemp(prefix='SYNTHETIC_',dir=base));directory=parent/'SYNTHETIC-batch'
    ledger=Ledger(directory,config,slots,parent)
    transport=SyntheticTransport(counter,mode);provider=Provider(config,counter,ledger,transport,directory/'raw')
    return provider,transport,sources,packets,rubrics,directory


def run_case(mode='normal',change=None,n=13):
    p,t,s,packets,rubrics,d=setup_case(mode,n,change)
    try:report=workflow.run_planned(p,s,lambda x:packets[x['unit_id']],rubrics,d)
    finally:p.ledger.close()
    return p,t,report,d


class RuntimeTests(unittest.TestCase):
    def test_full39_calls_through_raw_anchor_primary_and_third(self):
        p,t,r,d=run_case()
        self.assertEqual(t.calls,39);self.assertEqual(r['valid_final_ratings'],13)
        self.assertEqual(r['attempt_counts'],{'judge_output':26,'adjudicate_output':13})
        self.assertEqual(r['reserved_USD'],'1.1501568');self.assertFalse(r['batch_stopped'])
        self.assertEqual(len(list(d.glob('*.result.json'))),13)
        self.assertEqual(len(list((d/'ratings').glob('*.original.json'))),39)
        self.assertEqual(len(list((d/'raw').glob('*.request.json'))),39)
        self.assertTrue(all(read_json(f)['transport_kind']=='injected_synthetic_transport' for f in (d/'raw').glob('*.request.json')))
    def test_invalid_quote_keeps_raw_and_uses_no_extra_slot(self):
        p,t,r,d=run_case('bad_quote',n=1)
        data=read_json(next(d.glob('*.result.json')))
        self.assertEqual(data['original_ratings'][0]['status'],'evaluator_failure')
        self.assertIsNone(data['original_ratings'][0]['rating']);self.assertIsNotNone(data['original_ratings'][0]['raw_rating'])
        self.assertEqual(t.calls,3);self.assertEqual(r['valid_final_ratings'],1)
    def test_initial_input_N_minus1_stops_all13_without_transport(self):
        def change(c,counter,s,p):c['actions']['judge_output']['input_cap']=counter.count(qa.messages('judge_output',p[s[0]['unit_id']]))['input_tokens']-1
        p,t,r,d=run_case(change=change)
        self.assertEqual(t.calls,0);self.assertTrue(r['batch_stopped']);self.assertEqual(r['planned_questions'],13)
        self.assertEqual(r['valid_final_ratings'],0);self.assertEqual(len(list(d.glob('*.result.json'))),13)
        self.assertEqual(p.ledger.reserved,0)
    def test_initial_input_N_boundary_is_accepted(self):
        def change(c,counter,s,p):c['actions']['judge_output']['input_cap']=counter.count(qa.messages('judge_output',p[s[0]['unit_id']]))['input_tokens']
        p,t,r,d=run_case(change=change,n=1)
        self.assertEqual(t.calls,3);self.assertFalse(r['batch_stopped'])
    def test_third_full_AB_N_minus1_stops_after_two_calls(self):
        def change(c,counter,s,packets):
            packet=packets[s[0]['unit_id']];raw=wire(packet);rating=qa.validate_rating(raw,packet,s[0]['binding'])['rating']
            primary={'status':'valid','rating':rating,'source_binding':s[0]['binding']}
            body=qa.messages('adjudicate_output',qa.third_packet(packet,primary,primary))
            c['actions']['adjudicate_output']['input_cap']=counter.count(body)['input_tokens']-1
        p,t,r,d=run_case(change=change)
        self.assertEqual(t.calls,2);self.assertTrue(r['batch_stopped']);self.assertEqual(r['attempt_counts'],{'judge_output':2})
        self.assertEqual(r['planned_questions'],13)
    def test_unknown_alias_stops_and_reservation_is_retained(self):
        p,t,r,d=run_case('alias');self.assertEqual(t.calls,1);self.assertTrue(r['batch_stopped'])
        self.assertEqual(r['reserved_USD'],'0.024576');self.assertEqual(len(list((d/'raw').glob('*.http.json'))),1)
    def test_unknown_usage_stops_all_later_slots(self):
        p,t,r,d=run_case('missing_usage');self.assertEqual(t.calls,1);self.assertTrue(r['batch_stopped'])
    def test_usage_arithmetic_stops_all_later_slots(self):
        p,t,r,d=run_case('usage_arithmetic');self.assertEqual(t.calls,1);self.assertTrue(r['batch_stopped'])
    def test_unexpected_thinking_stops_all_later_slots(self):
        p,t,r,d=run_case('thinking');self.assertEqual(t.calls,1);self.assertTrue(r['batch_stopped'])
    def test_timeout_does_not_retry(self):
        p,t,r,d=run_case('timeout');self.assertEqual(t.calls,1);self.assertTrue(r['batch_stopped'])
        self.assertEqual(r['reserved_USD'],'0.024576')
    def test_counter_exception_stops_before_any_reservation(self):
        def change(c,counter,s,p):
            def fail(messages):raise Rejected('SYNTHETIC counter failure')
            counter.count=fail
        p,t,r,d=run_case(change=change);self.assertEqual(t.calls,0);self.assertTrue(r['batch_stopped']);self.assertEqual(p.ledger.reserved,0)
    def test_money_rejection_stops_without_refund_or_transport(self):
        def change(c,counter,s,p):c['proposed_hard_budget_USD']='0.001'
        p,t,r,d=run_case(change=change);self.assertEqual(t.calls,0);self.assertTrue(r['batch_stopped'])
    def test_tampered_source_binding_stops_before_transport(self):
        def change(c,counter,s,p):c['slot_bindings'][s[0]['question_id']+'_L_REF047_A']['observed_text_sha256']='0'*64
        p,t,r,d=run_case(change=change);self.assertEqual(t.calls,0);self.assertTrue(r['batch_stopped'])
    def test_third_cannot_substitute_new_primary_grades(self):
        p,t,s,packets,rubrics,d=setup_case(n=1)
        try:
            packet=packets[s[0]['unit_id']];rows=[]
            for label in ['A','B']:
                slot=s[0]['question_id']+'_L_REF047_'+label;raw,meta=p.send(slot,'judge_output',qa.messages('judge_output',packet))
                rating=qa.validate_rating(raw,packet,s[0]['binding'])['rating']
                row={'status':'valid','rating':rating,'raw_rating':raw,'source_binding':s[0]['binding']}
                p.record_primary(slot,row);rows.append(row)
            changed=copy.deepcopy(rows[0]);changed['rating']['points'][0]['score']=0
            payload=qa.third_packet(packet,*[{k:x[k] for k in ['status','rating','source_binding']} for x in [changed,rows[1]]])
            with self.assertRaisesRegex(Rejected,'differs from recorded'):p.send(s[0]['question_id']+'_L_REF047_C','adjudicate_output',qa.messages('adjudicate_output',payload))
            self.assertEqual(t.calls,2);self.assertTrue(p.ledger.broken)
        finally:p.ledger.close()
    def test_durable_original_failure_stops_and_preserves_raw(self):
        original=workflow.write_new
        def fail(path,value,root):
            if str(path).endswith('.original.json'):raise OSError('SYNTHETIC durable failure')
            return original(path,value,root)
        workflow.write_new=fail
        try:p,t,r,d=run_case()
        finally:workflow.write_new=original
        self.assertEqual(t.calls,1);self.assertTrue(r['batch_stopped']);self.assertEqual(r['planned_questions'],13)
        self.assertEqual(len(list((d/'raw').glob('*.response.json'))),1)
    def test_default_and_optimized_entry_never_execute(self):
        for flags in [[],['-O']]:
            result=subprocess.run([sys.executable,'-B',*flags,str(R/'execute_approved.py')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
        self.assertFalse(Path(read_json(R/'config/execution.json')['batch_output']).exists())
    def test_billing_or_profile_extensions_stop_after_one_attempt(self):
        for mode in ['cache_negative','cache_sum','reasoning_bool','details_list','new_billed_dimension','tools','multiple_choices','finish_unknown','finish_tool','bad_http_JSON']:
            with self.subTest(mode=mode):
                p,t,r,d=run_case(mode,n=1)
                self.assertEqual(t.calls,1);self.assertTrue(r['batch_stopped'])
    def test_bad_model_JSON_is_null_coding_not_unknown_usage(self):
        for mode in ['model_bad_json','model_duplicate_json','model_nan_json','finish_length']:
            with self.subTest(mode=mode):
                p,t,r,d=run_case(mode,n=1)
                self.assertEqual(t.calls,3);self.assertFalse(r['batch_stopped']);self.assertEqual(r['valid_final_ratings'],1)
                first=read_json(next(d.glob('*.result.json')))['original_ratings'][0]
                self.assertEqual(first['status'],'evaluator_failure');self.assertIsNone(first['rating'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeTests))
    print(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'success':result.wasSuccessful(),
                      'transport':'injected SYNTHETIC only','real_model_API_calls':0,'true_output_created':False},ensure_ascii=False,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
