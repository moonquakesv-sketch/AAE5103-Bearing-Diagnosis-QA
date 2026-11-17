"""Precommitted schema/control fixtures, not a semantic accuracy evaluation."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import httpx
import q01_flow as flow
import q01_intent as intent
import q01_intent_http as wire
import test_q01_flow as fixtures
from q01_token_counter import DeepSeekV41Counter

SECRET='SYNTHETIC_INTENT_SECRET'

def payload():
    return {'predicted_class':'outer_race','scenario_id':'SC-K01-OR-001',
            'question':'SYNTHETIC abstract task question.','explicitly_allowed_operating_context':{}}

def report():return {'status':'identified','task_types':['inspection'],'message':None}

def official(value=None,*,text=None,finish='stop',refusal=None):
    if value is None:value=report()
    message={'role':'assistant','content':json.dumps(value) if text is None else text}
    if refusal is not None:message['refusal']=refusal
    return {'status_code':200,'body':json.dumps({'id':'SYN-ID','model':'deepseek-flash',
      'usage':{'prompt_tokens':20,'completion_tokens':15,'total_tokens':35},
      'choices':[{'index':0,'message':message,'finish_reason':finish}]}),'headers':{'x-request-id':'SYN-REQUEST','Authorization':'Bearer '+SECRET}}

class Untouched:
    def __getattribute__(self,name):raise AssertionError('Blocked dispatch accessed '+name)

class Counter:
    def __init__(self,n=20):self.n=n
    @property
    def identity(self):return {'kind':'http_mock_token_counter','scope':'complete_chat_messages',
      'verified_models':['deepseek-flash'],'chat_template_sha256':'1'*64,'artifact_sha256':'2'*64}
    def count_messages(self,messages,model):return {'input_tokens':self.n,'model':model,
      'messages_sha256':flow.digest(messages),'chat_template_sha256':'1'*64}

class FakeGate:
    def __init__(self,folder,projection,qa_budget=None):
        self.run=folder;self.projection=projection;self.changed=False
        self._release_bytes=b'EXPLICIT_SYNTHETIC_MASTER'
        self.release={'qa_budget':qa_budget,'four_fields_sha256':flow.digest(projection),'file_hashes':{p:flow.file_hash(ROOT/p) for p in wire.CODE_FILES}}
        self.bound={}
    def require_frozen(self):
        if self.changed:raise intent.Error('freeze_changed','SYNTHETIC changed freeze')
    def require_derived(self):
        self.require_frozen()
        if any(flow.file_hash(self.run/p)!=h for p,h in self.bound.items()):raise intent.Error('recognition_binding_invalid','SYNTHETIC derived bytes changed')

class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage_run=Path(os.environ['Q01_INTENT_RUN_DIR']).absolute()
        if not cls.stage_run.is_relative_to(ROOT/'runs/q01_intent') or any(p.is_symlink() for p in [cls.stage_run,*cls.stage_run.parents]):raise ValueError('New unlinked intent run required.')
        cls.out=Path(tempfile.mkdtemp(prefix='contract_',dir=cls.stage_run/'outputs'))
        cls.profile=wire.read(ROOT/'configs/q01_deepseek.json');cls.cfg=wire.read(ROOT/'configs/q01_intent.json');cls.prompts=wire.read(ROOT/'configs/q01_intent_prompts.json')
        cls.obs=[]
    @classmethod
    def tearDownClass(cls):
        flow.write_record(cls.out/'observations.json',{'kind':'synthetic_contract_only','records':cls.obs,
          'source_hashes':{p:flow.file_hash(ROOT/p) for p in wire.CODE_FILES|wire.CONFIG_FILES},
          'actual_official_requests':0,'note':'Any real_provider metadata in gate fixtures is synthetic branch data, never a service result.'})
    def save(self,name,value):
        self.obs.append({'case':name,'value':value});flow.write_record(self.out/('case_'+name+'.json'),value)
    def router(self,exchange):return intent.ModelTaskRouter(self.profile,self.cfg,self.prompts,exchange,redact_values=(SECRET,))

    def test_frozen_model_output_cases(self):
        cases=wire.read(self.stage_run/'verification/expected_model_cases.json')['cases']
        for case in cases:
            calls=[]
            def exchange(endpoint,body,settings):calls.append(body);return official(case['output'])
            router=self.router(exchange);value=router.prepare_request(payload());self.save(case['case'],{'route':value,'provider':router.snapshot()})
            self.assertEqual(value['status'],case['status']);self.assertEqual(value['error_code'],case['error_code']);self.assertEqual(len(calls),1)
            data=flow.k02.strict_json(calls[0]['messages'][1]['content']);self.assertEqual(data,{'question':payload()['question']})
            if value['status']=='ready':
                self.assertEqual(value['model_report'],case['output']);self.assertEqual(value['request']['task_types'],[t for t in flow.k02.TASKS if t in case['output']['task_types']])
                self.assertEqual({k:value['request'][k] for k in intent.rules.INPUT_FIELDS},payload())
            else:self.assertIsNone(value['request'])

    def test_frozen_failures_consume_one_without_retry(self):
        expected=wire.read(self.stage_run/'verification/expected_failure_cases.json')['cases']
        for name,code in expected.items():
            calls=[]
            def exchange(endpoint,body,settings):
                calls.append(body)
                if name=='timeout':raise TimeoutError('SYNTHETIC timeout '+SECRET)
                if name=='rate_limit':return {'status_code':429,'body':'SYNTHETIC rate limit'}
                if name=='service_error':return {'status_code':503,'body':'SYNTHETIC service failure'}
                if name=='refusal':return official(refusal='SYNTHETIC refusal')
                if name=='truncation':return official(finish='length')
                if name=='empty_output':return official(text='')
                if name=='invalid_json':return official(text='```json\n{}\n```')
                if name=='duplicate_json_key':return official(text='{"status":"identified","status":"unsupported"}')
                if name=='unrecordable_response':return {'status_code':200,'body':object()}
                if name=='tool_calls':
                    v=official();b=json.loads(v['body']);b['choices'][0]['message']['tool_calls']=[{'id':'SYN'}];v['body']=json.dumps(b);return v
                if name=='multiple_choices':
                    v=official();b=json.loads(v['body']);b['choices']*=2;v['body']=json.dumps(b);return v
                if name=='credential_echo':return official({'status':'unsupported','task_types':[],'message':'SYNTHETIC '+SECRET})
                raise AssertionError(name)
            router=self.router(exchange);value=router.prepare_request(payload());again=router.prepare_request(payload())
            self.save('failure_'+name,{'route':value,'again':again,'provider':router.snapshot()})
            self.assertEqual(value['error_code'],code);self.assertIsNone(value['request']);self.assertEqual(again['error_code'],'budget_exhausted');self.assertEqual(len(calls),1)
            self.assertNotIn(SECRET,json.dumps(router.snapshot()));self.assertNotIn('Authorization',json.dumps(router.snapshot()))

    def test_invalid_inputs_are_pre_exchange_and_explicit_labels_unchanged(self):
        variants=[]
        for key in ['task_types','question_id','reference_answer','score']:
            value=payload();value[key]=['inspection'] if key=='task_types' else 'SYN';variants.append(value)
        bad=payload();bad['explicitly_allowed_operating_context']={'notes':'SYN'};variants.append(bad)
        variants.extend([{'x':float('nan')},None]);circular={};circular['x']=circular;variants.append(circular)
        for i,value in enumerate(variants):
            router=self.router(lambda *_:self.fail('Invalid input dispatched'))
            result=router.prepare_request(value);self.save('invalid_'+str(i),result);self.assertEqual(result['status'],'invalid_input');self.assertEqual(router.snapshot()['attempt_count'],0)
        invalid=copy.deepcopy(self.cfg);invalid['max_attempts']=True
        with self.assertRaises(ValueError):intent.ModelTaskRouter(self.profile,invalid,self.prompts,lambda *_:None)

    def test_no_fallback_and_no_downstream_for_blocked_states(self):
        for status in ['needs_clarification','unsupported']:
            router=self.router(lambda *_:official({'status':status,'task_types':[],'message':'SYNTHETIC restate request.'}))
            value=intent.run_model_flow(payload(),router,Untouched(),Untouched(),retriever=Untouched(),runtime=Untouched())
            self.save('blocked_'+status,value);self.assertIsNone(value['flow_record'])
        router=self.router(lambda *_:(_ for _ in ()).throw(TimeoutError('SYN')))
        value=intent.run_model_flow(payload(),router,Untouched(),Untouched(),retriever=Untouched());self.assertIsNone(value['flow_record'])

    def test_copy_and_synthetic_complete_chain_partial_failure(self):
        original=payload();before=copy.deepcopy(original)
        def mutating_exchange(endpoint,body,settings):body['messages'].clear();settings['parameters'].clear();return official()
        router=self.router(mutating_exchange);value=router.prepare_request(original);self.assertEqual(original,before);self.assertEqual(value['request']['question'],before['question']);self.assertTrue(router.attempts[0]['official_request']['messages'])
        for fail in [False,True]:
            router=self.router(lambda *_:official());ret=fixtures.retrieval('partial_task_coverage');ret['request']=dict(payload(),task_types=['inspection'])
            class Provider:
                def generate(self,p):return fixtures.answer()
                def verify(self,p):
                    if fail:raise TimeoutError('SYNTHETIC verify failed')
                    return {'verdict':'pass','issues':[]}
                def revise(self,p):raise AssertionError('Unexpected')
            value=intent.run_model_flow(payload(),router,flow.group_config('GV'),lambda *_:Provider(),retriever=fixtures.FakeRetriever(ret))
            self.save('chain_'+str(fail),value);rec=value['flow_record'];self.assertEqual(rec['counts']['retrieval'],1);self.assertEqual(rec['counts']['generate'],1);self.assertEqual(rec['counts']['verify'],1);self.assertEqual(rec['counts']['revise'],0)
            if fail:self.assertIsNone(rec['final_answer']);self.assertFalse(rec['publishable'])
            else:self.assertEqual(rec['status'],'evidence_insufficient');self.assertTrue(rec['publishable'])

    def test_mock_HTTP_caps_slot_redaction_and_errors(self):
        for name in ['ok','overcap','freeze_changed','timeout','invalid_utf8','rate_limit']:
            folder=self.out/('http_'+name);folder.mkdir();gate=FakeGate(folder,payload());gate.changed=name=='freeze_changed';counter=Counter(2049 if name=='overcap' else 20);posts=[]
            def handler(request):
                posts.append({'method':request.method,'body':json.loads(request.content),'headers_not_recorded':True})
                if name=='timeout':raise httpx.ReadTimeout('SYNTHETIC '+SECRET,request=request)
                if name=='invalid_utf8':return httpx.Response(200,content=b'\xff')
                if name=='rate_limit':return httpx.Response(429,content=b'SYNTHETIC rate limit')
                env=official();return httpx.Response(200,content=env['body'].encode(),headers={'x-request-id':'SYN-HTTP'})
            slot=wire.IntentSlot(folder/'slot',flow.digest(counter.identity));transport=wire.IntentHTTP(self.profile,self.cfg,self.prompts,slot,gate,token_counter=counter,http_transport=httpx.MockTransport(handler),test_credential=SECRET)
            router=intent.ModelTaskRouter(self.profile,self.cfg,self.prompts,transport.request,transport_kind='http_mock',redact_values=(SECRET,));value=router.prepare_request(payload())
            self.save('http_'+name,{'kind':'http_mock','route':value,'transport':transport.snapshot(),'provider':router.snapshot(),'posts':posts,'reserved':slot.reserved})
            self.assertNotIn(SECRET,json.dumps(transport.snapshot()));self.assertEqual(transport.identity['actual_official_http_requests'],0)
            expected=0 if name in ['overcap','freeze_changed'] else 1;self.assertEqual(len(posts),expected)
            if name=='ok':
                self.assertEqual(value['status'],'ready');self.assertEqual(slot.reserved,'0.0009216')
                second=intent.ModelTaskRouter(self.profile,self.cfg,self.prompts,transport.request,transport_kind='http_mock');denied=second.prepare_request(payload());self.assertEqual(denied['error_code'],'budget_exhausted');self.assertEqual(len(posts),1)
                self.save('http_ok_second_denial',{'kind':'http_mock','route':denied,'provider':second.snapshot(),'transport_after_denial':transport.snapshot(),'posts':posts})
            else:self.assertEqual(value['status'],'recognition_failed')

    def test_adapted_QA_budget_fixed_allocation_and_poison(self):
        cfg=wire.read(ROOT/'configs/q01_deepseek_budget.json');cfg.update(max_attempts=4,max_cost='0.018');budget=wire.ChainQABudget(self.out/'budget',cfg,'a'*64)
        try:
            with self.assertRaises(intent.Error):budget.begin_flow('T','a'*64)
            budget.begin_flow('GV','a'*64);budget.freeze('http_mock',{'kind':'synthetic_budget_fixture'},{'case':'budget_caps'})
            for action,out in [('generate',1024),('verify',512),('revise',1024),('verify',512)]:
                res=budget.reserve('GV',action,8192,out);budget.record_result(res,{'status':'failed','error_code':'synthetic_timeout','transport':'http_mock'})
            with self.assertRaises(intent.Error):budget.reserve('GV','verify',8192,512)
            self.assertEqual(budget.snapshot()['attempt_count'],4);self.assertEqual(budget.snapshot()['reserved_cost'],'0.0135168');self.save('qa_allocation',budget.snapshot())
        finally:budget.close()
        poisoned=wire.ChainQABudget(self.out/'poison',cfg,'b'*64)
        try:
            poisoned.begin_flow('GV','b'*64);original=poisoned.ledger_path.read_bytes();poisoned.ledger_path.write_bytes(original+b' ')
            with self.assertRaises(Exception):poisoned.reserve('GV','generate',8192,1024)
            poisoned.ledger_path.write_bytes(original)
            with self.assertRaises(intent.Error):poisoned.reserve('GV','generate',8192,1024)
            self.assertTrue(poisoned.snapshot()['broken']);self.save('qa_poisoned',poisoned.snapshot())
        finally:poisoned.close()
        bad=copy.deepcopy(cfg);bad['model_prices']['deepseek-flash']['input_per_million']='0'
        with self.assertRaises(ValueError):wire.ChainQABudget(self.out/'bad_price',bad,'c'*64)

    def test_derived_HTTP_binding_and_complete_card_projection(self):
        for tamper in [False,True]:
            folder=self.out/('qa_'+str(tamper));folder.mkdir();request=dict(payload(),task_types=['inspection']);route={'status':'ready','request':request,'model_report':report(),'source_four_fields_sha256':flow.digest(payload())}
            flow.write_record(folder/'derived_request.json',request);flow.write_record(folder/'recognition_record.json',route)
            cfg=wire.read(ROOT/'configs/q01_deepseek_budget.json');cfg.update(max_attempts=4,max_cost='0.018');gate=FakeGate(folder,payload(),cfg);gate.bound={p:flow.file_hash(folder/p) for p in ['derived_request.json','recognition_record.json']}
            if tamper:
                altered=copy.deepcopy(route);altered['model_report']['task_types']=['preparation'];(folder/'recognition_record.json').write_text(json.dumps(altered))
            posts=[]
            def handler(http_request):
                body=json.loads(http_request.content);posts.append(body);output={'verdict':'pass','issues':[]} if body['max_tokens']==512 else fixtures.answer();return httpx.Response(200,content=official(output)['body'].encode())
            budget=wire.ChainQABudget(folder/'ledger',cfg,flow.file_hash(folder/'derived_request.json'))
            try:
                provider=wire.DerivedQAProvider(self.profile,budget,flow.file_hash(folder/'derived_request.json'),gate=gate,token_counter=Counter(),http_transport=httpx.MockTransport(handler),test_credential=SECRET)
                ret=fixtures.retrieval('partial_task_coverage');ret['request']=request
                runtime={'record_kind':'provider_contract_mock','prompts':wire.read(ROOT/'configs/q01_provider_prompts.json'),'generation_settings':{'fixture':True}}
                rec=flow.run_flow(request,flow.group_config('GV'),provider,retriever=fixtures.FakeRetriever(ret),runtime=runtime)
                self.save('derived_'+str(tamper),{'kind':'http_mock','record':rec,'provider':provider.snapshot(),'posts':posts})
                self.assertEqual(len(posts),0 if tamper else 2)
                if tamper:self.assertFalse(rec['publishable']);self.assertIsNone(rec['final_answer'])
                else:
                    self.assertEqual(rec['status'],'evidence_insufficient');self.assertEqual(rec['counts']['verify'],1)
                    user=json.loads(posts[0]['messages'][1]['content']);self.assertEqual(set(user['request']),flow.k02.INPUT_FIELDS)
                    self.assertEqual(user['evidence_cards'][0]['source'],ret['results'][0]['source']);self.assertNotIn('rank',user['evidence_cards'][0])
            finally:budget.close()

    def test_master_gate_exact_file_set_source_freeze_and_derived_prerequisite(self):
        assets=ROOT/'runs/q01_provider/20261025_tokenizer_b03d7e91';counter=DeepSeekV41Counter(assets/'verification/counter_assets.json',self.profile,assets/'audit/counter_certificate.json')
        for case in ['valid','empty_files','projection_changed','freeze_changed']:
            folder=self.out/('master_'+case);folder.mkdir();(folder/'inputs').mkdir()
            for name in ['source_registration.json','four_fields.json']:(folder/'inputs'/name).write_bytes((self.stage_run/'inputs'/name).read_bytes())
            (folder/'controlled_chain.py').write_text('# SYNTHETIC non-executable driver identity fixture\n')
            qa=wire.read(ROOT/'configs/q01_deepseek_budget.json');qa.update(max_attempts=4,max_cost='0.018',approved_counter_sha256=flow.digest(counter.identity))
            paths=wire.CODE_FILES|wire.CONFIG_FILES|{(folder/'controlled_chain.py').relative_to(ROOT).as_posix()}
            release={'release':True,'file_hashes':{p:flow.file_hash(ROOT/p) for p in paths},'allocation':wire.ALLOCATION,'counter_identity_sha256':flow.digest(counter.identity),'qa_budget':qa,'four_fields_sha256':flow.digest(wire.read(folder/'inputs/four_fields.json')),'four_fields_file_sha256':flow.file_hash(folder/'inputs/four_fields.json')}
            if case=='empty_files':release['file_hashes']={}
            flow.write_record(folder/'live_release.json',release)
            if case=='empty_files':
                with self.assertRaises(intent.Error):wire.LiveGate(folder,counter)
                continue
            gate=wire.LiveGate(folder,counter);gate.freeze()
            with self.assertRaises(intent.Error):gate.require_derived()
            if case=='projection_changed':(folder/'inputs/four_fields.json').write_text('{}')
            elif case=='freeze_changed':
                with (folder/'master_first_request_freeze.json').open('ab') as f:f.write(b' ')
            if case!='valid':
                with self.assertRaises(intent.Error):gate.require_frozen()
            else:gate.require_frozen()
            self.save('master_'+case,{'kind':'offline_control_fixture','freeze_created':True,'no_inference_requests':True})

if __name__=='__main__':unittest.main()
