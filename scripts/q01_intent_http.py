"""New-stage intent slot and derived-request QA transport; old gates unchanged."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import time
from decimal import Decimal

import httpx
import q01_flow as flow
import q01_deepseek as codec
import q01_provider as common
import q01_deepseek_http as old
import q01_intent as intent

ROOT=old.ROOT
Error=common.ProviderError
CODE_FILES={'scripts/q01_flow.py','scripts/q01_provider.py','scripts/q01_deepseek.py',
 'scripts/q01_deepseek_http.py','scripts/q01_tasks.py','scripts/q01_intent.py',
 'scripts/q01_intent_http.py','scripts/q01_token_counter.py','scripts/retrieve_k02.py','scripts/validate_k01.py'}
CONFIG_FILES={'configs/q01_deepseek.json','configs/q01_deepseek_budget.json','configs/q01_provider_prompts.json',
 'configs/q01_offline.json','configs/q01_prompts.json','configs/k02_retrieval.json',
 'configs/q01_intent.json','configs/q01_intent_prompts.json'}
ALLOCATION={'identify_attempts':1,'identify_cost_usd':'0.002','qa_attempts':4,'qa_cost_usd':'0.018',
 'total_attempts':5,'total_cost_usd':'0.02','retries':0,'concurrency':1}


def put(path,value):
    path=old._no_links(path);raw=(flow.canonical(value)+'\n').encode('ascii')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        identity=(os.fstat(fd).st_dev,os.fstat(fd).st_ino)
        with os.fdopen(fd,'wb',closefd=False) as f:f.write(raw);f.flush();os.fsync(fd)
        current=path.lstat()
        if (current.st_dev,current.st_ino)!=identity or old._regular_bytes(path)!=raw:
            raise OSError('Durable record identity changed.')
    finally:os.close(fd)
    return raw


def read(path):return flow.k02.strict_json(old._regular_bytes(path))


class LiveGate:
    """Exact master release, source projection and first-request freeze."""
    def __init__(self,run,counter):
        self.run=old._no_links(run)
        if not self.run.is_relative_to(ROOT/'runs/q01_intent'):
            raise ValueError('A new intent run is required.')
        self.counter=counter;self.release_path=self.run/'live_release.json'
        self._release_bytes=old._regular_bytes(self.release_path);self.release=flow.k02.strict_json(self._release_bytes)
        self._freeze_bytes=None;self._derived_binding=None;self._derived_hashes=None
        self.check()

    def check(self):
        if old._regular_bytes(self.release_path)!=self._release_bytes:raise Error('release_changed','Master release bytes changed.')
        r=self.release;driver=(self.run/'controlled_chain.py').relative_to(ROOT).as_posix()
        if (r.get('release') is not True or set(r['file_hashes'])!=CODE_FILES|CONFIG_FILES|{driver}
                or r['allocation']!=ALLOCATION or r['counter_identity_sha256']!=flow.digest(self.counter.identity)):
            raise Error('release_invalid','Incomplete master release/ceiling/counter identity.')
        if any(flow.file_hash(ROOT/p)!=h for p,h in r['file_hashes'].items()):raise Error('release_changed','Released code/config/prompt changed.')
        registered=read(self.run/'inputs/source_registration.json')
        raw=old._regular_bytes(old.REQUEST_PATH);actual=hashlib.sha256(raw).hexdigest()
        if (registered['actual_sha256']!=old.REQUEST_SHA256 or actual!=old.REQUEST_SHA256
                or registered['path']!=old.REQUEST_PATH.relative_to(ROOT).as_posix()):
            raise Error('request_identity_mismatch','Only the registered pre-existing development source is allowed.')
        # Registration above existed before any source parsing; old task_types never consumed.
        source=flow.k02.strict_json(raw);four={k:source[k] for k in intent.rules.INPUT_FIELDS}
        try:self.projection=intent.validate_input(read(self.run/'inputs/four_fields.json'))
        except Exception:raise Error('request_identity_mismatch','Four-field projection is malformed.') from None
        if (self.projection!=four or flow.digest(four)!=r['four_fields_sha256']
                or flow.file_hash(self.run/'inputs/four_fields.json')!=r['four_fields_file_sha256']):
            raise Error('request_identity_mismatch','Four-field source projection changed.')
        self.profile=read(ROOT/'configs/q01_deepseek.json')
        self.intent_config=read(ROOT/'configs/q01_intent.json');self.prompts=read(ROOT/'configs/q01_intent_prompts.json')
        expected=read(ROOT/'configs/q01_deepseek_budget.json')
        expected.update(max_attempts=4,max_cost='0.018',approved_counter_sha256=r['counter_identity_sha256'])
        if expected!=r['qa_budget']:raise Error('release_invalid','QA allocation/prices differ from approved exact base budget.')
        if self._freeze_bytes is not None and old._regular_bytes(self.run/'master_first_request_freeze.json')!=self._freeze_bytes:
            raise Error('freeze_changed','First-request master freeze changed.')

    def freeze(self):
        self.check()
        self._freeze_bytes=put(self.run/'master_first_request_freeze.json',{
            'master_release_sha256':hashlib.sha256(self._release_bytes).hexdigest(),
            'release':self.release,'scope':'One identify and at most one derived GV chain; no historical batch replay.'})

    def require_frozen(self):
        self.check()
        if self._freeze_bytes is None:raise Error('freeze_required','Freeze all master controls before HTTP.')


    def bind_derived(self,actual_route,actual_provider_snapshot,actual_http_snapshot):
        self.require_frozen()
        if self._derived_binding is not None:raise Error('budget_exhausted','Derived chain already bound.')
        paths=['recognition_record.json','recognition_provider_snapshot.json','recognition_http_snapshot.json','derived_request.json']
        report=read(self.run/paths[0]);snapshot=read(self.run/paths[1]);http=read(self.run/paths[2])
        request=read(self.run/paths[3]);model=intent.validate_report(report['model_report'])
        if (report!=flow.json_snapshot(actual_route) or snapshot!=flow.json_snapshot(actual_provider_snapshot)
                or http!=flow.json_snapshot(actual_http_snapshot)):
            raise Error('recognition_binding_invalid','Persisted result is not the actual memory-held model call result.')
        if (report['status']!='ready' or model['status']!='identified' or report['request']!=request
                or snapshot['identity']['kind']!='real_provider' or snapshot['attempt_count']!=1
                or snapshot['attempts'][0]['status']!='success' or snapshot['attempts'][0]['output']!=model
                or snapshot['attempts'][0]['returned_model']!=self.intent_config['model']
                or http['actual_official_http_requests']!=1 or len(http['events'])!=1
                or not http['events'][0]['official_http_request_attempted'] or http['events'][0]['http_status']!=200):
            raise Error('recognition_binding_invalid','Only the actual successful single real task report can authorize QA.')
        self._derived_hashes={p:flow.file_hash(self.run/p) for p in paths}
        self._derived_binding=put(self.run/'identified_result_binding.json',{
            'file_hashes':self._derived_hashes,'master_release_sha256':hashlib.sha256(self._release_bytes).hexdigest()})

    def require_derived(self):
        self.require_frozen()
        if (self._derived_binding is None or self._derived_hashes is None
                or old._regular_bytes(self.run/'identified_result_binding.json')!=self._derived_binding
                or any(flow.file_hash(self.run/p)!=h for p,h in self._derived_hashes.items())):
            raise Error('recognition_binding_invalid','Actual model result/derived request binding changed.')


class ChainQABudget(old.SessionBudget):
    """New-namespace constructor; inherited append/freeze/reserve/result guards.

    This is an adapted class, not the legacy constructor run unchanged.
    Only one GV can start; four QA attempts and USD .018 are fixed allocations.
    """
    def __init__(self,run_dir,session_config,request_sha256):
        config=flow.json_snapshot(session_config)
        required={'max_attempts','currency','max_cost','allowed_groups','model_prices','approved_counter_sha256'}
        registered=read(ROOT/'configs/q01_deepseek_budget.json')
        if (type(config) is not dict or set(config)!=required or old._sensitive_registration(config)
                or type(config['max_attempts']) is not int or config['max_attempts']!=4
                or config['currency']!='USD' or config['max_cost']!='0.018'
                or config['allowed_groups']!=registered['allowed_groups']
                or config['model_prices']!=registered['model_prices'] or not old._sha(request_sha256)
                or (config['approved_counter_sha256'] is not None and not old._sha(config['approved_counter_sha256']))):
            raise ValueError('Exact QA-only 4-attempt/USD.018 allocation and registered model prices required.')
        run=old._no_links(run_dir)
        if not run.is_relative_to(ROOT/'runs/q01_intent'):raise ValueError('New chain QA directory required.')
        if run.exists():
            if not run.is_dir() or any(run.iterdir()):raise ValueError('New or empty QA ledger directory required.')
        else:run.mkdir(mode=0o700,parents=False,exist_ok=False)
        self.run_dir,self._config,self.request_sha256=run,config,request_sha256
        self.ledger_path=run/'session_ledger.jsonl'
        self._fd=os.open(self.ledger_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_APPEND|os.O_NOFOLLOW,0o600)
        fcntl.flock(self._fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self._file_identity=(os.fstat(self._fd).st_dev,os.fstat(self._fd).st_ino)
        self._ledger_sha=hashlib.sha256(b'').hexdigest()
        self._lock,self.exchange_lock=threading.RLock(),threading.Lock()
        self._closed,self._broken,self._sequence=False,False,0
        self._attempt_count,self._reserved_cost=0,Decimal('0')
        self._counts,self._flows,self._group_counts,self._denials=dict.fromkeys(old.TOTAL_LIMITS,0),{},{},[]
        self._freeze_path,self._freeze_bytes,self._freeze_binding=None,None,None
        self._append('init',{'config':config,'request_sha256':request_sha256,
             'ledger_policy':'New chain QA-only GV, exact4/.018; inherited exclusive/fsync/no refunds/no restart.'})

    def begin_flow(self,group,request_sha256):
        if group!='GV':raise Error('flow_not_allowed','Only the single derived GV is allowed in this QA allocation.')
        return super().begin_flow(group,request_sha256)



class IntentSlot:
    """One durable no-refund intent reservation, separately capped at USD .002."""
    def __init__(self,folder,counter_sha256):
        self.folder=old._no_links(folder)
        if not self.folder.is_relative_to(ROOT/'runs/q01_intent'):raise ValueError('New intent directory required.')
        self.folder.mkdir(exist_ok=False)
        self.config={'approved_counter_sha256':counter_sha256};self.used=False;self.raw=None;self.identity=None
        self.reserved='0';self.broken=False

    def reserve(self,config):
        if self.used or self.broken:raise Error('budget_exhausted','Intent slot already consumed.')
        self.used=True
        cost=(Decimal(config['max_input_tokens'])*Decimal('0.30')+Decimal(config['max_output_tokens'])*Decimal('1.20'))/1000000
        if cost>Decimal('0.002'):raise Error('cost_budget_exhausted','Intent allocation exceeded.')
        try:
            self.raw=put(self.folder/'reservation.json',{'action':'identify','attempt':1,'input_cap':config['max_input_tokens'],
             'output_cap':config['max_output_tokens'],'reserved_usd':str(cost),'no_refunds':True})
            st=(self.folder/'reservation.json').lstat();self.identity=(st.st_dev,st.st_ino);self.reserved=str(cost);self.check()
        except Exception:self.broken=True;raise

    def check(self):
        if self.broken:raise Error('session_unusable','Intent session poisoned.')
        if self.raw is not None:
            try:
                st=(self.folder/'reservation.json').lstat()
                if (st.st_dev,st.st_ino)!=self.identity or old._regular_bytes(self.folder/'reservation.json')!=self.raw:
                    raise OSError('Intent ledger changed.')
            except Exception:self.broken=True;raise Error('session_unusable','Intent reservation changed.') from None

    def result(self,value):
        try:self.check();put(self.folder/'result.json',value)
        except Exception:self.broken=True;raise


class IntentHTTP(old.OneShotDeepSeekHTTP):
    def __init__(self,profile,config,prompts,slot,gate,*,token_counter,http_transport=None,test_credential=None):
        super().__init__(profile,slot,token_counter=token_counter,http_transport=http_transport,test_credential=test_credential)
        intent.ModelTaskRouter(profile,config,prompts,lambda *_:None)
        if self.mode=='real_http' and type(gate) is not LiveGate:raise ValueError('Real intent transport needs the exact live master gate.')
        self.intent_config=flow.json_snapshot(config);self.prompts=flow.json_snapshot(prompts);self.gate=gate
        self._lock=threading.Lock()

    def request(self,endpoint,body,settings):
        if not self._lock.acquire(blocking=False):raise Error('concurrency_forbidden','Intent HTTP already active.')
        event={'action':'identify','transport':self.mode,'status':'preflight','httpx_post_invoked':False,
          'official_http_request_attempted':False,'http_status':'unknown','raw_response':'unknown','actual_bill':'unknown'}
        self.events.append(event);start=time.perf_counter();reserved=False
        try:
            self.gate.require_frozen();body=flow.json_snapshot(body)
            expected={'model':self.intent_config['model'],'messages':[{'role':'system','content':self.prompts['instructions']},
              {'role':'user','content':flow.canonical({'question':self.gate.projection['question']})}],
              'stream':False,'response_format':{'type':'json_object'},'max_tokens':self.intent_config['max_output_tokens'],**self.intent_config['parameters']}
            if endpoint!=codec.OFFICIAL_ENDPOINT or body!=expected or settings!=self.intent_config:
                raise Error('request_configuration_mismatch','Intent body/settings do not match frozen question-only projection.')
            _,counted=self._counter_result(body);event['token_count']=counted
            if counted['input_tokens']>self.intent_config['max_input_tokens']:raise Error('input_token_budget_exceeded','Intent input cap exceeded.')
            credential=self._test_credential if self.mode=='http_mock' else os.environ.get(self.credential_env)
            if type(credential) is not str or not credential.strip():raise Error('credential_missing','Configured credential unavailable.')
            self._redactor._redact_values=(credential,repr(credential.encode('utf-8'))[2:-1])
            self.budget.reserve(self.intent_config);reserved=True;self.gate.require_frozen();self.budget.check()
            event['official_request']=self._redactor._redact(body)
            transport=self._transport if self.mode=='http_mock' else httpx.HTTPTransport(retries=0,trust_env=False)
            with httpx.Client(transport=transport,trust_env=False,follow_redirects=False,timeout=httpx.Timeout(60)) as client:
                event.update(httpx_post_invoked=True,official_http_request_attempted=self.mode=='real_http')
                self.httpx_post_invocations+=1
                response=client.post(endpoint,json=body,headers={'Authorization':'Bearer '+credential,'Content-Type':'application/json'})
                raw=response.content;event['response_bytes_sha256']=hashlib.sha256(raw).hexdigest()
                try:text=raw.decode('utf-8')
                except UnicodeDecodeError:
                    event['raw_response']={'body_bytes_escaped':self._redactor._redact_text(repr(raw))}
                    raise Error('invalid_json','Invalid response UTF-8; no replacement decoding.') from None
                envelope={'status_code':response.status_code,'body':text}
                if 'x-request-id' in response.headers:envelope['request_id']=response.headers['x-request-id']
                event.update(raw_response=self._redactor._redact(envelope),http_status=response.status_code,status='http_response_received')
            self.budget.result({'status':event['status'],'http_status':response.status_code,'response_bytes_sha256':event['response_bytes_sha256']});reserved=False
            return envelope
        except Exception as error:
            code=error.code if isinstance(error,Error) else ('timeout' if isinstance(error,(httpx.TimeoutException,TimeoutError)) else 'transport_error')
            event.update(status='failed',error_code=code,diagnostic=self._redactor._redact_text(flow.error_text(error)))
            if reserved:
                try:self.budget.result({'status':'failed','error_code':code})
                except Exception:event['result_record_error']='unknown_or_failed'
            raise Error(code,event['diagnostic']) from None
        finally:event['elapsed_seconds']=time.perf_counter()-start;self._lock.release()


    @property
    def identity(self):
        value=super().identity
        value.update(role='task_identification',implementation_sha256=flow.file_hash(__file__))
        return value


class DerivedRequestHTTP(old.OneShotDeepSeekHTTP):
    def __init__(self,*args,gate,**kwargs):
        super().__init__(*args,**kwargs);self.gate=gate
        if self.mode=='real_http' and type(gate) is not LiveGate:raise ValueError('Exact live gate required for real derived QA.')

    def _context(self,body,context):
        self.gate.require_derived();context=flow.json_snapshot(context)
        required={'request_sha256','business_request_sha256','prompt_bundle_sha256','schema_sha256','code_hashes'}
        if set(context)!=required:raise Error('freeze_context_invalid','Complete QA context required.')
        hashes={str(p.relative_to(ROOT)):flow.file_hash(p) for p in old.CODE_PATHS}
        if (context['code_hashes']!=hashes or context['schema_sha256']!=flow.digest(old.SCHEMAS)
                or context['prompt_bundle_sha256']!=flow.file_hash(ROOT/'configs/q01_provider_prompts.json')
                or self.budget.config!=self.gate.release['qa_budget']):
            raise Error('freeze_context_invalid','QA code/prompt/schema/budget differs from release.')
        p=self.gate.run/'derived_request.json';raw=old._regular_bytes(p);h=hashlib.sha256(raw).hexdigest()
        self.budget._append('input_registration',{'path':str(p),'purpose':'Only the request derived from this actual task-model result; hash registered before parse.','actual_sha256':h})
        if context['request_sha256']!=h or self.budget.request_sha256!=h:raise Error('request_identity_mismatch','Derived request bytes changed.')
        request=flow.k02.strict_json(raw);flow.k02.validate_request(request)
        report=read(self.gate.run/'recognition_record.json');model=intent.validate_report(report['model_report'])
        business=flow.k02.strict_json(body['messages'][1]['content'])['request']
        if (report['status']!='ready' or model['status']!='identified' or report['request']!=request
                or report['source_four_fields_sha256']!=self.gate.release['four_fields_sha256']
                or {k:request[k] for k in intent.rules.INPUT_FIELDS}!=self.gate.projection
                or request['task_types']!=[t for t in flow.k02.TASKS if t in model['task_types']]
                or business!=request or context['business_request_sha256']!=flow.digest(request)):
            raise Error('request_identity_mismatch','QA is not bound to the actual model-derived tasks and original four fields.')
        context['code_hashes']=dict(hashes,**{p:self.gate.release['file_hashes'][p] for p in CODE_FILES})
        return context


class DerivedQAProvider(old.DeepSeekHTTPProvider):
    def __init__(self,config,budget,request_sha256,*,gate,token_counter,http_transport=None,test_credential=None):
        super().__init__(config,budget,'GV',request_sha256,token_counter=token_counter,
                         http_transport=http_transport,test_credential=test_credential)
        self._http=DerivedRequestHTTP(config,budget,gate=gate,token_counter=token_counter,
                         http_transport=http_transport,test_credential=test_credential)
        self.gate=gate

    @property
    def identity(self):
        value=super().identity
        value.update(chain_version='q01.model_chain.v1',chain_code_sha256=flow.file_hash(__file__),
          master_release_sha256=hashlib.sha256(self.gate._release_bytes).hexdigest() if hasattr(self.gate,'_release_bytes') else 'http_mock')
        return value
