"""One-shot model task identification and optional existing-flow dispatch.

No key, network, knowledge or evaluation access at import. The injected exchange
uses the official protocol; offline fixtures and real transport stay labelled.
"""
import datetime
import threading
import time

import q01_flow as flow
import q01_provider as common
import q01_deepseek as codec
import q01_tasks as rules

VERSION = 'q01.intent.v1'
REPORT_FIELDS = {'status', 'task_types', 'message'}
Error = common.ProviderError


def validate_input(payload):
    value = flow.json_snapshot(payload)
    common._unicode(value)
    if type(value) is not dict or set(value) != rules.INPUT_FIELDS:
        raise ValueError('Exactly the four business fields without existing task_types are required.')
    if type(value['question']) is not str or not value['question'].strip() or len(value['question']) > 2000:
        raise ValueError('A nonblank question of at most 2000 characters is required.')
    # Schema-only probe. These fixed enum values never enter the model or result.
    flow.k02.validate_request(dict(value, task_types=list(flow.k02.TASKS)))
    return value


def validate_report(value):
    value = flow.json_snapshot(value)
    common._unicode(value)
    if type(value) is not dict or set(value) != REPORT_FIELDS:
        raise Error('schema_mismatch', 'Exact task-report fields required.')
    status, tasks, message = value['status'], value['task_types'], value['message']
    if type(status) is not str or status not in ('identified', 'needs_clarification', 'unsupported'):
        raise Error('schema_mismatch', 'Unknown task status.')
    if (type(tasks) is not list or any(type(t) is not str for t in tasks)
            or len(tasks) != len(set(tasks)) or not set(tasks) <= set(flow.k02.TASKS)):
        raise Error('schema_mismatch', 'Task list must contain unique exact supported enum values.')
    if status == 'identified':
        if not tasks or message is not None:
            raise Error('schema_mismatch', 'Identified needs nonempty tasks and null message.')
    elif tasks or type(message) is not str or not message.strip():
        raise Error('schema_mismatch', 'Blocked status needs empty tasks and a nonblank message.')
    return value


class ModelTaskRouter:
    def __init__(self, provider_config, config, prompts, exchange, *, transport_kind='offline_fixture', redact_values=()):
        self.profile, _ = codec._mock_config(provider_config)
        self.config, self.prompts = flow.json_snapshot(config), flow.json_snapshot(prompts)
        common._unicode(self.config); common._unicode(self.prompts)
        required = {'version','model','endpoint','parameters','max_input_tokens','max_output_tokens',
                    'timeout_seconds','retries','concurrency','max_attempts','max_cost_usd',
                    'qa_max_attempts','qa_max_cost_usd'}
        if (type(self.config) is not dict or set(self.config) != required
                or self.config['model'] != self.profile['models']['generate']
                or self.config['endpoint'] != codec.OFFICIAL_ENDPOINT
                or self.config['parameters'] != self.profile['parameters']['generate']
                or self.config['parameters'] != {'thinking':{'type':'disabled'},'temperature':0}
                or type(self.config['max_input_tokens']) is not int or not 1 <= self.config['max_input_tokens'] <= 2048
                or type(self.config['max_output_tokens']) is not int or not 1 <= self.config['max_output_tokens'] <= 256
                or any(type(self.config[k]) is not int for k in ('max_attempts','retries','concurrency','timeout_seconds','qa_max_attempts'))
                or self.config['max_attempts'] != 1 or self.config['retries'] != 0 or self.config['concurrency'] != 1
                or self.config['timeout_seconds'] != 60 or self.config['max_cost_usd'] != '0.002'
                or self.config['qa_max_attempts'] != 4 or self.config['qa_max_cost_usd'] != '0.018'):
            raise ValueError('Intent profile and separate 1+4/.002+.018 ceiling must match registration.')
        if (type(self.prompts) is not dict or set(self.prompts) != {'version','instructions'}
                or any(type(v) is not str or not v.strip() for v in self.prompts.values())
                or transport_kind not in ('offline_fixture','http_mock','real_http') or not callable(exchange)):
            raise ValueError('Explicit prompt and trusted exchange identity required.')
        self.exchange, self.transport_kind = exchange, transport_kind
        self.helper = codec.DeepSeekChatProvider(self.profile, lambda *_: None, redact_values=redact_values)
        if self.helper._inner._redact(self.config) != self.config or self.helper._inner._redact(self.prompts) != self.prompts:
            raise ValueError('Sensitive configuration/prompt registration forbidden.')
        self.attempts, self.denials, self._used, self._lock = [], [], False, threading.Lock()

    @property
    def identity(self):
        return {'kind':'real_provider' if self.transport_kind=='real_http' else 'provider_contract_mock',
                'role':'task_identification','version':VERSION,'transport':self.transport_kind,
                'code_sha256':flow.file_hash(__file__),'config_sha256':flow.digest(self.config),
                'prompt_sha256':flow.digest(self.prompts),'semantic_accuracy_proven':False}

    def snapshot(self):
        return self.helper._inner._redact(flow.json_snapshot({'identity':self.identity,
            'attempt_count':len(self.attempts),'attempts':self.attempts,'denials':self.denials,
            'actual_bill':'unknown'}))

    def prepare_request(self, payload):
        result = {'status':'recognition_failed','request':None,'model_report':None,
                  'service_message':None,'error_code':None,'source_four_fields_sha256':None,
                  'identity':self.identity,'semantic_accuracy_proven':False}
        if not self._lock.acquire(blocking=False):
            result.update(error_code='concurrency_forbidden',service_message='Task recognition is already in progress.')
            self.denials.append({'code':'concurrency_forbidden'});return result
        try:
            try:
                value = validate_input(payload)
            except Exception:
                result.update(status='invalid_input',error_code='invalid_input',service_message='The four-field input is not valid.');return result
            result['source_four_fields_sha256'] = flow.digest(value)
            if self._used:
                self.denials.append({'code':'budget_exhausted'})
                result.update(error_code='budget_exhausted',service_message='Task-recognition attempt already consumed.');return result
            self._used = True
            body = {'model':self.config['model'],'messages':[
                {'role':'system','content':self.prompts['instructions']},
                {'role':'user','content':flow.canonical({'question':value['question']})}],
                'stream':False,'response_format':{'type':'json_object'},'max_tokens':self.config['max_output_tokens'],
                **flow.json_snapshot(self.config['parameters'])}
            row = {'attempt':1,'action':'identify','record_kind':self.identity['kind'],
                   'transport':self.transport_kind,'response_identity_origin':self.transport_kind,
                   'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'requested_model':body['model'],'returned_model':'unknown',
                   'request_id':'unknown','response_id':'unknown','token_usage':'unknown','actual_bill':'unknown',
                   'official_request':flow.json_snapshot(body),'official_request_sha256':flow.digest(body),
                   'raw_response':'unknown','status':'in_progress','error_code':None,'parse_diagnostic':'not_parsed'}
            self.attempts.append(row);started=time.perf_counter()
            try:
                try:
                    response = self.exchange(codec.OFFICIAL_ENDPOINT,flow.json_snapshot(body),flow.json_snapshot(self.config))
                finally:
                    owner = getattr(self.exchange,'__self__',None)
                    if owner is not None and hasattr(owner,'_redactor'):
                        self.helper._inner._redact_values = owner._redactor._redact_values
                normalized = self.helper._decode(response,row)
                status = normalized['status_code']
                if status == 429:raise Error('rate_limit','Service rate limit.')
                if not 200 <= status < 300:raise Error('service_error','Service returned an error status.')
                if not normalized['body'].strip():raise Error('empty_output','Empty response.')
                envelope = flow.k02.strict_json(normalized['body'])
                if envelope.get('refusal'):raise Error('refusal','Task service refused.')
                if envelope.get('truncated'):raise Error('truncated','Task output was truncated.')
                text = envelope.get('output_text')
                if type(text) is not str or not text.strip():raise Error('empty_output','No task output.')
                if self.helper._inner._redact_text(text) != text:raise Error('credential_echo','Sensitive output rejected.')
                try: report = validate_report(flow.k02.strict_json(text))
                except Error: raise
                except Exception: raise Error('invalid_json','Unparseable strict task JSON.') from None
                result['model_report'] = report
                if report['status']=='identified':
                    request = dict(value,task_types=[t for t in flow.k02.TASKS if t in report['task_types']])
                    flow.k02.validate_request(request);result.update(status='ready',request=request)
                else:result.update(status=report['status'],service_message=report['message'])
                row.update(status='success',output=report,parse_diagnostic='Strict task JSON/schema accepted; order canonicalized only in derived request.')
            except Exception as error:
                code=error.code if isinstance(error,Error) else ('timeout' if isinstance(error,TimeoutError) else 'transport_error')
                message=self.helper._inner._redact_text(flow.error_text(error))
                row.update(status='failed',error_code=code,parse_diagnostic=message)
                result.update(error_code=code,service_message='Task recognition failed; no downstream request dispatched.')
            finally:row['elapsed_seconds']=time.perf_counter()-started
            return flow.json_snapshot(result)
        finally:self._lock.release()


def run_model_flow(payload,router,config,provider_factory,*,retriever=None,runtime=None):
    route=router.prepare_request(payload)
    record=None
    if route['status']=='ready':
        provider=provider_factory(flow.json_snapshot(route['request']),flow.json_snapshot(route))
        record=flow.run_flow(route['request'],config,provider,retriever=retriever,runtime=runtime)
    return {'routing':route,'flow_record':record}
