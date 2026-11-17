"""Saved-answer scoring only. No retrieval, generation, retry, resume or implicit CLI."""
from pathlib import Path
import os
import time
from common import require, Rejected, digest, read_json, write_new, safe_path, snapshot
from baseline import resolve, fixed_ratio, family_macro
from quote_anchor import messages, validate_rating, third_packet
from rating_contract import source_binding
from provider import Counter, LiveGate, LiveHTTP, Ledger, Provider


def metrics(rating, points):
    scores={p['point_id']:p['score'] for p in rating['points']} if rating else {p['point_id']:None for p in points}
    result={'REF_cov':fixed_ratio(list(scores.values()))}
    for role,name in [('positive_knowledge','REF_knowledge'),('boundary','REF_boundary'),('source_status','REF_source_state')]:
        result[name]=fixed_ratio([scores[p['point_id']] for p in points if p['content_role']==role])
    result['REF_unsupported_claim']=fixed_ratio([{'full':0,'partial':0.5,'none':1,'outside_reference':None,'machine_unresolved':None}[x['support']] for x in rating['claims']]) if rating else {'status':'unknown','value':None,'denominator':None,'lower':None,'upper':None}
    result['REF_voluntary_citation_error']=fixed_ratio([{'supported':0,'contradicted':1,'fabricated_id':1,'unverifiable':None,'machine_unresolved':None}[x['status']] for x in rating['voluntary_citations']]) if rating else {'status':'unknown','value':None,'denominator':None,'lower':None,'upper':None}
    return result


def stopped_outcome(binding):
    return {'status':'not_attempted_batch_stopped','rating':None,'raw_rating':None,
            'source_binding':source_binding(binding),'anchor_metadata':None,'provider_metadata':None}


def evaluate_cell(provider, source, packet, rubric, output_root):
    started=time.monotonic(); originals=[]
    for label in ['A','B','C']:
        slot=source['question_id']+'_L_REF047_'+label
        binding=provider.config['slot_bindings'][slot]
        if provider.ledger.broken:
            original=stopped_outcome(binding)
        else:
            raw=None;meta=None
            try:
                if label=='C':
                    primary=[{k:row[k] for k in ['status','rating','source_binding']} for row in originals]
                    try:payload=third_packet(packet,*primary)
                    except Exception:
                        provider.ledger.broken=True
                        raise
                else:payload=packet
                raw,meta=provider.send(slot,binding['action'],messages(binding['action'],payload))
                anchored=validate_rating(raw,packet,source['binding'])
                original={'status':'valid','rating':anchored['rating'],'raw_rating':raw,
                          'source_binding':source_binding(binding),'anchor_metadata':{k:v for k,v in anchored.items() if k!='rating'},
                          'provider_metadata':meta}
            except Exception as exc:
                original={'status':'evaluator_failure','rating':None,'raw_rating':raw,
                          'source_binding':source_binding(binding),'anchor_metadata':None,'provider_metadata':meta,
                          'error_type':type(exc).__name__,
                          'reason':str(exc) if isinstance(exc,Rejected) else 'invalid wire schema or coding'}
        originals.append(original)
        try:
            write_new(Path(output_root)/'ratings'/(slot+'.original.json'),original,output_root)
            if label in ['A','B'] and not provider.ledger.broken:
                provider.record_primary(slot,original)
        except Exception:
            provider.ledger.broken=True
            raise
    final=resolve(*originals)
    result={'unit_id':source['unit_id'],'question_id':source['question_id'],'source_binding':source['binding'],
            'source_result_sha256':source['source_result_sha256'],'observed_text':packet['observed_text'],
            'original_ratings':originals,'final':final,'metrics':metrics(final['rating'],rubric['points']),
            'active_seconds':time.monotonic()-started,'new_answer_generated':False,'raw_wire_and_host_anchors_retained':True}
    write_new(Path(output_root)/(source['question_id']+'_L_REF047.result.json'),result,output_root)
    return result


def run_planned(provider, sources, packet_loader, rubrics, directory):
    """Used by the explicit live entry and isolated SYNTHETIC transport tests."""
    results=[];stop_reason=None
    for source in sources:
        packet=packet_loader(source)
        if provider.ledger.broken:
            originals=[stopped_outcome(provider.config['slot_bindings'][source['question_id']+'_L_REF047_'+x]) for x in ['A','B','C']]
            result={'unit_id':source['unit_id'],'question_id':source['question_id'],'source_binding':source['binding'],
                    'source_result_sha256':source['source_result_sha256'],'observed_text':packet['observed_text'],
                    'original_ratings':originals,'final':{'status':'evaluator_failure','rating':None},
                    'metrics':metrics(None,rubrics[source['question_id']]['points']),
                    'new_answer_generated':False,'not_attempted_batch_stopped':True}
            write_new(Path(directory)/(source['question_id']+'_L_REF047.result.json'),result,directory)
        else:
            try:result=evaluate_cell(provider,source,packet,rubrics[source['question_id']],directory)
            except Exception as exc:
                provider.ledger.broken=True
                stop_reason={'error_type':type(exc).__name__,'reason':'evaluation or durable-recording failure'}
                # Do not fabricate or replace an incomplete result; preserve original files and raw records.
                result={'final':{'rating':None},'metrics':metrics(None,rubrics[source['question_id']]['points']),
                        'incomplete_record':True}
            if provider.ledger.broken and stop_reason is None:
                stop_reason={'reason':'batch stopped after count/cap/identity/usage/transport uncertainty'}
        results.append((source,result))
    macros={name:family_macro([{'split':s['split'],'family_id':s['family_id'],'ratio':r['metrics'][name]} for s,r in results])
            for name in ['REF_cov','REF_knowledge','REF_boundary','REF_source_state']}
    report={'planned_questions':len(sources),'observed_result_units':sum(not r.get('not_attempted_batch_stopped') and not r.get('incomplete_record') for _,r in results),
            'valid_final_ratings':sum(r['final']['rating'] is not None for _,r in results),'fixed_macros':macros,
            'attempt_counts':provider.ledger.counts,'reserved_USD':str(provider.ledger.reserved),'actual_bill':'unknown',
            'batch_stopped':provider.ledger.broken,'stop_reason':stop_reason,'new_generations':0,
            'dynamic_metrics':'Per-unit coding only; unknown denominators retained, no invented pooled mean.',
            'all_original_A_B_third_and_raw_retained':True}
    write_new(Path(directory)/'reference_report.json',report,directory)
    return report


def execute_authorized(preparation, config, authorization):
    prep=safe_path(preparation)
    require(config['scope']['stage']=='E02' and config['scope']['new_answers']==0 and config['scope']['saved_answer_rescoring']==13,
            'only saved L13 ratings are executable')
    require(config['scope']['groups']==['L'] and set(config['actions'])=={'judge_output','adjudicate_output'},'no generation or other group')
    gate=LiveGate(prep,config,authorization)
    counter=Counter(prep.parents[2],read_json(prep/'config/counter_assets.json'))
    require(bool(os.environ.get(config['credential_environment_variable'], '').strip()),'credential not configured before identity consumption')
    require(authorization['batch_output']==config['batch_output'],'exact output authorization')
    directory=safe_path(config['batch_output'],prep.parents[2])
    safe_path(directory.parent).mkdir(parents=True,exist_ok=True)
    ledger=Ledger(directory,config,read_json(prep/'config/slots.json'),directory.parent,gate=gate)
    write_new(directory/'authorization.consumed.json',{'authorization_sha256':digest(authorization),'no_reuse':True},directory)
    transport=None
    try:
        transport=LiveHTTP(gate,config)
        provider=Provider(config,counter,ledger,transport,directory/'raw')
        sources=read_json(prep/'inputs/saved_answer_bindings.json')['sources']
        rubrics={r['question_id']:r for r in read_json(prep/'inputs/rubrics.json')['questions']}
        return run_planned(provider,sources,lambda s:read_json(prep/s['packet']),rubrics,directory)
    finally:
        ledger.close()
        if transport:transport.close()
