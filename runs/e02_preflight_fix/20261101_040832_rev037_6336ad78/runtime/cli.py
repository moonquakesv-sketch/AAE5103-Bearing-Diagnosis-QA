"""Separate future E02 launcher. Default status is read-only; run requires external release."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
from common import require, read_json, file_hash, digest, write_new, safe_path
from corpus import Corpus
from provider import Counter, Ledger, LiveGate, LiveHTTP, Provider
from engine import Runtime, final_packet, pair_packet
from evaluation import AutomaticEvaluation
from reporting import summarize

PREPARATION = Path(__file__).resolve().parents[1]
ROOT = PREPARATION.parents[2]


def load_materials(config, root=ROOT):
    for p,h in config['dataset']['files'].items():require(file_hash(root/p)==h,'dataset hash drift')
    directory=root/config['dataset']['directory']
    from common import strict_json
    requests={q['question_id']:q for q in [strict_json(l) for l in (directory/'requests.jsonl').read_bytes().splitlines()]}
    rubrics={q['question_id']:q for q in read_json(directory/'rubrics.json')['questions']}
    require(sorted(requests)==config['scope']['question_ids'] and len(requests)==config['scope']['Q'],'question scope drift')
    require(sum(len(q['points']) for q in rubrics.values())==config['dataset']['fixed_point_count'],'fixed denominator drift')
    return requests,rubrics


def execute_batch(runtime, mappings, requests, rubrics, directory, metric_ids):
    """Shared production state machine, exercised only with explicit synthetic fixtures here."""
    directory=Path(directory)
    batch_started=time.monotonic()
    for name in ['tested','evaluation','packets','timing']: (directory/name).mkdir(exist_ok=True)
    records=[];cache={};output_evaluations={};pair_evaluations={}
    for m in mappings:
        req=requests[m['question_id']]['business_request'];mode='text' if m['group'] in ['T','TV'] else 'graph'
        cache_key=(m['question_id'],m['repeat'],mode)
        start=time.monotonic()
        if runtime.provider.ledger.broken:
            record={'cell_key':m['cell_key'],'group':m['group'],'business_request':req,'actual_complete_cards':[],
             'retrieval':{'status':'not_run'},'status':'missing','draft':None,'final_answer':None,'service_message':None,
             'service_error_category':'batch_stopped','publishable':False,'provider_metadata':[],'verification_issues':[],
             'verification_observable':False,'tested_latency_seconds':None}
        else:
            try:
                if cache_key not in cache:
                    try:
                        cache[cache_key]={'status':'success','retrieval':runtime.corpus.retrieve(req,mode,3),'failure_type':None}
                    except Exception as exc:
                        cache[cache_key]={'status':'failed','retrieval':None,'failure_type':type(exc).__name__}
                require(cache[cache_key]['status']=='success','shared retrieval attempt failed')
                record=runtime.tested_cell(m['cell_key'],m['group'],req,cache[cache_key]['retrieval'])
                record['tested_latency_seconds']=time.monotonic()-start
            except Exception as exc:
                record={'cell_key':m['cell_key'],'group':m['group'],'business_request':req,'actual_complete_cards':[],
                 'retrieval':{'status':'retrieval_failed'},'status':'tested_failure','draft':None,'final_answer':None,
                 'service_message':'The service could not obtain an approved evidence bundle for this planned cell.',
                 'service_error_category':'retrieval_failure','publishable':False,'provider_metadata':[],'verification_issues':[],
                 'verification_observable':False,'failure_type':cache.get(cache_key,{}).get('failure_type') or type(exc).__name__,'tested_latency_seconds':time.monotonic()-start}
        record['collection_kind']='synthetic_fixture' if req['question'].startswith('SYNTHETIC:') else 'future_authorized_E02'
        write_new(directory/'tested'/(m['cell_key']+'.json'),record,directory)
        records.append(record)
    evaluator=AutomaticEvaluation(runtime,directory/'evaluation')
    # Primary A and B packages have only their own unit input, never another unit or peer rating.
    for m,record in zip(mappings,records):
        evaluation_started=time.monotonic()
        rubric=rubrics[m['question_id']]
        packet=final_packet(m['unit_id'],record,rubric,runtime.corpus.cards)
        write_new(directory/'packets'/(m['unit_id']+'_output.json'),packet,directory)
        output_evaluations[m['unit_id']]=evaluator.evaluate(m['unit_id'],'output',packet,record,rubric)
        if m['group'] in ['TV','GV']:
            packet=pair_packet(m['unit_id'],record,rubric,runtime.corpus.cards)
            write_new(directory/'packets'/(m['unit_id']+'_pair.json'),packet,directory)
            pair_evaluations[m['unit_id']]=evaluator.evaluate(m['unit_id'],'pair',packet,record,rubric)
        record['evaluation_latency_seconds']=time.monotonic()-evaluation_started
        evaluated=[output_evaluations[m['unit_id']]]+([pair_evaluations[m['unit_id']]] if m['group'] in ['TV','GV'] else [])
        record['evaluation_timing_censored']=any(
            'transport timeout' in (item.get('failure_reason') or '') or bool((item.get('metadata') or {}).get('timing_censored'))
            for evaluation in evaluated for item in evaluation.get('originals',[])+[evaluation.get('canonical',{})])
        record['workflow_active_latency_seconds']=(record['tested_latency_seconds']+record['evaluation_latency_seconds']) if record.get('tested_latency_seconds') is not None else None
        record['workflow_timing_censored']=bool(record.get('timing_censored')) or record['evaluation_timing_censored']
        write_new(directory/'timing'/(m['cell_key']+'.json'),
                  dict({k:m[k] for k in ['cell_key','unit_id','group','split']},
                       **{k:record.get(k) for k in ['tested_latency_seconds','timing_censored','evaluation_latency_seconds',
                          'evaluation_timing_censored','workflow_active_latency_seconds','workflow_timing_censored']}),
                  directory)
    result=summarize(records,output_evaluations,pair_evaluations,rubrics,runtime.corpus,mappings,runtime.provider.ledger,metric_ids)
    result['timing']['batch_wall_seconds']=time.monotonic()-batch_started
    write_new(directory/'automatic_report.json',result,directory)
    return result


def launch(authorization_path, output):
    config=read_json(PREPARATION/'config/execution.json')
    authorization=read_json(authorization_path)
    # No corpus query, credentials, output initialization or HTTP client before this gate.
    gate=LiveGate(PREPARATION,config,authorization)
    output_parent=safe_path(ROOT/config['output_parent'],ROOT/'runs')
    output=safe_path(output,output_parent)
    require(output.parent==output_parent and not output.exists(),'fresh immediate batch output required')
    require(authorization.get('batch_output')==str(output.relative_to(ROOT)),'exact future batch path authorization required')
    require(type(authorization.get('authorization_id')) is str and len(authorization['authorization_id'])==64 and all(c in '0123456789abcdef' for c in authorization['authorization_id']),'unique authorization id required')
    requests,rubrics=load_materials(config)
    counter=Counter(ROOT,read_json(PREPARATION/'config/counter_assets.json'))
    corpus=Corpus(ROOT,config['knowledge'],config['approved_contexts'])
    output_parent.mkdir(exist_ok=True)
    lockfd=os.open(safe_path(output_parent/'.e02_project_lock'),os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600)
    fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    transport=None;ledger=None
    try:
        used=output_parent/('authorization_'+authorization['authorization_id']+'.consumed.json')
        write_new(used,{'authorization_sha256':file_hash(authorization_path),'batch_output':str(output.relative_to(ROOT))},output_parent)
        transport=LiveHTTP(gate,config)
        schedule=read_json(PREPARATION/'config/schedule_private.json')
        ledger=Ledger(output,config,schedule['allowed_slots'],output_parent,gate=gate)
        runtime=Runtime(PREPARATION,config,Provider(config,counter,ledger,transport,output/'provider'),corpus)
        metrics=read_json(ROOT/'runs/e01_design/20261031_design_rev034_7c38b1a2/metrics/metric_catalog.json')
        return execute_batch(runtime,schedule['rows'],requests,rubrics,output,[m['metric_id'] for m in metrics])
    finally:
        if transport:transport.close()
        if ledger:ledger.close()
        os.close(lockfd)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('status')
    run=sub.add_parser('run');run.add_argument('--authorization',type=Path,required=True);run.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='status':
        from common import canonical
        print(canonical({'prepared_configuration':file_hash(PREPARATION/'config/execution.json'),'formal_experiment_released':False,'paid_execution_authorized':False,
                         'next':'independent cross-audit and separate exact paid release; this command does not inspect credentials or run questions'}))
    else:launch(args.authorization,args.output)


if __name__=='__main__':main()
