"""Compute all38 registered metrics from immutable records/coding, with explicit missingness."""
from collections import defaultdict
import statistics
from common import require, digest, canonical
import metric_math as mm
from engine import available_components


def ratio(n, d, status='ok'):
    r = mm.ratio(n, d, status)
    r.update(lower=r['value'], upper=r['value'])
    return r


def missing(status='missing'):
    return {'numerator': None, 'denominator': None, 'value': None, 'lower': None, 'upper': None, 'status': status}


def bounded(rows):
    n = len(rows)
    known = sum(x for x in rows if x is not None)
    u = sum(x is None for x in rows)
    r = ratio(None if u else known, n, 'machine_unresolved' if u else 'ok')
    r.update(lower=None if not n else known/n, upper=None if not n else (known+u)/n, unknown=u)
    return r


def canonical_rating(evaluation):
    final = (evaluation or {}).get('canonical', {})
    return final.get('rating') if final.get('status') in ['resolved_primary_agreement', 'resolved_third_agent'] else None


def evaluation_packages(evaluation, applicable=True):
    originals = (evaluation or {}).get('originals', [])
    final = (evaluation or {}).get('canonical', {})
    initial = {'assigned': 2 if applicable else 0, 'failed': sum(o['status']=='evaluator_failure' for o in originals),
               'unknown': max(0, (2 if applicable else 0)-len(originals))}
    review = final.get('review', {})
    assigned = int(review.get('assigned', bool(final.get('triggers')))) if applicable else 0
    state = review.get('status')
    third = {'assigned': assigned, 'failed': int(bool(assigned) and state=='evaluator_failure'),
             'unknown': int(bool(assigned) and state not in ['valid','evaluator_failure']),
             'assignment_unknown': applicable and evaluation is None}
    return {'initial': initial, 'review': third}


def package_rate(packages):
    phases = [p[k] for p in packages for k in ['initial','review']]
    n = sum(p['assigned'] for p in phases)
    failed = sum(p['failed'] for p in phases)
    unknown = sum(p['unknown'] for p in phases)
    result = ratio(None if unknown else failed, n, 'unknown' if unknown else 'ok')
    result.update(lower=failed/n if n else None, upper=(failed+unknown)/n if n else None, unknown=unknown)
    return result


def timing_stats(values, planned=None, censored=None):
    observed = [x for x in values if x is not None]
    planned = len(values) if planned is None else planned
    return {'unit':'seconds','planned':planned,'observed':len(observed),'missing':planned-len(observed),
            'censored':censored or [],'coverage':ratio(len(observed),planned),
            'mean':statistics.mean(observed) if observed else None,
            'median':statistics.median(observed) if observed else None,
            'minimum':min(observed) if observed else None,'maximum':max(observed) if observed else None}


def stage_observations(events, ledger, selected_slots=None):
    reserves = {e['slot']:e for e in events if e['event']=='reserve' and
                (selected_slots is None or e['slot'] in selected_slots)}
    usage = {}
    timing = {}
    local = {}
    for e in events:
        if e['event']!='result' or e['slot'] not in reserves:
            continue
        meta = e['metadata']
        if e['status']=='received':
            usage[e['slot']] = meta
        if 'latency_seconds' in meta:
            timing[e['slot']] = meta
        if 'local_input_tokens' in meta:
            local[e['slot']] = meta['local_input_tokens']
    by_action = {}
    time_by_action = {}
    def totals(values):
        known = [x for x in values if type(x) is int and x>=0]
        return {'total':sum(known) if len(known)==len(values) else None,'known_sum':sum(known),
                'assigned':len(values),'observed':len(known),'missing':len(values)-len(known)}
    for action in ledger.counts:
        slots = [s for s in reserves if ledger.allowed[s]==action]
        categories = {}
        for name, field in [('prompt_tokens','prompt_tokens'),('completion_tokens','completion_tokens'),
                            ('total_tokens','total_tokens'),('cache_hit_tokens','prompt_cache_hit_tokens'),
                            ('cache_miss_tokens','prompt_cache_miss_tokens'),('reasoning_tokens','reasoning_tokens')]:
            values = []
            for slot in slots:
                u = usage.get(slot, {}).get('usage', {})
                detail = u.get('completion_tokens_details')
                values.append(detail.get(field) if name=='reasoning_tokens' and isinstance(detail,dict)
                              else (None if name=='reasoning_tokens' else u.get(field)))
            categories[name] = totals(values)
        categories['local_input_tokens'] = totals([local.get(s) for s in slots])
        by_action[action] = categories
        censored = [{'slot':s,'observed_lower_bound_seconds':timing[s]['latency_seconds']} for s in slots
                    if s in timing and timing[s].get('timing_censored')]
        time_by_action[action] = timing_stats([timing[s]['latency_seconds'] if s in timing and not timing[s].get('timing_censored') else None for s in slots],
                                              len(slots),censored)
    return by_action, time_by_action


def phase_timing(records):
    phases={'tested_total':('tested_latency_seconds','timing_censored'),
            'evaluation_total':('evaluation_latency_seconds','evaluation_timing_censored'),
            'workflow_active_total':('workflow_active_latency_seconds','workflow_timing_censored')}
    return {name:timing_stats([r.get(field) if not r.get(censor) else None for r in records],
                censored=[{'cell_key':r['cell_key'],'observed_lower_bound_seconds':r.get(field)}
                          for r in records if r.get(censor)])
            for name,(field,censor) in phases.items()}


def coverage_roles(points):
    return {role:bounded([p['score'] for p in points if p['content_role']==role])
            for role in ['positive_knowledge','boundary','source_status']}

def output_metrics(record, rubric, evaluation, canonical_cards):
    points = rubric['points']
    positive = [p for p in points if p['content_role'] == 'positive_knowledge']
    cards = record['actual_complete_cards'];ids={c['evidence_id'] for c in cards}
    rel = {c['evidence_id'] for p in positive for c in p['citations']}
    r = {}
    r['R_rel'] = ratio(sum(bool(ids & {c['evidence_id'] for c in p['citations']}) for p in positive), len(positive))
    r['P_rel'] = ratio(len(ids & rel), len(ids))
    r['R_sup'] = ratio(sum(any(set(c) <= ids for c in p['support_any_of']) for p in positive), len(positive))
    tasks = set(p['task_type'] for p in positive)
    related_by_task = {t:{c['evidence_id'] for p in positive if p['task_type']==t for c in p['citations']} for t in tasks}
    r['R_task'] = ratio(sum(any(c['evidence_id'] in related_by_task[t] and c['task_type']==t for c in cards) for t in tasks),len(tasks))
    requested = record['business_request']['task_types']
    r['all_requested_task_structure'] = ratio(sum(any(c['task_type']==t for c in cards) for t in requested),len(requested))
    r['R_card'] = ratio(sum(c == canonical_cards.get(c['evidence_id']) for c in cards),len(cards))
    cond_keys = [(c['evidence_id'],k) for c in cards if c['evidence_id'] in rel for k in
                 ['conditions','limitations']+[f'necessary_conditions/{i}' for i in range(len(c.get('necessary_conditions',[])))]]
    def field(c,k):
        return c['necessary_conditions'][int(k.split('/')[1])] if '/' in k else c[k]
    byid={c['evidence_id']:c for c in cards}
    r['R_cond'] = ratio(sum(field(byid[e],k)==field(canonical_cards[e],k) for e,k in cond_keys),len(cond_keys))
    if record.get('retrieval',{}).get('status') in ['not_run','retrieval_failed']:
        for k in ['P_rel','R_card','R_cond']:r[k]=missing('missing')
    rating = canonical_rating(evaluation)
    failed = record['status'] in ['tested_failure','missing']
    r['service_completion'] = ratio(int(record['status'] in ['completed','reasonable_insufficiency']),1)
    r['tested_failure_rate'] = ratio(int(record['status']=='tested_failure'),1)
    r['missing_rate'] = ratio(int(record['status']=='missing'),1)
    r['publishable'] = ratio(int(record['publishable']),1)
    r['model_judged_release_adherence'] = ratio(0,1) if not record['publishable'] else bounded([rating['release_adherence'] if rating else None])
    timed = record.get('tested_latency_seconds') is not None and not record.get('timing_censored',False)
    r['latency'] = ratio(record['tested_latency_seconds'],1) if timed else missing('right_censored' if record.get('timing_censored') else 'missing')
    r['timing_coverage'] = ratio(int(timed),1)
    status = 'evaluator_failure' if not evaluation or evaluation['canonical']['status']=='evaluator_failure' else 'machine_unresolved'
    semantic_names = ['model_judged_U_claim','C_id_err','model_judged_C_support','model_judged_C_cov','model_judged_A_cond','model_judged_A_overref']
    if rating is None:
        r['model_judged_A_cov'] = bounded([0 if failed else None for _ in points])
        r['model_judged_A_bound'] = bounded([0 if failed else None for p in points if p['content_role']=='boundary'])
        applicable_insuff=any(c['applicable'] for c in rubric['insufficiency_criteria'])
        r['model_judged_A_insuff'] = bounded([0 if failed else None]*3 if applicable_insuff else [])
        for name in semantic_names:r[name]=missing(status)
    else:
        r['model_judged_A_cov'] = bounded([x['score'] for x in rating['point_rows']])
        r['model_judged_A_bound'] = bounded([x['score'] for x in rating['boundary_rows']])
        r['model_judged_A_insuff'] = bounded([x['score'] for x in rating['insufficiency']['rows']])
        if failed:
            for name in semantic_names:r[name]=missing('tested_failure')
        else:
            cm = mm.claim_metrics(rating['claims'],rating['citation_pairs'],ids)
            for name,v in cm.items():r[name]=v
            external = [c for c in rating['claims'] if c['requires_external']]
            def cited(c):
                if c['support']=='machine_unresolved':return None
                pairs={p['evidence_id']:p for p in rating['citation_pairs'] if p['claim_id']==c['id']}
                return int(c['support']=='full' and any(combo and set(combo)<=set(pairs) and
                           all(pairs[e]['id_valid'] and (pairs[e]['support']=='full' if len(combo)==1 else pairs[e]['support'] in ['full','partial']) for e in combo)
                           for combo in c['support_any_of']))
            r['model_judged_C_cov'] = bounded([cited(c) for c in external])
            r['model_judged_A_cond'] = bounded([x['score'] for x in rating['conditions']])
            targets=next(i for i in rubric['insufficiency_criteria'] if i['criterion_id']=='I03')['requested_supported_component_ids']
            fully=bool(targets) and set(available_components(rubric,cards))==set(targets)
            r['model_judged_A_overref']=ratio(int(rating['full_refusal']),1) if fully else ratio(0,0)
    originals=evaluation.get('originals',[]) if evaluation else []
    r['evaluator_failure_rate']=package_rate([evaluation_packages(evaluation)])
    rowstates=[x['status'] for x in rating['point_rows']] if rating else [status]*len(points)
    r['machine_unresolved_rate']=ratio(sum(x=='machine_unresolved' for x in rowstates),len(points))
    if len(originals)==2 and all(o['status']=='valid' for o in originals):
        left={x['id']:x['score'] for x in originals[0]['rating']['point_rows']+originals[0]['rating']['insufficiency']['rows']}
        right={x['id']:x['score'] for x in originals[1]['rating']['point_rows']+originals[1]['rating']['insufficiency']['rows']}
        comparable=[k for k in left if left[k] is not None and right[k] is not None]
        r['judge_agreement']=ratio(sum(left[k]==right[k] for k in comparable),len(comparable))
        r['judge_absolute_difference']=ratio(sum(abs(left[k]-right[k]) for k in comparable),len(comparable))
    else:
        r['judge_agreement']=missing('evaluator_failure');r['judge_absolute_difference']=missing('evaluator_failure')
    return r


def pair_metrics(record,evaluation):
    keys=['model_judged_V_fix','model_judged_V_miss','model_judged_V_detect','model_judged_V_false','model_judged_V_new_run','model_judged_V_new_count']
    if record['group'] not in ['TV','GV']:return {k:ratio(0,0) for k in keys}
    value=canonical_rating(evaluation)
    if not value or not record.get('draft') or value['evaluation_status']!='complete':
        r={k:missing('evaluator_failure' if not value else 'machine_unresolved') for k in keys}
        r['model_judged_V_new_run'].update(lower=0,upper=1,range_scope='full_planned_pair')
        return r
    result=mm.paired([e['error_id'] for e in value['draft_errors']],[e['error_id'] for e in value['final_errors']],
                     value['report_error_mapping'],value['final_observable'],value['report_observable'])
    r={'model_judged_'+k:v for k,v in result.items() if k in ['V_fix','V_miss','V_detect','V_false']}
    if result['new_errors'] is None:
        r.update(model_judged_V_new_run=missing(),model_judged_V_new_count=missing())
        r['model_judged_V_new_run'].update(lower=0,upper=1,range_scope='full_planned_pair')
    else:
        r.update(model_judged_V_new_run=ratio(int(bool(result['new_errors'])),1),model_judged_V_new_count=ratio(len(result['new_errors']),1))
    return r


def accounting(events, record):
    key=record['cell_key']
    reserves={e['slot']:e for e in events if e['event']=='reserve' and e['slot'].startswith(key+'_')}
    known={e['slot']:e['metadata'] for e in events if e['event']=='result' and e['status']=='received' and e['slot'] in reserves}
    if not reserves:
        return {'token_usage':ratio(0,1),'cost':missing('unknown'),'usage_coverage':ratio(0,1),'billed_cost_coverage':ratio(0,1)}
    complete=len(known)==len(reserves)
    return {'token_usage':ratio(sum(m['usage']['total_tokens'] for m in known.values()),1) if complete else missing('unknown'),
            'cost':missing('unknown'),'usage_coverage':ratio(int(complete),1),'billed_cost_coverage':ratio(0,1)}


def aggregate(rows):
    # Each metric preserves all planned cells. Unknown diagnostic denominators are not invented.
    results={}
    names=sorted({k for row in rows for k in row['metrics']})
    for group in ['T','TV','G','GV']:
        selected=[r for r in rows if r['group']==group]
        group_result={}
        for name in names:
            applicable=[r for r in selected if r['metrics'][name]['status']!='not_applicable']
            observed=[r for r in applicable if r['metrics'][name].get('value') is not None]
            family=defaultdict(lambda:defaultdict(list))
            for row in applicable:family[(row['split'],row['family_id'])][row['question_id']].append(row['metrics'][name])
            summaries=[]
            for (split,fam),questions in family.items():
                qvalues=[];qlow=[];qhigh=[]
                for values in questions.values():
                    qvalues.append(sum(v['value'] for v in values)/len(values) if all(v.get('value') is not None for v in values) else None)
                    qlow.append(sum(v['lower'] for v in values)/len(values) if all(v.get('lower') is not None for v in values) else None)
                    qhigh.append(sum(v['upper'] for v in values)/len(values) if all(v.get('upper') is not None for v in values) else None)
                summaries.append({'split':split,'family_id':fam,'question_count':len(questions),'value':sum(qvalues)/len(qvalues) if all(v is not None for v in qvalues) else None,
                 'lower':sum(qlow)/len(qlow) if all(v is not None for v in qlow) else None,'upper':sum(qhigh)/len(qhigh) if all(v is not None for v in qhigh) else None})
            split_summaries={}
            for split in ['debug','operational_reserve']:
                fs=[x for x in summaries if x['split']==split]
                split_summaries[split]={'families':len(fs),**{k:sum(f[k] for f in fs)/len(fs) if fs and all(f[k] is not None for f in fs) else None for k in ['value','lower','upper']}}
            nums=[r['metrics'][name]['numerator'] for r in observed];dens=[r['metrics'][name]['denominator'] for r in observed]
            group_result[name]={'planned_cells':len(selected),'applicable_cells':len(applicable),'observed_cells':len(observed),
              'not_applicable_cells':len(selected)-len(applicable),'unknown_or_missing_cells':len(applicable)-len(observed),
              'status_counts':{s:sum(r['metrics'][name]['status']==s for r in selected) for s in sorted({r['metrics'][name]['status'] for r in selected})},
              'diagnostic_observed_mean':sum(r['metrics'][name]['value'] for r in observed)/len(observed) if observed else None,
              'observed_micro':ratio(sum(nums),sum(dens)) if observed else missing(), 'families':summaries,'split_macro':split_summaries}
            if name=='latency':
                group_result[name]['observed_timing']=timing_stats([r['metrics'][name].get('value') for r in selected])
                group_result[name]['observed_timing_by_split']={split:timing_stats([r['metrics'][name].get('value') for r in selected if r['split']==split])
                                                               for split in ['debug','operational_reserve']}
        results[group]=group_result
    return results


def summarize(records, outputs, pairs, rubrics, corpus, mappings, ledger, metric_ids):
    events=[__import__('json').loads(l) for l in ledger.expected_bytes.splitlines()]
    mapped={m['cell_key']:m for m in mappings};rows=[]
    retrievals={}
    for record in records:
        m=mapped[record['cell_key']];q=rubrics[m['question_id']]
        metrics=output_metrics(record,q,outputs.get(m['unit_id']),corpus.cards)
        metrics.update(pair_metrics(record,pairs.get(m['unit_id'])))
        packages={'output':evaluation_packages(outputs.get(m['unit_id'])),
                  'pair':evaluation_packages(pairs.get(m['unit_id']),record['group'] in ['TV','GV'])}
        metrics['evaluator_failure_rate']=package_rate(list(packages.values()))
        metrics.update(accounting(events,record))
        if record.get('retrieval',{}).get('status') not in ['not_run','retrieval_failed']:
            retrievals[(m['question_id'],m['repeat'], 'text' if m['group'] in ['T','TV'] else 'graph')]=record['actual_complete_cards']
        metrics['evidence_length_difference']=missing()
        for value in metrics.values():
            if value.get('value') is not None:
                value.setdefault('lower',value['value']);value.setdefault('upper',value['value'])
        require(set(metrics)==set(metric_ids),'metric implementation does not cover all registered IDs')
        coding=canonical_rating(outputs.get(m['unit_id']))
        bypoint={x['id']:x for x in coding['point_rows']} if coding else {}
        ps=[]
        for p in q['points']:
            original=bypoint.get(p['point_id'])
            if original:score,state=original['score'],original['status']
            elif record['status'] in ['tested_failure','missing']:score,state=0,record['status']
            else:score,state=None,'evaluator_failure'
            ps.append({'point_id':p['point_id'],'task_type':p['task_type'],'content_role':p['content_role'],'score':score,'status':state})
        rows.append(dict(m,metrics=metrics,point_scores=ps,canonical_coding=coding,evaluation_packages=packages,
                         coverage_by_role=coverage_roles(ps),
                         predicted_class=record['business_request']['predicted_class'],task_types=record['business_request']['task_types']))
    differences=[]
    for q,repeat in sorted({(m['question_id'],m['repeat']) for m in mappings}):
        t=retrievals.get((q,repeat,'text'));g=retrievals.get((q,repeat,'graph'))
        if t is None or g is None:continue
        delta=len(canonical(t).encode())-len(canonical(g).encode());chars=len(canonical(t))-len(canonical(g))
        differences.append({'question_id':q,'repeat':repeat,'T_minus_G_UTF8_bytes':delta,'T_minus_G_characters':chars,'token_difference':'not_computed_from_bytes'})
        for row in rows:
            if row['question_id']==q and row['repeat']==repeat:row['metrics']['evidence_length_difference']=ratio(delta,1)
    planned_pairs=[m for m in mappings if m['group'] in ['TV','GV']]
    observed_new=[r['metrics']['model_judged_V_new_run']['value'] for r in rows if r['group'] in ['TV','GV']]
    added=sum(v==1 for v in observed_new);unknown=sum(v is None for v in observed_new)
    strata=defaultdict(list)
    for row in rows:
        for p in row['point_scores']:
            for dim,value in [('class',row['predicted_class']),('task',p['task_type']),('family',row['family_id'])]:
                strata[(row['group'],row['split'],dim,value)].append((row['question_id'],p))
    point_strata=[]
    for (group,split,dim,value),items in sorted(strata.items()):
        scores=[p['score'] for _,p in items];b=bounded(scores)
        point_strata.append({'group':group,'split':split,'dimension':dim,'value':value,'questions':len({q for q,_ in items}),
            'planned':len(items),'knowledge':sum(p['content_role']=='positive_knowledge' for _,p in items),
            'boundary':sum(p['content_role']=='boundary' for _,p in items),'source_status':sum(p['content_role']=='source_status' for _,p in items),
            'A_cov':b,'coverage_by_role':coverage_roles([p for _,p in items]),
            'states':{s:sum(p['status']==s for _,p in items) for s in {p['status'] for _,p in items}}})
    usage_by_action, _ = stage_observations(events,ledger)
    censored_slots={e['slot'] for e in events if e['event']=='result' and e.get('metadata',{}).get('timing_censored')}
    timing_records=[]
    for record in records:
        m=mapped[record['cell_key']]
        evaluation_censored=bool(record.get('evaluation_timing_censored')) or any(s.startswith(m['unit_id']+'_') for s in censored_slots)
        timing_records.append(dict(record,evaluation_timing_censored=evaluation_censored,
                              workflow_timing_censored=bool(record.get('timing_censored')) or evaluation_censored))
    def timing_for(selected):
        prefixes={mapped[r['cell_key']][k]+'_' for r in selected for k in ['cell_key','unit_id']}
        slots={s for s in ledger.allowed if any(s.startswith(p) for p in prefixes)}
        return dict(phase_timing(selected),provider_by_action=stage_observations(events,ledger,slots)[1])
    timing=timing_for(timing_records)
    timing['by_group']={g:timing_for([r for r in timing_records if mapped[r['cell_key']]['group']==g]) for g in ['T','TV','G','GV']}
    timing['by_split']={s:timing_for([r for r in timing_records if mapped[r['cell_key']]['split']==s]) for s in ['debug','operational_reserve']}
    timing['by_group_split']={g:{s:timing_for([r for r in timing_records if mapped[r['cell_key']]['group']==g and mapped[r['cell_key']]['split']==s])
                               for s in ['debug','operational_reserve']} for g in ['T','TV','G','GV']}
    timing['per_cell']=[dict({k:mapped[r['cell_key']][k] for k in ['cell_key','unit_id','group','split']},
                       **{k:r.get(k) for k in ['tested_latency_seconds','timing_censored','evaluation_latency_seconds',
                          'evaluation_timing_censored','workflow_active_latency_seconds','workflow_timing_censored']}) for r in timing_records]
    timing['scope']='active phases; batch wall is separate; timeout observations retain lower bounds and do not enter complete means'
    role_rows={role:[dict(row,metrics={'model_judged_A_cov':row['coverage_by_role'][role]}) for row in rows]
               for role in ['positive_knowledge','boundary','source_status']}
    service_pairs=[]
    for record in records:
        if record['group'] not in ['TV','GV']:continue
        m=mapped[record['cell_key']];value=canonical_rating(pairs.get(m['unit_id']))
        errors=[e for e in value['final_errors'] if any(s['channel']=='service_message' for s in e['spans'])] if value else None
        draft_ids={e['error_id'] for e in value['draft_errors']} if value else set()
        aligned=bool(value) and value['evaluation_status']=='complete' and record.get('draft') is not None
        new_service=[e['error_id'] for e in errors if e['error_id'] not in draft_ids] if aligned else None
        service_pairs.append({'unit_id':m['unit_id'],'final_body_present':record['final_answer'] is not None,
          'service_message_present':bool(record.get('service_message')),'service_error_ids':[e['error_id'] for e in errors] if errors is not None else None,
          'observed_service_error_count':len(errors) if errors is not None else None,
          'new_service_error_ids':new_service,'new_service_error_count':len(new_service) if new_service is not None else None,
          'scope':'quoted service errors retained separately; absent body never earns correction credit or zero V_new'})
    failure_breakdown=[]
    for group in ['T','TV','G','GV']:
        for split in ['debug','operational_reserve']:
            selected=[row for row in rows if row['group']==group and row['split']==split]
            for kind in ['output','pair']:
                for phase in ['initial','review']:
                    counts=[row['evaluation_packages'][kind][phase] for row in selected]
                    totals={k:sum(x[k] for x in counts) for k in ['assigned','failed','unknown']}
                    n=totals['assigned'];f=totals['failed'];u=totals['unknown']
                    rate=ratio(None if u else f,n,'unknown' if u else 'ok')
                    rate.update(lower=f/n if n else None,upper=(f+u)/n if n else None)
                    failure_breakdown.append(dict(group=group,split=split,kind=kind,phase=phase,failure_rate=rate,
                        assignment_unknown=sum(bool(x.get('assignment_unknown')) for x in counts),**totals))
    return {'schema':'e02.automatic_report.rev037','rows':rows,'group_metrics':aggregate(rows),'evidence_length_pairs':differences,
      'role_coverage':{role:aggregate(rr) for role,rr in role_rows.items()},
      'service_pair_diagnostics':service_pairs,'evaluation_failure_breakdown':failure_breakdown,
      'timing':timing,
      'measurement_units':{'task_diagnostics':'whole-cell subsets containing requested task; point attribution is fixed_point_strata',
                           'machine_unresolved_rate':'fixed requested point rows; I/claim/condition/pair unresolved statuses remain in raw coding',
                           'judge_agreement':'comparable fixed point and I score rows, not semantic accuracy'},
      'fixed_point_strata':point_strata,
      'class_diagnostics':{c:aggregate([r for r in rows if r['predicted_class']==c]) for c in {r['predicted_class'] for r in rows}},
      'task_diagnostics':{t:aggregate([r for r in rows if t in r['task_types']]) for t in {t for r in rows for t in r['task_types']}},
      'V_new_full_planned_bounds':mm.new_error_bounds(len(planned_pairs),added,unknown),
      'accounting':{'total_reserved_USD':str(ledger.reserved),'attempts_by_action':ledger.counts,'actual_bill':'unknown',
       'usage_by_action':usage_by_action,
       'received_usage_by_action':{a:{name:values[name]['total'] for name in ['prompt_tokens','completion_tokens']} for a,values in usage_by_action.items()},
       'local_counter_billing_equivalence_proven':False},
      'limitations':['model_judged adherence, not accuracy or OEM/expert truth','one positive reserve family/card/cell','context independence not independent errors','debug and reserve separately reported','same questions/repeats do not create devices']}
