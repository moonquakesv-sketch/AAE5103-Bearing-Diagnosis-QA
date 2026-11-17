"""Offline arithmetic over explicit synthetic/model-coded rows. No model or project imports."""
from collections import defaultdict
import copy

UNKNOWN = {'machine_unresolved', 'evaluator_failure'}

def ratio(numerator, denominator, status='ok'):
    assert denominator >= 0
    return {'numerator': numerator, 'denominator': denominator,
            'value': None if denominator == 0 or numerator is None else numerator / denominator,
            'status': 'not_applicable' if denominator == 0 else status}

def coverage(rows):
    """Fixed points retained; evaluated-system failure is 0, judge failure is unknown."""
    known = 0.0
    counts = defaultdict(int)
    for r in rows:
        s = r['status']; counts[s] += 1
        if s in UNKNOWN:
            assert r['score'] is None
        elif s in {'tested_failure', 'missing'}:
            assert r['score'] == 0
        else:
            assert s == 'scored' and type(r['score']) in (int, float) and r['score'] in (0, .5, 1)
            assert r['score'] != .5 or r.get('half_credit_allowed') is True
            known += r['score']
    n = len(rows); u = sum(counts[x] for x in UNKNOWN)
    return {'planned': n, 'known_sum': known, 'unknown': u, 'counts': dict(counts),
            'lower': None if n == 0 else known/n,
            'upper': None if n == 0 else (known+u)/n,
            'status': 'not_applicable' if n == 0 else ('bounded' if u else 'ok')}

def supported(point, cards):
    return any(set(combo) <= set(cards) for combo in point['support_any_of'])

def retrieval(points, returned):
    positive = [p for p in points if p['kind'] == 'knowledge']
    rel = {c for p in positive for c in p['relevant_evidence_ids']}
    ids = set(returned)
    unresolved=[p for p in positive if p.get('source_status')=='machine_unresolved']
    known=[p for p in positive if p not in unresolved]
    lo=sum(supported(p, ids) for p in known)
    hi=lo+sum(supported(p,ids) for p in unresolved)
    r=ratio(None if unresolved else lo,len(positive),'machine_unresolved' if unresolved else 'ok')
    r.update(lower=None if not positive else lo/len(positive),upper=None if not positive else hi/len(positive),
             unknown=len(unresolved),source_eligible=len(known))
    return {'R_sup':r,
            'R_rel':ratio(sum(bool(ids & set(p['relevant_evidence_ids'])) for p in positive),len(positive)),
            'P_rel':ratio(len(ids & rel),len(ids))}

def claim_metrics(claims, pairs, returned):
    assert len({c['id'] for c in claims}) == len(claims)
    assert len({(p['claim_id'],p['evidence_id']) for p in pairs}) == len(pairs)
    assert all(p['claim_id'] in {c['id'] for c in claims} for p in pairs)
    weights = {'full':0,'partial':.5,'none':1}
    unknown = [c for c in claims if c['support'] == 'machine_unresolved']
    if unknown:
        u = ratio(None,len(claims),'machine_unresolved')
    else:
        u = ratio(sum(weights[c['support']] for c in claims),len(claims))
    known_weight=sum(weights[c['support']] for c in claims if c['support']!='machine_unresolved')
    u.update(lower=None if not claims else known_weight/len(claims),
             upper=None if not claims else (known_weight+len(unknown))/len(claims),unknown=len(unknown))
    bad = sum(p['evidence_id'] not in returned for p in pairs)
    scores = []
    for p in pairs:
        if p['evidence_id'] not in returned: scores.append(0)
        elif p['support'] == 'machine_unresolved': scores.append(None)
        else: scores.append({'full':1,'partial':.5,'none':0}[p['support']])
    cs=ratio(None if None in scores else sum(scores),len(pairs),'machine_unresolved' if None in scores else 'ok')
    cs.update(lower=None if not scores else sum(x for x in scores if x is not None)/len(scores),
              upper=None if not scores else (sum(x for x in scores if x is not None)+scores.count(None))/len(scores),
              unknown=scores.count(None))
    return {'model_judged_U_claim':u,'C_id_err':ratio(bad,len(pairs)),'model_judged_C_support':cs}

def paired(draft_errors, final_errors, issue_mapping, final_observable=True, report_observable=True):
    d=set(draft_errors); f=set(final_errors)
    assert report_observable or not issue_mapping,'missing report cannot supply observed issue mapping'
    assert len({m['issue_id'] for m in issue_mapping})==len(issue_mapping)
    assert all(m['error_id'] is None or m['error_id'] in d for m in issue_mapping)
    detected={m['error_id'] for m in issue_mapping if m['error_id'] is not None}
    fixed=d-f if final_observable else set()
    retained=d & f if final_observable else d
    return {'V_fix':ratio(len(fixed),len(d)), 'V_miss':ratio(len(retained),len(d)),
            'V_detect':ratio(len(detected),len(d)),
            'V_false':ratio(sum(m['error_id'] is None for m in issue_mapping),len(issue_mapping)) if report_observable
                      else {'numerator':None,'denominator':None,'value':None,'status':'missing'},
            'new_errors':sorted(f-d) if final_observable else None,
            'final_observable':final_observable}

def new_error_bounds(planned, known_added, unobservable):
    assert 0<=known_added<=planned and 0<=unobservable<=planned-known_added
    return {'lower':ratio(known_added,planned), 'upper':ratio(known_added+unobservable,planned)}

def aggregate(rows):
    """Question first, family second. Inputs are explicit fixed-R cell values/bounds."""
    byq=defaultdict(list)
    for r in rows: byq[(r['split'],r['family'],r['question'])].append(r)
    questions=[]; families=defaultdict(list)
    for (split,family,q),rs in sorted(byq.items()):
        lo=sum(r['lower'] for r in rs)/len(rs);hi=sum(r['upper'] for r in rs)/len(rs)
        questions.append({'split':split,'family':family,'question':q,'R':len(rs),'lower':lo,'upper':hi})
        families[(split,family)].append((lo,hi))
    fs=[{'split':s,'family':f,'questions':len(v),'lower':sum(x[0] for x in v)/len(v),
         'upper':sum(x[1] for x in v)/len(v)} for (s,f),v in sorted(families.items())]
    splits={s:{'families':len(v),'lower':sum(x['lower'] for x in v)/len(v),
               'upper':sum(x['upper'] for x in v)/len(v)}
            for s in {f['split'] for f in fs} if (v:=[f for f in fs if f['split']==s])}
    return {'questions':questions,'families':fs,'splits':splits}

def final_projection(record, canonical_cards=None, common_context=None, scoring_rubric=None):
    """Explicit whitelist; final null never imports draft into service text."""
    canonical_cards=canonical_cards or {}
    ids=[c.get('evidence_id') for c in record['actual_complete_cards']]
    assert len(ids)==len(set(ids)) and len(ids)<=3,'global unique complete-card budget exceeded'
    def reject_signals(value):
        forbidden={'group','group_id','split','retrieval_mode','ranking_score','rank_score','verification_pass','content_verification','cost','blind_mapping'}
        if isinstance(value,dict):
            assert not set(value)&forbidden,'nested configuration signal'
            for v in value.values():reject_signals(v)
        elif isinstance(value,list):
            for v in value:reject_signals(v)
    for card in record['actual_complete_cards']:
        assert card==canonical_cards.get(card.get('evidence_id')),'actual card differs from frozen complete card'
    assert record['common_context']==common_context,'common context differs from frozen context'
    assert record['rubric']==scoring_rubric,'scoring rubric differs from frozen projection'
    projected=copy.deepcopy({k:record[k] for k in ['unit_id','business_request','actual_complete_cards',
                                 'common_context','rubric','final_answer','service_message','service_error_category']}
    )
    reject_signals(projected)
    return projected

def pair_projection(record, canonical_cards, common_context, scoring_rubric):
    base=final_projection(record,canonical_cards,common_context,scoring_rubric)
    issue_fields={'issue_id','draft_quote','card_ids','kind','reason'}
    issues=[]
    for issue in record['verification_issues']:
        assert set(issue)==issue_fields,'verification issue contains unapproved metadata'
        issues.append(copy.deepcopy(issue))
    return {'pair_id':record['pair_id'],'business_request':base['business_request'],
            'actual_complete_cards':base['actual_complete_cards'],'common_context':base['common_context'],
            'rubric':base['rubric'],'A':{'draft':record['draft']},
            'B':{'final_answer':base['final_answer'],'service_message':base['service_message']},
            'verification_issues_subprojection':issues}
