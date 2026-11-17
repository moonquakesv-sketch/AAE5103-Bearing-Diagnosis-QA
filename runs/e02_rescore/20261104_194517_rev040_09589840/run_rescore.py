"""Single-use scoring-only E02 batch. Does not import CLI, Corpus or tested execution entry."""
from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal
import datetime
import json
import sys
import time

PREPARATION = Path(__file__).absolute().parent
PROJECT = PREPARATION.parents[2]
sys.path.insert(0, str(PREPARATION / 'runtime'))
from common import read_json, read_bytes, snapshot, digest, write_new, require, canonical, file_hash, safe_path
from provider import Counter, Ledger, Provider, LiveGate, LiveHTTP
from rescore_evaluation import Evaluation
from reporting import summarize, stage_observations


def validate_inputs(preparation, schedule, records, rubrics, corpus):
    import jsonschema
    from engine import final_packet, pair_packet
    for mapping in schedule['rows']:
        record = records[mapping['cell_key']]
        rubric = rubrics[mapping['question_id']]
        for kind in (['output', 'pair'] if mapping['group'] in ['TV', 'GV'] else ['output']):
            packet = read_json(preparation / 'inputs/packets' / (mapping['unit_id'] + '_' + kind + '.json'))
            schema = read_json(preparation / 'schemas' / ('projection.json' if kind == 'output' else 'pair_projection.json'))
            jsonschema.Draft202012Validator(schema).validate(packet)
            rebuilt = (final_packet if kind == 'output' else pair_packet)(mapping['unit_id'], record, rubric, corpus.cards)
            require(packet == rebuilt, 'scoring projection does not match frozen tested record/rubric/cards')
            if kind == 'output':
                require('draft' not in packet and 'group' not in packet and 'split' not in packet, 'hidden output projection metadata')


def combined_ledger(ledger, preparation):
    """Retain old tested cost/time; exclude failed OLD evaluation from the new metric ledger.

    Total historical failed-evaluation overhead remains separately reported, never erased.
    """
    import json
    old_events = [json.loads(line) for line in read_bytes(preparation / 'inputs/old_ledger.jsonl').splitlines()]
    tested_actions = {'generate', 'verify_initial', 'revise', 'verify_final'}
    old_reserves = {e['slot']: e for e in old_events if e['event'] == 'reserve' and e['action'] in tested_actions}
    selected = [e for e in old_events if e.get('slot') in old_reserves]
    old_schedule = read_json(PROJECT / 'runs/e02_preflight_fix/20261101_040832_rev037_6336ad78/config/schedule_private.json')
    old_allowed = {s: a for s, a in old_schedule['allowed_slots'].items() if a in tested_actions}
    counts = dict(ledger.counts)
    for event in old_reserves.values():
        counts[event['action']] = counts.get(event['action'], 0) + 1
    old_reserve = sum((Decimal(e['amount_USD']) for e in old_reserves.values()), Decimal(0))
    events = selected + [json.loads(line) for line in ledger.expected_bytes.splitlines()]
    return SimpleNamespace(expected_bytes=('\n'.join(canonical(e) for e in events) + '\n').encode(),
                           counts=counts, allowed=dict(old_allowed, **ledger.allowed),
                           reserved=old_reserve + ledger.reserved)


def preflight(preparation=PREPARATION):
    preparation = safe_path(preparation)
    config = read_json(preparation / 'config/execution.json')
    authorization = read_json(preparation / 'authorization.json')
    require(authorization.get('output_directory') == str((preparation / 'execution').relative_to(PROJECT)), 'single exact scoring output required')
    gate = LiveGate(preparation, config, authorization)
    require(authorization.get('source_batch') == config['scope']['source_batch'], 'source batch identity mismatch')
    # Validate all current source bytes and the native counter before consuming authorization or reading a key.
    counter = Counter(PROJECT, read_json(preparation / 'config/counter_assets.json'))
    schedule = read_json(preparation / 'config/schedule_private.json')
    rubric_file = read_json(preparation / 'inputs/rubrics.json')
    rubrics = {q['question_id']: q for q in rubric_file['questions']}
    cards = read_json(preparation / 'inputs/canonical_cards.json')
    corpus = SimpleNamespace(cards={card['evidence_id']: card for card in cards})
    records = {m['cell_key']: read_json(preparation / 'inputs/tested' / (m['cell_key'] + '.json')) for m in schedule['rows']}
    require(len(records) == 52 and sum(r['group'] in ['TV', 'GV'] for r in records.values()) == 26, 'frozen scope mismatch')
    require(set(config['actions']) == {'judge_output', 'judge_pair', 'adjudicate_output', 'adjudicate_pair'}, 'scoring-only roles required')
    validate_inputs(preparation, schedule, records, rubrics, corpus)
    require(not (preparation / 'authorization.consumed.json').exists(), 'authorization already consumed; no resume/reset')
    require(not (preparation / 'execution').exists(), 'new scoring directory required; no resume/reset')
    return preparation, config, authorization, gate, counter, schedule, rubrics, corpus, records


def run(preparation=PREPARATION):
    preparation, config, authorization, gate, counter, schedule, rubrics, corpus, records = preflight(preparation)
    # Missing credentials cause no consumed marker and no ledger. Values never leave LiveHTTP memory.
    transport = LiveHTTP(gate, config)
    ledger = None
    started = time.monotonic()
    try:
        write_new(preparation / 'authorization.consumed.json', {'authorization_sha256': file_hash(preparation / 'authorization.json'),
                  'output_directory': authorization['output_directory'], 'one_execution_only': True}, preparation)
        ledger = Ledger(preparation / 'execution', config, schedule['allowed_slots'], preparation, gate=gate)
        provider = Provider(config, counter, ledger, transport, ledger.directory / 'provider')
        evaluator = Evaluation(preparation, config, provider, ledger.directory / 'evaluation')
        (ledger.directory / 'progress').mkdir()
        outputs, pairs = {}, {}
        wall_start = datetime.datetime.now().astimezone().isoformat()
        for index, mapping in enumerate(schedule['rows'], 1):
            record = records[mapping['cell_key']]
            unit = mapping['unit_id']
            rubric = rubrics[mapping['question_id']]
            phase_started = time.monotonic()
            packet = read_json(preparation / 'inputs/packets' / (unit + '_output.json'))
            outputs[unit] = evaluator.evaluate(unit, 'output', packet, record, rubric)
            if mapping['group'] in ['TV', 'GV'] and not ledger.broken:
                packet = read_json(preparation / 'inputs/packets' / (unit + '_pair.json'))
                pairs[unit] = evaluator.evaluate(unit, 'pair', packet, record, rubric)
            record['evaluation_latency_seconds'] = time.monotonic() - phase_started
            tested_time = record.get('tested_latency_seconds')
            record['workflow_active_latency_seconds'] = (tested_time + record['evaluation_latency_seconds']) if tested_time is not None else None
            record['evaluation_timing_censored'] = any(o.get('metadata', {}).get('timing_censored', False)
                for e in [outputs.get(unit), pairs.get(unit)] if e for o in e['originals'] if o.get('metadata'))
            current_events = [json.loads(line) for line in ledger.expected_bytes.splitlines()]
            record['evaluation_timing_censored'] = record['evaluation_timing_censored'] or any(
                e.get('slot', '').startswith(unit + '_') and e.get('metadata', {}).get('timing_censored')
                for e in current_events if e['event'] == 'result')
            record['workflow_timing_censored'] = bool(record.get('timing_censored')) or record['evaluation_timing_censored']
            progress = {'cell': index, 'planned_cells': 52, 'cell_key': mapping['cell_key'],
                        'output_status': outputs[unit]['canonical']['status'],
                        'pair_status': pairs.get(unit, {}).get('canonical', {}).get('status'),
                        'reserved_calls': len(ledger.used), 'reserve_USD': str(ledger.reserved),
                        'ledger_broken': ledger.broken}
            write_new(ledger.directory / 'progress' / f'{index:03}.json', progress, ledger.directory)
            print(canonical(progress), flush=True)
            if ledger.broken:
                break
        metric_ids = [m['metric_id'] for m in read_json(preparation / 'inputs/metric_catalog.json')]
        combined = combined_ledger(ledger, preparation)
        report = summarize(list(records.values()), outputs, pairs, rubrics, corpus, schedule['rows'], combined, metric_ids)
        combined_events = [json.loads(line) for line in combined.expected_bytes.splitlines()]
        report['phase_accounting_by_cell'] = []
        for mapping in schedule['rows']:
            selected_tested = {s for s in combined.allowed if s.startswith(mapping['cell_key'] + '_')}
            selected_evaluation = {s for s in combined.allowed if s.startswith(mapping['unit_id'] + '_')}
            report['phase_accounting_by_cell'].append({
                'cell_key': mapping['cell_key'],
                'historical_tested_service_by_action': stage_observations(combined_events, combined, selected_tested)[0],
                'new_evaluation_by_action': stage_observations(combined_events, combined, selected_evaluation)[0]})
        report['measurement_units']['token_usage'] = 'Original service metric counts tested cell_key stages only; new evaluation units are separately reported in phase_accounting_by_cell and usage_by_action. No old evaluation spending is erased.'
        report.update(schema='e02.automatic_report.rev040.rescore_of_rev037',
                      source_tested_batch=config['scope']['source_batch'], new_tested_outputs=0,
                      original_tested_bytes_unchanged=True,
                      rescore_accounting={'reserved_USD': str(ledger.reserved), 'reserved_calls': len(ledger.used),
                                         'attempts_by_action': ledger.counts, 'hard_cap_USD': config['proposed_hard_budget_USD'],
                                         'actual_bill': 'unknown'},
                      historical_failed_evaluation_accounting=read_json(preparation / 'inputs/execution_summary.json'),
                      workflow_timing_note='Original tested activity plus new evaluation activity; the calendar gap is excluded. '
                                           'No new generation/retrieval/classification. Old failed evaluation overhead remains separately reported.',
                      rescore_started_at=wall_start, rescore_batch_wall_seconds=time.monotonic() - started,
                      stopped_for_transport_or_accounting=ledger.broken)
        write_new(ledger.directory / 'automatic_report.json', report, ledger.directory)
        write_new(ledger.directory / 'timing_records.json', list(records.values()), ledger.directory)
        write_new(ledger.directory / 'completion.json', {'batch_closed': True, 'ledger_broken': ledger.broken,
                  'output_codings': len(outputs), 'pair_codings': len(pairs),
                  'new_HTTP_slots_reserved': len(ledger.used), 'reserve_USD': str(ledger.reserved),
                  'new_tested_outputs': 0, 'extra_retries': 0,
                  'canonical_output_states': {s: sum(e['canonical']['status'] == s for e in outputs.values())
                                              for s in {e['canonical']['status'] for e in outputs.values()}},
                  'canonical_pair_states': {s: sum(e['canonical']['status'] == s for e in pairs.values())
                                            for s in {e['canonical']['status'] for e in pairs.values()}}}, ledger.directory)
    finally:
        transport.close()
        if ledger is not None:
            ledger.close()


if __name__ == '__main__':
    require(sys.argv[1:] in [[], ['--preflight']], 'No resume, retry, output substitution or extra batch arguments allowed')
    if sys.argv[1:] == ['--preflight']:
        preflight()
        print('PASS offline exact-release preflight; no credential access, consumption or network')
    else:
        run()
