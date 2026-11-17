"""One approved old52 scoring launch using the byte-identical accepted runtime core."""
from pathlib import Path
import sys


def main():
    if not sys.dont_write_bytecode or sys.flags.optimize != 0:
        raise SystemExit('Use existing Python -B without -O')
    if sys.argv[1:] != ['--execute-approved-REF4']:
        raise SystemExit('No execution; exactly --execute-approved-REF4 required for the single authorized batch.')
    root = Path(__file__).absolute().parent
    sys.path.insert(0, str(root))
    import verify_ready as admission
    admission.verify()
    admission.modules()
    from common import read_json, safe_path, digest, require, write_new
    from provider import LiveGate, Counter, Ledger, LiveHTTP, Provider
    import workflow
    config = read_json(root / 'config/execution.json')
    auth = read_json(root / 'authorization/authorization.json')
    require(config['scope']['groups'] == ['T', 'TV', 'G', 'GV'] and config['scope']['saved_answer_rescoring'] == 52,
            'only approved old52 scoring')
    sources = read_json(root / 'inputs/saved_answer_bindings.json')['sources']
    original_rubrics = {q['question_id']: q for q in read_json(root / 'inputs/rubrics.json')['questions']}
    rubrics = {s['question_id']: original_rubrics[s['original_question_id']] for s in sources}
    gate = LiveGate(root, config, auth)
    counter = Counter(root.parents[2], read_json(root / 'config/counter_assets.json'))
    directory = safe_path(config['batch_output'], root.parents[2])
    directory.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(directory, config, read_json(root / 'config/slots.json'), directory.parent, gate=gate)
    write_new(directory / 'authorization.consumed.json', {'authorization_sha256': digest(auth), 'no_reuse': True}, directory)
    transport = None
    try:
        transport = LiveHTTP(gate, config)
        provider = Provider(config, counter, ledger, transport, directory / 'raw')
        report = workflow.run_planned(provider, sources, lambda s: read_json(root / s['packet']), rubrics, directory)
        print(__import__('json').dumps(report, ensure_ascii=False, indent=2))
    finally:
        ledger.close()
        if transport:
            transport.close()


if __name__ == '__main__':
    main()
