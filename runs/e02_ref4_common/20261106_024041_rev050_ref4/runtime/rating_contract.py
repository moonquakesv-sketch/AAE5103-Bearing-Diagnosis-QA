"""Exact frozen-packet and current A/B bindings for ratings-only sends."""
from pathlib import Path
from common import require, read_json, digest, strict_json
import quote_anchor

ROOT = Path(__file__).absolute().parents[1]


def packet_for(config, binding):
    if config['scope']['stage'] == 'SYNTHETIC':
        packet = binding['synthetic_packet']
        require(packet['business_request']['question'].startswith('SYNTHETIC:'), 'synthetic packet required')
    else:
        require(config['scope']['stage'] == 'E02', 'unknown execution stage')
        require('synthetic_packet' not in binding, 'fixture binding forbidden in E02')
        unit = binding['unit_id']
        require(type(unit) is str and unit.startswith('N-LREF-') and '/' not in unit and '..' not in unit, 'frozen anonymous unit only')
        packet = read_json(ROOT / 'packets' / (unit + '.packet.json'))
    require(digest(packet) == binding['packet_sha256'], 'frozen packet hash mismatch')
    require(packet['unit_id'] == binding['unit_id'] and
            quote_anchor.source_hash(packet['observed_text']) == binding['observed_text_sha256'], 'frozen answer binding mismatch')
    return packet


def source_binding(binding):
    return {k: binding[k] for k in ['unit_id', 'observed_text_sha256', 'packet_sha256']}


def validate_messages(config, action, msgs, binding, primary_outcomes):
    require(action in ['judge_output', 'adjudicate_output'] and binding['action'] == action,
            'ratings-only registered role required')
    require(type(msgs) is list and len(msgs) == 2 and [m.get('role') for m in msgs] == ['system', 'user'], 'fresh two-message context required')
    payload = strict_json(msgs[1]['content'])
    packet = packet_for(config, binding)
    if action == 'judge_output':
        require(binding['label'] in ['A', 'B'] and payload == packet, 'initial frozen packet mismatch')
    else:
        require(binding['label'] == 'C' and set(payload) == {'packet', 'A', 'B'} and payload['packet'] == packet,
                'third frozen packet mismatch')
        unit = binding['unit_id']
        require(unit in primary_outcomes and set(primary_outcomes[unit]) == {'A', 'B'}, 'both current A/B outcomes required before third')
        require(payload['A'] == primary_outcomes[unit]['A'] and payload['B'] == primary_outcomes[unit]['B'],
                'third payload differs from recorded current A/B')
    require(msgs == quote_anchor.messages(action, payload), 'complete prompt/schema/message identity mismatch')
    return True
