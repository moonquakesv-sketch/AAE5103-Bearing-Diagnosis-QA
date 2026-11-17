"""Resolve supplied span IDs to exact source quotes; never infer scores or edit text."""
from pathlib import Path
import hashlib
import re
import jsonschema
from common import canonical, digest, require, snapshot, read_json
from baseline import validate_rating as validate_canonical_rating

ROOT = Path(__file__).absolute().parents[1]
WIRE_VERSION = 'ref.rating.span_ids.v1'


def source_hash(text):
    require(type(text) is str, 'observed text must be a string')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_spans(text):
    """Partition visible source text at newlines and punctuation followed by whitespace."""
    sha = source_hash(text)
    spans, cursor = [], 0

    def append(start, end):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append({'span_id': 'S' + sha[:12] + '-%04d' % len(spans),
                          'start': start, 'end': end, 'text': text[start:end]})

    for boundary in re.finditer(r'(?<=[.!?])\s+|[\r\n]+', text):
        append(cursor, boundary.start())
        cursor = boundary.end()
    append(cursor, len(text))
    require(bool(spans), 'observed text has no visible source spans')
    return spans


def prepare_packet(packet):
    require(type(packet) is dict and type(packet.get('unit_id')) is str and
            packet['unit_id'].startswith('N-'), 'anonymous unit binding required')
    require('quote_spans' not in packet and 'quote_source_sha256' not in packet, 'packet already anchored')
    text = packet['observed_text']
    result = snapshot(packet)
    result['quote_source_sha256'] = source_hash(text)
    result['quote_spans'] = [{'span_id': s['span_id'], 'text': s['text']} for s in build_spans(text)]
    binding = {'unit_id': result['unit_id'], 'observed_text_sha256': source_hash(text),
               'packet_sha256': digest(result)}
    return result, binding


def resolve_quote(quote, text, spans):
    require(type(quote) is dict and set(quote) == {'span_ids'}, 'wire quote requires span_ids only')
    ids = quote['span_ids']
    require(type(ids) is list and bool(ids) and all(type(x) is str for x in ids), 'nonempty span ID list required')
    require(len(ids) == len(set(ids)), 'duplicate span ID')
    positions = {s['span_id']: i for i, s in enumerate(spans)}
    require(all(x in positions for x in ids), 'unknown or wrong-source span ID')
    indices = [positions[x] for x in ids]
    require(indices == list(range(indices[0], indices[0] + len(indices))), 'quote range must be ordered and contiguous')
    start, end = spans[indices[0]]['start'], spans[indices[-1]]['end']
    value = text[start:end]
    require(start < end <= len(text) and bool(value.strip()), 'resolved quote is empty or out of bounds')
    return {'start': start, 'end': end, 'text': value}


def validate_rating(raw_rating, packet, expected_binding):
    """Keep raw untouched; derive canonical positions, then run unchanged semantic guards."""
    require(type(expected_binding) is dict, 'trusted frozen binding required')
    require(expected_binding.get('unit_id') == packet.get('unit_id'), 'anonymous unit binding mismatch')
    text = packet['observed_text']
    sha = source_hash(text)
    require(sha == expected_binding.get('observed_text_sha256') == packet.get('quote_source_sha256'),
            'observed source hash mismatch')
    require(digest(packet) == expected_binding.get('packet_sha256'), 'complete evaluation packet changed')
    spans = build_spans(text)
    require(packet.get('quote_spans') == [{'span_id': s['span_id'], 'text': s['text']} for s in spans],
            'supplied span registry differs from exact source')
    schema = read_json(ROOT / 'schemas/reference_rating_wire.json')
    jsonschema.Draft202012Validator(schema).validate(raw_rating)
    value = snapshot(raw_rating)
    value.pop('schema_version')
    count = 0
    for point in value['points']:
        point['quotes'] = [resolve_quote(q, text, spans) for q in point['quotes']]
        count += len(point['quotes'])
    for category in ['claims', 'voluntary_citations']:
        for row in value[category]:
            row['quote'] = resolve_quote(row['quote'], text, spans)
            count += 1
    validate_canonical_rating(value, packet)
    return {'rating': value, 'raw_rating_sha256': digest(raw_rating),
            'canonical_rating_sha256': digest(value), 'observed_text_sha256': sha,
            'packet_sha256': digest(packet), 'span_registry_sha256': digest(spans),
            'resolved_quote_count': count, 'schema_version': WIRE_VERSION,
            'semantic_judgment_recomputed': False}


def messages(action, packet):
    require(action in ['judge_output', 'adjudicate_output'], 'ratings-only preparation; generation prohibited')
    name = 'judge.txt' if action == 'judge_output' else 'adjudicator.txt'
    schema = read_json(ROOT / 'schemas/reference_rating_wire.json')
    system = (ROOT / 'prompts' / name).read_text() + '\nReturn strict JSON following this complete schema:\n' + canonical(schema)
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': canonical(packet)}]


def third_packet(packet, primary_a, primary_b):
    """Future third receives full new A/B outcomes; it never consumes historical ratings."""
    binding = {'unit_id': packet['unit_id'], 'observed_text_sha256': source_hash(packet['observed_text']),
               'packet_sha256': digest(packet)}
    for primary in [primary_a, primary_b]:
        require(type(primary) is dict and set(primary) == {'status', 'rating', 'source_binding'}, 'complete validated A/B outcome required')
        require(primary['status'] in ['valid', 'evaluator_failure', 'not_attempted_batch_stopped'], 'invalid A/B status')
        require((primary['rating'] is not None) == (primary['status'] == 'valid'), 'A/B null/status mismatch')
        require(primary['source_binding'] == binding, 'A/B belongs to a different frozen source')
        if primary['status'] == 'valid':
            validate_canonical_rating(primary['rating'], packet)
    return {'packet': snapshot(packet),
            'A': {k: snapshot(primary_a[k]) for k in ['status', 'rating']},
            'B': {k: snapshot(primary_b[k]) for k in ['status', 'rating']}}
