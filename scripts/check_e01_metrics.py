"""Recalculate explicit synthetic E01 counting examples, never real answers.

This module does not import a retriever, a classifier, a generator, or an API.
It checks arithmetic on synthetic labels; humans must supply semantic labels
for any future real evaluation under the separate scoring protocol.
"""
import argparse
import json
import math
from pathlib import Path
import sys


VERSION = 'e01.synthetic_metrics.v1_rev014'


class MetricsError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise MetricsError(message)


def number(value, where):
    require(type(value) in (int, float), where + ' must be numeric, not bool.')
    require(math.isfinite(value), where + ' must be finite.')
    require(value >= 0, where + ' must be non-negative.')
    return value


def count(value, where):
    require(type(value) is int and value >= 0,
            where + ' must be a non-negative integer, not bool.')
    return value


def ratio(numerator, denominator, bounded=True):
    """Keep the counts and distinguish no applicable units from zero credit."""
    number(numerator, 'numerator')
    number(denominator, 'denominator')
    require(not bounded or numerator <= denominator,
            'A bounded ratio numerator cannot exceed its denominator.')
    if denominator == 0:
        require(numerator == 0, 'A zero denominator requires zero numerator.')
        return {'numerator': numerator, 'denominator': denominator,
                'value': None, 'status': 'not_applicable'}
    return {'numerator': numerator, 'denominator': denominator,
            'value': numerator / denominator, 'status': 'measured'}


def count_metric(value):
    count(value, 'count')
    return ratio(value, 1, bounded=False)


def list_value(value, where):
    require(type(value) is list, where + ' must be a list.')
    return value


def ids(value, where):
    """IDs represent unique cards/errors: repeated IDs cannot inflate counts."""
    list_value(value, where)
    require(all(type(item) is str and bool(item.strip()) for item in value),
            where + ' must contain non-empty string IDs.')
    return set(value)


def object_fields(value, fields, where):
    require(type(value) is dict, where + ' must be an object.')
    require(set(value) == set(fields), where + ' has unexpected/missing fields.')


def score_ratio(scores):
    list_value(scores, 'scores')
    for score in scores:
        number(score, 'score')
        require(score in (0, 0.5, 1), 'Scores must be 0, 0.5, or 1.')
    return ratio(sum(scores), len(scores))


def groups(value, where):
    list_value(value, where)
    result = [ids(group, where + ' combination') for group in value]
    require(all(result), where + ' combinations cannot be empty.')
    return result


def support_score(atom, retrieved):
    object_fields(atom, ('relevant', 'support_any_of', 'partial_any_of'), 'atom')
    relevant = ids(atom['relevant'], 'atom.relevant')
    complete = groups(atom['support_any_of'], 'atom.support_any_of')
    partial = groups(atom['partial_any_of'], 'atom.partial_any_of')
    require(all(group <= relevant for group in complete + partial),
            'All supporting IDs must also be registered as relevant.')
    returned = ids(list(retrieved), 'retrieved')
    if any(group <= returned for group in complete):
        return 1.0
    if any(group <= returned for group in partial):
        return 0.5
    return 0.0


def retrieval_metrics(atoms, retrieved):
    list_value(atoms, 'atoms')
    returned = ids(retrieved, 'retrieved')
    support = [support_score(atom, returned) for atom in atoms]
    related = [bool(ids(atom['relevant'], 'relevant') & returned)
               for atom in atoms]
    union = set().union(*(ids(atom['relevant'], 'relevant') for atom in atoms))
    return {
        'related_coverage': ratio(sum(related), len(atoms)),
        'support_coverage': score_ratio(support),
        'relevant_precision': ratio(len(returned & union), len(returned)),
    }


def unsupported_claim_rate(labels):
    list_value(labels, 'claim_labels')
    require(all(type(label) is str and label in ('full', 'partial', 'unsupported')
                for label in labels), 'Unknown synthetic claim label.')
    return ratio(labels.count('unsupported') + 0.5 * labels.count('partial'),
                 len(labels))


def citation_id_error_rate(validity):
    list_value(validity, 'citation_id_valid')
    require(all(type(item) is bool for item in validity),
            'Citation validity must contain booleans.')
    return ratio(sum(not item for item in validity), len(validity))


def error_change_metrics(draft, final, reported):
    before = ids(draft, 'draft_errors')
    after = ids(final, 'final_errors')
    detected = ids(reported, 'verifier_issues')
    new_count = len(after - before)
    return {
        'correction_rate': ratio(len(before - after), len(before)),
        'unresolved_rate': ratio(len(before & after), len(before)),
        'detection_recall': ratio(len(before & detected), len(before)),
        'false_alarm_rate': ratio(len(detected - before), len(detected)),
        'new_error_count': count_metric(new_count),
        'new_error_run_rate': ratio(int(new_count > 0), 1),
        'new_errors_per_alignable_run': ratio(new_count, 1, bounded=False),
    }


EXAMPLE_FIELDS = {
    'S1': ('retrieved', 'atoms'),
    'S2': ('point_scores', 'claim_labels', 'citation_id_valid',
           'citation_support_scores', 'claim_citation_scores', 'condition_scores'),
    'S3': ('content_point_scores', 'claim_labels', 'citation_id_valid',
           'insufficiency_scores'),
    'S3_failure': ('synthetic_service_failure', 'required_insufficiency_items'),
    'S4': ('draft_errors', 'final_errors', 'verifier_issues',
           'planned_verification_runs', 'alignable_runs'),
    'S5': ('planned_repeats', 'positive_coverage_scores_with_failure_zero',
           'completed_repeats', 'planned_timing_runs', 'observed_latencies_seconds'),
}


def compute_example(example):
    require(type(example) is dict, 'Example must be an object.')
    identity = example.get('id')
    require(type(identity) is str and identity in EXAMPLE_FIELDS,
            'Only the declared S1-S5 synthetic examples are supported.')
    object_fields(example, ('id', 'expected') + EXAMPLE_FIELDS[identity], identity)
    if identity == 'S1':
        return retrieval_metrics(example['atoms'], example['retrieved'])
    if identity == 'S2':
        require(len(list_value(example['citation_id_valid'], 'citation_id_valid')) ==
                len(list_value(example['citation_support_scores'],
                               'citation_support_scores')),
                'Citation validity/support counts must match.')
        for valid, support in zip(example['citation_id_valid'],
                                  example['citation_support_scores']):
            require(valid is not False or support == 0,
                    'An invalid citation cannot receive semantic support credit.')
        return {
            'point_coverage': score_ratio(example['point_scores']),
            'unsupported_claim_rate': unsupported_claim_rate(example['claim_labels']),
            'citation_id_error_rate': citation_id_error_rate(example['citation_id_valid']),
            'citation_support': score_ratio(example['citation_support_scores']),
            'claim_citation_coverage': score_ratio(example['claim_citation_scores']),
            'condition_retention': score_ratio(example['condition_scores']),
        }
    if identity == 'S3':
        return {
            'content_coverage': score_ratio(example['content_point_scores']),
            'unsupported_claim_rate': unsupported_claim_rate(example['claim_labels']),
            'citation_id_error_rate': citation_id_error_rate(example['citation_id_valid']),
            'insufficiency_handling': score_ratio(example['insufficiency_scores']),
        }
    if identity == 'S3_failure':
        require(example['synthetic_service_failure'] is True,
                'The synthetic failure flag must be true.')
        required = count(example['required_insufficiency_items'],
                         'required_insufficiency_items')
        return {
            'insufficiency_handling': ratio(0, required),
            'failure_count': count_metric(1),
            'unsupported_claim_rate': {
                'numerator': None, 'denominator': None,
                'value': None, 'status': 'missing_due_to_failure'},
        }
    if identity == 'S4':
        require(count(example['planned_verification_runs'],
                      'planned_verification_runs') == 1 and
                count(example['alignable_runs'], 'alignable_runs') == 1,
                'S4 describes exactly one planned, alignable synthetic pair.')
        return error_change_metrics(example['draft_errors'], example['final_errors'],
                                    example['verifier_issues'])
    planned = count(example['planned_repeats'], 'planned_repeats')
    completed = count(example['completed_repeats'], 'completed_repeats')
    scores = list_value(example['positive_coverage_scores_with_failure_zero'],
                        'positive_coverage_scores_with_failure_zero')
    require(len(scores) == planned and completed <= planned,
            'Repeat counts must match the planned synthetic runs.')
    times = list_value(example['observed_latencies_seconds'], 'latencies')
    for elapsed in times:
        number(elapsed, 'latency')
    timing_planned = count(example['planned_timing_runs'], 'planned_timing_runs')
    require(len(times) <= timing_planned, 'Too many observed timing runs.')
    return {
        'question_mean_coverage': score_ratio(scores),
        'completion_rate': ratio(completed, planned),
        'latency_mean_seconds': ratio(sum(times), len(times), bounded=False),
        'timing_coverage': ratio(len(times), timing_planned),
        'independent_question_count': count_metric(1),
    }


def expected_matches(actual, expected):
    require(type(expected) is dict and set(expected) == set(actual),
            'Expected metrics must exactly match calculated metric names.')
    matches = {}
    for name, metric in actual.items():
        target = expected[name]
        if target is None:
            matches[name] = metric['value'] is None
        else:
            number(target, 'expected ' + name)
            matches[name] = (metric['value'] is not None and
                             abs(metric['value'] - target) <= 1e-12)
    return matches


def load_strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result

    def constant(value):
        raise MetricsError('Non-finite JSON constant: ' + value)

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except json.JSONDecodeError as error:
        raise MetricsError('Invalid synthetic JSON: ' + str(error)) from error


def check_bundle(bundle):
    object_fields(bundle, ('synthetic', 'examples'), 'bundle')
    require(bundle['synthetic'] is True, 'Only explicitly synthetic inputs accepted.')
    examples = list_value(bundle['examples'], 'examples')
    require(bool(examples), 'Examples cannot be empty.')
    seen = set()
    records = []
    for example in examples:
        actual = compute_example(example)
        require(example['id'] not in seen, 'Duplicate synthetic example ID.')
        seen.add(example['id'])
        matches = expected_matches(actual, example['expected'])
        records.append({'id': example['id'], 'actual': actual,
                        'expected': example['expected'], 'matches': matches,
                        'passed': all(matches.values())})
    return {'version': VERSION, 'synthetic': True,
            'scope': 'arithmetic_only_no_system_scoring',
            'examples': records, 'all_passed': all(item['passed'] for item in records)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        bundle = load_strict_json(args.examples.read_text(encoding='utf-8'))
        result = check_bundle(bundle)
        encoded = (json.dumps(result, ensure_ascii=True, allow_nan=False,
                              indent=2) + '\n').encode('utf-8')
        with args.output.open('xb') as stream:
            stream.write(encoded)
    except (MetricsError, OSError, UnicodeError) as error:
        print(json.dumps({'status': 'rejected', 'error': str(error)},
                         ensure_ascii=True), file=sys.stderr)
        return 2
    print(json.dumps({'synthetic': True, 'all_passed': result['all_passed'],
                      'example_count': len(result['examples'])}))
    return 0 if result['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
