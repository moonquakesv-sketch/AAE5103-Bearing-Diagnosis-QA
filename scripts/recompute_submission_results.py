"""Recompute saved classification metrics and five-group REF arithmetic, without inference or APIs."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import csv
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'runs/e02_ref4_acceptance/20261106_024041_rev050_ref4'


def read(path):
    return json.loads(path.read_bytes())


def main():
    spec = importlib.util.spec_from_file_location('saved_ref_arithmetic', REF / 'analyze_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    units = read(REF / 'unit_statuses.json')
    if Counter(u['group'] for u in units) != {g: 13 for g in module.GROUPS}:
        raise ValueError('Expected all 65 planned REF units')
    checks = {}
    for sensitive, stats_name, contrasts_name in [
            (False, 'reference_statistics.json', 'contrasts.json'),
            (True, 'sensitivity_statistics.json', 'sensitivity_contrasts.json')]:
        computed = module.fixed(units, mask=sensitive)
        checks[stats_name] = computed == read(REF / stats_name)
        checks[contrasts_name] = module.contrasts(computed) == read(REF / contrasts_name)
    d04 = ROOT / 'runs/d04/20261023_rev008'
    with (d04 / 'test_predictions.csv').open(newline='') as stream:
        predictions = list(csv.DictReader(stream))
    labels = ['normal', 'inner_race', 'outer_race', 'rolling_element']
    counts = Counter((r['true_label'], r['predicted_class']) for r in predictions)
    recall, f1 = [], []
    for label in labels:
        tp = counts[label, label]
        actual = sum(v for (truth, _), v in counts.items() if truth == label)
        predicted = sum(v for (_, guess), v in counts.items() if guess == label)
        recall.append(Fraction(tp, actual))
        f1.append(Fraction(2 * tp, actual + predicted))
    computed = {'accuracy': float(Fraction(sum(counts[c, c] for c in labels), len(predictions))),
                'macro_f1': float(sum(f1) / len(labels)),
                'balanced_accuracy': float(sum(recall) / len(labels))}
    stored = read(d04 / 'metrics.json')
    checks['D04_236_saved_predictions'] = len(predictions) == 236
    for key, value in computed.items():
        checks['D04_' + key] = value == stored[key]
    if not all(checks.values()):
        raise ValueError('Saved result recomputation differs: ' + repr(checks))
    unknown = [{'group': u['group'], 'question_id': u['question_id']} for u in units if not u['final_valid']]
    print(json.dumps({'status': 'PASS', 'checks': checks, 'REF_planned_units': len(units),
                      'REF_valid_final_units': sum(u['final_valid'] for u in units), 'REF_unknown_units': unknown,
                      'D04_saved_metrics': computed, 'new_inference': False,
                      'model_api_calls': 0, 'source_files_modified': False}, indent=2))


if __name__ == '__main__':
    main()
