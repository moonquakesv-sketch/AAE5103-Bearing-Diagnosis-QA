"""Read-only independent D04 recount. No model loading, prediction or image reads.

Intentionally does not call D03/D04 metrics, per_record, cache readers or scorers.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["normal", "inner_race", "outer_race", "rolling_element"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def table(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def recount(rows):
    matrix = [[0] * 4 for _ in range(4)]
    for r in rows:
        matrix[CLASSES.index(r["true_label"])][CLASSES.index(r["predicted_class"])] += 1
    support = [sum(row) for row in matrix]
    predicted = [sum(row[j] for row in matrix) for j in range(4)]
    recall = [matrix[i][i] / support[i] if support[i] else 0. for i in range(4)]
    f1 = [2 * matrix[i][i] / (support[i] + predicted[i]) if support[i] + predicted[i] else 0. for i in range(4)]
    return {"n_windows": len(rows), "macro_f1": sum(f1) / 4,
            "balanced_accuracy": sum(recall[i] for i in range(4) if support[i]) / sum(n > 0 for n in support),
            "accuracy": sum(matrix[i][i] for i in range(4)) / len(rows),
            "per_class_recall": dict(zip(CLASSES, recall)),
            "confusion_matrix_rows_true_columns_predicted": matrix}


def check_bindings(run, declared, compatibility=None):
    approved = {}
    if compatibility is not None:
        revision = read_json(ROOT / compatibility)
        require(revision["purpose"] == "D04_guard_only_readonly_revision", "Wrong compatibility purpose")
        require(revision["run"] == str(run.relative_to(ROOT))
                and revision["declaration_sha256"] == sha(run / "declaration.json")
                and revision["completion_sha256"] == sha(run / "completion.json"), "Compatibility targets another evaluation")
        allowed = {"scripts/d04_pipeline.py", "scripts/verify_d04.py", "docs/D04_运行协议.md"}
        require(bool(revision["changes"]) and set(revision["changes"]) <= allowed, "Compatibility may only change D04 guards/protocol")
        for path, change in revision["changes"].items():
            backup = Path(change["before_copy"])
            require(not backup.is_absolute() and ".." not in backup.parts, "Invalid compatibility backup path")
            require(change["before_sha256"] == declared["bound_hashes"][path]
                    == sha(ROOT / backup) == sha(run / "source" / path), "Compatibility original source differs")
            require(change["after_sha256"] == sha(ROOT / path), "Unapproved D04 revision")
            approved[path] = change["after_sha256"]
    for path, expected in declared["bound_hashes"].items():
        require(sha(ROOT / path) == approved.get(path, expected), "Bound input/code changed: " + path)


def check_checkpoint_commits(run, events):
    # Independent ledger validation: no import of the production recovery helper.
    original = {}
    for e in events:
        if e["event"] not in ("features_saved", "predictions_saved", "feature_checkpoint_reused", "prediction_checkpoint_reused"):
            continue
        require(type(e["offset"]) is int and e["offset"] >= 0, "Invalid checkpoint offset")
        kind = "features" if e["event"].startswith("feature") else "predictions"
        name = f"{e['offset']:03d}_{kind}.npz"
        if e["event"].endswith("_saved"):
            require(name not in original, "Duplicate checkpoint commit")
            require(kind == "features" or f"{e['offset']:03d}_features.npz" in original,
                    "Prediction commit without feature commit")
            original[name] = e["sha256"]
        else:
            require(name in original, "Checkpoint reuse without original commit")
            require(e["sha256"] == original[name], "Checkpoint reuse digest differs from original commit")
    actual = {str(p.relative_to(run / "checkpoints")) for p in (run / "checkpoints").rglob("*") if p.is_file()}
    require(actual == set(original), "Checkpoint files and original commits differ")
    for name, expected in original.items():
        require(sha(run / "checkpoints" / name) == expected, "Checkpoint changed since commit")
    return set(original)


def verify(run, compatibility=None):
    declared, complete = read_json(run / "declaration.json"), read_json(run / "completion.json")
    require(complete["declaration_sha256"] == sha(run / "declaration.json"), "Declaration changed")
    for path, expected in complete["output_hashes"].items():
        require(sha(run / path) == expected, "Completed output changed: " + path)
    check_bindings(run, declared, compatibility)
    samples = table(ROOT / "data/samples.csv")
    expected = [(i, r) for i, r in enumerate(samples) if r["split"] == "test"]
    require(len(expected) == 236 and len({r["sample_id"] for _, r in expected}) == 236, "Test population mismatch")
    records = {r["record_id"]: r for r in table(ROOT / "data/records.csv")}
    require(Counter(r["true_label"] for _, r in expected) == Counter({k: 59 for k in CLASSES}), "Class counts mismatch")
    require(len({r["record_id"] for _, r in expected}) == 4, "Record count mismatch")
    result, mapping = table(run / "test_predictions.csv"), table(run / "features/test_mapping.csv")
    ids = [r["sample_id"] for _, r in expected]
    require([r["sample_id"] for r in result] == [r["sample_id"] for r in mapping] == ids
            == declared["ordered_sample_ids"], "Prediction/mapping/sample order mismatch")
    meta = read_json(run / "features/test.json")
    with np.load(run / "features/test.npz", allow_pickle=False) as z:
        features, cache_ids, context = z["features"].copy(), z["sample_id"].tolist(), z["context"].item()
    require(cache_ids == ids and context == meta["context"] == sha(run / "declaration.json"), "Cache identity mismatch")
    require(features.shape == (236, 512) and features.dtype == np.float32 and np.isfinite(features).all(), "Invalid feature array")
    require(array_hash(features) == meta["array_sha256"] and meta["npz_sha256"] == sha(run / "features/test.npz")
            and meta["mapping_sha256"] == sha(run / "features/test_mapping.csv"), "Feature hashes mismatch")
    scores = []
    for i, ((manifest_index, source), output, mapped) in enumerate(zip(expected, result, mapping)):
        record = records[source["record_id"]]
        require(source["load_hp"] == record["load_hp"] == "3" and record["split"] == "test", "Wrong load/split")
        require(all(source[k] == record[k] for k in ("true_label", "group_id", "source_family_id", "channel")), "Record mapping mismatch")
        require(all(mapped[k] == v for k, v in source.items()), "Row provenance mismatch")
        require(mapped["manifest_index"] == str(manifest_index) and mapped["feature_row"] == str(i), "Row index mismatch")
        require(mapped["feature_sha256"] == array_hash(features[i]), "Feature row digest mismatch")
        require(sha(ROOT / source["model_image_path"]) == mapped["model_image_sha256"], "Input image bytes changed")
        require(all(output[k] == source[k] for k in ("sample_id", "record_id", "group_id", "source_family_id", "split", "true_label")), "Output label/source mismatch")
        require(output["model_version"] == declared["config"]["model_version"], "Wrong model version")
        score = json.loads(output["decision_score"])
        require(len(score) == 4 and np.isfinite(score).all(), "Invalid scores")
        scores.append(score)
    # Crosscheck immutable inference checkpoints, never rerun inference.
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    committed = check_checkpoint_commits(run, events)
    batch_size = declared["feature_config"]["batch_size"]
    require(committed == {f"{offset:03d}_{kind}.npz" for offset in range(0, 236, batch_size)
                          for kind in ("features", "predictions")}, "Unexpected checkpoint population")
    for offset in range(0, 236, batch_size):
        end = min(offset + batch_size, 236)
        with np.load(run / f"checkpoints/{offset:03d}_features.npz", allow_pickle=False) as z:
            require(z["sample_id"].tolist() == ids[offset:end] and np.array_equal(z["features"], features[offset:end]), "Feature checkpoint mapping mismatch")
        with np.load(run / f"checkpoints/{offset:03d}_predictions.npz", allow_pickle=False) as z:
            require(z["sample_id"].tolist() == ids[offset:end], "Prediction checkpoint IDs mismatch")
            require(z["predicted"].tolist() == [CLASSES.index(r["predicted_class"]) for r in result[offset:end]], "Predicted count/mapping mismatch")
            require(np.array_equal(z["scores"], np.array(scores[offset:end])), "CSV score serialization mismatch")
    actual = recount(result)
    saved = read_json(run / "metrics.json")
    require(saved["class_order"] == CLASSES, "Metric class order differs")
    for key, value in actual.items():
        if isinstance(value, float):
            require(abs(saved[key] - value) <= 1e-14, "Independent metric mismatch: " + key)
        else:
            require(saved[key] == value, "Independent metric mismatch: " + key)
    grouped = defaultdict(list)
    for r in result:
        grouped[r["record_id"]].append(r)
    per_record = read_json(run / "test_per_record.json")
    require(per_record["class_order"] == CLASSES and len(per_record["records"]) == 4, "Per-record population mismatch")
    require({r["record_id"] for r in per_record["records"]} == set(grouped), "Per-record IDs mismatch")
    record_checks = []
    for saved_record in per_record["records"]:
        rid = saved_record["record_id"]
        subset = grouped[rid]
        stats = recount(subset)
        correct = sum(r["true_label"] == r["predicted_class"] for r in subset)
        counts = {k: sum(r["predicted_class"] == k for r in subset) for k in CLASSES}
        require(saved_record["n_windows"] == len(subset) == 59 and saved_record["correct"] == correct, "Per-record counts differ")
        require(saved_record["accuracy_and_true_class_recall"] == stats["accuracy"]
                and saved_record["predicted_class_counts"] == counts
                and saved_record["confusion_matrix_rows_true_columns_predicted"] == stats["confusion_matrix_rows_true_columns_predicted"], "Per-record metrics differ")
        record_checks.append({"record_id": rid, "n_windows": len(subset), "correct": correct, "predicted_counts": counts})
    errors = [r for r in result if r["true_label"] != r["predicted_class"]]
    require(table(run / "misclassifications.csv") == errors, "Incomplete error list")
    error_report = read_json(run / "errors.json")
    require(error_report["misclassifications"] == errors and error_report["n_misclassified"] == len(errors)
            and error_report["failed_samples"] == [] and error_report["n_failed_samples"] == 0, "Error report mismatch")
    invariance = read_json(run / "state_invariance.json")
    require(invariance["status"] == "PASS" and invariance["model_before"] == invariance["model_after"]
            and invariance["network_before"] == invariance["network_after"] and invariance["fitting_methods_blocked"], "State invariance failed")
    model = ROOT / declared["config"]["model_run"]
    for name, before in declared["frozen_identity"].items():
        require(sha(model / name) == before == invariance["frozen_files_before"][name] == invariance["frozen_files_after"][name], "Frozen component changed")
    require(sum(e["event"] == "formal_test_started" for e in events) == 1
            and sum(e["event"] == "formal_test_completed" for e in events) == 1, "Incorrect evaluation count")
    return {"status": "PASS", "method": "independent csv/NPZ mapping checks and explicit integer confusion matrix / TP/FP/FN formulas; no production metric calls",
            "test_set_used": True, "model_version": declared["config"]["model_version"], "recomputed_metrics": actual,
            "record_checks": record_checks, "n_scores": len(scores) * 4, "n_misclassified": len(errors),
            "n_failed_samples": 0, "n_interruptions": sum(e["event"] == "execution_interrupted" for e in events),
            "n_resumptions": sum(e["event"] == "resume_requested" for e in events),
            "test_images_decoded_for_review": 0, "predictions_reexecuted_for_review": 0,
            "completion_sha256": sha(run / "completion.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compatibility", help="Explicit guard revision for read-only verification; never authorizes inference")
    args = parser.parse_args()
    report = verify(ROOT / args.run, args.compatibility)
    with (ROOT / args.output).open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": report["status"], "metrics": report["recomputed_metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
