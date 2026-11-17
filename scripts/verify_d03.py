"""Independent D03 orchestration: manual metrics, fresh official net, independent fit.

Uses the same mathematical libraries, not a second implementation of ResNet/SVM.
Does not use production input transforms, extraction, metrics, selection, or fitting.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
from PIL import Image
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision.models import resnet18

import d03_pipeline as p
from data_contract import require


def manual_metrics(y, predicted, classes):
    matrix = np.zeros((4, 4), dtype=np.int64)
    for actual, pred in zip(y, predicted):
        matrix[int(actual), int(pred)] += 1
    tp = np.diag(matrix).astype(float)
    support, predicted_support = matrix.sum(axis=1), matrix.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros(4), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted_support, out=np.zeros(4), where=(support + predicted_support) > 0)
    return {"macro_f1": float(f1.mean()), "balanced_accuracy": float(recall[support > 0].mean()),
            "accuracy": float(tp.sum() / matrix.sum()), "per_class_recall": dict(zip(classes, recall.tolist())),
            "confusion_matrix_rows_true_columns_predicted": matrix.tolist()}


def assert_metrics(actual, expected):
    for key in ("macro_f1", "balanced_accuracy", "accuracy"):
        require(abs(actual[key] - expected[key]) <= 1e-14, f"Independent {key} differs")
    require(actual["per_class_recall"] == expected["per_class_recall"], "Independent recall mapping differs")
    require(actual["confusion_matrix_rows_true_columns_predicted"] == expected["confusion_matrix_rows_true_columns_predicted"], "Independent confusion matrix differs")


def independent_features(model, rows, root, batch_size, device):
    captured = []
    hook = model.avgpool.register_forward_hook(lambda module, args, result: captured.append(result.flatten(1).cpu().numpy().copy()))
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), batch_size):
                tensors = []
                for row in rows[offset:offset + batch_size]:
                    require(row["split"] in ("train", "validation"), "Independent check refuses test decoding")
                    with Image.open(root / row["model_image_path"]) as image:
                        require(image.mode == "RGB" and image.size == (224, 224), "Independent input shape differs")
                        pixels = np.asarray(image).astype(np.float32)
                    normalized = (pixels / np.float32(255) - np.array([.485, .456, .406], dtype=np.float32)) \
                                 / np.array([.229, .224, .225], dtype=np.float32)
                    tensors.append(torch.from_numpy(normalized.transpose(2, 0, 1).copy()))
                model(torch.stack(tensors).to(device))  # original 1000-way fc retained; use avgpool hook only
    finally:
        hook.remove()
    return np.concatenate(captured)


def read_prediction(path, rows, classes):
    table = p.read_csv(path)
    require([r["sample_id"] for r in table] == [r["sample_id"] for r in rows], "Prediction row mapping differs")
    for entry, row in zip(table, rows):
        require(all(entry[k] == row[k] for k in ("record_id", "group_id", "source_family_id", "split", "true_label")), "Prediction metadata differs")
    predicted = np.array([classes.index(r["predicted_class"]) for r in table])
    scores = np.array([json.loads(r["decision_score"]) for r in table])
    require(scores.shape == (len(rows), 4) and np.isfinite(scores).all(), "Invalid score mapping")
    return predicted, scores


def verify(run):
    p.check_declaration(run)
    c, device = p.config(), p.setup(p.config())
    require(str(device) == p.read_json(run / "extraction.json")["device"], "Verification device changed")
    rows = p.audit_population()
    require(rows == p.read_json(run / "metadata_audit.json")["rows"], "Population changed")
    mappings, x, y = {}, {}, {}
    context = p.read_json(run / "feature_context.json")
    for split in p.SPLITS:
        expected = [r for r in rows if r["split"] == split]
        x[split], mappings[split] = p.load_cache(run, split, expected, context)
        y[split] = np.array([c["classes"].index(r["true_label"]) for r in mappings[split]])
    accessed = p.read_json(run / "extraction.json")["decoded_sample_ids"]
    require(accessed == [r["sample_id"] for split in p.SPLITS for r in mappings[split]], "Extraction access list differs")
    require(not set(accessed).intersection(r["sample_id"] for r in rows if r["split"] == "test"), "Test input decoded for extraction")

    # Full independent layer/input reference, retaining the original classification head.
    net = resnet18(weights=None)
    weight = run / "weights/resnet18-f37072fd.pth"
    identity = p.read_json(run / "weight_identity.json")
    require(p.sha(weight) == identity["file_sha256"] == context["weights"]["file_sha256"], "Weight identity differs")
    net.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
    net.eval().requires_grad_(False).to(device)
    before = p.state_hash(net)
    require(before == identity["official_full_state_sha256"], "Original complete network state differs")
    repeat = {}
    tolerance = c["repeat_tolerance"]
    for split in p.SPLITS:
        # Label permutation deliberately accompanies the repeat; input reference never reads labels.
        permuted = copy.deepcopy(mappings[split])
        for r in permuted:
            r["true_label"] = c["classes"][(c["classes"].index(r["true_label"]) + 1) % 4]
        reference = independent_features(net, permuted, p.ROOT, c["batch_size"], device)
        require(reference.shape == x[split].shape == (len(permuted), 512) and np.isfinite(reference).all(), "Independent features invalid")
        require(np.allclose(reference, x[split], **tolerance), "Independent preprocessing/layer features differ")
        repeat[split] = {"rows": len(permuted), "dimensions": 512, "max_absolute_difference": float(np.max(np.abs(reference - x[split]))),
                         "allclose": True, "array_exact_equal": bool(np.array_equal(reference, x[split])),
                         "labels_permuted": True, "reference": "fresh full official net; independent NumPy normalization; avgpool hook"}
    require(before == p.state_hash(net), "Original network changed in verification")
    # Exercise production extraction with changed validation labels, without fitting anything.
    production_net, _ = p.load_extractor(weight, c, device)
    changed = copy.deepcopy(mappings["validation"][:16])
    for r in changed:
        r["true_label"] = "intentionally_not_a_label"
    alternate = p.extract_rows(production_net, changed, c, device)
    require(np.allclose(alternate, x["validation"][:16], **tolerance), "Production extraction reads validation labels")

    pipe = joblib.load(run / "model/pipeline.joblib")
    scaler, svm = pipe.named_steps["scaler"], pipe.named_steps["svm"]
    require(pipe.classes_.tolist() == list(range(4)), "SVM order differs")
    classes_file = p.read_json(run / "model/classes.json")
    require(classes_file["class_order"] == classes_file["decision_score_order"] == c["classes"], "Class/score order differs")
    require(classes_file["id_to_class"] == {str(i): name for i, name in enumerate(c["classes"])}, "Class IDs differ")
    require(classes_file["model_version"] == c["model_version"], "Model version differs")
    standalone_scaler, standalone_svm = joblib.load(run / "model/scaler.joblib"), joblib.load(run / "model/svm.joblib")
    train64 = x["train"].astype(np.float64)
    for attr, expected in (("mean_", train64.mean(0)), ("var_", train64.var(0))):
        require(np.allclose(getattr(scaler, attr), expected, rtol=1e-12, atol=1e-12), f"Independent scaler {attr} differs")
    require(int(scaler.n_samples_seen_) == 232, "Scaler did not fit exactly training rows")
    provenance = p.read_json(run / "model/fit_provenance.json")
    require(provenance["fit_sample_ids"] == [r["sample_id"] for r in mappings["train"]], "Fit IDs include heldout data or changed order")
    # A large heldout distribution makes contamination detectable; no official data are modified.
    with tempfile.TemporaryDirectory(prefix="aae5103-d03-independent-negative-") as temp:
        contaminated_x = np.concatenate([x["train"], np.full_like(x["validation"], 10000)])
        contaminated = StandardScaler().fit(contaminated_x)
        joblib.dump(contaminated, Path(temp) / "contaminated_scaler.joblib")
        try:
            p.check_scaler(contaminated, x["train"])
        except ValueError as exc:
            contamination_reason = str(exc)
        else:
            raise AssertionError("Contaminated scaler accepted")
    report = p.read_json(run / "metrics.json")
    selected_saved = {}
    for split in p.SPLITS:
        predicted, score = read_prediction(run / f"{split}_predictions.csv", mappings[split], c["classes"])
        require(np.array_equal(predicted, pipe.predict(x[split])) and np.array_equal(score, pipe.decision_function(x[split])), "Reload predictions or scores differ")
        require(np.array_equal(predicted, standalone_svm.predict(standalone_scaler.transform(x[split])))
                and np.array_equal(score, standalone_svm.decision_function(standalone_scaler.transform(x[split]))), "Standalone components differ")
        assert_metrics(manual_metrics(y[split], predicted, c["classes"]), report[split])
        records_report = p.read_json(run / f"{split}_per_record.json")
        require(records_report["class_order"] == c["classes"], "Per-record order differs")
        require({r["record_id"] for r in records_report["records"]} == {r["record_id"] for r in mappings[split]}, "Per-record population differs")
        for record in records_report["records"]:
            mask = np.array([r["record_id"] == record["record_id"] for r in mappings[split]])
            expected = manual_metrics(y[split][mask], predicted[mask], c["classes"])
            require(record["n_windows"] == int(mask.sum()) and record["correct"] == int(np.sum(y[split][mask] == predicted[mask])), "Record counts differ")
            require(record["confusion_matrix_rows_true_columns_predicted"] == expected["confusion_matrix_rows_true_columns_predicted"]
                    and record["accuracy_and_true_class_recall"] == expected["accuracy"], "Record metrics differ")
        selected_saved[split] = (predicted, score)

    # Independently refit every finite candidate, still with training rows only.
    selection = p.read_json(run / "selection.json")
    require(len(selection["candidates"]) == len(c["candidates"]) == 8, "Candidate set changed")
    independent_results, refitted = [], []
    for index, candidate in enumerate(selection["candidates"]):
        require(candidate["candidate_index"] == index and candidate["params"] == c["candidates"][index], "Undeclared candidate")
        require(candidate["fit_sample_ids"] == provenance["fit_sample_ids"], "Candidate fit IDs differ")
        fresh_scaler = StandardScaler().fit(x["train"])
        fresh_svm = SVC(**c["svm"], **candidate["params"], random_state=c["seed"])
        fresh_svm.fit(fresh_scaler.transform(x["train"]), y["train"])
        for split in p.SPLITS:
            pred, scores = read_prediction(run / "candidates" / f"{index:02d}" / f"{split}_predictions.csv", mappings[split], c["classes"])
            require(np.array_equal(pred, fresh_svm.predict(fresh_scaler.transform(x[split]))), "Independent candidate refit predictions differ")
            require(np.array_equal(scores, fresh_svm.decision_function(fresh_scaler.transform(x[split]))), "Independent candidate refit scores differ")
            actual = manual_metrics(y[split], pred, c["classes"])
            assert_metrics(actual, candidate[split])
            if split == "validation":
                independent_results.append((index, actual["macro_f1"], actual["balanced_accuracy"]))
        refitted.append((fresh_scaler, fresh_svm))
    maximum_f1 = max(r[1] for r in independent_results)
    tied = [r for r in independent_results if maximum_f1 - r[1] <= c["selection"]["tie_atol"]]
    maximum_ba = max(r[2] for r in tied)
    tied = [r for r in tied if maximum_ba - r[2] <= c["selection"]["tie_atol"]]
    chosen = sorted(tied, key=lambda r: (c["candidates"][r[0]]["kernel"] != "linear", c["candidates"][r[0]]["C"], r[0]))[0][0]
    require(chosen == selection["selected_candidate_index"], "Independent selection differs")
    require(selection["selected_params"] == c["candidates"][chosen], "Selected parameter record differs")
    require(svm.kernel == c["candidates"][chosen]["kernel"] and svm.C == c["candidates"][chosen]["C"], "Actual saved model parameters differ")
    for split in p.SPLITS:
        ref_scaler, ref_svm = refitted[chosen]
        require(np.array_equal(selected_saved[split][0], ref_svm.predict(ref_scaler.transform(x[split]))), "Selected model differs from independent fit")
        require(np.array_equal(selected_saved[split][1], ref_svm.decision_function(ref_scaler.transform(x[split]))), "Selected model scores differ from independent fit")
    p.write_json(run / "independent_verification.json", {"status": "PASS", "created_at": p.now(),
                 "device": str(device), "tolerance": tolerance, "feature_repeat": repeat,
                 "network_full_state_before": before, "network_full_state_after": p.state_hash(net),
                 "production_validation_label_invariance_rows": 16,
                 "independent_scaler_training_rows": 232, "contaminated_scaler_rejected": contamination_reason,
                 "all_candidate_independent_refits": 8, "candidate_predictions_and_scores_exact": True,
                 "manual_metrics_and_confusion_matrices": "all 8 candidates train/validation plus selected model",
                 "selected_per_record_verified": 12, "model_reload_predictions_exact": True,
                 "model_reload_decision_scores_exact": True, "standalone_components_equal_pipeline": True,
                 "selected_candidate_index": chosen, "test_decoded_or_extracted_or_predicted_or_scored": False,
                 "cross_device_or_version_verified": False, "file_byte_reproduction_claimed": False,
                 "independence_boundary": "separate preprocessing/layer reference, fit orchestration and manual metrics; same PyTorch/sklearn algorithms"})
    print("Independent verification PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    require(os.getenv("PYTHONDONTWRITEBYTECODE") == "1", "Set PYTHONDONTWRITEBYTECODE=1")
    verify(p.run_dir(args.run))
