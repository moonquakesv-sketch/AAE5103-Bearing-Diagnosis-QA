"""D04 only: consume the fixed test set once with the original frozen D03 model.

D03's split guards remain intact. Numeric transforms, weight loading, metrics and
record summaries are reused; D04 owns the test authorization and persistent ledger.
"""
import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import csv
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import joblib
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import d03_pipeline as p
from data_contract import require

ROOT = p.ROOT
CONFIG = "configs/d04.json"
PROTOCOL = "docs/D04_运行协议.md"
CODE_FILES = ["scripts/d04_pipeline.py", "scripts/verify_d04.py", "scripts/check_d04.py",
              "tests/test_d04_pipeline.py", "tests/test_d04_guards.py", CONFIG, PROTOCOL]
CLASSES = ["normal", "inner_race", "outer_race", "rolling_element"]
MODEL = "runs/d03/20261022_rev006"


def config():
    c = p.read_json(ROOT / CONFIG)
    require(c["model_run"] == MODEL and c["compatibility"] == "runs/d03_fix/rev007/compatibility.json",
            "D04 must use original rev006 model and rev007 compatibility")
    require(c["samples_manifest"] == "data/samples.csv" and c["split"] == "test" and c["load_hp"] == 3,
            "Wrong manifest/split/load")
    require(c["class_order"] == CLASSES and (c["n_windows"], c["n_records"], c["windows_per_class"]) == (236, 4, 59),
            "Wrong fixed test population")
    require(c["model_version"] == p.config()["model_version"], "Wrong model version")
    return c


def run_dir(value):
    run = (ROOT / value).resolve()
    require(run.parent == (ROOT / "runs/d04").resolve(), "Use runs/d04/<run>")
    return run


def checked_identity(c):
    """All checks here precede torch.load or joblib.load."""
    model = ROOT / c["model_run"]
    require(p.sha(model / "freeze.json") == c["freeze_sha256"], "Original freeze identity changed")
    p.check_frozen_run(model, c["compatibility"])
    frozen = p.read_json(model / "freeze.json")
    require(frozen["selected_params"] == {"kernel": "linear", "C": 0.01}, "Wrong selected model")
    for name in ("model/scaler.joblib", "model/svm.joblib", "model/fit_provenance.json", "weight_identity.json"):
        path = f"{c['model_run']}/{name}"
        require(path in frozen["artifact_hashes"] and p.sha(ROOT / path) == frozen["artifact_hashes"][path],
                "Incomplete/changed frozen component: " + name)
    mapping = p.read_json(model / "model/classes.json")
    require(mapping["id_to_class"] == {str(i): label for i, label in enumerate(CLASSES)}
            and mapping["svm_classes"] == list(range(4)), "Wrong class ID mapping")
    return {name: p.sha(model / name) for name in
            ("freeze.json", "declaration.json", "model/pipeline.joblib", "model/scaler.joblib", "model/svm.joblib",
             "model/classes.json", "model/fit_provenance.json", "weight_identity.json", "weights/resnet18-f37072fd.pth")}


def validate_rows(rows, c, expected=None):
    require(len(rows) == c["n_windows"], "Missing/extra test samples")
    require(len({r["sample_id"] for r in rows}) == len(rows), "Duplicate sample IDs")
    require(all(r["split"] == "test" and int(r["load_hp"]) == 3 for r in rows), "Wrong split/load")
    require(Counter(r["true_label"] for r in rows) == Counter({k: 59 for k in CLASSES}), "Wrong class population")
    require(len({r["record_id"] for r in rows}) == 4, "Wrong record population")
    for label in CLASSES:
        group = [r for r in rows if r["true_label"] == label]
        require(len({r["record_id"] for r in group}) == 1, "Class spans unexpected records")
    require(len({r["model_image_path"] for r in rows}) == len(rows), "Duplicate image paths")
    if expected is not None:
        require(rows == expected, "Sample IDs/order/source changed after declaration")
    return rows


def test_rows(c):
    # This inherited audit checks metadata and compressed bytes, never decodes test images.
    return validate_rows([r for r in p.audit_population() if r["split"] == "test"], c)


def input_specs(rows):
    # Labels, record IDs, and label-bearing metadata never reach numeric inference.
    return [{"sample_id": r["sample_id"], "model_image_path": r["model_image_path"]} for r in rows]


def validate_features(x, n):
    require(x.shape == (n, 512) and x.dtype == np.float32 and np.isfinite(x).all(), "Invalid features: dimension/dtype/NaN")


def extract_features(net, inputs, c, device):
    require(bool(inputs) and all(set(r) == {"sample_id", "model_image_path"} for r in inputs), "Unexpected inference metadata")
    require(not net.training and all(not v.requires_grad for v in net.parameters()), "Network must be frozen/eval")
    before, parts = p.state_hash(net), []
    with torch.inference_mode():
        for start in range(0, len(inputs), c["batch_size"]):
            batch = inputs[start:start + c["batch_size"]]
            tensors = torch.stack([p.tensor_input(ROOT / r["model_image_path"], c) for r in batch]).to(device)
            x = net(tensors).cpu().numpy().copy()
            validate_features(x, len(batch))
            parts.append(x)
    require(p.state_hash(net) == before, "Network state changed")
    return np.concatenate(parts)


@contextmanager
def no_fitting():
    def forbidden(*args, **kwargs):
        raise RuntimeError("Fitting is forbidden throughout D04")
    with ExitStack() as stack:
        for cls, methods in ((Pipeline, ("fit", "fit_transform", "fit_predict")),
                             (StandardScaler, ("fit", "partial_fit", "fit_transform")), (SVC, ("fit",))):
            for method in methods:
                stack.enter_context(patch.object(cls, method, forbidden))
        yield


def model_state(pipe):
    return {"pipeline": joblib.hash(pipe), "scaler": joblib.hash(pipe.named_steps["scaler"]),
            "svm": joblib.hash(pipe.named_steps["svm"])}


def load_model(c):
    checked_identity(c)
    model = ROOT / c["model_run"]
    pipe = joblib.load(model / "model/pipeline.joblib")
    require(type(pipe) is Pipeline and list(pipe.named_steps) == ["scaler", "svm"], "Unexpected pipeline")
    scaler, svm = pipe.named_steps["scaler"], pipe.named_steps["svm"]
    require(type(scaler) is StandardScaler and type(svm) is SVC, "Unexpected fitted estimator types")
    require(svm.kernel == "linear" and svm.C == .01 and not svm.probability and svm.decision_function_shape == "ovr",
            "Wrong frozen SVM parameters")
    require(np.array_equal(pipe.classes_, np.arange(4)) and scaler.n_features_in_ == svm.n_features_in_ == 512,
            "Wrong estimator classes/features")
    require(int(scaler.n_samples_seen_) == 232, "Scaler fit population changed")
    for name in ("scaler", "svm"):
        standalone = joblib.load(model / f"model/{name}.joblib")
        require(joblib.hash(standalone) == joblib.hash(pipe.named_steps[name]), "Standalone component differs")
    return pipe


def predict_features(pipe, x):
    validate_features(x, len(x))
    before = model_state(pipe)
    predicted, score = pipe.predict(x), pipe.decision_function(x)
    require(predicted.shape == (len(x),) and np.isin(predicted, np.arange(4)).all(), "Invalid predicted IDs")
    require(score.shape == (len(x), 4) and np.isfinite(score).all(), "Invalid decision scores")
    require(model_state(pipe) == before, "Model/Scaler changed during prediction")
    return predicted, score


def bindings(run, c):
    files = CODE_FILES + [c["baseline"], c["compatibility"], "scripts/d03_pipeline.py", "scripts/d02_pipeline.py",
                         "scripts/data_contract.py", "scripts/validate_d01.py", "configs/d03.yaml", "configs/data.yaml",
                         c["samples_manifest"], "data/records.csv", "configs/cwt_color_limits.json"]
    files += [str((run / name).relative_to(ROOT)) for name in
              ("baseline_validation.json", "protected_before.json", "prechecks.json")]
    return {path: p.sha(ROOT / path) for path in files}


def declare(run):
    c = config()
    require(not list((ROOT / "runs/d04").glob("*/execution_started.json")), "Test already used; inspect existing evaluation")
    preflight = p.read_json(run / "baseline_validation.json")
    require(preflight["status"] == preflight["content_check"]["status"] == "PASS", "Baseline/content preflight failed")
    require(p.verify_snapshot(ROOT, c["baseline"]) == preflight["snapshot_check"], "Stale baseline preflight")
    checks = p.read_json(run / "prechecks.json")
    require(checks["status"] == "PASS" and checks["code_hashes"] == {f: p.sha(ROOT / f) for f in CODE_FILES}, "Stale/failed D04 checks")
    identity = checked_identity(c)
    rows = test_rows(c)
    p.write_json(run / "environment.json", p.environment(p.config()))
    p.write_json(run / "declaration.json", {"created_at": p.now(), "stage": "D04_pre_test_declaration", "config": c,
                 "bound_hashes": bindings(run, c), "frozen_identity": identity, "feature_config": p.config(),
                 "test_rows": rows, "ordered_sample_ids": [r["sample_id"] for r in rows],
                 "formal_test_previously_executed": False, "authorization": "User explicitly authorized D04 implementation and one formal test on 2026-10-23",
                 "test_image_decoding_before_declaration": 0, "test_features_before_declaration": 0,
                 "test_predictions_before_declaration": 0})
    print("D04 declaration saved before any test decoding, extraction or prediction", flush=True)


def check_declaration(run):
    declared = p.read_json(run / "declaration.json")
    c = config()
    require(declared["config"] == c and declared["feature_config"] == p.config(), "D04 configuration changed")
    require(declared["bound_hashes"] == bindings(run, c), "D04 input/code/evidence changed")
    require(declared["frozen_identity"] == checked_identity(c), "Frozen model identity changed")
    validate_rows(test_rows(c), c, declared["test_rows"])
    require(declared["ordered_sample_ids"] == [r["sample_id"] for r in declared["test_rows"]], "Declared sample order mismatch")
    return declared


def event(run, kind, **fields):
    with (run / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": p.now(), "event": kind, **fields}, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "Refusing to replace persisted arrays")
    # A crash may leave .partial; preserve it, and require explicit recovery.
    with path.with_suffix(".partial").open("xb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    path.with_suffix(".partial").rename(path)


def read_chunk(path, ids, context):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    require(arrays.pop("sample_id").tolist() == ids and arrays.pop("context").item() == context, "Stale checkpoint/order")
    return arrays


def verify_checkpoint_log(run):
    """Every checkpoint needs one original commit; reuse cannot establish trust."""
    committed = {}
    for line in (run / "events.jsonl").read_text().splitlines():
        item = json.loads(line)
        if item["event"] in ("features_saved", "feature_checkpoint_reused", "predictions_saved", "prediction_checkpoint_reused"):
            require(type(item["offset"]) is int and item["offset"] >= 0, "Invalid checkpoint offset")
            kind = "features" if item["event"].startswith("feature") else "predictions"
            path = run / f"checkpoints/{item['offset']:03d}_{kind}.npz"
            if item["event"].endswith("_saved"):
                require(path not in committed, "Duplicate checkpoint commit")
                if kind == "predictions":
                    require(run / f"checkpoints/{item['offset']:03d}_features.npz" in committed,
                            "Prediction commit without feature commit")
                committed[path] = item["sha256"]
            else:
                require(path in committed, "Checkpoint reuse without original commit")
                require(item["sha256"] == committed[path], "Checkpoint reuse digest differs from original commit")
    files = {path for path in (run / "checkpoints").rglob("*") if path.is_file()}
    require(files == set(committed), "Checkpoint files and original commits differ; preserve uncommitted/missing files")
    for path, digest in committed.items():
        require(p.sha(path) == digest, "Checkpoint changed after commit")
    return committed


@contextmanager
def evaluation_session(run, resume=False):
    """Hold one project-wide OS lock from registration through all evaluation I/O."""
    run = run_dir(run)
    base = (ROOT / "runs/d04").resolve()
    base.mkdir(parents=True, exist_ok=True)
    # Keep this inode after release. Unlinking it would permit two independent locks.
    with (base / ".evaluation.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another D04 evaluation is active; do not start or resume concurrently") from None
        try:
            require(not (run / "completion.json").exists(), "Evaluation complete; recompute metrics from saved predictions only")
            starts = list(base.glob("*/execution_started.json"))
            own = run / "execution_started.json"
            require(all(path == own for path in starts), "Another formal test exists; reuse its results")
            require(resume if own.exists() else not resume,
                    "Interrupted evaluation exists; inspect logs and use --resume" if own.exists() else "Nothing to resume")
            declared = check_declaration(run)
            if own.exists():
                require(p.read_json(own)["declaration_sha256"] == p.sha(run / "declaration.json"), "Recovery declaration changed")
                event(run, "resume_requested", policy="validate original commits and reuse checkpoints; never replace outputs")
            else:
                p.write_json(own, {"started_at": p.now(), "declaration_sha256": p.sha(run / "declaration.json"),
                             "model_version": declared["config"]["model_version"], "test_set_status": "consumed from this point; never unused holdout again"})
                event(run, "formal_test_started")
            yield declared
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def ensure_json(path, obj):
    if path.exists():
        require(p.read_json(path) == obj, "Persisted output differs; do not overwrite")
    else:
        p.write_json(path, obj)


def ensure_csv(path, rows, fields):
    if path.exists():
        expected = [{k: str(r[k]) for k in fields} for r in rows]
        require(p.read_csv(path) == expected, "Persisted table differs; do not overwrite")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def save_results(run, rows, x, pred, scores, context, c):
    ids = [r["sample_id"] for r in rows]
    path = run / "features/test.npz"
    if path.exists():
        old = read_chunk(path, ids, context)
        require(np.array_equal(old["features"], x), "Persisted full features differ")
    else:
        save_npz(path, features=x, sample_id=np.array(ids), context=np.array(context))
    mapping = [{**r, "feature_row": i, "feature_sha256": p.array_hash(x[i])} for i, r in enumerate(rows)]
    ensure_csv(run / "features/test_mapping.csv", mapping, list(mapping[0]))
    ensure_json(run / "features/test.json", {"split": "test", "shape": list(x.shape), "context": context,
                "array_sha256": p.array_hash(x), "npz_sha256": p.sha(path), "ordered_sample_ids": ids,
                "mapping_sha256": p.sha(run / "features/test_mapping.csv")})
    predictions = [{"sample_id": r["sample_id"], "record_id": r["record_id"], "group_id": r["group_id"],
                    "source_family_id": r["source_family_id"], "split": "test", "true_label": r["true_label"],
                    "predicted_class": CLASSES[int(pred[i])], "decision_score": json.dumps(scores[i].tolist()),
                    "model_version": c["model_version"]} for i, r in enumerate(rows)]
    ensure_csv(run / "test_predictions.csv", predictions, list(predictions[0]))
    errors = [r for r in predictions if r["true_label"] != r["predicted_class"]]
    ensure_csv(run / "misclassifications.csv", errors, list(predictions[0]))
    y = p.labels(rows, CLASSES)  # first label access in the numerical result path: scoring only
    ensure_json(run / "metrics.json", {"model_version": c["model_version"], "split": "test", **p.metrics(y, pred, CLASSES)})
    ensure_json(run / "test_per_record.json", {"class_order": CLASSES, "records": p.per_record(rows, y, pred, CLASSES)})
    ensure_json(run / "errors.json", {"n_windows": len(rows), "misclassifications": errors,
                "n_misclassified": len(errors), "failed_samples": [], "n_failed_samples": 0,
                "failure_policy": "Any processing error stops completion and is preserved in events.jsonl and CLI log"})


def evaluate(run, resume=False):
    with evaluation_session(run, resume) as declared:
        active_ids = []
        try:
            require(not list(run.rglob("*.partial")), "Uncommitted partial checkpoint; preserve and review before recovery")
            verify_checkpoint_log(run)
            c, fc, rows = declared["config"], declared["feature_config"], declared["test_rows"]
            context = p.sha(run / "declaration.json")
            device = p.setup(fc)
            require(p.environment(fc)["packages"] == p.read_json(run / "environment.json")["packages"], "Environment drift")
            with no_fitting():
                pipe = load_model(c)
                before = model_state(pipe)
                net, identity = p.load_extractor(ROOT / c["model_run"] / "weights/resnet18-f37072fd.pth", fc, device)
                original = p.read_json(ROOT / c["model_run"] / "weight_identity.json")
                require(all(identity[k] == original[k] for k in identity), "Extractor identity differs")
                network_before = p.state_hash(net)
                all_x, all_pred, all_scores = [], [], []
                for offset in range(0, len(rows), fc["batch_size"]):
                    batch = rows[offset:offset + fc["batch_size"]]
                    ids = [r["sample_id"] for r in batch]
                    active_ids = ids
                    event(run, "batch_started", offset=offset, sample_ids=ids)
                    feature_path = run / f"checkpoints/{offset:03d}_features.npz"
                    prediction_path = run / f"checkpoints/{offset:03d}_predictions.npz"
                    if feature_path.exists():
                        require(resume, "Unexpected preexisting feature checkpoint")
                        x = read_chunk(feature_path, ids, context)["features"]
                        event(run, "feature_checkpoint_reused", offset=offset, sha256=p.sha(feature_path))
                    else:
                        require(not prediction_path.exists(), "Prediction checkpoint without features")
                        x = extract_features(net, input_specs(batch), fc, device)
                        save_npz(feature_path, features=x, sample_id=np.array(ids), context=np.array(context))
                        event(run, "features_saved", offset=offset, count=len(batch), sha256=p.sha(feature_path))
                    validate_features(x, len(batch))
                    feature_context = p.obj_hash({"declaration": context, "features": p.array_hash(x)})
                    if prediction_path.exists():
                        require(resume, "Unexpected preexisting prediction checkpoint")
                        saved = read_chunk(prediction_path, ids, feature_context)
                        pred, scores = saved["predicted"], saved["scores"]
                        event(run, "prediction_checkpoint_reused", offset=offset, sha256=p.sha(prediction_path))
                    else:
                        pred, scores = predict_features(pipe, x)
                        save_npz(prediction_path, predicted=pred, scores=scores, sample_id=np.array(ids), context=np.array(feature_context))
                        event(run, "predictions_saved", offset=offset, count=len(batch), sha256=p.sha(prediction_path))
                    require(pred.shape == (len(batch),) and np.isin(pred, np.arange(4)).all()
                            and scores.shape == (len(batch), 4) and np.isfinite(scores).all(), "Invalid cached predictions")
                    all_x.append(x)
                    all_pred.append(pred)
                    all_scores.append(scores)
                    print(f"Committed test windows {offset + len(batch)}/{len(rows)}", flush=True)
                x, pred, scores = np.concatenate(all_x), np.concatenate(all_pred), np.concatenate(all_scores)
                validate_features(x, 236)
                require(model_state(pipe) == before and p.state_hash(net) == network_before, "Fitted/network state changed")
                require(checked_identity(c) == declared["frozen_identity"], "Frozen files changed during evaluation")
                save_results(run, rows, x, pred, scores, context, c)
                ensure_json(run / "state_invariance.json", {"status": "PASS", "model_before": before, "model_after": model_state(pipe),
                            "network_before": network_before, "network_after": p.state_hash(net), "weight_identity": identity,
                            "frozen_files_before": declared["frozen_identity"], "frozen_files_after": checked_identity(c),
                            "fitting_methods_blocked": True, "n_feature_rows": len(x), "n_prediction_rows": len(pred)})
            outputs = {str(path.relative_to(run)): p.sha(path) for path in sorted(run.rglob("*")) if path.is_file()
                       and (path.suffix == ".npz" or path.name in ("test.json", "test_mapping.csv", "test_predictions.csv",
                           "misclassifications.csv", "metrics.json", "test_per_record.json", "errors.json", "state_invariance.json"))}
            p.write_json(run / "completion.json", {"completed_at": p.now(), "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW",
                         "declaration_sha256": context, "n_windows": 236, "n_records": 4, "output_hashes": outputs,
                         "test_set_used": True, "model_version": c["model_version"], "formal_evaluations": 1})
            event(run, "formal_test_completed", n_windows=236)
        except BaseException as exc:
            event(run, "execution_interrupted", sample_ids=active_ids, error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("declare", "evaluate"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require(os.getenv("PYTHONDONTWRITEBYTECODE") == "1", "Set PYTHONDONTWRITEBYTECODE=1")
    run = run_dir(args.run)
    require(not args.resume or args.stage == "evaluate", "--resume is evaluate-only")
    if args.stage == "declare":
        declare(run)
    else:
        evaluate(run, args.resume)


if __name__ == "__main__":
    main()
