"""Pre-test D04 checks; real positive smoke uses validation only, never test images."""
import argparse
import ast
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

import d04_pipeline as q


def check(run):
    hashes = {f: q.p.sha(q.ROOT / f) for f in q.CODE_FILES}
    # Full regression belongs to this exact code revision and exits nonzero on failure.
    with (run / "regression_pretest.log").open("x") as log:
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=q.ROOT,
                                stdout=log, stderr=subprocess.STDOUT)
    q.require(result.returncode == 0, "Pre-test regression failed; inspect regression_pretest.log")
    c, fc = q.config(), q.p.config()
    official = q.ROOT / c["model_run"]
    # Metadata comes from the old validated audit, with every test row excluded.
    validation = [r for r in q.p.read_json(official / "metadata_audit.json")["rows"] if r["split"] == "validation"][:16]
    allowed = {(q.ROOT / r["model_image_path"]).resolve() for r in validation}
    original_tensor = q.p.tensor_input
    decoded = []

    def guarded_tensor(path, config):
        q.require(Path(path).resolve() in allowed, "Pre-test image access outside validation smoke set")
        decoded.append(str(Path(path).relative_to(q.ROOT)))
        return original_tensor(path, config)

    with q.no_fitting(), patch.object(q.p, "tensor_input", guarded_tensor):
        pipe = q.load_model(c)
        before = q.model_state(pipe)
        device = q.p.setup(fc)
        net, identity = q.p.load_extractor(official / "weights/resnet18-f37072fd.pth", fc, device)
        state_before = q.p.state_hash(net)
        x = q.extract_features(net, q.input_specs(validation), fc, device)
        with np.load(official / "features/validation.npz", allow_pickle=False) as z:
            q.require(np.array_equal(x, z["features"][:16]), "Validation feature smoke differs")
        predicted, scores = q.predict_features(pipe, x)
        old = q.p.read_csv(official / "validation_predictions.csv")[:16]
        q.require(predicted.tolist() == [q.CLASSES.index(r["predicted_class"]) for r in old], "Validation prediction smoke differs")
        import json
        q.require(np.array_equal(scores, np.array([json.loads(r["decision_score"]) for r in old])), "Validation score smoke differs")
        q.require(q.model_state(pipe) == before and q.p.state_hash(net) == state_before, "Smoke changed fitted/network state")
    # Audit the small numeric call graph. Labels only enter save_results' scoring.
    tree = ast.parse((q.ROOT / "scripts/d04_pipeline.py").read_text())
    numeric = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("input_specs", "extract_features", "predict_features")]
    for node in numeric:
        q.require(not any(isinstance(n, ast.Constant) and n.value in ("true_label", "label", "record_id") for n in ast.walk(node)), "Labels in numeric function")
        q.require(not any(isinstance(n, ast.Attribute) and n.attr in ("fit", "partial_fit", "fit_transform") for n in ast.walk(node)), "Fit call in numeric function")
    q.require(hashes == {f: q.p.sha(q.ROOT / f) for f in q.CODE_FILES}, "Code changed during checks")
    q.p.write_json(run / "prechecks.json", {"status": "PASS", "created_at": q.p.now(), "code_hashes": hashes,
                    "regression_log_sha256": q.p.sha(run / "regression_pretest.log"), "validation_smoke_rows": 16,
                    "features_exact": True, "predictions_exact": True, "decision_scores_exact": True,
                    "decoded_validation_paths": decoded, "test_images_decoded": 0, "test_features_extracted": 0,
                    "test_predictions": 0, "fitting_blocked": True, "numeric_label_boundary_ast": "PASS",
                    "state_unchanged": True, "original_weight_identity": identity})
    print("D04 prechecks PASS; only 16 validation images decoded; no test inference", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    check(q.run_dir(args.run))
