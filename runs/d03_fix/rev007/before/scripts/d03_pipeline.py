"""D03 frozen-feature baseline. Deliberately exposes no test extraction/scoring entry."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision.models import ResNet18_Weights, resnet18

from data_contract import require
import d02_pipeline as d02

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/d03.yaml"
PROTOCOL = "docs/D03_运行协议.md"
CODE = "scripts/d03_pipeline.py"
BASELINE = "runs/d02_fix/rev005/deliverables.sha256"
INPUTS = ["configs/data.yaml", "data/records.csv", "data/samples.csv",
          "configs/cwt_color_limits.json", "runs/d02/cwru_de_007_v0.1/train_selection.json",
          "runs/d02/cwru_de_007_v0.1/heldout_generation.json",
          "runs/d02_fix/rev005/compatibility.json", BASELINE]
SPLITS = ("train", "validation")
ROW_FIELDS = ["sample_id", "record_id", "group_id", "source_family_id", "split",
              "true_label", "load_hp", "channel", "start_sample", "end_sample",
              "eligible_window_index", "processed_fs_hz", "preprocess_version",
              "raw_sha256", "signal_sha256_float64", "window_sha256_float64",
              "log_power_sha256_float64", "model_pixels_sha256", "color_limits_sha256",
              "model_image_path", "review_image_path"]


def now():
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()


def sha(path):
    return d02.file_hash(path)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def obj_hash(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def array_hash(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    require(bool(rows), "Cannot write empty result table")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def config(root=ROOT):
    c = yaml.safe_load((root / CONFIG).read_text())
    old = yaml.safe_load((root / "configs/data.yaml").read_text())
    contract = old["feature_input_contract"]
    require(c["weights"] == contract["weights"] == "ResNet18_Weights.IMAGENET1K_V1", "Weight contract differs")
    require(c["input"]["mean"] == contract["mean"] and c["input"]["std"] == contract["std"], "Normalization contract differs")
    require(not any(c["input"][k] for k in ("resize", "crop", "augmentation")), "Extra transforms forbidden")
    require(c["input"]["divide_by"] == 255.0 and c["input"]["mode"] == "RGB"
            and c["input"]["width"] == c["input"]["height"] == 224, "Image contract differs")
    require(c["classes"] == old["labels"] and c["allowed_feature_splits"] == list(SPLITS), "Class/split contract differs")
    require(c["feature_dim"] == 512 and c["device"] in ("cpu", "mps"), "Feature/device contract differs")
    require(c["scaler"]["fit_split"] == "train" and not c["selection"]["refit_train_plus_validation"], "Fit boundary differs")
    return c


def environment(c):
    return {"created_at": now(), "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("torch", "torchvision", "scikit-learn", "numpy", "scipy", "Pillow", "joblib", "PyYAML", "matplotlib")},
            "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(), "requested_device": c["device"],
            "seed": c["seed"], "torch_threads": c["torch_threads"],
            "PYTHONDONTWRITEBYTECODE": os.getenv("PYTHONDONTWRITEBYTECODE")}


def setup(c):
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    torch.set_num_threads(c["torch_threads"])
    torch.use_deterministic_algorithms(True)
    device = c["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    return torch.device(device)


def run_dir(value):
    p = (ROOT / value).resolve()
    require(p.is_relative_to((ROOT / "runs/d03").resolve()) and p != ROOT / "runs/d03", "Output must be inside runs/d03/<new-run>")
    return p


def bound_hashes(root=ROOT):
    return {p: sha(root / p) for p in INPUTS + [CONFIG, PROTOCOL, CODE]}


def check_declaration(run):
    declared = read_json(run / "declaration.json")
    require(declared["bound_hashes"] == bound_hashes(), "Declaration input/code/config changed")
    require(declared["config"] == config(), "Declared configuration changed")
    return declared


def audit_population(root=ROOT):
    """Metadata and bytes only for test; never calls Image.open on test."""
    rows, records = read_csv(root / "data/samples.csv"), read_csv(root / "data/records.csv")
    cfg = yaml.safe_load((root / "configs/data.yaml").read_text())
    plans = (read_json(root / INPUTS[4])["records"] + read_json(root / INPUTS[5])["records"])
    d02.validate_sample_population(rows, records, plans)
    require(len(records) == 16 and len(rows) == 704, "Frozen population count changed")
    by_record = {r["record_id"]: r for r in records}
    counts, groups = Counter(), defaultdict(set)
    seen_paths = set()
    colors_hash = sha(root / "configs/cwt_color_limits.json")
    audited = []
    for index, s in enumerate(rows):
        r = by_record[s["record_id"]]
        for k in ("group_id", "source_family_id", "true_label", "split", "load_hp", "channel"):
            require(s[k] == r[k], f"Sample/record mismatch: {k} {s['sample_id']}")
        require(s["split"] in ("train", "validation", "test") and s["true_label"] in cfg["labels"], "Unknown label/split")
        require(int(s["load_hp"]) in {"train": (0, 1), "validation": (2,), "test": (3,)}[s["split"]], "Load/split mismatch")
        require(s["raw_sha256"] == r["sha256"] and s["signal_sha256_float64"] == r["signal_sha256_float64"], "Signal provenance mismatch")
        require(s["preprocess_version"] == cfg["preprocess_version"] and s["color_limits_sha256"] == colors_hash,
                "Preprocessing/colour identity mismatch")
        require(int(s["processed_fs_hz"]) == 12000, "Sampling rate mismatch")
        expected_path = f"data/processed/{cfg['preprocess_version']}/model/{s['sample_id']}.png"
        require(s["model_image_path"] == expected_path and s["model_image_path"] != s["review_image_path"], "Model/review path mismatch")
        require(s["model_image_path"] not in seen_paths, "Duplicate input image path")
        seen_paths.add(s["model_image_path"])
        require((root / s["review_image_path"]).is_file(), "Missing review image (existence check only)")
        groups[s["group_id"]].add(s["split"])
        counts[(s["split"], s["true_label"])] += 1
        # Reading compressed file bytes here is integrity checking, not test feature extraction.
        audited.append({**{k: s[k] for k in ROW_FIELDS}, "manifest_index": index,
                        "model_image_sha256": sha(root / expected_path)})
    require(all(len(v) == 1 for v in groups.values()), "Source group crosses splits")
    for split, count in (("train", 58), ("validation", 59), ("test", 59)):
        require(all(counts[(split, label)] == count for label in cfg["labels"]), "Class/sample set incomplete")
    return audited


def declare(run):
    c = config()
    pre = read_json(run / "d02_preflight.json")
    require(pre["status"] == "PASS" and pre["snapshot_check"]["baseline"] == BASELINE, "D02 preflight required")
    require(pre["snapshot_check"]["baseline_sha256"] == sha(ROOT / BASELINE), "Baseline identity changed")
    run.mkdir(parents=True, exist_ok=True)
    rows = audit_population()
    write_json(run / "metadata_audit.json", {"status": "PASS", "created_at": now(), "rows": rows,
               "counts": dict(Counter(r["split"] for r in rows)), "test_image_access": "compressed bytes only; no decoding",
               "source_group_overlap": False, "source_families_are_not_verified_specimens": True})
    write_json(run / "environment.json", environment(c))
    write_json(run / "declaration.json", {"created_at": now(), "config": c, "bound_hashes": bound_hashes(),
               "validation_scores_seen": False, "candidate_count": len(c["candidates"]),
               "preflight_sha256": sha(run / "d02_preflight.json"), "metadata_audit_sha256": sha(run / "metadata_audit.json")})
    print("Declared candidate set before validation scoring", flush=True)


def tensor_input(path, c):
    # Reject incompatible inputs; no conversion, resizing, cropping or default weight transforms.
    with Image.open(path) as image:
        require(image.format == "PNG" and image.mode == "RGB" and image.size == (224, 224), "Expected 224x224 RGB PNG")
        pixels = np.array(image, dtype=np.uint8, copy=True)
    x = torch.from_numpy(pixels).permute(2, 0, 1).to(torch.float32).div(255.0)
    mean = torch.tensor(c["input"]["mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(c["input"]["std"], dtype=torch.float32)[:, None, None]
    return (x - mean) / std


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def load_extractor(weight_path, c, device):
    weight_path = Path(weight_path)
    weight_hash = sha(weight_path)
    require(weight_hash.startswith("f37072fd"), "Not the fixed official ResNet18 V1 weight checksum")
    model = resnet18(weights=None)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)  # fully overwrite initialization, no fallback
    official_state_hash = state_hash(model)
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    identity = {"enum": c["weights"], "url": ResNet18_Weights.IMAGENET1K_V1.url,
                "file_sha256": weight_hash, "official_full_state_sha256": official_state_hash,
                "extractor_state_sha256": state_hash(model), "strict_load": True,
                "feature_layer": c["feature_layer"], "feature_dim": 512}
    return model, identity


def extract_rows(model, rows, c, device, root=ROOT):
    require(bool(rows) and all(r["split"] in SPLITS for r in rows), "Test feature extraction forbidden")
    before = state_hash(model)
    require(not model.training and all(not p.requires_grad for p in model.parameters()), "Extractor is not frozen/eval")
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), c["batch_size"]):
            batch = rows[start:start + c["batch_size"]]
            x = torch.stack([tensor_input(root / r["model_image_path"], c) for r in batch]).to(device)
            y = model(x).cpu().numpy().copy()
            require(y.shape == (len(batch), 512) and np.isfinite(y).all(), "Invalid feature shape/values")
            output.append(y)
    require(before == state_hash(model), "Network parameters or buffers changed during extraction")
    return np.concatenate(output)


def cache_context(run, c, identity, device):
    return {"bound_hashes": check_declaration(run)["bound_hashes"], "weights": identity,
            "actual_device": str(device), "batch_size": c["batch_size"], "dtype": "float32",
            "seed": c["seed"], "torch_threads": c["torch_threads"],
            "environment_packages": read_json(run / "environment.json")["packages"]}


def save_cache(run, split, features, rows, context):
    require(split in SPLITS and all(r["split"] == split for r in rows), "Invalid cache split")
    p = run / "features" / f"{split}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        np.savez_compressed(f, features=features, sample_id=np.array([r["sample_id"] for r in rows]))
    mappings = [{**r, "feature_row": i, "feature_sha256": array_hash(features[i])} for i, r in enumerate(rows)]
    write_csv(run / "features" / f"{split}_mapping.csv", mappings)
    write_json(run / "features" / f"{split}.json", {"split": split, "shape": list(features.shape),
               "array_sha256": array_hash(features), "npz_sha256": sha(p), "context": context,
               "mapping_sha256": sha(run / "features" / f"{split}_mapping.csv"),
               "ordered_sample_ids": [r["sample_id"] for r in rows]})


def load_cache(run, split, expected_rows, expected_context):
    require(split in SPLITS, "Test cache loading forbidden")
    meta = read_json(run / "features" / f"{split}.json")
    p = run / "features" / f"{split}.npz"
    mappings_path = run / "features" / f"{split}_mapping.csv"
    require(meta["split"] == split and meta["context"] == expected_context, "Stale cache context")
    require(sha(p) == meta["npz_sha256"] and sha(mappings_path) == meta["mapping_sha256"], "Cache file changed")
    with np.load(p, allow_pickle=False) as data:
        x, ids = data["features"].copy(), data["sample_id"].tolist()
    mappings = read_csv(mappings_path)
    require(ids == meta["ordered_sample_ids"] == [r["sample_id"] for r in expected_rows], "Feature/sample row order mismatch")
    require(x.shape == (len(expected_rows), 512) and x.dtype == np.float32 and np.isfinite(x).all(), "Invalid cached features")
    require(array_hash(x) == meta["array_sha256"] and list(x.shape) == meta["shape"], "Feature checksum mismatch")
    require(len(mappings) == len(expected_rows), "Missing mapping rows")
    for i, (mapping, expected) in enumerate(zip(mappings, expected_rows)):
        require(all(mapping[k] == str(v) for k, v in expected.items()), "Feature metadata mapping mismatch")
        require(mapping["feature_row"] == str(i) and mapping["feature_sha256"] == array_hash(x[i]), "Feature row hash mismatch")
    return x, mappings


def extract(run, weights=None, download=False):
    check_declaration(run)
    c, device = config(), setup(config())
    rows = audit_population()
    require(rows == read_json(run / "metadata_audit.json")["rows"], "Inputs changed after declaration")
    cache = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    path = Path(weights).resolve() if weights else cache
    origin = "explicit existing fixed weight file" if weights else "existing Torch cache"
    if not path.is_file():
        require(download, "Fixed pretrained weights missing; explicitly allow --download")
        path = run / "weights/resnet18-f37072fd.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        require(not path.exists(), "Weight destination exists")
        torch.hub.download_url_to_file(ResNet18_Weights.IMAGENET1K_V1.url, path, hash_prefix="f37072fd", progress=False)
        origin = "downloaded from fixed official enum URL"
    model, identity = load_extractor(path, c, device)
    # Carry the exact original weight bytes with the model for future extraction.
    local = run / "weights/resnet18-f37072fd.pth"
    if path.resolve() != local.resolve():
        local.parent.mkdir(parents=True, exist_ok=True)
        require(not local.exists(), "Weight copy exists")
        shutil.copyfile(path, local)
    write_json(run / "weight_identity.json", {**identity, "origin": origin,
               "original_path": str(path), "portable_path": "weights/resnet18-f37072fd.pth",
               "torch_cache_checked": str(cache), "torch_cache_present": cache.is_file(), "actual_device": str(device)})
    context = cache_context(run, c, identity, device)
    write_json(run / "feature_context.json", context)
    decoded = []
    before = state_hash(model)
    for split in SPLITS:
        selected = [r for r in rows if r["split"] == split]
        features = extract_rows(model, selected, c, device)
        save_cache(run, split, features, selected, context)
        decoded.extend(r["sample_id"] for r in selected)
        print(f"Extracted {split}: {features.shape}", flush=True)
    write_json(run / "extraction.json", {"status": "PASS", "created_at": now(), "device": str(device),
               "state_before": before, "state_after": state_hash(model), "parameters_frozen": True,
               "decoded_sample_ids": decoded, "test_features_extracted": 0,
               "layer": c["feature_layer"], "context_sha256": sha(run / "feature_context.json")})


def labels(rows, classes):
    return np.array([classes.index(r["true_label"]) for r in rows], dtype=np.int64)


def fit_pipeline(x, y, rows, candidate, c):
    require(len(x) == len(y) == len(rows) and all(r["split"] == "train" for r in rows), "Only training rows may fit")
    require(len({r["sample_id"] for r in rows}) == len(rows), "Duplicate fit IDs")
    require(np.array_equal(y, labels(rows, c["classes"])), "Training labels misaligned")
    require(x.shape == (len(rows), 512) and np.isfinite(x).all(), "Invalid training features")
    svc_args = dict(c["svm"])
    svc_args.update(candidate, random_state=c["seed"])
    pipe = Pipeline([("scaler", StandardScaler(with_mean=c["scaler"]["with_mean"], with_std=c["scaler"]["with_std"])),
                     ("svm", SVC(**svc_args))])
    pipe.fit(x, y)
    check_scaler(pipe.named_steps["scaler"], x)
    require(np.array_equal(pipe.classes_, np.arange(4)), "SVC class order differs")
    require(pipe.named_steps["svm"].fit_status_ == 0, "SVM did not converge")
    return pipe


def check_scaler(scaler, train):
    require(int(scaler.n_samples_seen_) == len(train), "Scaler sample count includes non-training rows")
    # float64 reference to sklearn population statistics, including zero-variance convention.
    reference = train.astype(np.float64)
    mean, var = reference.mean(axis=0), reference.var(axis=0)
    require(np.allclose(scaler.mean_, mean, rtol=1e-12, atol=1e-12), "Scaler mean is not training-only")
    require(np.allclose(scaler.var_, var, rtol=1e-12, atol=1e-12), "Scaler variance is not training-only")
    expected_scale = np.sqrt(var)
    # Current features contain exact zero-variance dimensions; sklearn assigns scale=1.
    expected_scale[var == 0] = 1
    require(np.allclose(scaler.scale_, expected_scale, rtol=1e-12, atol=1e-12), "Scaler scale differs")


def metrics(y, pred, classes):
    return {"n_windows": len(y), "class_order": classes, "macro_f1": float(f1_score(y, pred, labels=np.arange(4), average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "accuracy": float(np.mean(y == pred)),
            "per_class_recall": dict(zip(classes, recall_score(y, pred, labels=np.arange(4), average=None, zero_division=0).tolist())),
            "confusion_matrix_rows_true_columns_predicted": confusion_matrix(y, pred, labels=np.arange(4)).tolist()}


def select_candidate(results, c):
    eps = c["selection"]["tie_atol"]
    best_f1 = max(r["validation"]["macro_f1"] for r in results)
    pool = [r for r in results if best_f1 - r["validation"]["macro_f1"] <= eps]
    best_ba = max(r["validation"]["balanced_accuracy"] for r in pool)
    pool = [r for r in pool if best_ba - r["validation"]["balanced_accuracy"] <= eps]
    return min(pool, key=lambda r: (r["params"]["kernel"] != "linear", r["params"]["C"], r["candidate_index"]))


def prediction_rows(pipe, x, rows, c):
    require(all(r["split"] in SPLITS for r in rows), "Test prediction forbidden")
    pred = pipe.predict(x)
    score = pipe.decision_function(x)
    require(score.shape == (len(rows), 4) and np.isfinite(score).all(), "Invalid decision scores")
    result = []
    for r, predicted, s in zip(rows, pred, score):
        result.append({"sample_id": r["sample_id"], "record_id": r["record_id"], "group_id": r["group_id"],
                       "source_family_id": r["source_family_id"], "split": r["split"], "true_label": r["true_label"],
                       "predicted_class": c["classes"][predicted], "decision_score": json.dumps(s.tolist()),
                       "model_version": c["model_version"]})
    return result, pred, score


def per_record(rows, y, pred, classes):
    output = []
    for rid in dict.fromkeys(r["record_id"] for r in rows):
        mask = np.array([r["record_id"] == rid for r in rows])
        group = [r for r in rows if r["record_id"] == rid]
        require(len({r["true_label"] for r in group}) == 1, "Mixed true labels in record")
        m = confusion_matrix(y[mask], pred[mask], labels=np.arange(4))
        output.append({"record_id": rid, "group_id": group[0]["group_id"], "source_family_id": group[0]["source_family_id"],
                       "split": group[0]["split"], "true_label": group[0]["true_label"], "load_hp": group[0]["load_hp"],
                       "n_windows": int(mask.sum()), "correct": int(np.sum(y[mask] == pred[mask])),
                       "accuracy_and_true_class_recall": float(np.mean(y[mask] == pred[mask])),
                       "predicted_class_counts": {label: int(np.sum(pred[mask] == i)) for i, label in enumerate(classes)},
                       "confusion_matrix_rows_true_columns_predicted": m.tolist()})
    return output


def plot_matrices(run, reports, c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    short = ["normal", "inner", "outer", "rolling"]
    for ax, split in zip(axes, SPLITS):
        matrix = np.array(reports[split]["confusion_matrix_rows_true_columns_predicted"])
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for (i, j), n in np.ndenumerate(matrix):
            ax.text(j, i, str(n), ha="center", va="center", color="white" if n > matrix.max() / 2 else "black")
        ax.set(xticks=range(4), yticks=range(4), xticklabels=short, yticklabels=short,
               xlabel="Predicted class", ylabel="True class", title=f"{split}: {reports[split]['n_windows']} windows")
    fig.suptitle(c["model_version"] + "\nTrain=fit diagnostic; validation=selected; test not evaluated", fontsize=10)
    fig.savefig(run / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def train(run):
    check_declaration(run)
    c = config()
    rows = audit_population()
    require(rows == read_json(run / "metadata_audit.json")["rows"], "Input identity changed")
    context = read_json(run / "feature_context.json")
    require(context["bound_hashes"] == bound_hashes(), "Feature source changed")
    require(context["weights"]["file_sha256"] == sha(run / "weights/resnet18-f37072fd.pth"), "Actual weight bytes changed")
    data, mapping = {}, {}
    for split in SPLITS:
        data[split], mapping[split] = load_cache(run, split, [r for r in rows if r["split"] == split], context)
    y = {split: labels(mapping[split], c["classes"]) for split in SPLITS}
    results, models = [], []
    for index, params in enumerate(c["candidates"]):
        pipe = fit_pipeline(data["train"], y["train"], mapping["train"], params, c)
        item = {"candidate_index": index, "params": params, "fit_sample_ids": [r["sample_id"] for r in mapping["train"]]}
        for split in SPLITS:
            predictions, pred, scores = prediction_rows(pipe, data[split], mapping[split], c)
            # Candidate identity is separate from the final selected model version.
            for row in predictions:
                row["model_version"] += f"/candidate_{index:02d}"
            write_csv(run / "candidates" / f"{index:02d}" / f"{split}_predictions.csv", predictions)
            item[split] = metrics(y[split], pred, c["classes"])
            item[f"{split}_per_record"] = per_record(mapping[split], y[split], pred, c["classes"])
        results.append(item)
        models.append(pipe)
        print(f"Candidate {index}: {params} validation macro-F1={item['validation']['macro_f1']:.6f}", flush=True)
    selected = select_candidate(results, c)
    pipe = models[selected["candidate_index"]]  # no refit, especially no train+validation fit
    model_dir = run / "model"
    model_dir.mkdir(exist_ok=False)
    with (model_dir / "pipeline.joblib").open("xb") as f:
        joblib.dump(pipe, f, compress=3)
    with (model_dir / "scaler.joblib").open("xb") as f:
        joblib.dump(pipe.named_steps["scaler"], f, compress=3)
    with (model_dir / "svm.joblib").open("xb") as f:
        joblib.dump(pipe.named_steps["svm"], f, compress=3)
    write_json(model_dir / "classes.json", {"model_version": c["model_version"], "class_order": c["classes"],
               "id_to_class": {str(i): label for i, label in enumerate(c["classes"])}, "svm_classes": pipe.classes_.tolist(),
               "decision_score_order": c["classes"], "decision_score_kind": "uncalibrated OVR decision_function; not probabilities",
               "prediction_rule": "SVC.predict OVO voting, break_ties=False; not forced argmax of OVR score"})
    write_json(run / "selection.json", {"created_at": now(), "declaration_sha256": sha(run / "declaration.json"),
               "rules": c["selection"], "candidates": results, "selected_candidate_index": selected["candidate_index"],
               "selected_params": selected["params"], "selection_reason": "max validation macro-F1, then balanced accuracy, then declared deterministic tie rules",
               "train_plus_validation_refit": False, "test_used": False})
    report = {}
    reloaded = joblib.load(model_dir / "pipeline.joblib")
    for split in SPLITS:
        predictions, pred, score = prediction_rows(pipe, data[split], mapping[split], c)
        require(np.array_equal(pred, reloaded.predict(data[split])) and np.array_equal(score, reloaded.decision_function(data[split])), "Reload output differs")
        write_csv(run / f"{split}_predictions.csv", predictions)
        report[split] = metrics(y[split], pred, c["classes"])
        write_json(run / f"{split}_per_record.json", {"class_order": c["classes"], "records": per_record(mapping[split], y[split], pred, c["classes"])})
    write_json(run / "metrics.json", {"model_version": c["model_version"], **report, "test": "not executed"})
    scaler = pipe.named_steps["scaler"]
    write_json(model_dir / "fit_provenance.json", {"fit_split": "train", "fit_sample_ids": [r["sample_id"] for r in mapping["train"]],
               "n_samples_seen": int(scaler.n_samples_seen_), "scaler_mean": scaler.mean_.tolist(), "scaler_var": scaler.var_.tolist(),
               "scaler_scale": scaler.scale_.tolist(), "fit_features_sha256": array_hash(data["train"]),
               "selected_candidate_index": selected["candidate_index"], "svm_effective_gamma": pipe.named_steps["svm"]._gamma,
               "svm_parameters": pipe.named_steps["svm"].get_params(), "scaler_parameters": scaler.get_params(),
               "refit_after_selection": False, "reload_predictions_and_scores_exact": True})
    plot_matrices(run, report, c)


def infer(run, split, output):
    require(split in SPLITS, "Test inference is outside this execution scope")
    check_declaration(run)
    c = config()
    rows = audit_population()
    context = read_json(run / "feature_context.json")
    require(context["bound_hashes"] == bound_hashes(), "Stale feature configuration")
    x, mappings = load_cache(run, split, [r for r in rows if r["split"] == split], context)
    pipe = joblib.load(run / "model/pipeline.joblib")
    mapping = read_json(run / "model/classes.json")
    require(mapping["class_order"] == c["classes"] and np.array_equal(pipe.classes_, np.arange(4)), "Model class identity differs")
    predictions, _, _ = prediction_rows(pipe, x, mappings, c)
    target = Path(output).resolve()
    require(target.is_relative_to((ROOT / "runs/d03").resolve()), "Inference output must be inside runs/d03")
    write_csv(target, predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("declare", "extract", "train", "infer"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--output")
    args = parser.parse_args()
    require(os.getenv("PYTHONDONTWRITEBYTECODE") == "1", "Set PYTHONDONTWRITEBYTECODE=1")
    run = run_dir(args.run)
    if args.stage == "declare":
        declare(run)
    elif args.stage == "extract":
        extract(run, args.weights, args.download)
    elif args.stage == "train":
        train(run)
    else:
        require(args.split and args.output, "infer requires --split and --output")
        infer(run, args.split, args.output)


if __name__ == "__main__":
    main()
