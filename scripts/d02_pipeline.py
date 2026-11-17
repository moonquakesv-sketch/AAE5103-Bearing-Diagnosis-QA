"""D02 only: staged train pilot, train-only color fit, frozen held-out rendering.

Run from project root in the existing aae5103 environment; see docs/D02_运行协议.md.
No classifiers, weights, learned features or classification scores are used.
"""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/aae5103-matplotlib")
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
import pywt
from scipy.io import loadmat
import yaml

from data_contract import require, resample_signal, validate_config
from validate_d01 import digest, validate_record_metadata, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
VERSION = "cwru_de_007_v0.1"
RUN = ROOT / "runs/d02" / VERSION
FIELDS = ["sample_id", "record_id", "group_id", "source_family_id", "split", "true_label",
          "load_hp", "channel", "start_sample", "end_sample", "eligible_window_index",
          "processed_fs_hz", "preprocess_version", "raw_sha256", "signal_sha256_float64",
          "window_sha256_float64", "log_power_sha256_float64", "model_pixels_sha256",
          "color_limits_sha256", "model_image_path", "review_image_path"]


def now():
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(a):
    return digest(np.ascontiguousarray(a, dtype="<f8").tobytes())


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_inputs():
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text(encoding="utf-8"))
    validate_config(cfg)
    require(cfg["preprocess_version"] == VERSION, "New processing version requires a new run directory")
    rows = read_csv(ROOT / "data/records.csv")
    validate_record_metadata(rows, cfg, read_json(ROOT / "runs/d01/source_audit.json"))
    return cfg, rows


def input_hashes():
    return {name: file_hash(ROOT / name) for name in (
        "configs/data.yaml", "data/records.csv", "scripts/d02_pipeline.py",
        "scripts/data_contract.py", "scripts/validate_d01.py", "runs/d01/source_audit.json")}


# A validation-only revision must not silently replace the historical train freeze.
UNCHANGED_PROCESSING_FUNCTIONS = (
    "frequencies", "log_power", "model_pixels", "render", "load_record",
    "eligible_windows", "choose_balanced", "sample_id", "metadata", "fit_color_limits")


def processing_fingerprints(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return {node.name: digest(ast.dump(node, include_attributes=False).encode())
            for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name in UNCHANGED_PROCESSING_FUNCTIONS}


def check_processing_inputs(expected, compatibility=None):
    current = input_hashes()
    if expected == current:
        return None
    require(compatibility is not None, "Frozen processing inputs changed; explicit validation-revision compatibility required")
    path = ROOT / compatibility
    approved = read_json(path)
    require(approved["purpose"] == "D02_validation_only_revision"
            and approved["frozen_input_hashes"] == expected
            and approved["approved_input_hashes"] == current, "Unapproved processing input change")
    require(set(expected) == set(current)
            and {k for k in expected if expected[k] != current[k]} == {"scripts/d02_pipeline.py"},
            "Compatibility may change only the audited pipeline code")
    prior = ROOT / approved["prior_pipeline_path"]
    require(file_hash(prior) == expected["scripts/d02_pipeline.py"], "Archived pipeline changed")
    fingerprints = processing_fingerprints(ROOT / "scripts/d02_pipeline.py")
    require(set(fingerprints) == set(UNCHANGED_PROCESSING_FUNCTIONS)
            and fingerprints == processing_fingerprints(prior)
            == approved["unchanged_processing_functions"], "Numerical processing or rendering changed")
    return approved


def frequencies(cfg):
    c = cfg["cwt"]
    return np.geomspace(c["frequency_min_hz"], c["frequency_max_hz"], c["n_frequencies"])


def log_power(values, cfg):
    c = cfg["cwt"]
    values = np.asarray(values, dtype=np.float64)
    require(len(values) == cfg["segmentation"]["window_samples"], "Wrong window length")
    require(np.isfinite(values).all() and np.ptp(values) > 0, "Invalid CWT window")
    centered = values - values.mean()
    pad = c["pad_samples_each_side"]
    padded = np.pad(centered, pad, mode=c["pad_mode"])
    freqs = frequencies(cfg)
    scales = pywt.frequency2scale(c["wavelet"], freqs / cfg["sampling"]["target_fs_hz"])
    coefs, actual = pywt.cwt(padded, scales, c["wavelet"],
        sampling_period=c["sampling_period_seconds"], method=c["method"], precision=c["precision"])
    require(np.allclose(actual, freqs, rtol=1e-12), "CWT frequency coordinate mismatch")
    matrix = 10 * np.log10(np.maximum(np.abs(coefs[:, pad:pad + len(values)]) ** 2, 1e-12))
    require(matrix.shape == (len(freqs), len(values)) and np.isfinite(matrix).all(), "Invalid CWT matrix")
    return matrix


def model_pixels(matrix, limits, cfg):
    # CWT rows ascend in Hz. Reverse BEFORE scalar interpolation: image row 0 is 3000 Hz.
    scalar = Image.fromarray(np.asarray(matrix[::-1], dtype=np.float32))
    scalar = np.asarray(scalar.resize((cfg["image"]["width"], cfg["image"]["height"]),
                                     resample=Image.Resampling.BILINEAR))
    normalized = np.clip((scalar - limits["vmin"]) / (limits["vmax"] - limits["vmin"]), 0, 1)
    return colormaps[cfg["image"]["colormap"]](normalized, bytes=True)[:, :, :3].copy()


def render(matrix, limits, cfg, model_path, review_path):
    require(not model_path.exists() and not review_path.exists(), "Refusing to overwrite images")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = model_pixels(matrix, limits, cfg)
    Image.fromarray(pixels).save(model_path)
    fs = cfg["sampling"]["target_fs_hz"]
    freqs = frequencies(cfg)
    # Pixel centers sit on the actual log-frequency grid; edges are geometric midpoints.
    ratio = freqs[1] / freqs[0]
    edges = np.concatenate(([freqs[0] / np.sqrt(ratio)], np.sqrt(freqs[:-1] * freqs[1:]),
                            [freqs[-1] * np.sqrt(ratio)]))
    times = (np.arange(matrix.shape[1] + 1) - .5) / fs
    # Fixed axes prevent deferred pcolormesh rasterization from obscuring labels
    # when automatic layout changes the axes rectangle during savefig.
    fig = Figure(figsize=(6.4, 4.2), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([.13, .14, .68, .82])
    cax = fig.add_axes([.84, .14, .025, .82])
    mesh = ax.pcolormesh(times, edges, matrix, shading="flat",
        cmap=cfg["image"]["colormap"], norm=Normalize(limits["vmin"], limits["vmax"], clip=True))
    ax.set_yscale("log")
    ax.set_ylim(edges[0], edges[-1])
    ax.set_xlim(times[0], times[-1])
    ticks = [100, 200, 500, 1000, 2000, 3000]
    ax.set_yticks(ticks, [str(x) for x in ticks])
    ax.set_xlabel("Time within window (s)")
    ax.set_ylabel("Frequency (Hz, log scale)")
    fig.colorbar(mesh, cax=cax, label="CWT log-power (dB, uncalibrated)")
    fig.savefig(review_path, metadata={"Software": "AAE5103 D02"})
    fig.clear()
    return digest(pixels.tobytes())


def load_record(row, cfg):
    path = ROOT / row["raw_path"]
    require(file_hash(path) == row["sha256"], f"Raw hash mismatch: {row['record_id']}")
    values = np.asarray(loadmat(path)[row["channel"]], dtype=np.float64).ravel()
    require(array_hash(values) == row["signal_sha256_float64"], "Selected signal hash mismatch")
    require(np.isfinite(values).all() and np.ptp(values) > 0, "Invalid full record")
    kind = "normal" if row["true_label"] == "normal" else "fault"
    result = resample_signal(values, cfg, int(row["original_fs_hz"]), kind)
    require(np.isfinite(result).all(), "Invalid resampled record")
    return result


def eligible_windows(values, cfg):
    seg = cfg["segmentation"]
    edge = cfg["sampling"]["discard_edge_samples_each_side"]
    length, stride = seg["window_samples"], seg["stride_samples"]
    valid, rejected = [], []
    for index, start in enumerate(range(edge, len(values) - edge - length + 1, stride)):
        end = start + length
        window = values[start:end]
        reason = "nonfinite" if not np.isfinite(window).all() else "constant" if np.ptp(window) == 0 else None
        entry = {"eligible_window_index": index, "start_sample": start, "end_sample": end}
        (rejected if reason else valid).append({**entry, "reason": reason} if reason else entry)
    return valid, rejected


def choose_balanced(plans):
    require(plans and all(p["split"] == "train" for p in plans), "Balance input must be train only")
    count = min(len(p["valid_windows"]) for p in plans)
    require(count > 0, "No balanced training population")
    for p in plans:
        indices = np.floor(np.linspace(0, len(p["valid_windows"]) - 1, count)).astype(int)
        require(len(set(indices)) == count, "Repeated balance selection")
        p["selected_windows"] = [p["valid_windows"][i] for i in indices]
    return count


def sample_id(row, window):
    # Opaque stable identifier; labels and load do not participate in the identifier.
    identity = f"{row['record_id']}:{row['channel']}:{window['start_sample']}:{window['end_sample']}:{VERSION}"
    return "s_" + digest(identity.encode())[:24]


def metadata(row, window, values, matrix, colors_hash, pixels_hash=""):
    sid = sample_id(row, window)
    return {"sample_id": sid, **{k: row[k] for k in (
        "record_id", "group_id", "source_family_id", "split", "true_label", "load_hp", "channel")},
        **window, "processed_fs_hz": 12000, "preprocess_version": VERSION,
        "raw_sha256": row["sha256"], "signal_sha256_float64": row["signal_sha256_float64"],
        "window_sha256_float64": array_hash(values), "log_power_sha256_float64": array_hash(matrix),
        "model_pixels_sha256": pixels_hash, "color_limits_sha256": colors_hash,
        "model_image_path": f"data/processed/{VERSION}/model/{sid}.png",
        "review_image_path": f"data/processed/{VERSION}/review/{sid}.png"}


def fit_color_limits(matrices, sample_rows, cfg):
    require(len(matrices) == len(sample_rows) > 0, "Invalid fit population")
    require(all(r["split"] == "train" for r in sample_rows), "Held-out data cannot fit colors")
    require(len({r["sample_id"] for r in sample_rows}) == len(sample_rows), "Repeated color fit sample")
    values = np.concatenate([m.ravel() for m in matrices])
    quantiles = cfg["image"]["color_limits"]["quantiles"]
    lo, hi = np.quantile(values, quantiles, method="linear")
    require(np.isfinite([lo, hi]).all() and hi > lo, "Degenerate color limits")
    return {"schema_version": 1, "preprocess_version": VERSION, "fit_split": "train",
        "quantiles": quantiles, "quantile_method": "linear", "vmin": float(lo), "vmax": float(hi),
        "population": "all_pixels_of_balanced_train_log_power_matrices_before_resize",
        "matrix_dtype": "float64", "pixel_count": int(values.size), "window_count": len(sample_rows),
        "sample_ids": [r["sample_id"] for r in sample_rows],
        "matrix_hashes": {r["sample_id"]: r["log_power_sha256_float64"] for r in sample_rows},
        "input_hashes": input_hashes(), "fit_at": now()}


def environment():
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "device": "CPU", "packages": {p: importlib.metadata.version(p) for p in (
            "numpy", "scipy", "PyWavelets", "Pillow", "matplotlib", "PyYAML")}}


def decoded_color_levels(pixels, cfg, approximate=False):
    """Recover scalar color order from actual RGB, not from a second CWT resize."""
    palette = colormaps[cfg["image"]["colormap"]](np.arange(256) / 255, bytes=True)[:, :3]
    colors, inverse = np.unique(np.asarray(pixels)[..., :3].reshape(-1, 3), axis=0, return_inverse=True)
    distance = ((colors[:, None].astype(float) - palette[None].astype(float)) ** 2).sum(axis=2)
    indices = distance.argmin(axis=1)
    require(approximate or np.all(distance[np.arange(len(colors)), indices] == 0),
            "Model pixels outside the configured palette")
    return indices[inverse].reshape(pixels.shape[:2])


def check_model_direction(pixels, cfg):
    height, width = cfg["image"]["height"], cfg["image"]["width"]
    require(pixels.shape == (height, width, 3), "Wrong synthetic model dimensions")
    levels = decoded_color_levels(pixels, cfg)
    rows = []
    for col in (slice(width // 8, 3 * width // 8), slice(5 * width // 8, 7 * width // 8)):
        score = levels[:, col].mean(axis=1)
        rows.append(float(np.flatnonzero(score == score.max()).mean()))
    freq = frequencies(cfg)
    low, high = freq[0] / np.sqrt(freq[1] / freq[0]), freq[-1] * np.sqrt(freq[1] / freq[0])
    expected = [np.log(high / tone) / np.log(high / low) * height - .5 for tone in (300, 1500)]
    require(all(abs(a - b) < 6 for a, b in zip(rows, expected)) and rows[1] < rows[0],
            "Actual model RGB time/frequency direction failed")
    return rows, list(map(float, expected))


def check_review_coordinates(path, cfg):
    with Image.open(path) as im:
        require(im.size == (640, 420), "Wrong synthetic review dimensions")
        pixels = np.asarray(im.convert("RGB"))
    # Independent analytic coordinates for the documented fixed axes rectangle.
    top, height, left, width = .04 * 420, .82 * 420, .13 * 640, .68 * 640
    freq = frequencies(cfg)
    low, high = freq[0] / np.sqrt(freq[1] / freq[0]), freq[-1] * np.sqrt(freq[1] / freq[0])
    measured = []
    for fraction in (.25, .75):
        x = round(left + width * fraction)
        y0, y1 = int(np.ceil(top)) + 3, int(np.floor(top + height)) - 3
        levels = decoded_color_levels(pixels[y0:y1, x - 8:x + 8], cfg, approximate=True)
        score = levels.mean(axis=1)
        y = float(np.flatnonzero(score == score.max()).mean()) + y0 + .5
        measured.append(float(np.exp(np.log(high) - (y - top) / height * np.log(high / low))))
    require(all(abs(value - tone) / tone < .1 for value, tone in zip(measured, (300, 1500))),
            "Actual review PNG time/log-frequency coordinates failed")
    return measured


def synthetic_checks(cfg, output=None):
    n = cfg["segmentation"]["window_samples"]
    fs = cfg["sampling"]["target_fs_hz"]
    t = np.arange(n) / fs
    signal = np.where(np.arange(n) < n // 2, np.sin(2*np.pi*300*t), np.sin(2*np.pi*1500*t))
    matrix = log_power(signal, cfg)
    freq = frequencies(cfg)
    cols = [slice(n//8, 3*n//8), slice(5*n//8, 7*n//8)]
    peaks = [float(freq[np.argmax(np.mean(m[:, c], axis=1))]) for m, c in [(matrix, col) for col in cols]]
    require(all(abs(p - target) / target < .06 for p, target in zip(peaks, [300, 1500])), "Tone location failed")
    limits = {"vmin": float(matrix.min()), "vmax": float(matrix.max())}
    pixels = model_pixels(matrix, limits, cfg)
    rows, expected = check_model_direction(pixels, cfg)
    require(np.array_equal(matrix, log_power(signal, cfg)), "CWT is not deterministic")
    with tempfile.TemporaryDirectory(prefix="aae5103-synthetic-") as temporary:
        destination = Path(output) if output is not None else Path(temporary)
        model_path, review_path = destination / "synthetic_model.png", destination / "synthetic_review.png"
        render(matrix, limits, cfg, model_path, review_path)
        with Image.open(model_path) as im:
            require(im.mode == "RGB", "Wrong saved synthetic model mode")
            saved_pixels = np.asarray(im)
            require(np.array_equal(saved_pixels, pixels), "Saved synthetic model pixels differ")
            check_model_direction(saved_pixels, cfg)
        review_peaks = check_review_coordinates(review_path, cfg)
    return {"status": "PASS", "tones_hz": [300,1500], "peaks_hz": peaks,
        "model_peak_rows_left_right": rows, "expected_rows": expected,
        "model_direction_checked_on": "actual returned and decoded saved RGB pixels",
        "review_measured_peaks_hz": review_peaks,
        "frequency_order": "highest_at_top", "time_order": "increasing_to_right",
        "matrix_shape_after_reflect_and_crop": list(matrix.shape), "repeat_array_equal": True,
        "log_power_sha256": array_hash(matrix), "pixel_sha256": digest(pixels.tobytes())}


def pilot():
    cfg, rows = load_inputs()
    require(not (RUN / "pilot.json").exists(), "Pilot already exists")
    preflight = read_json(RUN / "d01_preflight.json")
    require(preflight["status"] == "PASS", "D01 preflight required")
    require(verify_snapshot(ROOT, "runs/d01_fix/deliverables.sha256")["status"] == "PASS", "D01 changed")
    out = RUN / "pilot"
    out.mkdir(exist_ok=False)
    report = {"scope": "four train windows only; provisional preview limits are not final fit",
              "created_at": now(), "input_hashes": input_hashes(), "synthetic": synthetic_checks(cfg, out), "windows": []}
    montage = Image.new("RGB", (448,448))
    for i, rid in enumerate(("cwru_097", "cwru_105", "cwru_130", "cwru_118")):
        row = next(r for r in rows if r["record_id"] == rid)
        require(row["split"] == "train", "Pilot must be train only")
        values = load_record(row, cfg)
        valid, _ = eligible_windows(values, cfg)
        window = valid[0]
        chunk = values[window["start_sample"]:window["end_sample"]]
        matrix = log_power(chunk, cfg)
        limits = dict(zip(("vmin", "vmax"), map(float, np.quantile(matrix, [.01,.99]))))
        p = out / f"{rid}_model.png"
        pixel_hash = render(matrix, limits, cfg, p, out / f"{rid}_review.png")
        repeat = log_power(chunk, cfg)
        require(np.array_equal(matrix, repeat) and digest(model_pixels(repeat, limits, cfg).tobytes()) == pixel_hash,
                "Pilot repeated array/pixels differ")
        montage.paste(Image.open(p), ((i % 2)*224,(i//2)*224))
        report["windows"].append({"record_id": rid, **window, "split": "train",
            "matrix_sha256": array_hash(matrix), "pixel_sha256": pixel_hash,
            "repeat_array_equal": True, "repeat_pixels_equal": True, "preview_limits": limits})
    montage.save(out / "train_montage.png")
    write_json(RUN / "pilot.json", {**report, "status": "PASS"})
    print("Pilot PASS; inspect only train pilot images before proceeding.", flush=True)


def train(compatibility=None):
    cfg, rows = load_inputs()
    pilot_report = read_json(RUN / "pilot.json")
    require(pilot_report["status"] == "PASS", "Pilot differs")
    check_processing_inputs(pilot_report["input_hashes"], compatibility)
    approval = read_json(RUN / "pilot_review.json")
    require(approval["status"] == "PASS" and approval["scope"] == "train_only", "Training visual review required")
    require(not (RUN / "training_freeze.json").exists(), "Training version already frozen")
    display_check = synthetic_checks(cfg)
    plans, signals = [], {}
    for row in rows:
        if row["split"] != "train":
            continue
        values = load_record(row, cfg)
        valid, rejected = eligible_windows(values, cfg)
        plans.append({"record_id": row["record_id"], "split": "train", "processed_length": len(values),
                      "valid_windows": valid, "rejected_windows": rejected})
        signals[row["record_id"]] = values
    cap = choose_balanced(plans)
    write_json(RUN / "train_selection.json", {"train_windows_per_record": cap, "records": plans})
    matrices, sample_rows = [], []
    matrix_dir = RUN / "train_matrices"
    matrix_dir.mkdir(exist_ok=False)
    for plan in plans:
        row = next(r for r in rows if r["record_id"] == plan["record_id"])
        for window in plan["selected_windows"]:
            chunk = signals[row["record_id"]][window["start_sample"]:window["end_sample"]]
            matrix = log_power(chunk, cfg)
            sample = metadata(row, window, chunk, matrix, "")
            np.save(matrix_dir / (sample["sample_id"] + ".npy"), matrix, allow_pickle=False)
            sample_rows.append(sample)
            matrices.append(matrix)
        print(f"train CWT {row['record_id']}: {cap} windows", flush=True)
    limits = fit_color_limits(matrices, sample_rows, cfg)
    color_path = ROOT / "configs/cwt_color_limits.json"
    write_json(color_path, limits)
    colors_hash = file_hash(color_path)
    for i, (sample, matrix) in enumerate(zip(sample_rows, matrices)):
        sample["color_limits_sha256"] = colors_hash
        sample["model_pixels_sha256"] = render(matrix, limits, cfg,
            ROOT / sample["model_image_path"], ROOT / sample["review_image_path"])
        if (i+1) % 29 == 0:
            print(f"train images {i+1}/{len(sample_rows)}", flush=True)
    write_csv(RUN / "train_samples.csv", sample_rows)
    np.save(RUN / "frequencies_hz.npy", frequencies(cfg), allow_pickle=False)
    files = [color_path, RUN / "train_samples.csv", RUN / "train_selection.json", RUN / "frequencies_hz.npy",
             RUN / "pilot.json", RUN / "pilot_review.json"]
    files += list(matrix_dir.glob("*.npy"))
    files += [ROOT / s[k] for s in sample_rows for k in ("model_image_path", "review_image_path")]
    write_json(RUN / "training_freeze.json", {"status": "PASS", "frozen_at": now(),
        "preprocess_version": VERSION, "numeric_configuration_changed": False,
        "input_hashes": input_hashes(), "environment": environment(),
        "fit_split": "train", "training_windows": len(sample_rows), "train_windows_per_record": cap,
        "heldout_processed_at_freeze": False, "color_limits_sha256": colors_hash,
        "synthetic_display_check": display_check,
        "pilot_compatibility_sha256": file_hash(ROOT / compatibility) if compatibility else None,
        "files": {relative(p): file_hash(p) for p in sorted(files)}})
    print("Training freeze PASS; held-out rendering is now allowed.", flush=True)


def check_freeze(compatibility=None):
    frozen = read_json(RUN / "training_freeze.json")
    require(frozen["status"] == "PASS", "Frozen processing inputs changed")
    approved = check_processing_inputs(frozen["input_hashes"], compatibility)
    if approved:
        require(file_hash(RUN / "training_freeze.json") == approved["training_freeze_sha256"],
                "Historical training freeze changed")
    for name, expected in frozen["files"].items():
        require(file_hash(ROOT / name) == expected, f"Frozen training artifact changed: {name}")
    return frozen


def heldout():
    cfg, records = load_inputs()
    frozen = check_freeze()
    require(not (ROOT / "data/samples.csv").exists(), "Samples manifest already exists")
    limits = read_json(ROOT / "configs/cwt_color_limits.json")
    samples = read_csv(RUN / "train_samples.csv")
    plans = []
    started = now()
    for row in records:
        if row["split"] == "train":
            continue
        values = load_record(row, cfg)
        valid, rejected = eligible_windows(values, cfg)
        plans.append({"record_id": row["record_id"], "split": row["split"], "processed_length": len(values),
                      "selected_windows": valid, "rejected_windows": rejected})
        for window in valid:
            chunk = values[window["start_sample"]:window["end_sample"]]
            matrix = log_power(chunk, cfg)
            sample = metadata(row, window, chunk, matrix, frozen["color_limits_sha256"])
            sample["model_pixels_sha256"] = render(matrix, limits, cfg,
                ROOT / sample["model_image_path"], ROOT / sample["review_image_path"])
            samples.append(sample)
        print(f"mechanical render {row['record_id']} ({row['split']}): {len(valid)}", flush=True)
    require(file_hash(ROOT / "configs/cwt_color_limits.json") == frozen["color_limits_sha256"], "Colors changed")
    write_csv(ROOT / "data/samples.csv", samples)
    write_json(RUN / "heldout_generation.json", {"status": "PASS", "started_at": started, "finished_at": now(),
        "training_freeze_sha256": file_hash(RUN / "training_freeze.json"),
        "color_limits_sha256_before_and_after": frozen["color_limits_sha256"],
        "color_fit_calls": 0, "visual_tuning": False, "classification_scores_computed": False,
        "records": plans, "sample_counts": dict(Counter(s["split"] for s in samples))})


def validate_sample_population(samples, records, plans):
    by_record = {r["record_id"]: r for r in records}
    require(len(by_record) == len(records), "Repeated source records")
    require(all(s["record_id"] in by_record for s in samples), "Unknown sample record")
    require(len({s["sample_id"] for s in samples}) == len(samples), "Repeated sample IDs")
    require(len(plans) == len(by_record) and {p["record_id"] for p in plans} == set(by_record),
            "Selection plan record set differs")
    expected = []
    for plan in plans:
        row = by_record[plan["record_id"]]
        for w in plan["selected_windows"]:
            expected.append((sample_id(row, w), row["record_id"], w["start_sample"],
                             w["end_sample"], w["eligible_window_index"]))
    actual = [(s["sample_id"], s["record_id"], int(s["start_sample"]),
               int(s["end_sample"]), int(s["eligible_window_index"])) for s in samples]
    require(len(expected) == len(set(expected)) == len(actual) and set(actual) == set(expected),
            "Sample population differs from complete expected selection")
    return len(expected)


def compare_png_pixels(expected_path, actual_path, description):
    with Image.open(expected_path) as expected, Image.open(actual_path) as actual:
        expected.load()
        actual.load()
        require(expected.format == actual.format == "PNG" and expected.size == actual.size
                and expected.mode == actual.mode and np.array_equal(np.asarray(expected), np.asarray(actual)),
                f"{description}: decoded PNG pixels differ")


def verify_sample_images(sample, matrix, colors, cfg):
    """Rebuild both views from the verified raw-derived matrix; never trust a self-reported hash alone."""
    with tempfile.TemporaryDirectory(prefix="aae5103-d02-verify-") as temporary:
        temporary = Path(temporary)
        model_path, review_path = temporary / "model.png", temporary / "review.png"
        pixels_hash = render(matrix, colors, cfg, model_path, review_path)
        require(pixels_hash == sample["model_pixels_sha256"], "Repeated model pixels differ")
        compare_png_pixels(model_path, ROOT / sample["model_image_path"], "Model image")
        compare_png_pixels(review_path, ROOT / sample["review_image_path"], "Review image")


def validate(output=None, compatibility=None):
    output = ROOT / output if output is not None else RUN / "acceptance.json"
    require(not output.exists(), "Acceptance output already exists; choose a new output path")
    cfg, records = load_inputs()
    frozen = check_freeze(compatibility)
    samples = read_csv(ROOT / "data/samples.csv")
    colors = read_json(ROOT / "configs/cwt_color_limits.json")
    require(np.array_equal(np.load(RUN / "frequencies_hz.npy"), frequencies(cfg)), "Frequency vector mismatch")
    train_plans = read_json(RUN / "train_selection.json")["records"]
    held = read_json(RUN / "heldout_generation.json")
    require(held["color_fit_calls"] == 0 and not held["visual_tuning"], "Heldout was used for fitting/tuning")
    require(held["started_at"] >= frozen["frozen_at"], "Heldout preceded freeze")
    expected_count = validate_sample_population(samples, records, train_plans + held["records"])
    plan_by_id = {p["record_id"]: p for p in train_plans + held["records"]}
    regenerated, groups, counts = [], {}, Counter()
    for row in records:
        rid = row["record_id"]
        values = load_record(row, cfg)
        valid, rejected = eligible_windows(values, cfg)
        plan = plan_by_id[rid]
        require(len(values) == plan["processed_length"] and rejected == plan["rejected_windows"], "Quality log mismatch")
        selected = plan["selected_windows"]
        if row["split"] == "train":
            require(valid == plan["valid_windows"], "Training eligible population changed")
            cap = frozen["train_windows_per_record"]
            expected = [valid[i] for i in np.floor(np.linspace(0,len(valid)-1,cap)).astype(int)]
        else:
            expected = valid
        require(selected == expected, "Selection differs from protocol")
        actual = [s for s in samples if s["record_id"] == rid]
        require(len(actual) == len(selected), "Manifest count mismatch")
        previous_end = -1
        for sample, window in zip(actual, selected):
            require(sample["sample_id"] == sample_id(row,window), "Unstable sample identity")
            for key in ("group_id", "source_family_id", "split", "true_label", "load_hp", "channel"):
                require(sample[key] == row[key], f"Metadata mismatch: {key}")
            for key in ("start_sample", "end_sample", "eligible_window_index"):
                require(int(sample[key]) == window[key], "Index mismatch")
            start, end = window["start_sample"], window["end_sample"]
            require(start >= previous_end and end <= len(values)-cfg["sampling"]["discard_edge_samples_each_side"], "Overlap or out of bounds")
            previous_end = end
            require(sample["color_limits_sha256"] == frozen["color_limits_sha256"], "Per-sample colors differ")
            require(sample["preprocess_version"] == VERSION and int(sample["processed_fs_hz"]) == 12000, "Version/fs mismatch")
            require(sample["raw_sha256"] == row["sha256"] and sample["signal_sha256_float64"] == row["signal_sha256_float64"], "Provenance mismatch")
            require(sample["window_sha256_float64"] == array_hash(values[start:end]), "Window fingerprint changed")
            groups.setdefault(sample["group_id"],set()).add(sample["split"])
            for key in ("model_image_path", "review_image_path"):
                kind = "model" if key == "model_image_path" else "review"
                require(sample[key] == f"data/processed/{VERSION}/{kind}/{sample['sample_id']}.png",
                        "Image path differs from sample identity")
                with Image.open(ROOT / sample[key]) as im:
                    im.load()
                    require(im.format == "PNG", "Not PNG")
                    if key == "model_image_path":
                        require(im.size == (224,224) and im.mode == "RGB", "Wrong model image format")
                        require(digest(im.tobytes()) == sample["model_pixels_sha256"], "Image pixel hash changed")
                    else:
                        require(im.size == (640,420), "Review image dimensions changed")
            matrix = log_power(values[start:end], cfg)
            require(array_hash(matrix) == sample["log_power_sha256_float64"], "Repeated numeric CWT differs")
            verify_sample_images(sample, matrix, colors, cfg)
            regenerated.append(sample["sample_id"])
            counts[(sample["split"],sample["true_label"])] += 1
        print(f"acceptance {rid}: {len(actual)} windows, all matrices/model/review pixels PASS", flush=True)
    require(len(regenerated) == len(samples) == expected_count == sum(counts.values()),
            "Not every manifest row was verified and counted")
    require(all(len(s) == 1 for s in groups.values()), "Source group crosses splits")
    training = [s for s in samples if s["split"] == "train"]
    require(colors["sample_ids"] == [s["sample_id"] for s in training], "Color fit membership differs")
    matrices = []
    for sample in training:
        matrix = np.load(RUN / "train_matrices" / (sample["sample_id"]+".npy"),allow_pickle=False)
        require(array_hash(matrix) == sample["log_power_sha256_float64"] == colors["matrix_hashes"][sample["sample_id"]], "Fit matrix changed")
        require(digest(model_pixels(matrix,colors,cfg).tobytes()) == sample["model_pixels_sha256"], "Training model image differs from fit matrix")
        matrices.append(matrix)
    refit = fit_color_limits(matrices,training,cfg)  # acceptance-only train refit; never saves/replaces colors
    require((refit["vmin"],refit["vmax"]) == (colors["vmin"],colors["vmax"]), "Exact quantile repeat differs")
    summary = {split: {label: counts[(split,label)] for label in cfg["labels"]} for split in ("train","validation","test")}
    report = {"status":"PASS", "checked_at":now(), "samples":len(samples),
        "expected_samples":expected_count, "verified_samples":len(regenerated), "counted_samples":sum(counts.values()),
        "per_split_class_counts":summary, "unique_source_groups":len(groups), "source_groups_crossing_splits":0,
        "all_windows_provenance_bounds_quality_and_nonoverlap":"PASS", "all_image_formats":"PASS",
        "all_model_decoded_pixels_from_raw":"PASS", "all_review_decoded_pixels_from_raw":"PASS",
        "train_fit_matrix_and_model_correspondence":"PASS", "exact_train_quantile_repeat":"PASS",
        "repeated_raw_to_matrix_and_both_image_pixels":regenerated, "repeated_windows":len(regenerated),
        "synthetic_display_check":synthetic_checks(cfg), "fit_split":"train", "heldout_refit_calls":0,
        "colors_sha256":frozen["color_limits_sha256"], "environment":environment(),
        "heldout_visual_tuning":False, "classification_training_or_scoring":False,
        "compatibility_sha256":file_hash(ROOT / compatibility) if compatibility else None,
        "reproducibility_scope":"all numeric arrays and decoded model/review pixels in the same environment; no cross-version file-byte promise"}
    write_json(output, report)
    print(json.dumps({"status":"PASS","samples":len(samples),"per_split_class_counts":summary}),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot","train","heldout","validate"))
    parser.add_argument("--output", type=Path, help="New acceptance report path (validate only)")
    parser.add_argument("--compatibility", type=Path, help="Explicit audited validation-only revision (validate/train)")
    args = parser.parse_args()
    require(args.output is None or args.stage == "validate", "--output is validate-only")
    require(args.compatibility is None or args.stage in ("train", "validate"), "Compatibility is train/validate-only")
    if args.stage == "validate":
        validate(args.output, args.compatibility)
    elif args.stage == "train":
        train(args.compatibility)
    else:
        {"pilot":pilot,"heldout":heldout}[args.stage]()


if __name__ == "__main__":
    main()
