"""Offline D01 provenance/config validation with immutable, versioned reports.

No sample generation, classifier fitting or held-out scoring. Snapshot comparison
and content checks are reported separately. Existing reports are never overwritten.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

import numpy as np
from scipy.io import loadmat
import pywt
import yaml

from data_contract import (
    ContractError, processed_length, require, resample_parameters,
    resample_signal, validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = "runs/d01_fix/deliverables.sha256"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_snapshot(root, baseline):
    entries = {}
    for line in (root / baseline).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = PurePosixPath(name)
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "Invalid snapshot hash")
        require(not path.is_absolute() and ".." not in path.parts, f"Invalid snapshot path: {name}")
        require(name not in entries, f"Duplicate snapshot path: {name}")
        entries[name] = expected
    require(bool(entries), "Empty snapshot")
    return entries


def verify_snapshot(root, baseline):
    entries = read_snapshot(root, baseline)
    differences = []
    for name, expected in entries.items():
        path = root / name
        actual = digest(path.read_bytes()) if path.is_file() else None
        if actual != expected:
            differences.append({"path": name, "expected": expected, "actual": actual})
    return {"status": "PASS" if not differences else "FAIL", "baseline": str(baseline),
            "baseline_sha256": digest((root / baseline).read_bytes()),
            "checked_files": len(entries), "differences": differences}


def validate_record_metadata(rows, cfg, audit):
    """Check per-record semantics before reading any vibration data."""
    validate_config(cfg)
    require(len(rows) == 16 and len({r["record_id"] for r in rows}) == 16,
            "Expected 16 distinct records")
    by_id = {r["record_id"]: r for r in audit["records"]}
    require(len(by_id) == len(audit["records"]) == 16, "Invalid source audit record set")
    require({r["record_id"] for r in rows} == set(by_id), "Manifest/audit record sets differ")
    selection = cfg["selection"]
    require(selection["fault_bearing"] == "drive_end" and selection["sensor"] == "DE",
            "D01 selection requires drive-end bearings and DE sensors")
    for row in rows:
        rid = row["record_id"]
        source = by_id[rid]
        for field in ("true_label", "official_name", "source_url", "raw_path", "channel", "sha256"):
            require(row[field] == source[field], f"{rid}: {field} differs from source audit")
        require(row["file_name"] == source["file"], f"{rid}: file_name differs from source audit")
        require(int(row["load_hp"]) == source["load_hp"], f"{rid}: load_hp differs from source audit")
        require(int(row["n_samples"]) == source["n_samples"], f"{rid}: n_samples differs from source audit")
        number = int(Path(row["file_name"]).stem)
        require(row["channel"] == f"X{number:03d}_DE_time", f"{rid}: wrong exact DE channel")
        require(row["sensor_location"] == "drive_end_housing", f"{rid}: wrong sensor location")
        label = row["true_label"]
        normal = label == "normal"
        page = "normal-baseline-data" if normal else "12k-drive-end-bearing-fault-data"
        require(row["metadata_url"] == f"https://engineering.case.edu/bearingdatacenter/{page}",
                f"{rid}: wrong metadata URL")
        require(row["fault_bearing"] == ("none" if normal else "drive_end"), f"{rid}: wrong fault bearing")
        require(row["fault_size_unit"] == "inch_diameter", f"{rid}: wrong fault size unit")
        expected_size = 0 if normal else selection["fault_diameter_in"]
        require(float(row["fault_size"]) == expected_size, f"{rid}: fault size differs from selection")
        expected_position = selection["outer_race_clock"] if label == "outer_race" else "not_applicable"
        require(row["fault_position"] == expected_position, f"{rid}: fault position differs from selection")
        prefix = {"normal": "Normal", "inner_race": "IR007", "outer_race": "OR007@6",
                  "rolling_element": "B007"}[label]
        require(row["official_name"] == f"{prefix}_{source['load_hp']}", f"{rid}: inconsistent official name")
        require(normal or selection["fault_diameter_in"] == .007, "D01 audit covers 0.007 inch only")
        require(label != "outer_race" or selection["outer_race_clock"] == "6:00",
                "D01 audit covers outer-race 6:00 only")
        kind = "normal" if normal else "fault"
        resample_parameters(cfg, int(row["original_fs_hz"]), kind)
        require(row["split"] in ("train", "validation", "test"), f"{rid}: unknown split")
        require(int(row["load_hp"]) in cfg["split"][f"{row['split']}_load_hp"], f"{rid}: wrong load split")
        require(row["group_id"] == rid, f"{rid}: group_id differs from selected record")
    return by_id


def validate_content(root):
    with (root / "data/records.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cfg = yaml.safe_load((root / "configs/data.yaml").read_text(encoding="utf-8"))
    audit = json.loads((root / "runs/d01/source_audit.json").read_text(encoding="utf-8"))
    by_id = validate_record_metadata(rows, cfg, audit)
    groups, hashes, families = defaultdict(set), defaultdict(set), defaultdict(set)
    count, windows, rpm_differences = Counter(), {}, []
    for row in rows:
        rid = row["record_id"]
        raw = root / row["raw_path"]
        data = loadmat(raw)
        values = data[row["channel"]].ravel()
        require(digest(raw.read_bytes()) == row["sha256"], f"{rid}: raw file hash mismatch")
        require(len(values) == int(row["n_samples"]) and np.isfinite(values).all()
                and np.ptp(values) > 0, f"{rid}: invalid signal")
        signal_hash = digest(np.asarray(values, dtype="<f8").tobytes())
        require(signal_hash == row["signal_sha256_float64"]
                == by_id[rid]["arrays"][row["channel"]]["sha256_float64"], f"{rid}: signal hash mismatch")
        groups[row["group_id"]].add(row["split"])
        hashes[signal_hash].add(row["split"])
        families[row["source_family_id"]].add(row["split"])
        count[(row["split"], row["true_label"])] += 1
        if row["rpm_mat"] and float(row["rpm_mat"]) != float(row["rpm_table_approx"]):
            rpm_differences.append({"record_id": rid, "table": float(row["rpm_table_approx"]),
                                    "MAT": float(row["rpm_mat"])})
        kind = "normal" if row["true_label"] == "normal" else "fault"
        parameters = resample_parameters(cfg, int(row["original_fs_hz"]), kind)
        n = processed_length(len(values), parameters)
        edge = cfg["sampling"]["discard_edge_samples_each_side"]
        length, stride = cfg["segmentation"]["window_samples"], cfg["segmentation"]["stride_samples"]
        nwin = max(0, 1 + (n - 2 * edge - length) // stride)
        require(nwin > 0, f"{rid}: no complete windows")
        windows[rid] = {"split": row["split"], "eligible_windows_estimate": nwin,
                        "processed_length_estimate": n}
    require(all(len(s) == 1 for s in groups.values()), "Source group spans splits")
    require(all(len(s) == 1 for s in hashes.values()) and len(hashes) == 16,
            "Selected signals are duplicated")
    for split, n in (("train", 2), ("validation", 1), ("test", 1)):
        require(all(count[(split, label)] == n for label in cfg["labels"]), f"{split}: incorrect class counts")
    x98 = loadmat(root / "data/raw/cwru/98.mat")
    x99 = loadmat(root / "data/raw/cwru/99.mat")
    for channel in ("X098_DE_time", "X098_FE_time"):
        require(np.array_equal(x98[channel], x99[channel]), f"Expected duplicate {channel} changed")
    row99 = next(r for r in rows if r["record_id"] == "cwru_099")
    require(row99["channel"] == "X099_DE_time" and "X098_DE_time" in row99["ignored_time_keys"],
            "99.mat must exclude the X098 duplicate")

    # Exercise the same parsed resampling parameters future D02 processing will use.
    fs = cfg["sampling"]["target_fs_hz"]
    original_fs = cfg["sampling"]["normal_original_fs_hz"]
    t = np.arange(original_fs, dtype=float) / original_fs
    low = resample_signal(np.sin(2 * np.pi * 1000 * t), cfg, original_fs, "normal")
    high = resample_signal(np.sin(2 * np.pi * 8000 * t), cfg, original_fs, "normal")
    region = slice(edge, len(low) - edge)
    rms = lambda x: float(np.sqrt(np.mean(x * x)))
    attenuation = 20 * np.log10(rms(high[region]) / rms(low[region]))
    require(attenuation < -40 and abs(rms(low[region]) - 2 ** -.5) < .01,
            "Synthetic resampling pass/stop-band check failed")
    cwt = cfg["cwt"]
    freqs = np.geomspace(cwt["frequency_min_hz"], cwt["frequency_max_hz"], cwt["n_frequencies"])
    scales = pywt.frequency2scale(cwt["wavelet"], freqs / fs)
    require(scales.min() >= 4, "CWT scale too small for this pilot")
    signal = np.sin(2 * np.pi * 1000 * np.arange(length) / fs)
    pad = cwt["pad_samples_each_side"]
    signal = np.pad(signal - signal.mean(), pad, mode=cwt["pad_mode"])
    require("precision" in inspect.signature(pywt.cwt).parameters, "PyWavelets lacks precision support")
    coefs, actual_freqs = pywt.cwt(signal, scales, cwt["wavelet"],
        sampling_period=cwt["sampling_period_seconds"], method=cwt["method"], precision=cwt["precision"])
    coefs = coefs[:, pad:pad + length]
    require(coefs.shape == (cwt["n_frequencies"], length)
            and np.isfinite(coefs).all() and np.allclose(actual_freqs, freqs), "CWT shape/frequencies invalid")
    peak = float(freqs[np.argmax(np.mean(np.abs(coefs[:, length // 8:-length // 8]) ** 2, axis=1))])
    require(abs(peak - 1000) / 1000 < .06, "Synthetic CWT frequency check failed")
    train_cap = min(w["eligible_windows_estimate"] for w in windows.values() if w["split"] == "train")
    for window in windows.values():
        n = window["eligible_windows_estimate"]
        window["planned_windows_after_train_balance"] = train_cap if window["split"] == "train" else n
        if window["split"] == "train":
            ids = np.floor(np.linspace(0, n - 1, train_cap)).astype(int)
            require(len(set(ids)) == train_cap and ids[0] >= 0 and ids[-1] < n, "Invalid balancing indices")
    pip = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    require(pip.returncode == 0, f"Dependency check failed: {pip.stdout} {pip.stderr}")
    return {"status": "PASS", "records": len(rows), "unique_selected_signal_hashes": len(hashes),
        "group_split_overlap": False,
        "per_class_record_counts": {s: {label: count[(s, label)] for label in cfg["labels"]}
                                   for s in ("train", "validation", "test")},
        "source_families_spanning_splits_not_confirmed_specimens": {f: sorted(s) for f, s in families.items()},
        "excluded_duplicate_arrays": audit["duplicate_arrays"], "rpm_table_vs_mat_differences": rpm_differences,
        "window_counts_are_predictions_not_generated_samples": windows,
        "planned_total_windows": sum(w["planned_windows_after_train_balance"] for w in windows.values()),
        "train_windows_per_record": train_cap,
        "synthetic_checks": {"anti_alias_8khz_attenuation_db": float(attenuation), "cwt_shape": list(coefs.shape),
                             "cwt_1khz_peak_hz": peak, "scale_min": float(scales.min()), "scale_max": float(scales.max())},
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "pip_check": pip.stdout.strip()}}


def output_path(root, output=None):
    if output is None:
        stamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%dT%H%M%S_%f")
        output = f"runs/d01_checks/{stamp}_{uuid.uuid4().hex[:8]}/validation.json"
    path = (root / output).resolve()
    require(not path.is_relative_to((root / "runs/d01").resolve()), "Historical runs/d01 output is protected")
    require(not path.exists(), f"Output already exists; refusing to overwrite: {path}")
    return path


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def execute(root, baseline=DEFAULT_BASELINE, output=None):
    path = output_path(root, output)
    report = {"status": "FAIL", "checked_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
              "scope": "D01 metadata and synthetic configuration checks; no samples or model evaluation",
              "snapshot_check": {"status": "NOT_RUN"}, "content_check": {"status": "NOT_RUN"}}
    try:
        report["snapshot_check"] = verify_snapshot(root, baseline)
        if report["snapshot_check"]["status"] == "PASS":
            try:
                report["content_check"] = validate_content(root)
            except (ContractError, KeyError, ValueError, OSError) as exc:
                report["content_check"] = {"status": "FAIL", "error": str(exc)}
            if report["content_check"]["status"] == "PASS":
                report["status"] = "PASS"
    except (ContractError, ValueError, OSError) as exc:
        report["snapshot_check"] = {"status": "FAIL", "error": str(exc)}
    write_report(path, report)
    return path, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="Frozen snapshot; paths relative to project root")
    parser.add_argument("--output", help="New report path; defaults to a unique runs/d01_checks subdirectory")
    args = parser.parse_args(argv)
    try:
        path, report = execute(ROOT, args.baseline, args.output)
    except (ContractError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "output": str(path),
                      "snapshot_check": report["snapshot_check"],
                      "content_check": {k: v for k, v in report["content_check"].items()
                                        if k in ("status", "error", "records", "planned_total_windows", "synthetic_checks")}},
                     ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
