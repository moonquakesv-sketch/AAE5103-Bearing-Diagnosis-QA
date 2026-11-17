"""Rebuild D02 in a NEW directory using immutable source data and the accepted pilot.

The original project is read-only. The color fit is repeated from training only,
then all images are mechanically regenerated. Compare arrays/pixels, not timestamps.
"""
import argparse
from pathlib import Path
import shutil
import d02_pipeline as p
from data_contract import require


def compare_outputs(source, target):
    expected = p.read_csv(source / "data/samples.csv")
    actual = p.read_csv(target / "data/samples.csv")
    comparison_fields = [f for f in p.FIELDS if f != "color_limits_sha256"]
    require(len(expected) == len(actual), "Sample count changed")
    require(all(all(a[f] == b[f] for f in comparison_fields) for a,b in zip(expected,actual)),
            "Sample indices, CWT matrices or model pixels changed")
    # Compare decoded source files themselves, including heldout review images.
    for row, repeated in zip(expected, actual):
        for key in ("model_image_path", "review_image_path"):
            p.compare_png_pixels(source / row[key], target / repeated[key],
                                 f"Source/reproduction {key}: {row['sample_id']}")
    expected_colors = p.read_json(source / "configs/cwt_color_limits.json")
    colors = p.read_json(target / "configs/cwt_color_limits.json")
    for key in ("vmin","vmax","sample_ids","matrix_hashes","pixel_count"):
        require(expected_colors[key] == colors[key], f"Repeated color fit differs: {key}")
    return {"status":"PASS", "checked_at":p.now(), "samples":len(actual),
        "comparison_fields":comparison_fields, "exact_color_quantiles_equal":True,
        "numeric_matrices_and_model_pixels_equal":True,
        "source_model_and_review_decoded_pixels_equal":True,
        "compared_model_images":len(actual), "compared_review_images":len(actual),
        "timestamps_and_color_json_file_hashes_may_differ":True,
        "png_file_bytes_compared":False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="A directory that does not exist")
    parser.add_argument("--compatibility", type=Path, help="Audited validation-only revision for historical inputs")
    args = parser.parse_args(argv)
    source, source_run = p.ROOT, p.RUN
    original_freeze = p.check_freeze(args.compatibility)
    target = args.output_root.resolve()
    require(not target.exists(), "Reproduction output already exists")
    require(not target.is_relative_to(source), "Use a separate directory outside the source project")
    target.mkdir(parents=True)
    copied = list(p.input_hashes()) + ["scripts/reproduce_d02.py"]
    compatibility = None
    if args.compatibility:
        compatibility = (source / args.compatibility).resolve().relative_to(source)
        copied += [str(compatibility), p.read_json(source / compatibility)["prior_pipeline_path"]]
    for name in copied:
        dest = target / name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source / name, dest)
    (target / "data/raw").symlink_to(source / "data/raw",target_is_directory=True)
    target_run = target / "runs/d02" / p.VERSION
    target_run.mkdir(parents=True)
    for name in ("pilot.json","pilot_review.json","d01_preflight.json"):
        shutil.copy2(source_run / name, target_run / name)
    shutil.copytree(source_run / "pilot",target_run / "pilot")
    p.ROOT, p.RUN = target, target_run
    p.write_json(target_run / "reproduction_provenance.json", {
        "created_at":p.now(), "scope":"reuse accepted train-only pilot; repeat train fit and all rendering",
        "original_freeze_sha256":p.file_hash(source_run / "training_freeze.json"),
        "accepted_pilot_sha256":p.file_hash(source_run / "pilot.json"),
        "validation_revision_compatibility_sha256":p.file_hash(source / compatibility) if compatibility else None,
        "raw_data":"symlink to original read-only input; no download or split change",
        "source_environment":original_freeze["environment"]})
    p.train(compatibility)
    p.heldout()
    p.validate()
    p.write_json(target_run / "comparison.json", compare_outputs(source, target))
    print(f"Full reproduction PASS: {target_run / 'comparison.json'}",flush=True)


if __name__ == "__main__":
    main()
