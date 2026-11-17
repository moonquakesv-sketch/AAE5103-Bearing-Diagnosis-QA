"""Regression coverage for the two independently reproduced D03 guard failures."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import d03_pipeline as p


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="d03-preflight-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "new-run"
        self.run.mkdir()
        self.baseline = "runs/new_revision/deliverables.sha256"
        required = set(p.INPUTS[:-1] + [p.CONFIG, p.PROTOCOL, p.CODE, "scripts/d02_pipeline.py",
                                       "scripts/data_contract.py", "scripts/validate_d01.py", "scripts/verify_d03.py"])
        for name in required:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(name)
        self.manifest = self.root / self.baseline
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text("".join(f"{p.sha(self.root / name)}  {name}\n" for name in sorted(required)))
        self.report = {"status": "PASS", "snapshot_check": p.verify_snapshot(self.root, self.baseline),
                       "content_check": {"status": "PASS"}}
        self.save_report()

    def save_report(self):
        (self.run / "d02_preflight.json").write_text(json.dumps(self.report))

    def test_current_baseline_not_hardcoded_to_rev005(self):
        self.assertNotEqual(self.baseline, p.BASELINE)
        self.assertEqual(p.check_preflight(self.run, self.root), self.baseline)

    def test_copied_pass_rejected_after_input_changed(self):
        (self.root / "data/records.csv").write_text("changed after PASS")
        with self.assertRaisesRegex(ValueError, "Stale preflight"):
            p.check_preflight(self.run, self.root)

    def test_baseline_rewritten_after_pass_rejected(self):
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Stale preflight"):
            p.check_preflight(self.run, self.root)

    def test_incomplete_baseline_cannot_claim_pass(self):
        self.manifest.write_text(f"{p.sha(self.root / p.CODE)}  {p.CODE}\n")
        self.report["snapshot_check"] = p.verify_snapshot(self.root, self.baseline)
        self.save_report()
        with self.assertRaisesRegex(ValueError, "lacks required"):
            p.check_preflight(self.run, self.root)

    def test_failed_content_check_rejected(self):
        self.report["content_check"]["status"] = "FAIL"
        self.save_report()
        with self.assertRaisesRegex(ValueError, "preflight required"):
            p.check_preflight(self.run, self.root)


class FrozenInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="guard-test-", dir=p.ROOT / "runs/d03")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.c = p.config()
        # Guard-only fixture: deliberately invalid joblib/NPZ bytes must never be
        # deserialized by a rejected inference. Real positive inference is tested
        # separately against the unchanged official model and a fresh full run.
        for name in ("model/pipeline.joblib", "weights/resnet18-f37072fd.pth"):
            dest = self.run / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"guard fixture")
        p.write_json(self.run / "model/classes.json", {"model_version": self.c["model_version"],
                     "class_order": self.c["classes"], "decision_score_order": self.c["classes"]})
        p.write_json(self.run / "declaration.json", {"config": self.c, "bound_hashes": p.bound_hashes()})
        for name in ("feature_context.json", "selection.json"):
            p.write_json(self.run / name, {})
        for split in p.SPLITS:
            for suffix in (".json", ".npz", "_mapping.csv"):
                path = self.run / "features" / f"{split}{suffix}"
                path.parent.mkdir(exist_ok=True)
                path.write_text("fixture")
        paths = [str(path.relative_to(p.ROOT)) for path in self.run.rglob("*") if path.is_file()]
        self.frozen = {"model_version": self.c["model_version"], "class_order": self.c["classes"],
                       "feature_input": self.c["input"], "feature_layer": self.c["feature_layer"], "feature_dim": 512,
                       "artifact_hashes": {**p.bound_hashes(), **{name: p.sha(p.ROOT / name) for name in paths}}}
        self.save_freeze()

    def save_freeze(self):
        (self.run / "freeze.json").write_text(json.dumps(self.frozen))

    def refused_before_load(self, message, compatibility=None):
        output = self.run / "must_not_exist.csv"
        with patch.object(p.joblib, "load", side_effect=AssertionError("deserialization occurred")) as loader:
            with self.assertRaisesRegex((ValueError, FileNotFoundError), message):
                p.infer(self.run, "validation", output, compatibility)
            loader.assert_not_called()
        self.assertFalse(output.exists())

    def test_complete_unmodified_freeze_passes_guard(self):
        self.assertEqual(p.check_frozen_run(self.run)["config"], self.c)

    def test_replaced_model_rejected_before_deserialization_and_output(self):
        (self.run / "model/pipeline.joblib").write_bytes(b"different classifier")
        self.refused_before_load("Frozen artifact changed.*pipeline.joblib")

    def test_missing_freeze_rejected_before_deserialization(self):
        (self.run / "freeze.json").unlink()
        self.refused_before_load("freeze.json")

    def test_class_mapping_change_rejected(self):
        (self.run / "model/classes.json").write_text('{}')
        self.refused_before_load("Frozen artifact changed.*classes.json")

    def test_missing_model_hash_rejected(self):
        del self.frozen["artifact_hashes"][str((self.run / "model/pipeline.joblib").relative_to(p.ROOT))]
        self.save_freeze()
        self.refused_before_load("Incomplete model freeze")

    def test_wrong_frozen_version_rejected(self):
        self.frozen["model_version"] = "unrelated-version"
        self.save_freeze()
        self.refused_before_load("version/classes differ")

    def test_changed_input_contract_rejected(self):
        self.frozen["feature_input"] = copy.deepcopy(self.c["input"])
        self.frozen["feature_input"]["mean"][0] = 0
        self.save_freeze()
        self.refused_before_load("feature contract differs")

    def test_compatibility_cannot_approve_a_model_change(self):
        revision = {"purpose": "D03_guard_only_revision", "freeze_sha256": p.sha(self.run / "freeze.json"),
                    "declaration_sha256": p.sha(self.run / "declaration.json"),
                    "changes": {str((self.run / "model/pipeline.joblib").relative_to(p.ROOT)): {}}}
        path = self.run / "compatibility.json"
        p.write_json(path, revision)
        self.refused_before_load("only change guard code/protocol", path)

    def test_test_split_still_rejected_before_freeze_or_load(self):
        with self.assertRaisesRegex(ValueError, "outside this execution scope"):
            p.infer(self.run, "test", self.run / "test.csv")


if __name__ == "__main__":
    unittest.main()
