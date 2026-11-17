"""D04 boundary tests use synthetic data or existing validation artifacts only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import d04_pipeline as q
import verify_d04 as v


def synthetic_rows():
    return [{"sample_id": f"synthetic_{label}_{i}", "record_id": f"r_{label}", "true_label": label,
             "split": "test", "load_hp": "3", "model_image_path": f"synthetic/{label}_{i}.png"}
            for label in q.CLASSES for i in range(59)]


class PopulationTests(unittest.TestCase):
    def setUp(self):
        self.c, self.rows = q.config(), synthetic_rows()

    def test_complete_synthetic_population(self):
        self.assertEqual(len(q.validate_rows(self.rows, self.c)), 236)

    def test_missing(self):
        with self.assertRaisesRegex(ValueError, "Missing/extra"):
            q.validate_rows(self.rows[:-1], self.c)

    def test_duplicate(self):
        self.rows[-1] = self.rows[-2]
        with self.assertRaisesRegex(ValueError, "Duplicate sample"):
            q.validate_rows(self.rows, self.c)

    def test_wrong_split(self):
        self.rows[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "Wrong split"):
            q.validate_rows(self.rows, self.c)

    def test_wrong_load(self):
        self.rows[0]["load_hp"] = "2"
        with self.assertRaisesRegex(ValueError, "Wrong split/load"):
            q.validate_rows(self.rows, self.c)

    def test_reordered_samples(self):
        with self.assertRaisesRegex(ValueError, "IDs/order/source"):
            q.validate_rows(list(reversed(self.rows)), self.c, self.rows)

    def test_label_changes_cannot_reach_numeric_inputs(self):
        changed = copy.deepcopy(self.rows)
        for r in changed:
            r["true_label"] = "invalid_label"
        self.assertEqual(q.input_specs(self.rows), q.input_specs(changed))
        self.assertEqual(set(q.input_specs(self.rows)[0]), {"sample_id", "model_image_path"})


class NumericTests(unittest.TestCase):
    def test_wrong_dimension(self):
        with self.assertRaisesRegex(ValueError, "Invalid features"):
            q.validate_features(np.zeros((2, 511), dtype=np.float32), 2)

    def test_nan(self):
        x = np.zeros((2, 512), dtype=np.float32)
        x[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            q.validate_features(x, 2)

    def test_wrong_dtype(self):
        with self.assertRaisesRegex(ValueError, "dtype"):
            q.validate_features(np.zeros((2, 512)), 2)

    def test_extra_metadata_rejected_before_image_read(self):
        with patch.object(q.p, "tensor_input", side_effect=AssertionError("image read")):
            with self.assertRaisesRegex(ValueError, "metadata"):
                q.extract_features(None, [{"sample_id": "s", "model_image_path": "x", "true_label": "normal"}], q.p.config(), "cpu")

    def test_all_fit_entrypoints_blocked(self):
        with q.no_fitting():
            for estimator, method in ((q.StandardScaler(), "fit"), (q.StandardScaler(), "partial_fit"),
                                      (q.StandardScaler(), "fit_transform"), (q.SVC(), "fit"),
                                      (q.Pipeline([]), "fit"), (q.Pipeline([]), "fit_predict")):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    getattr(estimator, method)(np.zeros((4, 512)), np.arange(4))

    def test_manual_metric_including_errors(self):
        rows = [{"true_label": a, "predicted_class": b} for a, b in zip(q.CLASSES, ["normal", "normal", "outer_race", "rolling_element"])]
        manual = v.recount(rows)
        self.assertEqual(manual["accuracy"], .75)
        self.assertEqual(manual["balanced_accuracy"], .75)
        self.assertAlmostEqual(manual["macro_f1"], 2 / 3)
        self.assertEqual(manual["confusion_matrix_rows_true_columns_predicted"], [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="d04-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.path = self.run / "checkpoints/000_features.npz"
        self.x = np.zeros((2, 512), dtype=np.float32)
        q.save_npz(self.path, features=self.x, sample_id=np.array(["a", "b"]), context=np.array("current"))

    def test_stale_context(self):
        with self.assertRaisesRegex(ValueError, "Stale checkpoint"):
            q.read_chunk(self.path, ["a", "b"], "old")

    def test_swapped_cache_rows(self):
        with self.assertRaisesRegex(ValueError, "checkpoint/order"):
            q.read_chunk(self.path, ["b", "a"], "current")

    def test_cache_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, "replace persisted"):
            q.save_npz(self.path, features=self.x)

    def test_checkpoint_tamper_rejected(self):
        q.event(self.run, "features_saved", offset=0, sha256=q.p.sha(self.path))
        self.path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed after commit"):
            q.verify_checkpoint_log(self.run)

    def test_complete_checkpoint_can_be_reused_without_predicting(self):
        q.event(self.run, "features_saved", offset=0, sha256=q.p.sha(self.path))
        q.verify_checkpoint_log(self.run)
        with patch.object(q, "extract_features", side_effect=AssertionError("recomputed")):
            self.assertTrue(np.array_equal(q.read_chunk(self.path, ["a", "b"], "current")["features"], self.x))


class FrozenIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="d04-identity-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = q.config()
        model = q.ROOT / self.c["model_run"]
        frozen = q.p.read_json(model / "freeze.json")
        compat = q.p.read_json(q.ROOT / self.c["compatibility"])
        paths = set(frozen["artifact_hashes"]) | {f"{self.c['model_run']}/freeze.json", self.c["compatibility"]}
        paths.update(change["before_copy"] for change in compat["changes"].values())
        for name in paths:
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(q.ROOT / name)

    def replace(self, name, content):
        target = self.root / self.c["model_run"] / name
        target.unlink()  # never modify the original behind a symlink
        target.write_bytes(content)

    def assert_refused(self):
        with patch.object(q, "ROOT", self.root), patch.object(q.p, "ROOT", self.root):
            with patch.object(q.joblib, "load", side_effect=AssertionError("deserialized")) as loader:
                with patch.object(q.torch, "load", side_effect=AssertionError("weight deserialized")) as weight_loader:
                    with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                        q.load_model(self.c)
                    loader.assert_not_called()
                    weight_loader.assert_not_called()

    def test_replaced_model_refused_before_deserialization(self):
        self.replace("model/pipeline.joblib", b"replacement")
        self.assert_refused()

    def test_wrong_class_mapping_refused_before_deserialization(self):
        self.replace("model/classes.json", json.dumps({"class_order": list(reversed(q.CLASSES))}).encode())
        self.assert_refused()

    def test_replaced_scaler_refused_before_deserialization(self):
        self.replace("model/scaler.joblib", b"replacement")
        self.assert_refused()

    def test_replaced_weight_refused_before_deserialization(self):
        self.replace("weights/resnet18-f37072fd.pth", b"replacement")
        self.assert_refused()


if __name__ == "__main__":
    unittest.main()
