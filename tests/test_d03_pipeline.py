"""Discriminating D03 counterexamples; all mutations are in fresh temporary dirs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import d03_pipeline as p


class D03Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aae5103-d03-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = p.config()

    def image(self, size=(224, 224), mode="RGB"):
        a = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        a[:, :, 0] = np.arange(size[0], dtype=np.uint8)[None, :]
        a[:, :, 1] = np.arange(size[1], dtype=np.uint8)[:, None]
        a[:, :, 2] = 255
        path = self.root / "input.png"
        im = Image.fromarray(a)
        im.convert(mode).save(path)
        return path, a

    def test_normalization_analytic_pixels_once_no_geometry(self):
        path, a = self.image()
        with patch.object(Image.Image, "resize", side_effect=AssertionError("resize")), \
             patch.object(Image.Image, "crop", side_effect=AssertionError("crop")):
            x = p.tensor_input(path, self.c).numpy()
        expected = (a.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
                    - np.array([.485, .456, .406], dtype=np.float32)[:, None, None]) \
                   / np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
        np.testing.assert_array_equal(x, expected)
        self.assertEqual(x.shape, (3, 224, 224))
        # This pattern distinguishes all four corners and a central location.
        for y, z in ((0, 0), (0, 223), (223, 0), (223, 223), (73, 131)):
            np.testing.assert_array_equal(x[:, y, z], expected[:, y, z])

    def test_wrong_size_rejected_instead_of_resized(self):
        path, _ = self.image(size=(225, 224))
        with self.assertRaisesRegex(ValueError, "224x224"):
            p.tensor_input(path, self.c)

    def test_non_rgb_rejected(self):
        path, _ = self.image(mode="RGBA")
        with self.assertRaisesRegex(ValueError, "RGB"):
            p.tensor_input(path, self.c)

    def test_test_rows_rejected_before_any_image_read(self):
        with patch.object(p.Image, "open", side_effect=AssertionError("test image read")):
            with self.assertRaisesRegex(ValueError, "Test feature"):
                p.extract_rows(torch.nn.Identity(), [{"split": "test"}], self.c, torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "Test prediction"):
                p.prediction_rows(None, None, [{"split": "test"}], self.c)
            with self.assertRaisesRegex(ValueError, "Test cache"):
                p.load_cache(self.root, "test", [], {})

    def training(self):
        rng = np.random.default_rng(22)
        x = rng.normal(size=(8, 512)).astype(np.float32)
        rows = [{"sample_id": str(i), "split": "train", "true_label": self.c["classes"][i // 2]} for i in range(8)]
        return x, p.labels(rows, self.c["classes"]), rows

    def test_scaler_heldout_contamination_counterexample(self):
        x, _, _ = self.training()
        good = StandardScaler().fit(x)
        p.check_scaler(good, x)
        contaminated = StandardScaler().fit(np.concatenate([x, np.full((8, 512), 1e6, dtype=np.float32)]))
        with self.assertRaisesRegex(ValueError, "Scaler"):
            p.check_scaler(contaminated, x)
        # Defeat the count guard: the statistics guard must still catch contamination.
        contaminated.n_samples_seen_ = len(x)
        with self.assertRaisesRegex(ValueError, "mean"):
            p.check_scaler(contaminated, x)

    def test_fit_rejects_validation_and_misaligned_labels(self):
        x, y, rows = self.training()
        rows[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "Only training"):
            p.fit_pipeline(x, y, rows, self.c["candidates"][0], self.c)
        rows[0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "labels misaligned"):
            p.fit_pipeline(x, np.roll(y, 2), rows, self.c["candidates"][0], self.c)

    def test_svm_fit_receives_training_rows_and_training_labels_only(self):
        x, y, rows = self.training()
        original_fit = p.SVC.fit
        received = []
        def spy(instance, fit_x, fit_y, *args, **kwargs):
            received.append((fit_x.copy(), fit_y.copy()))
            return original_fit(instance, fit_x, fit_y, *args, **kwargs)
        with patch.object(p.SVC, "fit", spy):
            model = p.fit_pipeline(x, y, rows, self.c["candidates"][0], self.c)
        self.assertEqual(len(received), 1)
        np.testing.assert_array_equal(received[0][1], y)
        self.assertEqual(received[0][0].shape, (8, 512))
        p.check_scaler(model.named_steps["scaler"], x)

    def cache(self):
        x, _, rows = self.training()
        p.save_cache(self.root, "train", x, rows, {"fixed": "context"})
        return x, rows

    def test_cache_order_reversal_even_when_id_set_unchanged(self):
        _, rows = self.cache()
        with self.assertRaisesRegex(ValueError, "row order"):
            p.load_cache(self.root, "train", rows[::-1], {"fixed": "context"})

    def test_cache_metadata_wrong_label_group_split_rejected(self):
        _, rows = self.cache()
        for key, value in (("true_label", "outer_race"), ("split", "validation"), ("sample_id", "missing")):
            altered = copy.deepcopy(rows)
            altered[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                p.load_cache(self.root, "train", altered, {"fixed": "context"})

    def test_cache_feature_swap_with_updated_npz_checksum_still_rejected(self):
        x, rows = self.cache()
        path = self.root / "features/train.npz"
        swapped = x.copy()
        swapped[[0, 1]] = swapped[[1, 0]]
        with path.open("wb") as f:
            np.savez_compressed(f, features=swapped, sample_id=np.array([r["sample_id"] for r in rows]))
        meta_path = self.root / "features/train.json"
        meta = p.read_json(meta_path)
        meta["npz_sha256"], meta["array_sha256"] = p.sha(path), p.array_hash(swapped)
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "row hash"):
            p.load_cache(self.root, "train", rows, {"fixed": "context"})

    def test_stale_cache_weights_or_config_rejected(self):
        _, rows = self.cache()
        with self.assertRaisesRegex(ValueError, "Stale cache"):
            p.load_cache(self.root, "train", rows, {"fixed": "different weight"})

    def test_cache_missing_duplicate_nonfinite_or_wrong_dimension(self):
        x, rows = self.cache()
        for bad_rows in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                p.load_cache(self.root, "train", bad_rows, {"fixed": "context"})
        for bad_x in (x[:, :-1], np.full_like(x, np.nan)):
            path = self.root / "features/train.npz"
            with path.open("wb") as f:
                np.savez_compressed(f, features=bad_x, sample_id=np.array([r["sample_id"] for r in rows]))
            meta_path = self.root / "features/train.json"
            meta = p.read_json(meta_path)
            meta["npz_sha256"] = p.sha(path)
            meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "Invalid cached"):
                p.load_cache(self.root, "train", rows, {"fixed": "context"})

    def test_metrics_manual_nonperfect_example_and_order(self):
        y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        pred = np.array([0, 1, 1, 1, 2, 3, 0, 3])
        result = p.metrics(y, pred, self.c["classes"])
        expected = [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]]
        self.assertEqual(result["confusion_matrix_rows_true_columns_predicted"], expected)
        self.assertAlmostEqual(result["balanced_accuracy"], .625)
        self.assertAlmostEqual(result["macro_f1"], (.5 + .8 + 2 / 3 + .5) / 4)
        self.assertEqual(list(result["per_class_recall"].values()), [.5, 1., .5, .5])

    def test_selection_secondary_and_tie_rules(self):
        def item(i, f1, ba):
            return {"candidate_index": i, "params": self.c["candidates"][i], "validation": {"macro_f1": f1, "balanced_accuracy": ba}}
        self.assertEqual(p.select_candidate([item(0, .8, .7), item(7, .9, .6)], self.c)["candidate_index"], 7)
        self.assertEqual(p.select_candidate([item(0, .9, .7), item(7, .9, .8)], self.c)["candidate_index"], 7)
        self.assertEqual(p.select_candidate([item(1, .9, .8), item(4, .9, .8)], self.c)["candidate_index"], 1)
        self.assertEqual(p.select_candidate([item(1, .9, .8), item(0, .9, .8)], self.c)["candidate_index"], 0)

    def test_population_omission_duplicate_unknown_wrong_metadata(self):
        # Files used by metadata audit are copied or linked read-only; only CSV is mutated.
        for relative in p.INPUTS[:7]:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative != "data/samples.csv":
                target.symlink_to(p.ROOT / relative)
        # Audit only hashes model files; all linked inputs are unchanged.
        (self.root / "data/processed").symlink_to(p.ROOT / "data/processed")
        original = p.read_csv(p.ROOT / "data/samples.csv")
        p.write_csv(self.root / "data/samples.csv", original)
        self.assertEqual(len(p.audit_population(self.root)), 704)
        mutations = [original[:-1], original + [original[0]]]
        for key, value in (("record_id", "unknown"), ("true_label", "rolling_element"),
                           ("group_id", "wrong"), ("source_family_id", "wrong"), ("split", "validation"),
                           ("model_image_path", original[0]["review_image_path"])):
            changed = copy.deepcopy(original)
            changed[0][key] = value
            mutations.append(changed)
        for changed in mutations:
            path = self.root / "data/samples.csv"
            path.unlink()
            p.write_csv(path, changed)
            with self.assertRaises(ValueError):
                p.audit_population(self.root)

    def test_saving_never_overwrites(self):
        p.write_json(self.root / "frozen.json", {"a": 1})
        with self.assertRaises(FileExistsError):
            p.write_json(self.root / "frozen.json", {"a": 2})

    def test_extraction_feature_shape_finiteness_eval_and_state_guards(self):
        path, _ = self.image()
        rows = [{"split": "validation", "model_image_path": str(path), "true_label": "normal"}]
        class Tiny(torch.nn.Module):
            def __init__(self, dim=512, bad=False, mutate=False):
                super().__init__()
                self.register_buffer("counter", torch.zeros(1))
                self.dim, self.bad, self.mutate = dim, bad, mutate
            def forward(self, x):
                if self.mutate:
                    self.counter.add_(1)
                result = x.mean(dim=(1, 2, 3))[:, None].repeat(1, self.dim)
                return result * float("nan") if self.bad else result
        model = Tiny().eval()
        original = p.extract_rows(model, rows, self.c, torch.device("cpu"))
        changed = copy.deepcopy(rows)
        changed[0]["true_label"] = "outer_race"
        np.testing.assert_array_equal(original, p.extract_rows(model, changed, self.c, torch.device("cpu")))
        for bad in (Tiny(dim=511).eval(), Tiny(bad=True).eval(), Tiny(mutate=True).eval(), Tiny()):
            with self.assertRaises(ValueError):
                p.extract_rows(bad, rows, self.c, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
