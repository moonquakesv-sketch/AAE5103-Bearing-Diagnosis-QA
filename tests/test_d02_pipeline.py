"""Checks that catch leakage, wrong coordinates, and incorrect population selection."""
import sys
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from matplotlib.axes import Axes
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import d02_pipeline as p
import reproduce_d02
from data_contract import ContractError


class D02Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load((p.ROOT / "configs/data.yaml").read_text())

    def test_reject_heldout_color_fit(self):
        with self.assertRaisesRegex(ContractError, "Held-out"):
            p.fit_color_limits([np.arange(10)], [{"split":"test","sample_id":"a"}], self.cfg)

    def test_fit_all_matrix_pixels_exact_quantile(self):
        matrices = [np.arange(8).reshape(2,4), np.arange(8,16).reshape(2,4)]
        rows = [{"split":"train","sample_id":str(i),"log_power_sha256_float64":p.array_hash(m)}
                for i,m in enumerate(matrices)]
        colors = p.fit_color_limits(matrices,rows,self.cfg)
        self.assertEqual((colors["vmin"],colors["vmax"]), (.15,14.85))
        self.assertEqual(colors["pixel_count"],16)

    def test_index_origin_and_incomplete_tail(self):
        x = np.arange(128 + 2*2048 + 128 + 17, dtype=float)
        valid,rejected = p.eligible_windows(x,self.cfg)
        self.assertEqual([(w["start_sample"],w["end_sample"]) for w in valid],[(128,2176),(2176,4224)])
        self.assertEqual(rejected,[])

    def test_quality_rejection_preserves_full_signal_indices(self):
        x = np.arange(6400,dtype=float)
        x[128:2176] = 1
        valid,rejected = p.eligible_windows(x,self.cfg)
        self.assertEqual(rejected[0]["reason"],"constant")
        self.assertEqual(valid[0]["start_sample"],2176)

    def test_balance_exact_floor_linspace_and_train_only(self):
        plans = [{"split":"train","valid_windows":list(range(n))} for n in (3,8)]
        self.assertEqual(p.choose_balanced(plans),3)
        self.assertEqual(plans[1]["selected_windows"],[0,3,7])
        with self.assertRaises(ContractError):
            p.choose_balanced([{"split":"validation","valid_windows":[1]}])

    def test_synthetic_time_frequency_direction(self):
        result = p.synthetic_checks(self.cfg)
        self.assertEqual(result["status"],"PASS")
        self.assertLess(result["model_peak_rows_left_right"][1],result["model_peak_rows_left_right"][0])

    def test_no_per_image_autoscale(self):
        matrix = np.ones((128,2048))
        a = p.model_pixels(matrix,{"vmin":0,"vmax":2},self.cfg)
        b = p.model_pixels(matrix,{"vmin":0,"vmax":4},self.cfg)
        self.assertFalse(np.array_equal(a,b))


class D02ReviewRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = p.ROOT
        cls.cfg = yaml.safe_load((p.ROOT / "configs/data.yaml").read_text())

    def test_validator_rejects_unknown_record_before_signal_or_image_io(self):
        samples = p.read_csv(self.source / "data/samples.csv")
        extra = copy.deepcopy(next(s for s in samples if s["split"] == "test"))
        extra.update(record_id="cwru_nonexistent", sample_id="s_deadbeefdeadbeefdeadbeef",
                     model_image_path="missing_model.png", review_image_path="missing_review.png")
        original_read = p.read_csv
        def read(path):
            return samples + [extra] if Path(path).name == "samples.csv" else original_read(path)
        frozen = p.read_json(p.RUN / "training_freeze.json")
        with tempfile.TemporaryDirectory() as temporary, patch.object(p, "read_csv", side_effect=read), \
                patch.object(p, "check_freeze", return_value=frozen), patch.object(p, "load_record") as load:
            with self.assertRaisesRegex(ContractError, "Unknown sample record"):
                p.validate(output=Path(temporary) / "acceptance.json")
            load.assert_not_called()

    def test_population_requires_complete_bidirectional_membership(self):
        records = [{"record_id":"r", "channel":"X001_DE_time"}]
        windows = [{"start_sample":128+i*2048,"end_sample":2176+i*2048,"eligible_window_index":i} for i in range(2)]
        plans = [{"record_id":"r","selected_windows":windows}]
        samples = [{"record_id":"r","sample_id":p.sample_id(records[0],w),**w} for w in windows]
        self.assertEqual(p.validate_sample_population(samples,records,plans),2)
        changed = copy.deepcopy(samples)
        changed[1]["start_sample"] += 1
        for bad in (samples[:1], samples+[samples[0]], changed):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                p.validate_sample_population(bad,records,plans)

    def test_reject_actual_model_vertical_flip(self):
        original = p.model_pixels
        with patch.object(p,"model_pixels",side_effect=lambda *a:original(*a)[::-1].copy()):
            with self.assertRaisesRegex(ContractError,"Actual model RGB"):
                p.synthetic_checks(self.cfg)

    def test_reject_actual_model_horizontal_flip(self):
        original = p.model_pixels
        with patch.object(p,"model_pixels",side_effect=lambda *a:original(*a)[:,::-1].copy()):
            with self.assertRaisesRegex(ContractError,"Actual model RGB"):
                p.synthetic_checks(self.cfg)

    def test_reject_review_linear_axis_for_log_frequency_grid(self):
        original = Axes.set_yscale
        def wrong_scale(ax, value, **kwargs):
            return original(ax, "linear" if value == "log" else value, **kwargs)
        with patch.object(Axes,"set_yscale",wrong_scale):
            with self.assertRaisesRegex(ContractError,"review PNG time/log-frequency"):
                p.synthetic_checks(self.cfg)

    def test_review_pixels_reconstructed_not_just_dimensions(self):
        t = np.arange(2048)/12000
        matrix = p.log_power(np.sin(2*np.pi*300*t),self.cfg)
        colors = {"vmin":-60,"vmax":5}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, review = root/"model.png", root/"review.png"
            sample={"model_image_path":"model.png","review_image_path":"review.png",
                    "model_pixels_sha256":p.render(matrix,colors,self.cfg,model,review)}
            with patch.object(p,"ROOT",root):
                p.verify_sample_images(sample,matrix,colors,self.cfg)
                Image.new("RGBA",(640,420),(0,0,0,255)).save(review)
                with self.assertRaisesRegex(ContractError,"Review image: decoded PNG pixels differ"):
                    p.verify_sample_images(sample,matrix,colors,self.cfg)

    def test_reproduction_compares_source_review_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,target=Path(temporary)/"source",Path(temporary)/"target"
            row={k:"same" for k in p.FIELDS}
            row.update(model_image_path="model.png",review_image_path="review.png")
            for root in (source,target):
                (root/"data").mkdir(parents=True)
                (root/"configs").mkdir()
                p.write_csv(root/"data/samples.csv",[row])
                p.write_json(root/"configs/cwt_color_limits.json",{
                    "vmin":0,"vmax":1,"sample_ids":["same"],"matrix_hashes":{},"pixel_count":10})
                Image.new("RGB",(224,224),"white").save(root/"model.png")
                Image.new("RGBA",(640,420),"white").save(root/"review.png")
            self.assertTrue(reproduce_d02.compare_outputs(source,target)["source_model_and_review_decoded_pixels_equal"])
            Image.new("RGBA",(640,420),"black").save(source/"review.png")
            with self.assertRaisesRegex(ContractError,"review_image_path.*decoded PNG pixels differ"):
                reproduce_d02.compare_outputs(source,target)

    def test_changed_code_needs_explicit_revision_authorization(self):
        expected = p.read_json(p.RUN/"training_freeze.json")["input_hashes"]
        with self.assertRaisesRegex(ContractError,"explicit validation-revision"):
            p.check_processing_inputs(expected)

    def test_compatibility_cannot_approve_numerical_function_change(self):
        expected={"scripts/d02_pipeline.py":"old","configs/data.yaml":"cfg"}
        current={"scripts/d02_pipeline.py":"new","configs/data.yaml":"cfg"}
        functions={name:"same" for name in p.UNCHANGED_PROCESSING_FUNCTIONS}
        altered={**functions,"model_pixels":"different"}
        approved={"purpose":"D02_validation_only_revision","frozen_input_hashes":expected,
                  "approved_input_hashes":current,"prior_pipeline_path":"before.py",
                  "unchanged_processing_functions":functions}
        with patch.object(p,"input_hashes",return_value=current),patch.object(p,"read_json",return_value=approved), \
                patch.object(p,"file_hash",return_value="old"), \
                patch.object(p,"processing_fingerprints",side_effect=[altered,functions]):
            with self.assertRaisesRegex(ContractError,"Numerical processing or rendering changed"):
                p.check_processing_inputs(expected,Path("compatibility.json"))


if __name__ == "__main__":
    unittest.main()
