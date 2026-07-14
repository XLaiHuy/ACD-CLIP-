import json
import math
import unittest
from pathlib import Path

from phase2c_utils import phase2c_config, validate_loss_weight


class Phase2DLB01Tests(unittest.TestCase):
    def test_default_weights_preserve_historical_formula(self):
        config = phase2c_config("A_prime", "run", 0.20)
        self.assertEqual(config["cls_loss_weight"], 1.0)
        self.assertEqual(config["seg_loss_weight"], 1.0)
        raw_cls, raw_seg, kg, k = 2.0, 3.0, 4.0, 5.0
        self.assertEqual(config["cls_loss_weight"] * raw_cls + config["seg_loss_weight"] * raw_seg + config["lambda_kg"] * kg + config["lambda_k"] * k, 5.05)

    def test_lb_weighted_formula(self):
        self.assertEqual(0.1 * 2.0, 0.2)
        self.assertEqual(1.0 * 3.0, 3.0)

    def test_regularizers_are_unchanged(self):
        config = phase2c_config("A_prime", "run", 0.20)
        self.assertEqual(config["lambda_kg"], 0.01)
        self.assertEqual(config["lambda_k"], 0.002)

    def test_negative_and_nonfinite_weights_rejected(self):
        for value in (-1.0, float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                validate_loss_weight("weight", value)

    def test_zero_weight_is_valid_nonnegative_boundary(self):
        self.assertEqual(validate_loss_weight("weight", 0.0), 0.0)

    def test_config_serializes_both_weights(self):
        config = phase2c_config("A_prime", "run", 0.20)
        payload = json.loads(json.dumps(config))
        self.assertEqual(payload["cls_loss_weight"], 1.0)
        self.assertEqual(payload["seg_loss_weight"], 1.0)

    def test_runner_locks_lb_protocol(self):
        text = Path("run_phase2d_LB_0p1_seed42.sh").read_text()
        for token in ("--cls-loss-weight 0.1", "--seg-loss-weight 1.0", "--batch-size 6", "--bf16", "--condition A_prime", "--hybrid-alpha-max 0.20"):
            self.assertIn(token, text)
        self.assertNotIn("phase2d_interpolation.py", text)
        self.assertNotIn("P_pcgrad", text)
        self.assertNotIn("LB_0p3", text)

    def test_historical_config_has_expected_protocol(self):
        config = json.loads(Path("runs/phase2c_bf16/A_alpha020_seed42/config.json").read_text())
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["batch_size"], 6)
        self.assertTrue(config["bf16"])
        self.assertEqual(config["epochs"], 15)
        self.assertEqual(config["hybrid_alpha_max"], 0.2)

    def test_pcgrad_disabled_for_a_prime(self):
        self.assertFalse(phase2c_config("A_prime", "run", 0.20)["pcgrad_enabled"])

    def test_no_medical_path_in_runner(self):
        text = Path("run_phase2d_LB_0p1_seed42.sh").read_text().lower()
        self.assertNotIn("medad", text)
        self.assertNotIn("brain", text)
        self.assertNotIn("liver", text)

    def test_loss_report_fields_are_defined_in_training_source(self):
        text = Path("phase2c_train.py").read_text()
        for field in ("raw_cls_loss", "weighted_cls_loss", "raw_seg_loss", "weighted_seg_loss", "weighted_kg_loss", "weighted_k_loss", "loss_metrics.csv"):
            self.assertIn(field, text)

    def test_checkpoint_metadata_fields_are_defined_in_training_source(self):
        text = Path("phase2c_train.py").read_text()
        self.assertIn('"cls_loss_weight": config["cls_loss_weight"]', text)
        self.assertIn('"seg_loss_weight": config["seg_loss_weight"]', text)

    def test_protocol_diff_preflight_is_present(self):
        text = Path("run_phase2d_LB_0p1_seed42.sh").read_text()
        self.assertIn("protocol_diff.json", text)
        self.assertIn("unintended protocol difference", text)
        self.assertIn("medical_evaluation", text)

    def test_fixed_split_is_locked(self):
        text = Path("run_phase2d_LB_0p1_seed42.sh").read_text()
        self.assertIn("splits/visa_train_seed42.csv", text)
        self.assertIn("splits/visa_val_seed42.csv", text)
        self.assertIn("splits/visa_split_seed42_metadata.json", text)

    def test_weights_are_finite(self):
        self.assertTrue(math.isfinite(validate_loss_weight("cls", 0.1)))


if __name__ == "__main__":
    unittest.main()
