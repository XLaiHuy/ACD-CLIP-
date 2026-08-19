from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import sabra.trust_v2.visa_audit as audit  # noqa: E402


class TrustV2M4FollowupTest(unittest.TestCase):
    def test_m4_appends_d_rel_after_selected_columns(self) -> None:
        selected = np.arange(18, dtype=np.float32).reshape(6, 3)
        d_rel = np.linspace(0.01, 0.06, 6, dtype=np.float32)
        m4 = audit.build_m4_features(selected, d_rel)
        np.testing.assert_array_equal(m4[:, :3], selected)
        np.testing.assert_array_equal(m4[:, 3], d_rel)
        self.assertEqual(m4.shape, (6, 4))

    def test_m4_is_not_e_plus_pcrr(self) -> None:
        selected = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        d_rel = np.asarray([0.7, 0.8], dtype=np.float32)
        m4 = audit.build_m4_features(selected, d_rel)
        e_plus_pcrr = np.column_stack([selected[:, 0], np.asarray([0.91, 0.92], dtype=np.float32)])
        self.assertEqual(m4.shape[1], selected.shape[1] + 1)
        self.assertFalse(np.array_equal(m4, e_plus_pcrr))

    def test_m4_preserves_selected_feature_order_and_m0_m3_inputs(self) -> None:
        models = {
            "M0_E": np.asarray([[1.0], [2.0]], dtype=np.float32),
            "M1": np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
            "M2": np.asarray([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32),
            "M3": np.asarray([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]], dtype=np.float32),
        }
        before = {name: value.tobytes() for name, value in models.items()}
        m4 = audit.build_m4_features(models["M3"], np.asarray([0.2, 0.4], dtype=np.float32))
        self.assertEqual(m4.shape[1], models["M3"].shape[1] + 1)
        for name, value in models.items():
            self.assertEqual(value.tobytes(), before[name])

    def test_loco_excludes_held_out_class_from_scaler_and_model(self) -> None:
        scaler_rows: list[int] = []
        model_targets: list[np.ndarray] = []

        class SpyScaler:
            def fit(self, values):
                scaler_rows.append(int(values.shape[0]))
                return self

            def transform(self, values, copy=False):
                return values

        class SpyModel:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, values, target):
                model_targets.append(np.asarray(target))
                return self

            def predict_proba(self, values):
                return np.column_stack([np.zeros(values.shape[0]), np.ones(values.shape[0])])

        classes = np.repeat(np.asarray(audit.EXPECTED_VISA_CLASSES), 2)
        target = classes.copy()
        features = np.arange(classes.size, dtype=np.float64)[:, None]
        with patch.object(audit, "StandardScaler", SpyScaler), patch.object(audit, "LogisticRegression", SpyModel):
            output = audit._loco(features, target, classes)
        self.assertEqual(output.shape, target.shape)
        self.assertEqual(scaler_rows, [classes.size - 2] * len(audit.EXPECTED_VISA_CLASSES))
        self.assertEqual(len(model_targets), len(audit.EXPECTED_VISA_CLASSES))
        for held, fitted_target in zip(audit.EXPECTED_VISA_CLASSES, model_targets):
            self.assertNotIn(held, fitted_target)

    def test_d_rel_is_cache_derived_and_gt_free(self) -> None:
        manifest = json.loads((ROOT / "runs/phase5/sabra/TRUST_V2_DEVELOPMENT/TRUST_V2_GT_FREE_MANIFEST.json").read_text())
        self.assertIn("D_rel", manifest["fields"])
        self.assertIn("D_rel", manifest["features"]["persistent"])
        shard = ROOT / "runs/phase5/sabra/TRUST_V2_DEVELOPMENT/cache" / f"{manifest['classes'][0]}.npz"
        with np.load(shard, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["D_rel"], np.abs(data["baseline_pgm"] - data["baseline_pcrr"]))
        source = (ROOT / "tools/sabra/trust_v2/numerical.py").read_text().lower()
        self.assertNotIn("mask_path", source)
        self.assertNotIn("medical", source)
        self.assertNotIn("mvtec_dataset", source)

    def test_followup_has_no_mvtec_or_medical_access(self) -> None:
        source = (ROOT / "tools/sabra/trust_v2/visa_audit.py").read_text().lower()
        self.assertNotIn("mvtec_dataset", source)
        self.assertNotIn("medical_dataset", source)
        self.assertEqual(json.loads((ROOT / "runs/phase5/sabra/TRUST_V2_DEVELOPMENT/TRUST_V2_GT_FREE_MANIFEST.json").read_text())["counters"]["MEDICAL_READS"], 0)


if __name__ == "__main__":
    unittest.main()
