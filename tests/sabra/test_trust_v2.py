from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from sabra.trust_v2.numerical import (  # noqa: E402
    PATCHES,
    compact_geometry_v2,
    construct_b1_v2,
    percentile_rank,
    relational_v2,
    trust_stability,
)
from sabra.trust_v2.visa_audit import EXPECTED_VISA_CLASSES, _model_metrics, _reserve_metadata_for_patch  # noqa: E402


class TrustV2NumericalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(4102)
        cls.features = rng.normal(size=(3, PATCHES, 8)).astype(np.float32)
        cls.features /= np.linalg.norm(cls.features, axis=-1, keepdims=True)
        cls.margins = rng.normal(size=(3, PATCHES)).astype(np.float32)
        ranks = np.stack([percentile_rank(x) for x in cls.margins])
        cls.d_rank = np.std(ranks, axis=0, ddof=0)
        cls.b1 = construct_b1_v2(cls.features, cls.d_rank, cls.margins)
        cls.geometry = compact_geometry_v2(cls.features, cls.b1)
        cls.relational = relational_v2(cls.geometry, cls.b1)

    def test_01_percentile_rank_ascending_ties(self) -> None:
        np.testing.assert_allclose(percentile_rank(np.array([2.0, 1.0, 1.0, 3.0])), [2 / 3, 1 / 6, 1 / 6, 1.0])

    def test_02_stable_ordering_is_deterministic(self) -> None:
        first = construct_b1_v2(self.features, self.d_rank, self.margins)
        second = construct_b1_v2(self.features, self.d_rank, self.margins)
        np.testing.assert_array_equal(first["peer_indices"], second["peer_indices"])
        np.testing.assert_array_equal(first["reserve_p16_index"], second["reserve_p16_index"])

    def test_03_b1_validity_implies_exact_peer_count(self) -> None:
        self.assertTrue(np.all(self.b1["valid_b1"] <= (self.b1["candidate_count"] >= 8)))

    def test_04_p9_is_exact_ninth_reserve(self) -> None:
        self.assertTrue(np.all(self.b1["valid_p9"] <= self.b1["valid_b1"]))
        self.assertTrue(np.all(self.b1["reserve_p9_index"][~self.b1["valid_p9"]] == -1))

    def test_05_p16_is_exact_sixteenth_reserve(self) -> None:
        self.assertTrue(np.all(self.b1["valid_p16"] <= self.b1["valid_p9"]))
        self.assertTrue(np.all(self.b1["reserve_p16_index"][~self.b1["valid_p16"]] == -1))

    def test_06_no_duplicate_peer_or_reserve_identity(self) -> None:
        for patch in np.flatnonzero(self.b1["valid_p16"]):
            values = list(self.b1["peer_indices"][patch]) + [self.b1["reserve_p9_index"][patch], self.b1["reserve_p16_index"][patch]]
            self.assertEqual(len(values), len(set(values)))

    def test_07_geometry_shapes_are_compact(self) -> None:
        self.assertEqual(self.geometry["query_peer_cos"].shape, (3, PATCHES, 8))
        self.assertEqual(self.geometry["peer_gram_upper"].shape, (3, PATCHES, 36))
        self.assertEqual(self.geometry["query_reserve_cos"].shape, (2, 3, PATCHES))
        self.assertEqual(self.geometry["reserve_to_peer_cos"].shape, (2, 3, PATCHES, 8))

    def test_08_compact_geometry_is_float32(self) -> None:
        self.assertTrue(all(value.dtype == np.float32 for value in self.geometry.values()))

    def test_09_pgm_and_pcrr_baselines_are_finite(self) -> None:
        self.assertTrue(np.isfinite(self.relational["baseline_pgm"]).all())
        self.assertTrue(np.isfinite(self.relational["baseline_pcrr"]).all())

    def test_10_pgm_p16_has_all_eight_replacements(self) -> None:
        self.assertEqual(self.relational["reserve_pgm_rank"].shape, (2, 8, PATCHES))

    def test_11_pcrr_p16_is_diagnostic_only_geometry(self) -> None:
        self.assertEqual(self.relational["reserve_pcrr_rank"].shape, (2, 8, PATCHES))

    def test_12_baseline_cdf_is_fixed_per_image(self) -> None:
        self.assertTrue(np.all((self.relational["baseline_pgm"] >= 0) & (self.relational["baseline_pgm"] <= 1)))

    def test_13_reserves_do_not_change_peer_order(self) -> None:
        self.assertTrue(np.all(self.b1["peer_indices"][:, 0] != self.b1["reserve_p9_index"]))

    def test_14_d_rel_is_absolute_pgm_pcrr_gap(self) -> None:
        np.testing.assert_allclose(self.relational["d_rel"], np.abs(self.relational["baseline_pgm"] - self.relational["baseline_pcrr"]))

    def test_15_stability_includes_slot_eight_influence(self) -> None:
        stability = trust_stability(self.relational, self.b1)
        expected = 1 - np.max(np.abs(self.relational["reserve_pgm_rank"][0] - self.relational["baseline_pgm"][None]), axis=0)
        expected[~self.b1["valid_p9"]] = 0
        np.testing.assert_allclose(stability["S9"], np.clip(expected, 0, 1))

    def test_16_robust_reserve_score_includes_baseline(self) -> None:
        stability = trust_stability(self.relational, self.b1)
        expected = np.minimum.reduce(np.concatenate([self.relational["baseline_pgm"][None], self.relational["reserve_pgm_rank"][0]], axis=0))
        expected[~self.b1["valid_p9"]] = 0
        np.testing.assert_allclose(stability["R9"], np.clip(expected, 0, 1))

    def test_17_invalid_reserves_are_zeroed(self) -> None:
        stability = trust_stability(self.relational, self.b1)
        self.assertTrue(np.all(stability["S16"][~self.b1["valid_p16"]] == 0))
        self.assertTrue(np.all(stability["R16"][~self.b1["valid_p16"]] == 0))

    def test_18_stage_profile_geometry_is_three_stage(self) -> None:
        self.assertEqual(self.geometry["query_peer_cos"].shape[0], 3)

    def test_19_d_rank_uses_population_std(self) -> None:
        ranks = np.stack([percentile_rank(x) for x in self.margins])
        np.testing.assert_allclose(self.d_rank, ranks.std(axis=0, ddof=0))

    def test_20_no_gt_fields_in_numerical_sidecar(self) -> None:
        source = (ROOT / "tools/sabra/trust_v2/numerical.py").read_text().lower()
        self.assertNotIn("mask_path", source)
        self.assertNotIn("medical", source)

    def test_21_no_mvtec_access_in_gt_free_builder(self) -> None:
        source = (ROOT / "tools/sabra/trust_v2/cache_builder.py").read_text().lower()
        self.assertNotIn("mvtec_root", source)
        self.assertNotIn("mvtec_dataset", source)

    def test_22_protocol_json_is_valid_and_frozen(self) -> None:
        path = ROOT / "runs/phase5/sabra/TRUST_V2_DEVELOPMENT/SABRA_TRUST_V2_PROTOCOL.json"
        self.assertEqual(json.loads(path.read_text())["status"], "frozen")

    def test_23_deterministic_reproduction_is_bytewise_for_core_arrays(self) -> None:
        again = relational_v2(compact_geometry_v2(self.features, self.b1), self.b1)
        for key in ("baseline_pgm", "baseline_pcrr", "reserve_pgm_rank"):
            np.testing.assert_array_equal(self.relational[key], again[key])

    def test_24_reporting_serializes_patch_specific_reserve_ids(self) -> None:
        p9 = np.asarray([101, 203, 307], dtype=np.int64)
        p16 = np.asarray([401, 503, 607], dtype=np.int64)
        row0 = _reserve_metadata_for_patch(p9, p16, 0)
        row1 = _reserve_metadata_for_patch(p9, p16, 1)
        self.assertEqual(row0, {"p9_index": 101, "p16_index": 401})
        self.assertEqual(row1, {"p9_index": 203, "p16_index": 503})
        self.assertNotEqual(row0, row1)

    def test_25_reporting_fix_does_not_mutate_science_or_predictions(self) -> None:
        scientific = {
            key: np.array(self.relational[key], copy=True)
            for key in ("baseline_pgm", "baseline_pcrr", "reserve_pgm_rank", "reserve_pcrr_rank")
        }
        stability = {key: np.array(value, copy=True) for key, value in trust_stability(self.relational, self.b1).items()}
        predictions = np.linspace(0.01, 0.99, 3 * PATCHES, dtype=np.float32)
        predictions_before = predictions.tobytes()
        _reserve_metadata_for_patch(np.asarray([11, 13]), np.asarray([17, 19]), 1)
        for key, value in scientific.items():
            self.assertEqual(value.tobytes(), self.relational[key].tobytes())
        for key, value in stability.items():
            self.assertEqual(value.tobytes(), trust_stability(self.relational, self.b1)[key].tobytes())
        self.assertEqual(predictions_before, predictions.tobytes())

    def test_26_authority_metrics_allow_undefined_zero_occupancy_spearman(self) -> None:
        classes = np.repeat(np.asarray(EXPECTED_VISA_CLASSES), 2)
        target = np.tile(np.asarray([0, 1], dtype=np.int8), len(EXPECTED_VISA_CLASSES))
        scores = np.tile(np.asarray([0.2, 0.8], dtype=np.float32), len(EXPECTED_VISA_CLASSES))
        metrics, rows = _model_metrics({"A": scores}, target, np.zeros_like(scores), classes)
        self.assertIsNone(metrics["A"]["occupancy_spearman_mean"])
        self.assertEqual(len(rows["rows"]), len(EXPECTED_VISA_CLASSES))


if __name__ == "__main__":
    unittest.main()

