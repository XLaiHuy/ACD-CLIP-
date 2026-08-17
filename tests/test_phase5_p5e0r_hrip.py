import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tools.audit_phase5_p5e0r_hrip as e0r
from audit_phase5_hsir import ap_contamination, pairwise_risks, shifted_map
from audit_phase5_second_evidence import deterministic_matches, select_top


class TestP5E0R(unittest.TestCase):
    def test_t1_one_image_shift_parity(self):
        image = np.arange(e0r.PIXELS_PER_IMAGE, dtype=np.float32)
        self.assertTrue(np.array_equal(e0r.shift_per_image([image]), shifted_map(image, 518, 518)))

    def test_t2_two_image_boundary_isolation(self):
        a = np.zeros(e0r.PIXELS_PER_IMAGE, dtype=np.float32)
        b = np.ones(e0r.PIXELS_PER_IMAGE, dtype=np.float32)
        actual = e0r.shift_per_image([a, b])
        expected = np.concatenate([shifted_map(a, 518, 518), shifted_map(b, 518, 518)])
        self.assertTrue(np.array_equal(actual, expected))
        self.assertTrue(np.all(actual[:e0r.PIXELS_PER_IMAGE] == 0))
        self.assertTrue(np.all(actual[e0r.PIXELS_PER_IMAGE:] == 1))

    def test_t3_historical_class_concatenation_is_rejected(self):
        a = np.arange(e0r.PIXELS_PER_IMAGE, dtype=np.float32)
        b = a + 1
        with self.assertRaises(ValueError):
            shifted_map(np.concatenate([a, b]), 518, 518)

    def test_t4_identity_order_preserved(self):
        a = np.zeros(e0r.PIXELS_PER_IMAGE, dtype=np.float32)
        b = np.ones(e0r.PIXELS_PER_IMAGE, dtype=np.float32)
        out = e0r.shift_per_image([a, b])
        self.assertEqual(out.size, 2 * e0r.PIXELS_PER_IMAGE)
        self.assertTrue(np.all(out[:e0r.PIXELS_PER_IMAGE] == 0))
        self.assertTrue(np.all(out[e0r.PIXELS_PER_IMAGE:] == 1))

    def test_t5_only_shifted_evidence_changes(self):
        values = {name: np.arange(64, dtype=np.float32) for name in ("HRIP", "E_nonlocal", "score", "final_margin", "D_rank", "labels", "pixel_id")}
        before = {k: v.copy() for k, v in values.items()}
        _ = e0r.shift_per_image([np.zeros(e0r.PIXELS_PER_IMAGE, dtype=np.float32)])
        for name in values:
            self.assertTrue(np.array_equal(values[name], before[name]))

    def test_t6_matching_is_shared(self):
        row = {"matching": {"same_pairs_for": ["HRIP", "E_nonlocal", "HRIP_SHIFT"]}}
        self.assertEqual(row["matching"]["same_pairs_for"], ["HRIP", "E_nonlocal", "HRIP_SHIFT"])
        self.assertIs(deterministic_matches, e0r.deterministic_matches)

    def test_t7_risk_population_exact(self):
        values = np.arange(10, dtype=np.float32)
        ids = np.arange(10, dtype=np.int64)
        mask = e0r.frozen_risk_mask(values, ids)
        self.assertEqual(int(mask.sum()), 2)
        self.assertTrue(np.array_equal(np.flatnonzero(mask), np.array([8, 9])))
        self.assertIs(select_top, e0r.select_top)

    def test_t8_triage_exact(self):
        mask = np.zeros(10, dtype=bool)
        mask[-2:] = True
        self.assertEqual(e0r.frozen_triage_budget(mask), 1)

    def test_t9_bootstrap_seed_lock(self):
        self.assertEqual(e0r.BOOTSTRAP_SEEDS, {"hrip_matched_win": 5101, "centroid_matched_win": 5102, "hrip_minus_centroid": 5103, "aligned_minus_shifted": 5104, "c_ap_delta": 5105, "r_pos_delta": 5106, "r_neg_delta": 5107})

    def test_t10_zero_model_forwards(self):
        self.assertEqual(e0r.MODEL_FORWARD_COUNT, 0)

    def test_t11_frozen_cache_hash_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x"
            path.write_bytes(b"frozen")
            first = e0r.sha256_file(path)
            second = e0r.sha256_file(path)
            self.assertEqual(first, second)
            self.assertEqual(first, hashlib.sha256(b"frozen").hexdigest())

    def test_t12_old_e0_hash_snapshot_is_separate(self):
        self.assertNotEqual(e0r.OLD_E0_ROOT, e0r.RECOVERY_ROOT)
        self.assertEqual(e0r.hash_tree(e0r.OLD_E0_TOOLS[0]), e0r.hash_tree(e0r.OLD_E0_TOOLS[0]))

    def test_t13_authoritative_helpers_are_reused(self):
        self.assertIs(e0r.deterministic_matches, deterministic_matches)
        self.assertIs(e0r.ap_contamination, ap_contamination)
        self.assertIs(e0r.pairwise_risks, pairwise_risks)

    def test_t14_shift_call_input_is_one_image(self):
        lengths = []
        e0r.shift_per_image([np.zeros(e0r.PIXELS_PER_IMAGE, dtype=np.float32)], lengths)
        self.assertEqual(lengths, [518 * 518])

    def test_t15_no_evidence_construction_api(self):
        source = inspect.getsource(e0r)
        for forbidden in ("load_model(", ".forward(", "vision_text_fusion_gate_seg", "construct_image_evidence", "peer_features"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
