#!/usr/bin/env python3
"""Pre-GT structural tests for P5FR1CE1 evaluator recovery.

These tests intentionally stop before class evaluation. They read only the
frozen input lock, canonical config document, and GT-free manifest. No GT mask,
image, scientific result, model, or evaluator metric may be opened/computed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = ROOT / "runs/phase5/hsir/P5FR1C_MVTEC_LATE_COMPLETION"
DERIVE_STATUS = Path("/tmp/p5fr1c_all_config_evidence/DERIVE_STATUS.json")
EVALUATOR_PATH = ROOT / "tools/audit_p5fr1c_mvtec_posthoc.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("p5fr1ce1_evaluator", EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load committed evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StopBeforeGT(RuntimeError):
    pass


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator = load_evaluator()

    def frozen_ids(self) -> list[str]:
        _, rows, ids = self.evaluator.load_configs()
        self.assertEqual(len(rows), 26)
        self.assertEqual(len(ids), 26)
        self.assertEqual(len(set(ids)), 26)
        return ids

    def test_T01_load_configs_exact_26_unique_ids(self) -> None:
        families, rows, ids = self.evaluator.load_configs()
        self.assertEqual([len(families[name]) for name in self.evaluator.FAMILIES], [8, 8, 6, 4])
        self.assertEqual(len(rows), 26)
        self.assertEqual(len(ids), 26)
        self.assertEqual(len(set(ids)), 26)

    def test_T02_rows_reconstruct_exact_returned_ids(self) -> None:
        _, rows, ids = self.evaluator.load_configs()
        reconstructed = [cfg["config_id"] for _, cfg in rows]
        self.assertEqual(reconstructed, ids)

    def test_T03_integrity_subchecks_completes_without_type_error(self) -> None:
        ids = self.frozen_ids()
        checks = self.evaluator.integrity_subchecks(ids)
        self.assertIsInstance(checks, dict)
        self.assertTrue(checks["derived_config_count"])

    def test_T04_integrity_config_order_is_exact(self) -> None:
        _, rows, ids = self.evaluator.load_configs()
        checks = self.evaluator.integrity_subchecks([cfg["config_id"] for _, cfg in rows])
        self.assertTrue(checks["derived_config_count"])
        self.assertEqual([cfg["config_id"] for _, cfg in rows], ids)

    def test_T05_startup_reaches_pre_gt_gate_without_metrics(self) -> None:
        gate_calls: list[list[str]] = []
        original_gate = self.evaluator.integrity_subchecks

        def record_gate(config_ids: list[str]) -> dict[str, bool]:
            gate_calls.append(list(config_ids))
            return original_gate(config_ids)

        with mock.patch.object(self.evaluator, "integrity_subchecks", side_effect=record_gate):
            with mock.patch.object(self.evaluator, "class_common", side_effect=StopBeforeGT):
                with mock.patch.object(self.evaluator, "load_mask", side_effect=AssertionError("mask opened")):
                    with mock.patch.object(self.evaluator, "deploy_native_logits", side_effect=AssertionError("model forward")):
                        with mock.patch.object(self.evaluator, "exact_auc_ap", side_effect=AssertionError("metric computed")):
                            with self.assertRaises(StopBeforeGT):
                                self.evaluator.evaluate()
        self.assertEqual(len(gate_calls), 1)
        self.assertEqual(len(gate_calls[0]), 26)

    def test_T06_gt_mask_read_counter_remains_zero(self) -> None:
        ids = self.frozen_ids()
        counters = {"load_mask": 0, "image_open": 0}

        def forbidden_mask(*args, **kwargs):
            counters["load_mask"] += 1
            raise AssertionError("mask read during pre-GT test")

        def forbidden_image(*args, **kwargs):
            counters["image_open"] += 1
            raise AssertionError("image opened during pre-GT test")

        with mock.patch.object(self.evaluator, "load_mask", side_effect=forbidden_mask):
            with mock.patch.object(self.evaluator.Image, "open", side_effect=forbidden_image):
                self.evaluator.integrity_subchecks(ids)
        self.assertEqual(counters, {"load_mask": 0, "image_open": 0})

    def test_T07_model_forwards_remain_zero(self) -> None:
        status = json.loads(DERIVE_STATUS.read_text())
        manifest = json.loads((NAMESPACE / "GT_FREE_DERIVED_MANIFEST.json").read_text())
        self.assertEqual(status["model_forwards"], 0)
        self.assertEqual(manifest["model_forwards"], 0)
        self.assertFalse(status["gt_accessed"])
        self.assertFalse(status["mask_accessed"])
        with mock.patch.object(self.evaluator, "deploy_native_logits", side_effect=AssertionError("model forward")):
            self.evaluator.integrity_subchecks(self.frozen_ids())


if __name__ == "__main__":
    unittest.main(verbosity=2)
