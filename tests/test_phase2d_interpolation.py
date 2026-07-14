import copy
import unittest

import torch

from phase2d_interpolation import interpolate_payload, validate_compatible_payloads


def payload(value, integer=7):
    return {
        "epoch": 13,
        "condition": "A_prime",
        "image_adapter": {"weight": torch.tensor([value, value + 2], dtype=torch.float16), "count": torch.tensor(integer, dtype=torch.int64)},
        "text_adapter": {"weight": torch.tensor([[value]], dtype=torch.float32)},
        "soft_prompt": {"weight": torch.tensor([value], dtype=torch.float64)},
        "optimizer_state": {"not": "resumable"},
    }


class Phase2DInterpolationTests(unittest.TestCase):
    def make(self, lambda_b):
        return interpolate_payload(payload(2), payload(6), lambda_b, "test", "a.pth", "a" * 64, "b.pth", "b" * 64, "commit", "2026-01-01T00:00:00+00:00")

    def test_endpoints_and_midpoint_use_fp32_then_original_dtype(self):
        self.assertTrue(torch.equal(self.make(0)["image_adapter"]["weight"], payload(2)["image_adapter"]["weight"]))
        self.assertTrue(torch.equal(self.make(1)["image_adapter"]["weight"], payload(6)["image_adapter"]["weight"]))
        midpoint = self.make(0.5)["image_adapter"]["weight"]
        self.assertTrue(torch.equal(midpoint, torch.tensor([4, 6], dtype=torch.float16)))
        self.assertEqual(midpoint.dtype, torch.float16)

    def test_inputs_are_not_mutated_and_nonfloating_is_copied(self):
        left, right = payload(2), payload(6)
        left_before, right_before = copy.deepcopy(left), copy.deepcopy(right)
        candidate = interpolate_payload(left, right, 0.5, "test", "a", "a" * 64, "b", "b" * 64, "commit", "time")
        self.assertEqual(left["optimizer_state"], left_before["optimizer_state"])
        self.assertTrue(torch.equal(left["image_adapter"]["weight"], left_before["image_adapter"]["weight"]))
        self.assertTrue(torch.equal(right["text_adapter"]["weight"], right_before["text_adapter"]["weight"]))
        self.assertTrue(torch.equal(candidate["image_adapter"]["count"], left["image_adapter"]["count"]))
        self.assertNotIn("optimizer_state", candidate)

    def test_contract_failures_raise(self):
        left, right = payload(2), payload(6)
        right["text_adapter"]["extra"] = torch.tensor([1.0])
        with self.assertRaises(ValueError):
            validate_compatible_payloads(left, right)
        right = payload(6)
        right["text_adapter"]["weight"] = torch.zeros(2)
        with self.assertRaises(ValueError):
            validate_compatible_payloads(left, right)
        right = payload(6, integer=8)
        with self.assertRaises(ValueError):
            validate_compatible_payloads(left, right)

    def test_metadata_and_determinism(self):
        first = self.make(0.25)
        second = self.make(0.25)
        self.assertEqual(first["phase2d_interpolation"], second["phase2d_interpolation"])
        self.assertEqual(first["phase2d_interpolation"]["candidate_name"], "test")
        self.assertFalse(first["phase2d_interpolation"]["training_performed"])
        self.assertFalse(first["phase2d_interpolation"]["optimizer_state_interpolated"])
        self.assertEqual(first["phase2d_interpolation"]["floating_tensor_count"], 3)
        self.assertEqual(first["phase2d_interpolation"]["non_floating_tensor_count"], 1)


if __name__ == "__main__":
    unittest.main()
