import unittest

import torch
from torch import nn

from phase2c_pcgrad import project_group
from phase2c_utils import normalized_config, phase2c_config


class Phase2CPcgradTests(unittest.TestCase):
    def test_p_matches_a_prime_except_pcgrad_metadata(self):
        a = phase2c_config("A_prime", "a", .20)
        p = phase2c_config("P", "p", .20)
        for key, value in a.items():
            if key not in {"condition", "save_path", "parent_condition", "pcgrad_enabled", "pcgrad_groups", "pcgrad_variant", "pcgrad_epsilon", "precision"}:
                self.assertEqual(value, p[key])

    def test_positive_gradient_is_exact_sum(self):
        parameter = nn.Parameter(torch.zeros(2))
        final, stats = project_group([("p", parameter)], [torch.tensor([1., 2.])], [torch.tensor([3., 4.])], [None])
        self.assertFalse(stats["projection_applied"])
        self.assertTrue(torch.allclose(final[0], torch.tensor([4., 6.])))

    def test_negative_gradient_projects_and_none_zero_are_safe(self):
        parameter = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        final, stats = project_group([("p", parameter)], [torch.tensor([1., 0.])], [torch.tensor([-1., 1.])], [None])
        self.assertTrue(stats["projection_applied"])
        self.assertGreaterEqual(stats["post_projection_cosine"], -1e-6)
        self.assertEqual(final[0].dtype, torch.float32)
        _, zero = project_group([("p", parameter)], [None], [None], [None])
        self.assertEqual(zero["number_of_valid_gradient_tensors"], 0)


if __name__ == "__main__":
    unittest.main()
