import unittest

from tools.audit_phase5_p5e0_hrip import run_synthetic_tests


class P5E0HripSyntheticTests(unittest.TestCase):
    def test_synthetic_structural_suite(self):
        result = run_synthetic_tests()
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["test_count"], 20)
        self.assertTrue(result["no_official_visA_forward"])


if __name__ == "__main__":
    unittest.main()
