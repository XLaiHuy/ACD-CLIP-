from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import sabra.trust_v2.visa_audit as audit  # noqa: E402


class TrustV2RecoveryV2Test(unittest.TestCase):
    def test_recovery_root_is_exact_and_old_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            expected = Path(temp) / "TRUST_V2_M4_RECOVERY_V2"
            with patch.object(audit, "RECOVERY_ROOT", expected):
                audit.configure_output_root(expected)
                self.assertEqual(audit.TRUST_ROOT, expected.resolve())
                with self.assertRaisesRegex(RuntimeError, "TRUST_V2_RECOVERY_OUTPUT_ROOT_INVALID"):
                    audit.configure_output_root(audit.DEVELOPMENT_ROOT)

    def test_existing_recovery_result_path_is_a_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            expected = Path(temp) / "TRUST_V2_M4_RECOVERY_V2"
            expected.mkdir()
            (expected / "DECISION.json").write_text("invalid placeholder")
            with patch.object(audit, "RECOVERY_ROOT", expected):
                with self.assertRaisesRegex(RuntimeError, "ARTIFACT_PATH_COLLISION"):
                    audit.configure_output_root(expected)

    def test_atomic_json_writer_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "DECISION.json"
            audit._write_json(path, {"status": "first"})
            with self.assertRaisesRegex(RuntimeError, "ARTIFACT_PATH_COLLISION"):
                audit._write_json(path, {"status": "second"})
            self.assertEqual(path.read_text().strip().find("first") >= 0, True)

    def test_recovery_runner_has_no_scientific_redefinition(self) -> None:
        source = (ROOT / "tools/sabra/trust_v2/recovery_v2.py").read_text()
        self.assertIn("configure_output_root", source)
        self.assertIn("visa_audit.run()", source)
        self.assertNotIn("PCRR", source)
        self.assertNotIn("D_rel", source)


if __name__ == "__main__":
    unittest.main()
