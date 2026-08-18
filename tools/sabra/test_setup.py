"""Small runtime tests for the SABRA setup-only data boundaries."""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from sabra.data import VisaEvaluationDataset, VisaEvidenceDataset, read_visa_metadata  # noqa: E402


class SetupDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
        parser.add_argument("--data-root", type=Path, default=Path("/workspace/data/VisA_20220922"))
        args, _ = parser.parse_known_args()
        cls.rows = read_visa_metadata(args.metadata)
        cls.data_root = args.data_root.resolve()

    def test_image_and_mask_are_deterministic(self) -> None:
        evidence = VisaEvidenceDataset(self.rows, self.data_root)
        first = evidence[0]["image"]
        second = evidence[0]["image"]
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(float((first - second).abs().max()), 0.0)

        anomaly_index = next(i for i, row in enumerate(self.rows) if int(row["label"]) == 1)
        evaluation = VisaEvaluationDataset(self.rows, self.data_root)
        first_eval = evaluation[anomaly_index]
        second_eval = evaluation[anomaly_index]
        self.assertTrue(torch.equal(first_eval["image"], second_eval["image"]))
        self.assertTrue(torch.equal(first_eval["mask"], second_eval["mask"]))
        self.assertEqual(float((first_eval["mask"] - second_eval["mask"]).abs().max()), 0.0)
        self.assertTrue(set(first_eval["mask"].unique().tolist()).issubset({0.0, 1.0}))

    def test_gt_free_path_cannot_read_mask_pixels(self) -> None:
        evidence = VisaEvidenceDataset(self.rows, self.data_root)
        real_open = PILImage.open

        def guarded_open(path, *args, **kwargs):
            path_text = str(path)
            if "/Masks/" in path_text or Path(path_text).suffix.lower() in {".png", ".bmp", ".tif", ".tiff"}:
                raise AssertionError(f"GT-free path attempted mask read: {path_text}")
            return real_open(path, *args, **kwargs)

        with mock.patch.object(PILImage, "open", side_effect=guarded_open):
            sample = evidence[0]
        self.assertNotIn("label", sample)
        self.assertNotIn("mask", sample)
        self.assertNotIn("mask_path", sample)
        self.assertEqual(tuple(sample["image"].shape), (3, 518, 518))


if __name__ == "__main__":
    unittest.main()
