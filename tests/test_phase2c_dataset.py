import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from dataset import TextAndImageDataset, deterministic_sample_id


class Phase2CDatasetTests(unittest.TestCase):
    def test_manifest_filter_and_deterministic_validation_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = "item/Data/Images/Normal/001.png"
            full = root / image_path
            full.parent.mkdir(parents=True)
            Image.new("RGB", (12, 9), (20, 80, 140)).save(full)
            record = {"image_path": image_path, "label": 0, "class_name": "item"}
            meta = root / "meta.jsonl"
            meta.write_text(json.dumps(record) + "\n")
            manifest = root / "manifest.csv"
            with open(manifest, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id"])
                writer.writeheader()
                writer.writerow({"sample_id": deterministic_sample_id(record)})
            dataset = TextAndImageDataset(
                str(root), str(meta), 16, manifest_path=str(manifest), augment=False
            )
            first = dataset[0]
            second = dataset[0]
            self.assertEqual(first["sample_id"], deterministic_sample_id(record))
            self.assertTrue(torch.equal(first["image"], second["image"]))
            self.assertTrue(torch.equal(first["mask"], second["mask"]))


if __name__ == "__main__":
    unittest.main()
