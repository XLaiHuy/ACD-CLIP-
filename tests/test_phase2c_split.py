import csv
import json
import tempfile
import unittest
from pathlib import Path

from phase2c_split import (
    assign_groups, build_split, load_source_records, read_manifest,
    split_records, validate_integrity, verify_split,
)


class Phase2CSplitTests(unittest.TestCase):
    def records(self):
        rows = []
        for category in ("alpha", "beta"):
            for label, folder in ((0, "Normal"), (1, "Anomaly")):
                for index in range(10):
                    rows.append({
                        "image_path": f"{category}/Data/Images/{folder}/{index:03d}.JPG",
                        "mask_path": f"{category}/Data/Masks/{folder}/{index:03d}.png" if label else "",
                        "label": label,
                        "class_name": category,
                        "series_id": f"{folder}-{index // 2}",
                    })
        return rows

    def write_source(self, directory):
        path = Path(directory) / "VisA.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in self.records()))
        return path

    def test_integrity_coverage_and_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_source(tmp)
            records1 = assign_groups(load_source_records(source))
            train1, val1 = split_records(records1, seed=42)
            records2 = assign_groups(load_source_records(source))
            train2, val2 = split_records(records2, seed=42)
            validate_integrity(records1, train1, val1)
            self.assertEqual(
                [row["sample_id"] for row in train1],
                [row["sample_id"] for row in train2],
            )
            self.assertEqual(
                [row["sample_id"] for row in val1],
                [row["sample_id"] for row in val2],
            )
            self.assertFalse(
                {row["group_id"] for row in train1} & {row["group_id"] for row in val1}
            )
            self.assertEqual(
                {(row["class_name"], row["label"]) for row in val1},
                {("alpha", 0), ("alpha", 1), ("beta", 0), ("beta", 1)},
            )

    def test_cli_artifacts_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_source(tmp)
            output = Path(tmp) / "splits"
            train, val, metadata = build_split(source, output, seed=42)
            self.assertTrue(verify_split(source, train, val, metadata))
            self.assertEqual(len(read_manifest(train)) + len(read_manifest(val)), len(self.records()))
            payload = json.loads(metadata.read_text())
            self.assertEqual(payload["source"]["total"], len(self.records()))


if __name__ == "__main__":
    unittest.main()
