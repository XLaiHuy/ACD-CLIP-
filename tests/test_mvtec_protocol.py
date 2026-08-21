from __future__ import annotations

import json

from PIL import Image

from evaluation.datasets import EXPECTED_MVTEC_CLASSES, MVTecDatasetAdapter


def test_mvtec_preflight_adapter(tmp_path):
    rows = []
    for class_name in EXPECTED_MVTEC_CLASSES:
        image_dir = tmp_path / class_name / "test" / "good"
        mask_dir = tmp_path / class_name / "ground_truth" / "defect"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        good = image_dir / "000.png"
        bad = tmp_path / class_name / "test" / "defect" / "001.png"
        bad.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), color=(0, 0, 0)).save(good)
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(bad)
        mask = mask_dir / "001_mask.png"
        Image.new("L", (8, 8), color=255).save(mask)
        rows.extend([
            {"image_path": str(good.relative_to(tmp_path)), "label": 0, "class_name": class_name},
            {"image_path": str(bad.relative_to(tmp_path)), "label": 1, "mask_path": str(mask.relative_to(tmp_path)), "class_name": class_name},
        ])
    metadata = tmp_path / "MVTec.jsonl"
    metadata.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = MVTecDatasetAdapter(tmp_path, metadata_path=metadata, image_size=8).preflight(inspect_limit=1)
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["model_inference"] is False
    assert tuple(report["class_names"]) == tuple(sorted(EXPECTED_MVTEC_CLASSES))
