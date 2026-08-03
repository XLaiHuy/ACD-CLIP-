import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_phase4_medical_splits import stratified_split_rows


def test_stratified_split_is_deterministic_and_keeps_complete_rows():
    rows = [
        {"image_path": f"good/{index}.png", "label": 0, "class_name": "Demo"}
        for index in range(10)
    ] + [
        {
            "image_path": f"ungood/{index}.png",
            "mask_path": f"label/{index}.png",
            "label": 1,
            "class_name": "Demo",
        }
        for index in range(10)
    ]

    val_a, test_a = stratified_split_rows(rows, 0.30, 0, "Demo")
    val_b, test_b = stratified_split_rows(rows, 0.30, 0, "Demo")

    assert val_a == val_b
    assert test_a == test_b
    assert len(val_a) == 6
    assert len(test_a) == 14
    assert {row["image_path"] for row in val_a}.isdisjoint({row["image_path"] for row in test_a})
    assert all(row["mask_path"].startswith("label/") for row in val_a if row["label"] == 1)
    assert sum(row["label"] == 0 for row in val_a) == 3
    assert sum(row["label"] == 1 for row in val_a) == 3
