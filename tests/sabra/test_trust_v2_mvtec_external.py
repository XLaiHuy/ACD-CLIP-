from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from sabra.trust_v2 import mvtec_external


def test_frozen_probability_is_deterministic_and_does_not_fit() -> None:
    params = {
        "scaler_mean": [0.0, 1.0],
        "scaler_scale": [2.0, 4.0],
        "logistic_coef": [[2.0, -1.0]],
        "logistic_intercept": [0.5],
    }
    values = np.asarray([[0.0, 1.0], [2.0, 5.0]], dtype=np.float32)
    expected = 1.0 / (1.0 + np.exp(-np.asarray([0.5, 1.5])))
    first = mvtec_external.frozen_probability(params, values)
    second = mvtec_external.frozen_probability(params, values)
    np.testing.assert_allclose(first, expected, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(first, second)


def test_gt_free_rows_drop_labels_and_masks() -> None:
    path = Path("/tmp/mvtec-metadata-test.jsonl")
    rows = [
        {
            "class_name": mvtec_external.EXPECTED_CLASSES[index % len(mvtec_external.EXPECTED_CLASSES)],
            "image_path": f"synthetic/{index:04d}.png",
            "label": 0,
            "mask_path": f"synthetic/mask-{index:04d}.png",
        }
        for index in range(1725)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    rows = mvtec_external._gt_free_rows(path)
    assert all(set(row) == {"class_name", "image_path"} for row in rows)


def test_gt_free_stage_contains_no_label_or_mask_access() -> None:
    tree = ast.parse(Path(mvtec_external.__file__).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_gt_free_stage")
    names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)}
    assert "label" not in names
    assert "mask" not in names
    assert "mask_path" not in attributes


def test_decision_ladder_uses_class_proportions() -> None:
    assert mvtec_external._decision_ladder(np.asarray([0.011, 0.012, 0.013])) == "SUPPORTED"
    assert mvtec_external._decision_ladder(np.asarray([-0.04, 0.02, 0.02])) == "FALSIFIED"
