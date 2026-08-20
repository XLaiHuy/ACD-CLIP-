from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from sabra.trust_v2 import mvtec_external
from sabra.trust_v2.numerical import percentile_rank as frozen_percentile_rank


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


def _need_fixture() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    native_margins = np.asarray(
        [
            [1.0, 1.0, 2.0, 3.0, 3.0, 4.0],
            [1.0, 1.0, 2.0, 3.0, 3.0, 4.0],
            [1.0, 1.0, 2.0, 3.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )
    record = {
        "baseline_pgm": np.linspace(0.1, 0.6, 6, dtype=np.float32),
        "peer_coherence": np.linspace(0.9, 0.95, 6, dtype=np.float32),
        "query_support_mean": np.linspace(0.8, 0.9, 6, dtype=np.float32),
        "peer_eigen_entropy": np.linspace(0.5, 0.7, 6, dtype=np.float32),
        "stage_query_profile_disagreement": np.linspace(0.01, 0.06, 6, dtype=np.float32),
        "D_rank": np.linspace(0.05, 0.3, 6, dtype=np.float32),
    }
    sensitivity = np.linspace(0.001, 0.006, 6, dtype=np.float32)
    return record, native_margins, sensitivity


def _need_parameters() -> dict[str, object]:
    return {
        "scaler_mean": [0.5, 0.0, 0.17, 0.003],
        "scaler_scale": [0.3, 2.0, 0.1, 0.002],
        "logistic_coef": [[0.5, 0.1, -0.2, 0.7]],
        "logistic_intercept": [-0.1],
    }


def _trust_parameters() -> dict[str, object]:
    return {
        "scaler_mean": [0.5, 0.9, 0.85, 0.6, 0.03],
        "scaler_scale": [0.2, 0.02, 0.1, 0.1, 0.02],
        "logistic_coef": [[0.4, -0.1, 0.2, -0.3, 0.5]],
        "logistic_intercept": [0.0],
    }


def test_need_feature_order_is_exact() -> None:
    assert mvtec_external.NEED_ORDER == (
        "margin_within_image_rank",
        "robust_margin_normalization",
        "D_rank",
        "deployment_sensitivity",
    )


def test_percentile_rank_ties_match_authoritative_path() -> None:
    record, native_margins, sensitivity = _need_fixture()
    actual = mvtec_external._need_feature_matrix(record, native_margins, sensitivity)
    expected_rank = frozen_percentile_rank(native_margins.mean(axis=0)).astype(np.float32)
    np.testing.assert_array_equal(actual[:, 0], expected_rank)
    np.testing.assert_array_equal(
        expected_rank,
        np.asarray([0.1, 0.1, 0.4, 0.7, 0.7, 1.0], dtype=np.float32),
    )


def test_deployment_sensitivity_wiring_is_unchanged() -> None:
    record, native_margins, sensitivity = _need_fixture()
    actual = mvtec_external._need_feature_matrix(record, native_margins, sensitivity)
    np.testing.assert_array_equal(actual[:, 3], sensitivity)


def test_need_feature_and_score_parity() -> None:
    record, native_margins, sensitivity = _need_fixture()
    mean_margin = native_margins.mean(axis=0)
    median = np.median(mean_margin)
    robust = (mean_margin - median) / (np.median(np.abs(mean_margin - median)) + 1e-6)
    expected = np.column_stack(
        [
            frozen_percentile_rank(mean_margin).astype(np.float32),
            robust.astype(np.float32),
            record["D_rank"],
            sensitivity,
        ]
    )
    actual = mvtec_external._need_feature_matrix(record, native_margins, sensitivity)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    freeze = {
        "trust_model": {"trust_model_parameters": _trust_parameters()},
        "need_c1_model_parameters": _need_parameters(),
    }
    scored = mvtec_external._score_record(record, freeze, native_margins, sensitivity)
    np.testing.assert_allclose(
        scored["Need_C1"],
        mvtec_external.frozen_probability(_need_parameters(), expected),
        rtol=0,
        atol=1e-7,
    )


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
