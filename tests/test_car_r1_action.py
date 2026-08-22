from __future__ import annotations

import numpy as np
import pytest

from tools.sabra_car.r1_common import (
    FEATURE_ORDER,
    PATCHES,
    apply_robust_scaler,
    classify_utility,
    fit_robust_scaler,
    select_threshold,
    stable_argmax_predictions,
    stack_features,
    threshold_actions,
    threshold_landscape,
    write_csv,
)


def test_r1_feature_order_and_supported_stability_masks_are_exact():
    shape = (1, PATCHES)
    source = {
        name: np.full(shape, index + 1, dtype=np.float32)
        for index, name in enumerate(FEATURE_ORDER[:9])
    }
    valid_p9 = np.ones(shape, dtype=bool)
    valid_p16 = np.ones(shape, dtype=bool)
    valid_p9[0, 0] = False
    valid_p16[0, 1] = False
    trust = {
        "valid_p9": valid_p9,
        "valid_p16": valid_p16,
        "S9": np.full(shape, 0.9, dtype=np.float32),
        "S16": np.full(shape, 0.8, dtype=np.float32),
    }
    features = stack_features(source, trust)
    assert features.shape == (1, PATCHES, 11)
    assert features[0, 2].tolist() == pytest.approx(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0.9, 0.8]
    )
    assert features[0, 0, 9] == 0.0
    assert features[0, 1, 10] == 0.0


def test_fold_robust_scaler_uses_linear_quantiles_and_iqr_floor():
    values = np.tile(np.arange(4, dtype=np.float64)[:, None], (1, 11))
    values[:, -1] = 7.0
    median, iqr = fit_robust_scaler(values)
    assert median[:-1].tolist() == pytest.approx([1.5] * 10)
    assert iqr[:-1].tolist() == pytest.approx([1.5] * 10)
    assert median[-1] == 7.0
    assert iqr[-1] == 1e-6
    transformed = apply_robust_scaler(np.array([[1.5] * 10 + [7.0]]), median, iqr)
    assert np.array_equal(transformed, np.zeros((1, 11)))


def test_stable_argmax_and_threshold_abstention_contract():
    probability = np.array([[0.5, 0.5, 0.0], [0.1, 0.2, 0.7]], dtype=np.float64)
    prediction, confidence = stable_argmax_predictions(
        probability, np.array([-1, 0, 1], dtype=np.int8)
    )
    assert prediction.tolist() == [-1, 1]
    assert threshold_actions(prediction, confidence, 0.6).tolist() == [0, 1]


def test_threshold_selection_uses_lowest_risk_qualified_threshold():
    oracle = np.r_[np.ones(50, dtype=np.int8), -np.ones(50, dtype=np.int8)]
    probability = np.zeros((100, 3), dtype=np.float64)
    probability[:50] = [0.05, 0.05, 0.90]
    probability[50:] = [0.30, 0.25, 0.45]
    _, _, rows = threshold_landscape(
        oracle, probability, np.array([-1, 0, 1], dtype=np.int8)
    )
    selected = select_threshold(rows)
    assert selected == 0.5
    selected_row = next(row for row in rows if row["threshold"] == selected)
    assert selected_row["coverage"] == 0.5
    assert selected_row["opposite_sign_rate"] == 0.0
    assert selected_row["relative_opposite_sign_reduction"] == 1.0


def test_r1_utility_threshold_and_csv_lf(tmp_path):
    utility = np.array([-2e-8, -1e-8, 0.0, 1e-8, 2e-8], dtype=np.float32)
    assert classify_utility(utility).tolist() == [-1, 0, 0, 0, 1]
    output = tmp_path / "rows.csv"
    write_csv(output, [{"threshold": 0.5, "pass": True}])
    payload = output.read_bytes()
    assert b"\r" not in payload
    assert payload.endswith(b"\n")
