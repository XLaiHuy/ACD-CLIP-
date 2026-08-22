from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.metrics import binary_average_precision, binary_auroc
from tools.sabra_car.r0_direction import (
    ALPHAS,
    PATCHES,
    canonical_loss_per_image,
    classify_actions,
    coordinate_radius_for_image,
    exact_metrics,
    intervention_delta,
    informative_coordinates,
    select_signed_alpha,
    write_csv,
)
from utils import calculate_seg_loss


def test_signed_action_thresholds_are_exact():
    values = np.array([-2e-8, -1e-8, 0.0, 1e-8, 2e-8], dtype=np.float32)
    assert classify_actions(values).tolist() == [-1, 0, 0, 0, 1]
    tensor = torch.from_numpy(values)
    assert classify_actions(tensor).tolist() == [-1, 0, 0, 0, 1]


def test_informative_coordinates_are_largest_and_stable():
    utility = np.array([[0.1, -0.4, 0.4], [0.2, -0.3, 0.0]], dtype=np.float32)
    assert informative_coordinates(utility, count=3) == [(0, 1), (0, 2), (1, 1)]



def test_csv_writer_uses_lf_line_endings(tmp_path):
    output = tmp_path / "rows.csv"
    write_csv(output, [{"name": "candle", "value": 1}])
    payload = output.read_bytes()
    assert payload == b"name,value\ncandle,1\n"
    assert b"\r" not in payload


def test_intervention_is_abnormal_only_and_shared():
    native = torch.randn(3, 2, PATCHES, 2)
    correction = torch.randn(2, PATCHES)
    delta = intervention_delta(correction, native)
    assert torch.count_nonzero(delta[..., 0]).item() == 0
    assert torch.equal(delta[0, ..., 1], correction)
    assert torch.equal(delta[0], delta[1])
    assert torch.equal(delta[1], delta[2])


def test_per_image_loss_matches_canonical_batch_mean():
    torch.manual_seed(7)
    logits = torch.randn(3, 2, 2, 9, 11)
    probability = torch.softmax(logits.mean(0), dim=1)
    mask = (torch.rand(2, 1, 9, 11) > 0.7).float()
    per_image = canonical_loss_per_image(probability, mask)
    canonical = calculate_seg_loss(probability, mask)
    assert torch.allclose(per_image.mean(), canonical, atol=2e-7, rtol=2e-7)


def test_vectorized_metrics_match_canonical_tie_semantics():
    scores = np.array([0.1, 0.4, 0.4, 0.8, 0.2, 0.8], dtype=np.float32)
    labels = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
    observed = exact_metrics(scores, labels)
    assert observed["pAP"] == pytest.approx(binary_average_precision(scores, labels), abs=1e-15)
    assert observed["pAUROC"] == pytest.approx(binary_auroc(scores, labels), abs=1e-15)


def test_signed_alpha_tie_selects_smaller_value():
    macros = [{"condition": "native", "macro_pAP": 0.5, "macro_pAUROC": 0.7, "mean_loss": 1.0}]
    for alpha in ALPHAS[1:]:
        macros.append({"condition": f"signed_alpha_{alpha:g}", "macro_pAP": 0.6, "macro_pAUROC": 0.7, "mean_loss": 0.9})
    assert select_signed_alpha(macros) == 0.125


def test_coordinate_radius_zero_action_and_zero_basis_choose_zero():
    margin = torch.zeros(4)
    mask = torch.tensor([0.0, 1.0, 0.0, 1.0])
    actions = torch.zeros(PATCHES, dtype=torch.int8)
    actions[0] = 1
    indices = torch.zeros((PATCHES, 1), dtype=torch.long)
    values = torch.zeros((PATCHES, 1), dtype=torch.float32)
    valid = torch.ones((PATCHES, 1), dtype=torch.bool)
    correction = coordinate_radius_for_image(margin, mask, actions, indices, values, valid, patch_batch=128)
    assert torch.count_nonzero(correction).item() == 0


def test_invalid_intervention_shapes_fail_closed():
    native = torch.randn(3, 1, PATCHES, 2)
    with pytest.raises(ValueError):
        intervention_delta(torch.zeros(1, PATCHES - 1), native)
