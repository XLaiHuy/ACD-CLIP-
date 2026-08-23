from __future__ import annotations

import numpy as np

from tools.sabra_cure import r1
from tools.sabra_cure.r2 import (
    MIS_COVERAGES,
    conservative_index,
    interval_actions,
    risk_metrics,
    select_operating_point,
    signed_direction,
)


def test_r2_uses_exact_frozen_r1_feature_contract():
    assert len(r1.FEATURE_ORDER) == 14
    assert r1.FEATURE_ORDER[0] == "margin_within_image_rank"
    assert r1.FEATURE_ORDER[-1].startswith("robust_peer_signed_margin_consensus")


def test_conservative_quantile_index_is_frozen_and_bounded():
    count = 9
    assert [conservative_index(count, m) for m in MIS_COVERAGES] == [9 - 1, 8, 7, 6, 5]


def test_interval_boundary_is_keep():
    mu = np.array([1.0, -1.0, 0.0, 2.0])
    sigma = np.array([1.0, 1.0, 1.0, 0.5])
    assert interval_actions(mu, sigma, 1.0).tolist() == [0, 0, 0, 1]


def test_selector_prefers_coverage_then_risk_then_smaller_miscoverage():
    # The three unsafe signs have the largest residuals; the selector chooses
    # the eligible candidate with the greatest safe coverage.
    mu = np.array([0.1, 0.2, 0.3, -1.0, -2.0, -3.0, 3.0, -0.5, 2.5, -2.5])
    y = np.array([-0.9, -1.8, -2.7, -1.0, -2.0, -3.0, 3.0, -0.5, 2.5, -2.5])
    sigma = np.full_like(mu, 0.1)
    selected, evidence = select_operating_point(mu, y, sigma)
    assert selected is not None
    assert selected["miscoverage"] == 0.30
    assert evidence["selection_status"] == "QUALIFIED"


def test_selector_uses_smaller_miscoverage_when_safe_coverage_ties():
    mu = np.array([0.1, 0.2, 0.3, 4.0, -4.0, 3.5, -3.5, 4.5, -4.5, 5.0])
    y = np.array([-0.9, -1.8, -2.7, 4.0, -4.0, 3.5, -3.5, 4.5, -4.5, 5.0])
    selected, _ = select_operating_point(mu, y, np.full_like(mu, 0.1))
    assert selected is not None
    assert selected["miscoverage"] == min(MIS_COVERAGES)


def test_risk_metrics_excludes_keep_from_accepted_wrong_sign_rate():
    actions = np.array([1, -1, 0, 1], dtype=np.int8)
    y = np.array([1.0, 1.0, -1.0, -1.0])
    result = risk_metrics(actions, y, signed_direction(y))
    assert result["acted"] == 3
    assert result["wrong_sign"] == 2
    assert result["opposite_sign_rate"] == 2 / 3


def test_full_and_accumulated_ridge_parity_remains_exact():
    assert r1.parity_fixture()["status"] == "PASS"
