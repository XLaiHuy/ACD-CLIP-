import json

import numpy as np
import pytest

from tools.sabra_cure import post_r2v2_diagnostic as p12
from tools.sabra_cure import post_r2v2_diagnostic_recovery as recovery


def test_exact_parent_and_frozen_contract_are_present():
    assert recovery.git("rev-parse", recovery.P12_TERMINAL) == recovery.P12_TERMINAL
    assert recovery.git("rev-parse", recovery.P12_PREREG) == recovery.P12_PREREG
    assert set(recovery.contract_hashes()) == set(recovery.CONTRACT)
    assert recovery.p12_immutable()


def test_p12_stop_and_single_historical_marker_are_preserved():
    summary = json.loads((recovery.ROOT / "results/sabra_cure/post_r2v2_diagnostic/summary.json").read_text())
    marker = json.loads((recovery.ROOT / "results/sabra_cure/post_r2v2_diagnostic/ATTEMPT_STARTED.json").read_text())
    assert summary["status"] == "DIAGNOSTIC_ENGINEERING_STOP" and marker["runs"] == 1


def test_d0_d4_masks_partition_signs_and_rejected_correct_logic():
    c = p12.masks_for(np.array([1, -1, 0, 0, 1], dtype=np.int8), np.array([1., 1., -1., 1., 0.]), np.array([1., -1., -1., -1., 1.]))
    assert c["accepted"].sum() + c["rejected"].sum() == 5
    assert c["accepted_correct"].sum() == c["accepted_wrong"].sum() == c["accepted_near_zero"].sum() == 1
    assert c["rejected_correct"].sum() == c["rejected_wrong"].sum() == 1


def test_five_bins_searchsorted_and_ranking_fixture_are_frozen():
    values = np.array([0., 1., 2., 3., 4., 5.])
    assert p12.assign(values, p12.qbounds(values)).tolist() == [0, 1, 2, 2, 4, 4]
    before = np.array([[[.4, .3], [.2, .1]]], dtype=np.float32)
    after = np.array([[[.4, .3], [.2, .8]]], dtype=np.float32)
    labels = np.array([[[0, 0], [0, 1]]], dtype=np.uint8)
    assert p12.rank_stats(before, after, labels)["positive_mean_rank_shift"] > 0


def test_spatial_fixture_and_class_aggregation_are_deterministic():
    assert recovery.spatial_pairs(np.array([[1, 1], [1, 0]], dtype=bool)) == 2
    rows = {"D0_NATIVE": [{"pixel_ap": .5, "pixel_auroc": .6, "mean_loss": .2, "per_image_ap_mean": .5}], "D1_PERSISTED_HARM_AWARE": [{"pixel_ap": .6, "pixel_auroc": .7, "mean_loss": .1, "per_image_ap_mean": .6}]}
    assert recovery.aggregate_conditions(rows)["D1_PERSISTED_HARM_AWARE"]["pixel_ap"] == .6


def test_atomic_serialization_and_post_marker_failure_capture(tmp_path):
    recovery.atomic_json(tmp_path / "x.json", {"b": 2, "a": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1, "b": 2}
    recovery.atomic_json(tmp_path / "ATTEMPT_STARTED.json", {"runs": 1})
    try:
        raise ValueError("fixture")
    except ValueError as exc:
        recovery.capture_failure(tmp_path, "fixture", "candle", exc)
    failure = json.loads((tmp_path / "ENGINEERING_FAILURE.json").read_text())
    assert failure["exception_type"] == "ValueError" and failure["last_completed_class"] == "candle"


def test_exactly_once_guard_and_fixed_freeze_constants(tmp_path):
    recovery.atomic_json(tmp_path / "ATTEMPT_STARTED.json", {"runs": 1})
    with pytest.raises(RuntimeError, match="attempt already exists"):
        recovery.attempt_guard(tmp_path)
    assert p12.r2.ALPHA == .25 and not (recovery.ROOT / "results/sabra_cure/post_r2v2_diagnostic_recovery/ATTEMPT_STARTED.json").exists()
