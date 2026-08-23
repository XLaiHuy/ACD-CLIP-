from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tools.sabra_cure.phase0_diagnostic import (
    CANDIDATE_ORDER,
    PARENT_SHA,
    action_metrics,
    acted_mask,
    binned_risk,
    confusion,
    corr,
    deterministic_probe,
    opposite_mask,
    spatial_concentration,
    threshold_prediction,
    transform_target,
)
from tools.sabra_car.r1_common import EXPECTED_CLASSES, FEATURE_ORDER, THRESHOLDS


def test_phase0_parent_and_frozen_inventory():
    assert PARENT_SHA == "48cd72b4609200d0a03d9ba3818f61b887c8ab1e"
    assert len(EXPECTED_CLASSES) == 12
    assert len(FEATURE_ORDER) == 11
    assert len(CANDIDATE_ORDER) == 5
    assert THRESHOLDS == (0.50, 0.60, 0.70, 0.80, 0.90)


def test_confusion_and_threshold_keep_semantics():
    oracle = np.array([-1, -1, 0, 0, 1, 1], dtype=np.int8)
    prediction = np.array([-1, 1, -1, 0, -1, 1], dtype=np.int8)
    confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.95], dtype=np.float32)
    assert confusion(oracle, prediction) == [[1, 0, 1], [1, 1, 0], [1, 0, 1]]
    thresholded = threshold_prediction(prediction, confidence, 0.75)
    assert thresholded.tolist() == [-1, 1, 0, 0, 0, 1]
    metrics = action_metrics(oracle, thresholded)
    assert metrics["coverage"] == 0.5
    assert metrics["opposite_sign_errors"] == 1
    assert metrics["opposite_sign_rate"] == 1 / 3


def test_target_transforms_are_finite_odd_and_sign_preserving():
    utility = np.array([-100.0, -2.0, -0.1, 0.0, 0.1, 2.0, 100.0])
    for name in (
        "T1_raw_clipped_robust_scaled",
        "T2_tanh_robust_scaled",
        "T3_signed_log_compressed",
    ):
        transformed = transform_target(utility, 2.0, name)
        assert np.isfinite(transformed).all()
        assert np.array_equal(np.sign(transformed), np.sign(utility))
        assert np.allclose(transformed, -transformed[::-1])


def test_correlation_and_spatial_contract():
    assert np.isclose(corr(np.arange(5), np.arange(5)), 1.0)
    assert corr(np.ones(5), np.arange(5)) is None
    opposite = np.zeros((1, 1369), dtype=bool)
    acted = np.ones((1, 1369), dtype=bool)
    opposite[0, :2] = True
    result = spatial_concentration(opposite, acted)
    assert result["adjacent_opposite_pairs"] == 1
    assert result["adjacent_acted_pairs"] == 2664


def test_binned_risk_does_not_alias_or_mutate_selection_masks():
    values = np.arange(100_000, dtype=np.float64)
    acted = np.zeros(100_000, dtype=bool)
    acted[::4] = True
    opposite = np.zeros(100_000, dtype=bool)
    rows = binned_risk(values, opposite, acted)
    assert np.isclose(sum(row["patch_fraction"] for row in rows), 1.0)
    assert all(np.isclose(row["coverage"], 0.25) for row in rows)
    assert np.count_nonzero(acted) == 25_000


def test_large_action_masks_preserve_predictions_and_match_counts():
    prediction = np.resize(np.array([-1, 0, 1, 1], dtype=np.int8), 100_000)
    oracle = np.resize(np.array([1, 0, 1, -1], dtype=np.int8), 100_000)
    original = prediction.copy()
    acted = acted_mask(prediction)
    opposite = opposite_mask(prediction, oracle)
    assert np.array_equal(prediction, original)
    assert np.count_nonzero(acted) == 75_000
    assert np.count_nonzero(opposite) == 50_000


def test_probe_is_bounded_and_explicitly_non_scientific():
    features = []
    utilities = []
    for index, _ in enumerate(EXPECTED_CLASSES):
        x = np.linspace(-1.0, 1.0, 32)
        features.append(np.column_stack([x, x * (index + 1)]))
        utilities.append((0.2 * x).astype(np.float32))
    result = deterministic_probe(
        features,
        utilities,
        ["x", "class_scaled_x"],
        "T2_tanh_robust_scaled",
    )
    assert result["label"] == "PROBE_ONLY_NOT_SCIENTIFIC_RESULT"
    assert result["cannot_satisfy_scientific_gate"] is True
    assert result["seed"] == 0
    assert len(result["folds"]) == 12


def test_implementation_has_no_forbidden_runtime_roots_or_training_calls():
    source = (
        Path(__file__).resolve().parents[1]
        / "tools/sabra_cure/phase0_diagnostic.py"
    ).read_text().lower()
    assert ".fit(" not in source
    assert "optimizer" not in source
    assert "clip forward" in source
    assert "new_clip_forwards" in source
    assert "mvtec_access_count" in source
    assert "medical_access_count" in source


def test_required_phase0_artifacts_and_machine_summary_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        "PHASE0_DIAGNOSTIC.md",
        "FEATURE_AUDIT.json",
        "RISK_DIAGNOSTIC.json",
        "UTILITY_AUDIT.json",
        "FORMULATION_COMPARISON.md",
        "DECISION_TREE_JOURNAL.md",
        "PHASE0_FINAL_DECISION.md",
        "MASTER_PREREGISTRATION_V1.md",
    ]
    assert all((root / "research/sabra_cure" / name).is_file() for name in required)
    summary = json.loads((root / "results/sabra_cure/phase0/summary.json").read_text())
    assert summary["status"] == "DIAGNOSTICS_COMPLETE"
    assert summary["parent_sha"] == PARENT_SHA
    assert summary["mvtec_access_count"] == 0
    assert summary["medical_access_count"] == 0


def test_master_preregistration_freezes_selected_contract():
    root = Path(__file__).resolve().parents[1]
    text = (root / "research/sabra_cure/MASTER_PREREGISTRATION_V1.md").read_text()
    required = [
        "FROZEN_BEFORE_IMPLEMENTATION_AND_RESULTS",
        PARENT_SHA,
        "signed_native_margin",
        "cross_stage_signed_margin_difference",
        "robust_peer_signed_margin_consensus",
        "P75(abs(u_train))",
        "two deterministic linear ridge heads",
        "1.0*||beta_mu||_2^2",
        "k in {0.5,1.0,1.5,2.0,3.0}",
        "Phase2B optimizer steps are exactly zero",
        "MVTec is forbidden through R3",
        "No gate may be relaxed after results",
    ]
    assert all(item in text for item in required)
    assert "No SABRA-CURE scientific fit or result existed" in text


def test_phase0_evidence_pointers_match_canonical_hashes():
    root = Path(__file__).resolve().parents[1]
    for name in ("RISK_DIAGNOSTIC", "UTILITY_AUDIT", "FEATURE_AUDIT"):
        pointer = json.loads((root / f"research/sabra_cure/{name}.json").read_text())
        artifact = root / pointer["canonical_artifact"]
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert digest == pointer["sha256"]
