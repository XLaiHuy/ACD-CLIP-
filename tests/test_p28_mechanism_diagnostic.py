from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from model.phase2b_runtime import deploy_native_logits
from tools.sabra_car import r0_direction
from tools.sabra_v2 import p28_mechanism_diagnostic as p28
from tools.sabra_v2.region_pool import pool_patch_map, symmetric_margin_delta, upsample_region_map


def _native_fixture(batch: int = 2) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(3, batch, 37 * 37, 2, dtype=torch.float32) * 0.1


def _actions_fixture(batch: int = 2) -> torch.Tensor:
    actions = torch.zeros(batch, 37 * 37, dtype=torch.int8)
    actions[0, 0] = -1
    actions[0, 1] = 1
    actions[1, 2] = 1
    return actions


def test_op_correction_matches_exact_r0_alpha_and_margin_semantics() -> None:
    actions = _actions_fixture()
    observed = p28.patch_correction_from_actions(actions)
    expected = torch.from_numpy(
        r0_direction.correction_from_actions(actions.numpy(), 0.25, positive_only=False)
    ).reshape(-1, 37, 37)
    torch.testing.assert_close(observed, expected)


def test_abnormal_only_delta_matches_r0_intervention() -> None:
    native = _native_fixture()
    correction = torch.randn(2, 37 * 37, dtype=torch.float32)
    observed = p28.abnormal_only_delta(native, correction)
    expected = r0_direction.intervention_delta(correction, native)
    torch.testing.assert_close(observed, expected)
    assert torch.count_nonzero(observed[..., 0]) == 0
    assert torch.equal(observed[0], observed[1])
    assert torch.equal(observed[1], observed[2])


def test_region_pool_and_reconstruction_match_p27_path() -> None:
    patch = torch.arange(2 * 3 * 37 * 37, dtype=torch.float32).reshape(2, 3, 37, 37)
    observed_region, observed_patch = p28.regionize_and_reconstruct(patch)
    expected_region = pool_patch_map(patch)
    expected_patch = upsample_region_map(expected_region)
    torch.testing.assert_close(observed_region, expected_region)
    torch.testing.assert_close(observed_patch, expected_patch)


def test_symmetric_reconstruction_preserves_margin_delta() -> None:
    native = _native_fixture()
    patch = torch.randn(3, 2, 37, 37)
    corrected = symmetric_margin_delta(native, patch)
    observed_margin = corrected[..., 1] - corrected[..., 0]
    native_margin = native[..., 1] - native[..., 0]
    torch.testing.assert_close((observed_margin - native_margin).reshape(3, 2, 37, 37), patch)


def test_native_reconstruction_from_cache_matches_frozen_native_probability() -> None:
    native_logits = _native_fixture(batch=1)
    expected, _ = deploy_native_logits(native_logits, domain="Industrial")
    observed = p28.native_probability_from_logits(native_logits)
    torch.testing.assert_close(observed, expected[:, 1])


def test_immutable_prediction_loader_checks_freeze_hash(tmp_path: Path) -> None:
    payload = {
        "schema_version": "P27_IMMUTABLE_HELD_PREDICTIONS_V1",
        "held_class": "candle",
        "gt_used": False,
        "mask_reads": 0,
        "records": [{
            "class_name": "candle",
            "image_path": "candle/Data/Images/Normal/001.JPG",
            "native_abnormal_probability": torch.zeros(518, 518),
            "p27_abnormal_probability": torch.ones(518, 518),
        }],
    }
    path = tmp_path / "predictions.pt"
    torch.save(payload, path)
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = p28.load_immutable_predictions(path, "candle", expected_hash)
    assert loaded[0]["image_path"] == payload["records"][0]["image_path"]
    with pytest.raises(ValueError, match="freeze hash"):
        p28.load_immutable_predictions(path, "candle", "0" * 64)


def test_vectorized_auroc_and_ap_match_trusted_reference() -> None:
    scores = np.asarray([0.9, 0.9, 0.4, 0.2, 0.2, 0.1], dtype=np.float32)
    labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.uint8)
    observed = p28.exact_metrics(scores, labels)
    expected = r0_direction.exact_metrics(scores, labels)
    assert observed == expected


def test_pair_ordering_gained_lost_matches_bruteforce_fixture() -> None:
    native_anomaly = np.asarray([0.2, 0.8, 0.4], dtype=np.float32)
    native_normal = np.asarray([0.3, 0.7, 0.1, 0.5], dtype=np.float32)
    state_anomaly = np.asarray([0.9, 0.6, 0.4], dtype=np.float32)
    state_normal = np.asarray([0.2, 0.75, 0.15, 0.45], dtype=np.float32)
    observed = p28.pair_ordering_change(native_anomaly, native_normal, state_anomaly, state_normal)
    gained = lost = 0
    for anomaly_n, anomaly_s in zip(native_anomaly, state_anomaly):
        for normal_n, normal_s in zip(native_normal, state_normal):
            base = anomaly_n > normal_n
            state = anomaly_s > normal_s
            gained += int(state and not base)
            lost += int(base and not state)
    assert observed["gained"] == gained
    assert observed["lost"] == lost
    assert observed["net"] == gained - lost


def test_top_rank_behavior_uses_fixed_descriptive_fractions() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.1, 0.0], dtype=np.float32)
    labels = np.asarray([1, 0, 1, 0, 0], dtype=np.uint8)
    result = p28.top_rank_behavior(scores, labels, fractions=(0.2, 0.4))
    assert result["0.2"]["count"] == 1
    assert result["0.2"]["anomaly_fraction"] == 1.0
    assert result["0.4"]["count"] == 2
    assert result["0.4"]["anomaly_fraction"] == 0.5


def test_no_optimizer_clip_or_phase2b_forward_in_p28_execution_path() -> None:
    source = Path(p28.__file__).read_text()
    tree = ast.parse(source)
    forbidden = ("torch.optim", "Optimizer", ".backward(", ".step(", "clip_model", "Phase2BForward", "encode_image")
    assert not any(token in source for token in forbidden)
    imported = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("open_clip" in module or "clip" in module.lower() for module in imported)


def test_mvtec_medical_firewall() -> None:
    p28.enforce_data_firewall(Path("/workspace/data/source/VisA"), [Path("/workspace/data/source/VisA")])
    with pytest.raises(RuntimeError, match="firewall"):
        p28.enforce_data_firewall(Path("/workspace/data/source/VisA"), [Path("/workspace/MVTec/test")])
    with pytest.raises(RuntimeError, match="firewall"):
        p28.enforce_data_firewall(Path("/workspace/data/source/VisA"), [Path("/workspace/Medical/test")])
