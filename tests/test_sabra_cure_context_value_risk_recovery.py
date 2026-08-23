import json

import numpy as np
import pytest

from tools.sabra_car.r0_direction import exact_average_precision
from tools.sabra_cure import context_value_risk as p14
from tools.sabra_cure import context_value_risk_recovery as p15


def reference_replacement(safe, expand, labels, image):
    candidate = safe.copy(); candidate[image] = expand[image]
    return exact_average_precision(candidate, labels)


def grouped_replacement(safe, expand, labels, image):
    bs, bp, bt = p15.score_groups(safe, labels)
    ds, dp, dt = p15.delta_groups(safe[image], expand[image], labels[image])
    return p15.ap_with_delta(bs, bp, bt, ds, dp, dt)


def test_p14_contract_hash_constants_features_and_alpha():
    assert p15.sha256(p15.P14_SOURCE) == p15.P14_SOURCE_SHA
    assert p15.ALPHA == .25 and p15.Q == (.5, .6, .7, .8, .9)
    assert p15.FEATURE_ORDER == p14.FEATURE_ORDER and len(p15.FEATURE_ORDER) == 16


def test_p14_target_formula_is_replacement_minus_safe():
    safe = np.array([[.1, .6], [.5, .2]], np.float32); expand = np.array([[.9, .1], [.5, .2]], np.float32); labels = np.array([[0, 1], [1, 0]], np.uint8)
    base = exact_average_precision(safe, labels)
    assert np.isclose(grouped_replacement(safe, expand, labels, 0) - base, reference_replacement(safe, expand, labels, 0) - base)


@pytest.mark.parametrize("score,label", [
    (np.array([.1, .3, .2], np.float32), np.array([0, 1, 1], np.uint8)),
    (np.array([.5, .5, .5, .1], np.float32), np.array([0, 1, 0, 1], np.uint8)),
    (np.array([0., -0., 1., 1., -2.], np.float32), np.array([1, 0, 1, 0, 1], np.uint8)),
])
def test_grouped_ap_synthetic_and_tie_parity(score, label):
    _, positive, total = p15.score_groups(score, label)
    assert p15.ap_from_groups(positive, total) == exact_average_precision(score, label)


@pytest.mark.parametrize("image", [0, 1])
def test_grouped_delta_normal_anomaly_nochange_and_crossing(image):
    safe = np.array([[.1, .8, .2, .2], [.4, .3, .7, .6]], np.float32)
    expand = np.array([[.9, .8, .2, .0], [.4, .3, .7, .6]], np.float32)
    labels = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], np.uint8)
    assert abs(grouped_replacement(safe, expand, labels, image) - reference_replacement(safe, expand, labels, image)) <= 1e-12


def test_delta_handles_created_and_deleted_score_groups():
    safe = np.array([[.1, .1], [.9, .9]], np.float32); expand = np.array([[.2, .8], [.9, .9]], np.float32); labels = np.array([[1, 0], [1, 0]], np.uint8)
    assert grouped_replacement(safe, expand, labels, 0) == reference_replacement(safe, expand, labels, 0)


def test_multi_image_policy_parity():
    safe = np.array([[.1, .8], [.4, .3], [.5, .9]], np.float32); expand = np.array([[.9, .8], [.4, .7], [.5, .9]], np.float32); labels = np.array([[0, 1], [1, 0], [0, 1]], np.uint8)
    bs, bp, bt = p15.score_groups(safe, labels); ds, dp, dt = p15.delta_groups(safe[[0, 1]], expand[[0, 1]], labels[[0, 1]])
    candidate = safe.copy(); candidate[[0, 1]] = expand[[0, 1]]
    assert p15.ap_with_delta(bs, bp, bt, ds, dp, dt) == exact_average_precision(candidate, labels)


def test_actions_thresholds_safe_expand_and_ridge_are_frozen():
    risk = np.arange(5., dtype=np.float64); mu = np.array([1., -1., 1., -1., 1.]); t20, t40 = p14.thresholds(risk)
    safe, expand = p14.actions(mu, risk, t20), p14.actions(mu, risk, t40)
    assert np.all((safe != 0) <= (expand != 0)) and t20 <= t40
    model = p14.fit(np.arange(12., dtype=float).reshape(6, 2), np.arange(6., dtype=float))
    assert np.isfinite(model["beta"]).all() and np.isfinite(model["intercept"])


def test_atomic_checkpoint_and_marker_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(p15, "git", lambda *_: "deadbeef")
    monkeypatch.setattr(p15, "input_hashes", lambda: {"x": "1"})
    marker = p15.marker_payload(); p15.atomic(tmp_path / "ATTEMPT_STARTED.json", marker)
    assert json.loads((tmp_path / "ATTEMPT_STARTED.json").read_text())["runs"] == 1
    p15.write_checkpoint(tmp_path, "candle", "image_target_group", ["capsules"], {"ok": True})
    assert (tmp_path / "checkpoints" / "outer_candle" / "image_target_group.json").exists()


def test_resume_rejects_changed_execution_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(p15, "git", lambda *_: "new")
    monkeypatch.setattr(p15, "input_hashes", lambda: {"input": "new"})
    p15.atomic(tmp_path / "ATTEMPT_STARTED.json", {"execution_base_sha": "old", "prereg_sha": p15.PREREG, "input_hashes": {"input": "new"}, "attempt_uuid": "a"})
    with pytest.raises(RuntimeError, match="identity mismatch"):
        p15.execute(tmp_path, resume=True)


def test_worker_policy_is_bounded_and_deterministic_contract():
    assert 1 <= p15.WORKERS <= 4
    score = np.array([.2, .1, .8, .4], np.float32); label = np.array([0, 1, 1, 0], np.uint8)
    one = p15.score_groups(score, label); many = p15.score_groups(score, label)
    assert all(np.array_equal(a, b) for a, b in zip(one, many))


def test_historical_p14_marker_terminal_and_firewall_are_preserved():
    assert (p15.ROOT / "results/sabra_cure/context_value_risk/ATTEMPT_STARTED.json").exists()
    assert (p15.ROOT / "research/sabra_cure/context_value_risk/P14_FINAL_DECISION.md").exists()
    assert p15.ALPHA == .25
