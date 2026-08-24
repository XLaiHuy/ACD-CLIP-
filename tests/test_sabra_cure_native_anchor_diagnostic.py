"""Pre-marker exactness regressions for the P21 grouped-count engine."""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from tools.sabra_car.r0_direction import exact_metrics
from tools.sabra_cure import native_anchor_diagnostic as p21


def _engine() -> p21.NativeAnchorEngine:
    labels = np.array([[[1, 0, 1]], [[0, 1, 0]], [[1, 0, 0]]], dtype=np.uint8)
    native = np.array([[[.1, .2, .1]], [[.3, .4, .5]], [[.5, .5, .2]]], dtype=np.float32)
    safe = np.array([[[.2, .2, .1]], [[.3, .6, .4]], [[.5, .4, .2]]], dtype=np.float32)
    expand = np.array([[[.3, .0, .2]], [[.2, .7, .6]], [[.6, .3, .1]]], dtype=np.float32)
    return p21.NativeAnchorEngine(native, {"NATIVE": native, "SAFE20": safe, "EXPAND40": expand}, labels)


def test_fixture_fast_reference_is_exact() -> None:
    result = p21.fixture()
    assert result["error"] == 0.0


def test_all_mixed_states_match_direct_frozen_metric() -> None:
    engine = _engine()
    for states in itertools.product(p21.A0, repeat=engine.n_images):
        state = np.asarray(states)
        fast = engine.ap(*engine.counts(state))
        direct = exact_metrics(engine.compose(state).reshape(-1), engine.masks.reshape(-1))["pAP"]
        assert abs(fast - direct) <= p21.EPS


def test_candidate_commit_and_revert_restore_exact_state() -> None:
    engine = _engine()
    state = np.asarray(["NATIVE", "SAFE20", "EXPAND40"])
    positive, total = engine.counts(state)
    original = engine.ap(positive, total)
    changed_positive, changed_total, candidate = engine.candidate(positive, total, 1, "SAFE20", "EXPAND40")
    assert abs(candidate - engine.ap(changed_positive, changed_total)) <= p21.EPS
    reverted_positive, reverted_total, _ = engine.candidate(changed_positive, changed_total, 1, "EXPAND40", "SAFE20")
    assert abs(original - engine.ap(reverted_positive, reverted_total)) <= p21.EPS


def test_sparse_delta_not_dense_image_score_matrix() -> None:
    engine = _engine()
    delta = engine.image_delta(0, "SAFE20")
    assert delta.index.ndim == 1
    assert hasattr(engine, "union")


def test_strict_coordinate_update_rejects_epsilon_tie() -> None:
    labels = np.array([[[1, 0]]], dtype=np.uint8)
    native = np.array([[[.1, .2]]], dtype=np.float32)
    same = native.copy()
    engine = p21.NativeAnchorEngine(native, {"NATIVE": native, "SAFE20": same, "EXPAND40": same}, labels)
    outcome = p21.coordinate(engine, np.array(["NATIVE"]))
    assert outcome["state"].tolist() == ["NATIVE"]
    assert outcome["changes"] == 0


def test_multistart_tie_prefers_lower_coverage_then_conservative_order() -> None:
    a = {"pap": .5, "state": np.array(["SAFE20", "NATIVE"])}
    b = {"pap": .5, "state": np.array(["NATIVE", "SAFE20"])}
    c = {"pap": .5, "state": np.array(["NATIVE", "NATIVE"])}
    assert p21.choose_seed([a, b, c]) is c


def test_safe30_family_tie_order_is_conservative() -> None:
    a = {"pap": .5, "state": np.array(["SAFE30"])}
    b = {"pap": .5, "state": np.array(["SAFE20"])}
    assert p21.choose_seed([a, b], p21.A1) is b


def test_ranknet_is_deterministic_and_orders_synthetic_values() -> None:
    feature = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    value = np.array([0.0, .1, .2, .3], dtype=np.float64)
    first, _ = p21.fit_ranknet(feature, value, [np.arange(4)])
    second, _ = p21.fit_ranknet(feature, value, [np.arange(4)])
    assert np.array_equal(first, second)
    assert np.all(np.diff(feature @ first) > 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA exactness fixture")
def test_gpu_grouped_ap_and_coordinate_trajectory_match_cpu() -> None:
    engine = _engine()
    state = np.array(["NATIVE", "SAFE20", "EXPAND40"])
    positive, total = engine.counts(state)
    cpu = engine.ap(positive, total)
    gpu = p21.gpu_ap(torch.as_tensor(positive, dtype=torch.float64, device="cuda"), torch.as_tensor(total, dtype=torch.float64, device="cuda"))
    assert abs(cpu - gpu) <= p21.EPS
    batch = p21.gpu_ap_batch(torch.stack((torch.as_tensor(positive, dtype=torch.float64, device="cuda"), torch.as_tensor(positive, dtype=torch.float64, device="cuda"))), torch.stack((torch.as_tensor(total, dtype=torch.float64, device="cuda"), torch.as_tensor(total, dtype=torch.float64, device="cuda"))))
    assert np.max(np.abs(batch - cpu)) <= p21.EPS
    left = p21.coordinate(engine, np.array(["NATIVE", "NATIVE", "NATIVE"]))
    right = p21.coordinate_gpu(engine, np.array(["NATIVE", "NATIVE", "NATIVE"]))
    assert left["state"].tolist() == right["state"].tolist()
    assert abs(left["pap"] - right["pap"]) <= p21.EPS
