import json

import numpy as np
import pytest

from tools.sabra_cure import context_value_risk as p14
from tools.sabra_cure import r2v2_harm as frozen


def test_parent_provenance_and_frozen_constants():
    assert p14.git("rev-parse", p14.PARENT) == p14.PARENT
    assert p14.git("rev-parse", p14.PREREG) == p14.PREREG
    assert p14.ALPHA == .25 and p14.Q == (.5, .6, .7, .8, .9)
    assert len(p14.FEATURE_ORDER) == 16


def test_history_is_immutable_and_target_feasibility_is_go():
    assert p14.protected()
    assert p14.feasibility()["status"] == "GO"


def test_tau_actions_subset_and_expansion_band_are_exact():
    risk = np.array([0., 1., 2., 3., 4.]); mu = np.array([1., -1., 1., -1., 1.])
    t20, t40 = p14.thresholds(risk)
    s, e = p14.actions(mu, risk, t20), p14.actions(mu, risk, t40)
    assert np.all((s != 0) <= (e != 0))
    assert np.array_equal((e != 0) & (s == 0), (risk > t20) & (risk <= t40))


def test_context_features_empty_band_and_finite_behavior(monkeypatch):
    fake = {"native_score_rank": np.linspace(0, 1, p14.PATCHES), "native_score": np.zeros(p14.PATCHES), "signed_native_margin": np.zeros(p14.PATCHES), "stage_disagreement": np.zeros(p14.PATCHES), "peer_support": np.zeros(p14.PATCHES)}
    monkeypatch.setattr(p14.p12, "source_fields", lambda *_: (fake, np.zeros((1, 1, 1)), np.array(["x"])))
    x = np.zeros((p14.PATCHES, 14)); mu = np.ones(p14.PATCHES); sigma = np.ones(p14.PATCHES); risk = np.zeros(p14.PATCHES)
    out = p14.fields("fixture", x, mu, sigma, risk, .1, .2, np.array(["x"]))
    assert out.shape == (1, 16) and np.all(np.isfinite(out)) and np.all(out[0, 3:] == 0)


def test_centered_ridge_and_training_scaler_parity():
    x = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.]])
    y = np.array([0., 1., 2., 3.])
    model = p14.fit(x, y)
    b, c = frozen.ridge((x - model["median"]) / model["iqr"], y)
    assert np.allclose(model["beta"], b) and model["intercept"] == c


def test_selection_tie_prefers_higher_q_and_no_expansion_fallback(monkeypatch):
    def deploy_stub(_name, _actions):
        pap = .7 if np.count_nonzero(_actions) else .6
        return np.zeros((1, 1, 1)), np.zeros((1, 1, 1), dtype=np.uint8), np.zeros(1), {"pixel_ap": pap, "pixel_auroc": .7, "mean_loss": .2}
    monkeypatch.setattr(p14, "deploy", deploy_stub)
    monkeypatch.setattr(p14, "safety", lambda *_: {"wrong_rate": 0., "relative_weighted_harm_reduction": 1.})
    z, o = np.zeros(p14.PATCHES, dtype=np.int8), np.ones(p14.PATCHES, dtype=np.int8)
    g = {"a": {"safe": z, "expand": o, "y": np.ones(p14.PATCHES), "mu": np.ones(p14.PATCHES)}, "b": {"safe": z, "expand": o, "y": np.ones(p14.PATCHES), "mu": np.ones(p14.PATCHES)}}
    selected, _ = p14.source_selection(g, {"a": np.array([0.]), "b": np.array([1.])})
    assert selected == .9
    monkeypatch.setattr(p14, "safety", lambda *_: {"wrong_rate": 1., "relative_weighted_harm_reduction": 0.})
    selected, record = p14.source_selection(g, {"a": np.array([0.]), "b": np.array([1.])})
    assert selected is None and record["selected"] == "NO_EXPANSION"


def test_atomic_failure_and_exactly_once_guard(tmp_path):
    p14.atomic(tmp_path / "ATTEMPT_STARTED.json", {"runs": 1, "numpy_flag": np.bool_(True)})
    with pytest.raises(RuntimeError, match="attempt exists"):
        p14.guard(tmp_path)
    payload = json.loads((tmp_path / "ATTEMPT_STARTED.json").read_text())
    assert payload["runs"] == 1 and payload["numpy_flag"] is True
