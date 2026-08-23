from __future__ import annotations

import numpy as np

from tools.sabra_cure.r1 import FEATURE_ORDER, LAMBDA, build_features, fit_ridge_accumulated, fit_ridge_full, parity_fixture, p75_scale, transform


def test_frozen_feature_contract_and_peer_fallback():
    assert len(FEATURE_ORDER) == 14
    assert FEATURE_ORDER[11].startswith("signed_native_margin")
    source = {name: np.ones((1, 1369)) * (index + 1) for index, name in enumerate(FEATURE_ORDER[:9])}
    source["native_margins"] = np.stack([np.full((1, 1369), 2.0), np.full((1, 1369), 3.0), np.full((1, 1369), 5.0)], axis=1)
    trust = {"valid_p9": np.ones((1, 1369), bool), "S9": np.ones((1, 1369)), "valid_p16": np.zeros((1, 1369), bool), "S16": np.ones((1, 1369)) * 99,
             "peer_indices": np.zeros((1, 1369, 2), dtype=np.int64), "valid_b1": np.zeros((1, 1369), bool)}
    x = build_features(source, trust)
    assert x.shape == (1, 1369, 14)
    assert np.all(x[..., 10] == 0.0)
    assert np.all(x[..., 11] == 10.0 / 3.0)
    assert np.all(x[..., 12] == 3.0)
    assert np.all(x[..., 13] == 10.0 / 3.0)


def test_target_and_sufficient_statistics_parity():
    utility = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = transform(utility, p75_scale(utility))
    assert np.isfinite(y).all() and np.max(np.abs(y)) <= 1.0
    assert np.array_equal(np.sign(y), np.sign(utility))
    assert parity_fixture()["status"] == "PASS"
    assert LAMBDA == 1.0


def test_accumulated_fit_matches_full_fit():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(101, len(FEATURE_ORDER)))
    y = rng.normal(size=101)
    full = fit_ridge_full(x, y)
    accumulated = fit_ridge_accumulated((x[:50], x[50:]), (y[:50], y[50:]))
    assert np.allclose(full[0], accumulated[0], atol=1e-10, rtol=0.0)
    assert abs(full[1] - accumulated[1]) <= 1e-10
