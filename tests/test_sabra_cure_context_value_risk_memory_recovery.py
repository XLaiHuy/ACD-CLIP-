import gc
import json
import weakref

import numpy as np
import pytest

from tools.sabra_cure import context_value_risk as p14
from tools.sabra_cure import context_value_risk_memory_recovery as p16
from tools.sabra_cure import context_value_risk_recovery as p15


def test_frozen_contract_and_p15_engine_identity():
    assert p16.sha256(p16.ROOT / "tools/sabra_cure/context_value_risk.py") == p16.P14_SHA
    assert p14.ALPHA == .25 and p15.Q == (.5, .6, .7, .8, .9)
    assert len(p15.FEATURE_ORDER) == 16


def test_grouped_ap_exact_parity_is_inherited():
    score = np.array([.2, .2, .8, .1], np.float32); label = np.array([0, 1, 1, 0], np.uint8)
    _, positive, total = p15.score_groups(score, label)
    assert p15.ap_from_groups(positive, total) == p15.exact_average_precision(score, label)


def test_safe_expand_ridge_and_alpha_are_unchanged():
    risk = np.arange(5.); mu = np.array([1., -1., 1., -1., 1.]); t20, t40 = p14.thresholds(risk)
    safe, expand = p14.actions(mu, risk, t20), p14.actions(mu, risk, t40)
    assert np.all((safe != 0) <= (expand != 0)) and t20 <= t40
    model = p14.fit(np.arange(12.).reshape(6, 2), np.arange(6.))
    assert np.isfinite(model["beta"]).all() and model["intercept"] == model["intercept"]


def test_two_fold_synthetic_lifecycle_releases_all_full_buffers(tmp_path):
    report = p16.synthetic_memory_fixture(tmp_path)
    assert report["status"] == "PASS"
    assert report["full_buffers_unreachable"] and report["post_finalize_gate"] and report["no_monotonic_growth"]


def test_completed_cache_refs_are_detected_then_released():
    class Cache: pass
    cache = Cache(); fold = {"groups": {"a": {"cache": cache}}, "target": {"cache": cache}}
    refs = p16.cache_refs(fold)
    with pytest.raises(RuntimeError, match="remains reachable"):
        p16.assert_no_live_completed_cache(refs)
    del fold, cache; gc.collect(); p16.assert_no_live_completed_cache(refs)


def test_compact_checkpoint_hash_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(p16, "git", lambda *_: "sha")
    monkeypatch.setattr(p16, "input_hashes", lambda: {"input": "hash"})
    directory = p16.fold_dir(tmp_path, "candle"); directory.mkdir(parents=True)
    for name in ("parameters.json", "downstream.json", "policy_selection.json"):
        p16.atomic(directory / name, {})
    np.savez_compressed(directory / "fold.npz", mu=np.zeros(1), y=np.zeros(1), utility=np.zeros(1), sigma=np.zeros(1), risk=np.zeros(1), safe20=np.zeros(1), expand40=np.zeros(1), v=np.zeros(1), vhat=np.zeros(1), expand_images=np.zeros(1, bool))
    hashes = {name: p16.sha256(directory / name) for name in ("parameters.json", "downstream.json", "policy_selection.json", "fold.npz")}
    p16.atomic(directory / "checkpoint.json", {"execution_base_sha": "sha", "input_hashes": {"input": "hash"}, "artifacts": hashes})
    assert p16.completed_folds(tmp_path) == ["candle"]


def test_marker_and_resume_identity_fields_are_frozen(monkeypatch):
    monkeypatch.setattr(p16, "git", lambda *_: "sha")
    monkeypatch.setattr(p16, "input_hashes", lambda: {"input": "hash"})
    payload = p16.marker()
    assert payload["runs"] == 1 and payload["execution_base_sha"] == "sha" and payload["prereg_sha"] == p16.PREREG


def test_firewall_and_memory_limits_are_static_contracts():
    assert p16.MAX_RSS == 14 * 1024**3 and p16.POST_SLACK == 1024**3
    assert p14.ALPHA == .25
