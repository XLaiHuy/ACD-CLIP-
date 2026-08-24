import numpy as np

from tools.sabra_cure import patch_actionability_r1 as p


def test_frozen_panel_and_feature_contracts():
    assert p.TARGET_PATCHES_PER_CLASS == 2000
    assert p.CAP_PER_IMAGE == 16
    assert p.STRATA == 5 and p.STRATUM_QUOTA == 80
    assert len(p.FEATURE_ORDER) == 32
    assert p.FEATURE_ORDER[:22] == p.r2v2_harm.HARM_ORDER
    assert p.ALPHA == .25


def test_quantile_bins_are_bounded_and_deterministic():
    values = np.array([[0., 1., 1., 2., 3.], [4., 5., 6., 7., 8.]])
    first = p._rank_bin(values)
    assert first.shape == values.shape
    assert np.array_equal(first, p._rank_bin(values))
    assert first.min() >= 0 and first.max() <= 4


def test_pair_construction_is_same_class_nonadjacent_and_capped():
    values = np.linspace(-1, 1, 2000)
    first = p._deterministic_pairs("fixture", values)
    second = p._deterministic_pairs("fixture", values)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert len(first[0]) <= p.PAIR_CAP
    assert np.all(values[first[0]] > values[first[1]])


def test_apply_metric_and_gate_math_is_defined():
    v = np.array([-2., -1., 1., 2., 3.])
    score = np.array([-2., -1., 1., 2., 3.])
    metric = p.q1_metrics(v, score)
    assert metric["support"] is True
    assert metric["spearman"] is not None and metric["spearman"] > .99
    assert metric["sign_auc"] == 1.0
    assert metric["bc20"] is not None


def test_sparse_candidate_is_noop_only_for_keep():
    basis = p.Basis(np.array([[0, 1]], dtype=np.int32), np.array([[.25, .5]], dtype=np.float32), np.array([[True, True]]))
    margin = np.array([0., 0.], dtype=np.float32)
    index, score = p.candidate_support_scores(margin, 0, 1, basis)
    assert np.array_equal(index, np.array([0, 1], dtype=np.int32))
    assert np.all(score > .5)
