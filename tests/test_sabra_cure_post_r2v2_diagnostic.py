import numpy as np

from tools.sabra_cure.post_r2v2_diagnostic import assign, correlation, masks_for, qbounds, rank_stats


def test_masks_partition_and_oracle_rejected_cohorts():
    result = masks_for(np.array([1, -1, 0, 0, 1], dtype=np.int8), np.array([1., 1., -1., 1., 0.]), np.array([1., -1., -1., -1., 1.]))
    assert result["accepted"].sum() == 3 and result["rejected"].sum() == 2
    assert result["accepted_correct"].sum() == 1 and result["accepted_wrong"].sum() == 1 and result["accepted_near_zero"].sum() == 1
    assert result["rejected_correct"].sum() == 1 and result["rejected_wrong"].sum() == 1


def test_five_bins_are_deterministic_and_exhaustive():
    values = np.array([0., 1., 2., 3., 4., 5.])
    bins = assign(values, qbounds(values))
    # NumPy's linear .60 quantile is represented infinitesimally below 3 here;
    # record the actual frozen searchsorted result rather than rounding edges.
    assert bins.tolist() == [0, 1, 2, 2, 4, 4]
    assert set(bins.tolist()) <= set(range(5))


def test_ranking_fixture_improves_positive_ordering_without_patch_ap_attribution():
    native = np.array([[[.4, .3], [.2, .1]]], dtype=np.float32)
    changed = np.array([[[.4, .3], [.2, .8]]], dtype=np.float32)
    labels = np.array([[[0, 0], [0, 1]]], dtype=np.uint8)
    metrics = rank_stats(native, changed, labels)
    assert metrics["positive_mean_rank_shift"] > 0
    assert metrics["top10_anomaly_enrichment_delta"] > 0


def test_correlations_null_for_undefined_and_finite_for_valid_inputs():
    assert correlation(np.ones(3), np.arange(3))["pearson"] is None
    assert correlation(np.arange(4), np.arange(4))["spearman"] == 1.0
