from __future__ import annotations

import numpy as np

from sabra.trust_v2 import numerical as exact
from sabra.trust_v2.fast_geometry import build_compact_record_fast, construct_b1_fast


def _compare_discrete(features: np.ndarray, margins: np.ndarray) -> None:
    d_rank = np.std(
        np.stack([exact.percentile_rank(margins[s]) for s in range(3)]),
        axis=0,
    ).astype(np.float32)
    left = exact.construct_b1_v2(features, d_rank, margins)
    right = construct_b1_fast(features, d_rank, margins)
    for key in (
        "candidate_count",
        "valid_b1",
        "valid_p9",
        "valid_p16",
        "peer_indices",
        "reserve_p9_index",
        "reserve_p16_index",
    ):
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)
    for key in ("p8_p9_similarity_gap", "p8_p16_similarity_gap"):
        np.testing.assert_allclose(left[key], right[key], rtol=1e-5, atol=1e-6, err_msg=key)


def test_discrete_edge_cases_match_exact() -> None:
    rng = np.random.default_rng(42)
    margins = rng.normal(size=(3, 1369)).astype(np.float32)
    cases = [
        np.ones((3, 1369, 16), dtype=np.float32),
        np.concatenate(
            [np.ones((3, 1369, 1), dtype=np.float32), np.zeros((3, 1369, 15), dtype=np.float32)],
            axis=-1,
        ),
        rng.normal(size=(3, 1369, 16)).astype(np.float32) * 1e-12,
    ]
    for features in cases:
        _compare_discrete(features, margins)


def test_full_record_normal_parity() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(3, 1369, 16)).astype(np.float32)
    margins = rng.normal(size=(3, 1369)).astype(np.float32)
    expected, expected_transient = exact.build_compact_record(features, margins, "synthetic")
    actual, actual_transient = build_compact_record_fast(features, margins, "synthetic")
    for key in ("candidate_count", "valid_b1", "valid_p9", "valid_p16", "peer_indices", "reserve_p9_index", "reserve_p16_index"):
        np.testing.assert_array_equal(expected[key], actual[key], err_msg=key)
    for key in ("D_rank", "baseline_pgm", "baseline_pcrr", "D_rel", "S9", "R9", "S16", "R16"):
        np.testing.assert_allclose(expected[key], actual[key], rtol=1e-4, atol=1e-5, err_msg=key)
    for key in ("query_peer_cos", "peer_gram_upper", "query_reserve_cos", "reserve_to_peer_cos"):
        np.testing.assert_allclose(expected_transient["geometry"][key], actual_transient["geometry"][key], rtol=1e-5, atol=1e-6, err_msg=key)


def test_batch_size_one_and_larger_repeated_execution() -> None:
    rng = np.random.default_rng(9)
    one_features = rng.normal(size=(3, 1369, 16)).astype(np.float32)
    one_margins = rng.normal(size=(3, 1369)).astype(np.float32)
    one_a, _ = build_compact_record_fast(one_features, one_margins, "one")
    one_b, _ = build_compact_record_fast(one_features, one_margins, "one")
    np.testing.assert_array_equal(one_a["peer_indices"], one_b["peer_indices"])
    batch_features = [rng.normal(size=(3, 1369, 16)).astype(np.float32) for _ in range(4)]
    batch_margins = [rng.normal(size=(3, 1369)).astype(np.float32) for _ in range(4)]
    outputs = [build_compact_record_fast(features, margins, str(index))[0] for index, (features, margins) in enumerate(zip(batch_features, batch_margins))]
    assert len(outputs) == 4
    assert all(output["peer_indices"].shape == (1369, 8) for output in outputs)
