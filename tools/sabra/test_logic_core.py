from __future__ import annotations

import unittest

import numpy as np

from p5f_geometry import pcrr, pgm
from sabra.logic_core_fixed import (
    compact_geometry,
    compute_relational_scores,
    construct_b1,
)


class LogicCoreTest(unittest.TestCase):
    def test_b1_has_exact_p9_and_no_fabrication(self) -> None:
        rng = np.random.default_rng(17)
        features = rng.normal(size=(3, 1369, 768)).astype(np.float32)
        features /= np.linalg.norm(features, axis=-1, keepdims=True)
        margins = rng.normal(size=(3, 1369)).astype(np.float32)
        ranks = np.stack([np.argsort(np.argsort(x)) for x in margins]).astype(np.float32)
        d_rank = ranks.std(axis=0)
        b1 = construct_b1(features, d_rank, margins)
        self.assertTrue(np.all(b1["valid_stability"] <= b1["valid_b1"]))
        self.assertTrue(np.all(b1["reserve_peer_index"][~b1["valid_stability"]] == -1))
        self.assertTrue(np.all(b1["peer_indices"][~b1["valid_b1"]] == -1))
        if np.any(b1["valid_stability"]):
            self.assertTrue(np.all(b1["reserve_peer_index"][b1["valid_stability"]] >= 0))

    def test_compact_scores_match_canonical_transforms(self) -> None:
        rng = np.random.default_rng(23)
        features = rng.normal(size=(3, 1369, 768)).astype(np.float32)
        features /= np.linalg.norm(features, axis=-1, keepdims=True)
        margins = rng.normal(size=(3, 1369)).astype(np.float32)
        d_rank = np.std(np.stack([np.argsort(np.argsort(x)) for x in margins]), axis=0)
        b1 = construct_b1(features, d_rank, margins)
        geometry = compact_geometry(features, b1)
        scores = compute_relational_scores(geometry, b1)
        canonical_pgm = pgm.transform(
            geometry["query_peer_cos"], geometry["peer_gram_upper"], b1["valid_b1"],
            {"config_id": "pgm_sum_whitened_mean", "whitened_aggregation": "sum_whitened", "stage_aggregation": "mean"},
        )["final"]
        canonical_pcrr = pcrr.transform(
            geometry["query_peer_cos"], geometry["peer_gram_upper"], b1["valid_b1"],
            {"config_id": "pcrr_witness_local_mean_mean", "witness_pool": "witness_local", "witness_aggregation": "mean", "stage_aggregation": "mean"},
        )["final"]
        np.testing.assert_allclose(scores["baseline_pgm"], canonical_pgm, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(scores["baseline_pcrr"], canonical_pcrr, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
