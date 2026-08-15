import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase5_second_evidence", ROOT / "tools/audit_phase5_second_evidence.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_deterministic_selection_and_exact_budget_without_gt():
    signal = np.array([0.4, 0.9, 0.9, 0.1], dtype=np.float32)
    pixel_id = np.array([10, 11, 12, 13], dtype=np.int64)
    selected_a = AUDIT.select_top(signal, pixel_id, 2)
    selected_b = AUDIT.select_top(signal, pixel_id, 2)
    assert selected_a.tolist() == selected_b.tolist() == [False, True, True, False]
    assert int(selected_a.sum()) == 2
    assert AUDIT.select_top(signal, pixel_id, 2).sum() == 2


def test_selector_does_not_depend_on_gt():
    signal = np.array([0.1, 0.8, 0.3, 0.7], dtype=np.float64)
    pixel_id = np.arange(signal.size, dtype=np.int64)
    first = AUDIT.select_top(signal, pixel_id, 2)
    second = AUDIT.select_top(signal, pixel_id, 2)
    assert np.array_equal(first, second)


def test_quantile_bins_and_deterministic_matching_same_bin():
    score = np.linspace(0.0, 1.0, 20)
    d_rank = np.linspace(1.0, 0.0, 20)
    labels = np.array([1] * 10 + [0] * 10, dtype=np.uint8)
    selected = np.ones(20, dtype=bool)
    pixel_id = np.arange(100, 120, dtype=np.int64)
    pos_a, neg_a = AUDIT.deterministic_matches("class", score, d_rank, labels, selected, pixel_id)
    pos_b, neg_b = AUDIT.deterministic_matches("class", score, d_rank, labels, selected, pixel_id)
    assert np.array_equal(pos_a, pos_b)
    assert np.array_equal(neg_a, neg_b)
    assert pos_a.size == neg_a.size
    score_bins = AUDIT.quantile_bins(score)
    rank_bins = AUDIT.quantile_bins(d_rank)
    assert all(score_bins[p] == score_bins[n] and rank_bins[p] == rank_bins[n] for p, n in zip(pos_a, neg_a))


def test_metric_tie_handling_and_orientation():
    evidence = np.array([0.8, 0.8, 0.2, 0.9])
    positive = np.array([0, 1, 2], dtype=np.int64)
    negative = np.array([1, 0, 3], dtype=np.int64)
    assert AUDIT.matched_win_rate(evidence, positive, negative) == (0.5 + 0.5 + 0.0) / 3


def test_spatial_shift_is_wraparound_and_fixed():
    values = np.arange(16, dtype=np.float32)
    shifted = np.roll(values.reshape(4, 4), (4 // 3, 4 // 3), axis=(0, 1)).reshape(-1)
    expected = np.roll(values.reshape(4, 4), (1, 1), axis=(0, 1)).reshape(-1)
    assert np.array_equal(shifted, expected)


def test_ap_parity_and_positive_only_oracle():
    scores = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)
    labels = np.array([1, 0, 1, 0], dtype=np.uint8)
    auc, ap = AUDIT.exact_auc_ap(scores, labels)
    evaluator_auc, evaluator_ap = AUDIT.project_exact_auc_ap(scores, labels)
    assert abs(auc - evaluator_auc) < 1e-12
    assert abs(ap - evaluator_ap) < 1e-12
    selected = np.array([False, True, False, False])
    bundle = AUDIT.oracle_bundle(labels, ap, np.argsort(-scores, kind="mergesort"), selected)
    assert bundle["positive_only_delta"] == 0.0


def test_local_nonconformity_shape_and_finiteness():
    import torch

    feature = torch.nn.functional.normalize(torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4), dim=0)
    evidence = AUDIT.local_nonconformity(feature)
    assert tuple(evidence.shape) == (4, 4)
    assert torch.isfinite(evidence).all()


def test_no_train_paths_and_candidate_completeness_contract():
    assert not any("train" in name.lower() for name in AUDIT.CANDIDATES)
    assert len(AUDIT.CANDIDATES) == 4

def test_real_runtime_shape_contract_37x37_patch_grid():
    import torch

    class Encoder:
        grid_size = (37, 37)

    class Model:
        image_encoder = Encoder()
        n_groups = 3

    stage_features = [torch.empty(1, 1369, 768) for _ in range(3)]
    features = torch.empty(3, 1, 1369, 768)
    native = torch.empty(3, 1, 1369, 2)
    native_margin = torch.empty(3, 1, 1369)
    model_prob = torch.empty(1, 518, 518)
    reconstructed = torch.empty(1, 2, 518, 518)
    final_logits = torch.empty(1, 2, 518, 518)
    record = AUDIT.validate_runtime_shapes(Model(), stage_features, features, native, native_margin, model_prob, reconstructed, final_logits, 518)
    assert record["patch_grid"] == [37, 37]
    assert record["native_stage_logits"] == [3, 1, 1369, 2]
    assert record["deployed_final_logits"] == [1, 2, 518, 518]
