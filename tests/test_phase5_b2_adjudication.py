import importlib.util
import inspect
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase5_b2_adjudication", ROOT / "tools/audit_phase5_b2_adjudication.py")
B2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B2)


def fixture():
    m = np.linspace(-1, 1, B2.PATCH_COUNT, dtype=np.float32)
    d = np.linspace(0, 1, B2.PATCH_COUNT, dtype=np.float32)
    e = np.linspace(1, 0, B2.PATCH_COUNT, dtype=np.float32)
    return m, d, e, np.ones(B2.PATCH_COUNT, dtype=bool)


def test_t1_to_t15_synthetic_contracts_pass():
    result = B2.run_unit_tests()
    assert result["status"] == "PASS"
    assert result["test_count"] == 15
    assert all(result["tests"].values())


def test_deterministic_risk_selection_has_exact_k_and_identity_ties():
    m, d, e, valid = fixture()
    _, info = B2.adjudicate_slots(m, d, e, valid)
    assert info["risk"].sum() == int(np.ceil(0.20 * B2.PATCH_COUNT))
    tied = np.ones(B2.PATCH_COUNT, dtype=np.float32)
    _, tied_info = B2.adjudicate_slots(m, tied, e, valid)
    assert np.array_equal(np.flatnonzero(tied_info["risk"]), np.arange(int(np.ceil(0.20 * B2.PATCH_COUNT))) )


def test_gt_firewall_signature_has_no_target_or_mask():
    assert set(inspect.signature(B2.adjudicate_slots).parameters) == {"m_bar", "d_rank", "evidence", "valid_reference"}
    assert "target" not in inspect.signature(B2.predictor_gt_free).parameters
    assert "mask" not in inspect.signature(B2.predictor_gt_free).parameters


def test_score_bins_and_patch_shift_are_deterministic():
    values = np.arange(100, dtype=np.float32)
    assert np.array_equal(B2.quantile_bins(values), B2.quantile_bins(values))
    grid = np.arange(B2.PATCH_COUNT, dtype=np.float32)
    shifted = B2.shifted_evidence(grid)
    assert shifted[12 * 37 + 12] == grid[0]


def test_same_cell_match_does_not_use_evidence():
    positive = np.zeros(B2.PATCH_COUNT, dtype=bool)
    positive[0] = True
    positive[1] = True
    eligible = np.ones(B2.PATCH_COUNT, dtype=bool)
    score_bin = np.zeros(B2.PATCH_COUNT, dtype=np.int64)
    rank_bin = np.zeros(B2.PATCH_COUNT, dtype=np.int64)
    a = B2.bridge_matches("candle", 4, positive, eligible, score_bin, rank_bin)
    b = B2.bridge_matches("candle", 4, positive, eligible, score_bin, rank_bin)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_evidence_orientation_and_slot_conservation():
    m, d, e, valid = fixture()
    corrected, info = B2.adjudicate_slots(m, d, e, valid)
    assert np.all(np.isfinite(corrected))
    for cell in info["cells"]:
        indices = np.asarray(cell["patches"])
        assert np.array_equal(np.sort(m[indices])[::-1], np.sort(corrected[indices])[::-1])


def test_native_invariants_hold_after_delta():
    m, d, e, valid = fixture()
    corrected, info = B2.adjudicate_slots(m, d, e, valid)
    native = torch.zeros((3, 1, B2.PATCH_COUNT, 2), dtype=torch.float32)
    native[:, 0, :, 0] = -0.5
    native[:, 0, :, 1] = torch.from_numpy(m - 0.5)
    after = B2.apply_delta_to_native(native, info["delta"])
    assert torch.equal(native[:, :, :, 0], after[:, :, :, 0])
    assert torch.equal(native[:, 0, ~info["acted"], :], after[:, 0, ~info["acted"], :])
    before_margin = native[:, 0, :, 1] - native[:, 0, :, 0]
    after_margin = after[:, 0, :, 1] - after[:, 0, :, 0]
    assert torch.allclose(after_margin.mean(0), torch.from_numpy(corrected), atol=B2.MARGIN_TOL, rtol=0)
    assert torch.equal(after_margin[0] - after_margin[1], before_margin[0] - before_margin[1])


def test_exact_ap_auroc_parity():
    auc, ap = B2.exact_auc_ap(np.array([0, 1, 2, 3], dtype=np.float32), np.array([0, 1, 0, 1], dtype=np.uint8))
    assert np.isclose(auc, 0.75)
    assert np.isclose(ap, 5.0 / 6.0)


def test_empty_cells_abstain_and_no_nan_inf():
    m, d, e, _ = fixture()
    corrected, info = B2.adjudicate_slots(m, d, e, np.zeros(B2.PATCH_COUNT, dtype=bool))
    assert info["acted_patch_count"] == 0
    assert np.all(np.isfinite(corrected))


def test_constants_encode_test_only_scope_and_all_classes():
    assert B2.EXPECTED_CLASSES == 12
    assert B2.EXPECTED_IMAGES == 2162
    assert B2.EXPECTED_NORMAL == 962
    assert B2.EXPECTED_ANOMALY == 1200
    assert "train" not in str(B2.PHASE5_ROOT / "TEST").lower()

def test_bridge_shape_regression_flattens_37x37_occupancy_at_boundary():
    mask = np.zeros((B2.IMAGE_SIZE, B2.IMAGE_SIZE), dtype=np.uint8)
    mask[:B2.PATCH_STRIDE, :B2.PATCH_STRIDE] = 1
    positive = B2.occupancy_from_mask(mask) > 0
    assert positive.shape == (B2.PATCH_COUNT,)
    eligible = np.ones(B2.PATCH_COUNT, dtype=bool)
    bins = np.zeros(B2.PATCH_COUNT, dtype=np.int64)
    pos, neg = B2.bridge_matches("candle", 0, positive, eligible, bins, bins)
    assert pos.ndim == 1 and neg.ndim == 1
