import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase5_b2_adjudication", ROOT / "tools/audit_phase5_b2_adjudication.py")
B2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B2)


def correct_win_rate(e_pos, e_neg):
    e_pos = np.asarray(e_pos, dtype=np.float64)
    e_neg = np.asarray(e_neg, dtype=np.float64)
    delta = e_pos - e_neg
    return float(np.mean((delta > 0) + 0.5 * (delta == 0)))


def test_f1_known_positive_wins():
    assert correct_win_rate([0.9, 0.8], [0.1, 0.2]) == 1.0


def test_f2_known_positive_losses():
    assert correct_win_rate([0.1, 0.2], [0.9, 0.8]) == 0.0


def test_f3_exact_ties():
    assert correct_win_rate([0.4, 0.4], [0.4, 0.4]) == 0.5


def test_f4_mixed_wins_losses_ties():
    assert correct_win_rate([0.9, 0.1, 0.5, 0.7], [0.2, 0.8, 0.5, 0.9]) == 0.375


def test_f5_aligned_and_shifted_evidence_differ():
    aligned = correct_win_rate([0.9, 0.8], [0.1, 0.2])
    shifted = correct_win_rate([0.1, 0.2], [0.9, 0.8])
    assert aligned == 1.0 and shifted == 0.0 and aligned != shifted


def test_f6_row_major_37x37_to_1369_identity():
    mask = np.zeros((B2.IMAGE_SIZE, B2.IMAGE_SIZE), dtype=np.uint8)
    patch_y, patch_x = 2, 3
    mask[patch_y * B2.PATCH_STRIDE, patch_x * B2.PATCH_STRIDE] = 1
    occupancy = B2.occupancy_from_mask(mask)
    patch_id = patch_y * B2.PATCH_GRID[1] + patch_x
    assert occupancy.shape == (B2.PATCH_COUNT,)
    assert occupancy[patch_id] > 0
    assert int(np.flatnonzero(occupancy > 0)[0]) == patch_id


def test_f7_matched_indices_retrieve_distinct_known_values():
    evidence = np.array([0.0, 0.2, 0.9, 0.1, 0.8, 0.7], dtype=np.float32)
    pos = np.array([2, 5], dtype=np.int64)
    neg = np.array([1, 4], dtype=np.int64)
    assert np.array_equal(evidence[pos], np.array([0.9, 0.7], dtype=np.float32))
    assert np.array_equal(evidence[neg], np.array([0.2, 0.8], dtype=np.float32))
    assert B2.matched_win(evidence, pos, neg) == 0.5
    assert correct_win_rate(evidence[pos], evidence[neg]) == 0.5


def test_f8_bootstrap_nonidentical_vector_has_nonzero_width():
    ci = B2.bootstrap_ci([0.2, 0.4, 0.6, 0.8], 5201)
    assert ci is not None
    assert ci[1] > ci[0]


def test_original_b2_call_collapses_to_self_comparison():
    e_pos = np.array([0.9, 0.8], dtype=np.float32)
    e_neg = np.array([0.1, 0.2], dtype=np.float32)
    original_call = B2.matched_win(e_pos, np.arange(e_pos.size), np.arange(e_neg.size))
    assert original_call == 0.5
    assert correct_win_rate(e_pos, e_neg) == 1.0
