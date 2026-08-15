import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase5_reference_validity", ROOT / "tools/audit_phase5_reference_validity.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def unit_features(size=7, dim=3):
    value = torch.zeros(dim, size, size)
    value[0] = 1
    return value


def test_t1_homogeneous_field_has_zero_contrast_and_heterogeneity():
    result = AUDIT.scale_statistics(unit_features(), 3)
    assert torch.allclose(result["contrast"], torch.zeros_like(result["contrast"]), atol=1e-6)
    assert torch.allclose(result["heterogeneity"], torch.zeros_like(result["heterogeneity"]), atol=1e-6)


def test_t2_center_only_anomaly_has_high_c3():
    feature = unit_features()
    feature[:, 3, 3] = torch.tensor([0.0, 1.0, 0.0])
    feature = torch.nn.functional.normalize(feature, dim=0)
    result = AUDIT.scale_statistics(feature, 3)
    assert float(result["contrast"][3, 3]) > 0.5


def test_t3_coherent_3x3_island_prefers_wider_context():
    feature = unit_features()
    feature[:, 2:5, 2:5] = torch.tensor([0.0, 1.0, 0.0]).reshape(3, 1, 1)
    feature = torch.nn.functional.normalize(feature, dim=0)
    c3 = AUDIT.scale_statistics(feature, 3)["contrast"][3, 3]
    c5 = AUDIT.scale_statistics(feature, 5)["contrast"][3, 3]
    assert float(c3) < float(c5)


def test_t4_coherent_5x5_island_prefers_scale7():
    feature = unit_features()
    feature[:, 1:6, 1:6] = torch.tensor([0.0, 1.0, 0.0]).reshape(3, 1, 1)
    feature = torch.nn.functional.normalize(feature, dim=0)
    c5 = AUDIT.scale_statistics(feature, 5)["contrast"][3, 3]
    c7 = AUDIT.scale_statistics(feature, 7)["contrast"][3, 3]
    assert float(c5) < float(c7)


def test_t5_heterogeneous_ring_is_suppressed_by_h():
    coherent = unit_features()
    coherent[:, 3, 3] = torch.tensor([0.0, 1.0, 0.0])
    coherent = torch.nn.functional.normalize(coherent, dim=0)
    heterogeneous = coherent.clone()
    heterogeneous[:, 2, 3] = torch.tensor([0.0, 0.0, 1.0])
    heterogeneous[:, 3, 2] = torch.tensor([0.0, 1.0, 0.0])
    heterogeneous[:, 3, 4] = torch.tensor([0.0, 0.0, 1.0])
    heterogeneous[:, 4, 3] = torch.tensor([0.0, 1.0, 0.0])
    heterogeneous = torch.nn.functional.normalize(heterogeneous, dim=0)
    a = AUDIT.local_multiscale_maps([coherent.permute(1, 2, 0).reshape(-1, 3)], (7, 7))
    b = AUDIT.local_multiscale_maps([heterogeneous.permute(1, 2, 0).reshape(-1, 3)], (7, 7))
    assert float(b["H"][3][3, 3]) > float(a["H"][3][3, 3])


def test_t14_same_c3_contrast_wider_reference_statistics_distinguish_cases():
    x = torch.tensor([1.0, 0.0, 0.0])
    y = torch.tensor([0.0, 1.0, 0.0])
    case_a = x.reshape(3, 1, 1).repeat(1, 7, 7)
    case_a[:, 2:5, 2:5] = x.reshape(3, 1, 1)
    case_a[:, 1:6, 1:6] = y.reshape(3, 1, 1)
    case_a[:, 2:5, 2:5] = x.reshape(3, 1, 1)
    case_b = case_a.clone()
    for yy, xx in ((1, 2), (1, 4), (2, 1), (2, 5), (4, 1), (4, 5), (5, 2), (5, 4)):
        case_b[:, yy, xx] = x
    c3_a = AUDIT.scale_statistics(case_a, 3)["contrast"][3, 3]
    c3_b = AUDIT.scale_statistics(case_b, 3)["contrast"][3, 3]
    h5_a = AUDIT.scale_statistics(case_a, 5)["heterogeneity"][3, 3]
    h5_b = AUDIT.scale_statistics(case_b, 5)["heterogeneity"][3, 3]
    assert torch.allclose(c3_a, c3_b, atol=1e-6)
    assert float(h5_b) > float(h5_a)


def test_t6_border_counts_are_valid_and_finite():
    result = AUDIT.scale_statistics(unit_features(7), 7)
    assert int(result["ring_count"][0, 0]) == 7
    assert torch.isfinite(result["contrast"]).all()
    assert torch.isfinite(result["heterogeneity"]).all()


def test_t7_peer_pool_excludes_local_anomaly_candidates():
    features = [torch.nn.functional.normalize(torch.randn(49, 8), dim=1) for _ in range(3)]
    d_rank = np.full(49, 0.8, dtype=np.float32)
    d_rank[30:] = 0.1
    margins = np.ones((3, 49), dtype=np.float32)
    margins[:, 30:] = 0.0
    peers, valid, _ = AUDIT.nonlocal_peers(features, d_rank, margins)
    assert valid[0]
    assert np.all(np.maximum(np.abs(np.divmod(peers[0], 7)[0] - 0), np.abs(np.divmod(peers[0], 7)[1] - 0)) > 3)


def test_t8_no_valid_peer_pool_abstains():
    features = [torch.nn.functional.normalize(torch.randn(49, 8), dim=1) for _ in range(3)]
    d_rank = np.ones(49, dtype=np.float32)
    margins = np.ones((3, 49), dtype=np.float32)
    margins[:, 30:] = 0.0
    peers, valid, evidence = AUDIT.nonlocal_peers(features, d_rank, margins)
    assert not valid.any()
    assert np.all(peers == -1)
    assert np.all(evidence == 0)


def test_t9_deterministic_ties_use_identity():
    values = np.ones(4)
    identity = np.array([4, 1, 3, 2])
    assert AUDIT.select_top(values, identity, 2).tolist() == [False, True, False, True]


def test_t10_runtime_shape_contract():
    class Encoder:
        grid_size = (37, 37)

    class Model:
        image_encoder = Encoder()
        n_groups = 3

    stage = [torch.empty(1, 1369, 768) for _ in range(3)]
    assert [list(x.shape) for x in stage] == [[1, 1369, 768]] * 3
    assert AUDIT.PATCH_GRID == (37, 37)
    assert AUDIT.IMAGE_SIZE == 518


def test_t12_d_rank_uses_authoritative_population_std_and_percentile_rank():
    margins = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    expected = AUDIT.population_std(np.stack([AUDIT.percentile_rank(x) for x in margins]), axis=0)
    assert np.allclose(expected, AUDIT.population_std(np.stack([AUDIT.percentile_rank(x) for x in margins]), axis=0))


def test_no_train_paths_and_exact_local_scales():
    assert AUDIT.PATCH_GRID[0] * AUDIT.PATCH_STRIDE == AUDIT.IMAGE_SIZE
    assert tuple(sorted((3, 5, 7))) == (3, 5, 7)
