from __future__ import annotations

import inspect

import numpy as np
import pytest

from tools.phase5_selective_adjudication import (
    PATCH_COUNT,
    STAGES,
    K,
    apply_positive_only_projection,
    deploy_pre_softmax,
    deploy_native_logits,
    select_disjoint,
    select_gt_free,
)


def _arrays():
    m = np.arange(PATCH_COUNT, dtype=np.float32)
    d = np.zeros(PATCH_COUNT, dtype=np.float32)
    valid = np.ones(PATCH_COUNT, dtype=bool)
    e = np.zeros(PATCH_COUNT, dtype=np.float32)
    stage = np.zeros((STAGES, PATCH_COUNT), dtype=np.float32)
    loo = np.zeros((K, PATCH_COUNT), dtype=np.float32)
    return m, d, valid, e, stage, loo


def _single_relation_arrays(m_values=(0.0, 1.0), e_values=(2.0, 1.0)):
    m, d, valid, e, stage, loo = _arrays()
    # Put the strict inversion in one exact cell and make all other patches
    # harmless by assigning their own distinct score/evidence order.
    m[:] = np.arange(PATCH_COUNT, dtype=np.float32)
    e[:] = np.arange(PATCH_COUNT, dtype=np.float32)
    m[0], m[1] = m_values
    e[0], e[1] = e_values
    stage[:] = e[None, :]
    loo[:] = e[None, :]
    return m, d, valid, e, stage, loo


def test_identity_and_satisfied_relation_have_no_action():
    m, d, valid, e, stage, loo = _arrays()
    result = select_gt_free(m, d, valid, e, stage, loo)
    assert result["selected"] == []


def test_one_inversion_is_certified_and_projected():
    m, d, valid, e, stage, loo = _single_relation_arrays()
    result = select_gt_free(m, d, valid, e, stage, loo)
    assert result["selected"]
    native = np.zeros((STAGES, PATCH_COUNT, 2), dtype=np.float32)
    corrected, delta = apply_positive_only_projection(native, m, result["selected"])
    i, j, cost = result["selected"][0]
    assert np.isclose(delta[i], m[j] - m[i])
    assert delta[j] == 0
    assert np.all(corrected[:, i, 1] > native[:, i, 1])
    assert np.array_equal(corrected[:, :, 0], native[:, :, 0])


def test_exact_tie_stage_or_loo_disagreement_abstains():
    m, d, valid, e, stage, loo = _single_relation_arrays()
    stage[0, 0] = stage[0, 1]
    assert select_gt_free(m, d, valid, e, stage, loo)["certified"] == []
    m, d, valid, e, stage, loo = _single_relation_arrays()
    loo[0, 0] = loo[0, 1]
    assert select_gt_free(m, d, valid, e, stage, loo)["certified"] == []


def test_conflicting_or_repeated_relations_rejected():
    with pytest.raises(ValueError):
        apply_positive_only_projection(np.zeros((STAGES, PATCH_COUNT, 2), dtype=np.float32), np.arange(PATCH_COUNT), [(0, 1, 1.0), (1, 2, 1.0)])


def test_minimum_cost_order_is_deterministic_and_disjoint():
    chosen = select_disjoint([(0, 1, 2.0), (2, 3, 1.0), (0, 4, 0.5), (5, 6, 1.0)])
    assert chosen == [(0, 4, 0.5), (2, 3, 1.0), (5, 6, 1.0)]
    assert len({p for i, j, _ in chosen for p in (i, j)}) == 2 * len(chosen)


def test_projection_preserves_unrelated_and_normal_logits():
    m, d, valid, e, stage, loo = _single_relation_arrays()
    selected = select_gt_free(m, d, valid, e, stage, loo)["selected"]
    native = np.random.default_rng(9).normal(size=(STAGES, PATCH_COUNT, 2)).astype(np.float32)
    corrected, delta = apply_positive_only_projection(native, m, selected)
    acted = np.flatnonzero(delta > 0)
    unrelated = np.setdiff1d(np.arange(PATCH_COUNT), acted)
    assert np.array_equal(corrected[:, :, 0], native[:, :, 0])
    assert np.array_equal(corrected[:, unrelated, 1], native[:, unrelated, 1])
    assert np.all(delta >= 0)


def test_nan_inf_and_shape_rejection():
    m, d, valid, e, stage, loo = _arrays()
    m[0] = np.nan
    with pytest.raises(ValueError):
        select_gt_free(m, d, valid, e, stage, loo)
    with pytest.raises(ValueError):
        apply_positive_only_projection(np.zeros((STAGES, PATCH_COUNT, 2), dtype=np.float32), np.zeros(PATCH_COUNT), [(0, 1, 1.0)])


def test_deployment_impulse_sign_and_gt_free_interfaces():
    native = np.zeros((STAGES, PATCH_COUNT, 2), dtype=np.float32)
    native[:, 100, 1] = 1.0
    prob = deploy_native_logits(native)
    assert prob.shape == (1, 2, 518, 518)
    assert np.all(np.isfinite(prob))
    baseline = deploy_pre_softmax(np.zeros_like(native))
    perturbed = deploy_pre_softmax(native)
    impulse_delta = (perturbed - baseline).numpy()
    # Positive anomaly-only impulse cannot create a negative pre-softmax
    # anomaly perturbation under the fixed blur/interpolation/mean operator.
    assert np.min(impulse_delta[:, 1]) >= -1e-7
    assert np.max(np.abs(impulse_delta[:, 0])) <= 1e-7
    assert np.all(prob >= 0.0) and np.all(prob <= 1.0)
    for name in ("select_gt_free", "apply_positive_only_projection"):
        parameters = inspect.signature(globals()[name]).parameters
        assert not {"gt", "mask", "label", "labels"}.intersection(parameters)
