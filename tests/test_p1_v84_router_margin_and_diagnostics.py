from __future__ import annotations

import time

import pytest
import torch

from model.h6.specialization_trajectory import (
    aggregate_trajectory_milestone,
    aggregate_utility_records,
    binary_auroc,
    capture_utility_record,
)
from model.h6.utility_routing import (
    act_teacher,
    router_target_distribution,
    routed_residual_correction,
    utility_router_loss,
    utility_teacher,
)


def _legacy_binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Pre-optimization reference used only for small exactness tests."""
    scores = scores.flatten().float()
    labels = labels.flatten().bool()
    positives = int(labels.sum().item())
    negatives = int((~labels).sum().item())
    if not positives or not negatives:
        return None
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    for value in scores.unique():
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    return float(
        ((ranks[labels].sum() - positives * (positives + 1) / 2.0)
         / float(positives * negatives)).item()
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base = torch.zeros(1, 1, 4)
    # Patches 0/1 have a useful F1; patch 1 has a tied best and patch 2 has no
    # useful factor.  This makes margin eligibility independently observable.
    residual = torch.tensor(
        [[[
            [8.0, 0.0, -4.0, -8.0],
            [8.0, 8.0, -4.0, -8.0],
            [-1.0, -2.0, -4.0, -8.0],
            [8.0, 4.0, -4.0, -8.0],
        ]]]
    )
    target = torch.ones(1, 4)
    valid = torch.ones_like(target, dtype=torch.bool)
    return base, residual, target, valid


def _teacher(**kwargs: object) -> dict[str, torch.Tensor]:
    base, residual, target, valid = _inputs()
    return utility_teacher(
        base, residual, target, valid, rho=0.05, epsilon=0.0,
        routed_probabilities=torch.softmax(
            torch.tensor([[[
                [2.0, 1.0, 0.0, -1.0],
                [2.0, 1.0, 0.0, -1.0],
                [2.0, 1.0, 0.0, -1.0],
                [2.0, 1.0, 0.0, -1.0],
            ]]]),
            dim=-1,
        ),
        **kwargs,
    )


def test_binary_auroc_matches_legacy_on_small_ties_and_edge_cases():
    scores = torch.tensor([0.1, 0.1, 0.2, 0.5, 0.5, 0.5, 0.9])
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 1], dtype=torch.bool)
    assert binary_auroc(scores, labels) == pytest.approx(
        _legacy_binary_auroc(scores, labels), abs=1e-7
    )
    assert binary_auroc(torch.ones(5), torch.tensor([0, 1, 0, 1, 0])) == pytest.approx(0.5)
    assert binary_auroc(torch.arange(5.0), torch.ones(5, dtype=torch.bool)) is None
    assert binary_auroc(torch.arange(5.0), torch.zeros(5, dtype=torch.bool)) is None


def test_margin_router_eligibility_has_exact_positive_and_relative_margin_boundary():
    payload = _teacher(router_confidence_mode="margin_rel", router_margin_rel_threshold=0.10)
    # Useful and separated / tied best / no positive gain / useful and separated.
    assert torch.equal(payload["informative"], torch.tensor([[[True, False, False, True]]]))
    assert payload["best_gain_rel"][0, 0, 2] <= 0.0
    assert payload["margin_rel"][0, 0, 1] == pytest.approx(0.0)


def test_margin_threshold_changes_only_router_informative_mask():
    low = _teacher(router_confidence_mode="margin_rel", router_margin_rel_threshold=0.10)
    high = _teacher(router_confidence_mode="margin_rel", router_margin_rel_threshold=0.90)
    assert not torch.equal(low["informative"], high["informative"])
    for key in (
        "candidate_logits", "gain_rel", "q_factor_utility", "q_router_utility",
        "q_utility", "responsibility", "normalized_entropy", "winner",
        "best_gain_rel", "second_gain_rel", "margin_abs", "margin_rel",
    ):
        assert torch.equal(low[key], high[key])


def test_router_tau_changes_q_but_not_margin_eligibility_or_act_teacher():
    baseline = _teacher(router_confidence_mode="margin_rel", router_tau_utility=0.05)
    changed = _teacher(router_confidence_mode="margin_rel", router_tau_utility=0.02)
    assert not torch.equal(baseline["q_router_utility"], changed["q_router_utility"])
    assert not torch.equal(baseline["normalized_entropy"], changed["normalized_entropy"])
    for key in ("gain_rel", "responsibility", "winner", "margin_abs", "margin_rel", "informative"):
        assert torch.equal(baseline[key], changed[key])
    for key in ("positive", "negative", "ambiguous", "support"):
        assert torch.equal(
            act_teacher(baseline, gain_threshold=0.0)[key],
            act_teacher(changed, gain_threshold=0.0)[key],
        )


def test_factor_tau_changes_responsibility_only_and_legacy_entropy_is_exact():
    baseline = _teacher(router_confidence_mode="margin_rel", factor_tau_utility=0.05)
    changed = _teacher(router_confidence_mode="margin_rel", factor_tau_utility=0.02)
    assert not torch.equal(baseline["responsibility"], changed["responsibility"])
    for key in ("gain_rel", "q_router_utility", "normalized_entropy", "winner", "margin_rel", "informative"):
        assert torch.equal(baseline[key], changed[key])
    legacy = _teacher()
    explicit = _teacher(router_confidence_mode="entropy", router_margin_rel_threshold=0.10)
    for key in legacy:
        assert torch.equal(legacy[key], explicit[key])


def test_margin_router_change_preserves_exact_act_noop_and_zero_logit_probability():
    payload = _teacher(router_confidence_mode="margin_rel")
    probabilities = torch.softmax(torch.randn_like(payload["gain_rel"]), dim=-1)
    residual = torch.randn_like(probabilities)
    correction = routed_residual_correction(torch.zeros_like(payload["best_gain_rel"]), probabilities, residual)
    assert torch.equal(correction, torch.zeros_like(correction))
    assert torch.equal(torch.sigmoid(torch.zeros_like(payload["best_gain_rel"])), torch.full_like(payload["best_gain_rel"], 0.5))


def test_patch_zscore_router_target_preserves_order_scale_translation_and_all_nonrouter_state():
    baseline = _teacher(router_confidence_mode="margin_rel")
    zscore = _teacher(router_confidence_mode="margin_rel", router_target_mode="patch_zscore_softmax")
    for key in ("candidate_logits", "gain_rel", "q_factor_utility", "responsibility", "winner", "margin_rel", "informative"):
        assert torch.equal(baseline[key], zscore[key])
    for key in ("positive", "negative", "ambiguous", "support"):
        assert torch.equal(act_teacher(baseline, gain_threshold=0.0)[key], act_teacher(zscore, gain_threshold=0.0)[key])
    gain = torch.tensor([[0.4, 0.1, -0.1, -0.4]])
    q, zero = router_target_distribution(gain, tau_utility=0.05, mode="patch_zscore_softmax")
    scaled, _ = router_target_distribution(7.0 * gain, tau_utility=0.05, mode="patch_zscore_softmax")
    shifted, _ = router_target_distribution(gain + 9.0, tau_utility=0.05, mode="patch_zscore_softmax")
    assert torch.isfinite(q).all() and not zero.any()
    assert torch.equal(q.argmax(dim=-1), gain.argmax(dim=-1))
    assert torch.equal(q.argsort(dim=-1), gain.argsort(dim=-1))
    assert torch.allclose(q, scaled) and torch.allclose(q, shifted)


def test_patch_zscore_zero_spread_is_finite_and_legacy_mode_is_exact():
    gain = torch.zeros(2, 4)
    q, zero = router_target_distribution(gain, tau_utility=0.05, mode="patch_zscore_softmax")
    legacy, legacy_zero = router_target_distribution(gain, tau_utility=0.05)
    assert torch.isfinite(q).all() and zero.all()
    assert torch.equal(q, torch.full_like(q, 0.25))
    assert torch.equal(legacy, torch.full_like(legacy, 0.25)) and not legacy_zero.any()


def test_fast_trajectory_defers_only_intermediate_cumulative_and_final_is_legacy_exact():
    payload = _teacher(router_confidence_mode="margin_rel")
    router = payload["q_utility"]
    record = capture_utility_record(payload, router, torch.ones(1, 4), utility_router_loss(router, payload))
    kwargs = dict(gain_threshold=0.02, entropy_threshold=0.98, router_confidence_mode="margin_rel")
    direct = aggregate_utility_records([record, record], **kwargs)
    legacy, legacy_recent = aggregate_trajectory_milestone(
        [record, record], [record], aggregation_mode="legacy", is_final=False, **kwargs
    )
    fast, fast_recent = aggregate_trajectory_milestone(
        [record, record], [record], aggregation_mode="fast", is_final=False, **kwargs
    )
    final, _ = aggregate_trajectory_milestone(
        [record, record], [record], aggregation_mode="fast", is_final=True, **kwargs
    )
    assert fast["cumulative_deferred"] is True
    for key in ("Base", "OracleMulti", "router_supervised_patch_count", "router_margin_pass_fraction"):
        assert legacy[key] == pytest.approx(direct[key])
        assert final[key] == pytest.approx(legacy[key])
        assert fast_recent[key] == pytest.approx(legacy_recent[key])
