import math

import torch
import torch.nn.functional as F

from model.h6.model import H6Progress1
from model.h6.utility_routing import (
    act_teacher,
    effective_number_act_loss,
    routed_residual_correction,
    support_normalized_utility_router_loss,
    utility_teacher,
)
from train import grad_accum_window_size, primary_anchored_factor_surgery


def _teacher(base, residual, target, *, gain_threshold=0.02):
    valid = torch.ones_like(target, dtype=torch.bool)
    utility = utility_teacher(
        base, residual, target, valid,
        rho=0.05, epsilon=0.0, gain_threshold=gain_threshold,
    )
    return utility, act_teacher(utility, gain_threshold=gain_threshold)


def _decoupling_inputs():
    base = torch.zeros(1, 1, 3)
    residual = torch.tensor(
        [[[[8.0, 4.0, 0.0, -4.0],
           [4.0, 2.0, 0.0, -2.0],
           [-8.0, -4.0, -2.0, -1.0]]]]
    )
    target = torch.ones(1, 3)
    valid = torch.ones_like(target, dtype=torch.bool)
    return base, residual, target, valid


def _decoupled_teacher(**kwargs):
    base, residual, target, valid = _decoupling_inputs()
    act_gain_threshold = kwargs.pop("act_gain_threshold", 0.02)
    utility = utility_teacher(
        base, residual, target, valid,
        rho=0.05, epsilon=0.0, entropy_threshold=1.01, **kwargs,
    )
    act = act_teacher(utility, gain_threshold=act_gain_threshold)
    return utility, act


def test_legacy_and_explicit_decoupled_defaults_are_exactly_equivalent():
    legacy, legacy_act = _decoupled_teacher(
        tau_utility=0.05, gain_threshold=0.02
    )
    explicit, explicit_act = _decoupled_teacher(
        tau_utility=0.05,
        factor_tau_utility=0.05,
        router_tau_utility=0.05,
        gain_threshold=0.02,
        router_gain_threshold=0.02,
    )
    for key in (
        "candidate_logits", "gain_rel", "responsibility", "q_utility",
        "normalized_entropy", "informative",
    ):
        assert torch.equal(legacy[key], explicit[key])
    for key in ("positive", "negative", "ambiguous"):
        assert torch.equal(legacy_act[key], explicit_act[key])


def test_factor_tau_changes_only_factor_responsibility():
    baseline, baseline_act = _decoupled_teacher()
    changed, changed_act = _decoupled_teacher(factor_tau_utility=0.01)
    assert not torch.equal(baseline["responsibility"], changed["responsibility"])
    for key in (
        "candidate_logits", "gain_rel", "q_utility", "normalized_entropy",
        "informative",
    ):
        assert torch.equal(baseline[key], changed[key])
    for key in ("positive", "negative", "ambiguous"):
        assert torch.equal(baseline_act[key], changed_act[key])


def test_router_tau_changes_only_router_teacher_confidence():
    baseline, baseline_act = _decoupled_teacher()
    changed, changed_act = _decoupled_teacher(router_tau_utility=0.01)
    assert not torch.equal(baseline["q_utility"], changed["q_utility"])
    assert not torch.equal(baseline["normalized_entropy"], changed["normalized_entropy"])
    for key in ("candidate_logits", "gain_rel", "responsibility"):
        assert torch.equal(baseline[key], changed[key])
    for key in ("positive", "negative", "ambiguous"):
        assert torch.equal(baseline_act[key], changed_act[key])


def test_router_gain_threshold_changes_only_informative_mask():
    baseline, baseline_act = _decoupled_teacher(router_gain_threshold=0.0)
    changed, changed_act = _decoupled_teacher(router_gain_threshold=0.30)
    assert not torch.equal(baseline["informative"], changed["informative"])
    for key in ("responsibility", "q_utility"):
        assert torch.equal(baseline[key], changed[key])
    for key in ("positive", "negative", "ambiguous"):
        assert torch.equal(baseline_act[key], changed_act[key])


def test_act_gain_threshold_changes_only_act_positive_and_ambiguous_masks():
    baseline, baseline_act = _decoupled_teacher(act_gain_threshold=0.02)
    changed, changed_act = _decoupled_teacher(act_gain_threshold=0.30)
    assert not torch.equal(baseline_act["positive"], changed_act["positive"])
    assert not torch.equal(baseline_act["ambiguous"], changed_act["ambiguous"])
    for key in ("q_utility", "informative", "responsibility"):
        assert torch.equal(baseline[key], changed[key])


def test_noop_and_identical_reference_residual_are_exact_zero():
    reference = torch.randn(3, 2, 5)
    factors = reference.unsqueeze(-1).expand(-1, -1, -1, 4).clone()
    residual = factors - reference.unsqueeze(-1)
    assert torch.equal(residual, torch.zeros_like(residual))


def test_act_zero_is_exact_identity_and_act_one_uses_routed_residual():
    probabilities = torch.softmax(torch.randn(3, 2, 5, 4), dim=-1)
    residual = torch.randn_like(probabilities)
    zero = routed_residual_correction(
        torch.zeros(3, 2, 5), probabilities, residual
    )
    one = routed_residual_correction(
        torch.ones(3, 2, 5), probabilities, residual
    )
    expected = (probabilities * residual).sum(dim=-1)
    assert torch.equal(zero, torch.zeros_like(zero))
    assert torch.allclose(one, expected)


def test_soft_act_interpolation_is_linear():
    probabilities = torch.softmax(torch.randn(1, 1, 3, 4), dim=-1)
    residual = torch.randn_like(probabilities)
    routed = (probabilities * residual).sum(dim=-1)
    actual = routed_residual_correction(
        torch.full_like(routed, 0.25), probabilities, residual
    )
    assert torch.allclose(actual, 0.25 * routed)


def test_residual_teacher_uses_residual_candidates_not_absolute_logits():
    base = torch.zeros(1, 1, 1)
    residual = torch.tensor([[[[-2.0, -1.0, 1.0, 2.0]]]])
    utility, _ = _teacher(base, residual, torch.ones(1, 1))
    assert torch.allclose(
        utility["candidate_logits"], base.unsqueeze(-1) + 0.05 * residual
    )
    absolute = residual + 9.0
    assert not torch.allclose(
        utility["candidate_logits"], base.unsqueeze(-1) + 0.05 * absolute
    )


def test_act_teacher_negative_positive_and_ambiguous_zones():
    payload = {
        "best_gain_rel": torch.tensor([[[-0.1, 0.01, 0.021]]]),
        "valid": torch.ones(1, 1, 3, dtype=torch.bool),
    }
    teacher = act_teacher(payload, gain_threshold=0.02)
    assert teacher["target"].tolist() == [[[0.0, 0.0, 1.0]]]
    assert teacher["negative"].tolist() == [[[True, False, False]]]
    assert teacher["ambiguous"].tolist() == [[[False, True, False]]]
    assert teacher["positive"].tolist() == [[[False, False, True]]]
    assert teacher["support"].tolist() == [[[True, False, True]]]


def test_router_supervision_requires_act_positive_and_informative_support():
    base = torch.zeros(1, 1, 2)
    # Anomaly patch 0 has a distinct useful positive residual; patch 1 is
    # forced harmful and therefore must not supervise which-factor routing.
    residual = torch.tensor([[[[8.0, -8.0, -8.0, -8.0],
                               [-8.0, -7.0, -6.0, -5.0]]]])
    utility, act = _teacher(base, residual, torch.ones(1, 2))
    assert act["positive"][0, 0, 0]
    assert act["negative"][0, 0, 1]
    assert utility["informative"][0, 0, 0]
    assert not utility["informative"][0, 0, 1]
    uniform_router = torch.full_like(residual, 0.25)
    loss = support_normalized_utility_router_loss(uniform_router, utility)
    expected = math.log(4.0) / 2.0
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_act_loss_effective_number_region_weighting_and_ambiguous_exclusion():
    logits = torch.tensor([[[0.0, 1.0, -1.0, 100.0]]])
    teacher = {
        "target": torch.tensor([[[1.0, 0.0, 1.0, 0.0]]]),
        "support": torch.tensor([[[True, True, True, False]]]),
    }
    y_patch = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    beta = 0.9
    actual = effective_number_act_loss(logits, teacher, y_patch, beta=beta)
    losses = F.binary_cross_entropy_with_logits(
        logits, teacher["target"], reduction="none"
    )
    normal_weight = 1.0
    anomaly_effective_n = (1.0 - beta ** 2) / (1.0 - beta)
    anomaly_weight = 1.0 / anomaly_effective_n
    expected = (
        losses[0, 0, 0] * normal_weight
        + losses[0, 0, 1] * anomaly_weight
        + losses[0, 0, 2] * anomaly_weight
    ) / (normal_weight + 2.0 * anomaly_weight)
    assert torch.allclose(actual, expected)


def test_v84a_metadata_and_minimum_act_capacity_contract():
    model = H6Progress1(
        n_groups=3, num_factors=4, bank_dim=16, router_dim=8,
        text_dim=16, vae_hidden_dim=16, vae_latent_dim=8,
        progress_version="P1-v8.4-A",
    )
    config = model.config_dict()
    assert model.act_head is not None
    assert config["progress_version"] == "P1-v8.4-A"
    assert config["local_correction_semantics"] == "act_times_routed_true_residual"
    assert config["noop_reference"] == "expected_noop_pre_expert_bank"
    assert config["act_probability_mode"] == "continuous_sigmoid"
    assert config["act_parameter_count"] == 49
    act_logits = model.act_head(torch.randn(3, 2, 7, 16)).squeeze(-1)
    assert torch.equal(act_logits, torch.zeros_like(act_logits))
    assert torch.equal(torch.sigmoid(act_logits), torch.full_like(act_logits, 0.5))
    assert torch.equal(model.rho_values(), torch.full((3,), 0.05))
    patches = F.normalize(torch.randn(3, 1, 7, 16), dim=-1)
    noop_text = F.normalize(torch.randn(3, 1, 16, 2), dim=2)
    noop_logits = model.h6_logit(patches, noop_text.unsqueeze(2))
    assert noop_logits.shape == (3, 1, 7)


def test_v83_path_has_no_act_parameters_and_keeps_absolute_semantics():
    model = H6Progress1(
        n_groups=3, num_factors=4, bank_dim=16, router_dim=8,
        text_dim=16, vae_hidden_dim=16, vae_latent_dim=8,
        progress_version="P1-v8.3",
    )
    config = model.config_dict()
    assert model.act_head is None
    assert config["act_enabled"] is False
    assert config["local_correction_semantics"] == "routed_absolute_factor_margin"


def test_primary_main_gradient_and_accumulation_contract_remain_exact():
    main = [torch.tensor([1.0, 0.0])]
    factor = [torch.tensor([-1.0, 1.0])]
    safe_main, _, _ = primary_anchored_factor_surgery(main, factor)
    assert torch.equal(safe_main[0], main[0])
    assert [grad_accum_window_size(i, 8, 6) for i in range(1, 9)] == [6] * 6 + [2] * 2
