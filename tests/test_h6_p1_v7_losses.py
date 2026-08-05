import torch
from model.h6.losses import (
    assigned_expert_loss,
    compute_final_expert_hard_cosine,
    dual_routing_balance_loss,
    expert_clip_anchor_loss,
    expert_dead_counts,
    expert_etf_loss,
    expert_patch_function_diagnostics,
    factor_bank_against_reference_diagnostics,
    factor_bank_comparison_diagnostics,
    sum_loss_components,
)


def test_assigned_loss_and_empty_positive_are_finite():
    logits = torch.randn(1, 2, 4, 4, requires_grad=True)
    routing = torch.softmax(torch.randn_like(logits), -1)
    mask = torch.zeros(2, 1, 8, 8)
    loss, terms = assigned_expert_loss(logits, routing, mask)
    assert torch.isfinite(loss) and torch.isfinite(terms["expert_abnormal_assigned"])
    loss.backward()


def test_etf_target_and_balance_are_differentiable():
    # tetrahedron exactly has off-diagonal -1/3.
    tetra = torch.tensor([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])
    delta = tetra[None, None].requires_grad_()
    etf, _ = expert_etf_loss(delta)
    probs = torch.softmax(torch.randn(1, 1, 4, 4, requires_grad=True), -1)
    balance, _ = dual_routing_balance_loss(probs, probs)
    assert etf < 1e-6 and torch.isfinite(balance)


def test_assigned_patch_diagnostics_come_from_mask_and_are_finite():
    logits = torch.zeros(1, 1, 4, 4)
    indices = torch.tensor([[[[0, 1], [0, 1], [2, 3], [2, 3]]]])
    mask = torch.zeros(1, 1, 2, 2); mask[..., 0, 0] = 1
    metrics = expert_patch_function_diagnostics(logits, indices, mask, margin=.05)
    assert metrics["expert_abnormal_patch_count"] == 1
    assert metrics["expert_normal_patch_count"] == 3
    assert torch.isfinite(metrics["selected_expert_loss"])


def test_final_expert_hard_cosine_uses_embedding_axis_and_preserves_states():
    hard = torch.zeros(1, 1, 4, 2)
    hard[..., 0, 0] = 1.0
    hard[..., 1, 1] = 1.0
    final = hard.unsqueeze(2).expand(-1, -1, 3, -1, -1).clone()
    cosine = compute_final_expert_hard_cosine(final, hard)
    assert cosine.shape == (1, 1, 2)
    assert torch.allclose(cosine, torch.ones_like(cosine))
    loss, anchor_cosine = expert_clip_anchor_loss(final, hard, min_cosine=.70)
    assert loss == 0
    assert torch.equal(anchor_cosine, cosine)


def test_anchor_penalty_only_hits_abnormal_state_and_bf16_is_finite():
    hard = torch.zeros(1, 1, 4, 2)
    hard[..., 0, 0] = 1.0
    hard[..., 1, 1] = 1.0
    final = hard.unsqueeze(2).expand(-1, -1, 2, -1, -1).clone()
    final[..., 1] = 0.0
    final[..., 2, 1] = 1.0  # abnormal state becomes orthogonal to hard abnormal.
    cosine = compute_final_expert_hard_cosine(final.bfloat16(), hard.bfloat16())
    assert torch.isfinite(cosine).all()
    assert cosine[..., 0].min() > .95
    assert cosine[..., 1].max() < .01
    loss, _ = expert_clip_anchor_loss(final, hard, min_cosine=.70)
    assert loss > 0


def test_zero_residual_final_cosine_matches_pre_expert_cosine():
    hard = torch.randn(2, 3, 8, 2)
    pre_expert = hard + .02 * torch.randn_like(hard)
    final = pre_expert.unsqueeze(2).expand(-1, -1, 4, -1, -1).clone()
    assert torch.allclose(
        compute_final_expert_hard_cosine(final, hard),
        torch.nn.functional.cosine_similarity(
            pre_expert.float().movedim(-1, -2), hard.float().movedim(-1, -2), dim=-1,
        ),
    )


def test_dead_experts_are_counts_not_usage_values():
    usage = torch.tensor([[.25, .25, .25, .25], [0., .5, .5, 0.]])
    mask, count = expert_dead_counts(usage, threshold=.01)
    assert count.tolist() == [0, 2]
    assert mask.tolist() == [[False, False, False, False], [True, False, False, True]]


def test_zero_schedule_weights_keep_raw_losses_out_of_total():
    raw = {"expert_advantage": torch.tensor(2.0), "expert_etf": torch.tensor(3.0)}
    weighted = {name: value * 0.0 for name, value in raw.items()}
    total = sum_loss_components({"task": torch.tensor(1.0), **weighted})
    assert total == 1.0
    assert raw["expert_advantage"] > 0 and raw["expert_etf"] > 0


def test_noop_bank_diagnostics_preserve_factor_and_state_axes():
    hard = torch.randn(2, 3, 8, 2)
    expected_noop = hard.unsqueeze(2).expand(-1, -1, 4, -1, -1).clone()
    against_hard = factor_bank_against_reference_diagnostics(expected_noop, hard)
    exact = factor_bank_comparison_diagnostics(expected_noop, expected_noop)
    assert against_hard["cos_mean"] > .99999
    assert against_hard["max_abs_diff"] == 0
    assert exact["cos_min"] > .99999 and exact["max_abs_diff"] == 0
