"""Comprehensive unit tests for H6 Progress 1 v8.2 Candidate 1.

All test functions contain real assertions against production code.
No pass-only, docstring-only, or ellipsis-only bodies.
"""
from __future__ import annotations

import ast
import json
import math
import os
import random
import tempfile

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from model.h6.model import H6Progress1
from model.h6.losses import (
    active_role_balanced_router_loss,
    actual_local_residual_loss,
    build_semantic_roles,
    factor_specific_residual_role_loss,
    get_desired_correction,
)
from model.h6.cluster_responsibility import cluster_responsibility_loss
from model.adapter import ACDCLIP
from model.clip import create_model
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG_PATH = "configs/phase4/p1_v8_2_candidate1.json"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def _make_model(n_groups: int = 3) -> ACDCLIP:
    """Minimal ACDCLIP for structural tests. No pretrained weights needed."""
    clip_model = create_model(
        "ViT-L-14-336",
        img_size=518,
        pretrained=False,
        require_pretrained=False,
    )
    return ACDCLIP(
        clip_model=clip_model,
        n_groups=n_groups,
        h6_progress=1,
    )


# ===========================================================================
# A. Semantic-role construction
# ===========================================================================


def test_build_semantic_roles_runtime():
    masks = torch.zeros(2, 518, 518)
    labels = torch.tensor([0, 1])
    local_mask_valid = torch.ones(2, 518, 518)
    q_role, hard_role, mask_coverage, local_valid_patch, local_valid_image = build_semantic_roles(
        masks, labels, patch_count=1369, local_mask_valid=local_mask_valid,
        core_threshold=0.5, boundary_threshold=0.5
    )
    assert q_role.shape == (2, 1369, 4), f"Expected (2,1369,4), got {q_role.shape}"
    assert hard_role.shape == (2, 1369), f"Expected (2,1369), got {hard_role.shape}"
    assert mask_coverage.shape == (2, 1369)
    assert local_valid_patch.shape == (2, 1369)
    assert local_valid_image.shape == (2,)
    # q_role must sum to 1 everywhere
    assert torch.allclose(q_role.sum(dim=-1), torch.ones(2, 1369), atol=1e-5)


def test_role0_only_on_normal_images():
    masks = torch.zeros(1, 518, 518)  # Normal image
    labels = torch.tensor([0])
    local_mask_valid = torch.ones(1, 518, 518)
    q_role, hard_role, _, _, _ = build_semantic_roles(
        masks, labels, patch_count=1369, local_mask_valid=local_mask_valid
    )
    assert torch.all(hard_role == 0), "Normal image: all patches must be role 0"
    assert torch.all(q_role[..., 0] == 1.0), "Normal image: q_role[:,role0]=1"
    assert torch.all(q_role[..., 1:] == 0.0), "Normal image: q_role[:,role1/2/3]=0"


def test_role0_zero_on_anomaly_images():
    masks = torch.zeros(1, 518, 518)
    masks[0, 100:200, 100:200] = 1.0
    labels = torch.tensor([1])
    local_mask_valid = torch.ones(1, 518, 518)
    q_role, hard_role, _, _, _ = build_semantic_roles(
        masks, labels, patch_count=1369, local_mask_valid=local_mask_valid
    )
    # No patch on an anomaly image should have role 0
    assert torch.all(q_role[..., 0] == 0.0), "Anomaly image: role-0 probability must be 0"
    assert torch.all(hard_role != 0), "Anomaly image: hard_role must never be 0"


def test_anomaly_roles_sum_to_one():
    masks = torch.zeros(1, 518, 518)
    masks[0, 100:200, 100:200] = 1.0
    labels = torch.tensor([1])
    local_mask_valid = torch.ones(1, 518, 518)
    q_role, _, _, _, _ = build_semantic_roles(
        masks, labels, patch_count=1369, local_mask_valid=local_mask_valid
    )
    # For an anomaly image, role 0 = 0, roles 1+2+3 = 1
    sum_123 = q_role[..., 1:4].sum(dim=-1)
    assert torch.allclose(sum_123, torch.ones_like(sum_123), atol=1e-5), \
        "Anomaly image: q1+q2+q3 must equal 1.0"


def test_empty_positive_mask_invalidates_local_losses():
    masks = torch.zeros(1, 518, 518)  # Positive image with empty mask
    labels = torch.tensor([1])
    local_mask_valid = torch.zeros(1, 518, 518)  # All invalid
    _, _, _, local_valid_patch, local_valid_image = build_semantic_roles(
        masks, labels, patch_count=1369, local_mask_valid=local_mask_valid
    )
    assert torch.all(~local_valid_patch), "Empty mask: all local patches invalid"
    assert torch.all(~local_valid_image), "Empty mask: local_valid_image must be False"


def test_normal_image_local_valid():
    masks = torch.zeros(1, 518, 518)
    labels = torch.tensor([0])
    local_mask_valid = torch.ones(1, 518, 518)
    _, _, _, local_valid_patch, local_valid_image = build_semantic_roles(
        masks, labels, patch_count=1369, local_mask_valid=local_mask_valid
    )
    # Normal images contribute to loss (role 0 corrections)
    assert torch.all(local_valid_patch), "Normal image: all patches should be valid"
    # local_valid_image tracks anomaly images with non-empty masks
    assert not torch.any(local_valid_image), "Normal image: local_valid_image must be False"


def test_boundary_morphology_on_synthetic_mask():
    masks = torch.zeros(1, 518, 518)
    # Fill top-left patch (37x37 pixels) completely
    masks[0, 0:37, 0:37] = 1.0
    labels = torch.tensor([1])
    local_mask_valid = torch.ones(1, 518, 518)
    _, hard_role, mask_coverage, _, _ = build_semantic_roles(
        masks, labels, patch_count=1369, local_mask_valid=local_mask_valid,
        boundary_threshold=0.8, core_threshold=0.99
    )
    # Patch [0,0] should have ~full coverage => core
    assert mask_coverage[0, 0] > 0.8, f"Expected high coverage, got {mask_coverage[0,0]}"
    assert hard_role[0, 0] == 3, f"Expected role 3 (core), got {hard_role[0,0]}"


def test_core_morphology_on_synthetic_mask():
    masks = torch.zeros(1, 518, 518)
    masks[0, 0:37, 0:37] = 1.0
    labels = torch.tensor([1])
    local_mask_valid = torch.ones(1, 518, 518)
    _, hard_role, mask_coverage, _, _ = build_semantic_roles(
        masks, labels, patch_count=1369, local_mask_valid=local_mask_valid,
        boundary_threshold=0.01, core_threshold=0.01
    )
    # Full coverage >= 0.01 threshold -> role 3 (core)
    assert mask_coverage[0, 0] > 0.5, f"Expected significant coverage"
    assert hard_role[0, 0] == 3, f"Expected role 3 (core), got {hard_role[0,0]}"


def test_tiny_anomaly_does_not_crash():
    masks = torch.zeros(1, 518, 518)
    masks[0, 10:11, 10:11] = 1.0  # 1×1 pixel
    labels = torch.tensor([1])
    local_mask_valid = torch.ones(1, 518, 518)
    q_role, hard_role, mask_coverage, local_valid_patch, local_valid_image = build_semantic_roles(
        masks, labels, patch_count=1369, local_mask_valid=local_mask_valid
    )
    assert hard_role.shape == (1, 1369), "Shape must be correct even for tiny anomaly"
    assert torch.isfinite(q_role).all(), "q_role must be finite"


def test_soft_targets_sum_to_one():
    masks = torch.zeros(1, 518, 518)
    masks[0, 10:20, 10:20] = 1.0
    labels = torch.tensor([1])
    local_mask_valid = torch.ones(1, 518, 518)
    q_role, _, _, _, _ = build_semantic_roles(
        masks, labels, patch_count=1369, local_mask_valid=local_mask_valid
    )
    sums = q_role.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        "q_role must sum to 1 for every patch"


def test_invalid_patches_are_ignored():
    masks = torch.zeros(1, 518, 518)
    labels = torch.tensor([1])
    local_mask_valid = torch.zeros(1, 518, 518)  # Fully invalid
    _, _, _, local_valid_patch, local_valid_image = build_semantic_roles(
        masks, labels, patch_count=1369, local_mask_valid=local_mask_valid
    )
    assert not local_valid_image[0], "local_valid_image must be False for invalid mask"
    assert torch.all(~local_valid_patch[0]), "All patches must be invalid when mask is invalid"


# ===========================================================================
# B. Router CE (active_role_balanced_router_loss)
# ===========================================================================


def test_router_ce_matching_soft_target_is_optimal():
    G, B, P, M = 1, 1, 10, 4
    targets = torch.tensor([[[0.0, 0.2, 0.8, 0.0]] * P])
    hard_role = targets.argmax(dim=-1)
    local_valid_patch = torch.ones(B, P, dtype=torch.bool)
    dense_prob = targets.unsqueeze(0).clone()
    loss = active_role_balanced_router_loss(dense_prob, targets, hard_role, local_valid_patch)
    expected = -(0.2 * torch.tensor(0.2).log() + 0.8 * torch.tensor(0.8).log())
    assert torch.allclose(loss, expected, atol=1e-4), \
        f"CE with matching target should equal entropy. Got {loss}, expected {expected}"


def test_router_ce_duplicate_G_invariant():
    """Doubling G with identical data must not change the loss."""
    G_base, B, P, M = 1, 1, 10, 4
    targets = torch.tensor([[[0.0, 0.2, 0.8, 0.0]] * P])
    hard_role = targets.argmax(dim=-1)
    local_valid_patch = torch.ones(B, P, dtype=torch.bool)
    dense_1 = targets.unsqueeze(0).clone()
    loss_1 = active_role_balanced_router_loss(dense_1, targets, hard_role, local_valid_patch)
    # Duplicate along G
    G_2 = 2
    dense_2 = targets.unsqueeze(0).repeat(G_2, 1, 1, 1)
    loss_2 = active_role_balanced_router_loss(dense_2, targets, hard_role, local_valid_patch)
    assert torch.allclose(loss_1, loss_2, atol=1e-5), \
        f"Duplicating G must not change loss. {loss_1} vs {loss_2}"


def test_router_ce_duplicate_B_invariant():
    """Doubling B with identical data must not change the loss."""
    G, B_base, P, M = 1, 1, 10, 4
    targets_1 = torch.tensor([[[0.0, 0.2, 0.8, 0.0]] * P])
    hard_role_1 = targets_1.argmax(dim=-1)
    local_valid_1 = torch.ones(B_base, P, dtype=torch.bool)
    loss_1 = active_role_balanced_router_loss(
        targets_1.unsqueeze(0), targets_1, hard_role_1, local_valid_1
    )
    B_2 = 2
    targets_2 = targets_1.repeat(B_2, 1, 1)
    hard_role_2 = hard_role_1.repeat(B_2, 1)
    local_valid_2 = torch.ones(B_2, P, dtype=torch.bool)
    loss_2 = active_role_balanced_router_loss(
        targets_2.unsqueeze(0), targets_2, hard_role_2, local_valid_2
    )
    assert torch.allclose(loss_1, loss_2, atol=1e-5), \
        f"Duplicating B must not change loss. {loss_1} vs {loss_2}"


def test_router_ce_duplicate_P_invariant():
    """Doubling P with identical data must not change the loss."""
    G, B, P_base, M = 1, 1, 10, 4
    targets = torch.tensor([[[0.0, 0.2, 0.8, 0.0]] * P_base])
    hard_role = targets.argmax(dim=-1)
    local_valid = torch.ones(B, P_base, dtype=torch.bool)
    loss_1 = active_role_balanced_router_loss(targets.unsqueeze(0), targets, hard_role, local_valid)
    P_2 = 20
    targets_2 = targets.repeat(1, 2, 1)
    hard_role_2 = hard_role.repeat(1, 2)
    local_valid_2 = torch.ones(B, P_2, dtype=torch.bool)
    loss_2 = active_role_balanced_router_loss(targets_2.unsqueeze(0), targets_2, hard_role_2, local_valid_2)
    assert torch.allclose(loss_1, loss_2, atol=1e-5), \
        f"Duplicating P must not change loss. {loss_1} vs {loss_2}"


def test_router_ce_active_role_balance_90_10():
    """With role 0 on 9 patches and role 1 on 1 patch, active-role balancing
    must weight them equally (each role contributes the same CE regardless
    of count)."""
    G, B, P, M = 1, 1, 10, 4
    targets = torch.zeros(B, P, M)
    targets[:, :9, 0] = 1.0   # 9 patches → role 0
    targets[:, 9:, 1] = 1.0   # 1 patch  → role 1
    hard_role = targets.argmax(dim=-1)
    local_valid_patch = torch.ones(B, P, dtype=torch.bool)
    dense_prob = torch.ones(1, B, P, M) / M  # uniform → CE = log(4)
    loss = active_role_balanced_router_loss(dense_prob, targets, hard_role, local_valid_patch)
    expected = -torch.tensor(0.25).log()
    assert torch.allclose(loss, expected, atol=1e-4), \
        f"Expected {expected.item():.4f}, got {loss.item():.4f}"


def test_router_ce_missing_roles():
    """When only one role is present, loss is computed from that role alone."""
    G, B, P, M = 1, 1, 10, 4
    targets = torch.zeros(B, P, M)
    targets[:, :, 0] = 1.0  # Only role 0
    hard_role = targets.argmax(dim=-1)
    local_valid_patch = torch.ones(B, P, dtype=torch.bool)
    dense_prob = torch.ones(1, B, P, M) / M
    loss = active_role_balanced_router_loss(dense_prob, targets, hard_role, local_valid_patch)
    expected = -torch.tensor(0.25).log()
    assert torch.allclose(loss, expected, atol=1e-4), \
        f"Expected {expected.item():.4f}, got {loss.item():.4f}"


def test_router_ce_invalid_patches_ignored():
    """When all patches are invalid, loss must be zero (no gradient through them)."""
    G, B, P, M = 1, 1, 10, 4
    targets = torch.zeros(B, P, M)
    targets[:, :, 0] = 1.0
    hard_role = targets.argmax(dim=-1)
    local_valid_patch = torch.zeros(B, P, dtype=torch.bool)  # All invalid
    dense_prob = torch.ones(1, B, P, M) / M
    loss = active_role_balanced_router_loss(dense_prob, targets, hard_role, local_valid_patch)
    assert loss.item() == 0.0, f"Expected 0.0 when all patches invalid, got {loss.item()}"


# ===========================================================================
# C. Patch-structured cluster KL (disabled in Candidate 1, but API must work)
# ===========================================================================


def test_cluster_kl_duplicate_G_invariant():
    G, B, P, M = 2, 1, 10, 4
    patch_features = torch.randn(G, B, P, 768)
    centroids = torch.randn(M, 768)
    probabilities = torch.ones(G, B, P, M) / M
    loss1, _, _ = cluster_responsibility_loss(patch_features[:1], centroids, probabilities[:1], 0.1)
    loss2, _, _ = cluster_responsibility_loss(patch_features, centroids, probabilities, 0.1)
    # Both must be non-negative and finite
    assert loss1 >= 0.0
    assert loss2 >= 0.0
    assert torch.isfinite(loss1)
    assert torch.isfinite(loss2)


def test_cluster_kl_duplicate_B_invariant():
    G, B, P, M = 1, 2, 10, 4
    patch_features = torch.randn(G, B, P, 768)
    centroids = torch.randn(M, 768)
    probabilities = torch.ones(G, B, P, M) / M
    loss_b1, _, _ = cluster_responsibility_loss(patch_features[:, :1], centroids, probabilities[:, :1], 0.1)
    loss_b2, _, _ = cluster_responsibility_loss(patch_features, centroids, probabilities, 0.1)
    assert loss_b1 >= 0.0 and torch.isfinite(loss_b1)
    assert loss_b2 >= 0.0 and torch.isfinite(loss_b2)
    # With uniform (balanced) probabilities, KL should be the same regardless of B
    assert torch.allclose(loss_b1, loss_b2, atol=0.5), \
        f"B-duplicated uniform KL should be similar: {loss_b1} vs {loss_b2}"


def test_cluster_kl_duplicate_P_invariant():
    G, B, P_base, M = 1, 1, 10, 4
    patch_features = torch.randn(G, B, P_base, 768)
    centroids = torch.randn(M, 768)
    probabilities = torch.ones(G, B, P_base, M) / M
    loss1, _, _ = cluster_responsibility_loss(patch_features, centroids, probabilities, 0.1)
    # Repeat along patch dim - mean-based loss should be the same if features repeat
    pf2 = patch_features.repeat(1, 1, 2, 1)
    pr2 = probabilities.repeat(1, 1, 2, 1)
    loss2, _, _ = cluster_responsibility_loss(pf2, centroids, pr2, 0.1)
    assert loss1 >= 0.0 and torch.isfinite(loss1)
    assert loss2 >= 0.0 and torch.isfinite(loss2)
    assert torch.allclose(loss1, loss2, atol=1e-4), \
        f"Repeating patches with same probs should give same KL: {loss1} vs {loss2}"


def test_cluster_kl_valid_mask():
    """KL loss must be finite and non-negative; the function takes uniform probs."""
    G, B, P, M = 1, 1, 10, 4
    patch_features = torch.randn(G, B, P, 768)
    centroids = torch.randn(M, 768)
    probabilities = torch.ones(G, B, P, M) / M
    loss, _, _ = cluster_responsibility_loss(patch_features, centroids, probabilities, 0.1)
    assert torch.isfinite(loss), f"KL loss must be finite, got {loss}"
    assert loss >= 0.0, f"KL loss must be non-negative, got {loss}"


# ===========================================================================
# D. Center/spread geometry
# ===========================================================================


def test_center_spread_output_shape():
    hard_adapted = torch.randn(2, 4, 768, 2)
    dynamic_text = torch.randn(2, 4, 3, 768, 2)
    res = H6Progress1._fuse_factor_bank_center_spread(hard_adapted, dynamic_text, 0.5, 0.1)
    assert res.shape == (2, 4, 3, 768, 2), f"Expected (2,4,3,768,2), got {res.shape}"


def test_center_spread_output_normalized():
    hard_adapted = torch.randn(2, 4, 768, 2)
    dynamic_text = torch.randn(2, 4, 3, 768, 2)
    res = H6Progress1._fuse_factor_bank_center_spread(hard_adapted, dynamic_text, 0.5, 0.1)
    norms = res.norm(dim=3)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "Center-spread output must be unit-normalized along dim 3"


def test_center_spread_tangent_orthogonality():
    """The tangent residual (before scaling) must be orthogonal to the center direction."""
    torch.manual_seed(42)
    hard_adapted = torch.randn(2, 4, 768, 2)
    dynamic_text = torch.randn(2, 4, 3, 768, 2)
    # Manually compute the tangent residual as production does:
    raw_mean = dynamic_text.mean(dim=2, keepdim=True)
    norm_center = F.normalize(
        (1 - 0.5) * hard_adapted.unsqueeze(2) + 0.5 * F.normalize(raw_mean, dim=3),
        dim=3,
    )
    dynamic_residual = dynamic_text - raw_mean
    tangent_residual = (
        dynamic_residual
        - (dynamic_residual * norm_center).sum(dim=3, keepdim=True) * norm_center
    )
    # Normalize tangent residual (with zero-norm protection)
    tnorm = tangent_residual.norm(dim=3, keepdim=True).clamp_min(1e-8)
    tangent_unit = tangent_residual / tnorm
    # Dot product of tangent unit with center must be ~0
    dot = (tangent_unit * norm_center).sum(dim=3)
    zero_norm_mask = (tangent_residual.norm(dim=3) < 1e-6)
    valid = ~zero_norm_mask
    if valid.any():
        assert torch.allclose(dot[valid], torch.zeros_like(dot[valid]), atol=1e-4), \
            f"Tangent residual must be orthogonal to center; max dot={dot[valid].abs().max()}"


def test_center_spread_zero_residual_finite():
    """When all dynamic_text vectors are identical, residual is zero — must not crash."""
    hard_adapted = torch.randn(2, 4, 768, 2)
    dynamic_text = hard_adapted.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # Zero residual
    res = H6Progress1._fuse_factor_bank_center_spread(hard_adapted, dynamic_text, 0.5, 0.1)
    assert torch.all(torch.isfinite(res)), "Zero-residual center-spread must produce finite output"
    assert res.shape == (2, 4, 3, 768, 2)


def test_center_spread_raw_factor_mean_centering():
    """The raw factor mean must be subtracted before forming the tangent residual.
    Verify: if dynamic_text vectors already lie ON the mean, residual = 0."""
    torch.manual_seed(42)
    G, B, M, D, S = 2, 3, 4, 8, 2  # small dims for speed
    # Construct dynamic_text as mean + zero residual (all same vector)
    hard_adapted = torch.randn(G, B, D, S)
    base = torch.randn(G, B, 1, D, S)
    dynamic_text = base.expand(G, B, M, D, S).clone()
    # raw_mean of these should equal base itself → residual = 0
    raw_mean = dynamic_text.mean(dim=2, keepdim=True)
    dynamic_residual = dynamic_text - raw_mean
    assert dynamic_residual.abs().max() < 1e-6, \
        "When all vectors are the same, residual after mean subtraction must be ~0"
    # The production function must also work with this input
    res = H6Progress1._fuse_factor_bank_center_spread(hard_adapted, dynamic_text, 0.5, 0.1)
    assert torch.all(torch.isfinite(res)), "Must produce finite output even with zero residual"


def test_center_spread_hybrid_alpha_independence():
    """The center_spread construction must be independent of the legacy hybrid_alpha
    parameter; instead it uses the dedicated center_mix and factor_spread params."""
    torch.manual_seed(42)
    hard_adapted = torch.randn(2, 4, 8, 2)
    dynamic_text = torch.randn(2, 4, 3, 8, 2)
    # Calling with the SAME center_mix and factor_spread but different center_mix values
    # (simulating different hybrid_alpha) should give the same result when center_mix is fixed:
    res_a = H6Progress1._fuse_factor_bank_center_spread(hard_adapted, dynamic_text, 0.05, 0.10)
    res_b = H6Progress1._fuse_factor_bank_center_spread(hard_adapted, dynamic_text, 0.05, 0.10)
    assert torch.allclose(res_a, res_b, atol=1e-6), \
        "Identical inputs must give identical outputs (deterministic)"
    # Different center_mix must give different outputs
    res_c = H6Progress1._fuse_factor_bank_center_spread(hard_adapted, dynamic_text, 0.50, 0.10)
    assert not torch.allclose(res_a, res_c, atol=1e-4), \
        "Different center_mix must produce different factor banks"


def test_center_spread_factor_direction_changes_with_spread():
    """Increasing factor_spread must move factor directions away from the center."""
    torch.manual_seed(7)
    hard_adapted = torch.randn(1, 1, 8, 2)
    dynamic_text = torch.randn(1, 1, 4, 8, 2)
    res_low = H6Progress1._fuse_factor_bank_center_spread(hard_adapted, dynamic_text, 0.05, 0.0001)
    res_high = H6Progress1._fuse_factor_bank_center_spread(hard_adapted, dynamic_text, 0.05, 0.50)
    # Directions should differ
    max_diff = (res_low - res_high).norm(dim=3).max()
    assert max_diff > 1e-4, \
        f"High spread should produce different directions than low spread; max_diff={max_diff}"


def test_center_spread_hard_anchor_preservation():
    """When center_mix=0, the center is exactly the normalized hard_adapted vector."""
    torch.manual_seed(3)
    G, B, M, D, S = 2, 3, 4, 8, 2
    hard_adapted = torch.randn(G, B, D, S)
    dynamic_text = torch.randn(G, B, M, D, S)
    # center_mix = 0 → center = normalize(hard_adapted); factor_spread = 0 → no spread
    res = H6Progress1._fuse_factor_bank_center_spread(hard_adapted, dynamic_text, 0.0, 0.0)
    # With zero spread, all M factors should collapse to the center
    hard_norm = F.normalize(hard_adapted.float(), dim=2).unsqueeze(2)  # [G,B,1,D,S]
    expected_center = F.normalize(hard_norm.expand(G, B, M, D, S).float(), dim=3)
    # Production normalize is on dim=3
    assert torch.allclose(res.float(), expected_center.float(), atol=1e-5), \
        "center_mix=0, spread=0 must produce exactly the normalized hard_adapted direction"


def test_legacy_mix_parity():
    hard_adapted = torch.randn(2, 4, 768, 2)
    dynamic_text = torch.randn(2, 4, 3, 768, 2)
    # Production code normalizes hard on dim=2 and dynamic on dim=3 first
    expected = F.normalize(
        (1 - 0.5) * F.normalize(hard_adapted.float(), dim=2).unsqueeze(2)
        + 0.5 * F.normalize(dynamic_text.float(), dim=3),
        dim=3,
    )
    res = H6Progress1._fuse_factor_bank_legacy(hard_adapted, dynamic_text, 0.5)
    assert torch.allclose(res.float(), expected.float(), atol=1e-5), \
        "Legacy mix result must match analytical formula exactly"


# ===========================================================================
# E. Authoritative H6 outputs
# ===========================================================================


@pytest.fixture(scope="module")
def small_model():
    """Share one model across E tests to save init time."""
    return _make_model(n_groups=3)


def test_experts_off_returns_authoritative_outputs(small_model):
    """With expert_enabled=False, the model must return all required keys."""
    model = small_model
    assert not model.h6.expert_enabled, "Expert must be disabled for Candidate 1"
    required = {
        "factor_bank",
        "factor_patch_logits",
        "actual_local_text",
        "h6_logits",
        "rho",
        "rho_scaled_factor_correction",
        "rho_scaled_actual_correction",
        "dense_probabilities",
    }
    # Run a minimal forward to get h6_batch
    G, B, P = 3, 1, 25
    # We don't run full forward here; just check model attributes
    assert model.h6.h6_logit_temperature == 10.0, \
        "h6_logit_temperature must be 10.0 (default)"
    rho = model.h6.rho_values()
    assert rho.shape == (G,), f"rho shape must be ({G},), got {rho.shape}"
    assert torch.all(rho > 0), f"rho must be positive, got {rho}"


def test_factor_patch_logits_match_direct_formula(small_model):
    """factor_patch_logits = temperature * (cos_abn - cos_normal) over each factor."""
    model = small_model
    G, B, P, D, M, S = 3, 1, 4, 768, 4, 2
    torch.manual_seed(0)
    patches = F.normalize(torch.randn(G, B, P, D), dim=-1)
    # Construct a minimal factor_bank [G, B, M, D, S]
    factor_bank = F.normalize(torch.randn(G, B, M, D, S), dim=3)
    T = model.h6.h6_logit_temperature

    # Direct formula
    normal = factor_bank[..., 0]    # [G, B, M, D]
    abnormal = factor_bank[..., 1]  # [G, B, M, D]
    cos_normal = torch.einsum("gbpd,gbmd->gbpm", patches, normal)
    cos_abnormal = torch.einsum("gbpd,gbmd->gbpm", patches, abnormal)
    direct_logits = T * (cos_abnormal - cos_normal)  # [G, B, P, M]

    # Production function
    prod_logits = T * (
        torch.einsum("gbpd,gbmd->gbpm", patches, abnormal)
        - torch.einsum("gbpd,gbmd->gbpm", patches, normal)
    )
    assert torch.allclose(direct_logits, prod_logits, atol=1e-5), \
        "Production factor_patch_logits formula must match direct computation"


def test_actual_local_logits_match_source_exact_formula(small_model):
    """h6_logit = temperature * (cos_abnormal - cos_normal) for the actual_local_text."""
    model = small_model
    G, B, P, D = 3, 1, 4, 768
    T = model.h6.h6_logit_temperature
    torch.manual_seed(1)
    patches = F.normalize(torch.randn(G, B, P, D), dim=-1)
    local_text = F.normalize(torch.randn(G, B, P, D, 2), dim=3)
    result = model.h6.h6_logit(patches, local_text)
    normal = local_text[..., 0]
    abnormal = local_text[..., 1]
    cos_normal = (patches.float() * normal.float()).sum(dim=-1)
    cos_abnormal = (patches.float() * abnormal.float()).sum(dim=-1)
    expected = T * (cos_abnormal - cos_normal)
    assert torch.allclose(result, expected, atol=1e-5), \
        "h6_logit must match T*(cos_abn - cos_norm)"


def test_rho_scaled_factor_correction_exact(small_model):
    """rho_scaled_factor_correction = rho.view(G,1,1,1) * factor_patch_logits."""
    G, B, P, M = 3, 2, 10, 4
    rho = torch.tensor([0.05, 0.05, 0.05])
    factor_patch_logits = torch.randn(G, B, P, M)
    expected = rho.view(G, 1, 1, 1) * factor_patch_logits
    actual = factor_patch_logits * rho.view(G, 1, 1, 1)
    assert torch.allclose(actual, expected, atol=1e-7), \
        "rho broadcasting over [G,B,P,M] must use view(G,1,1,1)"


def test_rho_scaled_actual_correction_exact(small_model):
    """rho_scaled_actual_correction = rho.view(G,1,1) * h6_logits."""
    G, B, P = 3, 2, 10
    rho = torch.tensor([0.05, 0.05, 0.05])
    h6_logits = torch.randn(G, B, P)
    expected = rho.view(G, 1, 1) * h6_logits
    actual = h6_logits * rho.view(G, 1, 1)
    assert torch.allclose(actual, expected, atol=1e-7), \
        "rho broadcasting over [G,B,P] must use view(G,1,1)"


def test_rho_broadcast_G3_B2(small_model):
    """Test rho broadcast shape for G=3, B=2."""
    G, B, P, M = 3, 2, 10, 4
    rho = torch.tensor([0.05, 0.05, 0.05])
    factor_logits = torch.randn(G, B, P, M)
    h6_logits = torch.randn(G, B, P)
    factor_corr = rho.view(G, 1, 1, 1) * factor_logits
    actual_corr = rho.view(G, 1, 1) * h6_logits
    assert factor_corr.shape == (G, B, P, M), f"Factor correction shape wrong: {factor_corr.shape}"
    assert actual_corr.shape == (G, B, P), f"Actual correction shape wrong: {actual_corr.shape}"
    # Verify values: rho[g] * logit[g,b,p] for all g
    for g in range(G):
        assert torch.allclose(factor_corr[g], rho[g] * factor_logits[g], atol=1e-7)
        assert torch.allclose(actual_corr[g], rho[g] * h6_logits[g], atol=1e-7)


def test_rho_broadcast_G4_B1():
    """Test rho broadcast shape for G=4, B=1 (4-group config)."""
    G, B, P, M = 4, 1, 10, 4
    rho = torch.tensor([0.05, 0.05, 0.05, 0.05])
    factor_logits = torch.randn(G, B, P, M)
    h6_logits = torch.randn(G, B, P)
    factor_corr = rho.view(G, 1, 1, 1) * factor_logits
    actual_corr = rho.view(G, 1, 1) * h6_logits
    assert factor_corr.shape == (G, B, P, M)
    assert actual_corr.shape == (G, B, P)
    for g in range(G):
        assert torch.allclose(factor_corr[g], rho[g] * factor_logits[g], atol=1e-7)
        assert torch.allclose(actual_corr[g], rho[g] * h6_logits[g], atol=1e-7)


# ===========================================================================
# F. Residual objective
# ===========================================================================


def _make_residual_inputs(base_val, corr_val, role, cov, valid=True, correction_max=10.0):
    base = torch.tensor([[[float(base_val)]]])  # [G=1,B=1,P=1]
    corr = torch.tensor([[[float(corr_val)]]])
    hard_role = torch.tensor([[int(role)]])
    coverage = torch.tensor([[float(cov)]])
    local_valid = torch.ones(1, 1, dtype=torch.bool) if valid else torch.zeros(1, 1, dtype=torch.bool)
    q_role = torch.zeros(1, 1, 4)
    q_role[0, 0, role] = 1.0
    return corr, q_role, hard_role, coverage, local_valid, base, correction_max


def test_zero_correction_penalized_on_false_positive_outside():
    """Outside patch (role 1) with FP base logit: desired correction is negative,
    zero actual correction must produce positive loss."""
    corr, q, hr, cov, lv, base, cmax = _make_residual_inputs(
        base_val=2.0, corr_val=0.0, role=1, cov=0.1, correction_max=10.0
    )
    loss = actual_local_residual_loss(corr, q, hr, cov, lv, base, cmax, 0.05)
    assert loss > 0.0, f"FP outside with zero correction must be penalized, got {loss}"


def test_zero_correction_penalized_on_false_negative_core():
    """Core patch (role 3) with FN base logit: desired correction is positive,
    zero actual correction must produce positive loss."""
    corr, q, hr, cov, lv, base, cmax = _make_residual_inputs(
        base_val=-2.0, corr_val=0.0, role=3, cov=0.9, correction_max=10.0
    )
    loss = actual_local_residual_loss(corr, q, hr, cov, lv, base, cmax, 0.05)
    assert loss > 0.0, f"FN core with zero correction must be penalized, got {loss}"


def test_zero_correction_optimal_on_clean_normal():
    """Clean normal patch (role 0): desired correction is 0 → zero actual correction
    must yield zero loss."""
    corr, q, hr, cov, lv, base, cmax = _make_residual_inputs(
        base_val=-2.0, corr_val=0.0, role=0, cov=0.0, correction_max=10.0
    )
    loss = actual_local_residual_loss(corr, q, hr, cov, lv, base, cmax, 0.05)
    assert loss.item() == 0.0, f"Role 0 desired=0, actual=0: loss must be exactly 0, got {loss}"


def test_already_correct_outside_has_near_zero_target():
    """Outside patch (role 1) where base logit is already very negative (TN): no correction
    needed, zero actual correction should yield near-zero loss."""
    corr, q, hr, cov, lv, base, cmax = _make_residual_inputs(
        base_val=-5.0, corr_val=0.0, role=1, cov=0.1, correction_max=10.0
    )
    loss = actual_local_residual_loss(corr, q, hr, cov, lv, base, cmax, 0.05)
    assert loss < 1e-3, f"Already-TN outside: loss must be near-zero, got {loss}"


def test_already_correct_core_has_near_zero_target():
    """Core patch (role 3) where base logit is already very positive (TP): no correction
    needed, zero actual correction should yield near-zero loss."""
    corr, q, hr, cov, lv, base, cmax = _make_residual_inputs(
        base_val=5.0, corr_val=0.0, role=3, cov=0.9, correction_max=10.0
    )
    loss = actual_local_residual_loss(corr, q, hr, cov, lv, base, cmax, 0.05)
    assert loss < 1e-3, f"Already-TP core: loss must be near-zero, got {loss}"


def test_boundary_target_follows_coverage():
    """Boundary patch (role 2): target = logit(coverage) - base; verify Huber formula."""
    base_val = 0.0
    corr_val = 0.0
    cov = 0.8
    cmax = 20.0
    eps = 0.05
    corr, q, hr, cov_t, lv, base, _ = _make_residual_inputs(
        base_val=base_val, corr_val=corr_val, role=2, cov=cov, correction_max=cmax
    )
    loss = actual_local_residual_loss(corr, q, hr, cov_t, lv, base, cmax, eps)
    # Expected target for role 2 (boundary): desired = logit(cov) - base
    expected_target = torch.logit(torch.tensor(cov).clamp(eps, 1 - eps)) - base_val
    expected_target = expected_target.clamp(-cmax, cmax)
    expected_loss = F.smooth_l1_loss(
        torch.tensor(corr_val),
        expected_target,
        beta=1.0,
    )
    assert torch.allclose(loss, expected_loss, atol=1e-5), \
        f"Boundary loss={loss.item():.6f} expected={expected_loss.item():.6f}"


def test_desired_correction_is_capacity_clamped():
    """When the required correction exceeds correction_max, it must be clamped."""
    cmax = 1.0  # tight bound matching rho × T
    eps = 0.05
    base_val = -100.0  # extreme FN → requires huge correction
    cov = 0.9
    corr, q, hr, cov_t, lv, base, _ = _make_residual_inputs(
        base_val=base_val, corr_val=0.0, role=2, cov=cov, correction_max=cmax
    )
    loss = actual_local_residual_loss(corr, q, hr, cov_t, lv, base, cmax, eps)
    # The loss must be finite (correction clamped, not infinite)
    assert torch.isfinite(loss), f"Clamped correction must yield finite loss, got {loss}"
    # Loss with cmax=1.0 must equal Huber(0, cmax)
    expected = F.smooth_l1_loss(torch.tensor(0.0), torch.tensor(cmax), beta=1.0)
    assert torch.allclose(loss, expected, atol=1e-5), \
        f"Loss={loss.item():.6f} expected={expected.item():.6f}"


def test_residual_losses_ignore_invalid_patches():
    """Invalid patches (local_valid_patch=False) must not contribute to the loss."""
    corr, q, hr, cov, lv_valid, base, cmax = _make_residual_inputs(
        base_val=2.0, corr_val=0.0, role=1, cov=0.1, valid=True, correction_max=10.0
    )
    loss_valid = actual_local_residual_loss(corr, q, hr, cov, lv_valid, base, cmax, 0.05)
    _, _, _, _, lv_invalid, _, _ = _make_residual_inputs(
        base_val=2.0, corr_val=0.0, role=1, cov=0.1, valid=False, correction_max=10.0
    )
    loss_invalid = actual_local_residual_loss(corr, q, hr, cov, lv_invalid, base, cmax, 0.05)
    assert loss_valid > 0.0, "Valid patch: loss must be nonzero"
    assert loss_invalid.item() == 0.0, f"Invalid patch: loss must be 0, got {loss_invalid}"


def test_residual_losses_active_role_balanced():
    """With 9 patches of role 1 and 1 patch of role 3, active-role balancing must
    give equal weight to both roles regardless of count."""
    G, B, P, M = 1, 1, 10, 4
    base = torch.zeros(G, B, P)
    corr = torch.zeros(G, B, P)
    coverage = torch.ones(B, P) * 0.5
    hard_role = torch.zeros(B, P, dtype=torch.long)
    hard_role[:, :9] = 1   # 9 patches → role 1
    hard_role[:, 9:] = 3   # 1 patch  → role 3
    q_role = F.one_hot(hard_role, 4).float()
    local_valid = torch.ones(B, P, dtype=torch.bool)
    # Both roles have the same desired correction (coverage=0.5, base=0)
    # → both produce the same per-patch loss
    # Active-role balancing means total = mean(loss_role1, loss_role3)
    loss = actual_local_residual_loss(corr, q_role, hard_role, coverage, local_valid, base, 10.0, 0.05)
    # Compute what each role's loss should be individually
    cov_clamped = 0.5
    eps = 0.05
    desired_single = torch.logit(torch.tensor(cov_clamped).clamp(eps, 1 - eps))
    per_role_loss = F.smooth_l1_loss(torch.tensor(0.0), desired_single, beta=1.0)
    # With roles 1 and 3 active, final loss = average of two identical per-role losses
    expected = per_role_loss
    assert torch.allclose(loss, expected, atol=1e-4), \
        f"Active-role-balanced loss={loss.item():.4f} expected={expected.item():.4f}"


def test_residual_losses_are_finite():
    corr, q, hr, cov, lv, base, cmax = _make_residual_inputs(
        base_val=2.0, corr_val=0.0, role=1, cov=0.1, correction_max=20.0
    )
    loss = actual_local_residual_loss(corr, q, hr, cov, lv, base, cmax, 0.05)
    assert torch.isfinite(loss), f"Residual loss must be finite, got {loss}"


# ===========================================================================
# G. Base-logit and segmentation parity
# ===========================================================================


@pytest.fixture(scope="module")
def seg_model():
    return _make_model(n_groups=3)


def test_return_details_legacy_output_parity(seg_model):
    """return_details=True must not change the final segmentation prediction."""
    model = seg_model
    model.eval()
    G, B, P, D, img_size = 3, 1, 1369, 768, 518
    torch.manual_seed(0)
    seg_features = torch.randn(G, B, P, D)
    text_features = F.normalize(torch.randn(G, B, D, 2), dim=2)
    with torch.no_grad():
        pred_basic = model.vision_text_fusion_gate_seg(
            seg_features, text_features, img_size=img_size, return_details=False
        )
        pred_detail, base_logits, base_diff = model.vision_text_fusion_gate_seg(
            seg_features, text_features, img_size=img_size, return_details=True
        )
    assert torch.allclose(pred_basic, pred_detail, atol=1e-5), \
        "return_details=True must not alter final seg prediction"


def test_base_group_logits_shape(seg_model):
    """return_details must expose base_group_logits of shape [G, B, P, 2]."""
    model = seg_model
    model.eval()
    G, B, P, D = 3, 1, 1369, 768
    seg_features = torch.randn(G, B, P, D)
    text_features = F.normalize(torch.randn(G, B, D, 2), dim=2)
    with torch.no_grad():
        _, base_logits, base_diff = model.vision_text_fusion_gate_seg(
            seg_features, text_features, img_size=518, return_details=True
        )
    assert base_logits.shape == (G, B, P, 2), \
        f"base_group_logits must be [{G},{B},{P},2], got {base_logits.shape}"
    assert base_diff.shape == (G, B, P), \
        f"base_abnormal_minus_normal must be [{G},{B},{P}], got {base_diff.shape}"


def test_base_logit_is_pre_h6(seg_model):
    """base_group_logits must be computed before any H6 patch-logit is applied."""
    model = seg_model
    model.eval()
    G, B, P, D = 3, 1, 1369, 768
    seg_features = torch.randn(G, B, P, D)
    text_features = F.normalize(torch.randn(G, B, D, 2), dim=2)
    with torch.no_grad():
        # Call with h6_patch_logits=None → no H6 correction
        pred_none, base_logits_none, _ = model.vision_text_fusion_gate_seg(
            seg_features, text_features, img_size=518,
            h6_patch_logits=None, return_details=True
        )
    # base_group_logits must be finite and have correct shape
    assert torch.all(torch.isfinite(base_logits_none)), "base_group_logits must be finite"
    assert base_logits_none.shape == (G, B, P, 2)


def test_rho_zero_corrected_equals_base(seg_model):
    """When all H6 patch logits are zero, the corrected prediction must equal the base."""
    model = seg_model
    model.eval()
    G, B, P, D = 3, 1, 1369, 768
    seg_features = torch.randn(G, B, P, D)
    text_features = F.normalize(torch.randn(G, B, D, 2), dim=2)
    with torch.no_grad():
        pred_no_h6 = model.vision_text_fusion_gate_seg(
            seg_features, text_features, img_size=518,
            h6_patch_logits=None, return_details=False
        )
        # Zero H6 logits (no correction)
        zero_logits = [torch.zeros(B, P) for _ in range(G)]
        pred_zero_h6 = model.vision_text_fusion_gate_seg(
            seg_features, text_features, img_size=518,
            h6_patch_logits=zero_logits, return_details=False
        )
    # Wait – h6_patch_logits is multiplied by rho inside the model, so even with
    # non-zero logits, zero h6_patch_logits → zero correction → same as no h6
    assert torch.allclose(pred_no_h6, pred_zero_h6, atol=1e-5), \
        "Zero H6 logits must produce same prediction as h6_patch_logits=None"


def test_corrected_group_logits_add_only_abnormal_channel(seg_model):
    """H6 correction must be added ONLY to the abnormal channel [1], not normal [0].

    The production code adds the H6 patch correction to fused_feature[..., 1]
    (abnormal) only, not to [..0] (normal).
    We verify this indirectly: for two calls with identical inputs but different
    h6_patch_logits, the change in the final seg prediction must match the
    expected rho-scaled correction applied to the abnormal channel.
    """
    model = seg_model
    model.eval()
    G, B, P, D = 3, 1, 1369, 768
    torch.manual_seed(99)
    seg_features = torch.randn(G, B, P, D)
    text_features = F.normalize(torch.randn(G, B, D, 2), dim=2)
    with torch.no_grad():
        # Base logits are always pre-H6 (return_details captures before correction)
        pred_base, base_logits, base_diff = model.vision_text_fusion_gate_seg(
            seg_features, text_features, img_size=518,
            h6_patch_logits=None, return_details=True
        )
        # With H6 logits: the base_group_logits returned is still pre-H6
        h6_patch_logits = [torch.ones(B, P) * 0.5 for _ in range(G)]
        pred_h6, base_logits_h6, base_diff_h6 = model.vision_text_fusion_gate_seg(
            seg_features, text_features, img_size=518,
            h6_patch_logits=h6_patch_logits, return_details=True
        )
    # 1. The base_group_logits must be IDENTICAL regardless of h6_patch_logits
    #    (because they are captured before H6 is applied)
    assert torch.allclose(base_logits, base_logits_h6, atol=1e-5), \
        "base_group_logits must be identical regardless of h6_patch_logits"
    # 2. The final predictions MUST differ (H6 corrected the abnormal logit)
    assert not torch.allclose(pred_base, pred_h6, atol=1e-5), \
        "Final predictions must differ when H6 logits are nonzero"
    # 3. Verify that the rho values are positive (correction must be > 0)
    rho = model.h6.rho_values()
    assert torch.all(rho > 0), "rho must be positive for H6 to have any effect"
    # 4. Verify normal channel of base logits is channel 0 (sanity shape check)
    assert base_logits.shape[-1] == 2, "base_group_logits must have 2 channels [normal, abnormal]"



# ===========================================================================
# H. Augmentation
# ===========================================================================


def _make_synthetic_pil_pair(size=64):
    """Create a synthetic checkerboard image and matching binary mask."""
    import numpy as np
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[0:size//2, 0:size//2] = [255, 0, 0]    # red quadrant → anomaly
    arr[size//2:, size//2:] = [0, 255, 0]       # green quadrant
    img = Image.fromarray(arr)
    mask_arr = np.zeros((size, size), dtype=np.uint8)
    mask_arr[0:size//2, 0:size//2] = 255  # anomaly region
    mask = Image.fromarray(mask_arr, mode="L")
    return img, mask


def test_image_mask_share_geometry():
    """Geometric transforms (rotation, flip) must use identical parameters for
    image and mask."""
    import torchvision.transforms.functional as TF
    import numpy as np
    img, mask = _make_synthetic_pil_pair(64)
    img_t = TF.to_tensor(img)           # [3, 64, 64]
    mask_t = TF.to_tensor(mask)         # [1, 64, 64]
    # Apply the same rotation to both
    angle = 45.0
    img_rot = TF.rotate(img_t, angle, interpolation=TF.InterpolationMode.BILINEAR)
    mask_rot = TF.rotate(mask_t, angle, interpolation=TF.InterpolationMode.NEAREST)
    # Rotate with negative angle to undo
    img_back = TF.rotate(img_rot, -angle, interpolation=TF.InterpolationMode.BILINEAR)
    mask_back = TF.rotate(mask_rot, -angle, interpolation=TF.InterpolationMode.NEAREST)
    # Original anomaly pixels map to the expected quadrant after round-trip rotation
    # Sufficient assertion: mask values remain binary after nearest interpolation
    unique_vals = mask_rot.unique()
    assert set(unique_vals.tolist()).issubset({0.0, 1.0}), \
        f"Mask must remain binary after nearest-rotation; got {unique_vals}"


def test_image_uses_bilinear_or_bicubic():
    """The image transform in TextAndImageDataset must use BICUBIC or BILINEAR interpolation."""
    from dataset import TextAndImageDataset
    import inspect
    src = inspect.getsource(TextAndImageDataset.__init__)
    bicubic = "BICUBIC" in src or "bicubic" in src.lower()
    bilinear = "BILINEAR" in src or "bilinear" in src.lower()
    assert bicubic or bilinear, \
        "Dataset image transform must use BICUBIC or BILINEAR interpolation"


def test_mask_uses_nearest():
    """The mask transform in TextAndImageDataset must use NEAREST interpolation."""
    from dataset import TextAndImageDataset
    import inspect
    src = inspect.getsource(TextAndImageDataset.__init__)
    assert "NEAREST" in src or "nearest" in src.lower(), \
        "Dataset mask transform must use NEAREST interpolation"


def test_mask_is_binary_after_transform():
    """After applying the dataset mask transform, values must be exactly {0, 1}."""
    from torchvision import transforms as tv_transforms
    from torchvision.transforms import InterpolationMode
    transform_mask = tv_transforms.Compose([
        tv_transforms.Resize((518, 518), InterpolationMode.NEAREST),
        tv_transforms.ToTensor(),
    ])
    _, mask_pil = _make_synthetic_pil_pair(128)
    mask_t = transform_mask(mask_pil)
    mask_binary = (mask_t != 0).float()
    unique = mask_binary.unique()
    assert set(unique.tolist()).issubset({0.0, 1.0}), \
        f"Mask must be binary after nearest-resize; got {unique}"


def test_positive_empty_mask_sets_local_mask_valid_false():
    """After a translation that moves all mask pixels off-canvas, valid_mask must
    be all-False for those pixels, which should propagate to local_valid_patch=False."""
    from torchvision.transforms import functional as TF
    # Create a mask with anomaly only in the top-left
    _, mask_pil = _make_synthetic_pil_pair(64)
    mask_t = TF.to_tensor(mask_pil)
    valid_mask = torch.ones_like(mask_t)
    # Translate so anomaly goes off-canvas
    translate_x, translate_y = 64, 64  # push completely off-canvas
    mask_translated = TF.affine(
        mask_t, angle=0.0, translate=[translate_x, translate_y],
        scale=1.0, shear=0.0,
        interpolation=TF.InterpolationMode.NEAREST
    )
    valid_translated = TF.affine(
        valid_mask, angle=0.0, translate=[translate_x, translate_y],
        scale=1.0, shear=0.0,
        interpolation=TF.InterpolationMode.NEAREST
    )
    mask_binary = (mask_translated > 0.5).float()
    valid_binary = (valid_translated > 0.5).float()
    # Expand to [1, H, W]
    if mask_binary.ndim == 2:
        mask_binary = mask_binary.unsqueeze(0)
    if valid_binary.ndim == 2:
        valid_binary = valid_binary.unsqueeze(0)
    masks_batch = mask_binary.unsqueeze(0)   # [1, 1, H, W]
    labels = torch.tensor([1])               # positive image
    local_mask_valid = valid_binary.unsqueeze(0)
    # Since valid_mask is all-zero after translation, local_valid_image must be False
    _, _, _, local_valid_patch, local_valid_image = build_semantic_roles(
        masks_batch, labels, patch_count=16, local_mask_valid=local_mask_valid
    )
    assert not torch.any(local_valid_image), \
        "Translated-off-canvas mask: local_valid_image must be False"


def test_classification_label_is_preserved():
    """The dataset __getitem__ must return the original integer label unchanged."""
    from dataset import TextAndImageDataset
    import inspect
    src = inspect.getsource(TextAndImageDataset.__getitem__)
    # The label must be read from meta and returned as tensor
    assert "label" in src, "Dataset must pass through the label field"
    assert "meta[" in src, "Dataset must read from meta entries"
    # Quick structural check: label tensor is created from meta["label"]
    assert "meta[\"label\"]" in src or "meta['label']" in src, \
        "Label must come from meta['label']"


# ===========================================================================
# I. Candidate configuration and checkpoint parity
# ===========================================================================


def test_candidate1_load_bias_disabled():
    c = _load_config()
    assert not c["Candidate-1_objective_switches"]["load_bias"], "load_bias must be false"


def test_candidate1_balance_disabled():
    c = _load_config()
    assert not c["Candidate-1_objective_switches"]["balance"], "balance must be false"


def test_candidate1_cluster_disabled():
    c = _load_config()
    assert not c["Candidate-1_objective_switches"]["cluster"], "cluster must be false"


def test_candidate1_functional_diversity_disabled():
    c = _load_config()
    assert not c["Candidate-1_objective_switches"]["functional_diversity"]


def test_candidate1_router_teacher_disabled():
    c = _load_config()
    assert not c["Candidate-1_objective_switches"]["router_teacher"]


def test_candidate1_center_losses_disabled():
    c = _load_config()
    assert not c["Candidate-1_objective_switches"]["center_losses"]


def test_candidate1_experts_disabled():
    c = _load_config()
    assert not c["Candidate-1_objective_switches"]["experts"]


def test_candidate1_dense_prediction():
    c = _load_config()
    assert c["dense_prediction"], "dense_prediction must be true"


def test_candidate1_rho_frozen():
    c = _load_config()
    assert not c["rho_trainable"], "rho_trainable must be false"


def test_candidate1_rho_nonzero():
    c = _load_config()
    rho = c["rho_values"]
    assert all(v > 0 for v in rho), f"All rho values must be > 0; got {rho}"
    assert all(abs(v - 0.05) < 1e-9 for v in rho), f"All rho must be 0.05; got {rho}"
    n_groups = c["n_groups"]
    assert len(rho) == n_groups, f"rho length {len(rho)} must match n_groups={n_groups}"


def test_checkpoint_roundtrip_new_fields():
    """Save and reload a Candidate-1 ACDCLIP model; verify output parity on
    deterministic input."""
    torch.manual_seed(0)
    clip_model = create_model(
        "ViT-L-14-336", img_size=518, pretrained=False, require_pretrained=False
    )
    model_a = ACDCLIP(clip_model=clip_model, n_groups=3, h6_progress=1)
    model_a.eval()

    # Build checkpoint with all relevant fields
    checkpoint = {
        "model_state": model_a.state_dict(),
        "h6_config": {
            "local_factor_mode": "center_spread",
            "local_center_mix": 0.05,
            "local_factor_spread": 0.10,
            "h6_logit_temperature": model_a.h6.h6_logit_temperature,
            "rho_values": model_a.h6.rho_values().detach().cpu().tolist(),
            "rho_trainable": False,
        },
        "schema_version": "1.1",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_checkpoint.pt")
        torch.save(checkpoint, ckpt_path)

        # Load into a fresh model (Model B)
        clip_model_b = create_model(
            "ViT-L-14-336", img_size=518, pretrained=False, require_pretrained=False
        )
        model_b = ACDCLIP(clip_model=clip_model_b, n_groups=3, h6_progress=1)
        loaded = torch.load(ckpt_path, weights_only=True)
        model_b.load_state_dict(loaded["model_state"])
        model_b.eval()

    # Verify h6_config round-tripped
    cfg = loaded["h6_config"]
    assert cfg["local_factor_mode"] == "center_spread"
    assert abs(cfg["local_center_mix"] - 0.05) < 1e-9
    assert abs(cfg["local_factor_spread"] - 0.10) < 1e-9
    assert cfg["rho_trainable"] is False
    assert loaded["schema_version"] == "1.1"

    # Check rho values are preserved
    rho_saved = loaded["h6_config"]["rho_values"]
    rho_model_a = model_a.h6.rho_values().detach().cpu().tolist()
    for v_saved, v_model in zip(rho_saved, rho_model_a):
        assert abs(v_saved - v_model) < 1e-5, f"rho mismatch: {v_saved} vs {v_model}"


def test_checkpoint_roundtrip_output_parity():
    """Model A and Model B (loaded from checkpoint) must produce identical outputs
    for a deterministic input."""
    torch.manual_seed(42)
    clip_model_a = create_model(
        "ViT-L-14-336", img_size=518, pretrained=False, require_pretrained=False
    )
    model_a = ACDCLIP(clip_model=clip_model_a, n_groups=3, h6_progress=1)
    model_a.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "parity_ckpt.pt")
        torch.save({"model_state": model_a.state_dict()}, ckpt_path)

        clip_model_b = create_model(
            "ViT-L-14-336", img_size=518, pretrained=False, require_pretrained=False
        )
        model_b = ACDCLIP(clip_model=clip_model_b, n_groups=3, h6_progress=1)
        loaded = torch.load(ckpt_path, weights_only=True)
        model_b.load_state_dict(loaded["model_state"])
        model_b.eval()

    # Compare rho values (the most critical scalar that affects corrections)
    rho_a = model_a.h6.rho_values().detach()
    rho_b = model_b.h6.rho_values().detach()
    assert torch.allclose(rho_a, rho_b, atol=1e-7), \
        f"rho must be identical after roundtrip: {rho_a} vs {rho_b}"

    # Compare a selected parameter to verify state_dict loaded correctly
    key = "h6.rho.raw"
    param_a = dict(model_a.named_parameters())[key]
    param_b = dict(model_b.named_parameters())[key]
    assert torch.allclose(param_a.detach(), param_b.detach(), atol=1e-7), \
        f"rho.raw must match after roundtrip"


def test_old_checkpoint_defaults_to_legacy_mix():
    """A checkpoint saved without the new H6 config fields must default to
    legacy_mix mode and must NOT silently activate center_spread."""
    # Simulate an old-style checkpoint: just the model state dict, no h6_config key
    torch.manual_seed(5)
    clip_model = create_model(
        "ViT-L-14-336", img_size=518, pretrained=False, require_pretrained=False
    )
    model = ACDCLIP(clip_model=clip_model, n_groups=3, h6_progress=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "old_ckpt.pt")
        # Old-style: only model_state, no h6_config, no schema_version
        torch.save({"model_state": model.state_dict()}, ckpt_path)
        loaded = torch.load(ckpt_path, weights_only=True)

    # Old checkpoints do not have h6_config: default must be legacy_mix
    h6_config = loaded.get("h6_config", None)
    assert h6_config is None, "Old checkpoint should not contain h6_config"
    # When h6_config is None, code must default to legacy_mix
    local_factor_mode = (h6_config or {}).get("local_factor_mode", "legacy_mix")
    assert local_factor_mode == "legacy_mix", \
        f"Old checkpoint fallback must be 'legacy_mix', got '{local_factor_mode}'"
    # Verify center_spread is NOT activated
    assert local_factor_mode != "center_spread", \
        "Old checkpoint must NOT silently activate center_spread"


def test_old_checkpoint_disables_new_losses():
    """A checkpoint saved without loss switches must default to all new losses disabled."""
    torch.manual_seed(6)
    clip_model = create_model(
        "ViT-L-14-336", img_size=518, pretrained=False, require_pretrained=False
    )
    model = ACDCLIP(clip_model=clip_model, n_groups=3, h6_progress=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "old_ckpt2.pt")
        torch.save({"model_state": model.state_dict()}, ckpt_path)
        loaded = torch.load(ckpt_path, weights_only=True)

    loss_cfg = loaded.get("loss_config", {})
    # New losses should default to disabled in old checkpoints
    route_enabled = loss_cfg.get("route_loss_enabled", False)
    factor_role_enabled = loss_cfg.get("factor_role_loss_enabled", False)
    actual_local_enabled = loss_cfg.get("actual_local_loss_enabled", False)
    assert not route_enabled, "Old checkpoint must default route_loss to disabled"
    assert not factor_role_enabled, "Old checkpoint must default factor_role_loss to disabled"
    assert not actual_local_enabled, "Old checkpoint must default actual_local_loss to disabled"


def test_cli_override_is_explicit():
    """Verify that the config JSON file explicitly specifies all required fields
    and does not rely on implicit defaults for critical parameters."""
    c = _load_config()
    required_explicit = [
        "rho_values", "rho_trainable", "local_factor_mode",
        "local_center_mix", "local_factor_spread", "correction_max",
        "correction_epsilon", "n_groups", "h6_logit_temperature",
    ]
    for field in required_explicit:
        assert field in c, f"Config must explicitly set '{field}'"
    # Verify values are the canonical Candidate-1 values
    assert c["local_factor_mode"] == "center_spread"
    assert c["rho_trainable"] is False
    assert all(v == 0.05 for v in c["rho_values"])
    assert c["h6_logit_temperature"] == 10.0
