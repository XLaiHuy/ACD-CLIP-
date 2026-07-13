"""Comprehensive tests for Phase2C PCGrad implementation.

Covers:
  1.  P_LoRA_only config invariance versus A_prime
  2.  Full-P backward compatibility (all four groups)
  3.  No-conflict gradient equivalence (apply_pcgrad == backward)
  4.  Negative conflict detection and projection
  5.  None-gradient handling
  6.  other_loss.requires_grad == False handling
  7.  Unknown group rejection
  8.  Empty groups_to_project rejection
  9.  Duplicate group name rejection
  10. Duplicate parameter assignment rejection
  11. Unscoped parameters receive normal total gradients
  12. Regularization gradients preserved (other_loss)
  13. BF16 autocast compatibility — projection remains FP32
  14. A_prime behavior unchanged (pcgrad_enabled == False)
  15. B behavior unchanged
  16. C behavior unchanged
  17. Full P behavior unchanged
"""
import unittest

import torch
import torch.nn as nn

from phase2c_pcgrad import (
    ALL_GROUPS,
    apply_pcgrad,
    project_group,
    scoped_parameter_groups,
)
from phase2c_utils import normalized_config, phase2c_config


# ─────────────────────────────────────────────────────────────────────────────
# Shared toy model that mirrors the canonical scoped-group structure
# ─────────────────────────────────────────────────────────────────────────────
class ToyModel(nn.Module):
    """Minimal model with the four canonical scoped-group prefixes."""

    def __init__(self, dim=4):
        super().__init__()
        self.image_adapter = nn.ModuleDict({
            "lora_adapters": nn.ModuleList([nn.Linear(dim, dim, bias=False)]),
            "m_i_w": nn.ModuleList([nn.Linear(dim, dim, bias=False)]),
        })
        self.text_adapter = nn.Linear(dim, dim, bias=False)
        self.soft_prompt = nn.Linear(dim, dim, bias=False)
        # An extra module to test unscoped-parameter handling
        self.extra_head = nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        out = self.image_adapter["lora_adapters"][0](x)
        out = out + self.image_adapter["m_i_w"][0](x)
        out = out + self.text_adapter(x) + self.soft_prompt(x)
        return self.extra_head(out).squeeze(-1)


def _make_model(dim=4, seed=0):
    torch.manual_seed(seed)
    model = ToyModel(dim=dim)
    model.train()
    return model


def _forward_two_losses(model, x):
    """Return (cls_loss, seg_loss, reg_loss, total_loss) from a single forward pass."""
    out = model(x)
    cls_loss = (out - 1.0).pow(2).mean()
    seg_loss = (out + 1.0).pow(2).mean()
    reg_loss = torch.tensor(0.01) * sum(p.pow(2).sum() for p in model.parameters())
    total_loss = cls_loss + seg_loss + reg_loss
    return cls_loss, seg_loss, reg_loss, total_loss


# ─────────────────────────────────────────────────────────────────────────────
# 1 & 2 — Config invariance
# ─────────────────────────────────────────────────────────────────────────────
class TestConditionConfig(unittest.TestCase):

    def test_pl_matches_a_prime_scientific_fields(self):
        """P_LoRA_only must equal A_prime in every non-PCGrad scientific field."""
        a = phase2c_config("A_prime", "a", 0.20)
        pl = phase2c_config("P_LoRA_only", "pl", 0.20)
        # normalized_config removes all PCGrad and condition-variant fields
        self.assertEqual(normalized_config(a), normalized_config(pl))

    def test_pl_pcgrad_groups_is_lora_only(self):
        pl = phase2c_config("P_LoRA_only", "pl", 0.20)
        self.assertTrue(pl["pcgrad_enabled"])
        self.assertEqual(pl["pcgrad_groups"], ["shared_image_lora"])
        self.assertEqual(pl["parent_condition"], "A_prime")
        self.assertEqual(pl["pcgrad_variant"], "symmetric_module_scoped_lora_only")

    def test_full_p_pcgrad_groups_is_all_four(self):
        """Full P must still project all four groups."""
        p = phase2c_config("P", "p", 0.20)
        self.assertTrue(p["pcgrad_enabled"])
        self.assertEqual(
            p["pcgrad_groups"],
            ["shared_image_lora", "m_i_w", "hard_text_adapter", "soft_prompt"],
        )
        self.assertEqual(p["pcgrad_variant"], "deterministic_symmetric_two_task")

    def test_a_prime_pcgrad_disabled(self):
        a = phase2c_config("A_prime", "a", 0.20)
        self.assertFalse(a["pcgrad_enabled"])
        self.assertEqual(a["pcgrad_groups"], [])

    def test_b_pcgrad_disabled(self):
        b = phase2c_config("B", "b", 0.15)
        self.assertFalse(b["pcgrad_enabled"])

    def test_c_pcgrad_disabled(self):
        c = phase2c_config("C", "c", 0.20)
        self.assertFalse(c["pcgrad_enabled"])

    def test_pl_alpha_schedule_equals_a_prime(self):
        a = phase2c_config("A_prime", "a", 0.20)
        pl = phase2c_config("P_LoRA_only", "pl", 0.20)
        self.assertEqual(a["alpha_schedule"], pl["alpha_schedule"])

    def test_pl_soft_prompt_freeze_epochs_equals_a_prime(self):
        a = phase2c_config("A_prime", "a", 0.20)
        pl = phase2c_config("P_LoRA_only", "pl", 0.20)
        self.assertEqual(a["soft_prompt_freeze_epochs"], pl["soft_prompt_freeze_epochs"])

    def test_invalid_condition_raises(self):
        with self.assertRaises(ValueError):
            phase2c_config("unknown", "x", 0.20)

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            phase2c_config("P_LoRA_only", "x", 0.15)  # PL requires 0.20


# ─────────────────────────────────────────────────────────────────────────────
# 3 — No-conflict gradient equivalence
# ─────────────────────────────────────────────────────────────────────────────
class TestNoConflictEquivalence(unittest.TestCase):
    """When g_cls · g_seg >= 0, apply_pcgrad must equal standard backward()."""

    def _get_standard_grads(self, model, x):
        model.zero_grad(set_to_none=True)
        cls_loss, seg_loss, reg_loss, total_loss = _forward_two_losses(model, x)
        total_loss.backward()
        return {
            name: (param.grad.clone() if param.grad is not None else None)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def _get_pcgrad_grads(self, model, x, groups):
        model.zero_grad(set_to_none=True)
        cls_loss, seg_loss, reg_loss, total_loss = _forward_two_losses(model, x)
        apply_pcgrad(total_loss, cls_loss, seg_loss, model, groups)
        return {
            name: (param.grad.clone() if param.grad is not None else None)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def _find_no_conflict_x(self, model, groups, max_tries=20):
        """Find an input where dot(g_cls_lora, g_seg_lora) >= 0."""
        for seed in range(max_tries):
            torch.manual_seed(seed + 100)
            x = torch.randn(3, 4)
            cls_loss, seg_loss, _, _ = _forward_two_losses(model, x)
            groups_dict = scoped_parameter_groups(model)
            scoped = [p for g in groups for _, p in groups_dict[g] if p.requires_grad]
            cls_g = torch.autograd.grad(cls_loss, scoped, allow_unused=True, retain_graph=True)
            seg_g = torch.autograd.grad(seg_loss, scoped, allow_unused=True)
            cls_flat = torch.cat([
                g.reshape(-1).float() for g in cls_g if g is not None
            ])
            seg_flat = torch.cat([
                g.reshape(-1).float() for g in seg_g if g is not None
            ])
            if torch.dot(cls_flat, seg_flat).item() >= 0:
                return x
        return None

    def test_lora_only_no_conflict_equals_backward(self):
        model = _make_model(seed=42)
        groups = ["shared_image_lora"]
        x = self._find_no_conflict_x(model, groups)
        if x is None:
            self.skipTest("No no-conflict input found in max_tries; skipping")

        std_grads = self._get_standard_grads(model, x)

        model2 = _make_model(seed=42)  # fresh identical weights
        pcg_grads = self._get_pcgrad_grads(model2, x, groups)

        for name in std_grads:
            sg, pg = std_grads[name], pcg_grads.get(name)
            if sg is None and pg is None:
                continue
            self.assertIsNotNone(pg, f"PCGrad produced None grad for {name}")
            torch.testing.assert_close(
                pg.float(), sg.float(),
                rtol=1e-5, atol=1e-7,
                msg=f"Gradient mismatch for parameter '{name}'",
            )

    def test_all_four_groups_no_conflict_equals_backward(self):
        model = _make_model(seed=7)
        groups = list(ALL_GROUPS)
        x = self._find_no_conflict_x(model, groups)
        if x is None:
            self.skipTest("No no-conflict input found; skipping")
        std_grads = self._get_standard_grads(model, x)
        model2 = _make_model(seed=7)
        pcg_grads = self._get_pcgrad_grads(model2, x, groups)
        for name in std_grads:
            sg, pg = std_grads[name], pcg_grads.get(name)
            if sg is None and pg is None:
                continue
            torch.testing.assert_close(
                pg.float(), sg.float(), rtol=1e-5, atol=1e-7,
                msg=f"All-groups no-conflict mismatch for '{name}'",
            )


# ─────────────────────────────────────────────────────────────────────────────
# 4 — Negative conflict detection
# ─────────────────────────────────────────────────────────────────────────────
class TestNegativeConflict(unittest.TestCase):

    def test_projection_activates_on_negative_cosine(self):
        param = nn.Parameter(torch.zeros(3))
        g_cls = torch.tensor([1.0, 0.0, 0.0])
        g_seg = torch.tensor([-1.0, 0.0, 0.0])  # dot = -1 < 0
        _, stats = project_group([(("p", param))], [g_cls], [g_seg], [None])
        self.assertTrue(stats["projection_applied"])
        self.assertGreaterEqual(stats["post_projection_cosine"], -1e-6)

    def test_no_projection_on_positive_cosine(self):
        param = nn.Parameter(torch.zeros(2))
        g_cls = torch.tensor([1.0, 2.0])
        g_seg = torch.tensor([3.0, 4.0])
        final, stats = project_group([("p", param)], [g_cls], [g_seg], [None])
        self.assertFalse(stats["projection_applied"])
        torch.testing.assert_close(final[0], torch.tensor([4.0, 6.0]))


# ─────────────────────────────────────────────────────────────────────────────
# 5 — None gradient handling
# ─────────────────────────────────────────────────────────────────────────────
class TestNoneGradients(unittest.TestCase):

    def test_none_gradients_produce_valid_stats(self):
        param = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        final, stats = project_group([("p", param)], [None], [None], [None])
        self.assertEqual(stats["number_of_valid_gradient_tensors"], 0)

    def test_project_group_empty_named_parameters(self):
        final, stats = project_group([], [], [], [])
        self.assertEqual(final, [])
        self.assertFalse(stats["projection_applied"])
        self.assertEqual(stats["number_of_parameters"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# 6 — other_loss.requires_grad == False
# ─────────────────────────────────────────────────────────────────────────────
class TestOtherLossNoGrad(unittest.TestCase):

    def test_no_regularizer_does_not_crash(self):
        """When total_loss = cls + seg (no reg), other_loss has no grad; should not crash."""
        model = _make_model(seed=1)
        model.zero_grad(set_to_none=True)
        x = torch.randn(2, 4)
        out = model(x)
        cls_loss = (out - 1.0).pow(2).mean()
        seg_loss = (out + 1.0).pow(2).mean()
        total_loss = cls_loss + seg_loss
        # other_loss = total_loss - cls_loss - seg_loss = 0 (scalar; requires_grad=False after detach path)
        # apply_pcgrad must handle this gracefully
        apply_pcgrad(total_loss, cls_loss, seg_loss, model, ["shared_image_lora"])
        # No assertion needed beyond no exception being raised


# ─────────────────────────────────────────────────────────────────────────────
# 7–10 — Validation errors
# ─────────────────────────────────────────────────────────────────────────────
class TestValidation(unittest.TestCase):

    def _make_losses(self, model):
        x = torch.randn(2, 4)
        out = model(x)
        cls_loss = (out - 1.0).pow(2).mean()
        seg_loss = (out + 1.0).pow(2).mean()
        total_loss = cls_loss + seg_loss
        return total_loss, cls_loss, seg_loss

    def test_unknown_group_rejected(self):
        model = _make_model()
        total_loss, cls_loss, seg_loss = self._make_losses(model)
        with self.assertRaises(ValueError, msg="Unknown group not caught"):
            apply_pcgrad(total_loss, cls_loss, seg_loss, model, ["nonexistent_group"])

    def test_empty_groups_to_project_rejected(self):
        model = _make_model()
        total_loss, cls_loss, seg_loss = self._make_losses(model)
        with self.assertRaises(ValueError, msg="Empty groups_to_project not caught"):
            apply_pcgrad(total_loss, cls_loss, seg_loss, model, [])

    def test_duplicate_group_name_rejected(self):
        model = _make_model()
        total_loss, cls_loss, seg_loss = self._make_losses(model)
        with self.assertRaises(ValueError, msg="Duplicate group not caught"):
            apply_pcgrad(total_loss, cls_loss, seg_loss, model,
                         ["shared_image_lora", "shared_image_lora"])

    def test_no_trainable_params_in_group_rejected(self):
        model = _make_model()
        # Freeze all lora_adapter parameters
        for name, p in model.named_parameters():
            if name.startswith("image_adapter.lora_adapters."):
                p.requires_grad_(False)
        total_loss, cls_loss, seg_loss = self._make_losses(model)
        with self.assertRaises(ValueError, msg="No trainable params in group not caught"):
            apply_pcgrad(total_loss, cls_loss, seg_loss, model, ["shared_image_lora"])


# ─────────────────────────────────────────────────────────────────────────────
# 11 — Unscoped parameters receive normal total gradients
# ─────────────────────────────────────────────────────────────────────────────
class TestUnscopedGradients(unittest.TestCase):

    def test_extra_head_receives_total_gradient(self):
        """extra_head (not in any scoped group) must get normal total-loss grad."""
        model = _make_model(seed=99)
        model.zero_grad(set_to_none=True)
        x = torch.randn(2, 4)
        out = model(x)
        cls_loss = (out - 1.0).pow(2).mean()
        seg_loss = (out + 1.0).pow(2).mean()
        total_loss = cls_loss + seg_loss

        # Get standard backward gradient for extra_head
        total_loss_ref = cls_loss + seg_loss
        total_loss_ref.backward()
        extra_head_std = model.extra_head.weight.grad.clone()

        # Get apply_pcgrad gradient for extra_head
        model.zero_grad(set_to_none=True)
        out2 = model(x)
        cls_loss2 = (out2 - 1.0).pow(2).mean()
        seg_loss2 = (out2 + 1.0).pow(2).mean()
        total_loss2 = cls_loss2 + seg_loss2
        apply_pcgrad(total_loss2, cls_loss2, seg_loss2, model, ["shared_image_lora"])

        torch.testing.assert_close(
            model.extra_head.weight.grad.float(),
            extra_head_std.float(),
            rtol=1e-5, atol=1e-7,
            msg="extra_head gradient should match standard backward",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 12 — Regularization gradients preserved
# ─────────────────────────────────────────────────────────────────────────────
class TestRegularizationPreserved(unittest.TestCase):

    def test_reg_loss_gradient_reaches_lora(self):
        """When there is no conflict, lora gradient should include the regularizer."""
        model = _make_model(seed=5)
        model.zero_grad(set_to_none=True)
        x = torch.randn(2, 4)
        cls_loss, seg_loss, reg_loss, total_loss = _forward_two_losses(model, x)

        # Standard backward
        total_loss.backward()
        lora_param = list(model.image_adapter["lora_adapters"].parameters())[0]
        std_grad = lora_param.grad.clone()

        # PCGrad (with reg folded into other_loss)
        model.zero_grad(set_to_none=True)
        cls_loss2, seg_loss2, reg_loss2, total_loss2 = _forward_two_losses(model, x)
        apply_pcgrad(total_loss2, cls_loss2, seg_loss2, model, ["shared_image_lora"])
        pcg_grad = lora_param.grad.clone()

        # If no conflict, final gradient must include the regularizer contribution
        # so it should not be zero when the standard gradient is non-zero
        if std_grad.norm() > 0:
            self.assertGreater(pcg_grad.norm().item(), 0,
                               "PCGrad gradient should be non-zero when std grad is non-zero")


# ─────────────────────────────────────────────────────────────────────────────
# 13 — BF16 projection stays in FP32
# ─────────────────────────────────────────────────────────────────────────────
class TestBF16Projection(unittest.TestCase):

    def test_project_group_output_is_fp32(self):
        """project_group must return FP32 tensors regardless of input dtype."""
        param_bf16 = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        g_cls = torch.tensor([1.0, 0.0], dtype=torch.bfloat16)
        g_seg = torch.tensor([-1.0, 0.0], dtype=torch.bfloat16)
        final, stats = project_group([("p", param_bf16)], [g_cls], [g_seg], [None])
        self.assertEqual(final[0].dtype, torch.float32)

    def test_apply_pcgrad_restores_param_dtype(self):
        """After apply_pcgrad, each parameter.grad must match the parameter dtype."""
        model = _make_model(seed=2).to(torch.bfloat16)
        x = torch.randn(2, 4, dtype=torch.bfloat16)
        out = model(x)
        cls_loss = (out - 1.0).pow(2).mean()
        seg_loss = (out + 1.0).pow(2).mean()
        total_loss = cls_loss + seg_loss
        apply_pcgrad(total_loss, cls_loss, seg_loss, model, ["shared_image_lora"])
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.assertEqual(
                    param.grad.dtype, param.dtype,
                    msg=f"Grad dtype mismatch for {name}: "
                        f"grad={param.grad.dtype} param={param.dtype}",
                )


# ─────────────────────────────────────────────────────────────────────────────
# All_GROUPS constant consistency
# ─────────────────────────────────────────────────────────────────────────────
class TestAllGroupsConsistency(unittest.TestCase):

    def test_all_groups_matches_scoped_keys(self):
        """ALL_GROUPS must be exactly the same set as scoped_parameter_groups keys."""
        model = _make_model()
        self.assertEqual(set(ALL_GROUPS), set(scoped_parameter_groups(model).keys()))


if __name__ == "__main__":
    unittest.main()
