import math
from pathlib import Path

import pytest
import torch
from torch import nn

from h2_clean.contract import (
    ANCHOR_FAMILY_NAMES,
    SafeImageAdapterAnchor,
    anchor_parameter_family,
    apply_family_safe_anchor_budget,
    collect_family_gradient_metrics,
    partition_image_adapter_parameters,
)


class DirectionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.direction_logits = nn.Parameter(torch.ones(2))


class FamilyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_adapters = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.m_i_w = nn.ParameterList([nn.Parameter(torch.ones(2))])
        self.seg_proj = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.det_proj = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.seg_layer_norms = nn.ModuleList([nn.LayerNorm(2)])
        self.det_layer_norms = nn.ModuleList([nn.LayerNorm(2)])
        self.vision_text_q = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.vision_text_k = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.dfg_ss2d_branches = nn.ModuleList([DirectionBranch()])
        self.dfg_raw_gamma = nn.ParameterList([nn.Parameter(torch.zeros(()))])
        self.vision_text_gate = nn.ModuleList([nn.Linear(2, 2, bias=False)])


class GlobalModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.large = nn.Parameter(torch.ones(4))
        self.zero_reference = nn.Parameter(torch.zeros(4))


def _named(module):
    return sorted(module.named_parameters(), key=lambda item: item[0])


def _maps(module):
    named = _named(module)
    task = {name: torch.zeros_like(parameter) for name, parameter in named}
    raw = {name: torch.zeros_like(parameter) for name, parameter in named}
    return named, task, raw


def test_family_partition_is_complete_disjoint_and_expected():
    module = FamilyModule()
    families = partition_image_adapter_parameters(module)
    flattened = [name for entries in families.values() for name, _ in entries]
    all_names = [name for name, _ in _named(module)]
    assert tuple(families) == ANCHOR_FAMILY_NAMES
    assert sorted(flattened) == all_names
    assert len(flattened) == len(set(flattened))
    assert all(families[family] for family in ANCHOR_FAMILY_NAMES)
    assert anchor_parameter_family("dfg_ss2d_branches.0.direction_logits") == "direction_logits"
    assert anchor_parameter_family("vision_text_gate.0.weight") == "remaining_image_adapter_params"


def test_global_anchor_formula_has_one_denominator_and_safe_zero_reference():
    module = GlobalModule()
    anchor = SafeImageAdapterAnchor.from_module(module)
    with torch.no_grad():
        module.large.add_(0.5)
        module.zero_reference.fill_(0.25)
    expected = (4 * 0.5**2 + 4 * 0.25**2) / (4.0 + anchor.eps)
    assert math.isclose(anchor.loss(module).item(), expected, rel_tol=1e-6, abs_tol=1e-8)
    anchor.loss(module).backward()
    assert torch.isfinite(module.large.grad).all()
    assert torch.isfinite(module.zero_reference.grad).all()


def test_theta_equal_reference_has_zero_anchor_gradient():
    module = FamilyModule()
    anchor = SafeImageAdapterAnchor.from_module(module)
    loss = anchor.loss(module)
    assert loss.requires_grad
    loss.backward()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)) for parameter in module.parameters())


def test_active_family_anchor_is_capped_before_optimizer_clipping():
    module = FamilyModule()
    named, task, raw = _maps(module)
    target_name, target = next(iter(partition_image_adapter_parameters(module)["lora_adapters"]))
    task[target_name].view(-1)[0] = 1.0
    raw[target_name].view(-1)[0] = 100.0
    metrics = apply_family_safe_anchor_budget(
        module,
        named,
        task_gradients=task,
        raw_anchor_gradients=raw,
        anchor_lambda=1.0e-3,
        rho=0.10,
    )
    row = metrics["families"]["lora_adapters"]
    assert row["task_grad_norm"] == pytest.approx(1.0)
    assert row["raw_gradient_ratio"] == pytest.approx(100.0)
    assert row["lambda_times_raw_over_task"] == pytest.approx(0.10)
    assert row["effective_ratio"] <= 0.10 + 1.0e-10
    assert metrics["max_effective_active_family_ratio"] <= 0.10 + 1.0e-10
    assert target.grad is not None
    assert torch.isfinite(target.grad).all()
    assert target.grad.view(-1)[0].item() == pytest.approx(1.1, rel=1e-6)


def test_near_zero_task_floor_removes_anchor_contribution():
    module = FamilyModule()
    named, task, raw = _maps(module)
    near_name, near_parameter = next(iter(partition_image_adapter_parameters(module)["lora_adapters"]))
    active_name, _ = next(iter(partition_image_adapter_parameters(module)["seg_proj"]))
    task[near_name].fill_(1.0e-14)
    raw[near_name].fill_(100.0)
    task[active_name].view(-1)[0] = 1.0
    metrics = apply_family_safe_anchor_budget(
        module,
        named,
        task_gradients=task,
        raw_anchor_gradients=raw,
        anchor_lambda=1.0,
        rho=0.10,
    )
    row = metrics["families"]["lora_adapters"]
    assert row["status"] == "TASK_NEAR_ZERO"
    assert row["effective_ratio"] == 0.0
    assert row["effective_anchor_grad_norm"] == 0.0
    assert near_parameter.grad is not None
    assert torch.allclose(near_parameter.grad, task[near_name])
    assert metrics["task_floor"] >= 1.0e-12


def test_lambda_zero_and_rho_zero_preserve_task_gradient_exactly():
    for anchor_lambda, rho in ((0.0, 0.10), (0.5, 0.0)):
        module = FamilyModule()
        named, task, raw = _maps(module)
        for index, (name, parameter) in enumerate(named):
            task[name].fill_(1.0 + index)
            raw[name].fill_(2.0 + index)
        metrics = apply_family_safe_anchor_budget(
            module,
            named,
            task_gradients=task,
            raw_anchor_gradients=raw,
            anchor_lambda=anchor_lambda,
            rho=rho,
        )
        assert metrics["family_partition_complete"]
        for name, parameter in named:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, task[name])


def test_audit_collection_does_not_change_existing_gradients():
    module = FamilyModule()
    named, task, raw = _maps(module)
    for name, parameter in named:
        parameter.grad = (task[name] + 3.0).clone()
    before = {name: parameter.grad.clone() for name, parameter in named}
    collect_family_gradient_metrics(
        module,
        named,
        task_gradients=task,
        raw_anchor_gradients=raw,
        anchor_lambda=1.0e-3,
    )
    for name, parameter in named:
        assert torch.equal(parameter.grad, before[name])


def test_fp16_like_unscaled_statistics_and_historical_clip_order():
    module = FamilyModule()
    named, task, raw = _maps(module)
    target_name, target = next(iter(partition_image_adapter_parameters(module)["lora_adapters"]))
    task[target_name] = torch.ones_like(target, dtype=torch.float16)
    raw[target_name] = torch.full_like(target, 100.0, dtype=torch.float16)
    metrics = apply_family_safe_anchor_budget(
        module,
        named,
        task_gradients=task,
        raw_anchor_gradients=raw,
        anchor_lambda=1.0e-3,
        rho=0.10,
    )
    assert metrics["families"]["lora_adapters"]["effective_ratio"] <= 0.10 + 1.0e-10
    assert target.grad.dtype == target.dtype
    torch.nn.utils.clip_grad_norm_(module.parameters(), 0.5)
    assert torch.isfinite(target.grad).all()


def test_nonfinite_task_or_anchor_gradient_is_rejected():
    module = FamilyModule()
    named, task, raw = _maps(module)
    target_name, _ = next(iter(partition_image_adapter_parameters(module)["lora_adapters"]))
    raw[target_name].fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        apply_family_safe_anchor_budget(
            module,
            named,
            task_gradients=task,
            raw_anchor_gradients=raw,
            anchor_lambda=1.0e-3,
        )


def test_training_source_keeps_budget_after_unscale_before_clip():
    source = (Path(__file__).resolve().parents[1] / "train.py").read_text()
    backward = source.index("scaler.scale(loss).backward(retain_graph=anchor_gradient_budget)")
    unscale = source.index("scaler.unscale_(optimizer)", backward)
    budget = source.index("if anchor_gradient_budget:", unscale)
    apply = source.index("apply_family_safe_anchor_budget", budget)
    clip = source.index("clip_module_grad(model.image_adapter", apply)
    assert backward < unscale < budget < apply < clip
    assert "torch.autograd.grad(" in source[budget:apply]
