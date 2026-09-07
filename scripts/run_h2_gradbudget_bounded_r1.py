#!/usr/bin/env python3
"""Run H2 NEXT MECHANISM -- BOUNDED R1.

This screen starts both arms from the exact Safe-Anchor E1 full state and
changes only the selected stage-2/stage-3 Conv-LoRA task gradient.  It is
deliberately source-only: no endpoint inference is performed by this file.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
os.sys.path.insert(0, str(REPO))

from dataset import get_text_and_image_dataset
from h2_clean.contract import (  # noqa: E402
    EpochWorkerInit,
    SafeImageAdapterAnchor,
    apply_family_safe_anchor_budget,
    make_dataloader_generator,
    sha256_file,
)
from h2_clean.gradbudget import budgeted_gradient  # noqa: E402
from h2_clean.precision import PrecisionPolicy  # noqa: E402
from model.adapter_modules import ConvLoraAdapter  # noqa: E402
from model.adapter import ACDCLIP  # noqa: E402
from model.clip import create_model  # noqa: E402
from train import (  # noqa: E402
    apply_soft_prompt_lr_policy,
    compute_hybrid_k_regularization,
    get_dfg_beta_for_epoch,
    get_hybrid_alpha_for_epoch,
    has_non_finite_grad,
    optimizer_state_is_finite,
)
from utils import (  # noqa: E402
    BinaryDiceLoss,
    FocalLoss,
    calculate_seg_loss_components,
    get_hybrid_soft_prompt_single_class_text_embedding,
)


IMG = 518
SEED = 0
MAX_ATTEMPTS = 500
START_E1 = REPO / "runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth"
START_E1_SHA256 = "7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35"
ANCHOR_LAMBDA = 0.0021633926715180626
ANCHOR_FAMILY_BUDGET = 0.10
GRAD_EPS = 1.0e-12
ARM_CONTROL = "A_GRADBUDGET_SHORT_R1_CONTROL"
ARM_CANDIDATE = "A_GRADBUDGET_SHORT_R1_CANDIDATE"
PROTOCOL_ID = "H2_NEXT_MECHANISM_BOUNDED_R1_LATE_CONVLORA_ABNORMAL_DICE"
CONTROL_CSV = REPO / "audit/H2_GRADBUDGET_R1_CONTROL.csv"
CANDIDATE_CSV = REPO / "audit/H2_GRADBUDGET_R1_CANDIDATE.csv"

FOCAL = FocalLoss()
DICE = BinaryDiceLoss()
IDENTITY_KEYS = (
    "attempt_index",
    "epoch",
    "batch",
    "file_names",
    "image_sha256",
    "mask_sha256",
    "labels",
)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(staging, path)


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def rng_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "python_random_state": payload["python_random_state"],
        "numpy_random_state": payload["numpy_random_state"],
        "torch_cpu_rng_state": payload["torch_cpu_rng_state"],
        "torch_cuda_rng_state_all": payload["torch_cuda_rng_state_all"],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    if torch.cuda.is_available() and state["torch_cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])


def make_model(payload: dict[str, Any], device: torch.device) -> ACDCLIP:
    clip = create_model(
        "ViT-L-14-336",
        img_size=IMG,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip.set_grad_checkpointing(True)
    model = ACDCLIP(
        clip_model=clip,
        n_groups=3,
        image_adapt_weight=0.2,
        text_adapt_weight=0.2,
        conv_lora_rank=8,
        conv_lora_alpha=2.0,
        conv_kernel_size_list=[3, 5],
        lora_rank=16,
        lora_alpha=2.0,
        dfg_mode="attn",
        dfg_attn_dim=256,
        dfg_attn_tau=8.0,
        use_ss2d_dfg=True,
        dfg_gamma_max=0.2,
        dfg_ss2d_fusion="weight_residual",
        dfg_beta=0.1,
        dfg_beta_schedule="warmup010",
        dfg_beta_target=0.1,
        dfg_beta_current=0.1,
        dfg_weight_residual_fp32=True,
        stage_fusion_weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        use_soft_prompt=False,
        soft_prompt_ctx_len=4,
        soft_prompt_init="phrase",
        soft_prompt_init_phrase="a photo of a",
    ).to(device).eval()
    for module in model.modules():
        if hasattr(module, "enable_fp16_numerical_islands"):
            module.enable_fp16_numerical_islands = False
    model.image_adapter.load_state_dict(payload["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(payload["text_adapter"], strict=True)
    model.soft_prompt.load_state_dict(payload["soft_prompt"], strict=True)
    model.dfg_beta = float(payload["dfg_beta_current"])
    model.prompt_mode = "hybrid"
    model.use_hybrid_soft_prompt = True
    model.use_soft_prompt = False
    model.hybrid_alpha_current = 0.0
    model.hybrid_alpha_max = 0.2
    model.soft_prompt_freeze_epochs = 3
    model.requires_grad_(False)
    model.image_adapter.requires_grad_(True)
    model.text_adapter.requires_grad_(True)
    model.soft_prompt.requires_grad_(False)
    return model


def make_optimizer(model: ACDCLIP, payload: dict[str, Any]):
    optimizer = torch.optim.Adam([
        {"name": "text_adapter", "params": model.text_adapter.parameters(), "lr": 0.0005},
        {"name": "image_adapter", "params": model.image_adapter.parameters(), "lr": 0.001},
        {"name": "soft_prompt", "params": model.soft_prompt.parameters(), "lr": 0.0, "constant_lr": 0.00005},
    ])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    scaler.load_state_dict(payload["scaler_state"])
    return optimizer, scheduler, scaler


def loader_for_epoch(dataset, epoch: int):
    generator = make_dataloader_generator(SEED)
    generator.manual_seed(SEED + 104729 * epoch)
    worker = EpochWorkerInit(SEED)
    worker.set_epoch(epoch)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=6,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker,
    )


def configure_epoch(model: ACDCLIP, optimizer, epoch: int) -> bool:
    model.eval()
    model.image_encoder.eval()
    model.clipmodel.eval()
    frozen = epoch <= 3
    model.hybrid_alpha_current = get_hybrid_alpha_for_epoch(
        epoch,
        hybrid_alpha_max=0.2,
        soft_prompt_freeze_epochs=3,
    )
    model.soft_prompt.requires_grad_(not frozen)
    model.text_adapter.requires_grad_(True)
    apply_soft_prompt_lr_policy(optimizer, frozen)
    model.set_dfg_beta(get_dfg_beta_for_epoch(epoch, "warmup010", 0.1, 0.1))
    return frozen


def task_components(model, image, mask, label, class_names, device, policy):
    by_class = {}
    kg_losses = []
    k_losses = []
    for class_name in sorted(set(class_names)):
        text, kg, _, components = get_hybrid_soft_prompt_single_class_text_embedding(
            model,
            "VisA",
            class_name,
            device,
            return_kg=True,
            return_components=True,
        )
        k_loss, _ = compute_hybrid_k_regularization(
            model,
            components["hard_text"],
            components["soft_text"],
            model.hybrid_alpha_current,
        )
        by_class[class_name] = text
        kg_losses.append(kg)
        k_losses.append(k_loss)
    text = torch.stack([by_class[name] for name in class_names]).permute(1, 0, 2, 3)
    kg_loss = torch.stack(kg_losses).mean()
    k_loss = torch.stack(k_losses).mean()
    with policy.autocast(device):
        seg_tokens, det_tokens = model(image)
        seg_features = torch.stack(seg_tokens)
        det_features = torch.stack(det_tokens)
        cls = torch.stack([
            torch.matmul(det_features[i].unsqueeze(1), text[i]).squeeze(1)
            for i in range(3)
        ]).mean(0)
        cls_loss = F.cross_entropy(cls, label)
        pred = model.vision_text_fusion_gate_seg(seg_features, text)
        parts = calculate_seg_loss_components(pred, mask)
        # Keep the historical scalar sum while avoiding the in-place autograd
        # accumulator, so each named component has an independent VJP.
        seg_loss = parts["focal"] + parts["normal_dice"] + parts["abnormal_dice"]
        loss_main = cls_loss + seg_loss
        task_loss = loss_main + 0.01 * kg_loss + 0.002 * k_loss
        # Algebraically this is the exact task objective minus abnormal Dice,
        # expressed without subtracting through the in-place accumulation node.
        rest_seg_loss = parts["focal"].clone()
        rest_seg_loss += parts["normal_dice"]
        rest_task = cls_loss + rest_seg_loss + 0.01 * kg_loss + 0.002 * k_loss
    return {
        "task_loss": task_loss,
        "rest_task": rest_task,
        "abnormal_dice": parts["abnormal_dice"],
        "focal": parts["focal"],
        "normal_dice": parts["normal_dice"],
        "classification": cls_loss,
        "segmentation": seg_loss,
        "kg": kg_loss,
        "k": k_loss,
    }


def old_task_from_components(parts: dict[str, torch.Tensor]) -> torch.Tensor:
    """The pre-refactor loss expression used by historical ``train.py``."""
    seg_loss = parts["focal"].clone()
    seg_loss += parts["normal_dice"]
    seg_loss += parts["abnormal_dice"]
    loss_main = parts["classification"] + seg_loss
    return loss_main + 0.01 * parts["kg"] + 0.002 * parts["k"]


def finite_model_parameters(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters())


def discover_parameter_scope(model: ACDCLIP) -> dict[str, Any]:
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, ConvLoraAdapter)
    ]
    expected_names = ["image_adapter.lora_adapters.0", "image_adapter.lora_adapters.1", "image_adapter.lora_adapters.2"]
    actual_names = [name for name, _ in modules]
    if actual_names != expected_names:
        raise RuntimeError(f"unexpected Conv-LoRA module ordering: {actual_names}")
    image_levels = list(getattr(model, "image_levels", []))
    if image_levels != [8, 16, 24]:
        raise RuntimeError(f"unexpected image stage levels: {image_levels}")
    rows = []
    selected_names = []
    for module_index, (module_name, module) in enumerate(modules):
        stage = f"stage{module_index + 1}"
        for local_name, parameter in sorted(module.named_parameters()):
            full_name = f"{module_name}.{local_name}"
            selected = module_index in (1, 2)
            rows.append({
                "parameter_name": full_name,
                "module": module_name,
                "stage": stage,
                "module_index": module_index,
                "shape": json.dumps(list(parameter.shape), separators=(",", ":")),
                "requires_grad": int(parameter.requires_grad),
                "selected": int(selected),
            })
            if selected:
                selected_names.append(full_name)
    if not rows or not selected_names:
        raise RuntimeError("Conv-LoRA scope discovery returned no parameters")
    if any(row["selected"] and row["stage"] == "stage1" for row in rows):
        raise RuntimeError("stage1 was selected into the late Conv-LoRA scope")
    return {
        "module_order": actual_names,
        "stage_levels": image_levels,
        "stage_mapping": {
            "stage1": expected_names[0],
            "stage2": expected_names[1],
            "stage3": expected_names[2],
        },
        "selected_stages": ["stage2", "stage3"],
        "selected_module_names": [expected_names[1], expected_names[2]],
        "selected_parameter_names": selected_names,
        "parameter_count": len(rows),
        "selected_parameter_count": len(selected_names),
        "rows": rows,
    }


def write_scope(scope: dict[str, Any]) -> None:
    fields = [
        "parameter_name", "module", "stage", "module_index", "shape",
        "requires_grad", "selected",
    ]
    with (REPO / "audit/H2_GRADBUDGET_R1_PARAMETER_SCOPE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scope["rows"])
    json_dump(REPO / "audit/H2_GRADBUDGET_R1_PARAMETER_SCOPE.json", {
        key: value for key, value in scope.items() if key != "rows"
    })


def scope_parameters(model: ACDCLIP, scope: dict[str, Any]):
    all_pairs = [
        (name, parameter)
        for name, parameter in sorted(model.image_adapter.named_parameters())
        if parameter.requires_grad
    ]
    selected_set = set(scope["selected_parameter_names"])
    pairs = [
        (name.split("image_adapter.", 1)[-1], parameter)
        for name, parameter in sorted(model.named_parameters())
        if name in selected_set and parameter.requires_grad
    ]
    if {name for name, _ in pairs} != {name.split("image_adapter.", 1)[-1] for name in scope["selected_parameter_names"]}:
        raise RuntimeError("selected Conv-LoRA parameter lookup is incomplete")
    return all_pairs, pairs


def norm_of(values: Iterable[torch.Tensor | None]) -> float:
    tensors = [value.detach().float() for value in values if value is not None]
    if not tensors:
        return 0.0
    total = torch.zeros((), device=tensors[0].device, dtype=torch.float32)
    for value in tensors:
        total = total + value.square().sum()
    return float(total.sqrt().item())


def cosine_of(left: Iterable[torch.Tensor | None], right: Iterable[torch.Tensor | None]) -> float | None:
    left_values = [value.detach().float() if value is not None else None for value in left]
    right_values = [value.detach().float() if value is not None else None for value in right]
    dot = None
    left_sq = None
    right_sq = None
    for left_value, right_value in zip(left_values, right_values):
        if left_value is None or right_value is None:
            continue
        term_dot = (left_value * right_value).sum()
        term_left = left_value.square().sum()
        term_right = right_value.square().sum()
        dot = term_dot if dot is None else dot + term_dot
        left_sq = term_left if left_sq is None else left_sq + term_left
        right_sq = term_right if right_sq is None else right_sq + term_right
    if dot is None or float(left_sq.item()) == 0.0 or float(right_sq.item()) == 0.0:
        return None
    return float((dot / (left_sq.sqrt() * right_sq.sqrt())).item())


def batch_identity(attempt_index: int, epoch: int, batch_idx: int, batch, image, mask, label) -> dict[str, Any]:
    return {
        "attempt_index": int(attempt_index),
        "epoch": int(epoch),
        "batch": int(batch_idx),
        "file_names": json.dumps(list(batch["file_name"]), separators=(",", ":")),
        "image_sha256": tensor_hash(image),
        "mask_sha256": tensor_hash(mask),
        "labels": json.dumps(label.detach().cpu().tolist(), separators=(",", ":")),
    }


def prepare_payload() -> dict[str, Any]:
    if not START_E1.is_file():
        raise FileNotFoundError(START_E1)
    actual = sha256_file(START_E1)
    if actual != START_E1_SHA256:
        raise RuntimeError(f"E1 SHA256 mismatch: {actual}")
    payload = torch.load(START_E1, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 1 or payload.get("precision") not in ("amp", "fp16"):
        raise RuntimeError("shared E1 identity mismatch")
    if not bool(payload.get("amp_enabled")) or payload.get("scaler_state") is None:
        raise RuntimeError("E1 is missing the historical FP16 GradScaler state")
    return payload


def new_model_and_scope(payload: dict[str, Any], device: torch.device):
    model = make_model(payload, device)
    scope = discover_parameter_scope(model)
    return model, scope


def compare_gradient_lists(left, right) -> dict[str, Any]:
    max_abs = 0.0
    max_rel = 0.0
    left_norm = norm_of(left)
    right_norm = norm_of(right)
    for left_value, right_value in zip(left, right):
        left_value = torch.zeros_like(right_value) if left_value is None else left_value.detach().float()
        right_value = torch.zeros_like(left_value) if right_value is None else right_value.detach().float()
        difference = (left_value - right_value).abs()
        max_abs = max(max_abs, float(difference.max().item()))
        denominator = right_value.abs().maximum(torch.tensor(1.0e-12, device=right_value.device))
        max_rel = max(max_rel, float((difference / denominator).max().item()))
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "left_norm": left_norm,
        "right_norm": right_norm,
        "cosine": cosine_of(left, right),
    }


def loss_parity(payload: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    restore_rng_state(rng_from_payload(payload))
    model = make_model(payload, device)
    optimizer, _, _ = make_optimizer(model, payload)
    configure_epoch(model, optimizer, 2)
    dataset = get_text_and_image_dataset("VisA", IMG, "train")
    batch = next(iter(loader_for_epoch(dataset, 2)))
    image = batch["image"].to(device)
    mask = batch["mask"].to(device)
    label = batch["label"].to(device)
    policy = PrecisionPolicy("fp16")
    parts = task_components(model, image, mask, label, batch["class_name"], device, policy)
    old_task = old_task_from_components(parts)
    ref_task = parts["task_loss"]
    anchor_loss = SafeImageAdapterAnchor.from_checkpoint(START_E1, device).loss(model.image_adapter)
    old_total = old_task + ANCHOR_LAMBDA * anchor_loss
    ref_total = ref_task + ANCHOR_LAMBDA * anchor_loss
    # The parity graph is retained until every component gradient comparison is complete.
    all_pairs, selected_pairs = scope_parameters(model, scope)
    all_parameters = [parameter for _, parameter in all_pairs]
    selected_parameters = [parameter for _, parameter in selected_pairs]
    old_selected = torch.autograd.grad(old_task, selected_parameters, retain_graph=True, allow_unused=True)
    ref_selected = torch.autograd.grad(ref_task, selected_parameters, retain_graph=True, allow_unused=True)
    rest_selected = torch.autograd.grad(parts["rest_task"], selected_parameters, retain_graph=True, allow_unused=True)
    residual_abn = [
        None if task_value is None and rest_value is None else
        (torch.zeros_like(rest_value) if task_value is None else task_value) -
        (torch.zeros_like(task_value) if rest_value is None else rest_value)
        for task_value, rest_value in zip(ref_selected, rest_selected)
    ]
    # This residual is the abnormal-Dice VJP implied by the scalar identity
    # task = rest + abnormal, while preserving the historical task backward.
    abn_selected = residual_abn
    direct_task = parts["rest_task"] + parts["abnormal_dice"]
    direct_task_selected = torch.autograd.grad(direct_task, selected_parameters, retain_graph=True, allow_unused=True)
    direct_sum = [
        None if rest_value is None and abn_value is None else
        (torch.zeros_like(abn_value) if rest_value is None else rest_value) +
        (torch.zeros_like(rest_value) if abn_value is None else abn_value)
        for rest_value, abn_value in zip(rest_selected, abn_selected)
    ]
    disabled, disabled_metrics = budgeted_gradient(
        selected_parameters,
        rest_selected,
        abn_selected,
        enabled=False,
        eps=GRAD_EPS,
    )
    old_vs_ref = compare_gradient_lists(old_selected, ref_selected)
    ref_vs_disabled = compare_gradient_lists(ref_selected, disabled)
    with torch.no_grad():
        scaled_rest = [None if value is None else value * 1024.0 for value in rest_selected]
        scaled_abn = [None if value is None else value * 1024.0 for value in abn_selected]
        _, scaled_metrics = budgeted_gradient(selected_parameters, scaled_rest, scaled_abn, enabled=True, eps=GRAD_EPS)
    synthetic_rest = [torch.tensor([3.0], device=device)]
    synthetic_abn = [torch.tensor([12.0], device=device)]
    _, synthetic_metrics = budgeted_gradient(
        [torch.nn.Parameter(torch.zeros(1, device=device))],
        synthetic_rest,
        synthetic_abn,
        enabled=True,
        eps=GRAD_EPS,
    )
    max_loss_identity = float((old_task - ref_task).detach().float().abs().item())
    max_total_identity = float((old_total - ref_total).detach().float().abs().item())
    result = {
        "protocol_id": PROTOCOL_ID,
        "loss_identity": {
            "old_task_expression": "loss_main + 0.01*kg_loss + 0.002*k_loss",
            "rest_task_expression": "classification + focal + normal_dice + 0.01*kg_loss + 0.002*k_loss",
            "candidate_task_expression": "rest_task + abnormal_dice",
            "old_task_vs_ref_task_abs": max_loss_identity,
            "old_total_task_plus_anchor_identity_abs": max_total_identity,
            "rest_plus_abnormal_vs_task_abs": float((direct_task - ref_task).detach().float().abs().item()),
            "dtypes": {name: str(parts[name].dtype) for name in ("task_loss", "rest_task", "abnormal_dice", "focal", "normal_dice")},
            "anchor_excluded_from_gradbudget_task": True,
        },
        "gradient_identity": {
            "scope": "stage2+stage3 Conv-LoRA only",
            "selected_parameter_count": len(selected_parameters),
            "old_vs_ref_selected": old_vs_ref,
            "ref_vs_direct_rest_plus_abnormal": compare_gradient_lists(ref_selected, direct_sum),
            "ref_vs_direct_task_expression": compare_gradient_lists(ref_selected, direct_task_selected),
            "abnormal_gradient_definition": "unscaled task gradient minus unscaled rest-task gradient on the same graph",
            "component_norms": {
                "rest": norm_of(rest_selected),
                "abnormal": norm_of(abn_selected),
                "task": norm_of(ref_selected),
                "rest_plus_abnormal": norm_of(direct_sum),
            },
            "ref_vs_disabled_budget": ref_vs_disabled,
            "disabled_alpha": disabled_metrics["alpha"],
            "disabled_budget_active": disabled_metrics["budget_active"],
            "safe_anchor_application_calls": 1,
        },
        "unscaled_alpha_test": {
            "scale_factor": 1024.0,
            "raw_ratio": scaled_metrics["raw_ratio"],
            "alpha": scaled_metrics["alpha"],
            "gradients_unscaled": scaled_metrics["gradients_unscaled"],
            "matches_unscaled_norm_ratio": math.isclose(
                float(scaled_metrics["raw_ratio"]),
                float(disabled_metrics["raw_ratio"]),
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            ),
        },
        "synthetic_active_budget_test": synthetic_metrics,
        "finite": {
            "losses": all(bool(torch.isfinite(parts[name]).all().item()) for name in ("task_loss", "rest_task", "abnormal_dice")),
            "selected_gradients": all(value is None or bool(torch.isfinite(value).all().item()) for value in (*old_selected, *ref_selected, *rest_selected, *abn_selected)),
        },
        "PASS": bool(
            max_loss_identity <= 1.0e-6
            and old_vs_ref["max_abs"] <= 1.0e-5
            and old_vs_ref["cosine"] is not None
            and old_vs_ref["cosine"] >= 0.999999
            and ref_vs_disabled["max_abs"] <= 1.0e-5
            and ref_vs_disabled["cosine"] is not None
            and ref_vs_disabled["cosine"] >= 0.999999
            and bool(scaled_metrics["gradients_unscaled"])
            and bool(synthetic_metrics["budget_active"])
        ),
    }
    json_dump(REPO / "audit/H2_GRADBUDGET_R1_LOSS_PARITY.json", result)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    if not result["PASS"]:
        raise RuntimeError("H2_GRADBUDGET_R1 loss/gradient parity failed")
    return result


def collect_batch_ids(payload: dict[str, Any], limit: int = 16) -> list[dict[str, Any]]:
    device = torch.device("cuda:0")
    restore_rng_state(rng_from_payload(payload))
    model = make_model(payload, device)
    del model
    dataset = get_text_and_image_dataset("VisA", IMG, "train")
    rows = []
    for epoch in range(2, 9):
        for batch_idx, batch in enumerate(loader_for_epoch(dataset, epoch)):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)
            rows.append(batch_identity(len(rows), epoch, batch_idx, batch, image, mask, label))
            if len(rows) == limit:
                return rows
    raise RuntimeError(f"batch preflight did not collect {limit} attempts")


def batch_preflight(payload: dict[str, Any]) -> dict[str, Any]:
    control = collect_batch_ids(payload, limit=16)
    candidate = collect_batch_ids(payload, limit=16)
    result = {
        "protocol_id": PROTOCOL_ID,
        "PREFLIGHT_BATCH_PARITY": "PASS" if control == candidate else "FAIL",
        "batch_count": 16,
        "identity_fields": list(IDENTITY_KEYS),
        "control": control,
        "candidate": candidate,
        "exact_attempt_identity": control == candidate,
    }
    json_dump(REPO / "audit/H2_GRADBUDGET_R1_BATCH_PREFLIGHT.json", result)
    if result["PREFLIGHT_BATCH_PARITY"] != "PASS":
        raise RuntimeError("H2_GRADBUDGET_R1 batch preflight failed")
    return result


def stage_norms(parameter_pairs, gradients):
    by_stage = {"stage2": [], "stage3": []}
    for (name, _), gradient in zip(parameter_pairs, gradients):
        stage = "stage2" if name.startswith("lora_adapters.1.") else "stage3" if name.startswith("lora_adapters.2.") else None
        if stage is not None:
            by_stage[stage].append(gradient)
    return {stage: norm_of(values) for stage, values in by_stage.items()}


def gradient_preflight(payload: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    restore_rng_state(rng_from_payload(payload))
    model = make_model(payload, device)
    optimizer, _, _ = make_optimizer(model, payload)
    configure_epoch(model, optimizer, 2)
    dataset = get_text_and_image_dataset("VisA", IMG, "train")
    _, selected_pairs = scope_parameters(model, scope)
    selected_parameters = [parameter for _, parameter in selected_pairs]
    rows = []
    for epoch in range(2, 9):
        configure_epoch(model, optimizer, epoch)
        for batch_idx, batch in enumerate(loader_for_epoch(dataset, epoch)):
            if len(rows) == 16:
                break
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)
            parts = task_components(model, image, mask, label, batch["class_name"], device, policy)
            identity = batch_identity(len(rows), epoch, batch_idx, batch, image, mask, label)
            task = torch.autograd.grad(parts["task_loss"], selected_parameters, retain_graph=True, allow_unused=True)
            rest = torch.autograd.grad(parts["rest_task"], selected_parameters, retain_graph=True, allow_unused=True)
            abnormal = [
                None if task_value is None and rest_value is None else
                (torch.zeros_like(rest_value) if task_value is None else task_value) -
                (torch.zeros_like(task_value) if rest_value is None else rest_value)
                for task_value, rest_value in zip(task, rest)
            ]
            _, metrics = budgeted_gradient(selected_parameters, rest, abnormal, enabled=True, eps=GRAD_EPS)
            rows.append({
                **identity,
                **metrics,
                "stage2_g_rest_norm": stage_norms(selected_pairs, rest)["stage2"],
                "stage3_g_rest_norm": stage_norms(selected_pairs, rest)["stage3"],
                "stage2_g_abn_norm": stage_norms(selected_pairs, abnormal)["stage2"],
                "stage3_g_abn_norm": stage_norms(selected_pairs, abnormal)["stage3"],
                "finite": int(all(value is None or bool(torch.isfinite(value).all().item()) for value in (*rest, *abnormal))),
                "optimizer_step": 0,
                "gradients_unscaled": int(metrics["gradients_unscaled"]),
            })
        if len(rows) == 16:
            break
    fields = sorted({key for row in rows for key in row})
    with (REPO / "audit/H2_GRADBUDGET_R1_GRADIENT_PREFLIGHT.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "protocol_id": PROTOCOL_ID,
        "rows": len(rows),
        "optimizer_steps": 0,
        "gradients_unscaled": all(bool(row["gradients_unscaled"]) for row in rows),
        "finite": all(bool(row["finite"]) for row in rows),
        "budget_active_rows": sum(bool(row["budget_active"]) for row in rows),
        "alpha_min": min(float(row["alpha"]) for row in rows),
        "alpha_max": max(float(row["alpha"]) for row in rows),
        "PASS": len(rows) == 16 and all(bool(row["finite"]) for row in rows),
    }
    json_dump(REPO / "audit/H2_GRADBUDGET_R1_GRADIENT_PREFLIGHT.json", result)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    if not result["PASS"]:
        raise RuntimeError("H2_GRADBUDGET_R1 gradient preflight failed")
    return result


def read_control_identities() -> list[dict[str, str]]:
    with CONTROL_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != MAX_ATTEMPTS:
        raise RuntimeError(f"control has {len(rows)} rows, expected {MAX_ATTEMPTS}")
    if not set(IDENTITY_KEYS).issubset(rows[0]):
        raise RuntimeError("control CSV is missing batch identity columns")
    return rows


def check_candidate_identity(identity: dict[str, Any], expected: dict[str, str]) -> str | None:
    for key in IDENTITY_KEYS:
        if str(identity[key]) != str(expected[key]):
            return f"{key} mismatch at attempt {identity['attempt_index']}"
    return None


def scalar_or_none(value: torch.Tensor) -> float | None:
    return float(value.detach().float().cpu().item()) if bool(torch.isfinite(value).all().item()) else None


def run_arm(payload: dict[str, Any], scope: dict[str, Any], root: Path, arm: str) -> dict[str, Any]:
    if arm not in (ARM_CONTROL, ARM_CANDIDATE):
        raise ValueError(arm)
    candidate = arm == ARM_CANDIDATE
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    restore_rng_state(rng_from_payload(payload))
    model = make_model(payload, device)
    optimizer, scheduler, scaler = make_optimizer(model, payload)
    anchor = SafeImageAdapterAnchor.from_checkpoint(START_E1, device)
    dataset = get_text_and_image_dataset("VisA", IMG, "train")
    expected = read_control_identities() if candidate else None
    _, selected_pairs = scope_parameters(model, scope)
    selected_parameters = [parameter for _, parameter in selected_pairs]

    rows = []
    attempted = 0
    successful = 0
    nonfinite_loss_skips = 0
    nonfinite_grad_skips = 0
    consecutive_grad_skips = 0
    max_consecutive_grad_skips = 0
    optimizer_state_failures = 0
    parameter_corruption = 0
    batch_mismatch = None
    numerical_failure = None
    safe_anchor_calls = 0
    stage_drift_outside_scope = True

    for epoch in range(2, 9):
        frozen = configure_epoch(model, optimizer, epoch)
        for batch_idx, batch in enumerate(loader_for_epoch(dataset, epoch)):
            if attempted >= MAX_ATTEMPTS or batch_mismatch is not None:
                break
            attempt_index = attempted
            attempted += 1
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)
            identity = batch_identity(attempt_index, epoch, batch_idx, batch, image, mask, label)
            if candidate:
                mismatch = check_candidate_identity(identity, expected[attempt_index])
                if mismatch is not None:
                    batch_mismatch = mismatch
                    rows.append({**identity, "arm": arm, "status": "batch_mismatch"})
                    break
            optimizer.zero_grad(set_to_none=True)
            parts = task_components(model, image, mask, label, batch["class_name"], device, policy)
            anchor_loss = anchor.loss(model.image_adapter)
            row = {
                **identity,
                "arm": arm,
                "status": "pending",
                "base_task_loss": scalar_or_none(parts["task_loss"]),
                "rest_task_loss": scalar_or_none(parts["rest_task"]),
                "abnormal_dice_loss": scalar_or_none(parts["abnormal_dice"]),
                "classification_loss": scalar_or_none(parts["classification"]),
                "focal_loss": scalar_or_none(parts["focal"]),
                "normal_dice_loss": scalar_or_none(parts["normal_dice"]),
                "anchor_loss": scalar_or_none(anchor_loss),
                "successful_step": 0,
                "nonfinite_loss_skip": 0,
                "nonfinite_grad_skip": 0,
                "consecutive_nonfinite_grad_skips": 0,
                "optimizer_state_finite": None,
                "parameters_finite": None,
                "safe_anchor_effective_ratio": None,
                "safe_anchor_calls": 0,
                "g_abn_norm": None,
                "g_rest_norm": None,
                "raw_ratio": None,
                "alpha": None,
                "computed_alpha": None,
                "budget_active": None,
                "effective_abn_norm": None,
                "effective_ratio": None,
                "cosine_abn_rest": None,
                "selected_task_norm_before": None,
                "selected_task_norm_after": None,
                "selected_gradient_finite": None,
                "gradients_unscaled": 1,
                "stage1_outside_scope": int(stage_drift_outside_scope),
            }
            if not bool(torch.isfinite(parts["task_loss"]).all().item()) or not bool(torch.isfinite(anchor_loss).all().item()):
                nonfinite_loss_skips += 1
                consecutive_grad_skips = 0
                row.update(status="nonfinite_loss_skip", nonfinite_loss_skip=1)
                optimizer.zero_grad(set_to_none=True)
                rows.append(row)
                continue

            backward_task = parts["task_loss"] if candidate else old_task_from_components(parts)
            scaler.scale(backward_task).backward(retain_graph=True)
            scaler.unscale_(optimizer)
            if has_non_finite_grad(optimizer):
                nonfinite_grad_skips += 1
                consecutive_grad_skips += 1
                max_consecutive_grad_skips = max(max_consecutive_grad_skips, consecutive_grad_skips)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                row.update(
                    status="nonfinite_grad_skip",
                    nonfinite_grad_skip=1,
                    consecutive_nonfinite_grad_skips=consecutive_grad_skips,
                )
                rows.append(row)
                if consecutive_grad_skips > 1:
                    numerical_failure = numerical_failure or "consecutive_nonfinite_gradient_skips"
                continue
            consecutive_grad_skips = 0

            all_image_pairs = [
                (name, parameter)
                for name, parameter in sorted(model.image_adapter.named_parameters())
                if parameter.requires_grad
            ]
            all_image_names = [name for name, _ in all_image_pairs]
            all_image_parameters = [parameter for _, parameter in all_image_pairs]
            task_gradients = tuple(
                None if parameter.grad is None else parameter.grad.detach().float().clone()
                for parameter in all_image_parameters
            )
            rest_gradients = torch.autograd.grad(parts["rest_task"], selected_parameters, retain_graph=True, allow_unused=True)
            selected_task_gradients = [
                task_gradients[all_image_names.index(name)]
                for name, _ in selected_pairs
            ]
            abnormal_gradients = [
                None if task_value is None and rest_value is None else
                (torch.zeros_like(rest_value) if task_value is None else task_value) -
                (torch.zeros_like(task_value) if rest_value is None else rest_value)
                for task_value, rest_value in zip(selected_task_gradients, rest_gradients)
            ]
            effective, metrics = budgeted_gradient(
                selected_parameters,
                rest_gradients,
                abnormal_gradients,
                enabled=candidate,
                eps=GRAD_EPS,
            )
            selected_task_before = selected_task_gradients
            selected_task_after = selected_task_before if not candidate else effective
            row.update({
                key: value for key, value in metrics.items()
                if key in ("g_abn_norm", "g_rest_norm", "raw_ratio", "alpha", "computed_alpha", "budget_active", "effective_abn_norm", "effective_ratio", "cosine_abn_rest", "gradients_unscaled")
            })
            row["selected_task_norm_before"] = norm_of(selected_task_before)
            row["selected_task_norm_after"] = norm_of(selected_task_after)
            row["selected_gradient_finite"] = int(all(value is None or bool(torch.isfinite(value).all().item()) for value in (*selected_task_after, *rest_gradients, *abnormal_gradients)))
            if not row["selected_gradient_finite"]:
                numerical_failure = numerical_failure or "nonfinite_selected_gradient"
                optimizer.zero_grad(set_to_none=True)
                row.update(status="nonfinite_selected_gradient", nonfinite_grad_skip=1)
                rows.append(row)
                continue
            if candidate:
                selected_index = {id(parameter): index for index, parameter in enumerate(all_image_parameters)}
                task_gradients = list(task_gradients)
                for (_, parameter), gradient in zip(selected_pairs, effective):
                    task_gradients[selected_index[id(parameter)]] = gradient
            raw_anchor_gradients = torch.autograd.grad(anchor_loss, all_image_parameters, retain_graph=False, allow_unused=True)
            safe_anchor_calls += 1
            anchor_metrics = apply_family_safe_anchor_budget(
                model.image_adapter,
                sorted(model.named_parameters()),
                task_gradients=dict(zip(all_image_names, task_gradients)),
                raw_anchor_gradients=dict(zip(all_image_names, raw_anchor_gradients)),
                anchor_lambda=ANCHOR_LAMBDA,
                rho=ANCHOR_FAMILY_BUDGET,
                total_trainable_parameters=None,
            )
            row["safe_anchor_calls"] = 1
            row["safe_anchor_effective_ratio"] = anchor_metrics["global_effective_ratio"]
            torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.text_adapter.parameters(), 1.0)
            if not frozen:
                torch.nn.utils.clip_grad_norm_(model.soft_prompt.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            successful += 1
            state_finite = bool(optimizer_state_is_finite(optimizer))
            params_finite = finite_model_parameters(model)
            if not state_finite:
                optimizer_state_failures += 1
                numerical_failure = numerical_failure or "nonfinite_optimizer_state"
            if not params_finite:
                parameter_corruption += 1
                numerical_failure = numerical_failure or "nonfinite_parameter_corruption"
            row.update({
                "status": "success" if state_finite and params_finite else "numerical_failure",
                "successful_step": 1,
                "optimizer_state_finite": int(state_finite),
                "parameters_finite": int(params_finite),
            })
            rows.append(row)
        if attempted >= MAX_ATTEMPTS or batch_mismatch is not None:
            break
        scheduler.step()
        apply_soft_prompt_lr_policy(optimizer, epoch <= 3)

    output_dir = root / arm
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol_id": PROTOCOL_ID,
        "arm": arm,
        "dataset": "VisA",
        "seed": SEED,
        "image_size": IMG,
        "batch_size": 6,
        "max_attempts": MAX_ATTEMPTS,
        "stage_fusion_weights_train": [1.0 / 3.0] * 3,
        "stage_fusion_weights_endpoint": [1.0 / 3.0] * 3,
        "stage_order": ["stage1", "stage2", "stage3"],
        "stage_selection": ["stage2", "stage3"],
        "selected_module_names": scope["selected_module_names"],
        "loss_components": ["classification", "focal", "normal_dice", "abnormal_dice", "lambda_kg_kg", "lambda_k_k"],
        "loss_rest_identity": "exact historical task loss minus abnormal Dice; Safe Anchor is separate",
        "mechanism": "alpha=min(1,||g_rest||_S/(||g_abn||_S+eps)); g_S=g_rest+alpha*g_abn",
        "mechanism_enabled": candidate,
        "gradient_scope": "stage2+stage3 Conv-LoRA only",
        "gradient_eps": GRAD_EPS,
        "safe_anchor": True,
        "anchor_lambda": ANCHOR_LAMBDA,
        "anchor_family_budget_rho": ANCHOR_FAMILY_BUDGET,
        "functional_feature_anchor": False,
        "precision_protocol": "HISTORICAL_MIXED_FP16_FP32_V1",
        "fp16_autocast": True,
        "parameters_fp32": True,
        "optimizer_state_fp32": True,
        "grad_scaler": True,
        "tf32": False,
        "bf16": False,
        "equal_train_fusion": True,
        "no_target_inference": True,
        "shared_e1_sha256": sha256_file(START_E1),
    }
    result = {
        "protocol_id": PROTOCOL_ID,
        "arm": arm,
        "config": config,
        "attempted_steps": attempted,
        "successful_steps": successful,
        "nonfinite_loss_skips": nonfinite_loss_skips,
        "nonfinite_grad_skips": nonfinite_grad_skips,
        "max_consecutive_nonfinite_grad_skips": max_consecutive_grad_skips,
        "optimizer_state_failures": optimizer_state_failures,
        "parameter_corruption": parameter_corruption,
        "safe_anchor_application_calls": safe_anchor_calls,
        "batch_match_gate": "FAIL" if batch_mismatch else "PASS",
        "batch_mismatch": batch_mismatch,
        "numerical_failure": numerical_failure,
        "stage1_outside_scope": stage_drift_outside_scope,
        "rows": rows,
    }
    torch.save({
        "model_state": {
            "image_adapter": model.image_adapter.state_dict(),
            "text_adapter": model.text_adapter.state_dict(),
            "soft_prompt": model.soft_prompt.state_dict(),
        },
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": config,
        "attempted_steps": attempted,
        "successful_steps": successful,
        "nonfinite_loss_skips": nonfinite_loss_skips,
        "nonfinite_grad_skips": nonfinite_grad_skips,
        "hybrid_alpha_current": float(model.hybrid_alpha_current),
        "dfg_beta_current": float(model.dfg_beta),
    }, output_dir / "final.pth")
    fields = sorted({key for row in rows for key in row})
    output_csv = CONTROL_CSV if not candidate else CANDIDATE_CSV
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_dump(output_dir / "summary.json", result)
    if attempted != MAX_ATTEMPTS:
        raise RuntimeError(f"expected exactly {MAX_ATTEMPTS} attempted steps, got {attempted}")
    if batch_mismatch is not None:
        raise RuntimeError("BATCH_MATCH_GATE=FAIL")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("scope", "parity", "batch-preflight", "gradient-preflight", "run"), required=True)
    parser.add_argument("--arm", choices=(ARM_CONTROL, ARM_CANDIDATE))
    parser.add_argument("--root", default="/workspace/h2_late_convlora_gradbudget_bounded_r1")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    payload = prepare_payload()
    device = torch.device("cuda:0")
    restore_rng_state(rng_from_payload(payload))
    model, scope = new_model_and_scope(payload, device)
    write_scope(scope)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if args.mode == "scope":
        print(json.dumps({"status": "PASS", "scope": {key: value for key, value in scope.items() if key != "rows"}}, sort_keys=True))
        return
    if args.mode == "parity":
        result = loss_parity(payload, scope)
    elif args.mode == "batch-preflight":
        result = batch_preflight(payload)
    elif args.mode == "gradient-preflight":
        result = gradient_preflight(payload, scope)
    else:
        if args.arm is None:
            raise ValueError("--arm is required for --mode run")
        result = run_arm(payload, scope, Path(args.root), args.arm)
    print(json.dumps({
        "status": "PASS",
        "mode": args.mode,
        "arm": args.arm,
        "attempted": result.get("attempted_steps", result.get("batch_count")),
        "successful": result.get("successful_steps"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
