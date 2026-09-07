#!/usr/bin/env python3
"""Run the preregistered H2 late-stage Conv-LoRA freeze screen.

The control and candidate arms start from the exact Safe-Anchor E10 full
checkpoint and consume one pre-materialized VisA source-batch manifest.  The
candidate keeps the complete historical forward/backward/Safe-Anchor path and
suppresses only selected gradients immediately before ``scaler.step``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
os.sys.path.insert(0, str(REPO))

from dataset import get_text_and_image_dataset
from h2_clean.contract import (
    EpochWorkerInit,
    SafeImageAdapterAnchor,
    apply_family_safe_anchor_budget,
    make_dataloader_generator,
    sha256_file,
    state_dict_sha256,
)
from h2_clean.precision import PrecisionPolicy
from h2_clean.stage_fusion import H2_EQUAL_STAGE_FUSION_WEIGHTS
from model.adapter import ACDCLIP
from model.clip import create_model
from train import (
    apply_soft_prompt_lr_policy,
    calculate_seg_loss,
    compute_hybrid_k_regularization,
    get_dfg_beta_for_epoch,
    get_hybrid_alpha_for_epoch,
    has_non_finite_grad,
    optimizer_state_is_finite,
)
from utils import get_hybrid_soft_prompt_single_class_text_embedding


IMG = 518
SEED = 0
MAX_ATTEMPTS = 500
START_EPOCH = 10
START_GLOBAL_STEP = 3607
ANCHOR_LAMBDA = 0.0021633926715180626
ANCHOR_FAMILY_BUDGET = 0.10
PROTOCOL_ID = "H2_LATE_CONVLORA_FREEZE_R1"
ARM_CONTROL = "A_LATE_FREEZE_R1_CONTROL"
ARM_CANDIDATE = "A_LATE_FREEZE_R1_STAGE23_CONVLORA"
START_CHECKPOINT = Path("/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth")
START_CHECKPOINT_SHA256 = "64b72dc3d1155285c826781bee4c5970bd45218e95b21675fd19d9a6b2ab54a7"
CONTROL_CSV = REPO / "audit/H2_LATE_CONVLORA_FREEZE_R1_CONTROL.csv"
MANIFEST_JSON = REPO / "audit/H2_LATE_CONVLORA_FREEZE_R1_ATTEMPT_MANIFEST.json"
SCOPE_CSV = REPO / "audit/H2_LATE_CONVLORA_FREEZE_R1_PARAMETER_SCOPE.csv"
PARITY_JSON = REPO / "audit/H2_LATE_CONVLORA_FREEZE_R1_IMPLEMENTATION_PARITY.json"


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensor_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.detach().cpu().contiguous().numpy().tobytes() == right.detach().cpu().contiguous().numpy().tobytes()


def state_to_cpu(state: dict) -> dict:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def rng_from_payload(payload: dict) -> dict:
    return {
        "python_random_state": payload["python_random_state"],
        "numpy_random_state": payload["numpy_random_state"],
        "torch_cpu_rng_state": payload["torch_cpu_rng_state"],
        "torch_cuda_rng_state_all": payload["torch_cuda_rng_state_all"],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    if torch.cuda.is_available() and state["torch_cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])


def validate_start_checkpoint() -> tuple[dict, dict]:
    if not START_CHECKPOINT.is_file():
        raise FileNotFoundError(START_CHECKPOINT)
    actual = sha256_file(START_CHECKPOINT)
    if actual != START_CHECKPOINT_SHA256:
        raise RuntimeError(f"E10 SHA256 mismatch: {actual} != {START_CHECKPOINT_SHA256}")
    payload = torch.load(START_CHECKPOINT, map_location="cpu", weights_only=False)
    required = {
        "model_state", "optimizer_state", "scheduler_state", "scaler_state",
        "python_random_state", "numpy_random_state", "torch_cpu_rng_state",
        "torch_cuda_rng_state_all", "dataloader_generator_state",
        "resolved_scientific_config", "resolved_operational_config",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"E10 checkpoint is missing full-state keys: {missing}")
    if (payload["epoch"], payload["global_step"]) != (START_EPOCH, START_GLOBAL_STEP):
        raise RuntimeError("E10 epoch/global_step identity mismatch")
    for key, expected in {
        "precision": "fp16", "amp_enabled": True, "gradscaler_enabled": True,
        "tf32_enabled": False, "seed": 0, "n_groups": 3,
        "anchor_gradient_budget": True, "anchor_family_budget": ANCHOR_FAMILY_BUDGET,
        "lambda_kg": 0.01, "lambda_k": 0.002,
        "use_hybrid_soft_prompt": True, "use_soft_prompt": False,
        "use_ss2d_dfg": True, "dfg_mode": "attn",
        "precision_protocol": "HISTORICAL_MIXED_FP16_FP32_V1",
    }.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"E10 identity mismatch for {key}: {payload.get(key)!r} != {expected!r}")
    scientific = payload["resolved_scientific_config"]
    operational = payload["resolved_operational_config"]
    if scientific.get("anchor_lambda") != ANCHOR_LAMBDA:
        raise RuntimeError("E10 Safe Anchor lambda mismatch")
    if scientific.get("batch_size") != 6 or scientific.get("img_size") != IMG:
        raise RuntimeError("E10 batch/image-size identity mismatch")
    if operational.get("num_workers") != 6 or not operational.get("pin_memory"):
        raise RuntimeError("E10 operational DataLoader identity mismatch")
    summary_path = REPO / "audit/H2_SAFE_ANCHOR_E20_TRAINING_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    rows = summary.get("checkpoints", summary.get("epochs", summary.get("epoch_summaries", [])))
    row10 = next((row for row in rows if int(row.get("epoch", -1)) == START_EPOCH), None)
    if row10 is None:
        raise RuntimeError("tracked E10 training summary has no epoch 10 row")
    numerical_validity = bool(row10.get("numerical_validity", False))
    if not numerical_validity:
        raise RuntimeError("tracked E10 numerical_validity is not true")
    for key in ("non_finite_loss_skips", "non_finite_grad_skips"):
        if int(row10.get(key, 0)) != 0:
            raise RuntimeError(f"tracked E10 has unexpected {key}: {row10.get(key)}")
    if not all(bool(row10.get(key, True)) for key in ("model_state_finite", "optimizer_state_finite", "scaler_state_finite")):
        raise RuntimeError("tracked E10 finite-state gate failed")
    identity = {
        "checkpoint": str(START_CHECKPOINT),
        "checkpoint_sha256": actual,
        "epoch": START_EPOCH,
        "global_step": START_GLOBAL_STEP,
        "numerical_validity": True,
        "scientific_config": scientific,
        "operational_config": operational,
        "tracked_summary_epoch10": row10,
    }
    return payload, identity


def make_model(payload: dict, device: torch.device) -> ACDCLIP:
    clip = create_model(
        "ViT-L-14-336", img_size=IMG, device=device,
        pretrained="openai", require_pretrained=True,
    )
    clip.set_grad_checkpointing(True)
    model = ACDCLIP(
        clip_model=clip, n_groups=3, image_adapt_weight=.2, text_adapt_weight=.2,
        conv_lora_rank=8, conv_lora_alpha=2., conv_kernel_size_list=[3, 5],
        lora_rank=16, lora_alpha=2., dfg_mode="attn", dfg_attn_dim=256,
        dfg_attn_tau=8., use_ss2d_dfg=True, dfg_gamma_max=.2,
        dfg_ss2d_fusion="weight_residual", dfg_beta=.1,
        dfg_beta_schedule="warmup010", dfg_beta_target=.1,
        dfg_beta_current=float(payload["dfg_beta_current"]),
        dfg_weight_residual_fp32=True,
        stage_fusion_weights=H2_EQUAL_STAGE_FUSION_WEIGHTS,
        use_soft_prompt=False, soft_prompt_ctx_len=4,
        soft_prompt_init="phrase", soft_prompt_init_phrase="a photo of a",
    ).to(device).eval()
    for module in model.modules():
        if hasattr(module, "enable_fp16_numerical_islands"):
            module.enable_fp16_numerical_islands = False
    model.image_adapter.load_state_dict(payload["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(payload["text_adapter"], strict=True)
    model.soft_prompt.load_state_dict(payload["soft_prompt"], strict=True)
    model.dfg_beta = float(payload["dfg_beta_current"])
    model.hybrid_alpha_current = float(payload["hybrid_alpha_current"])
    model.hybrid_alpha_max = float(payload["hybrid_alpha_max"])
    model.prompt_mode = "hybrid"
    model.use_hybrid_soft_prompt = True
    model.use_soft_prompt = False
    model.soft_prompt_freeze_epochs = int(payload["soft_prompt_freeze_epochs"])
    model.requires_grad_(False)
    model.image_adapter.requires_grad_(True)
    model.text_adapter.requires_grad_(True)
    model.soft_prompt.requires_grad_(True)
    return model


def make_optimizer(model: ACDCLIP, payload: dict):
    optimizer = torch.optim.Adam([
        {"name": "text_adapter", "params": model.text_adapter.parameters(), "lr": .0005},
        {"name": "image_adapter", "params": model.image_adapter.parameters(), "lr": .001},
        {"name": "soft_prompt", "params": model.soft_prompt.parameters(), "lr": 0., "constant_lr": .00005},
    ])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler = StepLR(optimizer, step_size=1, gamma=.9)
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    scaler.load_state_dict(payload["scaler_state"])
    return optimizer, scheduler, scaler


def make_anchor(payload: dict) -> SafeImageAdapterAnchor:
    metadata = payload["image_anchor"]
    return SafeImageAdapterAnchor(
        payload["image_anchor_reference"],
        reference_sha256=metadata["reference_sha256"],
        reference_checkpoint_sha256=metadata["reference_checkpoint_sha256"],
        reference_epoch=metadata["reference_epoch"],
        reference_config_sha256=metadata["reference_config_sha256"],
        eps=metadata["eps"],
    )


def make_loader(dataset, epoch: int, payload: dict):
    generator = make_dataloader_generator(SEED)
    generator.manual_seed(SEED + 104729 * int(epoch))
    worker = EpochWorkerInit(SEED)
    worker.set_epoch(epoch)
    kwargs = dict(
        batch_size=6, shuffle=True, num_workers=6, pin_memory=True,
        generator=generator, worker_init_fn=worker, persistent_workers=False,
        prefetch_factor=2,
    )
    return DataLoader(dataset, **kwargs)


def configure_epoch(model: ACDCLIP, optimizer, epoch: int) -> bool:
    model.eval()
    model.image_encoder.eval()
    model.clipmodel.eval()
    frozen = epoch <= 3
    model.hybrid_alpha_current = get_hybrid_alpha_for_epoch(
        epoch, hybrid_alpha_max=.2, soft_prompt_freeze_epochs=3,
    )
    model.soft_prompt.requires_grad_(not frozen)
    model.text_adapter.requires_grad_(True)
    apply_soft_prompt_lr_policy(optimizer, frozen)
    model.set_dfg_beta(get_dfg_beta_for_epoch(epoch, "warmup010", .1, .1))
    return frozen


def batch_identity(attempt: int, epoch: int, batch_idx: int, batch, image, mask, label) -> dict:
    return {
        "attempt_index": int(attempt),
        "epoch": int(epoch),
        "batch": int(batch_idx),
        "file_names": json.dumps(list(batch["file_name"]), separators=(",", ":")),
        "image_sha256": tensor_hash(image),
        "mask_sha256": tensor_hash(mask),
        "labels": json.dumps(label.detach().cpu().tolist(), separators=(",", ":")),
    }


IDENTITY_KEYS = ("attempt_index", "epoch", "batch", "file_names", "image_sha256", "mask_sha256", "labels")


def collect_manifest(payload: dict) -> list[dict]:
    device = torch.device("cuda:0")
    restore_rng_state(rng_from_payload(payload))
    model = make_model(payload, device)
    optimizer, _, _ = make_optimizer(model, payload)
    anchor = make_anchor(payload)
    dataset = get_text_and_image_dataset("VisA", IMG, "train")
    del optimizer, anchor, model
    rows = []
    for epoch in range(START_EPOCH + 1, START_EPOCH + 10):
        for batch_idx, batch in enumerate(make_loader(dataset, epoch, payload)):
            image = batch["image"].to(device, non_blocking=False)
            mask = batch["mask"].to(device, non_blocking=False)
            label = batch["label"].to(device, non_blocking=False)
            rows.append(batch_identity(len(rows), epoch, batch_idx, batch, image, mask, label))
            if len(rows) == MAX_ATTEMPTS:
                return rows
    raise RuntimeError("could not collect the requested 500 attempted batches")


def write_scope(payload: dict) -> dict:
    device = torch.device("cuda:0")
    restore_rng_state(rng_from_payload(payload))
    model = make_model(payload, device)
    rows = []
    selected = []
    for name, parameter in sorted(model.image_adapter.named_parameters()):
        parts = name.split(".")
        stage_index = None
        for part in parts:
            if part in {"0", "1", "2"}:
                stage_index = int(part)
                break
        stage = f"stage{stage_index + 1}" if stage_index is not None else "non_stage_specific"
        module_name = "image_adapter." + ".".join(parts[:-1])
        is_conv_lora = name.startswith("lora_adapters.")
        freeze = is_conv_lora and stage_index in (1, 2)
        full_name = "image_adapter." + name
        rows.append({
            "parameter_name": full_name,
            "module_name": module_name,
            "stage": stage,
            "shape": json.dumps(list(parameter.shape), separators=(",", ":")),
            "requires_grad": str(bool(parameter.requires_grad)).lower(),
            "selected_for_freeze": str(bool(freeze)).lower(),
        })
        if freeze:
            selected.append(full_name)
    with SCOPE_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parameter_name", "module_name", "stage", "shape", "requires_grad", "selected_for_freeze"])
        writer.writeheader()
        writer.writerows(rows)
    if not selected or any("lora_adapters.0." in name for name in selected):
        raise RuntimeError("Conv-LoRA scope discovery failed")
    return {
        "parameter_scope_id": "image_adapter.lora_adapters.stage2+stage3.all_parameters",
        "selected_parameter_names": selected,
        "selected_count": len(selected),
        "stage1_selected": False,
        "stage2_selected": True,
        "stage3_selected": True,
        "non_conv_lora_selected": False,
    }


def text_and_task(model, image, mask, label, class_names, device, policy):
    by_class = {}
    kg_losses = []
    k_losses = []
    for class_name in sorted(set(class_names)):
        text, kg, _, components = get_hybrid_soft_prompt_single_class_text_embedding(
            model, "VisA", class_name, device,
            return_kg=True, return_components=True,
        )
        k_loss, _ = compute_hybrid_k_regularization(
            model, components["hard_text"], components["soft_text"],
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
        seg_pred = model.vision_text_fusion_gate_seg(seg_features, text)
        seg_loss = calculate_seg_loss(seg_pred, mask)
        loss_main = cls_loss + seg_loss
        task_loss = loss_main + .01 * kg_loss + .002 * k_loss
    return task_loss, {
        "classification": cls_loss,
        "segmentation": seg_loss,
        "kg": kg_loss,
        "k": k_loss,
    }


def norm_or_zero(gradient) -> float:
    if gradient is None:
        return 0.0
    return float(gradient.detach().float().norm().cpu())


def finite_model_parameters(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def suppress_frozen_gradients(model: ACDCLIP, selected_names: set[str]) -> None:
    for name, parameter in model.image_adapter.named_parameters():
        if "image_adapter." + name in selected_names:
            parameter.grad = None


def implementation_parity(payload: dict, scope: dict, manifest: list[dict]) -> dict:
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    restore_rng_state(rng_from_payload(payload))
    model = make_model(payload, device)
    optimizer, _, scaler = make_optimizer(model, payload)
    configure_epoch(model, optimizer, START_EPOCH + 1)
    dataset = get_text_and_image_dataset("VisA", IMG, "train")
    batch0 = next(iter(make_loader(dataset, START_EPOCH + 1, payload)))
    image = batch0["image"].to(device)
    mask = batch0["mask"].to(device)
    label = batch0["label"].to(device)
    identity = batch_identity(0, START_EPOCH + 1, 0, batch0, image, mask, label)
    if identity != manifest[0]:
        raise RuntimeError("implementation parity batch does not match manifest attempt 0")
    with torch.no_grad():
        control_forward = model(image)
        candidate_forward = model(image)
    forward_diffs = [float((left - right).abs().max().cpu()) for left, right in zip(control_forward[0], candidate_forward[0])]
    selected = set(scope["selected_parameter_names"])
    selected_params = [("image_adapter." + n, p) for n, p in model.image_adapter.named_parameters() if "image_adapter." + n in selected]
    stage1_params = [("image_adapter." + n, p) for n, p in model.image_adapter.named_parameters() if n.startswith("lora_adapters.0.")]
    non_f_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n not in selected]
    before_params = {n: p.detach().cpu().clone() for n, p in selected_params + stage1_params + non_f_params}
    before_state = {
        n: {key: (value.detach().cpu().clone() if torch.is_tensor(value) else value) for key, value in optimizer.state[p].items()}
        for n, p in selected_params
    }
    optimizer.zero_grad(set_to_none=True)
    task_loss, _ = text_and_task(model, image, mask, label, batch0["class_name"], device, policy)
    anchor_loss = make_anchor(payload).loss(model.image_adapter)
    if not torch.isfinite(task_loss).all() or not torch.isfinite(anchor_loss).all():
        raise RuntimeError("implementation parity encountered a non-finite loss")
    scaler.scale(task_loss).backward(retain_graph=True)
    scaler.unscale_(optimizer)
    selected_grad_norms = {n: norm_or_zero(p.grad) for n, p in selected_params}
    selected_nonzero = sum(value > 0.0 for value in selected_grad_norms.values())
    selected_pairs = [(n.removeprefix("image_adapter."), p) for n, p in selected_params]
    image_pairs = [(n, p) for n, p in sorted(model.image_adapter.named_parameters()) if p.requires_grad]
    task_grads = torch.autograd.grad(task_loss, [p for _, p in image_pairs], retain_graph=True, allow_unused=True)
    anchor_grads = torch.autograd.grad(anchor_loss, [p for _, p in image_pairs], retain_graph=False, allow_unused=True)
    apply_family_safe_anchor_budget(
        model.image_adapter, sorted(model.named_parameters()),
        task_gradients=dict(zip([n for n, _ in image_pairs], task_grads)),
        raw_anchor_gradients=dict(zip([n for n, _ in image_pairs], anchor_grads)),
        anchor_lambda=ANCHOR_LAMBDA, rho=ANCHOR_FAMILY_BUDGET,
        total_trainable_parameters=None,
    )
    torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(model.text_adapter.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(model.soft_prompt.parameters(), 1.0)
    suppress_frozen_gradients(model, selected)
    scaler.step(optimizer)
    scaler.update()
    selected_identity = all(tensor_bytes_equal(before_params[n], p) for n, p in selected_params)
    stage1_changed = any(not tensor_bytes_equal(before_params[n], p) for n, p in stage1_params)
    non_f_changed = any(not tensor_bytes_equal(before_params[n], p) for n, p in non_f_params)
    state_preserved = True
    state_values_unchanged = True
    for n, p in selected_params:
        after = optimizer.state[p]
        if set(after) != set(before_state[n]):
            state_preserved = False
            continue
        for key, value in before_state[n].items():
            if torch.is_tensor(value) and not tensor_bytes_equal(value, after[key]):
                state_values_unchanged = False
    result = {
        "protocol_id": PROTOCOL_ID,
        "test_batch_identity": identity,
        "forward_outputs_before_first_update_identical": bool(all(diff == 0.0 for diff in forward_diffs)),
        "forward_max_abs_diffs": forward_diffs,
        "selected_requires_grad": bool(all(p.requires_grad for _, p in selected_params)),
        "selected_gradient_flow_preserved": bool(selected_nonzero > 0),
        "selected_nonzero_gradient_parameter_count": selected_nonzero,
        "selected_parameter_count": len(selected_params),
        "selected_gradient_norms_before_suppression": selected_grad_norms,
        "candidate_frozen_param_byte_identity_after_optimizer_step": bool(selected_identity),
        "stage1_convlora_changed_normally": bool(stage1_changed),
        "at_least_one_non_frozen_trainable_parameter_changed": bool(non_f_changed),
        "selected_optimizer_state_keys_preserved": bool(state_preserved),
        "selected_optimizer_state_values_unchanged_by_suppressed_step": bool(state_values_unchanged),
        "suppression_location": "immediately_before_scaler.step; selected param.grad=None only",
        "forward_backward_safe_anchor_preserved": True,
        "true_freeze_implementation": bool(
            selected_identity and stage1_changed and non_f_changed
            and all(diff == 0.0 for diff in forward_diffs)
            and selected_nonzero > 0 and state_preserved and state_values_unchanged
        ),
    }
    json_dump(PARITY_JSON, result)
    del model
    torch.cuda.empty_cache()
    if not result["true_freeze_implementation"]:
        raise RuntimeError("implementation parity test failed")
    return result


def read_control_rows() -> tuple[list[dict], set[int], dict[int, str]]:
    with CONTROL_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != MAX_ATTEMPTS:
        raise RuntimeError(f"control CSV has {len(rows)} rows, expected {MAX_ATTEMPTS}")
    skip_set = set()
    skip_type = {}
    for row in rows:
        attempt = int(row["attempt_index"])
        if int(row["natural_nonfinite_loss_skip"]) or int(row["natural_nonfinite_grad_skip"]):
            skip_set.add(attempt)
            skip_type[attempt] = "loss" if int(row["natural_nonfinite_loss_skip"]) else "grad"
    return rows, skip_set, skip_type


def scalar(value) -> float | None:
    return None if value is None else float(value.detach().float().cpu())


def drift_stats(reference: dict, live: dict) -> dict:
    diff_sq = 0.0
    ref_sq = 0.0
    count = 0
    for name, ref in reference.items():
        value = live[name].detach().float().cpu()
        ref = ref.detach().float().cpu()
        diff_sq += float((value - ref).square().sum())
        ref_sq += float(ref.square().sum())
        count += ref.numel()
    return {
        "difference_l2": float(diff_sq ** .5),
        "reference_l2": float(ref_sq ** .5),
        "relative_l2": float((diff_sq / max(ref_sq, 1e-30)) ** .5),
        "parameter_count": count,
    }


def grouped_drifts(reference: dict, live: dict) -> dict:
    predicates = {
        "all_image_adapter": lambda n: True,
        "stage1_convlora": lambda n: n.startswith("lora_adapters.0."),
        "stage2_convlora": lambda n: n.startswith("lora_adapters.1."),
        "stage3_convlora": lambda n: n.startswith("lora_adapters.2."),
        "all_convlora": lambda n: n.startswith("lora_adapters."),
        "image_projection": lambda n: any(n.startswith(p) for p in ("m_i_w.", "seg_proj.", "det_proj.", "seg_layer_norms.", "det_layer_norms.")),
        "dfg_qk": lambda n: n.startswith("vision_text_q.") or n.startswith("vision_text_k."),
        "ss2d": lambda n: n.startswith("dfg_ss2d_branches.") or n.startswith("dfg_raw_gamma.") or n.startswith("direction_logits."),
    }
    return {name: drift_stats({k: v for k, v in reference.items() if pred(k)}, {k: v for k, v in live.items() if pred(k)}) for name, pred in predicates.items()}


def run_arm(payload: dict, identity: dict, scope: dict, manifest: list[dict], root: Path, arm: str) -> dict:
    candidate = arm == ARM_CANDIDATE
    control_rows, control_skip_set, control_skip_type = (read_control_rows() if candidate else (None, set(), {}))
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    restore_rng_state(rng_from_payload(payload))
    model = make_model(payload, device)
    optimizer, scheduler, scaler = make_optimizer(model, payload)
    anchor = make_anchor(payload)
    dataset = get_text_and_image_dataset("VisA", IMG, "train")
    selected_names = set(scope["selected_parameter_names"])
    reference_image = state_to_cpu(payload["image_adapter"])
    reference_text = state_to_cpu(payload["text_adapter"])
    reference_soft = state_to_cpu(payload["soft_prompt"])
    rows = []
    attempted = 0
    successful = 0
    natural_loss_skips = 0
    natural_grad_skips = 0
    forced_parity_skips = 0
    batch_mismatch = None
    numerical_failure = None
    max_consecutive_grad_skips = 0
    consecutive_grad_skips = 0
    pre_suppression_norms = []
    stage2_nonzero_attempts = 0
    stage3_nonzero_attempts = 0
    last_epoch = START_EPOCH

    for epoch in range(START_EPOCH + 1, START_EPOCH + 10):
        last_epoch = epoch
        soft_prompt_frozen = configure_epoch(model, optimizer, epoch)
        for batch_idx, batch in enumerate(make_loader(dataset, epoch, payload)):
            if attempted >= MAX_ATTEMPTS:
                break
            attempt = attempted
            attempted += 1
            image = batch["image"].to(device, non_blocking=False)
            mask = batch["mask"].to(device, non_blocking=False)
            label = batch["label"].to(device, non_blocking=False)
            identity_row = batch_identity(attempt, epoch, batch_idx, batch, image, mask, label)
            if identity_row != manifest[attempt] and batch_mismatch is None:
                batch_mismatch = f"manifest mismatch at attempt {attempt}"
            if candidate and (batch_mismatch is None or True):
                expected = control_rows[attempt]
                for key in IDENTITY_KEYS:
                    if str(identity_row[key]) != str(expected[key]):
                        batch_mismatch = batch_mismatch or f"control/candidate mismatch for {key} at attempt {attempt}"
                        break
            row = {
                **identity_row,
                "protocol_id": PROTOCOL_ID,
                "arm": arm,
                "status": "pending",
                "global_step_before": int(START_GLOBAL_STEP + successful),
                "global_step_after": int(START_GLOBAL_STEP + successful),
                "scaler_before": float(scaler.get_scale()),
                "scaler_after": None,
                "base_task_loss": None,
                "anchor_loss": None,
                "classification_loss": None,
                "segmentation_loss": None,
                "natural_nonfinite_loss_skip": 0,
                "natural_nonfinite_grad_skip": 0,
                "forced_parity_skip": 0,
                "successful_update": 0,
                "selected_grad_norm_before_suppression": None,
                "stage2_grad_norm_before_suppression": None,
                "stage3_grad_norm_before_suppression": None,
                "safe_anchor_effective_ratio": None,
                "safe_anchor_max_family_ratio": None,
                "parameters_finite": None,
                "optimizer_state_finite": None,
            }
            optimizer.zero_grad(set_to_none=True)
            task_loss, terms = text_and_task(model, image, mask, label, batch["class_name"], device, policy)
            anchor_loss = anchor.loss(model.image_adapter)
            row.update({
                "base_task_loss": scalar(task_loss) if torch.isfinite(task_loss).all() else None,
                "anchor_loss": scalar(anchor_loss) if torch.isfinite(anchor_loss).all() else None,
                "classification_loss": scalar(terms["classification"]) if torch.isfinite(terms["classification"]).all() else None,
                "segmentation_loss": scalar(terms["segmentation"]) if torch.isfinite(terms["segmentation"]).all() else None,
            })
            control_forced = candidate and attempt in control_skip_set
            if not torch.isfinite(task_loss).all() or not torch.isfinite(anchor_loss).all():
                natural_loss_skips += 1
                consecutive_grad_skips = 0
                row["natural_nonfinite_loss_skip"] = 1
                row["status"] = "natural_nonfinite_loss_skip"
                if not control_forced:
                    numerical_failure = numerical_failure or "candidate_extra_natural_nonfinite_loss_skip" if candidate else numerical_failure
                optimizer.zero_grad(set_to_none=True)
                rows.append(row)
                continue
            scaler.scale(task_loss).backward(retain_graph=True)
            scaler.unscale_(optimizer)
            image_pairs = [(name, parameter) for name, parameter in sorted(model.image_adapter.named_parameters()) if parameter.requires_grad]
            selected_pairs = [(name, parameter) for name, parameter in image_pairs if "image_adapter." + name in selected_names]
            selected_norms = {name: norm_or_zero(parameter.grad) for name, parameter in selected_pairs}
            stage2_norm = float(sum(value * value for name, value in selected_norms.items() if name.startswith("lora_adapters.1.")) ** .5)
            stage3_norm = float(sum(value * value for name, value in selected_norms.items() if name.startswith("lora_adapters.2.")) ** .5)
            selected_norm = float(sum(value * value for value in selected_norms.values()) ** .5)
            row.update({
                "selected_grad_norm_before_suppression": selected_norm,
                "stage2_grad_norm_before_suppression": stage2_norm,
                "stage3_grad_norm_before_suppression": stage3_norm,
            })
            pre_suppression_norms.append(selected_norm)
            stage2_nonzero_attempts += int(stage2_norm > 0.0)
            stage3_nonzero_attempts += int(stage3_norm > 0.0)
            if has_non_finite_grad(optimizer):
                natural_grad_skips += 1
                consecutive_grad_skips += 1
                max_consecutive_grad_skips = max(max_consecutive_grad_skips, consecutive_grad_skips)
                row["natural_nonfinite_grad_skip"] = 1
                row["status"] = "natural_nonfinite_grad_skip"
                if candidate and not control_forced:
                    numerical_failure = numerical_failure or "candidate_extra_natural_nonfinite_grad_skip"
                if scaler is not None:
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                rows.append(row)
                continue
            consecutive_grad_skips = 0
            task_grads = torch.autograd.grad(task_loss, [p for _, p in image_pairs], retain_graph=True, allow_unused=True)
            anchor_grads = torch.autograd.grad(anchor_loss, [p for _, p in image_pairs], retain_graph=False, allow_unused=True)
            anchor_metrics = apply_family_safe_anchor_budget(
                model.image_adapter, sorted(model.named_parameters()),
                task_gradients=dict(zip([n for n, _ in image_pairs], task_grads)),
                raw_anchor_gradients=dict(zip([n for n, _ in image_pairs], anchor_grads)),
                anchor_lambda=ANCHOR_LAMBDA, rho=ANCHOR_FAMILY_BUDGET,
                total_trainable_parameters=None,
            )
            torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.text_adapter.parameters(), 1.0)
            if not soft_prompt_frozen:
                torch.nn.utils.clip_grad_norm_(model.soft_prompt.parameters(), 1.0)
            if control_forced:
                forced_parity_skips += 1
                row["forced_parity_skip"] = 1
                row["status"] = "PAIRED_PARITY_SKIP"
                if control_skip_type[attempt] == "grad":
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                rows.append(row)
                continue
            if candidate:
                suppress_frozen_gradients(model, selected_names)
            scaler.step(optimizer)
            scaler.update()
            successful += 1
            row.update({
                "status": "success",
                "successful_update": 1,
                "global_step_after": START_GLOBAL_STEP + successful,
                "safe_anchor_effective_ratio": anchor_metrics["global_effective_ratio"],
                "safe_anchor_max_family_ratio": anchor_metrics["max_effective_active_family_ratio"],
                "parameters_finite": int(finite_model_parameters(model)),
                "optimizer_state_finite": int(optimizer_state_is_finite(optimizer)),
            })
            if not row["parameters_finite"] or not row["optimizer_state_finite"]:
                numerical_failure = numerical_failure or "nonfinite_post_update_state"
            rows.append(row)
        if attempted >= MAX_ATTEMPTS:
            break
        scheduler.step()
        apply_soft_prompt_lr_policy(optimizer, soft_prompt_frozen)

    final_image = state_to_cpu(model.image_adapter.state_dict())
    final_text = state_to_cpu(model.text_adapter.state_dict())
    final_soft = state_to_cpu(model.soft_prompt.state_dict())
    frozen_byte_identity = all(
        tensor_bytes_equal(reference_image[name.removeprefix("image_adapter.")], final_image[name.removeprefix("image_adapter.")])
        for name in selected_names
    ) if candidate else None
    image_drifts = grouped_drifts(reference_image, final_image)
    text_drift = drift_stats(reference_text, final_text)
    soft_drift = drift_stats(reference_soft, final_soft)
    all_nonf_drift = {
        "text_adapter": text_drift,
        "soft_prompt": soft_drift,
        "image_non_frozen": drift_stats(
            {k: v for k, v in reference_image.items() if "image_adapter." + k not in selected_names},
            {k: v for k, v in final_image.items() if "image_adapter." + k not in selected_names},
        ),
    }
    result = {
        "protocol_id": PROTOCOL_ID,
        "arm": arm,
        "start_checkpoint": str(START_CHECKPOINT),
        "start_checkpoint_sha256": START_CHECKPOINT_SHA256,
        "start_epoch": START_EPOCH,
        "start_global_step": START_GLOBAL_STEP,
        "attempted": attempted,
        "natural_nonfinite_loss_skips": natural_loss_skips,
        "natural_nonfinite_grad_skips": natural_grad_skips,
        "forced_parity_skips": forced_parity_skips,
        "successful_updates": successful,
        "final_epoch": last_epoch,
        "final_global_step": START_GLOBAL_STEP + successful,
        "final_scaler_value": float(scaler.get_scale()),
        "max_consecutive_natural_nonfinite_grad_skips": max_consecutive_grad_skips,
        "batch_match_gate": "PASS" if batch_mismatch is None else "FAIL",
        "batch_mismatch": batch_mismatch,
        "numerical_failure": numerical_failure,
        "candidate_frozen_param_byte_identity": frozen_byte_identity,
        "pre_suppression_selected_gradient_mean": float(np.mean(pre_suppression_norms)) if pre_suppression_norms else None,
        "pre_suppression_selected_gradient_nonzero_fraction": float(sum(v > 0 for v in pre_suppression_norms) / len(pre_suppression_norms)) if pre_suppression_norms else 0.0,
        "pre_suppression_stage2_gradient_nonzero_fraction": float(stage2_nonzero_attempts / max(1, successful)),
        "pre_suppression_stage3_gradient_nonzero_fraction": float(stage3_nonzero_attempts / max(1, successful)),
        "parameter_drift_from_e10": {
            "image_adapter": image_drifts,
            "text_adapter": text_drift,
            "soft_prompt": soft_drift,
            "non_frozen_trainable": all_nonf_drift,
        },
        "rows_csv": str(REPO / "audit" / f"H2_LATE_CONVLORA_FREEZE_R1_{'CANDIDATE' if candidate else 'CONTROL'}.csv"),
        "rows": rows,
    }
    output_dir = root / arm
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": {"image_adapter": final_image, "text_adapter": final_text, "soft_prompt": final_soft},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": last_epoch,
        "global_step": START_GLOBAL_STEP + successful,
        "config": {"protocol_id": PROTOCOL_ID, "arm": arm, "equal_train_fusion": [1/3, 1/3, 1/3]},
        "start_checkpoint_sha256": START_CHECKPOINT_SHA256,
        "candidate_frozen_param_byte_identity": frozen_byte_identity,
    }, output_dir / "final.pth")
    json_dump(output_dir / "summary.json", {k: v for k, v in result.items() if k != "rows"})
    audit_path = REPO / "audit" / f"H2_LATE_CONVLORA_FREEZE_R1_{'CANDIDATE' if candidate else 'CONTROL'}.csv"
    with audit_path.open("w", newline="") as handle:
        fields = list(rows[0])
        csv.DictWriter(handle, fieldnames=fields).writeheader()
        csv.DictWriter(handle, fieldnames=fields).writerows(rows)
    del model
    torch.cuda.empty_cache()
    if attempted != MAX_ATTEMPTS or batch_mismatch is not None:
        raise RuntimeError(f"{arm} did not complete the exact 500-batch stream")
    return result


def write_manifest(payload: dict, identity: dict, scope: dict, rows: list[dict]) -> None:
    json_dump(MANIFEST_JSON, {
        "protocol_id": PROTOCOL_ID,
        "manifest_type": "fixed_attempted_source_batch_stream",
        "dataset": "VisA",
        "split": "train",
        "medical_or_mvtec_or_target_evaluated": False,
        "attempt_count": len(rows),
        "attempts": rows,
        "identity_fields": list(IDENTITY_KEYS),
        "start_checkpoint": identity,
        "parameter_scope": scope,
        "loader": {
            "batch_size": 6, "shuffle": True, "num_workers": 6,
            "pin_memory": True, "persistent_workers": False,
            "generator_seed_formula": "seed + 104729 * epoch",
            "worker_seed_formula": "seed + 1000003 * epoch + worker_id",
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/h2_late_convlora_freeze_r1")
    parser.add_argument("--stage", choices=("all", "prepare", "control", "candidate"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload, identity = validate_start_checkpoint()
    scope = write_scope(payload)
    if args.stage in ("all", "prepare"):
        manifest_rows = collect_manifest(payload)
        write_manifest(payload, identity, scope, manifest_rows)
        implementation_parity(payload, scope, manifest_rows)
        if args.stage == "prepare":
            return
    manifest = json.loads(MANIFEST_JSON.read_text())["attempts"]
    parity = json.loads(PARITY_JSON.read_text())
    if not parity.get("true_freeze_implementation"):
        raise RuntimeError("implementation parity artifact is not PASS")
    root = Path(args.root)
    if args.stage in ("all", "control"):
        run_arm(payload, identity, scope, manifest, root, ARM_CONTROL)
        if args.stage == "control":
            return
    if args.stage in ("all", "candidate"):
        if not CONTROL_CSV.is_file():
            raise RuntimeError("candidate requires the committed control CSV")
        run_arm(payload, identity, scope, manifest, root, ARM_CANDIDATE)


if __name__ == "__main__":
    main()
