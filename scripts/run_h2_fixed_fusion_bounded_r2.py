#!/usr/bin/env python3
"""Run the preregistered H2 fixed stage-fusion R2 screen.

The two arms share the exact E1 state, dataloader stream, optimizer state,
precision policy, and Safe Anchor contract.  The candidate changes only the
parameter-free pre-softmax aggregation of the three native segmentation-logit
stages.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

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
)
from h2_clean.precision import PrecisionPolicy
from h2_clean.stage_fusion import (
    H2_EQUAL_STAGE_FUSION_WEIGHTS,
    H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS,
)
from model.adapter import ACDCLIP
from model.clip import create_model
from train import (
    apply_soft_prompt_lr_policy,
    calculate_seg_loss,
    get_dfg_beta_for_epoch,
    get_hybrid_alpha_for_epoch,
    has_non_finite_grad,
    optimizer_state_is_finite,
)
from utils import get_hybrid_soft_prompt_single_class_text_embedding


IMG = 518
SEED = 0
MAX_ATTEMPTS = 500
ANCHOR_LAMBDA = 0.0021633926715180626
ANCHOR_FAMILY_BUDGET = 0.10
ARM_CONTROL = "A_FUSE_SHORT_R2_CONTROL"
ARM_CANDIDATE = "A_FUSE_SHORT_R2_CANDIDATE"
START_CHECKPOINT = REPO / "runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth"
CONTROL_CSV = REPO / "audit/H2_FUSION_R2_CONTROL.csv"


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(staging, path)


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def rng_from_payload(payload: dict) -> dict:
    return {
        "python_random_state": payload["python_random_state"],
        "numpy_random_state": payload["numpy_random_state"],
        "torch_cpu_rng_state": payload["torch_cpu_rng_state"],
        "torch_cuda_rng_state_all": payload["torch_cuda_rng_state_all"],
    }


def restore_rng_state(state: dict) -> None:
    import random

    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    if torch.cuda.is_available() and state["torch_cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])


def make_model(
    payload: dict,
    device: torch.device,
    stage_fusion_weights: tuple[float, float, float],
) -> ACDCLIP:
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
        dfg_beta_current=.1, dfg_weight_residual_fp32=True,
        stage_fusion_weights=stage_fusion_weights,
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
    model.prompt_mode = "hybrid"
    model.use_hybrid_soft_prompt = True
    model.use_soft_prompt = False
    model.hybrid_alpha_current = 0.0
    model.hybrid_alpha_max = .2
    model.soft_prompt_freeze_epochs = 3
    model.requires_grad_(False)
    model.image_adapter.requires_grad_(True)
    model.text_adapter.requires_grad_(True)
    model.soft_prompt.requires_grad_(False)
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


def loader_for_epoch(dataset, epoch: int):
    generator = make_dataloader_generator(SEED)
    generator.manual_seed(SEED + 104729 * epoch)
    worker = EpochWorkerInit(SEED)
    worker.set_epoch(epoch)
    return DataLoader(
        dataset, batch_size=6, shuffle=True, num_workers=0, pin_memory=True,
        generator=generator, worker_init_fn=worker,
    )


def configure_epoch(model, optimizer, epoch: int) -> bool:
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


def text_and_task(model, image, mask, label, class_names, device, policy):
    by_class = {}
    kg_losses = []
    k_losses = []
    from train import compute_hybrid_k_regularization

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
        seg = torch.stack(seg_tokens)
        det = torch.stack(det_tokens)
        cls = torch.stack([
            torch.matmul(det[i].unsqueeze(1), text[i]).squeeze(1)
            for i in range(3)
        ]).mean(0)
        cls_loss = F.cross_entropy(cls, label)
        pred = model.vision_text_fusion_gate_seg(seg, text)
        seg_loss = calculate_seg_loss(pred, mask)
        base = cls_loss + seg_loss + .01 * kg_loss + .002 * k_loss
    return base, {"classification": cls_loss, "segmentation": seg_loss}


def batch_identity(attempt_index: int, epoch: int, batch_idx: int, batch, image, mask, label) -> dict:
    return {
        "attempt_index": int(attempt_index),
        "epoch": int(epoch),
        "batch": int(batch_idx),
        "file_names": json.dumps(list(batch["file_name"]), separators=(",", ":")),
        "image_sha256": tensor_hash(image),
        "mask_sha256": tensor_hash(mask),
        "labels": json.dumps(label.detach().cpu().tolist(), separators=(",", ":")),
    }


def collect_preflight_batches(payload: dict, weights) -> list[dict]:
    device = torch.device("cuda:0")
    dataset = get_text_and_image_dataset("VisA", IMG, "train")
    restore_rng_state(rng_from_payload(payload))
    model = make_model(payload, device, weights)
    rows = []
    with torch.no_grad():
        for epoch in (2, 3):
            for batch_idx, batch in enumerate(loader_for_epoch(dataset, epoch)):
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                label = batch["label"].to(device)
                rows.append(batch_identity(len(rows), epoch, batch_idx, batch, image, mask, label))
                if len(rows) == 16:
                    return rows
    raise RuntimeError("preflight did not collect 16 batches")


def preflight(payload: dict) -> None:
    control = collect_preflight_batches(payload, H2_EQUAL_STAGE_FUSION_WEIGHTS)
    candidate = collect_preflight_batches(payload, H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS)
    passed = control == candidate
    result = {
        "protocol_id": "H2_FIXED_E1_RELIABILITY_RESIDUAL_STAGE_FUSION_R2",
        "PREFLIGHT_BATCH_PARITY": "PASS" if passed else "FAIL",
        "batch_count": 16,
        "control": control,
        "candidate": candidate,
        "identity_fields": ["file_names", "labels", "image_sha256", "mask_sha256"],
        "control_weights": list(H2_EQUAL_STAGE_FUSION_WEIGHTS),
        "candidate_weights": list(H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS),
    }
    json_dump(REPO / "audit/H2_FUSION_R2_PREFLIGHT.json", result)
    if not passed:
        raise RuntimeError("PREFLIGHT_BATCH_PARITY=FAIL")


def read_control_identities() -> list[dict]:
    with CONTROL_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"attempt_index", "epoch", "batch", "file_names", "image_sha256", "mask_sha256", "labels"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("control CSV is missing the required batch identity columns")
    return rows


def finite_model_parameters(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def run_arm(payload: dict, root: Path, arm: str) -> dict:
    if arm not in (ARM_CONTROL, ARM_CANDIDATE):
        raise ValueError(arm)
    candidate = arm == ARM_CANDIDATE
    weights = (
        H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS
        if candidate else H2_EQUAL_STAGE_FUSION_WEIGHTS
    )
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    restore_rng_state(rng_from_payload(payload))
    model = make_model(payload, device, weights)
    optimizer, scheduler, scaler = make_optimizer(model, payload)
    anchor = SafeImageAdapterAnchor.from_checkpoint(START_CHECKPOINT, device)
    dataset = get_text_and_image_dataset("VisA", IMG, "train")
    expected = read_control_identities() if candidate else None

    rows = []
    attempted = 0
    successful = 0
    nonfinite_loss_skips = 0
    nonfinite_grad_skips = 0
    consecutive_grad_skips = 0
    max_consecutive_grad_skips = 0
    parameter_corruption = 0
    optimizer_state_failures = 0
    batch_mismatch = None
    numerical_failure = None

    config = {
        "protocol_id": "H2_FIXED_E1_RELIABILITY_RESIDUAL_STAGE_FUSION_R2",
        "arm": arm,
        "dataset": "VisA",
        "seed": SEED,
        "image_size": IMG,
        "batch_size": 6,
        "precision_protocol": "HISTORICAL_MIXED_FP16_FP32_V1",
        "fp16_autocast": True,
        "grad_scaler": True,
        "bf16": False,
        "tf32": False,
        "safe_anchor": True,
        "anchor_lambda": ANCHOR_LAMBDA,
        "anchor_family_budget_rho": ANCHOR_FAMILY_BUDGET,
        "stage_fusion_weights": list(weights),
        "stage_fusion_position": "pre_softmax_interpolated_stage_logits",
        "stage_order": ["stage1", "stage2", "stage3"],
        "shared_e1_sha256": sha256_file(START_CHECKPOINT),
        "max_attempts": MAX_ATTEMPTS,
        "functional_anchor": False,
        "medical_or_mvtec": False,
    }

    for epoch in (2, 3, 4, 5, 6, 7, 8):
        frozen = configure_epoch(model, optimizer, epoch)
        for batch_idx, batch in enumerate(loader_for_epoch(dataset, epoch)):
            if attempted >= MAX_ATTEMPTS:
                break
            attempt_index = attempted
            attempted += 1
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)
            identity = batch_identity(attempt_index, epoch, batch_idx, batch, image, mask, label)
            if candidate:
                if attempt_index >= len(expected):
                    batch_mismatch = f"candidate has no control row for attempt {attempt_index}"
                else:
                    for key in ("attempt_index", "epoch", "batch", "file_names", "image_sha256", "mask_sha256", "labels"):
                        if str(identity[key]) != str(expected[attempt_index][key]):
                            batch_mismatch = f"{key} mismatch at attempt {attempt_index}"
                            break
                if batch_mismatch is not None:
                    numerical_failure = "BATCH_MATCH_GATE=FAIL"
                    rows.append({**identity, "status": "batch_mismatch"})
                    break

            optimizer.zero_grad(set_to_none=True)
            base, terms = text_and_task(model, image, mask, label, batch["class_name"], device, policy)
            anchor_loss = anchor.loss(model.image_adapter)
            row = {
                **identity,
                "arm": arm,
                "status": "pending",
                "base_task_loss": float(base.detach().float().cpu()) if torch.isfinite(base).item() else None,
                "anchor_loss": float(anchor_loss.detach().float().cpu()) if torch.isfinite(anchor_loss).item() else None,
                "classification_loss": float(terms["classification"].detach().float().cpu()) if torch.isfinite(terms["classification"]).item() else None,
                "segmentation_loss": float(terms["segmentation"].detach().float().cpu()) if torch.isfinite(terms["segmentation"]).item() else None,
                "successful_step": 0,
                "nonfinite_loss_skip": 0,
                "nonfinite_grad_skip": 0,
                "consecutive_nonfinite_grad_skips": 0,
                "optimizer_state_finite": None,
                "parameters_finite": None,
                "safe_anchor_effective_ratio": None,
                "safe_anchor_max_family_ratio": None,
            }
            if not torch.isfinite(base).all() or not torch.isfinite(anchor_loss).all():
                nonfinite_loss_skips += 1
                consecutive_grad_skips = 0
                row.update(status="nonfinite_loss_skip", nonfinite_loss_skip=1)
                optimizer.zero_grad(set_to_none=True)
                rows.append(row)
                continue

            scaler.scale(base).backward(retain_graph=True)
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
                if numerical_failure is None and consecutive_grad_skips > 1:
                    numerical_failure = "consecutive_nonfinite_gradient_skips"
                continue

            consecutive_grad_skips = 0
            pairs = [
                (name, parameter)
                for name, parameter in sorted(model.image_adapter.named_parameters())
                if parameter.requires_grad
            ]
            names = [name for name, _ in pairs]
            parameters = [parameter for _, parameter in pairs]
            task_grads = torch.autograd.grad(base, parameters, retain_graph=True, allow_unused=True)
            anchor_grads = torch.autograd.grad(anchor_loss, parameters, allow_unused=True)
            anchor_metrics = apply_family_safe_anchor_budget(
                model.image_adapter,
                sorted(model.named_parameters()),
                task_gradients=dict(zip(names, task_grads)),
                raw_anchor_gradients=dict(zip(names, anchor_grads)),
                anchor_lambda=ANCHOR_LAMBDA,
                rho=ANCHOR_FAMILY_BUDGET,
                total_trainable_parameters=None,
            )
            torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.text_adapter.parameters(), 1.0)
            if not frozen:
                torch.nn.utils.clip_grad_norm_(model.soft_prompt.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            successful += 1
            state_finite = optimizer_state_is_finite(optimizer)
            params_finite = finite_model_parameters(model)
            if not state_finite:
                optimizer_state_failures += 1
                numerical_failure = numerical_failure or "nonfinite_optimizer_state"
            if not params_finite:
                parameter_corruption += 1
                numerical_failure = numerical_failure or "nonfinite_parameter_corruption"
            row.update(
                status="success" if state_finite and params_finite else "numerical_failure",
                successful_step=1,
                optimizer_state_finite=int(state_finite),
                parameters_finite=int(params_finite),
                safe_anchor_effective_ratio=anchor_metrics["global_effective_ratio"],
                safe_anchor_max_family_ratio=anchor_metrics["max_effective_active_family_ratio"],
            )
            rows.append(row)
        if attempted >= MAX_ATTEMPTS or batch_mismatch is not None:
            break
        scheduler.step()
        apply_soft_prompt_lr_policy(optimizer, epoch <= 3)

    output_dir = root / arm
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "arm": arm,
        "config": config,
        "attempted_steps": attempted,
        "successful_steps": successful,
        "nonfinite_loss_skips": nonfinite_loss_skips,
        "nonfinite_grad_skips": nonfinite_grad_skips,
        "max_consecutive_nonfinite_grad_skips": max_consecutive_grad_skips,
        "optimizer_state_failures": optimizer_state_failures,
        "parameter_corruption": parameter_corruption,
        "batch_match_gate": "FAIL" if batch_mismatch else "PASS",
        "batch_mismatch": batch_mismatch,
        "numerical_failure": numerical_failure,
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
    with (REPO / ("audit/H2_FUSION_R2_CONTROL.csv" if not candidate else "audit/H2_FUSION_R2_CANDIDATE.csv")).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_dump(output_dir / "summary.json", result)
    if attempted != MAX_ATTEMPTS:
        raise RuntimeError(f"expected exactly {MAX_ATTEMPTS} attempted steps, got {attempted}")
    if batch_mismatch:
        raise RuntimeError("BATCH_MATCH_GATE=FAIL")
    print(json.dumps({
        "status": "PASS",
        "arm": arm,
        "attempted": attempted,
        "successful": successful,
        "nonfinite_loss_skips": nonfinite_loss_skips,
        "nonfinite_grad_skips": nonfinite_grad_skips,
    }, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--arm", choices=(ARM_CONTROL, ARM_CANDIDATE))
    parser.add_argument("--shared-e1", default=str(START_CHECKPOINT))
    parser.add_argument("--root", default="/workspace/h2_fixed_fusion_bounded_r2")
    args = parser.parse_args()
    if args.shared_e1 != str(START_CHECKPOINT):
        raise ValueError("R2 requires the preregistered shared E1 checkpoint")
    if not START_CHECKPOINT.is_file():
        raise FileNotFoundError(START_CHECKPOINT)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    payload = torch.load(START_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 1 or payload.get("precision") not in ("amp", "fp16"):
        raise RuntimeError("shared E1 identity mismatch")
    if args.mode == "preflight":
        preflight(payload)
        return
    if args.arm is None:
        raise ValueError("--arm is required for --mode run")
    run_arm(payload, Path(args.root), args.arm)


if __name__ == "__main__":
    main()
