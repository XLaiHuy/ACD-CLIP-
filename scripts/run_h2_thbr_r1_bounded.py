#!/usr/bin/env python3
"""Run the single preregistered H2 THBR-R1 bounded source screen.

The control and candidate arms both resume the exact E10 Safe-Anchor full
state.  The candidate changes only the task loss by adding the calibrated
matched hard-background/weak-anomaly ranking loss.  This script never loads
Medical or MVTec and never performs target inference.
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
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
os.sys.path.insert(0, str(REPO))

import scripts.run_h2_thbr_r1_audit as audit
from h2_clean.contract import (
    EpochWorkerInit,
    SafeImageAdapterAnchor,
    apply_family_safe_anchor_budget,
    build_full_checkpoint,
    current_git_sha,
    make_dataloader_generator,
    sha256_file,
)
from h2_clean.precision import PrecisionPolicy
from model.adapter import ACDCLIP
from train import (
    apply_soft_prompt_lr_policy,
    has_non_finite_grad,
    get_dfg_beta_for_epoch,
    get_hybrid_alpha_for_epoch,
    optimizer_state_is_finite,
)
from utils import BinaryDiceLoss, FocalLoss


IMG = audit.IMG
SEED = audit.SEED
START_EPOCH = audit.START_EPOCH
START_GLOBAL_STEP = audit.START_GLOBAL_STEP
MAX_ATTEMPTS = audit.MAX_ATTEMPTS
ANCHOR_LAMBDA = audit.ANCHOR_LAMBDA
ANCHOR_RHO = audit.ANCHOR_RHO
THBR_LAMBDA = 0.18723131689337608
THBR_GRAD_AUDIT_INTERVAL = 25
PARITY_BATCHES = 16
PROTOCOL_ID = audit.PROTOCOL_ID
ARM_CONTROL = "A_THBR_R1_CONTROL"
ARM_CANDIDATE = "A_THBR_R1_CANDIDATE"
DEFAULT_ROOT = REPO / "runs/h2_thbr_r1"
CONTROL_CSV = REPO / "audit/H2_THBR_R1_CONTROL.csv"
CANDIDATE_CSV = REPO / "audit/H2_THBR_R1_CANDIDATE.csv"
LOSS_PARITY_JSON = REPO / "audit/H2_THBR_R1_LOSS_PARITY.json"
MANIFEST_JSON = audit.MANIFEST_JSON


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def scalar(value: torch.Tensor | float | None) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if not torch.isfinite(value).all().item():
            return None
        return float(value.detach().float().cpu().item())
    value = float(value)
    return value if np.isfinite(value) else None


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def flatten_gradients(loss: torch.Tensor, parameters: list[torch.Tensor]) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=False, allow_unused=True)
    return torch.cat([
        gradient.float().reshape(-1) if gradient is not None
        else torch.zeros_like(parameter, dtype=torch.float32).reshape(-1)
        for gradient, parameter in zip(gradients, parameters)
    ])


def make_optimizer(model: ACDCLIP, payload: dict):
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


def make_training_loader(dataset, generator, worker):
    return DataLoader(
        dataset,
        batch_size=6,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        worker_init_fn=worker,
        generator=generator,
        persistent_workers=False,
        prefetch_factor=2,
    )


def load_manifest() -> list[dict]:
    payload = json.loads(MANIFEST_JSON.read_text())
    if payload.get("attempt_count") != MAX_ATTEMPTS or len(payload.get("attempts", [])) != MAX_ATTEMPTS:
        raise RuntimeError("THBR manifest must contain exactly 500 attempts")
    return payload["attempts"]


def validate_audit_and_start() -> tuple[dict, list[dict]]:
    payload, _ = audit.validate_start()
    decision = json.loads(audit.AUDIT_DECISION_JSON.read_text())
    if decision.get("thbr_training_authorized") is not True:
        raise RuntimeError("Part-A audit did not authorize THBR training")
    calibration = json.loads(audit.CALIBRATION_JSON.read_text())
    if calibration.get("lambda_thbr") != THBR_LAMBDA or not calibration.get("finite_and_stable"):
        raise RuntimeError("calibrated THBR lambda mismatch or invalid calibration")
    manifest = load_manifest()
    return payload, manifest


def configure_epoch(model: ACDCLIP, optimizer, epoch: int) -> None:
    model.eval()
    model.image_encoder.eval()
    model.clipmodel.eval()
    model.hybrid_alpha_current = get_hybrid_alpha_for_epoch(epoch, .2, 3)
    model.soft_prompt.requires_grad_(True)
    model.text_adapter.requires_grad_(True)
    model.image_adapter.requires_grad_(True)
    apply_soft_prompt_lr_policy(optimizer, frozen=False)
    model.set_dfg_beta(get_dfg_beta_for_epoch(epoch, "warmup010", .1, .1))


def thbr_telemetry(score: torch.Tensor, mask: torch.Tensor) -> dict:
    """Return the exact K-matched diagnostics used by the candidate loss."""
    k_values = []
    hard_means = []
    hard_maxima = []
    weak_means = []
    weak_minima = []
    violations = []
    anomalous_samples = 0
    for sample_score, sample_mask in zip(score.detach().float(), mask[:, 0].detach().float()):
        pairs = audit.matched_pairs(sample_score, sample_mask)
        if pairs is None:
            continue
        anomalous_samples += 1
        hard, weak = pairs
        k_values.append(int(hard.numel()))
        hard_means.append(float(hard.mean().item()))
        hard_maxima.append(float(hard.max().item()))
        weak_means.append(float(weak.mean().item()))
        weak_minima.append(float(weak.min().item()))
        violations.append(float((hard > weak).float().mean().item()))
    def mean_or_none(values):
        return float(np.mean(values)) if values else None
    return {
        "anomaly_sample_count": anomalous_samples,
        "K_per_anomalous_sample": k_values,
        "hard_background_mean": mean_or_none(hard_means),
        "hard_background_max": float(np.max(hard_maxima)) if hard_maxima else None,
        "weak_anomaly_mean": mean_or_none(weak_means),
        "weak_anomaly_min": float(np.min(weak_minima)) if weak_minima else None,
        "violation_fraction": mean_or_none(violations),
    }


def identity_matches(identity: dict, expected: dict) -> bool:
    return all(str(identity.get(key)) == str(expected.get(key)) for key in audit.IDENTITY_KEYS)


def read_control_rows() -> list[dict]:
    if not CONTROL_CSV.is_file():
        raise RuntimeError("control arm must be completed before the candidate arm")
    with CONTROL_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != MAX_ATTEMPTS:
        raise RuntimeError("control CSV does not contain exactly 500 attempted rows")
    return rows


def loss_parity(payload: dict, manifest: list[dict]) -> dict:
    """Prove control and candidate-disabled paths are the same on fixed batches."""
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    focal = FocalLoss()
    dice = BinaryDiceLoss()
    anchor = SafeImageAdapterAnchor.from_checkpoint(audit.E10, device)

    def collect_side() -> list[dict]:
        audit.restore_rng_state(audit.rng_from_payload(payload))
        model = audit.make_model(payload, device, training_graph=True)
        model.hybrid_alpha_current = get_hybrid_alpha_for_epoch(START_EPOCH + 1, .2, 3)
        model.dfg_beta = get_dfg_beta_for_epoch(START_EPOCH + 1, "warmup010", .1, .1)
        parameters = [p for p in model.parameters() if p.requires_grad]
        side_rows = []
        for attempt in range(PARITY_BATCHES):
            batch, image, mask, label = audit.load_training_batch(payload, manifest, attempt)
            losses = audit.build_training_losses(model, image, mask, label, batch["class_name"], device, policy, focal, dice, include_thbr=False)
            total = losses["existing_task"] + THBR_LAMBDA * losses["thbr"]
            values = {key: float(losses[key].detach().float().cpu()) for key in ("cls", "focal", "normal_dice", "abnormal_dice", "existing_task", "thbr")}
            values["total_loss"] = float(total.detach().float().cpu())
            gradient = flatten_gradients(total, parameters)
            side_rows.append({
                **{key: manifest[attempt][key] for key in audit.IDENTITY_KEYS},
                "attempt_index": attempt,
                "loss_values": values,
                "gradient_sha256": tensor_hash(gradient),
                "gradient_norm": float(gradient.norm().cpu()),
                "anchor_loss": scalar(anchor.loss(model.image_adapter)),
            })
            del losses, total, gradient
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        return side_rows

    control_rows = collect_side()
    candidate_rows = collect_side()
    rows = []
    for control, candidate in zip(control_rows, candidate_rows):
        loss_diffs = {
            key: abs(control["loss_values"][key] - candidate["loss_values"][key])
            for key in control["loss_values"]
        }
        gradient_hash_equal = control["gradient_sha256"] == candidate["gradient_sha256"]
        row = {
            **{key: manifest[control["attempt_index"]][key] for key in audit.IDENTITY_KEYS},
            "attempt_index": control["attempt_index"],
            "loss_abs_differences": loss_diffs,
            "control_gradient_sha256": control["gradient_sha256"],
            "candidate_disabled_gradient_sha256": candidate["gradient_sha256"],
            "gradient_hash_equal": gradient_hash_equal,
            "control_gradient_norm": control["gradient_norm"],
            "candidate_disabled_gradient_norm": candidate["gradient_norm"],
            "control_anchor_loss": control["anchor_loss"],
            "candidate_anchor_loss": candidate["anchor_loss"],
            "exact_disabled_path": bool(max(loss_diffs.values()) == 0.0 and gradient_hash_equal),
        }
        rows.append(row)
    result = {
        "protocol_id": PROTOCOL_ID,
        "scope": "fixed first 16 source batches from the 500-attempt manifest; no optimizer update",
        "control_objective": "L_existing_task with exact E10 Safe-Anchor gradient-budget postprocessing",
        "candidate_disabled_objective": "L_existing_task + lambda_thbr * 0 where THBR is disabled",
        "thbr_lambda": THBR_LAMBDA,
        "margin": None,
        "temperature": None,
        "max_abs_loss_difference": float(max(max(row["loss_abs_differences"].values()) for row in rows)),
        "gradient_equality_method": "SHA256 of flattened FP32 parameter gradient bytes; equality implies exact zero difference",
        "max_abs_gradient_difference": 0.0 if all(row["gradient_hash_equal"] for row in rows) else None,
        "exact_disabled_path_all_batches": bool(all(row["exact_disabled_path"] for row in rows)),
        "no_optimizer_update": True,
        "rows": rows,
    }
    result["parity_pass"] = bool(result["exact_disabled_path_all_batches"])
    json_dump(LOSS_PARITY_JSON, result)
    if not result["parity_pass"]:
        raise RuntimeError("THBR loss parity failed")
    return result


def finite_model_parameters(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def run_arm(payload: dict, manifest: list[dict], root: Path, arm: str) -> dict:
    if arm not in (ARM_CONTROL, ARM_CANDIDATE):
        raise ValueError(arm)
    candidate = arm == ARM_CANDIDATE
    control_rows = read_control_rows() if candidate else None
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    audit.restore_rng_state(audit.rng_from_payload(payload))
    model = audit.make_model(payload, device, training_graph=True)
    optimizer, scheduler, scaler = make_optimizer(model, payload)
    anchor = SafeImageAdapterAnchor.from_checkpoint(audit.E10, device)
    dataset = audit.get_text_and_image_dataset("VisA", IMG, "train")
    data_generator = make_dataloader_generator(SEED)
    data_generator.set_state(payload["dataloader_generator_state"])
    worker = EpochWorkerInit(SEED)
    loader = make_training_loader(dataset, data_generator, worker)
    # Exact continuation restores RNG after model/optimizer construction;
    # epoch reseeding below then follows train.py's E10 stream contract.
    audit.restore_rng_state(audit.rng_from_payload(payload))
    rows = []
    attempted = 0
    successful = 0
    nonfinite_loss_skips = 0
    nonfinite_grad_skips = 0
    extra_natural_skips = []
    batch_mismatch = None
    numerical_failure = None
    natural_skip_indices = []
    global_step = START_GLOBAL_STEP
    max_consecutive_grad_skips = 0
    consecutive_grad_skips = 0

    while attempted < MAX_ATTEMPTS:
        made_progress = False
        for epoch in range(START_EPOCH + 1, START_EPOCH + 10):
            if attempted >= MAX_ATTEMPTS:
                break
            configure_epoch(model, optimizer, epoch)
            worker.set_epoch(epoch)
            data_generator.manual_seed(SEED + 104729 * epoch)
            for batch_idx, batch in enumerate(loader):
                if attempted >= MAX_ATTEMPTS:
                    break
                made_progress = True
                attempt = attempted
                attempted += 1
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                label = batch["label"].to(device)
                identity = audit.batch_identity(attempt, epoch, batch_idx, batch, image, mask, label)
                expected = manifest[attempt]
                if not identity_matches(identity, expected):
                    batch_mismatch = f"manifest mismatch at attempt {attempt}"
                    row = {**identity, "arm": arm, "status": "batch_mismatch", "successful_step": 0}
                    rows.append(row)
                    break
                expected_control = control_rows[attempt] if candidate else None
                control_skip = bool(expected_control and expected_control.get("status", "").endswith("skip"))
                if candidate and not identity_matches(identity, expected_control):
                    batch_mismatch = f"control/candidate identity mismatch at attempt {attempt}"
                    rows.append({**identity, "arm": arm, "status": "batch_mismatch", "successful_step": 0})
                    break

                optimizer.zero_grad(set_to_none=True)
                losses = audit.build_training_losses(
                    model, image, mask, label, batch["class_name"], device, policy,
                    FocalLoss(), BinaryDiceLoss(), include_thbr=candidate,
                )
                observed_thbr = losses["thbr"] if candidate else audit.thbr_raw(losses["seg_pred"][:, 1].detach(), mask)
                weighted_thbr = THBR_LAMBDA * observed_thbr if candidate else torch.zeros_like(losses["existing_task"])
                total_loss = losses["existing_task"] + weighted_thbr
                anchor_loss = anchor.loss(model.image_adapter)
                telemetry = thbr_telemetry(losses["seg_pred"][:, 1], mask)
                row = {
                    **identity,
                    "arm": arm,
                    "status": "pending",
                    "thbr_enabled": candidate,
                    "thbr_raw": scalar(observed_thbr),
                    "thbr_weighted": scalar(weighted_thbr),
                    "total_loss": scalar(total_loss),
                    "existing_task_loss": scalar(losses["existing_task"]),
                    "anchor_loss": scalar(anchor_loss),
                    "thbr_grad_norm": None,
                    "successful_step": 0,
                    "nonfinite_loss_skip": 0,
                    "nonfinite_grad_skip": 0,
                    "paired_control_skip": 0,
                    "optimizer_state_finite": None,
                    "parameters_finite": None,
                    "global_step_before": global_step,
                    **telemetry,
                }
                if candidate and attempt % THBR_GRAD_AUDIT_INTERVAL == 0 and torch.isfinite(observed_thbr).all().item():
                    image_parameters = [p for p in model.image_adapter.parameters() if p.requires_grad]
                    thbr_gradients = torch.autograd.grad(observed_thbr, image_parameters, retain_graph=True, allow_unused=True)
                    thbr_norm = torch.zeros((), device=device, dtype=torch.float32)
                    for gradient in thbr_gradients:
                        if gradient is not None:
                            thbr_norm = thbr_norm + gradient.float().square().sum()
                    row["thbr_grad_norm"] = float(thbr_norm.sqrt().detach().cpu())

                loss_is_finite = bool(torch.isfinite(total_loss).all().item() and torch.isfinite(anchor_loss).all().item())
                if not loss_is_finite:
                    nonfinite_loss_skips += 1
                    natural_skip_indices.append(attempt)
                    row.update(status="nonfinite_loss_skip", nonfinite_loss_skip=1)
                    if candidate and not control_skip:
                        extra_natural_skips.append(attempt)
                        numerical_failure = numerical_failure or "candidate_extra_natural_skip"
                    optimizer.zero_grad(set_to_none=True)
                    rows.append(row)
                    continue
                if control_skip:
                    row.update(status="paired_control_skip", paired_control_skip=1)
                    rows.append(row)
                    continue

                scaler.scale(total_loss).backward(retain_graph=True)
                scaler.unscale_(optimizer)
                if has_non_finite_grad(optimizer):
                    nonfinite_grad_skips += 1
                    consecutive_grad_skips += 1
                    max_consecutive_grad_skips = max(max_consecutive_grad_skips, consecutive_grad_skips)
                    natural_skip_indices.append(attempt)
                    row.update(status="nonfinite_grad_skip", nonfinite_grad_skip=1, consecutive_nonfinite_grad_skips=consecutive_grad_skips)
                    if candidate and not control_skip:
                        extra_natural_skips.append(attempt)
                        numerical_failure = numerical_failure or "candidate_extra_natural_skip"
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    rows.append(row)
                    continue
                consecutive_grad_skips = 0
                image_named = [(name, parameter) for name, parameter in sorted(model.image_adapter.named_parameters()) if parameter.requires_grad]
                image_names = [name for name, _ in image_named]
                image_parameters = [parameter for _, parameter in image_named]
                task_gradients = torch.autograd.grad(total_loss, image_parameters, retain_graph=True, allow_unused=True)
                raw_anchor_gradients = torch.autograd.grad(anchor_loss, image_parameters, retain_graph=False, allow_unused=True)
                anchor_metrics = apply_family_safe_anchor_budget(
                    model.image_adapter,
                    sorted(model.named_parameters()),
                    task_gradients=dict(zip(image_names, task_gradients)),
                    raw_anchor_gradients=dict(zip(image_names, raw_anchor_gradients)),
                    anchor_lambda=ANCHOR_LAMBDA,
                    rho=ANCHOR_RHO,
                    total_trainable_parameters=None,
                )
                torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.text_adapter.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.soft_prompt.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                successful += 1
                global_step += 1
                state_finite = optimizer_state_is_finite(optimizer)
                params_finite = finite_model_parameters(model)
                if not state_finite or not params_finite:
                    numerical_failure = numerical_failure or "post_step_nonfinite_state"
                row.update(
                    status="success" if state_finite and params_finite else "numerical_failure",
                    successful_step=1,
                    optimizer_state_finite=int(state_finite),
                    parameters_finite=int(params_finite),
                    safe_anchor_effective_ratio=anchor_metrics["global_effective_ratio"],
                    safe_anchor_max_family_ratio=anchor_metrics["max_effective_active_family_ratio"],
                    global_step_after=global_step,
                )
                rows.append(row)
                del losses
                torch.cuda.empty_cache()
            if batch_mismatch or attempted >= MAX_ATTEMPTS:
                break
            scheduler.step()
        if not made_progress:
            break

    output_dir = root / arm
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(payload["resolved_scientific_config"])
    config["implementation_git_sha"] = current_git_sha(str(REPO))
    config["working_tree_diff_sha256"] = None
    operational = dict(payload.get("resolved_operational_config", {}))
    operational.update({
        "save_path": str(output_dir), "resume": str(audit.E10), "max_batches": MAX_ATTEMPTS,
        "num_workers": 6, "cuda_device": 0, "pin_memory": True,
        "persistent_workers": False, "prefetch_factor": 2, "non_blocking_copy": False,
        "anchor_reference_path": str(audit.E10), "protocol_horizon": MAX_ATTEMPTS,
    })
    checkpoint = build_full_checkpoint(
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        epoch=epoch if rows else START_EPOCH, global_step=global_step,
        config=config, parent_config=payload.get("parent_scientific_config"),
        operational_config=operational, repo=str(REPO),
        clip_sha256=payload.get("clip_sha256"), dataset_manifest_sha256=payload.get("dataset_manifest_sha256"),
        dataloader_generator=data_generator, anchor=anchor,
        anchor_lambda=ANCHOR_LAMBDA, seed=SEED, precision="fp16",
        precision_protocol="HISTORICAL_MIXED_FP16_FP32_V1",
        later_transformer_fp32_islands=False, tf32_enabled=False,
    )
    checkpoint.update({
        "arm": arm,
        "protocol_id": PROTOCOL_ID,
        "thbr": {
            "enabled": candidate, "lambda_thbr": THBR_LAMBDA,
            "margin": None, "temperature": None,
            "matched_rule": "K=min(anomaly_pixels,background_pixels); top-K background with bottom-K anomaly; sorted hard descending/weak ascending",
        },
        "start_checkpoint": str(audit.E10),
        "start_checkpoint_sha256": audit.E10_SHA256,
        "attempted_steps": attempted,
        "successful_steps": successful,
        "nonfinite_loss_skips": nonfinite_loss_skips,
        "nonfinite_grad_skips": nonfinite_grad_skips,
        "natural_skip_indices": natural_skip_indices,
        "extra_natural_skips": extra_natural_skips,
        "batch_match_gate": "FAIL" if batch_mismatch else "PASS",
    })
    atomic_torch_save(output_dir / "final.pth", checkpoint)
    output_csv = CANDIDATE_CSV if candidate else CONTROL_CSV
    fields = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol_id": PROTOCOL_ID, "arm": arm, "candidate_mechanism": candidate,
        "root": str(output_dir), "checkpoint": str(output_dir / "final.pth"),
        "checkpoint_sha256": sha256_file(output_dir / "final.pth"),
        "attempted_steps": attempted, "successful_steps": successful,
        "nonfinite_loss_skips": nonfinite_loss_skips, "nonfinite_grad_skips": nonfinite_grad_skips,
        "natural_skip_indices": natural_skip_indices, "extra_natural_skips": extra_natural_skips,
        "max_consecutive_nonfinite_grad_skips": max_consecutive_grad_skips,
        "batch_match_gate": "FAIL" if batch_mismatch else "PASS", "batch_mismatch": batch_mismatch,
        "numerical_failure": numerical_failure, "rows": rows,
        "config": {
            "start_checkpoint_sha256": audit.E10_SHA256,
            "lambda_thbr": THBR_LAMBDA, "anchor_lambda": ANCHOR_LAMBDA,
            "anchor_rho": ANCHOR_RHO, "precision": "fp16",
            "precision_protocol": "HISTORICAL_MIXED_FP16_FP32_V1", "gradscaler": True,
            "tf32": False, "bf16": False, "deterministic_algorithms": True,
            "loader": {"batch_size": 6, "num_workers": 6, "pin_memory": True, "persistent_workers": False, "prefetch_factor": 2},
            "train_stage_fusion": [1 / 3, 1 / 3, 1 / 3], "functional_feature_anchor": False,
            "margin": None, "temperature": None,
        },
    }
    json_dump(output_dir / "summary.json", summary)
    del model
    torch.cuda.empty_cache()
    if attempted != MAX_ATTEMPTS:
        raise RuntimeError(f"expected exactly {MAX_ATTEMPTS} attempted steps, got {attempted}")
    if batch_mismatch:
        raise RuntimeError("BATCH_MATCH_GATE=FAIL")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("parity", "run"), required=True)
    parser.add_argument("--arm", choices=(ARM_CONTROL, ARM_CANDIDATE))
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    payload, manifest = validate_audit_and_start()
    if args.mode == "parity":
        if args.arm is not None:
            raise ValueError("--arm is not used by parity mode")
        result = loss_parity(payload, manifest)
        print(json.dumps({"status": "PASS", "mode": "parity", "max_abs_loss_difference": result["max_abs_loss_difference"], "max_abs_gradient_difference": result["max_abs_gradient_difference"]}, sort_keys=True))
        return
    if args.arm is None:
        raise ValueError("--arm is required for run mode")
    result = run_arm(payload, manifest, Path(args.root), args.arm)
    print(json.dumps({"status": "PASS", "arm": args.arm, "attempted": result["attempted_steps"], "successful": result["successful_steps"], "nonfinite_loss_skips": result["nonfinite_loss_skips"], "nonfinite_grad_skips": result["nonfinite_grad_skips"]}, sort_keys=True))


if __name__ == "__main__":
    main()
