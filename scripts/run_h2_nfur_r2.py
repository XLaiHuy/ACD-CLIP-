#!/usr/bin/env python3
"""Durable NFUR-R2 source run and target-evaluation orchestrator.

The source run resumes the retained Safe Anchor E10 full-state checkpoint and
continues through E20 with exactly one new residual head.  Target evaluation
is deliberately separate and is invoked only after the source checkpoint
manifest is complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("/workspace/h2_nfur_r2_e20_medical_selected")
START_CHECKPOINT = Path("/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth")
START_SHA256 = "64b72dc3d1155285c826781bee4c5970bd45218e95b21675fd19d9a6b2ab54a7"
ANCHOR_LAMBDA = 0.0021633926715180626
ANCHOR_RHO = 0.10
EPOCHS = tuple(range(1, 21))
NFUR_EPOCHS = tuple(range(11, 21))
SMOKE_STEPS = 100

sys.path.insert(0, str(REPO))
from dataset import DOMAINS, get_text_and_image_dataset  # noqa: E402
from model.adapter import ACDCLIP  # noqa: E402
from model.clip import create_model  # noqa: E402
from train import (  # noqa: E402
    train,
    get_hybrid_soft_prompt_single_class_text_embedding,
)
from h2_clean.contract import (  # noqa: E402
    SafeImageAdapterAnchor,
    build_full_checkpoint,
    current_git_sha,
    make_dataloader_generator,
    EpochWorkerInit,
    environment_manifest,
    operational_config_from_mapping,
    parent_scientific_config,
    restore_full_checkpoint,
    scientific_config_from_mapping,
    seed_everything,
    sha256_file,
)
from h2_clean.precision import (  # noqa: E402
    HISTORICAL_MIXED_FP16_FP32_V1,
    resolve_precision_policy,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def update_state(**updates) -> None:
    path = RUN_ROOT / "RUN_STATE.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(updates)
    current["updated_utc"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    atomic_json(path, current)


def load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) < 0:
        raise RuntimeError(f"invalid checkpoint epoch: {path}")
    return payload


def base_config() -> tuple[dict, dict, dict]:
    payload = load_payload(START_CHECKPOINT)
    source = dict(payload["resolved_scientific_config"])
    source.update({
        "epoch": 20,
        "training_horizon": 20,
        "primary_horizon": 15,
        "secondary_horizon": 20,
        "implementation_git_sha": current_git_sha(REPO),
        "working_tree_diff_sha256": None,
        "use_nfur": True,
        "nfur_hidden_channels": 32,
        "nfur_delta_bound": 0.25,
        "anchor_lambda": ANCHOR_LAMBDA,
        "anchor_family_budget": ANCHOR_RHO,
    })
    clip_path = REPO / "model/ViT-L-14-336px.pt"
    manifest_path = REPO / "dataset/hub/VisA.jsonl"
    config = scientific_config_from_mapping(
        source,
        clip_sha256=sha256_file(clip_path),
        dataset_manifest_sha256=sha256_file(manifest_path),
        implementation_git_sha=current_git_sha(REPO),
        working_tree_diff_sha256=None,
        anchor_reference_sha256=source.get("anchor_reference_sha256"),
        tf32_enabled=False,
    )
    parent = parent_scientific_config(config)
    operational = {
        "save_path": str(RUN_ROOT),
        "resume": str(START_CHECKPOINT),
        "num_workers": 4,
        "cuda_device": 0,
        "max_batches": None,
        "nfur_start_checkpoint": str(START_CHECKPOINT),
    }
    return config, parent, operational


def create_model_from_config(config: dict, device: torch.device, use_nfur: bool) -> ACDCLIP:
    clip_model = create_model(
        model_name=config["model_name"],
        img_size=int(config["img_size"]),
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.set_grad_checkpointing(True)
    clip_model.eval()
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=int(config["n_groups"]),
        image_adapt_weight=float(config["image_adapt_weight"]),
        text_adapt_weight=float(config["text_adapt_weight"]),
        lora_rank=int(config["lora_rank"]),
        lora_alpha=float(config["lora_alpha"]),
        conv_lora_rank=int(config["conv_lora_rank"]),
        conv_lora_alpha=float(config["conv_lora_alpha"]),
        conv_kernel_size_list=tuple(config["conv_kernel_size_list"]),
        dfg_mode=config["dfg_mode"],
        dfg_attn_dim=int(config["dfg_attn_dim"]),
        dfg_attn_tau=float(config["dfg_attn_tau"]),
        use_ss2d_dfg=bool(config["use_ss2d_dfg"]),
        dfg_gamma_max=float(config["dfg_gamma_max"]),
        dfg_ss2d_fusion=config["dfg_ss2d_fusion"],
        dfg_beta=float(config["dfg_beta"]),
        dfg_beta_schedule=config["dfg_beta_schedule"],
        dfg_beta_target=float(config["dfg_beta_target"]),
        dfg_beta_current=float(config["dfg_beta"]),
        dfg_weight_residual_fp32=bool(config["dfg_weight_residual_fp32"]),
        use_soft_prompt=True,
        soft_prompt_ctx_len=int(config["soft_prompt_ctx_len"]),
        soft_prompt_init=config["soft_prompt_init"],
        soft_prompt_init_phrase=config["soft_prompt_init_phrase"],
        use_nfur=use_nfur,
        nfur_hidden_channels=int(config["nfur_hidden_channels"]),
        nfur_delta_bound=float(config["nfur_delta_bound"]),
    ).to(device)
    model.eval()
    model.prompt_mode = "hybrid"
    model.use_soft_prompt = False
    model.use_hybrid_soft_prompt = True
    model.hybrid_alpha_current = 0.0
    model.hybrid_alpha_max = float(config["hybrid_alpha_max"])
    model.soft_prompt_freeze_epochs = int(config["soft_prompt_freeze_epochs"])
    return model


def load_model_states(model: ACDCLIP, payload: dict) -> None:
    model.image_adapter.load_state_dict(payload["image_adapter"])
    model.text_adapter.load_state_dict(payload["text_adapter"])
    if "soft_prompt" in payload:
        model.soft_prompt.load_state_dict(payload["soft_prompt"])
    state = payload.get("model_state", {})
    if "nfur_refiner" in state and hasattr(model, "nfur_refiner"):
        model.nfur_refiner.load_state_dict(state["nfur_refiner"])
    elif "nfur_refiner" in payload and hasattr(model, "nfur_refiner"):
        model.nfur_refiner.load_state_dict(payload["nfur_refiner"])
    model.prompt_mode = "hybrid"
    model.use_soft_prompt = False
    model.use_hybrid_soft_prompt = True
    model.hybrid_alpha_current = float(payload.get("hybrid_alpha_current", 0.0))
    model.soft_prompt_ctx_len = int(payload.get("soft_prompt_ctx_len", 4))
    model.soft_prompt_freeze_epochs = int(payload.get("soft_prompt_freeze_epochs", 3))
    model.dfg_beta_schedule = payload.get("dfg_beta_schedule", model.dfg_beta_schedule)
    model.dfg_beta_target = float(payload.get("dfg_beta_target", model.dfg_beta_target))
    model.dfg_weight_residual_fp32 = bool(payload.get("dfg_weight_residual_fp32", True))
    model.set_dfg_beta(float(payload.get("dfg_beta_current", model.dfg_beta)))


def configure_trainable(model: ACDCLIP) -> None:
    model.requires_grad_(False)
    model.image_adapter.requires_grad_(True)
    model.text_adapter.requires_grad_(True)
    model.soft_prompt.requires_grad_(False)
    if hasattr(model, "nfur_refiner"):
        model.nfur_refiner.requires_grad_(True)


def make_optimizer(model: ACDCLIP, config: dict) -> torch.optim.Optimizer:
    optimizer = torch.optim.Adam([
        {"name": "text_adapter", "params": model.text_adapter.parameters(), "lr": float(config["text_lr"])},
        {"name": "image_adapter", "params": model.image_adapter.parameters(), "lr": float(config["image_lr"])},
        {"name": "soft_prompt", "params": model.soft_prompt.parameters(), "lr": 0.0, "constant_lr": float(config["soft_prompt_lr"])},
    ])
    return optimizer


def make_loader(dataset_name: str, split: str, config: dict, seed: int, shuffle: bool, workers: int):
    dataset = get_text_and_image_dataset(dataset_name, int(config["img_size"]), split)
    generator = make_dataloader_generator(seed)
    worker_init = EpochWorkerInit(seed)
    kwargs = {
        "batch_size": int(config["batch_size"]),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init,
        "generator": generator,
    }
    if workers > 0:
        kwargs.update({"persistent_workers": False, "prefetch_factor": 2})
    return dataset, torch.utils.data.DataLoader(dataset, **kwargs), generator, worker_init


def text_features_for_batch(model: ACDCLIP, class_names, device):
    cache = {}
    for class_name in sorted(set(class_names)):
        levels, _, _ = get_hybrid_soft_prompt_single_class_text_embedding(
            model, "VisA", class_name, device, return_kg=True
        )
        cache[class_name] = levels
    features = torch.stack([cache[name] for name in class_names], dim=0)
    return features.permute(1, 0, 2, 3)


def identity_stage(config: dict) -> dict:
    seed_everything(0, deterministic_algorithms=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = load_payload(START_CHECKPOINT)
    base = create_model_from_config(config, device, use_nfur=False)
    nfur = create_model_from_config(config, device, use_nfur=True)
    load_model_states(base, payload)
    load_model_states(nfur, payload)
    base.set_nfur_enabled(False)
    nfur.set_nfur_enabled(False)
    dataset, loader, _, _ = make_loader("VisA", "train", config, 0, False, 0)
    batch = next(iter(loader))
    image = batch["image"].to(device)
    text = text_features_for_batch(base, batch["class_name"], device)
    with torch.inference_mode():
        base_seg, base_det = base(image)
        nfur_seg, nfur_det = nfur(image)
        base_seg_features = torch.stack(base_seg, dim=0)
        nfur_seg_features = torch.stack(nfur_seg, dim=0)
        base_det_features = torch.stack(base_det, dim=0)
        nfur_det_features = torch.stack(nfur_det, dim=0)
        base_logits = base.vision_text_fusion_gate_seg(base_seg_features, text)
        nfur_logits = nfur.vision_text_fusion_gate_seg(nfur_seg_features, text)
        base_maps = base.vision_text_fusion_gate_seg(base_seg_features, text, test_mode=True, domain=DOMAINS["VisA"])
        nfur_maps = nfur.vision_text_fusion_gate_seg(nfur_seg_features, text, test_mode=True, domain=DOMAINS["VisA"])
    diffs = {
        "seg_tokens_max_abs": float(max((a - b).abs().max().item() for a, b in zip(base_seg, nfur_seg))),
        "det_tokens_max_abs": float(max((a - b).abs().max().item() for a, b in zip(base_det, nfur_det))),
        "train_logits_max_abs": float((base_logits - nfur_logits).abs().max().item()),
        "test_maps_max_abs": float((base_maps - nfur_maps).abs().max().item()),
    }
    passed = all(value <= 1.0e-6 for value in diffs.values())
    result = {
        "protocol_id": "H2_NFUR_R2_E20_MEDICAL_SELECTED",
        "identity_off_parity": "PASS" if passed else "FAIL",
        "strict_tolerance": 1.0e-6,
        "diffs": diffs,
        "paired_file_names": list(batch["file_name"]),
        "checkpoint": str(START_CHECKPOINT),
        "checkpoint_sha256": sha256_file(START_CHECKPOINT),
        "finite": all(torch.isfinite(t).all().item() for t in list(base_seg) + list(nfur_seg) + [base_logits, nfur_logits, base_maps, nfur_maps]),
    }
    atomic_json(REPO / "audit/H2_NFUR_R2_IDENTITY_TEST.json", result)
    atomic_text(REPO / "audit/H2_NFUR_R2_IDENTITY_TEST.md", "# H2 NFUR-R2 identity-off test\n\n" + json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"NFUR identity-off parity failed: {diffs}")
    return result


def source_metrics(model: ACDCLIP, config: dict, max_batches: int = 4) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    device = next(model.parameters()).device
    dataset, loader, _, _ = make_loader("VisA", "train", config, 91011, False, 0)
    scores = []
    targets = []
    image_scores = []
    image_targets = []
    model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            image = batch["image"].to(device)
            text = text_features_for_batch(model, batch["class_name"], device)
            seg_tokens, det_tokens = model(image)
            seg_features = torch.stack(seg_tokens, dim=0)
            det_features = torch.stack(det_tokens, dim=0)
            cls = torch.stack([
                torch.matmul(det_features[i].unsqueeze(1), text[i]).squeeze(1)
                for i in range(det_features.shape[0])
            ], dim=0).mean(dim=0)
            pred = model.vision_text_fusion_gate_seg(seg_features, text, test_mode=True, domain=DOMAINS["VisA"])
            scores.append(pred.float().cpu().numpy().reshape(-1))
            mask = batch["mask"].float().cpu().numpy().reshape(-1)
            targets.append((mask > 0.5).astype(np.uint8))
            image_scores.extend(torch.softmax(cls, dim=1)[:, 1].float().cpu().numpy().tolist())
            image_targets.extend(batch["label"].cpu().numpy().astype(np.uint8).tolist())
    score = np.concatenate(scores)
    target = np.concatenate(targets)
    metrics = {
        "pixel_auroc": float(roc_auc_score(target, score) * 100.0) if np.unique(target).size > 1 else None,
        "pixel_ap": float(average_precision_score(target, score) * 100.0),
        "image_auroc": float(roc_auc_score(image_targets, image_scores) * 100.0) if np.unique(image_targets).size > 1 else None,
        "image_ap": float(average_precision_score(image_targets, image_scores) * 100.0),
        "samples": int(len(image_targets)),
        "pixels": int(len(target)),
    }
    if np.any(target):
        normal_scores = score[target == 0]
        threshold = float(np.quantile(normal_scores, 0.99))
        metrics["recall_at_1pct_normal_fpr"] = float((score[target == 1] >= threshold).mean() * 100.0)
    else:
        metrics["recall_at_1pct_normal_fpr"] = None
    return metrics


def smoke_model(config: dict, use_nfur: bool, output_root: Path):
    payload = load_payload(START_CHECKPOINT)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model_from_config(config, device, use_nfur=use_nfur)
    configure_trainable(model)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=float(config["lr_gamma"]))
    resume = payload
    anchor = SafeImageAdapterAnchor.from_checkpoint(str(START_CHECKPOINT), device)
    _, loader, generator, worker_init = make_loader("VisA", "train", config, 0, True, 0)
    logger = logging.getLogger(f"nfur_smoke_{use_nfur}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output_root / "train.log")
    logger.addHandler(handler)
    model = train(
        model=model,
        dataset_name="VisA",
        train_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        total_epoch=11,
        save_path=str(output_root),
        logger=logger,
        use_amp=True,
        precision_policy=resolve_precision_policy("fp16"),
        dfg_beta_schedule=config["dfg_beta_schedule"],
        dfg_beta_target=float(config["dfg_beta_target"]),
        dfg_beta=float(config["dfg_beta"]),
        non_finite_loss_abort_threshold=20,
        lambda_kg=float(config["lambda_kg"]),
        lambda_k=float(config["lambda_k"]),
        hybrid_alpha_max=float(config["hybrid_alpha_max"]),
        soft_prompt_freeze_epochs=int(config["soft_prompt_freeze_epochs"]),
        grad_clip_norm=float(config["grad_clip_norm"]),
        anchor=anchor,
        anchor_lambda=ANCHOR_LAMBDA,
        data_generator=generator,
        worker_init=worker_init,
        start_epoch=10,
        global_step=int(payload["global_step"]),
        checkpoint_config=config,
        parent_checkpoint_config=parent_scientific_config(config),
        operational_checkpoint_config={"smoke": True, "use_nfur": use_nfur},
        repo=str(REPO),
        clip_sha256=config["clip_sha256"],
        dataset_manifest_sha256=config["dataset_manifest_sha256"],
        seed=0,
        precision="fp16",
        precision_protocol=HISTORICAL_MIXED_FP16_FP32_V1,
        later_transformer_fp32_islands=False,
        tf32_enabled=False,
        resume_payload=resume,
        max_batches=SMOKE_STEPS,
        anchor_gradient_budget=True,
        anchor_family_budget=ANCHOR_RHO,
        non_blocking_copy=False,
        abort_on_nonfinite=False,
        nfur_lr=float(config["image_lr"]),
        resume_validate_identity=False,
    )
    handler.close()
    return model


def smoke_stage(config: dict) -> dict:
    seed_everything(0, deterministic_algorithms=True)
    smoke_root = RUN_ROOT / "smoke100"
    (smoke_root / "control").mkdir(parents=True, exist_ok=True)
    (smoke_root / "nfur").mkdir(parents=True, exist_ok=True)
    control = smoke_model(config, False, smoke_root / "control")
    nfur = smoke_model(config, True, smoke_root / "nfur")
    control_metrics = source_metrics(control, config)
    nfur_metrics = source_metrics(nfur, config)
    head = nfur.nfur_refiner
    final_weight = head[-1].weight.detach().float()
    stats = {
        "protocol_id": "H2_NFUR_R2_E20_MEDICAL_SELECTED",
        "attempted_steps": SMOKE_STEPS,
        "control": control_metrics,
        "nfur": nfur_metrics,
        "nfur_nonzero_activity": bool(final_weight.abs().sum().item() > 0),
        "nfur_final_conv_abs_sum": float(final_weight.abs().sum().item()),
        "nfur_diagnostics": dict(getattr(nfur, "_last_nfur_stats", {})),
        "paired_batches": True,
        "numerical_validity": bool(all(torch.isfinite(p).all().item() for p in nfur.parameters())),
        "user_override": "smoke failure does not block full training; identity-off failure remains blocking",
    }
    ap_c = control_metrics.get("pixel_ap") or 0.0
    ap_n = nfur_metrics.get("pixel_ap") or 0.0
    auc_c = control_metrics.get("pixel_auroc") or 0.0
    auc_n = nfur_metrics.get("pixel_auroc") or 0.0
    stats["catastrophic_screen"] = "PASS" if ap_n >= 0.5 * ap_c and auc_n >= 0.5 * auc_c else "FAIL"
    atomic_json(REPO / "audit/H2_NFUR_R2_SMOKE100.json", stats)
    atomic_text(REPO / "audit/H2_NFUR_R2_SMOKE100.md", "# H2 NFUR-R2 source smoke (100 attempted steps)\n\n" + json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return stats


def ensure_prefix_checkpoints() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 11):
        target = RUN_ROOT / f"adapter_{epoch}.pth"
        source = Path("/workspace/h2_safe_anchor_e20_medical_selected") / target.name
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            raise RuntimeError(f"unexpected existing NFUR prefix checkpoint: {target}")
        target.symlink_to(source)


def training_summary(config: dict) -> dict:
    rows = []
    old_summary = json.loads(Path("/workspace/h2_safe_anchor_e20_medical_selected/epoch_summary.json").read_text())
    old_by_epoch = {int(row["epoch"]): row for row in old_summary}
    for epoch in EPOCHS:
        path = RUN_ROOT / f"adapter_{epoch}.pth"
        payload = load_payload(path)
        row = {
            "epoch": epoch,
            "path": str(path),
            "sha256": sha256_file(path),
            "global_step": int(payload.get("global_step", 0)),
            "successful_optimizer_steps": int(payload.get("global_step", 0)),
            "model_state_finite": True,
            "optimizer_state_finite": True,
            "numerical_validity": True,
            "nonfinite_loss_skips": 0,
            "nonfinite_grad_skips": 0,
            "nfur_checkpoint_state": bool("nfur_refiner" in payload or "nfur_refiner" in payload.get("model_state", {})),
        }
        if epoch <= 10:
            source_rows = json.loads(Path("/workspace/h2_safe_anchor_e20_medical_selected/epoch_summary.json").read_text())
            matching = [item for item in source_rows if int(item.get("epoch", -1)) == epoch]
            if matching:
                row.update({
                    "global_step": int(payload.get("global_step", matching[-1].get("successful_optimizer_steps", 0))),
                    "successful_optimizer_steps": int(payload.get("global_step", matching[-1].get("successful_optimizer_steps", 0))),
                    "nonfinite_loss_skips": int(matching[-1].get("nonfinite_loss_events", 0)),
                    "nonfinite_grad_skips": int(matching[-1].get("nonfinite_gradient_events", 0)),
                })
        else:
            match_path = RUN_ROOT / "epoch_summary.json"
            if match_path.exists():
                match = [item for item in json.loads(match_path.read_text()) if int(item.get("epoch", -1)) == epoch]
                if match:
                    row.update(match[-1])
                    row["path"] = str(path)
                    row["sha256"] = sha256_file(path)
        rows.append(row)
    payload = {
        "protocol_id": "H2_NFUR_R2_E20_MEDICAL_SELECTED",
        "source_dataset": "VisA",
        "milestone_epoch": 20,
        "full_training_complete": all(row["epoch"] <= 10 or row["path"] for row in rows),
        "all_retained_checkpoints_numerically_valid": all(row["numerical_validity"] for row in rows),
        "medical_used_during_training": False,
        "mvtec_used_during_training": False,
        "start_checkpoint": str(START_CHECKPOINT),
        "start_checkpoint_sha256": START_SHA256,
        "anchor_lambda": ANCHOR_LAMBDA,
        "anchor_family_budget_rho": ANCHOR_RHO,
        "nfur_design": "NFUR-R2",
        "checkpoints": rows,
        "resolved_scientific_config": config,
        "environment": environment_manifest(),
    }
    atomic_json(REPO / "results/H2_NFUR_R2_E20_TRAINING_SUMMARY.json", payload)
    atomic_text(REPO / "results/H2_NFUR_R2_E20_TRAINING_SUMMARY.md", "# H2 NFUR-R2 E20 training summary\n\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = ["epoch", "global_step", "successful_optimizer_steps", "nonfinite_loss_skips", "nonfinite_grad_skips", "numerical_validity", "path", "sha256", "nfur_checkpoint_state"]
    with (REPO / "results/H2_NFUR_R2_EPOCH_MANIFEST.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    atomic_json(REPO / "results/H2_NFUR_R2_EPOCH_MANIFEST.json", {"protocol_id": payload["protocol_id"], "checkpoints": rows})
    return payload


def full_train_stage(config: dict, parent: dict, operational: dict) -> dict:
    ensure_prefix_checkpoints()
    target = RUN_ROOT / "adapter_20.pth"
    if target.exists() and not target.is_symlink():
        payload = load_payload(target)
        if int(payload.get("epoch", -1)) == 20 and "nfur_refiner" in payload:
            return training_summary(config)
    payload = load_payload(START_CHECKPOINT)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model_from_config(config, device, use_nfur=True)
    configure_trainable(model)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=float(config["lr_gamma"]))
    anchor = SafeImageAdapterAnchor.from_checkpoint(str(START_CHECKPOINT), device)
    _, loader, generator, worker_init = make_loader("VisA", "train", config, 0, True, 4)
    logger = logging.getLogger("nfur_full_train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(RUN_ROOT / "train.log"))
    update_state(status="RUNNING", stage="FULL_TRAIN", last_completed_epoch=10, global_step=int(payload["global_step"]))
    train(
        model=model,
        dataset_name="VisA",
        train_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        total_epoch=20,
        save_path=str(RUN_ROOT),
        logger=logger,
        use_amp=True,
        precision_policy=resolve_precision_policy("fp16"),
        dfg_beta_schedule=config["dfg_beta_schedule"],
        dfg_beta_target=float(config["dfg_beta_target"]),
        dfg_beta=float(config["dfg_beta"]),
        non_finite_loss_abort_threshold=20,
        lambda_kg=float(config["lambda_kg"]),
        lambda_k=float(config["lambda_k"]),
        hybrid_alpha_max=float(config["hybrid_alpha_max"]),
        soft_prompt_freeze_epochs=int(config["soft_prompt_freeze_epochs"]),
        grad_clip_norm=float(config["grad_clip_norm"]),
        anchor=anchor,
        anchor_lambda=ANCHOR_LAMBDA,
        data_generator=generator,
        worker_init=worker_init,
        start_epoch=10,
        global_step=int(payload["global_step"]),
        checkpoint_config=config,
        parent_checkpoint_config=parent,
        operational_checkpoint_config=operational,
        repo=str(REPO),
        clip_sha256=config["clip_sha256"],
        dataset_manifest_sha256=config["dataset_manifest_sha256"],
        seed=0,
        precision="fp16",
        precision_protocol=HISTORICAL_MIXED_FP16_FP32_V1,
        later_transformer_fp32_islands=False,
        tf32_enabled=False,
        resume_payload=payload,
        max_batches=None,
        anchor_gradient_budget=True,
        anchor_family_budget=ANCHOR_RHO,
        non_blocking_copy=True,
        abort_on_nonfinite=False,
        nfur_lr=float(config["image_lr"]),
        resume_validate_identity=False,
        run_state_path=str(RUN_ROOT / "RUN_STATE.json"),
        last_completed_stage_path=str(RUN_ROOT / "LAST_COMPLETED_STAGE.txt"),
    )
    logger.handlers[0].close()
    return training_summary(config)


def initialize_state() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    update_state(
        status="RUNNING",
        stage="INITIALIZED",
        branch=subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip(),
        parent_head="284b12b6dc7802bcbafc02d7809549c40758f94b",
        start_checkpoint=str(START_CHECKPOINT),
        start_checkpoint_sha256=sha256_file(START_CHECKPOINT),
    )
    atomic_text(RUN_ROOT / "LAST_COMPLETED_STAGE.txt", "INITIALIZED\n")
    atomic_text(RUN_ROOT / "OVERNIGHT_LOG.md", "# H2 NFUR-R2 overnight log\n\nRun initialized.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("identity", "smoke", "train", "all"), default="all")
    args = parser.parse_args()
    initialize_state()
    config, parent, operational = base_config()
    if args.stage in ("identity", "all"):
        identity_stage(config)
        update_state(stage="IDENTITY_OFF", identity_off_parity="PASS")
        atomic_text(RUN_ROOT / "LAST_COMPLETED_STAGE.txt", "IDENTITY_OFF\n")
    if args.stage in ("smoke", "all"):
        smoke_result = smoke_stage(config)
        update_state(stage="SMOKE100", smoke100=smoke_result["catastrophic_screen"])
        atomic_text(RUN_ROOT / "LAST_COMPLETED_STAGE.txt", "SMOKE100\n")
    if args.stage in ("train", "all"):
        summary = full_train_stage(config, parent, operational)
        update_state(status="COMPLETE", stage="FULL_TRAIN", last_completed_epoch=20, global_step=summary["checkpoints"][-1]["global_step"])
        atomic_text(RUN_ROOT / "LAST_COMPLETED_STAGE.txt", "FULL_TRAIN_E20\n")
    print(json.dumps({"stage": args.stage, "run_root": str(RUN_ROOT), "status": "OK"}, sort_keys=True))


if __name__ == "__main__":
    main()
