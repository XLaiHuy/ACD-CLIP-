#!/usr/bin/env python3
"""CIR/TRAIN-{source}: train the frozen parent adapter with CIR in one path."""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import TextAndImageDataset
from model.checkpoint_utils import capture_rng_state, restore_rng_state, write_torch_checkpoint_atomic
from model.phase2b_legacy_bridge import assert_phase2b_gradient_contract, load_adapter_state
from model.phase2b_runtime import build_phase2b_trainable, configure_canonical_fp32, trainable_parameter_counts
from model.phase2b_schedule import apply_soft_prompt_lr_policy, get_dfg_beta_for_epoch, get_hybrid_alpha_for_epoch, grad_accum_window_size
from tools.cir_rmt.identity import checkpoint_metadata, config_sha256, load_cir_config, release_identity_fields, validate_checkpoint_identity
from tools.cir_rmt.parameter_anchor import ImageParameterAnchor, load_image_parameter_anchor
from tools.cir_rmt.runtime import forward_cir
from utils import calculate_seg_loss, make_dataloader_generator, seed_worker




def seed_everything(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}



def build_loader(source: str, source_root: Path, config: Mapping[str, Any], args: argparse.Namespace, generator: torch.Generator):
    source_key = str(source).lower()
    try:
        manifest_name = {"visa": "VisA.jsonl", "mvtec": "MVTec.jsonl"}[source_key]
    except KeyError as exc:
        raise ValueError("source must be visa or mvtec") from exc
    metadata = Path(__file__).resolve().parents[2] / "dataset" / "hub" / manifest_name
    dataset = TextAndImageDataset(str(source_root), str(metadata), int(config["img_size"]))
    kwargs: dict[str, Any] = {"batch_size": int(args.micro_batch_size), "shuffle": True, "num_workers": int(args.num_workers), "pin_memory": bool(args.pin_memory), "worker_init_fn": seed_worker, "generator": generator}
    if args.num_workers > 0:
        kwargs.update({"persistent_workers": bool(args.persistent_workers), "prefetch_factor": int(args.prefetch_factor)})
    return dataset, DataLoader(dataset, **kwargs)


def _text_with_regularizers(model: Any, class_names: list[str], config: Mapping[str, Any], device: torch.device):
    from train import _text_with_regularizers as parent_text
    return parent_text(model, class_names, config, device)


def _optimizer(model: Any, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.Adam([
        {"name": "image_adapter", "params": list(model.image_adapter.parameters()), "lr": float(config["image_lr"])},
        {"name": "text_adapter", "params": list(model.text_adapter.parameters()), "lr": float(config["text_lr"])},
        {"name": "soft_prompt", "params": list(model.soft_prompt.parameters()), "lr": float(config["soft_prompt_lr"]), "constant_lr": float(config["soft_prompt_lr"])},
    ])


def _set_epoch_state(model: Any, optimizer: torch.optim.Optimizer, config: Mapping[str, Any], epoch: int) -> tuple[float, float, bool]:
    alpha = get_hybrid_alpha_for_epoch(epoch, float(config.get("hybrid_alpha_max", 0.2)), int(config.get("soft_prompt_freeze_epochs", 3))) if config.get("use_hybrid_soft_prompt") else 0.0
    frozen = bool(config.get("use_hybrid_soft_prompt") and epoch <= int(config.get("soft_prompt_freeze_epochs", 3)))
    model.hybrid_alpha_current = float(alpha)
    model.prompt_mode = "hybrid" if config.get("use_hybrid_soft_prompt") else "soft" if config.get("use_soft_prompt") else "hard"
    model.soft_prompt.requires_grad_(not frozen and bool(config.get("soft_prompt_trainable", True)))
    model.text_adapter.requires_grad_(True)
    beta = get_dfg_beta_for_epoch(epoch, str(config.get("dfg_beta_schedule", "fixed")), float(config.get("dfg_beta_target", config.get("dfg_beta", 0.1))), float(config.get("dfg_beta", 0.1)))
    model.set_dfg_beta(beta)
    apply_soft_prompt_lr_policy(optimizer, frozen)
    return float(alpha), float(beta), frozen


def checkpoint_payload(model: Any, parent_config: Mapping[str, Any], cir_config: Mapping[str, Any], source: str, epoch: int, step: int, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, generator: torch.Generator, git_sha: str | None, image_anchor: ImageParameterAnchor | None = None, image_anchor_lambda: float = 0.0) -> dict[str, Any]:
    payload = dict(checkpoint_metadata(cir_config, source_dataset=source, epoch=epoch, git_sha=git_sha))
    payload.update({
        "checkpoint_version": 1, "parent_protocol": parent_config["protocol_version"], "model_name": parent_config["model_name"], "img_size": int(parent_config["img_size"]), "precision": "fp32", "amp_enabled": False, "tf32_enabled": False,
        "global_step": int(step), "parent_config": dict(parent_config), "parent_config_sha256": str(cir_config["parent_config_sha256"]),
        "resolved_scientific_config": dict(parent_config), "image_adapter": _cpu_state(model.image_adapter), "text_adapter": _cpu_state(model.text_adapter), "soft_prompt": _cpu_state(model.soft_prompt),
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
        "image_anchor": image_anchor.metadata(float(image_anchor_lambda)) if image_anchor is not None else {"enabled": False, "lambda_image_anchor": float(image_anchor_lambda), "scope": "image_adapter_parameters_only", "train_only": True},
    })
    payload.update(capture_rng_state(dataloader_generator=generator))
    return payload


def validate_resume(payload: Mapping[str, Any], cir_config: Mapping[str, Any], source: str, git_sha: str | None, image_anchor_lambda: float = 0.0, image_anchor: ImageParameterAnchor | None = None) -> None:
    validate_checkpoint_identity(payload, cir_config, source_dataset=source, expected_git_sha=git_sha or payload.get("git_sha"))
    if payload.get("precision") != "fp32" or payload.get("amp_enabled") is True or payload.get("tf32_enabled") is True:
        raise ValueError("CIR resume violates FP32 contract")
    saved_anchor = payload.get("image_anchor", {})
    saved_lambda = float(saved_anchor.get("lambda_image_anchor", 0.0))
    if abs(saved_lambda - float(image_anchor_lambda)) > 1.0e-15:
        raise ValueError(f"CIR resume image-anchor coefficient mismatch: saved={saved_lambda} current={float(image_anchor_lambda)}")
    if float(image_anchor_lambda) > 0.0:
        if image_anchor is None or saved_anchor.get("reference_checkpoint_sha256") != image_anchor.reference_checkpoint_sha256:
            raise ValueError("CIR resume image-anchor reference mismatch")


def _gradient_l2(gradients: list[torch.Tensor | None]) -> float:
    values = [gradient.detach().float().pow(2).sum() for gradient in gradients if gradient is not None]
    return float(torch.sqrt(torch.stack(values).sum()).item()) if values else 0.0


def _finite_state_mapping(state: Mapping[str, Any]) -> bool:
    for value in state.values():
        if isinstance(value, Mapping):
            if not _finite_state_mapping(value):
                return False
        elif isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            return False
    return True


def e10_catastrophic_failure_gate(
    *,
    model: Any,
    checkpoint_payload_value: Mapping[str, Any],
    cir_config: Mapping[str, Any],
    source: str,
    row: Mapping[str, Any],
    gradient_probe: Mapping[str, float] | None,
    git_sha: str | None,
) -> dict[str, Any]:
    """Abort only on structural E10 failures, never on small metric changes."""
    reasons: list[str] = []
    model_finite = all(bool(torch.isfinite(parameter.detach()).all()) for parameter in model.parameters())
    payload_finite = _finite_state_mapping(checkpoint_payload_value)
    if not model_finite or not payload_finite:
        reasons.append("nonfinite_model_or_checkpoint_state")
    try:
        validate_checkpoint_identity(checkpoint_payload_value, cir_config, source_dataset=source, expected_git_sha=git_sha, expected_epoch=10)
    except Exception as exc:  # pragma: no cover - exact exception text is serialized for the run audit
        reasons.append(f"checkpoint_identity_failure:{type(exc).__name__}:{exc}")
    if int(row.get("batches", 0)) < 1:
        reasons.append("no_completed_training_batch")
    if float(row.get("rmt_delta_abs_mean", 0.0)) <= 0.0:
        reasons.append("rmt_path_inactive")
    if gradient_probe is not None:
        base_norm = float(gradient_probe.get("base_grad_l2", 0.0))
        anchor_norm = float(gradient_probe.get("anchor_grad_l2", 0.0))
        ratio = float(gradient_probe.get("anchor_to_base_ratio", 0.0))
        if not all(np.isfinite(value) for value in (base_norm, anchor_norm, ratio)):
            reasons.append("nonfinite_gradient_probe")
        elif ratio > 10.0:
            reasons.append("severe_anchor_gradient_domination")
    return {
        "stage": "CIR/E10-CATASTROPHIC-FAILURE-GATE",
        "epoch": 10,
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "small_metric_differences_do_not_abort": True,
        "model_finite": model_finite,
        "checkpoint_state_finite": payload_finite,
        "gradient_probe": dict(gradient_probe or {}),
        "thresholds": {"anchor_to_base_gradient_ratio_abort_gt": 10.0},
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    cir_config = load_cir_config(args.config)
    parent_path = Path(cir_config.get("parent_config_path", "configs/phase2b_canonical_v1.json"))
    if not parent_path.is_absolute():
        parent_path = Path(__file__).resolve().parents[2] / parent_path
    parent_config = json.loads(parent_path.read_text(encoding="utf-8"))
    source = str(args.source).lower()
    parent_config.update({"dataset": "VisA" if source == "visa" else "MVTec", "seed": int(args.seed), "epochs": int(args.epochs or parent_config.get("epochs", 20)), "micro_batch_size": int(args.micro_batch_size), "batch_size": int(args.micro_batch_size), "grad_accum_steps": int(args.grad_accum_steps), "effective_batch_size": int(args.micro_batch_size * args.grad_accum_steps), "num_workers": int(args.num_workers), "pin_memory": bool(args.pin_memory), "persistent_workers": bool(args.persistent_workers), "prefetch_factor": int(args.prefetch_factor)})
    if parent_config["effective_batch_size"] != 6:
        raise ValueError("canonical CIR effective batch size must be six")
    candidate_epochs = tuple(int(item) for item in parent_config["candidate_epochs"])
    matched_horizon_e14 = bool(getattr(args, "matched_horizon_e14", False))
    if matched_horizon_e14:
        if int(parent_config["epochs"]) != 14:
            raise ValueError("matched-horizon E14 mode requires --epochs 14")
        candidate_epochs = tuple(epoch for epoch in candidate_epochs if epoch <= 14)
    if not candidate_epochs:
        raise ValueError("canonical CIR candidate epoch schedule must not be empty")
    if source == "visa" and not args.source_root.is_dir():
        raise FileNotFoundError(args.source_root)
    if source == "mvtec" and not args.source_root.is_dir():
        raise FileNotFoundError(args.source_root)
    configure_canonical_fp32(); seed_everything(int(args.seed))
    device = torch.device(args.device)
    print("=" * 60)
    print("ARCH       : " + str(cir_config["arch_id"]))
    print("STAGE      : CIR/TRAIN-" + source.upper())
    print("SOURCE     : " + ("VisA" if source == "visa" else "MVTec-AD"))
    print("CONFIG     : " + config_sha256(cir_config)[:12])
    print("FREEZE     : " + str(cir_config["architecture_freeze_sha256"])[:12])
    print("GIT        : " + str(args.git_sha)[:12])
    print("RMT alpha  : " + str(cir_config["rmt_transport_alpha"]) + " (" + str(cir_config["rmt_alpha_status"]) + ")")
    print("Peers      : " + str(cir_config["rmt_peer_count"]))
    print("Groups     : " + str(cir_config["n_groups"]))
    print("Score mode : " + str(cir_config["rmt_score_mode"]))
    print("Device     : " + str(device))
    print("MICRO BATCH : " + str(args.micro_batch_size))
    print("GRAD ACCUM  : " + str(args.grad_accum_steps))
    print("EFF. BATCH  : " + str(parent_config["effective_batch_size"]))
    print("WORKERS     : " + str(args.num_workers))
    print("PREFETCH    : " + str(args.prefetch_factor))
    print("=" * 60)
    model = build_phase2b_trainable(parent_config, args.clip_asset, device)
    image_anchor_lambda = float(getattr(args, "image_anchor_lambda", 0.0))
    if image_anchor_lambda < 0.0:
        raise ValueError("--image-anchor-lambda must be non-negative")
    image_anchor = None
    if image_anchor_lambda > 0.0:
        if getattr(args, "image_anchor_checkpoint", None) is None:
            raise ValueError("--image-anchor-checkpoint is required when --image-anchor-lambda is positive")
        image_anchor = load_image_parameter_anchor(args.image_anchor_checkpoint, model, device)
    generator = make_dataloader_generator(int(args.seed))
    dataset, loader = build_loader(source, args.source_root, parent_config, args, generator)
    optimizer = _optimizer(model, parent_config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=float(parent_config["lr_gamma"]))
    start_epoch, global_step = 1, 0
    if args.resume:
        # Keep RNG tensors on CPU; restore_rng_state requires a CPU ByteTensor.
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume(payload, cir_config, source, args.git_sha, image_anchor_lambda=image_anchor_lambda, image_anchor=image_anchor)
        load_adapter_state(model, payload); optimizer.load_state_dict(payload["optimizer_state"]); scheduler.load_state_dict(payload["scheduler_state"]); restore_rng_state(payload, dataloader_generator=generator)
        start_epoch, global_step = int(payload["epoch"]) + 1, int(payload.get("global_step", 0))
    base_run_root = Path(args.run_root) if args.run_root is not None else Path("runs/cir_rmt") / str(cir_config["arch_id"])
    run_root = base_run_root / source / f"seed{int(args.seed)}"
    checkpoint_root = run_root / "checkpoints"
    history = []
    smoke_target = None if args.smoke_steps is None else int(args.smoke_steps)
    if smoke_target is not None and smoke_target < 1:
        raise ValueError("--smoke-steps must request at least one optimizer step")
    smoke_start_step = int(global_step)
    train_started = time.perf_counter()
    e10_gate: dict[str, Any] | None = None
    quiet_progress = bool(getattr(args, "quiet_progress", False))
    for epoch in range(start_epoch, int(parent_config["epochs"]) + 1):
        _, beta, soft_frozen = _set_epoch_state(model, optimizer, parent_config, epoch)
        model.train(); model.clipmodel.eval(); model.image_encoder.eval(); optimizer.zero_grad(set_to_none=True)
        sums = {"loss": 0.0, "cls": 0.0, "seg": 0.0, "image_anchor": 0.0}
        count = 0
        active, mad_values, delta_values, saturation_values = [], [], [], []
        gradient_probe: dict[str, float] | None = None
        started = time.perf_counter()
        iterator = iter(loader); total_batches = len(loader)
        progress = tqdm(
            loader,
            desc=f"CIR/TRAIN-{source.upper()} E{epoch:02d}",
            leave=False,
            dynamic_ncols=True,
            disable=quiet_progress,
        )
        try:
            for batch_index, batch in enumerate(progress, start=1):
                image = batch["image"].to(device, non_blocking=bool(args.pin_memory)).float()
                masks = batch["mask"].to(device, non_blocking=bool(args.pin_memory)).float()
                labels = batch["label"].to(device, non_blocking=bool(args.pin_memory)).long()
                classes = [str(x) for x in batch["class_name"]]
                text, kg_loss, k_loss = _text_with_regularizers(model, classes, parent_config, device)
                output = forward_cir(model, image, classes, device, cir_config, domain="Industrial", require_grad=True, dataset_name=parent_config["dataset"], precomputed_text_features=text)
                if output.peer_valid.numel() and not bool(output.peer_valid.any()):
                    raise RuntimeError("CIR training has no valid K=8 peer set")
                active.append(float(output.delta.detach().abs().mean()))
                mad_values.append(float(output.delta_stats["mad"].detach().float().median()))
                delta_values.append(float(output.delta.detach().float().abs().median()))
                saturation_values.append(float((output.delta.detach().float().abs() > 0.95).float().mean()))
                cls_loss = F.cross_entropy(output.classification_logits.float(), labels)
                seg_loss = calculate_seg_loss(output.cir_training_segmentation_probability.float(), masks.float())
                base_loss = cls_loss + seg_loss + float(parent_config.get("lambda_kg", 0.001)) * kg_loss + float(parent_config.get("lambda_k", 0.0)) * k_loss
                image_anchor_loss = image_anchor.loss(model.image_adapter) if image_anchor is not None else base_loss.detach() * 0.0
                loss = base_loss + image_anchor_lambda * image_anchor_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite CIR loss at epoch={epoch} batch={batch_index}")
                if epoch == 10 and batch_index == 1:
                    image_parameters = [parameter for parameter in model.image_adapter.parameters() if parameter.requires_grad]
                    base_gradients = torch.autograd.grad(base_loss, image_parameters, retain_graph=True, allow_unused=True)
                    anchor_component = image_anchor_lambda * image_anchor_loss
                    if image_anchor is not None and anchor_component.requires_grad:
                        anchor_gradients = torch.autograd.grad(anchor_component, image_parameters, retain_graph=True, allow_unused=True)
                    else:
                        anchor_gradients = [None for _ in image_parameters]
                    base_norm = _gradient_l2(list(base_gradients))
                    anchor_norm = _gradient_l2(list(anchor_gradients))
                    gradient_probe = {"base_grad_l2": base_norm, "anchor_grad_l2": anchor_norm, "anchor_to_base_ratio": anchor_norm / max(base_norm, 1.0e-12)}
                divisor = grad_accum_window_size(batch_index, total_batches, int(args.grad_accum_steps))
                (loss / float(divisor)).backward()
                assert_phase2b_gradient_contract(model, soft_prompt_trainable=(not soft_frozen and bool(parent_config.get("soft_prompt_trainable", True))))
                if batch_index % int(args.grad_accum_steps) == 0 or batch_index == total_batches:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad and p.grad is not None], float(parent_config["grad_clip_norm"]))
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
                sums["loss"] += float(loss.detach()); sums["cls"] += float(cls_loss.detach()); sums["seg"] += float(seg_loss.detach()); sums["image_anchor"] += float(image_anchor_loss.detach()); count += 1
                elapsed_live = max(time.perf_counter() - started, 1e-9)
                vram_gb = (torch.cuda.memory_allocated(device) / float(1024 ** 3)) if device.type == "cuda" else None
                vram_text = f"{vram_gb:.1f}GB" if vram_gb is not None else "n/a"
                progress.set_postfix_str(
                    f"loss={float(loss.detach()):.3f} "
                    f"cls={float(cls_loss.detach()):.3f} "
                    f"seg={float(seg_loss.detach()):.3f} "
                    f"lr={float(optimizer.param_groups[0]['lr']):.1e} "
                    f"Δ={delta_values[-1]:.3f} "
                    f"MAD={mad_values[-1]:.4f} "
                    f"sat={100.0 * saturation_values[-1]:.2f}% "
                    f"{count * int(args.micro_batch_size) / elapsed_live:.1f} img/s "
                    f"VRAM={vram_text}"
                )
                if smoke_target is not None and global_step - smoke_start_step >= smoke_target:
                    break

        finally:
            progress.close()
        # Keep CIR identical to the canonical Phase2B timing: complete the
        # epoch, advance StepLR once, then record/save the post-step state.
        scheduler.step()
        elapsed = time.perf_counter() - started
        images_per_sec = float(count * int(args.micro_batch_size) / max(elapsed, 1e-9)) if count else 0.0
        peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        row = {
            "stage": f"CIR/TRAIN-{source.upper()}",
            "epoch": epoch,
            "mean_loss": sums["loss"] / max(count, 1),
            "mean_cls": sums["cls"] / max(count, 1),
            "mean_seg": sums["seg"] / max(count, 1),
            "mean_image_anchor": sums["image_anchor"] / max(count, 1),
            "lambda_image_anchor": image_anchor_lambda,
            "rmt_delta_abs_mean": float(np.mean(active)) if active else 0.0,
            "mad_p50": float(np.median(mad_values)) if mad_values else 0.0,
            "delta_p50": float(np.median(delta_values)) if delta_values else 0.0,
            "delta_saturation_percent": 100.0 * float(np.mean(saturation_values)) if saturation_values else 0.0,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "images_per_sec": images_per_sec,
            "peak_vram_bytes": peak_vram,
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
            "beta": beta,
            "elapsed_seconds": elapsed,
            "batches": count,
            "e10_gradient_probe": gradient_probe,
            "checkpoint_saved": False,
        }
        if epoch in candidate_epochs and smoke_target is None:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            candidate_checkpoint_payload = checkpoint_payload(model, parent_config, cir_config, source, epoch, global_step, optimizer, scheduler, generator, args.git_sha, image_anchor=image_anchor, image_anchor_lambda=image_anchor_lambda)
            checkpoint_path = checkpoint_root / f"epoch_{epoch:02d}.pth"
            write_torch_checkpoint_atomic(checkpoint_path, candidate_checkpoint_payload)
            row["checkpoint_saved"] = True
            if epoch == 10:
                e10_gate = e10_catastrophic_failure_gate(model=model, checkpoint_payload_value=candidate_checkpoint_payload, cir_config=cir_config, source=source, row=row, gradient_probe=gradient_probe, git_sha=args.git_sha)
                _write_json(run_root / "E10_CATASTROPHIC_FAILURE_GATE.json", e10_gate)
                if e10_gate["status"] != "PASS":
                    raise RuntimeError("E10 catastrophic-failure gate failed: " + "; ".join(str(reason) for reason in e10_gate["reasons"]))
            if quiet_progress and row["checkpoint_saved"]:
                print(f"MATCHED_HORIZON_CHECKPOINT epoch={epoch} path={checkpoint_path} elapsed_seconds={elapsed:.3f}", flush=True)
        run_root.mkdir(parents=True, exist_ok=True)
        write_torch_checkpoint_atomic(run_root / "last.pth", checkpoint_payload(model, parent_config, cir_config, source, epoch, global_step, optimizer, scheduler, generator, args.git_sha, image_anchor=image_anchor, image_anchor_lambda=image_anchor_lambda))
        history.append(row)
        if not quiet_progress or epoch in candidate_epochs:
            print(json.dumps(row, sort_keys=True), flush=True)
        if smoke_target is not None and global_step - smoke_start_step >= smoke_target:
            break
    if smoke_target is not None:
        completed_steps = int(global_step - smoke_start_step)
        if completed_steps < smoke_target:
            raise RuntimeError(
                f"CIR smoke reached only {completed_steps} optimizer steps; "
                f"required {smoke_target}"
            )
    else:
        completed_steps = None
    manifest = {"stage": f"CIR/TRAIN-{source.upper()}", "status": "SMOKE_PASS" if smoke_target is not None else "COMPLETED", "arch_id": cir_config["arch_id"], "source": source, "seed": int(args.seed), "config_sha256": config_sha256(cir_config), "git_sha": args.git_sha, "epochs": [row["epoch"] for row in history], "target_epochs": list(candidate_epochs), "trainable_parameters": trainable_parameter_counts(model), "image_anchor": image_anchor.metadata(image_anchor_lambda) if image_anchor is not None else {"enabled": False, "lambda_image_anchor": image_anchor_lambda, "scope": "image_adapter_parameters_only", "train_only": True}, "e10_catastrophic_failure_gate": e10_gate, "train_wall_seconds": time.perf_counter() - train_started, "max_epoch": int(parent_config["epochs"]), "candidate_checkpoint_policy": "E10/E12/E14 only in matched-horizon mode; last.pth is an atomic resume cursor", "history": history}
    _write_json(run_root / "run_manifest.json", manifest)
    if smoke_target is not None:
        _write_json(run_root / "G5_SMOKE.json", {
            "stage": "CIR/G5-SMOKE",
            "gate": "G5_REAL",
            "scope": "real",
            "real": True,
            "real_asset": True,
            "status": "PASS",
            "requested_steps": smoke_target,
            "steps_completed": completed_steps,
            "rmt_active": any(row["rmt_delta_abs_mean"] > 0 for row in history),
            "image_anchor": manifest["image_anchor"],
            "checkpoint_resume_contract": (run_root / "last.pth").is_file(),
            "identity": release_identity_fields(cir_config),
            "evidence": {"kind": "train_smoke_real", "real_execution": True, "artifact": {"checkpoint": str(run_root / "last.pth")}, "source": source, "steps_completed": completed_steps},
        })
    return {"run_root": str(run_root), "history": history, "global_step": global_step}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["visa", "mvtec"], required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cir_dfg_rmt_v1.json"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--micro-batch-size", "--batch-size", dest="micro_batch_size", type=int, default=6)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--matched-horizon-e14", action="store_true", help="Cap the run at E14 and retain only E10/E12/E14 candidate checkpoints.")
    parser.add_argument("--quiet-progress", action="store_true", help="Suppress per-batch progress; retain milestone and final records.")
    parser.add_argument("--image-anchor-checkpoint", type=Path)
    parser.add_argument("--image-anchor-lambda", type=float, default=0.0)
    parser.add_argument("--git-sha", default="WORKTREE_SHA")
    args = parser.parse_args(argv)
    if args.micro_batch_size * args.grad_accum_steps != 6:
        raise SystemExit("micro_batch_size * grad_accum_steps must equal effective batch size 6")
    train(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
