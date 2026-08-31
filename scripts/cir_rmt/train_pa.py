#!/usr/bin/env python3
"""Train the native Phase2B image-anchor control (PA).

PA is the factorial no-CIR control: it uses the canonical Phase2B training
forward and objective, with the same train-only image-parameter anchor used by
the selected CIR+anchor run.  The canonical ``train.py`` helpers are reused
directly so loader, text regularization, optimizer, clipping, prompt/DFG
schedules, and scheduler timing stay protocol-identical.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from model.checkpoint_utils import restore_rng_state, write_torch_checkpoint_atomic
from model.phase2b_legacy_bridge import assert_phase2b_gradient_contract, load_adapter_state
from model.phase2b_runtime import (
    build_phase2b_trainable,
    configure_canonical_fp32,
    forward_phase2b,
    load_json_config,
    trainable_parameter_counts,
)
from model.phase2b_schedule import grad_accum_window_size
from tools.cir_rmt.parameter_anchor import ImageParameterAnchor, load_image_parameter_anchor
from train import (
    _build_loader,
    _checkpoint_payload,
    _json_default,
    _make_optimizer,
    _set_epoch_state,
    _text_with_regularizers,
    _validate_resume,
    clip_trainable_gradients,
    seed_everything,
)
from utils import calculate_seg_loss


CONTROL_ID = "PA_PHASE2B_IMAGE_ANCHOR_V1"
EPOCHS = (10, 12, 14, 16, 18, 20)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _gradient_l2(gradients: list[torch.Tensor | None]) -> float:
    values = [gradient.detach().float().pow(2).sum() for gradient in gradients if gradient is not None]
    return float(torch.sqrt(torch.stack(values).sum()).item()) if values else 0.0


def _host_rss_bytes() -> int:
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * (1 if os.uname().sysname == "Darwin" else 1024)


def compose_pa_loss(
    base_loss: torch.Tensor,
    image_anchor_loss: torch.Tensor,
    image_anchor_lambda: float,
) -> torch.Tensor:
    """Compose PA's only intervention; lambda=0 is exactly native loss."""
    return base_loss + float(image_anchor_lambda) * image_anchor_loss


def _effective_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(load_json_config(args.config))
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.epochs is not None:
        config["epochs"] = int(args.epochs)
    config["micro_batch_size"] = int(args.micro_batch_size)
    config["batch_size"] = int(args.micro_batch_size)
    config["grad_accum_steps"] = int(args.grad_accum_steps)
    config["effective_batch_size"] = int(args.micro_batch_size) * int(args.grad_accum_steps)
    config["num_workers"] = int(args.num_workers)
    config["pin_memory"] = bool(args.pin_memory)
    config["persistent_workers"] = bool(args.persistent_workers)
    config["prefetch_factor"] = int(args.prefetch_factor)
    if config["effective_batch_size"] != 6:
        raise ValueError("micro_batch_size * grad_accum_steps must equal six")
    if str(config.get("precision")) != "fp32":
        raise ValueError("PA requires the canonical fp32 policy")
    return config


def _pa_checkpoint_payload(
    model: Any,
    config: Mapping[str, Any],
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    git_sha: str | None,
    image_anchor: ImageParameterAnchor,
    image_anchor_lambda: float,
) -> dict[str, Any]:
    payload = _checkpoint_payload(model, config, epoch, global_step, optimizer, scheduler, generator, git_sha)
    payload.update(
        {
            "control_identity": {
                "control_id": CONTROL_ID,
                "cir_training": False,
                "rmt_training": False,
                "training_forward": "native_phase2b",
                "inference_path": "native_phase2b",
                "alpha_inference": None,
            },
            "cir_training_enabled": False,
            "rmt_training_enabled": False,
            "image_anchor": image_anchor.metadata(image_anchor_lambda),
        }
    )
    return payload


def _validate_pa_resume(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    image_anchor: ImageParameterAnchor,
    image_anchor_lambda: float,
    *,
    allow_smoke_horizon_change: bool = False,
) -> None:
    resume_config = dict(config)
    if allow_smoke_horizon_change:
        saved_config = payload.get("resolved_scientific_config")
        if not isinstance(saved_config, Mapping) or "epochs" not in saved_config:
            raise ValueError("PA smoke resume checkpoint lacks its saved epoch horizon")
        resume_config["epochs"] = saved_config["epochs"]
    _validate_resume(payload, resume_config)
    control = payload.get("control_identity", {})
    if control.get("control_id") != CONTROL_ID or control.get("cir_training") is not False:
        raise ValueError("resume checkpoint is not a PA native-training control")
    metadata = payload.get("image_anchor", {})
    if abs(float(metadata.get("lambda_image_anchor", 0.0)) - float(image_anchor_lambda)) > 1e-15:
        raise ValueError("PA resume anchor lambda mismatch")
    if metadata.get("reference_checkpoint_sha256") != image_anchor.reference_checkpoint_sha256:
        raise ValueError("PA resume anchor reference mismatch")
    if metadata.get("scope") != "image_adapter_parameters_only" or metadata.get("train_only") is not True:
        raise ValueError("PA resume anchor scope mismatch")


def _gradient_probe(
    base_loss: torch.Tensor,
    weighted_anchor_loss: torch.Tensor,
    model: Any,
) -> dict[str, float]:
    parameters = [parameter for parameter in model.image_adapter.parameters() if parameter.requires_grad]
    base_gradients = torch.autograd.grad(base_loss, parameters, retain_graph=True, allow_unused=True)
    anchor_gradients = torch.autograd.grad(weighted_anchor_loss, parameters, retain_graph=True, allow_unused=True)
    base_norm = _gradient_l2(list(base_gradients))
    anchor_norm = _gradient_l2(list(anchor_gradients))
    return {
        "base_grad_l2": base_norm,
        "weighted_anchor_grad_l2": anchor_norm,
        "anchor_to_base_ratio": anchor_norm / max(base_norm, 1.0e-12),
    }


def train_pa(args: argparse.Namespace) -> dict[str, Any]:
    configure_canonical_fp32()
    config = _effective_config(args)
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = torch.device(args.device)
    if not args.visa_root.is_dir():
        raise FileNotFoundError(args.visa_root)
    if not args.clip_asset.is_file():
        raise FileNotFoundError(args.clip_asset)

    run_base = Path(args.run_root).expanduser().resolve()
    run_root = run_base / "visa" / f"seed{seed}"
    checkpoint_root = run_root / "checkpoints"
    last_path = run_root / "last.pth"
    manifest_path = run_root / "run_manifest.json"
    progress_path = run_root / "PROGRESS.json"
    run_root.mkdir(parents=True, exist_ok=True)

    model = build_phase2b_trainable(config, args.clip_asset, device)
    image_anchor = load_image_parameter_anchor(args.image_anchor_checkpoint, model, device)
    image_anchor_lambda = float(args.image_anchor_lambda)
    if abs(image_anchor_lambda - 0.001) > 1.0e-15:
        raise ValueError("PA requires the frozen image-anchor lambda=0.001")
    generator = torch.Generator()
    generator.manual_seed(seed)
    _dataset, loader, generator = _build_loader(args, config, generator)
    optimizer = _make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=float(config["lr_gamma"]))

    start_epoch = 1
    global_step = 0
    resume_epoch: int | None = None
    if args.resume is not None:
        # Keep RNG tensors on CPU; restore_rng_state requires a CPU ByteTensor.
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        _validate_pa_resume(
            payload,
            config,
            image_anchor,
            image_anchor_lambda,
            allow_smoke_horizon_change=args.smoke_steps is not None,
        )
        load_adapter_state(model, payload)
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        restore_rng_state(payload, dataloader_generator=generator)
        resume_epoch = int(payload["epoch"])
        start_epoch = resume_epoch + 1
        global_step = int(payload.get("global_step", 0))

    prior_manifest: dict[str, Any] = {}
    if args.resume is not None and manifest_path.is_file():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    history: list[dict[str, Any]] = list(prior_manifest.get("history", []))
    probe_epochs = {10, 16, 18, 20}
    smoke_target = None if args.smoke_steps is None else int(args.smoke_steps)
    if smoke_target is not None and smoke_target < 1:
        raise ValueError("--smoke-steps must be positive")
    smoke_start_step = global_step
    train_started = time.perf_counter()
    quiet = bool(args.quiet_progress)

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        alpha, beta, soft_frozen = _set_epoch_state(model, optimizer, config, epoch)
        model.train()
        model.clipmodel.eval()
        model.image_encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        sums = {"loss": 0.0, "base_loss": 0.0, "cls": 0.0, "seg": 0.0, "kg": 0.0, "k": 0.0, "anchor": 0.0}
        count = 0
        gradient_probe: dict[str, float] | None = None
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        total_batches = len(loader)
        progress = tqdm(loader, desc=f"PA/TRAIN-VISA E{epoch:02d}", leave=False, disable=quiet, dynamic_ncols=True)
        try:
            for batch_index, batch in enumerate(progress, start=1):
                image = batch["image"].to(device, non_blocking=bool(args.pin_memory)).float()
                masks = batch["mask"].to(device, non_blocking=bool(args.pin_memory)).float()
                labels = batch["label"].to(device, non_blocking=bool(args.pin_memory)).long()
                classes = [str(value) for value in batch["class_name"]]
                text, kg_loss, k_loss = _text_with_regularizers(model, classes, config, device)
                output = forward_phase2b(
                    model,
                    image,
                    classes,
                    device,
                    config,
                    domain="Industrial",
                    require_grad=True,
                    dataset_name="VisA",
                    precomputed_text_features=text,
                )
                if output.training_segmentation_probability is None:
                    raise RuntimeError("PA native forward did not return training segmentation probability")
                cls_loss = F.cross_entropy(output.classification_logits.float(), labels)
                seg_loss = calculate_seg_loss(output.training_segmentation_probability.float(), masks.float())
                base_loss = cls_loss + seg_loss + float(config.get("lambda_kg", 0.001)) * kg_loss + float(config.get("lambda_k", 0.0)) * k_loss
                anchor_loss = image_anchor.loss(model.image_adapter)
                weighted_anchor_loss = image_anchor_lambda * anchor_loss
                loss = compose_pa_loss(base_loss, anchor_loss, image_anchor_lambda)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite PA loss at epoch={epoch} batch={batch_index}")
                if epoch in probe_epochs and batch_index == 1:
                    gradient_probe = _gradient_probe(base_loss, weighted_anchor_loss, model)
                divisor = grad_accum_window_size(batch_index, total_batches, int(args.grad_accum_steps))
                (loss / float(divisor)).backward()
                assert_phase2b_gradient_contract(
                    model,
                    soft_prompt_trainable=(not soft_frozen and bool(config.get("soft_prompt_trainable", True))),
                )
                should_step = (batch_index % int(args.grad_accum_steps) == 0) or batch_index == total_batches
                if should_step:
                    clip_trainable_gradients(optimizer, float(config["grad_clip_norm"]))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                sums["loss"] += float(loss.detach())
                sums["base_loss"] += float(base_loss.detach())
                sums["cls"] += float(cls_loss.detach())
                sums["seg"] += float(seg_loss.detach())
                sums["kg"] += float(kg_loss.detach())
                sums["k"] += float(k_loss.detach())
                sums["anchor"] += float(anchor_loss.detach())
                count += 1
                if smoke_target is not None and global_step - smoke_start_step >= smoke_target:
                    break
        finally:
            progress.close()

        # Canonical Phase2B timing: epoch optimizer updates -> scheduler.step
        # once -> post-step state is serialized in history/checkpoints.
        scheduler.step()
        elapsed = time.perf_counter() - epoch_started
        row: dict[str, Any] = {
            "stage": "PA/TRAIN-VISA",
            "epoch": int(epoch),
            "mean_loss": sums["loss"] / max(count, 1),
            "mean_base_loss": sums["base_loss"] / max(count, 1),
            "mean_cls": sums["cls"] / max(count, 1),
            "mean_seg": sums["seg"] / max(count, 1),
            "mean_kg": sums["kg"] / max(count, 1),
            "mean_k": sums["k"] / max(count, 1),
            "mean_anchor_loss": sums["anchor"] / max(count, 1),
            "weighted_anchor_loss": image_anchor_lambda * sums["anchor"] / max(count, 1),
            "lambda_image_anchor": image_anchor_lambda,
            "anchor_reference_distance": float(image_anchor.loss(model.image_adapter).detach().cpu()),
            "grad_base_l2": None if gradient_probe is None else gradient_probe["base_grad_l2"],
            "weighted_anchor_grad_l2": None if gradient_probe is None else gradient_probe["weighted_anchor_grad_l2"],
            "anchor_to_base_gradient_ratio": None if gradient_probe is None else gradient_probe["anchor_to_base_ratio"],
            "gradient_probe": gradient_probe,
            "alpha": alpha,
            "beta": beta,
            "soft_prompt_frozen": soft_frozen,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "scheduler_last_epoch": int(scheduler.last_epoch),
            "scheduler_step_count": int(scheduler.state_dict()["_step_count"]),
            "elapsed_seconds": elapsed,
            "samples_per_sec": float(count * int(args.micro_batch_size) / max(elapsed, 1e-9)),
            "host_rss_bytes": _host_rss_bytes(),
            "cuda": {
                "allocated_peak": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "reserved_peak": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            },
            "batches": int(count),
            "checkpoint_saved": False,
        }
        if epoch in EPOCHS and smoke_target is None:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            candidate = _pa_checkpoint_payload(
                model, config, epoch, global_step, optimizer, scheduler, generator,
                args.git_sha, image_anchor, image_anchor_lambda,
            )
            write_torch_checkpoint_atomic(checkpoint_root / f"adapter_{epoch}.pth", candidate)
            row["checkpoint_saved"] = True
        last_payload = _pa_checkpoint_payload(
            model, config, epoch, global_step, optimizer, scheduler, generator,
            args.git_sha, image_anchor, image_anchor_lambda,
        )
        write_torch_checkpoint_atomic(last_path, last_payload)
        history.append(row)
        status = "SMOKE_PASS" if smoke_target is not None else "RUNNING"
        _write_json(
            progress_path,
            {
                "stage": "PA/TRAIN-VISA",
                "status": status,
                "current_epoch": int(epoch),
                "last_completed_epoch": int(epoch),
                "latest_checkpoint": str(last_path),
                "latest_checkpoint_sha256": __import__("hashlib").sha256(last_path.read_bytes()).hexdigest(),
                "seconds_per_epoch": float(elapsed),
                "images_per_second": float(row["samples_per_sec"]),
                "peak_vram_allocated": row["cuda"]["allocated_peak"],
                "peak_vram_reserved": row["cuda"]["reserved_peak"],
                "host_rss": row["host_rss_bytes"],
                "last_error_class": None,
                "recovery_count": int(prior_manifest.get("recovery_count", 0)),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        if not quiet or row["checkpoint_saved"]:
            print(json.dumps(row, sort_keys=True, default=_json_default), flush=True)
        if smoke_target is not None and global_step - smoke_start_step >= smoke_target:
            break

    completed_steps = None if smoke_target is None else int(global_step - smoke_start_step)
    if smoke_target is not None and completed_steps < smoke_target:
        raise RuntimeError(f"PA smoke reached {completed_steps} optimizer steps; required {smoke_target}")
    run_manifest = {
        "control_id": CONTROL_ID,
        "status": "SMOKE_PASS" if smoke_target is not None else "COMPLETED",
        "source": "visa",
        "seed": seed,
        "git_sha": args.git_sha,
        "implementation_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": config,
        "config_sha256": __import__("hashlib").sha256(json.dumps(config, sort_keys=True, default=_json_default).encode()).hexdigest(),
        "parent_config_sha256": __import__("hashlib").sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), default=_json_default).encode()).hexdigest(),
        "precision": "fp32",
        "amp": False,
        "tf32": False,
        "effective_batch_size": int(config["effective_batch_size"]),
        "candidate_epochs": list(EPOCHS),
        "image_anchor": image_anchor.metadata(image_anchor_lambda),
        "control_identity": {
            "cir_training": False,
            "rmt_training": False,
            "training_forward": "native_phase2b",
            "inference_path": "native_phase2b",
        },
        "resume_from_epoch": resume_epoch,
        "history": history,
        "train_wall_seconds": time.perf_counter() - train_started,
        "smoke_steps": completed_steps,
        "target_tuning_occurred": False,
        "mvtec": "NOT_RUN",
        "status_detail": "new PA scientific control; no P/C_OLD/A checkpoints were used as model initialization",
    }
    _write_json(manifest_path, run_manifest)
    _write_json(
        progress_path,
        {
            "stage": "PA/TRAIN-VISA",
            "status": run_manifest["status"],
            "current_epoch": int(history[-1]["epoch"]) if history else 0,
            "last_completed_epoch": int(history[-1]["epoch"]) if history else 0,
            "latest_checkpoint": str(last_path),
            "latest_checkpoint_sha256": __import__("hashlib").sha256(last_path.read_bytes()).hexdigest(),
            "recovery_count": int(prior_manifest.get("recovery_count", 0)),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return {"run_root": str(run_root), "history": history, "global_step": global_step}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2b_canonical_v1.json"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--image-anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--image-anchor-lambda", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--micro-batch-size", "--batch-size", dest="micro_batch_size", type=int, default=6)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--git-sha", default="WORKTREE_SHA")
    args = parser.parse_args(argv)
    if args.micro_batch_size * args.grad_accum_steps != 6:
        raise SystemExit("micro_batch_size * grad_accum_steps must equal effective batch size 6")
    if int(args.epochs) != 20 and args.smoke_steps is None:
        raise SystemExit("the PA scientific control must train through E20")
    train_pa(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
