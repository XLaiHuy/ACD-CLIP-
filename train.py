#!/usr/bin/env python3
"""Canonical VisA Phase2B trainer with resumable FP32 execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import TextAndImageDataset
from model.checkpoint_utils import capture_rng_state, restore_rng_state, write_torch_checkpoint_atomic
from model.phase2b_legacy_bridge import assert_phase2b_gradient_contract
from model.phase2b_runtime import (
    build_phase2b_trainable,
    configure_canonical_fp32,
    forward_phase2b,
    load_json_config,
    trainable_parameter_counts,
)
from model.phase2b_schedule import (
    apply_soft_prompt_lr_policy,
    get_dfg_beta_for_epoch,
    get_hybrid_alpha_for_epoch,
    grad_accum_window_size,
    scientific_config,
)
from utils import calculate_seg_loss, get_hybrid_soft_prompt_single_class_text_embedding, get_multiple_adapted_single_class_text_embedding, make_dataloader_generator, seed_worker


# Keep these aliases importable for bounded schedule/loader tests and audit tools.
__all__ = [
    "get_dfg_beta_for_epoch", "get_hybrid_alpha_for_epoch", "grad_accum_window_size",
    "train_phase2b", "main",
]


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _path_identity(path: Path) -> dict[str, str]:
    resolved = str(path.expanduser().resolve())
    return {"basename": path.name, "resolved_path_sha256": _sha256_bytes(resolved.encode("utf-8"))}


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _build_loader(args: argparse.Namespace, config: Mapping[str, Any], generator: torch.Generator | None = None):
    metadata_path = Path(__file__).resolve().parent / "dataset/hub/VisA.jsonl"
    dataset = TextAndImageDataset(str(args.visa_root), str(metadata_path), int(config["img_size"]))
    if generator is None:
        generator = make_dataloader_generator(int(config.get("seed", 0)))
    workers = int(args.num_workers)
    kwargs: dict[str, Any] = {
        "batch_size": int(args.micro_batch_size),
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": bool(args.pin_memory),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return dataset, DataLoader(dataset, **kwargs), generator


def _sample_weighted_regularizer(
    values_by_class: Mapping[str, torch.Tensor | float],
    class_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Average a per-class regularizer with one weight per batch sample."""
    if not class_names or not values_by_class:
        return torch.zeros((), device=device)
    counts = Counter(class_names)
    weighted: torch.Tensor | None = None
    for class_name, count in counts.items():
        value = values_by_class[class_name]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32, device=device)
        else:
            value = value.to(device=device)
        contribution = value * float(count)
        weighted = contribution if weighted is None else weighted + contribution
    if weighted is None:
        return torch.zeros((), device=device)
    return weighted / float(len(class_names))


def _text_with_regularizers(model: Any, class_names: list[str], config: Mapping[str, Any], device: torch.device):
    from utils import get_phase2b_global_text_features

    dataset_name = str(config.get("dataset", "VisA"))
    unique = list(dict.fromkeys(class_names))
    by_class: dict[str, torch.Tensor] = {}
    kg_by_class: dict[str, torch.Tensor] = {}
    for class_name in unique:
        if bool(config.get("use_hybrid_soft_prompt", False)):
            text, kg_loss, _stats = get_hybrid_soft_prompt_single_class_text_embedding(
                model, dataset_name, class_name, device, return_kg=True,
            )
            kg_by_class[class_name] = kg_loss
        elif bool(config.get("use_soft_prompt", False)):
            from utils import get_soft_prompt_single_class_text_embedding
            text, kg_loss, _stats = get_soft_prompt_single_class_text_embedding(model, dataset_name, class_name, device, return_kg=True)
            kg_by_class[class_name] = kg_loss
        else:
            text = get_multiple_adapted_single_class_text_embedding(model, dataset_name, class_name, device)
        by_class[class_name] = text
    features = torch.stack([by_class[name] for name in class_names], dim=0).permute(1, 0, 2, 3).float()
    kg_loss = _sample_weighted_regularizer(kg_by_class, class_names, device)
    # lambda_k is preregistered as zero.  Keep an explicit tensor in the graph
    # so the task objective remains visible and can be enabled only by a new
    # scientific protocol, never implicitly.
    k_loss = torch.zeros((), device=device)
    return features, kg_loss, k_loss


def _trainable_parameters_with_grad(optimizer: torch.optim.Optimizer) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            identity = id(parameter)
            if parameter.requires_grad and parameter.grad is not None and identity not in seen:
                seen.add(identity)
                parameters.append(parameter)
    return parameters


def clip_trainable_gradients(optimizer: torch.optim.Optimizer, max_norm: float) -> torch.Tensor:
    """Clip only unique trainable parameters that received gradients."""
    parameters = _trainable_parameters_with_grad(optimizer)
    if not parameters:
        return torch.zeros((), dtype=torch.float32)
    return torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))


def _checkpoint_payload(
    model: Any,
    config: Mapping[str, Any],
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    git_sha: str | None,
) -> dict[str, Any]:
    resolved = dict(config)
    config_hash = _sha256_bytes(json.dumps(resolved, sort_keys=True, default=_json_default).encode("utf-8"))
    payload: dict[str, Any] = {
        "checkpoint_version": 2,
        "protocol_version": resolved["protocol_version"],
        "epoch": int(epoch),
        "global_step": int(global_step),
        "git_sha": git_sha,
        "model_name": resolved["model_name"],
        "img_size": int(resolved["img_size"]),
        "n_groups": int(resolved["n_groups"]),
        "precision": "fp32",
        "amp_enabled": False,
        "tf32_enabled": False,
        "resolved_scientific_config": scientific_config(resolved),
        "config_sha256": config_hash,
        "dfg_mode": resolved["dfg_mode"],
        "dfg_attn_dim": resolved.get("dfg_attn_dim"),
        "dfg_attn_tau": resolved.get("dfg_attn_tau"),
        "use_ss2d_dfg": bool(resolved.get("use_ss2d_dfg", False)),
        "dfg_gamma_max": resolved.get("dfg_gamma_max"),
        "dfg_ss2d_fusion": resolved.get("dfg_ss2d_fusion"),
        "dfg_beta": resolved.get("dfg_beta"),
        "dfg_beta_current": float(getattr(model, "dfg_beta", resolved.get("dfg_beta", 0.0))),
        "dfg_beta_schedule": resolved.get("dfg_beta_schedule"),
        "dfg_beta_target": resolved.get("dfg_beta_target"),
        "use_hybrid_soft_prompt": bool(getattr(model, "use_hybrid_soft_prompt", False)),
        "use_soft_prompt": bool(getattr(model, "use_soft_prompt", False)),
        "hybrid_alpha_current": float(getattr(model, "hybrid_alpha_current", 0.0)),
        "hybrid_alpha_max": float(getattr(model, "hybrid_alpha_max", resolved.get("hybrid_alpha_max", 0.2))),
        "soft_prompt_freeze_epochs": int(getattr(model, "soft_prompt_freeze_epochs", resolved.get("soft_prompt_freeze_epochs", 3))),
        "prompt_mode": str(getattr(model, "prompt_mode", "hard")),
        "image_adapter": _cpu_state_dict(model.image_adapter),
        "text_adapter": _cpu_state_dict(model.text_adapter),
        "soft_prompt": _cpu_state_dict(model.soft_prompt),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    payload.update(capture_rng_state(dataloader_generator=generator))
    return payload


def _validate_resume(checkpoint: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    expected = scientific_config(config)
    actual = checkpoint.get("resolved_scientific_config")
    if actual is None:
        raise ValueError("resume checkpoint lacks resolved scientific config identity")
    if json.dumps(actual, sort_keys=True, default=_json_default) != json.dumps(expected, sort_keys=True, default=_json_default):
        raise ValueError("resume checkpoint scientific configuration is incompatible")
    if str(checkpoint.get("protocol_version")) != str(config.get("protocol_version")):
        raise ValueError("resume checkpoint protocol mismatch")
    if checkpoint.get("precision") != "fp32" or checkpoint.get("amp_enabled") is True or checkpoint.get("tf32_enabled") is True:
        raise ValueError("resume checkpoint violates canonical FP32 policy")


def _make_optimizer(model: Any, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    groups = [
        {"name": "image_adapter", "params": list(model.image_adapter.parameters()), "lr": float(config["image_lr"])},
        {"name": "text_adapter", "params": list(model.text_adapter.parameters()), "lr": float(config["text_lr"])},
        {"name": "soft_prompt", "params": list(model.soft_prompt.parameters()), "lr": float(config["soft_prompt_lr"]), "constant_lr": float(config["soft_prompt_lr"])},
    ]
    return torch.optim.Adam(groups)


def _set_epoch_state(model: Any, optimizer: torch.optim.Optimizer, config: Mapping[str, Any], epoch: int) -> tuple[float, float, bool]:
    alpha = get_hybrid_alpha_for_epoch(epoch, float(config.get("hybrid_alpha_max", 0.2)), int(config.get("soft_prompt_freeze_epochs", 3))) if bool(config.get("use_hybrid_soft_prompt", False)) else 0.0
    frozen = bool(config.get("use_hybrid_soft_prompt", False) and epoch <= int(config.get("soft_prompt_freeze_epochs", 3)))
    model.hybrid_alpha_current = float(alpha)
    model.prompt_mode = "hybrid" if config.get("use_hybrid_soft_prompt", False) else "soft" if config.get("use_soft_prompt", False) else "hard"
    model.soft_prompt.requires_grad_(not frozen and bool(config.get("soft_prompt_trainable", True)))
    model.text_adapter.requires_grad_(True)
    beta = get_dfg_beta_for_epoch(epoch, str(config.get("dfg_beta_schedule", "fixed")), float(config.get("dfg_beta_target", config.get("dfg_beta", 0.1))), float(config.get("dfg_beta", 0.1)))
    model.set_dfg_beta(beta)
    apply_soft_prompt_lr_policy(optimizer, frozen)
    return float(alpha), float(beta), frozen


def _cuda_stats(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"allocated_peak": None, "reserved_peak": None, "total_vram": None, "vram_percent": None}
    props = torch.cuda.get_device_properties(device)
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "allocated_peak": allocated,
        "reserved_peak": reserved,
        "total_vram": int(props.total_memory),
        "vram_percent": float(100.0 * reserved / max(props.total_memory, 1)),
    }


def _run_batch_preflight(args: argparse.Namespace, config: dict[str, Any], device: torch.device) -> int:
    if not args.clip_asset.exists() or not args.visa_root.exists():
        print("PERFORMANCE_PROBE=NOT_RUN_NO_ASSETS")
        return 0
    configure_canonical_fp32()
    seed_everything(int(config.get("seed", 0)))
    model = None
    try:
        model = build_phase2b_trainable(config, args.clip_asset, device)
        _dataset, loader, _generator = _build_loader(args, config)
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        timings = {"forward": [], "backward": [], "step": []}
        iterator = iter(loader)
        max_steps = min(2, max(0, int(args.smoke_steps or 2)))
        optimizer = _make_optimizer(model, config)
        _set_epoch_state(model, optimizer, config, epoch=1)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(max_steps):
            batch = next(iterator)
            image = batch["image"].to(device, non_blocking=bool(args.pin_memory)).float()
            masks = batch["mask"].to(device, non_blocking=bool(args.pin_memory)).float()
            labels = batch["label"].to(device, non_blocking=bool(args.pin_memory)).long()
            classes = list(batch["class_name"])
            started = time.perf_counter()
            text, kg, k = _text_with_regularizers(model, classes, config, device)
            result = forward_phase2b(model, image, classes, device, config, domain="Industrial", require_grad=True, dataset_name="VisA", precomputed_text_features=text)
            forward_done = time.perf_counter()
            loss = F.cross_entropy(result.classification_logits.float(), labels) + calculate_seg_loss(result.training_segmentation_probability, masks) + float(config.get("lambda_kg", 0.001)) * kg + float(config.get("lambda_k", 0.0)) * k
            loss.backward()
            backward_done = time.perf_counter()
            assert_phase2b_gradient_contract(model, soft_prompt_trainable=False)
            clip_trainable_gradients(optimizer, float(config.get("grad_clip_norm", 1.0)))
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step_done = time.perf_counter()
            timings["forward"].append(forward_done - started); timings["backward"].append(backward_done - forward_done); timings["step"].append(step_done - backward_done)
        stats = _cuda_stats(device)
        elapsed = sum(timings["forward"]) + sum(timings["backward"]) + sum(timings["step"])
        samples = max_steps * int(args.micro_batch_size)
        print(json.dumps({"requested_micro_batch": int(args.micro_batch_size), "accumulation": int(args.grad_accum_steps), "effective_batch": int(args.micro_batch_size) * int(args.grad_accum_steps), "forward_time": float(np.mean(timings["forward"])), "backward_time": float(np.mean(timings["backward"])), "step_time": float(np.mean(timings["step"])), "samples_per_sec": float(samples / max(elapsed, 1e-9)), "cuda_allocated_peak": stats["allocated_peak"], "cuda_reserved_peak": stats["reserved_peak"], "total_gpu_vram": stats["total_vram"], "vram_percent": stats["vram_percent"], "oom": False}, sort_keys=True))
        print("BATCH6_PREFLIGHT=PASS")
        return 0
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if device.type == "cuda":
            stats = _cuda_stats(device); torch.cuda.empty_cache()
        else:
            stats = _cuda_stats(device)
        print(json.dumps({"requested_micro_batch": int(args.micro_batch_size), "accumulation": int(args.grad_accum_steps), "effective_batch": int(args.micro_batch_size) * int(args.grad_accum_steps), "cuda_allocated_peak": stats["allocated_peak"], "cuda_reserved_peak": stats["reserved_peak"], "total_gpu_vram": stats["total_vram"], "vram_percent": stats["vram_percent"], "oom": True}, sort_keys=True))
        print("BATCH6_PREFLIGHT=FAIL_OOM")
        return 0
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def train_phase2b(args: argparse.Namespace) -> dict[str, Any]:
    configure_canonical_fp32()
    config = dict(load_json_config(args.config))
    if args.seed is not None: config["seed"] = int(args.seed)
    if args.epochs is not None: config["epochs"] = int(args.epochs)
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
        raise ValueError("canonical trainer requires fp32")
    seed_everything(int(config.get("seed", 0)))
    device = torch.device(args.device)
    if args.preflight_batch:
        return {"preflight_exit": _run_batch_preflight(args, config, device)}
    model = build_phase2b_trainable(config, args.clip_asset, device)
    generator = make_dataloader_generator(int(config.get("seed", 0)))
    _dataset, loader, generator = _build_loader(args, config, generator)
    optimizer = _make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=float(config["lr_gamma"]))
    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        _validate_resume(checkpoint, config)
        from model.phase2b_legacy_bridge import load_adapter_state
        load_adapter_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng_state(checkpoint, dataloader_generator=generator)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
    run_root = Path(args.run_root)
    checkpoint_root = run_root / "phase2b" / "checkpoints"
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        alpha, beta, soft_frozen = _set_epoch_state(model, optimizer, config, epoch)
        model.train(); model.clipmodel.eval(); model.image_encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        sums = {"loss": 0.0, "cls": 0.0, "seg": 0.0, "kg": 0.0, "k": 0.0}
        count = 0
        epoch_started = time.perf_counter()
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        progress = tqdm(loader, desc=f"E{epoch:02d}/{int(config['epochs'])}", leave=False)
        total_batches = len(loader)
        for batch_index, batch in enumerate(progress, start=1):
            image = batch["image"].to(device, non_blocking=bool(args.pin_memory)).float()
            masks = batch["mask"].to(device, non_blocking=bool(args.pin_memory)).float()
            labels = batch["label"].to(device, non_blocking=bool(args.pin_memory)).long()
            classes = list(batch["class_name"])
            text, kg_loss, k_loss = _text_with_regularizers(model, classes, config, device)
            result = forward_phase2b(model, image, classes, device, config, domain="Industrial", require_grad=True, dataset_name="VisA", precomputed_text_features=text)
            if result.training_segmentation_probability is None:
                raise RuntimeError("training forward did not return segmentation probability")
            cls_loss = F.cross_entropy(result.classification_logits.float(), labels.long())
            seg_loss = calculate_seg_loss(result.training_segmentation_probability.float(), masks.float())
            task_loss = cls_loss + seg_loss + float(config.get("lambda_kg", 0.001)) * kg_loss + float(config.get("lambda_k", 0.0)) * k_loss
            if not torch.isfinite(task_loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch} batch={batch_index}")
            divisor = grad_accum_window_size(batch_index, total_batches, int(args.grad_accum_steps))
            (task_loss / float(divisor)).backward()
            assert_phase2b_gradient_contract(
                model,
                soft_prompt_trainable=(not soft_frozen and bool(config.get("soft_prompt_trainable", True))),
            )
            should_step = (batch_index % int(args.grad_accum_steps) == 0) or batch_index == total_batches
            if should_step:
                clip_trainable_gradients(optimizer, float(config["grad_clip_norm"]))
                optimizer.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
            sums["loss"] += float(task_loss.detach()); sums["cls"] += float(cls_loss.detach()); sums["seg"] += float(seg_loss.detach()); sums["kg"] += float(kg_loss.detach()); sums["k"] += float(k_loss.detach()); count += 1
            elapsed = max(time.perf_counter() - epoch_started, 1e-9)
            postfix = {"loss": f"{sums['loss']/count:.4f}", "cls": f"{sums['cls']/count:.4f}", "seg": f"{sums['seg']/count:.4f}", "kg": f"{sums['kg']/count:.4f}", "alpha": f"{alpha:.3f}", "beta": f"{beta:.3f}", "bs": int(args.micro_batch_size), "img/s": f"{count*int(args.micro_batch_size)/elapsed:.1f}"}
            if device.type == "cuda": postfix["vram"] = f"{torch.cuda.memory_reserved(device)/1024**3:.1f}G"
            progress.set_postfix(postfix)
        scheduler.step()
        elapsed = time.perf_counter() - epoch_started
        row = {"epoch": int(epoch), "mean_loss": sums["loss"] / max(count, 1), "mean_cls": sums["cls"] / max(count, 1), "mean_seg": sums["seg"] / max(count, 1), "mean_kg": sums["kg"] / max(count, 1), "mean_k": sums["k"] / max(count, 1), "alpha": alpha, "beta": beta, "soft_prompt_frozen": soft_frozen, "learning_rates": [float(group["lr"]) for group in optimizer.param_groups], "elapsed_seconds": elapsed, "samples_per_sec": float(count * int(args.micro_batch_size) / max(elapsed, 1e-9)), "cuda": _cuda_stats(device), "checkpoint_saved": False}
        if epoch in tuple(int(item) for item in config["candidate_epochs"]):
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            payload = _checkpoint_payload(model, config, epoch, global_step, optimizer, scheduler, generator, args.git_sha)
            candidate_path = checkpoint_root / f"adapter_{epoch}.pth"
            write_torch_checkpoint_atomic(candidate_path, payload)
            row["checkpoint_saved"] = True
        run_root.mkdir(parents=True, exist_ok=True)
        write_torch_checkpoint_atomic(run_root / "phase2b" / "last.pth", _checkpoint_payload(model, config, epoch, global_step, optimizer, scheduler, generator, args.git_sha))
        history.append(row)
        print(f"epoch={epoch} mean_loss={row['mean_loss']:.6f} mean_cls={row['mean_cls']:.6f} mean_seg={row['mean_seg']:.6f} alpha={alpha:.3f} beta={beta:.3f} elapsed={elapsed:.1f}s checkpoint_saved={row['checkpoint_saved']}")
    config_bytes = json.dumps(config, sort_keys=True, default=_json_default).encode("utf-8")
    counts = trainable_parameter_counts(model)
    _write_json(run_root / "phase2b" / "config_resolved.json", config)
    _write_json(run_root / "phase2b" / "run_manifest.json", {"protocol_version": config["protocol_version"], "git_sha": args.git_sha, "dataset_role": "VisA_TRAIN", "precision": "fp32", "amp": False, "tf32": False, "effective_batch_size": config["effective_batch_size"], "micro_batch_size": config["micro_batch_size"], "grad_accum_steps": config["grad_accum_steps"], "num_workers": config["num_workers"], "pin_memory": config["pin_memory"], "persistent_workers": config["persistent_workers"], "prefetch_factor": config["prefetch_factor"], "dataset_root_identity": _path_identity(Path(args.visa_root)), "config_sha256": _sha256_bytes(config_bytes), "trainable_parameter_count": counts["trainable"], "frozen_parameter_count": counts["frozen"], "candidate_epochs": list(config["candidate_epochs"]), "history": history, "status": "COMPLETED"})
    return {"run_root": str(run_root), "history": history, "global_step": global_step}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2b_canonical_v1.json"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--micro-batch-size", "--batch-size", dest="micro_batch_size", type=int, default=6)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-batch", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--git-sha", default="WORKTREE_SHA")
    args = parser.parse_args(argv)
    if args.micro_batch_size * args.grad_accum_steps != 6:
        raise SystemExit("micro_batch_size * grad_accum_steps must equal effective batch size 6")
    result = train_phase2b(args)
    if args.preflight_batch:
        return int(result.get("preflight_exit", 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
