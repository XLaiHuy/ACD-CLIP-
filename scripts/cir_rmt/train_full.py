#!/usr/bin/env python3
"""CIR/TRAIN-{source}: train the frozen parent adapter with CIR in one path."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from dataset import TextAndImageDataset
from dataset.info import CLASS_NAMES
from model.checkpoint_utils import capture_rng_state, restore_rng_state, write_torch_checkpoint_atomic
from model.phase2b_legacy_bridge import assert_phase2b_gradient_contract, load_adapter_state
from model.phase2b_runtime import build_phase2b_trainable, configure_canonical_fp32, trainable_parameter_counts
from model.phase2b_schedule import apply_soft_prompt_lr_policy, get_dfg_beta_for_epoch, get_hybrid_alpha_for_epoch, grad_accum_window_size
from tools.cir_rmt.identity import checkpoint_metadata, config_sha256, load_cir_config, validate_checkpoint_identity
from tools.cir_rmt.runtime import forward_cir
from utils import calculate_seg_loss, make_dataloader_generator, seed_worker


TARGET_EPOCHS = (12, 14, 16, 18, 20)
IMAGE_SIZE = 518
MVTEC_CLASSES = tuple(CLASS_NAMES["MVTec"])


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


class MVTecTrainDataset(Dataset):
    """Normal-only MVTec train split; labels/masks never enter peer selection."""
    def __init__(self, root: Path, image_size: int = IMAGE_SIZE):
        self.root = (root / "mvtec_anomaly_detection") if (root / "mvtec_anomaly_detection").is_dir() else root
        self.samples = []
        for class_name in MVTEC_CLASSES:
            good = self.root / class_name / "train" / "good"
            if not good.is_dir():
                raise FileNotFoundError(f"MVTec normal train directory missing: {good}")
            for path in sorted(good.iterdir()):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((path, class_name))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        if not self.samples:
            raise ValueError("MVTec train split is empty")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, class_name = self.samples[int(index)]
        with Image.open(path) as handle:
            image = self.transform(handle.convert("RGB")).contiguous()
        return {"image": image, "mask": torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32), "label": torch.tensor(0, dtype=torch.int64), "class_name": class_name, "file_name": str(path), "local_mask_valid": torch.ones_like(torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE)))}


def build_loader(source: str, source_root: Path, config: Mapping[str, Any], args: argparse.Namespace, generator: torch.Generator):
    if source == "visa":
        metadata = Path(__file__).resolve().parents[2] / "dataset/hub/VisA.jsonl"
        dataset = TextAndImageDataset(str(source_root), str(metadata), int(config["img_size"]))
    elif source == "mvtec":
        dataset = MVTecTrainDataset(source_root, int(config["img_size"]))
    else:
        raise ValueError("source must be visa or mvtec")
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


def checkpoint_payload(model: Any, parent_config: Mapping[str, Any], cir_config: Mapping[str, Any], source: str, epoch: int, step: int, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, generator: torch.Generator, git_sha: str | None) -> dict[str, Any]:
    payload = dict(checkpoint_metadata(cir_config, source_dataset=source, epoch=epoch, git_sha=git_sha))
    payload.update({
        "checkpoint_version": 1, "parent_protocol": parent_config["protocol_version"], "model_name": parent_config["model_name"], "img_size": int(parent_config["img_size"]), "precision": "fp32", "amp_enabled": False, "tf32_enabled": False,
        "global_step": int(step), "parent_config": dict(parent_config), "parent_config_sha256": str(cir_config["parent_config_sha256"]),
        "resolved_scientific_config": dict(parent_config), "image_adapter": _cpu_state(model.image_adapter), "text_adapter": _cpu_state(model.text_adapter), "soft_prompt": _cpu_state(model.soft_prompt),
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
    })
    payload.update(capture_rng_state(dataloader_generator=generator))
    return payload


def validate_resume(payload: Mapping[str, Any], cir_config: Mapping[str, Any], source: str, git_sha: str | None) -> None:
    validate_checkpoint_identity(payload, cir_config, source_dataset=source, expected_git_sha=git_sha or payload.get("git_sha"))
    if payload.get("precision") != "fp32" or payload.get("amp_enabled") is True or payload.get("tf32_enabled") is True:
        raise ValueError("CIR resume violates FP32 contract")


def train(args: argparse.Namespace) -> dict[str, Any]:
    cir_config = load_cir_config(args.config)
    parent_path = Path(cir_config.get("parent_config_path", "configs/phase2b_canonical_v1.json"))
    if not parent_path.is_absolute():
        parent_path = Path(__file__).resolve().parents[2] / parent_path
    parent_config = json.loads(parent_path.read_text(encoding="utf-8"))
    source = str(args.source).lower()
    parent_config.update({"dataset": "VisA" if source == "visa" else "MVTec", "seed": int(args.seed), "epochs": int(args.epochs or parent_config.get("epochs", 20)), "micro_batch_size": int(args.micro_batch_size), "batch_size": int(args.micro_batch_size), "grad_accum_steps": int(args.grad_accum_steps), "effective_batch_size": int(args.micro_batch_size * args.grad_accum_steps), "num_workers": int(args.num_workers), "pin_memory": bool(args.pin_memory), "persistent_workers": bool(args.persistent_workers), "prefetch_factor": int(args.prefetch_factor)})
    if parent_config["effective_batch_size"] != 6:
        raise ValueError("canonical Phase2B effective batch size must be six")
    if source == "visa" and not args.source_root.is_dir():
        raise FileNotFoundError(args.source_root)
    if source == "mvtec" and not args.source_root.is_dir():
        raise FileNotFoundError(args.source_root)
    configure_canonical_fp32(); seed_everything(int(args.seed))
    device = torch.device(args.device)
    model = build_phase2b_trainable(parent_config, args.clip_asset, device)
    generator = make_dataloader_generator(int(args.seed))
    dataset, loader = build_loader(source, args.source_root, parent_config, args, generator)
    optimizer = _optimizer(model, parent_config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=float(parent_config["lr_gamma"]))
    start_epoch, global_step = 1, 0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        validate_resume(payload, cir_config, source, args.git_sha)
        load_adapter_state(model, payload); optimizer.load_state_dict(payload["optimizer_state"]); scheduler.load_state_dict(payload["scheduler_state"]); restore_rng_state(payload, dataloader_generator=generator)
        start_epoch, global_step = int(payload["epoch"]) + 1, int(payload.get("global_step", 0))
    run_root = Path(args.run_root) / source / f"seed{int(args.seed)}"
    checkpoint_root = run_root / "checkpoints"
    history = []
    smoke_target = None if args.smoke_steps is None else int(args.smoke_steps)
    if smoke_target is not None and smoke_target < 1:
        raise ValueError("--smoke-steps must request at least one optimizer step")
    smoke_start_step = int(global_step)
    for epoch in range(start_epoch, int(parent_config["epochs"]) + 1):
        _, beta, soft_frozen = _set_epoch_state(model, optimizer, parent_config, epoch)
        model.train(); model.clipmodel.eval(); model.image_encoder.eval(); optimizer.zero_grad(set_to_none=True)
        sums = {"loss": 0.0, "cls": 0.0, "seg": 0.0}; count = 0; active = []; started = time.perf_counter()
        iterator = iter(loader); total_batches = len(loader)
        for batch_index, batch in enumerate(tqdm(loader, desc=f"CIR/TRAIN-{source.upper()} E{epoch:02d}", leave=False), start=1):
            image = batch["image"].to(device, non_blocking=bool(args.pin_memory)).float()
            masks = batch["mask"].to(device, non_blocking=bool(args.pin_memory)).float()
            labels = batch["label"].to(device, non_blocking=bool(args.pin_memory)).long()
            classes = [str(x) for x in batch["class_name"]]
            text, kg_loss, k_loss = _text_with_regularizers(model, classes, parent_config, device)
            output = forward_cir(model, image, classes, device, cir_config, domain="Industrial", require_grad=True, dataset_name=parent_config["dataset"], precomputed_text_features=text)
            if output.peer_valid.numel() and not bool(output.peer_valid.any()):
                raise RuntimeError("CIR training has no valid K=8 peer set")
            active.append(float(output.delta.detach().abs().mean()))
            cls_loss = F.cross_entropy(output.classification_logits.float(), labels)
            seg_loss = calculate_seg_loss(output.cir_training_segmentation_probability.float(), masks.float())
            loss = cls_loss + seg_loss + float(parent_config.get("lambda_kg", 0.001)) * kg_loss + float(parent_config.get("lambda_k", 0.0)) * k_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite CIR loss at epoch={epoch} batch={batch_index}")
            divisor = grad_accum_window_size(batch_index, total_batches, int(args.grad_accum_steps))
            (loss / float(divisor)).backward()
            assert_phase2b_gradient_contract(model, soft_prompt_trainable=(not soft_frozen and bool(parent_config.get("soft_prompt_trainable", True))))
            if batch_index % int(args.grad_accum_steps) == 0 or batch_index == total_batches:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad and p.grad is not None], float(parent_config["grad_clip_norm"]))
                optimizer.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
            sums["loss"] += float(loss.detach()); sums["cls"] += float(cls_loss.detach()); sums["seg"] += float(seg_loss.detach()); count += 1
            if smoke_target is not None and global_step - smoke_start_step >= smoke_target:
                break
        elapsed = time.perf_counter() - started
        row = {"stage": f"CIR/TRAIN-{source.upper()}", "epoch": epoch, "mean_loss": sums["loss"] / max(count, 1), "mean_cls": sums["cls"] / max(count, 1), "mean_seg": sums["seg"] / max(count, 1), "rmt_delta_abs_mean": float(np.mean(active)) if active else 0.0, "beta": beta, "elapsed_seconds": elapsed, "checkpoint_saved": False}
        if epoch in TARGET_EPOCHS and smoke_target is None:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            write_torch_checkpoint_atomic(checkpoint_root / f"epoch_{epoch:02d}.pth", checkpoint_payload(model, parent_config, cir_config, source, epoch, global_step, optimizer, scheduler, generator, args.git_sha))
            row["checkpoint_saved"] = True
        run_root.mkdir(parents=True, exist_ok=True)
        write_torch_checkpoint_atomic(run_root / "last.pth", checkpoint_payload(model, parent_config, cir_config, source, epoch, global_step, optimizer, scheduler, generator, args.git_sha))
        history.append(row)
        print(json.dumps(row, sort_keys=True))
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
    manifest = {"stage": f"CIR/TRAIN-{source.upper()}", "status": "SMOKE_PASS" if smoke_target is not None else "COMPLETED", "arch_id": cir_config["arch_id"], "source": source, "seed": int(args.seed), "config_sha256": config_sha256(cir_config), "git_sha": args.git_sha, "epochs": [row["epoch"] for row in history], "target_epochs": list(TARGET_EPOCHS), "trainable_parameters": trainable_parameter_counts(model), "history": history}
    _write_json(run_root / "run_manifest.json", manifest)
    if smoke_target is not None:
        _write_json(run_root / "G5_SMOKE.json", {"stage": "CIR/G5-SMOKE", "status": "PASS", "requested_steps": smoke_target, "steps_completed": completed_steps, "rmt_active": any(row["rmt_delta_abs_mean"] > 0 for row in history), "checkpoint_resume_contract": (run_root / "last.pth").is_file()})
    return {"run_root": str(run_root), "history": history, "global_step": global_step}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["visa", "mvtec"], required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cir_dfg_rmt_v1.json"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/cir_rmt/CIR_DFG_RMT_V1"))
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
    parser.add_argument("--git-sha", default="WORKTREE_SHA")
    args = parser.parse_args(argv)
    if args.micro_batch_size * args.grad_accum_steps != 6:
        raise SystemExit("micro_batch_size * grad_accum_steps must equal effective batch size 6")
    train(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
