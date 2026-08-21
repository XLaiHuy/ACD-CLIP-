#!/usr/bin/env python3
"""VisA-only Phase2B trainer for the canonical protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TextAndImageDataset
from model.phase2b_runtime import build_phase2b_trainable, forward_phase2b, load_json_config, trainable_parameter_counts
from utils import calculate_seg_loss


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
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _path_identity(path: Path) -> dict[str, str]:
    resolved = str(path.expanduser().resolve())
    return {"basename": path.name, "resolved_path_sha256": _sha256_bytes(resolved.encode("utf-8"))}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_payload(model: Any, config: dict[str, Any], epoch: int, git_sha: str | None) -> dict[str, Any]:
    return {
        "checkpoint_version": 1,
        "protocol_version": config["protocol_version"],
        "epoch": int(epoch),
        "git_sha": git_sha,
        "model_name": config["model_name"],
        "img_size": int(config["img_size"]),
        "n_groups": int(config["n_groups"]),
        "dfg_mode": config["dfg_mode"],
        "dfg_attn_dim": config.get("dfg_attn_dim"),
        "dfg_attn_tau": config.get("dfg_attn_tau"),
        "use_ss2d_dfg": bool(config.get("use_ss2d_dfg", False)),
        "dfg_gamma_max": config.get("dfg_gamma_max"),
        "dfg_ss2d_fusion": config.get("dfg_ss2d_fusion"),
        "dfg_beta": config.get("dfg_beta"),
        "dfg_beta_current": config.get("dfg_beta"),
        "use_hybrid_soft_prompt": bool(config.get("use_hybrid_soft_prompt", False)),
        "use_soft_prompt": bool(config.get("use_soft_prompt", False)),
        "hybrid_alpha_current": 0.0,
        "prompt_mode": "hybrid" if config.get("use_hybrid_soft_prompt") else "hard",
        "image_adapter": model.image_adapter.state_dict(),
        "text_adapter": model.text_adapter.state_dict(),
        "soft_prompt": model.soft_prompt.state_dict(),
    }


def train_phase2b(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(load_json_config(args.config))
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.epochs is not None:
        config["epochs"] = int(args.epochs)
    if int(config.get("batch_size", 1)) != 1 or int(config.get("grad_accum_steps", 1)) != 6:
        raise ValueError("canonical trainer requires batch_size=1 and grad_accum_steps=6")
    if str(config.get("precision")) != "fp32":
        raise ValueError("canonical trainer requires fp32")
    seed_everything(int(config.get("seed", 0)))
    device = torch.device(args.device)
    model = build_phase2b_trainable(config, args.clip_asset, device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        from model.phase2b_legacy_bridge import load_adapter_state

        load_adapter_state(model, checkpoint)
    parameter_groups = [
        {"name": "image_adapter", "params": list(model.image_adapter.parameters()), "lr": float(config["image_lr"])},
        {"name": "text_adapter", "params": list(model.text_adapter.parameters()), "lr": float(config["text_lr"])},
        {"name": "soft_prompt", "params": [p for p in model.soft_prompt.parameters() if p.requires_grad], "lr": float(config["soft_prompt_lr"])},
    ]
    parameter_groups = [group for group in parameter_groups if group["params"]]
    optimizer = torch.optim.Adam(parameter_groups)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=float(config["lr_gamma"]))
    metadata_path = Path(__file__).resolve().parent / "dataset/hub/VisA.jsonl"
    dataset = TextAndImageDataset(str(args.visa_root), str(metadata_path), int(config["img_size"]))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    run_root = Path(args.run_root)
    checkpoint_root = run_root / "phase2b/checkpoints"
    history: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        model.clipmodel.eval()
        model.image_encoder.eval()
        loss_values: list[float] = []
        for batch_index, batch in enumerate(loader, start=1):
            result = forward_phase2b(model, batch["image"], list(batch["class_name"]), device, config, domain="Industrial", require_grad=True, dataset_name="VisA")
            labels = batch["label"].to(device).float()
            masks = batch["mask"].to(device)
            if result.training_segmentation_probability is None:
                raise RuntimeError("training forward did not return segmentation probability")
            cls_loss = F.binary_cross_entropy(result.classification_probability, labels)
            seg_loss = calculate_seg_loss(result.training_segmentation_probability, masks)
            loss = cls_loss + seg_loss
            (loss / 6.0).backward()
            if batch_index % 6 == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_values.append(float(loss.detach().cpu()))
        scheduler.step()
        history.append({"epoch": float(epoch), "loss": float(np.mean(loss_values) if loss_values else 0.0)})
        if epoch in tuple(config["candidate_epochs"]):
            checkpoint_path = checkpoint_root / f"adapter_{epoch}.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_checkpoint_payload(model, config, epoch, args.git_sha), checkpoint_path)

    config_bytes = json.dumps(config, sort_keys=True, default=_json_default).encode("utf-8")
    counts = trainable_parameter_counts(model)
    groups = [{"name": group["name"], "lr": float(group["lr"]), "parameter_count": sum(p.numel() for p in group["params"])} for group in parameter_groups]
    checkpoint_records = []
    for epoch in config["candidate_epochs"]:
        path = checkpoint_root / f"adapter_{int(epoch)}.pth"
        if path.is_file():
            checkpoint_records.append({"epoch": int(epoch), "path": str(path), "sha256": _file_sha256(path)})
    _write_json(run_root / "phase2b/config_resolved.json", config)
    _write_json(
        run_root / "phase2b/train_manifest.json",
        {
            "protocol_version": config["protocol_version"],
            "git_sha": args.git_sha,
            "dataset": "VisA",
            "dataset_root_identity": _path_identity(Path(args.visa_root)),
            "config_sha256": _sha256_bytes(config_bytes),
            "seed": int(config["seed"]),
            "precision": config["precision"],
            "effective_batch": int(config["batch_size"]) * int(config["grad_accum_steps"]),
            "parameter_group_summary": groups,
            "trainable_parameter_count": counts["trainable"],
            "frozen_parameter_count": counts["frozen"],
            "candidate_epochs": list(config["candidate_epochs"]),
            "saved_checkpoints": checkpoint_records,
            "history": history,
        },
    )
    return {"run_root": str(run_root), "history": history}


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
    parser.add_argument("--git-sha", default="WORKTREE_SHA")
    args = parser.parse_args(argv)
    train_phase2b(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
