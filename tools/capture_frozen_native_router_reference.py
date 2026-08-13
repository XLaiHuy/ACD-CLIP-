#!/usr/bin/env python3
"""Capture a no-grad native-CLIP patch reference for Router Stage B."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import get_text_and_image_dataset
from tools.audit_p1_v83_semantics import _model_from_checkpoint
from tools.audit_p1_v84a_post300 import _IndexedDataset, _state_hash
from tools.capture_fresh_router_cross_state import fresh_model
from utils import make_dataloader_generator, seed_worker


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", choices=("restored", "fresh"), required=True)
    parser.add_argument("--state-label", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen-reference capture")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = torch.load(args.source_capture, map_location="cpu", weights_only=False)
    required = ("image_id", "group_index", "patch_index", "region", "target", "teacher_probability")
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"source capture missing fields: {missing}")
    wanted = sorted({int(value) for value in source["image_id"].tolist()})
    if wanted != list(range(len(wanted))):
        raise ValueError("source image IDs must be a contiguous deterministic prefix")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = json.loads(args.config.read_text())
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    model = _model_from_checkpoint(checkpoint, device) if args.state == "restored" else fresh_model(checkpoint, device)
    model.eval()
    model.clipmodel.eval()
    # The native encode_image inference route has no backward and this
    # repository's legacy checkpoint wrapper returns an incompatible tuple.
    # Disable only that runtime memory wrapper; it does not alter visual
    # weights or the frozen-reference computation.
    model.image_encoder.set_grad_checkpointing(False)
    state_before = _state_hash(model)
    dataset = _IndexedDataset(get_text_and_image_dataset("VisA", int(checkpoint["img_size"]), "train"))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
                        worker_init_fn=seed_worker, generator=make_dataloader_generator(args.seed))
    rows = []
    with torch.inference_mode():
        for image_id, sample in enumerate(loader):
            if image_id >= len(wanted):
                break
            selected = source["image_id"].long() == image_id
            if not bool(selected.any()):
                continue
            native_patches, _ = model.forward_original(sample["image"].to(device, non_blocking=True))
            if len(native_patches) != 1:
                raise RuntimeError(f"expected one native patch tensor, got {len(native_patches)}")
            feature = native_patches[0]
            if feature.ndim != 3 or feature.shape[0] != 1:
                raise RuntimeError(f"native patch reference must be [1,P,D], got {tuple(feature.shape)}")
            patch_index = source["patch_index"][selected].long().to(device)
            if int(patch_index.max().item()) >= feature.shape[1]:
                raise RuntimeError("source patch index exceeds native reference geometry")
            rows.append(feature[0, patch_index].detach().float().cpu())
    reference = torch.cat(rows, dim=0)
    if reference.shape[0] != source["image_id"].numel():
        raise RuntimeError(f"matched rows mismatch: {reference.shape[0]} != {source['image_id'].numel()}")
    state_after = _state_hash(model)
    if state_before != state_after:
        raise RuntimeError("inference mutated model state")
    output = {
        "audit": "FROZEN_NATIVE_CLIP_ROUTER_REFERENCE_CAPTURE", "state": args.state_label, "state_mode": args.state,
        "source_checkpoint": str(args.checkpoint.resolve()), "source_checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config.resolve()), "config_sha256": sha256(args.config),
        "source_capture": str(args.source_capture.resolve()), "source_capture_sha256": sha256(args.source_capture),
        "source_commit": checkpoint.get("git_sha"), "seed": args.seed, "native_reference": reference,
        "image_id": source["image_id"].long().clone(), "group_index": source["group_index"].long().clone(),
        "patch_index": source["patch_index"].long().clone(), "region": source["region"].to(torch.uint8).clone(),
        "target": source["target"].float().clone(), "teacher_probability": source["teacher_probability"].float().clone(),
        "utility_gap": source.get("utility_gap", torch.zeros_like(source["target"], dtype=torch.float32)).float().clone(),
        "metadata": {"images": len(wanted), "patches": int(reference.shape[0]), "feature_dim": int(reference.shape[1]),
                     "backward": False, "optimizer_steps": 0, "model_state_unchanged": True,
                     "native_path": "ACDCLIP.forward_original -> pretrained CLIP visual only",
                     "reference_gradient_checkpointing_disabled": True,
                     "tf32_disabled": not torch.backends.cuda.matmul.allow_tf32},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "metadata": output["metadata"]}, indent=2))


if __name__ == "__main__":
    main()
