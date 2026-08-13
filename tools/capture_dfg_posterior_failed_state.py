#!/usr/bin/env python3
"""Capture detached DFG base posteriors for an existing failed-state support.

This is intentionally inference-only.  It supplements the existing direct-head
forensic capture rather than re-running a dataset-scale audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import get_text_and_image_dataset
from tools.audit_p1_v83_semantics import _model_from_checkpoint
from tools.audit_p1_v84a_post300 import _IndexedDataset, _seed, _state_hash
from utils import get_phase2b_global_text_features, make_dataloader_generator, seed_worker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    source = torch.load(args.source_capture, map_location="cpu", weights_only=False)
    wanted = sorted({int(x) for x in source["image_id"].tolist()})
    if wanted != list(range(len(wanted))):
        raise RuntimeError(f"source support must be a contiguous prefix, got {wanted[:8]}...")
    _seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _model_from_checkpoint(checkpoint, device)
    model.requires_grad_(False)
    model.eval()
    before = _state_hash(model)
    dataset = _IndexedDataset(get_text_and_image_dataset("VisA", 518, "train"))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(args.seed),
    )
    chunks: list[torch.Tensor] = []
    image_ids: list[int] = []
    with torch.inference_mode():
        for image_id, sample in enumerate(loader):
            if image_id >= len(wanted):
                break
            image = sample["image"].to(device, non_blocking=True)
            classes = [sample["class_name"][0]]
            visual = model(image, return_phase4_features=True)
            seg_features = torch.stack(visual["seg_tokens"], dim=0)
            text_global = get_phase2b_global_text_features(
                model,
                "VisA",
                classes,
                device,
                use_hybrid_soft_prompt=True,
                use_soft_prompt=False,
            ).to(dtype=seg_features.dtype)
            _, base_group_logits, _ = model.vision_text_fusion_gate_seg(
                seg_features, text_global, img_size=518, h6_patch_logits=None, return_details=True
            )
            chunks.append(base_group_logits.float().cpu())
            image_ids.append(image_id)
    if image_ids != wanted:
        raise RuntimeError(f"captured image ids {image_ids} but expected {wanted}")
    base_group_logits = torch.cat(chunks, dim=1)
    posterior = F.softmax(base_group_logits, dim=-1)
    if not torch.isfinite(base_group_logits).all() or not torch.isfinite(posterior).all():
        raise RuntimeError("non-finite DFG capture")
    probability_error = float((posterior.sum(dim=-1) - 1.0).abs().max().item())
    after = _state_hash(model)
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "source_capture": str(args.source_capture.resolve()),
        "source_capture_sha256": hashlib.sha256(args.source_capture.read_bytes()).hexdigest(),
        "seed": args.seed,
        "image_ids": image_ids,
        "base_group_logits": base_group_logits,
        "dfg_posterior": posterior,
        "metadata": {
            "optimizer_steps": 0,
            "backward": False,
            "model_state_unchanged": before == after,
            "posterior_sum_max_abs_error": probability_error,
            "tf32_disabled": not torch.backends.cuda.matmul.allow_tf32,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "shape": list(base_group_logits.shape),
        "metadata": payload["metadata"],
    }, indent=2))


if __name__ == "__main__":
    main()
