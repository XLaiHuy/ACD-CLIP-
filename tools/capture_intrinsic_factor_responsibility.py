#!/usr/bin/env python3
"""Bounded TRAIN-only capture of intrinsic factor compatibility state scores."""
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
from tools.audit_p1_v83_semantics import _model_from_checkpoint, _seed
from tools.audit_p1_v84a_post300 import _IndexedDataset, _state_hash
from utils import make_dataloader_generator, seed_worker


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bounded capture")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = torch.load(args.source_capture, map_location="cpu", weights_only=False)
    required = ("image_id", "group_index", "patch_index", "delta", "target", "teacher_probability")
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"source capture missing: {missing}")
    image_ids = source["image_id"].long()
    wanted = sorted({int(value) for value in image_ids.tolist()})
    if wanted != list(range(len(wanted))):
        raise ValueError("source support must be a contiguous image prefix")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _seed(args.seed)
    device = torch.device("cuda:0")
    model = _model_from_checkpoint(checkpoint, device)
    model.requires_grad_(False)
    model.eval()
    model.clipmodel.eval()
    before = _state_hash(model)
    dataset = _IndexedDataset(get_text_and_image_dataset("VisA", int(checkpoint["img_size"]), "train"))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
                        worker_init_fn=seed_worker, generator=make_dataloader_generator(args.seed))
    state_rows, bank_rows, residual_rows, factor_rows = [], [], [], []
    with torch.inference_mode():
        for image_id, sample in enumerate(loader):
            if image_id >= len(wanted):
                break
            selected = image_ids == image_id
            image = sample["image"].to(device, non_blocking=True)
            visual = model(image, return_phase4_features=True)
            h6 = model.h6.build_batch(
                model, "VisA", [str(sample["class_name"][0])], visual,
                hybrid_alpha=float(checkpoint["hybrid_alpha_current"]), update_load_bias=False,
            )
            patches = F.normalize(torch.stack(visual["seg_tokens"]).float(), dim=-1)
            bank = F.normalize(h6["active_factor_bank"].float(), dim=3)
            scale = float(model.h6.h6_logit_temperature)
            states = scale * torch.einsum("gbpd,gbmds->gbpms", patches, bank)
            group = source["group_index"][selected].long().to(device)
            patch = source["patch_index"][selected].long().to(device)
            state_rows.append(states[group, 0, patch].cpu())
            bank_rows.append(bank[group, 0].permute(0, 1, 3, 2).cpu())
            residual_rows.append(h6["factor_residual_logits"][group, 0, patch].float().cpu())
            factor_rows.append(h6["factor_patch_logits"][group, 0, patch].float().cpu())
    state = torch.cat(state_rows)
    bank = torch.cat(bank_rows)
    residual = torch.cat(residual_rows)
    factor = torch.cat(factor_rows)
    if not torch.allclose(residual, source["delta"].float(), atol=2e-5, rtol=2e-5):
        raise RuntimeError(f"restored residual mismatch max={float((residual-source['delta'].float()).abs().max())}")
    if not torch.allclose(factor, state[..., 1] - state[..., 0], atol=2e-5, rtol=2e-5):
        raise RuntimeError("state contrast does not reconstruct factor logits")
    after = _state_hash(model)
    if before != after:
        raise RuntimeError("inference capture mutated model state")
    output = {
        "audit": "INTRINSIC_FACTOR_RESPONSIBILITY_CAPTURE",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config.resolve()), "config_sha256": sha256(args.config),
        "source_capture": str(args.source_capture.resolve()), "source_capture_sha256": sha256(args.source_capture),
        "source_commit": checkpoint.get("git_sha"), "seed": args.seed,
        "state_similarity": state, "active_factor_bank": bank,
        "factor_patch_logits": factor, "delta": residual,
        "image_id": source["image_id"].long().clone(), "group_index": source["group_index"].long().clone(),
        "patch_index": source["patch_index"].long().clone(), "target": source["target"].float().clone(),
        "teacher_probability": source["teacher_probability"].float().clone(),
        "learned_probability": source["probabilities"].float().clone(),
        "act_probability": source["act_probability"].float().clone(), "base_logit": source["base_logit"].float().clone(),
        "metadata": {"images": len(wanted), "patches": int(state.shape[0]), "state_shape": list(state.shape),
                     "similarity_scale": scale, "model_forwards": len(wanted), "backward": False,
                     "optimizer_steps": 0, "model_state_unchanged": True, "tf32_disabled": True,
                     "residual_source_max_abs_error": float((residual-source["delta"].float()).abs().max()),
                     "state_contrast_factor_logit_max_abs_error": float((factor-(state[..., 1]-state[..., 0])).abs().max())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "metadata": output["metadata"]}, indent=2))


if __name__ == "__main__":
    main()
