"""Tiny GT-free frozen-forward smoke for the qualified P27R1 child topology."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.sabra.data import VisaEvidenceDataset, read_visa_metadata
from tools.sabra_v2.cuda_runtime import probe_current_cuda
from tools.sabra_v2.train_region_distill import ROOT, _load_frozen_phase2b


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--p26-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    runtime = probe_current_cuda()
    rows = read_visa_metadata(args.metadata)
    dataset = VisaEvidenceDataset(rows[:1], args.visa_root)
    sample = dataset[0]
    if set(sample) != {"image", "class_name", "image_path", "index"}:
        raise RuntimeError("runtime smoke must remain GT-free")
    device = torch.device("cuda")
    phase2b, config = _load_frozen_phase2b(args.p26_checkpoint, args.clip_asset, device)
    from model.phase2b_runtime import forward_phase2b

    frozen = forward_phase2b(
        phase2b,
        sample["image"].unsqueeze(0),
        [sample["class_name"]],
        device,
        config,
        domain="Industrial",
        require_grad=False,
    )
    if tuple(frozen.seg_features.shape) != (3, 1, 1369, 768) or frozen.seg_features.dtype != torch.float32:
        raise RuntimeError("frozen seg_features contract failed")
    if tuple(frozen.native_logits.shape) != (3, 1, 1369, 2) or frozen.native_logits.dtype != torch.float32:
        raise RuntimeError("frozen native_logits contract failed")
    return {
        "schema_version": "P27R1_EXACT_PATH_SMOKE_V1",
        "status": "PASS",
        "engineering_only": True,
        "gt_used": False,
        "mask_reads": 0,
        "teacher_steps": 0,
        "adapter_training_steps": 0,
        "predictions_written": 0,
        "scores_computed": 0,
        "sample_id": f"{sample['class_name']}:{sample['image_path']}",
        "seg_features_shape": list(frozen.seg_features.shape),
        "seg_features_dtype": str(frozen.seg_features.dtype),
        "native_logits_shape": list(frozen.native_logits.shape),
        "native_logits_dtype": str(frozen.native_logits.dtype),
        "runtime": runtime,
    }


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
