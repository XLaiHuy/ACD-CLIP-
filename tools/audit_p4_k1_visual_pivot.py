#!/usr/bin/env python3
"""Inference-only Stage 1.7V audit: local visual evidence versus K1 text utility."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from audit_p4_k1_oracle_utility import (
    DeterministicVisATrainDataset,
    _auroc,
    _build_model,
    _git_sha,
    _pearson,
    _sha256,
)
from utils import configure_canonical_fp32, get_phase2b_global_text_features


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/phase4/k1/short64_seed0_attempt5"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase4/k1/stage1_7r/K1_VISUAL_TEXT_PIVOT_AUDIT.json"))
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    run_dir, manifest_path = args.run_dir.resolve(), args.manifest.resolve()
    config_path, checkpoint_path = run_dir / "config.json", run_dir / "adapter_1.pth"
    config, checkpoint = json.loads(config_path.read_text()), torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = json.loads(manifest_path.read_text())
    if config["h6_progress_version"] != "P4-CSF-K1" or not torch.cuda.is_available():
        raise RuntimeError("visual/text pivot audit requires CUDA and the trained K1 checkpoint")
    configure_canonical_fp32()
    device = torch.device(f"cuda:{config['cuda_device']}")
    model = _build_model(config, checkpoint, device)
    loader = DataLoader(DeterministicVisATrainDataset(manifest, config["img_size"]), batch_size=1, shuffle=False, num_workers=0)
    all_posterior, all_target = [], []
    image_rows, per_class = [], defaultdict(list)
    for image_index, raw in enumerate(loader):
        image, mask = raw["image"].to(device=device, dtype=torch.float32), raw["mask"].to(device=device, dtype=torch.float32)
        class_name, file_name, label = raw["class_name"][0], raw["file_name"][0], int(raw["label"].item())
        summaries = []
        with torch.inference_mode():
            visual = model(image, return_phase4_features=True)
            base_text = get_phase2b_global_text_features(model, "VisA", [class_name], device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
            batch = model.h6.build_batch(model, "VisA", [class_name], visual, hybrid_alpha=0.0, base_text_features=base_text)
            for group, patches in enumerate(visual["seg_tokens"]):
                side = int(patches.shape[1] ** 0.5)
                target = F.adaptive_avg_pool2d(mask, (side, side)).flatten(1)
                base_logit = batch["base_group_logits"][group, ..., 1] - batch["base_group_logits"][group, ..., 0]
                dynamic_logit = batch["dynamic_abnormal_logits"][group] - batch["base_group_logits"][group, ..., 0]
                posterior = torch.sigmoid(base_logit)
                utility = F.binary_cross_entropy_with_logits(base_logit, target, reduction="none") - F.binary_cross_entropy_with_logits(dynamic_logit, target, reduction="none")
                anomaly = target > 0
                all_posterior.append(posterior.flatten().cpu())
                all_target.append(anomaly.flatten().cpu())
                summaries.append({"base_posterior_mean": float(posterior.mean()), "base_posterior_anomaly_mean": float(posterior[anomaly].mean()) if anomaly.any() else None, "base_posterior_normal_mean": float(posterior[~anomaly].mean()), "anomaly_patch_fraction": float(anomaly.float().mean()), "oracle_utility_anomaly_mean": float(utility[anomaly].mean()) if anomaly.any() else None, "oracle_utility_normal_mean": float(utility[~anomaly].mean())})
        anomaly_parts = [row for row in summaries if row["oracle_utility_anomaly_mean"] is not None]
        row = {"image_index": image_index, "class_name": class_name, "file_name": file_name, "label": label, "base_posterior_mean": _mean([item["base_posterior_mean"] for item in summaries]), "base_posterior_normal_mean": _mean([item["base_posterior_normal_mean"] for item in summaries]), "anomaly_patch_fraction": _mean([item["anomaly_patch_fraction"] for item in summaries]), "oracle_utility_normal_mean": _mean([item["oracle_utility_normal_mean"] for item in summaries]), "base_posterior_anomaly_mean": _mean([item["base_posterior_anomaly_mean"] for item in anomaly_parts]), "oracle_utility_anomaly_mean": _mean([item["oracle_utility_anomaly_mean"] for item in anomaly_parts])}
        image_rows.append(row)
        per_class[class_name].append(row)
    posterior, target = torch.cat(all_posterior), torch.cat(all_target).bool()
    anomaly_images = [row for row in image_rows if row["oracle_utility_anomaly_mean"] is not None]
    by_class = {name: {"images": len(rows), "anomaly_images": sum(row["label"] for row in rows), "base_posterior_normal_mean": _mean([row["base_posterior_normal_mean"] for row in rows]), "base_posterior_anomaly_mean": _mean([row["base_posterior_anomaly_mean"] for row in rows if row["base_posterior_anomaly_mean"] is not None]), "oracle_utility_anomaly_mean": _mean([row["oracle_utility_anomaly_mean"] for row in rows if row["oracle_utility_anomaly_mean"] is not None]), "oracle_utility_normal_mean": _mean([row["oracle_utility_normal_mean"] for row in rows])} for name, rows in sorted(per_class.items())}
    report = {"decision": "STAGE1_7V_VISUAL_TEXT_PIVOT_AUDIT_COMPLETE", "provenance": {"repo_sha": _git_sha(), "script_sha256": _sha256(Path(__file__).resolve()), "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)), "checkpoint_sha256": _sha256(checkpoint_path), "checkpoint_git_sha": checkpoint.get("git_sha"), "config_sha256": _sha256(config_path), "manifest_path": str(manifest_path.relative_to(REPO_ROOT)), "manifest_sha256": _sha256(manifest_path), "dataset": "VisA", "split": "train", "seed": args.seed, "precision": "fp32; TF32 off; AMP off; deterministic no-augmentation Train transform", "optimizer_steps": 0}, "base_local_visual_evidence": {"patch_anomaly_auroc": _auroc(posterior, target), "posterior_normal_mean": float(posterior[~target].mean()), "posterior_anomaly_mean": float(posterior[target].mean())}, "text_branch_counterfactual": {"disabled": "exact current Phase2B base/A0", "enabled": "original K1 A1; use K1_ORACLE_UTILITY_AUDIT.json for region utility"}, "error_concentration": {"anomaly_mask_fraction_vs_anomaly_utility_pearson": _pearson(torch.tensor([row["anomaly_patch_fraction"] for row in anomaly_images]), torch.tensor([row["oracle_utility_anomaly_mean"] for row in anomaly_images])), "class_summary": by_class, "foreground_background_annotation_available": False, "texture_annotation_available": False}, "per_image": image_rows, "interpretation": "The audit can support only a local-visual versus text-conditioning pivot diagnosis; it has no foreground/background or texture labels and therefore does not claim either mechanism."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"base_local_visual_evidence": report["base_local_visual_evidence"], "error_concentration": {"mask_fraction_vs_utility": report["error_concentration"]["anomaly_mask_fraction_vs_anomaly_utility_pearson"]}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
