#!/usr/bin/env python3
"""Inference-only Stage 1.7 direct-residual-gate counterfactual for K1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from audit_p4_k1_oracle_utility import (
    DeterministicVisATrainDataset,
    _build_model,
    _bootstrap_mean,
    _git_sha,
    _sha256,
)
from utils import configure_canonical_fp32, get_phase2b_global_text_features


def _mean(values):
    value = torch.cat(values).float()
    return {"count": int(value.numel()), "mean": float(value.mean()), "std": float(value.std(unbiased=False))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/phase4/k1/short64_seed0_attempt5"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase4/k1/stage1_7r/K1_DIRECT_RESIDUAL_GATE_AUDIT.json"))
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    run_dir, manifest_path = args.run_dir.resolve(), args.manifest.resolve()
    config_path, checkpoint_path = run_dir / "config.json", run_dir / "adapter_1.pth"
    config = json.loads(config_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = json.loads(manifest_path.read_text())
    if config["h6_progress_version"] != "P4-CSF-K1":
        raise RuntimeError("direct residual audit requires the trained K1 checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("direct residual audit requires CUDA")
    configure_canonical_fp32()
    device = torch.device(f"cuda:{config['cuda_device']}")
    model = _build_model(config, checkpoint, device)
    loader = DataLoader(DeterministicVisATrainDataset(manifest, config["img_size"]), batch_size=1, shuffle=False, num_workers=0)
    variants = ("BASE", "ORIGINAL_K1", "SEMANTIC_MIXTURE_NOOP", "DIRECT_RESIDUAL_GATE")
    regions = ("all", "normal", "anomaly")
    bce = {variant: {region: [] for region in regions} for variant in variants}
    gate = {region: [] for region in regions}
    image_rows = []
    for image_index, raw in enumerate(loader):
        image = raw["image"].to(device=device, dtype=torch.float32)
        mask = raw["mask"].to(device=device, dtype=torch.float32)
        class_name, file_name, label = raw["class_name"][0], raw["file_name"][0], int(raw["label"].item())
        per_image = {variant: {region: [] for region in regions} for variant in variants}
        per_gate = {region: [] for region in regions}
        with torch.inference_mode():
            visual = model(image, return_phase4_features=True)
            base_text = get_phase2b_global_text_features(model, "VisA", [class_name], device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
            batch = model.h6.build_batch(model, "VisA", [class_name], visual, hybrid_alpha=0.0, base_text_features=base_text)
            base_cross = base_text.permute(1, 0, 2, 3).float()
            dynamic_cross = batch["dynamic_text"].permute(1, 0, 2, 3).float()
            rho = model.h6.rho_values().float()
            for group, v in enumerate(visual["seg_tokens"]):
                v = v.float()
                weights_n, weights_a = batch["base_dfg_weights_normal"][group], batch["base_dfg_weights_abnormal"][group]
                a0 = model.apply_dfg_weights(base_cross, weights_n, weights_a)[..., 1]
                a1 = model.apply_dfg_weights(dynamic_cross, weights_n, weights_a)[..., 1]
                margin = torch.einsum("bpd,bd->bp", v, a1 - a0)
                alpha = torch.sigmoid(margin)
                mix = F.normalize((1.0 - alpha).unsqueeze(-1) * a0.unsqueeze(1) + alpha.unsqueeze(-1) * a1.unsqueeze(1), dim=-1)
                base_logits = batch["base_group_logits"][group]
                base_abnormal = base_logits[..., 1]
                original_abnormal = batch["dynamic_abnormal_logits"][group]
                mixture_abnormal = torch.einsum("bpd,bpd->bp", 10.0 * v, mix)
                direct_abnormal = base_abnormal + rho[group] * alpha * (original_abnormal - base_abnormal)
                final = {
                    "BASE": base_abnormal,
                    "ORIGINAL_K1": base_abnormal + rho[group] * (original_abnormal - base_abnormal),
                    "SEMANTIC_MIXTURE_NOOP": base_abnormal + rho[group] * (mixture_abnormal - base_abnormal),
                    "DIRECT_RESIDUAL_GATE": direct_abnormal,
                }
                side = int(v.shape[1] ** 0.5)
                target = F.adaptive_avg_pool2d(mask, (side, side)).flatten(1)
                valid = torch.ones_like(target, dtype=torch.bool)
                masks = {"all": valid, "normal": valid & (target == 0), "anomaly": valid & (target > 0)}
                normal = base_logits[..., 0]
                for region, region_mask in masks.items():
                    for variant, abnormal in final.items():
                        values = F.binary_cross_entropy_with_logits(abnormal - normal, target, reduction="none")[region_mask].cpu()
                        bce[variant][region].append(values)
                        per_image[variant][region].append(values)
                    gate[region].append(alpha[region_mask].cpu())
                    per_gate[region].append(alpha[region_mask].cpu())
        row = {"image_index": image_index, "class_name": class_name, "file_name": file_name, "label": label}
        for region in regions:
            if per_gate[region]:
                row[f"gate_{region}_mean"] = float(torch.cat(per_gate[region]).mean())
                for variant in variants:
                    row[f"{variant}_{region}_bce"] = float(torch.cat(per_image[variant][region]).mean())
        image_rows.append(row)
    report_variants = {}
    for variant in variants:
        report_variants[variant] = {"bce": {region: _mean(bce[variant][region]) for region in regions}}
        report_variants[variant]["gain_base_minus_final"] = {region: report_variants["BASE"]["bce"][region]["mean"] - report_variants[variant]["bce"][region]["mean"] for region in regions}
    image_gains = {}
    for variant in variants[1:]:
        image_gains[variant] = {}
        for region in regions:
            rows = [row for row in image_rows if f"{variant}_{region}_bce" in row]
            gains = torch.tensor([row[f"BASE_{region}_bce"] - row[f"{variant}_{region}_bce"] for row in rows])
            image_gains[variant][region] = _bootstrap_mean(gains, args.seed)
    report = {
        "decision": "DIRECT_RESIDUAL_GATE_AUDIT_COMPLETE",
        "provenance": {"repo_sha": _git_sha(), "script_sha256": _sha256(Path(__file__).resolve()), "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)), "checkpoint_sha256": _sha256(checkpoint_path), "checkpoint_git_sha": checkpoint.get("git_sha"), "config_sha256": _sha256(config_path), "manifest_path": str(manifest_path.relative_to(REPO_ROOT)), "manifest_sha256": _sha256(manifest_path), "dataset": "VisA", "split": "train", "seed": args.seed, "precision": "fp32; TF32 off; AMP off; deterministic no-augmentation Train transform", "optimizer_steps": 0},
        "gate_definition": "sigmoid(m=score(A1)-score(A0)); gate=0 is exact base and gate=1 is exact original K1 residual",
        "variants": report_variants,
        "gate": {region: _mean(gate[region]) for region in regions},
        "image_level_gain_bootstrap": image_gains,
        "per_image": image_rows,
        "interpretation": "Counterfactual only. No new parameters, no semantic identity change, and no training.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"original": report_variants["ORIGINAL_K1"]["gain_base_minus_final"], "mixture": report_variants["SEMANTIC_MIXTURE_NOOP"]["gain_base_minus_final"], "direct": report_variants["DIRECT_RESIDUAL_GATE"]["gain_base_minus_final"], "gate": report["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
