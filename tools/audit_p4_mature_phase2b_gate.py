#!/usr/bin/env python3
"""Offline Stage 1.9A audit of a mature Phase2B segmentation map as K1 gate.

This script changes no model parameters.  It evaluates the documented Phase2B
e10 ``current_shared`` hybrid-alpha=.20 segmentation operator on the frozen
Stage 1.7R VisA-Train manifest, and compares its map with the already-trained
K1 A0/A1 counterfactual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from audit_p4_k1_oracle_utility import (
    DeterministicVisATrainDataset,
    _auroc,
    _bootstrap_mean,
    _build_model as build_k1_model,
    _git_sha,
    _pearson,
    _sha256,
    _spearman,
)
from dataset import DOMAINS
from phase2b_anchor_diagnosis import build_model as build_phase2b_model
from phase2b_anchor_diagnosis import get_class_text_embedding, load_checkpoint
from utils import configure_canonical_fp32, get_phase2b_global_text_features


SCRIPT_VERSION = "stage1_9a_mature_phase2b_direct_gate_v1"
REGIONS = ("all", "normal", "anomaly")
VARIANTS = ("BASE", "ORIGINAL_K1", "SEMANTIC_MIXTURE_NOOP", "MATURE_P2B_DIRECT_GATE")


def _summary(values: list[torch.Tensor]) -> dict[str, Any]:
    value = torch.cat(values).float()
    quantiles = torch.quantile(value, torch.tensor([0.1, 0.5, 0.9]))
    return {
        "count": int(value.numel()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "p10": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p90": float(quantiles[2]),
    }


def _utility_metrics(score: torch.Tensor, utility: torch.Tensor) -> dict[str, Any]:
    if score.numel() != utility.numel():
        raise ValueError("score and utility must have identical length")
    positive = utility > 0
    return {
        "count": int(score.numel()),
        "utility_mean": float(utility.mean()),
        "fraction_a1_better": float(positive.float().mean()),
        "pearson": _pearson(score, utility),
        "spearman": _spearman(score, utility),
        "auroc_a1_better": _auroc(score, positive),
        "mean_score_a1_better": float(score[positive].mean()) if positive.any() else None,
        "mean_score_a0_better": float(score[~positive].mean()) if (~positive).any() else None,
    }


def _phase2b_args(cuda_device: int) -> SimpleNamespace:
    """Exact documented e10 architecture and fixed ``current_shared`` prompt."""
    return SimpleNamespace(
        model_name="ViT-L-14-336", img_size=518, cuda_device=cuda_device,
        n_groups=3, lora_rank=16, lora_alpha=2.0, conv_lora_rank=8,
        conv_lora_alpha=2.0, conv_kernel_size_list=[3, 5],
        soft_prompt_ctx_len=4, soft_prompt_init="phrase",
        soft_prompt_init_phrase="a photo of a", dfg_mode="attn",
        dfg_attn_dim=256, dfg_attn_tau=8.0, use_ss2d_dfg=True,
        dfg_gamma_max=0.2, dfg_ss2d_fusion="weight_residual", dfg_beta=0.10,
        dfg_beta_schedule="warmup010", dfg_beta_target=0.10,
        fixed_prompt_config="current_shared", prompt_configs=["current_shared"],
    )


def _mean_auroc(values: list[float | None]) -> dict[str, Any]:
    valid = [value for value in values if value is not None]
    return {"eligible_images": len(valid), "mean": None if not valid else float(sum(valid) / len(valid)),
            "median": None if not valid else float(torch.tensor(valid).median())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k1-run-dir", type=Path, default=Path("runs/phase4/k1/short64_seed0_attempt5"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"))
    parser.add_argument("--phase2b-checkpoint", type=Path, default=Path("runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/adapter_10.pth"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase4/k1/stage1_9a/K1_MATURE_PHASE2B_DIRECT_GATE_AUDIT.json"))
    parser.add_argument("--gate-mode", choices=("raw", "image_centered"), default="raw")
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    k1_run_dir, manifest_path = args.k1_run_dir.resolve(), args.manifest.resolve()
    phase2b_path, output_path = args.phase2b_checkpoint.resolve(), args.output.resolve()
    k1_config_path, k1_checkpoint_path = k1_run_dir / "config.json", k1_run_dir / "adapter_1.pth"
    k1_config = json.loads(k1_config_path.read_text())
    k1_checkpoint = torch.load(k1_checkpoint_path, map_location="cpu", weights_only=False)
    phase2b_checkpoint = torch.load(phase2b_path, map_location="cpu", weights_only=False)
    manifest = json.loads(manifest_path.read_text())
    if k1_config.get("h6_progress_version") != "P4-CSF-K1":
        raise RuntimeError("requires the valid trained K1 checkpoint")
    if int(phase2b_checkpoint.get("epoch", -1)) != 10 or not phase2b_checkpoint.get("use_hybrid_soft_prompt"):
        raise RuntimeError("requires the documented Phase2B e10 hybrid checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1.9A requires CUDA for frozen inference")

    torch.manual_seed(args.seed)
    configure_canonical_fp32()
    device = torch.device(f"cuda:{k1_config['cuda_device']}")
    k1 = build_k1_model(k1_config, k1_checkpoint, device)
    p2b_args = _phase2b_args(k1_config["cuda_device"])
    phase2b = build_phase2b_model(p2b_args, device)
    loaded_epoch = load_checkpoint(phase2b, phase2b_path, p2b_args, device)
    if loaded_epoch != 10 or phase2b.prompt_mode != "hybrid" or abs(float(phase2b.hybrid_alpha_current) - 0.2) > 1e-8:
        raise RuntimeError("Phase2B checkpoint prompt state did not reproduce e10 hybrid alpha=.20")
    loader = DataLoader(DeterministicVisATrainDataset(manifest, k1_config["img_size"]), batch_size=1, shuffle=False, num_workers=0)

    bces = {variant: {region: [] for region in REGIONS} for variant in VARIANTS}
    gates = {region: [] for region in REGIONS}
    p2b_scores = {region: [] for region in REGIONS}
    utilities = {region: [] for region in REGIONS}
    targets = {region: [] for region in REGIONS}
    raw_affinity = {region: [] for region in REGIONS}
    per_image: list[dict[str, Any]] = []
    per_class: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for image_index, raw in enumerate(loader):
        image = raw["image"].to(device=device, dtype=torch.float32)
        mask = raw["mask"].to(device=device, dtype=torch.float32)
        class_name, file_name, label = raw["class_name"][0], raw["file_name"][0], int(raw["label"].item())
        image_bces = {variant: {region: [] for region in REGIONS} for variant in VARIANTS}
        image_scores, image_utility, image_target = [], [], []
        image_gates = {region: [] for region in REGIONS}
        with torch.inference_mode():
            # Exact mature Phase2B segmentation probability: the historical
            # model method includes DFG, medical-domain smoothing, interpolation,
            # group averaging, and abnormal-class softmax.
            p2b_tokens, _ = phase2b(image)
            p2b_text = get_class_text_embedding(phase2b, "VisA", class_name, device, "hybrid", 0.20)
            p2b_cross = p2b_text.unsqueeze(1).repeat(1, image.shape[0], 1, 1)
            p2b_map = phase2b.vision_text_fusion_gate_seg(
                torch.stack(p2b_tokens, dim=0), p2b_cross, test_mode=True, domain=DOMAINS["VisA"]
            )
            if args.gate_mode == "raw":
                gate_map = p2b_map
            else:
                # Parameter-free contrast above each image's Phase2B center.
                center = p2b_map.mean(dim=(-2, -1), keepdim=True)
                gate_map = ((p2b_map - center) / (1.0 - center).clamp_min(1e-12)).clamp(0.0, 1.0)

            visual = k1(image, return_phase4_features=True)
            base_text = get_phase2b_global_text_features(k1, "VisA", [class_name], device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
            batch = k1.h6.build_batch(k1, "VisA", [class_name], visual, hybrid_alpha=0.0, base_text_features=base_text)
            base_cross, dynamic_cross = base_text.permute(1, 0, 2, 3).float(), batch["dynamic_text"].permute(1, 0, 2, 3).float()
            rho = k1.h6.rho_values().float()
            for group, patches in enumerate(visual["seg_tokens"]):
                patches = patches.float()
                side = int(patches.shape[1] ** 0.5)
                target = F.adaptive_avg_pool2d(mask, (side, side)).flatten(1)
                # Full-resolution p2b probabilities are averaged over the exact
                # K1 patch grid, then shared across groups because p2b has one
                # final (already group-averaged) segmentation probability map.
                p2b_patch = F.adaptive_avg_pool2d(p2b_map.unsqueeze(1), (side, side)).flatten(1)
                gate_patch = F.adaptive_avg_pool2d(gate_map.unsqueeze(1), (side, side)).flatten(1)
                weights_n, weights_a = batch["base_dfg_weights_normal"][group], batch["base_dfg_weights_abnormal"][group]
                a0 = k1.apply_dfg_weights(base_cross, weights_n, weights_a)[..., 1]
                a1 = k1.apply_dfg_weights(dynamic_cross, weights_n, weights_a)[..., 1]
                base_logits = batch["base_group_logits"][group]
                normal, base_abnormal = base_logits[..., 0], base_logits[..., 1]
                original_abnormal = batch["dynamic_abnormal_logits"][group]
                affinity = torch.einsum("bpd,bd->bp", patches, a1 - a0)
                noop_alpha = torch.sigmoid(affinity)
                mixture = F.normalize((1.0 - noop_alpha).unsqueeze(-1) * a0.unsqueeze(1) + noop_alpha.unsqueeze(-1) * a1.unsqueeze(1), dim=-1)
                mixture_abnormal = torch.einsum("bpd,bpd->bp", 10.0 * patches, mixture)
                final = {
                    "BASE": base_abnormal,
                    "ORIGINAL_K1": base_abnormal + rho[group] * (original_abnormal - base_abnormal),
                    "SEMANTIC_MIXTURE_NOOP": base_abnormal + rho[group] * (mixture_abnormal - base_abnormal),
                    "MATURE_P2B_DIRECT_GATE": base_abnormal + rho[group] * gate_patch * (original_abnormal - base_abnormal),
                }
                utility = F.binary_cross_entropy_with_logits(base_abnormal - normal, target, reduction="none") - F.binary_cross_entropy_with_logits(original_abnormal - normal, target, reduction="none")
                region_masks = {"all": torch.ones_like(target, dtype=torch.bool), "normal": target == 0, "anomaly": target > 0}
                for region, region_mask in region_masks.items():
                    # Do not create a per-image region record for a genuinely
                    # empty region (for example, anomaly patches on a normal
                    # image). Empty tensors are harmless in global pooling but
                    # would otherwise turn the image bootstrap into NaN.
                    if not bool(region_mask.any()):
                        continue
                    for variant, abnormal in final.items():
                        value = F.binary_cross_entropy_with_logits(abnormal - normal, target, reduction="none")[region_mask].cpu()
                        bces[variant][region].append(value)
                        image_bces[variant][region].append(value)
                    gates[region].append(gate_patch[region_mask].cpu())
                    image_gates[region].append(gate_patch[region_mask].cpu())
                    p2b_scores[region].append(p2b_patch[region_mask].cpu())
                    utilities[region].append(utility[region_mask].cpu())
                    targets[region].append((target[region_mask] > 0).cpu())
                    raw_affinity[region].append(affinity[region_mask].cpu())
                image_scores.append(p2b_patch.flatten().cpu())
                image_utility.append(utility.flatten().cpu())
                image_target.append((target > 0).flatten().cpu())

        row: dict[str, Any] = {"image_index": image_index, "class_name": class_name, "file_name": file_name, "label": label}
        flat_score, flat_target, flat_utility = torch.cat(image_scores), torch.cat(image_target), torch.cat(image_utility)
        row["p2b_patch_anomaly_auroc"] = _auroc(flat_score, flat_target)
        row["p2b_utility_auroc"] = _auroc(flat_score, flat_utility > 0)
        row["p2b_utility_pearson"] = _pearson(flat_score, flat_utility)
        for region in REGIONS:
            if image_gates[region]:
                row[f"p2b_gate_{region}_mean"] = float(torch.cat(image_gates[region]).mean())
                for variant in VARIANTS:
                    row[f"{variant}_{region}_bce"] = float(torch.cat(image_bces[variant][region]).mean())
        per_image.append(row)
        per_class[class_name].append(row)

    report_variants: dict[str, Any] = {}
    for variant in VARIANTS:
        report_variants[variant] = {"bce": {region: _summary(bces[variant][region]) for region in REGIONS}}
        report_variants[variant]["gain_base_minus_final"] = {
            region: report_variants["BASE"]["bce"][region]["mean"] - report_variants[variant]["bce"][region]["mean"]
            for region in REGIONS
        }
    image_bootstrap: dict[str, Any] = {}
    for variant in VARIANTS[1:]:
        image_bootstrap[variant] = {}
        for region in REGIONS:
            rows = [row for row in per_image if f"{variant}_{region}_bce" in row]
            image_gain = torch.tensor([row[f"BASE_{region}_bce"] - row[f"{variant}_{region}_bce"] for row in rows])
            image_bootstrap[variant][region] = _bootstrap_mean(image_gain, args.seed)

    localization = {
        "patch_anomaly_auroc": _auroc(torch.cat(p2b_scores["all"]), torch.cat(targets["all"])),
        "per_image_auroc": _mean_auroc([row["p2b_patch_anomaly_auroc"] for row in per_image]),
        "p2b_probability": {region: _summary(p2b_scores[region]) for region in REGIONS},
    }
    alignment = {
        "mature_p2b": {region: _utility_metrics(torch.cat(p2b_scores[region]), torch.cat(utilities[region])) for region in REGIONS},
        "prior_raw_semantic_affinity": {region: _utility_metrics(torch.cat(raw_affinity[region]), torch.cat(utilities[region])) for region in REGIONS},
        "per_image_utility_auroc": _mean_auroc([row["p2b_utility_auroc"] for row in per_image]),
    }
    class_summary = {}
    for class_name, rows in sorted(per_class.items()):
        local_auc = [row["p2b_patch_anomaly_auroc"] for row in rows]
        utility_auc = [row["p2b_utility_auroc"] for row in rows]
        class_summary[class_name] = {
            "images": len(rows), "anomaly_label_images": sum(row["label"] for row in rows),
            "p2b_patch_anomaly_auroc": _mean_auroc(local_auc),
            "p2b_utility_auroc": _mean_auroc(utility_auc),
            "p2b_gate_normal_mean": float(torch.tensor([row["p2b_gate_normal_mean"] for row in rows]).mean()),
            "p2b_gate_anomaly_mean": float(torch.tensor([row["p2b_gate_anomaly_mean"] for row in rows if "p2b_gate_anomaly_mean" in row]).mean()) if any("p2b_gate_anomaly_mean" in row for row in rows) else None,
        }
    source_paths = [
        Path(__file__).resolve(), REPO_ROOT / "phase2b_anchor_diagnosis.py", REPO_ROOT / "model/adapter.py",
        REPO_ROOT / "tools/audit_p4_k1_oracle_utility.py", REPO_ROOT / "tools/audit_p4_k1_direct_residual_gate.py",
    ]
    report = {
        "decision": "STAGE1_9A_MATURE_PHASE2B_DIRECT_GATE_AUDIT_COMPLETE",
        "provenance": {
            "repo_sha": _git_sha(), "script_version": SCRIPT_VERSION,
            "script_sha256": _sha256(Path(__file__).resolve()),
            "source_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_paths},
            "dataset": "VisA", "split": "train", "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
            "manifest_sha256": _sha256(manifest_path), "seed": args.seed,
            "precision": "fp32; TF32 off; AMP off; deterministic no-augmentation Train transform", "optimizer_steps": 0,
        },
        "phase2b_candidate": {
            "checkpoint_path": str(phase2b_path.relative_to(REPO_ROOT)), "checkpoint_sha256": _sha256(phase2b_path),
            "checkpoint_epoch": loaded_epoch, "checkpoint_git_sha": phase2b_checkpoint.get("git_sha"),
            "documented_selection": "Phase2B e10 current_shared; hybrid alpha=.20 segmentation branch; documentation reports 6-medical mean pixel AUC/AP 90.98/40.35.",
            "verified_checkpoint_metadata": {key: phase2b_checkpoint.get(key) for key in ("n_groups", "dfg_mode", "dfg_attn_dim", "dfg_attn_tau", "use_ss2d_dfg", "dfg_ss2d_fusion", "dfg_beta_current", "prompt_mode", "use_hybrid_soft_prompt", "hybrid_alpha_current", "soft_prompt_ctx_len")},
        },
        "k1_candidate": {"checkpoint_path": str(k1_checkpoint_path.relative_to(REPO_ROOT)), "checkpoint_sha256": _sha256(k1_checkpoint_path), "checkpoint_git_sha": k1_checkpoint.get("git_sha"), "config_sha256": _sha256(k1_config_path), "rho": [float(value) for value in k1.h6.rho_values().cpu()]},
        "map_operator": "Exact Phase2B vision_text_fusion_gate_seg(test_mode=True, domain=DOMAINS['VisA']): DFG-adjusted group logits, medical blur sigma=1.5/kernel=9, bilinear interpolation align_corners=True, group mean, abnormal softmax probability. K1 transfer is F.adaptive_avg_pool2d of that final full-resolution probability to each 26x26 K1 patch grid, shared across K1 groups.",
        "gate_mode": args.gate_mode,
        "gate_formula": "L_final = L_base + rho * g_patch * (L_A1 - L_base); g_patch is frozen and detached by inference_mode.",
        "gate_calibration": "raw: g=p2b. image_centered: g=clamp((p2b-mean_image(p2b))/(1-mean_image(p2b)),0,1), with no learned parameters or threshold.",
        "localization": localization,
        "k1_utility_alignment": alignment,
        "variants": report_variants,
        "mature_p2b_gate": {region: _summary(gates[region]) for region in REGIONS},
        "image_level_gain_bootstrap": image_bootstrap,
        "per_class": class_summary,
        "per_image": per_image,
        "interpretation": "Inference-only attribution. Positive gain is BCE(BASE)-BCE(variant). Utility is BCE(A0)-BCE(A1) before rho; positive utility favors the K1 dynamic semantic.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": report["decision"], "localization": localization,
        "p2b_alignment_all": alignment["mature_p2b"]["all"],
        "gains": {name: report_variants[name]["gain_base_minus_final"] for name in VARIANTS},
        "gate": report["mature_p2b_gate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
