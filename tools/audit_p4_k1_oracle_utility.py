#!/usr/bin/env python3
"""Deterministic image-level Stage 1.7R oracle utility audit for trained K1.

This is deliberately inference-only.  Its manifest is selected from the
VisA *TRAIN* JSONL by a public hash rule, never from model outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tv_transforms
from torchvision.transforms import InterpolationMode

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.info import DATA_PATH
from model.adapter import ACDCLIP
from model.checkpoint_utils import load_adapter_checkpoint
from model.clip import create_model
from utils import configure_canonical_fp32, get_phase2b_global_text_features


SCRIPT_VERSION = "stage1_7r_oracle_utility_v1"
PROBE_GROUPS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _rank_key(seed: int, row: dict[str, Any]) -> str:
    value = f"{seed}|{row['class_name']}|{row['label']}|{row['image_path']}"
    return hashlib.sha256(value.encode()).hexdigest()


def build_manifest(meta_path: Path, output: Path, seed: int, per_class_label: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["class_name"], int(row["label"]))].append(row)
    selected: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, int]] = {}
    for class_name in sorted({row["class_name"] for row in rows}):
        coverage[class_name] = {}
        for label in (0, 1):
            candidates = sorted(buckets[(class_name, label)], key=lambda row: _rank_key(seed, row))
            if len(candidates) < per_class_label:
                raise RuntimeError(f"{class_name}/label={label} has only {len(candidates)} candidates")
            coverage[class_name][str(label)] = len(candidates)
            for row in candidates[:per_class_label]:
                selected.append({
                    "class_name": row["class_name"], "label": int(row["label"]),
                    "image_path": row["image_path"], "mask_path": row.get("mask_path"),
                    "selection_hash": _rank_key(seed, row),
                })
    selected.sort(key=lambda row: (row["class_name"], row["label"], row["selection_hash"]))
    manifest = {
        "schema_version": 1,
        "selection_rule": "take the lowest SHA256(seed|class_name|label|image_path) rows per class/label",
        "seed": seed,
        "dataset": "VisA",
        "split": "train",
        "meta_path": str(meta_path.relative_to(REPO_ROOT)),
        "meta_sha256": _sha256(meta_path),
        "per_class_label": per_class_label,
        "candidate_counts": coverage,
        "samples": selected,
    }
    manifest["ordered_samples_sha256"] = hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class DeterministicVisATrainDataset(Dataset):
    """Uses Train metadata/images but removes all stochastic training augmentation."""

    def __init__(self, manifest: dict[str, Any], image_size: int) -> None:
        self.samples = manifest["samples"]
        self.data_root = Path(DATA_PATH["VisA"])
        self.image_size = image_size
        self.image_transform = tv_transforms.Compose([
            tv_transforms.Resize((image_size, image_size), InterpolationMode.BICUBIC),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        self.mask_transform = tv_transforms.Compose([
            tv_transforms.Resize((image_size, image_size), InterpolationMode.NEAREST),
            tv_transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        image = self.image_transform(Image.open(self.data_root / row["image_path"]).convert("RGB"))
        if row["label"]:
            mask = (self.mask_transform(Image.open(self.data_root / row["mask_path"]).convert("L")) > 0).float()
        else:
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        return {
            "image": image,
            "mask": mask,
            "local_mask_valid": torch.ones_like(mask),
            "label": torch.tensor(row["label"], dtype=torch.int64),
            "file_name": row["image_path"],
            "class_name": row["class_name"],
        }


def _build_model(config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device) -> ACDCLIP:
    parameters = inspect.signature(ACDCLIP.__init__).parameters
    kwargs = {
        name: config[name] for name, parameter in parameters.items()
        if name not in {"self", "clip_model", "kwargs"}
        and parameter.kind is not inspect.Parameter.VAR_KEYWORD and name in config
    }
    kwargs.update({
        "dfg_beta_current": 0.0,
        "dfg_weight_residual_fp32": True,
        "h6_role_topology": config["h6_role_topology"],
        "h6_role_teacher_scale": config["h6_role_teacher_scale"],
        "h6_intrinsic_factor_responsibility": config["h6_intrinsic_factor_responsibility"],
        "h6_prediction_routing": config["h6_prediction_routing"],
        "diagnostics_mode": config["h6_diagnostics_mode"],
        "diagnostics_interval": config["h6_diagnostics_interval"],
        "h6_cluster_responsibility": config["h6_cluster_responsibility"],
        "h6_cluster_temperature": config["h6_cluster_temperature"],
        "h6_router_boundary_mode": config["h6_router_boundary_mode"],
        "h6_router_boundary_trust_scale": config["h6_router_boundary_trust_scale"],
    })
    clip = create_model(config["model_name"], img_size=config["img_size"], device=device, pretrained="openai", require_pretrained=True, precision="fp32")
    if config["grad_checkpointing"]:
        clip.set_grad_checkpointing(True)
    model = ACDCLIP(clip_model=clip, **kwargs).to(device)
    model.use_soft_prompt = False
    model.use_hybrid_soft_prompt = True
    model.prompt_mode = "h6_dynamic"
    load_adapter_checkpoint(model, checkpoint)
    model.set_dfg_beta(0.0)
    model.eval()
    model.clipmodel.eval()
    return model


def _mean(values: torch.Tensor) -> float | None:
    return None if values.numel() == 0 else float(values.float().mean())


def _median(values: torch.Tensor) -> float | None:
    return None if values.numel() == 0 else float(values.float().median())


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2 or x.std(unbiased=False) == 0 or y.std(unbiased=False) == 0:
        return None
    return float(torch.corrcoef(torch.stack((x.float(), y.float())))[0, 1])


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2:
        return None
    return _pearson(torch.argsort(torch.argsort(x)).float(), torch.argsort(torch.argsort(y)).float())


def _auroc(scores: torch.Tensor, positive: torch.Tensor) -> float | None:
    positive = positive.bool()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _utility_metrics(score: torch.Tensor, utility: torch.Tensor) -> dict[str, Any]:
    if not utility.numel():
        return {"count": 0, "mean_utility": None, "median_utility": None, "fraction_a1_better": None, "fraction_a0_better": None, "pearson": None, "spearman": None, "sign_agreement": None, "auroc_a1_better": None}
    a1_better = utility > 0
    return {
        "count": int(utility.numel()),
        "mean_utility": _mean(utility),
        "median_utility": _median(utility),
        "fraction_a1_better": _mean(a1_better.float()),
        "fraction_a0_better": _mean((utility < 0).float()),
        "pearson": _pearson(score, utility),
        "spearman": _spearman(score, utility),
        "sign_agreement": _mean(((score > 0) == a1_better).float()),
        "auroc_a1_better": _auroc(score, a1_better),
    }


def _bootstrap_mean(values: torch.Tensor, seed: int, draws: int = 1000) -> dict[str, Any]:
    if not values.numel():
        return {"n_images": 0, "mean": None, "ci95": None}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    picks = torch.randint(values.numel(), (draws, values.numel()), generator=generator)
    boot = values[picks].mean(dim=1)
    return {"n_images": int(values.numel()), "mean": float(values.mean()), "ci95": [float(torch.quantile(boot, 0.025)), float(torch.quantile(boot, 0.975))]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/phase4/k1/short64_seed0_attempt5"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase4/k1/stage1_7r"))
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--per-class-label", type=int, default=2)
    parser.add_argument("--build-manifest-only", action="store_true")
    args = parser.parse_args()
    run_dir, output_dir = args.run_dir.resolve(), args.output_dir.resolve()
    meta_path = REPO_ROOT / "dataset/hub/VisA.jsonl"
    manifest_path = output_dir / "visa_train_audit_manifest.json"
    manifest = build_manifest(meta_path, manifest_path, args.seed, args.per_class_label)
    if args.build_manifest_only:
        print(json.dumps({"manifest": str(manifest_path), "sample_count": len(manifest["samples"]), "sha256": manifest["ordered_samples_sha256"]}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1.7R oracle audit requires CUDA")
    config_path, checkpoint_path = run_dir / "config.json", run_dir / "adapter_1.pth"
    config = json.loads(config_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if config["h6_progress_version"] != "P4-CSF-K1":
        raise RuntimeError("oracle audit requires the trained K1 checkpoint")
    configure_canonical_fp32()
    device = torch.device(f"cuda:{config['cuda_device']}")
    model = _build_model(config, checkpoint, device)
    loader = DataLoader(DeterministicVisATrainDataset(manifest, config["img_size"]), batch_size=1, shuffle=False, num_workers=0)
    patch = {region: {name: [] for name in ("margin", "tangent", "contrastive_local", "utility", "final_utility")} for region in ("all", "normal", "anomaly")}
    image_rows: list[dict[str, Any]] = []
    image_region_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(loader):
        image = raw["image"].to(device=device, dtype=torch.float32)
        mask = raw["mask"].to(device=device, dtype=torch.float32)
        class_name, file_name, label = raw["class_name"][0], raw["file_name"][0], int(raw["label"].item())
        with torch.inference_mode():
            visual = model(image, return_phase4_features=True)
            base_text = get_phase2b_global_text_features(model, "VisA", [class_name], device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
            batch = model.h6.build_batch(model, "VisA", [class_name], visual, hybrid_alpha=0.0, base_text_features=base_text, debug=True)
            base_cross, dynamic_cross = base_text.permute(1, 0, 2, 3).float(), batch["dynamic_text"].permute(1, 0, 2, 3).float()
            state_norm = float(batch["state_delta_raw"].float().norm(dim=-1).mean().item())
            class_norm = float(batch["class_delta_raw"].float().norm(dim=-1).mean().item())
            image_region_measurements = {region: [] for region in ("all", "normal", "anomaly")}
            image_group_measurements = []
            for group in range(PROBE_GROUPS):
                weights_n, weights_a = batch["base_dfg_weights_normal"][group], batch["base_dfg_weights_abnormal"][group]
                a0 = model.apply_dfg_weights(base_cross, weights_n, weights_a)[..., 1]
                a1 = model.apply_dfg_weights(dynamic_cross, weights_n, weights_a)[..., 1]
                v = visual["seg_tokens"][group].float()
                residual = a1.unsqueeze(1) - a0.unsqueeze(1)
                tangent_r = residual - (residual * a0.unsqueeze(1)).sum(dim=-1, keepdim=True) * a0.unsqueeze(1)
                tangent_v = v - (v * a0.unsqueeze(1)).sum(dim=-1, keepdim=True) * a0.unsqueeze(1)
                margin = (v * residual).sum(dim=-1)
                tangent = (tangent_v * tangent_r).sum(dim=-1)
                side = int(v.shape[1] ** 0.5)
                target = F.adaptive_avg_pool2d(mask, (side, side)).flatten(1)
                valid = torch.ones_like(target, dtype=torch.bool)
                base_logit = batch["base_group_logits"][group, ..., 1] - batch["base_group_logits"][group, ..., 0]
                posterior = torch.sigmoid(base_logit).detach()
                z_anomaly = torch.einsum("bp,bpd->bd", posterior, v) / posterior.sum(dim=1, keepdim=True).clamp_min(1e-6)
                z_normal = torch.einsum("bp,bpd->bd", 1.0 - posterior, v) / (1.0 - posterior).sum(dim=1, keepdim=True).clamp_min(1e-6)
                local_contrast_direction = z_anomaly - z_normal
                contrastive_local = torch.einsum("bpd,bd->bp", v, local_contrast_direction)
                a1_logit = batch["dynamic_abnormal_logits"][group] - batch["base_group_logits"][group, ..., 0]
                base_bce = F.binary_cross_entropy_with_logits(base_logit, target, reduction="none")
                a1_bce = F.binary_cross_entropy_with_logits(a1_logit, target, reduction="none")
                final_bce = F.binary_cross_entropy_with_logits(batch["final_group_logits"][group, ..., 1] - batch["final_group_logits"][group, ..., 0], target, reduction="none")
                utility, final_utility = base_bce - a1_bce, base_bce - final_bce
                masks = {"all": valid, "normal": valid & (target == 0), "anomaly": valid & (target > 0)}
                for region, region_mask in masks.items():
                    patch[region]["margin"].append(margin[region_mask].cpu())
                    patch[region]["tangent"].append(tangent[region_mask].cpu())
                    patch[region]["utility"].append(utility[region_mask].cpu())
                    patch[region]["contrastive_local"].append(contrastive_local[region_mask].cpu())
                    patch[region]["final_utility"].append(final_utility[region_mask].cpu())
                    image_region_measurements[region].append({"utility_mean": _mean(utility[region_mask]), "margin_mean": _mean(margin[region_mask]), "tangent_mean": _mean(tangent[region_mask]), "contrastive_local_mean": _mean(contrastive_local[region_mask]), "count": int(region_mask.sum())})
                group_rows.append({"image_index": index, "group": group, "utility_mean": _mean(utility[valid]), "margin_mean": _mean(margin[valid]), "tangent_mean": _mean(tangent[valid]), "state_delta_raw_l2": state_norm, "class_delta_raw_l2": class_norm, "anomaly_patch_count": int(masks["anomaly"].sum())})
            for region, measurements in image_region_measurements.items():
                if sum(row["count"] for row in measurements): image_region_rows.append({"image_index": index, "class_name": class_name, "file_name": file_name, "label": label, "region": region, "utility_mean": sum(row["utility_mean"] * row["count"] for row in measurements) / sum(row["count"] for row in measurements), "margin_mean": sum(row["margin_mean"] * row["count"] for row in measurements) / sum(row["count"] for row in measurements), "tangent_mean": sum(row["tangent_mean"] * row["count"] for row in measurements) / sum(row["count"] for row in measurements), "contrastive_local_mean": sum(row["contrastive_local_mean"] * row["count"] for row in measurements) / sum(row["count"] for row in measurements), "patch_count": sum(row["count"] for row in measurements)})
                if region == "all":
                    image_rows.append({"image_index": index, "class_name": class_name, "file_name": file_name, "label": label, "utility_mean": image_region_rows[-1]["utility_mean"], "margin_mean": image_region_rows[-1]["margin_mean"], "tangent_mean": image_region_rows[-1]["tangent_mean"], "state_delta_raw_l2_mean": state_norm, "class_delta_raw_l2": class_norm, "anomaly_patch_count": sum(row["count"] for row in image_region_measurements["anomaly"])})
    patch_report = {}
    for region, values in patch.items():
        utility = torch.cat(values["utility"])
        patch_report[region] = {"raw_affinity": _utility_metrics(torch.cat(values["margin"]), utility), "tangent_residual": _utility_metrics(torch.cat(values["tangent"]), utility), "contrastive_local": _utility_metrics(torch.cat(values["contrastive_local"]), utility), "final_rho_utility_mean": _mean(torch.cat(values["final_utility"]))}
    image_t = {key: torch.tensor([row[key] for row in image_rows], dtype=torch.float32) for key in ("utility_mean", "margin_mean", "tangent_mean", "state_delta_raw_l2_mean", "class_delta_raw_l2")}
    image_report = {"raw_affinity": _utility_metrics(image_t["margin_mean"], image_t["utility_mean"]), "tangent_residual": _utility_metrics(image_t["tangent_mean"], image_t["utility_mean"]), "utility_bootstrap": _bootstrap_mean(image_t["utility_mean"], args.seed), "raw_state_magnitude_vs_utility_pearson": _pearson(image_t["state_delta_raw_l2_mean"], image_t["utility_mean"]), "raw_class_magnitude_vs_utility_pearson": _pearson(image_t["class_delta_raw_l2"], image_t["utility_mean"])}
    source_paths = [REPO_ROOT / "model/h6/model.py", REPO_ROOT / "model/h6/conditional_semantics.py", REPO_ROOT / "tools/audit_p4_k1_attribution.py", Path(__file__).resolve()]
    image_region_report = {}
    for region in ("all", "normal", "anomaly"):
        rows = [row for row in image_region_rows if row["region"] == region]
        values = {key: torch.tensor([row[key] for row in rows], dtype=torch.float32) for key in ("utility_mean", "margin_mean", "tangent_mean", "contrastive_local_mean")}
        image_region_report[region] = {"raw_affinity": _utility_metrics(values["margin_mean"], values["utility_mean"]), "tangent_residual": _utility_metrics(values["tangent_mean"], values["utility_mean"]), "contrastive_local": _utility_metrics(values["contrastive_local_mean"], values["utility_mean"]), "utility_bootstrap": _bootstrap_mean(values["utility_mean"], args.seed)}

    report = {"decision": "STAGE1_7R_ORACLE_AUDIT_COMPLETE", "provenance": {"repo_sha": _git_sha(), "script_version": SCRIPT_VERSION, "script_sha256": _sha256(Path(__file__).resolve()), "source_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_paths}, "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)), "checkpoint_sha256": _sha256(checkpoint_path), "checkpoint_git_sha": checkpoint.get("git_sha"), "config_sha256": _sha256(config_path), "dataset": "VisA", "split": "train", "manifest_path": str(manifest_path.relative_to(REPO_ROOT)), "manifest_sha256": _sha256(manifest_path), "seed": args.seed, "precision": "fp32; TF32 off; AMP off; deterministic no-augmentation Train transform", "optimizer_steps": 0}, "manifest_summary": {"image_count": len(image_rows), "anomaly_label_count": sum(row["label"] for row in image_rows), "normal_label_count": sum(not row["label"] for row in image_rows), "images_with_anomaly_patch": sum(row["anomaly_patch_count"] > 0 for row in image_rows)}, "patch_level": patch_report, "image_level": image_report, "image_level_by_region": image_region_report, "per_image": image_rows, "per_image_region": image_region_rows, "per_group": group_rows, "interpretation": "u=BCE(A0)-BCE(A1), with A1 evaluated as the original K1 dynamic abnormal predictor before rho interpolation; positive u favors A1."}
    output_path = output_dir / "K1_ORACLE_UTILITY_AUDIT.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": report["decision"], "manifest": report["manifest_summary"], "image_raw": image_report["raw_affinity"], "image_tangent": image_report["tangent_residual"], "patch_raw": patch_report["all"]["raw_affinity"], "patch_tangent": patch_report["all"]["tangent_residual"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
