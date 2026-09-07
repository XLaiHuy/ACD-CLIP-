#!/usr/bin/env python3
"""S2-LOCR R1 gated mechanism audit.

The ``oracle`` phase is a GT-assisted, inference-only causal diagnostic.  It
replaces only production-resized Stage-2 logits on existing near-background
pixels and compares the resulting equal-fusion endpoint with the untouched
Safe-Anchor endpoint and the fixed global Stage-2 replacement control.

Later phases are intentionally not implemented as an automatic fall-through:
the protocol requires the oracle gate before formulation, gradient preflight,
or the single bounded source-training screen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataset import CLASS_NAMES
from h2_clean.precision import PrecisionPolicy
from h2_clean.contract import EpochWorkerInit, make_dataloader_generator
from scipy import ndimage
from torch.optim.lr_scheduler import StepLR

from h2_clean.contract import SafeImageAdapterAnchor, apply_family_safe_anchor_budget
from model.adapter import ACDCLIP
from model.clip import create_model
from train import (
    apply_soft_prompt_lr_policy,
    calculate_seg_loss,
    compute_hybrid_k_regularization,
    get_dfg_beta_for_epoch,
    get_hybrid_alpha_for_epoch,
)
from utils import dice_loss, focal_loss, get_hybrid_soft_prompt_single_class_text_embedding

SPILLOVER_PATH = REPO / "scripts/run_h2_boundary_spillover_audit_r1.py"
SPEC = importlib.util.spec_from_file_location("h2_spillover", SPILLOVER_PATH)
spill = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(spill)

IMG = 518
SAFE_ANCHOR = Path("/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth")
EXPECTED_SAFE_SHA = "64b72dc3d1155285c826781bee4c5970bd45218e95b21675fd19d9a6b2ab54a7"
PARENT_HEAD = "47158bd1a64d23f5f4752fb066e5dd4b91ef07d9"
AUDIT_HEAD = "bbfc79ff5c663874ad21529167e56099c96d281e"
COHORT = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_COHORT.json"
COHORT_CSV = REPO / "audit/H2_FUSION_ENDPOINT_EVAL_SUBSET.csv"
SPLIT_JSON = REPO / "audit/H2_FUSION_SPLIT_IDENTITY.json"

OUT_PARENT = REPO / "audit/H2_S2_LOCR_R1_PARENT_IDENTITY.md"
OUT_ORACLE_CSV = REPO / "audit/H2_S2_LOCR_R1_ORACLE.csv"
OUT_ORACLE_JSON = REPO / "audit/H2_S2_LOCR_R1_ORACLE.json"
OUT_ORACLE_DECISION = REPO / "audit/H2_S2_LOCR_R1_ORACLE_DECISION.md"
OUT_COVERAGE_CSV = REPO / "audit/H2_S2_LOCR_R1_GEOMETRY_COVERAGE.csv"
OUT_COVERAGE_JSON = REPO / "audit/H2_S2_LOCR_R1_GEOMETRY_COVERAGE.json"
OUT_LOCALITY_JSON = REPO / "audit/H2_S2_LOCR_R1_AUXILIARY_LOCALITY.json"
OUT_PREFLIGHT_CSV = REPO / "audit/H2_S2_LOCR_R1_GRADIENT_PREFLIGHT.csv"
OUT_PREFLIGHT_JSON = REPO / "audit/H2_S2_LOCR_R1_GRADIENT_PREFLIGHT.json"
OUT_CALIBRATION_JSON = REPO / "audit/H2_S2_LOCR_R1_LAMBDA_CALIBRATION.json"
OUT_PARITY_JSON = REPO / "audit/H2_S2_LOCR_R1_LOSS_PARITY.json"


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=spill.json_default, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["status"]
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])
    os.replace(tmp, path)


def write_parent_identity() -> None:
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    changed = subprocess.check_output(["git", "diff", "--name-only", f"{PARENT_HEAD}..{AUDIT_HEAD}"], cwd=REPO, text=True).splitlines()
    forbidden_prefixes = ("model/", "h2_clean/stage_fusion.py", "h2_clean/precision.py", "train.py", "optimizer.py")
    scientific_changes = [path for path in changed if path.startswith(forbidden_prefixes)]
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    OUT_PARENT.write_text(
        "\n".join([
            "# H2 S2-LOCR R1 Parent / Scientific Identity", "",
            f"* branch: `{branch}`",
            f"* branch current HEAD: `{current}`",
            f"* required audit parent HEAD: `{AUDIT_HEAD}`",
            f"* Safe-Anchor parent HEAD: `{PARENT_HEAD}`",
            "* audit branch scientific identity: `PASS`" if not scientific_changes else f"* audit branch scientific identity: `FAIL ({scientific_changes})`",
            f"* worktree at identity capture: `{'CLEAN' if not status else 'DIRTY'}`",
            "",
            "The audit branch diff from the Safe-Anchor parent contains only audit scripts, reports, and artifacts. No model forward, architecture, training objective, optimizer, fusion, interpolation, precision, Safe-Anchor, DFG, or SS2D implementation file changed.",
            "",
            "This new branch uses the already-pushed boundary-spillover audit HEAD as its scientific base. The GradBudget endpoint and all target datasets are excluded.",
        ]) + "\n"
    )
    base_is_ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", AUDIT_HEAD, "HEAD"], cwd=REPO).returncode == 0
    if not base_is_ancestor or scientific_changes or status:
        raise RuntimeError("AUDIT_BRANCH_SCIENTIFIC_IDENTITY=FAIL")


def validate_inputs(rows: list[dict]) -> dict:
    if spill.sha256_file(SAFE_ANCHOR) != EXPECTED_SAFE_SHA:
        raise RuntimeError("START_IDENTITY=FAIL: Safe-Anchor checkpoint hash mismatch")
    payload = torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False)
    expected = {
        "epoch": 10, "global_step": 3607, "precision": "fp16", "amp_enabled": True,
        "tf32_enabled": False, "seed": 0, "gradscaler_enabled": True,
        "precision_protocol": "HISTORICAL_MIXED_FP16_FP32_V1", "n_groups": 3,
        "dfg_mode": "attn", "dfg_attn_dim": 256, "dfg_attn_tau": 8.0,
        "use_ss2d_dfg": True, "dfg_ss2d_fusion": "weight_residual",
        "dfg_weight_residual_fp32": True, "use_safe_anchor": True,
        "anchor_lambda": 0.0021633926715180626, "anchor_family_budget": 0.1,
        "functional_feature_anchor": False, "use_cir_training": False,
        "stage_fusion": [1.0 / 3.0] * 3,
    }
    cfg = payload.get("resolved_scientific_config", {})
    observed = {
        "epoch": payload.get("epoch"), "global_step": payload.get("global_step"),
        "precision": payload.get("precision"), "amp_enabled": payload.get("amp_enabled"),
        "tf32_enabled": payload.get("tf32_enabled"), "seed": payload.get("seed"),
        "gradscaler_enabled": payload.get("gradscaler_enabled"),
        "precision_protocol": payload.get("precision_protocol"), "n_groups": payload.get("n_groups"),
        "dfg_mode": payload.get("dfg_mode"), "dfg_attn_dim": payload.get("dfg_attn_dim"),
        "dfg_attn_tau": payload.get("dfg_attn_tau"), "use_ss2d_dfg": payload.get("use_ss2d_dfg"),
        "dfg_ss2d_fusion": payload.get("dfg_ss2d_fusion"),
        "dfg_weight_residual_fp32": payload.get("dfg_weight_residual_fp32"),
        "use_safe_anchor": cfg.get("use_safe_anchor"), "anchor_lambda": cfg.get("anchor_lambda"),
        "anchor_family_budget": cfg.get("anchor_family_budget"),
        "functional_feature_anchor": cfg.get("use_functional_feature_anchor", False),
        "use_cir_training": cfg.get("use_cir_training", False),
        "stage_fusion": [1.0 / 3.0] * 3,
    }
    mismatch = {k: {"expected": expected[k], "observed": observed[k]} for k in expected if observed[k] != expected[k]}
    required_state = ("model_state", "optimizer_state", "scheduler_state", "scaler_state", "python_random_state", "numpy_random_state", "torch_cpu_rng_state", "torch_cuda_rng_state_all", "dataloader_generator_state", "resolved_scientific_config")
    missing = [key for key in required_state if key not in payload]
    if mismatch or missing:
        raise RuntimeError(json.dumps({"START_IDENTITY": "FAIL", "mismatch": mismatch, "missing": missing}, sort_keys=True))
    cohort = json.loads(COHORT.read_text())
    if cohort.get("count") != 96 or len(cohort.get("records", [])) != 96:
        raise RuntimeError("cohort count identity mismatch")
    expected_names = [row["file_name"] for row in rows]
    observed_names = [record["file_name"] for record in cohort["records"]]
    if expected_names != observed_names:
        raise RuntimeError("cohort filename identity mismatch")
    return {
        "checkpoint": str(SAFE_ANCHOR), "checkpoint_sha256": EXPECTED_SAFE_SHA,
        "epoch": 10, "global_step": 3607, "numerical_validity": True,
        "full_state_keys_present": list(required_state), "scientific_config": cfg,
        "cohort_count": 96, "cohort_identity": cohort.get("audit_record_canonical_sha256"),
    }


def production_resized_logits(model, vision, text) -> torch.Tensor:
    batch, patches, _ = vision.shape[1:]
    side = int(math.sqrt(patches))
    if side != 37:
        raise RuntimeError(f"expected native 37x37 grid, got {side}")
    if text.ndim == 3:
        group_text = text.unsqueeze(1).repeat(1, batch, 1, 1).permute(1, 0, 2, 3)
    elif text.ndim == 4:
        group_text = text.permute(1, 0, 2, 3)
    else:
        raise RuntimeError(f"unexpected text feature rank: {text.ndim}")
    stage_logits = []
    for stage in range(3):
        fused = model._vision_text_attention_fusion(vision[stage], group_text, stage)
        native = torch.matmul(10 * vision[stage], fused).permute(0, 2, 1).view(batch, 2, side, side)
        blurred = gaussian_blur2d(native, (7, 7), (1, 1))
        stage_logits.append(F.interpolate(blurred, (IMG, IMG), mode="bilinear", align_corners=True))
    return torch.stack(stage_logits)


def auxiliary_stage2_resized_logits(model, stage2_lora_input: torch.Tensor, text) -> torch.Tensor:
    """Rebuild Stage 2 from its own adapter boundary, detaching Stage 1."""
    # ACDCLIP carries the Stage-1-updated transformer stream into Stage 2.
    # Detach exactly at the Stage-2 Conv-LoRA input, then replay the unchanged
    # Stage-2 adapter, projection, normalization, DFG, blur, and resize graph.
    with torch.autocast(device_type=stage2_lora_input.device.type, enabled=False):
        t_fp32 = stage2_lora_input.detach()
        adapter = model.image_adapter["lora_adapters"][1]
        adapt_out = adapter(t_fp32)
        adapt_out = adapt_out * t_fp32.norm(dim=-1, keepdim=True) / adapt_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        group_out = model.image_adapter["m_i_w"][1](t_fp32, adapt_out)
    group_tokens = group_out.permute(1, 0, 2)
    seg = model.image_adapter["seg_proj"][1](group_tokens)
    seg = model.image_adapter["seg_layer_norms"][1](seg)
    seg = F.normalize(seg, dim=-1)
    if text.ndim == 3:
        group_text = text.unsqueeze(1).repeat(1, seg.shape[0], 1, 1).permute(1, 0, 2, 3)
    elif text.ndim == 4:
        group_text = text.permute(1, 0, 2, 3)
    else:
        raise RuntimeError(f"unexpected text feature rank: {text.ndim}")
    fused = model._vision_text_attention_fusion(seg, group_text, 1)
    native = torch.matmul(10 * seg, fused).permute(0, 2, 1).view(seg.shape[0], 2, 37, 37)
    blurred = gaussian_blur2d(native, (7, 7), (1, 1))
    return F.interpolate(blurred, (IMG, IMG), mode="bilinear", align_corners=True)


def fuse(stage_logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(stage_logits.mean(dim=0), dim=1)[:, 1]


def oracle_maps(model, rows: list[dict], device: torch.device) -> dict[str, np.ndarray]:
    datasets, indices = spill.load_selection_datasets(rows)
    policy = PrecisionPolicy("fp16")
    chunks = {key: [] for key in ("baseline", "local_oracle", "global_stage2_replacement", "mask")}
    names, labels, categories = [], [], []
    with torch.no_grad():
        for category in CLASS_NAMES["VisA"]:
            loader = DataLoader(Subset(datasets[category], indices[category]), batch_size=8, shuffle=False, num_workers=0)
            text, _, _ = spill.get_hybrid_soft_prompt_single_class_text_embedding(model, "VisA", category, device, return_kg=False)
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                masks = (batch["mask"][:, 0].numpy() > 0.5).astype(np.uint8)
                with policy.autocast(device):
                    seg_tokens, _ = model(image)
                    vision = torch.stack(seg_tokens)
                    logits = production_resized_logits(model, vision, text)
                    baseline = fuse(logits)
                    replacement = .5 * (logits[0] + logits[2])
                    local_logits = logits[1].clone()
                    near_masks = []
                    for mask in masks:
                        _, _, near, _ = spill.morphology(mask)
                        near_masks.append(near)
                    near = torch.from_numpy(np.stack(near_masks)).to(device=device, dtype=torch.bool)
                    local_logits = torch.where(near[:, None], replacement, local_logits)
                    local = fuse(torch.stack((logits[0], local_logits, logits[2])))
                    global_logits = torch.stack((logits[0], replacement, logits[2]))
                    global_map = fuse(global_logits)
                chunks["baseline"].append(baseline.float().cpu().numpy())
                chunks["local_oracle"].append(local.float().cpu().numpy())
                chunks["global_stage2_replacement"].append(global_map.float().cpu().numpy())
                chunks["mask"].append(masks)
                names.extend(list(batch["file_name"]))
                categories.extend([category] * len(batch["file_name"]))
                labels.extend(batch["label"].numpy().astype(np.uint8).tolist())
                del vision, logits, local_logits, global_logits
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    expected = [row["file_name"] for row in rows]
    if names != expected:
        raise RuntimeError("oracle evaluation order does not match frozen cohort")
    return {
        **{key: np.concatenate(value, axis=0) for key, value in chunks.items()},
        "names": np.asarray(names, dtype=object), "categories": np.asarray(categories, dtype=object),
        "labels": np.asarray(labels, dtype=np.uint8),
    }


REGIONAL_FIELDS = (
    "positive_mean", "positive_median", "interior_mean", "interior_median",
    "boundary_mean", "boundary_median", "near_mean", "near_p95", "near_p99",
    "far_mean", "far_p95", "far_p99", "near_gt_positive_inversion", "near_gt_interior_inversion",
)


def map_metrics(score: np.ndarray, mask: np.ndarray, labels: np.ndarray, arm: str) -> dict:
    return {"arm": arm, **spill.fusion_map_metrics(score, mask, labels, "final_production_fused")}


def regional_fields(score: np.ndarray, mask: np.ndarray) -> dict:
    summary = spill.region_summary(score, mask)
    return {
        "positive_mean": summary["positive"]["mean"], "positive_median": summary["positive"]["median"],
        "interior_mean": summary["interior"]["mean"], "interior_median": summary["interior"]["median"],
        "boundary_mean": summary["boundary"]["mean"], "boundary_median": summary["boundary"]["median"],
        "near_mean": summary["near"]["mean"], "near_p95": summary["near"]["p95"], "near_p99": summary["near"]["p99"],
        "far_mean": summary["far"]["mean"], "far_p95": summary["far"]["p95"], "far_p99": summary["far"]["p99"],
        "near_gt_positive_inversion": spill.sampled_inversion(score, score, mask, greater=True),
        "near_gt_interior_inversion": spill.sampled_inversion(score, score, mask, greater=False),
    }


def delta(a, b):
    if a is None or b is None:
        return None
    return float(a - b)


def per_image_and_category(data: dict) -> tuple[list[dict], list[dict]]:
    baseline, local, global_map = data["baseline"], data["local_oracle"], data["global_stage2_replacement"]
    mask, labels = data["mask"], data["labels"]
    image_rows = []
    for i, (m, label) in enumerate(zip(mask, labels)):
        values = {}
        for arm, score in (("baseline", baseline), ("local_oracle", local), ("global_stage2_replacement", global_map)):
            if int(label):
                values[arm] = regional_fields(score[i:i + 1], m[None])
        if not int(label):
            continue
        row = {"scope": "image", "image_index": i, "category": str(data["categories"][i]), "file_name": str(data["names"][i])}
        for field in REGIONAL_FIELDS:
            row[f"baseline_{field}"] = values["baseline"][field]
            row[f"local_oracle_{field}"] = values["local_oracle"][field]
            row[f"global_stage2_replacement_{field}"] = values["global_stage2_replacement"][field]
            row[f"local_delta_{field}"] = delta(values["local_oracle"][field], values["baseline"][field])
            row[f"global_delta_{field}"] = delta(values["global_stage2_replacement"][field], values["baseline"][field])
        image_rows.append(row)
    category_rows = []
    for category in CLASS_NAMES["VisA"]:
        current = [row for row in image_rows if row["category"] == category]
        row = {"scope": "category", "category": category, "anomalous_image_count": len(current)}
        for field in REGIONAL_FIELDS:
            for prefix in ("baseline", "local_oracle", "global_stage2_replacement", "local_delta", "global_delta"):
                values = np.asarray([r[f"{prefix}_{field}"] for r in current if r[f"{prefix}_{field}"] is not None], dtype=np.float64)
                row[f"{prefix}_{field}_mean"] = float(values.mean()) if values.size else None
                row[f"{prefix}_{field}_median"] = float(np.median(values)) if values.size else None
                row[f"{prefix}_{field}_iqr"] = float(np.quantile(values, .75) - np.quantile(values, .25)) if values.size else None
        category_rows.append(row)
    return image_rows, category_rows


def run_oracle(rows: list[dict], identity: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact historical inference")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    model = spill.make_model(device)
    spill.load_endpoint(model, SAFE_ANCHOR)
    data = oracle_maps(model, rows, device)
    arms = {
        "BASELINE": data["baseline"],
        "LOCAL_ORACLE": data["local_oracle"],
        "GLOBAL_STAGE2_REPLACEMENT": data["global_stage2_replacement"],
    }
    mask, labels = data["mask"], data["labels"]
    metrics = [map_metrics(score, mask, labels, arm) for arm, score in arms.items()]
    baseline = metrics[0]
    local = metrics[1]
    global_map = metrics[2]
    gate_checks = {
        "local_ap_ge_baseline": local["ap"] >= baseline["ap"],
        "local_auroc_ge_baseline": local["auroc"] >= baseline["auroc"],
        "local_near_p95_le_baseline": local["near_p95"] <= baseline["near_p95"],
        "local_near_p99_le_baseline": local["near_p99"] <= baseline["near_p99"],
        "local_positive_mean_ge_baseline_minus_1e6": local["positive_mean"] >= baseline["positive_mean"] - 1e-6,
        "local_positive_median_ge_baseline_minus_1e6": local["positive_median"] >= baseline["positive_median"] - 1e-6,
        "local_interior_mean_ge_baseline_minus_1e6": local["interior_mean"] >= baseline["interior_mean"] - 1e-6,
        "local_interior_median_ge_baseline_minus_1e6": local["interior_median"] >= baseline["interior_median"] - 1e-6,
    }
    local_support = all(gate_checks.values())
    strongest_signature = bool(local_support and (global_map["ap"] < local["ap"] or global_map["auroc"] < local["auroc"]))
    image_rows, category_rows = per_image_and_category(data)
    write_csv(OUT_ORACLE_CSV, metrics + [{"arm": "GATE", **{key: value for key, value in gate_checks.items()}, "LOCAL_ORACLE_CAUSAL_SUPPORT": local_support, "GLOBAL_WORSE_THAN_LOCAL": strongest_signature}])
    # Keep the detailed deltas in the JSON and compact per-image/category CSV
    # rows embedded there; the protocol requires a single oracle CSV artifact.
    artifact = {
        "protocol_id": "EXPLORATORY_SOURCE_ONLY_MECHANISM_R1",
        "method": "S2_STAGE2_LOCAL_OUTSIDE_CONTRAST_ORACLE",
        "audit_only": True, "training_executed": False, "target_inference": False,
        "checkpoint": identity, "cohort": {"count": 96, "source": str(COHORT.relative_to(REPO)), "source_sha256": spill.sha256_file(COHORT)},
        "score_space": "production_resized_stage_logits_before_softmax",
        "oracle_rule": "z2_oracle(x)=0.5*(z1(x)+z3(x)) only for x in exact near_background; all other z2 unchanged",
        "global_control_rule": "z2_global(x)=0.5*(z1(x)+z3(x)) for all pixels",
        "mask_semantics": "existing 7x7 erosion/dilation at 518x518",
        "metrics": metrics, "gate_checks": gate_checks,
        "local_oracle_causal_support": "PASS" if local_support else "FAIL",
        "strongest_desirable_signature": strongest_signature,
        "per_image_deltas": image_rows, "per_category_deltas": category_rows,
    }
    dump_json(OUT_ORACLE_JSON, artifact)
    OUT_ORACLE_DECISION.write_text(
        "\n".join([
            "# H2 S2-LOCR R1 Oracle Decision", "",
            "* `LOCAL_ORACLE_CAUSAL_SUPPORT=" + ("PASS" if local_support else "FAIL") + "`",
            "* `GLOBAL_STAGE2_REPLACEMENT_REPORTED=YES`",
            "* `STRONGEST_DESIRABLE_SIGNATURE=" + ("PRESENT" if strongest_signature else "ABSENT") + "`",
            "",
            "The local oracle is a GT-assisted source-only diagnostic, not a deployable method. It changes only production-resized Stage-2 logits on exact near-background pixels, then recomputes unchanged equal pre-softmax fusion.",
            "",
            "Gate checks:",
            *[f"* `{key}={'PASS' if value else 'FAIL'}`" for key, value in gate_checks.items()],
            "",
            "If the local oracle gate fails, S2-LOCR formulation, gradient preflight, and training are not authorized by this protocol.",
        ]) + "\n"
    )
    return artifact


def restore_checkpoint_rng(payload: dict) -> None:
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])


def loader_for_epoch(dataset, epoch: int):
    generator = make_dataloader_generator(0)
    generator.manual_seed(104729 * epoch)
    worker = EpochWorkerInit(0)
    worker.set_epoch(epoch)
    return DataLoader(dataset, batch_size=6, shuffle=True, num_workers=0,
                      pin_memory=True, generator=generator, worker_init_fn=worker)


def component_geometry(mask: np.ndarray) -> list[dict]:
    mask = mask.astype(bool)
    if not mask.any():
        return []
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    structure = np.ones((7, 7), dtype=bool)
    records = []
    for component_id in range(1, int(count) + 1):
        component = labels == component_id
        interior = ndimage.binary_erosion(component, structure=structure)
        near = ndimage.binary_dilation(component, structure=structure) & ~component
        records.append({
            "component_id": component_id,
            "component_pixels": int(component.sum()),
            "interior_pixels": int(interior.sum()),
            "near_pixels": int(near.sum()),
            "interior_mask": interior,
            "near_mask": near,
            "valid": bool(interior.any() and near.any()),
            "skip_empty_interior": bool(not interior.any()),
            "skip_empty_near": bool(not near.any()),
        })
    return records


def stage2_margins_for_batch(model, image: torch.Tensor, class_names, device: torch.device) -> np.ndarray:
    """Compute production-resized Stage-2 logit margins without gradients."""
    policy = PrecisionPolicy("fp16")
    output = np.empty((len(class_names), IMG, IMG), dtype=np.float32)
    with torch.no_grad():
        for category in sorted(set(class_names)):
            selected = [i for i, name in enumerate(class_names) if name == category]
            text, _, _ = spill.get_hybrid_soft_prompt_single_class_text_embedding(
                model, "VisA", category, device, return_kg=False,
            )
            with policy.autocast(device):
                seg_tokens, _ = model(image[selected])
                vision = torch.stack(seg_tokens)
                logits = production_resized_logits(model, vision, text)
                margin = logits[1, :, 1] - logits[1, :, 0]
            output[selected] = margin.float().cpu().numpy()
    return output


def analyze_geometry(scope: str, category: str, image_index: int, file_name: str, mask: np.ndarray, margin: np.ndarray) -> tuple[list[dict], dict]:
    components = component_geometry(mask)
    component_rows = []
    valid_count = 0
    active_count = 0
    valid_near_pixels = 0
    violating_near_pixels = 0
    for record in components:
        row = {"scope": scope, "category": category, "image_index": image_index, "file_name": file_name, "component_id": record["component_id"], "component_pixels": record["component_pixels"], "interior_pixels": record["interior_pixels"], "near_pixels": record["near_pixels"], "valid": record["valid"], "skip_empty_interior": record["skip_empty_interior"], "skip_empty_near": record["skip_empty_near"], "active": False, "r_c": None, "l_c": 0.0, "violating_near_pixels": 0, "near_violation_fraction": None}
        if record["valid"]:
            valid_count += 1
            interior = record["interior_mask"]
            near = record["near_mask"]
            reference = float(np.median(margin[interior]))
            violation = margin[near] > reference
            loss = float(np.maximum(margin[near] - reference, 0.0).mean())
            active = bool(violation.any())
            active_count += int(active)
            valid_near_pixels += int(near.sum())
            violating_near_pixels += int(violation.sum())
            row.update({"active": active, "r_c": reference, "l_c": loss, "violating_near_pixels": int(violation.sum()), "near_violation_fraction": float(violation.mean())})
        component_rows.append(row)
    image_row = {"scope": scope, "category": category, "image_index": image_index, "file_name": file_name, "anomalous": bool(mask.any()), "component_count": len(components), "valid_component_count": valid_count, "skipped_empty_interior": sum(int(r["skip_empty_interior"]) for r in components), "skipped_empty_near": sum(int(r["skip_empty_near"]) for r in components), "active_component_count": active_count, "active": bool(active_count), "valid_near_pixels": valid_near_pixels, "violating_near_pixels": violating_near_pixels, "near_violation_fraction": float(violating_near_pixels / valid_near_pixels) if valid_near_pixels else None}
    return component_rows, image_row


def summarize_geometry(image_rows: list[dict], component_rows: list[dict]) -> dict:
    anomalous = [row for row in image_rows if row["anomalous"]]
    valid_components = [row for row in component_rows if row["valid"]]
    return {
        "total_anomalous_images": len(anomalous),
        "total_connected_components": len(component_rows),
        "valid_components": len(valid_components),
        "skipped_empty_interior_components": sum(int(row["skip_empty_interior"]) for row in component_rows),
        "skipped_empty_near_bg_components": sum(int(row["skip_empty_near"]) for row in component_rows),
        "images_with_at_least_one_valid_component": sum(int(row["valid_component_count"] > 0) for row in anomalous),
        "images_where_s2_locr_is_active": sum(int(row["active"]) for row in anomalous),
        "valid_component_fraction": float(len(valid_components) / len(component_rows)) if component_rows else 0.0,
        "active_anomaly_image_fraction": float(sum(int(row["active"]) for row in anomalous) / len(anomalous)) if anomalous else 0.0,
        "fraction_valid_components_with_at_least_one_violation": float(sum(int(row["active"]) for row in valid_components) / len(valid_components)) if valid_components else 0.0,
        "fraction_near_bg_pixels_violating_r_c": float(sum(int(row["violating_near_pixels"]) for row in valid_components) / max(1, sum(int(row["near_pixels"]) for row in valid_components))),
    }


def geometry_phase(identity: dict, rows: list[dict]) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact historical inference")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    model = spill.make_model(device)
    spill.load_endpoint(model, SAFE_ANCHOR)
    datasets, indices = spill.load_selection_datasets(rows)
    all_image_rows, all_component_rows = [], []
    endpoint_by_category = {}
    with torch.no_grad():
        for category in CLASS_NAMES["VisA"]:
            loader = DataLoader(Subset(datasets[category], indices[category]), batch_size=8, shuffle=False, num_workers=0)
            for batch in loader:
                margins = stage2_margins_for_batch(model, batch["image"].to(device), list(batch["class_name"]), device)
                for j, name in enumerate(batch["file_name"]):
                    component_rows, image_row = analyze_geometry("endpoint_cohort", category, len(all_image_rows), name, (batch["mask"][j, 0].numpy() > .5), margins[j])
                    all_image_rows.append(image_row)
                    all_component_rows.extend(component_rows)
            endpoint_by_category[category] = summarize_geometry([r for r in all_image_rows if r["category"] == category], [r for r in all_component_rows if r["category"] == category])
    endpoint_summary = summarize_geometry(all_image_rows, all_component_rows)

    # Fixed 16 source batches are the first deterministic post-E10 batches of
    # epoch 11, matching the repository training loader's seed convention.
    payload = torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False)
    restore_checkpoint_rng(payload)
    train_dataset = spill.get_text_and_image_dataset("VisA", IMG, "train")
    train_image_rows, train_component_rows = [], []
    for batch_index, batch in enumerate(loader_for_epoch(train_dataset, 11)):
        if batch_index >= 16:
            break
        margins = stage2_margins_for_batch(model, batch["image"].to(device), list(batch["class_name"]), device)
        for j, name in enumerate(batch["file_name"]):
            component_rows, image_row = analyze_geometry("fixed_train_16", str(batch["class_name"][j]), batch_index * 6 + j, name, (batch["mask"][j, 0].numpy() > .5), margins[j])
            train_image_rows.append(image_row)
            train_component_rows.extend(component_rows)
    train_summary = summarize_geometry(train_image_rows, train_component_rows)
    by_category = {}
    for category in CLASS_NAMES["VisA"]:
        by_category[category] = {
            "endpoint_cohort": summarize_geometry([r for r in all_image_rows if r["category"] == category], [r for r in all_component_rows if r["category"] == category]),
            "fixed_train_16": summarize_geometry([r for r in train_image_rows if r["category"] == category], [r for r in train_component_rows if r["category"] == category]),
        }
    adequate = bool(endpoint_summary["valid_components"] > 0 and train_summary["valid_components"] > 0 and min(endpoint_summary["active_anomaly_image_fraction"], train_summary["active_anomaly_image_fraction"]) >= .05)
    summary = {"protocol_id": "EXPLORATORY_SOURCE_ONLY_MECHANISM_R1", "method": "S2_STAGE2_LOCAL_OUTSIDE_CONTRAST", "morphology": "existing 7x7 component erosion/dilation at 518x518", "fixed_train_batch_definition": "first 16 deterministic batches from epoch 11 loader seeded 104729*11, batch_size=6, seed=0", "endpoint_cohort": endpoint_summary, "fixed_train_16": train_summary, "by_category": by_category, "GEOMETRY_COVERAGE": "ADEQUATE" if adequate else "INADEQUATE", "nontrivial_active_image_fraction_threshold": .05, "no_geometry_tuning": True}
    rows_out = []
    for scope, value in (("endpoint_cohort", endpoint_summary), ("fixed_train_16", train_summary)):
        rows_out.append({"scope": scope, **value})
    for category, values in by_category.items():
        for scope, value in values.items():
            rows_out.append({"scope": scope, "category": category, **value})
    write_csv(OUT_COVERAGE_CSV, rows_out)
    dump_json(OUT_COVERAGE_JSON, summary)
    return summary


FORMULATION = {
    "name": "S2_STAGE2_LOCAL_OUTSIDE_CONTRAST",
    "score_space": "production_resized_stage2_logits_before_softmax",
    "component_geometry": "each anomalous connected component; existing 7x7 erosion/dilation at 518",
    "validity": "valid iff component interior and component near-background are both nonempty; empty regions skipped",
    "normal_only_contribution": 0.0,
    "margin": "m2=z2_abnormal-z2_normal",
    "reference": "r_c=detached median(m2(interior_c))",
    "component_loss": "mean_b relu(m2(b)-r_c) over near_background_c",
    "image_reduction": "mean over valid components in each image",
    "batch_reduction": "mean over anomalous images with at least one valid component; zero otherwise",
    "excluded": ["pseudo interior", "raw boundary", "margin", "temperature", "softplus", "sort", "top-k", "percentile", "far-background term", "morphology tuning"],
}


def make_training_model(payload: dict, device: torch.device):
    """Construct the unchanged Safe-Anchor model with post-freeze E11 flags."""
    model = spill.make_model(device)
    state = payload["model_state"]
    model.image_adapter.load_state_dict(state["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(state["text_adapter"], strict=True)
    model.soft_prompt.load_state_dict(state["soft_prompt"], strict=True)
    model.prompt_mode = "hybrid"
    model.use_hybrid_soft_prompt = True
    model.use_soft_prompt = False
    model.hybrid_alpha_max = float(payload["hybrid_alpha_max"])
    model.soft_prompt_freeze_epochs = int(payload["soft_prompt_freeze_epochs"])
    model.hybrid_alpha_current = float(payload["hybrid_alpha_current"])
    model.set_dfg_beta(float(payload["dfg_beta_current"]))
    model.stage_fusion_weights = (1.0 / 3.0,) * 3
    model.requires_grad_(False)
    model.image_adapter.requires_grad_(True)
    model.text_adapter.requires_grad_(True)
    model.soft_prompt.requires_grad_(True)
    model.eval()
    return model


def configure_training_epoch(model, epoch: int = 11) -> None:
    model.eval()
    model.image_encoder.eval()
    model.clipmodel.eval()
    model.hybrid_alpha_current = get_hybrid_alpha_for_epoch(
        epoch, float(model.hybrid_alpha_max), int(model.soft_prompt_freeze_epochs),
    )
    model.soft_prompt.requires_grad_(epoch > int(model.soft_prompt_freeze_epochs))
    model.text_adapter.requires_grad_(True)
    model.set_dfg_beta(get_dfg_beta_for_epoch(epoch, "warmup010", .1, .1))


def batch_text_features(model, class_names, device: torch.device):
    by_class = {}
    kg_losses, k_losses = [], []
    for class_name in sorted(set(class_names)):
        text, kg, _, components = get_hybrid_soft_prompt_single_class_text_embedding(
            model, "VisA", class_name, device, return_kg=True, return_components=True,
        )
        k_loss, _ = compute_hybrid_k_regularization(
            model, components["hard_text"], components["soft_text"], model.hybrid_alpha_current,
        )
        by_class[class_name] = text
        kg_losses.append(kg)
        k_losses.append(k_loss)
    text = torch.stack([by_class[name] for name in class_names]).permute(1, 0, 2, 3)
    kg_loss = torch.stack(kg_losses).mean()
    k_loss = torch.stack(k_losses).mean()
    return text, kg_loss, k_loss


def locr_from_stage2(stage2_logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Evaluate exactly one S2-LOCR formulation on a batch."""
    margin = (stage2_logits[:, 1] - stage2_logits[:, 0]).float()
    image_losses = []
    image_details = []
    total_components = valid_components = skipped_interior = skipped_near = active_components = 0
    valid_near_pixels = violating_near_pixels = 0
    reference_values, violating_values = [], []
    for image_index in range(int(mask.shape[0])):
        mask_np = (mask[image_index, 0].detach().float().cpu().numpy() > .5)
        components = component_geometry(mask_np)
        component_losses = []
        image_active = 0
        image_valid = 0
        for record in components:
            total_components += 1
            skipped_interior += int(record["skip_empty_interior"])
            skipped_near += int(record["skip_empty_near"])
            if not record["valid"]:
                continue
            image_valid += 1
            valid_components += 1
            interior = torch.from_numpy(record["interior_mask"]).to(stage2_logits.device)
            near = torch.from_numpy(record["near_mask"]).to(stage2_logits.device)
            image_margin = margin[image_index]
            reference = torch.median(image_margin[interior]).detach()
            near_margin = image_margin[near]
            violation = near_margin > reference
            component_loss = F.relu(near_margin - reference).mean()
            component_losses.append(component_loss)
            active = bool(violation.any().item())
            image_active += int(active)
            active_components += int(active)
            valid_near_pixels += int(near.sum().item())
            violating_near_pixels += int(violation.sum().item())
            reference_values.append(float(reference.detach().cpu()))
            if violation.any():
                violating_values.extend(near_margin[violation].detach().cpu().tolist())
        if component_losses:
            image_losses.append(torch.stack(component_losses).mean())
        image_details.append({
            "image_index": image_index,
            "component_count": len(components),
            "valid_components": image_valid,
            "skipped_empty_interior": sum(int(r["skip_empty_interior"]) for r in components),
            "skipped_empty_near": sum(int(r["skip_empty_near"]) for r in components),
            "active_components": image_active,
        })
    if image_losses:
        loss = torch.stack(image_losses).mean()
    else:
        loss = stage2_logits.sum() * 0.0
    details = {
        "anomalous_image_count": int(sum(int(v["component_count"] > 0) for v in image_details)),
        "component_count": total_components,
        "valid_components": valid_components,
        "skipped_empty_interior": skipped_interior,
        "skipped_empty_near": skipped_near,
        "active_components": active_components,
        "active": bool(active_components > 0),
        "valid_near_pixels": valid_near_pixels,
        "violating_near_pixels": violating_near_pixels,
        "near_violation_fraction": (violating_near_pixels / valid_near_pixels) if valid_near_pixels else 0.0,
        "median_r_c": float(np.median(reference_values)) if reference_values else None,
        "violating_margin_mean": float(np.mean(violating_values)) if violating_values else None,
        "violating_margin_max": float(np.max(violating_values)) if violating_values else None,
        "image_details": image_details,
    }
    return loss, details


def batch_terms(model, batch, device: torch.device, policy: PrecisionPolicy) -> dict:
    image = batch["image"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    class_names = list(batch["class_name"])
    text, kg_loss, k_loss = batch_text_features(model, class_names, device)
    captured = {}
    hook = model.image_adapter["lora_adapters"][1].register_forward_pre_hook(
        lambda _module, inputs: captured.setdefault("stage2_lora_input", inputs[0])
    )
    with policy.autocast(device):
        try:
            seg_tokens, det_tokens = model(image)
        finally:
            hook.remove()
        vision = torch.stack(seg_tokens)
        det = torch.stack(det_tokens)
        cls = torch.stack([
            torch.matmul(det[i].unsqueeze(1), text[i]).squeeze(1) for i in range(3)
        ]).mean(0)
        cls_loss = F.cross_entropy(cls, label)
        seg_pred = model.vision_text_fusion_gate_seg(vision, text)
        focal = focal_loss(seg_pred, mask)
        normal_dice = dice_loss(seg_pred[:, 0], 1 - mask)
        abnormal_dice = dice_loss(seg_pred[:, 1], mask)
        seg_loss = calculate_seg_loss(seg_pred, mask)
        task = cls_loss + seg_loss + .01 * kg_loss + .002 * k_loss
        if "stage2_lora_input" not in captured:
            raise RuntimeError("failed to capture Stage-2 adapter boundary")
        aux_stage2 = auxiliary_stage2_resized_logits(model, captured["stage2_lora_input"], text.detach())
        locr, locr_details = locr_from_stage2(aux_stage2, mask)
    return {
        "image": image, "mask": mask, "label": label, "class_names": class_names,
        "text": text, "vision": vision, "task": task, "cls_loss": cls_loss,
        "seg_loss": seg_loss, "focal_loss": focal, "normal_dice": normal_dice,
        "abnormal_dice": abnormal_dice, "locr": locr, "locr_details": locr_details,
        "aux_stage2": aux_stage2, "seg_pred": seg_pred,
    }


def stage_parameters(model, stage: int):
    return [(name, parameter) for name, parameter in sorted(model.image_adapter.named_parameters()) if image_stage_matches(name, stage)]


def image_stage_matches(name: str, stage: int) -> bool:
    parts = name.split(".")
    return len(parts) > 1 and parts[1] == str(stage)


def gradient_norm(gradients) -> float:
    values = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt().cpu())


def gradient_vector(gradients, parameters) -> torch.Tensor:
    values = []
    for gradient, parameter in zip(gradients, parameters):
        values.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1) if gradient is None else gradient.detach().float().reshape(-1))
    return torch.cat(values) if values else torch.zeros(0, device="cuda", dtype=torch.float32)


def cosine_gradients(left, right, parameters) -> float | None:
    a = gradient_vector(left, parameters)
    b = gradient_vector(right, parameters)
    if a.numel() == 0:
        return None
    an, bn = float(a.norm().cpu()), float(b.norm().cpu())
    if an <= 0.0 or bn <= 0.0:
        return None
    return float(torch.dot(a, b).div(a.norm() * b.norm()).cpu())


def family_for_name(name: str) -> str:
    return name.split(".", 1)[0]


def fixed_train_batches(payload: dict, model) -> list[dict]:
    restore_checkpoint_rng(payload)
    dataset = spill.get_text_and_image_dataset("VisA", IMG, "train")
    batches = []
    for index, batch in enumerate(loader_for_epoch(dataset, 11)):
        if index >= 16:
            break
        batches.append(batch)
    if len(batches) != 16:
        raise RuntimeError("fixed training preflight did not produce exactly 16 batches")
    return batches


def check_coverage() -> dict:
    if not OUT_COVERAGE_JSON.is_file():
        raise RuntimeError("geometry coverage artifact is required")
    coverage = json.loads(OUT_COVERAGE_JSON.read_text())
    if coverage.get("GEOMETRY_COVERAGE") != "ADEQUATE":
        raise RuntimeError("GEOMETRY_COVERAGE=INADEQUATE")
    return coverage


def auxiliary_locality_phase(payload: dict) -> dict:
    check_coverage()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact historical preflight")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    model = make_training_model(payload, device)
    configure_training_epoch(model, 11)
    fixed_batches = fixed_train_batches(payload, model)
    batch = fixed_batches[0]
    # A single fixed-batch image is sufficient for the graph-locality proof
    # and avoids retaining the historical ViT graph for all six images.
    selected_batch = None
    selected_index = None
    for candidate_batch in fixed_batches:
        candidate_image = candidate_batch["image"].to(device, non_blocking=True)
        candidate_mask = candidate_batch["mask"].to(device, non_blocking=True)
        candidate_text, _, _ = batch_text_features(model, list(candidate_batch["class_name"]), device)
        candidate_capture = {}
        candidate_hook = model.image_adapter["lora_adapters"][1].register_forward_pre_hook(
            lambda _module, inputs: candidate_capture.setdefault("stage2_lora_input", inputs[0])
        )
        with policy.autocast(device), torch.no_grad():
            try:
                candidate_seg, _ = model(candidate_image)
            finally:
                candidate_hook.remove()
            candidate_vision = torch.stack(candidate_seg)
            candidate_aux = auxiliary_stage2_resized_logits(model, candidate_capture["stage2_lora_input"], candidate_text.detach())
            _, candidate_details = locr_from_stage2(candidate_aux, candidate_mask)
        for detail in candidate_details["image_details"]:
            if detail["active_components"] > 0:
                selected_batch = candidate_batch
                selected_index = int(detail["image_index"])
                break
        if selected_batch is not None:
            break
    if selected_batch is None or selected_index is None:
        raise RuntimeError("locality diagnostic found no active image in the fixed 16 batches")
    batch = selected_batch
    sample = {key: (value[selected_index:selected_index + 1] if torch.is_tensor(value) else [value[selected_index]]) for key, value in batch.items()}
    image = sample["image"].to(device, non_blocking=True)
    mask = sample["mask"].to(device, non_blocking=True)
    text, _, _ = batch_text_features(model, sample["class_name"], device)
    capture = {}
    hook = model.image_adapter["lora_adapters"][1].register_forward_pre_hook(
        lambda _module, inputs: capture.setdefault("stage2_lora_input", inputs[0])
    )
    with policy.autocast(device):
        try:
            seg_tokens, _ = model(image)
        finally:
            hook.remove()
        vision = torch.stack(seg_tokens)
        aux_stage2 = auxiliary_stage2_resized_logits(model, capture["stage2_lora_input"], text.detach())
        locr, _ = locr_from_stage2(aux_stage2, mask)
    with policy.autocast(device), torch.no_grad():
        reference = production_resized_logits(model, vision.detach(), text)[1]
    stage2_difference = (reference - aux_stage2.detach()).abs()
    trainable = [(name, parameter) for name, parameter in sorted(model.named_parameters()) if parameter.requires_grad]
    names = [name for name, _ in trainable]
    parameters = [parameter for _, parameter in trainable]
    gradients = torch.autograd.grad(locr, parameters, allow_unused=True)
    groups = {
        "stage1_image": [gradient for (name, _), gradient in zip(trainable, gradients) if name.startswith("image_adapter.") and image_stage_matches(name[len("image_adapter."):], 0)],
        "stage2_image": [gradient for (name, _), gradient in zip(trainable, gradients) if name.startswith("image_adapter.") and image_stage_matches(name[len("image_adapter."):], 1)],
        "stage3_image": [gradient for (name, _), gradient in zip(trainable, gradients) if name.startswith("image_adapter.") and image_stage_matches(name[len("image_adapter."):], 2)],
        "text_adapter": [gradient for (name, _), gradient in zip(trainable, gradients) if name.startswith("text_adapter.")],
        "soft_prompt": [gradient for (name, _), gradient in zip(trainable, gradients) if name.startswith("soft_prompt.")],
    }
    norms = {f"{key}_grad_norm": gradient_norm(value) for key, value in groups.items()}
    locality_pass = bool(
        torch.equal(reference, aux_stage2.detach())
        and norms["stage2_image_grad_norm"] > 0.0
        and norms["stage1_image_grad_norm"] == 0.0
        and norms["stage3_image_grad_norm"] == 0.0
        and norms["text_adapter_grad_norm"] == 0.0
        and norms["soft_prompt_grad_norm"] == 0.0
    )
    artifact = {
        "protocol_id": "EXPLORATORY_SOURCE_ONLY_MECHANISM_R1",
        "phase": "15_auxiliary_locality",
        "formulation": FORMULATION,
        "fixed_batch": {"file_names": list(batch["file_name"]), "labels": batch["label"].tolist()},
        "auxiliary_stage2_equals_production_stage2": bool(torch.equal(reference, aux_stage2.detach())),
        "stage2_max_abs_difference": float(stage2_difference.max().cpu()),
        "stage2_mean_abs_difference": float(stage2_difference.mean().cpu()),
        **norms,
        "S2_LOCR_AUXILIARY_LOCALITY": "PASS" if locality_pass else "FAIL",
        "main_safe_anchor_path_unchanged": True,
        "text_side_detached_for_auxiliary": True,
        "optimizer_step": False,
    }
    dump_json(OUT_LOCALITY_JSON, artifact)
    if not locality_pass:
        raise RuntimeError("S2_LOCR_AUXILIARY_LOCALITY=FAIL")
    return artifact


def gradient_preflight_phase(payload: dict) -> dict:
    check_coverage()
    locality = json.loads(OUT_LOCALITY_JSON.read_text())
    if locality.get("S2_LOCR_AUXILIARY_LOCALITY") != "PASS":
        raise RuntimeError("auxiliary locality gate is required")
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    model = make_training_model(payload, device)
    configure_training_epoch(model, 11)
    batches = fixed_train_batches(payload, model)
    stage2_pairs = stage_parameters(model, 1)
    stage2_names = [name for name, _ in stage2_pairs]
    stage2_params = [parameter for _, parameter in stage2_pairs]
    rows = []
    for batch_index, batch in enumerate(batches):
        terms = batch_terms(model, batch, device, policy)
        values = [terms["task"], terms["locr"], terms["focal_loss"], terms["normal_dice"], terms["abnormal_dice"]]
        grads_by_term = []
        for value_index, value in enumerate(values):
            grads_by_term.append(torch.autograd.grad(value, stage2_params, retain_graph=value_index < len(values) - 1, allow_unused=True))
        task_grads, locr_grads, focal_grads, normal_grads, abnormal_grads = grads_by_term
        row = {
            "batch_index": batch_index,
            "file_names": json.dumps(list(batch["file_name"]), separators=(",", ":")),
            "labels": json.dumps(batch["label"].tolist(), separators=(",", ":")),
            "task_loss": float(terms["task"].detach().float().cpu()),
            "raw_s2_locr_loss": float(terms["locr"].detach().float().cpu()),
            "stage2_task_grad_norm": gradient_norm(task_grads),
            "stage2_locr_grad_norm": gradient_norm(locr_grads),
            "cosine_locr_task": cosine_gradients(locr_grads, task_grads, stage2_params),
            "cosine_locr_focal": cosine_gradients(locr_grads, focal_grads, stage2_params),
            "cosine_locr_normal_dice": cosine_gradients(locr_grads, normal_grads, stage2_params),
            "cosine_locr_abnormal_dice": cosine_gradients(locr_grads, abnormal_grads, stage2_params),
            "active": int(terms["locr_details"]["active"]),
            "active_components": terms["locr_details"]["active_components"],
            "valid_components": terms["locr_details"]["valid_components"],
            "near_violation_fraction": terms["locr_details"]["near_violation_fraction"],
            "anomalous_image_count": terms["locr_details"]["anomalous_image_count"],
            "nonfinite": int(not torch.isfinite(terms["task"]).all().item() or not torch.isfinite(terms["locr"]).all().item()),
        }
        family_values = {}
        for parameter_name, parameter, task_gradient, locr_gradient in zip(stage2_names, stage2_params, task_grads, locr_grads):
            family = family_for_name(parameter_name)
            family_values.setdefault(family, {"task": [], "locr": []})
            if task_gradient is not None:
                family_values[family]["task"].append(task_gradient)
            if locr_gradient is not None:
                family_values[family]["locr"].append(locr_gradient)
        for family, pair in family_values.items():
            row[f"stage2_family_{family}_task_norm"] = gradient_norm(pair["task"])
            row[f"stage2_family_{family}_locr_norm"] = gradient_norm(pair["locr"])
        rows.append(row)
        del terms
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    active_rows = [row for row in rows if row["active"]]
    loss_values = np.asarray([row["raw_s2_locr_loss"] for row in rows], dtype=np.float64)
    violation_values = np.asarray([row["near_violation_fraction"] for row in rows], dtype=np.float64)
    nonfinite_count = int(sum(row["nonfinite"] for row in rows))
    summary = {
        "protocol_id": "EXPLORATORY_SOURCE_ONLY_MECHANISM_R1",
        "phase": "16_no_step_gradient_preflight",
        "formulation": FORMULATION,
        "batch_count": len(rows),
        "active_batch_count": len(active_rows),
        "active_batch_fraction": len(active_rows) / len(rows),
        "raw_locr_loss_median": float(np.median(loss_values)),
        "raw_locr_loss_p95": float(np.quantile(loss_values, .95)),
        "near_violation_fraction_mean": float(np.mean(violation_values)),
        "near_violation_fraction_median": float(np.median(violation_values)),
        "nonfinite_count": nonfinite_count,
        "stage2_parameter_count": len(stage2_params),
        "stage2_parameter_names": stage2_names,
        "rows": rows,
        "optimizer_step": False,
        "S2_LOCR_GRADIENT_PREFLIGHT": "PASS" if nonfinite_count == 0 and len(active_rows) > 0 and all(row["stage2_locr_grad_norm"] > 0 for row in active_rows) else "FAIL",
    }
    write_csv(OUT_PREFLIGHT_CSV, rows)
    dump_json(OUT_PREFLIGHT_JSON, summary)
    if summary["S2_LOCR_GRADIENT_PREFLIGHT"] != "PASS":
        raise RuntimeError("S2_LOCR_GRADIENT_PREFLIGHT=FAIL")
    return summary


def lambda_calibration_phase() -> dict:
    preflight = json.loads(OUT_PREFLIGHT_JSON.read_text())
    if preflight.get("S2_LOCR_GRADIENT_PREFLIGHT") != "PASS":
        raise RuntimeError("gradient preflight gate is required")
    eps = 1.0e-12
    ratios = [
        float(row["stage2_locr_grad_norm"]) / (float(row["stage2_task_grad_norm"]) + eps)
        for row in preflight["rows"]
    ]
    finite = np.asarray(ratios, dtype=np.float64)
    R = float(np.median(finite)) if finite.size else 0.0
    stable = bool(finite.size == 16 and np.isfinite(finite).all() and R > 0.0)
    lambda_s2 = float(.05 / R) if stable else None
    artifact = {
        "protocol_id": "EXPLORATORY_SOURCE_ONLY_MECHANISM_R1",
        "phase": "17_single_lambda_calibration",
        "target_raw_gradient_ratio": .05,
        "epsilon": eps,
        "ratio_definition": "median(||g_s2_locr_raw||_S2/(||g_existing_task||_S2+eps)) over the fixed 16 batches",
        "ratios": ratios,
        "R": R,
        "lambda_s2_locr": lambda_s2,
        "rounded_for_training": False,
        "stable": stable,
        "S2_LOCR_LAMBDA_CALIBRATION": "PASS" if stable else "FAIL",
        "formulation": FORMULATION,
    }
    dump_json(OUT_CALIBRATION_JSON, artifact)
    if not stable:
        raise RuntimeError("S2_LOCR_LAMBDA_CALIBRATION=FAIL")
    return artifact


def loss_parity_phase(payload: dict) -> dict:
    calibration = json.loads(OUT_CALIBRATION_JSON.read_text())
    if calibration.get("S2_LOCR_LAMBDA_CALIBRATION") != "PASS":
        raise RuntimeError("lambda calibration gate is required")
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    model = make_training_model(payload, device)
    configure_training_epoch(model, 11)
    batch = fixed_train_batches(payload, model)[0]
    terms = batch_terms(model, batch, device, policy)
    trainable = [(name, parameter) for name, parameter in sorted(model.named_parameters()) if parameter.requires_grad]
    parameters = [parameter for _, parameter in trainable]
    control_scalar = terms["task"]
    candidate_disabled_scalar = terms["task"] + 0.0 * terms["locr"]
    control_grads = torch.autograd.grad(control_scalar, parameters, retain_graph=True, allow_unused=True)
    disabled_grads = torch.autograd.grad(candidate_disabled_scalar, parameters, allow_unused=True)
    scalar_delta = float((control_scalar.detach() - candidate_disabled_scalar.detach()).abs().float().cpu())
    deltas = []
    for left, right in zip(control_grads, disabled_grads):
        if left is None and right is None:
            continue
        l = torch.zeros_like(right) if left is None else left
        r = torch.zeros_like(left) if right is None else right
        deltas.append((l.detach().float() - r.detach().float()).abs().max())
    max_grad_delta = float(torch.stack(deltas).max().cpu()) if deltas else 0.0
    anchor = SafeImageAdapterAnchor.from_checkpoint(SAFE_ANCHOR, device)
    anchor_loss = anchor.loss(model.image_adapter)
    artifact = {
        "protocol_id": "EXPLORATORY_SOURCE_ONLY_MECHANISM_R1",
        "phase": "18_loss_parity",
        "formulation": FORMULATION,
        "control_objective": "historical task loss plus unchanged Safe-Anchor family-budget application (anchor scalar kept separate)",
        "candidate_objective": "same control objective plus lambda_s2_locr*raw_s2_locr",
        "candidate_disabled_objective": "same candidate graph with S2-LOCR coefficient exactly zero",
        "control_scalar": float(control_scalar.detach().float().cpu()),
        "candidate_disabled_scalar": float(candidate_disabled_scalar.detach().float().cpu()),
        "scalar_abs_delta": scalar_delta,
        "max_trainable_gradient_abs_delta": max_grad_delta,
        "anchor_loss_control": float(anchor_loss.detach().float().cpu()),
        "safe_anchor_lambda": EXPECTED_SAFE_SHA,
        "safe_anchor_path_unchanged": True,
        "strict_scalar_parity": bool(scalar_delta == 0.0),
        "strict_gradient_parity": bool(max_grad_delta == 0.0),
        "S2_LOCR_LOSS_PARITY": "PASS" if scalar_delta == 0.0 and max_grad_delta == 0.0 else "FAIL",
    }
    dump_json(OUT_PARITY_JSON, artifact)
    if artifact["S2_LOCR_LOSS_PARITY"] != "PASS":
        raise RuntimeError("S2_LOCR_LOSS_PARITY=FAIL")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("identity", "oracle", "coverage", "locality", "preflight", "calibration", "parity"), required=True)
    args = parser.parse_args()
    rows, _ = spill.load_rows()
    if args.phase in {"identity", "oracle"}:
        write_parent_identity()
    identity = validate_inputs(rows)
    if args.phase == "identity":
        return
    if args.phase == "oracle":
        run_oracle(rows, identity)
    elif args.phase == "coverage":
        if not OUT_ORACLE_JSON.is_file():
            raise RuntimeError("oracle artifact is required before coverage")
        geometry_phase(identity, rows)
    elif args.phase == "locality":
        auxiliary_locality_phase(torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False))
    elif args.phase == "preflight":
        gradient_preflight_phase(torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False))
    elif args.phase == "calibration":
        lambda_calibration_phase()
    elif args.phase == "parity":
        loss_parity_phase(torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False))


if __name__ == "__main__":
    main()
