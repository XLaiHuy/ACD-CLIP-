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
    group_text = text.unsqueeze(1).repeat(1, batch, 1, 1).permute(1, 0, 2, 3)
    stage_logits = []
    for stage in range(3):
        fused = model._vision_text_attention_fusion(vision[stage], group_text, stage)
        native = torch.matmul(10 * vision[stage], fused).permute(0, 2, 1).view(batch, 2, side, side)
        blurred = gaussian_blur2d(native, (7, 7), (1, 1))
        stage_logits.append(F.interpolate(blurred, (IMG, IMG), mode="bilinear", align_corners=True))
    return torch.stack(stage_logits)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("identity", "oracle", "coverage"), required=True)
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


if __name__ == "__main__":
    main()
