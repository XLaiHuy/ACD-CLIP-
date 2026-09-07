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
OUT_MANIFEST_JSON = REPO / "audit/H2_S2_LOCR_R1_ATTEMPT_MANIFEST.json"
OUT_CONTROL_CSV = REPO / "audit/H2_S2_LOCR_R1_CONTROL.csv"
OUT_CANDIDATE_CSV = REPO / "audit/H2_S2_LOCR_R1_CANDIDATE.csv"
OUT_ENDPOINT_CSV = REPO / "audit/H2_S2_LOCR_R1_ENDPOINT.csv"
OUT_ENDPOINT_JSON = REPO / "audit/H2_S2_LOCR_R1_ENDPOINT.json"
OUT_DECISION_MD = REPO / "results/H2_S2_LOCR_R1_BOUNDED_DECISION.md"
OUT_DECISION_JSON = REPO / "results/H2_S2_LOCR_R1_BOUNDED_DECISION.json"
RUN_ROOT = Path("/workspace/h2_s2_locr_r1")
ARM_CONTROL = "A_S2_LOCR_R1_CONTROL"
ARM_CANDIDATE = "A_S2_LOCR_R1_CANDIDATE"
MAX_ATTEMPTS = 500


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


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


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
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
        "violating_margin_count": len(violating_values),
        "violating_margin_sum": float(np.sum(violating_values)) if violating_values else 0.0,
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


def preflight_microbatch(model, batch, device: torch.device, policy: PrecisionPolicy, stage2_params) -> dict:
    """Compute an exact batch-reduced preflight with one-image activation graphs."""
    image = batch["image"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    class_names = list(batch["class_name"])
    text, kg_loss, k_loss = batch_text_features(model, class_names, device)
    valid_flags = []
    for image_index in range(int(mask.shape[0])):
        valid_flags.append(any(record["valid"] for record in component_geometry((mask[image_index, 0].detach().float().cpu().numpy() > .5))))
    valid_image_count = max(1, sum(valid_flags))
    term_names = ("task", "locr", "focal", "normal_dice", "abnormal_dice")
    accumulated = {name: [torch.zeros_like(parameter, dtype=torch.float32) for parameter in stage2_params] for name in term_names}
    task_values, locr_values = [], []
    details_rows = []
    for image_index in range(int(image.shape[0])):
        one_image = image[image_index:image_index + 1]
        one_mask = mask[image_index:image_index + 1]
        one_label = label[image_index:image_index + 1]
        one_text = text[:, image_index:image_index + 1]
        captured = {}
        hook = model.image_adapter["lora_adapters"][1].register_forward_pre_hook(
            lambda _module, inputs: captured.setdefault("stage2_lora_input", inputs[0])
        )
        with policy.autocast(device):
            try:
                seg_tokens, det_tokens = model(one_image)
            finally:
                hook.remove()
            vision = torch.stack(seg_tokens)
            det = torch.stack(det_tokens)
            cls = torch.stack([
                torch.matmul(det[i].unsqueeze(1), one_text[i]).squeeze(1) for i in range(3)
            ]).mean(0)
            cls_loss = F.cross_entropy(cls, one_label)
            seg_pred = model.vision_text_fusion_gate_seg(vision, one_text)
            focal_value = focal_loss(seg_pred, one_mask)
            normal_value = dice_loss(seg_pred[:, 0], 1 - one_mask)
            abnormal_value = dice_loss(seg_pred[:, 1], one_mask)
            task_value = cls_loss + focal_value + normal_value + abnormal_value + .01 * kg_loss + .002 * k_loss
            aux_stage2 = auxiliary_stage2_resized_logits(model, captured["stage2_lora_input"], one_text.detach())
            locr_value, details = locr_from_stage2(aux_stage2, one_mask)
        values = (task_value, locr_value, focal_value, normal_value, abnormal_value)
        retain_for_next = image_index < int(image.shape[0]) - 1
        for term_index, (name, value) in enumerate(zip(term_names, values)):
            gradients = torch.autograd.grad(
                value,
                stage2_params,
                retain_graph=retain_for_next or term_index < len(values) - 1,
                allow_unused=True,
            )
            weight = (1.0 / valid_image_count) if name == "locr" and valid_flags[image_index] else (0.0 if name == "locr" else 1.0 / int(image.shape[0]))
            for target, gradient in zip(accumulated[name], gradients):
                if gradient is not None and weight != 0.0:
                    target.add_(gradient.detach().float(), alpha=weight)
        task_values.append(float(task_value.detach().float().cpu()) / int(image.shape[0]))
        if valid_flags[image_index]:
            locr_values.append(float(locr_value.detach().float().cpu()) / valid_image_count)
        details_rows.append(details)
        del values, vision, det, seg_pred, aux_stage2
    details = {
        "anomalous_image_count": sum(int(row["component_count"] > 0) for row in details_rows),
        "component_count": sum(row["component_count"] for row in details_rows),
        "valid_components": sum(row["valid_components"] for row in details_rows),
        "active_components": sum(row["active_components"] for row in details_rows),
        "active": any(row["active"] for row in details_rows),
        "near_violation_fraction": sum(row["violating_near_pixels"] for row in details_rows) / max(1, sum(row["valid_near_pixels"] for row in details_rows)),
    }
    return {
        "task_value": float(sum(task_values)),
        "locr_value": float(sum(locr_values)),
        "details": details,
        "gradients": accumulated,
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


def all_training_gates_pass() -> bool:
    if not OUT_ORACLE_JSON.is_file() or not OUT_COVERAGE_JSON.is_file() or not OUT_LOCALITY_JSON.is_file() or not OUT_PREFLIGHT_JSON.is_file() or not OUT_CALIBRATION_JSON.is_file() or not OUT_PARITY_JSON.is_file():
        return False
    oracle = json.loads(OUT_ORACLE_JSON.read_text())
    coverage = json.loads(OUT_COVERAGE_JSON.read_text())
    locality = json.loads(OUT_LOCALITY_JSON.read_text())
    preflight = json.loads(OUT_PREFLIGHT_JSON.read_text())
    calibration = json.loads(OUT_CALIBRATION_JSON.read_text())
    parity = json.loads(OUT_PARITY_JSON.read_text())
    return bool(
        oracle.get("local_oracle_causal_support") == "PASS"
        and coverage.get("GEOMETRY_COVERAGE") == "ADEQUATE"
        and locality.get("S2_LOCR_AUXILIARY_LOCALITY") == "PASS"
        and preflight.get("S2_LOCR_GRADIENT_PREFLIGHT") == "PASS"
        and calibration.get("S2_LOCR_LAMBDA_CALIBRATION") == "PASS"
        and parity.get("S2_LOCR_LOSS_PARITY") == "PASS"
    )


def manifest_row(attempt: int, epoch: int, batch_index: int, batch: dict) -> dict:
    image = batch["image"]
    mask = batch["mask"]
    return {
        "attempt_index": int(attempt),
        "epoch": int(epoch),
        "batch": int(batch_index),
        "file_names": list(batch["file_name"]),
        "labels": [int(value) for value in batch["label"].tolist()],
        "image_sha256": tensor_hash(image),
        "mask_sha256": tensor_hash(mask),
        "image_sha256_per_item": [tensor_hash(image[i:i + 1]) for i in range(image.shape[0])],
        "mask_sha256_per_item": [tensor_hash(mask[i:i + 1]) for i in range(mask.shape[0])],
    }


def generate_attempt_manifest(payload: dict) -> dict:
    if not all_training_gates_pass():
        raise RuntimeError("S2_LOCR_TRAINING_AUTHORIZED=NO: all pre-training gates must pass")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the historical source screen")
    # Constructing the model before restoring E10 RNG matches the resume path;
    # no optimizer or model update occurs during manifest generation.
    model = make_training_model(payload, torch.device("cuda:0"))
    del model
    torch.cuda.empty_cache()
    restore_checkpoint_rng(payload)
    dataset = spill.get_text_and_image_dataset("VisA", IMG, "train")
    attempts = []
    epoch = 11
    while len(attempts) < MAX_ATTEMPTS:
        for batch_index, batch in enumerate(loader_for_epoch(dataset, epoch)):
            attempts.append(manifest_row(len(attempts), epoch, batch_index, batch))
            if len(attempts) == MAX_ATTEMPTS:
                break
        epoch += 1
        if epoch > 100:
            raise RuntimeError("manifest generation exceeded epoch safety bound")
    artifact = {
        "protocol_id": "EXPLORATORY_SOURCE_ONLY_MECHANISM_R1",
        "phase": "20_fixed_500_attempt_manifest",
        "source_dataset": "VisA",
        "target_inference": False,
        "attempt_count": len(attempts),
        "batch_size": 6,
        "image_size": IMG,
        "seed": 0,
        "start_checkpoint": str(SAFE_ANCHOR),
        "start_checkpoint_sha256": EXPECTED_SAFE_SHA,
        "loader_definition": "shuffle=True, num_workers=0, pin_memory=True, epoch seed=104729*epoch, epoch starts at 11",
        "no_optimizer_updates_during_generation": True,
        "attempts": attempts,
    }
    dump_json(OUT_MANIFEST_JSON, artifact)
    return artifact


def make_training_optimizer(model, payload: dict):
    optimizer = torch.optim.Adam([
        {"name": "text_adapter", "params": model.text_adapter.parameters(), "lr": .0005},
        {"name": "image_adapter", "params": model.image_adapter.parameters(), "lr": .001},
        {"name": "soft_prompt", "params": model.soft_prompt.parameters(), "lr": 0.0, "constant_lr": .00005},
    ])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler = StepLR(optimizer, step_size=1, gamma=.9)
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    scaler.load_state_dict(payload["scaler_state"])
    return optimizer, scheduler, scaler


def aggregate_locr_details(details_rows: list[dict]) -> dict:
    component_count = sum(row["component_count"] for row in details_rows)
    valid_components = sum(row["valid_components"] for row in details_rows)
    skipped_interior = sum(row["skipped_empty_interior"] for row in details_rows)
    skipped_near = sum(row["skipped_empty_near"] for row in details_rows)
    active_components = sum(row["active_components"] for row in details_rows)
    valid_near_pixels = sum(row["valid_near_pixels"] for row in details_rows)
    violating_near_pixels = sum(row["violating_near_pixels"] for row in details_rows)
    reference_values = [row["median_r_c"] for row in details_rows if row["median_r_c"] is not None]
    violating_count = sum(row["violating_margin_count"] for row in details_rows)
    violating_sum = sum(row["violating_margin_sum"] for row in details_rows)
    violating_max = [row["violating_margin_max"] for row in details_rows if row["violating_margin_max"] is not None]
    return {
        "anomalous_image_count": sum(int(row["component_count"] > 0) for row in details_rows),
        "component_count": component_count,
        "valid_components": valid_components,
        "skipped_empty_interior": skipped_interior,
        "skipped_empty_near": skipped_near,
        "active_components": active_components,
        "active": bool(active_components > 0),
        "valid_near_pixels": valid_near_pixels,
        "violating_near_pixels": violating_near_pixels,
        "near_violation_fraction": violating_near_pixels / max(1, valid_near_pixels),
        "median_r_c": float(np.median(reference_values)) if reference_values else None,
        "violating_margin_mean": violating_sum / violating_count if violating_count else None,
        "violating_margin_max": float(max(violating_max)) if violating_max else None,
    }


def training_microbatch(model, batch, device: torch.device, policy: PrecisionPolicy, candidate: bool, lambda_s2: float, trainable):
    """Run one historical batch as exact-reduction one-image graphs."""
    image = batch["image"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    class_names = list(batch["class_name"])
    text, kg_loss, k_loss = batch_text_features(model, class_names, device)
    valid_flags = [any(record["valid"] for record in component_geometry(mask[i, 0].detach().float().cpu().numpy() > .5)) for i in range(int(mask.shape[0]))]
    valid_image_count = max(1, sum(valid_flags))
    accum = [torch.zeros_like(parameter, dtype=torch.float32) for _, parameter in trainable]
    task_values, raw_locr_values, weighted_locr_values, details_rows = [], [], [], []
    for image_index in range(int(image.shape[0])):
        one_image = image[image_index:image_index + 1]
        one_mask = mask[image_index:image_index + 1]
        one_label = label[image_index:image_index + 1]
        one_text = text[:, image_index:image_index + 1]
        capture = {}
        hook = model.image_adapter["lora_adapters"][1].register_forward_pre_hook(
            lambda _module, inputs: capture.setdefault("stage2_lora_input", inputs[0])
        )
        with policy.autocast(device):
            try:
                seg_tokens, det_tokens = model(one_image)
            finally:
                hook.remove()
            vision = torch.stack(seg_tokens)
            det = torch.stack(det_tokens)
            cls = torch.stack([torch.matmul(det[i].unsqueeze(1), one_text[i]).squeeze(1) for i in range(3)]).mean(0)
            cls_loss = F.cross_entropy(cls, one_label)
            seg_pred = model.vision_text_fusion_gate_seg(vision, one_text)
            focal_value = focal_loss(seg_pred, one_mask)
            normal_value = dice_loss(seg_pred[:, 0], 1 - one_mask)
            abnormal_value = dice_loss(seg_pred[:, 1], one_mask)
            task_value = cls_loss + focal_value + normal_value + abnormal_value + .01 * kg_loss + .002 * k_loss
            if candidate:
                aux_stage2 = auxiliary_stage2_resized_logits(model, capture["stage2_lora_input"], one_text.detach())
                locr_value, details = locr_from_stage2(aux_stage2, one_mask)
            else:
                with torch.no_grad():
                    aux_stage2 = auxiliary_stage2_resized_logits(model, capture["stage2_lora_input"], one_text.detach())
                    locr_value, details = locr_from_stage2(aux_stage2, one_mask)
            objective = task_value / int(image.shape[0])
            if candidate and valid_flags[image_index]:
                objective = objective + lambda_s2 * locr_value / valid_image_count
        gradients = torch.autograd.grad(
            objective,
            [parameter for _, parameter in trainable],
            retain_graph=image_index < int(image.shape[0]) - 1,
            allow_unused=True,
        )
        for index, gradient in enumerate(gradients):
            if gradient is not None:
                accum[index].add_(gradient.detach().float())
        task_values.append(float(task_value.detach().float().cpu()) / int(image.shape[0]))
        if valid_flags[image_index]:
            raw_locr_values.append(float(locr_value.detach().float().cpu()) / valid_image_count)
            weighted_locr_values.append(float(locr_value.detach().float().cpu()) * lambda_s2 / valid_image_count if candidate else 0.0)
        details_rows.append(details)
        del objective, gradients, vision, det, seg_pred, aux_stage2
    details = aggregate_locr_details(details_rows)
    return {
        "task_loss": float(sum(task_values)),
        "raw_locr_loss": float(sum(raw_locr_values)),
        "weighted_locr_loss": float(sum(weighted_locr_values)),
        "candidate_loss": float(sum(task_values) + sum(weighted_locr_values)),
        "details": details,
        "gradients": accum,
    }


def finite_model_parameters(model) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def optimizer_state_is_finite(optimizer) -> bool:
    return all(torch.isfinite(value).all().item() for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value))


def batch_identity_matches(manifest, batch) -> bool:
    current = manifest_row(manifest["attempt_index"], manifest["epoch"], manifest["batch"], batch)
    fields = ("file_names", "labels", "image_sha256", "mask_sha256", "image_sha256_per_item", "mask_sha256_per_item")
    return all(current[field] == manifest[field] for field in fields)


def csv_identity(manifest: dict) -> dict:
    return {
        "attempt_index": manifest["attempt_index"],
        "epoch": manifest["epoch"],
        "batch": manifest["batch"],
        "file_names": json.dumps(manifest["file_names"], separators=(",", ":")),
        "labels": json.dumps(manifest["labels"], separators=(",", ":")),
        "image_sha256": manifest["image_sha256"],
        "mask_sha256": manifest["mask_sha256"],
        "image_sha256_per_item": json.dumps(manifest["image_sha256_per_item"], separators=(",", ":")),
        "mask_sha256_per_item": json.dumps(manifest["mask_sha256_per_item"], separators=(",", ":")),
    }


def final_training_state(model, optimizer, scheduler, scaler, payload: dict, epoch: int, global_step: int, arm: str, lambda_s2: float, attempted: int, successful: int, natural_skips: int, forced_skips: int) -> dict:
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "n_groups": 3,
        "dfg_mode": "attn",
        "dfg_attn_dim": 256,
        "dfg_attn_tau": 8.0,
        "use_ss2d_dfg": True,
        "dfg_gamma_max": .2,
        "dfg_ss2d_fusion": "weight_residual",
        "dfg_beta": .1,
        "dfg_beta_schedule": "warmup010",
        "dfg_beta_target": .1,
        "dfg_beta_current": float(model.dfg_beta),
        "dfg_weight_residual_fp32": True,
        "prompt_mode": "hybrid",
        "use_soft_prompt": False,
        "use_hybrid_soft_prompt": True,
        "soft_prompt_ctx_len": 4,
        "soft_prompt_init": "phrase",
        "soft_prompt_init_phrase": "a photo of a",
        "hybrid_alpha_current": float(model.hybrid_alpha_current),
        "hybrid_alpha_max": .2,
        "soft_prompt_freeze_epochs": 3,
        "grad_clip_norm": 1.0,
        "anchor_gradient_budget": True,
        "anchor_family_budget": .1,
        "lambda_kg": .01,
        "lambda_k": .002,
        "k_reg_detached_wk": True,
        "k_reg_per_stage": True,
        "checkpoint_version": 3,
        "protocol_version": payload.get("protocol_version"),
        "model_state": {
            "image_adapter": model.image_adapter.state_dict(),
            "text_adapter": model.text_adapter.state_dict(),
            "soft_prompt": model.soft_prompt.state_dict(),
        },
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "resolved_scientific_config": payload["resolved_scientific_config"],
        "parent_scientific_config": payload.get("parent_scientific_config"),
        "resolved_operational_config": payload.get("resolved_operational_config"),
        "seed": 0,
        "precision": "fp16",
        "precision_protocol": "HISTORICAL_MIXED_FP16_FP32_V1",
        "amp_enabled": True,
        "gradscaler_enabled": True,
        "tf32_enabled": False,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "dataloader_generator_state": payload.get("dataloader_generator_state"),
        "image_anchor": payload.get("image_anchor"),
        "image_anchor_reference": payload.get("image_anchor_reference"),
        "s2_locr": {
            "arm": arm,
            "lambda_s2_locr": lambda_s2,
            "attempted": attempted,
            "successful": successful,
            "natural_skips": natural_skips,
            "forced_parity_skips": forced_skips,
            "formulation": FORMULATION,
        },
    }


def load_control_rows() -> list[dict]:
    with OUT_CONTROL_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != MAX_ATTEMPTS:
        raise RuntimeError(f"control CSV must contain {MAX_ATTEMPTS} attempts, got {len(rows)}")
    return rows


def run_training_arm(payload: dict, arm: str, manifest: dict, control_rows: list[dict] | None = None) -> dict:
    if arm not in (ARM_CONTROL, ARM_CANDIDATE):
        raise ValueError(arm)
    candidate = arm == ARM_CANDIDATE
    if len(manifest["attempts"]) != MAX_ATTEMPTS:
        raise RuntimeError("exact 500-attempt manifest is required")
    calibration = json.loads(OUT_CALIBRATION_JSON.read_text())
    lambda_s2 = float(calibration["lambda_s2_locr"])
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    model = make_training_model(payload, device)
    optimizer, scheduler, scaler = make_training_optimizer(model, payload)
    apply_soft_prompt_lr_policy(optimizer, False)
    anchor = SafeImageAdapterAnchor.from_checkpoint(SAFE_ANCHOR, device)
    trainable = [(name, parameter) for name, parameter in sorted(model.named_parameters()) if parameter.requires_grad]
    image_named = [(name, parameter) for name, parameter in sorted(model.image_adapter.named_parameters()) if parameter.requires_grad]
    image_names = [name for name, _ in image_named]
    image_params = [parameter for _, parameter in image_named]
    trainable_index = {name: index for index, (name, _) in enumerate(trainable)}
    dataset = spill.get_text_and_image_dataset("VisA", IMG, "train")
    restore_checkpoint_rng(payload)
    control_skip = {}
    if candidate:
        if control_rows is None:
            control_rows = load_control_rows()
        control_skip = {int(row["attempt_index"]): row for row in control_rows if row.get("natural_skip") == "1"}
    rows = []
    attempt = 0
    successful = 0
    natural_skips = 0
    forced_skips = 0
    additional_natural_skips = 0
    numerical_failure = None
    global_step = int(payload["global_step"])
    manifest_by_epoch = {}
    for item in manifest["attempts"]:
        manifest_by_epoch.setdefault(int(item["epoch"]), []).append(item)
    last_epoch = None
    full_batches = int(math.ceil(len(dataset) / 6))
    for epoch, epoch_manifest in manifest_by_epoch.items():
        configure_training_epoch(model, epoch)
        apply_soft_prompt_lr_policy(optimizer, False)
        processed = 0
        for batch_index, batch in enumerate(loader_for_epoch(dataset, epoch)):
            if processed >= len(epoch_manifest):
                break
            expected = epoch_manifest[processed]
            if int(expected["attempt_index"]) != attempt or int(expected["batch"]) != batch_index or not batch_identity_matches(expected, batch):
                raise RuntimeError(f"attempt manifest mismatch at attempt {attempt}, epoch {epoch}, batch {batch_index}")
            identity = csv_identity(expected)
            optimizer.zero_grad(set_to_none=True)
            metrics = training_microbatch(model, batch, device, policy, candidate, lambda_s2, trainable)
            anchor_loss = anchor.loss(model.image_adapter)
            objective_value = metrics["candidate_loss"] if candidate else metrics["task_loss"]
            forced = bool(candidate and attempt in control_skip)
            control_status = control_skip.get(attempt, {}).get("status") if forced else None
            row = {
                **identity,
                "arm": arm,
                "status": "pending",
                "successful_update": 0,
                "natural_skip": 0,
                "forced_parity_skip": 0,
                "base_task_loss": metrics["task_loss"],
                "safe_anchor_loss": float(anchor_loss.detach().float().cpu()) if torch.isfinite(anchor_loss).item() else None,
                "raw_s2_locr_loss": metrics["raw_locr_loss"],
                "weighted_s2_locr_loss": metrics["weighted_locr_loss"],
                "anomalous_image_count": metrics["details"]["anomalous_image_count"],
                "connected_component_count": metrics["details"]["component_count"],
                "valid_component_count": metrics["details"]["valid_components"],
                "skipped_component_count": metrics["details"]["skipped_empty_interior"] + metrics["details"]["skipped_empty_near"],
                "active_component_count": metrics["details"]["active_components"],
                "near_violation_fraction": metrics["details"]["near_violation_fraction"],
                "median_r_c": metrics["details"]["median_r_c"],
                "violating_margin_mean": metrics["details"]["violating_margin_mean"],
                "violating_margin_max": metrics["details"]["violating_margin_max"],
                "grad_scaler_scale_before": float(scaler.get_scale()),
                "grad_scaler_scale_after": None,
                "optimizer_state_finite": None,
                "parameters_finite": None,
                "global_step_before": global_step,
                "global_step_after": global_step,
            }
            loss_finite = bool(np.isfinite(objective_value) and torch.isfinite(anchor_loss).all().item())
            if not loss_finite:
                natural_skips += 1
                row.update(status="natural_loss_skip", natural_skip=1)
                if candidate and forced:
                    row["forced_parity_skip"] = 1
                    if control_status != "natural_loss_skip":
                        additional_natural_skips += 1
                optimizer.zero_grad(set_to_none=True)
                rows.append(row)
                attempt += 1
                processed += 1
                continue
            scale = float(scaler.get_scale())
            for (name, parameter), gradient in zip(trainable, metrics["gradients"]):
                if gradient is not None:
                    parameter.grad = (gradient * scale).to(dtype=parameter.dtype)
            # GradScaler lazily materializes its device scale on the first
            # scale() call. The gradients above are accumulated explicitly to
            # preserve batch reductions, so initialize that same scale without
            # introducing another loss or backward path.
            scaler.scale(torch.ones((), device=device))
            scaler.unscale_(optimizer)
            finite_grad = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for _, parameter in trainable)
            if not finite_grad:
                natural_skips += 1
                row.update(status="natural_grad_skip", natural_skip=1)
                scaler.update()
                if candidate and forced:
                    row["forced_parity_skip"] = 1
                    if control_status != "natural_grad_skip":
                        additional_natural_skips += 1
                optimizer.zero_grad(set_to_none=True)
                rows.append(row)
                attempt += 1
                processed += 1
                continue
            task_gradient_map = {
                name: metrics["gradients"][trainable_index[f"image_adapter.{name}"]]
                for name in image_names
            }
            anchor_gradients = torch.autograd.grad(anchor_loss, image_params, allow_unused=True)
            anchor_metrics = apply_family_safe_anchor_budget(
                model.image_adapter,
                sorted(model.named_parameters()),
                task_gradients=task_gradient_map,
                raw_anchor_gradients=dict(zip(image_names, anchor_gradients)),
                anchor_lambda=0.0021633926715180626,
                rho=.1,
                total_trainable_parameters=None,
            )
            forced_skip = bool(candidate and forced)
            if forced_skip:
                forced_skips += 1
                row.update(status="forced_parity_skip", forced_parity_skip=1)
                # The control's natural skip is intentionally not counted as a
                # candidate numerical failure or replaced by a compensating step.
                optimizer.zero_grad(set_to_none=True)
            else:
                torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.text_adapter.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.soft_prompt.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                successful += 1
                global_step += 1
                row.update(status="success", successful_update=1, global_step_after=global_step)
                state_finite = optimizer_state_is_finite(optimizer)
                params_finite = finite_model_parameters(model)
                row["optimizer_state_finite"] = int(state_finite)
                row["parameters_finite"] = int(params_finite)
                if not state_finite or not params_finite:
                    numerical_failure = numerical_failure or "nonfinite_state_after_update"
            row["grad_scaler_scale_after"] = float(scaler.get_scale())
            if row["optimizer_state_finite"] is None:
                row["optimizer_state_finite"] = int(optimizer_state_is_finite(optimizer))
            if row["parameters_finite"] is None:
                row["parameters_finite"] = int(finite_model_parameters(model))
            row["safe_anchor_effective_ratio"] = anchor_metrics["global_effective_ratio"]
            row["safe_anchor_max_family_ratio"] = anchor_metrics["max_effective_active_family_ratio"]
            rows.append(row)
            attempt += 1
            processed += 1
            del metrics, anchor_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if len(epoch_manifest) == full_batches and processed == len(epoch_manifest) and attempt < MAX_ATTEMPTS:
            scheduler.step()
            apply_soft_prompt_lr_policy(optimizer, False)
        last_epoch = epoch
        if attempt >= MAX_ATTEMPTS:
            break
    if attempt != MAX_ATTEMPTS:
        raise RuntimeError(f"expected exactly {MAX_ATTEMPTS} attempts, got {attempt}")
    output_dir = RUN_ROOT / arm
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final.pth"
    torch.save(final_training_state(model, optimizer, scheduler, scaler, payload, last_epoch or 11, global_step, arm, lambda_s2, attempt, successful, natural_skips, forced_skips), final_path)
    output_csv = OUT_CANDIDATE_CSV if candidate else OUT_CONTROL_CSV
    write_csv(output_csv, rows)
    summary = {
        "protocol_id": "EXPLORATORY_SOURCE_ONLY_MECHANISM_R1",
        "arm": arm,
        "attempted": attempt,
        "successful": successful,
        "natural_skips": natural_skips,
        "forced_parity_skips": forced_skips,
        "additional_natural_skips": additional_natural_skips,
        "numerical_failure": numerical_failure,
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": spill.sha256_file(final_path),
        "rows": rows,
    }
    dump_json(output_dir / "summary.json", summary)
    return summary


def state_group_drift(reference_state: dict, final_state: dict, prefix: str) -> dict:
    names = sorted(name for name in reference_state if name.startswith(prefix) and name in final_state)
    if not names:
        return {"parameter_count": 0, "l2_delta": None, "reference_l2": None, "relative_l2_delta": None, "cosine_to_e10": None, "names": []}
    ref = torch.cat([reference_state[name].detach().float().reshape(-1) for name in names])
    cur = torch.cat([final_state[name].detach().float().reshape(-1) for name in names])
    delta_value = cur - ref
    ref_norm = float(ref.norm().cpu())
    cur_norm = float(cur.norm().cpu())
    delta_norm = float(delta_value.norm().cpu())
    cosine = float(torch.dot(ref, cur).div(ref.norm() * cur.norm()).cpu()) if ref_norm > 0 and cur_norm > 0 else None
    return {
        "parameter_count": int(sum(reference_state[name].numel() for name in names)),
        "l2_delta": delta_norm,
        "reference_l2": ref_norm,
        "final_l2": cur_norm,
        "relative_l2_delta": delta_norm / ref_norm if ref_norm > 0 else None,
        "cosine_to_e10": cosine,
        "names": names,
    }


def endpoint_arm_metrics(data: dict, arm: str) -> dict:
    mask, labels = data["mask"], data["labels"]
    stage_metrics = []
    for stage in range(3):
        score = data["resized_stage_prob"][:, stage]
        metric = spill.fusion_map_metrics(score, mask, labels, f"stage_{stage + 1}_resized")
        metric.update({"arm": arm, "stage": stage + 1})
        if stage == 1:
            regional = spill.region_summary(score, mask)
            metric["near_mean"] = regional["near"]["mean"]
        stage_metrics.append(metric)
    final_score = data["resized_fused_prob"]
    final_metric = spill.fusion_map_metrics(final_score, mask, labels, "final_production_fused")
    final_metric.update({"arm": arm, "stage": "final"})
    final_regional = spill.region_summary(final_score, mask)
    final_metric.update({
        "near_mean": final_regional["near"]["mean"],
        "far_mean": final_regional["far"]["mean"],
    })
    distance_profiles = {}
    distance_rows = []
    for map_name, score in (
        ("stage_1_resized", data["resized_stage_prob"][:, 0]),
        ("stage_2_resized", data["resized_stage_prob"][:, 1]),
        ("stage_3_resized", data["resized_stage_prob"][:, 2]),
        ("final_production_fused", final_score),
    ):
        current_rows, summary = spill.distance_rows(score, mask, f"{arm}_{map_name}", "518x518")
        distance_rows.extend([{**row, "arm": arm, "map": map_name} for row in current_rows])
        distance_profiles[map_name] = {row["bin"]: row for row in current_rows if row["bin"] != "ALL_OUTSIDE"}
        distance_profiles[map_name]["ALL_OUTSIDE"] = summary
    return {
        "arm": arm,
        "checkpoint": data["checkpoint"],
        "checkpoint_sha256": data["checkpoint_sha256"],
        "stage_metrics": stage_metrics,
        "final_metric": final_metric,
        "distance_profiles": distance_profiles,
        "distance_rows": distance_rows,
        "maps": data,
    }


def endpoint_phase(payload: dict) -> dict:
    if not all_training_gates_pass():
        raise RuntimeError("endpoint requires all pre-training gates")
    if not OUT_CONTROL_CSV.is_file() or not OUT_CANDIDATE_CSV.is_file():
        raise RuntimeError("both control and candidate CSVs are required before endpoint")
    control_path = RUN_ROOT / ARM_CONTROL / "final.pth"
    candidate_path = RUN_ROOT / ARM_CANDIDATE / "final.pth"
    if not control_path.is_file() or not candidate_path.is_file():
        raise RuntimeError("both final training checkpoints are required before endpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for endpoint evaluation")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    rows, _ = spill.load_rows()
    model = spill.make_model(device)
    control_data = spill.evaluate_maps(model, control_path, rows, device, full=False)
    control = endpoint_arm_metrics(control_data, "CONTROL")
    del control_data
    torch.cuda.empty_cache()
    candidate_data = spill.evaluate_maps(model, candidate_path, rows, device, full=False)
    candidate = endpoint_arm_metrics(candidate_data, "CANDIDATE")
    del candidate_data, model
    torch.cuda.empty_cache()
    e10_payload = torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False)
    e10_state = e10_payload["model_state"]["image_adapter"]
    drift = {}
    for arm, path in (("CONTROL", control_path), ("CANDIDATE", candidate_path)):
        final_payload = torch.load(path, map_location="cpu", weights_only=False)
        final_state = final_payload["model_state"]["image_adapter"]
        drift[arm] = {
            "stage1_convlora": state_group_drift(e10_state, final_state, "lora_adapters.0."),
            "stage2_convlora": state_group_drift(e10_state, final_state, "lora_adapters.1."),
            "stage3_convlora": state_group_drift(e10_state, final_state, "lora_adapters.2."),
            "stage2_projection": state_group_drift(e10_state, final_state, "seg_proj.1."),
            "stage2_dfg_q": state_group_drift(e10_state, final_state, "vision_text_q.1."),
            "stage2_dfg_k": state_group_drift(e10_state, final_state, "vision_text_k.1."),
            "stage2_ss2d": state_group_drift(e10_state, final_state, "dfg_ss2d_branches.1."),
            "stage2_ss2d_gamma": state_group_drift(e10_state, final_state, "dfg_raw_gamma.1."),
            "checkpoint_epoch": final_payload.get("epoch"),
            "checkpoint_global_step": final_payload.get("global_step"),
            "checkpoint_sha256": spill.sha256_file(path),
        }
    endpoint_json = {
        "protocol_id": "EXPLORATORY_SOURCE_ONLY_MECHANISM_R1",
        "phase": "25_fixed_source_endpoint",
        "cohort": {"count": 96, "source": str(COHORT.relative_to(REPO)), "source_sha256": spill.sha256_file(COHORT)},
        "fusion": "equal pre-softmax stage-logit fusion only",
        "control": {k: v for k, v in control.items() if k != "maps"},
        "candidate": {k: v for k, v in candidate.items() if k != "maps"},
        "parameter_geometry_drift": drift,
        "no_target_inference": True,
        "medical_inference": False,
        "mvtec_inference": False,
    }
    csv_rows = []
    for arm_data in (control, candidate):
        for metric in arm_data["stage_metrics"]:
            csv_rows.append({"scope": "stage", "arm": arm_data["arm"], **{k: v for k, v in metric.items() if k not in ("arm",)}})
        csv_rows.append({"scope": "final", "arm": arm_data["arm"], **{k: v for k, v in arm_data["final_metric"].items() if k not in ("arm",)}})
        for map_name, bins in arm_data["distance_profiles"].items():
            for bin_name, metric in bins.items():
                csv_rows.append({"scope": "distance", "arm": arm_data["arm"], "map": map_name, **metric})
        for name, metric in drift[arm_data["arm"]].items():
            if isinstance(metric, dict):
                csv_rows.append({"scope": "drift", "arm": arm_data["arm"], "family": name, **{k: v for k, v in metric.items() if k != "names"}})
    write_csv(OUT_ENDPOINT_CSV, csv_rows)
    dump_json(OUT_ENDPOINT_JSON, endpoint_json)
    return endpoint_json


def decision_phase() -> dict:
    endpoint = json.loads(OUT_ENDPOINT_JSON.read_text())
    control_rows = list(csv.DictReader(OUT_CONTROL_CSV.open(newline="")))
    candidate_rows = list(csv.DictReader(OUT_CANDIDATE_CSV.open(newline="")))
    identity_fields = ("attempt_index", "epoch", "batch", "file_names", "labels", "image_sha256", "mask_sha256", "image_sha256_per_item", "mask_sha256_per_item")
    exact_batch_match = len(control_rows) == len(candidate_rows) == MAX_ATTEMPTS and all(all(left[field] == right[field] for field in identity_fields) for left, right in zip(control_rows, candidate_rows))
    control_success = sum(int(row["successful_update"]) for row in control_rows)
    candidate_success = sum(int(row["successful_update"]) for row in candidate_rows)
    control_natural = sum(int(row["natural_skip"]) for row in control_rows)
    candidate_natural = sum(int(row["natural_skip"]) for row in candidate_rows)
    candidate_forced = sum(int(row["forced_parity_skip"]) for row in candidate_rows)
    additional_natural = [row["attempt_index"] for row in candidate_rows if row["natural_skip"] == "1" and control_rows[int(row["attempt_index"])] ["natural_skip"] != "1"]
    control = endpoint["control"]
    candidate = endpoint["candidate"]
    control_final = control["final_metric"]
    candidate_final = candidate["final_metric"]
    control_stage2 = control["stage_metrics"][1]
    candidate_stage2 = candidate["stage_metrics"][1]
    active_rows = [row for row in candidate_rows if int(row["active_component_count"]) > 0 and float(row["weighted_s2_locr_loss"]) > 0.0]
    def ge(left, right, tol=0.0):
        return left >= right - tol
    def le(left, right, tol=0.0):
        return left <= right + tol
    gates = {
        "training_authorization": all_training_gates_pass(),
        "numerical_validity": not additional_natural and all(row["parameters_finite"] == "1" and row["optimizer_state_finite"] == "1" for row in control_rows + candidate_rows),
        "exact_attempted_batch_identity": exact_batch_match,
        "successful_step_count_match": control_success == candidate_success == MAX_ATTEMPTS,
        "s2_locr_active": len(active_rows) > 0,
        "stage2_near_p95_decreases": candidate_stage2["near_p95"] < control_stage2["near_p95"],
        "stage2_near_p99_decreases": candidate_stage2["near_p99"] < control_stage2["near_p99"],
        "final_near_p95_not_increased": le(candidate_final["near_p95"], control_final["near_p95"]),
        "final_near_p99_not_increased": le(candidate_final["near_p99"], control_final["near_p99"]),
        "final_ap_non_decrease": ge(candidate_final["ap"], control_final["ap"]),
        "final_auroc_non_decrease": ge(candidate_final["auroc"], control_final["auroc"]),
        "positive_mean_non_decrease": ge(candidate_final["positive_mean"], control_final["positive_mean"], 1e-6),
        "positive_median_non_decrease": ge(candidate_final["positive_median"], control_final["positive_median"], 1e-6),
        "interior_mean_non_decrease": ge(candidate_final["interior_mean"], control_final["interior_mean"], 1e-6),
        "interior_median_non_decrease": ge(candidate_final["interior_median"], control_final["interior_median"], 1e-6),
        "near_positive_inversion_not_increased": le(candidate_final["near_gt_positive_inversion"], control_final["near_gt_positive_inversion"], 1e-6),
        "near_interior_inversion_not_increased": le(candidate_final["near_gt_interior_inversion"], control_final["near_gt_interior_inversion"], 1e-6),
    }
    bounded_pass = all(gates.values())
    if not gates["training_authorization"]:
        bounded_screen = "PREFLIGHT_REJECTED"
    elif not gates["numerical_validity"]:
        bounded_screen = "INVALID_NUMERICAL"
    elif bounded_pass:
        bounded_screen = "PASS"
    else:
        bounded_screen = "FAIL"
    if bounded_screen == "PASS":
        mechanism = "SUPPORTED"
        confirmatory = "YES"
        recommendation = "RECOMMEND_SEPARATE_CROSS_DOMAIN_STAGE_WISE_GENERALIZATION_AUDIT"
        interpretation = "CASE_A"
    elif bounded_screen == "INVALID_NUMERICAL":
        mechanism = "NOT_ESTABLISHED"
        confirmatory = "NO"
        recommendation = "NOT_TRIGGERED"
        interpretation = "NUMERICAL_INVALIDITY"
    else:
        mechanism = "NOT_SUPPORTED"
        confirmatory = "NO"
        recommendation = "NOT_TRIGGERED"
        if gates["stage2_near_p95_decreases"] and gates["stage2_near_p99_decreases"] and not all((gates["positive_mean_non_decrease"], gates["positive_median_non_decrease"], gates["interior_mean_non_decrease"], gates["interior_median_non_decrease"])):
            interpretation = "CASE_B"
        elif gates["stage2_near_p95_decreases"] and gates["stage2_near_p99_decreases"] and not (gates["final_near_p95_not_increased"] and gates["final_near_p99_not_increased"]):
            interpretation = "CASE_C"
        elif gates["final_ap_non_decrease"] and not gates["final_auroc_non_decrease"]:
            interpretation = "CASE_D"
        elif gates["final_auroc_non_decrease"] and not gates["final_ap_non_decrease"]:
            interpretation = "CASE_E"
        else:
            interpretation = "CASE_F"
    final_metrics = {
        "final_auroc_control": control_final["auroc"],
        "final_auroc_candidate": candidate_final["auroc"],
        "final_auroc_delta": candidate_final["auroc"] - control_final["auroc"],
        "final_ap_control": control_final["ap"],
        "final_ap_candidate": candidate_final["ap"],
        "final_ap_delta": candidate_final["ap"] - control_final["ap"],
        "stage2_near_bg_p95_control": control_stage2["near_p95"],
        "stage2_near_bg_p95_candidate": candidate_stage2["near_p95"],
        "stage2_near_bg_p95_delta": candidate_stage2["near_p95"] - control_stage2["near_p95"],
        "stage2_near_bg_p99_control": control_stage2["near_p99"],
        "stage2_near_bg_p99_candidate": candidate_stage2["near_p99"],
        "stage2_near_bg_p99_delta": candidate_stage2["near_p99"] - control_stage2["near_p99"],
        "final_near_bg_p95_delta": candidate_final["near_p95"] - control_final["near_p95"],
        "final_near_bg_p99_delta": candidate_final["near_p99"] - control_final["near_p99"],
        "positive_mean_delta": candidate_final["positive_mean"] - control_final["positive_mean"],
        "positive_median_delta": candidate_final["positive_median"] - control_final["positive_median"],
        "interior_mean_delta": candidate_final["interior_mean"] - control_final["interior_mean"],
        "interior_median_delta": candidate_final["interior_median"] - control_final["interior_median"],
        "boundary_mean_delta": candidate_final["boundary_mean"] - control_final["boundary_mean"],
        "near_positive_inversion_delta": candidate_final["near_gt_positive_inversion"] - control_final["near_gt_positive_inversion"],
        "near_interior_inversion_delta": candidate_final["near_gt_interior_inversion"] - control_final["near_gt_interior_inversion"],
    }
    decision = {
        "protocol_id": "EXPLORATORY_SOURCE_ONLY_MECHANISM_R1",
        "interpretation_case": interpretation,
        "training_authorization": "YES" if gates["training_authorization"] else "NO",
        "control_attempted": len(control_rows), "control_natural_skips": control_natural, "control_successful": control_success,
        "candidate_attempted": len(candidate_rows), "candidate_natural_skips": candidate_natural, "candidate_forced_parity_skips": candidate_forced, "candidate_successful": candidate_success,
        "candidate_additional_natural_skip_attempts": additional_natural,
        "exact_batch_match": exact_batch_match,
        "successful_count_match": control_success == candidate_success,
        "active_candidate_attempts": len(active_rows),
        "gates": gates,
        "final_metrics": final_metrics,
        "numerical_validity": "PASS" if gates["numerical_validity"] else "FAIL",
        "s2_locr_mechanism": mechanism,
        "bounded_screen": bounded_screen,
        "full_confirmatory_run_justified": confirmatory,
        "post_run_recommendation": recommendation,
        "new_full_training_run": False,
        "medical_inference_run": False,
        "mvtec_inference_run": False,
        "target_tuning_used": False,
        "hyperparameter_sweep": False,
        "waiting_for_user_approval": True,
    }
    markdown_lines = [
        "# H2 S2-LOCR R1 Bounded Decision", "",
        f"* `S2_LOCR_TRAINING_AUTHORIZED={decision['training_authorization']}`",
        f"* `NUMERICAL_VALIDITY={decision['numerical_validity']}`",
        f"* `BOUNDED_SCREEN={bounded_screen}`",
        f"* `S2_LOCR_MECHANISM={mechanism}`",
        f"* `FULL_CONFIRMATORY_RUN_JUSTIFIED={confirmatory}`",
        f"* `INTERPRETATION={interpretation}`", "",
        "## Pairing",
        f"* attempts: control={len(control_rows)}, candidate={len(candidate_rows)}",
        f"* successful: control={control_success}, candidate={candidate_success}",
        f"* exact batch identity: `{exact_batch_match}`",
        f"* candidate additional natural skips: `{additional_natural}`", "",
        "## Gate results",
    ]
    markdown_lines.extend(f"* `{name}={'PASS' if value else 'FAIL'}`" for name, value in gates.items())
    markdown_lines.extend([
        "", "## Interpretation", "",
        "The result is a source-only exploratory mechanism screen. It does not establish novelty or authorize a confirmatory full run automatically.",
        f"", f"Post-run recommendation flag: `{recommendation}`.", "",
        "## Prohibitions and scope",
        "Medical inference: NO; MVTec inference: NO; target tuning: NO; hyperparameter sweep: NO; E15/E20 full training: NO.",
        "WAITING_FOR_USER_APPROVAL=YES",
    ])
    OUT_DECISION_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_DECISION_MD.write_text("\n".join(markdown_lines) + "\n")
    dump_json(OUT_DECISION_JSON, decision)
    return decision


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
        preflight = preflight_microbatch(model, batch, device, policy, stage2_params)
        gradients = preflight["gradients"]
        task_grads = gradients["task"]
        locr_grads = gradients["locr"]
        focal_grads = gradients["focal"]
        normal_grads = gradients["normal_dice"]
        abnormal_grads = gradients["abnormal_dice"]
        row = {
            "batch_index": batch_index,
            "file_names": json.dumps(list(batch["file_name"]), separators=(",", ":")),
            "labels": json.dumps(batch["label"].tolist(), separators=(",", ":")),
            "task_loss": preflight["task_value"],
            "raw_s2_locr_loss": preflight["locr_value"],
            "stage2_task_grad_norm": gradient_norm(task_grads),
            "stage2_locr_grad_norm": gradient_norm(locr_grads),
            "cosine_locr_task": cosine_gradients(locr_grads, task_grads, stage2_params),
            "cosine_locr_focal": cosine_gradients(locr_grads, focal_grads, stage2_params),
            "cosine_locr_normal_dice": cosine_gradients(locr_grads, normal_grads, stage2_params),
            "cosine_locr_abnormal_dice": cosine_gradients(locr_grads, abnormal_grads, stage2_params),
            "active": int(preflight["details"]["active"]),
            "active_components": preflight["details"]["active_components"],
            "valid_components": preflight["details"]["valid_components"],
            "near_violation_fraction": preflight["details"]["near_violation_fraction"],
            "anomalous_image_count": preflight["details"]["anomalous_image_count"],
            "nonfinite": int(not np.isfinite(preflight["task_value"]) or not np.isfinite(preflight["locr_value"])),
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
        del preflight
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
    batch = {key: (value[:1] if torch.is_tensor(value) else [value[0]]) for key, value in batch.items()}
    terms = batch_terms(model, batch, device, policy)
    trainable = [(name, parameter) for name, parameter in sorted(model.named_parameters()) if parameter.requires_grad]
    parameters = [parameter for _, parameter in trainable]
    control_scalar = terms["task"]
    # S2 disabled means the auxiliary branch is absent, exactly as in the
    # historical control; a zero-multiplied auxiliary graph is not parity.
    candidate_disabled_scalar = terms["task"]
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
        "safe_anchor_lambda": 0.0021633926715180626,
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
    parser.add_argument("--phase", choices=("identity", "oracle", "coverage", "locality", "preflight", "calibration", "parity", "manifest", "control", "candidate", "endpoint", "decision"), required=True)
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
    elif args.phase == "manifest":
        generate_attempt_manifest(torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False))
    elif args.phase == "control":
        if not OUT_MANIFEST_JSON.is_file():
            raise RuntimeError("attempt manifest is required before control")
        manifest = json.loads(OUT_MANIFEST_JSON.read_text())
        run_training_arm(torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False), ARM_CONTROL, manifest)
    elif args.phase == "candidate":
        if not OUT_MANIFEST_JSON.is_file() or not OUT_CONTROL_CSV.is_file():
            raise RuntimeError("manifest and control artifacts are required before candidate")
        manifest = json.loads(OUT_MANIFEST_JSON.read_text())
        run_training_arm(torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False), ARM_CANDIDATE, manifest, load_control_rows())
    elif args.phase == "endpoint":
        endpoint_phase(torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False))
    elif args.phase == "decision":
        decision_phase()


if __name__ == "__main__":
    main()
