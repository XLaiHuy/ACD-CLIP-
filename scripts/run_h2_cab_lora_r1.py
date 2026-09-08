#!/usr/bin/env python3
"""CAB-LoRA R1: bounded, source-only Stage-2 Conv-LoRA screen.

The driver deliberately stops at the 500-attempt bounded decision.  It does
not contain a Medical or MVTec path.  CAB is an auxiliary representation loss
implemented by replaying only the Stage-2 Conv-LoRA block from detached
Stage-2 inputs; no production forward, DFG, SS2D, projection, fusion, or
evaluation code is modified.
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
from scipy import ndimage
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


spill = load_module("h2_cab_spill", REPO / "scripts/run_h2_boundary_spillover_audit_r1.py")
s2 = load_module("h2_cab_s2_helpers", REPO / "scripts/run_h2_s2_locr_r1.py")

from h2_clean.contract import SafeImageAdapterAnchor, apply_family_safe_anchor_budget
from h2_clean.precision import PrecisionPolicy
from train import calculate_seg_loss


IMG = 518
NATIVE = 37
PATCH_FOOTPRINT = 14
CONTEXT_HALO = 14
CONTEXT_TRANSITION = 14
MAX_POSITIONS_PER_IMAGE = 4
MAX_ATTEMPTS = 500
SEED = 0
ANCHOR_LAMBDA = 0.0021633926715180626
ANCHOR_FAMILY_BUDGET = 0.10
CAB_TARGET_GRADIENT_RATIO = 0.025
GRAD_EPS = 1.0e-12
PARENT_HEAD = "284b12b6dc7802bcbafc02d7809549c40758f94b"
BRANCH = "research/h2-cab-lora-r1-bounded"
SAFE_ANCHOR = Path("/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth")
SAFE_SHA = "64b72dc3d1155285c826781bee4c5970bd45218e95b21675fd19d9a6b2ab54a7"
RUN_ROOT = Path("/workspace/h2_cab_lora_r1_bounded")
CONTROL = "CAB_R1_CONTROL"
CANDIDATE = "CAB_R1_CANDIDATE"

OUT_IDENTITY_MD = REPO / "audit/H2_CAB_R1_IDENTITY_OFF.md"
OUT_IDENTITY_JSON = REPO / "audit/H2_CAB_R1_IDENTITY_OFF.json"
OUT_RESEARCH = REPO / "audit/H2_CAB_R1_RESEARCH.md"
OUT_PROTOCOL_MD = REPO / "audit/H2_CAB_R1_PROTOCOL.md"
OUT_PROTOCOL_JSON = REPO / "audit/H2_CAB_R1_PROTOCOL.json"
OUT_GRAPH_MD = REPO / "audit/H2_CAB_R1_GRAPH_PROVENANCE.md"
OUT_CF_MD = REPO / "audit/H2_CAB_R1_COUNTERFACTUAL_AUDIT.md"
OUT_CF_JSON = REPO / "audit/H2_CAB_R1_COUNTERFACTUAL_AUDIT.json"
OUT_PREFLIGHT_MD = REPO / "audit/H2_CAB_R1_GRADIENT_PREFLIGHT.md"
OUT_PREFLIGHT_JSON = REPO / "audit/H2_CAB_R1_GRADIENT_PREFLIGHT.json"
OUT_CALIBRATION_JSON = REPO / "audit/H2_CAB_R1_LAMBDA_CALIBRATION.json"
OUT_MANIFEST_JSON = RUN_ROOT / "attempt_manifest.json"
OUT_SCREEN_CSV = REPO / "audit/H2_CAB_R1_PAIRED_SCREEN.csv"
OUT_MECHANISM_MD = REPO / "audit/H2_CAB_R1_MECHANISM_ANALYSIS.md"
OUT_MECHANISM_JSON = REPO / "audit/H2_CAB_R1_MECHANISM_ANALYSIS.json"
OUT_DECISION_MD = REPO / "results/H2_CAB_R1_BOUNDED_DECISION.md"
OUT_DECISION_JSON = REPO / "results/H2_CAB_R1_BOUNDED_DECISION.json"
STATE_JSON = RUN_ROOT / "RUN_STATE.json"
LAST_STAGE = RUN_ROOT / "LAST_COMPLETED_STAGE.txt"
OVERNIGHT_LOG = RUN_ROOT / "OVERNIGHT_LOG.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=json_default, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])
    os.replace(temporary, path)


def stats(values) -> dict:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if not data.size:
        return {"count": 0, "mean": None, "median": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p50": float(np.quantile(data, .50)),
        "p95": float(np.quantile(data, .95)),
        "p99": float(np.quantile(data, .99)),
        "max": float(data.max()),
    }


def set_stage(stage: str, status: str = "RUNNING") -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    state = {
        "branch": BRANCH,
        "parent_head": PARENT_HEAD,
        "safe_anchor": str(SAFE_ANCHOR),
        "safe_anchor_sha256": SAFE_SHA,
        "stage": stage,
        "status": status,
        "updated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    dump_json(STATE_JSON, state)
    LAST_STAGE.write_text(stage + "\n", encoding="utf-8")


def restore_rng(payload: dict) -> None:
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])


def validate_parent_and_start() -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    if branch != BRANCH or status:
        raise RuntimeError(f"CAB identity requires clean {BRANCH}, got branch={branch!r}, status={status!r}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", PARENT_HEAD, head], cwd=REPO).returncode != 0:
        raise RuntimeError("CAB parent HEAD is not an ancestor")
    if sha256_file(SAFE_ANCHOR) != SAFE_SHA:
        raise RuntimeError("Safe Anchor SHA-256 mismatch")
    payload = torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False)
    expected = {
        "epoch": 10, "global_step": 3607, "precision": "fp16", "amp_enabled": True,
        "tf32_enabled": False, "seed": 0, "gradscaler_enabled": True,
        "precision_protocol": "HISTORICAL_MIXED_FP16_FP32_V1", "n_groups": 3,
        "dfg_mode": "attn", "dfg_attn_dim": 256, "dfg_attn_tau": 8.0,
        "use_ss2d_dfg": True, "dfg_ss2d_fusion": "weight_residual",
        "dfg_weight_residual_fp32": True, "use_safe_anchor": True,
        "anchor_lambda": ANCHOR_LAMBDA, "anchor_family_budget": ANCHOR_FAMILY_BUDGET,
        "use_functional_feature_anchor": False, "use_cir_training": False,
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
        "use_functional_feature_anchor": cfg.get("use_functional_feature_anchor", False),
        "use_cir_training": cfg.get("use_cir_training", False),
    }
    mismatch = {key: {"expected": expected[key], "observed": observed[key]} for key in expected if expected[key] != observed[key]}
    required = ("model_state", "optimizer_state", "scheduler_state", "scaler_state", "python_random_state", "numpy_random_state", "torch_cpu_rng_state", "torch_cuda_rng_state_all", "dataloader_generator_state", "resolved_scientific_config")
    missing = [key for key in required if key not in payload]
    if mismatch or missing:
        raise RuntimeError(json.dumps({"mismatch": mismatch, "missing": missing}, sort_keys=True))
    return {"branch": branch, "head": head, "status": "PASS", "payload": payload, "observed": observed, "required_state_keys": required}


def protocol_artifacts() -> None:
    OUT_RESEARCH.write_text(
        """# H2 CAB-LoRA R1 Research Record

Frozen root-cause evidence says the frozen ViT already contains contextual
information, while Conv-LoRA is the primary trainable context amplifier and
Stage 2 is the strongest observed spillover site. S2-LOCR showed that direct
Stage-2 score suppression can reduce near-background responses while also
reducing anomaly/interior evidence. The Functional Anchor showed that global
feature preservation is non-selective. NFUR-R2 was not supported, so CAB-R1
does not add a downstream refinement head.

CAB-R1 therefore tests one falsifiable hypothesis: penalize only the positive
increase in context sensitivity introduced by Stage-2 Conv-LoRA, relative to
the frozen/pre-adapter representation. Native ViT contextuality is not
matched globally and no score suppression, feature teacher, DFG change, SS2D
change, routing, or multi-stage term is present.

The experiment is source-only on VisA train. It is a bounded 500-attempt
CONTROL versus CAB-LoRA R1 screen and stops at the decision artifact. Medical
and MVTec inference are prohibited before a passing bounded decision and are
not part of this driver.
""", encoding="utf-8")
    protocol = {
        "protocol_id": "H2_CAB_LORA_R1_BOUNDED",
        "branch": BRANCH,
        "parent_head": PARENT_HEAD,
        "source_dataset": "VisA train",
        "target_inference_before_decision": False,
        "mechanism": "CAB-LoRA R1",
        "formula": "mean_i ReLU((1-cos(z_post(x_i),z_post(x_tilde_i)))-(1-cos(z_pre(x_i),z_pre(x_tilde_i))))",
        "pre_adapter_tensor": "Stage-2 Conv-LoRA input t=x[1:] after transformer block 16, before image_adapter.lora_adapters[1]",
        "post_adapter_tensor": "norm-matched Stage-2 Conv-LoRA output merged with detached m_i_w[1] coefficient",
        "protected_geometry": "native token footprint rows [14r,14r+13], cols [14c,14c+13]",
        "native_grid": [37, 37],
        "patch_footprint": [14, 14],
        "counterfactual": "cyclic peer context from the same source batch, with union of selected token footprints plus 14px halo protected and a 14px distance-based smooth transition",
        "selected_tokens": "up to 4 deterministic row-major valid near-background tokens per image; zero-footprint and intersecting existing 7x7 near-background region",
        "intervention_scope": ["Stage-2 Conv-LoRA only"],
        "unchanged_modules": ["Stage-1 Conv-LoRA", "Stage-3 Conv-LoRA", "DFG", "DFG Q/K", "SS2D", "segmentation projection", "prompt branch", "stage fusion", "inference smoothing", "interpolation", "NFUR", "classification head", "evaluation code"],
        "max_attempts": MAX_ATTEMPTS,
        "arms": ["CONTROL", "CAB-LoRA R1"],
        "matched_contract": ["starting checkpoint", "seed", "attempt manifest", "batch order", "augmentations", "optimizer", "LR", "scheduler", "AMP", "gradient clipping", "Safe Anchor", "DFG", "SS2D", "prompt settings", "attempt count"],
        "cab_disabled_default": True,
        "cab_target_gradient_ratio": CAB_TARGET_GRADIENT_RATIO,
        "cab_target_gradient_ratio_rationale": "2.5% is a single conservative source-only calibration target: CAB acts at representation level upstream of logits, and a small ratio reduces the risk of suppressing useful anomaly-context relationships; it is not the prior S2-LOCR 5% value.",
        "no_sweep": True,
        "no_medical_tuning": True,
        "no_mvtec_tuning": True,
        "full_train_justified_only_if_bounded_pass": True,
    }
    dump_json(OUT_PROTOCOL_JSON, protocol)
    OUT_PROTOCOL_MD.write_text(
        "# H2 CAB-LoRA R1 Protocol\n\n" + json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_model_for_training(payload: dict, device: torch.device):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = s2.make_training_model(payload, device)
    model.stage_fusion_weights = (1.0 / 3.0,) * 3
    model.cab_enabled = False
    model.eval()
    return model


def first_source_batch(payload: dict) -> dict:
    restore_rng(payload)
    dataset = spill.get_text_and_image_dataset("VisA", IMG, "train")
    return next(iter(s2.loader_for_epoch(dataset, 11)))


def fixed_source_batches(payload: dict, count: int = 16) -> list[dict]:
    restore_rng(payload)
    dataset = spill.get_text_and_image_dataset("VisA", IMG, "train")
    rows = []
    for batch in s2.loader_for_epoch(dataset, 11):
        rows.append(batch)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"expected {count} fixed source batches, got {len(rows)}")
    return rows


def capture_stage2_input(model, image: torch.Tensor, policy: PrecisionPolicy, no_grad: bool) -> torch.Tensor:
    captured = {}
    hook = model.image_adapter["lora_adapters"][1].register_forward_pre_hook(
        lambda _module, inputs: captured.setdefault("pre", inputs[0])
    )
    context = torch.no_grad() if no_grad else torch.enable_grad()
    with context, policy.autocast(image.device):
        try:
            model(image)
        finally:
            hook.remove()
    if "pre" not in captured:
        raise RuntimeError("Stage-2 Conv-LoRA pre tensor was not captured")
    return captured["pre"]


def replay_stage2_post(model, pre: torch.Tensor) -> torch.Tensor:
    """Replay only Stage-2 Conv-LoRA with a detached merge coefficient."""
    with torch.autocast(device_type=pre.device.type, enabled=False):
        x = pre.detach().float()
        adapter = model.image_adapter["lora_adapters"][1]
        adapt = adapter(x)
        adapt = adapt * x.norm(dim=-1, keepdim=True) / adapt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        merge = model.image_adapter["m_i_w"][1].i_w.detach().float()
        return (1.0 - merge) * x + merge * adapt


def valid_token_positions(mask: torch.Tensor) -> list[list[int]]:
    selected = []
    for current in mask[:, 0].detach().float().cpu().numpy():
        binary = current > 0.5
        near = ndimage.binary_dilation(binary, structure=np.ones((7, 7), dtype=bool)) & ~binary
        footprint = binary.reshape(NATIVE, PATCH_FOOTPRINT, NATIVE, PATCH_FOOTPRINT).any(axis=(1, 3))
        near_footprint = near.reshape(NATIVE, PATCH_FOOTPRINT, NATIVE, PATCH_FOOTPRINT).any(axis=(1, 3))
        valid = (~footprint) & near_footprint
        candidates = np.flatnonzero(valid.reshape(-1))
        selected.append([int(x) for x in candidates[:MAX_POSITIONS_PER_IMAGE]])
    return selected


def context_counterfactual(images: torch.Tensor, positions: list[list[int]]) -> tuple[torch.Tensor, dict]:
    """Replace peer context while preserving every selected local footprint."""
    donor = images.roll(shifts=-1, dims=0)
    output = images.clone()
    footprint_diffs, context_changes, protected_fractions = [], [], []
    pair_rows = []
    for batch_index, tokens in enumerate(positions):
        if not tokens:
            continue
        protected = np.zeros((IMG, IMG), dtype=bool)
        for token in tokens:
            row, col = divmod(int(token), NATIVE)
            top = max(0, PATCH_FOOTPRINT * row - CONTEXT_HALO)
            bottom = min(IMG, PATCH_FOOTPRINT * (row + 1) + CONTEXT_HALO)
            left = max(0, PATCH_FOOTPRINT * col - CONTEXT_HALO)
            right = min(IMG, PATCH_FOOTPRINT * (col + 1) + CONTEXT_HALO)
            protected[top:bottom, left:right] = True
        distance = ndimage.distance_transform_edt(~protected).astype(np.float32)
        alpha = np.clip(distance / float(CONTEXT_TRANSITION), 0.0, 1.0)
        alpha_t = torch.from_numpy(alpha).to(device=images.device, dtype=images.dtype)
        current = (1.0 - alpha_t) * images[batch_index] + alpha_t * donor[batch_index]
        current[:, protected] = images[batch_index][:, protected]
        output[batch_index] = current
        exact_diffs = []
        outside_diffs = []
        source = images[batch_index].detach().float().cpu().numpy()
        changed = current.detach().float().cpu().numpy()
        for token in tokens:
            row, col = divmod(int(token), NATIVE)
            top, bottom = PATCH_FOOTPRINT * row, PATCH_FOOTPRINT * (row + 1)
            left, right = PATCH_FOOTPRINT * col, PATCH_FOOTPRINT * (col + 1)
            footprint = np.zeros((IMG, IMG), dtype=bool)
            footprint[top:bottom, left:right] = True
            exact_diffs.append(float(np.abs(changed[:, footprint] - source[:, footprint]).max()))
            outside_diffs.append(float(np.abs(changed[:, ~footprint] - source[:, ~footprint]).mean()))
        footprint_diffs.extend(exact_diffs)
        context_changes.extend(outside_diffs)
        protected_fractions.append(float(protected.mean()))
        pair_rows.append({
            "batch_index": batch_index,
            "token_indices": tokens,
            "footprint_max_abs_diff": max(exact_diffs),
            "outside_footprint_mean_abs_diff": float(np.mean(outside_diffs)),
            "protected_pixel_fraction": float(protected.mean()),
            "donor_rule": "cyclic next image in same VisA source batch",
        })
    return output, {
        "pair_count": len(pair_rows),
        "token_count": len(footprint_diffs),
        "footprint_max_abs_diff": float(max(footprint_diffs)) if footprint_diffs else 0.0,
        "context_change_mean": float(np.mean(context_changes)) if context_changes else 0.0,
        "context_change_p95": float(np.quantile(context_changes, .95)) if context_changes else 0.0,
        "protected_pixel_fraction_mean": float(np.mean(protected_fractions)) if protected_fractions else 0.0,
        "rows": pair_rows,
    }


def single_image_batch(batch: dict, index: int) -> dict:
    return {
        key: (value[index:index + 1] if torch.is_tensor(value) else [value[index]])
        for key, value in batch.items()
    }


def task_terms(model, batch: dict, device: torch.device, policy: PrecisionPolicy) -> dict:
    image = batch["image"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    class_names = list(batch["class_name"])
    text, kg_loss, k_loss = s2.batch_text_features(model, class_names, device)
    captured = {}
    hook = model.image_adapter["lora_adapters"][1].register_forward_pre_hook(
        lambda _module, inputs: captured.setdefault("pre", inputs[0])
    )
    with policy.autocast(device):
        try:
            seg_tokens, det_tokens = model(image)
        finally:
            hook.remove()
        vision = torch.stack(seg_tokens)
        det = torch.stack(det_tokens)
        cls = torch.stack([
            torch.matmul(det[i].unsqueeze(1), text[i]).squeeze(1)
            for i in range(3)
        ]).mean(0)
        cls_loss = F.cross_entropy(cls, label)
        seg_pred = model.vision_text_fusion_gate_seg(vision, text)
        seg_loss = calculate_seg_loss(seg_pred, mask)
        task = cls_loss + seg_loss + 0.01 * kg_loss + 0.002 * k_loss
    if "pre" not in captured:
        raise RuntimeError("task graph did not capture Stage-2 pre tensor")
    return {
        "image": image, "mask": mask, "label": label, "class_names": class_names,
        "text": text, "task": task, "cls_loss": cls_loss, "seg_loss": seg_loss,
        "pre": captured["pre"], "seg_pred": seg_pred,
    }


def empty_cab_detail() -> dict:
    return {"token_count": 0, "active_token_count": 0, "active_token_fraction": 0.0, "cab_excess_mean": 0.0, "d_pre": stats([]), "d_post": stats([]), "positive_delta": stats([]), "positive_delta_sum": 0.0, "requires_grad": False, "finite": True}


def cab_pair_terms(model, pre: torch.Tensor, counter_pre: torch.Tensor, tokens: list[int], requires_grad: bool) -> tuple[torch.Tensor, dict]:
    if not tokens:
        zero = pre.sum() * 0.0
        return zero if requires_grad else zero.detach(), empty_cab_detail()
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        real_post = replay_stage2_post(model, pre)
        counter_post = replay_stage2_post(model, counter_pre)
    real_pre_values, counter_pre_values, real_post_values, counter_post_values = [], [], [], []
    for token in tokens:
        real_pre_values.append(pre[token, 0])
        counter_pre_values.append(counter_pre[token, 0])
        real_post_values.append(real_post[token, 0])
        counter_post_values.append(counter_post[token, 0])
    real_pre_values = torch.stack(real_pre_values).float()
    counter_pre_values = torch.stack(counter_pre_values).float()
    real_post_values = torch.stack(real_post_values).float()
    counter_post_values = torch.stack(counter_post_values).float()
    d_pre = 1.0 - F.cosine_similarity(real_pre_values.detach(), counter_pre_values.detach(), dim=-1)
    d_post = 1.0 - F.cosine_similarity(real_post_values, counter_post_values, dim=-1)
    positive_delta = F.relu(d_post - d_pre)
    loss = positive_delta.mean()
    delta_detached = (d_post.detach() - d_pre.detach()).float()
    active = delta_detached > 0.0
    detail = {
        "token_count": int(delta_detached.numel()),
        "active_token_count": int(active.sum().item()),
        "active_token_fraction": float(active.float().mean().item()),
        "cab_excess_mean": float(positive_delta.detach().float().mean().item()),
        "d_pre": stats(d_pre.detach().cpu().numpy()),
        "d_post": stats(d_post.detach().cpu().numpy()),
        "positive_delta": stats(positive_delta.detach().cpu().numpy()),
        "positive_delta_sum": float(positive_delta.detach().float().sum().item()),
        "requires_grad": bool(loss.requires_grad),
        "finite": bool(torch.isfinite(d_pre).all().item() and torch.isfinite(d_post).all().item() and torch.isfinite(loss).item()),
        "_d_pre_values": d_pre.detach().cpu().tolist(),
        "_d_post_values": d_post.detach().cpu().tolist(),
        "_positive_delta_values": positive_delta.detach().cpu().tolist(),
    }
    return loss if requires_grad else loss.detach(), detail


def aggregate_cab_details(details: list[dict]) -> dict:
    token_count = sum(item["token_count"] for item in details)
    active_count = sum(item["active_token_count"] for item in details)
    def pooled(key: str):
        values = []
        for item in details:
            values.extend(item.get(f"_{key}_values", []))
        return stats(values)
    return {
        "token_count": token_count,
        "active_token_count": active_count,
        "active_token_fraction": active_count / token_count if token_count else 0.0,
        "cab_excess_mean": float(sum(item["cab_excess_mean"] * item["token_count"] for item in details) / token_count) if token_count else 0.0,
        "d_pre": pooled("d_pre"), "d_post": pooled("d_post"), "positive_delta": pooled("positive_delta"),
        "finite": all(item["finite"] for item in details),
    }


def batch_gradient_terms(model, batch: dict, device: torch.device, policy: PrecisionPolicy, trainable_params, stage2_indices, candidate: bool) -> dict:
    """Compute one batch using one-image graphs to fit the historical GPU."""
    image = batch["image"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    positions = valid_token_positions(mask)
    counterfactual, locality = context_counterfactual(image, positions)
    batch_size = int(image.shape[0])
    valid_indices = [i for i, values in enumerate(positions) if values]
    valid_count = max(1, len(valid_indices))
    task_accum = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in trainable_params]
    cab_accum = [torch.zeros_like(trainable_params[index], dtype=torch.float32) for index in stage2_indices]
    task_values, cab_details = [], []
    for index in range(batch_size):
        one = single_image_batch(batch, index)
        terms = task_terms(model, one, device, policy)
        task_grads = torch.autograd.grad(terms["task"], trainable_params, allow_unused=True)
        for target, gradient in zip(task_accum, task_grads):
            if gradient is not None:
                target.add_(gradient.detach().float(), alpha=1.0 / batch_size)
        task_values.append(float(terms["task"].detach().float().cpu()) / batch_size)
        if positions[index]:
            counter_pre = capture_stage2_input(model, counterfactual[index:index + 1], policy, no_grad=True)
            cab_loss, detail = cab_pair_terms(model, terms["pre"], counter_pre, positions[index], requires_grad=candidate)
            if candidate:
                cab_grads = torch.autograd.grad(cab_loss, [trainable_params[j] for j in stage2_indices], allow_unused=True)
                for target, gradient in zip(cab_accum, cab_grads):
                    if gradient is not None:
                        target.add_(gradient.detach().float(), alpha=1.0 / valid_count)
            detail["weighted_loss"] = float(cab_loss.detach().float().cpu()) / valid_count
            cab_details.append(detail)
        else:
            cab_details.append(empty_cab_detail())
        del terms, task_grads
        if candidate and positions[index]:
            del cab_loss, counter_pre
        torch.cuda.empty_cache()
    return {
        "task_gradients": task_accum,
        "cab_stage2_gradients": cab_accum,
        "task_loss": float(sum(task_values)),
        "cab_loss": float(sum(item.get("weighted_loss", 0.0) for item in cab_details)),
        "cab_detail": aggregate_cab_details(cab_details),
        "locality": locality,
        "positions": positions,
    }


def stage2_lora_parameters(model):
    return [(name, parameter) for name, parameter in sorted(model.image_adapter["lora_adapters"][1].named_parameters())]


def grad_norm(values) -> float:
    tensors = [value.detach().float() for value in values if value is not None]
    if not tensors:
        return 0.0
    return float(torch.stack([value.square().sum() for value in tensors]).sum().sqrt().item())


def combined_gradients(task_grads, cab_grads, candidate: bool, lambda_cab: float):
    output = []
    for task_grad, cab_grad in zip(task_grads, cab_grads):
        if task_grad is None and cab_grad is None:
            output.append(None)
        elif task_grad is None:
            output.append(float(lambda_cab) * cab_grad.detach().float() if candidate else torch.zeros_like(cab_grad.detach().float()))
        elif cab_grad is None or not candidate:
            output.append(task_grad.detach().float())
        else:
            output.append(task_grad.detach().float() + float(lambda_cab) * cab_grad.detach().float())
    return output


def identity_phase(payload: dict) -> dict:
    set_stage("IDENTITY_OFF")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocol_artifacts()
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    batch = single_image_batch(first_source_batch(payload), 0)
    model = make_model_for_training(payload, device)
    terms_a = task_terms(model, batch, device, policy)
    outputs_a = production_outputs(model, terms_a["pre"], terms_a["text"], terms_a["image"], payload, policy, use_full_forward=False)
    # A separate disabled-by-default invocation uses exactly the same model and
    # input, with no CAB branch or CAB graph attached.
    terms_b = task_terms(model, batch, device, policy)
    outputs_b = production_outputs(model, terms_b["pre"], terms_b["text"], terms_b["image"], payload, policy, use_full_forward=False)
    labels = {"stage1_logits": outputs_a["stage_logits"][0], "stage2_logits": outputs_a["stage_logits"][1], "stage3_logits": outputs_a["stage_logits"][2], "final_fused_logits": outputs_a["final_logits"], "final_anomaly_map": outputs_a["final_map"], "task_loss": terms_a["task"]}
    labels_b = {"stage1_logits": outputs_b["stage_logits"][0], "stage2_logits": outputs_b["stage_logits"][1], "stage3_logits": outputs_b["stage_logits"][2], "final_fused_logits": outputs_b["final_logits"], "final_anomaly_map": outputs_b["final_map"], "task_loss": terms_b["task"]}
    diffs = {key: float((labels[key].detach().float() - labels_b[key].detach().float()).abs().max().item()) for key in labels}
    artifact = {
        "protocol_id": "H2_CAB_LORA_R1_BOUNDED",
        "cab_enabled": False,
        "identity_off_parity": "PASS" if all(value == 0.0 for value in diffs.values()) else "FAIL",
        "strict_tolerance": 1.0e-6,
        "diffs_max_abs": diffs,
        "finite": all(torch.isfinite(value).all().item() for value in labels.values()),
        "graph_shapes": {"stage2_pre": list(terms_a["pre"].shape), "stage2_post_replay": list(replay_stage2_post(model, terms_a["pre"]).shape), "stage_logits": list(outputs_a["stage_logits"].shape), "final_logits": list(outputs_a["final_logits"].shape), "final_map": list(outputs_a["final_map"].shape)},
        "checkpoint": str(SAFE_ANCHOR), "checkpoint_sha256": SAFE_SHA,
        "source_batch_file_names": list(batch["file_name"]),
    }
    dump_json(OUT_IDENTITY_JSON, artifact)
    OUT_IDENTITY_MD.write_text("# H2 CAB-LoRA R1 Identity-Off Audit\n\n" + json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_GRAPH_MD.write_text(
        """# H2 CAB-LoRA R1 Graph Provenance

The production path is `model/adapter.py:ACDCLIP.forward`. The ViT stream
enters Stage 2 at `t=x[1:]` after transformer block 16, shape
`[1369,B,1024]`, before `image_adapter[\"lora_adapters\"][1]`. This is the
pre-adapter tensor `z_pre`; it is captured with a forward pre-hook only for
instrumentation and is detached for CAB.

The raw Stage-2 Conv-LoRA output is norm-matched to `t` exactly as the
production path does. CAB's `z_post` uses that output and the Stage-2
`m_i_w[1]` merge coefficient, but the coefficient and pre tensor are
detached. Consequently the CAB graph contains only Stage-2 Conv-LoRA
parameters. It stops before Stage-2 projection, DFG Q/K, SS2D, Stage 3,
stage fusion, and interpolation.

Stage-2 logits are the unchanged `_vision_text_attention_fusion` output
converted to `[B,2,37,37]`; the production score path is Gaussian 7x7,
bilinear resize to 518 with `align_corners=True`, equal pre-softmax fusion,
then softmax. CAB never edits these logits.

The identity artifact records exact disabled parity for Stage-1/2/3 logits,
final fused logits, final anomaly map, and task loss. CAB is disabled by
default and has no production-forward branch.
""", encoding="utf-8")
    if artifact["identity_off_parity"] != "PASS":
        raise RuntimeError("CAB_R1_IDENTITY_OFF=FAIL")
    set_stage("IDENTITY_OFF", "PASS")
    return artifact


def production_outputs(model, pre, text, image, payload, policy, use_full_forward=False):
    # Re-run the exact production forward to obtain all stage tokens. The
    # optional args make the call site explicit that CAB does not alter it.
    with torch.no_grad(), policy.autocast(image.device):
        seg_tokens, _ = model(image)
        vision = torch.stack(seg_tokens)
        group_text = text.permute(1, 0, 2, 3)
        stage_logits = []
        for stage in range(3):
            fused = model._vision_text_attention_fusion(vision[stage], group_text, stage)
            native = torch.matmul(10 * vision[stage], fused).permute(0, 2, 1).view(image.shape[0], 2, NATIVE, NATIVE)
            stage_logits.append(F.interpolate(
                spill.gaussian_blur2d(native, (7, 7), (1, 1)),
                (IMG, IMG), mode="bilinear", align_corners=True,
            ))
        stage_logits = torch.stack(stage_logits)
        final_logits = stage_logits.mean(dim=0)
        final_map = F.softmax(final_logits, dim=1)[:, 1]
    return {"stage_logits": stage_logits, "final_logits": final_logits, "final_map": final_map}


def counterfactual_audit_phase(payload: dict) -> dict:
    set_stage("COUNTERFACTUAL_AUDIT")
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    model = make_model_for_training(payload, device)
    rows, locality_rows = [], []
    for batch_index, batch in enumerate(fixed_source_batches(payload, 16)):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        positions = valid_token_positions(mask)
        _, locality = context_counterfactual(image, positions)
        rows.append({"batch_index": batch_index, "file_names": list(batch["file_name"]), "selected_token_count": sum(len(x) for x in positions), "positions": positions, **{key: value for key, value in locality.items() if key != "rows"}})
        locality_rows.extend(locality["rows"])
    footprint = [row["footprint_max_abs_diff"] for row in rows]
    changes = [row["context_change_mean"] for row in rows if row["selected_token_count"]]
    artifact = {
        "protocol_id": "H2_CAB_LORA_R1_BOUNDED",
        "audit_batches": len(rows),
        "construction": "cyclic peer context from same VisA source batch; union of selected footprints plus 14px halo protected; 14px smooth distance transition",
        "selection": "existing source training masks only; zero-footprint native token intersecting existing 7x7 near-background",
        "red_team": {
            "hard outside-footprint swap": "rejected: creates a direct seam adjacent to the protected token",
            "hard peer-image replacement": "rejected: same seam risk and stronger synthetic cue",
            "naive spatial roll": "rejected: wrap-around border cue",
            "crop_or_resize": "rejected: can alter the protected local footprint",
            "chosen_peer_context": "same-source cyclic peer with protected halo and smooth transition; no labels or category names choose the donor",
        },
        "local_footprint_identity": "PASS" if max(footprint or [0.0]) == 0.0 else "FAIL",
        "context_changed": "PASS" if bool(changes) and float(np.mean(changes)) > 1.0e-3 else "FAIL",
        "no_target_information": True,
        "source_domain_compatible": True,
        "deterministic": True,
        "computational_form": "one counterfactual image per source image, preserving up to four selected token footprints; no per-token full-image explosion",
        "summary": {"footprint_max_abs_diff": float(max(footprint or [0.0])), "context_change_mean": float(np.mean(changes)) if changes else 0.0, "context_change_p95": float(np.quantile(changes, .95)) if changes else 0.0, "selected_token_count": int(sum(row["selected_token_count"] for row in rows))},
        "rows": rows,
        "pair_rows": locality_rows,
    }
    dump_json(OUT_CF_JSON, artifact)
    OUT_CF_MD.write_text("# H2 CAB-LoRA R1 Counterfactual Audit\n\n" + json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if artifact["local_footprint_identity"] != "PASS" or artifact["context_changed"] != "PASS":
        raise RuntimeError("CAB_R1_COUNTERFACTUAL_VALIDITY=FAIL")
    set_stage("COUNTERFACTUAL_AUDIT", "PASS")
    return artifact


def gradient_preflight_phase(payload: dict) -> dict:
    set_stage("GRADIENT_PREFLIGHT")
    cf = json.loads(OUT_CF_JSON.read_text())
    if cf["local_footprint_identity"] != "PASS" or cf["context_changed"] != "PASS":
        raise RuntimeError("counterfactual audit gate failed")
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    model = make_model_for_training(payload, device)
    trainable = [(name, parameter) for name, parameter in sorted(model.named_parameters()) if parameter.requires_grad]
    trainable_params = [parameter for _, parameter in trainable]
    stage2_pairs = stage2_lora_parameters(model)
    stage2_params = [parameter for _, parameter in stage2_pairs]
    rows, ratios = [], []
    fixed_batches = fixed_source_batches(payload, 16)
    for batch_index, batch in enumerate(fixed_batches):
        result = batch_gradient_terms(model, batch, device, policy, trainable_params, [index for index, (name, _) in enumerate(trainable) if name.startswith("image_adapter.lora_adapters.1.")], True)
        detail = result["cab_detail"]
        if not detail["finite"]:
            raise RuntimeError(f"nonfinite CAB preflight batch {batch_index}")
        stage2_indices_local = [index for index, (name, _) in enumerate(trainable) if name.startswith("image_adapter.lora_adapters.1.")]
        task_norm = grad_norm([result["task_gradients"][index] for index in stage2_indices_local])
        cab_norm = grad_norm(result["cab_stage2_gradients"])
        ratio = cab_norm / (task_norm + GRAD_EPS)
        result["ratio"] = ratio
        ratios.append(ratio)
        rows.append({"batch_index": batch_index, "file_names": list(batch["file_name"]), "task_loss": result["task_loss"], "cab_loss": result["cab_loss"], "cab_active": detail["active_token_count"] > 0, "active_token_fraction": detail["active_token_fraction"], "token_count": detail["token_count"], "d_pre": detail["d_pre"], "d_post": detail["d_post"], "positive_delta": detail["positive_delta"], "stage2_task_grad_norm": task_norm, "stage2_cab_raw_grad_norm": cab_norm, "raw_cab_to_task_ratio": ratio, "locality": result["locality"]})
        del result
        torch.cuda.empty_cache()
    # Scope the raw CAB graph directly on one active source image. Because
    # the pre tensor and merge coefficient are detached in cab_pair_terms,
    # any nonzero gradient outside Stage-2 Conv-LoRA is an implementation bug.
    scope_rows = []
    for batch_index, batch in enumerate(fixed_batches):
        positions = valid_token_positions(batch["mask"])
        active_index = next((index for index, values in enumerate(positions) if values), None)
        if active_index is None:
            continue
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        counterfactual, _ = context_counterfactual(image, positions)
        terms = task_terms(model, single_image_batch(batch, active_index), device, policy)
        counter_pre = capture_stage2_input(model, counterfactual[active_index:active_index + 1], policy, no_grad=True)
        cab_loss, detail = cab_pair_terms(model, terms["pre"], counter_pre, positions[active_index], True)
        all_cab = torch.autograd.grad(cab_loss, trainable_params, allow_unused=True)
        predicates = {
            "stage2_convlora": lambda n: n.startswith("image_adapter.lora_adapters.1."),
            "stage2_projection": lambda n: n.startswith("image_adapter.seg_proj.1."),
            "stage2_dfg_q": lambda n: n.startswith("image_adapter.vision_text_q.1."),
            "stage2_dfg_k": lambda n: n.startswith("image_adapter.vision_text_k.1."),
            "stage2_ss2d": lambda n: n.startswith("image_adapter.dfg_ss2d_branches.1.") or n.startswith("image_adapter.dfg_raw_gamma.1."),
            "stage1_convlora": lambda n: n.startswith("image_adapter.lora_adapters.0."),
            "stage3_convlora": lambda n: n.startswith("image_adapter.lora_adapters.2."),
            "prompt_text": lambda n: n.startswith("text_adapter.") or n.startswith("soft_prompt."),
            "stage2_merge_weight": lambda n: n.startswith("image_adapter.m_i_w.1."),
        }
        scope = {group: grad_norm([gradient for (name, _), gradient in zip(trainable, all_cab) if predicate(name)]) for group, predicate in predicates.items()}
        scope_rows.append({"batch_index": batch_index, **scope})
        del terms, cab_loss, all_cab, counter_pre
        torch.cuda.empty_cache()
        if len(scope_rows) >= 4:
            break
    if not scope_rows:
        raise RuntimeError("no active batch available for CAB gradient scope audit")
    scope_max = {key: max(row[key] for row in scope_rows) for key in scope_rows[0] if key != "batch_index"}
    scope_pass = scope_max["stage2_convlora"] > 0.0 and all(value == 0.0 for key, value in scope_max.items() if key != "stage2_convlora")
    active_count = sum(int(row["cab_active"]) for row in rows)
    artifact = {
        "protocol_id": "H2_CAB_LORA_R1_BOUNDED",
        "phase": "pre-optimizer-step gradient and activity audit",
        "batch_count": len(rows),
        "active_batch_count": active_count,
        "active_batch_fraction": active_count / len(rows),
        "active_token_fraction": stats([row["active_token_fraction"] for row in rows]),
        "d_pre_distribution": {key: stats([row["d_pre"][key] for row in rows if row["d_pre"][key] is not None]) for key in ("mean", "median", "p95", "p99")},
        "d_post_distribution": {key: stats([row["d_post"][key] for row in rows if row["d_post"][key] is not None]) for key in ("mean", "median", "p95", "p99")},
        "positive_delta_distribution": {key: stats([row["positive_delta"][key] for row in rows if row["positive_delta"][key] is not None]) for key in ("mean", "median", "p95", "p99")},
        "gradient_scope_max_norm": scope_max,
        "gradient_scope_rows": scope_rows,
        "ratios": ratios,
        "raw_cab_to_task_ratio_median": float(np.median(ratios)),
        "all_finite": all(np.isfinite(ratios)) and all(row["locality"]["footprint_max_abs_diff"] == 0.0 for row in rows),
        "CAB_ACTIVITY": "PASS" if active_count > 0 and sum(row["token_count"] for row in rows) > 0 else "FAIL",
        "CAB_GRADIENT_SCOPE": "PASS" if scope_pass else "FAIL",
        "rows": rows,
    }
    dump_json(OUT_PREFLIGHT_JSON, artifact)
    OUT_PREFLIGHT_MD.write_text("# H2 CAB-LoRA R1 Gradient Preflight\n\n" + json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if artifact["CAB_ACTIVITY"] != "PASS" or artifact["CAB_GRADIENT_SCOPE"] != "PASS" or not artifact["all_finite"]:
        raise RuntimeError("CAB_R1 gradient/activity preflight failed")
    set_stage("GRADIENT_PREFLIGHT", "PASS")
    return artifact


def lambda_calibration_phase() -> dict:
    set_stage("LAMBDA_CALIBRATION")
    preflight = json.loads(OUT_PREFLIGHT_JSON.read_text())
    if preflight["CAB_GRADIENT_SCOPE"] != "PASS" or preflight["CAB_ACTIVITY"] != "PASS":
        raise RuntimeError("gradient preflight gate failed")
    ratios = np.asarray(preflight["ratios"], dtype=np.float64)
    R = float(np.median(ratios))
    lam = float(CAB_TARGET_GRADIENT_RATIO / R) if np.isfinite(R) and R > 0.0 else None
    artifact = {
        "protocol_id": "H2_CAB_LORA_R1_BOUNDED",
        "phase": "single deterministic source-only calibration",
        "ratio_definition": "median over fixed 16 VisA source batches of ||g_CAB||_Stage2ConvLoRA/(||g_task||_Stage2ConvLoRA+1e-12)",
        "raw_ratios": ratios.tolist(),
        "R": R,
        "target_effective_ratio": CAB_TARGET_GRADIENT_RATIO,
        "lambda_cab": lam,
        "rounded_for_training": False,
        "rationale": "CAB is an upstream representation-level relative bound; 2.5% is a conservative frozen target to reduce anomaly-suppression risk and is independently chosen rather than reusing S2-LOCR's 5%.",
        "no_sweep": True,
        "CAB_LAMBDA_CALIBRATION": "PASS" if lam is not None and np.isfinite(lam) and lam > 0.0 else "FAIL",
    }
    dump_json(OUT_CALIBRATION_JSON, artifact)
    if artifact["CAB_LAMBDA_CALIBRATION"] != "PASS":
        raise RuntimeError("CAB lambda calibration failed")
    set_stage("LAMBDA_CALIBRATION", "PASS")
    return artifact


def make_attempt_manifest(payload: dict) -> dict:
    set_stage("ATTEMPT_MANIFEST")
    # Dataset transforms use the checkpoint-restored Python/NumPy/Torch RNGs;
    # restore them before materializing hashes so training replays the exact
    # augmented tensors rather than only the DataLoader permutation.
    restore_rng(payload)
    dataset = spill.get_text_and_image_dataset("VisA", IMG, "train")
    attempts, epoch = [], 11
    while len(attempts) < MAX_ATTEMPTS:
        for batch_index, batch in enumerate(s2.loader_for_epoch(dataset, epoch)):
            image, mask = batch["image"], batch["mask"]
            attempts.append({
                "attempt_index": len(attempts), "epoch": epoch, "batch": batch_index,
                "file_names": list(batch["file_name"]), "labels": [int(x) for x in batch["label"].tolist()],
                "image_sha256": tensor_sha256(image), "mask_sha256": tensor_sha256(mask),
                "image_sha256_per_item": [tensor_sha256(image[i:i+1]) for i in range(image.shape[0])],
                "mask_sha256_per_item": [tensor_sha256(mask[i:i+1]) for i in range(mask.shape[0])],
            })
            if len(attempts) == MAX_ATTEMPTS:
                break
        epoch += 1
    artifact = {"protocol_id": "H2_CAB_LORA_R1_BOUNDED", "source_dataset": "VisA train", "attempt_count": len(attempts), "batch_size": 6, "image_size": IMG, "seed": SEED, "start_checkpoint": str(SAFE_ANCHOR), "start_checkpoint_sha256": SAFE_SHA, "loader_definition": "shuffle=True, num_workers=0, pin_memory=True, epoch seed=104729*epoch, starts at epoch 11", "target_inference": False, "attempts": attempts}
    dump_json(OUT_MANIFEST_JSON, artifact)
    set_stage("ATTEMPT_MANIFEST", "PASS")
    return artifact


def manifest_row_identity(expected: dict, batch: dict) -> bool:
    image, mask = batch["image"], batch["mask"]
    return (
        expected["file_names"] == list(batch["file_name"])
        and expected["labels"] == [int(x) for x in batch["label"].tolist()]
        and expected["image_sha256"] == tensor_sha256(image)
        and expected["mask_sha256"] == tensor_sha256(mask)
        and expected["image_sha256_per_item"] == [tensor_sha256(image[i:i+1]) for i in range(image.shape[0])]
        and expected["mask_sha256_per_item"] == [tensor_sha256(mask[i:i+1]) for i in range(mask.shape[0])]
    )


def train_arm(payload: dict, manifest: dict, arm: str, control_rows: list[dict] | None = None) -> dict:
    candidate = arm == CANDIDATE
    calibration = json.loads(OUT_CALIBRATION_JSON.read_text())
    lambda_cab = float(calibration["lambda_cab"])
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    model = make_model_for_training(payload, device)
    optimizer, scheduler, scaler = s2.make_training_optimizer(model, payload)
    anchor = SafeImageAdapterAnchor.from_checkpoint(SAFE_ANCHOR, device)
    trainable = [(name, parameter) for name, parameter in sorted(model.named_parameters()) if parameter.requires_grad]
    trainable_params = [parameter for _, parameter in trainable]
    stage2_indices = [index for index, (name, _) in enumerate(trainable) if name.startswith("image_adapter.lora_adapters.1.")]
    image_named = [(name, parameter) for name, parameter in sorted(model.image_adapter.named_parameters()) if parameter.requires_grad]
    image_names = [name for name, _ in image_named]
    image_params = [parameter for _, parameter in image_named]
    dataset = spill.get_text_and_image_dataset("VisA", IMG, "train")
    restore_rng(payload)
    control_by_attempt = {int(row["attempt_index"]): row for row in (control_rows or [])}
    rows, epoch_groups = [], {}
    for item in manifest["attempts"]:
        epoch_groups.setdefault(int(item["epoch"]), []).append(item)
    attempt = successful = natural_skips = forced_skips = additional_natural = 0
    global_step = int(payload["global_step"])
    for epoch, expected_items in epoch_groups.items():
        s2.configure_training_epoch(model, epoch)
        processed = 0
        for batch_index, batch in enumerate(s2.loader_for_epoch(dataset, epoch)):
            if processed >= len(expected_items):
                break
            expected = expected_items[processed]
            if int(expected["attempt_index"]) != attempt or int(expected["batch"]) != batch_index or not manifest_row_identity(expected, batch):
                raise RuntimeError(f"attempt manifest mismatch arm={arm} attempt={attempt} epoch={epoch} batch={batch_index}")
            optimizer.zero_grad(set_to_none=True)
            result = batch_gradient_terms(model, batch, device, policy, trainable_params, stage2_indices, candidate)
            combined = list(result["task_gradients"])
            if candidate:
                for local_index, global_index in enumerate(stage2_indices):
                    combined[global_index] = combined[global_index] + lambda_cab * result["cab_stage2_gradients"][local_index]
            anchor_loss = anchor.loss(model.image_adapter)
            finite = bool(np.isfinite(result["task_loss"]) and np.isfinite(result["cab_loss"]) and torch.isfinite(anchor_loss).item() and all(g is None or torch.isfinite(g).all().item() for g in combined))
            forced = bool(candidate and int(expected["attempt_index"]) in control_by_attempt and control_by_attempt[int(expected["attempt_index"])] ["status"] != "success")
            row = {
                "attempt_index": attempt, "epoch": epoch, "batch": batch_index, "arm": arm,
                "file_names": json.dumps(list(batch["file_name"]), separators=(",", ":")),
                "labels": json.dumps([int(x) for x in batch["label"].tolist()], separators=(",", ":")),
                "image_sha256": expected["image_sha256"], "mask_sha256": expected["mask_sha256"],
                "task_loss": result["task_loss"], "cab_loss": result["cab_loss"],
                "cab_excess_mean": result["cab_detail"]["cab_excess_mean"], "cab_active_token_fraction": result["cab_detail"]["active_token_fraction"], "cab_token_count": result["cab_detail"]["token_count"],
                "d_pre_mean": result["cab_detail"]["d_pre"]["mean"], "d_post_mean": result["cab_detail"]["d_post"]["mean"], "positive_delta_p95": result["cab_detail"]["positive_delta"]["p95"],
                "context_change_mean": result["locality"]["context_change_mean"], "footprint_max_abs_diff": result["locality"]["footprint_max_abs_diff"],
                "finite_before_update": int(finite), "status": "pending", "successful_update": 0, "natural_skip": 0, "forced_parity_skip": 0,
                "global_step_before": global_step, "global_step_after": global_step,
            }
            if not finite:
                natural_skips += 1
                row.update({"status": "natural_skip", "natural_skip": 1})
                if candidate and forced:
                    row["forced_parity_skip"] = 1
                optimizer.zero_grad(set_to_none=True)
            elif forced:
                forced_skips += 1
                row.update({"status": "forced_parity_skip", "forced_parity_skip": 1})
                optimizer.zero_grad(set_to_none=True)
            else:
                scale = float(scaler.get_scale())
                for parameter, gradient in zip(trainable_params, combined):
                    parameter.grad = None if gradient is None else (gradient * scale).to(dtype=parameter.dtype)
                scaler.scale(torch.ones((), device=device))
                scaler.unscale_(optimizer)
                image_indices = [index for index, (name, _) in enumerate(trainable) if name.startswith("image_adapter.")]
                image_task_map = {
                    name: combined[index]
                    for (name, _), index in zip(image_named, image_indices)
                }
                raw_anchor = torch.autograd.grad(anchor_loss, image_params, allow_unused=True)
                anchor_metrics = apply_family_safe_anchor_budget(model.image_adapter, sorted(model.named_parameters()), task_gradients=image_task_map, raw_anchor_gradients=dict(zip(image_names, raw_anchor)), anchor_lambda=ANCHOR_LAMBDA, rho=ANCHOR_FAMILY_BUDGET, total_trainable_parameters=None)
                torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.text_adapter.parameters(), 1.0)
                if any(parameter.requires_grad for parameter in model.soft_prompt.parameters()):
                    torch.nn.utils.clip_grad_norm_(model.soft_prompt.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                successful += 1
                global_step += 1
                row.update({"status": "success", "successful_update": 1, "global_step_after": global_step, "safe_anchor_effective_ratio": anchor_metrics["global_effective_ratio"]})
                if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters() if parameter.requires_grad):
                    raise RuntimeError(f"nonfinite parameter after {arm} attempt {attempt}")
                if not all(torch.isfinite(value).all().item() for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)):
                    raise RuntimeError(f"nonfinite optimizer state after {arm} attempt {attempt}")
            rows.append(row)
            attempt += 1
            processed += 1
            del result, combined, anchor_loss
            torch.cuda.empty_cache()
            if attempt >= MAX_ATTEMPTS:
                break
        if processed == len(expected_items) and attempt < MAX_ATTEMPTS:
            scheduler.step()
        if attempt >= MAX_ATTEMPTS:
            break
    additional_natural = sum(1 for row in rows if row["natural_skip"] and control_by_attempt.get(int(row["attempt_index"]), {}).get("status") == "success")
    final_path = RUN_ROOT / arm / "final.pth"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "protocol_id": "H2_CAB_LORA_R1_BOUNDED", "arm": arm, "epoch": rows[-1]["epoch"], "global_step": global_step,
        "model_state": {"image_adapter": model.image_adapter.state_dict(), "text_adapter": model.text_adapter.state_dict(), "soft_prompt": model.soft_prompt.state_dict()},
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
        "resolved_scientific_config": payload["resolved_scientific_config"], "parent_scientific_config": payload.get("parent_scientific_config"), "resolved_operational_config": payload.get("resolved_operational_config"),
        "seed": SEED, "precision": "fp16", "amp_enabled": True, "gradscaler_enabled": True, "tf32_enabled": False,
        "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(), "torch_cpu_rng_state": torch.get_rng_state(), "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "cab": {"enabled": candidate, "lambda_cab": lambda_cab, "target_gradient_ratio": CAB_TARGET_GRADIENT_RATIO, "counterfactual": "cyclic peer with protected halo and smooth transition", "scope": "Stage-2 Conv-LoRA only"},
        "attempted": attempt, "successful": successful, "natural_skips": natural_skips, "forced_parity_skips": forced_skips, "additional_natural_skips": additional_natural,
    }
    torch.save(state, final_path)
    summary = {**state, "final_checkpoint": str(final_path), "final_checkpoint_sha256": sha256_file(final_path), "rows": rows, "model_state": None, "optimizer_state": None, "scheduler_state": None, "scaler_state": None, "python_random_state": None, "numpy_random_state": None, "torch_cpu_rng_state": None, "torch_cuda_rng_state_all": None}
    dump_json(final_path.parent / "summary.json", summary)
    return summary


def paired_screen_phase(payload: dict) -> dict:
    set_stage("PAIRED_SCREEN")
    manifest = json.loads(OUT_MANIFEST_JSON.read_text())
    control = train_arm(payload, manifest, CONTROL)
    candidate = train_arm(payload, manifest, CANDIDATE, control["rows"])
    if len(control["rows"]) != MAX_ATTEMPTS or len(candidate["rows"]) != MAX_ATTEMPTS:
        raise RuntimeError("paired screen did not complete exactly 500 attempts per arm")
    paired = []
    for left, right in zip(control["rows"], candidate["rows"]):
        fields = ("attempt_index", "epoch", "batch", "file_names", "labels", "image_sha256", "mask_sha256")
        if any(left[field] != right[field] for field in fields):
            raise RuntimeError(f"paired identity mismatch at attempt {left['attempt_index']}")
        paired.append({
            **{field: left[field] for field in fields},
            "control_status": left["status"], "candidate_status": right["status"],
            "control_task_loss": left["task_loss"], "candidate_task_loss": right["task_loss"],
            "control_cab_excess_mean": left["cab_excess_mean"], "candidate_cab_excess_mean": right["cab_excess_mean"],
            "control_d_pre_mean": left["d_pre_mean"], "candidate_d_pre_mean": right["d_pre_mean"],
            "control_d_post_mean": left["d_post_mean"], "candidate_d_post_mean": right["d_post_mean"],
            "control_cab_active_token_fraction": left["cab_active_token_fraction"], "candidate_cab_active_token_fraction": right["cab_active_token_fraction"],
            "control_context_change_mean": left["context_change_mean"], "candidate_context_change_mean": right["context_change_mean"],
            "control_successful_update": left["successful_update"], "candidate_successful_update": right["successful_update"],
        })
    write_csv(OUT_SCREEN_CSV, paired)
    artifact = {"protocol_id": "H2_CAB_LORA_R1_BOUNDED", "attempt_count_per_arm": MAX_ATTEMPTS, "exact_batch_identity": True, "control": {key: control[key] for key in ("attempted", "successful", "natural_skips", "forced_parity_skips", "additional_natural_skips", "final_checkpoint", "final_checkpoint_sha256")}, "candidate": {key: candidate[key] for key in ("attempted", "successful", "natural_skips", "forced_parity_skips", "additional_natural_skips", "final_checkpoint", "final_checkpoint_sha256")}, "paired_csv": str(OUT_SCREEN_CSV.relative_to(REPO)), "paired_rows": paired}
    dump_json(RUN_ROOT / "paired_screen_summary.json", artifact)
    set_stage("PAIRED_SCREEN", "PASS")
    return artifact


def full_region_values(score: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    values = {name: [] for name in ("positive", "interior", "boundary", "near", "far")}
    for current, current_mask in zip(score, mask):
        boundary, interior, near, far = spill.morphology(current_mask)
        for name, region in (("positive", current_mask.astype(bool)), ("interior", interior), ("boundary", boundary), ("near", near), ("far", far)):
            if region.any():
                values[name].append(current[region].reshape(-1))
    return {name: np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32) for name, chunks in values.items()}


def pair_metric(score: np.ndarray, mask: np.ndarray, positive: str, negative: str) -> dict:
    values = full_region_values(score, mask)
    x = np.concatenate((values[positive], values[negative]))
    y = np.concatenate((np.ones(values[positive].size, dtype=np.uint8), np.zeros(values[negative].size, dtype=np.uint8)))
    return spill.binary_metrics(x, y)


def fpr_recall(score: np.ndarray, mask: np.ndarray) -> dict:
    values = full_region_values(score, mask)
    result = {}
    for name, rate in (("1pct", .01), ("5pct", .05)):
        threshold = float(np.quantile(values["far"], 1.0 - rate)) if values["far"].size else None
        result[name] = {"threshold": threshold, "recall": float((values["positive"] > threshold).mean()) if threshold is not None and values["positive"].size else None}
    return result


def source_metric(final_score: np.ndarray, stage2_score: np.ndarray, mask: np.ndarray, labels: np.ndarray) -> dict:
    global_metric = spill.binary_metrics(final_score, mask)
    final_regions = full_region_values(final_score, mask)
    stage2_regions = full_region_values(stage2_score, mask)
    fpr = fpr_recall(final_score, mask)
    return {
        "pixel_ap": global_metric["ap"], "pixel_auroc": global_metric["auroc"], "pixel_count": global_metric["pixel_count"],
        "anomaly_vs_near": {"ap": pair_metric(final_score, mask, "positive", "near")["ap"], "auroc": pair_metric(final_score, mask, "positive", "near")["auroc"]},
        "interior_vs_near": {"auroc": pair_metric(final_score, mask, "interior", "near")["auroc"]},
        "boundary_vs_near": {"auroc": pair_metric(final_score, mask, "boundary", "near")["auroc"]},
        "stage2_near_p95": stats(stage2_regions["near"])["p95"], "stage2_near_p99": stats(stage2_regions["near"])["p99"],
        "final_near_p95": stats(final_regions["near"])["p95"], "final_near_p99": stats(final_regions["near"])["p99"],
        "positive_mean": stats(final_regions["positive"])["mean"], "positive_median": stats(final_regions["positive"])["median"],
        "interior_mean": stats(final_regions["interior"])["mean"], "interior_median": stats(final_regions["interior"])["median"],
        "boundary_mean": stats(final_regions["boundary"])["mean"], "boundary_median": stats(final_regions["boundary"])["median"],
        "recall_at_1pct_fpr": fpr["1pct"]["recall"], "recall_at_5pct_fpr": fpr["5pct"]["recall"],
        "near_gt_positive_inversion": spill.sampled_inversion(final_score, final_score, mask, greater=True),
        "near_gt_interior_inversion": spill.sampled_inversion(final_score, final_score, mask, greater=False),
    }


def endpoint_phase(payload: dict) -> dict:
    set_stage("SOURCE_ENDPOINT")
    paired = json.loads((RUN_ROOT / "paired_screen_summary.json").read_text())
    if paired["control"]["attempted"] != MAX_ATTEMPTS or paired["candidate"]["attempted"] != MAX_ATTEMPTS:
        raise RuntimeError("endpoint requires complete paired screen")
    rows, _ = spill.load_rows()
    if len(rows) != 96:
        raise RuntimeError("source endpoint must use frozen 96-image VisA cohort")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    model = spill.make_model(device)
    arms = {}
    for arm, key in ((CONTROL, "control"), (CANDIDATE, "candidate")):
        path = Path(paired[key]["final_checkpoint"])
        data = spill.evaluate_maps(model, path, rows, device, full=False)
        arms[arm] = {"checkpoint": str(path), "checkpoint_sha256": sha256_file(path), "metrics": source_metric(data["resized_fused_prob"], data["resized_stage_prob"][:, 1], data["mask"], data["labels"]), "cohort_count": len(rows), "target_inference": False}
        del data
        torch.cuda.empty_cache()
    dump_json(RUN_ROOT / "source_endpoint.json", arms)
    set_stage("SOURCE_ENDPOINT", "PASS")
    return arms


def decision_phase(payload: dict) -> dict:
    set_stage("DECISION")
    preflight = json.loads(OUT_PREFLIGHT_JSON.read_text())
    calibration = json.loads(OUT_CALIBRATION_JSON.read_text())
    paired = json.loads((RUN_ROOT / "paired_screen_summary.json").read_text())
    endpoint = json.loads((RUN_ROOT / "source_endpoint.json").read_text())
    control_rows = paired["control"]
    candidate_rows = paired["candidate"]
    cmetric = endpoint[CONTROL]["metrics"]
    xmetric = endpoint[CANDIDATE]["metrics"]
    screen = json.loads((RUN_ROOT / "paired_screen_summary.json").read_text())["paired_rows"] if "paired_rows" in json.loads((RUN_ROOT / "paired_screen_summary.json").read_text()) else list(csv.DictReader(OUT_SCREEN_CSV.open(newline="")))
    control_excess = np.asarray([float(row["control_cab_excess_mean"]) for row in screen], dtype=np.float64)
    candidate_excess = np.asarray([float(row["candidate_cab_excess_mean"]) for row in screen], dtype=np.float64)
    candidate_active = np.asarray([float(row["candidate_cab_active_token_fraction"]) for row in screen], dtype=np.float64)
    def ge(a, b): return a >= b - 1.0e-6
    def le(a, b): return a <= b + 1.0e-6
    gates = {
        "CAB_R1_IDENTITY_OFF": json.loads(OUT_IDENTITY_JSON.read_text())["identity_off_parity"] == "PASS",
        "CAB_R1_COUNTERFACTUAL_VALIDITY": json.loads(OUT_CF_JSON.read_text())["local_footprint_identity"] == "PASS" and json.loads(OUT_CF_JSON.read_text())["context_changed"] == "PASS",
        "CAB_R1_GRADIENT_SCOPE": preflight["CAB_GRADIENT_SCOPE"] == "PASS",
        "CAB_R1_NUMERICAL_VALIDITY": control_rows["natural_skips"] == 0 and candidate_rows["natural_skips"] == 0 and candidate_rows["additional_natural_skips"] == 0 and control_rows["successful"] == candidate_rows["successful"] == MAX_ATTEMPTS,
        "CAB_R1_ACTIVITY": preflight["CAB_ACTIVITY"] == "PASS" and bool(np.any(candidate_active > 0.0)),
        "CAB_R1_CONTEXT_AMPLIFICATION_REDUCED": bool(np.mean(candidate_excess) < np.mean(control_excess)),
        "CAB_R1_NEAR_BACKGROUND": le(xmetric["stage2_near_p95"], cmetric["stage2_near_p95"]) and le(xmetric["stage2_near_p99"], cmetric["stage2_near_p99"]) and le(xmetric["final_near_p95"], cmetric["final_near_p95"]) and le(xmetric["final_near_p99"], cmetric["final_near_p99"]),
        "CAB_R1_ANOMALY_PRESERVATION": ge(xmetric["positive_mean"], cmetric["positive_mean"]) and ge(xmetric["positive_median"], cmetric["positive_median"]) and ge(xmetric["interior_mean"], cmetric["interior_mean"]) and ge(xmetric["interior_median"], cmetric["interior_median"]),
        "CAB_R1_PIXEL_AP": ge(xmetric["pixel_ap"], cmetric["pixel_ap"]),
        "CAB_R1_PIXEL_AUROC": ge(xmetric["pixel_auroc"], cmetric["pixel_auroc"]),
        "CAB_R1_ANOMALY_VS_NEAR": ge(xmetric["anomaly_vs_near"]["auroc"], cmetric["anomaly_vs_near"]["auroc"]) and ge(xmetric["anomaly_vs_near"]["ap"], cmetric["anomaly_vs_near"]["ap"]),
        "CAB_R1_BOUNDARY_INTERIOR": ge(xmetric["interior_vs_near"]["auroc"], cmetric["interior_vs_near"]["auroc"]) and ge(xmetric["boundary_vs_near"]["auroc"], cmetric["boundary_vs_near"]["auroc"]),
    }
    all_pass = all(gates.values())
    decision = {
        "protocol_id": "H2_CAB_LORA_R1_BOUNDED",
        "CAB_R1_IDENTITY_OFF": "PASS" if gates["CAB_R1_IDENTITY_OFF"] else "FAIL",
        "CAB_R1_COUNTERFACTUAL_VALIDITY": "PASS" if gates["CAB_R1_COUNTERFACTUAL_VALIDITY"] else "FAIL",
        "CAB_R1_GRADIENT_SCOPE": "PASS" if gates["CAB_R1_GRADIENT_SCOPE"] else "FAIL",
        "CAB_R1_NUMERICAL_VALIDITY": "PASS" if gates["CAB_R1_NUMERICAL_VALIDITY"] else "FAIL",
        "CAB_R1_ACTIVITY": "PASS" if gates["CAB_R1_ACTIVITY"] else "FAIL",
        "CAB_R1_CONTEXT_AMPLIFICATION_REDUCED": "PASS" if gates["CAB_R1_CONTEXT_AMPLIFICATION_REDUCED"] else "FAIL",
        "CAB_R1_NEAR_BACKGROUND": "PASS" if gates["CAB_R1_NEAR_BACKGROUND"] else "FAIL",
        "CAB_R1_ANOMALY_PRESERVATION": "PASS" if gates["CAB_R1_ANOMALY_PRESERVATION"] else "FAIL",
        "CAB_R1_PIXEL_AP": "PASS" if gates["CAB_R1_PIXEL_AP"] else "FAIL",
        "CAB_R1_PIXEL_AUROC": "PASS" if gates["CAB_R1_PIXEL_AUROC"] else "FAIL",
        "CAB_R1_ANOMALY_VS_NEAR": "PASS" if gates["CAB_R1_ANOMALY_VS_NEAR"] else "FAIL",
        "CAB_R1_BOUNDARY_INTERIOR": "PASS" if gates["CAB_R1_BOUNDARY_INTERIOR"] else "FAIL",
        "CAB_R1_MECHANISM_SUPPORTED": "YES" if all_pass else "NO",
        "CAB_R1_FULL_TRAIN_JUSTIFIED": "YES" if all_pass else "NO",
        "FINAL_INTERPRETATION": "CAB_SUPPORTED" if all_pass else "CAB_NOT_SUPPORTED",
        "gates": gates,
        "control": {"attempted": control_rows["attempted"], "successful": control_rows["successful"], "natural_skips": control_rows["natural_skips"], "checkpoint": control_rows["final_checkpoint"], "checkpoint_sha256": control_rows["final_checkpoint_sha256"], "metrics": cmetric},
        "candidate": {"attempted": candidate_rows["attempted"], "successful": candidate_rows["successful"], "natural_skips": candidate_rows["natural_skips"], "checkpoint": candidate_rows["final_checkpoint"], "checkpoint_sha256": candidate_rows["final_checkpoint_sha256"], "metrics": xmetric},
        "mechanism": {"control_cab_excess_mean": float(np.mean(control_excess)), "candidate_cab_excess_mean": float(np.mean(candidate_excess)), "delta_candidate_minus_control": float(np.mean(candidate_excess) - np.mean(control_excess)), "candidate_active_token_fraction_mean": float(np.mean(candidate_active)), "preflight": {"d_pre": preflight["d_pre_distribution"], "d_post": preflight["d_post_distribution"], "positive_delta": preflight["positive_delta_distribution"]}},
        "lambda_calibration": calibration,
        "source_only": True,
        "medical_inference_run": False,
        "mvtec_inference_run": False,
        "target_tuning_used": False,
        "hyperparameter_sweep": False,
        "prohibited_after_fail": ["Medical inference", "MVTec inference", "lambda retry", "lambda sweep", "architecture expansion", "multi-stage CAB", "DFG/SS2D modification", "NFUR", "full E15/E20"],
    }
    dump_json(OUT_DECISION_JSON, decision)
    OUT_DECISION_MD.write_text("# H2 CAB-LoRA R1 Bounded Decision\n\n" + json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MECHANISM_JSON.write_text(json.dumps(decision["mechanism"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MECHANISM_MD.write_text("# H2 CAB-LoRA R1 Mechanism Analysis\n\n" + json.dumps(decision["mechanism"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    set_stage("DECISION", "PASS")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("identity", "counterfactual", "preflight", "calibration", "manifest", "control", "candidate", "screen", "endpoint", "decision"), required=True)
    args = parser.parse_args()
    identity = validate_parent_and_start()
    payload = identity["payload"]
    if args.phase == "identity":
        identity_phase(payload)
    elif args.phase == "counterfactual":
        counterfactual_audit_phase(payload)
    elif args.phase == "preflight":
        gradient_preflight_phase(payload)
    elif args.phase == "calibration":
        lambda_calibration_phase()
    elif args.phase == "manifest":
        make_attempt_manifest(payload)
    elif args.phase in {"control", "candidate", "screen"}:
        if args.phase == "screen":
            paired_screen_phase(payload)
        else:
            manifest = json.loads(OUT_MANIFEST_JSON.read_text())
            if args.phase == "control":
                train_arm(payload, manifest, CONTROL)
            else:
                control = json.loads((RUN_ROOT / CONTROL / "summary.json").read_text())
                train_arm(payload, manifest, CANDIDATE, control["rows"])
    elif args.phase == "endpoint":
        endpoint_phase(payload)
    elif args.phase == "decision":
        decision_phase(payload)


if __name__ == "__main__":
    main()
