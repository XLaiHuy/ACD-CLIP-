#!/usr/bin/env python3
"""H2 root-cause leakage and S2-LOCR strength audit.

This is a source-only diagnostic driver.  It reuses the exact cohorts and
retained endpoints from the committed audits, never trains a model, and does
not run Medical or MVTec inference.  The implementation deliberately keeps
all counterfactuals local to mathematically valid module boundaries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from h2_clean.precision import PrecisionPolicy


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


spill = load_module("h2_rootcause_spillover", REPO / "scripts/run_h2_boundary_spillover_audit_r1.py")
exact = load_module("h2_rootcause_exact", REPO / "scripts/run_h2_stagewise_causal_localization_audit_r1.py")

IMG = 518
NATIVE = 37
PATCH_FOOTPRINT = 14
BATCH = 8
PARENT_HEAD = "3c97eed761df4669d4143aa7b5d7b586f97d001c"
BRANCH = "research/h2-rootcause-s2-strength-audit-r1"
SAFE_ANCHOR = Path("/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth")
SAFE_SHA = "64b72dc3d1155285c826781bee4c5970bd45218e95b21675fd19d9a6b2ab54a7"
COHORT_A = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_COHORT_A.json"
COHORT_B = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_COHORT_B.json"
R1_ENDPOINT = REPO / "audit/H2_S2_LOCR_R1_ENDPOINT.json"
R1_DECISION = REPO / "results/H2_S2_LOCR_R1_BOUNDED_DECISION.json"

OUT_IDENTITY = REPO / "audit/H2_ROOTCAUSE_S2_R1_PARENT_IDENTITY.md"
OUT_GRAPH_MD = REPO / "audit/H2_ROOTCAUSE_S2_R1_GRAPH_PROVENANCE.md"
OUT_GRAPH_JSON = REPO / "audit/H2_ROOTCAUSE_S2_R1_GRAPH_PROVENANCE.json"
OUT_LEAK_CSV = REPO / "audit/H2_ROOTCAUSE_S2_R1_LEAKAGE_ONSET.csv"
OUT_LEAK_JSON = REPO / "audit/H2_ROOTCAUSE_S2_R1_LEAKAGE_ONSET.json"
OUT_MODULE_CSV = REPO / "audit/H2_ROOTCAUSE_S2_R1_MODULE_ATTRIBUTION.csv"
OUT_MODULE_JSON = REPO / "audit/H2_ROOTCAUSE_S2_R1_MODULE_ATTRIBUTION.json"
OUT_CONTEXT_CSV = REPO / "audit/H2_ROOTCAUSE_S2_R1_CONTEXT_COUNTERFACTUAL.csv"
OUT_CONTEXT_JSON = REPO / "audit/H2_ROOTCAUSE_S2_R1_CONTEXT_COUNTERFACTUAL.json"
OUT_STRENGTH_CSV = REPO / "audit/H2_ROOTCAUSE_S2_R1_STRENGTH_PATH.csv"
OUT_STRENGTH_JSON = REPO / "audit/H2_ROOTCAUSE_S2_R1_STRENGTH_PATH.json"
OUT_PARETO_MD = REPO / "audit/H2_ROOTCAUSE_S2_R1_PARETO.md"
OUT_PARETO_JSON = REPO / "audit/H2_ROOTCAUSE_S2_R1_PARETO.json"
OUT_DECISION_MD = REPO / "results/H2_ROOTCAUSE_S2_R1_DECISION.md"
OUT_DECISION_JSON = REPO / "results/H2_ROOTCAUSE_S2_R1_DECISION.json"

RUN_ROOT = Path("/workspace/h2_rootcause_s2_strength_audit_r1")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=json_default, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])
    os.replace(temporary, path)


def stats(values) -> dict:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"count": 0, **{name: None for name in ("mean", "median", "p90", "p95", "p99")}}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, .90)),
        "p95": float(np.quantile(x, .95)),
        "p99": float(np.quantile(x, .99)),
    }


def safe_delta(left, right):
    if left is None or right is None:
        return None
    return float(left - right)


def safe_ratio(left, right):
    if left is None or right is None or abs(float(right)) < 1e-12:
        return None
    return float(left / right)


def auroc(scores, labels) -> float | None:
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.uint8).reshape(-1)
    valid = np.isfinite(x)
    x, y = x[valid], y[valid]
    pos, neg = int(y.sum()), int((1 - y).sum())
    if not pos or not neg:
        return None
    ranks = rankdata(x, method="average")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def exact_binary(scores, labels) -> dict:
    return exact._external_exact_binary_metrics(((scores, labels),))


def morphology(mask: np.ndarray):
    return spill.morphology(mask)


def native_mask(mask: np.ndarray) -> np.ndarray:
    return spill.flatten_masks_native(mask).astype(bool)


def exact_occupancy(mask: np.ndarray) -> np.ndarray:
    if mask.shape[1:] != (IMG, IMG) or IMG != NATIVE * PATCH_FOOTPRINT:
        raise RuntimeError("fixed 37x37 x 14x14 footprint mapping is unavailable")
    return mask.astype(np.float32).reshape(-1, NATIVE, PATCH_FOOTPRINT, NATIVE, PATCH_FOOTPRINT).mean(axis=(2, 4))


def load_cohort(path: Path, cohort_id: str) -> list[dict]:
    data = json.loads(path.read_text())
    if data.get("count") != 96 or len(data.get("records", [])) != 96:
        raise RuntimeError(f"{cohort_id} is not the retained 96-image cohort")
    if data.get("cohort_id") != cohort_id:
        raise RuntimeError(f"cohort id mismatch for {cohort_id}")
    return data["records"]


def endpoint_identity() -> dict:
    endpoint = json.loads(R1_ENDPOINT.read_text())
    result = {}
    for label in ("control", "candidate"):
        path = Path(endpoint[label]["checkpoint"])
        observed = sha_file(path)
        committed = endpoint[label]["checkpoint_sha256"]
        if observed != committed:
            raise RuntimeError(f"retained {label} endpoint hash mismatch")
        result[label] = {
            "path": str(path), "sha256": observed, "committed_sha256": committed,
            "arm": endpoint[label].get("arm"), "epoch": torch.load(path, map_location="cpu", weights_only=False).get("epoch"),
        }
    return result


def identity_phase() -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    if head != PARENT_HEAD:
        raise RuntimeError(f"identity phase must start at parent {PARENT_HEAD}, got {head}")
    if branch != BRANCH or status:
        raise RuntimeError("identity phase requires clean new audit branch")
    if sha_file(SAFE_ANCHOR) != SAFE_SHA:
        raise RuntimeError("Safe Anchor hash mismatch")
    a = load_cohort(COHORT_A, "A")
    b = load_cohort(COHORT_B, "B")
    names_a, names_b = {x["file_name"] for x in a}, {x["file_name"] for x in b}
    if names_a & names_b:
        raise RuntimeError("retained Cohort A/B are not disjoint")
    endpoints = endpoint_identity()
    decision = json.loads(R1_DECISION.read_text())
    canonical = lambda records: sha_bytes("\n".join(
        f"{r['category']}|{r['label_type']}|{r['file_name']}|{r['image_sha256']}|{r['mask_sha256']}" for r in records
    ).encode())
    artifact = {
        "protocol_id": "H2_ROOTCAUSE_S2_STRENGTH_AUDIT_R1",
        "branch": branch, "parent_head": PARENT_HEAD, "head_at_identity": head,
        "safe_anchor": {"path": str(SAFE_ANCHOR), "sha256": SAFE_SHA},
        "retained_endpoints": endpoints,
        "cohort_a": {"path": str(COHORT_A.relative_to(REPO)), "count": len(a), "canonical_sha256": canonical(a)},
        "cohort_b": {"path": str(COHORT_B.relative_to(REPO)), "count": len(b), "canonical_sha256": canonical(b)},
        "cohort_disjoint": True,
        "parent_decision_preserved": {
            "bounded_screen": decision.get("BOUNDED_SCREEN"),
            "mechanism": decision.get("S2_LOCR_MECHANISM"),
            "interpretation": decision.get("INTERPRETATION"),
        },
        "protocol_prohibitions": {
            "medical_inference": False, "mvtec_inference": False, "target_tuning": False,
            "new_training": False, "lambda_sweep": False, "architecture_change": False,
            "r1_decision_modified": False,
        },
        "status": "PASS",
    }
    OUT_IDENTITY.write_text("\n".join([
        "# H2 Root-Cause / S2-LOCR R1 Parent Identity", "",
        f"* branch: `{branch}`",
        f"* exact parent HEAD: `{PARENT_HEAD}`",
        "* parent identity: `PASS`",
        f"* Safe Anchor: `{SAFE_ANCHOR}` ({SAFE_SHA})",
        f"* retained Cohort A: `{COHORT_A.relative_to(REPO)}` ({canonical(a)})",
        f"* retained Cohort B: `{COHORT_B.relative_to(REPO)}` ({canonical(b)})",
        f"* retained endpoints: control `{endpoints['control']['sha256']}`, candidate `{endpoints['candidate']['sha256']}`",
        "* target inference: `NO`",
        "* training / lambda sweep: `NO`",
        "* committed S2-LOCR R1 decision: preserved unchanged (`BOUNDED_SCREEN=FAIL`, `CASE_B`, `NOT_SUPPORTED`)",
        "",
        "Cohorts are reused byte-for-byte from the previous source-only stagewise audit. No new category, image, target, score, or metric-based selection is performed.",
    ]) + "\n")
    dump_json(REPO / "audit/H2_ROOTCAUSE_S2_R1_PARENT_IDENTITY.json", artifact)
    return artifact


GRAPH_NODES = [
    {"id": "clip_patch_stage1", "kind": "feature", "shape": "[B,1369,1024]", "resolution": "37x37", "score_space": "feature; projected margin for audit", "source": "model/adapter.py:238-270", "semantics": "ViT transformer output after selected frozen resblock and before Stage-1 trainable image adaptation"},
    {"id": "clip_patch_stage2", "kind": "feature", "shape": "[B,1369,1024]", "resolution": "37x37", "score_space": "feature; projected margin for audit", "source": "model/adapter.py:238-270", "semantics": "selected resblock output before Stage-2 adapter; stream already includes prior stage replacement"},
    {"id": "clip_patch_stage3", "kind": "feature", "shape": "[B,1369,1024]", "resolution": "37x37", "score_space": "feature; projected margin for audit", "source": "model/adapter.py:238-270", "semantics": "selected resblock output before Stage-3 adapter; stream already includes prior stage replacements"},
    {"id": "convlora_raw", "kind": "feature", "shape": "[B,1369,1024]", "resolution": "37x37", "score_space": "feature; projected margin for audit", "source": "model/adapter.py:270-291; model/adapter_modules.py:79-92", "semantics": "learned Conv-LoRA output before norm matching and AddWeight"},
    {"id": "post_convlora", "kind": "feature", "shape": "[B,1369,1024]", "resolution": "37x37", "score_space": "feature; projected margin for audit", "source": "model/adapter.py:270-291", "semantics": "normalized learned adapter merged with base through m_i_w"},
    {"id": "pre_seg_projection", "kind": "feature", "shape": "[B,1369,1024]", "resolution": "37x37", "score_space": "feature; projected margin for audit", "source": "model/adapter.py:293-300", "semantics": "group token representation entering seg_proj"},
    {"id": "post_seg_projection", "kind": "feature", "shape": "[B,1369,768]", "resolution": "37x37", "score_space": "normalized feature; direct text margin", "source": "model/adapter.py:300-309", "semantics": "MLP projection, LayerNorm, then L2 normalization"},
    {"id": "dfg_input", "kind": "feature", "shape": "[B,1369,768]", "resolution": "37x37", "score_space": "normalized feature; direct text margin", "source": "model/adapter.py:535-555", "semantics": "same normalized segmentation tokens entering DFG"},
    {"id": "dfg_qk_compatibility", "kind": "compatibility", "shape": "[B] broadcast to 37x37", "resolution": "global/broadcast", "score_space": "abnormal-minus-normal Q/K compatibility", "source": "model/adapter.py:555-607", "semantics": "global DFG query/key compatibility before weighted text-value fusion"},
    {"id": "ss2d_input", "kind": "feature", "shape": "[B,768] broadcast to 37x37", "resolution": "global/broadcast", "score_space": "direct text margin", "source": "model/adapter.py:544-552", "semantics": "GAP visual vector entering the SS2D residual path"},
    {"id": "ss2d_output", "kind": "feature", "shape": "[B,768] broadcast to 37x37", "resolution": "global/broadcast", "score_space": "direct text margin", "source": "model/adapter_modules.py:166-188", "semantics": "SS2D branch output before its DFG weight residual"},
    {"id": "stage_logits_pre_blur", "kind": "logit", "shape": "[B,2,37,37]", "resolution": "37x37", "score_space": "normal/abnormal logit margin", "source": "model/adapter.py:365-369", "semantics": "per-stage production logits before Gaussian blur"},
    {"id": "stage_logits_post_blur", "kind": "logit", "shape": "[B,2,37,37]", "resolution": "37x37", "score_space": "normal/abnormal logit margin", "source": "model/adapter.py:372-379", "semantics": "per-stage logits after existing Gaussian7 sigma1"},
    {"id": "stage_map_post_resize", "kind": "logit", "shape": "[B,2,518,518]", "resolution": "518x518", "score_space": "normal/abnormal logit margin", "source": "model/adapter.py:372-379", "semantics": "per-stage logits after existing bilinear resize align_corners=True"},
    {"id": "final_fusion_logits", "kind": "logit", "shape": "[B,2,518,518]", "resolution": "518x518", "score_space": "equal pre-softmax normal/abnormal margin", "source": "h2_clean/stage_fusion.py; model/adapter.py:382-385", "semantics": "equal mean of post-resize stage logits; evaluator emits sigmoid margin"},
]


def graph_phase() -> dict:
    edges = [
        ["clip_patch_stage1", "convlora_raw"], ["convlora_raw", "post_convlora"],
        ["post_convlora", "pre_seg_projection"], ["pre_seg_projection", "post_seg_projection"],
        ["post_seg_projection", "dfg_input"], ["dfg_input", "dfg_qk_compatibility"],
        ["dfg_input", "ss2d_input"], ["ss2d_input", "ss2d_output"],
        ["dfg_qk_compatibility", "stage_logits_pre_blur"], ["ss2d_output", "stage_logits_pre_blur"],
        ["stage_logits_pre_blur", "stage_logits_post_blur"], ["stage_logits_post_blur", "stage_map_post_resize"],
        ["stage_map_post_resize", "final_fusion_logits"],
        ["clip_patch_stage2", "convlora_raw"], ["clip_patch_stage3", "convlora_raw"],
    ]
    artifact = {
        "protocol_id": "H2_ROOTCAUSE_S2_STRENGTH_AUDIT_R1",
        "parent_head": PARENT_HEAD, "graph_reconstructed_before_interpretation": True,
        "nodes": GRAPH_NODES, "edges": [{"from": x, "to": y} for x, y in edges],
        "normalization": {
            "image_scale": 10.0, "feature_margin": "10*(cosine(image_feature,text_abnormal)-cosine(image_feature,text_normal))",
            "logit_margin": "abnormal_logit-normal_logit", "final_probability": "sigmoid(final_logit_margin)",
        },
        "spatial_mapping": "518 = 37*14; occupancy is exact mean over each 14x14 input footprint; native region masks retain the existing nearest-37x37 semantics",
        "production_graph": "CLIP resblocks -> Conv-LoRA normalized residual/AddWeight -> seg_proj -> LayerNorm -> L2 -> DFG Q/K and optional SS2D weight residual -> two-class stage logits -> Gaussian7/sigma1 -> bilinear518 -> equal pre-softmax stage fusion -> softmax",
        "counterfactual_constraints": "Only additive learned residuals use bypass; DFG/segmentation projection use input/output or directional attribution; no arbitrary uniform attention or identity projection is introduced.",
    }
    dump_json(OUT_GRAPH_JSON, artifact)
    lines = ["# H2 Root-Cause / S2-LOCR R1 Feature and Logit Provenance", "", "The graph below is reconstructed from the frozen implementation before any diagnostic interpretation.", "", "```text", "CLIP patch tokens (37x37, 1024)", "  -> Conv-LoRA raw output -> norm-matched AddWeight output", "  -> seg_proj (1024->768) -> LayerNorm -> L2 normalized seg tokens", "  -> DFG GAP/Q/K (+ SS2D weight residual)", "  -> stage normal/abnormal logits (37x37)", "  -> Gaussian7/sigma1 -> bilinear518 stage maps", "  -> equal pre-softmax fusion -> final normal/abnormal logits/probability", "```", "", "| Node | Shape / resolution | Space | Source / semantics |", "|---|---|---|---|"]
    for node in GRAPH_NODES:
        lines.append(f"| `{node['id']}` | `{node['shape']}` / `{node['resolution']}` | `{node['score_space']}` | `{node['source']}; {node['semantics']} |")
    lines += ["", "The feature-point audit uses the existing text embeddings and a production-compatible direct normal/anomaly margin after the stage segmentation projection. Q/K compatibility and SS2D global vectors are reported in their native diagnostic semantics and broadcast only for region accounting; they are not treated as spatial pixel logits.", "", "Exact footprint rule: each native token `(r,c)` covers input rows `[14r,14r+13]` and columns `[14c,14c+13]`."]
    OUT_GRAPH_MD.write_text("\n".join(lines) + "\n")
    return artifact


def make_model(device: torch.device):
    model = spill.make_model(device)
    model.requires_grad_(False)
    model.eval()
    return model


def load_endpoint(model, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload)
    model.image_adapter.load_state_dict(state["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(state["text_adapter"], strict=True)
    model.soft_prompt.load_state_dict(state["soft_prompt"], strict=True)
    model.dfg_beta = float(payload.get("dfg_beta_current", getattr(model, "dfg_beta", .1)))
    model.hybrid_alpha_current = float(payload.get("hybrid_alpha_current", .2))
    model.hybrid_alpha_max = float(payload.get("hybrid_alpha_max", .2))
    model.stage_fusion_weights = (1.0 / 3.0,) * 3
    model.prompt_mode = "hybrid"
    model.use_hybrid_soft_prompt = True
    model.use_soft_prompt = False
    model.eval()
    model.requires_grad_(False)
    return payload


def selected_datasets(rows: list[dict]):
    datasets = spill.get_text_and_image_dataset("VisA", IMG, "test")
    indices = {category: [] for category in spill.CLASS_NAMES["VisA"]}
    for category in spill.CLASS_NAMES["VisA"]:
        by_name = {meta["image_path"]: i for i, meta in enumerate(datasets[category].meta)}
        for row in rows:
            if row["category"] != category:
                continue
            if row["file_name"] not in by_name:
                raise RuntimeError(f"retained cohort file absent: {row['file_name']}")
            index = by_name[row["file_name"]]
            expected = int(row["label"])
            if int(datasets[category].meta[index]["label"]) != expected:
                raise RuntimeError(f"retained cohort label mismatch: {row['file_name']}")
            indices[category].append(index)
    return datasets, indices


def install_capture_hooks(model, capture: dict):
    handles = []
    for stage, block_index in enumerate((7, 15, 23)):
        def block_hook(_module, _inputs, output, stage=stage):
            capture[f"pre_lora_{stage}"] = output[0][1:]
        def raw_hook(_module, _inputs, output, stage=stage):
            capture[f"convlora_raw_{stage}"] = output
        def merged_hook(_module, _inputs, output, stage=stage):
            capture[f"post_convlora_{stage}"] = output
        def projection_hook(_module, _inputs, output, stage=stage):
            capture[f"seg_proj_raw_{stage}"] = output
        handles.append(model.image_encoder.transformer.resblocks[block_index].register_forward_hook(block_hook))
        handles.append(model.image_adapter["lora_adapters"][stage].register_forward_hook(raw_hook))
        handles.append(model.image_adapter["m_i_w"][stage].register_forward_hook(merged_hook))
        handles.append(model.image_adapter["seg_proj"][stage].register_forward_hook(projection_hook))
    return handles


def direct_margin(features: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """Existing two-class cosine margin, returned as [B,L] or [B]."""
    image = F.normalize(features.float(), dim=-1)
    if text.ndim == 2:
        text = text.unsqueeze(0).expand(image.shape[0], -1, -1)
    normal = F.normalize(text[..., 0].float(), dim=-1)
    abnormal = F.normalize(text[..., 1].float(), dim=-1)
    if image.ndim == 2:
        return 10.0 * ((image * abnormal).sum(-1) - (image * normal).sum(-1))
    return 10.0 * (torch.einsum("bld,bd->bl", image, abnormal) - torch.einsum("bld,bd->bl", image, normal))


def text_margin_tokens(features: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """Project 1024-d image-side features through the existing seg path."""
    # Hooks expose [L,B,C]; convert to the production [B,L,C] convention.
    batch_features = features.permute(1, 0, 2)
    return direct_margin(batch_features, text)


def feature_margin(model, features_lbc: torch.Tensor, text: torch.Tensor, stage: int) -> torch.Tensor:
    features = features_lbc.permute(1, 0, 2)
    projected = model.image_adapter["seg_proj"][stage](features)
    projected = model.image_adapter["seg_layer_norms"][stage](projected)
    projected = F.normalize(projected, dim=-1)
    class_text = text
    if class_text.ndim == 4:
        # [B,G,D,2] -> the stage-specific class vectors [B,D,2].
        class_text = class_text[:, stage]
    else:
        class_text = class_text[stage].unsqueeze(0).expand(projected.shape[0], -1, -1)
    return direct_margin(projected, class_text)


def dfg_instrumented(model, img_feat: torch.Tensor, group_text: torch.Tensor, stage: int, *, beta_override: float | None = None):
    """Exact DFG output plus native compatibility diagnostics."""
    v_gap = img_feat.mean(dim=1)
    v_ss2d = model.image_adapter["dfg_ss2d_branches"][stage](img_feat) if model.use_ss2d_dfg else None
    text_normal = group_text[..., 0]
    text_abnormal = group_text[..., 1]
    k_normal = model.image_adapter["vision_text_k"][stage](text_normal)
    k_abnormal = model.image_adapter["vision_text_k"][stage](text_abnormal)
    scale = (model.dfg_attn_dim ** 0.5) * model.dfg_attn_tau
    q_gap = q_ss2d = None
    if model.use_ss2d_dfg and model.dfg_ss2d_fusion == "weight_residual":
        q_gap = model.image_adapter["vision_text_q"][stage](v_gap)
        q_ss2d = model.image_adapter["vision_text_q"][stage](v_ss2d)
        if model.dfg_weight_residual_fp32:
            q_gap_a, q_ss2d_a = q_gap.float(), q_ss2d.float()
            k_normal_a, k_abnormal_a = k_normal.float(), k_abnormal.float()
        else:
            q_gap_a, q_ss2d_a = q_gap, q_ss2d
            k_normal_a, k_abnormal_a = k_normal, k_abnormal
        gap_normal = model._attention_scores(q_gap_a, k_normal_a, scale)
        gap_abnormal = model._attention_scores(q_gap_a, k_abnormal_a, scale)
        ss_normal = model._attention_scores(q_ss2d_a, k_normal_a, scale)
        ss_abnormal = model._attention_scores(q_ss2d_a, k_abnormal_a, scale)
        gap_wn, gap_wa = F.softmax(gap_normal, 1), F.softmax(gap_abnormal, 1)
        ss_wn, ss_wa = F.softmax(ss_normal, 1), F.softmax(ss_abnormal, 1)
        beta = float(model.dfg_beta if beta_override is None else beta_override)
        wn = ((1 - beta) * gap_wn + beta * ss_wn).to(dtype=text_normal.dtype)
        wa = ((1 - beta) * gap_wa + beta * ss_wa).to(dtype=text_abnormal.dtype)
        qk_margin = .5 * ((gap_abnormal - gap_normal).mean(1) + (ss_abnormal - ss_normal).mean(1))
    else:
        q = model.image_adapter["vision_text_q"][stage](v_gap)
        scores_normal = model._attention_scores(q, k_normal, scale)
        scores_abnormal = model._attention_scores(q, k_abnormal, scale)
        wn, wa = F.softmax(scores_normal, 1), F.softmax(scores_abnormal, 1)
        qk_margin = (scores_abnormal - scores_normal).mean(1)
    fused_normal = torch.einsum("bn,bnd->bd", wn, text_normal)
    fused_abnormal = torch.einsum("bn,bnd->bd", wa, text_abnormal)
    fused = torch.stack([F.normalize(fused_normal, dim=-1), F.normalize(fused_abnormal, dim=-1)], dim=-1)
    return fused, {
        "qk_margin": qk_margin,
        "ss2d_input_margin": direct_margin(v_gap, group_text[:, stage]),
        "ss2d_output_margin": direct_margin(v_ss2d, group_text[:, stage]) if v_ss2d is not None else None,
        "v_gap": v_gap,
        "v_ss2d": v_ss2d,
    }


def production_from_tokens(model, seg_tokens: list[torch.Tensor], text: torch.Tensor, *, beta_override: float | None = None):
    group_text = text.unsqueeze(1).repeat(1, seg_tokens[0].shape[0], 1, 1).permute(1, 0, 2, 3) if text.ndim == 3 else text.permute(1, 0, 2, 3)
    native, blurred, resized, diagnostics = [], [], [], []
    for stage, features in enumerate(seg_tokens):
        fused, diag = dfg_instrumented(model, features, group_text, stage, beta_override=beta_override)
        logits = torch.matmul(10.0 * features, fused).permute(0, 2, 1).view(features.shape[0], 2, NATIVE, NATIVE)
        blur = gaussian_blur2d(logits, (7, 7), (1, 1))
        resize = F.interpolate(blur, (IMG, IMG), mode="bilinear", align_corners=True)
        native.append(logits)
        blurred.append(blur)
        resized.append(resize)
        diagnostics.append(diag)
    native = torch.stack(native)
    blurred = torch.stack(blurred)
    resized = torch.stack(resized)
    return native, blurred, resized, diagnostics


def collect_graph_cohort(model, rows: list[dict], device: torch.device) -> dict:
    datasets, indices = selected_datasets(rows)
    policy = PrecisionPolicy("fp16")
    capture = {}
    handles = install_capture_hooks(model, capture)
    chunks = {key: [] for key in ("native", "mask", "labels")}
    point_chunks = {"frozen_patch": [], "convlora_raw": [], "post_convlora": [], "pre_seg_projection": [], "post_seg_projection": [], "dfg_input": [], "dfg_qk_compatibility": [], "ss2d_input": [], "ss2d_output": [], "stage_logits_pre_blur": [], "stage_logits_post_blur": [], "stage_map_post_resize": [], "final_fusion_logits": []}
    names, categories = [], []
    try:
        with torch.no_grad():
            for category in spill.CLASS_NAMES["VisA"]:
                loader = DataLoader(Subset(datasets[category], indices[category]), batch_size=BATCH, shuffle=False, num_workers=0)
                text, _, _ = spill.get_hybrid_soft_prompt_single_class_text_embedding(model, "VisA", category, device, return_kg=False)
                for batch in loader:
                    capture.clear()
                    image = batch["image"].to(device, non_blocking=True)
                    with policy.autocast(device):
                        seg_tokens, _ = model(image)
                        native, blurred, resized, diagnostics = production_from_tokens(model, seg_tokens, text)
                        # Exact production parity check at the first batch only.
                        if not chunks["native"]:
                            group_text = text.unsqueeze(1).repeat(1, image.shape[0], 1, 1).permute(1, 0, 2, 3)
                            for stage in range(3):
                                ref = model._vision_text_attention_fusion(seg_tokens[stage], group_text, stage)
                                ours = torch.matmul(10.0 * seg_tokens[stage], ref).permute(0, 2, 1).view(image.shape[0], 2, NATIVE, NATIVE)
                                if not torch.allclose(ours, native[stage], atol=2e-5, rtol=2e-5):
                                    raise RuntimeError("instrumented DFG does not reproduce production native logits")
                    mask = (batch["mask"][:, 0].numpy() > .5).astype(np.uint8)
                    chunks["mask"].append(mask)
                    chunks["labels"].append(batch["label"].numpy().astype(np.uint8))
                    native_margin = (native[:, :, 1] - native[:, :, 0]).permute(1, 0, 2, 3)
                    blur_margin = (blurred[:, :, 1] - blurred[:, :, 0]).permute(1, 0, 2, 3)
                    resize_margin = (resized[:, :, 1] - resized[:, :, 0]).permute(1, 0, 2, 3)
                    chunks["native"].append(native_margin.float().cpu().numpy())
                    stage_values = {key: [] for key in point_chunks if key != "final_fusion_logits"}
                    for stage in range(3):
                        stage_values["frozen_patch"].append(feature_margin(model, capture[f"pre_lora_{stage}"], text, stage).view(-1, NATIVE, NATIVE).float().cpu().numpy())
                        stage_values["convlora_raw"].append(feature_margin(model, capture[f"convlora_raw_{stage}"], text, stage).view(-1, NATIVE, NATIVE).float().cpu().numpy())
                        stage_values["post_convlora"].append(feature_margin(model, capture[f"post_convlora_{stage}"], text, stage).view(-1, NATIVE, NATIVE).float().cpu().numpy())
                        stage_values["pre_seg_projection"].append(stage_values["post_convlora"][-1])
                        stage_values["post_seg_projection"].append(direct_margin(seg_tokens[stage], text[stage]).view(-1, NATIVE, NATIVE).float().cpu().numpy())
                        stage_values["dfg_input"].append(stage_values["post_seg_projection"][-1])
                        stage_values["dfg_qk_compatibility"].append(diagnostics[stage]["qk_margin"].view(-1, 1, 1).expand(-1, NATIVE, NATIVE).float().cpu().numpy())
                        stage_values["ss2d_input"].append(diagnostics[stage]["ss2d_input_margin"].view(-1, 1, 1).expand(-1, NATIVE, NATIVE).float().cpu().numpy())
                        output_margin = diagnostics[stage]["ss2d_output_margin"]
                        stage_values["ss2d_output"].append((output_margin if output_margin is not None else torch.zeros_like(diagnostics[stage]["ss2d_input_margin"])).view(-1, 1, 1).expand(-1, NATIVE, NATIVE).float().cpu().numpy())
                        stage_values["stage_logits_pre_blur"].append(native_margin[:, stage].float().cpu().numpy())
                        stage_values["stage_logits_post_blur"].append(blur_margin[:, stage].float().cpu().numpy())
                        stage_values["stage_map_post_resize"].append(resize_margin[:, stage].float().cpu().numpy())
                    for key, values in stage_values.items():
                        point_chunks[key].append(np.stack(values, axis=1))
                    point_chunks["final_fusion_logits"].append(resize_margin.mean(dim=1).float().cpu().numpy())
                    names.extend(batch["file_name"])
                    categories.extend([category] * len(batch["file_name"]))
                    del image, seg_tokens, native, blurred, resized
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()
    expected = [row["file_name"] for row in rows]
    if names != expected:
        raise RuntimeError("cohort evaluator order mismatch")
    return {
        "points": {key: np.concatenate(value, axis=0) for key, value in point_chunks.items()},
        "native_margin": np.concatenate(chunks["native"], axis=0),
        "mask": np.concatenate(chunks["mask"], axis=0), "labels": np.concatenate(chunks["labels"], axis=0),
        "names": np.asarray(names, dtype=object), "categories": np.asarray(categories, dtype=object),
    }


def point_region_arrays(score: np.ndarray, mask: np.ndarray, resolution: str):
    """Return exact fixed-semantic regions for a [N,H,W] point map."""
    if resolution == "native":
        occ = exact_occupancy(mask)
        regions = []
        for current_mask, current_occ in zip(mask, occ):
            nmask = native_mask(current_mask[None])[0]
            _, _, near, far = morphology(nmask)
            regions.append({
                "zero_near": (current_occ == 0) & near,
                "zero_far": (current_occ == 0) & far,
                "partial": (current_occ > 0) & (current_occ < 1),
                "full": current_occ == 1,
                "positive": nmask,
            })
        return regions
    if resolution == "full":
        occ_native = exact_occupancy(mask)
        occ = np.repeat(np.repeat(occ_native, PATCH_FOOTPRINT, axis=1), PATCH_FOOTPRINT, axis=2)
        regions = []
        for current_mask, current_occ in zip(mask, occ):
            _, _, near, far = morphology(current_mask)
            regions.append({
                "zero_near": (current_occ == 0) & near,
                "zero_far": (current_occ == 0) & far,
                "partial": (current_occ > 0) & (current_occ < 1),
                "full": current_occ == 1,
                "positive": current_mask.astype(bool),
            })
        return regions
    if resolution == "global":
        regions = []
        for current_mask in mask:
            nmask = native_mask(current_mask[None])[0]
            _, _, near, far = morphology(nmask)
            regions.append({
                "zero_near": near, "zero_far": far, "partial": np.zeros_like(nmask, bool),
                "full": np.zeros_like(nmask, bool), "positive": nmask,
            })
        return regions
    raise ValueError(resolution)


def point_resolution(point: str) -> str:
    if point in ("stage_map_post_resize", "final_fusion_logits"):
        return "full"
    if point in ("dfg_qk_compatibility", "ss2d_input", "ss2d_output"):
        return "global"
    return "native"


def summarize_point(score: np.ndarray, mask: np.ndarray, point: str) -> dict:
    resolution = point_resolution(point)
    regions = point_region_arrays(score, mask, resolution)
    result = {"point": point, "resolution": resolution, "regions": {}}
    for name in ("zero_near", "zero_far", "partial", "full"):
        values = [current[name] for current in regions]
        flat = np.concatenate([score[i][region].reshape(-1) for i, region in enumerate(values) if region.any()]) if any(region.any() for region in values) else np.empty(0)
        result["regions"][name] = stats(flat)
    near_values, far_values, ap_values, ap_labels = [], [], [], []
    for i, current in enumerate(regions):
        near = score[i][current["zero_near"]]
        far = score[i][current["zero_far"]]
        partial = score[i][current["partial"]]
        positive = score[i][current["positive"]]
        if near.size:
            near_values.append(near)
            ap_values.append(near); ap_labels.append(np.zeros(near.size, dtype=np.uint8))
        if far.size:
            far_values.append(far)
            ap_values.append(far); ap_labels.append(np.ones(far.size, dtype=np.uint8))
        if positive.size:
            ap_values.append(positive); ap_labels.append(np.ones(positive.size, dtype=np.uint8))
        if partial.size:
            ap_values.append(partial); ap_labels.append(np.ones(partial.size, dtype=np.uint8))
    result["zero_near_vs_zero_far_auroc"] = auroc(
        np.concatenate(near_values + far_values) if near_values and far_values else np.empty(0),
        np.concatenate([np.ones(sum(x.size for x in near_values), dtype=np.uint8), np.zeros(sum(x.size for x in far_values), dtype=np.uint8)]) if near_values and far_values else np.empty(0, dtype=np.uint8),
    )
    # The combined anomaly/partial-vs-near comparison is intentionally
    # unweighted and uses all retained pixels; no score threshold is tuned.
    result["anomaly_partial_vs_zero_near_auroc"] = auroc(
        np.concatenate(ap_values) if ap_values else np.empty(0),
        np.concatenate(ap_labels) if ap_labels else np.empty(0, dtype=np.uint8),
    )
    return result


LEAKAGE_POINTS = (
    ("stage1_frozen_patch", "frozen_patch", 0),
    ("stage1_convlora_raw", "convlora_raw", 0),
    ("stage1_post_convlora", "post_convlora", 0),
    ("stage1_post_seg_projection", "post_seg_projection", 0),
    ("stage1_dfg_input", "dfg_input", 0),
    ("stage1_dfg_qk_compatibility", "dfg_qk_compatibility", 0),
    ("stage1_ss2d_input", "ss2d_input", 0),
    ("stage1_ss2d_output", "ss2d_output", 0),
    ("stage1_logits_pre_blur", "stage_logits_pre_blur", 0),
    ("stage1_logits_post_blur", "stage_logits_post_blur", 0),
    ("stage1_map_post_resize", "stage_map_post_resize", 0),
    ("stage2_frozen_patch", "frozen_patch", 1),
    ("stage2_convlora_raw", "convlora_raw", 1),
    ("stage2_post_convlora", "post_convlora", 1),
    ("stage2_post_seg_projection", "post_seg_projection", 1),
    ("stage2_dfg_input", "dfg_input", 1),
    ("stage2_dfg_qk_compatibility", "dfg_qk_compatibility", 1),
    ("stage2_ss2d_input", "ss2d_input", 1),
    ("stage2_ss2d_output", "ss2d_output", 1),
    ("stage2_logits_pre_blur", "stage_logits_pre_blur", 1),
    ("stage2_logits_post_blur", "stage_logits_post_blur", 1),
    ("stage2_map_post_resize", "stage_map_post_resize", 1),
    ("stage3_frozen_patch", "frozen_patch", 2),
    ("stage3_convlora_raw", "convlora_raw", 2),
    ("stage3_post_convlora", "post_convlora", 2),
    ("stage3_post_seg_projection", "post_seg_projection", 2),
    ("stage3_dfg_input", "dfg_input", 2),
    ("stage3_dfg_qk_compatibility", "dfg_qk_compatibility", 2),
    ("stage3_ss2d_input", "ss2d_input", 2),
    ("stage3_ss2d_output", "ss2d_output", 2),
    ("stage3_logits_pre_blur", "stage_logits_pre_blur", 2),
    ("stage3_logits_post_blur", "stage_logits_post_blur", 2),
    ("stage3_map_post_resize", "stage_map_post_resize", 2),
    ("final_fusion_logits", "final_fusion_logits", None),
)


def leakage_phase() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact retained endpoint inference")
    device = torch.device("cuda:0")
    model = make_model(device)
    load_endpoint(model, SAFE_ANCHOR)
    cohort_data = {}
    for cohort_id, path in (("A", COHORT_A), ("B", COHORT_B)):
        cohort_data[cohort_id] = collect_graph_cohort(model, load_cohort(path, cohort_id), device)
    rows = []
    point_records = {}
    for display, key, stage in LEAKAGE_POINTS:
        for cohort_id in ("A", "B"):
            data = cohort_data[cohort_id]
            scores = data["points"][key] if stage is None else data["points"][key][:, stage]
            record = summarize_point(scores, data["mask"], display)
            record.update({"cohort": cohort_id, "stage": None if stage is None else stage + 1})
            point_records.setdefault(display, {})[cohort_id] = record
            row = {"point": display, "cohort": cohort_id, "stage": record["stage"], "resolution": record["resolution"]}
            for region in ("zero_near", "zero_far", "partial", "full"):
                for stat_name, value in record["regions"][region].items():
                    row[f"{region}_{stat_name}"] = value
            row["zero_near_vs_zero_far_auroc"] = record["zero_near_vs_zero_far_auroc"]
            row["anomaly_partial_vs_zero_near_auroc"] = record["anomaly_partial_vs_zero_near_auroc"]
            rows.append(row)
    onset = None
    onset_checks = []
    for display, _, _ in LEAKAGE_POINTS:
        a, b = point_records[display]["A"], point_records[display]["B"]
        check = bool(
            a["regions"]["zero_near"]["mean"] is not None and b["regions"]["zero_near"]["mean"] is not None
            and a["regions"]["zero_far"]["mean"] is not None and b["regions"]["zero_far"]["mean"] is not None
            and a["regions"]["zero_near"]["mean"] > a["regions"]["zero_far"]["mean"]
            and b["regions"]["zero_near"]["mean"] > b["regions"]["zero_far"]["mean"]
            and a["zero_near_vs_zero_far_auroc"] is not None and b["zero_near_vs_zero_far_auroc"] is not None
            and a["zero_near_vs_zero_far_auroc"] > .5 and b["zero_near_vs_zero_far_auroc"] > .5
        )
        onset_checks.append({"point": display, "both_cohorts_directionally_separated": check})
        if onset is None and check:
            onset = display
    artifact = {
        "protocol_id": "H2_ROOTCAUSE_S2_STRENGTH_AUDIT_R1", "checkpoint": str(SAFE_ANCHOR), "checkpoint_sha256": SAFE_SHA,
        "cohorts": {"A": {"count": 96, "source": str(COHORT_A.relative_to(REPO))}, "B": {"count": 96, "source": str(COHORT_B.relative_to(REPO))}},
        "score_definition": "feature points use existing two-class normal/anomaly text-margin after the stage's existing seg projection; logits use abnormal-normal; global Q/K and SS2D margins retain native diagnostic semantics and are broadcast only for region accounting",
        "rows": rows, "leakage_onset_point": onset or "NOT_ESTABLISHED",
        "onset_rule": "earliest graph point with zero-near mean > zero-far mean and zero-near-vs-zero-far AUROC > 0.5 on both retained cohorts; no tuned numerical threshold",
        "onset_checks": onset_checks,
    }
    write_csv(OUT_LEAK_CSV, rows)
    dump_json(OUT_LEAK_JSON, artifact)
    del model
    torch.cuda.empty_cache()
    return artifact
