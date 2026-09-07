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
from scipy import ndimage
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
    if subprocess.run(["git", "merge-base", "--is-ancestor", PARENT_HEAD, head], cwd=REPO).returncode != 0:
        raise RuntimeError(f"identity phase requires parent {PARENT_HEAD} as an ancestor, got {head}")
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
    # Hooks retain the historical AMP dtype; the diagnostic projection is
    # evaluated in the same FP32 island used by the production adapter path.
    features = features_lbc.permute(1, 0, 2).float()
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
    if point == "final_fusion_logits" or point.endswith("_map_post_resize"):
        return "full"
    if point.endswith(("_dfg_qk_compatibility", "_ss2d_input", "_ss2d_output")):
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


def seg_from_pre_lora(model, tokens_lbc: torch.Tensor, stage: int) -> torch.Tensor:
    """Replay only the valid Stage-X local adapter path from its input."""
    with torch.autocast(device_type=tokens_lbc.device.type, enabled=False):
        t = tokens_lbc.float()
        adapter = model.image_adapter["lora_adapters"][stage]
        delta_out = adapter(t)
        delta_out = delta_out * t.norm(dim=-1, keepdim=True) / delta_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        merged = model.image_adapter["m_i_w"][stage](t, delta_out)
        group = merged.permute(1, 0, 2)
        projected = model.image_adapter["seg_proj"][stage](group)
        projected = model.image_adapter["seg_layer_norms"][stage](projected)
        return F.normalize(projected, dim=-1)


def stage_from_seg(model, features: torch.Tensor, text: torch.Tensor, stage: int, *, beta_override: float | None = None):
    group_text = text.unsqueeze(1).repeat(1, features.shape[0], 1, 1).permute(1, 0, 2, 3) if text.ndim == 3 else text.permute(1, 0, 2, 3)
    fused, diagnostic = dfg_instrumented(model, features, group_text, stage, beta_override=beta_override)
    logits = torch.matmul(10.0 * features, fused).permute(0, 2, 1).view(features.shape[0], 2, NATIVE, NATIVE)
    blurred = gaussian_blur2d(logits, (7, 7), (1, 1))
    resized = F.interpolate(blurred, (IMG, IMG), mode="bilinear", align_corners=True)
    return logits, blurred, resized, diagnostic


def region_flat(score: np.ndarray, mask: np.ndarray, resolution: str) -> dict[str, np.ndarray]:
    regions = point_region_arrays(score, mask, resolution)
    out = {name: [] for name in ("zero_near", "zero_far", "partial", "full")}
    for index, current in enumerate(regions):
        for name in out:
            if current[name].any():
                out[name].append(score[index][current[name]].reshape(-1))
    return {name: np.concatenate(values) if values else np.empty(0, dtype=np.float32) for name, values in out.items()}


def module_row(cohort: str, stage: int, module: str, current: np.ndarray, variant: np.ndarray, mask: np.ndarray, resolution: str, comparison: str, valid_counterfactual: bool) -> dict:
    current_regions = region_flat(current, mask, resolution)
    variant_regions = region_flat(variant, mask, resolution)
    row = {"cohort": cohort, "stage": stage + 1, "module": module, "resolution": resolution, "comparison": comparison, "valid_counterfactual": valid_counterfactual}
    for name in ("zero_near", "zero_far", "partial", "full"):
        cs, vs = stats(current_regions[name]), stats(variant_regions[name])
        for stat_name in ("mean", "median", "p95", "p99"):
            row[f"current_{name}_{stat_name}"] = cs[stat_name]
            row[f"variant_{name}_{stat_name}"] = vs[stat_name]
            row[f"delta_{name}_{stat_name}"] = safe_delta(cs[stat_name], vs[stat_name])
        row[f"amplification_ratio_{name}"] = None
    row["amplification_ratio_zero_near_relative_to_zero_far"] = safe_ratio(row["delta_zero_near_mean"], row["delta_zero_far_mean"])
    row["near_minus_far_delta"] = safe_delta(row["delta_zero_near_mean"], row["delta_zero_far_mean"])
    row["near_selective_abs"] = None if row["delta_zero_near_mean"] is None or row["delta_zero_far_mean"] is None else abs(row["delta_zero_near_mean"]) > abs(row["delta_zero_far_mean"])
    return row


def collect_module_cohort(model, rows: list[dict], device: torch.device) -> dict:
    datasets, indices = selected_datasets(rows)
    policy = PrecisionPolicy("fp16")
    capture = {}
    handles = install_capture_hooks(model, capture)
    actual_native, actual_resize, actual_final, masks = [], [], [], []
    variants = {name: [] for name in ("convlora", "seg_projection", "dfg_qk", "ss2d", "no_blur")}
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
                        native, blurred, resized, _ = production_from_tokens(model, seg_tokens, text)
                        no_lora_native, no_lora_resize = [], []
                        no_ss2d_native, no_ss2d_resize = [], []
                        raw_projection = []
                        for stage in range(3):
                            base_tokens = capture[f"pre_lora_{stage}"]
                            # Conv-LoRA delta removed means the frozen/base
                            # operation t itself, never an identity 1024->768 projection.
                            base_group = base_tokens.permute(1, 0, 2).float()
                            base_projected = model.image_adapter["seg_proj"][stage](base_group)
                            base_projected = model.image_adapter["seg_layer_norms"][stage](base_projected)
                            base_seg = F.normalize(base_projected, dim=-1)
                            nl, nb, nr, _ = stage_from_seg(model, base_seg, text, stage)
                            no_lora_native.append(nl)
                            no_lora_resize.append(nr)
                            sl, sb, sr, _ = stage_from_seg(model, seg_tokens[stage], text, stage, beta_override=0.0)
                            no_ss2d_native.append(sl)
                            no_ss2d_resize.append(sr)
                            raw_projection.append(direct_margin(capture[f"seg_proj_raw_{stage}"].permute(1, 0, 2), text[stage]))
                    actual_native.append(((native[:, :, 1] - native[:, :, 0]).permute(1, 0, 2, 3)).float().cpu().numpy())
                    actual_resize.append(((resized[:, :, 1] - resized[:, :, 0]).permute(1, 0, 2, 3)).float().cpu().numpy())
                    actual_final.append(((resized[:, :, 1] - resized[:, :, 0]).mean(dim=0)).float().cpu().numpy())
                    masks.append((batch["mask"][:, 0].numpy() > .5).astype(np.uint8))
                    variants["convlora"].append(np.stack([x[:, 1].sub(x[:, 0]).float().cpu().numpy() for x in no_lora_native], axis=1))
                    variants["seg_projection"].append(np.stack([x.float().cpu().numpy().reshape(-1, NATIVE, NATIVE) for x in raw_projection], axis=1))
                    variants["dfg_qk"].append(np.stack([direct_margin(seg_tokens[s], text[s]).float().cpu().numpy().reshape(-1, NATIVE, NATIVE) for s in range(3)], axis=1))
                    variants["ss2d"].append(np.stack([x[:, 1].sub(x[:, 0]).float().cpu().numpy() for x in no_ss2d_native], axis=1))
                    variants["no_blur"].append(np.stack([F.interpolate(native[s], (IMG, IMG), mode="bilinear", align_corners=True)[:, 1].sub(F.interpolate(native[s], (IMG, IMG), mode="bilinear", align_corners=True)[:, 0]).float().cpu().numpy() for s in range(3)], axis=1))
                    del image, seg_tokens, native, blurred, resized
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()
    return {
        "native": np.concatenate(actual_native, axis=0), "resize": np.concatenate(actual_resize, axis=0),
        "final": np.concatenate(actual_final, axis=0), "mask": np.concatenate(masks, axis=0),
        "variants": {key: np.concatenate(value, axis=0) for key, value in variants.items()},
    }


def module_phase() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for module attribution")
    device = torch.device("cuda:0")
    model = make_model(device)
    load_endpoint(model, SAFE_ANCHOR)
    data = {cohort: collect_module_cohort(model, load_cohort(path, cohort), device) for cohort, path in (("A", COHORT_A), ("B", COHORT_B))}
    rows = []
    for cohort, current in data.items():
        mask = current["mask"]
        for stage in range(3):
            base_native = current["native"][:, stage]
            rows.append(module_row(cohort, stage, "convlora", base_native, current["variants"]["convlora"][:, stage], mask, "native", "current stage logits minus valid Conv-LoRA residual-removed local replay", True))
            rows.append(module_row(cohort, stage, "seg_projection", current["native"][:, stage], current["variants"]["seg_projection"][:, stage], mask, "native", "post-seg normalized margin versus pre-LayerNorm raw projection; representation attribution only, no identity bypass", False))
            rows.append(module_row(cohort, stage, "dfg_qk", base_native, current["variants"]["dfg_qk"][:, stage], mask, "native", "stage native margin versus exact DFG-input direct text margin", False))
            rows.append(module_row(cohort, stage, "ss2d", base_native, current["variants"]["ss2d"][:, stage], mask, "native", "current weight-residual DFG versus exact beta=0 GAP-only component", True))
            base_resize = current["resize"][:, stage]
            rows.append(module_row(cohort, stage, "interpolation", base_resize, current["variants"]["no_blur"][:, stage], mask, "full", "existing Gaussian7+resize versus same resize without blur", False))
            final = current["final"]
            rows.append(module_row(cohort, stage, "stage_fusion", final, base_resize, mask, "full", "equal pre-softmax final fusion versus the corresponding individual resized stage", False))
    statuses = {}
    for module in ("convlora", "seg_projection", "dfg_qk", "ss2d", "interpolation", "stage_fusion"):
        subset = [row for row in rows if row["module"] == module]
        supported_by_cohort = {}
        for cohort in ("A", "B"):
            cohort_rows = [row for row in subset if row["cohort"] == cohort]
            supported_by_cohort[cohort] = any(row["near_selective_abs"] for row in cohort_rows)
        statuses[module] = "SUPPORTED" if all(supported_by_cohort.values()) else "MIXED" if any(supported_by_cohort.values()) else "NOT_SUPPORTED"
    candidate_modules = ("convlora", "dfg_qk", "ss2d")
    candidate_rows = [row for row in rows if row["module"] in candidate_modules and row["near_minus_far_delta"] is not None]
    by_module = {module: float(np.mean([abs(row["near_minus_far_delta"]) for row in candidate_rows if row["module"] == module])) for module in candidate_modules}
    supported_modules = [module for module in candidate_modules if statuses[module] == "SUPPORTED"]
    primary = max(supported_modules, key=lambda module: by_module[module]) if supported_modules else "NONE"
    secondary_candidates = [module for module in candidate_modules if module != primary and module in by_module]
    secondary = max(secondary_candidates, key=lambda module: by_module[module]) if secondary_candidates else "NONE"
    artifact = {
        "protocol_id": "H2_ROOTCAUSE_S2_STRENGTH_AUDIT_R1", "rows": rows,
        "module_status": statuses, "module_selective_score": by_module,
        "primary_amplifier": primary, "secondary_amplifier": secondary,
        "amplification_definition": "current minus valid local variant for additive residuals; seg projection and DFG use pre/post representation attribution; amplification ratio is delta ZERO_NEAR relative to delta ZERO_FAR",
        "interpretation": "The status is descriptive. A supported module is not by itself a causal proof because Stage-2/3 transformer streams carry prior-stage effects.",
    }
    write_csv(OUT_MODULE_CSV, rows)
    dump_json(OUT_MODULE_JSON, artifact)
    del model
    torch.cuda.empty_cache()
    return artifact


def merged_from_pre_lora(model, tokens_lbc: torch.Tensor, stage: int) -> torch.Tensor:
    with torch.autocast(device_type=tokens_lbc.device.type, enabled=False):
        t = tokens_lbc.float()
        adapter = model.image_adapter["lora_adapters"][stage]
        delta_out = adapter(t)
        delta_out = delta_out * t.norm(dim=-1, keepdim=True) / delta_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return model.image_adapter["m_i_w"][stage](t, delta_out)


def context_plan(mask: np.ndarray):
    occ = exact_occupancy(mask[None])[0]
    nmask = native_mask(mask[None])[0]
    _, _, near, far = morphology(nmask)
    target = (occ == 0) & near
    anomaly = occ > 0
    target_dilated = ndimage.binary_dilation(target, structure=np.ones((3, 3), dtype=bool))
    context = anomaly & target_dilated
    context_indices = np.flatnonzero(context.reshape(-1))
    far_indices = np.flatnonzero(((occ == 0) & far).reshape(-1))
    n = min(int(context_indices.size), int(far_indices.size))
    return occ, target, context_indices[:n], far_indices, n


CONTEXT_POINTS = (
    "pre_lora",
    "convlora_output",
    "post_seg_projection",
    "dfg_qk_compatibility",
    "stage_logits_pre_blur",
    "stage_logits_post_blur",
    "stage_map_post_resize",
)


def context_point_values(model, tokens_lbc: torch.Tensor, text: torch.Tensor, stage: int, *, beta_override: float | None = None):
    merged = merged_from_pre_lora(model, tokens_lbc, stage)
    group = merged.permute(1, 0, 2)
    raw_seg = model.image_adapter["seg_proj"][stage](group)
    norm_seg = F.normalize(model.image_adapter["seg_layer_norms"][stage](raw_seg), dim=-1)
    native, blurred, resized, diag = stage_from_seg(model, norm_seg, text, stage, beta_override=beta_override)
    group_text = text.unsqueeze(1).repeat(1, norm_seg.shape[0], 1, 1).permute(1, 0, 2, 3) if text.ndim == 3 else text.permute(1, 0, 2, 3)
    return {
        "pre_lora": feature_margin(model, tokens_lbc, text, stage).view(-1, NATIVE, NATIVE),
        "convlora_output": feature_margin(model, merged, text, stage).view(-1, NATIVE, NATIVE),
        "post_seg_projection": direct_margin(norm_seg, text[stage]).view(-1, NATIVE, NATIVE),
        "dfg_qk_compatibility": diag["qk_margin"].view(-1, 1, 1).expand(-1, NATIVE, NATIVE),
        "stage_logits_pre_blur": (native[:, 1] - native[:, 0]),
        "stage_logits_post_blur": (blurred[:, 1] - blurred[:, 0]),
        "stage_map_post_resize": (resized[:, 1] - resized[:, 0]),
    }


def target_value(score: torch.Tensor | np.ndarray, target_native: np.ndarray, resolution: str) -> float | None:
    array = score.detach().float().cpu().numpy() if torch.is_tensor(score) else np.asarray(score, dtype=np.float32)
    if resolution == "full":
        target = np.repeat(np.repeat(target_native, PATCH_FOOTPRINT, axis=0), PATCH_FOOTPRINT, axis=1)
    else:
        target = target_native
    values = array[target]
    return float(values.mean()) if values.size else None


def context_phase() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for context counterfactual")
    device = torch.device("cuda:0")
    model = make_model(device)
    load_endpoint(model, SAFE_ANCHOR)
    details = []
    for cohort_id, path in (("A", COHORT_A), ("B", COHORT_B)):
        rows = load_cohort(path, cohort_id)
        datasets, indices = selected_datasets(rows)
        policy = PrecisionPolicy("fp16")
        capture = {}
        handles = install_capture_hooks(model, capture)
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
                            _, _, _, base_diag = production_from_tokens(model, seg_tokens, text)
                            base_points = []
                            for stage in range(3):
                                base_points.append({
                                    "pre_lora": feature_margin(model, capture[f"pre_lora_{stage}"], text, stage).view(-1, NATIVE, NATIVE),
                                    "convlora_output": feature_margin(model, capture[f"post_convlora_{stage}"], text, stage).view(-1, NATIVE, NATIVE),
                                    "post_seg_projection": direct_margin(seg_tokens[stage], text[stage]).view(-1, NATIVE, NATIVE),
                                    "dfg_qk_compatibility": base_diag[stage]["qk_margin"].view(-1, 1, 1).expand(-1, NATIVE, NATIVE),
                                })
                                native, blurred, resized, _ = stage_from_seg(model, seg_tokens[stage], text, stage)
                                base_points[-1].update({
                                    "stage_logits_pre_blur": native[:, 1] - native[:, 0],
                                    "stage_logits_post_blur": blurred[:, 1] - blurred[:, 0],
                                    "stage_map_post_resize": resized[:, 1] - resized[:, 0],
                                })
                            for image_index, current_mask in enumerate((batch["mask"][:, 0].numpy() > .5).astype(np.uint8)):
                                if not int(batch["label"][image_index]):
                                    continue
                                occ, target, context_indices, far_indices, count = context_plan(current_mask)
                                if count == 0:
                                    continue
                                target_values = {"cohort": cohort_id, "file_name": str(batch["file_name"][image_index]), "stage": None, "target_count": int(target.sum()), "context_count": int(count)}
                                # Each stage is replayed from the exact pre-LoRA
                                # tensor. Replacements are canonical spatial
                                # tokens, never prediction-score selected.
                                for stage in range(3):
                                    source = capture[f"pre_lora_{stage}"][:, image_index:image_index + 1, :].clone()
                                    candidate = source.clone()
                                    control = source.clone()
                                    flat_far = far_indices.tolist()
                                    flat_context = context_indices.tolist()
                                    # Candidate: anomaly-containing neighbours
                                    # receive same-image zero-far features.
                                    for j, flat_index in enumerate(flat_context):
                                        src = flat_far[j % len(flat_far)]
                                        candidate[flat_index, 0] = source[src, 0]
                                    # Control A: the same number of already-far
                                    # tokens are replaced by a cyclic far token.
                                    for j, flat_index in enumerate(flat_far[:count]):
                                        src = flat_far[(j + 1) % len(flat_far)]
                                        control[flat_index, 0] = source[src, 0]
                                    candidate_points = context_point_values(model, candidate, text, stage)
                                    control_points = context_point_values(model, control, text, stage)
                                    stage_base = {key: value[image_index:image_index + 1] for key, value in base_points[stage].items()}
                                    for point in CONTEXT_POINTS:
                                        resolution = "full" if point == "stage_map_post_resize" else "native"
                                        baseline_value = target_value(stage_base[point][0], target, resolution)
                                        candidate_value = target_value(candidate_points[point][0], target, resolution)
                                        control_value = target_value(control_points[point][0], target, resolution)
                                        details.append({
                                            "cohort": cohort_id, "file_name": str(batch["file_name"][image_index]), "stage": stage + 1, "point": point,
                                            "target_count": int(target.sum()), "context_count": int(count), "resolution": resolution,
                                            "baseline": baseline_value, "control_a": control_value, "candidate": candidate_value,
                                            "candidate_minus_control": None if candidate_value is None or control_value is None else candidate_value - control_value,
                                            "candidate_minus_baseline": None if candidate_value is None or baseline_value is None else candidate_value - baseline_value,
                                            "control_minus_baseline": None if control_value is None or baseline_value is None else control_value - baseline_value,
                                        })
                        del image, seg_tokens
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
        finally:
            for handle in handles:
                handle.remove()
    aggregate = []
    for cohort_id in ("A", "B"):
        for stage in (1, 2, 3):
            for point in CONTEXT_POINTS:
                subset = [row for row in details if row["cohort"] == cohort_id and row["stage"] == stage and row["point"] == point]
                row = {"cohort": cohort_id, "stage": stage, "point": point, "resolution": subset[0]["resolution"] if subset else None, "image_count": len(subset), "target_count_total": int(sum(x["target_count"] for x in subset)), "context_count_total": int(sum(x["context_count"] for x in subset))}
                for field in ("baseline", "control_a", "candidate", "candidate_minus_control", "candidate_minus_baseline", "control_minus_baseline"):
                    values = np.asarray([x[field] for x in subset if x[field] is not None], dtype=np.float64)
                    row[f"{field}_mean"] = float(values.mean()) if values.size else None
                    row[f"{field}_median"] = float(np.median(values)) if values.size else None
                    row[f"{field}_p95"] = float(np.quantile(values, .95)) if values.size else None
                aggregate.append(row)
    first_sensitive = "NOT_ESTABLISHED"
    for point in CONTEXT_POINTS:
        a = [x["candidate_minus_control_mean"] for x in aggregate if x["cohort"] == "A" and x["point"] == point and x["candidate_minus_control_mean"] is not None]
        b = [x["candidate_minus_control_mean"] for x in aggregate if x["cohort"] == "B" and x["point"] == point and x["candidate_minus_control_mean"] is not None]
        if a and b and np.mean(a) < 0 and np.mean(b) < 0:
            first_sensitive = point
            break
    support = "YES" if first_sensitive != "NOT_ESTABLISHED" else "NO"
    labels = {"convlora_output": "CONVLORA_AMPLIFIES_CONTEXT", "post_seg_projection": "SEG_PROJECTION_AMPLIFIES_CONTEXT", "dfg_qk_compatibility": "DFG_AMPLIFIES_CONTEXT", "stage_logits_pre_blur": "DFG_AMPLIFIES_CONTEXT", "stage_logits_post_blur": "MIXED", "stage_map_post_resize": "MIXED"}
    artifact = {
        "protocol_id": "H2_ROOTCAUSE_S2_STRENGTH_AUDIT_R1", "details": details, "rows": aggregate,
        "intervention": "For each anomalous ZERO_NEAR target, keep target token unchanged and replace canonical neighbouring occupancy>0 tokens with same-image canonical ZERO_FAR tokens; Control A replaces the same number of ZERO_FAR tokens cyclically; Control B is the untouched baseline.",
        "target_selection": "exact occupancy==0 and existing native near mask; no prediction-score selection",
        "contextual_propagation_support": support,
        "first_context_sensitive_module": first_sensitive,
        "interpretation": labels.get(first_sensitive, "NOT_ESTABLISHED") if support == "YES" else "NOT_ESTABLISHED",
    }
    write_csv(OUT_CONTEXT_CSV, aggregate)
    dump_json(OUT_CONTEXT_JSON, artifact)
    del model
    torch.cuda.empty_cache()
    return artifact


def endpoint_payloads():
    endpoint = json.loads(R1_ENDPOINT.read_text())
    control_path = Path(endpoint["control"]["checkpoint"])
    candidate_path = Path(endpoint["candidate"]["checkpoint"])
    control = torch.load(control_path, map_location="cpu", weights_only=False)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    return control, candidate, control_path, candidate_path


def parameter_signature(payload: dict) -> dict:
    state = payload["model_state"]
    return {
        module: {key: (tuple(value.shape), str(value.dtype)) for key, value in sorted(state[module].items())}
        for module in ("image_adapter", "text_adapter", "soft_prompt")
    }


def load_interpolated(model, control: dict, candidate: dict, alpha: float) -> None:
    if parameter_signature(control) != parameter_signature(candidate):
        raise RuntimeError("control/candidate parameter-key identity mismatch")
    for module in ("image_adapter", "text_adapter", "soft_prompt"):
        c_state, r_state = control["model_state"][module], candidate["model_state"][module]
        values = {key: c_state[key] + float(alpha) * (r_state[key] - c_state[key]) for key in c_state}
        getattr(model, module).load_state_dict(values, strict=True)
    model.dfg_beta = float(control.get("dfg_beta_current", .1))
    model.hybrid_alpha_current = float(control.get("hybrid_alpha_current", .2))
    model.hybrid_alpha_max = float(control.get("hybrid_alpha_max", .2))
    model.stage_fusion_weights = (1.0 / 3.0,) * 3
    model.eval()
    model.requires_grad_(False)
    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise RuntimeError(f"non-finite interpolated parameters at alpha={alpha}")


def collect_production_cohort(model, rows: list[dict], device: torch.device) -> dict:
    datasets, indices = selected_datasets(rows)
    policy = PrecisionPolicy("fp16")
    native, resized, masks, labels, names = [], [], [], [], []
    with torch.no_grad():
        for category in spill.CLASS_NAMES["VisA"]:
            loader = DataLoader(Subset(datasets[category], indices[category]), batch_size=BATCH, shuffle=False, num_workers=0)
            text, _, _ = spill.get_hybrid_soft_prompt_single_class_text_embedding(model, "VisA", category, device, return_kg=False)
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                with policy.autocast(device):
                    seg_tokens, _ = model(image)
                    stage_native, _, stage_resized, _ = production_from_tokens(model, seg_tokens, text)
                native.append((stage_native[:, :, 1] - stage_native[:, :, 0]).permute(1, 0, 2, 3).float().cpu().numpy())
                resized.append((stage_resized[:, :, 1] - stage_resized[:, :, 0]).permute(1, 0, 2, 3).float().cpu().numpy())
                masks.append((batch["mask"][:, 0].numpy() > .5).astype(np.uint8))
                labels.append(batch["label"].numpy().astype(np.uint8))
                names.extend(batch["file_name"])
                del image, seg_tokens, stage_native, stage_resized
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    if names != [row["file_name"] for row in rows]:
        raise RuntimeError("strength evaluator order mismatch")
    return {
        "native_margin": np.concatenate(native, axis=0), "resized_margin": np.concatenate(resized, axis=0),
        "mask": np.concatenate(masks, axis=0), "labels": np.concatenate(labels, axis=0),
        "names": np.asarray(names, dtype=object),
    }


def full_region_values(score: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    values = {name: [] for name in ("positive", "interior", "boundary", "near", "far")}
    for current, current_mask in zip(score, mask):
        boundary, interior, near, far = morphology(current_mask)
        for name, region in (("positive", current_mask.astype(bool)), ("interior", interior), ("boundary", boundary), ("near", near), ("far", far)):
            if region.any():
                values[name].append(current[region].reshape(-1))
    return {name: np.concatenate(current) if current else np.empty(0, dtype=np.float32) for name, current in values.items()}


def fpr_recalls(score: np.ndarray, mask: np.ndarray) -> dict:
    regions = full_region_values(score, mask)
    bg = regions["far"]
    result = {}
    for name, rate in (("1pct", .01), ("5pct", .05)):
        threshold = float(np.quantile(bg, 1.0 - rate)) if bg.size else None
        result[name] = {
            "threshold": threshold,
            "recall": float((regions["positive"] > threshold).mean()) if threshold is not None and regions["positive"].size else None,
            "background_count": int(bg.size), "positive_count": int(regions["positive"].size),
        }
    return result


def strength_metrics(data: dict) -> dict:
    score = 1.0 / (1.0 + np.exp(-np.clip(data["resized_margin"].mean(axis=1), -80.0, 80.0)))
    stage2 = 1.0 / (1.0 + np.exp(-np.clip(data["resized_margin"][:, 1], -80.0, 80.0)))
    mask, labels = data["mask"], data["labels"]
    pixel_labels = mask.astype(np.uint8)
    global_metric = exact_binary(score, pixel_labels)
    local = exact.anomaly_near_metrics(score, mask)
    regions = full_region_values(score, mask)
    stage2_regions = full_region_values(stage2, mask)
    fpr = fpr_recalls(score, mask)
    return {
        "pixel_ap": global_metric["ap"], "pixel_auroc": global_metric["auroc"],
        "anomaly_vs_near_ap": local["anomaly_vs_near"]["ap"], "anomaly_vs_near_auroc": local["anomaly_vs_near"]["auroc"],
        "interior_vs_near_auroc": local["interior_vs_near"]["auroc"], "boundary_vs_near_auroc": local["boundary_vs_near"]["auroc"],
        "stage2_near_p95": stats(stage2_regions["near"])["p95"], "stage2_near_p99": stats(stage2_regions["near"])["p99"],
        "final_near_p95": stats(regions["near"])["p95"], "final_near_p99": stats(regions["near"])["p99"],
        "positive_mean": stats(regions["positive"])["mean"], "positive_median": stats(regions["positive"])["median"],
        "interior_mean": stats(regions["interior"])["mean"], "interior_median": stats(regions["interior"])["median"],
        "boundary_mean": stats(regions["boundary"])["mean"], "boundary_median": stats(regions["boundary"])["median"],
        "recall_at_1pct_fpr": fpr["1pct"]["recall"], "recall_at_5pct_fpr": fpr["5pct"]["recall"],
        "fpr_threshold_1pct": fpr["1pct"]["threshold"], "fpr_threshold_5pct": fpr["5pct"]["threshold"],
        "pixel_count": global_metric["pixel_count"],
    }


ALPHAS = (0.00, 0.25, 0.50, 0.75, 1.00)


def strength_phase() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the retained endpoint interpolation path")
    control, candidate, control_path, candidate_path = endpoint_payloads()
    if parameter_signature(control) != parameter_signature(candidate):
        raise RuntimeError("control/candidate parameter-key identity mismatch")
    device = torch.device("cuda:0")
    model = make_model(device)
    rows = []
    for alpha in ALPHAS:
        load_interpolated(model, control, candidate, alpha)
        for cohort_id, path in (("A", COHORT_A), ("B", COHORT_B)):
            data = collect_production_cohort(model, load_cohort(path, cohort_id), device)
            metric = strength_metrics(data)
            rows.append({"alpha": alpha, "cohort": cohort_id, "checkpoint": "theta_control_plus_alpha_endpoint_delta", "control_checkpoint": str(control_path), "candidate_checkpoint": str(candidate_path), "parameter_key_identity": True, "finite_parameters": True, "architecture_identity": True, **metric})
    control_rows = {(row["cohort"], row["alpha"]): row for row in rows}
    decisions = []
    for alpha in ALPHAS[1:]:
        cohort_checks = {}
        for cohort in ("A", "B"):
            current = control_rows[(cohort, alpha)]
            base = control_rows[(cohort, 0.0)]
            checks = {
                "ap_gt_control": current["pixel_ap"] > base["pixel_ap"],
                "auroc_ge_control": current["pixel_auroc"] >= base["pixel_auroc"],
                "near_p95_lt_control": current["final_near_p95"] < base["final_near_p95"],
                "near_p99_lt_control": current["final_near_p99"] < base["final_near_p99"],
                "local_auroc_ge_control": current["anomaly_vs_near_auroc"] >= base["anomaly_vs_near_auroc"],
                "interior_ranking_ge_control": current["interior_vs_near_auroc"] >= base["interior_vs_near_auroc"],
                "boundary_ranking_ge_control": current["boundary_vs_near_auroc"] >= base["boundary_vs_near_auroc"],
                "recall_1pct_ge_control": current["recall_at_1pct_fpr"] >= base["recall_at_1pct_fpr"],
                "recall_5pct_ge_control": current["recall_at_5pct_fpr"] >= base["recall_at_5pct_fpr"],
            }
            cohort_checks[cohort] = checks
        decisions.append({"alpha": alpha, "cohort_checks": cohort_checks, "both_cohorts_supported": all(all(checks.values()) for checks in cohort_checks.values())})
    supported_alphas = [x["alpha"] for x in decisions if x["both_cohorts_supported"]]
    endpoint = next(x for x in decisions if x["alpha"] == 1.0)
    if supported_alphas:
        hypothesis = "SUPPORTED"
    elif endpoint["cohort_checks"]["A"]["ap_gt_control"] and endpoint["cohort_checks"]["B"]["ap_gt_control"]:
        hypothesis = "NOT_SUPPORTED"
    else:
        a_ap = [control_rows[("A", alpha)]["pixel_ap"] for alpha in ALPHAS]
        b_ap = [control_rows[("B", alpha)]["pixel_ap"] for alpha in ALPHAS]
        hypothesis = "MIXED" if any((control_rows[("A", alpha)]["pixel_ap"] > control_rows[("A", 0.0)]["pixel_ap"]) != (control_rows[("B", alpha)]["pixel_ap"] > control_rows[("B", 0.0)]["pixel_ap"]) for alpha in ALPHAS[1:]) else "INCONCLUSIVE"
    all_aps = {cohort: [control_rows[(cohort, alpha)]["pixel_ap"] for alpha in ALPHAS] for cohort in ("A", "B")}
    all_near = {cohort: [control_rows[(cohort, alpha)]["final_near_p95"] for alpha in ALPHAS] for cohort in ("A", "B")}
    intermediate_global_region = []
    for alpha in ALPHAS[1:-1]:
        ok = True
        for cohort in ("A", "B"):
            current, base = control_rows[(cohort, alpha)], control_rows[(cohort, 0.0)]
            ok = ok and current["pixel_ap"] > base["pixel_ap"] and current["pixel_auroc"] >= base["pixel_auroc"] and current["final_near_p95"] < base["final_near_p95"] and current["final_near_p99"] < base["final_near_p99"]
        if ok:
            intermediate_global_region.append(alpha)
    endpoint_worse_than_best = any(control_rows[(cohort, 1.0)]["pixel_ap"] < max(control_rows[(cohort, alpha)]["pixel_ap"] for alpha in ALPHAS[1:-1]) for cohort in ("A", "B"))
    if intermediate_global_region and endpoint_worse_than_best:
        curve = "EARLY_USEFUL_THEN_OVERSUPPRESSED"
    elif supported_alphas == [0.25, 0.50, 0.75, 1.00]:
        curve = "MONOTONIC_USEFUL"
    elif all(all_near[c][i] > all_near[c][0] for c in ("A", "B") for i in (1,)):
        curve = "IMMEDIATE_COUPLING"
    elif not supported_alphas:
        curve = "NO_USEFUL_REGION"
    else:
        curve = "NONLINEAR_UNSTABLE"
    artifact = {
        "protocol_id": "H2_ROOTCAUSE_S2_STRENGTH_AUDIT_R1", "alphas": list(ALPHAS), "rows": rows,
        "parameter_key_identity": parameter_signature(control) == parameter_signature(candidate),
        "interpolated_modules": ["image_adapter", "text_adapter", "soft_prompt"],
        "non_model_state_interpolation": False,
        "interpretation_warning": "Weight interpolation is a diagnostic endpoint direction, not lambda interpolation or a training trajectory.",
        "decisions": decisions, "supported_alphas_both_cohorts": supported_alphas,
        "s2_strength_hypothesis": hypothesis, "strength_curve": curve,
        "control_checkpoint_sha256": sha_file(control_path), "candidate_checkpoint_sha256": sha_file(candidate_path),
    }
    write_csv(OUT_STRENGTH_CSV, rows)
    dump_json(OUT_STRENGTH_JSON, artifact)
    del model
    torch.cuda.empty_cache()
    return artifact


def pareto_phase() -> dict:
    strength = refresh_strength_curve()
    rows = strength["rows"]
    by_key = {(float(row["alpha"]), row["cohort"]): row for row in rows}
    pareto_rows = []
    for row in rows:
        base = by_key[(0.0, row["cohort"])]
        current = dict(row)
        for field in ("pixel_ap", "pixel_auroc", "anomaly_vs_near_ap", "anomaly_vs_near_auroc", "interior_vs_near_auroc", "boundary_vs_near_auroc", "stage2_near_p95", "stage2_near_p99", "final_near_p95", "final_near_p99", "recall_at_1pct_fpr", "recall_at_5pct_fpr", "positive_mean", "interior_mean", "boundary_mean"):
            current[f"{field}_delta_vs_control"] = safe_delta(row.get(field), base.get(field))
        pareto_rows.append(current)
    supported = strength.get("supported_alphas_both_cohorts", [])
    best = None
    if supported:
        best = max(supported, key=lambda alpha: np.mean([by_key[(float(alpha), cohort)]["pixel_ap"] for cohort in ("A", "B")]))
    artifact = {
        "protocol_id": "H2_ROOTCAUSE_S2_STRENGTH_AUDIT_R1", "rows": pareto_rows,
        "best_source_pareto_alpha": best if best is not None else "NONE",
        "best_source_pareto_deltas": {
            "ap": None if best is None else float(np.mean([by_key[(float(best), c)]["pixel_ap"] - by_key[(0.0, c)]["pixel_ap"] for c in ("A", "B")])),
            "auroc": None if best is None else float(np.mean([by_key[(float(best), c)]["pixel_auroc"] - by_key[(0.0, c)]["pixel_auroc"] for c in ("A", "B")])),
            "near_p95": None if best is None else float(np.mean([by_key[(float(best), c)]["final_near_p95"] - by_key[(0.0, c)]["final_near_p95"] for c in ("A", "B")])),
            "interior_recall": None if best is None else float(np.mean([by_key[(float(best), c)]["interior_vs_near_auroc"] - by_key[(0.0, c)]["interior_vs_near_auroc"] for c in ("A", "B")])),
        },
        "strength_hypothesis": strength["s2_strength_hypothesis"], "strength_curve": strength["strength_curve"],
        "gate_definition": "A source Pareto alpha must be one of the preregistered intermediate values, improve AP on both cohorts, not reduce AUROC, reduce both final near-background tails, and preserve local/interior/boundary ranking plus fixed-FPR recall on both cohorts.",
        "weight_interpolation_warning": strength["interpretation_warning"],
    }
    dump_json(OUT_PARETO_JSON, artifact)
    lines = ["# H2 Root-Cause / S2-LOCR R1 Pareto Audit", "", f"* Hypothesis: `{artifact['strength_hypothesis']}`", f"* Curve: `{artifact['strength_curve']}`", f"* Best source Pareto alpha: `{artifact['best_source_pareto_alpha']}`", "", "The five alpha values are fixed endpoint weight interpolations, not lambda interpolation and not a training trajectory.", "", "| alpha | A AP | B AP | A AUROC | B AUROC | A final near p95 | B final near p95 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for alpha in ALPHAS:
        a, b = by_key[(alpha, "A")], by_key[(alpha, "B")]
        lines.append(f"| {alpha:g} | {a['pixel_ap']:.9f} | {b['pixel_ap']:.9f} | {a['pixel_auroc']:.9f} | {b['pixel_auroc']:.9f} | {a['final_near_p95']:.9f} | {b['final_near_p95']:.9f} |")
    lines += ["", "No alpha is called a source Pareto point unless every preregistered preservation check passes on both cohorts. Raw positive/interior means are retained as diagnostics but are not used as the sole gate.", "", f"`R2_TRAINING_AUTHORIZED=NO`; this report only authorizes a source-only formulation direction after final diagnosis."]
    OUT_PARETO_MD.write_text("\n".join(lines) + "\n")
    return artifact


def refresh_strength_curve() -> dict:
    """Recompute only the descriptive curve label from retained results."""
    strength = json.loads(OUT_STRENGTH_JSON.read_text())
    rows = strength["rows"]
    by_key = {(float(row["alpha"]), row["cohort"]): row for row in rows}
    intermediate_global_region = []
    for alpha in ALPHAS[1:-1]:
        ok = True
        for cohort in ("A", "B"):
            current, base = by_key[(alpha, cohort)], by_key[(0.0, cohort)]
            ok = ok and current["pixel_ap"] > base["pixel_ap"] and current["pixel_auroc"] >= base["pixel_auroc"] and current["final_near_p95"] < base["final_near_p95"] and current["final_near_p99"] < base["final_near_p99"]
        if ok:
            intermediate_global_region.append(alpha)
    endpoint_worse_than_best = any(by_key[(1.0, cohort)]["pixel_ap"] < max(by_key[(alpha, cohort)]["pixel_ap"] for alpha in ALPHAS[1:-1]) for cohort in ("A", "B"))
    if intermediate_global_region and endpoint_worse_than_best:
        curve = "EARLY_USEFUL_THEN_OVERSUPPRESSED"
    elif strength.get("supported_alphas_both_cohorts") == [0.25, 0.5, 0.75, 1.0]:
        curve = "MONOTONIC_USEFUL"
    elif all(by_key[(cohort, 0.25)]["final_near_p95"] > by_key[(cohort, 0.0)]["final_near_p95"] for cohort in ("A", "B")):
        curve = "IMMEDIATE_COUPLING"
    elif not strength.get("supported_alphas_both_cohorts"):
        curve = "NO_USEFUL_REGION"
    else:
        curve = "NONLINEAR_UNSTABLE"
    strength["strength_curve"] = curve
    dump_json(OUT_STRENGTH_JSON, strength)
    return strength


def finalization_phase() -> dict:
    leak = json.loads(OUT_LEAK_JSON.read_text())
    module = json.loads(OUT_MODULE_JSON.read_text())
    context = json.loads(OUT_CONTEXT_JSON.read_text())
    strength = json.loads(OUT_STRENGTH_JSON.read_text())
    pareto = json.loads(OUT_PARETO_JSON.read_text())
    onset = leak["leakage_onset_point"]
    first_context = context["first_context_sensitive_module"]
    primary_amp = module.get("primary_amplifier", "NONE")
    secondary_amp = module.get("secondary_amplifier", "NONE")
    root_cause = "MULTI_MODULE_CONTEXTUAL_COUPLING" if context["contextual_propagation_support"] == "YES" and onset != "NOT_ESTABLISHED" else "ROOT_CAUSE_NOT_ESTABLISHED"
    if onset.startswith("stage1_frozen_patch") and context["contextual_propagation_support"] == "YES":
        direction = "NATIVE_FOOTPRINT_UNCERTAINTY_REFINEMENT"
    elif primary_amp == "convlora":
        direction = "CONVLORA_LOCAL_CONTEXT_CONTROL"
    elif primary_amp == "dfg_qk":
        direction = "LOCALIZED_DFG_CONTEXT_GATING"
    else:
        direction = "NONE_YET"
    confidence = "MEDIUM" if root_cause != "ROOT_CAUSE_NOT_ESTABLISHED" else "LOW"
    decision = {
        "protocol_id": "H2_ROOTCAUSE_S2_STRENGTH_AUDIT_R1", "branch": BRANCH,
        "parent_head": PARENT_HEAD,
        "leakage_onset_point": onset, "first_context_sensitive_module": first_context,
        "primary_amplifier": primary_amp, "secondary_amplifier": secondary_amp,
        "primary_root_cause": root_cause, "root_cause_confidence": confidence,
        "module_status": module["module_status"], "contextual_propagation_support": context["contextual_propagation_support"],
        "s2_strength_hypothesis": strength["s2_strength_hypothesis"], "strength_curve": strength["strength_curve"],
        "best_source_pareto_alpha": pareto["best_source_pareto_alpha"], "best_source_pareto_deltas": pareto["best_source_pareto_deltas"],
        "r2_research_direction": direction, "r2_formulation_research_authorized": "YES", "r2_training_authorized": "NO",
        "protocol": {"new_training_run": "NO", "lambda_sweep": "NO", "medical_inference_run": "NO", "mvtec_inference_run": "NO", "target_tuning_used": "NO", "s2_locr_r1_decision_modified": "NO"},
    }
    dump_json(OUT_DECISION_JSON, decision)
    lines = ["# H2 Root-Cause / S2-LOCR R1 Final Decision", "", f"* Branch: `{BRANCH}`", f"* Parent: `{PARENT_HEAD}`", f"* Leakage onset: `{onset}`", f"* First context-sensitive module: `{first_context}`", f"* Primary amplifier: `{primary_amp}`; secondary: `{secondary_amp}`", f"* Primary root cause: `{root_cause}`", f"* Confidence: `{confidence}`", "", "## Module conclusions", ""]
    lines.extend(f"* `{key}={value}`" for key, value in module["module_status"].items())
    lines += [f"* `CONTEXTUAL_PROPAGATION_SUPPORT={context['contextual_propagation_support']}`", "", "## S2 strength", "", f"* `S2_STRENGTH_HYPOTHESIS={strength['s2_strength_hypothesis']}`", f"* `STRENGTH_CURVE={strength['strength_curve']}`", f"* `BEST_SOURCE_PARETO_ALPHA={pareto['best_source_pareto_alpha']}`", "", "## Joint decision", "", f"* `R2_RESEARCH_DIRECTION={direction}`", "* `R2_FORMULATION_RESEARCH_AUTHORIZED=YES`", "* `R2_TRAINING_AUTHORIZED=NO`", "", "The original S2-LOCR R1 decision is not modified. Weight interpolation is diagnostic only; it does not represent lambda interpolation or optimizer dynamics."]
    OUT_DECISION_MD.write_text("\n".join(lines) + "\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("provenance", "leakage", "module", "context", "strength", "pareto", "final"))
    args = parser.parse_args()
    if args.phase == "provenance":
        identity_phase(); graph_phase()
    elif args.phase == "leakage":
        leakage_phase()
    elif args.phase == "module":
        module_phase()
    elif args.phase == "context":
        context_phase()
    elif args.phase == "strength":
        strength_phase()
    elif args.phase == "pareto":
        pareto_phase()
    elif args.phase == "final":
        finalization_phase()


if __name__ == "__main__":
    main()
