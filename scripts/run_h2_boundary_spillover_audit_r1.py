#!/usr/bin/env python3
"""Audit-only H2 stage-wise boundary leakage and spatial spillover.

This script intentionally has no training, loss, optimizer, backward, or target
selection path.  It evaluates the already-frozen Safe-Anchor E10 endpoint and
the two previously retained comparison endpoints on the frozen 96-image VisA
cohort.  The primary score path is the existing VisA test path in
``ACDCLIP.vision_text_fusion_gate_seg``: native logits -> 7x7 Gaussian blur ->
bilinear resize -> equal pre-softmax fusion -> softmax.

Large score arrays are kept outside Git under /workspace.  The required
compact audit artifacts are written under audit/ and results/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from scipy import ndimage, stats
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataset import CLASS_NAMES, DATA_PATH, DOMAINS, get_text_and_image_dataset
from h2_clean.precision import PrecisionPolicy
from h2_clean.stage_fusion import H2_EQUAL_STAGE_FUSION_WEIGHTS, fuse_stage_logits
from model.adapter import ACDCLIP
from model.clip import create_model
from utils import get_hybrid_soft_prompt_single_class_text_embedding


IMG = 518
PATCH = 37
BATCH = 8
PAIR_SEED = 1729
PAIR_PER_IMAGE = 5000
COHORT_CSV = REPO / "audit/H2_FUSION_ENDPOINT_EVAL_SUBSET.csv"
SPLIT_JSON = REPO / "audit/H2_FUSION_SPLIT_IDENTITY.json"
SAFE_ANCHOR = Path("/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth")
LATE_FREEZE = Path("/workspace/h2_late_convlora_freeze_r1/A_LATE_FREEZE_R1_STAGE23_CONVLORA/final.pth")
THBR_CONTROL = Path("/tmp/h2-thbr-evidence/runs/h2_thbr_r1/A_THBR_R1_CONTROL/final.pth")
THBR_CANDIDATE = Path("/tmp/h2-thbr-evidence/runs/h2_thbr_r1/A_THBR_R1_CANDIDATE/final.pth")
EXTERNAL_ROOT = Path("/workspace/h2_boundary_spillover_audit_r1")
CACHE_PATH = EXTERNAL_ROOT / "safe_anchor_e10_maps.npz"

OUT_COHORT = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_COHORT.json"
OUT_SEMANTICS = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_MASK_SEMANTICS.md"
OUT_PROVENANCE = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_SCORE_MAP_PROVENANCE.md"
OUT_NATIVE_CSV = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_STAGE_NATIVE.csv"
OUT_NATIVE_JSON = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_STAGE_NATIVE.json"
OUT_DISTANCE_CSV = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_DISTANCE_PROFILE.csv"
OUT_DISTANCE_JSON = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_DISTANCE_PROFILE.json"
OUT_INTERP_CSV = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_INTERPOLATION.csv"
OUT_INTERP_JSON = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_INTERPOLATION.json"
OUT_FUSION_CSV = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_FUSION_ATTRIBUTION.csv"
OUT_FUSION_JSON = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_FUSION_ATTRIBUTION.json"
OUT_IMAGE_CSV = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_PER_IMAGE.csv"
OUT_CATEGORY_CSV = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_PER_CATEGORY.csv"
OUT_TRI_CSV = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_INTERVENTION_TRIANGULATION.csv"
OUT_TRI_JSON = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_INTERVENTION_TRIANGULATION.json"
OUT_DECISION_MD = REPO / "results/H2_BOUNDARY_SPILLOVER_R1_DECISION.md"
OUT_DECISION_JSON = REPO / "results/H2_BOUNDARY_SPILLOVER_R1_DECISION.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])
    os.replace(temporary, path)


def finite_or_none(value: float | int | np.number | None):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def stats_dict(values: np.ndarray, include_max: bool = True) -> dict:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    result = {"count": int(x.size)}
    if not x.size:
        for name in ("mean", "median", "p90", "p95", "p99", "max"):
            result[name] = None
        return result
    result.update({
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
    })
    if include_max:
        result["max"] = float(x.max())
    return result


def morphology(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    structure = np.ones((7, 7), dtype=bool)
    eroded = ndimage.binary_erosion(mask.astype(bool), structure=structure)
    dilated = ndimage.binary_dilation(mask.astype(bool), structure=structure)
    return mask.astype(bool) & ~eroded, eroded, dilated & ~mask.astype(bool), ~dilated


def safe_ratio(left, right):
    left = finite_or_none(left)
    right = finite_or_none(right)
    if left is None or right is None or abs(right) < 1e-12:
        return None
    return float(left / right)


def safe_delta(left, right):
    left = finite_or_none(left)
    right = finite_or_none(right)
    if left is None or right is None:
        return None
    return float(left - right)


def binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.uint8).reshape(-1)
    valid = np.isfinite(x)
    x, y = x[valid], y[valid]
    if not x.size or y.min() == y.max():
        return {"auroc": None, "ap": None, "pixel_count": int(x.size)}
    order = np.argsort(-x, kind="stable")
    sx, sy = x[order], y[order]
    tp = np.cumsum(sy, dtype=np.float64)
    fp = np.cumsum(1 - sy, dtype=np.float64)
    positives, negatives = float(tp[-1]), float(fp[-1])
    ap = float((tp / np.maximum(tp + fp, 1.0) * sy).sum() / positives)
    ends = np.r_[np.flatnonzero(sx[1:] != sx[:-1]), len(sx) - 1]
    tpr = np.r_[0.0, tp[ends] / positives]
    fpr = np.r_[0.0, fp[ends] / negatives]
    return {"auroc": float(np.trapezoid(tpr, fpr)), "ap": ap, "pixel_count": int(x.size)}


def load_rows() -> tuple[list[dict], dict[str, list[dict]]]:
    with COHORT_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 96 or any(row["role"] != "ENDPOINT_EVAL" for row in rows):
        raise RuntimeError("the frozen endpoint cohort is not the required 96-row ENDPOINT_EVAL subset")
    by_category = {category: [] for category in CLASS_NAMES["VisA"]}
    for row in rows:
        by_category[row["category"]].append(row)
    if any(len(by_category[category]) != 8 for category in by_category):
        raise RuntimeError("the frozen endpoint cohort does not contain 8 rows per category")
    split = json.loads(SPLIT_JSON.read_text())
    if split.get("endpoint_eval_count") != 96 or split.get("intersection_count") != 0:
        raise RuntimeError("the frozen split identity is not disjoint and 96-image")
    return rows, by_category


def source_mask_path(category: str, meta: dict) -> Path | None:
    if not int(meta["label"]):
        return None
    return Path(DATA_PATH["VisA"]) / meta["mask_path"]


def make_cohort_artifact(rows: list[dict], by_category: dict[str, list[dict]]) -> None:
    datasets = get_text_and_image_dataset("VisA", IMG, "test")
    records = []
    for row in rows:
        category = row["category"]
        dataset = datasets[category]
        index_by_name = {meta["image_path"]: index for index, meta in enumerate(dataset.meta)}
        if row["file_name"] not in index_by_name:
            raise RuntimeError(f"cohort file is absent from VisA test manifest: {row['file_name']}")
        index = index_by_name[row["file_name"]]
        meta = dataset.meta[index]
        expected_label = 1 if row["label_type"] == "anomaly" else 0
        if int(meta["label"]) != expected_label:
            raise RuntimeError(f"cohort label mismatch: {row['file_name']}")
        image_path = Path(DATA_PATH["VisA"]) / meta["image_path"]
        mask_path = source_mask_path(category, meta)
        transformed_mask = dataset[index]["mask"].numpy().astype(np.float32, copy=False)
        records.append({
            "category": category,
            "label": expected_label,
            "label_type": row["label_type"],
            "manifest_position": int(row["manifest_position"]),
            "file_name": row["file_name"],
            "image_source": str(image_path),
            "image_sha256": sha256_file(image_path),
            "mask_source": None if mask_path is None else str(mask_path),
            "mask_sha256": None if mask_path is None else sha256_file(mask_path),
            "resized_mask_sha256": sha256_bytes(np.ascontiguousarray(transformed_mask).tobytes()),
        })
    canonical = "\n".join(f"{r['category']}|{r['label_type']}|{r['file_name']}" for r in records).encode()
    artifact = {
        "protocol_id": "H2_BOUNDARY_SPILLOVER_AUDIT_R1",
        "dataset": "VisA",
        "cohort_source": str(COHORT_CSV.relative_to(REPO)),
        "cohort_source_sha256": sha256_file(COHORT_CSV),
        "split_identity_source": str(SPLIT_JSON.relative_to(REPO)),
        "split_canonical_sha256": json.loads(SPLIT_JSON.read_text())["canonical_sha256"],
        "count": len(records),
        "per_category_count": {category: len(by_category[category]) for category in CLASS_NAMES["VisA"]},
        "ordering": "frozen CSV order; no resampling, category selection, or target inference",
        "records": records,
        "audit_record_canonical_sha256": sha256_bytes(canonical),
    }
    dump_json(OUT_COHORT, artifact)


def write_semantics_and_provenance() -> None:
    OUT_SEMANTICS.write_text(
        """# H2 Boundary Spillover R1 — Mask Semantics

The audit uses the exact existing dataset mask path and the existing
morphology semantics; no morphology-radius sweep or new mask rule is
introduced. Source masks are resized to 518x518 with nearest-neighbor
interpolation by `BaseSingleClassDataset`, then binarized. Native-stage ground
truth is the exact existing nearest-neighbor resize of that source mask to the
37x37 patch grid (`F.interpolate(..., mode="nearest")`).

For every resolution, the existing 7x7 all-ones structuring element is used
exactly once: `boundary = mask & ~binary_erosion(mask)`, `interior =
binary_erosion(mask)`, `near_background = binary_dilation(mask) & ~mask`, and
`far_background = ~binary_dilation(mask)`. Normal images have an empty anomaly
mask and therefore contribute only to positive/negative or global metric
calculations, not anomaly-region interior/boundary/distance summaries. Empty
regions are reported as null.

The 1-pixel distance bin is the standard Euclidean distance-transform interval
`(0, 1]`; subsequent fixed bins are `(1,2]`, `(2,4]`, `(4,8]`, `(8,16]`, and
`(16, infinity)`.
"""
    )
    OUT_PROVENANCE.write_text(
        f"""# H2 Boundary Spillover R1 — Score-Map Provenance

## Frozen endpoint and inference identity

Primary endpoint: Safe-Anchor E10, loaded read-only from `{SAFE_ANCHOR}`
(SHA-256 `{sha256_file(SAFE_ANCHOR)}`). The frozen cohort is
`audit/H2_FUSION_ENDPOINT_EVAL_SUBSET.csv`, exactly 96 images, 8 per VisA
category, with the cohort artifact recording source image/mask hashes. The
prior freeze and THBR endpoints are used only as descriptive inference-only
comparison arms in the final triangulation.

Model construction is the committed H2 Safe-Anchor E10 architecture and
configuration: ViT-L/14-336, image size 518, three stages, DFG attention with
SS2D weight residual, hybrid text prompts, and equal stage fusion. No
architecture, feature detachment, fusion weight, or interpolation change is
made by this audit. `eval()`, `requires_grad_(False)`, `torch.no_grad()`, and
the historical FP16 autocast inference policy are used; there is no backward or
optimizer operation.

## Map inventory

* Stage 1/2/3 native anomaly maps: each stage's native 37x37 class logits from
  `_vision_text_attention_fusion`, converted with per-stage softmax. Native
  logits are retained conceptually for the equal pre-softmax native combined
  map.
* Resized stage maps: each native stage logit map receives the existing VisA
  test-mode 7x7 Gaussian blur (`sigma=1`) and is then bilinearly resized to
  518x518 with `align_corners=True`; per-stage softmax yields the stage anomaly
  probability.
* Pre-fusion combined map: equal mean of the three native stage logits followed
  by softmax for native-resolution descriptive measurements.
* Final resized/evaluator map: equal mean of the three resized, blurred stage
  logits followed by softmax. This is the current production VisA endpoint path
  in `vision_text_fusion_gate_seg(test_mode=True, domain="Industrial")`. There
  is no separate hidden final resize after this map.
* Fusion attribution additionally reports the equal pre-softmax fused anomaly
  logit margin (`logit_abnormal - logit_normal`) so that the pre-softmax map
  remains in its native score space rather than being silently redefined as a
  probability.

All interpolation statements are observational comparisons of the exact
current path; no alternative interpolation is evaluated or selected.
"""
    )


def make_model(device: torch.device) -> ACDCLIP:
    clip = create_model(
        "ViT-L-14-336", img_size=IMG, device=device,
        pretrained="openai", require_pretrained=True,
    )
    model = ACDCLIP(
        clip_model=clip, n_groups=3, image_adapt_weight=.2, text_adapt_weight=.2,
        conv_lora_rank=8, conv_lora_alpha=2., conv_kernel_size_list=[3, 5],
        lora_rank=16, lora_alpha=2., dfg_mode="attn", dfg_attn_dim=256,
        dfg_attn_tau=8., use_ss2d_dfg=True, dfg_gamma_max=.2,
        dfg_ss2d_fusion="weight_residual", dfg_beta=.1,
        dfg_beta_schedule="warmup010", dfg_beta_target=.1, dfg_beta_current=.1,
        dfg_weight_residual_fp32=True,
        stage_fusion_weights=H2_EQUAL_STAGE_FUSION_WEIGHTS,
        use_soft_prompt=False, soft_prompt_ctx_len=4,
        soft_prompt_init="phrase", soft_prompt_init_phrase="a photo of a",
    ).to(device).eval()
    for module in model.modules():
        if hasattr(module, "enable_fp16_numerical_islands"):
            module.enable_fp16_numerical_islands = False
    model.prompt_mode = "hybrid"
    model.use_hybrid_soft_prompt = True
    model.use_soft_prompt = False
    model.requires_grad_(False)
    return model


def load_endpoint(model: ACDCLIP, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload)
    model.image_adapter.load_state_dict(state["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(state["text_adapter"], strict=True)
    model.soft_prompt.load_state_dict(state["soft_prompt"], strict=True)
    model.dfg_beta = float(payload.get("dfg_beta_current", getattr(model, "dfg_beta", .1)))
    model.hybrid_alpha_current = float(payload.get("hybrid_alpha_current", .2))
    model.hybrid_alpha_max = .2
    model.stage_fusion_weights = H2_EQUAL_STAGE_FUSION_WEIGHTS
    model.eval()
    model.requires_grad_(False)
    return payload


def endpoint_identity(path: Path, label: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload)
    return {
        "label": label,
        "path": str(path),
        "sha256": sha256_file(path),
        "git_evidence": {
            "safe_anchor": "research/h2-safe-anchor-e20-medical-selected",
            "late_freeze": "research/h2-late-convlora-freeze-r1",
            "thbr": "research/h2-thbr-r1",
        }.get(label),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "checkpoint_version": payload.get("checkpoint_version"),
        "implementation_git_sha": payload.get("implementation_git_sha"),
        "model_state_finite": all(torch.isfinite(v).all().item() for module in state.values() if isinstance(module, dict) for v in module.values() if torch.is_tensor(v)),
    }


def native_and_resized_maps(model, vision, text):
    batch, patches, _ = vision.shape[1:]
    side = int(math.sqrt(patches))
    if side != PATCH:
        raise RuntimeError(f"unexpected native patch grid {side}")
    group_text = text.unsqueeze(1).repeat(1, batch, 1, 1).permute(1, 0, 2, 3)
    native_logits, resized_logits = [], []
    for stage in range(model.n_groups):
        fused = model._vision_text_attention_fusion(vision[stage], group_text, stage)
        logits = torch.matmul(10 * vision[stage], fused).permute(0, 2, 1).view(batch, 2, side, side)
        native_logits.append(logits)
        blurred = gaussian_blur2d(logits, (7, 7), (1, 1))
        resized_logits.append(F.interpolate(blurred, (IMG, IMG), mode="bilinear", align_corners=True))
    native_logits = torch.stack(native_logits)
    resized_logits = torch.stack(resized_logits)
    native_fused_logits = fuse_stage_logits(native_logits, H2_EQUAL_STAGE_FUSION_WEIGHTS)
    resized_fused_logits = fuse_stage_logits(resized_logits, H2_EQUAL_STAGE_FUSION_WEIGHTS)
    return {
        "native_stage_prob": F.softmax(native_logits, dim=2)[:, :, 1].permute(1, 0, 2, 3),
        "native_fused_prob": F.softmax(native_fused_logits, dim=1)[:, 1],
        "resized_stage_prob": F.softmax(resized_logits, dim=2)[:, :, 1].permute(1, 0, 2, 3),
        "resized_fused_prob": F.softmax(resized_fused_logits, dim=1)[:, 1],
        "resized_fused_margin": resized_fused_logits[:, 1] - resized_fused_logits[:, 0],
    }


def load_selection_datasets(rows: list[dict]):
    datasets = get_text_and_image_dataset("VisA", IMG, "test")
    indices = {}
    for category in CLASS_NAMES["VisA"]:
        by_name = {meta["image_path"]: i for i, meta in enumerate(datasets[category].meta)}
        indices[category] = []
        for row in rows:
            if row["category"] != category:
                continue
            if row["file_name"] not in by_name:
                raise RuntimeError(f"cohort file is absent: {row['file_name']}")
            i = by_name[row["file_name"]]
            if int(datasets[category].meta[i]["label"]) != (1 if row["label_type"] == "anomaly" else 0):
                raise RuntimeError(f"cohort label mismatch: {row['file_name']}")
            indices[category].append(i)
    return datasets, indices


def evaluate_maps(model: ACDCLIP, checkpoint: Path, rows: list[dict], device: torch.device, *, full: bool) -> dict:
    load_endpoint(model, checkpoint)
    datasets, indices = load_selection_datasets(rows)
    policy = PrecisionPolicy("fp16")
    chunks = {key: [] for key in (
        "native_stage_prob", "native_fused_prob", "resized_stage_prob",
        "resized_fused_prob", "resized_fused_margin", "mask",
    )}
    names = []
    categories = []
    labels = []
    with torch.no_grad():
        for category in CLASS_NAMES["VisA"]:
            selected = indices[category]
            loader = DataLoader(Subset(datasets[category], selected), batch_size=BATCH, shuffle=False, num_workers=0)
            text, _, _ = get_hybrid_soft_prompt_single_class_text_embedding(
                model, "VisA", category, device, return_kg=False,
            )
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                with policy.autocast(device):
                    seg_tokens, _ = model(image)
                    vision = torch.stack(seg_tokens)
                    maps = native_and_resized_maps(model, vision, text)
                chunks["native_stage_prob"].append(maps["native_stage_prob"].float().cpu().numpy())
                chunks["native_fused_prob"].append(maps["native_fused_prob"].float().cpu().numpy())
                chunks["resized_stage_prob"].append(maps["resized_stage_prob"].float().cpu().numpy())
                chunks["resized_fused_prob"].append(maps["resized_fused_prob"].float().cpu().numpy())
                chunks["resized_fused_margin"].append(maps["resized_fused_margin"].float().cpu().numpy())
                chunks["mask"].append((batch["mask"][:, 0].numpy() > 0.5).astype(np.uint8))
                names.extend(list(batch["file_name"]))
                categories.extend([category] * len(batch["file_name"]))
                labels.extend(batch["label"].numpy().astype(np.uint8).tolist())
                del vision, maps
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    expected = [row["file_name"] for row in rows]
    if names != expected:
        raise RuntimeError("evaluation order does not match frozen cohort order")
    result = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    result["names"] = np.asarray(names, dtype=object)
    result["categories"] = np.asarray(categories, dtype=object)
    result["labels"] = np.asarray(labels, dtype=np.uint8)
    result["checkpoint"] = str(checkpoint)
    result["checkpoint_sha256"] = sha256_file(checkpoint)
    result["full_maps"] = bool(full)
    return result


def save_baseline_cache(data: dict) -> None:
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        native_stage_prob=data["native_stage_prob"].astype(np.float32),
        native_fused_prob=data["native_fused_prob"].astype(np.float32),
        resized_stage_prob=data["resized_stage_prob"].astype(np.float32),
        resized_fused_prob=data["resized_fused_prob"].astype(np.float32),
        resized_fused_margin=data["resized_fused_margin"].astype(np.float32),
        mask=data["mask"].astype(np.uint8),
        labels=data["labels"].astype(np.uint8),
        categories=data["categories"].astype(str),
        names=data["names"].astype(str),
    )


def load_baseline_cache() -> dict:
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(f"missing external baseline cache: {CACHE_PATH}; run --phase native first")
    with np.load(CACHE_PATH, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def regions_for(score: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    values = {key: [] for key in ("positive", "negative", "boundary", "interior", "near", "far")}
    for s, m in zip(score, mask):
        boundary, interior, near, far = morphology(m)
        for key, region in (("positive", m.astype(bool)), ("negative", ~m.astype(bool)),
                            ("boundary", boundary), ("interior", interior),
                            ("near", near), ("far", far)):
            if region.any():
                values[key].append(s[region].astype(np.float32))
    return {key: np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
            for key, chunks in values.items()}


def region_summary(score: np.ndarray, mask: np.ndarray) -> dict:
    regions = regions_for(score, mask)
    out = {key: stats_dict(value) for key, value in regions.items()}
    out["ratios"] = {
        "near_p95_over_interior_p95": safe_ratio(out["near"]["p95"], out["interior"]["p95"]),
        "near_p99_over_interior_p99": safe_ratio(out["near"]["p99"], out["interior"]["p99"]),
        "near_mean_over_boundary_mean": safe_ratio(out["near"]["mean"], out["boundary"]["mean"]),
        "near_p99_over_boundary_p99": safe_ratio(out["near"]["p99"], out["boundary"]["p99"]),
    }
    out["contrasts"] = {
        "interior_mean_minus_near_mean": safe_delta(out["interior"]["mean"], out["near"]["mean"]),
        "boundary_mean_minus_near_mean": safe_delta(out["boundary"]["mean"], out["near"]["mean"]),
        "interior_p95_minus_near_p95": safe_delta(out["interior"]["p95"], out["near"]["p95"]),
        "boundary_p95_minus_near_p95": safe_delta(out["boundary"]["p95"], out["near"]["p95"]),
    }
    return out


def flatten_masks_native(mask: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[:, None]
    return (F.interpolate(tensor, (PATCH, PATCH), mode="nearest")[:, 0].numpy() > 0.5).astype(np.uint8)


def native_stage_phase(data: dict) -> tuple[list[dict], dict]:
    mask_native = flatten_masks_native(data["mask"])
    rows, structured = [], {"protocol_id": "H2_BOUNDARY_SPILLOVER_AUDIT_R1", "checkpoint": "SAFE_ANCHOR_E10", "resolution": "37x37", "mask_resolution": "nearest_37x37", "stage_rows": []}
    for stage in range(3):
        summary = region_summary(data["native_stage_prob"][:, stage], mask_native)
        row = {"checkpoint": "SAFE_ANCHOR_E10", "stage": stage + 1, "resolution": "37x37", "score_space": "per_stage_anomaly_probability"}
        for region in ("positive", "negative", "boundary", "interior", "near", "far"):
            for stat_name, value in summary[region].items():
                row[f"{region}_{stat_name}"] = value
        for group in ("ratios", "contrasts"):
            for name, value in summary[group].items():
                row[name] = value
        rows.append(row)
        structured["stage_rows"].append(row)
    structured["methods"] = {"morphology": "existing 7x7 erosion/dilation", "no_new_radius": True}
    write_csv(OUT_NATIVE_CSV, rows)
    dump_json(OUT_NATIVE_JSON, structured)
    return rows, structured


DISTANCE_BINS = (("1", 0.0, 1.0), ("2", 1.0, 2.0), ("3-4", 2.0, 4.0), ("5-8", 4.0, 8.0), ("9-16", 8.0, 16.0), (">16", 16.0, float("inf")))


def distance_rows(score: np.ndarray, mask: np.ndarray, map_name: str, resolution: str) -> tuple[list[dict], dict]:
    bin_values = {name: [] for name, _, _ in DISTANCE_BINS}
    all_distance, all_score = [], []
    for s, m in zip(score, mask):
        if not m.any():
            continue
        distance = ndimage.distance_transform_edt(~m.astype(bool))
        outside = ~m.astype(bool)
        all_distance.append(distance[outside].astype(np.float32))
        all_score.append(s[outside].astype(np.float32))
        for name, low, high in DISTANCE_BINS:
            selected = outside & (distance > low) & (distance <= high)
            if selected.any():
                bin_values[name].append(s[selected].astype(np.float32))
    rows = []
    for name, _, _ in DISTANCE_BINS:
        values = np.concatenate(bin_values[name]) if bin_values[name] else np.empty(0, dtype=np.float32)
        row = {"checkpoint": "SAFE_ANCHOR_E10", "map": map_name, "resolution": resolution, "bin": name}
        row.update(stats_dict(values))
        rows.append(row)
    if all_distance:
        distances = np.concatenate(all_distance).astype(np.float64)
        scores = np.concatenate(all_score).astype(np.float64)
        rng = np.random.default_rng(PAIR_SEED)
        if len(scores) > 200_000:
            chosen = rng.choice(len(scores), size=200_000, replace=False)
            distances, scores = distances[chosen], scores[chosen]
        if np.unique(distances).size > 1 and np.unique(scores).size > 1:
            spearman = float(stats.spearmanr(distances, scores).statistic)
            slope = float(np.polyfit(distances, scores, 1)[0])
        else:
            spearman, slope = None, None
        all_count = int(len(distances))
    else:
        spearman, slope, all_count = None, None, 0
    summary = {"checkpoint": "SAFE_ANCHOR_E10", "map": map_name, "resolution": resolution,
               "bin": "ALL_OUTSIDE", "count": all_count, "distance_spearman": spearman,
               "distance_linear_slope": slope, "sample_seed": PAIR_SEED, "sample_cap": 200_000}
    rows.append(summary)
    return rows, summary


def distance_phase(data: dict) -> tuple[list[dict], dict]:
    native_mask = flatten_masks_native(data["mask"])
    rows, summaries = [], []
    for stage in range(3):
        current, summary = distance_rows(data["native_stage_prob"][:, stage], native_mask, f"stage_{stage+1}", "37x37")
        rows.extend(current); summaries.append(summary)
    current, summary = distance_rows(data["resized_fused_prob"], data["mask"], "final_fused", "518x518")
    rows.extend(current); summaries.append(summary)
    structured = {"protocol_id": "H2_BOUNDARY_SPILLOVER_AUDIT_R1", "bin_definition": [(name, low, None if math.isinf(high) else high) for name, low, high in DISTANCE_BINS], "rows": rows, "summaries": summaries,
                  "interpretation": "negative Spearman/slope indicates descriptive score decay with distance from anomaly; this is not a causal claim"}
    write_csv(OUT_DISTANCE_CSV, rows)
    dump_json(OUT_DISTANCE_JSON, structured)
    return rows, structured


def interpolation_phase(data: dict) -> tuple[list[dict], dict]:
    native_mask = flatten_masks_native(data["mask"])
    rows = []
    for stage in range(3):
        native = region_summary(data["native_stage_prob"][:, stage], native_mask)
        resized = region_summary(data["resized_stage_prob"][:, stage], data["mask"])
        row = {"checkpoint": "SAFE_ANCHOR_E10", "stage": stage + 1, "native_resolution": "37x37", "resized_resolution": "518x518",
               "resized_path": "existing_test_mode_gaussian7_sigma1_then_bilinear_align_corners_true"}
        for prefix, summary in (("native", native), ("resized", resized)):
            for region in ("boundary", "interior", "near", "far"):
                for stat_name in ("mean", "p95", "p99"):
                    row[f"{prefix}_{region}_{stat_name}"] = summary[region][stat_name]
            row[f"{prefix}_near_p95_over_interior_p95"] = summary["ratios"]["near_p95_over_interior_p95"]
            row[f"{prefix}_near_p99_over_interior_p99"] = summary["ratios"]["near_p99_over_interior_p99"]
            row[f"{prefix}_interior_mean_minus_near_mean"] = summary["contrasts"]["interior_mean_minus_near_mean"]
        row["delta_near_mean_resized_minus_native"] = safe_delta(row["resized_near_mean"], row["native_near_mean"])
        row["delta_near_p95_resized_minus_native"] = safe_delta(row["resized_near_p95"], row["native_near_p95"])
        row["delta_near_p99_resized_minus_native"] = safe_delta(row["resized_near_p99"], row["native_near_p99"])
        row["delta_boundary_mean_resized_minus_native"] = safe_delta(row["resized_boundary_mean"], row["native_boundary_mean"])
        row["delta_interior_mean_resized_minus_native"] = safe_delta(row["resized_interior_mean"], row["native_interior_mean"])
        row["interpolation_amplification_ratio_p95"] = safe_delta(row["resized_near_p95_over_interior_p95"], row["native_near_p95_over_interior_p95"])
        row["interpolation_amplification_ratio_p99"] = safe_delta(row["resized_near_p99_over_interior_p99"], row["native_near_p99_over_interior_p99"])
        rows.append(row)
    structured = {"protocol_id": "H2_BOUNDARY_SPILLOVER_AUDIT_R1", "rows": rows,
                  "amplification_definition": "resized leakage ratio minus native leakage ratio, with raw near/boundary/interior deltas beside it",
                  "interpretation": "materiality is descriptive and is not used to alter interpolation"}
    write_csv(OUT_INTERP_CSV, rows)
    dump_json(OUT_INTERP_JSON, structured)
    return rows, structured


def sampled_inversion(score_a: np.ndarray, score_b: np.ndarray, masks: np.ndarray, *, greater: bool = True) -> float | None:
    rng = np.random.default_rng(PAIR_SEED)
    rates = []
    for a, b, mask in zip(score_a, score_b, masks):
        boundary, interior, near, far = morphology(mask)
        left = a[near]
        right = b[mask.astype(bool) if greater else interior]
        if not left.size or not right.size:
            continue
        nl = min(PAIR_PER_IMAGE, left.size)
        nr = min(PAIR_PER_IMAGE, right.size)
        li = rng.choice(left.size, size=nl, replace=left.size < nl)
        ri = rng.choice(right.size, size=nr, replace=right.size < nr)
        n = min(nl, nr)
        lhs, rhs = left[li[:n]], right[ri[:n]]
        rates.append(float((lhs > rhs).mean()))
    return float(np.mean(rates)) if rates else None


def fusion_map_metrics(score: np.ndarray, mask: np.ndarray, labels: np.ndarray, map_name: str) -> dict:
    summary = region_summary(score, mask)
    return {
        "map": map_name,
        "score_space": "logit_margin" if map_name == "equal_pre_softmax_fused" else "anomaly_probability",
        **binary_metrics(score, mask),
        "positive_mean": summary["positive"]["mean"], "positive_median": summary["positive"]["median"],
        "interior_mean": summary["interior"]["mean"], "interior_median": summary["interior"]["median"],
        "boundary_mean": summary["boundary"]["mean"], "boundary_median": summary["boundary"]["median"],
        "near_p95": summary["near"]["p95"], "near_p99": summary["near"]["p99"],
        "far_p95": summary["far"]["p95"], "far_p99": summary["far"]["p99"],
        "near_gt_positive_inversion": sampled_inversion(score, score, mask, greater=True),
        "near_gt_interior_inversion": sampled_inversion(score, score, mask, greater=False),
        "near_p95_over_interior_p95": summary["ratios"]["near_p95_over_interior_p95"],
        "near_p99_over_interior_p99": summary["ratios"]["near_p99_over_interior_p99"],
        "near_gt_far_p95": bool(summary["near"]["p95"] is not None and summary["far"]["p95"] is not None and summary["near"]["p95"] > summary["far"]["p95"]),
        "near_gt_interior_p95": bool(summary["near"]["p95"] is not None and summary["interior"]["p95"] is not None and summary["near"]["p95"] > summary["interior"]["p95"]),
    }


def fusion_phase(data: dict) -> tuple[list[dict], dict]:
    masks, labels = data["mask"], data["labels"]
    maps = {
        "stage_1_resized": data["resized_stage_prob"][:, 0],
        "stage_2_resized": data["resized_stage_prob"][:, 1],
        "stage_3_resized": data["resized_stage_prob"][:, 2],
        "equal_pre_softmax_fused": data["resized_fused_margin"],
        "final_production_fused": data["resized_fused_prob"],
    }
    rows = [fusion_map_metrics(score, masks, labels, name) for name, score in maps.items()]
    stage_rows = rows[:3]
    final = rows[-1]
    final["fusion_near_p95_delta_vs_worst_stage"] = final["near_p95"] - max(row["near_p95"] for row in stage_rows if row["near_p95"] is not None)
    final["fusion_near_p99_delta_vs_worst_stage"] = final["near_p99"] - max(row["near_p99"] for row in stage_rows if row["near_p99"] is not None)
    final["fusion_ratio_p95_delta_vs_worst_stage"] = final["near_p95_over_interior_p95"] - max(row["near_p95_over_interior_p95"] for row in stage_rows if row["near_p95_over_interior_p95"] is not None)
    final["fusion_ratio_p99_delta_vs_worst_stage"] = final["near_p99_over_interior_p99"] - max(row["near_p99_over_interior_p99"] for row in stage_rows if row["near_p99_over_interior_p99"] is not None)
    structured = {"protocol_id": "H2_BOUNDARY_SPILLOVER_AUDIT_R1", "rows": rows,
                  "fusion_delta_definition": "final production fused map minus the worst individual resized-stage map, with all component rows retained",
                  "pairing": {"seed": PAIR_SEED, "pairs_per_image_cap": PAIR_PER_IMAGE, "descriptive_only": True}}
    write_csv(OUT_FUSION_CSV, rows)
    dump_json(OUT_FUSION_JSON, structured)
    return rows, structured


def per_image_phase(data: dict) -> tuple[list[dict], list[dict], dict]:
    score, masks, labels = data["resized_fused_prob"], data["mask"], data["labels"]
    image_rows = []
    for index, (s, m, label, category, name) in enumerate(zip(score, masks, labels, data["categories"], data["names"])):
        if not int(label):
            continue
        boundary, interior, near, far = morphology(m)
        def one(region, stat):
            values = s[region]
            return finite_or_none(np.quantile(values, stat)) if values.size else None
        near_p95, near_p99 = one(near, .95), one(near, .99)
        far_p95 = one(far, .95)
        boundary_p95 = one(boundary, .95)
        interior_mean = float(s[interior].mean()) if interior.any() else None
        boundary_mean = float(s[boundary].mean()) if boundary.any() else None
        near_mean = float(s[near].mean()) if near.any() else None
        leakage = bool(near_p95 is not None and far_p95 is not None and boundary_p95 is not None and near_p95 > far_p95 and abs(near_p95 - boundary_p95) < abs(near_p95 - far_p95))
        # Deterministic per-image pair rates use the same fixed RNG stream order.
        pair_rate_interior = sampled_inversion(score[index:index+1], score[index:index+1], masks[index:index+1], greater=False)
        pair_rate_positive = sampled_inversion(score[index:index+1], score[index:index+1], masks[index:index+1], greater=True)
        image_rows.append({
            "checkpoint": "SAFE_ANCHOR_E10", "image_index": index, "category": str(category), "file_name": str(name),
            "near_p95": near_p95, "near_p99": near_p99, "far_p95": far_p95, "boundary_p95": boundary_p95,
            "interior_mean": interior_mean, "boundary_mean": boundary_mean, "near_mean": near_mean,
            "interior_mean_minus_near_mean": safe_delta(interior_mean, near_mean),
            "boundary_mean_minus_near_mean": safe_delta(boundary_mean, near_mean),
            "near_gt_positive_inversion": pair_rate_positive, "near_gt_interior_inversion": pair_rate_interior,
            "leakage_image_definition_pass": leakage,
        })
    category_rows = []
    for category in CLASS_NAMES["VisA"]:
        current = [row for row in image_rows if row["category"] == category]
        row = {"checkpoint": "SAFE_ANCHOR_E10", "category": category, "anomalous_image_count": len(current)}
        for field in ("near_p95", "near_p99", "far_p95", "boundary_p95", "interior_mean", "boundary_mean", "near_mean", "interior_mean_minus_near_mean", "boundary_mean_minus_near_mean", "near_gt_positive_inversion", "near_gt_interior_inversion"):
            values = np.asarray([r[field] for r in current if r[field] is not None], dtype=np.float64)
            row[f"{field}_mean"] = float(values.mean()) if values.size else None
            row[f"{field}_median"] = float(np.median(values)) if values.size else None
            row[f"{field}_iqr"] = float(np.quantile(values, .75) - np.quantile(values, .25)) if values.size else None
        leakage_values = [bool(r["leakage_image_definition_pass"]) for r in current]
        row["leakage_image_fraction"] = float(np.mean(leakage_values)) if leakage_values else None
        row["category_shows_leakage_majority_rule"] = bool(sum(leakage_values) >= max(1, math.ceil(len(leakage_values) / 2))) if leakage_values else False
        category_rows.append(row)
    image_fraction = float(np.mean([r["leakage_image_definition_pass"] for r in image_rows])) if image_rows else None
    category_fraction = float(np.mean([r["category_shows_leakage_majority_rule"] for r in category_rows])) if category_rows else None
    structured = {"protocol_id": "H2_BOUNDARY_SPILLOVER_AUDIT_R1", "image_leakage_definition": "near p95 > far p95 and distance from near p95 to boundary p95 is less than distance to far p95", "category_rule": "category shows leakage when at least half of its four anomalous images pass the image definition", "anomalous_image_fraction": image_fraction, "category_fraction_majority_rule": category_fraction}
    write_csv(OUT_IMAGE_CSV, image_rows)
    write_csv(OUT_CATEGORY_CSV, category_rows)
    dump_json(OUT_IMAGE_CSV.with_suffix(".json"), structured)
    return image_rows, category_rows, structured


def final_summary_for_comparator(model: ACDCLIP, path: Path, label: str, rows: list[dict], device: torch.device) -> dict:
    data = evaluate_maps(model, path, rows, device, full=False)
    score, mask, labels = data["resized_fused_prob"], data["mask"], data["labels"]
    metrics = fusion_map_metrics(score, mask, labels, "final_production_fused")
    _, distance = distance_rows(score, mask, label, "518x518")
    metrics.update({"checkpoint": label, "checkpoint_path": str(path), "checkpoint_sha256": sha256_file(path),
                    "distance_spearman": distance["distance_spearman"], "distance_linear_slope": distance["distance_linear_slope"],
                    "distance_leakage_1px_mean": next((r["mean"] for r in _ if r["bin"] == "1"), None),
                    "distance_leakage_3_4px_mean": next((r["mean"] for r in _ if r["bin"] == "3-4"), None)})
    return metrics


def severity_and_decision(data: dict, interp: dict, fusion: dict, image_struct: dict, tri_rows: list[dict], distance_struct: dict) -> tuple[dict, dict]:
    stage_rows = json.loads(OUT_NATIVE_JSON.read_text())["stage_rows"]
    ratios95 = np.asarray([r["near_p95_over_interior_p95"] for r in stage_rows if r["near_p95_over_interior_p95"] is not None], dtype=float)
    ratios99 = np.asarray([r["near_p99_over_interior_p99"] for r in stage_rows if r["near_p99_over_interior_p99"] is not None], dtype=float)
    ratio_available = bool(ratios95.size == 3 and ratios99.size == 3)
    if ratio_available:
        severity_metric = "requested_near_tail_over_interior_tail_ratios"
        severity95, severity99 = ratios95, ratios99
    else:
        # The prescribed native 7x7 morphology leaves no eroded interior on
        # this 37x37 cohort. Keep the requested ratios null and rank stages
        # using raw near tails as a transparent geometry-limited fallback.
        severity_metric = "geometry_limited_raw_near_p95_and_near_p99_fallback_no_native_interior"
        severity95 = np.asarray([r["near_p95"] for r in stage_rows if r["near_p95"] is not None], dtype=float)
        severity99 = np.asarray([r["near_p99"] for r in stage_rows if r["near_p99"] is not None], dtype=float)
    lo95, hi95 = float(severity95.min()), float(severity95.max())
    lo99, hi99 = float(severity99.min()), float(severity99.max())
    severity_rows = []
    for row in stage_rows:
        a = row["near_p95_over_interior_p95"] if ratio_available else row["near_p95"]
        b = row["near_p99_over_interior_p99"] if ratio_available else row["near_p99"]
        n95 = 0.0 if hi95 == lo95 else (a - lo95) / (hi95 - lo95)
        n99 = 0.0 if hi99 == lo99 else (b - lo99) / (hi99 - lo99)
        severity_rows.append({"checkpoint": "SAFE_ANCHOR_E10", "stage": row["stage"], "near_p95_over_interior_p95": row["near_p95_over_interior_p95"], "near_p99_over_interior_p99": row["near_p99_over_interior_p99"], "raw_near_p95": row["near_p95"], "raw_near_p99": row["near_p99"], "normalized_p95": n95, "normalized_p99": n99, "S_stage": .5 * (n95 + n99), "severity_metric": severity_metric})
    severity_rows.sort(key=lambda row: row["S_stage"], reverse=True)
    tri_rows_with_kind = list(tri_rows) + severity_rows
    write_csv(OUT_TRI_CSV, tri_rows_with_kind)
    tri_json = json.loads(OUT_TRI_JSON.read_text()) if OUT_TRI_JSON.exists() else {}
    tri_json["stage_severity"] = {"formula": "0.5*minmax_normalized(near_p95/interior_p95)+0.5*minmax_normalized(near_p99/interior_p99), within Safe-Anchor E10 and this cohort only", "ratio_available": ratio_available, "geometry_limited_fallback": None if ratio_available else severity_metric, "rows_ranked": severity_rows}

    final_row = next(row for row in tri_rows if row["checkpoint"] == "SAFE_ANCHOR_E10")
    distance_final = next(row for row in distance_struct["summaries"] if row["map"] == "final_fused")
    distance_1 = next(row for row in json.loads(OUT_DISTANCE_JSON.read_text())["rows"] if row["map"] == "final_fused" and row["bin"] == "1")
    distance_34 = next(row for row in json.loads(OUT_DISTANCE_JSON.read_text())["rows"] if row["map"] == "final_fused" and row["bin"] == "3-4")
    interp_rows = interp["rows"]
    native_ratio_values = [float(r["native_near_p99_over_interior_p99"]) for r in interp_rows if r["native_near_p99_over_interior_p99"] is not None]
    resized_ratio_values = [float(r["resized_near_p99_over_interior_p99"]) for r in interp_rows if r["resized_near_p99_over_interior_p99"] is not None]
    native_max = max(native_ratio_values) if native_ratio_values else None
    resized_max = max(resized_ratio_values) if resized_ratio_values else None
    fusion_max_stage = max(float(r["near_p99"]) for r in fusion["rows"][:3] if r["near_p99"] is not None)
    fusion_delta = float(final_row["near_p99"] - fusion_max_stage)
    # Fixed descriptive rubric, stated explicitly in the artifact. It is not a
    # tuned decision threshold and does not authorize an intervention.
    boundary_artifact_possible = bool(distance_1.get("mean") is not None and distance_34.get("mean") is not None and distance_34["mean"] <= 0.25 * max(distance_1["mean"], 1e-12) and distance_final.get("distance_spearman") is not None and distance_final["distance_spearman"] > -0.05)
    if native_max is not None and resized_max is not None:
        interpolation_likely = bool(resized_max - native_max > 0.10 and native_max < 1.0)
    else:
        interpolation_likely = bool(sum(float(r["resized_near_p95"]) > float(r["native_near_p95"]) + 0.10 and float(r["native_near_p95"]) < 0.10 for r in interp_rows) >= 2)
    stage_severity_gap = float(severity_rows[0]["S_stage"] - severity_rows[1]["S_stage"]) if len(severity_rows) > 1 else 0.0
    stage_local_likely = bool(stage_severity_gap > 0.10)
    fusion_likely = bool(fusion_delta > 0.01 or float(final_row.get("near_p95", 0.0)) > max(float(r["near_p95"]) for r in fusion["rows"][:3] if r["near_p95"] is not None) + 0.01)
    near_far_local = bool(final_row.get("near_p95") is not None and final_row.get("far_p95") is not None and final_row["near_p95"] > final_row["far_p95"] and final_row.get("near_p99") is not None and final_row.get("far_p99") is not None and final_row["near_p99"] > final_row["far_p99"])
    global_calibration = bool(final_row.get("near_p95") is not None and final_row.get("far_p95") is not None and abs(final_row["near_p95"] - final_row["far_p95"]) < 0.01)
    persistent = bool(distance_34.get("mean") is not None and distance_1.get("mean") is not None and distance_34["mean"] > distance_1["mean"] * 0.25)
    robust = bool((image_struct.get("anomalous_image_fraction") or 0.0) >= 0.25 and (image_struct.get("category_fraction_majority_rule") or 0.0) >= 0.25)
    native_local = bool(((native_max is not None and native_max > 1.0) or (native_max is None and max(float(r["near_p99"]) for r in stage_rows) > 2.0 * max(float(r["far_p99"]) for r in stage_rows))) and persistent)
    interpolation_minor = bool(any(float(r["delta_near_p95_resized_minus_native"]) > 0.005 for r in interp_rows))
    if boundary_artifact_possible:
        primary = "BOUNDARY_ANNOTATION_ARTIFACT_POSSIBLE"
    elif interpolation_likely:
        primary = "UPSAMPLING_AMPLIFICATION_LIKELY"
    elif stage_local_likely and native_local:
        primary = "STAGE_LOCAL_SPILLOVER_LIKELY"
    elif fusion_likely:
        primary = "FUSION_SPILLOVER_LIKELY"
    elif global_calibration and not near_far_local:
        primary = "GLOBAL_CALIBRATION_PROBLEM"
    elif near_far_local and persistent and native_local and robust:
        primary = "TRUE_LOCAL_BOUNDARY_SPILLOVER_SUPPORTED"
    else:
        primary = "NO_CLEAR_SPATIAL_SPILLOVER"
    confidence = "HIGH" if primary != "NO_CLEAR_SPATIAL_SPILLOVER" and robust and (near_far_local or interpolation_likely or fusion_likely or stage_local_likely) else ("MEDIUM" if primary != "NO_CLEAR_SPATIAL_SPILLOVER" else "LOW")
    boundary_dominant = primary == "BOUNDARY_ANNOTATION_ARTIFACT_POSSIBLE"
    local_authorized = primary in {"TRUE_LOCAL_BOUNDARY_SPILLOVER_SUPPORTED", "STAGE_LOCAL_SPILLOVER_LIKELY"} and not boundary_dominant and robust and (native_local or not interpolation_likely)
    decision = {
        "protocol_id": "H2_BOUNDARY_SPILLOVER_AUDIT_R1",
        "primary_diagnosis": primary,
        "spillover_confidence": confidence,
        "most_responsible_stage": f"STAGE{severity_rows[0]['stage']}" if primary == "STAGE_LOCAL_SPILLOVER_LIKELY" else ("MIXED" if primary == "TRUE_LOCAL_BOUNDARY_SPILLOVER_SUPPORTED" else "NONE"),
        "interpolation_contribution": "MATERIAL" if interpolation_likely else ("MINOR" if interpolation_minor else "NONE"),
        "fusion_contribution": "MATERIAL" if fusion_likely else "NONE",
        "boundary_annotation_artifact_dominant": boundary_dominant,
        "rubric_observations": {"boundary_artifact_possible": boundary_artifact_possible, "upsampling_amplification_likely": interpolation_likely, "stage_local_spillover_likely": stage_local_likely, "fusion_spillover_likely": fusion_likely, "global_calibration_problem": global_calibration, "near_far_local_signature": near_far_local, "native_persistence_beyond_1px": persistent, "robust_multiple_images_categories": robust},
        "thresholds_used_for_labels": {"boundary_artifact_3_4_mean_le_25pct_1px_and_spearman_gt_-0.05": True, "interpolation_ratio_amplification_gt_0.10_and_native_ratio_lt_1": True, "stage_S_gap_gt_0.10": True, "fusion_near_p99_delta_gt_0.01": True, "global_near_far_absolute_p95_delta_lt_0.01": True, "robust_image_fraction_ge_0.25_and_category_fraction_ge_0.25": True},
        "local_boundary_contrast_justified": "YES" if local_authorized else "NO",
        "inference_resampling_diagnostic_justified": "YES" if primary == "UPSAMPLING_AMPLIFICATION_LIKELY" else "NO",
        "fusion_diagnostic_justified": "YES" if primary == "FUSION_SPILLOVER_LIKELY" else "NO",
        "return_to_representation_diagnosis": "YES" if primary == "NO_CLEAR_SPATIAL_SPILLOVER" else "NO",
        "automatic_intervention_executed": "NO",
        "training_executed": "NO",
        "optimizer_step_executed": "NO",
        "backward_executed": "NO",
        "new_loss_or_regularizer": "NO",
        "medical_evaluation_executed": "NO",
        "mvtec_evaluation_executed": "NO",
        "target_inference_or_tuning": "NO",
        "epoch_selection_or_sweep": "NO",
        "morphology_radius_sweep": "NO",
        "threshold_or_topk_sweep": "NO",
        "waiting_for_user_approval": "YES",
    }
    tri_json["decision"] = decision
    dump_json(OUT_TRI_JSON, tri_json)
    markdown = "\n".join([
        "# H2 Boundary Spillover R1 Decision", "", "Audit-only conclusion", "",
        f"* **PRIMARY_DIAGNOSIS:** `{primary}`",
        f"* **SPILLOVER_CONFIDENCE:** `{confidence}`",
        f"* **MOST_RESPONSIBLE_STAGE:** `{decision['most_responsible_stage']}`",
        f"* **INTERPOLATION_CONTRIBUTION:** `{decision['interpolation_contribution']}`",
        f"* **FUSION_CONTRIBUTION:** `{decision['fusion_contribution']}`",
        f"* **BOUNDARY_ANNOTATION_ARTIFACT_DOMINANT:** `{decision['boundary_annotation_artifact_dominant']}`", "",
        "## Authorized next-step flags", "",
        f"* **LOCAL_BOUNDARY_CONTRAST_JUSTIFIED:** `{decision['local_boundary_contrast_justified']}`",
        f"* **INFERENCE_RESAMPLING_DIAGNOSTIC_JUSTIFIED:** `{decision['inference_resampling_diagnostic_justified']}`",
        f"* **FUSION_DIAGNOSTIC_JUSTIFIED:** `{decision['fusion_diagnostic_justified']}`",
        f"* **RETURN_TO_REPRESENTATION_DIAGNOSIS:** `{decision['return_to_representation_diagnosis']}`", "",
        "The decision is descriptive and uses the fixed cohort, fixed existing mask semantics, fixed current score path, and fixed endpoint-only comparisons. No intervention was implemented automatically. User approval is required before any future Local Boundary Contrast or other diagnostic is run.", "",
        "## Primary endpoint headline", "",
        f"Safe-Anchor E10 final near-background p95={final_row.get('near_p95')}, p99={final_row.get('near_p99')}; far-background p95={final_row.get('far_p95')}, p99={final_row.get('far_p99')}; final distance Spearman={distance_final.get('distance_spearman')}; image leakage fraction={image_struct.get('anomalous_image_fraction')}; category majority-rule fraction={image_struct.get('category_fraction_majority_rule')}.", "",
        "## Prohibitions", "",
        "TRAINING_EXECUTED=NO; OPTIMIZER_STEP_EXECUTED=NO; BACKWARD_EXECUTED=NO; NEW_LOSS_OR_REGULARIZER=NO; MEDICAL_EVALUATION_EXECUTED=NO; MVTec_EVALUATION_EXECUTED=NO; TARGET_INFERENCE_OR_TUNING=NO; EPOCH_SELECTION_OR_SWEEP=NO; MORPHOLOGY_RADIUS_SWEEP=NO; THRESHOLD_OR_TOPK_SWEEP=NO; AUTOMATIC_INTERVENTION_EXECUTED=NO.", "",
        "WAITING_FOR_USER_APPROVAL=YES",
    ]) + "\n"
    OUT_DECISION_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_DECISION_MD.write_text(markdown)
    dump_json(OUT_DECISION_JSON, decision)
    return decision, tri_json


def triangulation_phase(model: ACDCLIP, baseline: dict, rows: list[dict], device: torch.device, distance_struct: dict, interp_struct: dict, fusion_struct: dict, image_struct: dict) -> tuple[list[dict], dict]:
    endpoints = [("SAFE_ANCHOR_E10", SAFE_ANCHOR), ("LATE_FREEZE_STAGE23", LATE_FREEZE), ("THBR_CONTROL", THBR_CONTROL), ("THBR_CANDIDATE", THBR_CANDIDATE)]
    tri_rows = []
    for label, path in endpoints:
        if not path.is_file():
            raise FileNotFoundError(f"missing read-only comparison endpoint {path}")
        if label == "SAFE_ANCHOR_E10":
            data = baseline
        else:
            data = evaluate_maps(model, path, rows, device, full=False)
        metrics = fusion_map_metrics(data["resized_fused_prob"], data["mask"], data["labels"], "final_production_fused")
        profile_rows, summary = distance_rows(data["resized_fused_prob"], data["mask"], label, "518x518")
        one_px = next(row for row in profile_rows if row["bin"] == "1")
        tri_rows.append({"kind": "endpoint_comparison", "checkpoint": label, "checkpoint_path": str(path), "checkpoint_sha256": sha256_file(path),
                         "auroc": metrics["auroc"], "ap": metrics["ap"], "interior_mean": metrics["interior_mean"], "boundary_mean": metrics["boundary_mean"],
                         "near_p95": metrics["near_p95"], "near_p99": metrics["near_p99"], "far_p95": metrics["far_p95"], "far_p99": metrics["far_p99"],
                         "near_gt_positive_inversion": metrics["near_gt_positive_inversion"], "near_gt_interior_inversion": metrics["near_gt_interior_inversion"],
                         "distance_spearman": summary["distance_spearman"], "distance_linear_slope": summary["distance_linear_slope"], "distance_1px_mean": one_px["mean"]})
    tri_json = {"protocol_id": "H2_BOUNDARY_SPILLOVER_AUDIT_R1", "cohort_count": 96, "endpoint_rows": tri_rows,
                "comparison_scope": "same frozen cohort, descriptive endpoint inference only; no endpoint selection or target tuning",
                "evidence_refs": {"safe_anchor": "research/h2-safe-anchor-e20-medical-selected", "late_freeze": "research/h2-late-convlora-freeze-r1", "thbr": "research/h2-thbr-r1"}}
    dump_json(OUT_TRI_JSON, tri_json)
    decision, complete = severity_and_decision(baseline, interp_struct, fusion_struct, image_struct, tri_rows, distance_struct)
    complete["endpoint_rows"] = tri_rows
    dump_json(OUT_TRI_JSON, complete)
    return tri_rows, complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("provenance", "native", "distance", "interpolation", "fusion", "robustness", "triangulation", "all"), required=True)
    args = parser.parse_args()
    rows, by_category = load_rows()
    if args.phase in {"provenance", "all"}:
        make_cohort_artifact(rows, by_category)
        write_semantics_and_provenance()
    if args.phase == "provenance":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the historical inference-only score-map audit")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    model = make_model(device)
    if args.phase in {"native", "all"}:
        baseline = evaluate_maps(model, SAFE_ANCHOR, rows, device, full=True)
        save_baseline_cache(baseline)
        native_stage_phase(baseline)
    else:
        baseline = load_baseline_cache()
    if args.phase in {"distance", "all"}:
        distance_phase(baseline)
    if args.phase in {"interpolation", "all"}:
        interpolation_phase(baseline)
    if args.phase in {"fusion", "all"}:
        fusion_phase(baseline)
    if args.phase in {"robustness", "all"}:
        per_image_phase(baseline)
    if args.phase in {"triangulation", "all"}:
        if not OUT_DISTANCE_JSON.is_file():
            distance_phase(baseline)
        if not OUT_INTERP_JSON.is_file():
            interpolation_phase(baseline)
        if not OUT_FUSION_JSON.is_file():
            fusion_phase(baseline)
        if not OUT_IMAGE_CSV.is_file():
            _, _, image_struct = per_image_phase(baseline)
        else:
            image_struct = json.loads(OUT_IMAGE_CSV.with_suffix(".json").read_text())
        _, distance_struct = distance_phase(baseline) if not OUT_DISTANCE_JSON.is_file() else ([], json.loads(OUT_DISTANCE_JSON.read_text()))
        _, interp_struct = interpolation_phase(baseline) if not OUT_INTERP_JSON.is_file() else ([], json.loads(OUT_INTERP_JSON.read_text()))
        _, fusion_struct = fusion_phase(baseline) if not OUT_FUSION_JSON.is_file() else ([], json.loads(OUT_FUSION_JSON.read_text()))
        triangulation_phase(model, baseline, rows, device, distance_struct, interp_struct, fusion_struct, image_struct)


if __name__ == "__main__":
    main()
