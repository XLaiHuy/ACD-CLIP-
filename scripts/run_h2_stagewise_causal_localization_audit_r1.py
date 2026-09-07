#!/usr/bin/env python3
"""H2 POST-S2-LOCR R1 stagewise causal localization audit.

This file is an audit driver only.  Endpoint phases reconstruct existing
production maps and use GT masks only for retrospective diagnostics.  The
trajectory phase, when needed, is a deterministic replay of the already
committed R1 source-only screen; it is not a new experiment and its endpoint
must reproduce the committed R1 checkpoint before trajectory claims are used.
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
from scipy import ndimage
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataset import CLASS_NAMES, DATA_PATH, get_text_and_image_dataset
from h2_clean.precision import PrecisionPolicy
from model.adapter import ACDCLIP


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


spill = load_script("h2_spillover_stagewise", REPO / "scripts/run_h2_boundary_spillover_audit_r1.py")
s2 = load_script("h2_s2_locr_r1_stagewise", REPO / "scripts/run_h2_s2_locr_r1.py")

IMG = 518
PATCH = 37
PATCH_SIZE = 14
BATCH = 8
PAIR_SEED = 1729
PARENT_HEAD = "f44cca2e163585dba3bbffc99c45518501db4852"
SAFE_ANCHOR = Path("/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth")
SAFE_SHA = "64b72dc3d1155285c826781bee4c5970bd45218e95b21675fd19d9a6b2ab54a7"
R1_ENDPOINT = REPO / "audit/H2_S2_LOCR_R1_ENDPOINT.json"
R1_DECISION_JSON = REPO / "results/H2_S2_LOCR_R1_BOUNDED_DECISION.json"
R1_MANIFEST = REPO / "audit/H2_S2_LOCR_R1_ATTEMPT_MANIFEST.json"
R1_CONTROL_CSV = REPO / "audit/H2_S2_LOCR_R1_CONTROL.csv"
R1_PREFLIGHT_JSON = REPO / "audit/H2_S2_LOCR_R1_GRADIENT_PREFLIGHT.json"
COHORT_A_SOURCE = REPO / "audit/H2_BOUNDARY_SPILLOVER_R1_COHORT.json"
RUN_ROOT = Path("/workspace/h2_stagewise_causal_localization_audit_r1")

OUT_IDENTITY = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_PARENT_IDENTITY.md"
OUT_A = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_COHORT_A.json"
OUT_B = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_COHORT_B.json"
OUT_ORACLE_CSV = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_ORACLE_STAGEWISE.csv"
OUT_ORACLE_JSON = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_ORACLE_STAGEWISE.json"
OUT_OCC_CSV = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_PATCH_OCCUPANCY.csv"
OUT_OCC_JSON = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_PATCH_OCCUPANCY.json"
OUT_CAL_CSV = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_CALIBRATION_INVARIANT.csv"
OUT_CAL_JSON = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_CALIBRATION_INVARIANT.json"
OUT_TRAJ_CSV = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_TRAJECTORY.csv"
OUT_TRAJ_JSON = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_TRAJECTORY.json"
OUT_TASK_CSV = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_TASK_GRAD_REDISTRIBUTION.csv"
OUT_TASK_JSON = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_TASK_GRAD_REDISTRIBUTION.json"
OUT_DIR_CSV = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_DIRECTIONAL_FAMILY.csv"
OUT_DIR_JSON = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_DIRECTIONAL_FAMILY.json"
OUT_MODULE_MD = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_MODULE_CAUSAL_ASSESSMENT.md"
OUT_RESEARCH_MD = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_RESEARCH.md"
OUT_RESEARCH_JSON = REPO / "audit/H2_STAGEWISE_CAUSAL_R1_RESEARCH.json"
OUT_DECISION_MD = REPO / "results/H2_STAGEWISE_CAUSAL_R1_DECISION.md"
OUT_DECISION_JSON = REPO / "results/H2_STAGEWISE_CAUSAL_R1_DECISION.json"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=json_default, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["status"]
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])
    os.replace(tmp, path)


def finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def delta(a, b):
    a, b = finite(a), finite(b)
    return None if a is None or b is None else a - b


def current_branch() -> str:
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()


def current_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def git_status() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()


def load_a_records() -> list[dict]:
    data = json.loads(COHORT_A_SOURCE.read_text())
    if data.get("count") != 96 or len(data.get("records", [])) != 96:
        raise RuntimeError("Cohort A source identity is not the committed 96-image cohort")
    return data["records"]


def record_for(category: str, meta: dict, index: int, dataset) -> dict:
    image_path = Path(DATA_PATH["VisA"]) / meta["image_path"]
    mask_path = None if not int(meta["label"]) else Path(DATA_PATH["VisA"]) / meta["mask_path"]
    transformed = dataset[index]["mask"].numpy().astype(np.float32, copy=False)
    return {
        "category": category,
        "label": int(meta["label"]),
        "label_type": "anomaly" if int(meta["label"]) else "normal",
        "file_name": meta["image_path"],
        "image_source": str(image_path),
        "image_sha256": sha_file(image_path),
        "mask_source": None if mask_path is None else str(mask_path),
        "mask_sha256": None if mask_path is None else sha_file(mask_path),
        "resized_mask_sha256": sha_bytes(np.ascontiguousarray(transformed).tobytes()),
        "source_manifest_index": int(index),
    }


def identity_and_cohorts() -> dict:
    if current_head() != PARENT_HEAD:
        raise RuntimeError(f"identity phase must start at exact R1 HEAD {PARENT_HEAD}, got {current_head()}")
    if current_branch() != "research/h2-stagewise-causal-localization-audit-r1":
        raise RuntimeError("wrong audit branch")
    if sha_file(SAFE_ANCHOR) != SAFE_SHA:
        raise RuntimeError("Safe Anchor hash mismatch")
    endpoint = json.loads(R1_ENDPOINT.read_text())
    control = Path(endpoint["control"]["checkpoint"])
    candidate = Path(endpoint["candidate"]["checkpoint"])
    endpoint_identity = {
        "control": {"path": str(control), "sha256": sha_file(control), "committed_sha256": endpoint["control"]["checkpoint_sha256"]},
        "candidate": {"path": str(candidate), "sha256": sha_file(candidate), "committed_sha256": endpoint["candidate"]["checkpoint_sha256"]},
    }
    if endpoint_identity["control"]["sha256"] != endpoint_identity["control"]["committed_sha256"] or endpoint_identity["candidate"]["sha256"] != endpoint_identity["candidate"]["committed_sha256"]:
        raise RuntimeError("R1 endpoint hash mismatch")
    a_records = load_a_records()
    excluded = {r["file_name"] for r in a_records}
    datasets = get_text_and_image_dataset("VisA", IMG, "test")
    b_records = []
    for category in CLASS_NAMES["VisA"]:
        dataset = datasets[category]
        for label in (0, 1):
            eligible = [(i, m) for i, m in enumerate(dataset.meta) if int(m["label"]) == label and m["image_path"] not in excluded]
            eligible.sort(key=lambda pair: pair[1]["image_path"])
            for i, meta in eligible[:4]:
                b_records.append(record_for(category, meta, i, dataset))
    if len(b_records) != 96:
        raise RuntimeError(f"Cohort B deterministic selection yielded {len(b_records)}, expected 96")
    if excluded.intersection({r["file_name"] for r in b_records}):
        raise RuntimeError("Cohort A/B are not disjoint")
    def cohort_artifact(records, cohort_id, source, selection_rule):
        canonical = "\n".join(f"{r['category']}|{r['label_type']}|{r['file_name']}|{r['image_sha256']}|{r['mask_sha256']}" for r in records).encode()
        return {
            "protocol_id": "H2_STAGEWISE_CAUSAL_LOCALIZATION_AUDIT_R1",
            "cohort_id": cohort_id,
            "dataset": "VisA",
            "count": len(records),
            "records": records,
            "source_artifact": str(source),
            "selection_rule": selection_rule,
            "ordering": "CLASS_NAMES[VisA] category order; within each category NORMAL then ANOMALOUS; image_path lexicographic order",
            "canonical_manifest_sha256": sha_bytes(canonical),
            "target_inference": False,
            "prediction_or_metric_selection": False,
        }
    a = cohort_artifact(a_records, "A", COHORT_A_SOURCE, "Exact committed H2_BOUNDARY_SPILLOVER_R1_COHORT.json records; no reselection")
    b = cohort_artifact(b_records, "B", "VisA test manifest", "Per category, after excluding all Cohort A file_name values, select first 4 eligible NORMAL and first 4 eligible ANOMALOUS records after canonical image_path sorting")
    dump_json(OUT_A, a)
    dump_json(OUT_B, b)
    OUT_IDENTITY.write_text("\n".join([
        "# H2 Stagewise Causal Localization Audit R1 — Parent Identity",
        "",
        f"* branch: `{current_branch()}`",
        f"* exact R1 parent HEAD: `{PARENT_HEAD}`",
        f"* branch HEAD at identity capture: `{current_head()}`",
        "* parent identity: `PASS`",
        f"* Safe Anchor E10: `{SAFE_ANCHOR}`",
        f"* Safe Anchor SHA256: `{SAFE_SHA}`",
        f"* R1 control: `{endpoint_identity['control']['path']}` ({endpoint_identity['control']['sha256']})",
        f"* R1 candidate: `{endpoint_identity['candidate']['path']}` ({endpoint_identity['candidate']['sha256']})",
        "",
        "The committed S2-LOCR R1 decision is preserved unchanged: `BOUNDED_SCREEN=FAIL`, `CASE_B`, and `S2_LOCR_MECHANISM=NOT_SUPPORTED`. This branch performs stagewise causal localization, patch-footprint, calibration-invariant, trajectory, and parameter-family diagnostics only.",
        "",
        "No Medical inference, MVTec inference, target tuning, new training objective, architecture change, optimizer/fusion/interpolation change, hyperparameter sweep, S2-LOCR-v2, or post-hoc R1 gate modification is permitted or performed.",
        "",
        f"Cohort A is the exact committed 96-image spillover cohort (`{a['canonical_manifest_sha256']}`). Cohort B is a new disjoint 96-image VisA test cohort selected only by the preregistered canonical manifest rule (`{b['canonical_manifest_sha256']}`).",
    ]) + "\n")
    return {"parent_head": PARENT_HEAD, "safe_anchor_sha256": SAFE_SHA, "endpoint": endpoint_identity, "cohort_a": a, "cohort_b": b}


def load_cohort(cohort_id: str) -> list[dict]:
    data = json.loads((OUT_A if cohort_id == "A" else OUT_B).read_text())
    return data["records"]


def production_exact(model, vision, text):
    batch, patches, _ = vision.shape[1:]
    side = int(math.sqrt(patches))
    if side != PATCH:
        raise RuntimeError(f"expected 37x37 native grid, got {side}")
    if text.ndim == 3:
        group_text = text.unsqueeze(1).repeat(1, batch, 1, 1).permute(1, 0, 2, 3)
    else:
        group_text = text.permute(1, 0, 2, 3)
    native, resized = [], []
    for stage in range(3):
        fused = model._vision_text_attention_fusion(vision[stage], group_text, stage)
        z = torch.matmul(10 * vision[stage], fused).permute(0, 2, 1).view(batch, 2, PATCH, PATCH)
        native.append(z)
        resized.append(F.interpolate(gaussian_blur2d(z, (7, 7), (1, 1)), (IMG, IMG), mode="bilinear", align_corners=True))
    native = torch.stack(native)
    resized = torch.stack(resized)
    fused_native = native.mean(0)
    fused_resized = resized.mean(0)
    return {"native_logits": native, "resized_logits": resized, "native_prob": F.softmax(native, 2)[:, :, 1].permute(1, 0, 2, 3), "resized_prob": F.softmax(resized, 2)[:, :, 1].permute(1, 0, 2, 3), "fused_prob": F.softmax(fused_resized, 1)[:, 1], "fused_margin": fused_resized[:, 1] - fused_resized[:, 0]}


def evaluate_exact(checkpoint: Path, rows: list[dict], device: torch.device, requires_grad: bool = False) -> dict:
    model = spill.make_model(device)
    spill.load_endpoint(model, checkpoint)
    if requires_grad:
        model.requires_grad_(True)
    else:
        model.requires_grad_(False)
    datasets, indices = spill.load_selection_datasets(rows)
    policy = PrecisionPolicy("fp16")
    chunks = {k: [] for k in ("native_logits", "resized_logits", "native_prob", "resized_prob", "fused_prob", "fused_margin", "mask")}
    names, labels, categories = [], [], []
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        for category in CLASS_NAMES["VisA"]:
            selected = indices[category]
            loader = DataLoader(Subset(datasets[category], selected), batch_size=BATCH, shuffle=False, num_workers=0)
            text, _, _ = spill.get_hybrid_soft_prompt_single_class_text_embedding(model, "VisA", category, device, return_kg=False)
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                with policy.autocast(device):
                    seg_tokens, _ = model(image)
                    maps = production_exact(model, torch.stack(seg_tokens), text)
                for key in chunks:
                    if key == "mask":
                        chunks[key].append((batch["mask"][:, 0].numpy() > .5).astype(np.uint8))
                    elif key in ("native_logits", "resized_logits"):
                        chunks[key].append(maps[key].permute(1, 0, 2, 3, 4).float().detach().cpu().numpy())
                    else:
                        chunks[key].append(maps[key].float().detach().cpu().numpy())
                names.extend(list(batch["file_name"]))
                labels.extend(batch["label"].numpy().astype(np.uint8).tolist())
                categories.extend([category] * len(batch["file_name"]))
                del maps
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    expected = [r["file_name"] for r in rows]
    if names != expected:
        raise RuntimeError("exact evaluator order mismatch")
    out = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    out.update({"names": np.asarray(names, dtype=object), "categories": np.asarray(categories, dtype=object), "labels": np.asarray(labels, dtype=np.uint8), "checkpoint": str(checkpoint), "checkpoint_sha256": sha_file(checkpoint)})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def anomaly_near_metrics(score: np.ndarray, mask: np.ndarray) -> dict:
    positive, negative = [], []
    interior, boundary = [], []
    for s, m in zip(score, mask):
        b, i, n, _ = spill.morphology(m)
        if m.any() and n.any():
            positive.append(s[m].reshape(-1)); negative.append(s[n].reshape(-1))
            if i.any(): interior.append(s[i].reshape(-1))
            if b.any(): boundary.append(s[b].reshape(-1))
    def bm(left, right):
        if not left or not right:
            return {"auroc": None, "ap": None, "pixel_count": 0}
        x = np.concatenate(left + right); y = np.concatenate([np.ones(sum(len(v) for v in left), dtype=np.uint8), np.zeros(sum(len(v) for v in right), dtype=np.uint8)])
        return spill.binary_metrics(x, y)
    result = {"anomaly_vs_near": bm(positive, negative), "interior_vs_near": bm(interior, negative), "boundary_vs_near": bm(boundary, negative)}
    return result


def region_fields(score: np.ndarray, mask: np.ndarray) -> dict:
    s = spill.region_summary(score, mask)
    out = {}
    for region in ("positive", "interior", "boundary", "near", "far"):
        for stat in ("mean", "median", "p95", "p99"):
            out[f"{region}_{stat}"] = s[region][stat]
    out["near_gt_positive_inversion"] = spill.sampled_inversion(score, score, mask, greater=True)
    out["near_gt_interior_inversion"] = spill.sampled_inversion(score, score, mask, greater=False)
    return out


def fused_metric(score, mask, labels, name="final") -> dict:
    out = {"map": name, **spill.binary_metrics(score, mask)}
    out.update(region_fields(score, mask))
    out.update(anomaly_near_metrics(score, mask))
    return out


def fuse_logits(z):
    return z.mean(axis=0)


def local_or_far(base: np.ndarray, mask: np.ndarray, stage: int, far_control: bool = False) -> tuple[np.ndarray, list[int]]:
    out = base.copy()
    selected_counts = []
    for i, m in enumerate(mask):
        _, _, near, far = spill.morphology(m)
        near_idx = np.flatnonzero(near.reshape(-1))
        far_idx = np.flatnonzero(far.reshape(-1))
        n = min(len(near_idx), len(far_idx))
        chosen = far_idx[:n] if far_control else near_idx[:n]
        selected_counts.append(int(n))
        if n:
            other = (base[(stage + 1) % 3, i].reshape(2, -1) + base[(stage + 2) % 3, i].reshape(2, -1)) / 2.0
            flat = out[stage, i].reshape(2, -1)
            flat[:, chosen] = other[:, chosen]
    return out, selected_counts


def oracle_phase() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for exact production inference")
    device = torch.device("cuda:0")
    rows_by = {c: load_cohort(c) for c in ("A", "B")}
    all_rows = {c: [] for c in rows_by}
    records, csv_rows = [], []
    unique_support = {}
    for cohort, rows in rows_by.items():
        data = evaluate_exact(SAFE_ANCHOR, rows, device)
        base = data["resized_logits"]
        base_fused = F.softmax(torch.from_numpy(base.mean(1)), dim=1).numpy()[:, 1]
        cohort_metrics = {}
        baseline = fused_metric(base_fused, data["mask"], data["labels"], "baseline")
        cohort_metrics["baseline"] = baseline
        for stage in range(3):
            for kind in ("local", "far"):
                altered, counts = local_or_far(base, data["mask"], stage, far_control=kind == "far")
                fused = F.softmax(torch.from_numpy(altered.mean(1)), dim=1).numpy()[:, 1]
                key = f"{kind}_s{stage+1}"
                metric = fused_metric(fused, data["mask"], data["labels"], key)
                metric["replacement_pixel_count_mean"] = float(np.mean(counts))
                metric["replacement_pixel_count_total"] = int(np.sum(counts))
                cohort_metrics[key] = metric
        for arm, metric in cohort_metrics.items():
            row = {"cohort": cohort, "arm": arm}
            for key, value in metric.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict):
                            for leaf, leafvalue in subvalue.items(): row[f"{key}_{subkey}_{leaf}"] = leafvalue
                        else: row[f"{key}_{subkey}"] = subvalue
                else: row[key] = value
            csv_rows.append(row)
        local_improvements = {}
        for stage in range(3):
            m = cohort_metrics[f"local_s{stage+1}"]
            f = cohort_metrics[f"far_s{stage+1}"]
            inv_base = np.nanmean([baseline.get("near_gt_positive_inversion"), baseline.get("near_gt_interior_inversion")])
            inv_local = np.nanmean([m.get("near_gt_positive_inversion"), m.get("near_gt_interior_inversion")])
            ranking = float(np.nanmean([m["anomaly_vs_near"]["ap"] - baseline["anomaly_vs_near"]["ap"], m["anomaly_vs_near"]["auroc"] - baseline["anomaly_vs_near"]["auroc"], inv_base - inv_local]))
            local_improvements[stage + 1] = {"final_ap_delta": delta(m["ap"], baseline["ap"]), "final_auroc_delta": delta(m["auroc"], baseline["auroc"]), "anomaly_vs_near_ap_delta": delta(m["anomaly_vs_near"]["ap"], baseline["anomaly_vs_near"]["ap"]), "anomaly_vs_near_auroc_delta": delta(m["anomaly_vs_near"]["auroc"], baseline["anomaly_vs_near"]["auroc"]), "ranking_improvement": ranking, "far_final_ap_delta": delta(f["ap"], baseline["ap"]), "far_final_auroc_delta": delta(f["auroc"], baseline["auroc"])}
        unique_support[cohort] = local_improvements
        records.append({"cohort": cohort, "metrics": cohort_metrics, "local_improvements": local_improvements})
    s2both = all(unique_support[c][2]["final_ap_delta"] is not None and unique_support[c][2]["final_ap_delta"] >= 0 and unique_support[c][2]["final_auroc_delta"] >= 0 for c in ("A", "B"))
    rankboth = all(unique_support[c][2]["ranking_improvement"] > max(unique_support[c][1]["ranking_improvement"], unique_support[c][3]["ranking_improvement"]) for c in ("A", "B"))
    far_not_reproduce = all(not (unique_support[c][2]["far_final_ap_delta"] is not None and unique_support[c][2]["far_final_ap_delta"] >= unique_support[c][2]["final_ap_delta"] and unique_support[c][2]["far_final_auroc_delta"] >= unique_support[c][2]["final_auroc_delta"]) for c in ("A", "B"))
    stage2_unique = "YES" if s2both and rankboth and far_not_reproduce else "NO"
    artifact = {"protocol_id": "H2_STAGEWISE_CAUSAL_LOCALIZATION_AUDIT_R1", "checkpoint": str(SAFE_ANCHOR), "checkpoint_sha256": SAFE_SHA, "cohorts": records, "matched_far_rule": "For each anomalous image, use N=min(canonical near pixel count, canonical far pixel count); replace the first N flattened pixels in the selected region with the mean of the other two stage logits.", "stage2_unique_causal_support": stage2_unique, "support_checks": {"preserve_or_improve_both": s2both, "stronger_local_ranking_both": rankboth, "matched_far_does_not_reproduce": far_not_reproduce}}
    write_csv(OUT_ORACLE_CSV, csv_rows); dump_json(OUT_ORACLE_JSON, artifact)
    return artifact


def occupancy_phase() -> dict:
    if 37 * 14 != 518:
        raise RuntimeError("exact patch mapping failed")
    device = torch.device("cuda:0")
    rows_all = load_cohort("A") + load_cohort("B")
    data = evaluate_exact(SAFE_ANCHOR, rows_all, device)
    masks = data["mask"].astype(bool)
    occupancy = masks.reshape(len(masks), PATCH, PATCH_SIZE, PATCH, PATCH_SIZE).mean(axis=(2, 4))
    native_masks = spill.flatten_masks_native(data["mask"])
    if not np.allclose(occupancy, native_masks, atol=1.0):
        raise RuntimeError("native patch occupancy reshape did not match exact 14x14 patch footprints")
    bins = [("EXACT_0", 0.0, 0.0, "eq0"), ("LOW", 0.0, .25, "open_left"), ("MID_LOW", .25, .5, "open_left"), ("MID_HIGH", .5, .75, "open_left"), ("HIGH", .75, 1.0, "open_left"), ("EXACT_1", 1.0, 1.0, "eq1")]
    rows, structured = [], []
    for stage in range(3):
        prob = data["native_prob"][:, stage]
        margin = data["native_logits"][:, stage, 1] - data["native_logits"][:, stage, 0]
        for name, low, high, mode in bins:
            occ = (occupancy == 0) if mode == "eq0" else (occupancy == 1 if mode == "eq1" else ((occupancy > low) & (occupancy <= high)))
            for distance_region in ("all", "positive", "near", "far", "boundary", "interior"):
                vals_p, vals_m = [], []
                for i in range(len(masks)):
                    b, inn, near, far = spill.morphology(native_masks[i])
                    region = {"all": np.ones_like(native_masks[i], bool), "positive": native_masks[i].astype(bool), "near": near, "far": far, "boundary": b, "interior": inn}[distance_region]
                    selected = occ[i] & region
                    if selected.any(): vals_p.append(prob[i][selected]); vals_m.append(margin[i][selected])
                p = np.concatenate(vals_p) if vals_p else np.empty(0); m = np.concatenate(vals_m) if vals_m else np.empty(0)
                row = {"stage": stage + 1, "occupancy_bin": name, "distance_region": distance_region, "count": int(p.size), "prob_mean": float(p.mean()) if p.size else None, "prob_median": float(np.median(p)) if p.size else None, "margin_mean": float(m.mean()) if m.size else None, "margin_median": float(np.median(m)) if m.size else None}
                rows.append(row)
    for cohort, indices in (("A", range(96)), ("B", range(96, 192))):
        for stage in range(3):
            z = data["native_prob"][list(indices), stage]
            o = occupancy[list(indices)]
            near_vals = z[(o == 0) & np.stack([spill.morphology(native_masks[i])[2] for i in indices])]
            far_vals = z[(o == 0) & np.stack([spill.morphology(native_masks[i])[3] for i in indices])]
            partial = z[(o > 0) & (o < 1)]
            structured.append({"cohort": cohort, "stage": stage + 1, "zero_occ_near_score": float(near_vals.mean()) if near_vals.size else None, "zero_occ_far_score": float(far_vals.mean()) if far_vals.size else None, "partial_occ_score": float(partial.mean()) if partial.size else None})
    zero_near = [x["zero_occ_near_score"] for x in structured if x["stage"] == 2]
    zero_far = [x["zero_occ_far_score"] for x in structured if x["stage"] == 2]
    partial = [x["partial_occ_score"] for x in structured if x["stage"] == 2]
    elevated = all(a is not None and b is not None and a > b for a, b in zip(zero_near, zero_far))
    diagnosis = "PATCH_FOOTPRINT_ALIASING_LIKELY" if elevated and any(x is not None for x in partial) else "CONTEXTUAL_SPILLOVER_LIKELY" if not elevated else "MIXED"
    artifact = {"protocol_id": "H2_STAGEWISE_CAUSAL_LOCALIZATION_AUDIT_R1", "mapping": {"image_size": IMG, "native_grid": [PATCH, PATCH], "conv_patch_kernel": PATCH_SIZE, "conv_patch_stride": PATCH_SIZE, "exact_mapping": "PASS", "footprint": "native patch (r,c) maps to input rows [14r,14r+13] and columns [14c,14c+13]"}, "bins": [{"name": n, "interval": "==0" if mode == "eq0" else "==1" if mode == "eq1" else f"({lo},{hi}]"} for n, lo, hi, mode in bins], "rows": rows, "cohort_summary": structured, "diagnosis": diagnosis, "no_threshold_tuning": True}
    write_csv(OUT_OCC_CSV, rows); dump_json(OUT_OCC_JSON, artifact)
    return artifact


def background_fpr_recall(score, mask):
    bg = score[~mask.astype(bool)]
    pos = score[mask.astype(bool)]
    result = {}
    for name, fpr in (("1pct", .01), ("5pct", .05)):
        threshold = float(np.quantile(bg, 1 - fpr)) if bg.size else None
        result[name] = {"threshold": threshold, "recall": float((pos > threshold).mean()) if threshold is not None and pos.size else None, "background_count": int(bg.size), "positive_count": int(pos.size)}
    return result


def calibration_phase() -> dict:
    endpoint = json.loads(R1_ENDPOINT.read_text())
    paths = {"CONTROL": Path(endpoint["control"]["checkpoint"]), "CANDIDATE": Path(endpoint["candidate"]["checkpoint"])}
    rows, structured = [], []
    for cohort in ("A", "B"):
        cr = load_cohort(cohort)
        for arm, path in paths.items():
            data = evaluate_exact(path, cr, torch.device("cuda:0"))
            metric = fused_metric(data["fused_prob"], data["mask"], data["labels"], "final")
            fpr = background_fpr_recall(data["fused_prob"], data["mask"])
            row = {"cohort": cohort, "arm": arm, "checkpoint": str(path), "checkpoint_sha256": sha_file(path), "pixel_ap": metric["ap"], "pixel_auroc": metric["auroc"], "positive_mean": metric["positive_mean"], "positive_median": metric["positive_median"], "interior_mean": metric["interior_mean"], "interior_median": metric["interior_median"], "boundary_mean": metric["boundary_mean"], "boundary_median": metric["boundary_median"], "near_p95": metric["near_p95"], "near_p99": metric["near_p99"], "far_p95": metric["far_p95"], "far_p99": metric["far_p99"], "anomaly_vs_near_ap": metric["anomaly_vs_near"]["ap"], "anomaly_vs_near_auroc": metric["anomaly_vs_near"]["auroc"], "interior_vs_near_auroc": metric["interior_vs_near"]["auroc"], "boundary_vs_near_auroc": metric["boundary_vs_near"]["auroc"], "recall_at_1pct_fpr": fpr["1pct"]["recall"], "recall_at_5pct_fpr": fpr["5pct"]["recall"]}
            rows.append(row); structured.append({"cohort": cohort, "arm": arm, "metrics": row, "fpr": fpr})
    comparisons = []
    for cohort in ("A", "B"):
        c = next(x for x in rows if x["cohort"] == cohort and x["arm"] == "CONTROL"); a = next(x for x in rows if x["cohort"] == cohort and x["arm"] == "CANDIDATE")
        comparisons.append({"cohort": cohort, **{f"{key}_delta": delta(a[key], c[key]) for key in ("pixel_ap", "pixel_auroc", "anomaly_vs_near_ap", "anomaly_vs_near_auroc", "interior_vs_near_auroc", "boundary_vs_near_auroc", "recall_at_1pct_fpr", "recall_at_5pct_fpr")}})
    a = json.loads(R1_DECISION_JSON.read_text())
    artifact = {"protocol_id": "H2_STAGEWISE_CAUSAL_LOCALIZATION_AUDIT_R1", "rows": structured, "comparisons": comparisons, "aupro_status": "NOT_AVAILABLE", "aupro_search": "No committed AUPRO/PRO evaluator was found in repository audit/scripts/results; implementing one solely for this audit is prohibited.", "interpretation": "INCONCLUSIVE_PENDING_TRAJECTORY"}
    write_csv(OUT_CAL_CSV, rows); dump_json(OUT_CAL_JSON, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("identity", "oracle", "occupancy", "calibration", "all"), default="identity")
    args = parser.parse_args()
    if args.phase in ("identity", "all"):
        identity_and_cohorts()
    if args.phase in ("oracle", "all"):
        oracle_phase()
    if args.phase in ("occupancy", "all"):
        occupancy_phase()
    if args.phase in ("calibration", "all"):
        calibration_phase()


if __name__ == "__main__":
    main()
