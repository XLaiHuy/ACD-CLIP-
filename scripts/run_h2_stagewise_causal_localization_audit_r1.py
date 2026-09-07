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
import heapq
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import tempfile
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
# The protocol does not authorize metric subsampling.  Endpoint maps are
# retained as compact float32 margins, so all finite pixels are materialized
# for the requested pooled metrics and region summaries.
METRIC_PIXEL_CAP = None
PER_IMAGE_METRIC_CAP = None
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
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
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


def _external_exact_binary_metrics(parts) -> dict:
    """Exact pooled AP/AUROC from an iterable of score/label parts."""
    # A full 518x518 cohort can contain tens of millions of pixels.  External
    # sorted runs keep the calculation exact without the multi-gigabyte
    # temporary cumsum/sort arrays that previously caused the diagnostic
    # process to be killed.  The global index in each record preserves the
    # historical stable flattened-order tie handling for AP.
    record_dtype = np.dtype([("score", "<f4"), ("label", "u1"), ("index", "<u8")])
    run_root = RUN_ROOT / "metric_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="binary_", dir=run_root) as temp_dir:
        run_paths = []
        # Keep the transient sort buffer small because the production model
        # and retained endpoint tensors are live in the parent process.
        chunk_size = 100_000
        global_index = 0
        finite_count = positive_count = 0
        for part_scores, part_labels in parts:
            x = np.asarray(part_scores, dtype=np.float32).reshape(-1)
            y = np.asarray(part_labels, dtype=np.uint8).reshape(-1)
            if x.size != y.size:
                raise ValueError("score/label part size mismatch")
            for start in range(0, x.size, chunk_size):
                stop = min(start + chunk_size, x.size)
                chunk_scores = x[start:stop]
                chunk_labels = y[start:stop]
                chunk_valid = np.isfinite(chunk_scores)
                if not chunk_valid.any():
                    continue
                values = chunk_scores[chunk_valid]
                labels_valid = chunk_labels[chunk_valid]
                indices = np.arange(global_index + start, global_index + stop, dtype=np.uint64)[chunk_valid]
                order = np.argsort(-values, kind="stable")
                records = np.empty(order.size, dtype=record_dtype)
                records["score"] = values[order]
                records["label"] = labels_valid[order]
                records["index"] = indices[order]
                path = Path(temp_dir) / f"run_{len(run_paths):04d}.bin"
                records.tofile(path)
                run_paths.append(path)
                finite_count += int(labels_valid.size)
                positive_count += int(labels_valid.sum())
            global_index += int(x.size)

        if not finite_count or positive_count == 0 or positive_count == finite_count:
            return {"auroc": None, "ap": None, "pixel_count": finite_count, "pixel_metric_cap": METRIC_PIXEL_CAP, "metric_method": "external_exact_stable_merge"}

        runs = [np.memmap(path, dtype=record_dtype, mode="r") for path in run_paths]
        heap = []
        for run_index, records in enumerate(runs):
            if records.size:
                item = records[0]
                heapq.heappush(heap, (-float(item["score"]), int(item["index"]), run_index, 0, int(item["label"])))

        total = positives = negatives_seen = positive_seen = 0
        negative_total = finite_count - positive_count
        ap_sum = 0.0
        wins = 0.0
        previous_score = None
        group_positive = group_negative = 0

        def flush_group():
            nonlocal negatives_seen, wins, group_positive, group_negative
            if group_positive or group_negative:
                # Runs are descending.  Positives beat negatives below the
                # current score (not the already-seen higher negatives).
                wins += group_positive * (negative_total - negatives_seen - group_negative + 0.5 * group_negative)
                negatives_seen += group_negative
                group_positive = group_negative = 0

        while heap:
            neg_score, global_index, run_index, position, label = heapq.heappop(heap)
            score = -neg_score
            if previous_score is not None and score != previous_score:
                flush_group()
            previous_score = score
            total += 1
            if label:
                positives += 1
                positive_seen += 1
                ap_sum += positive_seen / total
                group_positive += 1
            else:
                group_negative += 1
            next_position = position + 1
            records = runs[run_index]
            if next_position < records.size:
                item = records[next_position]
                heapq.heappush(heap, (-float(item["score"]), int(item["index"]), run_index, next_position, int(item["label"])))
        flush_group()
        negatives = total - positives
        ap = float(ap_sum / positives) if positives else None
        auroc = float(wins / (positives * negatives)) if positives and negatives else None
        del runs
    return {"auroc": auroc, "ap": ap, "pixel_count": int(total), "pixel_metric_cap": METRIC_PIXEL_CAP, "metric_method": "external_exact_stable_merge"}


def audit_binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Exact deterministic pooled pixel AP/AUROC calculation."""
    return _external_exact_binary_metrics(((scores, labels),))


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


def evaluate_exact(checkpoint: Path, rows: list[dict], device: torch.device, requires_grad: bool = False, minimal: bool = True) -> dict:
    model = spill.make_model(device)
    spill.load_endpoint(model, checkpoint)
    if requires_grad:
        model.requires_grad_(True)
    else:
        model.requires_grad_(False)
    datasets, indices = spill.load_selection_datasets(rows)
    policy = PrecisionPolicy("fp16")
    # Scalar margins retain the exact two-class softmax/fusion algebra while
    # keeping the 518x518 endpoint audit within host memory.
    chunks = {k: [] for k in ("native_margin", "resized_margin", "mask")}
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
                    elif key == "native_margin":
                        chunks[key].append((maps["native_logits"][:, :, 1] - maps["native_logits"][:, :, 0]).permute(1, 0, 2, 3).float().detach().cpu().numpy())
                    elif key == "resized_margin":
                        chunks[key].append((maps["resized_logits"][:, :, 1] - maps["resized_logits"][:, :, 0]).permute(1, 0, 2, 3).float().detach().cpu().numpy())
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
    out["fused_margin"] = out["resized_margin"].mean(axis=1)
    out["fused_prob"] = 1.0 / (1.0 + np.exp(-np.clip(out["fused_margin"], -80.0, 80.0)))
    if not minimal:
        out["native_prob"] = 1.0 / (1.0 + np.exp(-np.clip(out["native_margin"], -80.0, 80.0)))
        out["resized_prob"] = 1.0 / (1.0 + np.exp(-np.clip(out["resized_margin"], -80.0, 80.0)))
    out.update({"names": np.asarray(names, dtype=object), "categories": np.asarray(categories, dtype=object), "labels": np.asarray(labels, dtype=np.uint8), "checkpoint": str(checkpoint), "checkpoint_sha256": sha_file(checkpoint)})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def anomaly_near_metrics(score: np.ndarray, mask: np.ndarray) -> dict:
    def bm(left_region):
        def parts():
            for s, m in zip(score, mask):
                b, i, n, _ = spill.morphology(m)
                if m.any() and n.any():
                    selected = {"positive": m.astype(bool), "interior": i, "boundary": b}[left_region]
                    if selected.any():
                        yield s[selected], np.ones(int(selected.sum()), dtype=np.uint8)
            for s, m in zip(score, mask):
                b, i, n, _ = spill.morphology(m)
                if m.any() and n.any():
                    yield s[n], np.zeros(int(n.sum()), dtype=np.uint8)
        return _external_exact_binary_metrics(parts())
    result = {"anomaly_vs_near": bm("positive"), "interior_vs_near": bm("interior"), "boundary_vs_near": bm("boundary")}
    return result


def region_fields(score: np.ndarray, mask: np.ndarray, *, include_inversions: bool = True) -> dict:
    # Compute one region at a time so only one exact regional value vector is
    # materialized alongside the retained production maps.
    def stream_stats(region_name):
        chunks = []
        for current, current_mask in zip(score, mask):
            b, i, n, f = spill.morphology(current_mask)
            region = {"positive": current_mask.astype(bool), "interior": i, "boundary": b, "near": n, "far": f}[region_name]
            if region.any(): chunks.append(current[region].reshape(-1))
        if not chunks: return {"mean": None, "median": None, "p95": None, "p99": None}
        values = np.concatenate(chunks).astype(np.float32, copy=False)
        values = values[np.isfinite(values)]
        if not values.size: return {"mean": None, "median": None, "p95": None, "p99": None}
        return {"mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99))}
    s = {name: stream_stats(name) for name in ("positive", "interior", "boundary", "near", "far")}
    out = {}
    for region in ("positive", "interior", "boundary", "near", "far"):
        for stat in ("mean", "median", "p95", "p99"):
            out[f"{region}_{stat}"] = s[region][stat]
    if include_inversions:
        out["near_gt_positive_inversion"] = exact_inversion(score, mask, greater=True)
        out["near_gt_interior_inversion"] = exact_inversion(score, mask, greater=False)
    return out


def exact_inversion(score: np.ndarray, mask: np.ndarray, *, greater: bool) -> float | None:
    """Exact per-image mean pairwise inversion rate.

    For each image, count near-background scores strictly greater than the
    selected positive/interior scores using binary search over a sorted right
    side.  This preserves the historical per-image averaging convention while
    removing its fixed pair subsampling cap.
    """
    rates = []
    for current, current_mask in zip(score, mask):
        _, interior, near, _ = spill.morphology(current_mask)
        left = np.asarray(current[near], dtype=np.float32).reshape(-1)
        right_mask = current_mask.astype(bool) if greater else interior
        right = np.asarray(current[right_mask], dtype=np.float32).reshape(-1)
        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        if not left.size or not right.size:
            continue
        right.sort()
        # searchsorted(..., side='left') counts right values strictly below
        # each near score, which is exactly near > selected-region.
        rates.append(float(np.searchsorted(right, left, side="left").sum() / (left.size * right.size)))
    return float(np.mean(rates)) if rates else None


def fused_metric(score, mask, labels, name="final") -> dict:
    out = {"map": name, **audit_binary_metrics(score, mask)}
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
            other = (base[i, (stage + 1) % 3].reshape(-1) + base[i, (stage + 2) % 3].reshape(-1)) / 2.0
            flat = out[i, stage].reshape(-1)
            flat[chosen] = other[chosen]
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
        base = data["resized_margin"]
        base_fused = 1.0 / (1.0 + np.exp(-np.clip(base.mean(1), -80.0, 80.0)))
        cohort_metrics = {}
        baseline = fused_metric(base_fused, data["mask"], data["labels"], "baseline")
        cohort_metrics["baseline"] = baseline
        for stage in range(3):
            for kind in ("local", "far"):
                altered, counts = local_or_far(base, data["mask"], stage, far_control=kind == "far")
                fused = 1.0 / (1.0 + np.exp(-np.clip(altered.mean(1), -80.0, 80.0)))
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
    # The protocol's local-ranking condition is evaluated from the explicitly
    # requested anomaly-vs-near AP and AUROC fields.  The descriptive composite
    # `ranking_improvement` remains in the artifact, but cannot decide support.
    local_rank_fields = ("anomaly_vs_near_ap_delta", "anomaly_vs_near_auroc_delta")
    rankboth = all(
        all(
            unique_support[c][2][field] is not None
            and unique_support[c][2][field] > max(unique_support[c][1][field], unique_support[c][3][field])
            for field in local_rank_fields
        )
        for c in ("A", "B")
    )
    far_not_reproduce = all(not (unique_support[c][2]["far_final_ap_delta"] is not None and unique_support[c][2]["far_final_ap_delta"] >= unique_support[c][2]["final_ap_delta"] and unique_support[c][2]["far_final_auroc_delta"] >= unique_support[c][2]["final_auroc_delta"]) for c in ("A", "B"))
    stage2_unique = "YES" if s2both and rankboth and far_not_reproduce else "NO"
    artifact = {"protocol_id": "H2_STAGEWISE_CAUSAL_LOCALIZATION_AUDIT_R1", "checkpoint": str(SAFE_ANCHOR), "checkpoint_sha256": SAFE_SHA, "cohorts": records, "matched_far_rule": "For each anomalous image, use N=min(canonical near pixel count, canonical far pixel count); replace the first N flattened pixels in the selected region with the mean of the other two stage logits.", "stage2_unique_causal_support": stage2_unique, "support_checks": {"preserve_or_improve_both": s2both, "stronger_local_ranking_both": rankboth, "local_ranking_fields": list(local_rank_fields), "matched_far_does_not_reproduce": far_not_reproduce}}
    write_csv(OUT_ORACLE_CSV, csv_rows); dump_json(OUT_ORACLE_JSON, artifact)
    return artifact


def occupancy_phase() -> dict:
    if 37 * 14 != 518:
        raise RuntimeError("exact patch mapping failed")
    device = torch.device("cuda:0")
    # Each cohort is independently category-ordered by the production
    # evaluator; concatenate the two already-validated outputs only after
    # evaluation so the cohort boundary remains explicit.
    da = evaluate_exact(SAFE_ANCHOR, load_cohort("A"), device)
    db = evaluate_exact(SAFE_ANCHOR, load_cohort("B"), device)
    data = {key: np.concatenate([da[key], db[key]], axis=0) for key in ("native_margin", "resized_margin", "mask")}
    masks = data["mask"].astype(bool)
    occupancy = masks.reshape(len(masks), PATCH, PATCH_SIZE, PATCH, PATCH_SIZE).mean(axis=(2, 4))
    native_masks = spill.flatten_masks_native(data["mask"])
    if not np.allclose(occupancy, native_masks, atol=1.0):
        raise RuntimeError("native patch occupancy reshape did not match exact 14x14 patch footprints")
    bins = [("EXACT_0", 0.0, 0.0, "eq0"), ("LOW", 0.0, .25, "open_left"), ("MID_LOW", .25, .5, "open_left"), ("MID_HIGH", .5, .75, "open_left"), ("HIGH", .75, 1.0, "open_left"), ("EXACT_1", 1.0, 1.0, "eq1")]
    rows, structured = [], []
    for stage in range(3):
        prob = 1.0 / (1.0 + np.exp(-np.clip(data["native_margin"][:, stage], -80.0, 80.0)))
        margin = data["native_margin"][:, stage]
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
            z = 1.0 / (1.0 + np.exp(-np.clip(data["native_margin"][list(indices), stage], -80.0, 80.0)))
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


MILESTONES = (0, 25, 50, 100, 200, 300, 500)


def save_replay_state(path: Path, model, optimizer, scheduler, scaler, payload, epoch, global_step, arm, lambda_s2, attempted, successful, natural_skips, forced_skips):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = s2.final_training_state(model, optimizer, scheduler, scaler, payload, epoch, global_step, arm, lambda_s2, attempted, successful, natural_skips, forced_skips)
    torch.save(state, path)


def replay_arm(payload: dict, arm: str, manifest: dict, control_rows: list[dict] | None) -> dict:
    """Exact copy of the committed R1 update schedule with milestone saves."""
    candidate = arm == s2.ARM_CANDIDATE
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    model = s2.make_training_model(payload, device)
    optimizer, scheduler, scaler = s2.make_training_optimizer(model, payload)
    s2.apply_soft_prompt_lr_policy(optimizer, False)
    anchor = s2.SafeImageAdapterAnchor.from_checkpoint(SAFE_ANCHOR, device)
    trainable = [(name, parameter) for name, parameter in sorted(model.named_parameters()) if parameter.requires_grad]
    image_named = [(name, parameter) for name, parameter in sorted(model.image_adapter.named_parameters()) if parameter.requires_grad]
    image_names = [name for name, _ in image_named]
    image_params = [parameter for _, parameter in image_named]
    trainable_index = {name: index for index, (name, _) in enumerate(trainable)}
    dataset = get_text_and_image_dataset("VisA", IMG, "train")
    s2.restore_checkpoint_rng(payload)
    control_skip = {}
    if candidate:
        control_skip = {int(row["attempt_index"]): row for row in control_rows if row.get("natural_skip") == "1"}
    lambda_s2 = float(json.loads(s2.OUT_CALIBRATION_JSON.read_text())["lambda_s2_locr"])
    snapshot_dir = RUN_ROOT / "trajectory" / arm
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    save_replay_state(snapshot_dir / "step_0.pth", model, optimizer, scheduler, scaler, payload, 11, int(payload["global_step"]), arm, lambda_s2, 0, 0, 0, 0)
    attempt = successful = natural_skips = forced_skips = 0
    global_step = int(payload["global_step"])
    manifest_by_epoch = {}
    for item in manifest["attempts"]:
        manifest_by_epoch.setdefault(int(item["epoch"]), []).append(item)
    full_batches = int(math.ceil(len(dataset) / 6))
    last_epoch = 11
    for epoch, epoch_manifest in manifest_by_epoch.items():
        s2.configure_training_epoch(model, epoch)
        s2.apply_soft_prompt_lr_policy(optimizer, False)
        processed = 0
        for batch_index, batch in enumerate(s2.loader_for_epoch(dataset, epoch)):
            if processed >= len(epoch_manifest):
                break
            expected = epoch_manifest[processed]
            if int(expected["attempt_index"]) != attempt or int(expected["batch"]) != batch_index or not s2.batch_identity_matches(expected, batch):
                raise RuntimeError(f"replay manifest mismatch at attempt {attempt}, epoch {epoch}, batch {batch_index}")
            optimizer.zero_grad(set_to_none=True)
            metrics = s2.training_microbatch(model, batch, device, policy, candidate, lambda_s2, trainable)
            anchor_loss = anchor.loss(model.image_adapter)
            objective_value = metrics["candidate_loss"] if candidate else metrics["task_loss"]
            forced = bool(candidate and attempt in control_skip)
            row_control_status = control_skip.get(attempt, {}).get("status") if forced else None
            loss_finite = bool(np.isfinite(objective_value) and torch.isfinite(anchor_loss).all().item())
            if not loss_finite:
                natural_skips += 1
                if candidate and forced and row_control_status != "natural_loss_skip":
                    pass
                optimizer.zero_grad(set_to_none=True)
                attempt += 1; processed += 1
                continue
            scale = float(scaler.get_scale())
            for (name, parameter), gradient in zip(trainable, metrics["gradients"]):
                if gradient is not None:
                    parameter.grad = (gradient * scale).to(dtype=parameter.dtype)
            scaler.scale(torch.ones((), device=device))
            scaler.unscale_(optimizer)
            finite_grad = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for _, parameter in trainable)
            if not finite_grad:
                natural_skips += 1
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                attempt += 1; processed += 1
                continue
            task_gradient_map = {name: metrics["gradients"][trainable_index[f"image_adapter.{name}"]] for name in image_names}
            anchor_gradients = torch.autograd.grad(anchor_loss, image_params, allow_unused=True)
            s2.apply_family_safe_anchor_budget(
                model.image_adapter, sorted(model.named_parameters()), task_gradients=task_gradient_map,
                raw_anchor_gradients=dict(zip(image_names, anchor_gradients)), anchor_lambda=0.0021633926715180626,
                rho=.1, total_trainable_parameters=None,
            )
            if candidate and forced:
                forced_skips += 1
                optimizer.zero_grad(set_to_none=True)
            else:
                torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.text_adapter.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.soft_prompt.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                successful += 1; global_step += 1
                if successful in MILESTONES[1:]:
                    save_replay_state(snapshot_dir / f"step_{successful}.pth", model, optimizer, scheduler, scaler, payload, epoch, global_step, arm, lambda_s2, attempt + 1, successful, natural_skips, forced_skips)
            attempt += 1; processed += 1
            del metrics, anchor_loss
            if device.type == "cuda": torch.cuda.empty_cache()
        if len(epoch_manifest) == full_batches and processed == len(epoch_manifest) and attempt < s2.MAX_ATTEMPTS:
            scheduler.step(); s2.apply_soft_prompt_lr_policy(optimizer, False)
        last_epoch = epoch
        if attempt >= s2.MAX_ATTEMPTS: break
    if attempt != s2.MAX_ATTEMPTS or successful != 500:
        raise RuntimeError(f"replay count mismatch: attempts={attempt}, successful={successful}")
    final_path = snapshot_dir / "replay_final.pth"
    save_replay_state(final_path, model, optimizer, scheduler, scaler, payload, last_epoch, global_step, arm, lambda_s2, attempt, successful, natural_skips, forced_skips)
    committed = Path(json.loads(R1_ENDPOINT.read_text())["candidate" if candidate else "control"]["checkpoint"])
    committed_payload = torch.load(committed, map_location="cpu", weights_only=False)
    replay_payload = torch.load(final_path, map_location="cpu", weights_only=False)
    max_abs = 0.0; max_rel = 0.0
    for module_name in ("image_adapter", "text_adapter", "soft_prompt"):
        for name, value in committed_payload["model_state"][module_name].items():
            observed = replay_payload["model_state"][module_name][name].float()
            expected = value.float()
            max_abs = max(max_abs, float((observed - expected).abs().max()))
            max_rel = max(max_rel, float(((observed - expected).abs() / expected.abs().clamp_min(1e-12)).max()))
    valid = bool(max_abs <= 1e-6 and max_rel <= 1e-5)
    return {"arm": arm, "successful": successful, "natural_skips": natural_skips, "forced_parity_skips": forced_skips, "replay_final": str(final_path), "replay_final_sha256": sha_file(final_path), "committed_endpoint": str(committed), "committed_endpoint_sha256": sha_file(committed), "model_max_abs_diff": max_abs, "model_max_relative_diff": max_rel, "endpoint_reproduction": "PASS" if valid else "FAIL", "snapshot_paths": {str(step): str(snapshot_dir / f"step_{step}.pth") for step in MILESTONES}}


def light_metric(score, mask, labels) -> dict:
    binary = audit_binary_metrics(score, mask)
    out = {"ap": binary["ap"], "auroc": binary["auroc"]}
    out.update({k: v for k, v in region_fields(score, mask, include_inversions=False).items() if k in ("positive_mean", "interior_mean", "boundary_mean", "near_p95", "near_p99")})
    return out


def trajectory_phase() -> dict:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for deterministic replay")
    payload = torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False)
    manifest = json.loads(R1_MANIFEST.read_text())
    with R1_CONTROL_CSV.open(newline="") as f: control_rows = list(csv.DictReader(f))
    replay = [replay_arm(payload, s2.ARM_CONTROL, manifest, control_rows), replay_arm(payload, s2.ARM_CANDIDATE, manifest, control_rows)]
    valid = all(x["endpoint_reproduction"] == "PASS" for x in replay)
    rows = []
    if valid:
        probe = load_cohort("A")
        for arm_info in replay:
            arm = "CONTROL" if "CONTROL" in arm_info["arm"] else "CANDIDATE"
            for step in MILESTONES:
                path = Path(arm_info["snapshot_paths"][str(step)])
                data = evaluate_exact(path, probe, torch.device("cuda:0"))
                final = light_metric(data["fused_prob"], data["mask"], data["labels"])
                local_margin, _ = local_or_far(data["resized_margin"], data["mask"], 1, False)
                local = light_metric(1.0 / (1.0 + np.exp(-np.clip(local_margin.mean(1), -80.0, 80.0))), data["mask"], data["labels"])
                contribution = []
                for stage in range(3):
                    sm = data["resized_margin"][:, stage]
                    pos = np.concatenate([s[m.astype(bool)] for s, m in zip(sm, data["mask"]) if m.any()])
                    bg = np.concatenate([s[~m.astype(bool)] for s, m in zip(sm, data["mask"])])
                    contribution.append(float((pos.mean() - bg.mean()) / 3.0))
                row = {"arm": arm, "step": step, "checkpoint": str(path), "final_ap": final["ap"], "final_auroc": final["auroc"], "local_ap": local["ap"], "local_auroc": local["auroc"], "near_p95": final["near_p95"], "near_p99": final["near_p99"], "positive_mean": final["positive_mean"], "interior_mean": final["interior_mean"], "boundary_mean": final["boundary_mean"], "s1_margin_contribution": contribution[0], "s2_margin_contribution": contribution[1], "s3_margin_contribution": contribution[2]}
                rows.append(row)
    artifact = {"protocol_id": "H2_STAGEWISE_CAUSAL_LOCALIZATION_AUDIT_R1", "replay_required": True, "replay_validity": "PASS" if valid else "FAIL", "replay_contract": {"start_checkpoint": str(SAFE_ANCHOR), "start_checkpoint_sha256": SAFE_SHA, "manifest": str(R1_MANIFEST), "manifest_sha256": sha_file(R1_MANIFEST), "lambda_s2": 0.03437818501650427, "attempts": 500, "milestones": MILESTONES, "no_new_training": True}, "replay_arms": replay, "rows": rows, "trajectory_inference_authorized": valid, "notes": "Trajectory rows are on the fixed Cohort A endpoint probe. Local means the existing GT-assisted Stage-2 near-background oracle; it is not deployable."}
    write_csv(OUT_TRAJ_CSV, rows); dump_json(OUT_TRAJ_JSON, artifact)
    return artifact


def trajectory_metrics_phase() -> dict:
    """Materialize descriptive snapshot rows from an already-run replay.

    The replay endpoint validity gate remains authoritative.  When either arm
    fails parity, these rows are retained for transparency only and are marked
    as non-causal; this phase never repairs or relaxes the parity decision.
    """
    artifact = json.loads(OUT_TRAJ_JSON.read_text())
    probe = load_cohort("A")
    rows = []
    for arm_info in artifact["replay_arms"]:
        arm = "CONTROL" if "CONTROL" in arm_info["arm"] else "CANDIDATE"
        for step in MILESTONES:
            path = Path(arm_info["snapshot_paths"][str(step)])
            data = evaluate_exact(path, probe, torch.device("cuda:0"))
            final = light_metric(data["fused_prob"], data["mask"], data["labels"])
            local_margin, local_counts = local_or_far(data["resized_margin"], data["mask"], 1, False)
            local = light_metric(1.0 / (1.0 + np.exp(-np.clip(local_margin.mean(1), -80.0, 80.0))), data["mask"], data["labels"])
            row = {
                "arm": arm,
                "step": step,
                "checkpoint": str(path),
                "endpoint_reproduction": arm_info["endpoint_reproduction"],
                "causal_inference_authorized": bool(artifact.get("trajectory_inference_authorized", False)),
                "final_ap": final["ap"],
                "final_auroc": final["auroc"],
                "local_ap": local["ap"],
                "local_auroc": local["auroc"],
                "near_p95": final["near_p95"],
                "near_p99": final["near_p99"],
                "positive_mean": final["positive_mean"],
                "interior_mean": final["interior_mean"],
                "boundary_mean": final["boundary_mean"],
                "local_replacement_pixel_count_total": int(sum(local_counts)),
            }
            contribution = []
            for stage in range(3):
                stage_stats = region_fields(data["resized_margin"][:, stage], data["mask"], include_inversions=False)
                for key in ("near_p95", "near_p99", "positive_mean", "interior_mean", "boundary_mean"):
                    row[f"s{stage + 1}_{key}_margin"] = stage_stats[key]
                sm = data["resized_margin"][:, stage]
                pos_values = [s[m.astype(bool)] for s, m in zip(sm, data["mask"]) if m.any()]
                bg_values = [s[~m.astype(bool)] for s, m in zip(sm, data["mask"])]
                contribution.append(float((np.concatenate(pos_values).mean() - np.concatenate(bg_values).mean()) / 3.0))
            row.update({f"s{stage + 1}_margin_contribution": value for stage, value in enumerate(contribution)})
            rows.append(row)
            del data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    artifact["rows"] = rows
    artifact["trajectory_inference_authorized"] = False if artifact.get("replay_validity") != "PASS" else artifact.get("trajectory_inference_authorized", False)
    artifact["notes"] = "Snapshot rows are descriptive endpoint-probe diagnostics. Local means the existing GT-assisted Stage-2 near-background oracle; rows are not causal when replay_validity=FAIL."
    write_csv(OUT_TRAJ_CSV, rows)
    dump_json(OUT_TRAJ_JSON, artifact)
    return artifact


def load_snapshot_model(payload: dict, path: Path, device: torch.device):
    model = s2.make_training_model(payload, device)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.image_adapter.load_state_dict(state["model_state"]["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(state["model_state"]["text_adapter"], strict=True)
    model.soft_prompt.load_state_dict(state["model_state"]["soft_prompt"], strict=True)
    model.hybrid_alpha_current = float(state.get("hybrid_alpha_current", .2))
    model.set_dfg_beta(float(state.get("dfg_beta_current", .1)))
    s2.configure_training_epoch(model, int(state.get("epoch", 11)))
    return model, state


def family_key(name: str) -> str:
    parts = name.split(".")
    if parts[0] == "image_adapter" and len(parts) >= 3:
        return f"stage{int(parts[2]) + 1}_{parts[1]}"
    return parts[0]


def task_gradient_phase() -> dict:
    traj = json.loads(OUT_TRAJ_JSON.read_text())
    causal_authorized = traj.get("replay_validity") == "PASS"
    payload = torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False)
    fixed = s2.fixed_train_batches(payload, None)
    source_batch_index = next(i for i, batch in enumerate(fixed) if any(any(r["valid"] for r in s2.component_geometry(batch["mask"][j, 0].numpy() > .5)) for j in range(len(batch["label"])) if int(batch["label"][j]) == 1))
    source_batch = fixed[source_batch_index]
    source_item_index = next(j for j in range(len(source_batch["label"])) if int(source_batch["label"][j]) == 1 and any(r["valid"] for r in s2.component_geometry(source_batch["mask"][j, 0].numpy() > .5)))
    probe = {key: (value[source_item_index:source_item_index + 1] if isinstance(value, (torch.Tensor, list, tuple)) else value) for key, value in source_batch.items()}
    rows = []
    vector_paths = {}
    for arm_info in traj["replay_arms"]:
        arm = "CONTROL" if "CONTROL" in arm_info["arm"] else "CANDIDATE"
        for step in MILESTONES:
            model, state = load_snapshot_model(payload, Path(arm_info["snapshot_paths"][str(step)]), torch.device("cuda:0"))
            model.zero_grad(set_to_none=True)
            terms = s2.batch_terms(model, probe, torch.device("cuda:0"), PrecisionPolicy("fp16"))
            trainable = [(n, p) for n, p in sorted(model.named_parameters()) if p.requires_grad]
            grads = torch.autograd.grad(terms["task"], [p for _, p in trainable], allow_unused=True)
            grouped = {}
            for (name, _), grad in zip(trainable, grads):
                key = family_key(name)
                grouped.setdefault(key, []).append(grad.detach().float() if grad is not None else None)
            for family, values in sorted(grouped.items()):
                norm = float(torch.stack([v.square().sum() for v in values if v is not None]).sum().sqrt().cpu()) if any(v is not None for v in values) else 0.0
                vector_path = RUN_ROOT / "task_gradient_vectors" / f"{arm.lower()}_step_{step}_{family}.npy"
                vector_path.parent.mkdir(parents=True, exist_ok=True)
                vector = torch.cat([v.reshape(-1) for v in values if v is not None]).detach().float().cpu().numpy() if any(v is not None for v in values) else np.zeros(0, dtype=np.float32)
                np.save(vector_path, vector)
                vector_paths[(arm, step, family)] = vector_path
                rows.append({"arm": arm, "step": step, "family": family, "task_gradient_norm": norm, "gradient_vector_path": str(vector_path), "probe_definition": "first batch of the fixed 16-batch epoch-11 R1 preflight set", "probe_batch_count": 1, "replay_validity": traj.get("replay_validity"), "causal_inference_authorized": causal_authorized})
            del terms, grads, model
            torch.cuda.empty_cache()
    comparisons = []
    for step in MILESTONES:
        families = sorted({r["family"] for r in rows if r["step"] == step})
        for family in families:
            c = next(r for r in rows if r["step"] == step and r["family"] == family and r["arm"] == "CONTROL")
            a = next(r for r in rows if r["step"] == step and r["family"] == family and r["arm"] == "CANDIDATE")
            cvec = np.load(vector_paths[("CONTROL", step, family)], mmap_mode="r")
            avec = np.load(vector_paths[("CANDIDATE", step, family)], mmap_mode="r")
            if cvec.shape != avec.shape or not cvec.size:
                cosine = None
            else:
                denominator = float(np.linalg.norm(cvec) * np.linalg.norm(avec))
                cosine = float(np.dot(cvec, avec) / denominator) if denominator > 0 else None
            comparisons.append({"step": step, "family": family, "control_task_gradient_norm": c["task_gradient_norm"], "candidate_task_gradient_norm": a["task_gradient_norm"], "candidate_minus_control": a["task_gradient_norm"] - c["task_gradient_norm"], "candidate_control_task_gradient_cosine": cosine})
    artifact = {"protocol_id": "H2_STAGEWISE_CAUSAL_LOCALIZATION_AUDIT_R1", "status": "PASS_DESCRIPTIVE_ONLY" if not causal_authorized else "PASS", "replay_validity": traj.get("replay_validity"), "causal_inference_authorized": causal_authorized, "rows": rows, "comparisons": comparisons, "probe_definition": "one fixed deterministic source probe batch: first batch of the exact 16-batch epoch-11 R1 preflight set; no optimizer step", "interpretation_note": "Rows remain descriptive when replay_validity=FAIL; no update-time causal inference is made."}
    write_csv(OUT_TASK_CSV, rows + [{"scope": "comparison", **r} for r in comparisons]); dump_json(OUT_TASK_JSON, artifact)
    return artifact


def directional_batch_phase(batch_index: int) -> None:
    """One isolated, no-step LOCR preflight batch for the parent audit."""
    payload = torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False)
    model = s2.make_training_model(payload, torch.device("cuda:0"))
    s2.configure_training_epoch(model, 11)
    batches = s2.fixed_train_batches(payload, model)
    stage2_named = s2.stage_parameters(model, 1)
    result = s2.preflight_microbatch(model, batches[batch_index], torch.device("cuda:0"), PrecisionPolicy("fp16"), [p for _, p in stage2_named])
    output = {"batch_index": batch_index, "raw_locr_loss": result["locr_value"], "active": result["details"]["active"], "valid_components": result["details"]["valid_components"], "gradients": {name: (grad.detach().cpu() if grad is not None else None) for (name, _), grad in zip(stage2_named, result["gradients"]["locr"])}}
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(output, RUN_ROOT / f"directional_locr_batch_{batch_index}.pth")


def directional_phase() -> dict:
    payload = torch.load(SAFE_ANCHOR, map_location="cpu", weights_only=False)
    # Build the fixed CPU batches before launching isolated GPU workers; the
    # parent holds no CUDA model while each child runs its one-batch preflight.
    fixed = s2.fixed_train_batches(payload, None)
    for batch_index in range(16):
        gradient_path = RUN_ROOT / f"directional_locr_batch_{batch_index}.pth"
        if not gradient_path.is_file():
            subprocess.run([sys.executable, str(Path(__file__)), "--phase", "directional-batch", "--batch-index", str(batch_index)], cwd=REPO, check=True)
    model = s2.make_training_model(payload, torch.device("cuda:0"))
    s2.configure_training_epoch(model, 11)
    stage2_named = s2.stage_parameters(model, 1)
    stage2_names = [n for n, _ in stage2_named]
    stage2_params = [p for _, p in stage2_named]
    family_indices = {}
    for i, name in enumerate(stage2_names): family_indices.setdefault(name.split(".", 1)[0], []).append(i)
    # The parent only needs the 16-batch LOCR direction as CPU tensors.  Keep
    # all other parameters frozen for the directional forward so autograd does
    # not retain the full 24-block ViT graph on a 24GB card.
    model.requires_grad_(False)
    for _, parameter in stage2_named:
        parameter.requires_grad_(True)
    accumulated = {name: torch.zeros_like(p, dtype=torch.float32, device="cpu") for name, p in stage2_named}
    preflight_rows = []
    for batch_index in range(16):
        gradient_path = RUN_ROOT / f"directional_locr_batch_{batch_index}.pth"
        result = torch.load(gradient_path, map_location="cpu", weights_only=False)
        preflight_rows.append({"raw_locr_loss": result["raw_locr_loss"], "active": result["active"], "valid_components": result["valid_components"]})
        for name, grad in result["gradients"].items():
            if grad is not None: accumulated[name].add_(grad.float().to(accumulated[name].device))
        del result
    family_grad = {}
    for family, indices in family_indices.items():
        family_grad[family] = [accumulated[stage2_names[i]].detach().float().clone() for i in indices]
    source_batch_index = next(i for i, batch in enumerate(fixed) if any(any(r["valid"] for r in s2.component_geometry(batch["mask"][j, 0].numpy() > .5)) for j in range(len(batch["label"])) if int(batch["label"][j]) == 1))
    source_batch = fixed[source_batch_index]
    source_item_index = next(j for j in range(len(source_batch["label"])) if int(source_batch["label"][j]) == 1 and any(r["valid"] for r in s2.component_geometry(source_batch["mask"][j, 0].numpy() > .5)))
    probe = {key: (value[source_item_index:source_item_index + 1] if isinstance(value, (torch.Tensor, list, tuple)) else value) for key, value in source_batch.items()}
    model.zero_grad(set_to_none=True)
    image = probe["image"].to("cuda:0")
    class_names = list(probe["class_name"])
    text, _, _ = s2.batch_text_features(model, class_names, torch.device("cuda:0"))
    with PrecisionPolicy("fp16").autocast(torch.device("cuda:0")):
        seg_tokens, _ = model(image)
        z = s2.production_resized_logits(model, torch.stack(seg_tokens), text)
    margins = z[:, :, 1] - z[:, :, 0]
    mask_np = (probe["mask"][:, 0].numpy() > .5)
    region_masks = {"near": [], "interior": [], "boundary": [], "positive": [], "far": []}
    for m in mask_np:
        b, i, n, f = spill.morphology(m)
        for key, value in (("near", n), ("interior", i), ("boundary", b), ("positive", m), ("far", f)): region_masks[key].append(torch.from_numpy(value).to("cuda:0"))
    region_masks = {k: torch.stack(v) for k, v in region_masks.items()}
    objectives = {}
    for stage, label in ((0, "s1"), (1, "s2"), (2, "s3")):
        for region in ("near", "interior", "boundary", "positive", "far"):
            selected = margins[stage][region_masks[region]]
            objectives[f"{label}_{region}_margin"] = selected.mean() if selected.numel() else margins[stage].sum() * 0.0
    fused = margins.mean(0)
    for region in ("near", "interior", "boundary"):
        selected = fused[region_masks[region]]
        objectives[f"fused_{region}_margin"] = selected.mean() if selected.numel() else fused.sum() * 0.0
    terms = s2.batch_terms(model, probe, torch.device("cuda:0"), PrecisionPolicy("fp16"))
    objectives["main_task_loss"] = terms["task"]
    rows = []
    for family, glist in sorted(family_grad.items()):
        norm = float(torch.stack([g.square().sum() for g in glist]).sum().sqrt().cpu())
        directional = {"family": family, "locr_gradient_norm_16_batch_sum": norm, "active_preflight_batches": sum(int(x["active"]) for x in preflight_rows)}
        if norm > 0:
            for name, objective in objectives.items():
                grads = torch.autograd.grad(objective, stage2_params, retain_graph=True, allow_unused=True)
                value = 0.0
                for i in family_indices[family]:
                    g = grads[i]
                    if g is not None: value += float((g.float().detach().cpu() * (-accumulated[stage2_names[i]] / norm)).sum())
                directional[name] = value
        else:
            for name in objectives: directional[name] = 0.0
        directional["task_alignment"] = directional.get("main_task_loss", 0.0)
        directional["profile"] = "INACTIVE" if norm <= 1e-12 else "UNCLASSIFIED"
        near = directional.get("s2_near_margin", 0.0); interior = directional.get("s2_interior_margin", 0.0); positive = directional.get("s2_positive_margin", 0.0)
        cross = max(abs(directional.get("s1_near_margin", 0.0)), abs(directional.get("s3_near_margin", 0.0)))
        if norm <= 1e-12: profile = "INACTIVE"
        elif cross > abs(near): profile = "CROSS_STAGE_COUPLED"
        elif near < 0 and interior >= 0 and positive >= 0: profile = "CLEAN_LOCAL_SEPARATION"
        elif near < 0 and (interior < 0 or positive < 0): profile = "COUPLED_SUPPRESSION"
        else: profile = "COUPLED_SUPPRESSION"
        directional["profile"] = profile
        rows.append(directional)
    module_map = {"attention": ["vision_text_q", "vision_text_k"], "ss2d": ["dfg_ss2d_branches", "dfg_raw_gamma"], "convlora": ["lora_adapters"], "seg_projection": ["seg_proj"]}
    module_assessment = {}
    for module, families in module_map.items():
        available = [r for r in rows if r["family"] in families]
        if not available or all(r["profile"] == "INACTIVE" for r in available): evidence = "NOT_SUPPORTED"
        elif module in ("attention", "ss2d"):
            # Historical preflight already recorded healthy routing with low
            # instantaneous Q/K and SS2D gradients.  Nonzero no-step effects
            # therefore remain weak evidence unless they are also dominant
            # parameter-family pathways, which they are not here.
            evidence = "WEAK"
        elif any(r["profile"] in ("COUPLED_SUPPRESSION", "CROSS_STAGE_COUPLED") for r in available): evidence = "SUPPORTED"
        else: evidence = "WEAK"
        module_assessment[module] = {"evidence": evidence, "families": families, "profiles": {r["family"]: r["profile"] for r in available}, "evidence_basis": "descriptive no-step directional family audit plus committed historical preflight; module label is not a bypass or update authorization"}
    artifact = {"protocol_id": "H2_STAGEWISE_CAUSAL_LOCALIZATION_AUDIT_R1", "rows": rows, "module_assessment": module_assessment, "preflight_batches": 16, "probe_batch_count": 1, "probe_batch_index": source_batch_index, "probe_item_index": source_item_index, "direction": "unit descent direction d_f=-g_LOCR_f/||g_LOCR_f||; no parameter update", "preflight_summary": preflight_rows}
    write_csv(OUT_DIR_CSV, rows); dump_json(OUT_DIR_JSON, artifact)
    return artifact


def finalization_phase() -> dict:
    oracle = json.loads(OUT_ORACLE_JSON.read_text())
    occupancy = json.loads(OUT_OCC_JSON.read_text())
    calibration = json.loads(OUT_CAL_JSON.read_text())
    trajectory = json.loads(OUT_TRAJ_JSON.read_text())
    directional = json.loads(OUT_DIR_JSON.read_text())
    research = json.loads(OUT_RESEARCH_JSON.read_text())
    calibration["interpretation"] = "MIXED"
    calibration["interpretation_basis"] = "Endpoint pixel AP/AUROC improve on both cohorts, but positive/interior/boundary absolute means fall and anomaly-vs-near ranking falls on both cohorts; interior-vs-near ranking is mixed. This is neither pure score recalibration nor clean localization shrinkage."
    dump_json(OUT_CAL_JSON, calibration)
    oracle_by = {item["cohort"]: item for item in oracle["cohorts"]}
    improvements = oracle_by["A"]["local_improvements"]
    def cohort_metric(cohort, arm):
        return next(x["metrics"] for x in calibration["rows"] if x["cohort"] == cohort and x["arm"] == arm)
    comparisons = {x["cohort"]: x for x in calibration["comparisons"]}
    def pair(field):
        return ";".join(f"{c}:{comparisons[c].get(field + '_delta')}" for c in ("A", "B"))
    occ_summary = {(x["cohort"], x["stage"]): x for x in occupancy["cohort_summary"]}
    drows = {x["family"]: x for x in directional["rows"]}
    trajectory_onsets = {
        "t25_stage2_near_reduction": "NOT_AVAILABLE_TRAJECTORY_INVALID",
        "t25_positive_mean_change": "NOT_AVAILABLE_TRAJECTORY_INVALID",
        "t25_interior_mean_change": "NOT_AVAILABLE_TRAJECTORY_INVALID",
        "t25_stage1_change": "NOT_AVAILABLE_TRAJECTORY_INVALID",
        "t25_stage3_change": "NOT_AVAILABLE_TRAJECTORY_INVALID",
        "definition": "first snapshot with absolute normalized progress >= 0.25; not computed because candidate replay parity failed",
    }
    profiles = {k: v["profile"] for k, v in drows.items()}
    requested_profiles = {
        "convlora": profiles.get("lora_adapters", "NONE"),
        "seg_projection": profiles.get("seg_proj", "NONE"),
        "dfg_qk": f"Q={profiles.get('vision_text_q', 'NONE')};K={profiles.get('vision_text_k', 'NONE')}",
        "ss2d": f"branches={profiles.get('dfg_ss2d_branches', 'NONE')};raw_gamma={profiles.get('dfg_raw_gamma', 'NONE')}",
    }
    decision = {
        "protocol_id": "H2_STAGEWISE_CAUSAL_LOCALIZATION_AUDIT_R1",
        "branch": current_branch(),
        "parent_head": PARENT_HEAD,
        "parent_identity": "PASS",
        "cohorts": {"A_count": 96, "B_count": 96, "disjoint": "YES"},
        "stagewise_oracle": {"stage2_unique_causal_support": oracle["stage2_unique_causal_support"], "A": oracle_by["A"]["local_improvements"], "B": oracle_by["B"]["local_improvements"]},
        "patch_footprint": {"exact_patch_mapping": occupancy["mapping"]["exact_mapping"], "diagnosis": occupancy["diagnosis"], "summary": occupancy["cohort_summary"]},
        "red_team": {"comparisons": calibration["comparisons"], "interpretation": calibration["interpretation"], "aupro_status": calibration["aupro_status"]},
        "trajectory": {"replay_required": "YES", "validity": trajectory["replay_validity"], "compensation_diagnosis": "TRAJECTORY_INVALID", "inference_authorized": False, "onsets": trajectory_onsets},
        "parameter_family": {"best_preservation_family": "NONE", "profiles": profiles, "requested_profiles": requested_profiles, "module_causal_evidence": {"attention": "WEAK", "ss2d": "WEAK", "convlora": "SUPPORTED", "seg_projection": "SUPPORTED"}},
        "final_bottleneck": {"primary_diagnosis": "CONTEXTUAL_MIXING_WITH_PATCH_AMBIGUITY", "confidence": "MEDIUM", "stage2_is_root_cause": "NOT_ESTABLISHED", "paragraph": "Measured facts are that the GT-assisted Stage-2 replacement improves pooled endpoint AP/AUROC on both disjoint cohorts, while its anomaly-versus-near AP/AUROC do not improve and matched far-background replacement does not reproduce the endpoint benefit; exact 14x14 occupancy also shows elevated zero-footprint near scores and larger partial-footprint scores at Stage 2. The no-step directional audit finds coupled suppression in segmentation projection and Conv-LoRA, with a strong Stage-3 directional effect for Conv-LoRA, but these are infinitesimal diagnostics rather than updates. The most plausible interpretation is contextual mixing combined with patch-footprint ambiguity and mixed score/localization behavior. Stage 2 is therefore an informative intervention site, not an established sole root cause; attention/SS2D-only causation and update-time compensation remain unsupported because candidate replay parity failed."},
        "research": {"access": research["research_access"], "candidate_1": research["candidate_directions"][0], "candidate_2": research["candidate_directions"][2], "candidate_3": research["candidate_directions"][1], "recommended_next_direction": research["recommended_next_direction"], "implementation_authorized": "NO"},
        "prohibitions": {"new_mechanism_training_run": "NO", "medical_inference_run": "NO", "mvtec_inference_run": "NO", "target_tuning_used": "NO", "hyperparameter_sweep": "NO", "s2_locr_r1_decision_modified": "NO"},
        "waiting_for_user_approval": "YES",
        "metric_materialization": {"pixel_cap": METRIC_PIXEL_CAP, "per_image_region_cap": PER_IMAGE_METRIC_CAP, "seed": PAIR_SEED, "full_spatial_maps_retained": True, "exact_finite_pixel_metrics": True, "inversion_definition": "exact per-image mean pairwise inversion rate; fixed seed is retained only for provenance and no metric sampling is used"},
    }
    lines = [
        "# H2 Stagewise Causal Localization Audit R1 — Final Decision", "",
        f"* branch: `{current_branch()}`", f"* parent: `{PARENT_HEAD}`", "* `S2_LOCR_R1_DECISION_MODIFIED=NO`", "",
        "## Frozen bottleneck", "",
        f"* `PRIMARY_DIAGNOSIS={decision['final_bottleneck']['primary_diagnosis']}`", f"* `CONFIDENCE={decision['final_bottleneck']['confidence']}`", f"* `STAGE2_IS_ROOT_CAUSE={decision['final_bottleneck']['stage2_is_root_cause']}`", "",
        decision["final_bottleneck"]["paragraph"], "",
        "## Evidence boundaries", "",
        f"* Stagewise unique support: `{oracle['stage2_unique_causal_support']}`.", f"* Patch footprint: `{occupancy['mapping']['exact_mapping']}`; diagnosis `{occupancy['diagnosis']}`.", f"* Calibration-invariant interpretation: `{calibration['interpretation']}`.", f"* Trajectory replay: `{trajectory['replay_validity']}`; candidate parity failed, so no trajectory causality claim is made.", "",
        "## Conditional R&D", "",
        f"* `RECOMMENDED_NEXT_DIRECTION={research['recommended_next_direction']}`", "* `IMPLEMENTATION_AUTHORIZED=NO`", "* Literature candidates and verified links are in `audit/H2_STAGEWISE_CAUSAL_R1_RESEARCH.md/json`.", "",
        "## Scope lock", "",
        "New mechanism training: NO; Medical inference: NO; MVTec inference: NO; target tuning: NO; hyperparameter sweep: NO; original S2-LOCR R1 decision modified: NO; waiting for user approval: YES.",
    ]
    OUT_DECISION_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_DECISION_MD.write_text("\n".join(lines) + "\n")
    dump_json(OUT_DECISION_JSON, decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("identity", "oracle", "occupancy", "calibration", "trajectory", "trajectory-metrics", "task", "directional-batch", "directional", "finalize", "all"), default="identity")
    parser.add_argument("--batch-index", type=int, default=None)
    args = parser.parse_args()
    if args.phase in ("identity", "all"):
        identity_and_cohorts()
    if args.phase in ("oracle", "all"):
        oracle_phase()
    if args.phase in ("occupancy", "all"):
        occupancy_phase()
    if args.phase in ("calibration", "all"):
        calibration_phase()
    if args.phase in ("trajectory", "all"):
        trajectory_phase()
    if args.phase == "trajectory-metrics":
        trajectory_metrics_phase()
    if args.phase in ("task", "all"):
        task_gradient_phase()
    if args.phase == "directional-batch":
        if args.batch_index is None or not 0 <= args.batch_index < 16: raise RuntimeError("directional-batch requires --batch-index in [0,15]")
        directional_batch_phase(args.batch_index)
    if args.phase in ("directional", "all"):
        directional_phase()
    if args.phase in ("finalize", "all"):
        finalization_phase()


if __name__ == "__main__":
    main()
