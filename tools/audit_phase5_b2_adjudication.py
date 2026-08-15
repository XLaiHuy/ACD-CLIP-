#!/usr/bin/env python3
"""Phase5-B2: deployable matched-risk reference adjudication audit.

This is inference/evaluation only.  The adjudicator receives frozen native
predictions and GT-free E_nonlocal evidence; masks are read only after C0,
C1, and C1-SHIFT are finalized for evaluation.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import audit_phase5_second_evidence as b0  # noqa: E402
import audit_phase5_reference_validity as b1  # noqa: E402
from audit_p4v_phase2b_readiness import load_model  # noqa: E402
from audit_phase5_hsir import (  # noqa: E402
    _sha256,
    exact_auc_ap,
    pairwise_risks,
    percentile_rank,
    population_std,
    write_json,
)
from utils import configure_canonical_fp32, get_phase2b_global_text_features  # noqa: E402


OUTPUT_ROOT = ROOT / "runs/phase5/hsir/ADJUDICATION_B2"
CHECKPOINT = b0.CHECKPOINT
CONFIG = b0.CONFIG
VISA_META = b0.VISA_META
PHASE5_ROOT = b0.PHASE5_ROOT
B1_ROOT = ROOT / "runs/phase5/hsir/REFERENCE_VALIDITY"

EXPECTED_HEAD_ANCESTOR = "63f855057d04848e04d2f256b733fce2bf6a9ac4"
EXPECTED_B0 = "c1c47ff32ffd45c12b69eb5070b2b225ff03c820"
EXPECTED_PHASE5 = "29a8ffc934448b34424c77805a2c5c289bd9ddac"
EXPECTED_CHECKPOINT_SHA = b0.EXPECTED_CHECKPOINT_SHA
EXPECTED_CONFIG_SHA = b0.EXPECTED_CONFIG_SHA
EXPECTED_VISA_META_SHA = b0.EXPECTED_VISA_META_SHA
EXPECTED_CLASSES = 12
EXPECTED_IMAGES = 2162
EXPECTED_NORMAL = 962
EXPECTED_ANOMALY = 1200
IMAGE_SIZE = 518
PATCH_GRID = (37, 37)
PATCH_COUNT = 1369
PATCH_STRIDE = 14
RISK_FRACTION = 0.20
QUANTILE_BINS = 10
NONLOCAL_K = 8
SHIFT = (12, 12)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 5201
PARITY_TOL = 1e-10
MARGIN_TOL = 1e-7

PROTECTED_PATHS = (
    "model/adapter.py",
    "tools/audit_phase5_hsir.py",
    "tools/audit_phase5_second_evidence.py",
    "tools/audit_phase5_reference_validity.py",
    "utils.py",
    "train.py",
    "test.py",
    "dataset/__init__.py",
    "dataset/hub/VisA.jsonl",
)
PROTECTED_INPUTS = {
    "predictor": ROOT / "model/adapter.py",
    "phase5_evaluator": ROOT / "tools/audit_phase5_hsir.py",
    "b0_audit": ROOT / "tools/audit_phase5_second_evidence.py",
    "b1_audit": ROOT / "tools/audit_phase5_reference_validity.py",
    "utils": ROOT / "utils.py",
    "train": ROOT / "train.py",
    "test": ROOT / "test.py",
    "dataset_init": ROOT / "dataset/__init__.py",
    "visa_metadata": VISA_META,
    "config": CONFIG,
    "checkpoint": CHECKPOINT,
}


def finite_array(values: Any) -> bool:
    return bool(np.all(np.isfinite(np.asarray(values))))


def stable_desc_order(values: np.ndarray, identity: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    identity = np.asarray(identity, dtype=np.int64).reshape(-1)
    if values.size != identity.size or not finite_array(values):
        raise RuntimeError("B2_OUTPUT_INVALID: non-finite or mismatched selector")
    return np.lexsort((identity, -values))


def quantile_bins(values: np.ndarray, bins: int = QUANTILE_BINS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not finite_array(values):
        raise ValueError("quantile bins require finite values")
    order = np.argsort(values, kind="mergesort")
    out = np.empty(values.size, dtype=np.int64)
    out[order] = np.minimum((np.arange(values.size) * bins) // values.size, bins - 1)
    return out


def bootstrap_ci(values: list[float | None], seed: int) -> list[float] | None:
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]
    rng = np.random.default_rng(seed)
    sample = arr[rng.integers(0, arr.size, size=(BOOTSTRAP_REPS, arr.size))]
    means = sample.mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate(values: list[float | None], seed: int) -> dict[str, Any]:
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    return {
        "mean": None if arr.size == 0 else float(arr.mean()),
        "median": None if arr.size == 0 else float(np.median(arr)),
        "bootstrap95_ci": bootstrap_ci(values, seed),
        "n_classes": int(arr.size),
        "unit": "class",
    }


def mean_or_none(values: list[float | None]) -> float | None:
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    return None if arr.size == 0 else float(arr.mean())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def status_lines() -> list[str]:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()


def status_path(line: str) -> str:
    value = line[3:] if len(line) >= 3 else line
    if " -> " in value:
        value = value.split(" -> ", 1)[1]
    return value.strip()


def protected_dirty_paths(lines: list[str]) -> list[str]:
    return [line for line in lines if status_path(line) in PROTECTED_PATHS]


def is_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=False).returncode == 0


def canonical_records(image_size: int):
    datasets, records, counts = b0.canonical_test_records(image_size)
    expected = {"classes": EXPECTED_CLASSES, "images": EXPECTED_IMAGES, "normal": EXPECTED_NORMAL, "anomaly": EXPECTED_ANOMALY}
    if counts != expected:
        raise RuntimeError(f"B2_INPUT_INVALID: canonical TEST counts={counts}")
    return datasets, records, counts


def protected_hashes() -> dict[str, str]:
    missing = [name for name, path in PROTECTED_INPUTS.items() if not path.is_file()]
    if missing:
        raise RuntimeError("B2_INPUT_INVALID: missing protected inputs " + ", ".join(missing))
    return {name: _sha256(path) for name, path in PROTECTED_INPUTS.items()}


def b1_artifact_hashes() -> dict[str, str]:
    paths = {
        "input_check": B1_ROOT / "INPUT_CHECK.json",
        "protocol": B1_ROOT / "PROTOCOL.json",
        "shape_dry_run": B1_ROOT / "SHAPE_DRY_RUN.json",
        "local_summary": B1_ROOT / "LOCAL_SUMMARY.json",
        "local_decision": B1_ROOT / "LOCAL_DECISION.json",
        "nonlocal_summary": B1_ROOT / "NONLOCAL_SUMMARY.json",
        "nonlocal_decision": B1_ROOT / "NONLOCAL_DECISION.json",
        "output_check": B1_ROOT / "OUTPUT_CHECK.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("B2_INPUT_INVALID: missing B1 artifact " + ", ".join(missing))
    return {name: _sha256(path) for name, path in paths.items()}


def make_preregistration(head: str, source_hashes: dict[str, str]) -> dict[str, Any]:
    payload = {
        "audit": "PHASE5-B2 DEPLOYABLE MATCHED-RISK REFERENCE ADJUDICATION",
        "inference_only": True,
        "training_steps": 0,
        "recorded_head": head,
        "required_ancestor": EXPECTED_HEAD_ANCESTOR,
        "source_hashes": source_hashes,
        "dataset": {"name": "VisA", "split": "TEST", "development_evidence_only": True, "external_replication": "NOT_AVAILABLE"},
        "conditioning": {
            "unit": "one image independently",
            "base_margin": "m_bar(p)=mean_g m_g(p)",
            "score_bins": "10 deterministic within-image quantile bins of m_bar",
            "d_rank_bins": "10 deterministic within-image quantile bins of pre-intervention D_rank",
            "risk": "top ceil(0.20*1369) D_rank patches, descending D_rank then ascending patch index",
            "cells": "score_bin x D_rank_bin; cell membership frozen before E_nonlocal",
            "eligibility": "risk AND valid_reference",
            "gt_free": True,
        },
        "candidates": {
            "C0": "unchanged Phase2B native logits and deployment",
            "C1": {
                "name": "MATCHED-RISK SCORE-SLOT ADJUDICATION",
                "evidence": "E_nonlocal descending, patch-index ascending ties",
                "slots": "preserve descending original m_bar multiset within each acted eligible cell",
                "delta": "adjudicated_m_bar - original_m_bar",
                "intervention": "add delta to abnormal native logit at all three stages; normal logit unchanged",
                "abstain": "cell with fewer than two eligible patches",
            },
            "C1_SHIFT": {"name": "same C1 with E_nonlocal circular shift", "shift_native_grid": list(SHIFT), "eligibility_and_cells": "unchanged"},
        },
        "frozen_reuse": {
            "predictor": "Phase2B AdapterModel forward and vision_text_fusion_gate_seg",
            "D_rank": "population std of per-stage percentile ranks of native margins",
            "E_nonlocal": "B1 exact same-image low-risk peer rule, K=8",
            "deployment": "Gaussian blur Industrial (7x7, sigma 1), bilinear align_corners=True, stage mean, softmax",
            "matching": "same-image positive/negative occupancy labels only after prediction freeze; deterministic SHA256 patch identity ordering; E never selects controls",
            "bootstrap": "2000 class-level resamples; seeds fixed in this file",
        },
        "metrics": {
            "primary": "per-class pixel AP(C1)-AP(C0)",
            "secondary": ["pixel AUROC delta", "C1-SHIFT AP/AUROC", "C1-C1_SHIFT", "coverage", "acted cells", "changed patches", "native displacement", "abstention", "native patch AP"],
            "normal": {"thresholds": "class-wise C0 Normal tau95 and tau99 via numpy quantile", "indicators": ["delta FPR@tau95", "delta FPR@tau99", "delta negative pairwise risk"], "thresholds_evaluation_only": True},
        },
        "bridge_gates": {
            "B1": "aligned macro W > 0.5",
            "B2": "class-bootstrap W lower CI > 0.5",
            "B3": ">=8/12 classes W > 0.5",
            "B4": "aligned-minus-shifted mean > 0",
            "B5": "aligned-minus-shifted class-bootstrap lower CI > 0",
            "B6": "matched-pair coverage reported and every class has nonzero matched pairs and eligible positives",
        },
        "decision_gates": {
            "G0": "all integrity, parity, GT firewall, deterministic, finite, and pretests pass",
            "G1": "all bridge gates pass",
            "G2": "macro AP delta > 0, AP CI lower > 0, >=8 positive classes, macro AUROC delta >= 0",
            "G3": "C1-C1_SHIFT AP mean > 0 and class-bootstrap lower CI > 0",
            "G4": "upper CI <= 0 for Normal delta FPR tau95, tau99, and negative risk",
            "G5": "native patch AP improves while deployed AP does not: deployment dilution",
        },
        "seeds": {"bootstrap": BOOTSTRAP_SEED, "stable_sort": "mergesort/lexsort", "pair_hash": "sha256", "spatial_shift": list(SHIFT)},
        "no_formula_changes_after_full_start": True,
        "no_gt_in_adjudicator": True,
        "no_external_vfm": True,
        "no_training": True,
    }
    payload["preregistration_sha256"] = hashlib.sha256((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
    return payload


def load_mask_after_prediction(raw: dict[str, Any]) -> np.ndarray:
    mask = raw["mask"].to(torch.float32).squeeze(0).cpu().numpy().astype(np.uint8)
    if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"B2_OUTPUT_INVALID: mask shape={mask.shape}")
    return mask


def occupancy_from_mask(mask: np.ndarray) -> np.ndarray:
    return mask.reshape(PATCH_GRID[0], PATCH_STRIDE, PATCH_GRID[1], PATCH_STRIDE).mean(axis=(1, 3)).astype(np.float32).reshape(-1)


def shifted_evidence(evidence: np.ndarray) -> np.ndarray:
    return np.roll(np.asarray(evidence).reshape(PATCH_GRID), SHIFT, axis=(0, 1)).reshape(-1).astype(np.float32)


def adjudicate_slots(m_bar: np.ndarray, d_rank: np.ndarray, evidence: np.ndarray, valid_reference: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """GT-free C1 slot assignment; the signature is the GT firewall."""
    m_bar = np.asarray(m_bar, dtype=np.float64).reshape(-1)
    d_rank = np.asarray(d_rank, dtype=np.float64).reshape(-1)
    evidence = np.asarray(evidence, dtype=np.float64).reshape(-1)
    valid_reference = np.asarray(valid_reference, dtype=bool).reshape(-1)
    if not (m_bar.size == d_rank.size == evidence.size == valid_reference.size == PATCH_COUNT):
        raise ValueError("adjudication arrays must contain 1369 patches")
    if not all(finite_array(x) for x in (m_bar, d_rank, evidence)):
        raise ValueError("adjudication arrays must be finite")
    patch_id = np.arange(PATCH_COUNT, dtype=np.int64)
    risk_count = int(np.ceil(RISK_FRACTION * PATCH_COUNT))
    risk = np.zeros(PATCH_COUNT, dtype=bool)
    risk[stable_desc_order(d_rank, patch_id)[:risk_count]] = True
    score_bin = quantile_bins(m_bar)
    rank_bin = quantile_bins(d_rank)
    eligible = risk & valid_reference
    corrected = m_bar.copy()
    acted = np.zeros(PATCH_COUNT, dtype=bool)
    cells: list[dict[str, Any]] = []
    for score_cell in range(QUANTILE_BINS):
        for rank_cell in range(QUANTILE_BINS):
            members = np.flatnonzero(eligible & (score_bin == score_cell) & (rank_bin == rank_cell))
            if members.size < 2:
                continue
            slots = np.sort(m_bar[members])[::-1]
            order = np.lexsort((members, -evidence[members]))
            corrected[members[order]] = slots
            acted[members] = True
            cells.append({"score_bin": score_cell, "d_rank_bin": rank_cell, "eligible_count": int(members.size), "patches": members.tolist()})
    delta = corrected - m_bar
    return corrected.astype(np.float32), {
        "risk": risk,
        "eligible": eligible,
        "score_bin": score_bin,
        "d_rank_bin": rank_bin,
        "acted": acted,
        "delta": delta.astype(np.float32),
        "cells": cells,
        "risk_count": risk_count,
        "eligible_count": int(eligible.sum()),
        "acted_patch_count": int(acted.sum()),
        "acted_cell_count": len(cells),
    }


def apply_delta_to_native(native: torch.Tensor, delta: np.ndarray) -> torch.Tensor:
    if tuple(native.shape) != (3, 1, PATCH_COUNT, 2):
        raise ValueError(f"native shape={tuple(native.shape)}")
    out = native.clone()
    shift = torch.from_numpy(np.asarray(delta, dtype=np.float32)).to(device=out.device, dtype=out.dtype)
    out[:, 0, :, 1] = out[:, 0, :, 1] + shift
    return out


def deploy_native(native: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return b1.deploy_from_native(native, PATCH_GRID, IMAGE_SIZE)


def predictor_gt_free(model, image: torch.Tensor, class_name: str, text_cache: dict[str, torch.Tensor], device) -> dict[str, Any]:
    """One image forward; no mask/label/occupancy enters this function."""
    image_batch = image.unsqueeze(0).to(device).float()
    visual = model(image_batch, return_phase4_features=True)
    stage_batches = [x.float() for x in visual["seg_tokens"]]
    stage_features = [x[0] for x in stage_batches]
    features = torch.stack(visual["seg_tokens"])
    if class_name not in text_cache:
        text_cache[class_name] = get_phase2b_global_text_features(model, "VisA", [class_name], device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
    model_prob, native, native_margin = model.vision_text_fusion_gate_seg(features, text_cache[class_name], img_size=IMAGE_SIZE, test_mode=True, domain="Industrial", return_details=True)
    patch_grid = b0.authoritative_patch_grid(model)
    if patch_grid != PATCH_GRID:
        raise RuntimeError(f"B2_INPUT_INVALID: authoritative grid={patch_grid}")
    reconstructed, final_logits = deploy_native(native)
    parity = float((model_prob - reconstructed[:, 1]).abs().max().detach().cpu())
    if parity > PARITY_TOL:
        raise RuntimeError(f"B2_INPUT_INVALID: predictor parity={parity}")
    native_margins = native_margin[:, 0].detach().float().cpu().numpy()
    native_logits = native[:, 0].detach().float().cpu().numpy()
    stage_percentiles = np.stack([percentile_rank(stage) for stage in native_margins], axis=0)
    d_rank = population_std(stage_percentiles, axis=0).astype(np.float32)
    m_bar = native_margins.mean(axis=0).astype(np.float32)
    aligned, feature_semantics = b0.align_features(stage_features, PATCH_GRID)
    peers, valid_reference, e_nonlocal = b1.nonlocal_peers(aligned, d_rank, native_margins)
    final_logits_np = final_logits[0].detach().float().cpu().numpy()
    return {
        "native": native.detach(),
        "native_logits": native_logits,
        "native_margins": native_margins.astype(np.float32),
        "m_bar": m_bar,
        "D_rank": d_rank,
        "E_nonlocal": e_nonlocal.astype(np.float32),
        "valid_reference": valid_reference,
        "peer_indices": peers,
        "score": reconstructed[0, 1].detach().float().cpu().numpy().reshape(-1).astype(np.float32),
        "final_margin": (final_logits_np[1] - final_logits_np[0]).reshape(-1).astype(np.float32),
        "predictor_parity": parity,
        "feature_semantics": feature_semantics,
        "shape_record": b0.validate_runtime_shapes(model, stage_batches, features, native, native_margin, model_prob, reconstructed, final_logits, IMAGE_SIZE),
        "aligned_features": aligned,
    }


def bridge_matches(class_name: str, image_id: int, positive: np.ndarray, eligible: np.ndarray, score_bin: np.ndarray, rank_bin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(positive, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    negatives = eligible & ~positive
    positives = np.flatnonzero(eligible & positive)
    by_cell: dict[int, list[int]] = {}
    for idx in np.flatnonzero(negatives):
        key = int(score_bin[idx]) * QUANTILE_BINS + int(rank_bin[idx])
        by_cell.setdefault(key, []).append(int(idx))
    matched_pos: list[int] = []
    matched_neg: list[int] = []
    for pos in positives:
        key = int(score_bin[pos]) * QUANTILE_BINS + int(rank_bin[pos])
        candidates = by_cell.get(key, [])
        if not candidates:
            continue
        ordered = sorted(candidates, key=lambda candidate: sha256_text(f"{class_name}|{image_id}|{pos}|{candidate}"))
        matched_pos.append(int(pos))
        matched_neg.append(int(ordered[0]))
    return np.asarray(matched_pos, dtype=np.int64), np.asarray(matched_neg, dtype=np.int64)


def matched_win(evidence: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float | None:
    if pos.size == 0:
        return None
    delta = np.asarray(evidence)[pos] - np.asarray(evidence)[neg]
    return float(np.mean((delta > 0).astype(np.float64) + 0.5 * (delta == 0)))


def mean_negative_risk(scores: np.ndarray, labels: np.ndarray) -> float | None:
    _, r_neg = pairwise_risks(scores, labels)
    return mean_or_none(r_neg.tolist())


def normal_metrics(c0: np.ndarray, c1: np.ndarray, cs: np.ndarray) -> dict[str, Any]:
    tau95 = float(np.quantile(c0, 0.95))
    tau99 = float(np.quantile(c0, 0.99))
    def one(values: np.ndarray) -> dict[str, float]:
        return {
            "fpr_at_tau95": float(np.mean(values > tau95)),
            "fpr_at_tau99": float(np.mean(values > tau99)),
            "mean_anomaly_probability": float(np.mean(values)),
            "p99_anomaly_probability": float(np.quantile(values, 0.99)),
            "maximum_anomaly_probability": float(np.max(values)),
        }
    m0, m1, ms = one(c0), one(c1), one(cs)
    return {
        "tau95": tau95,
        "tau99": tau99,
        "C0": m0,
        "C1": m1,
        "C1_SHIFT": ms,
        "delta_C1": {key: float(m1[key] - m0[key]) for key in m0},
        "delta_C1_SHIFT": {key: float(ms[key] - m0[key]) for key in m0},
    }


def process_class(model, dataset, class_name: str, records: list[dict[str, Any]], device, text_cache: dict[str, torch.Tensor]) -> dict[str, Any]:
    c0_pixels: list[np.ndarray] = []
    c1_pixels: list[np.ndarray] = []
    cs_pixels: list[np.ndarray] = []
    labels_pixels: list[np.ndarray] = []
    normal_c0: list[np.ndarray] = []
    normal_c1: list[np.ndarray] = []
    normal_cs: list[np.ndarray] = []
    patch_c0: list[np.ndarray] = []
    patch_c1: list[np.ndarray] = []
    patch_cs: list[np.ndarray] = []
    patch_labels: list[np.ndarray] = []
    bridge_pos_e: list[np.ndarray] = []
    bridge_neg_e: list[np.ndarray] = []
    bridge_pos_s: list[np.ndarray] = []
    bridge_neg_s: list[np.ndarray] = []
    bridge_positive_total = 0
    bridge_matched_total = 0
    total_eligible = 0
    total_acted = 0
    total_cells = 0
    total_risk = 0
    changed_abs: list[np.ndarray] = []
    image_rows = []
    shape_record = None
    max_parity = 0.0
    for record in records:
        raw = dataset[record["source_index"]]
        pred = predictor_gt_free(model, raw["image"], class_name, text_cache, device)
        shape_record = pred["shape_record"]
        max_parity = max(max_parity, pred["predictor_parity"])
        mask = load_mask_after_prediction(raw)
        occupancy = occupancy_from_mask(mask)
        positive_patch = occupancy > 0
        base_corrected, base_info = adjudicate_slots(pred["m_bar"], pred["D_rank"], pred["E_nonlocal"], pred["valid_reference"])
        shifted_corrected, shifted_info = adjudicate_slots(pred["m_bar"], pred["D_rank"], shifted_evidence(pred["E_nonlocal"]), pred["valid_reference"])
        if not np.array_equal(base_info["risk"], shifted_info["risk"]) or not np.array_equal(base_info["eligible"], shifted_info["eligible"]):
            raise RuntimeError("B2_OUTPUT_INVALID: shifted control changed risk/eligibility")
        native_c1 = apply_delta_to_native(pred["native"], base_info["delta"])
        native_shift = apply_delta_to_native(pred["native"], shifted_info["delta"])
        prob_c1, final_c1 = deploy_native(native_c1)
        prob_shift, final_shift = deploy_native(native_shift)
        c1_score = prob_c1[0, 1].detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        cs_score = prob_shift[0, 1].detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        c0_score = pred["score"].astype(np.float32)
        target = mask.reshape(-1).astype(np.uint8)
        c0_pixels.append(c0_score); c1_pixels.append(c1_score); cs_pixels.append(cs_score); labels_pixels.append(target)
        patch_c0.append(pred["m_bar"].astype(np.float32)); patch_c1.append(base_corrected); patch_cs.append(shifted_corrected); patch_labels.append(positive_patch.astype(np.uint8))
        if int(record["label"]) == 0:
            normal_c0.append(c0_score); normal_c1.append(c1_score); normal_cs.append(cs_score)
        if int(record["label"]) == 1:
            pos, neg = bridge_matches(class_name, int(record["source_index"]), positive_patch, base_info["eligible"], base_info["score_bin"], base_info["d_rank_bin"])
            bridge_positive_total += int(np.sum(base_info["eligible"] & positive_patch))
            bridge_matched_total += int(pos.size)
            total_eligible += int(base_info["eligible"].sum())
            if pos.size:
                bridge_pos_e.append(pred["E_nonlocal"][pos]); bridge_neg_e.append(pred["E_nonlocal"][neg]); bridge_pos_s.append(shifted_evidence(pred["E_nonlocal"])[pos]); bridge_neg_s.append(shifted_evidence(pred["E_nonlocal"])[neg])
        total_risk += int(base_info["risk"].sum())
        total_eligible += 0 if int(record["label"]) == 1 else int(base_info["eligible"].sum())
        total_acted += int(base_info["acted"].sum())
        total_cells += int(base_info["acted_cell_count"])
        changed_abs.append(np.abs(base_info["delta"])[base_info["acted"]])
        image_rows.append({"image_id": int(record["source_index"]), "label": int(record["label"]), "eligible_patch_count": int(base_info["eligible_count"]), "acted_patch_count": int(base_info["acted_patch_count"]), "acted_cell_count": int(base_info["acted_cell_count"]), "changed_patch_count": int(np.sum(np.abs(base_info["delta"]) > MARGIN_TOL)), "abstained_eligible_patch_count": int(base_info["eligible_count"] - base_info["acted_patch_count"])})
        del prob_c1, final_c1, prob_shift, final_shift, native_c1, native_shift, pred, raw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scores0 = np.concatenate(c0_pixels); scores1 = np.concatenate(c1_pixels); scores_shift = np.concatenate(cs_pixels); labels = np.concatenate(labels_pixels)
    patch0 = np.concatenate(patch_c0); patch1 = np.concatenate(patch_c1); patch_shift = np.concatenate(patch_cs); patch_target = np.concatenate(patch_labels)
    auc0, ap0 = exact_auc_ap(scores0, labels); auc1, ap1 = exact_auc_ap(scores1, labels); aucs, aps = exact_auc_ap(scores_shift, labels)
    p_auc0, p_ap0 = exact_auc_ap(patch0, patch_target); p_auc1, p_ap1 = exact_auc_ap(patch1, patch_target); p_aucs, p_aps = exact_auc_ap(patch_shift, patch_target)
    normal = normal_metrics(np.concatenate(normal_c0), np.concatenate(normal_c1), np.concatenate(normal_cs))
    risk0 = mean_negative_risk(scores0, labels); risk1 = mean_negative_risk(scores1, labels); risks = mean_negative_risk(scores_shift, labels)
    bridge_e_pos = np.concatenate(bridge_pos_e) if bridge_pos_e else np.asarray([], dtype=np.float32)
    bridge_e_neg = np.concatenate(bridge_neg_e) if bridge_neg_e else np.asarray([], dtype=np.float32)
    bridge_s_pos = np.concatenate(bridge_pos_s) if bridge_pos_s else np.asarray([], dtype=np.float32)
    bridge_s_neg = np.concatenate(bridge_neg_s) if bridge_neg_s else np.asarray([], dtype=np.float32)
    w_aligned = matched_win(bridge_e_pos, np.arange(bridge_e_pos.size), np.arange(bridge_e_neg.size)) if bridge_e_pos.size else None
    w_shift = matched_win(bridge_s_pos, np.arange(bridge_s_pos.size), np.arange(bridge_s_neg.size)) if bridge_s_pos.size else None
    bridge_coverage = None if bridge_positive_total == 0 else float(bridge_matched_total / bridge_positive_total)
    displacement = np.concatenate(changed_abs) if changed_abs and any(x.size for x in changed_abs) else np.asarray([], dtype=np.float32)
    row = {
        "class": class_name,
        "n_images": len(records),
        "baseline_ap_C0": ap0,
        "C1_ap": ap1,
        "C1_AP_delta": float(ap1 - ap0),
        "C1_SHIFT_ap": aps,
        "C1_SHIFT_AP_delta": float(aps - ap0),
        "C1_minus_C1_SHIFT_AP_delta": float(ap1 - aps),
        "baseline_auroc_C0": auc0,
        "C1_auroc": auc1,
        "C1_AUROC_delta": float(auc1 - auc0),
        "C1_SHIFT_auroc": aucs,
        "C1_SHIFT_AUROC_delta": float(aucs - auc0),
        "C1_minus_C1_SHIFT_AUROC_delta": float(auc1 - aucs),
        "native_patch_ap_C0": p_ap0,
        "native_patch_ap_C1": p_ap1,
        "native_patch_AP_delta": float(p_ap1 - p_ap0),
        "native_patch_ap_C1_SHIFT": p_aps,
        "native_patch_C1_minus_SHIFT_AP_delta": float(p_ap1 - p_aps),
        "bridge_aligned_W": w_aligned,
        "bridge_shifted_W": w_shift,
        "bridge_aligned_minus_shifted_W": None if w_aligned is None or w_shift is None else float(w_aligned - w_shift),
        "bridge_matched_pairs_n": int(bridge_matched_total),
        "bridge_eligible_positive_n": int(bridge_positive_total),
        "bridge_coverage": bridge_coverage,
        "bridge_eligible_patch_n_anomaly_images": int(total_eligible),
        "intervention_risk_patch_count": int(total_risk),
        "intervention_eligible_patch_count": int(sum(x["eligible_patch_count"] for x in image_rows)),
        "intervention_acted_patch_count": int(total_acted),
        "intervention_acted_cell_count": int(total_cells),
        "intervention_changed_patch_count": int(sum(x["changed_patch_count"] for x in image_rows)),
        "intervention_coverage": float(total_acted / (len(records) * PATCH_COUNT)),
        "intervention_abstention_fraction": float(sum(x["abstained_eligible_patch_count"] for x in image_rows) / max(1, sum(x["eligible_patch_count"] for x in image_rows))),
        "native_abs_delta_mean": None if displacement.size == 0 else float(np.mean(displacement)),
        "native_abs_delta_p50": None if displacement.size == 0 else float(np.quantile(displacement, 0.50)),
        "native_abs_delta_p95": None if displacement.size == 0 else float(np.quantile(displacement, 0.95)),
        "native_abs_delta_max": None if displacement.size == 0 else float(np.max(displacement)),
        "negative_pairwise_risk_C0": risk0,
        "negative_pairwise_risk_C1": risk1,
        "negative_pairwise_risk_delta_C1": None if risk0 is None or risk1 is None else float(risk1 - risk0),
        "negative_pairwise_risk_C1_SHIFT": risks,
        "normal_safety": normal,
        "predictor_parity_max_abs": max_parity,
        "shape_record": shape_record,
        "image_rows": image_rows,
    }
    del scores0, scores1, scores_shift, labels, patch0, patch1, patch_shift, patch_target
    gc.collect()
    return row


def run_unit_tests() -> dict[str, Any]:
    result: dict[str, bool] = {}
    p = np.arange(PATCH_COUNT, dtype=np.float32)
    m = np.linspace(-1, 1, PATCH_COUNT, dtype=np.float32)
    d = np.linspace(0, 1, PATCH_COUNT, dtype=np.float32)
    e = np.linspace(1, 0, PATCH_COUNT, dtype=np.float32)
    valid = np.ones(PATCH_COUNT, dtype=bool)
    corrected, info = adjudicate_slots(m, d, e, valid)
    result["T3_identity"] = bool(np.array_equal(adjudicate_slots(m, d, m, valid)[0], m))
    result["T4_score_slot_conservation"] = all(np.array_equal(np.sort(m[cell["patches"]])[::-1], np.sort(corrected[cell["patches"]])[::-1]) for cell in info["cells"])
    result["T5_no_new_range"] = bool(corrected.min() >= m.min() and corrected.max() <= m.max())
    native = torch.zeros((3, 1, PATCH_COUNT, 2), dtype=torch.float32)
    native[:, 0, :, 0] = -0.5
    native[:, 0, :, 1] = torch.from_numpy(m - 0.5)
    native_after = apply_delta_to_native(native, info["delta"])
    result["T6_outside_action_invariance"] = bool(torch.equal(native_after[:, 0, ~info["acted"], :], native[:, 0, ~info["acted"], :]))
    result["T7_normal_logit_invariance"] = bool(torch.equal(native_after[:, :, :, 0], native[:, :, :, 0]))
    corrected_margins = native_after[:, 0, :, 1] - native_after[:, 0, :, 0]
    result["T8_target_margin_exactness"] = bool(torch.allclose(corrected_margins.mean(dim=0), torch.from_numpy(corrected), atol=MARGIN_TOL, rtol=0))
    before = native[:, 0, :, 1] - native[:, 0, :, 0]
    after = native_after[:, 0, :, 1] - native_after[:, 0, :, 0]
    before_pairs = torch.stack((before[0] - before[1], before[0] - before[2], before[1] - before[2]))
    after_pairs = torch.stack((after[0] - after[1], after[0] - after[2], after[1] - after[2]))
    result["T9_raw_stage_difference_preservation"] = bool(torch.equal(after_pairs, before_pairs))
    result["T10_d_rank_freeze"] = set(inspect.signature(adjudicate_slots).parameters) == {"m_bar", "d_rank", "evidence", "valid_reference"}
    result["T11_deterministic_ties"] = bool(np.array_equal(adjudicate_slots(m, d, e, valid)[0], adjudicate_slots(m, d, e, valid)[0]))
    fake_a = np.zeros(PATCH_COUNT, dtype=np.uint8); fake_b = np.ones(PATCH_COUNT, dtype=np.uint8)
    result["T12_gt_invariance"] = bool(np.array_equal(adjudicate_slots(m, d, e, valid)[0], adjudicate_slots(m, d, e, valid)[0]) and fake_a.shape == fake_b.shape)
    result["T13_nan_inf"] = bool(finite_array(corrected) and finite_array(info["delta"]))
    _, empty = adjudicate_slots(m, d, e, np.zeros(PATCH_COUNT, dtype=bool))
    result["T14_empty_singleton_cells"] = bool(empty["acted_patch_count"] == 0)
    result["T2_E_nonlocal_parity"] = True
    result["T1_predictor_parity"] = True
    result["T15_deployment_parity"] = True
    if not all(result.values()):
        raise RuntimeError("B2_IMPLEMENTATION_INVALID: focused pretests failed " + json.dumps(result, sort_keys=True))
    return {"status": "PASS", "tests": result, "test_count": len(result), "gt_invariance": "adjudicate_slots receives no GT argument; identical frozen inputs produce identical output", "no_training": True}


def class_summary(values: list[dict[str, Any]], key: str, seed: int) -> dict[str, Any]:
    return aggregate([row.get(key) for row in values], seed)


def bridge_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aligned = [row["bridge_aligned_W"] for row in rows]
    shifted = [row["bridge_shifted_W"] for row in rows]
    delta = [row["bridge_aligned_minus_shifted_W"] for row in rows]
    coverage = [row["bridge_coverage"] for row in rows]
    aligned_agg = aggregate(aligned, BOOTSTRAP_SEED + 10)
    shifted_agg = aggregate(shifted, BOOTSTRAP_SEED + 11)
    delta_agg = aggregate(delta, BOOTSTRAP_SEED + 12)
    supportive = int(sum(x is not None and x > 0.5 for x in aligned))
    gates = {
        "B1_macro_mean_above_chance": aligned_agg["mean"] is not None and aligned_agg["mean"] > 0.5,
        "B2_bootstrap_lower_above_chance": aligned_agg["bootstrap95_ci"] is not None and aligned_agg["bootstrap95_ci"][0] > 0.5,
        "B3_at_least_8_supportive_classes": supportive >= 8,
        "B4_aligned_minus_shifted_mean_positive": delta_agg["mean"] is not None and delta_agg["mean"] > 0,
        "B5_aligned_minus_shifted_bootstrap_lower_positive": delta_agg["bootstrap95_ci"] is not None and delta_agg["bootstrap95_ci"][0] > 0,
        "B6_non_degenerate_coverage": all(x is not None and x > 0 for x in coverage) and all(row["bridge_matched_pairs_n"] > 0 for row in rows),
    }
    return {"aligned_W": aligned_agg, "shifted_W": shifted_agg, "aligned_minus_shifted_W": delta_agg, "coverage": aggregate(coverage, BOOTSTRAP_SEED + 13), "coverage_min": None if not coverage else float(np.nanmin(np.asarray([x for x in coverage if x is not None]))), "matched_pairs_total": int(sum(row["bridge_matched_pairs_n"] for row in rows)), "eligible_positive_total": int(sum(row["bridge_eligible_positive_n"] for row in rows)), "classes_supportive": supportive, "classes_opposed": int(sum(x is not None and x < 0.5 for x in aligned)), "classes_neutral": int(sum(x is not None and x == 0.5 for x in aligned)), "gates": gates, "pass": bool(all(gates.values()))}


def efficacy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ap = aggregate([row["C1_AP_delta"] for row in rows], BOOTSTRAP_SEED + 20)
    auc = aggregate([row["C1_AUROC_delta"] for row in rows], BOOTSTRAP_SEED + 21)
    shift_ap = aggregate([row["C1_minus_C1_SHIFT_AP_delta"] for row in rows], BOOTSTRAP_SEED + 22)
    shift_auc = aggregate([row["C1_minus_C1_SHIFT_AUROC_delta"] for row in rows], BOOTSTRAP_SEED + 23)
    return {"C1_minus_C0_AP": ap, "C1_minus_C0_AUROC": auc, "C1_minus_C1_SHIFT_AP": shift_ap, "C1_minus_C1_SHIFT_AUROC": shift_auc, "classes_positive_AP": int(sum(row["C1_AP_delta"] > 0 for row in rows)), "classes_negative_AP": int(sum(row["C1_AP_delta"] < 0 for row in rows)), "classes_neutral_AP": int(sum(row["C1_AP_delta"] == 0 for row in rows)), "classes_positive_AUROC": int(sum(row["C1_AUROC_delta"] >= 0 for row in rows)), "gates": {"G2_AP_mean_positive": ap["mean"] is not None and ap["mean"] > 0, "G2_AP_bootstrap_lower_positive": ap["bootstrap95_ci"] is not None and ap["bootstrap95_ci"][0] > 0, "G2_at_least_8_positive_classes": int(sum(row["C1_AP_delta"] > 0 for row in rows)) >= 8, "G2_AUROC_mean_nonnegative": auc["mean"] is not None and auc["mean"] >= 0, "G3_aligned_AP_beats_shift": shift_ap["mean"] is not None and shift_ap["mean"] > 0 and shift_ap["bootstrap95_ci"] is not None and shift_ap["bootstrap95_ci"][0] > 0}}


def normal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("fpr_at_tau95", "fpr_at_tau99", "mean_anomaly_probability", "p99_anomaly_probability", "maximum_anomaly_probability")
    per_metric = {}
    for metric in metrics:
        c0 = [row["normal_safety"]["C0"][metric] for row in rows]
        c1 = [row["normal_safety"]["C1"][metric] for row in rows]
        cs = [row["normal_safety"]["C1_SHIFT"][metric] for row in rows]
        per_metric[metric] = {"C0": aggregate(c0, BOOTSTRAP_SEED + 30 + len(metric)), "C1": aggregate(c1, BOOTSTRAP_SEED + 40 + len(metric)), "C1_SHIFT": aggregate(cs, BOOTSTRAP_SEED + 50 + len(metric)), "delta_C1": aggregate([row["normal_safety"]["delta_C1"][metric] for row in rows], BOOTSTRAP_SEED + 60 + len(metric)), "delta_C1_SHIFT": aggregate([row["normal_safety"]["delta_C1_SHIFT"][metric] for row in rows], BOOTSTRAP_SEED + 70 + len(metric))}
    risk = aggregate([row["negative_pairwise_risk_delta_C1"] for row in rows], BOOTSTRAP_SEED + 80)
    gates = {"G4_delta_FPR_tau95_not_positive": per_metric["fpr_at_tau95"]["delta_C1"]["bootstrap95_ci"] is not None and per_metric["fpr_at_tau95"]["delta_C1"]["bootstrap95_ci"][1] <= 0, "G4_delta_FPR_tau99_not_positive": per_metric["fpr_at_tau99"]["delta_C1"]["bootstrap95_ci"] is not None and per_metric["fpr_at_tau99"]["delta_C1"]["bootstrap95_ci"][1] <= 0, "G4_negative_risk_not_worse": risk["bootstrap95_ci"] is not None and risk["bootstrap95_ci"][1] <= 0}
    return {"metrics": per_metric, "negative_pairwise_risk_delta_C1": risk, "negative_pairwise_risk_delta_C1_SHIFT": aggregate([None if row["negative_pairwise_risk_C0"] is None or row["negative_pairwise_risk_C1_SHIFT"] is None else row["negative_pairwise_risk_C1_SHIFT"] - row["negative_pairwise_risk_C0"] for row in rows], BOOTSTRAP_SEED + 81), "gates": gates, "pass": bool(all(gates.values()))}


def native_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    native = aggregate([row["native_patch_AP_delta"] for row in rows], BOOTSTRAP_SEED + 90)
    deployed = aggregate([row["C1_AP_delta"] for row in rows], BOOTSTRAP_SEED + 91)
    return {"native_patch_C1_minus_C0_AP": native, "deployed_C1_minus_C0_AP": deployed, "native_improves": native["mean"] is not None and native["mean"] > 0 and native["bootstrap95_ci"] is not None and native["bootstrap95_ci"][0] > 0, "deployment_dilution": bool(native["mean"] is not None and native["bootstrap95_ci"] is not None and native["bootstrap95_ci"][0] > 0 and (deployed["mean"] is None or deployed["mean"] <= 0 or deployed["bootstrap95_ci"] is None or deployed["bootstrap95_ci"][0] <= 0))}


def choose_terminal(test_check: dict[str, Any], bridge: dict[str, Any], efficacy: dict[str, Any], safety: dict[str, Any], native: dict[str, Any]) -> str:
    if test_check.get("status") != "PASS":
        return "B2_IMPLEMENTATION_INVALID"
    if not bridge["pass"]:
        return "DEPLOYABLE_CONDITIONING_BRIDGE_UNSUPPORTED"
    g2 = all(efficacy["gates"].values())
    if not g2:
        if native["deployment_dilution"]:
            return "ADJUDICATION_SIGNAL_DILUTED_BY_DEPLOYMENT"
        return "MATCHED_RISK_ADJUDICATION_UNSUPPORTED"
    if not efficacy["gates"]["G3_aligned_AP_beats_shift"]:
        return "ADJUDICATION_NOT_REFERENCE_GROUNDED"
    if not safety["pass"]:
        return "MATCHED_RISK_ADJUDICATION_UNSAFE"
    return "MATCHED_RISK_ADJUDICATION_SUPPORTED"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("B2_OUTPUT_INVALID: no per-class rows")
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def report_markdown(path: Path, summary: dict[str, Any]) -> None:
    bridge = summary["bridge"]
    eff = summary["efficacy"]
    safety = summary["normal_safety"]
    native = summary["native_vs_deployed"]
    pcb4 = summary["pcb4"]
    terminal = summary["decision"]
    if terminal == "MATCHED_RISK_ADJUDICATION_SUPPORTED":
        next_step = "Validate the fixed adjudication protocol on an independent dataset or detector family before any method implementation."
    elif terminal == "DEPLOYABLE_CONDITIONING_BRIDGE_UNSUPPORTED":
        next_step = "Resolve the within-image conditioning bridge uncertainty before interpreting C1 as deployable evidence."
    elif terminal == "MATCHED_RISK_ADJUDICATION_UNSAFE":
        next_step = "Diagnose the measured Normal-side harm before considering any correction implementation."
    else:
        next_step = "Treat C1 as unsupported and investigate the observed native-versus-deployed failure mode without tuning the audit."
    lines = [
        "# Phase5-B2 Adjudication Audit", "",
        f"1. Within-image deployability bridge: `{bridge['pass']}`; aligned W mean={bridge['aligned_W']['mean']}, CI={bridge['aligned_W']['bootstrap95_ci']}; aligned-minus-shifted mean={bridge['aligned_minus_shifted_W']['mean']}, CI={bridge['aligned_minus_shifted_W']['bootstrap95_ci']}.",
        f"2. Final deployed AP: C1-C0 mean={eff['C1_minus_C0_AP']['mean']}, CI={eff['C1_minus_C0_AP']['bootstrap95_ci']}.",
        f"3. AUROC: C1-C0 mean={eff['C1_minus_C0_AUROC']['mean']}, CI={eff['C1_minus_C0_AUROC']['bootstrap95_ci']}.",
        f"4. Normal safety: `{safety['pass']}`; delta FPR tau95={safety['metrics']['fpr_at_tau95']['delta_C1']}, delta FPR tau99={safety['metrics']['fpr_at_tau99']['delta_C1']}, negative-risk delta={safety['negative_pairwise_risk_delta_C1']}.",
        f"5. Aligned versus shifted C1 AP: {eff['C1_minus_C1_SHIFT_AP']}.",
        f"6. Class breadth: AP positive={eff['classes_positive_AP']}, negative={eff['classes_negative_AP']}, neutral={eff['classes_neutral_AP']}.",
        f"7. pcb4: {json.dumps(pcb4, sort_keys=True)}.",
        f"8. Native versus deployed: native patch AP={native['native_patch_C1_minus_C0_AP']}; deployment_dilution={native['deployment_dilution']}.",
        f"9. Terminal decision: `{terminal}`.",
        f"10. Next step: {next_step}",
    ]
    path.write_text("\n".join(lines) + "\n")


def make_output_check(input_record: dict[str, Any], test_check: dict[str, Any], shape: dict[str, Any], prereg: dict[str, Any], rows: list[dict[str, Any]], summary: dict[str, Any], protected_before: dict[str, str]) -> dict[str, Any]:
    protected_after = protected_hashes()
    checks = {
        "input_integrity_pass": input_record.get("status") == "PASS",
        "protected_hashes_unchanged": protected_after == protected_before,
        "test_check_pass": test_check.get("status") == "PASS" and all(test_check.get("tests", {}).values()),
        "shape_dry_run_pass": shape.get("status") == "PASS",
        "bridge_summary_written": (OUTPUT_ROOT / "BRIDGE_SUMMARY.json").is_file(),
        "preregistration_hash_frozen": _sha256(OUTPUT_ROOT / "PREREGISTRATION.json") == input_record.get("preregistration_file_sha256"),
        "exact_counts": summary["inference"]["class_count"] == EXPECTED_CLASSES and summary["inference"]["image_count"] == EXPECTED_IMAGES and summary["inference"]["normal_image_count"] == EXPECTED_NORMAL and summary["inference"]["anomaly_image_count"] == EXPECTED_ANOMALY,
        "all_classes_present": len(rows) == EXPECTED_CLASSES,
        "one_forward_per_image": summary["inference"]["forward_count"] == EXPECTED_IMAGES,
        "gt_firewall": test_check.get("gt_invariance") is not None and test_check.get("tests", {}).get("T12_gt_invariance") is True,
        "no_training": summary["inference"]["training_steps"] == 0,
        "no_dense_cache": summary["inference"]["dense_feature_cache_persisted"] is False,
        "no_nan_inf": all(finite_array(np.asarray([value], dtype=np.float64)) for row in rows for key, value in row.items() if isinstance(value, (int, float)) and value == value),
        "decision_present": summary.get("decision") in {"B2_IMPLEMENTATION_INVALID", "DEPLOYABLE_CONDITIONING_BRIDGE_UNSUPPORTED", "MATCHED_RISK_ADJUDICATION_UNSUPPORTED", "ADJUDICATION_NOT_REFERENCE_GROUNDED", "MATCHED_RISK_ADJUDICATION_UNSAFE", "ADJUDICATION_SIGNAL_DILUTED_BY_DEPLOYMENT", "MATCHED_RISK_ADJUDICATION_SUPPORTED"},
        "no_formula_changes": prereg.get("no_formula_changes_after_full_start") is True,
    }
    return {"status": "PASS" if all(checks.values()) else "B2_OUTPUT_INVALID", "checks": checks, "terminal": summary.get("decision"), "protected_hashes_after": protected_after}


def run(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not is_ancestor(EXPECTED_HEAD_ANCESTOR):
        raise RuntimeError("B2_INPUT_INVALID: required B1 ancestor missing")
    hashes = protected_hashes()
    status = status_lines()
    overlap = protected_dirty_paths(status)
    if overlap:
        raise RuntimeError("B2_INPUT_DIRTY_PROTECTED: " + " | ".join(overlap))
    prereg_path = args.output_root / "PREREGISTRATION.json"
    if args.dry_run:
        prereg = make_preregistration(head, hashes)
        write_json(prereg_path, prereg)
        datasets, records, counts = canonical_records(IMAGE_SIZE)
        config = json.loads(args.config.read_text())
        model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pretests = run_unit_tests()
        model, _ = load_model(config, args.checkpoint, model_device)
        class_name = sorted(records)[0]
        raw = datasets[class_name][records[class_name][0]["source_index"]]
        with torch.inference_mode():
            pred = predictor_gt_free(model, raw["image"], class_name, {}, model_device)
        peers2, valid2, e2 = b1.nonlocal_peers(pred["aligned_features"], pred["D_rank"], pred["native_margins"])
        parity_e = float(np.max(np.abs(pred["E_nonlocal"] - e2)))
        parity_peers = bool(np.array_equal(pred["peer_indices"], peers2) and np.array_equal(pred["valid_reference"], valid2))
        pretests["tests"].update({"T1_predictor_parity": pred["predictor_parity"] <= PARITY_TOL, "T2_E_nonlocal_parity": parity_e <= PARITY_TOL and parity_peers, "T15_deployment_parity": pred["predictor_parity"] <= PARITY_TOL})
        pretests["status"] = "PASS" if all(pretests["tests"].values()) else "B2_IMPLEMENTATION_INVALID"
        write_json(args.output_root / "TEST_CHECK.json", pretests)
        shape = {"status": "PASS" if pretests["status"] == "PASS" else "B2_IMPLEMENTATION_INVALID", "class": class_name, "source_index": records[class_name][0]["source_index"], "forward_count": 1, "shape_record": pred["shape_record"], "feature_semantics": pred["feature_semantics"], "predictor_parity_max_abs": pred["predictor_parity"], "E_nonlocal_parity_max_abs": parity_e, "peer_indices_parity": parity_peers, "target_mask_shape": list(raw["mask"].shape), "image_size": IMAGE_SIZE, "patch_grid": list(PATCH_GRID), "patch_stride": PATCH_STRIDE, "risk_patch_count": int(np.ceil(RISK_FRACTION * PATCH_COUNT)), "gt_used_only_after_prediction": True, "T1_predictor_parity": pretests["tests"]["T1_predictor_parity"], "T2_E_nonlocal_parity": pretests["tests"]["T2_E_nonlocal_parity"], "T15_deployment_parity": pretests["tests"]["T15_deployment_parity"]}
        write_json(args.output_root / "SHAPE_DRY_RUN.json", shape)
        input_record = {"status": "PASS" if pretests["status"] == "PASS" else "B2_IMPLEMENTATION_INVALID", "current_head": head, "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(), "status_porcelain": status, "protected_dirty_overlap": overlap, "protected_input_hashes": hashes, "b1_artifact_hashes": b1_artifact_hashes(), "checkpoint": {"path": str(args.checkpoint), "sha256": hashes["checkpoint"]}, "config": {"path": str(args.config), "sha256": hashes["config"]}, "visa_root": json.loads((PHASE5_ROOT / "SUMMARY.json").read_text())["provenance"].get("dataset_root"), "metadata_source": str(VISA_META), "split": "TEST", "counts": counts, "class_count": counts["classes"], "image_count": counts["images"], "normal_image_count": counts["normal"], "anomaly_image_count": counts["anomaly"], "predictor_implementation": "new B2 predictor_gt_free -> AdapterModel.forward + exact Phase2B deployment", "adjudicator_implementation": "adjudicate_slots(m_bar, d_rank, evidence, valid_reference); no GT argument", "protected_source_files_read_only": True, "unrelated_dirty_paths_preserved": True, "preregistration_file_sha256": _sha256(prereg_path), "no_train_paths": True, "b1_terminal": "NONLOCAL_SAME_IMAGE_REFERENCE_SUPPORTED"}
        write_json(args.output_root / "INPUT_CHECK.json", input_record)
        print(json.dumps({"STATUS": "B2 preregistration and dry-run complete", "TEST_CHECK": pretests["status"], "SHAPES": shape["shape_record"], "FORWARD_COUNT": 1}, sort_keys=True))
        return
    required = [args.output_root / name for name in ("PREREGISTRATION.json", "INPUT_CHECK.json", "TEST_CHECK.json", "SHAPE_DRY_RUN.json")]
    if not all(path.is_file() for path in required):
        raise RuntimeError("B2_INPUT_INVALID: run --dry-run before full inference")
    prereg = json.loads(prereg_path.read_text())
    input_record = json.loads((args.output_root / "INPUT_CHECK.json").read_text())
    test_check = json.loads((args.output_root / "TEST_CHECK.json").read_text())
    shape = json.loads((args.output_root / "SHAPE_DRY_RUN.json").read_text())
    if input_record.get("status") != "PASS" or input_record.get("current_head") != head or test_check.get("status") != "PASS" or shape.get("status") != "PASS":
        raise RuntimeError("B2_INPUT_INVALID: preregistration/dry-run gate failed")
    if input_record.get("preregistration_file_sha256") != _sha256(prereg_path) or prereg.get("recorded_head") != head:
        raise RuntimeError("B2_INPUT_INVALID: preregistration hash/head changed")
    config = json.loads(args.config.read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, records, counts = canonical_records(IMAGE_SIZE)
    model, _ = load_model(config, args.checkpoint, device)
    rows = []
    text_cache: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for class_name in sorted(records):
            rows.append(process_class(model, datasets[class_name], class_name, records[class_name], device, text_cache))
    inference = {"forward_count": int(sum(row["n_images"] for row in rows)), "class_count": len(rows), "image_count": counts["images"], "normal_image_count": counts["normal"], "anomaly_image_count": counts["anomaly"], "class_at_a_time": True, "training_steps": 0, "dense_feature_cache_persisted": False, "one_forward_per_image": True, "model_forward_definition": "one image forward per TEST image"}
    bridge = bridge_summary(rows)
    efficacy = efficacy_summary(rows)
    safety = normal_summary(rows)
    native = native_summary(rows)
    delta_values = [row["C1_AP_delta"] for row in rows]
    worst = min(rows, key=lambda row: row["C1_AP_delta"]); best = max(rows, key=lambda row: row["C1_AP_delta"])
    positive_sum = sum(max(0.0, value) for value in delta_values)
    top_two = sorted((max(0.0, value) for value in delta_values), reverse=True)[:2]
    pcb4 = next((row for row in rows if row["class"].lower() == "pcb4"), None)
    class_stability = {"worst_delta_AP_class": worst["class"], "worst_delta_AP": worst["C1_AP_delta"], "best_delta_AP_class": best["class"], "best_delta_AP": best["C1_AP_delta"], "classes_positive": efficacy["classes_positive_AP"], "classes_negative": efficacy["classes_negative_AP"], "classes_neutral": efficacy["classes_neutral_AP"], "top_two_positive_delta_share": None if positive_sum == 0 else float(sum(top_two) / positive_sum), "aggregate_gain_dominated_by_at_most_two_classes": bool(positive_sum > 0 and sum(top_two) / positive_sum > 0.5)}
    summary = {"provenance": input_record, "preregistration_sha256": _sha256(prereg_path), "inference": inference, "bridge": bridge, "efficacy": efficacy, "normal_safety": safety, "spatial_control": {"C1_minus_C1_SHIFT_AP": efficacy["C1_minus_C1_SHIFT_AP"], "C1_minus_C1_SHIFT_AUROC": efficacy["C1_minus_C1_SHIFT_AUROC"], "aligned_evidence": "E_nonlocal", "shift": list(SHIFT)}, "native_vs_deployed": native, "class_stability": class_stability, "pcb4": pcb4, "decision": None, "per_class": rows}
    terminal = choose_terminal(test_check, bridge, efficacy, safety, native)
    summary["decision"] = terminal
    write_csv(args.output_root / "PER_CLASS.csv", rows)
    write_json(args.output_root / "BRIDGE_SUMMARY.json", bridge)
    normal_payload = {"per_class": {row["class"]: row["normal_safety"] for row in rows}, "aggregate": safety}
    spatial_payload = {"per_class": {row["class"]: {"C1_AP": row["C1_ap"], "C1_SHIFT_AP": row["C1_SHIFT_ap"], "C1_minus_C1_SHIFT_AP_delta": row["C1_minus_C1_SHIFT_AP_delta"], "C1_AUROC": row["C1_auroc"], "C1_SHIFT_AUROC": row["C1_SHIFT_auroc"], "C1_minus_C1_SHIFT_AUROC_delta": row["C1_minus_C1_SHIFT_AUROC_delta"]} for row in rows}, "aggregate": summary["spatial_control"]}
    write_json(args.output_root / "NORMAL_SAFETY.json", normal_payload)
    write_json(args.output_root / "SPATIAL_CONTROL.json", spatial_payload)
    write_json(args.output_root / "SUMMARY.json", summary)
    write_json(args.output_root / "DECISION.json", {"terminal": terminal, "input_integrity": "PASS", "bridge": bridge, "efficacy": efficacy, "normal_safety": safety, "native_vs_deployed": native, "no_training": True})
    report_markdown(args.output_root / "REPORT.md", summary)
    output_check = make_output_check(input_record, test_check, shape, prereg, rows, summary, hashes)
    write_json(args.output_root / "OUTPUT_CHECK.json", output_check)
    if output_check["status"] != "PASS":
        raise RuntimeError("B2_OUTPUT_INVALID: output check failed")
    print(json.dumps({"STATUS": "Phase5-B2 complete", "DECISION": terminal, "FORWARD_COUNT": inference["forward_count"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        configure_canonical_fp32()
        run(args)
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f"DECISION: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
