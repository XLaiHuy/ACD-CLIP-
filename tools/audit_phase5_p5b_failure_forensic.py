#!/usr/bin/env python3
"""Post-hoc forensic decomposition of the frozen P5B selective action.

This module is deliberately cache-only.  It reuses the frozen R0 selector,
positive-only tie projection, deployment operator, B2 occupancy semantics,
and exact metric helpers.  GT is loaded only after the GT-free action is
constructed for an image; it is never passed to selection or action code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import audit_phase5_b2_adjudication as b2  # noqa: E402
import audit_phase5_p5b_full_eval as full  # noqa: E402
from audit_phase5_hsir import ap_contamination, exact_auc_ap, pairwise_risks  # noqa: E402
from phase5_selective_adjudication import (  # noqa: E402
    apply_positive_only_projection,
    deploy_pre_softmax,
)


OUTPUT_ROOT = ROOT / "runs/phase5/hsir/P5B_FAILURE_FORENSIC_C0"
CACHE_ROOT = Path("/tmp/p5_r0_run2")
FULL_EVAL_ROOT = ROOT / "runs/phase5/hsir/P5B_FULL_EVAL"
START_HEAD = "cb8ce3518751eb0eb5224d918061160d4cd0bc7b"
EXPECTED_CACHE_SHA = "cfbd66b04c04b314756d151b759d95041afc2a69a8dc411e24896a7b4f931365"
EXPECTED_CACHE_SCHEMA = "P5B_R0_GT_FREE_CACHE_v1"
EXPECTED_IMAGES = 2162
EXPECTED_CLASSES = 12
EXPECTED_NORMAL = 962
EXPECTED_ANOMALY = 1200
PATCH_COUNT = 37 * 37
IMAGE_SIZE = 518
STAGES = 3
SHIFT = (12, 12)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 7701
CASE_NAMES = ("N0", "NN", "AA", "AN", "NA")
CASE_CODES = {name: idx for idx, name in enumerate(CASE_NAMES)}
TRANSITION_NAMES = ("WRONG", "TIE", "RIGHT")
CONDITIONS = (
    "C0", "P5", "P5_without_N0", "P5_without_NN", "P5_without_AA",
    "P5_without_AN", "P5_without_NA", "CF_N0", "CF_NN", "CF_AA",
    "CF_AN", "CF_NA", "O1_MIXED_ONLY_TIE", "O2_CORRECT_MIXED_ONLY_TIE",
    "O3_TARGET_POSITIVE_TIE", "O4_NO_NORMAL_IMAGE_ACTION",
    "O5_CORRECT_MIXED_STRICT_MIN", "T1_AN_TIE", "T2_AN_STRICT_MIN",
)
PROFILE_THRESHOLDS = {
    "dominant_fraction": 0.50,
    "substantial_case_fraction": 0.25,
    "negligible_leverage_fraction": 0.01,
    "clear_tie_gain": 1e-6,
    "spatial_inside_fraction": 0.50,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def branch() -> str:
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()


def head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def clean_tree() -> bool:
    return not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()


def only_forensic_tool_untracked() -> bool:
    lines = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    return lines == ["?? tools/audit_phase5_p5b_failure_forensic.py"]


def source_hashes() -> dict[str, str]:
    paths = (
        "model/adapter.py",
        "tools/audit_phase5_b2_adjudication.py",
        "tools/audit_phase5_b3_action_mismatch.py",
        "tools/audit_phase5_reference_validity.py",
        "tools/audit_phase5_second_evidence.py",
        "tools/audit_phase5_hsir.py",
        "tools/phase5_selective_adjudication.py",
        "tools/audit_phase5_p5b_full_eval.py",
    )
    return {path: sha256(ROOT / path) for path in paths}


def artifact_hashes() -> dict[str, str]:
    names = (
        "INPUT_CHECK.json", "PROTOCOL.json", "CACHE_CHECK.json", "PER_CLASS.csv",
        "SUMMARY.json", "NORMAL_SAFETY.json", "ACTION_DIAGNOSTICS.json",
        "DEPLOYMENT_ANALYSIS.json", "DECISION.json", "OUTPUT_CHECK.json", "REPORT.md",
    )
    return {name: sha256(FULL_EVAL_ROOT / name) for name in names if (FULL_EVAL_ROOT / name).is_file()}


def bootstrap_ci(values: Iterable[float | None], seed: int = BOOTSTRAP_SEED) -> list[float] | None:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(BOOTSTRAP_REPS, arr.size))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate(values: Iterable[float | None], seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    return {
        "mean": None if arr.size == 0 else float(arr.mean()),
        "median": None if arr.size == 0 else float(np.median(arr)),
        "p05": None if arr.size == 0 else float(np.quantile(arr, 0.05)),
        "p25": None if arr.size == 0 else float(np.quantile(arr, 0.25)),
        "p75": None if arr.size == 0 else float(np.quantile(arr, 0.75)),
        "p95": None if arr.size == 0 else float(np.quantile(arr, 0.95)),
        "min": None if arr.size == 0 else float(arr.min()),
        "max": None if arr.size == 0 else float(arr.max()),
        "n": int(arr.size),
    }


def class_bootstrap(class_values: dict[str, float | None], seed: int) -> dict[str, Any]:
    values = [v for v in class_values.values() if v is not None and np.isfinite(v)]
    return {**aggregate(values, seed), "bootstrap95_ci": bootstrap_ci(values, seed), "unit": "class", "n_classes": len(values)}


def stable_ranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    order = np.lexsort((np.arange(values.size, dtype=np.int64), -values.astype(np.float64)))
    rank = np.empty(values.size, dtype=np.int32)
    rank[order] = np.arange(values.size, dtype=np.int32)
    percentile = np.empty(values.size, dtype=np.float64)
    percentile[order] = np.arange(values.size, dtype=np.float64) / max(1, values.size - 1)
    return rank, percentile


def shifted(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return np.roll(values.reshape(37, 37), SHIFT, axis=(0, 1)).reshape(-1).astype(values.dtype)
    if values.ndim == 2 and values.shape[1] == PATCH_COUNT:
        return np.stack([shifted(row) for row in values], axis=0).astype(values.dtype)
    raise ValueError(f"unsupported shift shape {values.shape}")


def peer_jaccard(peer_indices: np.ndarray, i: int, j: int) -> float:
    left = set(int(x) for x in peer_indices[i].tolist())
    right = set(int(x) for x in peer_indices[j].tolist())
    union = left | right
    return 1.0 if not union else float(len(left & right) / len(union))


def transition(before_correct: bool, after_correct: bool) -> str:
    if not before_correct and after_correct:
        return "rescued"
    if not before_correct:
        return "missed"
    if after_correct:
        return "preserved"
    return "broken"


def state_index(positive_score: float, negative_score: float) -> int:
    if positive_score < negative_score:
        return 0
    if positive_score == negative_score:
        return 1
    return 2


def transition_matrix(scores0: np.ndarray, scores1: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Count every affected positive-negative pair without materializing a matrix."""
    scores0 = np.asarray(scores0, dtype=np.float32).reshape(-1)
    scores1 = np.asarray(scores1, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    changed = scores0 != scores1
    pos = np.flatnonzero(labels)
    neg = np.flatnonzero(~labels)
    acted_pos = pos[changed[pos]]
    acted_neg = neg[changed[neg]]
    unaffected_pos = pos[~changed[pos]]
    out = np.zeros((3, 3), dtype=np.int64)
    for p in acted_pos:
        old = np.searchsorted(np.sort(scores0[neg]), scores0[p], side="left")
        # Direct vector comparisons retain exact float32 tie semantics.
        a = np.asarray([state_index(scores0[p], x) for x in scores0[neg]], dtype=np.int8)
        b = np.asarray([state_index(scores1[p], x) for x in scores1[neg]], dtype=np.int8)
        np.add.at(out, (a, b), 1)
        del old
    for n in acted_neg:
        a = np.asarray([state_index(x, scores0[n]) for x in scores0[unaffected_pos]], dtype=np.int8)
        b = np.asarray([state_index(x, scores1[n]) for x in scores1[unaffected_pos]], dtype=np.int8)
        np.add.at(out, (a, b), 1)
    return out


def transition_matrix_fast(scores0: np.ndarray, scores1: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of transition_matrix."""
    scores0 = np.asarray(scores0, dtype=np.float32).reshape(-1)
    scores1 = np.asarray(scores1, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    changed = scores0 != scores1
    pos = np.flatnonzero(labels)
    neg = np.flatnonzero(~labels)
    out = np.zeros((3, 3), dtype=np.int64)
    acted_pos = pos[changed[pos]]
    acted_neg = neg[changed[neg]]
    unaffected_pos = pos[~changed[pos]]
    neg0, neg1 = scores0[neg], scores1[neg]
    for p in acted_pos:
        old = np.where(scores0[p] < neg0, 0, np.where(scores0[p] == neg0, 1, 2))
        new = np.where(scores1[p] < neg1, 0, np.where(scores1[p] == neg1, 1, 2))
        for a in range(3):
            for b in range(3):
                out[a, b] += int(np.sum((old == a) & (new == b)))
    pos0, pos1 = scores0[unaffected_pos], scores1[unaffected_pos]
    for n in acted_neg:
        old = np.where(pos0 < scores0[n], 0, np.where(pos0 == scores0[n], 1, 2))
        new = np.where(pos1 < scores1[n], 0, np.where(pos1 == scores1[n], 1, 2))
        for a in range(3):
            for b in range(3):
                out[a, b] += int(np.sum((old == a) & (new == b)))
    return out


def auc_credit(matrix: np.ndarray) -> float:
    weights = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    return float(np.sum(matrix * weights[None, :] - matrix * weights[:, None]))


def deploy_delta_batch(delta_cases: np.ndarray) -> np.ndarray:
    """Exact linear deployment of abnormal-only native deltas, batched by case."""
    delta_cases = np.asarray(delta_cases, dtype=np.float32)
    if delta_cases.ndim != 2 or delta_cases.shape[1] != PATCH_COUNT:
        raise ValueError(f"delta_cases shape={delta_cases.shape}")
    import torch
    import torch.nn.functional as F
    from model.adapter import gaussian_blur2d
    tensor = torch.from_numpy(delta_cases).float().reshape(delta_cases.shape[0], 1, 37, 37)
    outputs = []
    for _ in range(STAGES):
        blurred = gaussian_blur2d(tensor, (7, 7), (1, 1))
        outputs.append(F.interpolate(blurred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True))
    return torch.stack(outputs).mean(dim=0).squeeze(1).numpy()


def apply_delta(native: np.ndarray, delta: np.ndarray) -> np.ndarray:
    out = np.asarray(native, dtype=np.float32).copy()
    out[:, :, 1] += np.asarray(delta, dtype=np.float32)[None, :]
    return out


def condition_delta(name: str, case_delta: dict[str, np.ndarray], full_delta: np.ndarray, m_bar: np.ndarray, pairs: list[tuple[int, int, float]], case_by_target: dict[int, str]) -> np.ndarray:
    if name == "C0":
        return np.zeros(PATCH_COUNT, dtype=np.float32)
    if name == "P5":
        return full_delta.copy()
    if name.startswith("P5_without_"):
        return full_delta - case_delta[name[len("P5_without_"):]]
    if name.startswith("CF_"):
        return case_delta[name[3:]].copy()
    if name == "O1_MIXED_ONLY_TIE":
        return case_delta["AN"] + case_delta["NA"]
    if name == "O2_CORRECT_MIXED_ONLY_TIE":
        return case_delta["AN"].copy()
    if name == "O3_TARGET_POSITIVE_TIE":
        return case_delta["AN"] + case_delta["AA"]
    if name == "O4_NO_NORMAL_IMAGE_ACTION":
        return full_delta - case_delta["N0"]
    if name == "O5_CORRECT_MIXED_STRICT_MIN":
        return strict_delta_for_pairs(pairs, case_by_target, m_bar, only_cases={"AN"})
    if name == "T1_AN_TIE":
        return case_delta["AN"].copy()
    if name == "T2_AN_STRICT_MIN":
        return strict_delta_for_pairs(pairs, case_by_target, m_bar, only_cases={"AN"})
    raise KeyError(name)


def strict_delta_for_pairs(pairs: list[tuple[int, int, float]], case_by_target: dict[int, str], m_bar: np.ndarray, only_cases: set[str]) -> np.ndarray:
    delta = np.zeros(PATCH_COUNT, dtype=np.float32)
    for i, j, _ in pairs:
        case = case_by_target.get(int(i))
        if case not in only_cases:
            continue
        target = np.nextafter(np.float32(m_bar[j]), np.float32(np.inf))
        delta[i] = np.float32(target - np.float32(m_bar[i]))
    return delta


def relation_record(class_name: str, class_idx: int, image_index: int, source_index: int, variant: str, arrays: dict[str, np.ndarray], pair: tuple[int, int, float], labels: np.ndarray, ranks: np.ndarray, score_percentile: np.ndarray, e_percentile: np.ndarray, e_values: np.ndarray, e_stage: np.ndarray, e_loo: np.ndarray, shifted_flag: bool) -> dict[str, Any]:
    i, j, cost = int(pair[0]), int(pair[1]), float(pair[2])
    ri, ci = divmod(i, 37)
    rj, cj = divmod(j, 37)
    li, lj = bool(labels[i]), bool(labels[j])
    if li is False and lj is False:
        case = "N0" if int(labels.sum()) == 0 else "NN"
    elif li and lj:
        case = "AA"
    elif li and not lj:
        case = "AN"
    else:
        case = "NA"
    score_gap = float(np.float32(arrays["m_bar"][j]) - np.float32(arrays["m_bar"][i]))
    stage_gap = np.asarray(e_stage[:, i] - e_stage[:, j], dtype=np.float64)
    loo_gap = np.asarray(e_loo[:, i] - e_loo[:, j], dtype=np.float64)
    peer = arrays.get("peer_indices")
    return {
        "class": class_name, "class_idx": class_idx, "image_index": image_index,
        "image_id": source_index, "variant": variant, "patch_i": i, "patch_j": j,
        "row_i": ri, "col_i": ci, "row_j": rj, "col_j": cj,
        "case": case, "target_label": int(li), "comparator_label": int(lj),
        "m_bar_i": float(arrays["m_bar"][i]), "m_bar_j": float(arrays["m_bar"][j]),
        "base_rank_i": int(ranks[i]), "base_rank_j": int(ranks[j]),
        "signed_base_rank_gap": int(ranks[i] - ranks[j]), "abs_base_rank_gap": int(abs(int(ranks[i] - ranks[j]))),
        "signed_score_gap": score_gap, "abs_score_gap": abs(score_gap),
        "D_rank_i": float(arrays["D_rank"][i]), "D_rank_j": float(arrays["D_rank"][j]),
        "valid_reference_i": bool(arrays["valid_reference"][i]), "valid_reference_j": bool(arrays["valid_reference"][j]),
        "E_i": float(e_values[i]), "E_j": float(e_values[j]), "E_gap": float(e_values[i] - e_values[j]),
        "E_stage_min_gap": float(stage_gap.min()), "E_stage_mean_gap": float(stage_gap.mean()), "E_stage_std_gap": float(stage_gap.std()),
        "E_LOO_min_gap": float(loo_gap.min()), "E_LOO_mean_gap": float(loo_gap.mean()), "E_LOO_std_gap": float(loo_gap.std()),
        "score_percentile_i": float(score_percentile[i]), "score_percentile_j": float(score_percentile[j]),
        "E_percentile_i": float(e_percentile[i]), "E_percentile_j": float(e_percentile[j]),
        "delta_row": int(ri - rj), "delta_col": int(ci - cj),
        "chebyshev_distance": int(max(abs(ri - rj), abs(ci - cj))), "euclidean_distance": float(math.hypot(ri - rj, ci - cj)),
        "peer_jaccard": None if peer is None else peer_jaccard(peer, i, j),
        "action_delta": score_gap, "base_correct_mixed": bool((li and not lj) is False) if li != lj else None,
        "shifted_evidence": bool(shifted_flag),
    }


def image_case_deltas(pairs: list[tuple[int, int, float]], labels: np.ndarray, m_bar: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, dict[int, str]]:
    deltas = {name: np.zeros(PATCH_COUNT, dtype=np.float32) for name in CASE_NAMES}
    case_by_target: dict[int, str] = {}
    for i, j, _ in pairs:
        i, j = int(i), int(j)
        li, lj = bool(labels[i]), bool(labels[j])
        if not li and not lj:
            case = "N0" if int(labels.sum()) == 0 else "NN"
        elif li and lj:
            case = "AA"
        elif li:
            case = "AN"
        else:
            case = "NA"
        gap = np.float32(m_bar[j] - m_bar[i])
        deltas[case][i] = gap
        case_by_target[i] = case
    return deltas, sum(deltas.values(), np.zeros(PATCH_COUNT, dtype=np.float32)), case_by_target


def summarize_breakpoints(values: list[float]) -> dict[str, Any]:
    return {"count": int(len(values)), **aggregate(values)}


def feature_auc(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if not a.size or not b.size:
        return None
    values = np.concatenate([a, b])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], sorted_values.size]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
    u = float(ranks[:a.size].sum() - a.size * (a.size + 1) / 2.0)
    return float(u / (a.size * b.size))


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        out = np.empty(v.size, dtype=np.float64)
        sorted_v = v[order]
        starts = np.r_[0, np.flatnonzero(sorted_v[1:] != sorted_v[:-1]) + 1]
        ends = np.r_[starts[1:], v.size]
        for s, e in zip(starts, ends): out[order[s:e]] = (s + e - 1) / 2.0
        return out
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return None
    return pearson(rank(x), rank(y))


def protocol_payload(input_check: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "P5B_FAILURE_FORENSIC_C0_v1",
        "audit": "POST-HOC FORENSIC DECOMPOSITION OF FROZEN P5B SELECTIVE ACTION",
        "post_hoc_forensic": True,
        "candidate_selection_allowed": False,
        "gt_dependent_diagnostics_not_deployable": True,
        "model_forwards": 0,
        "training_steps": 0,
        "starting_head": START_HEAD,
        "dataset": {"name": "VisA", "split": "TEST", "classes": EXPECTED_CLASSES, "images": EXPECTED_IMAGES, "normal": EXPECTED_NORMAL, "anomaly": EXPECTED_ANOMALY},
        "frozen_reuse": {
            "cache": EXPECTED_CACHE_SCHEMA,
            "selector": "exact phase5_selective_adjudication.select_gt_free; no formula changes",
            "projection": "exact positive-only native abnormal-logit uplift by m_bar[j]-m_bar[i]",
            "deployment": "exact Gaussian blur 7x7 sigma=1 -> bilinear resize align_corners=True -> stage mean -> softmax",
            "occupancy": "exact B2 mask reshape 37x14x37x14 mean over stride axes; occupancy > 0",
            "shift": list(SHIFT),
            "K": 8, "risk_fraction": 0.20, "bins": 10, "unanimity": "3 stages and 8 LOO views",
            "ties": "stable descending m_bar then ascending patch ID; float32 score comparisons",
        },
        "case_taxonomy": {
            "order": list(CASE_NAMES),
            "N0": "Normal image, target i=N and comparator j=N",
            "NN": "Anomaly image, target i=N and comparator j=N",
            "AA": "Anomaly image, target i=A and comparator j=A",
            "AN": "Anomaly image, target i=A and comparator j=N",
            "NA": "Anomaly image, target i=N and comparator j=A",
            "partition_rule": "exactly one case per selected relation; GT read only after frozen action",
        },
        "global_rank_states": list(TRANSITION_NAMES),
        "descriptive_strata": {"rank_gap": ["1", "2", "3", "4-5", "6-10", ">10"]},
        "counterfactuals": {"tie": "m'_i=m_j", "strict_min": "m'_i=nextafter_float32(m_j,+inf); no epsilon or sweep", "oracle_conditions": "diagnostic only, never deployable"},
        "profile_thresholds_frozen_before_analysis": PROFILE_THRESHOLDS,
        "bootstrap": {"unit": "class", "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
        "forbidden": [
            "No candidate implementation or selector/action modification",
            "No tuning, margin sweep, threshold search, AP optimization, learned model, training, or new model forward",
            "No GT in inference, selector, action, eligibility, evidence, or deployment construction",
            "No dense maps or masks committed",
        ],
        "input_check_sha256": hashlib.sha256(json.dumps(input_check, indent=2, sort_keys=True, allow_nan=False).encode()).hexdigest(),
    }
    payload["protocol_sha256"] = hashlib.sha256(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return payload


def input_check_protocol() -> dict[str, Any]:
    status_lines = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    checks: dict[str, Any] = {
        "branch_exact": branch() == "autopilot/p5-minimal-reference-adjudication",
        "head_exact": head() == START_HEAD,
        "working_tree_clean_before_tool": only_forensic_tool_untracked(),
        "author_ident_valid": bool(subprocess.check_output(["git", "var", "GIT_AUTHOR_IDENT"], cwd=ROOT, text=True).strip()),
        "committer_ident_valid": bool(subprocess.check_output(["git", "var", "GIT_COMMITTER_IDENT"], cwd=ROOT, text=True).strip()),
        "cache_manifest_exists": (CACHE_ROOT / "CACHE_MANIFEST.json").is_file(),
        "full_eval_outputs_exist": all((FULL_EVAL_ROOT / name).is_file() for name in ("INPUT_CHECK.json", "PROTOCOL.json", "CACHE_CHECK.json", "PER_CLASS.csv", "SUMMARY.json", "NORMAL_SAFETY.json", "ACTION_DIAGNOSTICS.json", "DEPLOYMENT_ANALYSIS.json", "DECISION.json", "OUTPUT_CHECK.json", "REPORT.md")),
        "protected_sources_exist": all((ROOT / name).is_file() for name in source_hashes()),
    }
    if checks["cache_manifest_exists"]:
        manifest = json.loads((CACHE_ROOT / "CACHE_MANIFEST.json").read_text())
        checks.update({
            "cache_sha_exact": sha256(CACHE_ROOT / "CACHE_MANIFEST.json") == EXPECTED_CACHE_SHA,
            "cache_schema_exact": manifest.get("schema_version") == EXPECTED_CACHE_SCHEMA,
            "cache_finalized": bool(manifest.get("finalized")),
            "cache_images": manifest.get("scientific_unique_image_forwards") == EXPECTED_IMAGES,
            "cache_training_zero": manifest.get("training_steps") == 0,
        })
    else:
        checks.update({"cache_sha_exact": False, "cache_schema_exact": False, "cache_finalized": False, "cache_images": False, "cache_training_zero": False})
    checks["full_eval_terminal_archived"] = json.loads((FULL_EVAL_ROOT / "DECISION.json").read_text()).get("terminal") == "P5B_SELECTIVE_ADJUDICATION_UNSUPPORTED" if (FULL_EVAL_ROOT / "DECISION.json").is_file() else False
    checks["full_eval_zero_forwards"] = json.loads((FULL_EVAL_ROOT / "SUMMARY.json").read_text()).get("model_forwards") == 0 if (FULL_EVAL_ROOT / "SUMMARY.json").is_file() else False
    status = "PASS" if all(checks.values()) else "P5B_FORENSIC_INPUT_INVALID"
    return {"schema_version": "P5B_FAILURE_FORENSIC_C0_INPUT_v1", "status": status, "preflight": {"branch": branch(), "head": head(), "expected_head": START_HEAD, "working_tree_clean_before_tool": only_forensic_tool_untracked()}, "checks": checks, "cache": {"root": str(CACHE_ROOT), "manifest_sha256": sha256(CACHE_ROOT / "CACHE_MANIFEST.json") if (CACHE_ROOT / "CACHE_MANIFEST.json").is_file() else None, "schema_version": EXPECTED_CACHE_SCHEMA}, "source_hashes": source_hashes() if all((ROOT / name).is_file() for name in source_hashes()) else {}, "full_eval_artifact_hashes": artifact_hashes(), "model_forwards": 0, "training_steps": 0}


def run_protocol() -> None:
    if head() != START_HEAD or branch() != "autopilot/p5-minimal-reference-adjudication" or not only_forensic_tool_untracked():
        raise RuntimeError("P5B_FORENSIC_PROTOCOL_PREFLIGHT_BLOCKED")
    inp = input_check_protocol()
    if inp["status"] != "PASS":
        write_json(OUTPUT_ROOT / "INPUT_CHECK.json", inp)
        raise RuntimeError(inp["status"])
    write_json(OUTPUT_ROOT / "INPUT_CHECK.json", inp)
    write_json(OUTPUT_ROOT / "PROTOCOL.json", protocol_payload(inp))
    print(json.dumps({"status": "PASS", "model_forwards": 0, "training_steps": 0, "protocol": str(OUTPUT_ROOT / "PROTOCOL.json")}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("protocol", "forensic"), required=True)
    parser.add_argument("--protocol-commit")
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    args = parser.parse_args()
    if args.mode == "protocol":
        run_protocol()
        return
    raise RuntimeError("P5B_FORENSIC_NOT_YET_IMPLEMENTED")


if __name__ == "__main__":
    main()
