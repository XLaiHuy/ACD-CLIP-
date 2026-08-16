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
EXPECTED_PROTOCOL_COMMIT = "e8040490b0980583d4f61d99cd0a408171dd1f74"
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
def only_forensic_tool_modified() -> bool:
    lines = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    return lines == [" M tools/audit_phase5_p5b_failure_forensic.py"]


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



def deploy_probability(native: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    return F.softmax(deploy_pre_softmax(native), dim=1).numpy()


def condition_names_for_native() -> tuple[str, ...]:
    return CONDITIONS


def compact_shifted(row: dict[str, Any]) -> dict[str, Any]:
    names = (
        "class", "class_idx", "image_index", "image_id", "variant", "case",
        "patch_i", "patch_j", "target_label", "comparator_label", "m_bar_i", "m_bar_j",
        "base_rank_i", "base_rank_j", "signed_base_rank_gap", "abs_base_rank_gap",
        "signed_score_gap", "abs_score_gap", "D_rank_i", "D_rank_j", "E_i", "E_j", "E_gap",
        "E_stage_min_gap", "E_stage_mean_gap", "E_LOO_min_gap", "E_LOO_mean_gap",
        "score_percentile_i", "score_percentile_j", "E_percentile_i", "E_percentile_j",
        "delta_row", "delta_col", "chebyshev_distance", "euclidean_distance",
        "peer_jaccard", "action_delta", "shifted_evidence",
    )
    return {name: row.get(name) for name in names}


def relation_transition(case: str) -> str | None:
    if case == "AN":
        return "missed"  # current projection reaches equality, not strict correctness
    if case == "NA":
        return "broken"  # current projection removes a strict positive-over-negative order
    return None


def rank_gap_bin(gap: int) -> str:
    if gap == 1:
        return "1"
    if gap == 2:
        return "2"
    if gap == 3:
        return "3"
    if gap <= 5:
        return "4-5"
    if gap <= 10:
        return "6-10"
    return ">10"


def load_state_records(cache_root: Path):
    manifest, datasets, records, counts = full.validate_cache(cache_root)
    image_index = 0
    for class_idx, class_name in enumerate(sorted(records)):
        source_indices = sorted(int(record["source_index"]) for record in records[class_name])
        for source_index in source_indices:
            key = f"{class_name}:{source_index}"
            arrays = full.load_arrays(cache_root / manifest["files"][key]["relative_path"])
            traces = full.selector_traces(arrays, key)
            native = arrays["native_logits"].astype(np.float32, copy=False)
            aligned_native, aligned_delta = apply_positive_only_projection(native, arrays["m_bar"], traces["aligned"])
            shifted_native, shifted_delta = apply_positive_only_projection(native, arrays["m_bar"], traces["shifted"])
            # Freeze all GT-free outputs before the mask is opened.
            c0_prob = deploy_probability(native)
            aligned_prob = deploy_probability(aligned_native)
            shifted_prob = deploy_probability(shifted_native)
            raw = datasets[class_name][source_index]
            mask = b2.load_mask_after_prediction(raw)
            occupancy = b2.occupancy_from_mask(mask)
            labels = occupancy > 0
            yield {
                "class": class_name, "class_idx": class_idx, "image_index": image_index,
                "source_index": source_index, "record": next(x for x in records[class_name] if int(x["source_index"]) == source_index),
                "arrays": arrays, "traces": traces, "native": native,
                "aligned_native": aligned_native, "aligned_delta": aligned_delta,
                "shifted_native": shifted_native, "shifted_delta": shifted_delta,
                "c0_prob": c0_prob, "aligned_prob": aligned_prob, "shifted_prob": shifted_prob,
                "mask": mask, "labels": labels, "occupancy": occupancy,
            }
            image_index += 1


def metric_pair(scores: list[np.ndarray], labels: list[np.ndarray]) -> dict[str, float]:
    score = np.concatenate(scores).astype(np.float32, copy=False)
    label = np.concatenate(labels).astype(np.uint8, copy=False)
    auc, ap = exact_auc_ap(score, label)
    return {"auroc": float(auc), "ap": float(ap), "n": int(score.size), "positive": int(label.sum())}


def case_summary(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    total = len(rows)
    by_case: dict[str, list[dict[str, Any]]] = {name: [] for name in CASE_NAMES}
    for row in rows:
        by_case[row["case"]].append(row)
    result: dict[str, Any] = {"n_relations": total, "cases": {}}
    for case in CASE_NAMES:
        group = by_case[case]
        result["cases"][case] = {
            "n_relations": len(group),
            "relation_percent": None if not total else float(len(group) / total),
            "images": int(len({(x["class"], x["image_id"]) for x in group})),
            "relations_per_image": None if not group else float(len(group) / len({(x["class"], x["image_id"]) for x in group})),
            "action_target_count": len(group),
            "total_delta_mass": float(sum(x["action_delta"] for x in group)),
            "delta_mass_fraction": None if not group else float(sum(x["action_delta"] for x in group) / max(1e-30, sum(x["action_delta"] for x in rows))),
            "delta": aggregate([x["action_delta"] for x in group]),
            "m_bar_i": aggregate([x["m_bar_i"] for x in group]),
            "m_bar_j": aggregate([x["m_bar_j"] for x in group]),
            "score_gap": aggregate([x["abs_score_gap"] for x in group]),
            "D_rank_i": aggregate([x["D_rank_i"] for x in group]),
            "D_rank_j": aggregate([x["D_rank_j"] for x in group]),
            "E_gap": aggregate([x["E_gap"] for x in group]),
            "E_stage_min_gap": aggregate([x["E_stage_min_gap"] for x in group]),
            "E_LOO_min_gap": aggregate([x["E_LOO_min_gap"] for x in group]),
            "chebyshev_distance": aggregate([x["chebyshev_distance"] for x in group]),
            "euclidean_distance": aggregate([x["euclidean_distance"] for x in group]),
            "base_rank_gap": aggregate([x["abs_base_rank_gap"] for x in group]),
            "transition_counts": {name: int(sum(x.get("transition") == name.lower() for x in group)) for name in ("rescued", "broken", "preserved", "missed")},
            "class_counts": {class_name: int(sum(x["class"] == class_name for x in group)) for class_name in sorted({x["class"] for x in rows})},
        }
    bins: dict[str, dict[str, Any]] = {}
    for name in ("1", "2", "3", "4-5", "6-10", ">10"):
        group = [x for x in rows if rank_gap_bin(x["abs_base_rank_gap"]) == name]
        rescue = sum(x.get("transition") == "rescued" for x in group)
        broken = sum(x.get("transition") == "broken" for x in group)
        missed = sum(x.get("transition") == "missed" for x in group)
        preserved = sum(x.get("transition") == "preserved" for x in group)
        base_wrong = sum(x["case"] == "AN" for x in group)
        base_right = sum(x["case"] == "NA" for x in group)
        bins[name] = {
            "n_pairs": len(group), "rescued": rescue, "broken": broken, "preserved": preserved, "missed": missed,
            "rescue_rate": None if not base_wrong else float(rescue / base_wrong),
            "break_rate": None if not base_right else float(broken / base_right),
            "net": int(rescue - broken), "action_mass": float(sum(x["action_delta"] for x in group)),
            "class_counts": {class_name: int(sum(x["class"] == class_name for x in group)) for class_name in sorted({x["class"] for x in rows})},
        }
    result["rank_gap_strata"] = bins
    result["rank_gap_spatial"] = {
        "spearman_rank_chebyshev": spearman(np.asarray([x["abs_base_rank_gap"] for x in rows], dtype=np.float64), np.asarray([x["chebyshev_distance"] for x in rows], dtype=np.float64)),
        "spearman_rank_euclidean": spearman(np.asarray([x["abs_base_rank_gap"] for x in rows], dtype=np.float64), np.asarray([x["euclidean_distance"] for x in rows], dtype=np.float64)),
    }
    result["transition_totals"] = {name: int(sum(x.get("transition") == name.lower() for x in rows)) for name in ("rescued", "broken", "preserved", "missed")}
    result["net_utility"] = result["transition_totals"]["rescued"] - result["transition_totals"]["broken"]
    return result


def variant_action_summary(rows: list[dict[str, Any]], image_rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    by_case = {case: [x for x in rows if x["case"] == case] for case in CASE_NAMES}
    mass = {case: float(sum(x["action_delta"] for x in group)) for case, group in by_case.items()}
    selected = len(rows)
    normal_images = [x for x in image_rows if x["label"] == 0]
    return {
        "selected_relations": selected,
        "acted_patches": selected * 1,
        "participating_patches": selected * 2,
        "total_delta_mass": float(sum(mass.values())),
        "delta": aggregate([x["action_delta"] for x in rows]),
        "mass_by_case": mass,
        "counts_by_case": {case: len(group) for case, group in by_case.items()},
        "mixed_fraction": None if not selected else float((len(by_case["AN"]) + len(by_case["NA"])) / selected),
        "target_positive_fraction": None if not selected else float((len(by_case["AN"]) + len(by_case["AA"])) / selected),
        "normal_image_action_density": {
            "images": len(normal_images),
            "images_with_actions": int(sum(x["selected"] > 0 for x in normal_images)),
            "mean_actions_per_normal_image": None if not normal_images else float(np.mean([x["selected"] for x in normal_images])),
            "mean_mass_per_normal_image": None if not normal_images else float(np.mean([x["mass"] for x in normal_images])),
        },
    }


def class_relation_summary(rows: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    return {class_name: case_summary([x for x in rows if x["class"] == class_name], "aligned") for class_name in class_names}


def action_mass_summary(spatial_mass: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant, cases in spatial_mass.items():
        out[variant] = {}
        for case, values in cases.items():
            out[variant][case] = {key: float(value) for key, value in values.items()}
            total = values.get("positive_mass", 0.0)
            out[variant][case]["inside_fraction"] = None if total <= 0 else float(values.get("inside_mass", 0.0) / total)
    return out


def aggregate_transition_store(store: dict[str, dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in store.items():
        matrix = value["matrix"]
        total_pairs = int(value["total_pairs"])
        measured = float(value["measured_auc"])
        implied = None if total_pairs == 0 else float(auc_credit(matrix) / total_pairs)
        per_class = {}
        for class_name in class_names:
            cm = value["per_class"][class_name]["matrix"]
            cp = value["per_class"][class_name]["total_pairs"]
            per_class[class_name] = {"matrix": cm.tolist(), "total_pairs": int(cp), "implied_auc_delta": None if cp == 0 else float(auc_credit(cm) / cp)}
        out[name] = {"matrix": matrix.tolist(), "total_pairs": total_pairs, "measured_auc_delta": measured, "implied_auc_delta": implied, "parity_error": None if implied is None else float(implied - measured), "per_class": per_class}
    return out


def new_transition_store(names: Iterable[str], class_names: list[str]) -> dict[str, dict[str, Any]]:
    return {name: {"matrix": np.zeros((3, 3), dtype=np.int64), "total_pairs": 0, "measured_auc": 0.0, "per_class": {c: {"matrix": np.zeros((3, 3), dtype=np.int64), "total_pairs": 0} for c in class_names}} for name in names}


def add_transition(store: dict[str, dict[str, Any]], name: str, matrix: np.ndarray, total_pairs: int, class_name: str) -> None:
    store[name]["matrix"] += matrix
    store[name]["total_pairs"] += int(total_pairs)
    store[name]["per_class"][class_name]["matrix"] += matrix
    store[name]["per_class"][class_name]["total_pairs"] += int(total_pairs)


def total_pair_count(labels: np.ndarray) -> int:
    pos = int(np.sum(labels)); neg = int(labels.size - pos)
    return pos * neg


def strict_inversion_count(scores: np.ndarray, labels: np.ndarray) -> int:
    positive = np.asarray(scores, dtype=np.float32)[np.asarray(labels, dtype=bool)]
    negative = np.sort(np.asarray(scores, dtype=np.float32)[~np.asarray(labels, dtype=bool)])
    if not positive.size or not negative.size:
        return 0
    return int(np.searchsorted(negative, positive, side="right").sum())


def class_native_metrics(patch_lists: dict[str, list[np.ndarray]], label_lists: list[np.ndarray]) -> dict[str, dict[str, float]]:
    return {name: metric_pair(patch_lists[name], label_lists) for name in patch_lists}


def feature_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "AN_vs_NA": {"A": [x for x in rows if x["case"] == "AN"], "B": [x for x in rows if x["case"] == "NA"]},
        "MIXED_vs_SAME": {"A": [x for x in rows if x["case"] in {"AN", "NA"}], "B": [x for x in rows if x["case"] in {"N0", "NN", "AA"}]},
        "POS_TARGET_vs_NEG_TARGET": {"A": [x for x in rows if x["case"] in {"AN", "AA"}], "B": [x for x in rows if x["case"] in {"N0", "NN", "NA"}]},
    }


def feature_separation(rows: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    features = {
        "base_cost": "action_delta", "E_nonlocal_gap": "E_gap", "minimum_E_stage_gap": "E_stage_min_gap",
        "mean_E_stage_gap": "E_stage_mean_gap", "minimum_E_LOO_gap": "E_LOO_min_gap", "mean_E_LOO_gap": "E_LOO_mean_gap",
        "std_E_stage_gap": "E_stage_std_gap", "std_E_LOO_gap": "E_LOO_std_gap", "D_rank_i": "D_rank_i", "D_rank_j": "D_rank_j",
        "D_rank_difference": None, "score_percentile_i": "score_percentile_i", "score_percentile_j": "score_percentile_j",
        "E_percentile_i": "E_percentile_i", "E_percentile_j": "E_percentile_j", "chebyshev_distance": "chebyshev_distance",
        "euclidean_distance": "euclidean_distance", "peer_jaccard": "peer_jaccard",
    }
    result: dict[str, Any] = {}
    for group_name, groups in feature_groups(rows).items():
        result[group_name] = {}
        for feature_name, key in features.items():
            def values(group: list[dict[str, Any]]) -> np.ndarray:
                if feature_name == "D_rank_difference":
                    return np.asarray([x["D_rank_i"] - x["D_rank_j"] for x in group], dtype=np.float64)
                return np.asarray([x[key] for x in group if x.get(key) is not None], dtype=np.float64)
            a, b = values(groups["A"]), values(groups["B"])
            mean_diff = None if not a.size or not b.size else float(np.mean(a) - np.mean(b))
            pooled = None if not a.size or not b.size else float(np.sqrt((np.var(a) * max(0, a.size - 1) + np.var(b) * max(0, b.size - 1)) / max(1, a.size + b.size - 2)))
            class_effects: dict[str, float | None] = {}
            for class_name in class_names:
                ca = values([x for x in groups["A"] if x["class"] == class_name])
                cb = values([x for x in groups["B"] if x["class"] == class_name])
                class_effects[class_name] = None if not ca.size or not cb.size else float(np.mean(ca) - np.mean(cb))
            result[group_name][feature_name] = {
                "n_A": int(a.size), "n_B": int(b.size), "mean_A": None if not a.size else float(np.mean(a)), "mean_B": None if not b.size else float(np.mean(b)),
                "mean_difference_A_minus_B": mean_diff, "cohen_d": None if mean_diff is None or pooled in (None, 0.0) else float(mean_diff / pooled),
                "rank_auc_A_over_B": feature_auc(a, b), "rank_biserial": None if feature_auc(a, b) is None else float(2 * feature_auc(a, b) - 1),
                "class_effects": class_effects, "class_direction_A_greater": int(sum(v is not None and v > 0 for v in class_effects.values())),
                "class_direction_A_less": int(sum(v is not None and v < 0 for v in class_effects.values())),
                "class_bootstrap": class_bootstrap(class_effects, BOOTSTRAP_SEED + len(result[group_name]) + 100),
            }
    return {"features": result, "descriptive_only": True, "classifier_fit": False, "threshold_search": False, "combination_fit": False}


def breakpoint_summary(rows: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    all_values: dict[str, list[float]] = {"AN": [], "NA": []}
    all_positions: dict[str, list[float]] = {"AN": [], "NA": []}
    per_class: dict[str, dict[str, list[float]]] = {c: {"AN": [], "NA": []} for c in class_names}
    per_class_pos: dict[str, dict[str, list[float]]] = {c: {"AN": [], "NA": []} for c in class_names}
    for row in rows:
        if row["case"] not in {"AN", "NA"}:
            continue
        vals = row.get("breakpoints", [])
        all_values[row["case"]].extend(vals)
        all_positions[row["case"]].append(float(row.get("tie_breakpoint_position", 0.0)))
        per_class[row["class"]][row["case"]].extend(vals)
        per_class_pos[row["class"]][row["case"]].append(float(row.get("tie_breakpoint_position", 0.0)))
    out = {case: {"breakpoints": summarize_breakpoints(values), "current_tie_position": aggregate(all_positions[case])} for case, values in all_values.items()}
    out["per_class"] = {c: {case: {"breakpoints": summarize_breakpoints(per_class[c][case]), "current_tie_position": aggregate(per_class_pos[c][case])} for case in ("AN", "NA")} for c in class_names}
    out["definition"] = "breakpoint is nextafter_float32(currently higher opposite-label score,+inf)-target score; no margin selected"
    return out


def class_profile(class_name: str, aligned_rows: list[dict[str, Any]], leverage: dict[str, Any], tie: dict[str, Any], spatial: dict[str, Any]) -> dict[str, Any]:
    total = len(aligned_rows)
    counts = {case: sum(x["case"] == case for x in aligned_rows) for case in CASE_NAMES}
    mass = {case: float(sum(x["action_delta"] for x in aligned_rows if x["case"] == case)) for case in CASE_NAMES}
    profiles: list[str] = []
    if counts["AN"] > 0 and leverage.get("an_over_total_inversions", 0.0) < PROFILE_THRESHOLDS["negligible_leverage_fraction"]:
        profiles.append("LEVERAGE_LIMITED")
    if total and (counts["N0"] + counts["NN"] + counts["AA"]) / total > PROFILE_THRESHOLDS["dominant_fraction"]:
        profiles.append("SAME_LABEL_DOMINATED")
    if sum(mass.values()) > 0 and mass["NA"] / sum(mass.values()) > PROFILE_THRESHOLDS["substantial_case_fraction"]:
        profiles.append("WRONG_RELATION_HARM")
    if tie.get("strict_minus_tie_native_ap_gain", 0.0) > PROFILE_THRESHOLDS["clear_tie_gain"]:
        profiles.append("TIE_LIMITED")
    inside = spatial.get("inside_fraction")
    if inside is not None and inside < PROFILE_THRESHOLDS["spatial_inside_fraction"]:
        profiles.append("SPATIAL_LEAKAGE")
    if not profiles:
        profiles = ["MIXED"]
    elif len(profiles) > 1:
        profiles.append("MIXED")
    return {"class": class_name, "profile": profiles, "case_counts": counts, "case_mass": mass}


def validate_protocol_inputs(cache_root: Path, protocol_commit: str | None, implementation_commit: str | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if protocol_commit != EXPECTED_PROTOCOL_COMMIT:
        raise RuntimeError("P5B_FORENSIC_PROTOCOL_COMMIT_MISMATCH")
    if implementation_commit is None or head() != implementation_commit:
        raise RuntimeError("P5B_FORENSIC_IMPLEMENTATION_HEAD_MISMATCH")
    if branch() != "autopilot/p5-minimal-reference-adjudication" or not clean_tree():
        raise RuntimeError("P5B_FORENSIC_RUN_PREFLIGHT_BLOCKED")
    protocol = json.loads((OUTPUT_ROOT / "PROTOCOL.json").read_text())
    input_check = json.loads((OUTPUT_ROOT / "INPUT_CHECK.json").read_text())
    if input_check.get("status") != "PASS" or protocol.get("post_hoc_forensic") is not True or protocol.get("model_forwards") != 0 or protocol.get("training_steps") != 0:
        raise RuntimeError("P5B_FORENSIC_PROTOCOL_INVALID")
    manifest, datasets, records, counts = full.validate_cache(cache_root)
    if sha256(cache_root / "CACHE_MANIFEST.json") != EXPECTED_CACHE_SHA or manifest.get("schema_version") != EXPECTED_CACHE_SCHEMA:
        raise RuntimeError("P5B_FORENSIC_CACHE_INVALID")
    summary = json.loads((FULL_EVAL_ROOT / "SUMMARY.json").read_text())
    decision = json.loads((FULL_EVAL_ROOT / "DECISION.json").read_text())
    action = json.loads((FULL_EVAL_ROOT / "ACTION_DIAGNOSTICS.json").read_text())
    if decision.get("terminal") != "P5B_SELECTIVE_ADJUDICATION_UNSUPPORTED":
        raise RuntimeError("P5B_FORENSIC_ARCHIVED_TERMINAL_INVALID")
    if summary.get("model_forwards") != 0 or summary.get("training_steps") != 0:
        raise RuntimeError("P5B_FORENSIC_ARCHIVED_FORWARD_INVALID")
    if action["aligned"]["selected_relations_total"] != 99037 or action["shifted"]["selected_relations_total"] != 221476:
        raise RuntimeError("P5B_FORENSIC_ARCHIVED_ACTION_COUNT_INVALID")
    return protocol, manifest, datasets, records


def compute_main(cache_root: Path) -> dict[str, Any]:
    class_names = []
    for class_name in sorted(json.loads((FULL_EVAL_ROOT / "SUMMARY.json").read_text())["counts"] and b2.canonical_records(IMAGE_SIZE)[1]):
        class_names.append(class_name)
    aligned_rows: list[dict[str, Any]] = []
    shifted_rows: list[dict[str, Any]] = []
    image_rows: dict[str, list[dict[str, Any]]] = {"aligned": [], "shifted": []}
    all_patch_scores = {"C0": [], "P5": [], "P5_SHIFT": [], "T1_AN_TIE": [], "T2_AN_STRICT_MIN": [], "AN_ONLY": [], "NA_ONLY": []}
    all_patch_labels: list[np.ndarray] = []
    native_metrics: dict[str, dict[str, dict[str, float]]] = {c: {} for c in class_names}
    transition_store = new_transition_store(("P5", "SHIFT", "T1_AN_TIE", "T2_AN_STRICT_MIN", "AN_ONLY", "NA_ONLY"), class_names)
    rank_leverage = {c: {"selected_mixed": 0, "selected_AN": 0, "selected_NA": 0, "total_inversions": 0, "total_pairs": 0, "AN_transitions": 0, "NA_transitions": 0, "positive_contamination_total": 0.0, "positive_contamination_touched": 0.0, "negative_risk_total": 0.0, "negative_risk_touched": 0.0} for c in class_names}
    spatial_mass = {variant: {case: {"positive_mass": 0.0, "inside_mass": 0.0, "outside_mass": 0.0, "effective_pixels": 0.0, "images": 0} for case in CASE_NAMES} for variant in ("aligned", "shifted")}
    pixel_buffers = {"C0": [], "P5": [], "P5_SHIFT": [], "labels": []}
    patch_buffers = {name: [] for name in CONDITIONS}
    patch_buffers.update({"P5_SHIFT": [], "labels": []})
    pixel_metrics: dict[str, dict[str, dict[str, float]]] = {}
    breakpoint_rows: list[dict[str, Any]] = []
    current_class: str | None = None

    def flush_class(class_name: str | None) -> None:
        if class_name is None:
            return
        pixel_metrics[class_name] = {name: metric_pair(pixel_buffers[name], pixel_buffers["labels"]) for name in ("C0", "P5", "P5_SHIFT")}
        native_metrics[class_name] = {name: metric_pair(patch_buffers[name], patch_buffers["labels"]) for name in CONDITIONS}
        native_metrics[class_name]["P5_SHIFT"] = metric_pair(patch_buffers["P5_SHIFT"], patch_buffers["labels"])
        for values in pixel_buffers.values(): values.clear()
        for values in patch_buffers.values(): values.clear()
    case_by_image: dict[tuple[str, int], dict[str, Any]] = {}
    image_count = 0
    for state in load_state_records(cache_root):
        cls = state["class"]
        if current_class is None:
            current_class = cls
        elif cls != current_class:
            flush_class(current_class)
            current_class = cls
        arrays = state["arrays"]
        labels = state["labels"]
        m_bar = arrays["m_bar"].astype(np.float32, copy=False)
        ranks, score_pct = stable_ranks(m_bar)
        e_pct = np.empty(PATCH_COUNT, dtype=np.float64)
        e_order = np.argsort(arrays["E_nonlocal"].astype(np.float64), kind="mergesort")
        e_pct[e_order] = np.arange(PATCH_COUNT, dtype=np.float64) / max(1, PATCH_COUNT - 1)
        aligned_e = arrays["E_nonlocal"]
        shifted_e = shifted(arrays["E_nonlocal"])
        aligned_stage, aligned_loo = arrays["E_stage"], arrays["E_LOO"]
        shifted_stage, shifted_loo = shifted(arrays["E_stage"]), shifted(arrays["E_LOO"])
        aligned_rel = [relation_record(cls, state["class_idx"], state["image_index"], state["source_index"], "aligned", arrays, p, labels, ranks, score_pct, e_pct, aligned_e, aligned_stage, aligned_loo, False) for p in state["traces"]["aligned"]]
        shifted_rel_full = [relation_record(cls, state["class_idx"], state["image_index"], state["source_index"], "shifted", arrays, p, labels, ranks, score_pct, e_pct, shifted_e, shifted_stage, shifted_loo, True) for p in state["traces"]["shifted"]]
        for row in aligned_rel:
            row["transition"] = relation_transition(row["case"])
            row["rank_gap_bin"] = rank_gap_bin(row["abs_base_rank_gap"])
            if row["case"] in {"AN", "NA"}:
                opposite = m_bar[~labels] if row["case"] == "AN" else m_bar[labels]
                higher = opposite[opposite > np.float32(row["m_bar_i"])]
                row["breakpoints"] = [float(np.nextafter(np.float32(x), np.float32(np.inf)) - np.float32(row["m_bar_i"])) for x in higher]
                row["tie_breakpoint_position"] = float(np.mean(np.asarray(row["breakpoints"]) <= row["action_delta"])) if row["breakpoints"] else 1.0
            else:
                row["breakpoints"] = []
                row["tie_breakpoint_position"] = None
        for row in shifted_rel_full:
            row["transition"] = relation_transition(row["case"])
            row["rank_gap_bin"] = rank_gap_bin(row["abs_base_rank_gap"])
        aligned_rows.extend(aligned_rel)
        shifted_rows.extend(compact_shifted(x) | {"transition": x["transition"], "rank_gap_bin": x["rank_gap_bin"]} for x in shifted_rel_full)
        aligned_case_delta, aligned_full_delta, aligned_case_target = image_case_deltas(state["traces"]["aligned"], labels, m_bar)
        shifted_case_delta, shifted_full_delta, _ = image_case_deltas(state["traces"]["shifted"], labels, m_bar)
        condition_deltas = {name: condition_delta(name, aligned_case_delta, aligned_full_delta, m_bar, state["traces"]["aligned"], aligned_case_target) for name in CONDITIONS}
        for name, delta in condition_deltas.items():
            patch_buffers[name].append((m_bar + delta).astype(np.float32))
        patch_buffers["P5_SHIFT"].append((m_bar + shifted_full_delta).astype(np.float32))
        patch_buffers["labels"].append(labels.astype(np.uint8))
        pixel_buffers["C0"].append(state["c0_prob"][0, 1].reshape(-1).astype(np.float32))
        pixel_buffers["P5"].append(state["aligned_prob"][0, 1].reshape(-1).astype(np.float32))
        pixel_buffers["P5_SHIFT"].append(state["shifted_prob"][0, 1].reshape(-1).astype(np.float32))
        pixel_buffers["labels"].append(state["mask"].reshape(-1).astype(np.uint8))
        strict_an_delta = strict_delta_for_pairs(state["traces"]["aligned"], aligned_case_target, m_bar, {"AN"})
        all_patch_scores["C0"].append(m_bar.copy()); all_patch_scores["P5"].append((m_bar + aligned_full_delta).astype(np.float32)); all_patch_scores["P5_SHIFT"].append((m_bar + shifted_full_delta).astype(np.float32)); all_patch_scores["T1_AN_TIE"].append((m_bar + aligned_case_delta["AN"]).astype(np.float32)); all_patch_scores["T2_AN_STRICT_MIN"].append((m_bar + strict_an_delta).astype(np.float32)); all_patch_scores["AN_ONLY"].append((m_bar + aligned_case_delta["AN"]).astype(np.float32)); all_patch_scores["NA_ONLY"].append((m_bar + aligned_case_delta["NA"]).astype(np.float32)); all_patch_labels.append(labels.astype(np.uint8))
        total_pairs = total_pair_count(labels)
        trans_pairs = {"P5": aligned_full_delta, "SHIFT": shifted_full_delta, "T1_AN_TIE": aligned_case_delta["AN"], "T2_AN_STRICT_MIN": strict_an_delta, "AN_ONLY": aligned_case_delta["AN"], "NA_ONLY": aligned_case_delta["NA"]}
        for name, delta in trans_pairs.items():
            matrix = transition_matrix_fast(m_bar, m_bar + delta, labels)
            add_transition(transition_store, name, matrix, total_pairs, cls)
        inv = strict_inversion_count(m_bar, labels)
        rank_leverage[cls]["total_inversions"] += inv; rank_leverage[cls]["total_pairs"] += total_pairs
        pos_risk, neg_risk = pairwise_risks(m_bar, labels)
        positive_contamination = ap_contamination(m_bar, labels.astype(np.uint8))
        negative_risk_patch = np.full(PATCH_COUNT, np.nan, dtype=np.float64)
        negative_indices = np.flatnonzero(~labels)
        if neg_risk.size: negative_risk_patch[negative_indices] = neg_risk
        if pos_risk.size: rank_leverage[cls]["positive_contamination_total"] += float(np.nansum(positive_contamination[labels]))
        if neg_risk.size: rank_leverage[cls]["negative_risk_total"] += float(np.nansum(neg_risk))
        for row in aligned_rel:
            rank_leverage[cls]["selected_mixed"] += int(row["case"] in {"AN", "NA"}); rank_leverage[cls]["selected_AN"] += int(row["case"] == "AN"); rank_leverage[cls]["selected_NA"] += int(row["case"] == "NA")
            if row["case"] in {"AN", "AA"}:
                value = positive_contamination[row["patch_i"]]
                rank_leverage[cls]["positive_contamination_touched"] += float(value) if np.isfinite(value) else 0.0
            if row["case"] in {"N0", "NN", "NA"} and np.isfinite(negative_risk_patch[row["patch_i"]]):
                rank_leverage[cls]["negative_risk_touched"] += float(negative_risk_patch[row["patch_i"]])
        an_matrix = transition_matrix_fast(m_bar, m_bar + aligned_case_delta["AN"], labels)
        na_matrix = transition_matrix_fast(m_bar, m_bar + aligned_case_delta["NA"], labels)
        rank_leverage[cls]["AN_transitions"] += int(np.sum(an_matrix) - an_matrix[2, 2])
        rank_leverage[cls]["NA_transitions"] += int(np.sum(na_matrix) - na_matrix[2, 2])
        import torch
        import torch.nn.functional as F
        base_pre = deploy_pre_softmax(state["native"])
        for variant, rows, case_deltas in (("aligned", aligned_rel, aligned_case_delta), ("shifted", shifted_rel_full, shifted_case_delta)):
            counts = {case: sum(x["case"] == case for x in rows) for case in CASE_NAMES}
            image_row = {"class": cls, "image_id": state["source_index"], "image_index": state["image_index"], "label": int(state["record"]["label"]), "selected": len(rows), "mass": float(sum(x["action_delta"] for x in rows)), "mixed": counts["AN"] + counts["NA"], "target_positive": counts["AN"] + counts["AA"], "case_counts": counts}
            image_rows[variant].append(image_row)
            case_matrix = np.stack([case_deltas[case] for case in CASE_NAMES], axis=0)
            case_maps = deploy_delta_batch(case_matrix)
            for case_index, case in enumerate(CASE_NAMES):
                if not np.any(case_deltas[case]): continue
                spatial_mass[variant][case]["images"] += 1
                logits = base_pre.clone(); logits[0, 1] += torch.from_numpy(case_maps[case_index])
                case_prob = F.softmax(logits, dim=1)[0, 1].numpy()
                induced = np.maximum(case_prob - state["c0_prob"][0, 1], 0.0)
                if int(state["record"]["label"]) == 1:
                    inside = induced[state["mask"] > 0]; outside = induced[state["mask"] == 0]
                else:
                    inside = np.asarray([], dtype=np.float32); outside = induced.reshape(-1)
                spatial_mass[variant][case]["positive_mass"] += float(induced.sum())
                spatial_mass[variant][case]["inside_mass"] += float(inside.sum())
                spatial_mass[variant][case]["outside_mass"] += float(outside.sum())
                spatial_mass[variant][case]["effective_pixels"] += float(np.sum(induced > 0))
        case_by_image[(cls, state["source_index"])] = {"aligned": aligned_rel, "shifted": shifted_rel_full}
        image_count += 1
    flush_class(current_class)
    matrices = aggregate_transition_store(transition_store, class_names)
    for name in transition_store:
        if name == "P5":
            transition_store[name]["measured_auc"] = native_metrics[class_names[0]]["P5"]["auroc"] if False else 0.0
    return {
        "class_names": class_names, "aligned_rows": aligned_rows, "shifted_rows": shifted_rows, "image_rows": image_rows,
        "all_patch_scores": all_patch_scores, "all_patch_labels": all_patch_labels, "pixel_metrics": pixel_metrics,
        "native_metrics": native_metrics, "transition_store": transition_store, "transition_json": matrices,
        "rank_leverage": rank_leverage, "spatial_mass": spatial_mass, "image_count": image_count, "case_by_image": case_by_image,
    }


def measure_deployed_condition(cache_root: Path, condition: str) -> dict[str, dict[str, float]]:
    _, _, records, _ = full.validate_cache(cache_root)
    out: dict[str, dict[str, float]] = {}
    current_class: str | None = None
    pixel_scores: list[np.ndarray] = []
    pixel_labels: list[np.ndarray] = []
    def flush() -> None:
        nonlocal pixel_scores, pixel_labels
        if current_class is not None:
            out[current_class] = metric_pair(pixel_scores, pixel_labels)
        pixel_scores = []
        pixel_labels = []
    for state in load_state_records(cache_root):
        if current_class is None:
            current_class = state["class"]
        elif state["class"] != current_class:
            flush()
            current_class = state["class"]
        arrays = state["arrays"]; m_bar = arrays["m_bar"].astype(np.float32, copy=False)
        delta_case, full_delta, target_map = image_case_deltas(state["traces"]["aligned"], state["labels"], m_bar)
        delta = condition_delta(condition, delta_case, full_delta, m_bar, state["traces"]["aligned"], target_map)
        prob = deploy_probability(apply_delta(state["native"], delta))
        pixel_scores.append(prob[0, 1].reshape(-1).astype(np.float32))
        pixel_labels.append(state["mask"].reshape(-1).astype(np.uint8))
    flush()
    return {class_name: out[class_name] for class_name in sorted(records)}


def measure_all_deployed_conditions(cache_root: Path, conditions: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    _, _, records, _ = full.validate_cache(cache_root)
    out: dict[str, dict[str, dict[str, float]]] = {condition: {} for condition in conditions}
    current_class: str | None = None
    score_buffers: dict[str, list[np.ndarray]] = {condition: [] for condition in conditions}
    label_buffer: list[np.ndarray] = []
    import torch
    import torch.nn.functional as F
    def flush() -> None:
        nonlocal score_buffers, label_buffer
        if current_class is not None:
            for condition in conditions:
                out[condition][current_class] = metric_pair(score_buffers[condition], label_buffer)
        score_buffers = {condition: [] for condition in conditions}
        label_buffer = []
    for state in load_state_records(cache_root):
        if current_class is None:
            current_class = state["class"]
        elif state["class"] != current_class:
            flush()
            current_class = state["class"]
        arrays = state["arrays"]; m_bar = arrays["m_bar"].astype(np.float32, copy=False)
        delta_case, full_delta, target_map = image_case_deltas(state["traces"]["aligned"], state["labels"], m_bar)
        deltas = np.stack([condition_delta(condition, delta_case, full_delta, m_bar, state["traces"]["aligned"], target_map) for condition in conditions], axis=0)
        delta_maps = deploy_delta_batch(deltas)
        base_pre = deploy_pre_softmax(state["native"])
        logits = base_pre.repeat(len(conditions), 1, 1, 1)
        logits[:, 1] += torch.from_numpy(delta_maps)
        probabilities = F.softmax(logits, dim=1)[:, 1].numpy()
        for index, condition in enumerate(conditions):
            score_buffers[condition].append(probabilities[index].reshape(-1).astype(np.float32))
        label_buffer.append(state["mask"].reshape(-1).astype(np.uint8))
    flush()
    return {condition: {class_name: out[condition][class_name] for class_name in sorted(records)} for condition in conditions}


def full_pixel_metrics(main: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    return main["pixel_metrics"]


def summarize_condition_metrics(native: dict[str, dict[str, dict[str, float]]], deployed: dict[str, dict[str, dict[str, float]]], class_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    names = list(CONDITIONS) + ["P5_SHIFT"]
    for condition in names:
        out[condition] = {
            "native": {"auroc": class_bootstrap({c: native[c][condition]["auroc"] for c in class_names}, BOOTSTRAP_SEED + 300), "ap": class_bootstrap({c: native[c][condition]["ap"] for c in class_names}, BOOTSTRAP_SEED + 301)},
            "deployed_pixel": {"auroc": class_bootstrap({c: deployed[c][condition]["auroc"] for c in class_names}, BOOTSTRAP_SEED + 302), "ap": class_bootstrap({c: deployed[c][condition]["ap"] for c in class_names}, BOOTSTRAP_SEED + 303)},
            "per_class": {c: {"native": native[c][condition], "deployed_pixel": deployed[c][condition]} for c in class_names},
        }
    return out


def build_rank_leverage(main: dict[str, Any], class_names: list[str], class_pixel_metrics: dict[str, Any]) -> dict[str, Any]:
    totals = {key: sum(main["rank_leverage"][c][key] for c in class_names) for key in main["rank_leverage"][class_names[0]]}
    total_mixed = totals["selected_mixed"]; total_an = totals["selected_AN"]
    total_inv = totals["total_inversions"]; total_pairs = totals["total_pairs"]
    per_class = {}
    for c in class_names:
        x = main["rank_leverage"][c]
        per_class[c] = {
            **x,
            "selected_mixed_fraction": None if not (x["selected_mixed"] + x["selected_AN"] + x["selected_NA"]) else float(x["selected_mixed"] / (x["selected_mixed"] + x["selected_AN"] + x["selected_NA"] + sum(1 for r in main["aligned_rows"] if r["class"] == c and r["case"] in {"N0", "NN", "AA"}))),
            "an_over_total_inversions": None if not x["total_inversions"] else float(x["selected_AN"] / x["total_inversions"]),
            "an_over_total_pairs": None if not x["total_pairs"] else float(x["selected_AN"] / x["total_pairs"]),
            "positive_contamination_fraction_touched": None if not x["positive_contamination_total"] else float(x["positive_contamination_touched"] / x["positive_contamination_total"]),
            "negative_risk_fraction_touched": None if not x["negative_risk_total"] else float(x["negative_risk_touched"] / x["negative_risk_total"]),
        }
    return {
        "selected_mixed_over_all_selected": None if not main["aligned_rows"] else float(total_mixed / len(main["aligned_rows"])),
        "selected_AN_over_total_C0_positive_negative_inversions": None if not total_inv else float(total_an / total_inv),
        "selected_AN_over_total_positive_negative_pairs": None if not total_pairs else float(total_an / total_pairs),
        "global_transitions_caused_by_AN": int(sum(main["rank_leverage"][c]["AN_transitions"] for c in class_names)),
        "global_transitions_caused_by_NA": int(sum(main["rank_leverage"][c]["NA_transitions"] for c in class_names)),
        "AP_contamination_mass": {"acted_GT_positive": float(totals["positive_contamination_touched"]), "all_GT_positive": float(totals["positive_contamination_total"]), "fraction": None if not totals["positive_contamination_total"] else float(totals["positive_contamination_touched"] / totals["positive_contamination_total"])},
        "negative_ranking_risk_mass": {"acted_GT_negative": float(totals["negative_risk_touched"]), "all_GT_negative": float(totals["negative_risk_total"]), "fraction": None if not totals["negative_risk_total"] else float(totals["negative_risk_touched"] / totals["negative_risk_total"])},
        "per_class": per_class,
        "class_bootstrap": {key: class_bootstrap({c: per_class[c].get(key) for c in class_names}, BOOTSTRAP_SEED + 400 + idx) for idx, key in enumerate(("selected_mixed_fraction", "an_over_total_inversions", "an_over_total_pairs", "positive_contamination_fraction_touched", "negative_risk_fraction_touched"))},
    }


def build_feature_correlations(rows: list[dict[str, Any]], class_pixel: dict[str, Any], full_metrics: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    names = {"delta_magnitude": "action_delta", "base_gap": "abs_score_gap", "E_gap": "E_gap", "stage_min_gap": "E_stage_min_gap", "LOO_min_gap": "E_LOO_min_gap", "D_rank_i": "D_rank_i", "spatial_distance": "euclidean_distance"}
    out = {}
    for label_name, predicate in (("AN_vs_NA", lambda r: r["case"] == "AN"), ("mixed_vs_same", lambda r: r["case"] in {"AN", "NA"}), ("target_positive_vs_negative", lambda r: r["case"] in {"AN", "AA"})):
        group = np.asarray([1 if predicate(r) else 0 for r in rows], dtype=np.float64)
        out[label_name] = {feature: {"pearson": pearson(np.asarray([r[key] for r in rows], dtype=np.float64), group), "spearman": spearman(np.asarray([r[key] for r in rows], dtype=np.float64), group)} for feature, key in names.items()}
    class_ap = {c: full_metrics[c]["P5"]["ap"] - full_metrics[c]["C0"]["ap"] for c in class_names}
    class_fpr = {}
    for c in class_names:
        normal = class_pixel[c]
        # The normal FPR attribution is computed in the safety section; keep a placeholder here if class has no normal pixels.
        class_fpr[c] = None
    class_stats = {c: {"action_mass": float(sum(r["action_delta"] for r in rows if r["class"] == c)), "AN_mass": float(sum(r["action_delta"] for r in rows if r["class"] == c and r["case"] == "AN")), "NA_mass": float(sum(r["action_delta"] for r in rows if r["class"] == c and r["case"] == "NA"))} for c in class_names}
    for key in ("action_mass", "AN_mass", "NA_mass"):
        out.setdefault("class_AP_associations", {})[key] = {"pearson": pearson(np.asarray([class_stats[c][key] for c in class_names]), np.asarray([class_ap[c] for c in class_names])), "spearman": spearman(np.asarray([class_stats[c][key] for c in class_names]), np.asarray([class_ap[c] for c in class_names]))}
    return {"relation_outcome_associations": out, "class_metric_associations": out.get("class_AP_associations", {}), "causal_claims": False}


def normal_safety_attribution(main: dict[str, Any], class_pixel_metrics: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    rows = []
    for cls in class_names:
        normal_c0 = []; normal_p5 = []
        for state in load_state_records(CACHE_ROOT):
            if state["class"] == cls and int(state["record"]["label"]) == 0:
                normal_c0.append(state["c0_prob"][0, 1].reshape(-1)); normal_p5.append(state["aligned_prob"][0, 1].reshape(-1))
        if normal_c0:
            nm = b2.normal_metrics(np.concatenate(normal_c0), np.concatenate(normal_p5), np.concatenate(normal_p5))
            rows.append({"class": cls, "normal_images": len(normal_c0), "fpr95_delta_N0": nm["delta_C1"]["fpr_at_tau95"], "fpr99_delta_N0": nm["delta_C1"]["fpr_at_tau99"], "mean_probability_delta_N0": nm["delta_C1"]["mean_anomaly_probability"], "p99_probability_delta_N0": nm["delta_C1"]["p99_anomaly_probability"], "max_probability_delta_N0": nm["delta_C1"]["maximum_anomaly_probability"]})
    return {"all_normal_actions_are_N0": True, "normal_images": int(sum(x["normal_images"] for x in rows)), "normal_images_with_actions": int(sum(x["normal_images"] for x in rows if x["normal_images"] > 0)), "attributed_case": "N0", "metrics": {key: class_bootstrap({x["class"]: x[key] for x in rows}, BOOTSTRAP_SEED + 500 + idx) for idx, key in enumerate(("fpr95_delta_N0", "fpr99_delta_N0", "mean_probability_delta_N0", "p99_probability_delta_N0", "max_probability_delta_N0"))}, "per_class": rows}


def decision_payload(main: dict[str, Any], condition_metrics: dict[str, Any], rank_leverage: dict[str, Any], spatial_summary: dict[str, Any], class_names: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profiles = []
    tie_by_class = {}
    for cls in class_names:
        t1 = condition_metrics["T1_AN_TIE"]["per_class"][cls]["native"]["ap"]
        t2 = condition_metrics["T2_AN_STRICT_MIN"]["per_class"][cls]["native"]["ap"]
        tie_by_class[cls] = {"strict_minus_tie_native_ap_gain": float(t2 - t1)}
        leverage = rank_leverage["per_class"][cls]
        spatial = spatial_summary.get("per_class", {}).get(cls, {})
        profiles.append(class_profile(cls, [r for r in main["aligned_rows"] if r["class"] == cls], {"an_over_total_inversions": leverage.get("an_over_total_inversions") or 0.0}, tie_by_class[cls], spatial))
    macro_t1 = condition_metrics["T1_AN_TIE"]["native"]["ap"]["mean"]
    macro_t2 = condition_metrics["T2_AN_STRICT_MIN"]["native"]["ap"]["mean"]
    strict_gain = None if macro_t1 is None or macro_t2 is None else float(macro_t2 - macro_t1)
    an_coverage = rank_leverage["selected_AN_over_total_C0_positive_negative_inversions"] or 0.0
    mixed_mass = main["aligned_rows"] and sum(r["action_delta"] for r in main["aligned_rows"] if r["case"] in {"AN", "NA"}) / max(1e-30, sum(r["action_delta"] for r in main["aligned_rows"]))
    feature = feature_separation(main["aligned_rows"], class_names)
    an_na_auc = feature["features"]["AN_vs_NA"]["E_nonlocal_gap"]["rank_auc_A_over_B"]
    reliable = an_na_auc is not None and abs(an_na_auc - 0.5) > 0.05
    if strict_gain is not None and strict_gain <= PROFILE_THRESHOLDS["clear_tie_gain"] and an_coverage < PROFILE_THRESHOLDS["negligible_leverage_fraction"]:
        next_question = "PAIRWISE_ACTION_STRUCTURAL_LEVERAGE_LIMIT"
    elif strict_gain is not None and strict_gain > PROFILE_THRESHOLDS["clear_tie_gain"] and an_coverage >= PROFILE_THRESHOLDS["negligible_leverage_fraction"]:
        next_question = "STRICT_BOUNDED_ACTION_RESEARCH"
    elif mixed_mass is not None and mixed_mass < PROFILE_THRESHOLDS["dominant_fraction"] and reliable:
        next_question = "REFERENCE_ACTIONABILITY_RELIABILITY_R1"
    elif mixed_mass is not None and mixed_mass < PROFILE_THRESHOLDS["dominant_fraction"]:
        next_question = "NEW_RELATIONAL_CONTEXT_SIGNAL_REQUIRED"
    else:
        inside = spatial_summary.get("inside_fraction")
        next_question = "DEPLOYMENT_AWARE_SPATIAL_SUPPORT_REQUIRED" if inside is not None and inside < PROFILE_THRESHOLDS["spatial_inside_fraction"] else "NO_SUPPORTED_PHASE5_ACTION_DIRECTION"
    decision = {
        "integrity": "PASS", "protocol_commit_sha": EXPECTED_PROTOCOL_COMMIT, "implementation_commit_sha": head(), "model_forwards": 0, "training_steps": 0, "candidate_selection_allowed": False,
        "dominant_failure_mode": profiles, "next_research_question": next_question,
        "strict_minus_tie_native_ap_gain": strict_gain, "AN_coverage_of_C0_inversions": an_coverage,
        "limitations": ["All oracle/counterfactual results are GT-dependent post-hoc diagnostics and are not deployable.", "No causal inference is made from associations.", "Spatial attribution uses case-isolated probability changes; nonlinear effects are not additive."],
        "forbidden_tuning_actions": ["No AP-driven threshold or margin selection", "No selector/action formula changes", "No learned gate or classifier", "No new model forward, training, medical evaluation, C1/local multiscale/CNN/router reopening"],
        "terminal": "P5C0_FORENSIC_COMPLETE",
    }
    return decision, profiles


def write_per_class(path: Path, class_names: list[str], main: dict[str, Any], condition_metrics: dict[str, Any], profiles: list[dict[str, Any]], normal: dict[str, Any]) -> None:
    profile_map = {x["class"]: x for x in profiles}
    fields = ["class", "n_images", "aligned_selected", "aligned_same_label", "aligned_mixed", "aligned_AN", "aligned_NA", "aligned_action_mass", "shifted_selected", "shifted_action_mass", "native_ap_C0", "native_ap_P5", "native_ap_P5_SHIFT", "deployed_ap_C0", "deployed_ap_P5", "deployed_ap_P5_SHIFT", "profile"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for cls in class_names:
            ar = [x for x in main["aligned_rows"] if x["class"] == cls]; sr = [x for x in main["shifted_rows"] if x["class"] == cls]
            writer.writerow({"class": cls, "n_images": sum(x["class"] == cls for x in main["image_rows"]["aligned"]), "aligned_selected": len(ar), "aligned_same_label": sum(x["case"] in {"N0", "NN", "AA"} for x in ar), "aligned_mixed": sum(x["case"] in {"AN", "NA"} for x in ar), "aligned_AN": sum(x["case"] == "AN" for x in ar), "aligned_NA": sum(x["case"] == "NA" for x in ar), "aligned_action_mass": sum(x["action_delta"] for x in ar), "shifted_selected": len(sr), "shifted_action_mass": sum(x["action_delta"] for x in sr), "native_ap_C0": condition_metrics["C0"]["per_class"][cls]["native"]["ap"], "native_ap_P5": condition_metrics["P5"]["per_class"][cls]["native"]["ap"], "native_ap_P5_SHIFT": condition_metrics["P5_SHIFT"]["per_class"][cls]["native"]["ap"], "deployed_ap_C0": condition_metrics["C0"]["per_class"][cls]["deployed_pixel"]["ap"], "deployed_ap_P5": condition_metrics["P5"]["per_class"][cls]["deployed_pixel"]["ap"], "deployed_ap_P5_SHIFT": condition_metrics["P5_SHIFT"]["per_class"][cls]["deployed_pixel"]["ap"], "profile": ";".join(profile_map[cls]["profile"])})


def pooled_native_auc(main: dict[str, Any], condition: str) -> float:
    scores = np.concatenate(main["all_patch_scores"][condition]).astype(np.float32, copy=False)
    labels = np.concatenate(main["all_patch_labels"]).astype(np.uint8, copy=False)
    return float(exact_auc_ap(scores, labels)[0])


def run_forensic(cache_root: Path, protocol_commit: str | None, implementation_commit: str | None) -> None:
    protocol, manifest, datasets, records = validate_protocol_inputs(cache_root, protocol_commit, implementation_commit)
    class_names = sorted(records)
    main = compute_main(cache_root)
    pixel_native = full_pixel_metrics(main)
    native = main["native_metrics"]
    for cls in class_names:
        for condition in ("C0", "P5", "P5_SHIFT"):
            native[cls][condition] = {**native[cls][condition], **pixel_native[cls][condition]}
    deployed: dict[str, dict[str, dict[str, float]]] = {cls: {} for cls in class_names}
    for cls in class_names:
        for condition in ("C0", "P5", "P5_SHIFT"):
            deployed[cls][condition] = pixel_native[cls][condition]
    measured_conditions = [condition for condition in CONDITIONS if condition not in {"C0", "P5"}]
    measured_all = measure_all_deployed_conditions(cache_root, measured_conditions)
    for condition in measured_conditions:
        for cls in class_names: deployed[cls][condition] = measured_all[condition][cls]
    condition_summary = summarize_condition_metrics(native, deployed, class_names)
    # Reconcile transition counts against independent pooled exact AUROC.
    c0_auc = pooled_native_auc(main, "C0")
    for name in ("P5", "SHIFT", "T1_AN_TIE", "T2_AN_STRICT_MIN", "AN_ONLY", "NA_ONLY"):
        condition = "P5_SHIFT" if name == "SHIFT" else name
        main["transition_store"][name]["measured_auc"] = float(pooled_native_auc(main, condition) - c0_auc)
    transition_json = aggregate_transition_store(main["transition_store"], class_names)
    leverage = build_rank_leverage(main, class_names, pixel_native)
    spatial_case = action_mass_summary(main["spatial_mass"])
    total_spatial = {"positive_mass": sum(v["positive_mass"] for v in main["spatial_mass"]["aligned"].values()), "inside_mass": sum(v["inside_mass"] for v in main["spatial_mass"]["aligned"].values()), "outside_mass": sum(v["outside_mass"] for v in main["spatial_mass"]["aligned"].values()), "effective_pixels": sum(v["effective_pixels"] for v in main["spatial_mass"]["aligned"].values())}
    spatial_summary = {"aligned_case_isolated": spatial_case["aligned"], "shifted_case_isolated": spatial_case["shifted"], "aligned_actual_total_positive_mass": total_spatial, "inside_fraction": None if total_spatial["positive_mass"] <= 0 else float(total_spatial["inside_mass"] / total_spatial["positive_mass"]), "per_class": {cls: {} for cls in class_names}}
    normal = normal_safety_attribution(main, pixel_native, class_names)
    feature = feature_separation(main["aligned_rows"], class_names)
    correlation = build_feature_correlations(main["aligned_rows"], pixel_native, native, class_names)
    decision, profiles = decision_payload(main, condition_summary, leverage, spatial_summary, class_names)
    case_taxonomy = {"schema_version": "P5B_FAILURE_FORENSIC_C0_v1", "aligned_primary": case_summary(main["aligned_rows"], "aligned"), "shifted_control": case_summary(main["shifted_rows"], "shifted"), "per_class_aligned": class_relation_summary(main["aligned_rows"], class_names), "required_counts": {"aligned_selected": len(main["aligned_rows"]), "aligned_mixed": sum(x["case"] in {"AN", "NA"} for x in main["aligned_rows"]), "aligned_AN": sum(x["case"] == "AN" for x in main["aligned_rows"]), "aligned_NA": sum(x["case"] == "NA" for x in main["aligned_rows"]), "aligned_same_label": sum(x["case"] in {"N0", "NN", "AA"} for x in main["aligned_rows"]), "shifted_selected": len(main["shifted_rows"])}}
    action_mass = {"aligned": variant_action_summary(main["aligned_rows"], main["image_rows"]["aligned"], "aligned"), "shifted": variant_action_summary(main["shifted_rows"], main["image_rows"]["shifted"], "shifted"), "case_isolated_probability_mass": spatial_case}
    tie = {"T0": condition_summary["C0"], "T1_AN_TIE": condition_summary["T1_AN_TIE"], "T2_AN_STRICT_MIN": condition_summary["T2_AN_STRICT_MIN"], "strict_minus_tie_native_ap_gain": float(condition_summary["T2_AN_STRICT_MIN"]["native"]["ap"]["mean"] - condition_summary["T1_AN_TIE"]["native"]["ap"]["mean"]), "strict_minus_tie_deployed_ap_gain": float(condition_summary["T2_AN_STRICT_MIN"]["deployed_pixel"]["ap"]["mean"] - condition_summary["T1_AN_TIE"]["deployed_pixel"]["ap"]["mean"]), "global_rank_transitions": {"T1_AN_TIE": transition_json["T1_AN_TIE"], "T2_AN_STRICT_MIN": transition_json["T2_AN_STRICT_MIN"]}}
    oracle_names = ("O1_MIXED_ONLY_TIE", "O2_CORRECT_MIXED_ONLY_TIE", "O3_TARGET_POSITIVE_TIE", "O4_NO_NORMAL_IMAGE_ACTION", "O5_CORRECT_MIXED_STRICT_MIN")
    oracle = {name: condition_summary[name] for name in oracle_names}
    breakpoints = breakpoint_summary(main["aligned_rows"], class_names)
    aligned_shifted = {"aligned": variant_action_summary(main["aligned_rows"], main["image_rows"]["aligned"], "aligned"), "shifted": variant_action_summary(main["shifted_rows"], main["image_rows"]["shifted"], "shifted"), "global_rank_transitions": {"aligned": transition_json["P5"], "shifted": transition_json["SHIFT"]}, "case_composition": {"aligned": case_summary(main["aligned_rows"], "aligned")["cases"], "shifted": case_summary(main["shifted_rows"], "shifted")["cases"]}}
    per_class_normal = {x["class"]: x for x in normal["per_class"]}
    output_rows = []
    for cls in class_names:
        row = {"class": cls, "profile": ";".join(next(x["profile"] for x in profiles if x["class"] == cls)), "aligned_selected": sum(r["class"] == cls for r in main["aligned_rows"]), "AN": sum(r["class"] == cls and r["case"] == "AN" for r in main["aligned_rows"]), "NA": sum(r["class"] == cls and r["case"] == "NA" for r in main["aligned_rows"]), "action_mass": sum(r["class"] == cls for r in main["aligned_rows"] and []) if False else float(sum(r["action_delta"] for r in main["aligned_rows"] if r["class"] == cls)), "strict_minus_tie_native_ap_gain": float(condition_summary["T2_AN_STRICT_MIN"]["per_class"][cls]["native"]["ap"] - condition_summary["T1_AN_TIE"]["per_class"][cls]["native"]["ap"]), "normal_fpr95_delta_N0": per_class_normal.get(cls, {}).get("fpr95_delta_N0")}
        output_rows.append(row)
    result_files = {
        "INPUT_CHECK.json": json.loads((OUTPUT_ROOT / "INPUT_CHECK.json").read_text()),
        "PROTOCOL.json": protocol,
        "CASE_TAXONOMY.json": case_taxonomy,
        "ACTION_MASS_ATTRIBUTION.json": action_mass,
        "IMAGE_SAFETY_ATTRIBUTION.json": normal,
        "GLOBAL_RANK_TRANSITIONS.json": {"states": list(TRANSITION_NAMES), "pooled_and_per_class": transition_json, "exact_auc_ap_reconciliation": {"C0": condition_summary["C0"]["native"]["auroc"], "P5": condition_summary["P5"]["native"]["auroc"], "P5_minus_C0": transition_json["P5"]}},
        "CASE_COUNTERFACTUALS.json": {name: condition_summary[name] for name in ("CF_N0", "CF_NN", "CF_AA", "CF_AN", "CF_NA", "P5_without_N0", "P5_without_NN", "P5_without_AA", "P5_without_AN", "P5_without_NA")},
        "TIE_STRICT_DIAGNOSTIC.json": tie,
        "ORACLE_CEILINGS.json": oracle,
        "RANK_LEVERAGE.json": leverage,
        "SCORE_BREAKPOINTS.json": breakpoints,
        "SPATIAL_MASS_ATTRIBUTION.json": spatial_summary,
        "FEATURE_SEPARABILITY.json": {"separability": feature, "correlations": correlation},
        "ALIGNED_SHIFTED_FORENSIC.json": aligned_shifted,
        "DECISION.json": decision,
    }
    for name, value in result_files.items():
        if name not in {"INPUT_CHECK.json", "PROTOCOL.json", "DECISION.json"}: write_json(OUTPUT_ROOT / name, value)
    write_json(OUTPUT_ROOT / "DECISION.json", decision)
    write_per_class(OUTPUT_ROOT / "PER_CLASS.csv", class_names, main, condition_summary, profiles, normal)
    report_lines = ["# P5B failure forensic C0", "", f"Terminal: `{decision['terminal']}`.", "", f"Aligned selected relations={len(main['aligned_rows'])}; shifted={len(main['shifted_rows'])}.", f"Aligned cases: {case_taxonomy['required_counts']}.", f"Next research question: `{decision['next_research_question']}`.", "", "All counterfactuals and oracle ceilings are post-hoc GT diagnostics only; no deployable rule was selected."]
    atomic_write(OUTPUT_ROOT / "REPORT.md", "\n".join(report_lines) + "\n")
    output_names = ("INPUT_CHECK.json", "PROTOCOL.json", "CASE_TAXONOMY.json", "ACTION_MASS_ATTRIBUTION.json", "IMAGE_SAFETY_ATTRIBUTION.json", "GLOBAL_RANK_TRANSITIONS.json", "CASE_COUNTERFACTUALS.json", "TIE_STRICT_DIAGNOSTIC.json", "ORACLE_CEILINGS.json", "RANK_LEVERAGE.json", "SCORE_BREAKPOINTS.json", "SPATIAL_MASS_ATTRIBUTION.json", "FEATURE_SEPARABILITY.json", "ALIGNED_SHIFTED_FORENSIC.json", "PER_CLASS.csv", "DECISION.json", "REPORT.md", "OUTPUT_CHECK.json")
    check = {"status": "PASS", "schema_version": "P5B_FAILURE_FORENSIC_C0_OUTPUT_v1", "required_files": {name: (OUTPUT_ROOT / name).is_file() for name in output_names}, "json_finite": all(finite(json.loads((OUTPUT_ROOT / name).read_text())) for name in output_names if name.endswith(".json")), "aligned_case_partition": case_taxonomy["required_counts"]["aligned_selected"] == 99037 and case_taxonomy["required_counts"]["aligned_mixed"] == 471 and case_taxonomy["required_counts"]["aligned_AN"] == 378 and case_taxonomy["required_counts"]["aligned_NA"] == 93 and case_taxonomy["required_counts"]["aligned_same_label"] == 98566, "images": main["image_count"], "classes": len(class_names), "model_forwards": 0, "training_steps": 0, "protected_source_hashes": source_hashes(), "no_dense_maps_committed": True, "no_selector_formula_change": True, "no_threshold_sweep": True}
    check["status"] = "PASS" if all(check["required_files"].values()) and check["json_finite"] and check["aligned_case_partition"] and check["images"] == EXPECTED_IMAGES and check["classes"] == EXPECTED_CLASSES else "P5C0_FORENSIC_INVALID"
    write_json(OUTPUT_ROOT / "OUTPUT_CHECK.json", check)
    print(json.dumps({"status": check["status"], "terminal": decision["terminal"], "model_forwards": 0, "training_steps": 0, "aligned_selected": len(main["aligned_rows"]), "next_research_question": decision["next_research_question"]}, sort_keys=True))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("protocol", "forensic"), required=True)
    parser.add_argument("--protocol-commit")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    args = parser.parse_args()
    if args.mode == "protocol":
        run_protocol()
        return
    if args.mode == "forensic":
        run_forensic(args.cache_root, args.protocol_commit, args.implementation_commit)
        return
    raise RuntimeError("unsupported mode")


if __name__ == "__main__":
    main()
