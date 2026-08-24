"""P25R exact patch-actionability identifiability study.

This module is deliberately separate from the frozen P14--P23 runners.  It
uses the immutable VisA source cache, R0 sparse deployment basis and source
labels only after the GT-free panel membership has been frozen.  No CLIP or
Phase2B optimisation is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.sabra_car.r0_direction import MARGIN_SCALE, deploy_correction, exact_metrics, load_masks, metadata_and_root
from tools.sabra_cure import context_value_risk_recovery as p15
from tools.sabra_cure import r1, r2, r2v2_harm


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sabra_cure/patch_actionability_r1"
DOC = ROOT / "research/sabra_cure/patch_actionability_r1"
BRANCH = "research/p25r-sabra-cure-patch-actionability-v1"
PARENT = "c0dd9ee86806346276f07bad8a7d1ea56327590d"
ALPHA = 0.25
PATCHES = r1.PATCHES
TARGET_PATCHES_PER_CLASS = 2000
CAP_PER_IMAGE = 16
STRATA = 5
STRATUM_QUOTA = TARGET_PATCHES_PER_CLASS // (STRATA * STRATA)
FEATURE_ORDER = (*r2v2_harm.HARM_ORDER,
                 "harm_risk", "harm_policy_action",
                 "support_native_rank_median", "support_native_rank_q90",
                 "signed_delta_mean_over_image_iqr", "abs_delta_q90_over_image_iqr",
                 "support_rank_shift_median", "support_rank_shift_abs_q90",
                 "top5_boundary_cross_fraction", "top20_boundary_cross_fraction")
PAIR_CAP = 8192
PAIR_L2 = 1.0
Q1_GATES = {"support_classes": 10, "median_spearman": .20, "positive_spearman": 9,
            "macro_sign_auc": .65, "macro_bc20": .35, "bc20_classes": 9}


def git(*args: str) -> str:
    return r1.git(*args)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def finite(label: str, *items: np.ndarray) -> None:
    if not all(np.isfinite(np.asarray(item)).all() for item in items):
        raise RuntimeError(f"P25R_ENGINEERING_STOP non-finite {label}")


def _rank_bin(values: np.ndarray) -> np.ndarray:
    """Frozen linear-quantile, right-closed quintile assignment."""
    original = np.asarray(values, dtype=np.float64)
    flat = original.reshape(-1)
    cuts = np.quantile(flat, [.2, .4, .6, .8], method="linear")
    return np.minimum(np.searchsorted(cuts, flat, side="right"), 4).astype(np.int8).reshape(original.shape)


def _selection_key(class_name: str, image_path: str, patch: int) -> bytes:
    return hashlib.sha256(class_name.encode() + b"\0" + image_path.encode() + b"\0" + str(int(patch)).encode()).digest()


def panel_membership(class_name: str) -> dict[str, np.ndarray]:
    """Build the pre-target GT-free 25-stratum panel for one source class."""
    source_path = r1.SOURCE_ROOT / "gt_free_cache" / f"{class_name}.npz"
    with np.load(source_path, allow_pickle=False) as data:
        paths = data["image_path"].astype(str)
        rank = np.asarray(data["margin_within_image_rank"], dtype=np.float64)
        sensitivity = np.asarray(data["deployment_sensitivity"], dtype=np.float64)
    if rank.shape != sensitivity.shape or rank.shape[1] != PATCHES:
        raise RuntimeError("P25R_ENGINEERING_STOP panel cache shape")
    finite("panel inputs", rank, sensitivity)
    rows, patches = np.indices(rank.shape, dtype=np.int64)
    rb, sb = _rank_bin(rank), _rank_bin(sensitivity)
    chosen: list[tuple[int, int, int, int]] = []
    global_per_image: dict[int, int] = {}
    for rbin in range(STRATA):
        for sbin in range(STRATA):
            mask = (rb == rbin) & (sb == sbin)
            candidates = [( _selection_key(class_name, str(paths[i]), int(j)), int(i), int(j))
                          for i, j in zip(rows[mask], patches[mask])]
            candidates.sort(key=lambda item: (item[0], str(paths[item[1]]), item[2]))
            selected: list[tuple[int, int]] = []
            for _, image, patch in candidates:
                if global_per_image.get(image, 0) >= CAP_PER_IMAGE:
                    continue
                selected.append((image, patch)); global_per_image[image] = global_per_image.get(image, 0) + 1
                if len(selected) == STRATUM_QUOTA:
                    break
            if len(selected) != STRATUM_QUOTA:
                raise RuntimeError(f"P25R_TARGET_PANEL_NO_GO {class_name} stratum={rbin},{sbin} selected={len(selected)}")
            chosen.extend((image, patch, rbin, sbin) for image, patch in selected)
    chosen.sort(key=lambda item: (str(paths[item[0]]), item[1]))
    image = np.asarray([item[0] for item in chosen], dtype=np.int32)
    patch = np.asarray([item[1] for item in chosen], dtype=np.int32)
    rank_stratum = np.asarray([item[2] for item in chosen], dtype=np.int8)
    sensitivity_stratum = np.asarray([item[3] for item in chosen], dtype=np.int8)
    counts = np.bincount(image, minlength=len(paths))
    if len(image) != TARGET_PATCHES_PER_CLASS or int(counts.max(initial=0)) > CAP_PER_IMAGE:
        raise RuntimeError("P25R_TARGET_PANEL_NO_GO panel quota/cap")
    return {"image_path": paths[image], "image_index": image, "patch_index": patch,
            "rank_stratum": rank_stratum, "sensitivity_stratum": sensitivity_stratum}


def panel_digest(panel: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in ("image_path", "image_index", "patch_index", "rank_stratum", "sensitivity_stratum"):
        values = np.asarray(panel[key])
        h.update(key.encode() + b"\0")
        h.update(values.astype("U").tobytes() if values.dtype.kind in "US" else values.tobytes())
    return h.hexdigest()


def build_all_panels(output: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"classes": {}, "target_patches_per_class": TARGET_PATCHES_PER_CLASS,
                                "cap_per_image": CAP_PER_IMAGE, "strata": STRATA,
                                "stratum_quota": STRATUM_QUOTA, "uses_gt": False}
    for name in r1.CLASSES:
        panel = panel_membership(name)
        atomic_npz(output / "panels" / f"{name}.npz", **panel)
        payload["classes"][name] = {"count": int(len(panel["patch_index"])), "digest": panel_digest(panel),
                                    "max_selected_per_image": int(np.bincount(panel["image_index"]).max()),
                                    "stratum_counts": np.bincount(panel["rank_stratum"] * STRATA + panel["sensitivity_stratum"], minlength=25).tolist()}
    payload["panel_digest"] = hashlib.sha256(json.dumps(payload["classes"], sort_keys=True).encode()).hexdigest()
    payload["status"] = "PASS"
    atomic_json(output / "panel_membership.json", payload)
    return payload


@dataclass
class Basis:
    indices: np.ndarray
    values: np.ndarray
    valid: np.ndarray

    def support(self, patch: int) -> tuple[np.ndarray, np.ndarray]:
        mask = self.valid[int(patch)]
        return self.indices[int(patch), mask], self.values[int(patch), mask]


def load_basis() -> Basis:
    path = ROOT / "results/sabra_car/r0/sparse_deployment_basis.npz"
    with np.load(path, allow_pickle=False) as data:
        basis = Basis(np.asarray(data["indices"], dtype=np.int32), np.asarray(data["values"], dtype=np.float32), np.asarray(data["valid"], dtype=bool))
    if basis.indices.shape[0] != PATCHES or basis.values.shape != basis.indices.shape or basis.valid.shape != basis.indices.shape:
        raise RuntimeError("P25R_ENGINEERING_STOP sparse basis")
    return basis


def _deployed_margin(native_logits: np.ndarray, device: torch.device) -> np.ndarray:
    native = torch.from_numpy(np.asarray(native_logits, dtype=np.float32)[None]).permute(1, 0, 2, 3).to(device)
    with torch.no_grad():
        _, logits = deploy_correction(native, torch.zeros((1, PATCHES), device=device))
    return (logits[0, 1] - logits[0, 0]).reshape(-1).cpu().numpy().astype(np.float32)


def candidate_support_scores(margin: np.ndarray, patch: int, sign: int, basis: Basis) -> tuple[np.ndarray, np.ndarray]:
    index, values = basis.support(patch)
    shifted = np.asarray(margin, dtype=np.float32)[index] + np.float32(sign * ALPHA * MARGIN_SCALE) * values
    probability = (1.0 / (1.0 + np.exp(-shifted.astype(np.float64)))).astype(np.float32)
    return index, probability


def _candidate_delta(base_image: np.ndarray, labels_image: np.ndarray, margin: np.ndarray, patch: int, sign: int, basis: Basis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index, candidate = candidate_support_scores(margin, patch, sign, basis)
    return p15.delta_groups(np.asarray(base_image, dtype=np.float32).reshape(-1)[index], candidate, np.asarray(labels_image, dtype=np.uint8).reshape(-1)[index])


def target_class(class_name: str, panel: dict[str, np.ndarray], output: Path, basis: Basis, direct_check: int = 128) -> dict[str, Any]:
    """Generate the exact source-only V_j target after membership is frozen."""
    source_path = r1.SOURCE_ROOT / "gt_free_cache" / f"{class_name}.npz"
    utility_path = r1.UTILITY_ROOT / f"{class_name}.npz"
    with np.load(source_path, allow_pickle=False) as source, np.load(utility_path, allow_pickle=False) as utility_data:
        logits = np.asarray(source["native_logits"], dtype=np.float32)
        scores = np.asarray(source["native_pixel_probability"], dtype=np.float32)
        paths = source["image_path"].astype(str)
        utility = np.asarray(utility_data["utility"], dtype=np.float64)
        if not np.array_equal(paths, utility_data["image_path"].astype(str)):
            raise RuntimeError("P25R_ENGINEERING_STOP target path alignment")
    metadata, data_root = metadata_and_root(r2.DATA_ROOT)
    masks = load_masks(paths, metadata, data_root)
    base_s, base_p, base_t = p15.score_groups(scores, masks)
    base_ap = p15.ap_from_groups(base_p, base_t)
    values = np.empty(len(panel["patch_index"]), dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices_by_image: dict[int, np.ndarray] = {}
    for image in np.unique(panel["image_index"]):
        indices_by_image[int(image)] = np.flatnonzero(panel["image_index"] == image)
    started = time.perf_counter(); parity_error = 0.0
    for image, rows in indices_by_image.items():
        margin = _deployed_margin(logits[image], device)
        for row in rows:
            patch = int(panel["patch_index"][row]); sign = int(np.sign(utility[image, patch]))
            if sign == 0:
                values[row] = 0.0; continue
            cs, cp, ct = _candidate_delta(scores[image], masks[image], margin, patch, sign, basis)
            values[row] = p15.ap_with_delta(base_s, base_p, base_t, cs, cp, ct) - base_ap
            if row < direct_check:
                correction = torch.zeros((1, PATCHES), device=device); correction[0, patch] = float(sign * ALPHA * MARGIN_SCALE)
                native = torch.from_numpy(logits[image:image + 1]).permute(1, 0, 2, 3).to(device)
                with torch.no_grad():
                    probability, _ = deploy_correction(native, correction)
                direct = scores.copy(); direct[image] = probability[0, 1].cpu().numpy().astype(np.float32)
                parity_error = max(parity_error, abs(float(exact_metrics(direct.reshape(-1), masks.reshape(-1))["pAP"]) - float(values[row] + base_ap)))
    finite("targets", values)
    target_path = output / "targets" / f"{class_name}.npz"
    atomic_npz(target_path, image_path=panel["image_path"], image_index=panel["image_index"], patch_index=panel["patch_index"],
               rank_stratum=panel["rank_stratum"], sensitivity_stratum=panel["sensitivity_stratum"], V=values)
    row = {"class": class_name, "count": int(len(values)), "base_pap": float(base_ap), "elapsed_seconds": time.perf_counter() - started,
           "direct_fast_max_abs_error": float(parity_error), "direct_checks": min(direct_check, len(values)), "target_sha256": sha256(target_path)}
    if parity_error > 1e-12:
        raise RuntimeError(f"P25R_ENGINEERING_STOP target direct parity {class_name}: {parity_error}")
    return row


def _rank_fraction(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    rank = np.empty(len(flat), dtype=np.float64); rank[order] = np.arange(len(flat), dtype=np.float64)
    return rank / max(1, len(flat) - 1)


def impact_features(native_score: np.ndarray, margin: np.ndarray, patch: int, action: int, basis: Basis) -> np.ndarray:
    """The frozen F25--F32 GT-free score-impact construction for one action."""
    index, candidate = candidate_support_scores(margin, patch, action, basis)
    base = np.asarray(native_score, dtype=np.float32).reshape(-1)
    support = base[index].astype(np.float64)
    delta = candidate.astype(np.float64) - support
    iq = max(float(np.subtract(*np.quantile(base, [.75, .25], method="linear"))), 1e-8)
    before_rank = _rank_fraction(base)[index]
    altered = base.copy(); altered[index] = candidate
    after_rank = _rank_fraction(altered)[index]
    q95 = float(np.quantile(base, .95, method="linear")); q80 = float(np.quantile(base, .80, method="linear"))
    return np.asarray([np.median(before_rank), np.quantile(before_rank, .90, method="linear"),
                       delta.mean() / iq, np.quantile(np.abs(delta), .90, method="linear") / iq,
                       np.median(after_rank - before_rank), np.quantile(np.abs(after_rank - before_rank), .90, method="linear"),
                       np.mean((support <= q95) & (candidate > q95)) + np.mean((support > q95) & (candidate <= q95)),
                       np.mean((support <= q80) & (candidate > q80)) + np.mean((support > q80) & (candidate <= q80))], dtype=np.float64)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or x.std() == 0.0 or y.std() == 0.0:
        return None
    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort"); out = np.empty(len(v), dtype=np.float64); out[order] = np.arange(len(v), dtype=np.float64)
        starts = np.r_[0, np.flatnonzero(v[order][1:] != v[order][:-1]) + 1]; stops = np.r_[starts[1:], len(v)]
        for a, b in zip(starts, stops): out[order[a:b]] = (a + b - 1) / 2.0
        return out
    return r1.pearson(rank(x), rank(y))


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64); labels = np.asarray(labels, dtype=bool)
    p, n = int(labels.sum()), int((~labels).sum())
    if p == 0 or n == 0: return None
    order = np.argsort(scores, kind="mergesort"); ranks = np.empty(len(scores), dtype=np.float64); ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sorted_scores = scores[order]; starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]; stops = np.r_[starts[1:], len(scores)]
    for a, b in zip(starts, stops): ranks[order[a:b]] = (a + b + 1) / 2.0
    return float((ranks[labels].sum() - p * (p + 1) / 2.0) / (p * n))


def _panel_rows(panel: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(panel["image_index"], dtype=np.int64) * PATCHES + np.asarray(panel["patch_index"], dtype=np.int64)


def _load_panel(output: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(output / "panels" / f"{name}.npz", allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _load_target(output: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(output / "targets" / f"{name}.npz", allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _impact_rows(name: str, image_index: np.ndarray, patch_index: np.ndarray, actions: np.ndarray, basis: Basis) -> np.ndarray:
    """Materialise F25--F32 from only frozen detector/action evidence."""
    source_path = r1.SOURCE_ROOT / "gt_free_cache" / f"{name}.npz"
    with np.load(source_path, allow_pickle=False) as data:
        logits = np.asarray(data["native_logits"], dtype=np.float32)
        score = np.asarray(data["native_pixel_probability"], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = np.zeros((len(patch_index), 8), dtype=np.float64)
    for image in np.unique(image_index):
        rows = np.flatnonzero(image_index == image)
        margin = _deployed_margin(logits[int(image)], device)
        for row in rows:
            action = int(actions[row])
            if action != 0:
                result[row] = impact_features(score[int(image)], margin, int(patch_index[row]), action, basis)
            else:
                # KEEP is a valid action: zero score displacement while retaining the support's native ranks.
                support, _ = basis.support(int(patch_index[row]))
                ranks = _rank_fraction(score[int(image)]) [support]
                result[row, :2] = (np.median(ranks), np.quantile(ranks, .90, method="linear"))
    finite("impact features", result)
    return result


def _feature_rows_for_outer(held: str, name: str, outer: dict[str, Any], shards: dict[str, r1.Shard], output: Path, basis: Basis) -> tuple[np.ndarray, np.ndarray]:
    """Return P25R rows and their frozen R2-v2 actions, without reading V."""
    panel = _load_panel(output, name)
    rows = _panel_rows(panel)
    if name == held:
        x = r2v2_harm.harm_features(shards[held].x, outer["mu"], outer["sigma"])
        risk = np.asarray(outer["risk_h"], dtype=np.float64)
        action = r2v2_harm.action(outer["mu"], risk, float(outer["tau_harm"]))
    else:
        group = next(item for item in outer["level1"] if item["name"] == name)
        x = np.asarray(group["f"], dtype=np.float64)
        risk = np.asarray(group["r_h"], dtype=np.float64)
        action = r2v2_harm.action(group["mu"], risk, float(outer["tau_harm"]))
    actions = action[rows].astype(np.int8)
    f24 = actions.astype(np.float64)
    impact = _impact_rows(name, panel["image_index"], panel["patch_index"], actions, basis)
    features = np.column_stack((x[rows], risk[rows], f24, impact)).astype(np.float64)
    if features.shape != (len(rows), len(FEATURE_ORDER)):
        raise RuntimeError("P25R_ENGINEERING_STOP 32D feature shape")
    finite("32D feature rows", features)
    return features, actions


def _feature_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    q25, median, q75 = np.quantile(x, [.25, .5, .75], axis=0, method="linear")
    iqr = np.maximum(q75 - q25, 1e-6)
    finite("rank scaler", median, iqr)
    return median, iqr


def _deterministic_pairs(name: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within-class, non-adjacent-decile, balanced deterministic pair set."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    cuts = np.quantile(values, np.arange(1, 10) / 10.0, method="linear")
    bins = np.searchsorted(cuts, values, side="right")
    blocks = [np.flatnonzero(bins == item) for item in range(10)]
    families = [(lo, hi) for lo in range(10) for hi in range(lo + 2, 10) if len(blocks[lo]) and len(blocks[hi])]
    if not families:
        raise RuntimeError("P25R_ENGINEERING_STOP no non-adjacent decile pairs")
    quota = max(1, PAIR_CAP // len(families))
    left: list[int] = []; right: list[int] = []
    for lo, hi in families:
        a = sorted(blocks[lo].tolist(), key=lambda i: hashlib.sha256(f"{name}:L:{lo}:{i}".encode()).digest())
        b = sorted(blocks[hi].tolist(), key=lambda i: hashlib.sha256(f"{name}:H:{hi}:{i}".encode()).digest())
        for index in range(min(quota, len(a) * len(b))):
            low, high = a[index % len(a)], b[(index // len(a)) % len(b)]
            left.append(high); right.append(low)
    order = sorted(range(len(left)), key=lambda i: hashlib.sha256(f"{name}:P:{left[i]}:{right[i]}".encode()).digest())[:PAIR_CAP]
    i = np.asarray([left[item] for item in order], dtype=np.int64)
    j = np.asarray([right[item] for item in order], dtype=np.int64)
    weights = np.maximum(values[i] - values[j], 1e-12)
    return i, j, weights


def fit_ranker(x: np.ndarray, groups: list[tuple[str, np.ndarray]]) -> dict[str, Any]:
    """Frozen zero-init float64 linear pairwise logistic ranker (CPU L-BFGS)."""
    x = np.asarray(x, dtype=np.float64)
    median, iqr = _feature_scaler(x)
    standardized = (x - median) / iqr
    dx: list[np.ndarray] = []; weight: list[np.ndarray] = []
    offset = 0
    provenance: list[dict[str, Any]] = []
    for name, values in groups:
        count = len(values); left, right, pair_weight = _deterministic_pairs(name, values)
        dx.append(standardized[offset + left] - standardized[offset + right]); weight.append(pair_weight)
        provenance.append({"class": name, "pairs": int(len(left)), "decile_rule": "same/adjacent skipped", "cap": PAIR_CAP})
        offset += count
    if offset != len(x):
        raise RuntimeError("P25R_ENGINEERING_STOP ranker group alignment")
    difference = np.concatenate(dx).astype(np.float64); weights = np.concatenate(weight).astype(np.float64)
    # torch L-BFGS is deterministic on CPU for this pure float64 linear objective.
    torch.set_num_threads(1)
    design = torch.from_numpy(difference); w = torch.from_numpy(weights / weights.mean())
    parameter = torch.zeros((x.shape[1],), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([parameter], lr=1.0, max_iter=100, tolerance_grad=1e-12,
                                  tolerance_change=1e-14, line_search_fn="strong_wolfe")
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = (w * torch.nn.functional.softplus(-(design @ parameter))).mean() + .5 * PAIR_L2 * parameter.square().sum()
        loss.backward(); return loss
    loss = float(optimizer.step(closure).detach().item())
    beta = parameter.detach().numpy().copy(); finite("rank beta", beta)
    return {"median": median, "iqr": iqr, "beta": beta, "pairs": provenance, "loss": loss,
            "pair_count": int(len(difference)), "feature_order": list(FEATURE_ORDER)}


def rank_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    result = ((np.asarray(x, dtype=np.float64) - np.asarray(model["median"])) / np.asarray(model["iqr"])) @ np.asarray(model["beta"])
    finite("rank prediction", result)
    return result


def q1_metrics(v: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    v = np.asarray(v, dtype=np.float64); score = np.asarray(score, dtype=np.float64)
    support = bool(np.any(v > 0.0) and np.any(v < 0.0))
    ordered = np.argsort(-score, kind="mergesort")
    top = ordered[:max(1, int(np.ceil(.20 * len(score))))]
    positive_total = float(np.maximum(v, 0.0).sum())
    bc20 = None if positive_total <= 0.0 else float(np.maximum(v[top], 0.0).sum() / positive_total)
    return {"support": support, "spearman": spearman(score, v), "sign_auc": auc(score, v > 0.0), "bc20": bc20,
            "top20_count": int(len(top)), "positive_benefit_total": positive_total}


def q1_fold(held: str, shards: dict[str, r1.Shard], output: Path, basis: Basis) -> dict[str, Any]:
    """Strict outer LOCO Q1: held V opens only after model is frozen."""
    outer = r2v2_harm.outer(held, shards)
    names = [name for name in r1.CLASSES if name != held]
    source_x: list[np.ndarray] = []; source_v: list[np.ndarray] = []
    for name in names:
        features, _ = _feature_rows_for_outer(held, name, outer, shards, output, basis)
        source_x.append(features); source_v.append(np.asarray(_load_target(output, name)["V"], dtype=np.float64))
    model = fit_ranker(np.concatenate(source_x), list(zip(names, source_v)))
    # The held target file is deliberately read only at this point.
    held_x, held_actions = _feature_rows_for_outer(held, held, outer, shards, output, basis)
    held_v = np.asarray(_load_target(output, held)["V"], dtype=np.float64)
    score = rank_predict(model, held_x)
    metric = q1_metrics(held_v, score)
    stored = {"held": held, "outer_training": names, "feature_order": list(FEATURE_ORDER),
              "model": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.items()},
              "metrics": metric, "held_actions_nonkeep": int(np.count_nonzero(held_actions)),
              "held_count": int(len(held_v))}
    atomic_npz(output / "q1" / "folds" / f"{held}.npz", image_index=_load_target(output, held)["image_index"],
               patch_index=_load_target(output, held)["patch_index"], V=held_v, score=score, actions=held_actions)
    atomic_json(output / "q1" / "parameters" / f"{held}.json", stored)
    return stored


def evaluate_q1(folds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [folds[name]["metrics"] for name in r1.CLASSES]
    if any(row["spearman"] is None or row["sign_auc"] is None or row["bc20"] is None for row in rows):
        raise RuntimeError("P25R_ENGINEERING_STOP undefined Q1 metric")
    metrics = {"support_classes": int(sum(row["support"] for row in rows)),
               "median_spearman": float(np.median([row["spearman"] for row in rows])),
               "positive_spearman_classes": int(sum(row["spearman"] > 0.0 for row in rows)),
               "macro_sign_auc": float(np.mean([row["sign_auc"] for row in rows])),
               "macro_bc20": float(np.mean([row["bc20"] for row in rows])),
               "bc20_gt_20_classes": int(sum(row["bc20"] > .20 for row in rows)),
               "per_class": {name: folds[name]["metrics"] for name in r1.CLASSES}}
    gates = {"Q1_G1_SUPPORT": metrics["support_classes"] >= Q1_GATES["support_classes"],
             "Q1_G2_MEDIAN_SPEARMAN": metrics["median_spearman"] >= Q1_GATES["median_spearman"],
             "Q1_G3_POSITIVE_SPEARMAN": metrics["positive_spearman_classes"] >= Q1_GATES["positive_spearman"],
             "Q1_G4_SIGN_AUC": metrics["macro_sign_auc"] >= Q1_GATES["macro_sign_auc"],
             "Q1_G5_BC20": metrics["macro_bc20"] >= Q1_GATES["macro_bc20"],
             "Q1_G6_BC20_BREADTH": metrics["bc20_gt_20_classes"] >= Q1_GATES["bc20_classes"]}
    return {"metrics": metrics, "gates": gates, "pass": bool(all(gates.values()))}


def _local_equals_remote() -> bool:
    return git("rev-parse", "HEAD") == git("rev-parse", f"origin/{BRANCH}")


def _historical_immutable() -> bool:
    protected = ("results/sabra_car/r0", "results/sabra_cure/r1", "results/sabra_cure/r2", "results/sabra_cure/r2v2_harm",
                 "results/sabra_cure/post_r2v2_diagnostic", "tools/sabra_cure/r1.py", "tools/sabra_cure/r2.py", "tools/sabra_cure/r2v2_harm.py")
    return os.system(" ".join(["git", "diff", "--quiet", PARENT, "--", *protected]) + " >/dev/null 2>&1") == 0


def _verify_start(require_clean: bool) -> None:
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("START_STATE_FAILURE wrong P25R branch")
    if git("merge-base", "--is-ancestor", PARENT, "HEAD") != "":
        raise RuntimeError("START_STATE_FAILURE P25 parent absent")
    if require_clean and (not _local_equals_remote() or git("status", "--porcelain")):
        raise RuntimeError("P25R_ENGINEERING_STOP unpublished or dirty")
    if not _historical_immutable():
        raise RuntimeError("P25R_ENGINEERING_STOP historical artifact mutation")
    if (OUT / "ATTEMPT_STARTED.json").exists():
        raise RuntimeError("P25R_ENGINEERING_STOP attempt already exists")


def candle_parity(output: Path, count: int = 128) -> dict[str, Any]:
    """Pre-marker real-source direct-versus-fast engineering parity, no target is persisted."""
    panel = _load_panel(output, "candle")
    basis = load_basis(); name = "candle"
    source_path = r1.SOURCE_ROOT / "gt_free_cache" / f"{name}.npz"; utility_path = r1.UTILITY_ROOT / f"{name}.npz"
    with np.load(source_path, allow_pickle=False) as source, np.load(utility_path, allow_pickle=False) as utility_data:
        logits = np.asarray(source["native_logits"], dtype=np.float32); scores = np.asarray(source["native_pixel_probability"], dtype=np.float32)
        paths = source["image_path"].astype(str); utility = np.asarray(utility_data["utility"], dtype=np.float64)
    metadata, data_root = metadata_and_root(r2.DATA_ROOT); masks = load_masks(paths, metadata, data_root)
    base_s, base_p, base_t = p15.score_groups(scores, masks); base_ap = p15.ap_from_groups(base_p, base_t)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); max_error = 0.0; started = time.perf_counter(); cache: dict[int, np.ndarray] = {}
    for row in range(min(int(count), len(panel["patch_index"]))):
        image, patch = int(panel["image_index"][row]), int(panel["patch_index"][row]); sign = int(np.sign(utility[image, patch]))
        if sign == 0: continue
        margin = cache.setdefault(image, _deployed_margin(logits[image], device))
        cs, cp, ct = _candidate_delta(scores[image], masks[image], margin, patch, sign, basis)
        fast = p15.ap_with_delta(base_s, base_p, base_t, cs, cp, ct)
        correction = torch.zeros((1, PATCHES), device=device); correction[0, patch] = float(sign * ALPHA * MARGIN_SCALE)
        native = torch.from_numpy(logits[image:image + 1]).permute(1, 0, 2, 3).to(device)
        with torch.no_grad(): probability, _ = deploy_correction(native, correction)
        direct = scores.copy(); direct[image] = probability[0, 1].cpu().numpy().astype(np.float32)
        max_error = max(max_error, abs(float(exact_metrics(direct.reshape(-1), masks.reshape(-1))["pAP"]) - fast))
    result = {"status": "PASS" if max_error <= 1e-12 else "FAIL", "class": name, "count": int(min(count, len(panel["patch_index"]))),
              "max_abs_error": float(max_error), "elapsed_seconds": time.perf_counter() - started, "base_pap": float(base_ap),
              "uses_target_labels": False, "firewall": {"mvtec": 0, "medical": 0, "clip": 0, "phase2b_steps": 0}}
    if result["status"] != "PASS": raise RuntimeError("P25R_ENGINEERING_STOP candle parity")
    return result


def performance_benchmark(output: Path, parity: dict[str, Any] | None = None) -> dict[str, Any]:
    parity = candle_parity(output, 128) if parity is None else parity
    seconds_per_target = float(parity["elapsed_seconds"]) / max(1, int(parity["count"]))
    projected_target_seconds = seconds_per_target * TARGET_PATCHES_PER_CLASS * len(r1.CLASSES)
    # Q1 uses 12 R2-v2 nested folds plus one 22k-row pairwise ranker per fold; this is deliberately conservative.
    projected_q1_seconds = 12.0 * 240.0
    projected_total_hours = (projected_target_seconds + projected_q1_seconds) / 3600.0
    result = {"status": "PASS" if projected_total_hours <= 4.0 else "P25R_PERFORMANCE_NO_GO",
              "target_seconds_per_patch": seconds_per_target, "projected_target_hours": projected_target_seconds / 3600.0,
              "projected_q1_hours": projected_q1_seconds / 3600.0, "projected_total_hours": projected_total_hours,
              "preferred_hours": 2.0, "hard_no_go_hours": 4.0, "parity": parity,
              "additional_clip_forwards": 0, "phase2b_training_steps": 0}
    atomic_json(output / "performance_benchmark.json", result)
    return result


def pre_execution_audit(output: Path) -> dict[str, Any]:
    _verify_start(require_clean=True)
    if not (output / "panel_membership.json").is_file() or any(not (output / "panels" / f"{name}.npz").is_file() for name in r1.CLASSES):
        raise RuntimeError("P25R_ENGINEERING_STOP committed panel membership absent")
    parity = candle_parity(output, 128); benchmark = performance_benchmark(output, parity)
    payload = {"status": "PASS" if parity["status"] == "PASS" and benchmark["status"] == "PASS" else benchmark["status"],
               "parent_sha": PARENT, "branch": BRANCH, "audited_head": git("rev-parse", "HEAD"),
               "local_equals_remote": True, "worktree_clean": True, "panel_membership_sha256": sha256(output / "panel_membership.json"),
               "runner_sha256": sha256(ROOT / "tools/sabra_cure/patch_actionability_r1.py"),
               "tests_sha256": sha256(ROOT / "tests/test_sabra_cure_patch_actionability_r1.py"),
               "panel_count": TARGET_PATCHES_PER_CLASS * len(r1.CLASSES), "features": len(FEATURE_ORDER), "feature_order": list(FEATURE_ORDER),
               "candle_direct_fast_parity": parity, "performance": benchmark, "historical_immutable": True,
               "mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0}
    atomic_json(output / "pre_execution_audit.json", payload)
    if payload["status"] != "PASS": raise RuntimeError(payload["status"])
    return payload


def execute_once(output: Path) -> dict[str, Any]:
    """The sole P25R target/Q1 attempt.  Q2 is intentionally reached only on Q1 pass."""
    _verify_start(require_clean=True)
    audit_path = output / "pre_execution_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("P25R_ENGINEERING_STOP published pre-execution audit absent")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS" or audit.get("runner_sha256") != sha256(ROOT / "tools/sabra_cure/patch_actionability_r1.py") or audit.get("tests_sha256") != sha256(ROOT / "tests/test_sabra_cure_patch_actionability_r1.py"):
        raise RuntimeError("P25R_ENGINEERING_STOP stale pre-execution audit")
    attempt = {"status": "ATTEMPT_STARTED", "attempt_uuid": uuid.uuid4().hex, "runs": 1,
               "execution_base_sha": git("rev-parse", "HEAD"), "panel_sha256": sha256(output / "panel_membership.json"),
               "preregistration_sha": "c83ee53cfcde951d6375724e9048990fe235752d"}
    atomic_json(output / "ATTEMPT_STARTED.json", attempt)
    basis = load_basis(); target_rows: dict[str, Any] = {}
    for name in r1.CLASSES:
        target_rows[name] = target_class(name, _load_panel(output, name), output, basis)
        print(json.dumps({"event": "P25R_TARGET_CLASS_COMPLETE", "class": name}), flush=True)
    atomic_json(output / "target_summary.json", {"status": "PASS", "classes": target_rows, "target_total": 24000})
    shards, provenance = r1.load_shards(True); folds: dict[str, Any] = {}
    for held in r1.CLASSES:
        folds[held] = q1_fold(held, shards, output, basis)
        print(json.dumps({"event": "P25R_Q1_FOLD_COMPLETE", "held": held}), flush=True)
    q1 = evaluate_q1(folds); atomic_json(output / "q1_summary.json", q1)
    if q1["pass"]:
        # A Q1 pass without the separately frozen full-patch Q2 implementation is an engineering failure,
        # never a scientific shortcut or policy result.
        raise RuntimeError("P25R_ENGINEERING_STOP Q2 required but not implemented")
    summary = {"status": "P25_PATCH_BENEFIT_NOT_IDENTIFIABLE", "attempt": attempt, "pre_execution_audit": audit["status"],
               "target_summary": target_rows, "q1": q1, "q2": {"status": "NOT_RUN_Q1_FAILED"}, "folds_completed": 12,
               "provenance": provenance, "firewall": {"mvtec_accessed": False, "medical_accessed": False},
               "freeze": {"additional_clip_forwards": 0, "phase2b_training_steps": 0}}
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-panels", action="store_true")
    parser.add_argument("--pre-audit", action="store_true")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--candle-parity", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if sum((args.build_panels, args.pre_audit, args.run_once, args.candle_parity)) != 1:
        parser.error("choose exactly one action")
    output = args.output.resolve()
    if args.build_panels: result = build_all_panels(output)
    elif args.pre_audit: result = pre_execution_audit(output)
    elif args.candle_parity: result = candle_parity(output)
    else: result = execute_once(output)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
