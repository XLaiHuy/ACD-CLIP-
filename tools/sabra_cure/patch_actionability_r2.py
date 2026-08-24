"""P25R2 exact-batched native-anchored patch actionability recovery.

The target engine deliberately calls the frozen deployment operator for every
candidate row (batched by image).  It never uses P25R's retired sparse
post-deployment basis for an authoritative candidate score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.sabra_car.r0_direction import MARGIN_SCALE, deploy_correction, exact_metrics, load_masks, metadata_and_root
from tools.sabra_cure import context_value_risk_recovery as p15
from tools.sabra_cure import patch_actionability_r1 as p25r_features
from tools.sabra_cure import r1, r2, r2v2_harm


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sabra_cure/patch_actionability_r2"
DOC = ROOT / "research/sabra_cure/patch_actionability_r2"
BRANCH = "research/p25r2-sabra-cure-exact-patch-actionability-v1"
PARENT = "87d3c15b6fe4f62762bc87760960c1f83eda90d3"
ALPHA = .25
PATCHES = r1.PATCHES
TARGET_PATCHES_PER_CLASS = 2000
CAP_PER_IMAGE = 16
STRATA = 5
STRATUM_QUOTA = 80
NUMERICAL_FIXTURE = ("candle", 128)
BENEFIT_EPS = 1e-10
FEATURE_ORDER = p25r_features.FEATURE_ORDER


def git(*args: str) -> str:
    return r1.git(*args)


def _default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False, default=_default)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(label: str, *values: np.ndarray) -> None:
    if not all(np.isfinite(np.asarray(value)).all() for value in values):
        raise RuntimeError(f"P25R2_ENGINEERING_STOP non-finite {label}")


def _bin(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    cuts = np.quantile(array.reshape(-1), [.2, .4, .6, .8], method="linear")
    return np.minimum(np.searchsorted(cuts, array.reshape(-1), side="right"), 4).astype(np.int8).reshape(array.shape)


def _key(class_name: str, image_path: str, patch: int) -> bytes:
    return hashlib.sha256(class_name.encode() + b"\0" + image_path.encode() + b"\0" + str(int(patch)).encode()).digest()


def panel_for_class(name: str) -> dict[str, np.ndarray]:
    """GT-free deterministic 25-stratum panel with one global image cap."""
    with np.load(r1.SOURCE_ROOT / "gt_free_cache" / f"{name}.npz", allow_pickle=False) as data:
        paths = data["image_path"].astype(str)
        rank = np.asarray(data["margin_within_image_rank"], dtype=np.float64)
        sensitivity = np.asarray(data["deployment_sensitivity"], dtype=np.float64)
    if rank.shape != sensitivity.shape or rank.shape[1] != PATCHES:
        raise RuntimeError("P25R2_TARGET_PANEL_NO_GO source feature shape")
    finite("panel", rank, sensitivity)
    rb, sb = _bin(rank), _bin(sensitivity)
    ii, pp = np.indices(rank.shape, dtype=np.int64)
    cap: dict[int, int] = {}; chosen: list[tuple[int, int, int, int]] = []
    for rbin in range(STRATA):
        for sbin in range(STRATA):
            mask = (rb == rbin) & (sb == sbin)
            candidates = [(_key(name, str(paths[i]), int(j)), int(i), int(j)) for i, j in zip(ii[mask], pp[mask])]
            candidates.sort(key=lambda row: (row[0], str(paths[row[1]]), row[2]))
            picked: list[tuple[int, int]] = []
            for _, image, patch in candidates:
                if cap.get(image, 0) >= CAP_PER_IMAGE:
                    continue
                cap[image] = cap.get(image, 0) + 1; picked.append((image, patch))
                if len(picked) == STRATUM_QUOTA:
                    break
            if len(picked) != STRATUM_QUOTA:
                raise RuntimeError(f"P25R2_TARGET_PANEL_NO_GO {name} stratum={rbin},{sbin} count={len(picked)}")
            chosen.extend((image, patch, rbin, sbin) for image, patch in picked)
    chosen.sort(key=lambda row: (str(paths[row[0]]), row[1]))
    image = np.asarray([row[0] for row in chosen], dtype=np.int32)
    patch = np.asarray([row[1] for row in chosen], dtype=np.int32)
    rs = np.asarray([row[2] for row in chosen], dtype=np.int8); ss = np.asarray([row[3] for row in chosen], dtype=np.int8)
    counts = np.bincount(image, minlength=len(paths))
    if len(image) != TARGET_PATCHES_PER_CLASS or int(counts.max(initial=0)) > CAP_PER_IMAGE:
        raise RuntimeError("P25R2_TARGET_PANEL_NO_GO panel count/cap")
    return {"image_path": paths[image], "image_index": image, "patch_index": patch, "rank_stratum": rs, "sensitivity_stratum": ss}


def panel_hash(panel: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in ("image_path", "image_index", "patch_index", "rank_stratum", "sensitivity_stratum"):
        item = np.asarray(panel[key]); h.update(key.encode() + b"\0")
        h.update(item.astype("U").tobytes() if item.dtype.kind in "US" else item.tobytes())
    return h.hexdigest()


def build_panels(out: Path) -> dict[str, Any]:
    classes: dict[str, Any] = {}
    for name in r1.CLASSES:
        panel = panel_for_class(name)
        atomic_npz(out / "panels" / f"{name}.npz", **panel)
        classes[name] = {"count": int(len(panel["patch_index"])), "panel_hash": panel_hash(panel),
                         "cap": int(np.bincount(panel["image_index"]).max()),
                         "strata": np.bincount(panel["rank_stratum"] * STRATA + panel["sensitivity_stratum"], minlength=25).tolist()}
    result = {"status": "PASS", "parent_sha": PARENT, "uses_gt": False, "patches_per_class": TARGET_PATCHES_PER_CLASS,
              "cap_per_image": CAP_PER_IMAGE, "strata": 25, "classes": classes,
              "global_panel_digest": hashlib.sha256(json.dumps(classes, sort_keys=True).encode()).hexdigest(),
              "firewall": {"mvtec": 0, "medical": 0, "clip": 0, "phase2b_steps": 0}}
    atomic_json(out / "panel_feasibility.json", result)
    return result


def load_panel(out: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(out / "panels" / f"{name}.npz", allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def deployment_batch(native_logits: np.ndarray, patches: np.ndarray, signs: np.ndarray, device: torch.device) -> np.ndarray:
    """Actual frozen deployment; each batch row has one non-zero correction."""
    patches = np.asarray(patches, dtype=np.int64).reshape(-1); signs = np.asarray(signs, dtype=np.float32).reshape(-1)
    if len(patches) == 0 or patches.shape != signs.shape or np.any((patches < 0) | (patches >= PATCHES)):
        raise RuntimeError("P25R2_ENGINEERING_STOP candidate batch")
    native = torch.from_numpy(np.asarray(native_logits, dtype=np.float32)).to(device)
    batch = len(patches)
    repeated = native[:, None].expand(-1, batch, -1, -1).contiguous()
    correction = torch.zeros((batch, PATCHES), dtype=torch.float32, device=device)
    correction[torch.arange(batch, device=device), torch.from_numpy(patches).to(device)] = torch.from_numpy(signs).to(device) * float(ALPHA * MARGIN_SCALE)
    with torch.inference_mode():
        probability, _ = deploy_correction(repeated, correction)
    result = probability[:, 1].cpu().numpy().astype(np.float32, copy=False)
    if result.shape != (batch, 518, 518):
        raise RuntimeError("P25R2_ENGINEERING_STOP deployed candidate shape")
    return result


def class_state(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(r1.SOURCE_ROOT / "gt_free_cache" / f"{name}.npz", allow_pickle=False) as source, np.load(r1.UTILITY_ROOT / f"{name}.npz", allow_pickle=False) as utility:
        logits = np.asarray(source["native_logits"], dtype=np.float32); native = np.asarray(source["native_pixel_probability"], dtype=np.float32)
        paths = source["image_path"].astype(str); u = np.asarray(utility["utility"], dtype=np.float64)
        if not np.array_equal(paths, utility["image_path"].astype(str)):
            raise RuntimeError("P25R2_ENGINEERING_STOP source/utility alignment")
    meta, root = metadata_and_root(r2.DATA_ROOT); masks = load_masks(paths, meta, root)
    scores, positive, total = p15.score_groups(native, masks)
    return logits, native, paths, u, masks, scores, positive, total


def candidate_ap(base_scores: np.ndarray, base_positive: np.ndarray, base_total: np.ndarray, native_image: np.ndarray, candidate_image: np.ndarray, mask_image: np.ndarray) -> float:
    delta_s, delta_p, delta_t = p15.delta_groups(native_image, candidate_image, mask_image)
    return p15.ap_with_delta(base_scores, base_positive, base_total, delta_s, delta_p, delta_t)


def _order_equal(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.array_equal(np.argsort(np.asarray(first).reshape(-1), kind="mergesort"), np.argsort(np.asarray(second).reshape(-1), kind="mergesort")))


def numerical_audit(out: Path, fixture_count: int = 128) -> dict[str, Any]:
    """Pre-prereg batch=1 reference versus exact batched frozen deployment."""
    panel = load_panel(out, "candle"); selected = np.arange(min(fixture_count, len(panel["patch_index"])), dtype=np.int64)
    logits, native, paths, utility, masks, base_s, base_p, base_t = class_state("candle")
    native_ap = p15.ap_from_groups(base_p, base_t); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    max_score = 0.0; max_ap = 0.0; order_disagreements = 0; sign_disagreements = 0; ap_rows: list[float] = []; started = time.perf_counter()
    for image in np.unique(panel["image_index"][selected]):
        rows = selected[panel["image_index"][selected] == image]
        patches = panel["patch_index"][rows]; signs = np.sign(utility[int(image), patches]).astype(np.float32)
        batched = deployment_batch(logits[int(image)], patches, signs, device)
        single = np.concatenate([deployment_batch(logits[int(image)], np.asarray([patch]), np.asarray([sign]), device) for patch, sign in zip(patches, signs)], axis=0)
        for offset, row in enumerate(rows):
            score_error = float(np.max(np.abs(single[offset] - batched[offset]))); max_score = max(max_score, score_error)
            first = candidate_ap(base_s, base_p, base_t, native[int(image)], single[offset], masks[int(image)])
            second = candidate_ap(base_s, base_p, base_t, native[int(image)], batched[offset], masks[int(image)])
            difference = abs(first - second); max_ap = max(max_ap, difference); ap_rows.append(difference)
            if not _order_equal(single[offset], batched[offset]): order_disagreements += 1
    benefit_eps = max(1e-10, 20.0 * max_ap)
    # Only after the numerical envelope is known can sign agreement be assessed.
    for difference in ap_rows:
        if difference > benefit_eps: sign_disagreements += 1
    # Independent class-replacement parity of one actual batch=1 candidate.
    image = int(panel["image_index"][0]); patch = int(panel["patch_index"][0]); sign = float(np.sign(utility[image, patch]))
    candidate = deployment_batch(logits[image], np.asarray([patch]), np.asarray([sign]), device)[0]
    delta_ap = candidate_ap(base_s, base_p, base_t, native[image], candidate, masks[image])
    full = native.copy(); full[image] = candidate
    direct_ap = float(exact_metrics(full.reshape(-1), masks.reshape(-1))["pAP"])
    replacement_error = abs(delta_ap - direct_ap)
    result = {"status": "PASS" if max_score == 0.0 and max_ap <= benefit_eps and order_disagreements == 0 and replacement_error <= 1e-12 else "FAIL",
              "fixture": {"class": "candle", "candidates": int(len(selected)), "selection": "first deterministic panel rows"},
              "device": str(device), "batch_reference": 1, "production_batch": CAP_PER_IMAGE,
              "max_deployed_score_abs_error": max_score, "stable_score_order_disagreements": order_disagreements,
              "max_exact_ap_difference": max_ap, "benefit_eps": benefit_eps, "benefit_sign_disagreements_away_from_eps": sign_disagreements,
              "native_ap": native_ap, "class_replacement_direct_ap_error": replacement_error,
              "elapsed_seconds": time.perf_counter() - started,
              "firewall": {"mvtec": 0, "medical": 0, "clip": 0, "phase2b_steps": 0}}
    atomic_json(out / "numerical_audit.json", result)
    return result


def performance_audit(out: Path) -> dict[str, Any]:
    """Measure exact batched deployment plus exact grouped AP on a fixed fixture."""
    panel = load_panel(out, "candle"); selected = np.arange(min(128, len(panel["patch_index"])), dtype=np.int64)
    logits, native, _, utility, masks, base_s, base_p, base_t = class_state("candle")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); deploy_seconds = 0.0; ap_seconds = 0.0; count = 0
    for image in np.unique(panel["image_index"][selected]):
        rows = selected[panel["image_index"][selected] == image]; patches = panel["patch_index"][rows]; signs = np.sign(utility[int(image), patches]).astype(np.float32)
        started = time.perf_counter(); scores = deployment_batch(logits[int(image)], patches, signs, device); deploy_seconds += time.perf_counter() - started
        started = time.perf_counter()
        for score in scores: candidate_ap(base_s, base_p, base_t, native[int(image)], score, masks[int(image)])
        ap_seconds += time.perf_counter() - started; count += len(rows)
    target_hours = (deploy_seconds + ap_seconds) / count * (TARGET_PATCHES_PER_CLASS * len(r1.CLASSES)) / 3600.0
    # Conservative fixed engineering allowance for 12 nested ranker folds and conditional Q2.
    estimated_total = target_hours + .75
    result = {"status": "PASS" if estimated_total <= 4.0 else "P25R2_PERFORMANCE_NO_GO", "fixture_candidates": count,
              "deployment_seconds_per_candidate": deploy_seconds / count, "ap_seconds_per_candidate": ap_seconds / count,
              "candidates_per_second": count / (deploy_seconds + ap_seconds), "projected_target_hours": target_hours,
              "projected_total_hours": estimated_total, "host_rss_note": "class-local bounded state", "device": str(device),
              "firewall": {"mvtec": 0, "medical": 0, "clip": 0, "phase2b_steps": 0}}
    atomic_json(out / "performance_audit.json", result)
    return result


def target_class(name: str, out: Path, batch_size: int = CAP_PER_IMAGE) -> dict[str, Any]:
    """Generate one compact exact V shard using actual batched deployment maps."""
    panel = load_panel(out, name)
    logits, native, paths, utility, masks, base_s, base_p, base_t = class_state(name)
    native_ap = p15.ap_from_groups(base_p, base_t)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = len(panel["patch_index"]); candidate_ap_values = np.empty(count, dtype=np.float64)
    signs = np.sign(utility[panel["image_index"], panel["patch_index"]]).astype(np.int8)
    started = time.perf_counter()
    for image in np.unique(panel["image_index"]):
        rows = np.flatnonzero(panel["image_index"] == image)
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            score_batch = deployment_batch(logits[int(image)], panel["patch_index"][batch_rows], signs[batch_rows], device)
            for local, row in enumerate(batch_rows):
                candidate_ap_values[row] = candidate_ap(base_s, base_p, base_t, native[int(image)], score_batch[local], masks[int(image)])
    values = candidate_ap_values - native_ap
    finite("target", values, candidate_ap_values)
    target_path = out / "targets" / f"{name}.npz"
    atomic_npz(target_path, image_path=panel["image_path"], image_index=panel["image_index"], patch_index=panel["patch_index"],
               rank_stratum=panel["rank_stratum"], sensitivity_stratum=panel["sensitivity_stratum"], oracle_direction=signs,
               native_ap=np.full(count, native_ap, dtype=np.float64), candidate_ap=candidate_ap_values, V=values)
    return {"class": name, "count": int(count), "native_ap": native_ap, "elapsed_seconds": time.perf_counter() - started,
            "target_sha256": sha256(target_path), "positive": int(np.count_nonzero(values > BENEFIT_EPS)),
            "negative": int(np.count_nonzero(values < -BENEFIT_EPS)), "near_zero": int(np.count_nonzero(np.abs(values) <= BENEFIT_EPS))}


def load_target(out: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(out / "targets" / f"{name}.npz", allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _rank_fraction(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1); order = np.argsort(flat, kind="mergesort")
    out = np.empty(len(flat), dtype=np.float64); out[order] = np.arange(len(flat), dtype=np.float64)
    return out / max(1, len(flat) - 1)


def fast_impact_rows(name: str, panel: dict[str, np.ndarray], actions: np.ndarray) -> np.ndarray:
    """Fixed GT-free action-impact proxies; never an authoritative target path."""
    basis = p25r_features.load_basis()
    with np.load(r1.SOURCE_ROOT / "gt_free_cache" / f"{name}.npz", allow_pickle=False) as source:
        native = np.asarray(source["native_pixel_probability"], dtype=np.float32)
        logits = np.asarray(source["native_logits"], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = np.zeros((len(actions), 8), dtype=np.float64)
    for image in np.unique(panel["image_index"]):
        rows = np.flatnonzero(panel["image_index"] == image); base = native[int(image)].reshape(-1)
        rank = _rank_fraction(base); sorted_score = np.sort(base, kind="mergesort")
        margin = p25r_features._deployed_margin(logits[int(image)], device)
        iqr = max(float(np.subtract(*np.quantile(base, [.75, .25], method="linear"))), 1e-8)
        q95, q80 = np.quantile(base, [.95, .80], method="linear")
        for row in rows:
            patch, action = int(panel["patch_index"][row]), int(actions[row]); support, values = basis.support(patch)
            before = base[support].astype(np.float64); before_rank = rank[support]
            if action == 0:
                result[row, :2] = (np.median(before_rank), np.quantile(before_rank, .90, method="linear")); continue
            # This proxy is a fixed GT-free feature only.  P25R2 V always uses deployment_batch.
            shifted = torch.sigmoid(torch.from_numpy(margin[support]).to(device) + float(action * ALPHA * MARGIN_SCALE) * torch.from_numpy(values).to(device)).cpu().numpy().astype(np.float32)
            delta = shifted.astype(np.float64) - before
            moved_rank = np.searchsorted(sorted_score, shifted, side="right") / max(1, len(sorted_score) - 1)
            result[row] = (np.median(before_rank), np.quantile(before_rank, .90, method="linear"),
                           delta.mean() / iqr, np.quantile(np.abs(delta), .90, method="linear") / iqr,
                           np.median(moved_rank - before_rank), np.quantile(np.abs(moved_rank - before_rank), .90, method="linear"),
                           np.mean((before <= q95) & (shifted > q95)) + np.mean((before > q95) & (shifted <= q95)),
                           np.mean((before <= q80) & (shifted > q80)) + np.mean((before > q80) & (shifted <= q80)))
    finite("impact proxy", result)
    return result


def feature_rows_for_outer(held: str, name: str, outer: dict[str, Any], shards: dict[str, r1.Shard], out: Path) -> tuple[np.ndarray, np.ndarray]:
    panel = load_panel(out, name); row = panel["image_index"].astype(np.int64) * PATCHES + panel["patch_index"].astype(np.int64)
    if name == held:
        base = r2v2_harm.harm_features(shards[held].x, outer["mu"], outer["sigma"]); risk = np.asarray(outer["risk_h"], dtype=np.float64)
    else:
        group = next(item for item in outer["level1"] if item["name"] == name)
        base, risk = np.asarray(group["f"], dtype=np.float64), np.asarray(group["r_h"], dtype=np.float64)
    action = r2v2_harm.action(base[:, 14], risk, float(outer["tau_harm"]))[row]
    impact = fast_impact_rows(name, panel, action)
    features = np.column_stack((base[row], risk[row], action.astype(np.float64), impact)).astype(np.float64)
    if features.shape != (len(row), len(FEATURE_ORDER)):
        raise RuntimeError("P25R2_ENGINEERING_STOP 32D schema")
    finite("feature rows", features)
    return features, action


def q1_fold(held: str, shards: dict[str, r1.Shard], out: Path) -> dict[str, Any]:
    """Held V is deliberately opened only after the source ranker is frozen."""
    outer = r2v2_harm.outer(held, shards); names = [name for name in r1.CLASSES if name != held]
    source_features: list[np.ndarray] = []; source_values: list[np.ndarray] = []
    for name in names:
        x, _ = feature_rows_for_outer(held, name, outer, shards, out)
        source_features.append(x); source_values.append(np.asarray(load_target(out, name)["V"], dtype=np.float64))
    model = p25r_features.fit_ranker(np.concatenate(source_features), list(zip(names, source_values)))
    held_x, action = feature_rows_for_outer(held, held, outer, shards, out)
    held_target = load_target(out, held); score = p25r_features.rank_predict(model, held_x); values = np.asarray(held_target["V"], dtype=np.float64)
    metric = p25r_features.q1_metrics(values, score)
    metric["pearson"] = r1.pearson(score, values)
    metric["positive_count"] = int(np.count_nonzero(values > BENEFIT_EPS)); metric["negative_count"] = int(np.count_nonzero(values < -BENEFIT_EPS)); metric["near_zero_count"] = int(np.count_nonzero(np.abs(values) <= BENEFIT_EPS)); metric["score_variance"] = float(np.var(score))
    serial_model = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.items()}
    result = {"held": held, "outer_training": names, "feature_order": list(FEATURE_ORDER), "model": serial_model, "metrics": metric,
              "held_nonkeep_actions": int(np.count_nonzero(action)), "held_count": int(len(values))}
    atomic_npz(out / "q1" / "folds" / f"{held}.npz", image_index=held_target["image_index"], patch_index=held_target["patch_index"], V=values, score=score, actions=action)
    atomic_json(out / "q1" / "parameters" / f"{held}.json", result)
    return result


def evaluate_q1(folds: dict[str, Any]) -> dict[str, Any]:
    rows = [folds[name]["metrics"] for name in r1.CLASSES]
    if any(row["spearman"] is None or row["sign_auc"] is None or row["bc20"] is None for row in rows):
        raise RuntimeError("P25R2_ENGINEERING_STOP undefined Q1 metric")
    metrics = {"support_classes": int(sum(row["positive_count"] > 0 and row["negative_count"] > 0 for row in rows)),
               "median_spearman": float(np.median([row["spearman"] for row in rows])), "positive_spearman_classes": int(sum(row["spearman"] > 0 for row in rows)),
               "macro_sign_auc": float(np.mean([row["sign_auc"] for row in rows])), "macro_bc20": float(np.mean([row["bc20"] for row in rows])),
               "bc20_gt_20_classes": int(sum(row["bc20"] > .20 for row in rows)), "per_class": {name: folds[name]["metrics"] for name in r1.CLASSES}}
    gates = {"G1_SUPPORT": metrics["support_classes"] >= 10, "G2_MEDIAN_SPEARMAN": metrics["median_spearman"] >= .20,
             "G3_POSITIVE_SPEARMAN": metrics["positive_spearman_classes"] >= 9, "G4_SIGN_AUC": metrics["macro_sign_auc"] >= .65,
             "G5_BC20": metrics["macro_bc20"] >= .35, "G6_BC20_BREADTH": metrics["bc20_gt_20_classes"] >= 9}
    return {"metrics": metrics, "gates": gates, "pass": bool(all(gates.values()))}


def _policy_metrics(action: np.ndarray, y: np.ndarray, baseline: np.ndarray) -> dict[str, float | int | None]:
    action = np.asarray(action, dtype=np.int8); y = np.asarray(y, dtype=np.float64); baseline = np.asarray(baseline, dtype=np.int8)
    acted = action != 0; wrong = acted & (action * np.sign(y) < 0); base_wrong = (baseline != 0) & (baseline * np.sign(y) < 0)
    harm = float(np.sum(np.abs(y[wrong]))); base_harm = float(np.sum(np.abs(y[base_wrong])))
    return {"acted": int(acted.sum()), "coverage": float(acted.mean()), "wrong_sign": int(wrong.sum()),
            "wrong_rate": None if not np.any(acted) else float(wrong[acted].mean()), "weighted_harm": harm,
            "relative_weighted_harm_reduction": None if base_harm <= 0.0 else float(1.0 - harm / base_harm)}


def select_source_policy(groups: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Frozen 3x3 source-only grid; its labels never include the outer held class."""
    risk = np.concatenate([group["risk"] for group in groups]); score = np.concatenate([group["score"] for group in groups])
    candidates: list[dict[str, Any]] = []
    for risk_quantile in (.4, .6, .8):
        risk_threshold = float(np.quantile(risk, risk_quantile, method="linear"))
        for benefit_quantile in (.8, .9, .95):
            benefit_threshold = float(np.quantile(score, benefit_quantile, method="linear")); per_class = []
            for group in groups:
                selected = (group["base_action"] != 0) & (group["risk"] <= risk_threshold) & (group["score"] > benefit_threshold)
                action = np.where(selected, group["base_action"], 0).astype(np.int8)
                safety = _policy_metrics(action, group["y"], group["base_action"])
                values = group["V"][selected]
                per_class.append({"name": group["name"], "safety": safety, "mean_V": None if not len(values) else float(values.mean()),
                                  "positive_V": int(np.count_nonzero(values > BENEFIT_EPS))})
            wrong_values = [row["safety"]["wrong_rate"] for row in per_class if row["safety"]["wrong_rate"] is not None]
            reduction_values = [row["safety"]["relative_weighted_harm_reduction"] for row in per_class if row["safety"]["relative_weighted_harm_reduction"] is not None]
            positive_classes = sum(row["positive_V"] > 0 for row in per_class)
            criterion = float(np.mean([row["mean_V"] for row in per_class if row["mean_V"] is not None])) if any(row["mean_V"] is not None for row in per_class) else -np.inf
            wrong = float(np.mean(wrong_values)) if wrong_values else 1.0; reduction = float(np.mean(reduction_values)) if reduction_values else -np.inf
            candidates.append({"risk_quantile": risk_quantile, "benefit_quantile": benefit_quantile, "risk_threshold": risk_threshold,
                               "benefit_threshold": benefit_threshold, "wrong_rate": wrong, "harm_reduction": reduction,
                               "positive_source_classes": positive_classes, "criterion": criterion, "per_class": per_class,
                               "eligible": wrong <= .05 and reduction >= .50})
    eligible = [row for row in candidates if row["eligible"]]
    selected = min(eligible, key=lambda row: (-row["criterion"], -row["positive_source_classes"], row["risk_quantile"], -row["benefit_quantile"])) if eligible else None
    return selected, candidates


def q2_fold(held: str, shards: dict[str, r1.Shard], out: Path) -> dict[str, Any]:
    """Conditional Q2 with source-only selection and one held deployment."""
    params = json.loads((out / "q1" / "parameters" / f"{held}.json").read_text()); raw = params["model"]
    model = {key: np.asarray(value, dtype=np.float64) if key in {"median", "iqr", "beta"} else value for key, value in raw.items()}
    outer = r2v2_harm.outer(held, shards); names = [name for name in r1.CLASSES if name != held]; groups: list[dict[str, Any]] = []
    for name in names:
        x, action = feature_rows_for_outer(held, name, outer, shards, out); group = next(item for item in outer["level1"] if item["name"] == name)
        groups.append({"name": name, "score": p25r_features.rank_predict(model, x), "risk": np.asarray(group["r_h"], dtype=np.float64),
                       "base_action": action, "y": np.asarray(group["y"], dtype=np.float64), "V": np.asarray(load_target(out, name)["V"], dtype=np.float64)})
    selected, candidates = select_source_policy(groups)
    held_x, held_base_action = feature_rows_for_outer(held, held, outer, shards, out); held_score = p25r_features.rank_predict(model, held_x)
    if selected is None:
        held_action = np.zeros_like(held_base_action, dtype=np.int8)
        policy = {"status": "NO_ELIGIBLE_SOURCE_POLICY", "candidates": candidates, "selected": None}
    else:
        held_risk = np.asarray(outer["risk_h"], dtype=np.float64)[load_panel(out, held)["image_index"].astype(np.int64) * PATCHES + load_panel(out, held)["patch_index"].astype(np.int64)]
        chosen = (held_base_action != 0) & (held_risk <= selected["risk_threshold"]) & (held_score > selected["benefit_threshold"])
        held_action = np.where(chosen, held_base_action, 0).astype(np.int8)
        policy = {"status": "SELECTED", "candidates": candidates, "selected": selected}
    panel = load_panel(out, held)
    atomic_npz(out / "q2" / "actions" / f"{held}.npz", image_path=panel["image_path"], image_index=panel["image_index"], patch_index=panel["patch_index"], actions=held_action, benefit_score=held_score)
    # Held GT opens only after actions were serialized above.
    logits, native, paths, _, masks, _, _, _ = class_state(held)
    full_action = np.zeros((len(paths), PATCHES), dtype=np.int8); full_action[panel["image_index"], panel["patch_index"]] = held_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from tools.sabra_car.r0_direction import evaluate_correction
    score_map, _ = evaluate_correction(logits, masks, (full_action.astype(np.float32) * float(ALPHA * MARGIN_SCALE)), device, 4)
    deployed = exact_metrics(score_map.reshape(-1), masks.reshape(-1)); native_metric = exact_metrics(native.reshape(-1), masks.reshape(-1))
    index = panel["image_index"].astype(np.int64) * PATCHES + panel["patch_index"].astype(np.int64)
    safety = _policy_metrics(held_action, np.asarray(outer["y"])[index], held_base_action)
    result = {"held": held, "selection": policy, "metrics": {"native_pap": float(native_metric["pAP"]), "pap": float(deployed["pAP"]),
              "native_pauroc": float(native_metric["pAUROC"]), "pauroc": float(deployed["pAUROC"]), "delta_pap": float(deployed["pAP"] - native_metric["pAP"])}, "safety": safety,
              "action_sha256": sha256(out / "q2" / "actions" / f"{held}.npz")}
    atomic_json(out / "q2" / "parameters" / f"{held}.json", result)
    return result


def evaluate_q2(folds: dict[str, Any]) -> dict[str, Any]:
    metrics = {"macro_native_pap": float(np.mean([folds[name]["metrics"]["native_pap"] for name in r1.CLASSES])),
               "macro_pap": float(np.mean([folds[name]["metrics"]["pap"] for name in r1.CLASSES])),
               "macro_native_pauroc": float(np.mean([folds[name]["metrics"]["native_pauroc"] for name in r1.CLASSES])),
               "macro_pauroc": float(np.mean([folds[name]["metrics"]["pauroc"] for name in r1.CLASSES])),
               "nonregressing": int(sum(folds[name]["metrics"]["delta_pap"] >= 0.0 for name in r1.CLASSES)),
               "improving": int(sum(folds[name]["metrics"]["delta_pap"] > 0.0 for name in r1.CLASSES)),
               "wrong_rate": float(np.mean([folds[name]["safety"]["wrong_rate"] if folds[name]["safety"]["wrong_rate"] is not None else 1.0 for name in r1.CLASSES])),
               "harm_reduction": float(np.mean([folds[name]["safety"]["relative_weighted_harm_reduction"] if folds[name]["safety"]["relative_weighted_harm_reduction"] is not None else 0.0 for name in r1.CLASSES]))}
    harm_summary = json.loads((ROOT / "results/sabra_cure/r2v2_harm/summary.json").read_text())
    harm_only = float(harm_summary["metrics"]["macro_pap"]["harm"])
    gates = {"G1_AUDIT": True, "G2_WRONG": metrics["wrong_rate"] <= .05, "G3_HARM": metrics["harm_reduction"] >= .50,
             "G4_PAP": metrics["macro_pap"] >= metrics["macro_native_pap"] + .0025, "G5_NONREGRESSION": metrics["nonregressing"] >= 9,
             "G6_IMPROVING": metrics["improving"] >= 7, "G7_AUROC": metrics["macro_pauroc"] >= metrics["macro_native_pauroc"] - .005,
             "G8_HARM_ONLY": metrics["macro_pap"] > harm_only}
    return {"metrics": metrics | {"harm_only_pap": harm_only}, "gates": gates, "pass": bool(all(gates.values()))}


def progress(out: Path, stage: str, targets: int, q1: int, q2: int, overall: float, status: str, event: str, started: float) -> None:
    atomic_json(out / "PROGRESS.json", {"current_stage": stage, "target_classes_completed": targets, "q1_folds_completed": q1,
                                          "q2_folds_completed": q2, "task_progress_percent": overall, "overall_progress_percent": overall,
                                          "last_event": event, "elapsed_seconds": time.perf_counter() - started, "status": status,
                                          "firewall": {"mvtec": 0, "medical": 0, "additional_clip": 0, "phase2b": 0}})


def pre_execution_audit(out: Path) -> dict[str, Any]:
    if git("branch", "--show-current") != BRANCH or git("merge-base", "--is-ancestor", PARENT, "HEAD") != "":
        raise RuntimeError("START_STATE_FAILURE P25R2 branch/parent")
    if git("rev-parse", "HEAD") != git("rev-parse", f"origin/{BRANCH}") or git("status", "--porcelain"):
        raise RuntimeError("P25R2_ENGINEERING_STOP unpublished or dirty execution base")
    numerical = json.loads((out / "numerical_audit.json").read_text()); performance = json.loads((out / "performance_audit.json").read_text()); panel = json.loads((out / "panel_feasibility.json").read_text())
    if numerical["status"] != "PASS" or performance["status"] != "PASS" or panel["status"] != "PASS":
        raise RuntimeError("P25R2_ENGINEERING_STOP preprereg evidence")
    result = {"status": "PASS", "parent_sha": PARENT, "audited_head": git("rev-parse", "HEAD"), "prereg_sha": "233d16b",
              "runner_sha256": sha256(ROOT / "tools/sabra_cure/patch_actionability_r2.py"), "tests_sha256": sha256(ROOT / "tests/test_sabra_cure_patch_actionability_r2.py"),
              "panel_digest": panel["global_panel_digest"], "panel_classes": len(panel["classes"]), "benefit_eps": BENEFIT_EPS,
              "numerical": {key: numerical[key] for key in ("status", "max_deployed_score_abs_error", "stable_score_order_disagreements", "max_exact_ap_difference", "class_replacement_direct_ap_error")},
              "performance": performance, "feature_count": len(FEATURE_ORDER), "feature_order": list(FEATURE_ORDER),
              "firewall": {"mvtec": 0, "medical": 0, "clip": 0, "phase2b_steps": 0}}
    atomic_json(out / "pre_execution_audit.json", result)
    return result


def execute_once(out: Path) -> dict[str, Any]:
    if (out / "ATTEMPT_STARTED.json").exists() or (out / "summary.json").exists():
        raise RuntimeError("P25R2_ENGINEERING_STOP attempt already exists")
    audit = json.loads((out / "pre_execution_audit.json").read_text())
    if audit.get("status") != "PASS" or audit.get("runner_sha256") != sha256(ROOT / "tools/sabra_cure/patch_actionability_r2.py") or audit.get("tests_sha256") != sha256(ROOT / "tests/test_sabra_cure_patch_actionability_r2.py"):
        raise RuntimeError("P25R2_ENGINEERING_STOP missing/stale execution audit")
    started = time.perf_counter(); marker = {"status": "ATTEMPT_STARTED", "attempt_uuid": uuid.uuid4().hex, "runs": 1,
                                             "parent_sha": PARENT, "prereg_sha": audit["prereg_sha"], "execution_base_sha": git("rev-parse", "HEAD"),
                                             "panel_hash": audit["panel_digest"], "benefit_eps": BENEFIT_EPS, "production_backend": "cuda", "production_batch": CAP_PER_IMAGE}
    atomic_json(out / "ATTEMPT_STARTED.json", marker); progress(out, "TARGET", 0, 0, 0, 75.0, "RUNNING", "marker_created", started)
    target_summary: dict[str, Any] = {}
    for index, name in enumerate(r1.CLASSES, start=1):
        target_summary[name] = target_class(name, out)
        progress(out, "TARGET", index, 0, 0, 75.0 + 10.0 * index / 12.0, "RUNNING", f"target_complete:{name}", started)
        print(json.dumps({"event": "P25R2_TARGET_COMPLETE", "class": name}), flush=True)
    atomic_json(out / "target_summary.json", {"status": "PASS", "classes": target_summary, "targets": 24000})
    shards, provenance = r1.load_shards(True); q1_folds: dict[str, Any] = {}
    for index, held in enumerate(r1.CLASSES, start=1):
        q1_folds[held] = q1_fold(held, shards, out)
        progress(out, "Q1", 12, index, 0, 85.0 + 10.0 * index / 12.0, "RUNNING", f"q1_complete:{held}", started)
        print(json.dumps({"event": "P25R2_Q1_COMPLETE", "held": held}), flush=True)
    q1 = evaluate_q1(q1_folds); atomic_json(out / "q1_summary.json", q1)
    if not q1["pass"]:
        summary = {"status": "P25_PATCH_BENEFIT_NOT_IDENTIFIABLE", "attempt": marker, "target_summary": target_summary, "q1": q1,
                   "q2": {"status": "NOT_REQUIRED_Q1_FAILED"}, "folds_completed": 12, "provenance": provenance,
                   "firewall": {"mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0}}
        atomic_json(out / "summary.json", summary); progress(out, "TERMINAL", 12, 12, 0, 100.0, "COMPLETED", summary["status"], started); return summary
    q2_folds: dict[str, Any] = {}
    for index, held in enumerate(r1.CLASSES, start=1):
        q2_folds[held] = q2_fold(held, shards, out)
        progress(out, "Q2", 12, 12, index, 95.0 + 5.0 * index / 12.0, "RUNNING", f"q2_complete:{held}", started)
        print(json.dumps({"event": "P25R2_Q2_COMPLETE", "held": held}), flush=True)
    q2 = evaluate_q2(q2_folds); atomic_json(out / "q2_summary.json", q2)
    status = "P25_PATCH_ACTIONABILITY_IDENTIFIED" if q2["pass"] else "P25_PATCH_BENEFIT_NOT_POLICY_TRANSFERABLE"
    summary = {"status": status, "attempt": marker, "target_summary": target_summary, "q1": q1, "q2": q2, "folds_completed": 12,
               "provenance": provenance, "firewall": {"mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0}}
    atomic_json(out / "summary.json", summary); progress(out, "TERMINAL", 12, 12, 12, 100.0, "COMPLETED", status, started); return summary


def preprereg_audit(out: Path) -> dict[str, Any]:
    if git("branch", "--show-current") != BRANCH or git("merge-base", "--is-ancestor", PARENT, "HEAD") != "":
        raise RuntimeError("START_STATE_FAILURE P25R2 provenance")
    panels = build_panels(out); numerical = numerical_audit(out); performance = performance_audit(out)
    result = {"status": "PASS" if panels["status"] == numerical["status"] == performance["status"] == "PASS" else "FAIL",
              "parent_sha": PARENT, "panel": panels, "numerical": numerical, "performance": performance,
              "no_attempt_marker": not (out / "ATTEMPT_STARTED.json").exists(),
              "firewall": {"mvtec": 0, "medical": 0, "clip": 0, "phase2b_steps": 0}}
    atomic_json(out / "preprereg_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--preprereg-audit", action="store_true"); parser.add_argument("--pre-execution-audit", action="store_true"); parser.add_argument("--run-once", action="store_true"); parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if sum((args.preprereg_audit, args.pre_execution_audit, args.run_once)) != 1: parser.error("choose exactly one action")
    output = args.output.resolve()
    result = preprereg_audit(output) if args.preprereg_audit else pre_execution_audit(output) if args.pre_execution_audit else execute_once(output)
    print(json.dumps(result, indent=2, sort_keys=True, default=_default))


if __name__ == "__main__":
    main()
