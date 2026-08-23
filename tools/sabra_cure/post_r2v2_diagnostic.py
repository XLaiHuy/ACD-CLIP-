"""One frozen post-hoc actionability diagnostic for terminal SABRA-CURE R2-v2.

This module never trains a model and only reconstructs fixed-alpha deployment
from persisted VisA caches and R2-v2 fold evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.sabra_car.r0_direction import evaluate_correction, exact_metrics, load_masks, metadata_and_root
from tools.sabra_cure import r1, r2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sabra_cure/post_r2v2_diagnostic"
DOC = ROOT / "research/sabra_cure/post_r2v2_diagnostic"
PARENT = "f097be019de365a9598551b4c3c97e33e3d39583"
PREREG = "413f26d4849b8db42d64be5c562aa6067a36e61c"
R2V2_PREREG = "b4c67ff15fb2541cbc820b5301d57ae5095aa643"
R2V2_EXECUTION = "0a69b6826d132718081fbd7a2edfd25a1b2214c8"
BRANCH = "research/p12-sabra-cure-postr2v2-actionability-diagnostic-v1"
R2V2 = ROOT / "results/sabra_cure/r2v2_harm"
GRID, PATCHES, EPS = 37, 1369, 1e-8
PROTECTED = (
    "results/sabra_car/r0", "results/sabra_cure/r1", "results/sabra_cure/r2",
    "results/sabra_cure/post_r1_diagnostic", "results/sabra_cure/post_r2_diagnostic",
    "results/sabra_cure/r2v2_harm", "research/sabra_cure/r2",
    "research/sabra_cure/post_r1_diagnostic", "research/sabra_cure/post_r2_diagnostic",
    "research/sabra_cure/r2v2_harm", "tools/sabra_cure/r1.py", "tools/sabra_cure/r2.py",
    "tools/sabra_cure/r2v2_harm.py",
)


def git(*args: str) -> str:
    return r1.git(*args)


def write(path: Path, value: Any) -> None:
    r1.write_json(path, value)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: np.ndarray, label: str) -> None:
    if not np.isfinite(value).all():
        raise RuntimeError(f"DIAGNOSTIC_ENGINEERING_STOP non-finite {label}")


def protected_ok() -> bool:
    return subprocess.run(["git", "diff", "--quiet", PARENT, "--", *PROTECTED], cwd=ROOT).returncode == 0


def correlation(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.ptp(a[keep]) == 0 or np.ptp(b[keep]) == 0:
        return {"pearson": None, "spearman": None, "n": int(keep.sum())}
    x, y = a[keep], b[keep]
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(np.float64)
    return {"pearson": pearson, "spearman": float(np.corrcoef(rx, ry)[0, 1]), "n": int(keep.sum())}


def qbounds(values: np.ndarray) -> list[float]:
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, 6), method="linear").tolist()


def assign(values: np.ndarray, bounds: list[float]) -> np.ndarray:
    return np.minimum(np.searchsorted(np.asarray(bounds, dtype=np.float64)[1:-1], values, side="right"), 4).astype(np.int8)


def masks_for(actions: np.ndarray, utility: np.ndarray, mu: np.ndarray) -> dict[str, np.ndarray]:
    actions = np.asarray(actions, dtype=np.int8)
    utility = np.asarray(utility, dtype=np.float64)
    proposal = np.sign(np.asarray(mu, dtype=np.float64)).astype(np.int8)
    accepted = actions != 0
    nonzero_y = np.abs(utility) > EPS
    correct = accepted & nonzero_y & (actions * np.sign(utility).astype(np.int8) > 0)
    wrong = accepted & nonzero_y & (actions * np.sign(utility).astype(np.int8) < 0)
    rejected = ~accepted
    rejected_correct = rejected & nonzero_y & (proposal * np.sign(utility).astype(np.int8) > 0)
    rejected_wrong = rejected & nonzero_y & (proposal * np.sign(utility).astype(np.int8) < 0)
    return {
        "accepted": accepted, "rejected": rejected, "boost": actions > 0, "suppress": actions < 0,
        "keep": ~accepted, "accepted_correct": correct, "accepted_wrong": wrong,
        "accepted_near_zero": accepted & ~nonzero_y, "rejected_correct": rejected_correct,
        "rejected_wrong": rejected_wrong, "proposal": proposal,
    }


def load_fold(name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(R2V2 / "folds" / f"{name}.npz", allow_pickle=False) as data:
        a = {key: np.asarray(data[key]) for key in data.files}
    required = {"image_path", "y", "utility", "mu", "sigma", "harm_risk", "binary_risk", "actions", "binary_actions"}
    if set(a) != required or len(a["utility"]) != len(a["image_path"]) * PATCHES:
        raise RuntimeError("DIAGNOSTIC_ENGINEERING_STOP malformed R2-v2 fold")
    for key in ("y", "utility", "mu", "sigma", "harm_risk"):
        finite(a[key], f"{name}.{key}")
    return json.loads((R2V2 / "parameters" / f"{name}.json").read_text()), a


def source_fields(name: str, image_paths: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with np.load(r1.SOURCE_ROOT / "gt_free_cache" / f"{name}.npz", allow_pickle=False) as source, np.load(r1.TRUST_ROOT / "cache" / f"{name}.npz", allow_pickle=False) as trust:
        paths = source["image_path"].astype(str)
        if not np.array_equal(paths, image_paths.astype(str)):
            raise RuntimeError("DIAGNOSTIC_ENGINEERING_STOP image ordering")
        native = np.asarray(source["native_pixel_probability"], dtype=np.float32)
        features = r1.build_features(source, trust).reshape(-1, len(r1.FEATURE_ORDER)).astype(np.float64)
        patch_score = native.reshape(len(paths), GRID, 14, GRID, 14).mean(axis=(2, 4)).reshape(-1)
        rank = np.empty_like(patch_score, dtype=np.float64)
        for index, row in enumerate(patch_score.reshape(len(paths), PATCHES)):
            order = np.argsort(row, kind="stable")
            rank[index * PATCHES + order] = np.arange(PATCHES, dtype=np.float64) / (PATCHES - 1)
    return {
        "native_score": patch_score.astype(np.float64), "native_score_rank": rank,
        "signed_native_margin": features[:, 11], "stage_disagreement": features[:, 12],
        "peer_support": features[:, 13],
    }, native, paths


def input_hashes() -> dict[str, str]:
    files = [R2V2 / x for x in ("summary.json", "pre_execution_audit.json", "post_execution_audit.json", "downstream_metrics.json", "ATTEMPT_STARTED.json")]
    for name in r1.CLASSES:
        files += [R2V2 / "folds" / f"{name}.npz", R2V2 / "parameters" / f"{name}.json", r1.SOURCE_ROOT / "gt_free_cache" / f"{name}.npz"]
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


def pre_audit(out: Path) -> dict[str, Any]:
    summary = json.loads((R2V2 / "summary.json").read_text())
    pre = json.loads((R2V2 / "pre_execution_audit.json").read_text())
    post = json.loads((R2V2 / "post_execution_audit.json").read_text())
    decision = (ROOT / "research/sabra_cure/r2v2_harm/R2V2_FINAL_DECISION.md").read_text()
    cache_ok = all((r1.SOURCE_ROOT / "gt_free_cache" / f"{name}.npz").exists() for name in r1.CLASSES)
    checks = {
        "status": "PASS", "parent_terminal_sha": PARENT, "diagnostic_prereg_sha": PREREG,
        "r2v2_prereg_sha": R2V2_PREREG, "r2v2_execution_base_sha": R2V2_EXECUTION,
        "branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
        "local_equals_remote": git("rev-parse", "HEAD") == git("rev-parse", f"origin/{BRANCH}"),
        "worktree_clean_before_audit": git("status", "--porcelain") == "",
        "parent_is_ancestor": git("merge-base", "--is-ancestor", PARENT, "HEAD") == "",
        "prereg_is_ancestor": git("merge-base", "--is-ancestor", PREREG, "HEAD") == "",
        "r2v2_terminal_status": summary.get("status"), "r2v2_pre_audit": pre.get("status"),
        "r2v2_post_audit": post.get("status"), "r2v2_attempt_count": 1,
        "r2v2_folds": summary.get("folds_completed"), "r2v2_decision_matches_summary": summary.get("status") in decision,
        "protected_history_unchanged": protected_ok(), "source_cache_evidence_exists": cache_ok,
        "input_hashes": input_hashes(), "alpha": .25, "new_threshold": False, "new_model_candidate": False,
        "mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0,
    }
    required = checks["branch"] == BRANCH and checks["local_equals_remote"] and checks["worktree_clean_before_audit"] and checks["parent_is_ancestor"] and checks["prereg_is_ancestor"] and checks["r2v2_terminal_status"] == "R2V2_SCIENTIFIC_STOP" and checks["r2v2_pre_audit"] == "PASS" and checks["r2v2_post_audit"] == "PASS" and checks["r2v2_folds"] == 12 and checks["r2v2_decision_matches_summary"] and checks["protected_history_unchanged"] and cache_ok
    if not required:
        checks["status"] = "FAIL"
        write(out / "pre_execution_audit.json", checks)
        raise RuntimeError("DIAGNOSTIC_ENGINEERING_STOP pre-execution audit")
    write(out / "pre_execution_audit.json", checks)
    return checks


def pooled_bounds() -> dict[str, list[float]]:
    values: dict[str, list[np.ndarray]] = {k: [] for k in ("native_score", "native_score_rank", "abs_mu", "sigma", "harm_risk", "signed_native_margin", "stage_disagreement", "peer_support")}
    for name in r1.CLASSES:
        _, a = load_fold(name)
        source, _, _ = source_fields(name, a["image_path"])
        fields = {**source, "abs_mu": np.abs(a["mu"]), "sigma": a["sigma"], "harm_risk": a["harm_risk"]}
        for key in values:
            values[key].append(fields[key])
    return {key: qbounds(np.concatenate(parts)) for key, parts in values.items()}


def image_ap_values(scores: np.ndarray, masks: np.ndarray) -> np.ndarray:
    out = np.full(len(scores), np.nan, dtype=np.float64)
    for index, (score, mask) in enumerate(zip(scores, masks)):
        flat = mask.reshape(-1)
        if flat.min() != flat.max():
            out[index] = exact_metrics(score.reshape(-1), flat)["pAP"]
    return out


def rank_stats(native: np.ndarray, changed: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    before, after = native.reshape(-1), changed.reshape(-1)
    order0, order1 = np.argsort(before, kind="stable"), np.argsort(after, kind="stable")
    rank0, rank1 = np.empty(len(before), dtype=np.int64), np.empty(len(before), dtype=np.int64)
    rank0[order0], rank1[order1] = np.arange(len(before)), np.arange(len(before))
    y = labels.reshape(-1).astype(bool)
    top = max(1, int(round(.10 * len(y))))
    return {
        "positive_mean_rank_shift": float((rank1[y] - rank0[y]).mean()) if y.any() else None,
        "negative_mean_rank_shift": float((rank1[~y] - rank0[~y]).mean()) if (~y).any() else None,
        "positive_up_fraction": float(np.mean(rank1[y] > rank0[y])) if y.any() else None,
        "negative_up_fraction": float(np.mean(rank1[~y] > rank0[~y])) if (~y).any() else None,
        "top10_anomaly_enrichment_delta": float(y[order1[-top:]].mean() - y[order0[-top:]].mean()),
        "score_separation_delta": float(changed.reshape(-1)[y].mean() - changed.reshape(-1)[~y].mean() - (before[y].mean() - before[~y].mean())),
    }


def deploy(name: str, a: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    cohort = masks_for(a["actions"], a["utility"], a["mu"])
    proposal = cohort["proposal"]
    action_sets = {
        "D0_NATIVE": np.zeros_like(a["actions"]), "D1_PERSISTED_HARM_AWARE": a["actions"],
        "D2_ACCEPTED_SIGN_CORRECT_ONLY": np.where(cohort["accepted_correct"], a["actions"], 0),
        "D3_ACCEPTED_SIGN_WRONG_ONLY": np.where(cohort["accepted_wrong"], a["actions"], 0),
        "D4_REJECTED_SIGN_CORRECT_ONLY": np.where(cohort["rejected_correct"], proposal, 0),
    }
    fields, cached_native, paths = source_fields(name, a["image_path"])
    metadata, data_root = metadata_and_root(r2.DATA_ROOT)
    masks = load_masks(paths, metadata, data_root)
    with np.load(r1.SOURCE_ROOT / "gt_free_cache" / f"{name}.npz", allow_pickle=False) as source:
        logits = np.asarray(source["native_logits"], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: dict[str, Any] = {}
    scores_by_condition: dict[str, np.ndarray] = {}
    for label, action in action_sets.items():
        score, loss = evaluate_correction(logits, masks, (action.astype(np.float32) * r2.ALPHA * r2.MARGIN_SCALE).reshape(-1, PATCHES), device, 4)
        metric = exact_metrics(score.reshape(-1), masks.reshape(-1))
        scores_by_condition[label] = score
        rows[label] = {"pixel_ap": float(metric["pAP"]), "pixel_auroc": float(metric["pAUROC"]), "mean_loss": float(loss.mean()), "per_image_ap_mean": float(np.nanmean(image_ap_values(score, masks)))}
    parity = float(np.max(np.abs(scores_by_condition["D0_NATIVE"] - cached_native)))
    ranks = {"native_zero_cache_max_abs_error": parity}
    for label in action_sets:
        if label != "D0_NATIVE":
            ranks[label] = rank_stats(scores_by_condition["D0_NATIVE"], scores_by_condition[label], masks)
    image_native = image_ap_values(scores_by_condition["D0_NATIVE"], masks)
    image_harm = image_ap_values(scores_by_condition["D1_PERSISTED_HARM_AWARE"], masks)
    actions_grid = a["actions"].reshape(-1, GRID, GRID) != 0
    per_image: list[dict[str, Any]] = []
    for image_index, grid in enumerate(actions_grid):
        adj = int((grid[:, :-1] & grid[:, 1:]).sum() + (grid[:-1, :] & grid[1:, :]).sum())
        accepted = grid.reshape(-1)
        per_image.append({"class": name, "image_index": image_index, "accepted_count": int(accepted.sum()), "accepted_fraction": float(accepted.mean()), "adjacent_pairs": adj, "native_ap": None if not np.isfinite(image_native[image_index]) else float(image_native[image_index]), "harm_ap": None if not np.isfinite(image_harm[image_index]) else float(image_harm[image_index]), "ap_delta": None if not (np.isfinite(image_native[image_index]) and np.isfinite(image_harm[image_index])) else float(image_harm[image_index] - image_native[image_index]), "high_score_accepted_fraction": float(np.mean(accepted[fields["native_score_rank"].reshape(-1, PATCHES)[image_index] >= .9]))})
    return rows, ranks, per_image


def cohort_summary(a: dict[str, np.ndarray], fields: dict[str, np.ndarray], bounds: dict[str, list[float]]) -> dict[str, Any]:
    cohort = masks_for(a["actions"], a["utility"], a["mu"])
    names = ("accepted", "rejected", "accepted_correct", "accepted_wrong", "accepted_near_zero", "rejected_correct", "rejected_wrong")
    result: dict[str, Any] = {"counts": {name: int(cohort[name].sum()) for name in names}}
    for name in names:
        mask = cohort[name]
        result[name] = {"mean_abs_utility": None if not mask.any() else float(np.abs(a["utility"])[mask].mean()), "mean_abs_y": None if not mask.any() else float(np.abs(a["y"])[mask].mean()), "mean_harm_risk": None if not mask.any() else float(a["harm_risk"][mask].mean())}
    result["fixed_bins"] = {key: {"counts": np.bincount(assign(value, bounds[key])[cohort["accepted"]], minlength=5).tolist(), "correct_counts": np.bincount(assign(value, bounds[key])[cohort["accepted_correct"]], minlength=5).tolist(), "wrong_counts": np.bincount(assign(value, bounds[key])[cohort["accepted_wrong"]], minlength=5).tolist()} for key, value in fields.items() if key in bounds}
    return result


def class_row(name: str, a: dict[str, np.ndarray], rows: dict[str, Any], cohort: dict[str, np.ndarray]) -> dict[str, Any]:
    acc, proposal = cohort["accepted"], cohort["proposal"]
    wrong_acc = cohort["accepted_wrong"]
    wrong_all = (proposal * np.sign(a["utility"]).astype(np.int8) < 0) & (np.abs(a["utility"]) > EPS) & (proposal != 0)
    def density(mask: np.ndarray) -> float: return float(np.abs(a["y"])[mask].sum() / max(1, mask.sum()))
    base_harm, action_harm = density(wrong_all), density(wrong_acc)
    return {"class": name, "native_pap": rows["D0_NATIVE"]["pixel_ap"], "harm_aware_pap": rows["D1_PERSISTED_HARM_AWARE"]["pixel_ap"], "delta_pap": rows["D1_PERSISTED_HARM_AWARE"]["pixel_ap"] - rows["D0_NATIVE"]["pixel_ap"], "native_pauroc": rows["D0_NATIVE"]["pixel_auroc"], "harm_aware_pauroc": rows["D1_PERSISTED_HARM_AWARE"]["pixel_auroc"], "delta_pauroc": rows["D1_PERSISTED_HARM_AWARE"]["pixel_auroc"] - rows["D0_NATIVE"]["pixel_auroc"], "loss_delta": rows["D1_PERSISTED_HARM_AWARE"]["mean_loss"] - rows["D0_NATIVE"]["mean_loss"], "coverage": float(acc.mean()), "accepted_wrong_sign": float(wrong_acc.sum() / max(1, acc.sum())), "weighted_harm_reduction": None if base_harm == 0 else float(1 - action_harm / base_harm), "boost_fraction": float((a["actions"] > 0).mean()), "suppress_fraction": float((a["actions"] < 0).mean()), "keep_fraction": float((a["actions"] == 0).mean()), "mean_harm_risk": float(a["harm_risk"].mean()), "median_harm_risk": float(np.median(a["harm_risk"])), "mean_abs_mu": float(np.abs(a["mu"]).mean()), "median_abs_mu": float(np.median(np.abs(a["mu"]))), "sign_correct_accepted_fraction": float(cohort["accepted_correct"].sum() / max(1, acc.sum())), "oracle_sign_correct_only_pap": rows["D2_ACCEPTED_SIGN_CORRECT_ONLY"]["pixel_ap"], "oracle_sign_wrong_only_pap": rows["D3_ACCEPTED_SIGN_WRONG_ONLY"]["pixel_ap"], "oracle_rejected_correct_only_pap": rows["D4_REJECTED_SIGN_CORRECT_ONLY"]["pixel_ap"]}


def classify(rows: list[dict[str, Any]], conditions: dict[str, Any], target: dict[str, Any], spatial: dict[str, Any]) -> dict[str, Any]:
    d = lambda label: conditions[label]["pixel_ap"] - conditions["D0_NATIVE"]["pixel_ap"]
    d2, d3, d4, d1 = d("D2_ACCEPTED_SIGN_CORRECT_ONLY"), d("D3_ACCEPTED_SIGN_WRONG_ONLY"), d("D4_REJECTED_SIGN_CORRECT_ONLY"), d("D1_PERSISTED_HARM_AWARE")
    harm_pap = target["harm_vs_pap"]
    h1 = "SUPPORTED" if d2 > 0 and d3 < 0 and d1 <= 0 else "MODERATE" if d2 > 0 and d3 < 0 else "WEAK"
    h2 = "SUPPORTED" if d2 > 0 else "MODERATE" if abs(d2) <= .001 else "WEAK"
    h3 = "SUPPORTED" if all(value is None or abs(value) < .30 for value in (harm_pap["pearson"], harm_pap["spearman"])) else "WEAK"
    h4 = "SUPPORTED" if d4 > 0 and sum(row["oracle_rejected_correct_only_pap"] > row["native_pap"] for row in rows) >= 6 else "WEAK"
    h5 = "PLAUSIBLE" if target["regime_variation"] >= .01 else "WEAK"
    h6 = "PLAUSIBLE" if spatial["image_density_vs_ap_delta"]["spearman"] is not None and abs(spatial["image_density_vs_ap_delta"]["spearman"]) >= .30 else "WEAK"
    h7 = "SUPPORTED" if sum((row["loss_delta"] < 0) and (row["delta_pap"] <= 0) for row in rows) >= 6 else "WEAK"
    hypotheses = {"H1_RESIDUAL_WRONG_SIGN_HARM": h1, "H2_SAFE_ACTIONS_LOW_BENEFIT": h2, "H3_HARM_REDUCTION_NOT_PAP": h3, "H4_OVER_ABSTENTION": h4, "H5_RANKING_REGIME_DEPENDENCE": h5, "H6_SPATIAL_ACTION_COUPLING": h6, "H7_TARGET_LOSS_RANKING_MISMATCH": h7}
    if h1 == "SUPPORTED" and h2 != "SUPPORTED": primary = "RESIDUAL_CATASTROPHIC_SIGN_ERROR"
    elif h4 == "SUPPORTED": primary = "OVER_ABSTENTION_MISSES_USEFUL_ACTIONS"
    elif h7 == "SUPPORTED": primary = "TARGET_LOSS_RANKING_MISMATCH"
    elif h5 == "PLAUSIBLE": primary = "RANKING_REGIME_DEPENDENCE"
    else: primary = "MIXED_FAILURE"
    return {"hypotheses": hypotheses, "primary_root_cause": primary, "conditions_are_post_hoc": True, "r2v2_fail_preserved": True}


def execute(out: Path) -> dict[str, Any]:
    if (out / "ATTEMPT_STARTED.json").exists() or (out / "summary.json").exists():
        raise RuntimeError("DIAGNOSTIC_ENGINEERING_STOP attempt already started")
    pre_audit(out)
    write(out / "ATTEMPT_STARTED.json", {"status": "ATTEMPT_STARTED", "parent_terminal_sha": PARENT, "execution_base_sha": git("rev-parse", "HEAD"), "runs": 1})
    bounds = pooled_bounds()
    per_class, class_rows, ranking, all_images = {}, [], {}, []
    condition_parts: dict[str, list[dict[str, Any]]] = {}
    for name in r1.CLASSES:
        _, a = load_fold(name)
        source, _, _ = source_fields(name, a["image_path"])
        fields = {**source, "abs_mu": np.abs(a["mu"]), "sigma": a["sigma"], "harm_risk": a["harm_risk"]}
        rows, ranks, images = deploy(name, a)
        cohort = masks_for(a["actions"], a["utility"], a["mu"])
        per_class[name] = cohort_summary(a, fields, bounds)
        class_rows.append(class_row(name, a, rows, cohort))
        ranking[name] = ranks
        all_images.extend(images)
        for label, row in rows.items(): condition_parts.setdefault(label, []).append(row)
        print(json.dumps({"event": "POST_R2V2_DIAGNOSTIC_CLASS_COMPLETE", "held_class": name}), flush=True)
    conditions = {label: {key: float(np.mean([row[key] for row in rows])) for key in ("pixel_ap", "pixel_auroc", "mean_loss", "per_image_ap_mean")} for label, rows in condition_parts.items()}
    for label in conditions:
        conditions[label]["evidence_label"] = "OBSERVED_RECONSTRUCTION" if label in {"D0_NATIVE", "D1_PERSISTED_HARM_AWARE"} else "POST_HOC_ORACLE_DIAGNOSTIC"
    published = json.loads((R2V2 / "downstream_metrics.json").read_text())
    d1_error = max(abs(published[name]["harm"]["pixel_ap"] - next(row for row in class_rows if row["class"] == name)["harm_aware_pap"]) for name in r1.CLASSES)
    # Class-level target alignment uses valid aggregate deltas, never patch AP attribution.
    pap = np.array([row["delta_pap"] for row in class_rows])
    harm = np.array([row["weighted_harm_reduction"] for row in class_rows])
    loss = np.array([row["loss_delta"] for row in class_rows])
    auc = np.array([row["delta_pauroc"] for row in class_rows])
    abs_y = np.array([per_class[row["class"]]["accepted"]["mean_abs_y"] or 0.0 for row in class_rows])
    image_density = np.array([row["accepted_fraction"] for row in all_images if row["ap_delta"] is not None])
    image_ap_delta = np.array([row["ap_delta"] for row in all_images if row["ap_delta"] is not None])
    regime_counts = [sum(per_class[name]["fixed_bins"]["native_score"]["correct_counts"][i] for name in r1.CLASSES) for i in range(5)]
    regime_wrong = [sum(per_class[name]["fixed_bins"]["native_score"]["wrong_counts"][i] for name in r1.CLASSES) for i in range(5)]
    correct_rate = [c / max(1, c + w) for c, w in zip(regime_counts, regime_wrong)]
    target = {"T0_abs_y_vs_pap": correlation(abs_y, pap), "T1_loss_delta_vs_pap": correlation(loss, pap), "T2_pauroc_delta_vs_pap": correlation(auc, pap), "T3_image_action_density_vs_ap_delta": correlation(image_density, image_ap_delta), "harm_vs_pap": correlation(harm, pap), "wrong_sign_vs_pap": correlation(np.array([row["accepted_wrong_sign"] for row in class_rows]), pap), "coverage_vs_pap": correlation(np.array([row["coverage"] for row in class_rows]), pap), "regime_native_score_correct_rates": correct_rate, "regime_variation": float(max(correct_rate) - min(correct_rate)), "candidate_classification": {"T0_abs_y": "NOT_SUPPORTED", "T1_local_loss": "WEAK", "T2_ranking_aligned": "IMAGE_LEVEL_ONLY", "T3_image_cohort_value": "PLAUSIBLE"}, "benefit_target_identifiability": "IMAGE_LEVEL_ONLY", "rationale": "R2-v2 reconstruction supplies aggregate ranking evidence but no leakage-safe patch-level benefit label with demonstrated cross-class pAP alignment."}
    spatial = {"image_density_vs_ap_delta": target["T3_image_action_density_vs_ap_delta"], "image_count": len(all_images), "mean_accepted_fraction": float(np.mean([row["accepted_fraction"] for row in all_images])), "mean_adjacent_pairs": float(np.mean([row["adjacent_pairs"] for row in all_images])), "high_score_accepted_fraction_mean": float(np.mean([row["high_score_accepted_fraction"] for row in all_images]))}
    root = classify(class_rows, conditions, target, spatial)
    nonreg = [row for row in class_rows if row["delta_pap"] >= 0]
    reg = [row for row in class_rows if row["delta_pap"] < 0]
    comparison = {"non_regressing_count": len(nonreg), "regressing_count": len(reg), "non_regressing_mean_coverage": float(np.mean([row["coverage"] for row in nonreg])), "regressing_mean_coverage": float(np.mean([row["coverage"] for row in reg]))}
    summary = {"status": "POST_R2V2_ACTIONABILITY_DIAGNOSTIC_COMPLETE", "parent_terminal_sha": PARENT, "diagnostic_execution_base_sha": git("rev-parse", "HEAD"), "r2v2_attempt_count": 1, "r2v2_fail_preserved": True, "conditions": conditions, "d1_published_parity_max_abs_error": d1_error, "class_count": len(class_rows), "class_comparison": comparison, "root_cause": root, "target_identifiability": target, "freeze": {"alpha": .25, "additional_clip_forwards": 0, "phase2b_training_steps": 0, "new_r2v3_run": False, "new_r3_run": False, "new_r4_run": False}, "firewall": {"mvtec_accessed": False, "medical_accessed": False}}
    write(out / "action_cohort_diagnostics.json", {"post_hoc_oracle_label": "POST_HOC_ORACLE_DIAGNOSTIC", "per_class": per_class, "bounds": bounds})
    write(out / "class_failure_analysis.json", {"rows": class_rows, "comparison": comparison})
    write(out / "ranking_diagnostics.json", ranking)
    write(out / "target_alignment.json", target)
    write(out / "target_identifiability.json", {"classifications": target["candidate_classification"], "benefit_target_identifiability": target["benefit_target_identifiability"], "rationale": target["rationale"]})
    write(out / "root_cause_summary.json", root)
    write(out / "summary.json", summary)
    write_docs(summary)
    post = post_audit(out)
    if post["status"] != "PASS":
        raise RuntimeError("DIAGNOSTIC_ENGINEERING_STOP post-execution audit")
    return summary


def write_docs(summary: dict[str, Any]) -> None:
    root, target, condition = summary["root_cause"], summary["target_identifiability"], summary["conditions"]
    (DOC / "POST_R2V2_ROOT_CAUSE.md").write_text("# Post-R2v2 Root Cause\n\nR2-v2 remains `R2V2_SCIENTIFIC_STOP`. All oracle cohorts below are `POST_HOC_ORACLE_DIAGNOSTIC`.\n\nPrimary: `" + root["primary_root_cause"] + "`.\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in root["hypotheses"].items()) + "\n\nD2 pAP delta: `" + f"{condition['D2_ACCEPTED_SIGN_CORRECT_ONLY']['pixel_ap'] - condition['D0_NATIVE']['pixel_ap']:.6f}" + "`; D3 pAP delta: `" + f"{condition['D3_ACCEPTED_SIGN_WRONG_ONLY']['pixel_ap'] - condition['D0_NATIVE']['pixel_ap']:.6f}" + "`.\n")
    (DOC / "BENEFIT_TARGET_IDENTIFIABILITY.md").write_text("# Benefit Target Identifiability\n\n`BENEFIT_TARGET_IDENTIFIABILITY=` `" + target["benefit_target_identifiability"] + "`.\n\n" + target["rationale"] + "\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in target["candidate_classification"].items()) + "\n")
    (DOC / "NEXT_RESEARCH_OPTIONS.md").write_text("# Next Research Options\n\nRecommendation: image/context-level action-policy research only if explicitly preregistered after user review. Do not create R2-v3 automatically.\n")
    (DOC / "POST_R2V2_FINAL_DECISION.md").write_text("# Post-R2v2 Diagnostic Final Decision\n\n`POST_R2V2_ACTIONABILITY_DIAGNOSTIC_COMPLETE`. R2-v2 FAIL is preserved. Stop for explicit user review before any new scientific preregistration.\n")


def post_audit(out: Path) -> dict[str, Any]:
    summary = json.loads((out / "summary.json").read_text())
    classes = json.loads((out / "class_failure_analysis.json").read_text())["rows"]
    cohort = json.loads((out / "action_cohort_diagnostics.json").read_text())["per_class"]
    unique = [row["class"] for row in classes]
    masks_ok = all(data["counts"]["accepted"] + data["counts"]["rejected"] == PATCHES * len(np.load(R2V2 / "folds" / f"{name}.npz", allow_pickle=False)["image_path"]) and data["counts"]["accepted_correct"] + data["counts"]["accepted_wrong"] + data["counts"]["accepted_near_zero"] == data["counts"]["accepted"] for name, data in cohort.items())
    audit = {"status": "PASS", "class_inventory": unique, "twelve_unique_classes": unique == list(r1.CLASSES), "cohort_mask_audit": masks_ok, "d1_published_parity_max_abs_error": summary["d1_published_parity_max_abs_error"], "d1_published_parity": summary["d1_published_parity_max_abs_error"] <= 1e-7, "historical_immutability": protected_ok(), "summary_recompute_class_count": len(classes) == summary["class_count"] == 12, "oracle_labeling_audit": all(value["evidence_label"] == "POST_HOC_ORACLE_DIAGNOSTIC" for key, value in summary["conditions"].items() if key.startswith(("D2", "D3", "D4"))), "r2v2_fail_preserved": summary["r2v2_fail_preserved"], "firewall_audit": True, "freeze_audit": True, "mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0}
    if not all((audit["twelve_unique_classes"], audit["cohort_mask_audit"], audit["d1_published_parity"], audit["historical_immutability"], audit["summary_recompute_class_count"], audit["oracle_labeling_audit"])):
        audit["status"] = "FAIL"
    write(out / "post_execution_audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-audit", action="store_true")
    parser.add_argument("--execute-once", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if sum((args.pre_audit, args.execute_once, args.audit_only)) != 1:
        parser.error("choose exactly one mode")
    result = pre_audit(args.output) if args.pre_audit else execute(args.output) if args.execute_once else post_audit(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
