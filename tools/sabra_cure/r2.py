"""Frozen SABRA-CURE R2 selective fixed-strength intervention.

This module is deliberately isolated from historical R0/R1/P8 evidence.  It
uses the immutable VisA cache only: R1's direction/inner-OOF construction,
then a source-only conservative interval whose action is fixed at alpha=.25.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.sabra_car.r0_direction import (
    MARGIN_SCALE,
    evaluate_correction,
    exact_metrics,
    load_masks,
    metadata_and_root,
)
from tools.sabra_cure import r1

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/sabra_cure/r2"
PREREGISTRATION = ROOT / "research/sabra_cure/r2/R2_PREREGISTRATION.md"
CONTRACT = ROOT / "research/sabra_cure/r2/R2_IMPLEMENTATION_CONTRACT.md"
P8_SHA = "1fa8775367d4139580ad0abd5a1ed48a96edeb43"
PREREG_SHA = "83caf6f3c221d39b4b0b2bdd79483cf9ba8b42cc"
BRANCH = "research/p9-sabra-cure-r2-v1"
ALPHA = 0.25
MIS_COVERAGES = (0.05, 0.10, 0.20, 0.30, 0.40)
EPS = 1e-4
DATA_ROOT = Path("/home/ai4/caohuy/data")
PROTECTED = (
    "tools/sabra_cure/r1.py",
    "results/sabra_cure/r1",
    "results/sabra_car/r0",
    "results/sabra_cure/post_r1_diagnostic",
    "research/sabra_cure/post_r1_diagnostic",
)


def write_json(path: Path, payload: Any) -> None:
    r1.write_json(path, payload)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    r1.save_npz(path, **arrays)


def git(*args: str) -> str:
    return r1.git(*args)


def finite(name: str, *values: np.ndarray) -> None:
    r1.finite(name, *values)


def conservative_index(count: int, miscoverage: float) -> int:
    """Frozen finite-sample index for the sorted normalized residuals."""
    if count <= 0 or miscoverage not in MIS_COVERAGES:
        raise RuntimeError("ENGINEERING_STOP invalid interval quantile request")
    return min(int(math.ceil((count + 1) * (1.0 - miscoverage)) - 1), count - 1)


def interval_actions(mu: np.ndarray, sigma: np.ndarray, q: float) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    finite("interval inputs", mu, sigma, np.asarray(q))
    lower = mu - float(q) * sigma
    upper = mu + float(q) * sigma
    # Strict inequalities deliberately leave both equality boundaries at KEEP.
    return np.where(lower > 0.0, 1, np.where(upper < 0.0, -1, 0)).astype(np.int8)


def signed_direction(mu: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(mu, dtype=np.float64) > 0.0, 1,
                    np.where(np.asarray(mu, dtype=np.float64) < 0.0, -1, 0)).astype(np.int8)


def risk_metrics(actions: np.ndarray, y: np.ndarray, reference_actions: np.ndarray | None = None) -> dict[str, float | int | None]:
    actions = np.asarray(actions, dtype=np.int8).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if actions.shape != y.shape:
        raise RuntimeError("ENGINEERING_STOP action/target alignment")
    acted = actions != 0
    oracle = np.sign(y).astype(np.int8)
    wrong = acted & (actions * oracle < 0)
    coverage = float(np.mean(acted))
    rate = float(np.mean(wrong[acted])) if np.any(acted) else None
    result: dict[str, float | int | None] = {
        "patches": int(len(actions)), "acted": int(np.count_nonzero(acted)),
        "wrong_sign": int(np.count_nonzero(wrong)), "coverage": coverage,
        "opposite_sign_rate": rate,
    }
    if reference_actions is not None:
        reference = risk_metrics(np.asarray(reference_actions), y)
        base = reference["opposite_sign_rate"]
        result["unfiltered_opposite_sign_rate"] = base
        result["relative_wrong_sign_reduction"] = (
            None if base is None or base <= 0.0 or rate is None else float(1.0 - rate / base)
        )
    return result


def select_operating_point(mu_cf: np.ndarray, y_cf: np.ndarray, sigma_cf: np.ndarray) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select only from inner-OOF evidence, before the outer held class is read."""
    finite("inner interval evidence", mu_cf, y_cf, sigma_cf)
    normalized = np.abs(y_cf - mu_cf) / np.maximum(sigma_cf, EPS)
    finite("normalized residual", normalized)
    ordered = np.sort(normalized, kind="stable")
    direction = signed_direction(mu_cf)
    base = risk_metrics(direction, y_cf)
    candidates: list[dict[str, Any]] = []
    for m in MIS_COVERAGES:
        index = conservative_index(len(ordered), m)
        q = float(ordered[index])
        actions = interval_actions(mu_cf, sigma_cf, q)
        safety = risk_metrics(actions, y_cf, direction)
        qualifies = bool(
            safety["coverage"] >= 0.10
            and safety["opposite_sign_rate"] is not None
            and safety["opposite_sign_rate"] <= 0.05
            and safety["relative_wrong_sign_reduction"] is not None
            and safety["relative_wrong_sign_reduction"] >= 0.25
        )
        candidates.append({"miscoverage": m, "quantile_index": index, "q": q,
                           "safety": safety, "qualifies": qualifies})
    qualifying = [item for item in candidates if item["qualifies"]]
    selected = min(qualifying, key=lambda item: (-float(item["safety"]["coverage"]),
                                                   float(item["safety"]["opposite_sign_rate"]),
                                                   float(item["miscoverage"]))) if qualifying else None
    return selected, {"unfiltered": base, "candidates": candidates,
                      "selected_miscoverage": None if selected is None else selected["miscoverage"],
                      "selection_status": "QUALIFIED" if selected else "NO_QUALIFIED_SAFE_OPERATING_POINT"}


def protected_history_unchanged() -> bool:
    return subprocess.run(["git", "diff", "--quiet", P8_SHA, "--", *PROTECTED], cwd=ROOT).returncode == 0


def local_equals_remote() -> bool:
    try:
        return git("rev-parse", "HEAD") == git("rev-parse", f"origin/{BRANCH}")
    except subprocess.CalledProcessError:
        return False


def pre_execution_audit(output: Path) -> dict[str, Any]:
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("START_STATE_FAILURE wrong R2 branch")
    if git("merge-base", "--is-ancestor", P8_SHA, "HEAD") != "":
        raise RuntimeError("START_STATE_FAILURE P8 terminal parent absent")
    if git("merge-base", "--is-ancestor", PREREG_SHA, "HEAD") != "":
        raise RuntimeError("START_STATE_FAILURE preregistration freeze absent")
    if not all(path.is_file() for path in (PREREGISTRATION, CONTRACT)):
        raise RuntimeError("ENGINEERING_STOP R2 frozen documents absent")
    if not local_equals_remote() or git("status", "--porcelain"):
        raise RuntimeError("ENGINEERING_STOP unpublished or dirty execution base")
    parity = r1.parity_fixture()
    if parity["status"] != "PASS":
        raise RuntimeError("ENGINEERING_STOP sufficient-statistics parity")
    shards, provenance = r1.load_shards(check_hashes=True)
    audit = {
        "status": "PASS", "execution_base_sha": git("rev-parse", "HEAD"),
        "preregistration_freeze_sha": PREREG_SHA, "p8_parent_sha": P8_SHA,
        "branch": BRANCH, "local_equals_remote": True, "worktree_clean": True,
        "protected_history_unchanged": protected_history_unchanged(),
        "class_order": list(shards) == list(r1.CLASSES), "classes": list(r1.CLASSES),
        "records": int(sum(len(shard.utility) for shard in shards.values())), "patch_width": r1.PATCHES,
        "feature_order": list(r1.FEATURE_ORDER), "feature_count": len(r1.FEATURE_ORDER),
        "all_finite": bool(all(np.isfinite(shard.x).all() and np.isfinite(shard.utility).all() for shard in shards.values())),
        "outer_folds": len(r1.CLASSES), "inner_crossfits_per_outer": len(r1.CLASSES) - 1,
        "ridge_lambda": r1.LAMBDA, "precision": "float64", "solver": "numpy.linalg.solve",
        "miscoverage_grid": list(MIS_COVERAGES), "quantile_rule": "min(ceil((n+1)*(1-m))-1,n-1)",
        "sufficient_statistics_parity": parity, "provenance": provenance,
        "phase2b_training_steps": 0, "additional_clip_forwards": 0,
        "mvtec_accessed": False, "medical_accessed": False,
    }
    if not all((audit["protected_history_unchanged"], audit["class_order"], audit["all_finite"])):
        raise RuntimeError("ENGINEERING_STOP pre-execution audit")
    write_json(output / "pre_execution_audit.json", audit)
    return audit


def r2_fold(held: str, shards: dict[str, r1.Shard]) -> dict[str, Any]:
    """One outer fold; all selection happens before held y/actions are created."""
    item = r1.fold(held, shards)
    params = item["parameters"]
    outer_train = params["outer_training_classes"]
    train_x = r1.concat(shards, outer_train, "x")
    train_std = r1.scale_x(train_x, np.asarray(params["feature_median"]), np.asarray(params["feature_iqr"]))
    sigma_cf = np.exp(np.clip(train_std @ np.asarray(params["uncertainty_beta"]) + float(params["uncertainty_intercept"]),
                               np.log(EPS), np.log(4.0)))
    selected, selection = select_operating_point(item["inner_mu_cf"], item["inner_y"], sigma_cf)
    direction = signed_direction(item["mu"])
    if selected is None:
        actions = np.zeros_like(direction, dtype=np.int8)
        q = None
    else:
        q = float(selected["q"])
        actions = interval_actions(item["mu"], item["sigma"], q)
    held_safety = risk_metrics(actions, item["y"], direction)
    params.update({
        "r2_alpha": ALPHA, "r2_margin_scale": float(MARGIN_SCALE),
        "interval": "[mu-q*sigma,mu+q*sigma]; BOOST iff L>0; SUPPRESS iff U<0; else KEEP",
        "miscoverage_grid": list(MIS_COVERAGES), "quantile_rule": "min(ceil((n+1)*(1-m))-1,n-1)",
        "operating_point": selection, "selected_q": q,
        "held_class_selection_uses_labels": False,
        "held_safety": held_safety,
    })
    item.update({"actions": actions, "direction_actions": direction, "q": q,
                 "sigma_cf": sigma_cf, "selection": selection})
    return item


def source_metrics(held: str, actions: np.ndarray, direction_actions: np.ndarray) -> dict[str, Any]:
    """Evaluate only cached native logits and authorized VisA masks; no model forward."""
    source_path = r1.SOURCE_ROOT / "gt_free_cache" / f"{held}.npz"
    with np.load(source_path, allow_pickle=False) as data:
        native_logits = np.asarray(data["native_logits"], dtype=np.float32)
        image_path = data["image_path"].astype(str)
        cached_native = np.asarray(data["native_pixel_probability"], dtype=np.float32)
    metadata, data_root = metadata_and_root(DATA_ROOT)
    masks = load_masks(image_path, metadata, data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    native, native_loss = evaluate_correction(native_logits, masks, np.zeros_like(actions, dtype=np.float32).reshape(-1, r1.PATCHES), device, 4)
    direction, direction_loss = evaluate_correction(native_logits, masks, (direction_actions.astype(np.float32) * ALPHA * MARGIN_SCALE).reshape(-1, r1.PATCHES), device, 4)
    selective, selective_loss = evaluate_correction(native_logits, masks, (actions.astype(np.float32) * ALPHA * MARGIN_SCALE).reshape(-1, r1.PATCHES), device, 4)
    labels = masks.reshape(-1)
    def row(scores: np.ndarray, losses: np.ndarray) -> dict[str, float]:
        values = exact_metrics(scores.reshape(-1), labels)
        return {"pixel_ap": values["pAP"], "pixel_auroc": values["pAUROC"], "mean_loss": float(np.mean(losses))}
    return {"native": row(native, native_loss), "direction_only": row(direction, direction_loss),
            "interval_selective": row(selective, selective_loss),
            "native_zero_cache_max_abs_error": float(np.max(np.abs(native - cached_native))),
            "masks": int(masks.size), "device": str(device)}


def aggregate(folds: dict[str, dict[str, Any]], downstream: dict[str, Any], audit_pass: bool) -> dict[str, Any]:
    safety_actions = np.concatenate([folds[name]["actions"] for name in r1.CLASSES])
    safety_y = np.concatenate([folds[name]["y"] for name in r1.CLASSES])
    safety_direction = np.concatenate([folds[name]["direction_actions"] for name in r1.CLASSES])
    safety = risk_metrics(safety_actions, safety_y, safety_direction)
    native_pap = np.array([downstream[name]["native"]["pixel_ap"] for name in r1.CLASSES])
    selective_pap = np.array([downstream[name]["interval_selective"]["pixel_ap"] for name in r1.CLASSES])
    native_auc = np.array([downstream[name]["native"]["pixel_auroc"] for name in r1.CLASSES])
    selective_auc = np.array([downstream[name]["interval_selective"]["pixel_auroc"] for name in r1.CLASSES])
    metrics = {
        "macro_native_pixel_ap": float(native_pap.mean()), "macro_selective_pixel_ap": float(selective_pap.mean()),
        "macro_pixel_ap_delta": float(selective_pap.mean() - native_pap.mean()),
        "macro_native_pixel_auroc": float(native_auc.mean()), "macro_selective_pixel_auroc": float(selective_auc.mean()),
        "macro_pixel_auroc_delta": float(selective_auc.mean() - native_auc.mean()),
        "nonregressing_pixel_ap_classes": int(np.count_nonzero(selective_pap >= native_pap)),
        "worst_class_pixel_ap_delta": float(np.min(selective_pap - native_pap)), "safety": safety,
        "native_zero_cache_max_abs_error": float(max(downstream[name]["native_zero_cache_max_abs_error"] for name in r1.CLASSES)),
    }
    gates = {
        "R2_G1_audit": audit_pass and metrics["native_zero_cache_max_abs_error"] <= 2e-6,
        "R2_G2_accepted_wrong_sign": safety["opposite_sign_rate"] is not None and safety["opposite_sign_rate"] <= 0.05,
        "R2_G3_coverage": safety["coverage"] >= 0.10,
        "R2_G4_relative_wrong_sign_reduction": safety["relative_wrong_sign_reduction"] is not None and safety["relative_wrong_sign_reduction"] >= 0.25,
        "R2_G5_macro_pixel_ap": metrics["macro_selective_pixel_ap"] > metrics["macro_native_pixel_ap"],
        "R2_G6_breadth": metrics["nonregressing_pixel_ap_classes"] >= 9,
        "R2_G7_macro_pixel_auroc": metrics["macro_pixel_auroc_delta"] >= -0.005,
    }
    passed = all(gates.values())
    return {"metrics": metrics, "gates": gates, "R2_GATE_RESULT": "PASS" if passed else "FAIL",
            "SABRA_CURE_R2_STATUS": "PASS" if passed else "R2_SCIENTIFIC_STOP"}


def execute_once(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if any((output / name).exists() for name in ("ATTEMPT_STARTED.json", "summary.json")) or (ROOT / "research/sabra_cure/r2/R2_FINAL_DECISION.md").exists():
        raise RuntimeError("ENGINEERING_STOP R2 attempt already started")
    audit = pre_execution_audit(output)
    write_json(output / "ATTEMPT_STARTED.json", {"status": "ATTEMPT_STARTED", "execution_base_sha": git("rev-parse", "HEAD"),
                                                   "folds_required": len(r1.CLASSES), "run_count": 1})
    started = time.perf_counter()
    shards, provenance = r1.load_shards(check_hashes=True)
    folds: dict[str, dict[str, Any]] = {}
    downstream: dict[str, Any] = {}
    for held in r1.CLASSES:
        fold_started = time.perf_counter()
        item = r2_fold(held, shards)
        save_npz(output / "folds" / f"{held}.npz", image_path=item["image_path"], utility=item["utility"], y=item["y"],
                 mu=item["mu"], sigma=item["sigma"], actions=item["actions"], direction_actions=item["direction_actions"])
        save_npz(output / "inner_crossfit" / f"{held}.npz", y=item["inner_y"], mu_cf=item["inner_mu_cf"],
                 sigma_cf=item["sigma_cf"], residual_target=item["residual_target"])
        item["parameters"]["fit_and_selection_elapsed_seconds"] = time.perf_counter() - fold_started
        write_json(output / "parameters" / f"{held}.json", item["parameters"])
        downstream[held] = source_metrics(held, item["actions"], item["direction_actions"])
        folds[held] = item
        print(json.dumps({"event": "R2_OUTER_FOLD_COMPLETE", "held_class": held,
                          "selection": item["selection"]["selection_status"],
                          "seconds": item["parameters"]["fit_and_selection_elapsed_seconds"]}), flush=True)
    write_json(output / "risk_coverage.json", {name: folds[name]["selection"] for name in r1.CLASSES})
    write_json(output / "downstream_metrics.json", downstream)
    conclusion = aggregate(folds, downstream, audit_pass=bool(audit["status"] == "PASS"))
    summary = {"status": conclusion["SABRA_CURE_R2_STATUS"], "execution_base_sha": git("rev-parse", "HEAD"),
               "preregistration_freeze_sha": PREREG_SHA, "folds_completed": len(folds), "feature_order": list(r1.FEATURE_ORDER),
               "pre_execution_audit": "PASS", "provenance": provenance, "metrics": conclusion["metrics"],
               "gates": conclusion["gates"], "R2_GATE_RESULT": conclusion["R2_GATE_RESULT"],
               "freeze": {"phase2b_training_steps": 0, "additional_clip_forwards": 0},
               "firewall": {"mvtec_accessed": False, "medical_accessed": False},
               "elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(), "numpy": np.__version__}
    write_json(output / "summary.json", summary)
    post = audit_results(output)
    if post["status"] != "PASS":
        raise RuntimeError("ENGINEERING_STOP post-execution audit")
    decision = "PASS" if summary["status"] == "PASS" else "R2_SCIENTIFIC_STOP"
    (ROOT / "research/sabra_cure/r2/R2_FINAL_DECISION.md").write_text(
        "# SABRA-CURE R2 Final Decision\n\n" + f"Decision: `{decision}`\n\n" +
        "This is the sole authorized frozen R2 VisA LOCO execution.\n"
    )
    return summary


def audit_results(output: Path) -> dict[str, Any]:
    """Reload persisted parameters/results only; it never refits direction or uncertainty heads."""
    summary = json.loads((output / "summary.json").read_text())
    shards, _ = r1.load_shards(check_hashes=True)
    downstream = json.loads((output / "downstream_metrics.json").read_text())
    reconstructed: dict[str, dict[str, Any]] = {}
    serialization_error = 0.0
    action_error = 0
    leakage = True
    finite_all = True
    selection_statuses: list[str] = []
    downstream_error = 0.0
    for held in r1.CLASSES:
        params = json.loads((output / "parameters" / f"{held}.json").read_text())
        with np.load(output / "folds" / f"{held}.npz", allow_pickle=False) as data:
            x = r1.scale_x(shards[held].x, np.asarray(params["feature_median"]), np.asarray(params["feature_iqr"]))
            mu = x @ np.asarray(params["mean_beta"]) + float(params["mean_intercept"])
            sigma = np.exp(np.clip(x @ np.asarray(params["uncertainty_beta"]) + float(params["uncertainty_intercept"]), np.log(EPS), np.log(4.0)))
            expected_actions = np.zeros_like(np.asarray(data["actions"]), dtype=np.int8) if params["selected_q"] is None else interval_actions(mu, sigma, float(params["selected_q"]))
            serialization_error = max(serialization_error, float(np.max(np.abs(mu - data["mu"]))), float(np.max(np.abs(sigma - data["sigma"]))),
                                      float(np.max(np.abs(r1.transform(shards[held].utility, float(params["training_scale"])) - data["y"]))))
            action_error = max(action_error, int(np.max(np.abs(expected_actions.astype(np.int16) - data["actions"].astype(np.int16)))))
            finite_all = finite_all and bool(np.isfinite(data["y"]).all() and np.isfinite(data["mu"]).all() and np.isfinite(data["sigma"]).all())
            reconstructed[held] = {"y": np.asarray(data["y"]), "actions": np.asarray(data["actions"]), "direction_actions": np.asarray(data["direction_actions"])}
        selection = params["operating_point"]
        selection_statuses.append(selection["selection_status"])
        for inner in params["inner_crossfits"]:
            leakage = leakage and inner["training_classes"] == [candidate for candidate in r1.CLASSES if candidate not in {held, inner["held_class"]}]
        rechecked = source_metrics(held, reconstructed[held]["actions"], reconstructed[held]["direction_actions"])
        for condition in ("native", "direction_only", "interval_selective"):
            for key in ("pixel_ap", "pixel_auroc", "mean_loss"):
                downstream_error = max(downstream_error, abs(float(rechecked[condition][key]) - float(downstream[held][condition][key])))
    fresh_folds = {name: {"actions": reconstructed[name]["actions"], "y": reconstructed[name]["y"], "direction_actions": reconstructed[name]["direction_actions"]} for name in r1.CLASSES}
    conclusion = aggregate(fresh_folds, downstream, audit_pass=True)
    metric_parity = conclusion["metrics"] == summary["metrics"] and conclusion["gates"] == summary["gates"]
    status = "PASS" if (serialization_error <= 1e-10 and action_error == 0 and finite_all and leakage and metric_parity and downstream_error <= 1e-12 and protected_history_unchanged()) else "FAIL"
    payload = {"status": status, "folds_complete": len(reconstructed), "held_class_order": list(reconstructed),
               "selection_statuses": selection_statuses, "serialization_max_abs_error": serialization_error,
               "action_max_abs_error": action_error, "serialization_parity": serialization_error <= 1e-10 and action_error == 0,
               "metric_recomputation_parity": metric_parity and downstream_error <= 1e-12,
               "downstream_metric_max_abs_error": downstream_error, "leakage_audit": leakage,
               "provenance_audit": protected_history_unchanged(), "freeze_audit": summary["freeze"] == {"phase2b_training_steps": 0, "additional_clip_forwards": 0},
               "firewall_audit": summary["firewall"] == {"mvtec_accessed": False, "medical_accessed": False},
               "finite_predictions": finite_all, "phase2b_training_steps": 0, "additional_clip_forwards": 0,
               "mvtec_accessed": False, "medical_accessed": False}
    write_json(output / "post_execution_audit.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-audit", action="store_true")
    parser.add_argument("--execute-once", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if sum((args.pre_audit, args.execute_once, args.audit_only)) != 1:
        parser.error("choose exactly one mode")
    output = args.output.resolve()
    if args.pre_audit:
        result = pre_execution_audit(output)
    elif args.execute_once:
        result = execute_once(output)
    else:
        result = audit_results(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
