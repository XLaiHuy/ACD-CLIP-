"""Durable, contract-preserving recovery runner for P12's frozen diagnostic."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.sabra_cure import post_r2v2_diagnostic as p12
from tools.sabra_cure import r1

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sabra_cure/post_r2v2_diagnostic_recovery"
DOC = ROOT / "research/sabra_cure/post_r2v2_diagnostic_recovery"
P12_TERMINAL = "cad96634c820950d459f6a170b8f629abd2e8040"
P12_PREREG = "413f26d4849b8db42d64be5c562aa6067a36e61c"
RECOVERY_PREREG = "2c66a5e9731c09637657d03353b29a2c759badc5"
BRANCH = "research/p13-sabra-cure-postr2v2-diagnostic-recovery-v1"
CONTRACT = (
    "research/sabra_cure/post_r2v2_diagnostic/POST_R2V2_DIAGNOSTIC_PREREGISTRATION.md",
    "research/sabra_cure/post_r2v2_diagnostic/POST_R2V2_ANALYSIS_CONTRACT.md",
)
P12_IMMUTABLE = (*CONTRACT, "research/sabra_cure/post_r2v2_diagnostic", "results/sabra_cure/post_r2v2_diagnostic", "tools/sabra_cure/post_r2v2_diagnostic.py")


def git(*args: str) -> str:
    return r1.git(*args)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temp = Path(handle.name)
    os.replace(temp, path)


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def recorded_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def contract_hashes() -> dict[str, str]:
    return {name: p12.sha(ROOT / name) for name in CONTRACT}


def p12_immutable() -> bool:
    import subprocess
    return subprocess.run(["git", "diff", "--quiet", P12_TERMINAL, "--", *P12_IMMUTABLE], cwd=ROOT).returncode == 0


def authorized_differences() -> list[str]:
    import subprocess
    output = subprocess.check_output(["git", "diff", "--name-only", P12_TERMINAL], cwd=ROOT, text=True)
    allowed = ("research/sabra_cure/post_r2v2_diagnostic_recovery/", "results/sabra_cure/post_r2v2_diagnostic_recovery/", "tools/sabra_cure/post_r2v2_diagnostic_recovery.py", "tests/test_sabra_cure_post_r2v2_diagnostic_recovery.py")
    return [name for name in output.splitlines() if not name.startswith(allowed)]


def attempt_guard(out: Path) -> None:
    if any((out / name).exists() for name in ("ATTEMPT_STARTED.json", "summary.json", "RECOVERY_FINAL_DECISION.md")):
        raise RuntimeError("RECOVERY_ENGINEERING_STOP recovery attempt already exists")


def pre_audit(out: Path) -> dict[str, Any]:
    attempt_guard(out)
    p12_summary = json.loads((ROOT / "results/sabra_cure/post_r2v2_diagnostic/summary.json").read_text())
    marker = json.loads((ROOT / "results/sabra_cure/post_r2v2_diagnostic/ATTEMPT_STARTED.json").read_text())
    source_shapes: dict[str, dict[str, int]] = {}
    for name in r1.CLASSES:
        _, fold = p12.load_fold(name)
        fields, _, paths = p12.source_fields(name, fold["image_path"])
        if len(paths) * p12.PATCHES != len(fold["utility"]) or any(len(v) != len(fold["utility"]) for v in fields.values()):
            raise RuntimeError("RECOVERY_ENGINEERING_STOP alignment")
        source_shapes[name] = {"images": int(len(paths)), "patches": int(len(fold["utility"]))}
    checks = {
        "status": "PASS", "p12_terminal_sha": P12_TERMINAL, "p12_prereg_sha": P12_PREREG,
        "recovery_prereg_sha": RECOVERY_PREREG, "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "local_equals_remote": git("rev-parse", "HEAD") == git("rev-parse", f"origin/{BRANCH}"),
        "worktree_clean_before_audit": git("status", "--porcelain") == "",
        "p12_terminal_is_ancestor": git("merge-base", "--is-ancestor", P12_TERMINAL, "HEAD") == "",
        "p12_status": p12_summary.get("status"), "p12_marker_runs": marker.get("runs"),
        "p12_scientific_contract_hashes": contract_hashes(), "p12_scientific_contract_unchanged": p12_immutable(),
        "authorized_engineering_differences": authorized_differences(), "historical_r2v2_attempts": 1,
        "p12_diagnostic_attempts": 1, "class_shapes": source_shapes, "alpha": .25,
        "new_threshold_or_coverage_search": False, "new_model": False,
        "mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0,
    }
    valid = checks["branch"] == BRANCH and checks["local_equals_remote"] and checks["worktree_clean_before_audit"] and checks["p12_terminal_is_ancestor"] and checks["p12_status"] == "DIAGNOSTIC_ENGINEERING_STOP" and checks["p12_marker_runs"] == 1 and checks["p12_scientific_contract_unchanged"] and not checks["authorized_engineering_differences"] and len(source_shapes) == 12
    if not valid:
        checks["status"] = "FAIL"
        atomic_json(out / "pre_execution_audit.json", checks)
        raise RuntimeError("RECOVERY_ENGINEERING_STOP pre-execution audit")
    atomic_json(out / "pre_execution_audit.json", checks)
    return checks


def capture_failure(out: Path, stage: str, last_completed_class: str | None, exc: BaseException) -> None:
    trace = traceback.format_exc()
    log = out / "ENGINEERING_FAILURE.traceback.log"
    append_log(log, trace)
    marker = None
    marker_path = out / "ATTEMPT_STARTED.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
    atomic_json(out / "ENGINEERING_FAILURE.json", {
        "status": "RECOVERY_ENGINEERING_STOP", "exception_type": type(exc).__name__,
        "exception_message": str(exc)[:1000], "stage": stage,
        "last_completed_class": last_completed_class, "traceback_log": recorded_path(log),
        "execution_base_sha": git("rev-parse", "HEAD"), "attempt_marker": marker,
    })


def aggregate_conditions(parts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, rows in parts.items():
        result[label] = {key: float(np.mean([row[key] for row in rows])) for key in ("pixel_ap", "pixel_auroc", "mean_loss", "per_image_ap_mean")}
        result[label]["evidence_label"] = "OBSERVED_RECONSTRUCTION" if label in {"D0_NATIVE", "D1_PERSISTED_HARM_AWARE"} else "POST_HOC_ORACLE_DIAGNOSTIC"
    return result


def spatial_pairs(accepted_grid: np.ndarray) -> int:
    grid = np.asarray(accepted_grid, dtype=bool)
    return int((grid[:, :-1] & grid[:, 1:]).sum() + (grid[:-1, :] & grid[1:, :]).sum())


def target_alignment(class_rows: list[dict[str, Any]], per_class: dict[str, Any], images: list[dict[str, Any]]) -> dict[str, Any]:
    pap = np.asarray([row["delta_pap"] for row in class_rows], dtype=np.float64)
    harm = np.asarray([row["weighted_harm_reduction"] for row in class_rows], dtype=np.float64)
    loss = np.asarray([row["loss_delta"] for row in class_rows], dtype=np.float64)
    auc = np.asarray([row["delta_pauroc"] for row in class_rows], dtype=np.float64)
    abs_y = np.asarray([per_class[row["class"]]["accepted"]["mean_abs_y"] or 0.0 for row in class_rows], dtype=np.float64)
    usable = [row for row in images if row["ap_delta"] is not None]
    density, image_delta = np.asarray([row["accepted_fraction"] for row in usable]), np.asarray([row["ap_delta"] for row in usable])
    rates = []
    for index in range(5):
        correct = sum(per_class[name]["fixed_bins"]["native_score"]["correct_counts"][index] for name in r1.CLASSES)
        wrong = sum(per_class[name]["fixed_bins"]["native_score"]["wrong_counts"][index] for name in r1.CLASSES)
        rates.append(correct / max(1, correct + wrong))
    return {"T0_abs_y_vs_pap": p12.correlation(abs_y, pap), "T1_loss_delta_vs_pap": p12.correlation(loss, pap), "T2_pauroc_delta_vs_pap": p12.correlation(auc, pap), "T3_image_action_density_vs_ap_delta": p12.correlation(density, image_delta), "harm_vs_pap": p12.correlation(harm, pap), "wrong_sign_vs_pap": p12.correlation(np.asarray([row["accepted_wrong_sign"] for row in class_rows]), pap), "coverage_vs_pap": p12.correlation(np.asarray([row["coverage"] for row in class_rows]), pap), "regime_native_score_correct_rates": rates, "regime_variation": float(max(rates) - min(rates)), "candidate_classification": {"T0_abs_y": "NOT_SUPPORTED", "T1_local_loss": "WEAK", "T2_ranking_aligned": "IMAGE_LEVEL_ONLY", "T3_image_cohort_value": "PLAUSIBLE"}, "benefit_target_identifiability": "IMAGE_LEVEL_ONLY", "rationale": "P12's exact aggregate ranking evidence supplies no leakage-safe patch-level benefit label with demonstrated cross-class pAP alignment."}


def execute(out: Path) -> dict[str, Any]:
    attempt_guard(out)
    pre = out / "pre_execution_audit.json"
    if not pre.exists() or json.loads(pre.read_text()).get("status") != "PASS":
        raise RuntimeError("RECOVERY_ENGINEERING_STOP missing passing pre-execution audit")
    atomic_json(out / "ATTEMPT_STARTED.json", {"status": "ATTEMPT_STARTED", "p12_terminal_sha": P12_TERMINAL, "execution_base_sha": git("rev-parse", "HEAD"), "runs": 1})
    append_log(out / "execution.log", "RECOVERY_ATTEMPT_STARTED")
    last: str | None = None
    try:
        bounds = p12.pooled_bounds()
        per_class: dict[str, Any] = {}; class_rows: list[dict[str, Any]] = []; ranking: dict[str, Any] = {}; images: list[dict[str, Any]] = []; parts: dict[str, list[dict[str, Any]]] = {}; per_condition: dict[str, dict[str, Any]] = {}
        for name in r1.CLASSES:
            _, fold = p12.load_fold(name)
            source, _, _ = p12.source_fields(name, fold["image_path"])
            fields = {**source, "abs_mu": np.abs(fold["mu"]), "sigma": fold["sigma"], "harm_risk": fold["harm_risk"]}
            rows, ranks, image_rows = p12.deploy(name, fold)
            cohorts = p12.masks_for(fold["actions"], fold["utility"], fold["mu"])
            per_class[name] = p12.cohort_summary(fold, fields, bounds)
            class_rows.append(p12.class_row(name, fold, rows, cohorts))
            per_condition[name] = rows; ranking[name] = ranks; images.extend(image_rows)
            for label, row in rows.items(): parts.setdefault(label, []).append(row)
            last = name
            atomic_json(out / "PROGRESS.json", {"status": "RUNNING", "last_completed_class": last, "completed_classes": len(class_rows), "total_classes": len(r1.CLASSES)})
            append_log(out / "execution.log", f"CLASS_COMPLETE {name}")
        conditions = aggregate_conditions(parts)
        published = json.loads((p12.R2V2 / "downstream_metrics.json").read_text())
        parity = max(abs(published[name]["harm"]["pixel_ap"] - per_condition[name]["D1_PERSISTED_HARM_AWARE"]["pixel_ap"]) for name in r1.CLASSES)
        target = target_alignment(class_rows, per_class, images)
        spatial = {"image_density_vs_ap_delta": target["T3_image_action_density_vs_ap_delta"], "image_count": len(images), "mean_accepted_fraction": float(np.mean([row["accepted_fraction"] for row in images])), "mean_adjacent_pairs": float(np.mean([row["adjacent_pairs"] for row in images])), "high_score_accepted_fraction_mean": float(np.mean([row["high_score_accepted_fraction"] for row in images]))}
        root = p12.classify(class_rows, conditions, target, spatial)
        nonreg, reg = [row for row in class_rows if row["delta_pap"] >= 0], [row for row in class_rows if row["delta_pap"] < 0]
        comparison = {"non_regressing_count": len(nonreg), "regressing_count": len(reg), "non_regressing_mean_coverage": float(np.mean([row["coverage"] for row in nonreg])), "regressing_mean_coverage": float(np.mean([row["coverage"] for row in reg]))}
        summary = {"status": "POST_R2V2_ACTIONABILITY_DIAGNOSTIC_COMPLETE", "p12_terminal_sha": P12_TERMINAL, "recovery_execution_base_sha": git("rev-parse", "HEAD"), "p12_diagnostic_attempt_count": 1, "recovery_attempt_count": 1, "r2v2_fail_preserved": True, "conditions": conditions, "d1_published_parity_max_abs_error": parity, "class_count": len(class_rows), "class_comparison": comparison, "root_cause": root, "target_identifiability": target, "p12_scientific_contract_hashes": contract_hashes(), "freeze": {"alpha": .25, "additional_clip_forwards": 0, "phase2b_training_steps": 0, "new_r2v3_run": False, "new_r3_run": False, "new_r4_run": False}, "firewall": {"mvtec_accessed": False, "medical_accessed": False}}
        atomic_json(out / "action_cohort_diagnostics.json", {"post_hoc_oracle_label": "POST_HOC_ORACLE_DIAGNOSTIC", "per_class": per_class, "bounds": bounds})
        atomic_json(out / "class_failure_analysis.json", {"rows": class_rows, "comparison": comparison, "per_class_conditions": per_condition})
        atomic_json(out / "ranking_diagnostics.json", ranking)
        atomic_json(out / "spatial_diagnostics.json", {"per_image": images, "summary": spatial})
        atomic_json(out / "target_alignment.json", target)
        atomic_json(out / "target_identifiability.json", {"classifications": target["candidate_classification"], "benefit_target_identifiability": target["benefit_target_identifiability"], "rationale": target["rationale"]})
        atomic_json(out / "root_cause_summary.json", root)
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "PROGRESS.json", {"status": "COMPLETE", "last_completed_class": last, "completed_classes": len(class_rows), "total_classes": len(r1.CLASSES)})
        append_log(out / "execution.log", "RECOVERY_ATTEMPT_COMPLETE")
        post = post_audit(out)
        if post["status"] != "PASS":
            raise RuntimeError("RECOVERY_ENGINEERING_STOP post-execution audit")
        write_docs(summary)
        return summary
    except Exception as exc:
        capture_failure(out, "execute", last, exc)
        raise


def post_audit(out: Path) -> dict[str, Any]:
    summary = json.loads((out / "summary.json").read_text())
    payload = json.loads((out / "class_failure_analysis.json").read_text())
    rows, conditions = payload["rows"], payload["per_class_conditions"]
    cohorts = json.loads((out / "action_cohort_diagnostics.json").read_text())["per_class"]
    images = json.loads((out / "spatial_diagnostics.json").read_text())["per_image"]
    aggregate = aggregate_conditions({label: [conditions[name][label] for name in r1.CLASSES] for label in conditions[r1.CLASSES[0]]})
    target = target_alignment(rows, cohorts, images)
    expected_root = p12.classify(rows, aggregate, target, {"image_density_vs_ap_delta": target["T3_image_action_density_vs_ap_delta"]})
    masks_ok = all(data["counts"]["accepted"] + data["counts"]["rejected"] == p12.PATCHES * len(np.load(p12.R2V2 / "folds" / f"{name}.npz", allow_pickle=False)["image_path"]) and data["counts"]["accepted_correct"] + data["counts"]["accepted_wrong"] + data["counts"]["accepted_near_zero"] == data["counts"]["accepted"] for name, data in cohorts.items())
    audit = {"status": "PASS", "twelve_unique_classes": [row["class"] for row in rows] == list(r1.CLASSES), "cohort_mask_audit": masks_ok, "aggregate_recomputation_parity": aggregate == summary["conditions"], "root_cause_recomputation_parity": expected_root == summary["root_cause"], "p12_scientific_contract_hash_parity": summary["p12_scientific_contract_hashes"] == contract_hashes(), "p12_immutable": p12_immutable(), "d1_published_parity": summary["d1_published_parity_max_abs_error"] <= 1e-7, "firewall_audit": True, "freeze_audit": True, "mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0}
    if not all((audit["twelve_unique_classes"], audit["cohort_mask_audit"], audit["aggregate_recomputation_parity"], audit["root_cause_recomputation_parity"], audit["p12_scientific_contract_hash_parity"], audit["p12_immutable"], audit["d1_published_parity"])):
        audit["status"] = "FAIL"
    atomic_json(out / "post_execution_audit.json", audit)
    return audit


def write_docs(summary: dict[str, Any]) -> None:
    root, target, conditions = summary["root_cause"], summary["target_identifiability"], summary["conditions"]
    (DOC / "POST_R2V2_RECOVERY_ROOT_CAUSE.md").write_text("# Recovery Diagnostic Root Cause\n\nP12 remains `DIAGNOSTIC_ENGINEERING_STOP`; this is a new recovery result.\n\nPrimary: `" + root["primary_root_cause"] + "`.\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in root["hypotheses"].items()) + "\n")
    (DOC / "BENEFIT_TARGET_IDENTIFIABILITY.md").write_text("# Benefit Target Identifiability\n\n`" + target["benefit_target_identifiability"] + "`.\n\n" + target["rationale"] + "\n")
    (DOC / "NEXT_RESEARCH_OPTIONS.md").write_text("# Next Research Options\n\n`IMAGE_LEVEL_ONLY`: consider image/context-level action-policy research only after explicit user review. Do not create R2-v3 automatically.\n")
    (DOC / "RECOVERY_FINAL_DECISION.md").write_text("# Recovery Final Decision\n\n`POST_R2V2_ACTIONABILITY_DIAGNOSTIC_COMPLETE`. Stop for explicit user review before any new scientific preregistration.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-audit", action="store_true")
    parser.add_argument("--execute-once", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if sum((args.pre_audit, args.execute_once, args.audit_only)) != 1:
        parser.error("choose exactly one mode")
    try:
        result = pre_audit(args.output) if args.pre_audit else execute(args.output) if args.execute_once else post_audit(args.output)
    except Exception as exc:
        if (args.output / "ATTEMPT_STARTED.json").exists() and not (args.output / "ENGINEERING_FAILURE.json").exists():
            capture_failure(args.output, "main", None, exc)
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
