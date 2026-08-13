#!/usr/bin/env python3
"""Offline terminal visual-pivot attribution after the mature P2B K1-gate audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from audit_p4_k1_oracle_utility import _git_sha, _sha256


def _gate_metrics(report: dict) -> dict:
    variant = report["variants"]["MATURE_P2B_DIRECT_GATE"]
    bootstrap = report["image_level_gain_bootstrap"]["MATURE_P2B_DIRECT_GATE"]
    return {
        "normal_gain": variant["gain_base_minus_final"]["normal"],
        "anomaly_gain": variant["gain_base_minus_final"]["anomaly"],
        "normal_image_bootstrap": bootstrap["normal"],
        "anomaly_image_bootstrap": bootstrap["anomaly"],
        "normal_gate_mean": report["mature_p2b_gate"]["normal"]["mean"],
        "anomaly_gate_mean": report["mature_p2b_gate"]["anomaly"]["mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("runs/phase4/k1/stage1_9a/K1_MATURE_PHASE2B_DIRECT_GATE_AUDIT.json"))
    parser.add_argument("--centered", type=Path, default=Path("runs/phase4/k1/stage1_9a/K1_MATURE_PHASE2B_CENTERED_GATE_AUDIT.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase4/k1/stage1_9a/K1_VISUAL_ONLY_PIVOT_AUDIT.json"))
    args = parser.parse_args()
    raw_path, centered_path, output_path = args.raw.resolve(), args.centered.resolve(), args.output.resolve()
    raw, centered = json.loads(raw_path.read_text()), json.loads(centered_path.read_text())
    localization = raw["localization"]
    class_auc = {name: entry["p2b_patch_anomaly_auroc"]["mean"] for name, entry in raw["per_class"].items()}
    if any(value is None for value in class_auc.values()):
        raise RuntimeError("The fixed manifest must retain meaningful per-class anomaly localization.")
    raw_gate, centered_gate = _gate_metrics(raw), _gate_metrics(centered)
    if not (raw_gate["normal_gain"] < 0 and centered_gate["normal_gain"] < 0):
        raise RuntimeError("This terminal pivot applies only after both permitted gates fail strict Normal safety.")
    report = {
        "decision": "PHASE4_VISUAL_PIVOT_CANDIDATE_IDENTIFIED",
        "terminal_text_semantic_state": "CALIBRATED_P2B_GATE_FAIL; PHASE4_TEXT_SEMANTIC_LINE_STOPPED",
        "provenance": {
            "repo_sha": _git_sha(), "script_sha256": _sha256(Path(__file__).resolve()),
            "raw_audit_path": str(raw_path.relative_to(REPO_ROOT)), "raw_audit_sha256": _sha256(raw_path),
            "centered_audit_path": str(centered_path.relative_to(REPO_ROOT)), "centered_audit_sha256": _sha256(centered_path),
            "optimizer_steps": 0,
        },
        "evidence": {
            "mature_phase2b_patch_anomaly_auroc": localization["patch_anomaly_auroc"],
            "mature_phase2b_per_image_auc": localization["per_image_auroc"],
            "mature_phase2b_per_class_auc_min": min(class_auc.values()),
            "mature_phase2b_per_class_auc": class_auc,
            "mature_phase2b_vs_k1_utility_anomaly": raw["k1_utility_alignment"]["mature_p2b"]["anomaly"],
            "raw_soft_gate": raw_gate,
            "one_allowed_centered_calibration": centered_gate,
        },
        "selected_future_candidate": {
            "name": "VisualAD-style visual-only adaptation",
            "why_this_one": "The mature Phase2B visual localization is strong and consistent, yet every permitted text-semantic residual gate remains strictly unsafe on Normal patches. The supported next question is therefore whether a visual-side adaptation can retain local evidence without a dynamic-text correction.",
            "not_implemented": True,
        },
        "excluded_without_evidence": {
            "foreground_background_separation": "No foreground/background annotations or audit evidence were available.",
            "frequency_texture_adaptation": "No frequency or texture failure evidence was measured.",
            "additional_dynamic_text_capacity": "Disallowed after strict Normal-safety failure; K2/OT/Router/ACT are not authorized.",
        },
        "next_action": "Stop. Do not train the visual candidate without a new explicit authorization.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": report["decision"], "candidate": report["selected_future_candidate"]["name"], "raw_normal_gain": raw_gate["normal_gain"], "centered_normal_gain": centered_gate["normal_gain"], "localization_auroc": localization["patch_anomaly_auroc"]}, indent=2))


if __name__ == "__main__":
    main()
