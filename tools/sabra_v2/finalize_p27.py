"""Create compact terminal P27 evidence from a completed immutable runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.sabra.data import EXPECTED_VISA_CLASSES
from tools.sabra_v2.train_region_distill import _sha256


STATUSES = ("P27_SUPPORTED", "P27_MIXED", "P27_NOT_SUPPORTED", "P27_ENGINEERING_STOP")


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = json.loads(args.runtime_result.read_text())
    if result.get("schema_version") != "P27_COMPLETE_SCIENTIFIC_RESULT_V1":
        raise RuntimeError("terminalization requires a complete P27 scientific result")
    if result.get("post_audit", {}).get("status") != "PASS":
        raise RuntimeError("terminalization requires a passing post-run audit")
    folds = result.get("fold_records", [])
    if [fold.get("held_class") for fold in folds] != list(EXPECTED_VISA_CLASSES):
        raise RuntimeError("terminalization requires exactly the frozen 12-fold order")
    benchmark = json.loads(args.benchmark.read_text())
    if not benchmark.get("parity", {}).get("gradient_and_step_within_tolerance"):
        raise RuntimeError("terminalization requires passing cache parity")
    args.output.mkdir(parents=True, exist_ok=False)
    fold_metrics: dict[str, Any] = {}
    for fold in folds:
        held = fold["held_class"]
        metric_path = Path(fold["score"]["result"]["metrics_path"])
        payload = json.loads(metric_path.read_text())
        if payload["prediction_sha256"] != fold["prediction_sha256_before_and_after_score"]:
            raise RuntimeError(f"prediction/metric provenance mismatch for {held}")
        fold_metrics[held] = payload
    summary = {
        "schema_version": "P27_TERMINAL_EVIDENCE_V1",
        "final_status": args.status,
        "observed": result["aggregate"],
        "interpretation": args.interpretation,
        "engineering": {
            "frozen_parent_sha": "1151373f2c4968268f52cdc3e538c7ebcef7b8f0",
            "recovery_branch": "research/p27-cache-performance-recovery-v1",
            "scientific_execution_base_sha": result["attempt"]["execution_base_sha"],
            "cache_parity": benchmark["parity"],
            "cache": benchmark["cache"],
            "timing": benchmark["timing"],
            "projection": benchmark["projection"],
            "actual_scientific_wall_seconds": result["actual_scientific_wall_seconds"],
        },
        "attempt": result["attempt"],
        "post_audit": result["post_audit"],
        "fold_metrics": fold_metrics,
        "runtime_result_sha256": _sha256(args.runtime_result),
    }
    summary_path = args.output / "P27_TERMINAL_EVIDENCE.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output / "P27_ATTEMPT_STARTED.json").write_text(json.dumps(result["attempt"], indent=2, sort_keys=True) + "\n")
    (args.output / "P27_POST_AUDIT.json").write_text(json.dumps(result["post_audit"], indent=2, sort_keys=True) + "\n")
    (args.output / "P27_FOLD_METRICS.json").write_text(json.dumps(fold_metrics, indent=2, sort_keys=True) + "\n")
    macro = result["aggregate"]["macro"]
    breadth = result["aggregate"]["breadth"]
    report = [
        "# P27 Final Execution Report", "", "## Observed", "",
        f"- Native macro pAP: `{macro['native_pAP']}`",
        f"- P27 macro pAP: `{macro['p27_pAP']}`",
        f"- Delta pAP: `{macro['delta_pAP']}`",
        f"- Native macro pAUROC: `{macro['native_pAUROC']}`",
        f"- P27 macro pAUROC: `{macro['p27_pAUROC']}`",
        f"- Delta pAUROC: `{macro['delta_pAUROC']}`",
        f"- Improving / non-regressing / regressing pAP classes: `{breadth['improving_pAP']} / {breadth['non_regressing_pAP']} / {breadth['regressing_pAP']}`",
        "", "## Interpretation", "", args.interpretation, "", f"Final status: `{args.status}`", "",
        "Full engineering, per-class, attempt, and audit evidence is in `P27_TERMINAL_EVIDENCE.json`.",
    ]
    (args.output / "P27_FINAL_EXECUTION_REPORT.md").write_text("\n".join(report) + "\n")
    return {"status": args.status, "terminal_evidence": str(summary_path), "files": sorted(path.name for path in args.output.iterdir())}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-result", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", choices=STATUSES, required=True)
    parser.add_argument("--interpretation", required=True)
    return parser


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
