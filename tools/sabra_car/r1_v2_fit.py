"""Execute the single SABRA-CAR R1 Scalable Solver Protocol v2 fit."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from tools.sabra_car.r1_common import EXPECTED_CLASSES, write_json
from tools.sabra_car.r1_fit import (
    V2_MAX_ITER,
    V2_SOLVER,
    FoldConvergenceError,
    _git_head,
    run,
)

ROOT = Path(__file__).resolve().parents[2]
V2_PROTOCOL = "SABRA-CAR R1 Scalable Solver Protocol v2"
V2_PREREG_SHA = "fefeab35b58d4aa6be4ceddfaaa0994fa456d180"
PARENT_TERMINAL_SHA = "782b8b81aa5b03c88a4417e5d7106e19ceff83ce"
ORIGINAL_S0_SHA = "08ca99ff69d6d85184f5d145830876befb413628"
V2_OUTPUT = ROOT / "results/sabra_car/r1_v2_newton_cholesky"
V1_STOP = ROOT / "results/sabra_car/r1_recovery_v1/COMPUTATIONAL_STOP.json"
ORIGINAL_FAILURE = ROOT / "results/sabra_car/r1/FIT_FAILED.json"


def v2_arguments(output: Path = V2_OUTPUT) -> argparse.Namespace:
    return argparse.Namespace(
        source_root=Path(
            "/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0/sabra_source"
        ),
        trust_root=ROOT / "runs/phase5/sabra/TRUST_V2_DEVELOPMENT",
        utility_root=ROOT / "results/sabra_car/r0/utility",
        output=output.resolve(),
        skip_hashes=False,
        max_iter=V2_MAX_ITER,
        solver=V2_SOLVER,
        recovery_protocol=V2_PROTOCOL,
        protocol_prereg_sha=V2_PREREG_SHA,
        emit_progress=True,
    )


def v2_identity() -> dict[str, Any]:
    return {
        "protocol": V2_PROTOCOL,
        "protocol_prereg_sha": V2_PREREG_SHA,
        "parent_terminal_sha": PARENT_TERMINAL_SHA,
        "original_s0_sha": ORIGINAL_S0_SHA,
        "r1_v1_status": "COMPUTATIONAL_STOP",
        "r1_v1_artifact": str(V1_STOP.relative_to(ROOT)),
        "original_r1_artifact": str(ORIGINAL_FAILURE.relative_to(ROOT)),
        "solver": V2_SOLVER,
        "max_iter": V2_MAX_ITER,
        "C": 1.0,
        "penalty": "l2",
        "class_weight": "balanced",
        "tol": 1e-4,
        "fit_intercept": True,
        "random_state": 0,
        "max_v2_attempts": 1,
        "medical_reads": 0,
        "mvtec_reads": 0,
        "phase2b_training_steps": 0,
    }


def completed_folds(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_name in EXPECTED_CLASSES:
        path = output / "parameters" / f"{class_name}.json"
        if path.is_file():
            rows.append(json.loads(path.read_text()))
    return rows


def ensure_attempt_available(output: Path) -> None:
    if output.exists():
        raise RuntimeError(f"R1-v2 attempt already exists or was started: {output}")
    if not ORIGINAL_FAILURE.is_file() or not V1_STOP.is_file():
        raise RuntimeError("published R1/R1-v1 terminal evidence is missing")


def execute_once(output: Path = V2_OUTPUT) -> dict[str, Any]:
    output = output.resolve()
    ensure_attempt_available(output)
    output.mkdir(parents=True)
    identity = v2_identity() | {
        "status": "ATTEMPT_STARTED",
        "git_head": _git_head(),
        "folds_required": len(EXPECTED_CLASSES),
        "folds_converged": 0,
        "r1_predictions_produced": False,
        "r1_metrics_produced": False,
        "r1_gate_evaluated": False,
    }
    write_json(output / "ATTEMPT_STARTED.json", identity)
    try:
        selection = run(v2_arguments(output))
    except FoldConvergenceError as error:
        folds = completed_folds(output)
        result = identity | {
            "status": "COMPUTATIONAL_STOP",
            "r1_v2_execution_status": "COMPUTATIONAL_STOP",
            "folds_converged": len(folds),
            "completed_folds": folds,
            "failed_fold": error.details,
            "fold_predictions_complete": False,
            "r1_predictions_produced": False,
            "r1_metrics_produced": False,
            "r1_gate_evaluated": False,
            "next_stage": "NONE",
        }
        write_json(output / "COMPUTATIONAL_STOP.json", result)
        return result
    except Exception as error:
        folds = completed_folds(output)
        result = identity | {
            "status": "ENGINEERING_FAILURE",
            "r1_v2_execution_status": "ENGINEERING_FAILURE",
            "folds_converged": len(folds),
            "completed_folds": folds,
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "r1_predictions_produced": False,
            "r1_metrics_produced": False,
            "r1_gate_evaluated": False,
            "next_stage": "NONE",
        }
        write_json(output / "ENGINEERING_FAILURE.json", result)
        raise
    folds = completed_folds(output)
    if len(folds) != len(EXPECTED_CLASSES):
        raise RuntimeError("R1-v2 fit returned without twelve completed folds")
    result = identity | {
        "status": "FIT_COMPLETE",
        "r1_v2_execution_status": "READY_FOR_ORIGINAL_R1_EVALUATION",
        "folds_converged": len(folds),
        "completed_folds": folds,
        "fold_predictions_complete": True,
        "r1_predictions_produced": True,
        "risk_selection_status": selection["status"],
        "selected_threshold": selection["selected_threshold"],
        "r1_metrics_produced": True,
        "r1_gate_evaluated": False,
        "next_stage": "ORIGINAL_R1_EVALUATION",
    }
    write_json(output / "FIT_COMPLETE.json", result)
    return result


def main() -> None:
    print(json.dumps(execute_once(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
