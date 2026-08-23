"""Execute the single preregistered SABRA-CAR R1 Recovery Protocol v1 fit."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from tools.sabra_car.r1_common import write_json
from tools.sabra_car.r1_fit import (
    ORIGINAL_MAX_ITER,
    RECOVERY_MAX_ITER,
    RECOVERY_PROTOCOL,
    FoldConvergenceError,
    _git_head,
    run,
)

ROOT = Path(__file__).resolve().parents[2]
RECOVERY_PREREG_SHA = "6f8fed381838a0221bb5289fb23d2243a6b5f0ef"
ORIGINAL_S0_SHA = "08ca99ff69d6d85184f5d145830876befb413628"
RECOVERY_OUTPUT = ROOT / "results/sabra_car/r1_recovery_v1"
ORIGINAL_FAILURE = ROOT / "results/sabra_car/r1/FIT_FAILED.json"


def recovery_arguments(output: Path = RECOVERY_OUTPUT) -> argparse.Namespace:
    return argparse.Namespace(
        source_root=Path(
            "/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0/sabra_source"
        ),
        trust_root=ROOT / "runs/phase5/sabra/TRUST_V2_DEVELOPMENT",
        utility_root=ROOT / "results/sabra_car/r0/utility",
        output=output.resolve(),
        skip_hashes=False,
        max_iter=RECOVERY_MAX_ITER,
        recovery_protocol=RECOVERY_PROTOCOL,
    )


def recovery_identity() -> dict[str, Any]:
    return {
        "recovery_protocol": RECOVERY_PROTOCOL,
        "recovery_prereg_sha": RECOVERY_PREREG_SHA,
        "original_s0_sha": ORIGINAL_S0_SHA,
        "original_r1_status": "INCONCLUSIVE_SOLVER_FAILURE_BEFORE_SCIENTIFIC_RESULT",
        "original_failure_artifact": str(ORIGINAL_FAILURE.relative_to(ROOT)),
        "original_max_iter": ORIGINAL_MAX_ITER,
        "recovery_max_iter": RECOVERY_MAX_ITER,
        "max_recovery_attempts": 1,
        "solver": "lbfgs",
        "tol": 1e-4,
        "C": 1.0,
        "penalty": "l2",
        "class_weight": "balanced",
        "random_state": 0,
        "medical_reads": 0,
        "mvtec_reads": 0,
        "phase2b_training_steps": 0,
    }


def ensure_attempt_available(output: Path) -> None:
    if output.exists():
        raise RuntimeError(
            f"R1 Recovery v1 attempt already exists or was started: {output}"
        )
    if not ORIGINAL_FAILURE.is_file():
        raise RuntimeError("original R1 failure evidence is missing")


def execute_once(output: Path = RECOVERY_OUTPUT) -> dict[str, Any]:
    output = output.resolve()
    ensure_attempt_available(output)
    output.mkdir(parents=True)
    identity = recovery_identity() | {
        "status": "ATTEMPT_STARTED",
        "git_head": _git_head(),
        "r1_predictions_produced": False,
        "r1_metrics_produced": False,
        "r1_gate_evaluated": False,
    }
    write_json(output / "ATTEMPT_STARTED.json", identity)
    try:
        selection = run(recovery_arguments(output))
    except FoldConvergenceError as error:
        result = identity | {
            "status": "COMPUTATIONAL_STOP",
            "r1_recovery_status": "COMPUTATIONAL_STOP",
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
        result = identity | {
            "status": "ENGINEERING_FAILURE",
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
    result = identity | {
        "status": "FIT_COMPLETE",
        "r1_recovery_status": "READY_FOR_ORIGINAL_R1_EVALUATION",
        "fold_predictions_complete": True,
        "r1_predictions_produced": True,
        "risk_selection_status": selection["status"],
        "selected_threshold": selection["selected_threshold"],
        "r1_metrics_produced": True,
        "r1_gate_evaluated": False,
        "next_stage": "ORIGINAL_R1_EVALUATION",
    }
    write_json(output / "RECOVERY_FIT_COMPLETE.json", result)
    return result


def main() -> None:
    print(json.dumps(execute_once(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
