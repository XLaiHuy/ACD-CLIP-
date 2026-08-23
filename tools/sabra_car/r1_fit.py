"""Fit preregistered SABRA-CAR R1 LOCO multinomial action predictors."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from tools.sabra_car.r1_common import (
    EXPECTED_CLASSES,
    FEATURE_ORDER,
    apply_robust_scaler,
    fit_robust_scaler,
    load_r1_shards,
    select_threshold,
    threshold_landscape,
    write_csv,
    write_json,
)

ORIGINAL_MAX_ITER = 1000
RECOVERY_MAX_ITER = 5000
RECOVERY_PROTOCOL = "SABRA-CAR R1 Recovery Protocol v1"
ROOT = Path(__file__).resolve().parents[2]


class FoldConvergenceError(RuntimeError):
    def __init__(self, details: dict[str, Any]) -> None:
        super().__init__(
            f"R1 fold exhausted max_iter: {details['held_out_class']} "
            f"at {details['max_iter']} iterations"
        )
        self.details = details


def estimator(max_iter: int = ORIGINAL_MAX_ITER) -> LogisticRegression:
    if max_iter not in (ORIGINAL_MAX_ITER, RECOVERY_MAX_ITER):
        raise ValueError(f"unauthorized R1 max_iter: {max_iter}")
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        fit_intercept=True,
        class_weight="balanced",
        random_state=0,
        solver="lbfgs",
        max_iter=max_iter,
        multi_class="multinomial",
        tol=1e-4,
    )


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def recovery_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "recovery_protocol", None) is None:
        return {}
    return {
        "recovery_protocol": args.recovery_protocol,
        "original_r1_status": "INCONCLUSIVE_SOLVER_FAILURE_BEFORE_SCIENTIFIC_RESULT",
        "original_max_iter": ORIGINAL_MAX_ITER,
        "recovery_max_iter": args.max_iter,
        "max_recovery_attempts": 1,
    }


def valid_existing_fold(path: Path, image_path: np.ndarray, oracle: np.ndarray) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as data:
        return bool(
            data["probability"].shape == oracle.shape + (3,)
            and np.array_equal(data["image_path"].astype(str), image_path.astype(str))
            and np.array_equal(data["oracle_action"].astype(np.int8), oracle)
            and tuple(data["classes"].astype(np.int8).tolist()) == (-1, 0, 1)
            and np.isfinite(data["probability"]).all()
            and np.allclose(data["probability"].sum(-1), 1.0, atol=1e-6, rtol=0.0)
        )


def fit_fold(
    held_out: str,
    shards: dict[str, Any],
    output: Path,
    max_iter: int = ORIGINAL_MAX_ITER,
) -> dict[str, Any]:
    train_names = [name for name in EXPECTED_CLASSES if name != held_out]
    train_features = np.concatenate(
        [shards[name].features.reshape(-1, len(FEATURE_ORDER)) for name in train_names], axis=0
    )
    train_labels = np.concatenate(
        [shards[name].oracle_action.reshape(-1) for name in train_names], axis=0
    )
    median, iqr = fit_robust_scaler(train_features)
    standardized_train = apply_robust_scaler(train_features, median, iqr)
    held = shards[held_out]
    standardized_test = apply_robust_scaler(
        held.features.reshape(-1, len(FEATURE_ORDER)), median, iqr
    )
    model = estimator(max_iter)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(standardized_train, train_labels)
    elapsed = time.perf_counter() - started
    convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    n_iter = model.n_iter_.tolist()
    if convergence:
        details = {
            "held_out_class": held_out,
            "classes": model.classes_.tolist(),
            "n_iter": n_iter,
            "max_iter": max_iter,
            "fit_elapsed_seconds": elapsed,
            "warnings": [str(item.message) for item in caught],
        }
        if np.max(model.n_iter_) >= max_iter:
            raise FoldConvergenceError(details)
        raise RuntimeError(
            f"R1 fold emitted a convergence warning before max_iter: {details}"
        )
    if tuple(model.classes_.tolist()) != (-1, 0, 1):
        raise RuntimeError(
            f"R1 fold class contract failed: held_out={held_out} "
            f"classes={model.classes_.tolist()}"
        )
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise RuntimeError(f"non-finite R1 fitted parameters: {held_out}")
    probability = model.predict_proba(standardized_test).astype(np.float32)
    probability = probability.reshape(held.oracle_action.shape + (3,))
    if not np.isfinite(probability).all() or not np.allclose(
        probability.sum(-1), 1.0, atol=1e-6, rtol=0.0
    ):
        raise RuntimeError(f"invalid R1 probabilities: {held_out}")
    fold_path = output / "folds" / f"{held_out}.npz"
    save_npz(
        fold_path,
        probability=probability,
        oracle_action=held.oracle_action.astype(np.int8),
        image_path=held.image_path,
        classes=model.classes_.astype(np.int8),
    )
    parameters = {
        "held_out_class": held_out,
        "training_classes": train_names,
        "feature_order": list(FEATURE_ORDER),
        "median": median.tolist(),
        "iqr": iqr.tolist(),
        "classes": model.classes_.tolist(),
        "coefficient": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "n_iter": n_iter,
        "max_iter": max_iter,
        "converged": bool(np.max(model.n_iter_) <= max_iter),
        "training_patches": int(len(train_labels)),
        "held_out_patches": int(held.oracle_action.size),
        "training_class_counts": {
            str(value): int(np.count_nonzero(train_labels == value)) for value in (-1, 0, 1)
        },
        "fit_elapsed_seconds": elapsed,
        "warnings": [str(item.message) for item in caught],
    }
    write_json(output / "parameters" / f"{held_out}.json", parameters)
    del train_features, train_labels, standardized_train, standardized_test, model
    return parameters


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    shards, provenance = load_r1_shards(
        args.source_root, args.trust_root, args.utility_root, verify_hashes=not args.skip_hashes
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fold_parameters: list[dict[str, Any]] = []
    for held_out in EXPECTED_CLASSES:
        held = shards[held_out]
        fold_path = args.output / "folds" / f"{held_out}.npz"
        parameter_path = args.output / "parameters" / f"{held_out}.json"
        if valid_existing_fold(fold_path, held.image_path, held.oracle_action) and parameter_path.exists():
            fold_parameters.append(json.loads(parameter_path.read_text()))
            continue
        if fold_path.exists() or parameter_path.exists():
            raise RuntimeError(f"invalid partial R1 fold artifact: {held_out}")
        fold_parameters.append(fit_fold(held_out, shards, args.output, args.max_iter))
    probabilities: list[np.ndarray] = []
    oracle: list[np.ndarray] = []
    for class_name in EXPECTED_CLASSES:
        with np.load(args.output / "folds" / f"{class_name}.npz", allow_pickle=False) as data:
            probabilities.append(np.asarray(data["probability"], dtype=np.float32).reshape(-1, 3))
            oracle.append(np.asarray(data["oracle_action"], dtype=np.int8).reshape(-1))
            classes = np.asarray(data["classes"], dtype=np.int8)
    probability = np.concatenate(probabilities, axis=0)
    oracle_action = np.concatenate(oracle, axis=0)
    prediction, confidence, threshold_rows = threshold_landscape(oracle_action, probability, classes)
    selected = select_threshold(threshold_rows)
    write_csv(args.output / "threshold_landscape.csv", threshold_rows)
    selection = {
        "selected_threshold": selected,
        "status": "RISK_GATE_PASS" if selected is not None else "RISK_GATE_FAIL",
        "threshold_rows": threshold_rows,
        "unfiltered_argmax_action_counts": {
            str(value): int(np.count_nonzero(prediction == value)) for value in (-1, 0, 1)
        },
        "confidence_range": [float(confidence.min()), float(confidence.max())],
        "medical_reads": 0,
        "mvtec_reads": 0,
        "phase2b_training_steps": 0,
        **recovery_metadata(args),
    }
    write_json(args.output / "selection.json", selection)
    runtime = {
        "status": "COMPLETE",
        "elapsed_seconds": time.perf_counter() - started,
        "fold_fit_seconds": {
            row["held_out_class"]: row["fit_elapsed_seconds"] for row in fold_parameters
        },
        "fold_n_iter": {row["held_out_class"]: row["n_iter"] for row in fold_parameters},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "git_head": _git_head(),
        "medical_reads": 0,
        "mvtec_reads": 0,
        "phase2b_training_steps": 0,
        **recovery_metadata(args),
    }
    write_json(args.output / "runtime.json", runtime)
    write_json(
        args.output / "provenance.json",
        provenance
        | {
            "git_head": runtime["git_head"],
            "solver": {
                "environment": "Thai",
                "python": runtime["python"],
                "numpy": runtime["numpy"],
                "sklearn": runtime["sklearn"],
                "estimator": repr(estimator(args.max_iter)),
            },
        }
        | recovery_metadata(args),
    )
    return selection


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0/sabra_source"),
    )
    result.add_argument(
        "--trust-root",
        type=Path,
        default=ROOT / "runs/phase5/sabra/TRUST_V2_DEVELOPMENT",
    )
    result.add_argument(
        "--utility-root",
        type=Path,
        default=ROOT / "results/sabra_car/r0/utility",
    )
    result.add_argument("--output", type=Path, default=ROOT / "results/sabra_car/r1")
    result.add_argument("--skip-hashes", action="store_true")
    result.set_defaults(max_iter=ORIGINAL_MAX_ITER, recovery_protocol=None)
    return result


def main() -> None:
    args = parser().parse_args()
    args.output = args.output.resolve()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
