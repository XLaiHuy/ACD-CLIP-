"""Pure-NumPy contracts shared by SABRA-CAR R1 fitting and evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_CLASSES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)
FEATURE_ORDER = (
    "margin_within_image_rank",
    "robust_margin_normalization",
    "D_rank",
    "deployment_sensitivity",
    "E",
    "peer_coherence",
    "query_support_mean",
    "peer_eigen_entropy",
    "stage_query_profile_disagreement",
    "supported_p9_stability",
    "supported_p16_stability",
)
SOURCE_FEATURES = FEATURE_ORDER[:9]
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
PATCHES = 1369
EPSILON = 1e-8


@dataclass(frozen=True)
class R1Shard:
    class_name: str
    image_path: np.ndarray
    features: np.ndarray
    oracle_action: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def classify_utility(utility: np.ndarray) -> np.ndarray:
    utility = np.asarray(utility, dtype=np.float32)
    return np.where(utility > EPSILON, 1, np.where(utility < -EPSILON, -1, 0)).astype(np.int8)


def stack_features(source: dict[str, np.ndarray], trust: dict[str, np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(source[name], dtype=np.float32) for name in SOURCE_FEATURES]
    p9 = np.where(
        np.asarray(trust["valid_p9"], dtype=bool),
        np.asarray(trust["S9"], dtype=np.float32),
        np.float32(0.0),
    )
    p16 = np.where(
        np.asarray(trust["valid_p16"], dtype=bool),
        np.asarray(trust["S16"], dtype=np.float32),
        np.float32(0.0),
    )
    arrays.extend([p9, p16])
    shape = arrays[0].shape
    if len(shape) != 2 or shape[1] != PATCHES or any(item.shape != shape for item in arrays):
        raise ValueError("R1 feature shapes must all be [images,1369]")
    result = np.stack(arrays, axis=-1)
    if result.shape[-1] != len(FEATURE_ORDER) or not np.isfinite(result).all():
        raise ValueError("R1 features are non-finite or have the wrong width")
    return result.astype(np.float32, copy=False)


def _load_manifest(path: Path, trust: bool) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("GT_FREE_CACHE_FINALIZED") is not True or manifest.get("immutable") is not True:
        raise RuntimeError(f"cache is not finalized and immutable: {path}")
    if manifest.get("record_count") != 2162 or tuple(manifest.get("classes", ())) != EXPECTED_CLASSES:
        raise RuntimeError(f"cache record/class contract failed: {path}")
    if trust:
        gate = manifest.get("finalization_gate", {})
        if (
            gate.get("gt_free_only") is not True
            or gate.get("baseline_parity") != "PASS"
            or gate.get("p16_parity") != "PASS"
            or gate.get("medical_opened") is not False
            or gate.get("mvtec_opened") is not False
        ):
            raise RuntimeError("Trust-v2 finalization gate failed")
        counters = manifest.get("counters", {})
        if counters.get("MEDICAL_READS") != 0 or counters.get("MVTEC_READS_BEFORE_FREEZE") != 0:
            raise RuntimeError("Trust-v2 forbidden data counter is nonzero")
    else:
        if (
            manifest.get("dataset") != "VisA"
            or manifest.get("medical_reads") != 0
            or manifest.get("labels_read") is not False
            or manifest.get("mask_paths_read") is not False
            or manifest.get("mask_pixels_read") is not False
        ):
            raise RuntimeError("canonical GT-free manifest contract failed")
    return manifest


def load_r1_shards(
    source_root: Path,
    trust_root: Path,
    utility_root: Path,
    verify_hashes: bool = True,
) -> tuple[dict[str, R1Shard], dict[str, Any]]:
    source_manifest_path = source_root / "GT_FREE_MANIFEST.json"
    trust_manifest_path = trust_root / "TRUST_V2_GT_FREE_MANIFEST.json"
    source_manifest = _load_manifest(source_manifest_path, trust=False)
    trust_manifest = _load_manifest(trust_manifest_path, trust=True)
    output: dict[str, R1Shard] = {}
    records = 0
    shard_hashes: dict[str, dict[str, str]] = {}
    for class_name in EXPECTED_CLASSES:
        source_path = source_root / "gt_free_cache" / f"{class_name}.npz"
        trust_path = trust_root / "cache" / f"{class_name}.npz"
        utility_path = utility_root / f"{class_name}.npz"
        if verify_hashes:
            source_hash = sha256_file(source_path)
            trust_hash = sha256_file(trust_path)
            if source_hash != source_manifest["shards"][class_name]:
                raise RuntimeError(f"source shard hash failed: {class_name}")
            if trust_hash != trust_manifest["shards"][class_name]:
                raise RuntimeError(f"trust shard hash failed: {class_name}")
        else:
            source_hash = source_manifest["shards"][class_name]
            trust_hash = trust_manifest["shards"][class_name]
        with (
            np.load(source_path, allow_pickle=False) as source_data,
            np.load(trust_path, allow_pickle=False) as trust_data,
            np.load(utility_path, allow_pickle=False) as utility_data,
        ):
            source = {name: np.asarray(source_data[name]) for name in SOURCE_FEATURES}
            trust = {
                name: np.asarray(trust_data[name])
                for name in ("valid_p9", "valid_p16", "S9", "S16")
            }
            source_paths = source_data["image_path"].astype(str)
            trust_paths = trust_data["image_path"].astype(str)
            utility_paths = utility_data["image_path"].astype(str)
            if not np.array_equal(source_paths, trust_paths) or not np.array_equal(source_paths, utility_paths):
                raise RuntimeError(f"cross-cache path alignment failed: {class_name}")
            features = stack_features(source, trust)
            actions = classify_utility(np.asarray(utility_data["utility"], dtype=np.float32))
            if actions.shape != features.shape[:2]:
                raise RuntimeError(f"oracle action shape failed: {class_name}")
            output[class_name] = R1Shard(
                class_name=class_name,
                image_path=np.array(source_paths, copy=True),
                features=np.array(features, copy=True),
                oracle_action=np.array(actions, copy=True),
            )
            records += len(source_paths)
            shard_hashes[class_name] = {"source": source_hash, "trust": trust_hash}
    if records != 2162:
        raise RuntimeError(f"R1 record count failed: {records}")
    provenance = {
        "status": "PASS",
        "records": records,
        "classes": list(EXPECTED_CLASSES),
        "feature_order": list(FEATURE_ORDER),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "trust_manifest_sha256": sha256_file(trust_manifest_path),
        "shards": shard_hashes,
        "medical_reads": 0,
        "mvtec_reads": 0,
        "phase2b_training_steps": 0,
    }
    return output, provenance


def fit_robust_scaler(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(train_features)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_ORDER) or not np.isfinite(values).all():
        raise ValueError("invalid R1 training feature matrix")
    q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75], axis=0, method="linear")
    iqr = np.maximum(q75 - q25, 1e-6)
    return median.astype(np.float64), iqr.astype(np.float64)


def apply_robust_scaler(features: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    result = (np.asarray(features, dtype=np.float64) - median) / iqr
    if not np.isfinite(result).all():
        raise ValueError("non-finite standardized R1 features")
    return result


def stable_argmax_predictions(probability: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float64)
    classes = np.asarray(classes, dtype=np.int8)
    if tuple(classes.tolist()) != (-1, 0, 1) or probability.ndim != 2 or probability.shape[1] != 3:
        raise ValueError("R1 probability class contract failed")
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("R1 probabilities are invalid")
    index = np.argmax(probability, axis=1)
    return classes[index], probability[np.arange(len(index)), index]


def threshold_actions(prediction: np.ndarray, confidence: np.ndarray, threshold: float) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.int8)
    confidence = np.asarray(confidence, dtype=np.float64)
    if prediction.shape != confidence.shape:
        raise ValueError("prediction/confidence shape mismatch")
    return np.where(confidence >= threshold, prediction, 0).astype(np.int8)


def risk_row(
    oracle: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
    threshold: float | None,
    unfiltered_opposite_rate: float | None = None,
) -> dict[str, Any]:
    oracle = np.asarray(oracle, dtype=np.int8).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.int8).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    selected = prediction if threshold is None else threshold_actions(prediction, confidence, threshold)
    acted = selected != 0
    acted_count = int(np.count_nonzero(acted))
    opposite = acted & (oracle != 0) & (selected == -oracle)
    opposite_count = int(np.count_nonzero(opposite))
    rate = float(opposite_count / acted_count) if acted_count else None
    coverage = float(acted_count / len(selected))
    if unfiltered_opposite_rate is None:
        reduction = None
    elif unfiltered_opposite_rate == 0.0:
        reduction = None
    elif rate is None:
        reduction = 1.0
    else:
        reduction = float(1.0 - rate / unfiltered_opposite_rate)
    return {
        "threshold": "unfiltered" if threshold is None else float(threshold),
        "coverage": coverage,
        "acted_patches": acted_count,
        "opposite_sign_errors": opposite_count,
        "opposite_sign_rate": rate,
        "relative_opposite_sign_reduction": reduction,
    }


def threshold_landscape(
    oracle: np.ndarray, probability: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    prediction, confidence = stable_argmax_predictions(probability, classes)
    unfiltered = risk_row(oracle, prediction, confidence, None)
    baseline_rate = unfiltered["opposite_sign_rate"]
    rows = [unfiltered]
    for threshold in THRESHOLDS:
        row = risk_row(oracle, prediction, confidence, threshold, baseline_rate)
        zero_exception = baseline_rate == 0.0 and row["opposite_sign_rate"] == 0.0
        row["risk_gate_pass"] = bool(
            row["coverage"] >= 0.10
            and row["opposite_sign_rate"] is not None
            and row["opposite_sign_rate"] <= 0.05
            and (zero_exception or (row["relative_opposite_sign_reduction"] is not None and row["relative_opposite_sign_reduction"] >= 0.25))
        )
        rows.append(row)
    unfiltered["risk_gate_pass"] = False
    return prediction, confidence, rows


def select_threshold(rows: list[dict[str, Any]]) -> float | None:
    candidates = [row for row in rows if row.get("risk_gate_pass")]
    if not candidates:
        return None
    selected = min(candidates, key=lambda row: (float(row["threshold"]), -float(row["coverage"])))
    return float(selected["threshold"])
