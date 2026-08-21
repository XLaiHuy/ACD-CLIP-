"""Minimal Recovery-v2 MVTec orchestration.

This adapter is deliberately thin.  It composes the already frozen Phase2B
loader, the frozen Trust-v2 numerical sidecar, and the frozen VisA-trained
parameters.  The GT-free pass retains only compact evidence.  Labels and
masks are read by a separate post-pass, after the GT-free manifest exists.

No scaler, model, threshold, feature order, or correction magnitude is fit
on MVTec.  Medical data is not imported or traversed by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ModuleNotFoundError:  # deterministic exact metric fallback
    from evaluation.metrics import binary_average_precision, binary_auroc

    def roc_auc_score(labels, scores):
        return binary_auroc(scores, labels)

    def average_precision_score(labels, scores):
        return binary_average_precision(scores, labels)

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

try:
    from sabra.cache_runner import _forward_one, _load_model as _load_historical_model  # noqa: E402
except (ImportError, ModuleNotFoundError) as exc:  # canonical wrapper has no historical cache loader
    _historical_import_error = exc

    def _forward_one(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("historical cache runtime is unavailable in the canonical package") from _historical_import_error

    def _load_historical_model(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("historical cache runtime is unavailable in the canonical package") from _historical_import_error
from model.checkpoint_loader import load_checkpoint_for_evaluation  # noqa: E402
from sabra.data import IMAGE_SIZE, VisaEvidenceDataset, safe_data_path  # noqa: E402
from sabra.trust_v2.backend import compact_record_builder, validate_backend  # noqa: E402
from sabra.trust_v2.numerical import percentile_rank  # noqa: E402
from utils import get_phase2b_global_text_features  # noqa: E402

METADATA = ROOT / "dataset/hub/MVTec.jsonl"
FREEZE_ROOT = ROOT / "runs/phase5/sabra/TRUST_V2_M4_RECOVERY_V2"
EXTERNAL_ROOT = ROOT / "runs/phase5/sabra/TRUST_V2_EXTERNAL_MVTEC"
MVTEC_METADATA_SHA256 = "3a5e304ea16bba82e6e525d188698e91ca92b718696f8c257ed435d235b4cc2c"
EXPECTED_CLASSES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)
FEATURE_ORDER = (
    "E",
    "peer_coherence",
    "query_support_mean",
    "peer_eigen_entropy",
    "stage_query_profile_disagreement",
)
NEED_ORDER = (
    "margin_within_image_rank",
    "robust_margin_normalization",
    "D_rank",
    "deployment_sensitivity",
)
SCORE_NAMES = ("E_raw", "T_v2", "Need_C1", "A1_N_E_raw", "A2_N_T_v2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"ARTIFACT_PATH_COLLISION {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_npz_once(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise RuntimeError(f"ARTIFACT_PATH_COLLISION {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result.astype(np.float32)


def frozen_probability(parameters: dict[str, Any], values: np.ndarray) -> np.ndarray:
    """Apply a persisted StandardScaler + logistic model without fitting."""
    matrix = np.asarray(values, dtype=np.float64)
    mean = np.asarray(parameters["scaler_mean"], dtype=np.float64)
    scale = np.asarray(parameters["scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(parameters["logistic_coef"], dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(parameters["logistic_intercept"], dtype=np.float64).reshape(-1)[0])
    if matrix.shape[-1] != coefficient.size:
        raise ValueError(f"frozen feature width {coefficient.size} != input {matrix.shape[-1]}")
    if mean.size != coefficient.size or scale.size != coefficient.size:
        raise ValueError("frozen scaler width does not match frozen coefficients")
    return _sigmoid(((matrix - mean) / scale) @ coefficient + intercept)


def frozen_contract() -> dict[str, Any]:
    freeze = json.loads((FREEZE_ROOT / "EXTERNAL_VALIDATION_FREEZE.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "CANDIDATE_FROZEN":
        raise RuntimeError("RECOVERY_V2_FREEZE_INVALID")
    if freeze.get("selected_model") != "M1_E_Credibility":
        raise RuntimeError("RECOVERY_V2_SELECTED_MODEL_INVALID")
    if tuple(freeze.get("selected_model_feature_order", ())) != FEATURE_ORDER:
        raise RuntimeError("RECOVERY_V2_FEATURE_ORDER_INVALID")
    if freeze.get("PCRR_STATUS") != "DROP":
        raise RuntimeError("RECOVERY_V2_PCRR_STATUS_INVALID")
    if sha256_file(METADATA) != MVTEC_METADATA_SHA256:
        raise RuntimeError("MVTEC_METADATA_HASH_INVALID")
    trust = freeze["trust_model"]["trust_model_parameters"]
    need = freeze["need_c1_model_parameters"]
    if tuple(need.get("feature_order", ())) != NEED_ORDER:
        raise RuntimeError("RECOVERY_V2_NEED_FEATURE_ORDER_INVALID")
    if int(trust["n_features_in"]) != len(FEATURE_ORDER):
        raise RuntimeError("RECOVERY_V2_TRUST_PARAMETER_WIDTH_INVALID")
    if int(need["n_features_in"]) != len(NEED_ORDER):
        raise RuntimeError("RECOVERY_V2_NEED_PARAMETER_WIDTH_INVALID")
    return freeze


def _gt_free_rows(metadata_path: Path = METADATA) -> list[dict[str, str]]:
    """Sanitize metadata before the evidence pass; labels and mask paths drop."""
    rows: list[dict[str, str]] = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        rows.append({"class_name": str(raw["class_name"]), "image_path": str(raw["image_path"])})
    rows.sort(key=lambda row: (row["class_name"], row["image_path"]))
    if tuple(sorted({row["class_name"] for row in rows})) != tuple(sorted(EXPECTED_CLASSES)):
        raise RuntimeError("MVTEC_CLASS_SET_INVALID")
    if len(rows) != 1725 or len({(row["class_name"], row["image_path"]) for row in rows}) != len(rows):
        raise RuntimeError("MVTEC_METADATA_ROW_COUNT_INVALID")
    return rows


def resolve_data_root(data_root: Path, rows: Iterable[dict[str, str]]) -> Path:
    root = data_root.resolve()
    first = next(iter(rows), None)
    if first is None:
        raise RuntimeError("MVTEC_METADATA_EMPTY")
    if (root / first["image_path"]).is_file():
        return root
    wrapped = root / "mvtec_anomaly_detection"
    if (wrapped / first["image_path"]).is_file():
        return wrapped
    raise FileNotFoundError(f"MVTec image root does not contain {first['image_path']}")


def _mvt_text(model: torch.nn.Module, class_name: str, device: torch.device) -> torch.Tensor:
    return get_phase2b_global_text_features(
        model,
        "MVTec",
        [class_name],
        device,
        use_hybrid_soft_prompt=True,
        use_soft_prompt=False,
    ).float()


def _need_feature_matrix(
    record: dict[str, Any],
    native_margins: np.ndarray,
    deployment_sensitivity: np.ndarray,
) -> np.ndarray:
    """Build the exact frozen Need C1 feature vector for one image."""
    mean_margin = np.asarray(native_margins, dtype=np.float32).mean(axis=0)
    ranks = percentile_rank(mean_margin).astype(np.float32)
    median = float(np.median(mean_margin))
    mad = float(np.median(np.abs(mean_margin - median)))
    robust = (mean_margin - median) / (mad + 1e-6)
    return np.column_stack(
        [
            ranks,
            robust.astype(np.float32),
            np.asarray(record["D_rank"], dtype=np.float32),
            np.asarray(deployment_sensitivity, dtype=np.float32),
        ]
    )


def _score_record(
    record: dict[str, Any],
    freeze: dict[str, Any],
    native_margins: np.ndarray,
    deployment_sensitivity: np.ndarray,
) -> dict[str, np.ndarray]:
    trust_features = np.column_stack(
        [
            np.asarray(record["baseline_pgm"], dtype=np.float32),
            np.asarray(record["peer_coherence"], dtype=np.float32),
            np.asarray(record["query_support_mean"], dtype=np.float32),
            np.asarray(record["peer_eigen_entropy"], dtype=np.float32),
            np.asarray(record["stage_query_profile_disagreement"], dtype=np.float32),
        ]
    )
    trust_params = freeze["trust_model"]["trust_model_parameters"]
    trust = frozen_probability(trust_params, trust_features)
    need_features = _need_feature_matrix(record, native_margins, deployment_sensitivity)
    need = frozen_probability(freeze["need_c1_model_parameters"], need_features)
    e_raw = np.asarray(record["baseline_pgm"], dtype=np.float32)
    return {
        "E_raw": e_raw,
        "T_v2": trust,
        "Need_C1": need,
        "A1_N_E_raw": need * e_raw,
        "A2_N_T_v2": need * trust,
    }


def run_gt_free_stage(
    data_root: Path,
    output_root: Path,
    freeze: dict[str, Any],
    backend: str = "exact",
    checkpoint: Path | None = None,
    allow_nonfinal_checkpoint: bool = False,
) -> dict[str, Any]:
    backend = validate_backend(backend)
    build_compact_record = compact_record_builder(backend)
    rows = _gt_free_rows()
    root = resolve_data_root(data_root, rows)
    dataset = VisaEvidenceDataset(rows, root, image_size=IMAGE_SIZE)
    device = torch.device("cuda")
    if checkpoint is None:
        model, _, _ = _load_historical_model(device)
        checkpoint_identity = None
        evaluation_role = "HISTORICAL_FROZEN_EXTERNAL_VALIDATION"
    else:
        model, checkpoint_identity, _ = load_checkpoint_for_evaluation(
            checkpoint,
            device,
            expected_epoch=None if allow_nonfinal_checkpoint else 20,
        )
        evaluation_role = "POST_TRAINING_PREVIOUSLY_OBSERVED_EXTERNAL_BENCHMARK"
    text_by_class = {name: _mvt_text(model, name, device) for name in EXPECTED_CLASSES}
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in EXPECTED_CLASSES}
    for index, row in enumerate(rows):
        sample = dataset[index]
        text = text_by_class[row["class_name"]]
        result = _forward_one(model, text, sample["image"], device)
        record, _ = build_compact_record(
            np.asarray(result["features"], dtype=np.float32),
            np.asarray(result["native_margin"], dtype=np.float32),
            row["image_path"],
        )
        native_margins = np.asarray(result["native_margin"], dtype=np.float32)
        deployment_sensitivity = np.asarray(result["sensitivity"], dtype=np.float32)
        scores = _score_record(record, freeze, native_margins, deployment_sensitivity)
        arrays = {
            "image_path": np.asarray([row["image_path"]], dtype="U256"),
            "native_margins": native_margins[None],
            "D_rank": np.asarray(record["D_rank"], dtype=np.float32)[None],
            "deployment_sensitivity": deployment_sensitivity[None],
            **{name: values[None].astype(np.float32) for name, values in scores.items()},
        }
        records[row["class_name"]].append({"arrays": arrays, "image_path": row["image_path"]})
        if (index + 1) % 100 == 0:
            print(json.dumps({"stage": "GT_FREE", "images": index + 1, "total": len(rows)}, sort_keys=True), flush=True)
    shard_root = output_root / "GT_FREE"
    shard_root.mkdir(parents=True, exist_ok=False)
    for class_name, class_records in records.items():
        merged = {
            key: np.concatenate([item["arrays"][key] for item in class_records], axis=0)
            for key in class_records[0]["arrays"]
        }
        _write_npz_once(shard_root / f"{class_name}.npz", merged)
    manifest = {
        "status": "PASS",
        "stage": "GT_FREE",
        "gt_free": True,
        "backend": backend,
        "labels_read": False,
        "mask_paths_read": False,
        "mask_pixels_read": 0,
        "MVTec_image_reads": len(rows),
        "medical_reads": 0,
        "scientific_metrics_observed": False,
        "classes": list(EXPECTED_CLASSES),
        "images": len(rows),
        "frozen_selected_model": freeze["selected_model"],
        "frozen_feature_order": list(FEATURE_ORDER),
        "evaluation_role": evaluation_role,
        "checkpoint_identity": checkpoint_identity,
    }
    _write_json_once(output_root / "GT_FREE_MANIFEST.json", manifest)
    return manifest


def _mask_for_row(root: Path, row: dict[str, Any]) -> np.ndarray:
    if int(row["label"]) == 0:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask_path = safe_data_path(root, str(row["mask_path"]))
    with Image.open(mask_path) as handle:
        mask = np.asarray(handle.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST))
    return (mask > 0).astype(np.uint8)


def _metric(scores: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.int8).reshape(-1)
    if np.unique(target).size < 2:
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(target, scores)),
        "average_precision": float(average_precision_score(target, scores)),
    }


def _upsample_patch(values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).reshape(1, 1, 37, 37)
    return F.interpolate(tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True).numpy()[0, 0]


def _decision_ladder(deltas: np.ndarray) -> str:
    values = np.asarray(deltas, dtype=np.float64)
    positive = int(np.sum(values > 0))
    nonnegative = int(np.sum(values >= 0))
    catastrophic = bool(np.any(values <= -0.03))
    n = values.size
    if catastrophic:
        return "FALSIFIED"
    if values.mean() >= 0.010 and np.median(values) >= 0.005 and positive >= int(np.ceil(2 * n / 3)):
        return "SUPPORTED"
    if values.mean() >= 0.005 and np.median(values) >= 0 and positive >= int(np.ceil(0.60 * n)):
        return "PROMISING_BUT_UNCERTAIN"
    if values.mean() > 0 and np.median(values) >= 0 and nonnegative >= int(np.ceil(0.50 * n)):
        return "WEAK_POSITIVE_EVIDENCE"
    if values.mean() < 0 or np.median(values) < 0:
        return "FALSIFIED"
    return "INCONCLUSIVE"


def evaluate_ground_truth(data_root: Path, output_root: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads((output_root / "GT_FREE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or not manifest.get("gt_free") or manifest.get("scientific_metrics_observed"):
        raise RuntimeError("GT_FREE_FIREWALL_INVALID")
    rows: list[dict[str, Any]] = []
    for line in METADATA.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows.sort(key=lambda row: (str(row["class_name"]), str(row["image_path"])))
    root = resolve_data_root(data_root, _gt_free_rows())
    per_class: dict[str, dict[str, Any]] = {}
    shard_root = output_root / "GT_FREE"
    for class_name in EXPECTED_CLASSES:
        with np.load(shard_root / f"{class_name}.npz", allow_pickle=False) as shard:
            class_rows = [row for row in rows if str(row["class_name"]) == class_name]
            if len(class_rows) != len(shard["image_path"]):
                raise RuntimeError(f"GT_FREE_ROW_COUNT_INVALID {class_name}")
            labels = np.asarray([int(row["label"]) for row in class_rows], dtype=np.int8)
            masks = np.stack([_mask_for_row(root, row) for row in class_rows])
            metrics: dict[str, Any] = {"class": class_name, "images": int(labels.size)}
            for name in SCORE_NAMES:
                patch_scores = np.asarray(shard[name], dtype=np.float32)
                image_scores = patch_scores.max(axis=1)
                pixel_scores = np.stack([_upsample_patch(item) for item in patch_scores])
                image_metric = _metric(image_scores, labels)
                pixel_metric = _metric(pixel_scores, masks)
                metrics[name] = {"image": image_metric, "pixel": pixel_metric}
            e_auc = metrics["E_raw"]["image"]["auroc"]
            t_auc = metrics["T_v2"]["image"]["auroc"]
            metrics["trust_image_auroc_delta"] = None if e_auc is None or t_auc is None else float(t_auc - e_auc)
            raw_a1_auc = metrics["A1_N_E_raw"]["image"]["auroc"]
            a2_auc = metrics["A2_N_T_v2"]["image"]["auroc"]
            metrics["authority_raw_image_auroc_delta"] = None if raw_a1_auc is None or a2_auc is None else float(a2_auc - raw_a1_auc)
            per_class[class_name] = metrics
    trust_deltas = np.asarray([per_class[name]["trust_image_auroc_delta"] for name in EXPECTED_CLASSES], dtype=np.float64)
    raw_authority = np.asarray([per_class[name]["authority_raw_image_auroc_delta"] for name in EXPECTED_CLASSES], dtype=np.float64)
    result = {
        "status": "PASS",
        "scientific_validity": "VALID_TRUST_EXTERNAL_AUTHORITY_PRIMARY_COMPARATOR_NOT_PERSISTED",
        "trust": {
            "baseline": "E_raw_PGM_rank",
            "candidate": "T_v2",
            "class_deltas": trust_deltas.tolist(),
            "decision": _decision_ladder(trust_deltas),
        },
        "authority_v2_external": "UNRESOLVED_MISSING_FROZEN_M0_E_CALIBRATOR",
        "authority_secondary_raw": {
            "baseline": "A1_N_E_raw",
            "candidate": "A2_N_T_v2",
            "class_deltas": raw_authority.tolist(),
            "decision": _decision_ladder(raw_authority),
        },
        "medical_reads": 0,
        "MVTec_gt_reads": len(rows),
        "evaluation_role": manifest.get("evaluation_role"),
        "checkpoint_identity": manifest.get("checkpoint_identity"),
        "per_class": per_class,
    }
    _write_json_once(output_root / "PER_CLASS_METRICS.json", per_class)
    _write_json_once(output_root / "RESULT.json", result)
    return result


def run(
    data_root: Path,
    output_root: Path,
    backend: str = "exact",
    checkpoint: Path | None = None,
    allow_nonfinal_checkpoint: bool = False,
) -> dict[str, Any]:
    freeze = frozen_contract()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"ARTIFACT_PATH_COLLISION {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_gt_free_stage(
        data_root,
        output_root,
        freeze,
        backend=backend,
        checkpoint=checkpoint,
        allow_nonfinal_checkpoint=allow_nonfinal_checkpoint,
    )
    return evaluate_ground_truth(data_root, output_root, freeze)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="MVTec root; defaults to MVTEC_ROOT")
    parser.add_argument("--backend", choices=["exact", "fast"], default="exact")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Explicit post-training checkpoint; never falls back when supplied")
    parser.add_argument("--allow-nonfinal-checkpoint", action="store_true", help="Diagnostic-only override of the expected epoch-20 guard")
    parser.add_argument("--output-root", type=Path, default=EXTERNAL_ROOT)
    args = parser.parse_args()
    data_root = args.data_root or (Path(os.environ["MVTEC_ROOT"]) if os.environ.get("MVTEC_ROOT") else None)
    if data_root is None:
        raise SystemExit("MVTec root is required via --data-root or MVTEC_ROOT")
    print(json.dumps(
        run(
            data_root,
            args.output_root,
            backend=args.backend,
            checkpoint=args.checkpoint,
            allow_nonfinal_checkpoint=args.allow_nonfinal_checkpoint,
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
