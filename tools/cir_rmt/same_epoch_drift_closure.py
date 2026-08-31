#!/usr/bin/env python3
"""Compare P, C_OLD, and image-anchored A at the same training epochs.

The comparison is source-only and uses the frozen deterministic 96-image VisA
sample.  It deliberately reports descriptive distances rather than selecting a
hyperparameter or using target labels.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model.phase2b_runtime import deploy_native_logits
from scripts.cir_rmt.eval_full import ManifestDataset
from tools.cir_rmt.identity import config_sha256, load_cir_config
from tools.cir_rmt.pre_full_run_diagnostics import (
    IMAGE_SIZE,
    _capture,
    _concat,
    _load_payload,
    _linear_cka,
    _make_model,
    _norm_ratio,
    _pairwise_geometry_corr,
    _sample_indices,
)
from tools.cir_rmt.runtime import forward_cir


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPOCHS = (10, 12, 14, 16, 18, 20)
GEOMETRY_MAX_ROWS = 2048


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _cosines(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    numerator = np.sum(x * y, axis=-1)
    denominator = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return numerator / np.maximum(denominator, 1.0e-12)


def _geometry_input(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(value.shape[0], -1)
    if value.shape[0] <= GEOMETRY_MAX_ROWS:
        return value
    indices = np.linspace(0, value.shape[0] - 1, GEOMETRY_MAX_ROWS, dtype=np.int64)
    return value[indices]


def _feature_summary(x: np.ndarray, y: np.ndarray, *, seed: int) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xf = x.reshape(-1, x.shape[-1])
    yf = y.reshape(-1, y.shape[-1])
    cosine = _cosines(xf, yf)
    gx = _geometry_input(x)
    gy = _geometry_input(y)
    return {
        "mean_cosine": _finite(float(np.nanmean(cosine))),
        "median_cosine": _finite(float(np.nanmedian(cosine))),
        "norm_ratio": _norm_ratio(xf, yf),
        "linear_cka": _linear_cka(xf, yf, seed=seed),
        "pairwise_geometry_corr": _pairwise_geometry_corr(gx, gy),
        "mean_abs_delta": _finite(float(np.mean(np.abs(x - y)))),
        "geometry_rows": int(min(gx.shape[0], gy.shape[0])),
    }


def _parent_config(cir_config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(cir_config.get("parent_config_path", "configs/phase2b_canonical_v1.json")))
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def _sample(archive: Path, per_category: int, seed: int) -> tuple[list[int], list[dict[str, Any]]]:
    identity = json.loads((archive / "SOURCE_SAMPLE_IDENTITY.json").read_text(encoding="utf-8"))
    if int(identity["per_category"]) != int(per_category) or int(identity["sample_seed"]) != int(seed):
        raise ValueError("requested sample does not match frozen source identity")
    indices, rows = _sample_indices(ROOT / "dataset/hub/VisA.jsonl", int(per_category), int(seed))
    frozen_indices = [int(row["manifest_index"]) for row in identity["selection"]]
    if indices != frozen_indices or [str(row["image_path"]) for row in rows] != [str(row["image_path"]) for row in identity["selection"]]:
        raise ValueError("frozen 96-image source sample is not reproducible")
    return indices, rows


def _capture_checkpoint(
    checkpoint: Path,
    *,
    parent_config: Mapping[str, Any],
    cir_config: Mapping[str, Any],
    dataset: Any,
    clip_asset: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    payload = _load_payload(checkpoint)
    model = _make_model(parent_config, payload, clip_asset, device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    captures: list[dict[str, np.ndarray]] = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device).float()
            names = [str(value) for value in batch["class_name"]]
            output = forward_cir(model, image, names, device, cir_config, domain="Industrial", dataset_name="VisA")
            native_probability, _ = deploy_native_logits(output.native_logits, image_size=IMAGE_SIZE, domain="Industrial")
            captures.append(_capture(output, native_probability, output.cir_segmentation_probability))
            del output, image
    data = _concat(captures)
    del model, loader, captures, payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return data


def _payload_vector(payload: Mapping[str, Any], key: str) -> np.ndarray:
    values = payload.get(key, {})
    names = sorted(values)
    if not names:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate([values[name].detach().float().cpu().numpy().reshape(-1).astype(np.float64) for name in names])


def _parameter_rows(epoch: int, p_payload: Mapping[str, Any], c_payload: Mapping[str, Any], a_payload: Mapping[str, Any], p14_payload: Mapping[str, Any], hashes: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component in ("image_adapter", "text_adapter", "soft_prompt"):
        p = _payload_vector(p_payload, component)
        p14 = _payload_vector(p14_payload, component)
        for comparison, payload in (("C_OLD", c_payload), ("A", a_payload)):
            y = _payload_vector(payload, component)
            diff = y - p
            p14_diff = y - p14
            pnorm = float(np.linalg.norm(p))
            ynorm = float(np.linalg.norm(y))
            p14norm = float(np.linalg.norm(p14))
            rows.append({
                "epoch": epoch,
                "reference": f"P_E{epoch:02d}",
                "comparison": f"{comparison}_E{epoch:02d}",
                "component": component,
                "parameter_count": int(p.size),
                "l2_distance": float(np.linalg.norm(diff)),
                "normalized_l2": float(np.linalg.norm(diff) / max(pnorm, 1.0e-12)),
                "cosine_flattened": float(np.dot(p, y) / max(pnorm * ynorm, 1.0e-12)),
                "relative_update_magnitude": float(np.linalg.norm(diff) / max(ynorm, 1.0e-12)),
                "max_abs_delta": float(np.max(np.abs(diff))) if diff.size else 0.0,
                "diagnostic_reference": "P_E14",
                "diagnostic_l2_to_p_e14": float(np.linalg.norm(p14_diff)),
                "diagnostic_normalized_l2_to_p_e14": float(np.linalg.norm(p14_diff) / max(p14norm, 1.0e-12)),
                "parent_checkpoint_sha256": hashes["parent"],
                "old_cir_checkpoint_sha256": hashes["old_cir"],
                "anchor_checkpoint_sha256": hashes["anchor"],
            })
    return rows


def _feature_rows(epoch: int, reference: Mapping[str, np.ndarray], comparison: Mapping[str, np.ndarray], name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(signal: str, axis: str, x: np.ndarray, y: np.ndarray) -> None:
        summary = _feature_summary(x, y, seed=epoch)
        rows.append({
            "epoch": epoch,
            "reference": f"P_E{epoch:02d}",
            "comparison": f"{name}_E{epoch:02d}",
            "signal": signal,
            "axis": axis,
            "n_images": int(x.shape[0]),
            **summary,
        })

    for stage in range(3):
        add(f"Seg_stage{stage}_pooled", "feature", reference["seg_pooled"][stage], comparison["seg_pooled"][stage])
        add(f"Seg_stage{stage}_patch_subsample", "patch_feature", reference["seg_patch"][stage], comparison["seg_patch"][stage])
        add(f"Det_stage{stage}", "feature", reference["det"][stage], comparison["det"][stage])
        add(f"DFG_native_weights_stage{stage}", "group_class", reference["native_weights"][stage], comparison["native_weights"][stage])
        add(f"DFG_group_margins_stage{stage}", "patch_group", reference["group_margins"][stage], comparison["group_margins"][stage])
        add(f"DFG_native_fused_margin_stage{stage}", "patch", reference["native_margin"][stage], comparison["native_margin"][stage])
        add(f"DFG_transported_margin_stage{stage}", "patch", reference["cir_margin"][stage], comparison["cir_margin"][stage])
    add("Text_descriptors", "descriptor", reference["text"].transpose(0, 1, 3, 2), comparison["text"].transpose(0, 1, 3, 2))
    return rows


def _median_feature_rows(rows: Sequence[Mapping[str, Any]], epoch: int, comparison: str) -> list[dict[str, Any]]:
    """Add explicit per-epoch medians without hiding the per-stage rows."""
    selected = [row for row in rows if int(row["epoch"]) == epoch and row["comparison"] == f"{comparison}_E{epoch:02d}" and row["signal"] != "Text_descriptors"]
    if not selected:
        return []
    output: dict[str, Any] = {
        "epoch": epoch,
        "reference": f"P_E{epoch:02d}",
        "comparison": f"{comparison}_E{epoch:02d}",
        "signal": "MEDIAN_NON_TEXT_SIGNALS",
        "axis": "summary",
        "n_images": int(selected[0]["n_images"]),
    }
    for key in ("mean_cosine", "median_cosine", "norm_ratio", "linear_cka", "pairwise_geometry_corr", "mean_abs_delta", "geometry_rows"):
        values = [float(row[key]) for row in selected if row.get(key) not in (None, "")]
        output[key] = float(np.median(values)) if values else None
    return [output]


def _classification(parameter_rows: Sequence[Mapping[str, Any]], feature_rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    if not parameter_rows or not feature_rows:
        return "INCONCLUSIVE", "Required same-epoch parameter or feature rows are missing."
    p_image = [row for row in parameter_rows if row["component"] == "image_adapter"]
    p_old = np.asarray([float(row["normalized_l2"]) for row in p_image if str(row["comparison"]).startswith("C_OLD")], dtype=np.float64)
    p_anchor = np.asarray([float(row["normalized_l2"]) for row in p_image if str(row["comparison"]).startswith("A_")], dtype=np.float64)
    if not p_old.size or not p_anchor.size:
        return "INCONCLUSIVE", "Image-adapter comparison rows are incomplete."
    anchor_param_closer = float(np.nanmedian(p_anchor)) < float(np.nanmedian(p_old))
    non_text = [row for row in feature_rows if row["signal"] != "Text_descriptors"]
    old = np.asarray([1.0 - float(row["mean_cosine"]) for row in non_text if str(row["comparison"]).startswith("C_OLD")], dtype=np.float64)
    anchor = np.asarray([1.0 - float(row["mean_cosine"]) for row in non_text if str(row["comparison"]).startswith("A_")], dtype=np.float64)
    if not old.size or not anchor.size:
        return "INCONCLUSIVE", "Non-text feature comparison rows are incomplete."
    finite = np.isfinite(old) & np.isfinite(anchor)
    closer_fraction = float(np.mean(anchor[finite] < old[finite])) if finite.any() else float("nan")
    if anchor_param_closer and closer_fraction >= 0.8:
        return "PRESERVATION_SUPPORTED", f"Anchor image parameters are closer to P at the median and anchor feature drift is lower for {closer_fraction:.1%} of non-text signals."
    if anchor_param_closer and closer_fraction > 0.0:
        return "PRESERVATION_PARTIAL", f"Anchor image parameters are closer to P, but lower feature drift occurs for only {closer_fraction:.1%} of non-text signals."
    if anchor_param_closer:
        return "TRAJECTORY_REGULARIZATION_ONLY", "Image parameters are closer to P, but the measured feature trajectory does not show same-epoch preservation."
    return "PRESERVATION_NOT_SUPPORTED", "The image-anchor checkpoint is not closer to P in the same-epoch image-parameter comparison."


def run(args: argparse.Namespace) -> None:
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = args.config.expanduser().resolve()
    cir_config = load_cir_config(config_path)
    parent_config = _parent_config(cir_config)
    sample_archive = args.sample_archive.expanduser().resolve()
    indices, selected = _sample(sample_archive, int(args.per_category), int(args.sample_seed))
    base_dataset = ManifestDataset(args.source_root.expanduser().resolve(), ROOT / "dataset/hub/VisA.jsonl", IMAGE_SIZE)
    dataset = Subset(base_dataset, indices)
    epochs = tuple(sorted(set(int(value) for value in args.epochs)))
    if not epochs or any(epoch not in DEFAULT_EPOCHS for epoch in epochs):
        raise ValueError(f"epochs must be a subset of {DEFAULT_EPOCHS}")
    device = torch.device(args.device)
    parameter_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    checkpoint_identity: dict[str, Any] = {}
    parent_root = args.parent_run_root.expanduser().resolve()
    old_root = args.old_cir_run_root.expanduser().resolve()
    anchor_root = args.anchor_run_root.expanduser().resolve()
    p14_path = parent_root / "phase2b" / "checkpoints" / "adapter_14.pth"
    p14_payload = _load_payload(p14_path)
    for epoch in epochs:
        p_path = parent_root / "phase2b" / "checkpoints" / f"adapter_{epoch}.pth"
        old_path = old_root / "visa" / "seed0" / "checkpoints" / f"epoch_{epoch:02d}.pth"
        anchor_path = anchor_root / "visa" / "seed0" / "checkpoints" / f"epoch_{epoch:02d}.pth"
        for path in (p_path, old_path, anchor_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        hashes = {"parent": _sha256(p_path), "old_cir": _sha256(old_path), "anchor": _sha256(anchor_path)}
        checkpoint_identity[str(epoch)] = {key: value for key, value in hashes.items()}
        p_payload, old_payload, anchor_payload = _load_payload(p_path), _load_payload(old_path), _load_payload(anchor_path)
        parameter_rows.extend(_parameter_rows(epoch, p_payload, old_payload, anchor_payload, p14_payload, hashes))
        p_features = _capture_checkpoint(p_path, parent_config=parent_config, cir_config=cir_config, dataset=dataset, clip_asset=args.clip_asset.expanduser().resolve(), device=device, batch_size=int(args.batch_size), num_workers=int(args.num_workers))
        old_features = _capture_checkpoint(old_path, parent_config=parent_config, cir_config=cir_config, dataset=dataset, clip_asset=args.clip_asset.expanduser().resolve(), device=device, batch_size=int(args.batch_size), num_workers=int(args.num_workers))
        anchor_features = _capture_checkpoint(anchor_path, parent_config=parent_config, cir_config=cir_config, dataset=dataset, clip_asset=args.clip_asset.expanduser().resolve(), device=device, batch_size=int(args.batch_size), num_workers=int(args.num_workers))
        feature_rows.extend(_feature_rows(epoch, p_features, old_features, "C_OLD"))
        feature_rows.extend(_feature_rows(epoch, p_features, anchor_features, "A"))
        feature_rows.extend(_median_feature_rows(feature_rows, epoch, "C_OLD"))
        feature_rows.extend(_median_feature_rows(feature_rows, epoch, "A"))
        print(f"completed same-epoch closure E{epoch:02d} ({len(indices)} source images)", flush=True)
    parameter_fields = ["epoch", "reference", "comparison", "component", "parameter_count", "l2_distance", "normalized_l2", "cosine_flattened", "relative_update_magnitude", "max_abs_delta", "diagnostic_reference", "diagnostic_l2_to_p_e14", "diagnostic_normalized_l2_to_p_e14", "parent_checkpoint_sha256", "old_cir_checkpoint_sha256", "anchor_checkpoint_sha256"]
    feature_fields = ["epoch", "reference", "comparison", "signal", "axis", "n_images", "mean_cosine", "median_cosine", "norm_ratio", "linear_cka", "pairwise_geometry_corr", "mean_abs_delta", "geometry_rows"]
    _write_csv(output / "SAME_EPOCH_PARAMETER_DRIFT.csv", parameter_rows, parameter_fields)
    _write_csv(output / "SAME_EPOCH_FEATURE_DRIFT.csv", feature_rows, feature_fields)
    conclusion, rationale = _classification(parameter_rows, feature_rows) if len(epochs) == len(DEFAULT_EPOCHS) else ("INCONCLUSIVE", "E10/E12/E14 closure is an interim record; E16/E18/E20 are required for the final classification.")
    lines = [
        "# Representation preservation closure",
        "",
        f"Status: {'PASS' if len(epochs) == len(DEFAULT_EPOCHS) else 'INTERIM'}.",
        "",
        "Scope: source-only, deterministic 96-image VisA sample; P is matched Phase2B, C_OLD is the previously trained CIR run, and A is the E14 image-parameter-anchor continuation.",
        "",
        "The parameter rows compare each same-epoch checkpoint to P at the same epoch. `diagnostic_*_to_p_e14` is a descriptive common-reference view using the parent E14 checkpoint; it is not a training target or selection rule.",
        "",
        f"CONCLUSION: {conclusion}",
        f"Rationale: {rationale}",
        "",
        "No target-domain labels or Medical metrics are used in this closure. The tables report association and representation distance; they do not prove causal transfer preservation.",
    ]
    (output / "REPRESENTATION_PRESERVATION_CLOSURE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    identity = {
        "status": "PASS" if len(epochs) == len(DEFAULT_EPOCHS) else "INTERIM",
        "scope": "SOURCE_ONLY",
        "epochs": list(epochs),
        "n_images": len(indices),
        "sample_seed": int(args.sample_seed),
        "per_category": int(args.per_category),
        "selection_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "config_sha256": config_sha256(cir_config),
        "clip_asset_sha256": _sha256(args.clip_asset.expanduser().resolve()),
        "checkpoint_identity": checkpoint_identity,
        "conclusion": conclusion,
        "medical": "NOT_RUN",
        "mvtec": "NOT_RUN",
    }
    (output / "SAME_EPOCH_CLOSURE_STATUS.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--old-cir-run-root", type=Path, required=True)
    parser.add_argument("--anchor-run-root", type=Path, required=True)
    parser.add_argument("--sample-archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", nargs="+", type=int, default=list(DEFAULT_EPOCHS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=9014)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
