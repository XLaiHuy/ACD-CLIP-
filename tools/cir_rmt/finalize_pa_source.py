#!/usr/bin/env python3
"""Assemble the PA source factorial and source-only representation freeze."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
EPOCHS = (10, 12, 14, 16, 18, 20)
METRICS = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")
METHODS = ("P", "C_OLD_0", "PA", "A0")
ALL_FROZEN_METHODS = ("P", "C_OLD_0", "C_OLD_05", "A0", "A05")
TARGETS = ("Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    if value is None or value == "" or str(value).lower() in {"none", "nan"}:
        return None
    return float(value)


def _checkpoint_path(args: argparse.Namespace, method: str, epoch: int) -> Path:
    if method == "P":
        return args.parent_run_root / "phase2b" / "checkpoints" / f"adapter_{epoch}.pth"
    if method == "PA":
        return args.pa_run_root / "visa" / "seed0" / "checkpoints" / f"adapter_{epoch}.pth"
    if method.startswith("C_OLD"):
        return args.old_cir_run_root / "visa" / "seed0" / "checkpoints" / f"epoch_{epoch:02d}.pth"
    return args.anchor_run_root / "visa" / "seed0" / "checkpoints" / f"epoch_{epoch:02d}.pth"


def _load_source_matrix(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    rows = _rows(path)
    expected = {(epoch, method) for epoch in EPOCHS for method in ALL_FROZEN_METHODS}
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["epoch"]), row["method"])
        if key not in expected:
            continue
        result[key] = {**row, **{metric: _float(row.get(metric)) for metric in METRICS}}
    if set(result) != expected:
        raise ValueError(f"frozen source matrix incomplete: missing={sorted(expected - set(result))}")
    return result


def _load_pa_source(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    rows = _rows(path)
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["epoch"]), "PA")
        if key in result:
            raise ValueError(f"duplicate PA source row: {key}")
        result[key] = {**row, **{metric: _float(row.get(metric)) for metric in METRICS}}
    expected = {(epoch, "PA") for epoch in EPOCHS}
    if set(result) != expected:
        raise ValueError(f"PA source results incomplete: missing={sorted(expected - set(result))}")
    return result


def _verify_hashes(args: argparse.Namespace, frozen: Mapping[tuple[int, str], Mapping[str, Any]], pa: Mapping[tuple[int, str], Mapping[str, Any]]) -> None:
    for epoch in EPOCHS:
        for method in ALL_FROZEN_METHODS:
            path = _checkpoint_path(args, method, epoch)
            expected = str(frozen[(epoch, method)].get("checkpoint_sha256", ""))
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"frozen source checkpoint hash mismatch: {method} E{epoch}")
        path = _checkpoint_path(args, "PA", epoch)
        if not path.is_file() or _sha256(path) != str(pa[(epoch, "PA")]["checkpoint_sha256"]):
            raise ValueError(f"PA source checkpoint hash mismatch: E{epoch}")


def _factorial_rows(frozen: Mapping[tuple[int, str], Mapping[str, Any]], pa: Mapping[tuple[int, str], Mapping[str, Any]], evaluator_git_sha: str, source_sha: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        for metric in METRICS:
            p = float(frozen[(epoch, "P")][metric])
            c = float(frozen[(epoch, "C_OLD_0")][metric])
            pa_value = float(pa[(epoch, "PA")][metric])
            a = float(frozen[(epoch, "A0")][metric])
            rows.append({
                "epoch": epoch,
                "metric": metric,
                "P_native_no_anchor": p,
                "CIR_native_no_anchor": c,
                "PA_native_anchor": pa_value,
                "CIR_native_anchor": a,
                "CIR_no_anchor_minus_P": c - p,
                "anchor_no_CIR_minus_P": pa_value - p,
                "anchor_with_CIR_minus_CIR_no_anchor": a - c,
                "CIR_with_anchor_minus_PA": a - pa_value,
                "interaction": a - c - pa_value + p,
                "evaluator_git_sha": evaluator_git_sha,
                "frozen_source_matrix_sha256": source_sha,
            })
    return rows


def _parameter_vector(path: Path, component: str = "image_adapter") -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get(component)
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint lacks {component}: {path}")
    values = [value.detach().float().reshape(-1).numpy() for _, value in sorted(state.items()) if isinstance(value, torch.Tensor)]
    if not values:
        raise ValueError(f"empty {component} state: {path}")
    return np.concatenate(values).astype(np.float64, copy=False)


def _parameter_drift(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    anchor_e14 = _parameter_vector(_checkpoint_path(args, "P", 14))
    for epoch in EPOCHS:
        p = _parameter_vector(_checkpoint_path(args, "P", epoch))
        pa = _parameter_vector(_checkpoint_path(args, "PA", epoch))
        distance = float(np.linalg.norm(pa - p))
        p_norm = float(np.linalg.norm(p))
        cosine = float(np.dot(p, pa) / max(np.linalg.norm(p) * np.linalg.norm(pa), 1.0e-12))
        anchor_distance = float(np.linalg.norm(pa - anchor_e14))
        anchor_norm = float(np.linalg.norm(anchor_e14))
        rows.append({
            "epoch": epoch,
            "reference": f"P_E{epoch:02d}",
            "comparison": f"PA_E{epoch:02d}",
            "component": "image_adapter",
            "parameter_count": int(p.size),
            "l2_distance": distance,
            "normalized_l2": distance / max(p_norm, 1.0e-12),
            "cosine_flattened": cosine,
            "relative_update_magnitude": distance / max(p_norm, 1.0e-12),
            "max_abs_delta": float(np.max(np.abs(pa - p))),
            "diagnostic_reference": "P_E14",
            "diagnostic_l2_to_p_e14": anchor_distance,
            "diagnostic_normalized_l2_to_p_e14": anchor_distance / max(anchor_norm, 1.0e-12),
            "parent_checkpoint_sha256": _sha256(_checkpoint_path(args, "P", epoch)),
            "pa_checkpoint_sha256": _sha256(_checkpoint_path(args, "PA", epoch)),
        })
        del p, pa
    return rows


def _combined_feature_drift(output: Path, frozen_archive: Path) -> None:
    new_rows = _rows(output / "PA_SOURCE_FEATURE_DRIFT.csv")
    old_path = frozen_archive / "SAME_EPOCH_FEATURE_DRIFT.csv"
    old_rows = _rows(old_path) if old_path.is_file() else []
    rows: list[dict[str, Any]] = []
    fields = ["epoch", "reference", "comparison", "signal", "axis", "n_images", "mean_cosine", "norm_ratio", "linear_cka", "pairwise_geometry_corr", "mean_abs_delta", "geometry_rows", "source"]
    for row in old_rows:
        rows.append({**row, "source": "frozen_P_C_OLD_A_feature_drift"})
    for row in new_rows:
        rows.append({**row, "source": "new_P_PA_native_feature_drift"})
    _write_csv(output / "PA_FACTORIAL_DRIFT.csv", rows, fields)


def _write_reports(args: argparse.Namespace, output: Path, factorial: Sequence[Mapping[str, Any]], pa: Mapping[tuple[int, str], Mapping[str, Any]], frozen_source_sha: str, evaluator_git_sha: str, pa_config_sha: str) -> None:
    frozen_archive = args.frozen_source_archive.expanduser().resolve()
    primary = [row for row in factorial if row["metric"] == "pixel_auroc"]
    selected = max(primary, key=lambda row: (float(pa[(int(row["epoch"]), "PA")]["pixel_auroc"]), float(pa[(int(row["epoch"]), "PA")]["pixel_ap"]), -int(row["epoch"])))
    selected_epoch = int(selected["epoch"])
    (output / "PA_FACTORIAL_REPRESENTATION.md").write_text(
        "\n".join([
            "# PA factorial representation and source analysis",
            "",
            "Status: PASS for the source-only stage.",
            "",
            "The four primary source cells are P (native/no anchor), C_OLD_0 (CIR/native/no anchor), PA (native/image anchor), and A0 (CIR/native/image anchor). The primary CIR-with-anchor contrast is A0 - PA. The 2x2 interaction is A0 - C_OLD_0 - PA + P.",
            "",
            "PA was forwarded with the canonical native Phase2B path only. It has no CIR/RMT transport, peer search, delta, or alpha inference. The PA feature rows therefore measure the native representation change associated with the train-only image anchor relative to the matched P checkpoint; they do not establish a target-domain causal result.",
            "",
            f"Frozen P/C_OLD/A representation context was reused from `{frozen_archive / 'SAME_EPOCH_FEATURE_DRIFT.csv'}`. New P-vs-PA compact feature rows are in `PA_SOURCE_FEATURE_DRIFT.csv` and combined in `PA_FACTORIAL_DRIFT.csv`. Parameter rows are in `PA_PARAMETER_DRIFT.csv`. Frozen source matrix SHA256: `{frozen_source_sha}`.",
            "",
            f"Source-only Medical freeze selection rule: highest PA pixel AUROC, tie-break by PA pixel AP, then earliest epoch. This selected E{selected_epoch:02d} only as a preregistered reporting anchor; all six PA epochs remain required for Medical evaluation.",
            "",
            "No Medical or MVTec data were accessed by this source stage.",
            "",
        ]) + "\n",
        encoding="utf-8",
    )
    source_lines = [
        "# Source factorial interpretation",
        "",
        "Status: PASS. The fixed 96-image VisA sample is used only to characterize the 2x2 training factorial before Medical access.",
        "",
        "P = native Phase2B without anchor; C_OLD_0 = CIR training without anchor; PA = native Phase2B with the frozen P_E14 image anchor; A0 = CIR training with that same anchor.",
        "",
        "Primary contrast: CIR_WITH_ANCHOR = A0 - PA. Interaction = A0 - C_OLD_0 - PA + P. These source effects are diagnostic associations, not target-domain evidence and not a permission to tune on Medical.",
        "",
        "See SOURCE_FACTORIAL_2X2.csv for every epoch and metric, and PRE_PA_MEDICAL_FREEZE.json for the target-blind freeze created after this source stage.",
        "",
    ]
    (output / "SOURCE_FACTORIAL_INTERPRETATION.md").write_text("\n".join(source_lines), encoding="utf-8")
    freeze = {
        "status": "FROZEN",
        "freeze_scope": "before_medical",
        "source_sample": "fixed 96-image VisA sample seed 9014",
        "source_matrix_sha256": frozen_source_sha,
        "pa_source_results": str(output / "PA_SOURCE_RESULTS.csv"),
        "pa_source_results_sha256": _sha256(output / "PA_SOURCE_RESULTS.csv"),
        "selected_reporting_epoch": selected_epoch,
        "selection_rule": "highest PA source pixel AUROC; tie by PA source pixel AP; then earliest epoch",
        "candidate_epochs": list(EPOCHS),
        "medical_targets": list(TARGETS),
        "parent_config_sha256": "d24cf942684b0be3c12838699ec6fe452697bd7f0a58eabbf316fb79b1b18cdb",
        "pa_effective_config_sha256": pa_config_sha,
        "anchor_reference_checkpoint_sha256": _sha256(_checkpoint_path(args, "P", 14)),
        "architecture_freeze_sha256": "f6de6ee8f1998f591c077efeff50fa9741a9f8bad34603ba145ec54ef961ba86",
        "clip_asset_sha256": "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02",
        "evaluator_git_sha": evaluator_git_sha,
        "medical_status": "NOT_RUN",
        "mvtec_status": "NOT_RUN",
        "target_tuning_occurred": False,
    }
    (output / "PRE_PA_MEDICAL_FREEZE.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "PRE_PA_MEDICAL_FREEZE.md").write_text(
        "\n".join([
            "# Pre-PA Medical freeze",
            "",
            "Status: PASS. Source-only freeze completed before Medical access.",
            "",
            f"PA source evaluation covered E10/E12/E14/E16/E18/E20 on the fixed 96-image VisA sample. The source-only reporting epoch is E{selected_epoch:02d} under the rule: highest PA pixel AUROC, tie by PA pixel AP, then earliest epoch. This does not cherry-pick the Medical benchmark: all six PA checkpoints must be evaluated.",
            "",
            "Primary Medical factorial: P vs C_OLD_0 vs PA vs A0, with CIR-with-anchor = A0 - PA and interaction = A0 - C_OLD_0 - PA + P. Existing P/C_OLD/A Medical rows are frozen and will be reused; only the 36 PA cells are authorized for new evaluation.",
            "",
            "Identity: VisA source, seed 0, ViT-L/14@336, image size 518, FP32, AMP=false, TF32=false, effective batch 6, Adam/StepLR canonical schedule, exact P_E14 image anchor lambda=0.001. Target tuning: NO. MVTec: NOT RUN.",
            "",
            "The full machine-readable freeze is PRE_PA_MEDICAL_FREEZE.json. No Medical or MVTec data were accessed before this freeze.",
            "",
        ]) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    output = args.output_root.expanduser().resolve()
    frozen_archive = args.frozen_source_archive.expanduser().resolve()
    frozen = _load_source_matrix(frozen_archive / "FINAL_SOURCE_MATRIX.csv")
    pa = _load_pa_source(output / "PA_SOURCE_RESULTS.csv")
    _verify_hashes(args, frozen, pa)
    evaluator_git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    frozen_source_sha = _sha256(frozen_archive / "FINAL_SOURCE_MATRIX.csv")
    pa_manifest = json.loads((args.pa_run_root.expanduser().resolve() / "visa" / "seed0" / "run_manifest.json").read_text(encoding="utf-8"))
    pa_config_sha = str(pa_manifest.get("config_sha256", ""))
    if not pa_config_sha:
        raise ValueError("PA run manifest lacks config_sha256")
    factorial = _factorial_rows(frozen, pa, evaluator_git_sha, frozen_source_sha)
    fields = ["epoch", "metric", "P_native_no_anchor", "CIR_native_no_anchor", "PA_native_anchor", "CIR_native_anchor", "CIR_no_anchor_minus_P", "anchor_no_CIR_minus_P", "anchor_with_CIR_minus_CIR_no_anchor", "CIR_with_anchor_minus_PA", "interaction", "evaluator_git_sha", "frozen_source_matrix_sha256"]
    _write_csv(output / "SOURCE_FACTORIAL_2X2.csv", factorial, fields)
    _write_csv(output / "SOURCE_FACTORIAL_INTERACTION.csv", [row for row in factorial], fields)
    deltas = []
    for row in factorial:
        if row["metric"] in {"pixel_auroc", "pixel_ap"}:
            deltas.append({"epoch": row["epoch"], "metric": row["metric"], "CIR_with_anchor_minus_PA": row["CIR_with_anchor_minus_PA"], "interaction": row["interaction"], "status": "source_association"})
    _write_csv(output / "PA_SOURCE_FACTORIAL_DELTAS.csv", deltas, ["epoch", "metric", "CIR_with_anchor_minus_PA", "interaction", "status"])
    parameter_rows = _parameter_drift(args)
    _write_csv(output / "PA_PARAMETER_DRIFT.csv", parameter_rows, ["epoch", "reference", "comparison", "component", "parameter_count", "l2_distance", "normalized_l2", "cosine_flattened", "relative_update_magnitude", "max_abs_delta", "diagnostic_reference", "diagnostic_l2_to_p_e14", "diagnostic_normalized_l2_to_p_e14", "parent_checkpoint_sha256", "pa_checkpoint_sha256"])
    _combined_feature_drift(output, frozen_archive)
    _write_reports(args, output, factorial, pa, frozen_source_sha, evaluator_git_sha, pa_config_sha)
    (output / "PA_SOURCE_FACTORIAL_STATUS.json").write_text(json.dumps({"status": "PASS", "source_only": True, "epochs": list(EPOCHS), "primary_methods": list(METHODS), "medical": "FROZEN_AFTER_SOURCE", "mvtec": "NOT_RUN", "frozen_source_matrix_sha256": frozen_source_sha}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-source-archive", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--pa-run-root", type=Path, required=True)
    parser.add_argument("--old-cir-run-root", type=Path, required=True)
    parser.add_argument("--anchor-run-root", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
