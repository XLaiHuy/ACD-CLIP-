#!/usr/bin/env python3
"""Deterministically export completed canonical Phase2B+SABRA results.

This module only reads completed JSON/CSV artifacts.  It never loads a model,
opens a dataset sample, reruns an evaluator, or changes scientific results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCIENTIFIC_CODE_SHA = "4aa9b465ddeb072e9218b74982306d6324c62375"
METRIC_NAMES = ("pAUROC", "pAP", "iAUROC", "iAP")
RAW_METRIC_NAMES = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metric is not finite: {value!r}")
    return result


def _metric_mapping(payload: Mapping[str, Any]) -> dict[str, float | None]:
    aliases = {
        "pAUROC": ("pixel_auroc", "pAUROC", "p_auroc"),
        "pAP": ("pixel_ap", "pAP", "p_ap"),
        "iAUROC": ("image_auroc", "iAUROC", "i_auroc"),
        "iAP": ("image_ap", "iAP", "i_ap"),
    }
    output: dict[str, float | None] = {}
    for name, candidates in aliases.items():
        found = next((candidate for candidate in candidates if candidate in payload), None)
        output[name] = _finite_or_none(payload[found]) if found is not None else None
    return output


def delta_metrics(phase2b: Mapping[str, Any], sabra: Mapping[str, Any]) -> dict[str, float | None]:
    left = _metric_mapping(phase2b)
    right = _metric_mapping(sabra)
    return {
        name: (None if left[name] is None or right[name] is None else right[name] - left[name])
        for name in METRIC_NAMES
    }


def _mean_defined(rows: Sequence[Mapping[str, float | None]]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        output[name] = None if not values else sum(values) / len(values)
    return output


def _compare_metrics(payload: Mapping[str, Any]) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Read the compare evaluator's macro fields without recomputing metrics."""
    phase2b = payload.get("phase2b_macro", payload.get("phase2b_metrics"))
    sabra = payload.get("sabra_macro", payload.get("sabra_metrics"))
    if isinstance(phase2b, Mapping) and isinstance(sabra, Mapping):
        return _metric_mapping(phase2b), _metric_mapping(sabra)
    # A compact fixture may contain only per-class compare dictionaries.
    phase2b_classes = payload.get("phase2b")
    sabra_classes = payload.get("sabra")
    if not isinstance(phase2b_classes, Mapping) or not isinstance(sabra_classes, Mapping):
        raise ValueError("Medical metrics.json lacks compare phase2b/sabra macro fields")
    phase2b_rows = [_metric_mapping(value) for value in phase2b_classes.values() if isinstance(value, Mapping)]
    sabra_rows = [_metric_mapping(value) for value in sabra_classes.values() if isinstance(value, Mapping)]
    if not phase2b_rows or not sabra_rows:
        raise ValueError("Medical compare metrics contain no per-class rows")
    return _mean_defined(phase2b_rows), _mean_defined(sabra_rows)


def _class_metrics(payload: Mapping[str, Any], method: str) -> list[dict[str, float | None]]:
    values = payload.get(method)
    if not isinstance(values, Mapping):
        return []
    return [_metric_mapping(value) for value in values.values() if isinstance(value, Mapping)]


def _reference_rows(path: Path) -> dict[str, dict[str, float | None]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        items = payload["rows"]
    elif isinstance(payload, list):
        items = payload
    elif isinstance(payload, Mapping):
        items = [{"Dataset": key, **value} for key, value in payload.items() if isinstance(value, Mapping)]
    else:
        raise ValueError("ACD-CLIP reference JSON must be a list, rows object, or dataset mapping")
    output: dict[str, dict[str, float | None]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("ACD-CLIP reference rows must be objects")
        name = item.get("Dataset", item.get("dataset"))
        if not name:
            raise ValueError("ACD-CLIP reference row lacks Dataset")
        metrics = item.get("metrics") if isinstance(item.get("metrics"), Mapping) else item
        output[str(name)] = _metric_mapping(metrics)
    return output


def _environment(preflight: Mapping[str, Any] | None) -> dict[str, Any]:
    if preflight is not None:
        return {
            "python": preflight.get("python", sys.executable),
            "python_version": preflight.get("python_version", platform.python_version()),
            "torch": preflight.get("torch"),
            "cuda": preflight.get("cuda"),
            "gpu": preflight.get("gpu"),
            "vram_gib": preflight.get("vram_gib"),
        }
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": None,
        "cuda": None,
        "gpu": None,
        "vram_gib": None,
    }


def _csv_value(value: Any) -> Any:
    # csv.DictWriter emits None as an empty field; retain that undefined metric
    # representation rather than converting it to zero.
    return value


def export_results(
    run_root: Path,
    *,
    code_sha: str = SCIENTIFIC_CODE_SHA,
    acdclip_reference: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    selection_path = run_root / "phase2b_selection" / "phase2b_selection.json"
    freeze_path = run_root / "sabra_lambda" / "SABRA_FREEZE.json"
    if not selection_path.is_file() or not freeze_path.is_file():
        raise FileNotFoundError("completed Phase2B selection and SABRA freeze are required")
    selection = load_json(selection_path)
    freeze = load_json(freeze_path)
    if selection.get("status") != "FROZEN":
        raise ValueError("Phase2B selection is not FROZEN")
    selected_checkpoint = Path(str(selection.get("selected_checkpoint", ""))).expanduser()
    selected_sha = str(selection.get("selected_checkpoint_sha256", ""))
    if not selected_checkpoint.is_file() or not selected_sha:
        raise ValueError("Phase2B selection lacks a selected checkpoint SHA256")
    if sha256_file(selected_checkpoint) != selected_sha:
        raise ValueError("selected checkpoint SHA256 mismatch")

    # Importing this validator reads only the freeze JSON and canonical
    # contracts; it does not load a model or dataset.
    from tools.sabra.artifacts import validate_sabra_freeze

    validate_sabra_freeze(freeze, checkpoint_sha256=selected_sha)
    if freeze.get("provenance", {}).get("git_sha") != code_sha:
        raise ValueError("SABRA freeze provenance SHA does not match exporter code SHA")
    if int(freeze["phase2b"]["selected_epoch"]) != int(selection["selected_epoch"]):
        raise ValueError("SABRA freeze selected epoch does not match Phase2B selection")
    if freeze.get("relational", {}).get("backend") != "fast":
        raise ValueError("canonical export requires frozen SABRA backend=fast")

    from dataset.info import CLASS_NAMES, is_medical_dataset

    medical_datasets = tuple(name for name in CLASS_NAMES if is_medical_dataset(name))
    rows: list[dict[str, Any]] = []
    phase2b_classes: list[dict[str, float | None]] = []
    sabra_classes: list[dict[str, float | None]] = []
    for dataset in medical_datasets:
        metrics_path = run_root / "medical" / dataset / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing completed Medical metrics: {metrics_path}")
        payload = load_json(metrics_path)
        phase2b, sabra = _compare_metrics(payload)
        phase2b_classes.extend(_class_metrics(payload, "phase2b"))
        sabra_classes.extend(_class_metrics(payload, "sabra"))
        rows.append({"Dataset": dataset, "Phase2B": phase2b, "SABRA": sabra, "Delta": delta_metrics(phase2b, sabra)})

    overall_phase2b = _mean_defined(phase2b_classes) if phase2b_classes else _mean_defined([row["Phase2B"] for row in rows])
    overall_sabra = _mean_defined(sabra_classes) if sabra_classes else _mean_defined([row["SABRA"] for row in rows])
    overall = {"Dataset": "OVERALL_MACRO", "Phase2B": overall_phase2b, "SABRA": overall_sabra, "Delta": delta_metrics(overall_phase2b, overall_sabra)}

    reference = _reference_rows(acdclip_reference) if acdclip_reference is not None else {}
    for row in rows + [overall]:
        ref = reference.get(str(row["Dataset"]))
        if ref is not None:
            row["ACD-CLIP"] = ref
            row["Delta_P2B"] = {
                name: (None if row["Phase2B"][name] is None or ref[name] is None else row["Phase2B"][name] - ref[name])
                for name in METRIC_NAMES
            }
            row["Delta_SABRA"] = row["Delta"]

    preflight_path = run_root / "manifests" / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError(f"missing required preflight manifest: {preflight_path}")
    preflight = load_json(preflight_path)
    if preflight.get("scientific_code_sha", preflight.get("git_sha")) != code_sha:
        raise ValueError("preflight manifest scientific code SHA does not match exporter code SHA")
    workflow_package_sha = preflight.get("workflow_package_sha")
    if not isinstance(workflow_package_sha, str) or len(workflow_package_sha) != 40:
        raise ValueError("preflight manifest lacks a valid workflow package SHA")
    if preflight.get("scientific_code_verified") is not True:
        raise ValueError("preflight manifest does not certify the scientific code identity")
    if preflight.get("cuda_available") is not True:
        raise ValueError("preflight manifest does not certify CUDA availability")
    if preflight.get("matmul_tf32") is not False or preflight.get("cudnn_tf32") is not False:
        raise ValueError("preflight manifest does not certify TF32 disabled")
    sm_value = float(freeze["correction"]["margin_scale"])
    provenance = {
        "code_sha": code_sha,
        "scientific_code_sha": code_sha,
        "workflow_package_sha": workflow_package_sha,
        "scientific_code_verified": True,
        "phase2b_selected_epoch": int(selection["selected_epoch"]),
        "phase2b_selected_checkpoint": str(selected_checkpoint.resolve()),
        "phase2b_selected_checkpoint_sha256": selected_sha,
        "trust": {
            "feature_order": freeze["trust"]["feature_order"],
            "predictor": freeze["trust"].get("predictor"),
            "settings": freeze["trust"].get("settings"),
        },
        "need": {
            "feature_order": freeze["need"]["feature_order"],
            "predictor": freeze["need"].get("predictor"),
            "settings": freeze["need"].get("settings"),
        },
        "sabra_backend": freeze["relational"]["backend"],
        "s_m": sm_value,
        "margin_scale": {
            "value": sm_value,
            "definition": freeze["correction"]["margin_scale_definition"],
        },
        "selected_lambda": freeze["correction"]["lambda"],
        "sabra_freeze_sha256": sha256_file(freeze_path),
        **_environment(preflight),
        "phase2b_batch": 6,
        "grad_accumulation": 1,
        "effective_batch": 6,
        "precision": "fp32",
        "tf32": False,
        "source": "VisA",
        "development": "MVTecAD",
        "final": "Medical",
    }

    output = {"code_sha": code_sha, "rows": rows, "overall": overall, "provenance": provenance}
    if reference:
        output["acdclip_reference"] = str(acdclip_reference.resolve())

    final_dir = run_root / "final"
    targets = [final_dir / name for name in ("summary.csv", "summary.json", "deltas.csv", "provenance.json")]
    if not force and any(path.exists() for path in targets):
        raise FileExistsError("final export exists; pass --force for an explicit deterministic replacement")
    if dry_run:
        print(f"DRY_RUN: would write {', '.join(str(path) for path in targets)}")
        return output

    final_dir.mkdir(parents=True, exist_ok=True)
    base_fields = ["Dataset"] + [f"Phase2B_{name}" for name in METRIC_NAMES] + [f"SABRA_{name}" for name in METRIC_NAMES] + [f"Delta_{name}" for name in METRIC_NAMES]
    optional_fields = []
    if reference:
        optional_fields = (
            [f"ACD-CLIP_{name}" for name in METRIC_NAMES]
            + [f"Delta_P2B_{name}" for name in METRIC_NAMES]
            + [f"Delta_SABRA_{name}" for name in METRIC_NAMES]
        )
    fields = base_fields + optional_fields
    with (final_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows + [overall]:
            flat = {"Dataset": row["Dataset"]}
            flat.update({f"Phase2B_{name}": _csv_value(row["Phase2B"][name]) for name in METRIC_NAMES})
            flat.update({f"SABRA_{name}": _csv_value(row["SABRA"][name]) for name in METRIC_NAMES})
            flat.update({f"Delta_{name}": _csv_value(row["Delta"][name]) for name in METRIC_NAMES})
            if reference and "ACD-CLIP" in row:
                flat.update({f"ACD-CLIP_{name}": _csv_value(row["ACD-CLIP"][name]) for name in METRIC_NAMES})
                flat.update({f"Delta_P2B_{name}": _csv_value(row["Delta_P2B"][name]) for name in METRIC_NAMES})
                flat.update({f"Delta_SABRA_{name}": _csv_value(row["Delta_SABRA"][name]) for name in METRIC_NAMES})
            writer.writerow(flat)

    with (final_dir / "deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["Dataset"] + [f"delta_{name}" for name in METRIC_NAMES]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows + [overall]:
            writer.writerow({"Dataset": row["Dataset"], **{f"delta_{name}": _csv_value(row["Delta"][name]) for name in METRIC_NAMES}})

    (final_dir / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (final_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"EXPORT_STATUS=PASS datasets={len(rows)} output={final_dir}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("RUN_ROOT", "/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0")))
    parser.add_argument("--acdclip-reference-json", type=Path)
    parser.add_argument("--code-sha", default=SCIENTIFIC_CODE_SHA)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    dry_run = bool(args.dry_run or os.environ.get("DRY_RUN") == "1")
    export_results(
        args.run_root,
        code_sha=str(args.code_sha),
        acdclip_reference=args.acdclip_reference_json,
        force=bool(args.force),
        dry_run=dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
