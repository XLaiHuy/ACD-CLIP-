#!/usr/bin/env python3
"""Thin final evaluator CLI for Phase2B, SABRA, and compare mode."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from evaluation.datasets import MVTecDatasetAdapter, resolve_mvtec_root
from evaluation.evaluator import evaluate_records, image_score
from model.phase2b_runtime import forward_phase2b, load_json_config, load_phase2b_checkpoint, sha256_file
from tools.sabra.artifacts import load_json, validate_sabra_freeze
from tools.sabra.pipeline import compare_forward

MEDICAL_DATASETS = {"Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"}


def _write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    rows = []
    if "per_class" in result:
        rows = [{"class_name": key, **value} for key, value in result["per_class"].items()]
    elif "phase2b" in result and isinstance(result["phase2b"], dict):
        classes = sorted(result["phase2b"])
        for name in classes:
            rows.append({"class_name": name, **{f"phase2b_{key}": value for key, value in result["phase2b"][name].items()}, **{f"sabra_{key}": value for key, value in result["sabra"][name].items()}})
    if rows:
        with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def validate_medical_guard(dataset: str, freeze: dict[str, Any], checkpoint_sha256: str) -> None:
    if dataset in MEDICAL_DATASETS:
        validate_sabra_freeze(freeze, checkpoint_sha256=checkpoint_sha256)


def _iter_medical(dataset: str, data_root: Path) -> Iterable[dict[str, Any]]:
    # This adapter is wired for the later final run but is never invoked by
    # setup tests.  Its labels are consumed only after the image-only runtime
    # forward has produced a relational record.
    os.environ.setdefault("ACDCLIP_DATA_ROOT", str(data_root.expanduser().resolve()))
    from dataset import get_text_and_image_dataset

    datasets = get_text_and_image_dataset(dataset, 518, stage="test")
    for class_name, class_dataset in datasets.items():
        for sample in class_dataset:
            yield {"image": sample["image"], "mask": sample["mask"], "label": int(sample["label"]), "class_name": class_name, "image_path": sample["file_name"]}


def _iter_samples(dataset: str, data_root: Path | None) -> Iterable[dict[str, Any]]:
    if dataset == "MVTec":
        root = resolve_mvtec_root(data_root)
        if root is None:
            raise ValueError("MVTec evaluation requires --data-root/--mvtec-root or ACDCLIP_MVTEC_ROOT")
        yield from MVTecDatasetAdapter(root)
        return
    if dataset in MEDICAL_DATASETS:
        if data_root is None:
            raise ValueError("Medical evaluation requires --data-root")
        yield from _iter_medical(dataset, data_root)
        return
    raise ValueError(f"unsupported evaluation dataset: {dataset}")


def run_model_records(
    dataset: str,
    data_root: Path | None,
    selection: Mapping[str, Any],
    freeze: Mapping[str, Any] | None,
    config_path: Path,
    clip_asset: Path,
    device: torch.device,
    method: str,
) -> list[dict[str, Any]]:
    config = load_json_config(config_path)
    checkpoint_path = Path(str(selection["selected_checkpoint"])).expanduser()
    model = load_phase2b_checkpoint(checkpoint_path, config, clip_asset, device)
    domain = "Medical" if dataset in MEDICAL_DATASETS else "Industrial"
    records: list[dict[str, Any]] = []
    for sample in _iter_samples(dataset, data_root):
        image = sample["image"].unsqueeze(0).to(device)
        forward = forward_phase2b(model, image, [sample["class_name"]], device, config, domain=domain, require_grad=False, dataset_name=dataset)
        native_pixels = forward.native_segmentation_probability[0].detach().cpu().numpy()
        native_cls = float(forward.classification_probability[0].detach().cpu())
        payload: dict[str, Any] = {
            "class_name": sample["class_name"],
            "pixel_labels": sample["mask"].reshape(-1).numpy().astype(np.int8),
            "image_labels": np.asarray([sample["label"]], dtype=np.int8),
            "image_path": sample["image_path"],
            "phase2b": {"pixel_scores": native_pixels.reshape(-1), "image_scores": np.asarray([image_score(native_cls, float(native_pixels.max()), domain)])},
        }
        if method in {"sabra", "compare"}:
            if freeze is None:
                raise ValueError("SABRA method requires a validated freeze")
            composed = compare_forward(forward, freeze, domain=domain)
            corrected_pixels = composed["corrected_probability"][0].detach().cpu().numpy()
            payload["sabra"] = {
                "pixel_scores": corrected_pixels.reshape(-1),
                "image_scores": np.asarray([image_score(native_cls, float(corrected_pixels.max()), domain)]),
            }
        if method == "phase2b":
            payload = {**payload, "pixel_scores": payload["phase2b"]["pixel_scores"], "image_scores": payload["phase2b"]["image_scores"]}
        records.append(payload)
    return records


def _add_report_metadata(result: dict[str, Any], dataset: str, selection: Mapping[str, Any], freeze_path: Path | None) -> dict[str, Any]:
    result = dict(result)
    result["dataset"] = dataset
    result["role"] = "FINAL_ZERO_SHOT" if dataset in MEDICAL_DATASETS else "DEVELOPMENT"
    result["phase2b_checkpoint_sha256"] = selection.get("selected_checkpoint_sha256")
    if freeze_path is not None:
        result["sabra_freeze_sha256"] = sha256_file(freeze_path)
    if "phase2b_macro" in result:
        result["phase2b_metrics"] = result["phase2b_macro"]
    if "sabra_macro" in result:
        result["sabra_metrics"] = result["sabra_macro"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["phase2b", "sabra", "compare"], default="phase2b")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--phase2b-selection", type=Path, required=True)
    parser.add_argument("--sabra-freeze", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2b_canonical_v1.json"))
    parser.add_argument("--clip-asset", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric-mode", choices=["exact"], default="exact")
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument("--records-json", type=Path)
    args = parser.parse_args(argv)
    if args.pixel_stride != 1:
        raise SystemExit("canonical selection/final reports require --pixel-stride 1")
    selection = load_json(args.phase2b_selection)
    if selection.get("status") != "FROZEN":
        raise SystemExit("Phase2B selection must be FROZEN")
    freeze = None
    if args.method in {"sabra", "compare"}:
        if args.sabra_freeze is None:
            raise SystemExit("--sabra-freeze is required for SABRA methods")
        freeze = load_json(args.sabra_freeze)
        validate_sabra_freeze(freeze, checkpoint_sha256=selection.get("selected_checkpoint_sha256"))
    if args.dataset in MEDICAL_DATASETS and freeze is not None:
        validate_medical_guard(args.dataset, freeze, selection.get("selected_checkpoint_sha256"))
    if args.records_json is not None:
        records = json.loads(args.records_json.read_text(encoding="utf-8"))
        result = evaluate_records(records, method=args.method)
    else:
        if args.clip_asset is None:
            raise SystemExit("real model inference requires --clip-asset")
        records = run_model_records(args.dataset, args.data_root, selection, freeze, args.config, args.clip_asset, torch.device(args.device), args.method)
        result = evaluate_records(records, method=args.method)
    _write_outputs(args.output_dir, _add_report_metadata(result, args.dataset, selection, args.sabra_freeze))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
