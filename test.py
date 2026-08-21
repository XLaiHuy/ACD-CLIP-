#!/usr/bin/env python3
"""Final evaluator for native Phase2B, frozen SABRA, and compare mode."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dataset import get_text_and_image_dataset
from dataset.info import CLASS_NAMES, dataset_domain, is_medical_dataset
from evaluation.datasets import MVTecDatasetAdapter, resolve_mvtec_root
from evaluation.evaluator import evaluate_records, image_score
from model.phase2b_runtime import forward_phase2b, load_json_config, load_phase2b_checkpoint, sha256_file
from tools.sabra.artifacts import load_json, validate_sabra_freeze
from tools.sabra.pipeline import compare_forward

MEDICAL_DATASETS = tuple(name for name in CLASS_NAMES if is_medical_dataset(name))


def _exact_auc_ap_from_sorted_chunks(
    chunks: Any,
    total_pos: int | None = None,
    total_neg: int | None = None,
) -> tuple[float | None, float | None]:
    """Exact tie-aware AUROC/AP over sorted on-disk chunks, returned in percent.

    The historical audit passes ``[(score_npy, label_npy), ...]`` plus global
    positive/negative counts.  A two-array call remains supported for compact
    unit tests.  All chunks are concatenated once and delegated to the shared
    exact metric implementation; no threshold-by-threshold rescans occur.
    """
    from evaluation.metrics import binary_average_precision, binary_auroc
    two_array_call = total_neg is None and total_pos is not None and not isinstance(total_pos, (int, np.integer))
    if (total_pos is None and total_neg is None) or two_array_call:
        scores, labels = chunks, total_pos
        auc = binary_auroc(scores, labels, allow_undefined=True)
        ap = binary_average_precision(scores, labels, allow_undefined=True)
    else:
        score_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        for score_path, label_path in chunks:
            score_parts.append(np.asarray(np.load(score_path), dtype=np.float64).reshape(-1))
            label_parts.append(np.asarray(np.load(label_path), dtype=np.int8).reshape(-1))
        scores = np.concatenate(score_parts) if score_parts else np.empty(0, dtype=np.float64)
        labels = np.concatenate(label_parts) if label_parts else np.empty(0, dtype=np.int8)
        if scores.size != int(total_pos) + int(total_neg):
            raise ValueError("metric chunk counts do not match total_pos + total_neg")
        auc = binary_auroc(scores, labels, allow_undefined=True)
        ap = binary_average_precision(scores, labels, allow_undefined=True)
    return (None if auc is None else 100.0 * float(auc), None if ap is None else 100.0 * float(ap))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    if "per_class" in result:
        rows = [{"class_name": key, **value} for key, value in result["per_class"].items()]
    elif "phase2b" in result and isinstance(result["phase2b"], dict):
        for name in sorted(result["phase2b"]):
            rows.append({
                "class_name": name,
                **{f"phase2b_{key}": value for key, value in result["phase2b"][name].items()},
                **{f"sabra_{key}": value for key, value in result["sabra"][name].items()},
            })
    if rows:
        with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def validate_medical_guard(dataset: str, freeze: Mapping[str, Any], checkpoint_sha256: str) -> None:
    if is_medical_dataset(dataset):
        validate_sabra_freeze(freeze, checkpoint_sha256=checkpoint_sha256)


def _iter_medical(dataset: str, data_root: Path) -> Iterable[dict[str, Any]]:
    os.environ.setdefault("ACDCLIP_DATA_ROOT", str(data_root.expanduser().resolve()))
    datasets = get_text_and_image_dataset(dataset, 518, stage="test")
    for class_name, class_dataset in datasets.items():
        for sample in class_dataset:
            yield {
                "image": sample["image"],
                "mask": sample["mask"],
                "label": int(sample["label"]),
                "class_name": class_name,
                "image_path": sample["file_name"],
            }


class _MedicalSampleDataset(Dataset):
    """Map-style Medical adapter preserving the historical class/sample order."""

    def __init__(self, dataset: str, data_root: Path) -> None:
        os.environ.setdefault("ACDCLIP_DATA_ROOT", str(data_root.expanduser().resolve()))
        datasets = get_text_and_image_dataset(dataset, 518, stage="test")
        self.datasets = datasets
        self.entries = [
            (class_name, index)
            for class_name, class_dataset in datasets.items()
            for index in range(len(class_dataset))
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        class_name, sample_index = self.entries[int(index)]
        sample = self.datasets[class_name][sample_index]
        return {
            "image": sample["image"],
            "mask": sample["mask"],
            "label": int(sample["label"]),
            "class_name": class_name,
            "image_path": sample["file_name"],
        }


def _build_inference_dataset(dataset: str, data_root: Path | None) -> Dataset:
    if dataset == "MVTec":
        root = resolve_mvtec_root(data_root)
        if root is None:
            raise ValueError("MVTec evaluation requires --data-root/--mvtec-root or ACDCLIP_MVTEC_ROOT")
        return MVTecDatasetAdapter(root)
    if is_medical_dataset(dataset):
        if data_root is None:
            raise ValueError("Medical evaluation requires --data-root")
        return _MedicalSampleDataset(dataset, data_root)
    raise ValueError(f"unsupported evaluation dataset: {dataset}")


def _iter_samples(dataset: str, data_root: Path | None) -> Iterable[dict[str, Any]]:
    yield from _build_inference_dataset(dataset, data_root)


def _verify_selected_checkpoint(selection: Mapping[str, Any]) -> tuple[Path, str]:
    path_value = selection.get("selected_checkpoint")
    expected = selection.get("selected_checkpoint_sha256")
    if not path_value or not expected:
        raise ValueError("selection must contain selected_checkpoint and selected_checkpoint_sha256")
    path = Path(str(path_value)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"checkpoint SHA256 mismatch: expected {expected}, got {actual}")
    return path.resolve(), actual


def _cuda_runtime_stats(device: torch.device, started: float, images: int) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - started, 1e-9)
    stats: dict[str, Any] = {
        "images": int(images),
        "elapsed_seconds": float(elapsed),
        "samples_per_sec": float(images / elapsed),
        "eta_seconds": None,
        "peak_allocated_vram": None,
        "peak_reserved_vram": None,
        "total_vram": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        stats.update({
            "peak_allocated_vram": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram": int(torch.cuda.max_memory_reserved(device)),
            "total_vram": int(props.total_memory),
        })
    return stats


def run_model_records(
    dataset: str,
    data_root: Path | None,
    selection: Mapping[str, Any],
    freeze: Mapping[str, Any] | None,
    config_path: Path,
    clip_asset: Path,
    device: torch.device,
    method: str,
    *,
    batch_size: int = 6,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    runtime_stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    checkpoint_path, checkpoint_sha = _verify_selected_checkpoint(selection)
    config = load_json_config(config_path)
    model = load_phase2b_checkpoint(checkpoint_path, config, clip_asset, device)
    model.eval()
    domain = dataset_domain(dataset)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    inference_dataset = _build_inference_dataset(dataset, data_root)
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(device.type == "cuda" if pin_memory is None else pin_memory),
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    loader = DataLoader(inference_dataset, **loader_kwargs)
    progress = tqdm(total=len(inference_dataset), desc=f"{dataset} {method}", unit="img")
    try:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=device.type == "cuda").float()
            class_names = [str(value) for value in batch["class_name"]]
            forward = forward_phase2b(
                model,
                image,
                class_names,
                device,
                config,
                domain=domain,
                require_grad=False,
                dataset_name=dataset,
            )
            native_pixels = forward.deployed_segmentation_probability.detach().cpu().numpy().astype(np.float32)
            native_cls = forward.classification_probability.detach().cpu().numpy().astype(np.float32)
            masks = batch["mask"].detach().cpu().numpy().astype(np.int8)
            labels = batch["label"].detach().cpu().numpy().reshape(-1).astype(np.int8)
            paths = [str(value) for value in batch["image_path"]]
            corrected_pixels: np.ndarray | None = None
            if method in {"sabra", "compare"}:
                if freeze is None:
                    raise ValueError("SABRA method requires a validated freeze")
                # compare_forward consumes this single Batch-6 Phase2B forward.
                composed = compare_forward(forward, freeze, domain=domain)
                corrected_pixels = composed["corrected_probability"].detach().cpu().numpy().astype(np.float32)
            for index, class_name in enumerate(class_names):
                native_pixel = native_pixels[index]
                native_image_score = image_score(float(native_cls[index]), float(native_pixel.max()), domain)
                payload: dict[str, Any] = {
                    "class_name": class_name,
                    "pixel_labels": masks[index].reshape(-1),
                    "image_labels": np.asarray([labels[index]], dtype=np.int8),
                    "image_path": paths[index],
                    "phase2b": {
                        "pixel_scores": native_pixel.reshape(-1),
                        "image_scores": np.asarray([native_image_score]),
                    },
                }
                if corrected_pixels is not None:
                    corrected_pixel = corrected_pixels[index]
                    payload["sabra"] = {
                        "pixel_scores": corrected_pixel.reshape(-1),
                        "image_scores": np.asarray([image_score(float(native_cls[index]), float(corrected_pixel.max()), domain)]),
                    }
                if method == "phase2b":
                    payload = {
                        **payload,
                        "pixel_scores": payload["phase2b"]["pixel_scores"],
                        "image_scores": payload["phase2b"]["image_scores"],
                    }
                records.append(payload)
            progress.update(len(class_names))
            elapsed = max(time.perf_counter() - started, 1e-9)
            rate = len(records) / elapsed
            remaining = max(len(inference_dataset) - len(records), 0)
            progress.set_postfix({"img/s": f"{rate:.2f}", "eta": f"{remaining / max(rate, 1e-9):.0f}s"})
    finally:
        progress.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if runtime_stats is not None:
        runtime_stats.update(_cuda_runtime_stats(device, started, len(records)))
        runtime_stats["checkpoint_sha256"] = checkpoint_sha
        runtime_stats["domain"] = domain
    return records


def _metric_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(right) - float(left)


def _add_report_metadata(
    result: dict[str, Any],
    dataset: str,
    selection: Mapping[str, Any],
    freeze_path: Path | None,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = dict(result)
    output["dataset"] = dataset
    output["role"] = "FINAL_ZERO_SHOT" if is_medical_dataset(dataset) else "DEVELOPMENT"
    output["phase2b_checkpoint_sha256"] = selection.get("selected_checkpoint_sha256")
    if freeze_path is not None:
        output["sabra_freeze_sha256"] = sha256_file(freeze_path)
    if "phase2b_macro" in output:
        output["phase2b_metrics"] = output["phase2b_macro"]
    if "sabra_macro" in output:
        output["sabra_metrics"] = output["sabra_macro"]
    if runtime is not None:
        output["runtime"] = dict(runtime)
    return output


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
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric-mode", choices=["exact"], default="exact")
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument("--records-json", type=Path, help="debug-only precomputed records")
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
    if args.dataset not in {"MVTec", *MEDICAL_DATASETS}:
        raise SystemExit(f"unsupported canonical test dataset: {args.dataset}")
    if is_medical_dataset(args.dataset) and freeze is not None:
        validate_medical_guard(args.dataset, freeze, str(selection.get("selected_checkpoint_sha256")))
    runtime: dict[str, Any] = {"metric_mode": args.metric_mode, "pixel_stride": int(args.pixel_stride)}
    if args.records_json is not None:
        records = json.loads(args.records_json.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise SystemExit("--records-json must contain a list")
        runtime["records_json_debug"] = True
    else:
        if args.clip_asset is None:
            raise SystemExit("real model inference requires --clip-asset")
        records = run_model_records(
            args.dataset,
            args.data_root,
            selection,
            freeze,
            args.config,
            args.clip_asset,
            torch.device(args.device),
            args.method,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            runtime_stats=runtime,
        )
    result = evaluate_records(
        records,
        method=args.method,
        allow_undefined_image_metrics=is_medical_dataset(args.dataset),
    )
    _write_outputs(args.output_dir, _add_report_metadata(result, args.dataset, selection, args.sabra_freeze, runtime))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
