#!/usr/bin/env python3
"""Run the target-blind fixed VisA source gate for the H2 master arms.

This evaluator intentionally imports the historical H2 model, prompt, and
deployment path through ``train_h2_anchor_cir``.  It evaluates only the frozen
VisA source sample and never opens Medical or MVTec data.  The three E10 arms
are reported as R (native H2 control), RA (image-parameter anchor), and RCA
(anchor plus train-time CIR).  Deployment is native H2 alpha=0 for every arm;
CIR is a training-time intervention in this master experiment.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from evaluation.metrics import binary_metrics, macro_metrics
from scripts.cir_rmt import train_h2_anchor_cir as runner


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "exact_h2_anchor_cir_master_v1.json"
DEFAULT_SAMPLE = ROOT / "research_artifacts" / "cir_rmt_v2" / "pre_full_run_root_cause_lock_20260831" / "SOURCE_SAMPLE_IDENTITY.json"
DEFAULT_OUTPUT = ROOT / "research_artifacts" / "h2_anchor_cir_master_20260901"
METHODS = ("R", "RA", "RCA")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
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


def _metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    auc, ap = binary_metrics(scores.reshape(-1), labels.reshape(-1))
    return {"auroc": _finite(auc), "ap": _finite(ap)}


def _cosine_mean(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
    y = np.asarray(y, dtype=np.float64).reshape(-1, y.shape[-1])
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return float(np.mean(np.sum(x * y, axis=1) / np.maximum(denom, 1e-12)))


def _parameter_summary(
    payload: Mapping[str, Any],
    e0: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    current = payload["image_adapter"]
    initial = e0["image_adapter"]
    ref = reference["image_adapter"] if reference is not None else None
    sq_initial = 0.0
    sq_current = 0.0
    sq_reference = 0.0
    sq_from_reference = 0.0
    max_abs_initial = 0.0
    max_abs_reference = 0.0
    for name, tensor in current.items():
        value = tensor.detach().float().cpu()
        start = initial[name].detach().float().cpu()
        diff_initial = value - start
        sq_initial += float(torch.sum(diff_initial * diff_initial).item())
        sq_current += float(torch.sum(value * value).item())
        max_abs_initial = max(max_abs_initial, float(diff_initial.abs().max().item()))
        if ref is not None:
            diff_reference = value - ref[name].detach().float().cpu()
            sq_from_reference += float(torch.sum(diff_reference * diff_reference).item())
            sq_reference += float(torch.sum(ref[name].detach().float().cpu() ** 2).item())
            max_abs_reference = max(max_abs_reference, float(diff_reference.abs().max().item()))
    return {
        "parameter_scope": "image_adapter",
        "l2_from_e0": math.sqrt(sq_initial),
        "relative_l2_from_e0": math.sqrt(sq_initial) / max(math.sqrt(sq_current), 1e-12),
        "max_abs_from_e0": max_abs_initial,
        "l2_from_r_e10": math.sqrt(sq_from_reference) if ref is not None else None,
        "relative_l2_from_r_e10": math.sqrt(sq_from_reference) / max(math.sqrt(sq_reference), 1e-12) if ref is not None else None,
        "max_abs_from_r_e10": max_abs_reference if ref is not None else None,
    }


def _sample_context(sample_path: Path, h2_repo: Path) -> tuple[list[str], dict[str, list[str]], dict[str, Any]]:
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    metadata = (h2_repo / "dataset" / "hub" / "VisA.jsonl").resolve()
    if runner.sha256_file(metadata) != str(sample["manifest_sha256"]):
        raise ValueError("fixed source sample does not match the H2 VisA manifest SHA")
    selected_by_class: dict[str, list[str]] = {}
    for row in sample["selection"]:
        selected_by_class.setdefault(str(row["class_name"]), []).append(str(row["image_path"]))
    categories = [str(value) for value in sample["categories"]]
    if set(categories) != set(selected_by_class) or sum(len(v) for v in selected_by_class.values()) != int(sample["n_images"]):
        raise ValueError("fixed source sample category/count identity is invalid")
    return categories, selected_by_class, sample


def _load_checkpoint(path: Path, cfg: Mapping[str, Any], config_sha: str, epoch: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("h2_contract_sha256") != config_sha:
        raise ValueError(f"checkpoint/config SHA mismatch: {path}")
    if payload.get("epoch") != epoch:
        raise ValueError(f"source decomposition requires E{epoch} checkpoint: {path}")
    if payload.get("architecture_freeze_sha256") != cfg["architecture_freeze_sha256"]:
        raise ValueError(f"architecture freeze mismatch: {path}")
    if payload.get("deployment_alpha") != 0.0:
        raise ValueError(f"source gate requires native deployment alpha=0: {path}")
    return payload




def _evaluate_one_validated(
    method: str,
    checkpoint_path: Path,
    payload: Mapping[str, Any],
    cfg: Mapping[str, Any],
    modules: Mapping[str, Any],
    h2_dataset: Any,
    domains: Mapping[str, str],
    categories: Sequence[str],
    selected_by_class: Mapping[str, Sequence[str]],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = runner.build_model(cfg, modules, device)
    model.image_adapter.load_state_dict(payload["image_adapter"])
    model.text_adapter.load_state_dict(payload["text_adapter"])
    model.soft_prompt.load_state_dict(payload["soft_prompt"])
    model.prompt_mode = "hybrid"
    model.use_hybrid_soft_prompt = True
    model.use_soft_prompt = True
    model.hybrid_alpha_current = 0.0
    model.dfg_beta = float(payload.get("dfg_beta_current", cfg["dfg_beta"]))
    model.eval()

    metadata = str((Path(cfg["h2_repo_path"]).resolve() / cfg["manifest_path"]).resolve())
    per_class: dict[str, dict[str, Any]] = {}
    pixel_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    image_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    pooled_seg: list[np.ndarray] = []
    pooled_det: list[np.ndarray] = []
    image_paths: list[str] = []

    with torch.inference_mode():
        text_embeddings = modules["get_multiple_adapted_text_embedding"](model, "VisA", device)
        for class_name in categories:
            dataset = h2_dataset(
                data_path=str(Path(cfg["source_root"]).resolve()),
                meta_path=metadata,
                img_size=int(cfg["img_size"]),
                class_name=class_name,
            )
            selected_paths = set(str(value) for value in selected_by_class[class_name])
            indices = [index for index, row in enumerate(dataset.meta) if str(row["image_path"]) in selected_paths]
            if len(indices) != len(selected_paths):
                raise ValueError(f"fixed source selection mismatch for class {class_name}")
            loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
            class_pixels: list[np.ndarray] = []
            class_masks: list[np.ndarray] = []
            class_images: list[np.ndarray] = []
            class_labels: list[np.ndarray] = []
            class_paths: list[str] = []
            for batch in loader:
                image = batch["image"].to(device, non_blocking=device.type == "cuda")
                mask = batch["mask"].detach().cpu().numpy()[:, 0].astype(np.uint8, copy=False)
                labels = torch.as_tensor(batch["label"]).detach().cpu().numpy().astype(np.int64, copy=False)
                names = [str(value) for value in batch["class_name"]]
                if set(names) != {class_name}:
                    raise ValueError("source gate DataLoader mixed class identities")
                seg_tokens, det_tokens = model(image)
                seg_features = torch.stack(seg_tokens, dim=0)
                det_features = torch.stack(det_tokens, dim=0)
                class_text = text_embeddings[class_name].unsqueeze(dim=1).repeat(1, image.shape[0], 1, 1)
                cls_logits = torch.stack([
                    torch.matmul(det_features[index].unsqueeze(1), class_text[index]).squeeze(1)
                    for index in range(det_features.shape[0])
                ], dim=0).mean(dim=0)
                cls_prob = torch.softmax(cls_logits, dim=1)[:, 1]
                seg_prob = model.vision_text_fusion_gate_seg(
                    seg_features,
                    class_text,
                    test_mode=True,
                    domain=domains["VisA"],
                )
                pixel = seg_prob.detach().float().cpu().numpy()
                image_score = 0.9 * cls_prob.detach().float().cpu().numpy() + 0.1 * pixel.reshape(pixel.shape[0], -1).max(axis=1)
                pooled_seg.append(seg_features.detach().float().mean(dim=2).cpu().numpy())
                pooled_det.append(det_features.detach().float().cpu().numpy())
                class_pixels.append(pixel)
                class_masks.append(mask)
                class_images.append(image_score)
                class_labels.append(labels)
                class_paths.extend(str(value) for value in batch["file_name"])
                del image, seg_tokens, det_tokens, seg_features, det_features, class_text, cls_logits, cls_prob, seg_prob
            class_pixel = np.concatenate(class_pixels, axis=0)
            class_mask = np.concatenate(class_masks, axis=0)
            class_image = np.concatenate(class_images, axis=0)
            class_label = np.concatenate(class_labels, axis=0)
            class_pixel_metrics = _metrics(class_pixel, class_mask)
            class_image_metrics = _metrics(class_image, class_label)
            per_class[class_name] = {
                "n_images": int(class_label.size),
                "pixel_auroc": class_pixel_metrics["auroc"],
                "pixel_ap": class_pixel_metrics["ap"],
                "image_auroc": class_image_metrics["auroc"],
                "image_ap": class_image_metrics["ap"],
            }
            pixel_parts.append(class_pixel)
            mask_parts.append(class_mask)
            image_parts.append(class_image)
            label_parts.append(class_label)
            image_paths.extend(class_paths)
            del dataset, loader, class_pixels, class_masks, class_images, class_labels, class_pixel, class_mask, class_image, class_label
    pooled_pixels = np.concatenate(pixel_parts, axis=0)
    pooled_masks = np.concatenate(mask_parts, axis=0)
    pooled_images = np.concatenate(image_parts, axis=0)
    pooled_labels = np.concatenate(label_parts, axis=0)
    pooled_pixel_metrics = _metrics(pooled_pixels, pooled_masks)
    pooled_image_metrics = _metrics(pooled_images, pooled_labels)
    macro = macro_metrics({name: {key: value for key, value in row.items() if key in {"pixel_auroc", "pixel_ap", "image_auroc", "image_ap"}} for name, row in per_class.items()})
    arrays = {
        "seg_pooled": np.concatenate(pooled_seg, axis=1),
        "det": np.concatenate(pooled_det, axis=1),
        "image_paths": image_paths,
    }
    result = {
        "method": method,
        "epoch": int(payload["epoch"]),
        "n_images": int(pooled_labels.size),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": runner.sha256_file(checkpoint_path),
        "pixel_auroc": pooled_pixel_metrics["auroc"],
        "pixel_ap": pooled_pixel_metrics["ap"],
        "image_auroc": pooled_image_metrics["auroc"],
        "image_ap": pooled_image_metrics["ap"],
        "macro_pixel_auroc": macro["pixel_auroc"],
        "macro_pixel_ap": macro["pixel_ap"],
        "macro_image_auroc": macro["image_auroc"],
        "macro_image_ap": macro["image_ap"],
        "deployment": "native_h2_alpha0",
        "rmt_inference": "NOT_APPLIED; RCA CIR IS TRAINING_ONLY",
    }
    del model, text_embeddings, pixel_parts, mask_parts, image_parts, label_parts, pooled_pixels, pooled_masks, pooled_images, pooled_labels
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, per_class, arrays


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = dict(cfg)
    config_sha = runner.sha256_file(config_path)
    h2_repo = Path(cfg["h2_repo_path"]).resolve()
    modules = dict(runner._load_h2_modules(h2_repo))
    h2_utils = __import__("utils", fromlist=["get_multiple_adapted_text_embedding"])
    modules["get_multiple_adapted_text_embedding"] = h2_utils.get_multiple_adapted_text_embedding
    dataset_module = __import__("dataset", fromlist=["BaseSingleClassDataset", "DOMAINS"])
    h2_dataset = dataset_module.BaseSingleClassDataset
    domains = dataset_module.DOMAINS
    categories, selected_by_class, sample = _sample_context(args.source_sample.resolve(), h2_repo)
    e0_path = args.e0.resolve()
    if not e0_path.is_file():
        raise FileNotFoundError(e0_path)
    e0 = torch.load(e0_path, map_location="cpu", weights_only=False)
    if e0.get("snapshot_kind") != "exact_h2_initialization_only" or e0.get("h2_contract_sha256") != config_sha:
        raise ValueError("E0 source-gate identity mismatch")
    epochs = tuple(sorted({int(epoch) for epoch in args.epochs}))
    if not epochs or any(epoch < 1 for epoch in epochs):
        raise ValueError("--epochs must contain positive epoch numbers")
    methods = tuple(dict.fromkeys(str(method) for method in args.methods))
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"--methods must be drawn from {METHODS}")
    checkpoints = {
        (method, epoch): (Path(args.run_root).resolve() / method / f"adapter_{epoch}.pth")
        for method in methods
        for epoch in epochs
    }
    missing_checkpoints = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing_checkpoints and not args.allow_missing:
        raise FileNotFoundError(missing_checkpoints[0])
    checkpoints = {key: path for key, path in checkpoints.items() if path.is_file()}
    payloads = {
        key: _load_checkpoint(path, cfg, config_sha, key[1])
        for key, path in checkpoints.items()
    }
    device = torch.device(args.device)
    result_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    snapshots: dict[tuple[str, int], dict[str, Any]] = {}
    for epoch in epochs:
        for method in methods:
            if (method, epoch) not in checkpoints:
                continue
            result, per_class, arrays = _evaluate_one_validated(
                method,
                checkpoints[(method, epoch)],
                payloads[(method, epoch)],
                cfg,
                modules,
                h2_dataset,
                domains,
                categories,
                selected_by_class,
                device,
                int(args.batch_size),
            )
            result_rows.append(result)
            snapshots[(method, epoch)] = arrays
            for class_name in categories:
                class_rows.append({"method": method, "epoch": epoch, "class_name": class_name, **per_class[class_name], "checkpoint_sha256": result["checkpoint_sha256"]})

    parameter_rows = []
    for epoch in epochs:
        reference = payloads.get(("R", epoch))
        for method in methods:
            if (method, epoch) not in payloads:
                continue
            summary = _parameter_summary(payloads[(method, epoch)], e0, reference if method != "R" else None)
            row = next(row for row in result_rows if row["method"] == method and row["epoch"] == epoch)
            parameter_rows.append({"method": method, "epoch": epoch, **summary, "checkpoint_sha256": row["checkpoint_sha256"]})

    feature_rows = []
    for epoch in epochs:
        baseline = snapshots.get(("R", epoch))
        if baseline is None:
            continue
        for method in methods:
            if (method, epoch) not in snapshots:
                continue
            current = snapshots[(method, epoch)]
            for signal in ("seg_pooled", "det"):
                x = baseline[signal]
                y = current[signal]
                feature_rows.append({
                    "method": method,
                    "epoch": epoch,
                    "signal": signal,
                    "mean_cosine_vs_r": 1.0 if method == "R" else _cosine_mean(x, y),
                    "mean_abs_delta_vs_r": 0.0 if method == "R" else float(np.mean(np.abs(x.astype(np.float64) - y.astype(np.float64)))),
                    "n_images": int(x.shape[1]),
                })

    output = args.output.resolve()
    _write_csv(output / "SOURCE_DECOMPOSITION.csv", [
        "method", "epoch", "n_images", "checkpoint_sha256", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap",
        "macro_pixel_auroc", "macro_pixel_ap", "macro_image_auroc", "macro_image_ap", "deployment", "rmt_inference",
    ], result_rows)
    _write_csv(output / "SOURCE_DECOMPOSITION_PER_CLASS.csv", [
        "method", "epoch", "class_name", "n_images", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap", "checkpoint_sha256",
    ], class_rows)
    _write_csv(output / "SOURCE_PARAMETER_DRIFT.csv", [
        "method", "epoch", "parameter_scope", "l2_from_e0", "relative_l2_from_e0", "max_abs_from_e0",
        "l2_from_r_e10", "relative_l2_from_r_e10", "max_abs_from_r_e10", "checkpoint_sha256",
    ], parameter_rows)
    _write_csv(output / "SOURCE_FEATURE_DRIFT.csv", [
        "method", "epoch", "signal", "mean_cosine_vs_r", "mean_abs_delta_vs_r", "n_images",
    ], feature_rows)
    status = {
        "status": "PASS",
        "gate": "SOURCE_ONLY_TARGET_BLIND",
        "epochs": list(epochs),
        "source_dataset": "VisA",
        "source_sample_path": str(args.source_sample.resolve()),
        "source_sample_sha256": runner.sha256_file(args.source_sample.resolve()),
        "source_sample_n_images": int(sample["n_images"]),
        "source_sample_categories": categories,
        "source_sample_selection_note": sample.get("protocol_note"),
        "methods": list(methods),
        "missing_checkpoints": missing_checkpoints,
        "deployment_alpha": 0.0,
        "rmt_inference_effect": "NOT_MEASURED_IN_THIS_MASTER; RCA CIR IS TRAINING_ONLY",
        "anchor_gradient_ratio": "NOT_MEASURED; trainer records loss and CIR telemetry, not per-objective gradient decomposition",
        "medical_evaluation": "NOT_RUN",
        "mvtec_evaluation": "NOT_RUN",
        "config_sha256": config_sha,
        "architecture_freeze_sha256": cfg["architecture_freeze_sha256"],
        "checkpoint_sha256": {f"{row['method']}_E{row['epoch']}": row["checkpoint_sha256"] for row in result_rows},
        "finite_metrics": all(all(value is not None for value in (row["pixel_auroc"], row["pixel_ap"], row["image_auroc"], row["image_ap"])) for row in result_rows),
        "source_gate_rule": "fixed sample identity, exact native-H2 alpha0 metrics, complete R/RA/RCA candidate checkpoint and identity audit; no pre-registered performance threshold",
    }
    _write_json(output / "SOURCE_DECOMPOSITION_STATUS.json", status)
    report_lines = [
        "# H2 Master Source Decomposition",
        "",
        "Status: PASS (source-only audit).",
        "",
        "The gate uses the frozen deterministic VisA sample only. All arms use the historical H2 native deployment with alpha=0; RCA CIR is train-time only. Therefore this file does not estimate an inference-time RMT effect.",
        "",
        "| method | epoch | pixel AUROC | pixel AP | image AUROC | image AP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        report_lines.append(f"| {row['method']} | {row['epoch']} | {row['pixel_auroc']:.8f} | {row['pixel_ap']:.8f} | {row['image_auroc']:.8f} | {row['image_ap']:.8f} |")
    report_lines.extend([
        "",
        "R is the native H2 control. RA isolates the image-parameter anchor training effect relative to R. RCA adds train-time CIR relative to RA.",
        "",
        "Anchor-gradient ratio: NOT_MEASURED. The trainer records total loss components and RCA peer/delta telemetry, but not separate per-objective gradient norms.",
        "",
        "Medical and MVTec: NOT_RUN. Candidate continuation decisions are target-blind and must be recorded before target evaluation.",
        "",
        "See SOURCE_DECOMPOSITION.csv, SOURCE_DECOMPOSITION_PER_CLASS.csv, SOURCE_PARAMETER_DRIFT.csv, and SOURCE_FEATURE_DRIFT.csv for the compact evidence tables.",
    ])
    (output / "SOURCE_DECOMPOSITION.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status["status"], "output": str(output), "rows": result_rows}, sort_keys=True, default=_json_default))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs" / "h2_anchor_cir_master_20260901")
    parser.add_argument("--e0", type=Path, default=ROOT / "runs" / "h2_anchor_cir_master_20260901" / "common" / "e0.pth")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, nargs="+", default=[10])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
