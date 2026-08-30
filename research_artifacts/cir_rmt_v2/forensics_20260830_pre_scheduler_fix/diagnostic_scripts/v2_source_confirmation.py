#!/usr/bin/env python3
"""CIR_DFG_RMT_V2 bounded, preregistered VisA source confirmation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from evaluation.evaluator import evaluate_records, image_score
from evaluation.metrics import binary_average_precision, binary_auroc
from model.phase2b_runtime import build_phase2b_trainable, configure_canonical_fp32, deploy_native_logits
from tools.cir_rmt.core import (
    V1_TRANSPORT_DIRECTION,
    V2_TRANSPORT_DIRECTION,
    cir_logits_from_native_weights,
    transport_pair,
)
from tools.cir_rmt.identity import config_sha256, git_identity, load_cir_config, release_identity_fields
from tools.cir_rmt.runtime import forward_cir

ALPHAS = (0.0, 0.10, 0.25, 0.50)
V1_CONTROL_ALPHA = 0.50
IMAGE_SIZE = 518
PATCH_SIDE = 37
NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
NORM_STD = (0.26862954, 0.26130258, 0.27577711)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return {name: float(np.quantile(array, q)) for name, q in (("p01", 0.01), ("p05", 0.05), ("p50", 0.50), ("p95", 0.95), ("p99", 0.99))}


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty_like(values, dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _spearman(scores: np.ndarray, labels: np.ndarray) -> float | None:
    left = _rankdata(scores)
    right = _rankdata(labels)
    denom = float(left.std() * right.std())
    if denom == 0.0:
        return None
    return float(np.mean((left - left.mean()) * (right - right.mean())) / denom)


def _gt_stats(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=np.int8).reshape(-1)
    normal = scores[target == 0]
    anomaly = scores[target == 1]
    return {
        "normal_mean": float(normal.mean()),
        "anomaly_mean": float(anomaly.mean()),
        "normal_median": float(np.median(normal)),
        "anomaly_median": float(np.median(anomaly)),
        "normal_abs_mean": float(np.abs(normal).mean()),
        "anomaly_abs_mean": float(np.abs(anomaly).mean()),
        "normal_abs_median": float(np.median(np.abs(normal))),
        "anomaly_abs_median": float(np.median(np.abs(anomaly))),
        "signed_mean_diff_anomaly_minus_normal": float(anomaly.mean() - normal.mean()),
        "signed_median_diff_anomaly_minus_normal": float(np.median(anomaly) - np.median(normal)),
        "signed_auroc": float(binary_auroc(scores, target)),
        "absolute_auroc": float(binary_auroc(np.abs(scores), target)),
        "signed_ap": float(binary_average_precision(scores, target)),
        "absolute_ap": float(binary_average_precision(np.abs(scores), target)),
        "spearman_signed": _spearman(scores, target),
        "spearman_absolute": _spearman(np.abs(scores), target),
        "finite": bool(np.isfinite(scores).all()),
    }


def _metric_macro(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = evaluate_records(records, method="phase2b")
    return {**evaluated["macro"], "per_class": evaluated["per_class"]}


def _select_rows(manifest: Path, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_class: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        by_class.setdefault(str(row["class_name"]), {}).setdefault(int(row["label"]), []).append(row)
    rng = random.Random(int(seed))
    selected: list[dict[str, Any]] = []
    shortfall: dict[str, dict[str, int]] = {}
    for class_name in sorted(by_class):
        for label in (0, 1):
            candidates = sorted(by_class[class_name].get(label, []), key=lambda item: str(item["image_path"]))
            take = min(5, len(candidates))
            if take < 5:
                shortfall.setdefault(class_name, {})[str(label)] = take
            chosen = rng.sample(candidates, take)
            selected.extend(sorted(chosen, key=lambda item: str(item["image_path"])))
    selected.sort(key=lambda item: (str(item["class_name"]), int(item["label"]), str(item["image_path"])))
    if len(selected) < 2 or not any(int(row["label"]) == 0 for row in selected) or not any(int(row["label"]) == 1 for row in selected):
        raise RuntimeError("VisA confirmation subset lacks both labels")
    return selected


def _load_batch(rows: list[dict[str, Any]], root: Path, image_tf: Any, mask_tf: Any) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str]]:
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    labels: list[int] = []
    names: list[str] = []
    for row in rows:
        with Image.open(root / row["image_path"]) as handle:
            images.append(image_tf(handle.convert("RGB")))
        if int(row["label"]):
            with Image.open(root / row["mask_path"]) as handle:
                masks.append(mask_tf(handle.convert("L")).gt(0).to(torch.float32))
        else:
            masks.append(torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32))
        labels.append(int(row["label"]))
        names.append(str(row["class_name"]))
    return torch.stack(images), torch.stack(masks), labels, names


def _delta_map(delta: torch.Tensor) -> torch.Tensor:
    patch = delta.mean(dim=(0, 3)).reshape(delta.shape[1], 1, PATCH_SIDE, PATCH_SIDE)
    return F.interpolate(patch, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True)[:, 0]


def _peer_diagnostics(output: Any, config: Mapping[str, Any], totals: dict[str, int], arrays: dict[str, list[np.ndarray]]) -> None:
    arrays["mad"].append(output.delta_stats["mad"].detach().float().cpu().numpy().reshape(-1))
    arrays["z"].append(output.delta_stats["z"].detach().float().cpu().numpy().reshape(-1))
    arrays["delta_raw"].append(output.delta.detach().float().cpu().numpy().reshape(-1))
    arrays["candidate"].append(output.peer_candidate_count.detach().cpu().numpy().reshape(-1))
    arrays["valid"].append(output.peer_valid.detach().cpu().numpy().reshape(-1))
    arrays["peer_margins"].append(output.peer_margins.detach().float().cpu().numpy().reshape(-1))
    peer_indices = output.peer_indices.detach().cpu().numpy()
    radius = int(config["rmt_spatial_radius"])
    for batch_index in range(peer_indices.shape[0]):
        for patch_index in range(peer_indices.shape[1]):
            values = peer_indices[batch_index, patch_index].tolist()
            totals["slots"] += len(values)
            totals["self"] += sum(int(value == patch_index) for value in values)
            totals["duplicate"] += len(values) - len(set(values))
            py, px = divmod(patch_index, PATCH_SIDE)
            totals["spatial"] += sum(int(max(abs(divmod(value, PATCH_SIDE)[0] - py), abs(divmod(value, PATCH_SIDE)[1] - px)) <= radius) for value in values)


def _transport_diagnostics(output: Any, alpha: float, direction: str) -> dict[str, float]:
    native = output.native_weights.detach().float()
    delta = output.delta.detach().float()
    native_patch = native.unsqueeze(2).expand(-1, -1, delta.shape[2], -1, -1)
    normal, abnormal = transport_pair(native_patch[..., 0], native_patch[..., 1], delta, alpha, transport_direction=direction)
    native_normal = native_patch[..., 0]
    native_abnormal = native_patch[..., 1]
    native_n_entropy = (-(native_normal * native_normal.clamp_min(1e-8).log()).sum(-1)).mean()
    native_a_entropy = (-(native_abnormal * native_abnormal.clamp_min(1e-8).log()).sum(-1)).mean()
    transport_n_entropy = (-(normal * normal.clamp_min(1e-8).log()).sum(-1)).mean()
    transport_a_entropy = (-(abnormal * abnormal.clamp_min(1e-8).log()).sum(-1)).mean()
    return {
        "weight_l1_shift": float(0.5 * ((normal - native_normal).abs().mean() + (abnormal - native_abnormal).abs().mean())),
        "normal_entropy_shift": float(transport_n_entropy - native_n_entropy),
        "abnormal_entropy_shift": float(transport_a_entropy - native_a_entropy),
        "normal_active_fraction": float((normal - native_normal).abs().sum(-1).gt(1e-7).float().mean()),
        "abnormal_active_fraction": float((abnormal - native_abnormal).abs().sum(-1).gt(1e-7).float().mean()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config = load_cir_config(args.config)
    if str(config["arch_id"]) != "CIR_DFG_RMT_V2":
        raise ValueError("V2 source confirmation requires CIR_DFG_RMT_V2 config")
    if str(config.get("rmt_transport_direction")) != V2_TRANSPORT_DIRECTION:
        raise ValueError("V2 source confirmation requires abnormal_minus_normal_plus direction")
    visa_root = args.visa_root.expanduser().resolve()
    manifest = root / "dataset/hub/VisA.jsonl"
    asset = args.clip_asset.expanduser().resolve()
    if not asset.is_file() or not visa_root.is_dir():
        raise FileNotFoundError("real VisA/CLIP assets are required")
    rows = _select_rows(manifest, args.seed)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print("=" * 60, flush=True)
    print("ARCH       : CIR_DFG_RMT_V2", flush=True)
    print("EXPERIMENT : SOURCE SIGN CONFIRMATION", flush=True)
    print("SOURCE     : VisA", flush=True)
    print("DIRECTION  : abnormal - alpha*delta", flush=True)
    print("             normal   + alpha*delta", flush=True)
    print(f"SAMPLES    : {len(rows)}", flush=True)
    print(f"CONFIG     : {config_sha256(config)[:12]}", flush=True)
    print(f"GIT        : {git_identity()['head'][:12]}", flush=True)
    print(f"DEVICE     : {args.device}", flush=True)
    print("=" * 60, flush=True)
    per_class: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = per_class.setdefault(str(row["class_name"]), {"normal": 0, "anomaly": 0})
        bucket["anomaly" if int(row["label"]) else "normal"] += 1
    subset_manifest = {
        "architecture": "CIR_DFG_RMT_V2",
        "status": "SOURCE_CONFIRMATION_SUBSET",
        "seed": int(args.seed),
        "selection": "class-stratified, five normal and five anomaly per class where available, sampled from sorted paths",
        "total": len(rows),
        "normal": sum(int(row["label"]) == 0 for row in rows),
        "anomaly": sum(int(row["label"]) == 1 for row in rows),
        "classes": sorted(per_class),
        "per_class": per_class,
        "manifest": str(manifest),
        "visa_root": str(visa_root),
        "rows": rows,
        "identity": release_identity_fields(config),
        "clip_asset": str(asset),
    }
    _write_json(output_root / "visa_subset_manifest.json", subset_manifest)

    image_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    configure_canonical_fp32()
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)
    parent_config = json.loads((root / str(config["parent_config_path"])).read_text(encoding="utf-8"))
    parent_config["dataset"] = "VisA"
    model = build_phase2b_trainable(parent_config, asset, device)
    model.eval()

    records_by_alpha: dict[float, list[dict[str, Any]]] = {alpha: [] for alpha in ALPHAS}
    component_records: dict[float, dict[str, list[dict[str, Any]]]] = {
        alpha: {"classification_only": [], "pixel_max": [], "deployed_final": []} for alpha in ALPHAS
    }
    v1_records: list[dict[str, Any]] = []
    delta_maps: list[np.ndarray] = []
    gt_masks: list[np.ndarray] = []
    delta_arrays: dict[str, list[np.ndarray]] = {name: [] for name in ("mad", "z", "delta_raw", "candidate", "valid", "peer_margins")}
    peer_totals = {"slots": 0, "self": 0, "duplicate": 0, "spatial": 0}
    margin_changes: dict[float, list[np.ndarray]] = {alpha: [] for alpha in ALPHAS}
    v1_margin_changes: list[np.ndarray] = []
    transport_rows: dict[float, list[dict[str, float]]] = {alpha: [] for alpha in ALPHAS}
    nonfinite = 0
    cache_check_max = None
    cache_check_mean = None
    started = time.perf_counter()
    for start in range(0, len(rows), int(args.batch_size)):
        batch_rows = rows[start : start + int(args.batch_size)]
        images, masks, labels, names = _load_batch(batch_rows, visa_root, image_tf, mask_tf)
        output = forward_cir(model, images.to(device), names, device, config, domain="Industrial", require_grad=False, dataset_name="VisA")
        _peer_diagnostics(output, config, peer_totals, delta_arrays)
        delta_maps.append(_delta_map(output.delta.detach().float()).cpu().numpy())
        gt_masks.append(masks[:, 0].numpy().astype(np.int8))
        if not torch.isfinite(output.delta).all() or not torch.isfinite(output.native_logits).all():
            nonfinite += 1
        with torch.inference_mode():
            _, native_deployed_logits = deploy_native_logits(output.native_logits, image_size=IMAGE_SIZE, domain="Industrial")
            native_margin_map = native_deployed_logits[:, 1] - native_deployed_logits[:, 0]
            for alpha in ALPHAS:
                logits, _ = cir_logits_from_native_weights(
                    output.seg_features, output.text_features, output.native_weights, output.delta,
                    alpha, score_mode="optimized", eps=float(config["rmt_eps"]), transport_direction=V2_TRANSPORT_DIRECTION,
                )
                probabilities, deployed_logits = deploy_native_logits(logits, image_size=IMAGE_SIZE, domain="Industrial")
                maps = probabilities[:, 1]
                margin_changes[alpha].append((deployed_logits[:, 1] - deployed_logits[:, 0] - native_margin_map).detach().float().cpu().numpy())
                if alpha == float(config["rmt_transport_alpha"]):
                    cache_check = (logits - output.cir_logits).detach().float().abs()
                    cache_check_max = float(cache_check.max())
                    cache_check_mean = float(cache_check.mean())
                for index, row in enumerate(batch_rows):
                    pixel_scores = maps[index].detach().float().cpu().numpy().reshape(-1)
                    pixel_labels = masks[index].numpy().astype(np.int8).reshape(-1)
                    cls = float(output.classification_probability[index].detach().float().cpu())
                    pmax = float(pixel_scores.max())
                    final_score = float(image_score(cls, pmax, "Industrial"))
                    record = {"class_name": str(row["class_name"]), "pixel_scores": pixel_scores, "pixel_labels": pixel_labels, "image_scores": [final_score], "image_labels": [int(row["label"])], "image_path": str(row["image_path"])}
                    records_by_alpha[alpha].append(record)
                    component_records[alpha]["classification_only"].append({**record, "image_scores": [cls]})
                    component_records[alpha]["pixel_max"].append({**record, "image_scores": [pmax]})
                    component_records[alpha]["deployed_final"].append(record)
                transport_rows[alpha].append(_transport_diagnostics(output, alpha, V2_TRANSPORT_DIRECTION))
            v1_logits, _ = cir_logits_from_native_weights(
                output.seg_features, output.text_features, output.native_weights, output.delta,
                V1_CONTROL_ALPHA, score_mode="optimized", eps=float(config["rmt_eps"]), transport_direction=V1_TRANSPORT_DIRECTION,
            )
            v1_probabilities, v1_deployed_logits = deploy_native_logits(v1_logits, image_size=IMAGE_SIZE, domain="Industrial")
            v1_margin_changes.append((v1_deployed_logits[:, 1] - v1_deployed_logits[:, 0] - native_margin_map).detach().float().cpu().numpy())
            for index, row in enumerate(batch_rows):
                pixel_scores = v1_probabilities[index, 1].detach().float().cpu().numpy().reshape(-1)
                pmax = float(pixel_scores.max()); cls = float(output.classification_probability[index].detach().float().cpu())
                v1_records.append({"class_name": str(row["class_name"]), "pixel_scores": pixel_scores, "pixel_labels": masks[index].numpy().astype(np.int8).reshape(-1), "image_scores": [float(image_score(cls, pmax, "Industrial"))], "image_labels": [int(row["label"])], "image_path": str(row["image_path"])})
        del output, images, masks
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started

    metrics = {str(alpha): _metric_macro(records_by_alpha[alpha]) for alpha in ALPHAS}
    components = {str(alpha): {name: _metric_macro(records) for name, records in component_records[alpha].items()} for alpha in ALPHAS}
    v1_metrics = _metric_macro(v1_records)
    delta = np.concatenate(delta_maps, axis=0).reshape(-1)
    gt = np.concatenate(gt_masks, axis=0).reshape(-1)
    diagnostics = {
        "delta_gt": _gt_stats(delta, gt),
        "delta_quantiles": _quantiles(delta),
        "delta_abs_gt_095": float(np.mean(np.abs(delta) > 0.95)),
        "delta_abs_gt_099": float(np.mean(np.abs(delta) > 0.99)),
        "mad_quantiles": _quantiles(np.concatenate(delta_arrays["mad"])),
        "near_zero_mad_fraction": float(np.mean(np.concatenate(delta_arrays["mad"]) <= float(config["rmt_eps"]))),
        "z_quantiles": _quantiles(np.concatenate(delta_arrays["z"])),
        "peer_candidate_quantiles": _quantiles(np.concatenate(delta_arrays["candidate"])),
        "invalid_peer_fraction": float(1.0 - np.mean(np.concatenate(delta_arrays["valid"]).astype(np.float64))),
        "duplicate_peer_violation_fraction": float(peer_totals["duplicate"] / max(peer_totals["slots"], 1)),
        "self_peer_violation_fraction": float(peer_totals["self"] / max(peer_totals["slots"], 1)),
        "spatial_violation_fraction": float(peer_totals["spatial"] / max(peer_totals["slots"], 1)),
        "nonfinite_delta_stats": {name: int(np.count_nonzero(~np.isfinite(np.concatenate(values)))) for name, values in delta_arrays.items() if name not in {"candidate", "valid"}},
        "nonfinite_batches": int(nonfinite),
        "throughput_images_per_sec": float(len(rows) / max(elapsed, 1e-9)),
        "delta_map_definition": "mean over stage/group axes, bilinear 37x37 to 518x518",
    }
    margin_summary: dict[str, Any] = {}
    for alpha in ALPHAS:
        changes = np.concatenate(margin_changes[alpha], axis=0).reshape(-1)
        margin_summary[str(alpha)] = _gt_stats(changes, gt)
    margin_summary["v1_direction_alpha_0.5"] = _gt_stats(np.concatenate(v1_margin_changes, axis=0).reshape(-1), gt)
    transport_summary = {str(alpha): {key: float(np.mean([row[key] for row in rows_for_alpha])) for key in rows_for_alpha[0]} for alpha, rows_for_alpha in transport_rows.items()}
    baseline = metrics["0.0"]
    improvements = {}
    for alpha in ALPHAS[1:]:
        row = metrics[str(alpha)]
        improvements[str(alpha)] = {
            "pixel_auroc_improves": float(row["pixel_auroc"]) > float(baseline["pixel_auroc"]),
            "pixel_ap_improves": float(row["pixel_ap"]) > float(baseline["pixel_ap"]),
            "image_auroc_drop": float(baseline["image_auroc"]) - float(row["image_auroc"]),
            "image_ap_drop": float(baseline["image_ap"]) - float(row["image_ap"]),
        }
    both_pixel = [str(alpha) for alpha in ALPHAS[1:] if improvements[str(alpha)]["pixel_auroc_improves"] and improvements[str(alpha)]["pixel_ap_improves"]]
    image_safe = [alpha for alpha in both_pixel if improvements[alpha]["image_auroc_drop"] <= 0.02 and improvements[alpha]["image_ap_drop"] <= 0.02]
    if len(both_pixel) >= 2 and image_safe:
        decision = "V2_SIGN_CONFIRMED"
    elif len(both_pixel) >= 2:
        decision = "LOCALIZATION_IMAGE_CONFLICT"
    elif len(both_pixel) < 2:
        decision = "SIGN_NOT_CONFIRMED"
    else:
        decision = "INCONCLUSIVE"
    decision_payload = {
        "decision": decision,
        "rule": "two or more nonzero alphas must improve both pixel metrics; one must keep image AUROC/AP drops <=0.02",
        "baseline_alpha": 0.0,
        "both_pixel_improvements": both_pixel,
        "image_safe_candidates": image_safe,
        "improvements": improvements,
        "architecture_freeze_allowed": decision == "V2_SIGN_CONFIRMED",
        "alpha_status": "PROVISIONAL",
        "release_lock": False,
    }
    result = {
        "architecture": "CIR_DFG_RMT_V2",
        "experiment": "SOURCE SIGN CONFIRMATION",
        "source": "VisA",
        "direction": V2_TRANSPORT_DIRECTION,
        "identity": release_identity_fields(config),
        "git": git_identity(),
        "clip_asset": str(asset),
        "visa_root": str(visa_root),
        "subset": subset_manifest,
        "alpha_grid": list(ALPHAS),
        "metrics": metrics,
        "image_score_decomposition": components,
        "v1_direction_control_alpha_0.5": v1_metrics,
        "delta_diagnostics": diagnostics,
        "transport_diagnostics": transport_summary,
        "transport_margin_change": margin_summary,
        "cache_check": {"ordinary_forward_vs_cached_alpha_0.5_max_abs": cache_check_max, "ordinary_forward_vs_cached_alpha_0.5_mean_abs": cache_check_mean, "allclose_atol_1e-5": bool(cache_check_max is not None and cache_check_max <= 1e-5)},
        "decision": decision_payload,
    }
    _write_json(output_root / "alpha_results.json", {"identity": result["identity"], "alpha_grid": result["alpha_grid"], "metrics": metrics, "v1_direction_control_alpha_0.5": v1_metrics, "cache_check": result["cache_check"]})
    with (output_root / "alpha_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["alpha", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for alpha in ALPHAS:
            writer.writerow({"alpha": alpha, **{key: metrics[str(alpha)][key] for key in fields[1:]}})
    _write_json(output_root / "delta_audit.json", {"identity": result["identity"], "delta_diagnostics": diagnostics, "transport_margin_change": margin_summary})
    _write_json(output_root / "image_score_decomposition.json", {"identity": result["identity"], "decomposition": components})
    _write_json(output_root / "decision.json", {"identity": result["identity"], **decision_payload})
    report_lines = [
        "# CIR_DFG_RMT_V2 source confirmation",
        "",
        "Status: " + decision,
        "",
        "This bounded confirmation uses VisA only, the fixed class-stratified 120-image subset, and the preregistered alpha grid 0/0.10/0.25/0.50. The V2 direction is abnormal - alpha*delta; normal + alpha*delta. Alpha remains PROVISIONAL and release lock remains FALSE.",
        "",
        "| alpha | pixel AUROC | pixel AP | image AUROC | image AP |",
        "|---:|---:|---:|---:|---:|",
    ]
    for alpha in ALPHAS:
        row = metrics[str(alpha)]
        report_lines.append(f"| {alpha:.2f} | {row['pixel_auroc']:.6f} | {row['pixel_ap']:.6f} | {row['image_auroc']:.6f} | {row['image_ap']:.6f} |")
    report_lines += ["", "Decision rule: " + decision_payload["rule"], "", "V1 terminal remains immutable; no release gate or full training was launched."]
    (output_root / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cir_dfg_rmt_v2.json"))
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/cir_rmt/CIR_DFG_RMT_V2/source_confirmation"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"decision": result["decision"]["decision"], "subset_total": result["subset"]["total"], "metrics": {alpha: {key: result["metrics"][alpha][key] for key in ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")} for alpha in result["metrics"]}, "output_root": str(args.output_root)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
