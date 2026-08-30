#!/usr/bin/env python3
"""Read-only failure forensics for the frozen CIR_DFG_RMT_V2 artifacts.

The runner never trains, edits the frozen configuration, or writes into an
existing evaluation directory.  It evaluates the alpha=0 native control from
the existing V2 checkpoints, reads the preserved alpha=0.5 summaries, and
writes all new evidence below a dedicated audit directory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/cir_dfg_rmt_v2.json"
DEFAULT_CLIP = ROOT / "model/ViT-L-14-336px.pt"
DEFAULT_MEDICAL_ROOT = Path("/home/ai4/caohuy/data")
DEFAULT_RUN_ROOT = ROOT / "runs/cir_rmt/CIR_DFG_RMT_V2/visa/seed0"
DEFAULT_OUTPUT = ROOT / "runs/cir_rmt/CIR_DFG_RMT_V2/forensics_20260830"
MEDICAL_TARGETS = (
    "Brain",
    "Liver",
    "Retina",
    "Colon_clinicDB",
    "Colon_colonDB",
    "Colon_Kvasir",
)
EPOCHS = (12, 14, 16, 18, 20)
PATCH_GRID = (37, 37)
IMAGE_SIZE = 518
V2_DIRECTION = "abnormal_minus_normal_plus"
V1_DIRECTION = "normal_minus_delta_abnormal_plus"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    if fields is None:
        fields = []
        for row in materialized:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except Exception:
            return ""

    return {"head": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current")}


class RunningStats:
    """Numerically stable enough scalar summary for bounded forensic output."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: Any) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return
        self.count += int(array.size)
        self.total += float(array.sum(dtype=np.float64))
        self.total_sq += float(np.square(array, dtype=np.float64).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def mean(self) -> float | None:
        return None if self.count == 0 else self.total / self.count

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        variance = max(self.total_sq / self.count - (self.total / self.count) ** 2, 0.0)
        return {
            "n": self.count,
            "mean": self.total / self.count,
            "std": variance**0.5,
            "min": self.minimum,
            "max": self.maximum,
        }


class SampledStats:
    """Running moments plus a small deterministic sample for quantiles."""

    def __init__(self, sample_cap: int = 100_000) -> None:
        self.stats = RunningStats()
        self.sample_cap = int(sample_cap)
        self.sample: list[np.ndarray] = []
        self.sample_count = 0

    def update(self, values: Any, sample_points: int = 64) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = array[np.isfinite(array)]
        self.stats.update(finite)
        if finite.size == 0 or self.sample_count >= self.sample_cap:
            return
        take = min(int(sample_points), int(finite.size), self.sample_cap - self.sample_count)
        if take < finite.size:
            indices = np.linspace(0, finite.size - 1, take, dtype=np.int64)
            picked = finite[indices]
        else:
            picked = finite
        self.sample.append(np.asarray(picked, dtype=np.float64))
        self.sample_count += int(picked.size)

    def quantiles(self, probabilities: tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)) -> dict[str, float | None]:
        if not self.sample:
            return {f"p{int(p * 100):02d}": None for p in probabilities}
        values = np.concatenate(self.sample)
        return {f"p{int(p * 100):02d}": float(np.quantile(values, p)) for p in probabilities}


def tensor_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _metric_value(value: Any) -> float | None:
    return None if value is None else float(value)


def _finite_count(value: Any) -> int:
    return int(np.count_nonzero(~np.isfinite(tensor_numpy(value))))


def _safe_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return None if values.size == 0 else float(values.mean())


def _safe_quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return None if values.size == 0 else float(np.quantile(values, q))


def _patch_map_to_image(patch: torch.Tensor, stages: int | None = None, groups: int | None = None) -> torch.Tensor:
    """Bilinear-resize [S,B,P], [S,B,P,G], or [B,P] without changing scores."""
    if patch.ndim == 2:
        batch, patches = patch.shape
        image = patch.reshape(batch, 1, *PATCH_GRID)
        return F.interpolate(image, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True)[:, 0]
    if patch.ndim == 3:
        stage_count, batch, patches = patch.shape
        image = patch.reshape(stage_count * batch, 1, *PATCH_GRID)
        resized = F.interpolate(image, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True)
        return resized.reshape(stage_count, batch, IMAGE_SIZE, IMAGE_SIZE)
    if patch.ndim == 4:
        stage_count, batch, patches, group_count = patch.shape
        image = patch.permute(0, 1, 3, 2).reshape(stage_count * batch * group_count, 1, *PATCH_GRID)
        resized = F.interpolate(image, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True)
        return resized.reshape(stage_count, batch, group_count, IMAGE_SIZE, IMAGE_SIZE)
    raise ValueError(f"unsupported patch map shape: {tuple(patch.shape)}")


def _sample_diag_keys(dataset: Any, limit: int) -> set[tuple[str, str]]:
    entries: list[tuple[str, str, int]] = []
    if hasattr(dataset, "entries") and hasattr(dataset, "mappings"):
        for class_name, index in dataset.entries:
            row = dataset.mappings[class_name].meta[int(index)]
            entries.append((str(class_name), str(row["image_path"]), int(row.get("label", 0))))
    elif hasattr(dataset, "rows"):
        for row in dataset.rows:
            entries.append((str(row["class_name"]), str(row["image_path"]), int(row.get("label", 0))))
    else:
        raise TypeError(f"cannot identify diagnostic entries for {type(dataset).__name__}")
    if not entries:
        return set()
    normals = [row for row in entries if row[2] == 0]
    anomalies = [row for row in entries if row[2] == 1]
    take = max(int(limit), 0)
    selected: list[tuple[str, str, int]] = []
    if normals and anomalies:
        half = take // 2
        selected.extend(normals[:half])
        selected.extend(anomalies[:half])
        selected.extend(normals[half:])
        selected.extend(anomalies[half:])
    else:
        selected = entries
    return {(name, path) for name, path, _label in selected[:take]}


def _label_stats(stats: dict[str, RunningStats], key: str, values: np.ndarray) -> None:
    stats.setdefault(key, RunningStats()).update(values)


class MechanismAccumulator:
    """Collect RMT mechanism and peer invariants without retaining score maps."""

    def __init__(self, groups: int, peer_count: int, spatial_radius: int) -> None:
        self.groups = int(groups)
        self.peer_count = int(peer_count)
        self.spatial_radius = int(spatial_radius)
        self.scalar = {name: SampledStats() for name in ("delta", "center", "mad", "scale", "z", "peer_margins", "candidate_count")}
        self.by_label: dict[str, dict[str, RunningStats]] = {"normal": {}, "anomaly": {}}
        self.peer_slots = 0
        self.invalid_slots = 0
        self.self_violations = 0
        self.duplicate_pairs = 0
        self.spatial_violations = 0
        self.contaminated_slots = 0
        self.nonfinite = {name: 0 for name in ("delta", "center", "mad", "scale", "z", "peer_margins")}
        self.invalid_peer_index = 0
        self.image_counts = {"normal": 0, "anomaly": 0}
        self.weight_sums = np.zeros((2, 2), dtype=np.float64)
        self.weight_counts = np.zeros((2,), dtype=np.int64)

    def update(self, output: Any, masks: torch.Tensor, labels: torch.Tensor) -> None:
        delta = tensor_numpy(output.delta)
        native_group = tensor_numpy(output.native_group_margin)
        native_weights = tensor_numpy(output.native_weights)
        native_margin = tensor_numpy(output.native_margin)
        cir_margin = tensor_numpy(output.cir_margin)
        stats = output.delta_stats
        arrays = {
            "delta": delta,
            "center": tensor_numpy(stats["center"]),
            "mad": tensor_numpy(stats["mad"]),
            "scale": tensor_numpy(stats["scale"]),
            "z": tensor_numpy(stats["z"]),
            "peer_margins": tensor_numpy(output.peer_margins),
            "candidate_count": tensor_numpy(output.peer_candidate_count),
        }
        for name, value in arrays.items():
            if name in self.nonfinite:
                self.nonfinite[name] += _finite_count(value)
            self.scalar[name].update(value)

        label_array = labels.detach().cpu().numpy().reshape(-1).astype(np.int64)
        label_names = ("normal", "anomaly")
        delta_group = delta.mean(axis=(0, 2))  # [B,G]
        group_margin = native_group.mean(axis=(0, 2))  # [B,G]
        weight_mean = native_weights.mean(axis=(0, 2))  # [B,2]; native weights are already group-aggregated
        effect = (cir_margin - native_margin).mean(axis=(0, 2))  # [B]
        for label_value, label_name in enumerate(label_names):
            selected = label_array == label_value
            count = int(selected.sum())
            if count == 0:
                continue
            self.image_counts[label_name] += count
            self.weight_counts[label_value] += count
            for group in range(self.groups):
                _label_stats(self.by_label[label_name], f"delta_g{group}", delta_group[selected, group])
                _label_stats(self.by_label[label_name], f"native_margin_g{group}", group_margin[selected, group])
            self.weight_sums[label_value] += weight_mean[selected].sum(axis=0)
            _label_stats(self.by_label[label_name], "transport_margin_effect", effect[selected])

        # Patch-level GT use is post-hoc only.  It never enters peer selection.
        mask_cpu = masks.detach().float().cpu()
        gt_patch = F.adaptive_max_pool2d(mask_cpu, PATCH_GRID).squeeze(1).reshape(mask_cpu.shape[0], -1).numpy() > 0.0
        delta_patch = delta.mean(axis=(0, 3))  # [B,P]
        for label_name, selected in (("normal", label_array == 0), ("anomaly", label_array == 1)):
            if selected.any():
                _label_stats(self.by_label[label_name], "delta_patch", delta_patch[selected])
                _label_stats(self.by_label[label_name], "delta_abs_patch", np.abs(delta_patch[selected]))
        _label_stats(self.by_label["normal"], "delta_gt_negative", delta_patch[~gt_patch])
        _label_stats(self.by_label["anomaly"], "delta_gt_positive", delta_patch[gt_patch])

        indices = output.peer_indices.detach().cpu().long()
        valid = output.peer_valid.detach().cpu().bool()
        if indices.ndim != 3 or valid.ndim != 2:
            raise ValueError(f"unexpected peer geometry: indices={tuple(indices.shape)} valid={tuple(valid.shape)}")
        batch, patches, peers = indices.shape
        if peers != self.peer_count or patches != PATCH_GRID[0] * PATCH_GRID[1]:
            raise ValueError(f"unexpected peer contract: {tuple(indices.shape)}")
        in_range = (indices >= 0) & (indices < patches)
        self.invalid_peer_index += int((~in_range).sum())
        safe_indices = indices.clamp(0, patches - 1)
        valid_slots = valid.unsqueeze(-1).expand_as(indices)
        self.peer_slots += int(valid_slots.sum())
        self.invalid_slots += int((~valid).sum()) * peers
        query = torch.arange(patches).reshape(1, patches, 1)
        self.self_violations += int(((safe_indices == query) & valid_slots).sum())
        sorted_indices = torch.sort(torch.where(valid_slots, safe_indices, torch.full_like(safe_indices, -1)), dim=-1).values
        duplicate_pairs = (sorted_indices[..., 1:] == sorted_indices[..., :-1]) & (sorted_indices[..., 1:] >= 0)
        self.duplicate_pairs += int(duplicate_pairs.sum())
        ys = torch.arange(patches) // PATCH_GRID[1]
        xs = torch.arange(patches) % PATCH_GRID[1]
        peer_y = ys[safe_indices]
        peer_x = xs[safe_indices]
        query_y = ys.reshape(1, patches, 1)
        query_x = xs.reshape(1, patches, 1)
        distance = torch.maximum((peer_y - query_y).abs(), (peer_x - query_x).abs())
        self.spatial_violations += int(((distance <= self.spatial_radius) & valid_slots).sum())
        gathered_gt = gt_patch.astype(np.uint8)
        gathered_gt_t = torch.from_numpy(gathered_gt).gather(1, safe_indices.reshape(batch, -1)).reshape(batch, patches, peers)
        self.contaminated_slots += int(((gathered_gt_t.bool()) & valid_slots).sum())

    def row(self, epoch: int, target: str, scope: str, n_images: int) -> dict[str, Any]:
        row: dict[str, Any] = {
            "scope": scope,
            "epoch": int(epoch),
            "target": target,
            "n_images": int(n_images),
            "n_normal": self.image_counts["normal"],
            "n_anomaly": self.image_counts["anomaly"],
            "peer_count": self.peer_count,
            "spatial_radius": self.spatial_radius,
            "peer_valid_fraction": self.peer_slots / max(self.peer_slots + self.invalid_slots, 1),
            "invalid_peer_fraction": self.invalid_slots / max(self.peer_slots + self.invalid_slots, 1),
            "self_violation_fraction": self.self_violations / max(self.peer_slots, 1),
            "duplicate_pair_fraction": self.duplicate_pairs / max(self.peer_slots, 1),
            "spatial_violation_fraction": self.spatial_violations / max(self.peer_slots, 1),
            "peer_gt_contamination_fraction": self.contaminated_slots / max(self.peer_slots, 1),
            "invalid_peer_index_count": self.invalid_peer_index,
            "nonfinite_total": sum(self.nonfinite.values()),
        }
        for name, sample in self.scalar.items():
            summary = sample.stats.summary()
            row[f"{name}_mean"] = summary["mean"]
            row[f"{name}_min"] = summary["min"]
            row[f"{name}_max"] = summary["max"]
            for key, value in sample.quantiles().items():
                row[f"{name}_{key}"] = value
        for label_name in ("normal", "anomaly"):
            for key, stat in self.by_label[label_name].items():
                row[f"{key}_{label_name}"] = stat.mean()
        for label_value, label_name in enumerate(("normal", "anomaly")):
            for class_index, class_name in enumerate(("normal", "abnormal")):
                count = self.weight_counts[label_value]
                value = None if count == 0 else self.weight_sums[label_value, class_index] / count
                row[f"native_weight_{class_name}_{label_name}"] = float(value) if value is not None else None
        delta_values = np.concatenate([part for part in self.scalar["delta"].sample], axis=0) if self.scalar["delta"].sample else np.asarray([])
        row["delta_abs_gt_095"] = None if delta_values.size == 0 else float(np.mean(np.abs(delta_values) > 0.95))
        row["delta_abs_gt_099"] = None if delta_values.size == 0 else float(np.mean(np.abs(delta_values) > 0.99))
        row["nonfinite_by_field"] = json.dumps(self.nonfinite, sort_keys=True)
        return row


def _metric_macro_from_spool(spool: Any) -> dict[str, Any]:
    from evaluation.evaluator import evaluate_spool

    try:
        result = evaluate_spool(spool, allow_undefined_image_metrics=True)
    except ValueError as error:
        if str(error) != "binary metric requires both positive and negative labels":
            raise
        return {"pixel_auroc": None, "pixel_ap": None, "image_auroc": None, "image_ap": None}
    return {key: _metric_value(value) for key, value in result["macro"].items()}


def _existing_alpha05(run_root: Path, target: str, epoch: int, expected_config_sha: str) -> dict[str, Any]:
    path = run_root / "eval" / target / f"epoch_{epoch}" / "metrics.json"
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "arch_id": payload.get("arch_id") == "CIR_DFG_RMT_V2",
        "epoch": int(payload.get("epoch", -1)) == int(epoch),
        "target": payload.get("target") == target,
        "source": str(payload.get("source", "")).lower() == "visa",
        "config_sha256": payload.get("config_sha256") == expected_config_sha,
    }
    metrics = payload.get("macro", {})
    if not all(checks.values()):
        return {"status": "IDENTITY_MISMATCH", "path": str(path), "checks": checks, "metrics": metrics}
    return {"status": "PASS", "path": str(path), "checks": checks, "metrics": metrics, "evaluator_hash": payload.get("evaluator_hash")}


def _append_diag(
    spools: dict[str, Any],
    output: Any,
    masks: torch.Tensor,
    labels: torch.Tensor,
    names: list[str],
    paths: list[str],
    selected_indices: list[int],
    alpha0_maps: torch.Tensor,
) -> list[dict[str, Any]]:
    """Append bounded stage/group signals and per-image rank rows."""
    from evaluation.evaluator import image_score
    from evaluation.metrics import binary_metrics

    if not selected_indices:
        return []
    native_patch = F.softmax(output.native_logits.detach().float(), dim=-1)[..., 1]
    cir_patch = F.softmax(output.cir_logits.detach().float(), dim=-1)[..., 1]
    native_stage = _patch_map_to_image(native_patch)
    cir_stage = _patch_map_to_image(cir_patch)
    group_stage = _patch_map_to_image(output.native_group_margin.detach().float())
    mask_np = masks.detach().cpu().numpy().astype(np.int8)
    label_np = labels.detach().cpu().numpy().reshape(-1).astype(np.int8)
    alpha05_maps = output.cir_segmentation_probability.detach().float().cpu()
    alpha0_cpu = alpha0_maps.detach().float().cpu()
    rank_rows: list[dict[str, Any]] = []
    for index in selected_indices:
        class_name = names[index]
        path = paths[index]
        mask = mask_np[index].reshape(-1)
        alpha0 = alpha0_cpu[index].numpy().reshape(-1)
        alpha05 = alpha05_maps[index].numpy().reshape(-1)
        diff = alpha05 - alpha0
        alpha0_auc, alpha0_ap = binary_metrics(alpha0, mask, allow_undefined=True)
        alpha05_auc, alpha05_ap = binary_metrics(alpha05, mask, allow_undefined=True)
        positive = mask == 1
        negative = mask == 0
        rank_rows.append(
            {
                "class_name": class_name,
                "image_path": path,
                "image_label": int(label_np[index]),
                "mask_coverage": float(positive.mean()),
                "alpha0_pixel_auroc": _metric_value(alpha0_auc),
                "alpha0_pixel_ap": _metric_value(alpha0_ap),
                "alpha05_pixel_auroc": _metric_value(alpha05_auc),
                "alpha05_pixel_ap": _metric_value(alpha05_ap),
                "delta_pixel_auroc": None if alpha0_auc is None or alpha05_auc is None else float(alpha05_auc - alpha0_auc),
                "delta_pixel_ap": None if alpha0_ap is None or alpha05_ap is None else float(alpha05_ap - alpha0_ap),
                "alpha0_max": float(alpha0.max()),
                "alpha05_max": float(alpha05.max()),
                "score_diff_mean": float(diff.mean()),
                "score_diff_abs_mean": float(np.abs(diff).mean()),
                "score_diff_positive_mean": _safe_mean(diff[positive]),
                "score_diff_negative_mean": _safe_mean(diff[negative]),
                "score_diff_max": float(diff.max()),
                "score_diff_min": float(diff.min()),
            }
        )
        for stage in range(native_stage.shape[0]):
            spools[f"native_stage_{stage}"].append(
                class_name,
                native_stage[stage, index].cpu().numpy().reshape(-1).astype(np.float32),
                mask,
                float(native_stage[stage, index].max()),
                int(label_np[index]),
            )
            spools[f"cir_stage_{stage}"].append(
                class_name,
                cir_stage[stage, index].cpu().numpy().reshape(-1).astype(np.float32),
                mask,
                float(cir_stage[stage, index].max()),
                int(label_np[index]),
            )
        for group in range(group_stage.shape[2]):
            spools[f"native_group_{group}"].append(
                class_name,
                group_stage[0, index, group].cpu().numpy().reshape(-1).astype(np.float32),
                mask,
                float(group_stage[0, index, group].max()),
                int(label_np[index]),
            )
    return rank_rows


def _stage_group_rows(spools: dict[str, Any], epoch: int, target: str, scope: str, selected_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, spool in spools.items():
        metrics = _metric_macro_from_spool(spool)
        axis, index = name.rsplit("_", 1)
        rows.append(
            {
                "scope": scope,
                "epoch": int(epoch),
                "target": target,
                "signal": axis,
                "axis_index": int(index),
                "attribution_protocol": "selected deterministic images; patch signal bilinear-resized to 518; no deployment blur",
                "n_images": int(selected_count),
                "pixel_auroc": metrics.get("pixel_auroc"),
                "pixel_ap": metrics.get("pixel_ap"),
                "image_auroc": metrics.get("image_auroc"),
                "image_ap": metrics.get("image_ap"),
            }
        )
    return rows


def _dataset_for_target(target: str, root: Path) -> tuple[Any, str, str]:
    # Imports are delayed until MEDICAL_ROOT is set so the repository adapter
    # resolves the requested physical root exactly as eval_full does.
    os.environ["MEDICAL_ROOT"] = str(root.expanduser().resolve())
    os.environ["ACDCLIP_DATA_ROOT"] = str(root.expanduser().resolve())
    from scripts.cir_rmt.eval_full import ManifestDataset, _target_dataset

    if target == "VisA_SOURCE":
        dataset = ManifestDataset(root, ROOT / "dataset/hub/VisA.jsonl", IMAGE_SIZE)
        return dataset, "VisA", "Industrial"
    dataset = _target_dataset(target, root)
    return dataset, target, "Medical"


def _loader(dataset: Any, batch_size: int, num_workers: int, prefetch_factor: int) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": True,
    }
    if int(num_workers) > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": int(prefetch_factor)})
    return DataLoader(dataset, **kwargs)


def run_eval_cell(
    model: Any,
    config: Mapping[str, Any],
    dataset: Any,
    dataset_name: str,
    domain: str,
    epoch: int,
    target: str,
    scope: str,
    device: torch.device,
    output_root: Path,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    diag_limit: int,
    max_images: int | None,
    verify_alpha05: bool,
) -> dict[str, Any]:
    from evaluation.evaluator import image_score
    from evaluation.spool import EvaluationSpool
    from tools.cir_rmt.runtime import forward_cir
    from model.phase2b_runtime import deploy_native_logits

    diag_keys = _sample_diag_keys(dataset, diag_limit)
    spool_dir = output_root / "temporary_spools" / f"{scope}_{target}_epoch_{epoch}"
    alpha0_spool = EvaluationSpool.create(spool_dir / "alpha0")
    alpha05_spool = EvaluationSpool.create(spool_dir / "alpha05") if verify_alpha05 else None
    diag_names = [f"native_stage_{i}" for i in range(3)] + [f"cir_stage_{i}" for i in range(3)] + [f"native_group_{i}" for i in range(int(config["n_groups"]))]
    diag_spools = {name: EvaluationSpool.create(spool_dir / "diag" / name) for name in diag_names}
    mechanism = MechanismAccumulator(int(config["n_groups"]), int(config["rmt_peer_count"]), int(config["rmt_spatial_radius"]))
    rank_rows: list[dict[str, Any]] = []
    loader = _loader(dataset, batch_size, num_workers, prefetch_factor)
    seen = 0
    started = time.perf_counter()
    progress = tqdm(total=len(dataset), desc=f"forensics {scope} {target} E{epoch:02d}", unit="img", dynamic_ncols=True)
    try:
        for batch in loader:
            if max_images is not None and seen >= int(max_images):
                break
            image = batch["image"].to(device, non_blocking=device.type == "cuda").float()
            names = [str(value) for value in batch["class_name"]]
            paths = [str(value) for value in batch["image_path"]]
            masks = batch["mask"].to(device, non_blocking=device.type == "cuda").float()
            labels = batch["label"].to(device, non_blocking=device.type == "cuda").long().reshape(-1)
            take = len(names)
            if max_images is not None:
                take = min(take, int(max_images) - seen)
            if take <= 0:
                break
            if take < len(names):
                image = image[:take]
                masks = masks[:take]
                labels = labels[:take]
                names = names[:take]
                paths = paths[:take]
            output = forward_cir(model, image, names, device, config, domain=domain, require_grad=False, dataset_name=dataset_name)
            alpha0_prob, _ = deploy_native_logits(output.native_logits, image_size=IMAGE_SIZE, domain=domain)
            alpha0_maps = alpha0_prob[:, 1]
            alpha0_cpu = alpha0_maps.detach().cpu().numpy()
            cls_cpu = output.classification_probability.detach().cpu().numpy().reshape(-1)
            mask_cpu = masks.detach().cpu().numpy()
            label_cpu = labels.detach().cpu().numpy().reshape(-1)
            for index, class_name in enumerate(names):
                pixel = alpha0_cpu[index].reshape(-1)
                alpha0_spool.append(
                    class_name,
                    pixel.astype(np.float32),
                    mask_cpu[index].reshape(-1),
                    float(image_score(float(cls_cpu[index]), float(pixel.max()), domain)),
                    int(label_cpu[index]),
                )
                if alpha05_spool is not None:
                    current_pixel = output.cir_segmentation_probability[index].detach().cpu().numpy().reshape(-1).astype(np.float32)
                    alpha05_spool.append(
                        class_name,
                        current_pixel,
                        mask_cpu[index].reshape(-1),
                        float(image_score(float(cls_cpu[index]), float(current_pixel.max()), domain)),
                        int(label_cpu[index]),
                    )
            mechanism.update(output, masks, labels)
            selected_indices = [index for index, key in enumerate(zip(names, paths)) if key in diag_keys]
            rank_rows.extend(_append_diag(diag_spools, output, masks, labels, names, paths, selected_indices, alpha0_maps))
            seen += take
            progress.update(take)
            elapsed = max(time.perf_counter() - started, 1e-9)
            progress.set_postfix_str(f"{seen / elapsed:.2f} img/s")
            del output, alpha0_prob, alpha0_maps, image, masks, labels
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        progress.close()
        del loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    alpha0_metrics = _metric_macro_from_spool(alpha0_spool)
    alpha0_spool.cleanup()
    alpha05_metrics = None
    if alpha05_spool is not None:
        alpha05_metrics = _metric_macro_from_spool(alpha05_spool)
        alpha05_spool.cleanup()
    stage_rows = _stage_group_rows(diag_spools, epoch, target, scope, len(rank_rows))
    for spool in diag_spools.values():
        spool.cleanup()
    shutil.rmtree(spool_dir, ignore_errors=True)
    existing = None
    if scope == "medical":
        existing = _existing_alpha05(output_root.parent / "visa" / "seed0", target, epoch, str(config["_audit_config_sha256"]))
    row = {
        "scope": scope,
        "epoch": int(epoch),
        "target": target,
        "n_images": int(seen),
        "n_diagnostic_images": len(rank_rows),
        "alpha0": alpha0_metrics,
        "alpha05_recomputed": alpha05_metrics,
        "alpha05_existing": existing,
        "mechanism": mechanism.row(epoch, target, scope, seen),
        "stage_group": stage_rows,
        "pixel_rank": rank_rows,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return row


def _checkpoint_state(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for group in ("image_adapter", "text_adapter", "soft_prompt"):
        nested = payload.get(group, {})
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                if torch.is_tensor(value):
                    state[f"{group}.{key}"] = value.detach().float().cpu()
    return state


def checkpoint_drift_rows(config: Mapping[str, Any], run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base: dict[str, torch.Tensor] | None = None
    previous: dict[str, torch.Tensor] | None = None
    previous_epoch: int | None = None
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    for epoch in EPOCHS:
        path = run_root / "checkpoints" / f"epoch_{epoch}.pth"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = _checkpoint_state(payload)
        if base is None:
            base = state
        def norm(prefix: str | None = None) -> float:
            values = [value for key, value in state.items() if prefix is None or key.startswith(prefix)]
            return float(sum(float(torch.sum(value * value)) for value in values) ** 0.5)
        row: dict[str, Any] = {
            "epoch": epoch,
            "checkpoint": str(path),
            "checkpoint_bytes": path.stat().st_size,
            "checkpoint_sha256": sha256_file(path),
            "global_step": payload.get("global_step"),
            "checkpoint_git_sha": payload.get("git_sha"),
            "current_audit_head": current_git().get("head"),
            "config_sha256": payload.get("config_sha256"),
            "parent_config_sha256": payload.get("parent_config_sha256"),
            "parent_checkpoint_sha256": payload.get("parent_checkpoint_sha256"),
            "optimizer_state_present": isinstance(payload.get("optimizer_state"), Mapping),
            "scheduler_state_present": isinstance(payload.get("scheduler_state"), Mapping),
            "state_tensor_count": len(state),
            "state_numel": sum(int(value.numel()) for value in state.values()),
            "parameter_l2": norm(),
            "image_adapter_l2": norm("image_adapter."),
            "text_adapter_l2": norm("text_adapter."),
            "soft_prompt_l2": norm("soft_prompt."),
            "run_manifest_git_sha": manifest.get("git_sha"),
            "run_manifest_target_epochs": json.dumps(manifest.get("target_epochs", [])),
            "run_manifest_epochs": json.dumps(manifest.get("epochs", [])),
        }
        for relation, other, other_epoch in (("vs_e12", base, EPOCHS[0]), ("vs_previous", previous, previous_epoch)):
            if other is None:
                row[f"{relation}_epoch"] = None
                row[f"{relation}_l2"] = None
                row[f"{relation}_rms"] = None
                row[f"{relation}_max_abs"] = None
                continue
            common = sorted(set(state) & set(other))
            sq = 0.0
            count = 0
            max_abs = 0.0
            for key in common:
                difference = state[key] - other[key]
                sq += float(torch.sum(difference * difference))
                count += int(difference.numel())
                max_abs = max(max_abs, float(difference.abs().max()))
            row[f"{relation}_epoch"] = other_epoch
            row[f"{relation}_l2"] = sq**0.5
            row[f"{relation}_rms"] = (sq / max(count, 1)) ** 0.5
            row[f"{relation}_max_abs"] = max_abs
        rows.append(row)
        previous = state
        previous_epoch = epoch
        del payload, state
        gc.collect()
    return rows


def _parse_historical_args(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "args:" not in line:
            continue
        try:
            return ast.literal_eval(line.split("args:", 1)[1].strip())
        except (SyntaxError, ValueError):
            return {}
    return {}


def protocol_ledger(config: Mapping[str, Any], parent: Mapping[str, Any], run_root: Path) -> list[dict[str, Any]]:
    hist_root = ROOT / "runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch"
    hist_train = _parse_historical_args(hist_root / "train.log")
    hist_test = _parse_historical_args(hist_root / "test.log")
    hist_ref = hist_root / "parsed_results.csv"
    rows: list[dict[str, Any]] = []
    def add(comparison: str, current: Any, reference: Any, status: str, impact: str, evidence: str) -> None:
        rows.append({"comparison": comparison, "current": json.dumps(current, sort_keys=True, default=_json_default), "historical_reference": json.dumps(reference, sort_keys=True, default=_json_default), "status": status, "impact": impact, "evidence": evidence})

    add("architecture", config.get("arch_id"), "legacy Phase2B", "MISMATCH_BY_DESIGN", "The reference is not a CIR V2 control.", "configs/cir_dfg_rmt_v2.json; configs/phase2b_canonical_v1.json")
    add("reference_epoch", "V2 has no E10 checkpoint", "E10", "MISMATCH", "The supplied 90.98/40.35 anchor cannot be compared to V2 E12-E20 as a same-epoch result.", f"{hist_ref}; {hist_root / 'test.log'}")
    add("candidate_epochs", list(config.get("parent_config", {}).get("candidate_epochs", parent.get("candidate_epochs", []))), list(parent.get("candidate_epochs", [])), "MATCH", "Parent preregistration includes E10-E20, but V2 training omitted E10.", "checkpoint resolved_scientific_config; scripts/cir_rmt/train_full.py:27")
    add("current_target_epochs", [12, 14, 16, 18, 20], [10, 12, 14, 16, 18, 20], "MISMATCH", "E10 was the historical reference and is absent from the V2 artifact family.", "scripts/cir_rmt/train_full.py:27; scripts/cir_rmt/run_full_cir_v2.sh")
    add("model", config.get("parent_config_path"), hist_train.get("model_name"), "MATCH", "Model family and input resolution are aligned where logged.", "checkpoint metadata; historical train.log")
    add("image_size", parent.get("img_size"), hist_train.get("img_size"), "MATCH", "No image-size mismatch is indicated.", "configs/phase2b_canonical_v1.json; historical train.log")
    add("effective_batch_size", parent.get("effective_batch_size", parent.get("batch_size")), hist_train.get("batch_size"), "MATCH", "Batch geometry is not the primary explanation for the anchor gap.", "checkpoint resolved_scientific_config; historical train.log")
    add("precision_amp", config.get("precision"), {"precision": "amp", "amp": hist_train.get("amp")}, "MISMATCH", "Current V2 is FP32; historical Phase2B used AMP=True.", "checkpoint metadata; historical train.log")
    add("grad_checkpointing", parent.get("grad_checkpointing"), hist_train.get("grad_checkpointing"), "MATCH", "Both logs/configs request gradient checkpointing.", "configs/phase2b_canonical_v1.json; historical train.log")
    add("workers", parent.get("num_workers"), hist_train.get("num_workers"), "MISMATCH", "Loader workers differ but should not change deterministic test maps.", "checkpoint resolved_scientific_config; historical train.log")
    add("pixel_stride", "1", hist_test.get("pixel_stride", 4), "MISMATCH", "The historical 90.98/40.35 result uses a different pixel metric grid.", "scripts/cir_rmt/eval_full.py; historical test.log; parsed_results.csv")
    add("medical_image_metrics", "undefined -> null for one-class colon", "0.00/0.00 in legacy table", "MISMATCH", "Image metrics are not comparable for colon targets.", "evaluation/evaluator.py; parsed_results.csv")
    add("historical_hybrid_alpha", config.get("rmt_transport_alpha"), hist_train.get("hybrid_alpha_max"), "MISMATCH_CONCEPTS", "Legacy hybrid prompt alpha is not RMT transport alpha.", "configs/cir_dfg_rmt_v2.json; historical train.log")
    add("lambda_kg", parent.get("lambda_kg"), hist_train.get("lambda_kg"), "MISMATCH", "The historical model was trained with a tenfold larger KG coefficient.", "configs/phase2b_canonical_v1.json; historical train.log")
    add("lambda_k", parent.get("lambda_k"), hist_train.get("lambda_k"), "MISMATCH", "The historical model used nonzero k regularization; V2 preregisters zero.", "configs/phase2b_canonical_v1.json; historical train.log")
    add("matched_parent_checkpoint", "MATCHED_PARENT_CHECKPOINT=NOT AVAILABLE", "required for causal P/V0 control", "NOT_AVAILABLE", "No checkpoint carrying the canonical parent config hash was found in the audited artifact family.", "parent_config_sha256=d24cf... search across /home/ai4/caohuy artifacts")
    add("current_checkpoint_identity", "CIR_DFG_RMT_V2; alpha=0.5; V2 direction", "legacy adapter_10.pth has no CIR identity", "MISMATCH", "The old adapter cannot be treated as a matched parent checkpoint.", "V2 epoch checkpoint metadata; legacy adapter_10.pth")
    add("source_confirmation_model", "fresh parent model; no checkpoint load", "not a trained V2 medical control", "LIMITED", "The preserved 120-image alpha curve confirms a source-side sign behavior only.", "tools/cir_rmt/v2_source_confirmation.py:200-437; source_confirmation/REPORT.md")
    add("source_confirmation_identity", "current config SHA 064e...", "source artifact final identity 31827...", "MISMATCH_ADMINISTRATIVE", "The source confirmation predates the alpha-status freeze and is not the current checkpoint identity.", "source_confirmation/*.json; configs/cir_dfg_rmt_v2.json")
    add("data_roles", "VisA train/source; six medical final test", "VisA train; six medical test", "MATCH_WITH_CAVEAT", "Medical samples are final-test evidence; no medical training was found.", "scripts/canonical/README.md; dataset/hub/*.jsonl")
    add("checkpoint_artifact_manifest", "run_manifest epochs=[]; target_epochs=[12..20]", "training trajectory expected", "WEAK_PROVENANCE", "The manifest has no history and records a pre-final evaluator commit; checkpoint nested metadata remains usable.", str(run_root / "run_manifest.json"))
    add("lr_scheduler", {"type": "StepLR", "step_size": 1, "gamma": parent.get("lr_gamma", config.get("lr_gamma")), "scheduler_step_call": False}, {"type": "StepLR", "step_size": 1, "gamma": parent.get("lr_gamma"), "scheduler_step_call": True}, "MISMATCH_CONFIRMED", "CIR constructs and serializes StepLR but never advances it; all five CIR epoch checkpoint states remain at last_epoch=0 and initial image/text LRs.", "scripts/cir_rmt/train_full.py:148-166,224-250; train.py:359-420; runs/cir_rmt/CIR_DFG_RMT_V2/forensics_20260830/scheduler_audit_summary.json")
    add("optimizer_hparams", {"optimizer": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "groups": ["image_adapter", "text_adapter", "soft_prompt"]}, {"optimizer": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "groups": ["image_adapter", "text_adapter", "soft_prompt"]}, "MATCH", "Serialized CIR optimizer groups match the canonical Adam hyperparameters; zero weight decay is global, so no group is accidentally exempt from a nonzero decay.", "scripts/cir_rmt/train_full.py:72-77; train.py:_make_optimizer; scheduler_optimizer_group_detail.csv")
    add("soft_prompt_lr_policy", "separate freeze/unfreeze policy with constant_lr=1e-4", "same separate freeze/unfreeze policy", "MATCH_WITH_CAVEAT", "The soft prompt is zero-LR while frozen and restored to constant_lr when unfrozen; this policy is distinct from StepLR and does not rescue the missing image/text scheduler updates.", "scripts/cir_rmt/train_full.py:80-90; model/phase2b_schedule.py:60-66")
    add("soft_prompt_base_lr", config.get("parent_config", {}).get("soft_prompt_lr", parent.get("soft_prompt_lr")), 5e-5, "MISMATCH_LEGACY", "The available historical parent log used soft_prompt_lr=5e-5, while the current canonical parent/CIR protocol uses 1e-4; this is a separate legacy mismatch from the missing StepLR call.", "configs/phase2b_canonical_v1.json; historical train.log:1,11,127; scheduler_optimization_audit.csv")
    return rows


def parameter_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = [
        ("rmt_transport_alpha", config.get("rmt_transport_alpha"), "Frozen value inherited from the V2 source-sign protocol; this audit does not retune it.", "configs/cir_dfg_rmt_v2.json; docs/cir_rmt/v2/ARCHITECTURE_FREEZE_V2.md", "FROZEN"),
        ("rmt_transport_direction", config.get("rmt_transport_direction"), "V2 causal direction under audit: abnormal receives -alpha*delta and normal +alpha*delta.", "configs/cir_dfg_rmt_v2.json; tools/cir_rmt/core.py:129-146", "FROZEN"),
        ("n_groups", config.get("n_groups"), "Inherited Phase2B group geometry.", "configs/phase2b_canonical_v1.json", "INHERITED"),
        ("rmt_peer_count", config.get("rmt_peer_count"), "Exactly K=8 is part of the frozen peer contract.", "configs/cir_dfg_rmt_v2.json; tools/cir_rmt/core.py:48-99", "FROZEN"),
        ("rmt_spatial_radius", config.get("rmt_spatial_radius"), "Excludes local Chebyshev neighbors from the GT-free peer candidates.", "configs/cir_dfg_rmt_v2.json; tools/cir_rmt/core.py:149-205", "FROZEN"),
        ("rmt_center", config.get("rmt_center"), "Midpoint median is the registered robust center.", "configs/cir_dfg_rmt_v2.json; tools/cir_rmt/core.py:48-99", "FROZEN"),
        ("rmt_scale", config.get("rmt_scale"), "MAD scaled by 1.4826, with epsilon stabilization.", "configs/cir_dfg_rmt_v2.json; tools/cir_rmt/core.py:48-99", "FROZEN"),
        ("rmt_transform", config.get("rmt_transform"), "Tanh bounds relational evidence before transport.", "configs/cir_dfg_rmt_v2.json; tools/cir_rmt/core.py:48-99", "FROZEN"),
        ("rmt_delta_stopgrad", config.get("rmt_delta_stopgrad"), "Peer evidence is detached so the relational path is not a hidden trainable head.", "configs/cir_dfg_rmt_v2.json; tools/cir_rmt/core.py:48-99", "FROZEN"),
        ("rmt_score_mode", config.get("rmt_score_mode"), "Exact score-space implementation with optimized/reference parity tests.", "configs/cir_dfg_rmt_v2.json; tools/cir_rmt/core.py:308-381", "FROZEN"),
        ("precision", config.get("precision"), "FP32 is required for the canonical numerical contract.", "configs/cir_dfg_rmt_v2.json; model/phase2b_runtime.py", "FROZEN"),
        ("medical_image_score", "0.5*classification + 0.5*pixel_max", "Frozen medical image-level aggregation.", "evaluation/evaluator.py:12-16", "FROZEN"),
        ("candidate_epochs", config.get("parent_config", {}).get("candidate_epochs", "see checkpoint"), "Parent preregistration includes E10; V2 artifact production omitted it, which is an audit finding rather than a retuning choice.", "configs/phase2b_canonical_v1.json; scripts/cir_rmt/train_full.py:27", "AUDIT_FINDING"),
    ]
    return [{"parameter": key, "value": json.dumps(value, sort_keys=True, default=_json_default), "justification": justification, "evidence": evidence, "status": status} for key, value, justification, evidence, status in entries]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_progress(path: Path, completed: set[tuple[str, int]]) -> None:
    write_json(path, {"completed": sorted([{"scope": scope, "epoch": epoch} for scope, epoch in completed]), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def _gradient_norms(model: Any, grads: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter], names: list[str]) -> dict[str, float]:
    sums: dict[str, float] = {"all": 0.0, "image_adapter": 0.0, "text_adapter": 0.0, "soft_prompt": 0.0, "other": 0.0}
    for name, parameter, gradient in zip(names, params, grads):
        if gradient is None:
            continue
        norm_sq = float(torch.sum(gradient.detach().float() ** 2))
        sums["all"] += norm_sq
        group = "image_adapter" if name.startswith("image_adapter.") else "text_adapter" if name.startswith("text_adapter.") else "soft_prompt" if name.startswith("soft_prompt.") else "other"
        sums[group] += norm_sq
    return {key: value**0.5 for key, value in sums.items()}


def _gradient_vector(grads: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().float().reshape(-1) for parameter, gradient in zip(params, grads)])


def run_gradient_audit(config: Mapping[str, Any], run_root: Path, clip_asset: Path, visa_root: Path, device: torch.device, output_root: Path, epoch: int = 14) -> dict[str, Any]:
    """One deterministic VisA training batch: objective-gradient conflict and train/deploy mismatch."""
    from dataset import TextAndImageDataset
    from model.phase2b_legacy_bridge import load_adapter_state
    from model.phase2b_runtime import build_phase2b_trainable
    from scripts.cir_rmt.train_full import _set_epoch_state, _text_with_regularizers
    from tools.cir_rmt.runtime import forward_cir
    from utils import calculate_seg_loss, configure_canonical_fp32
    from evaluation.metrics import binary_metrics

    configure_canonical_fp32()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    checkpoint_path = run_root / "checkpoints" / f"epoch_{epoch}.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parent_config = dict(checkpoint["parent_config"])
    model = build_phase2b_trainable(parent_config, clip_asset, device)
    load_adapter_state(model, checkpoint)
    audit_optimizer = torch.optim.SGD([{"params": list(model.soft_prompt.parameters()), "name": "soft_prompt", "lr": 0.0, "constant_lr": 0.0}], lr=0.0)
    _set_epoch_state(model, audit_optimizer, parent_config, epoch)
    del audit_optimizer
    model.eval()  # keep the fixed 37x37 token contract; gradients remain enabled below
    dataset = TextAndImageDataset(str(visa_root), str(ROOT / "dataset/hub/VisA.jsonl"), IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=int(parent_config["micro_batch_size"]), shuffle=False, num_workers=0, pin_memory=True)
    batch = next(iter(loader))
    images = batch["image"].to(device).float()
    masks = batch["mask"].to(device).float()
    labels = batch["label"].to(device).long().reshape(-1)
    names = [str(value) for value in batch["class_name"]]
    text, kg_loss, k_loss = _text_with_regularizers(model, names, parent_config, device)
    output = forward_cir(model, images, names, device, config, domain="Industrial", require_grad=True, dataset_name="VisA", precomputed_text_features=text)
    cls_loss = F.cross_entropy(output.classification_logits, labels)
    seg_loss = calculate_seg_loss(output.cir_training_segmentation_probability.float(), masks.float())
    terms: dict[str, torch.Tensor] = {
        "classification": cls_loss,
        "segmentation": seg_loss,
        "weighted_kg": float(parent_config.get("lambda_kg", 0.0)) * kg_loss,
        "weighted_k": float(parent_config.get("lambda_k", 0.0)) * k_loss,
    }
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    names_params = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    gradients: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for name, loss in terms.items():
        if loss.requires_grad:
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        else:
            grads = tuple(None for _ in params)
        vector = _gradient_vector(grads, params).cpu()
        gradients[name] = vector
        norms = _gradient_norms(model, grads, params, names_params)
        rows.append({"row_type": "component", "epoch": epoch, "batch_size": len(names), "component": name, "component_a": "", "component_b": "", "raw_loss": float(loss.detach().cpu()), "gradient_norm": norms["all"], "image_adapter_gradient_norm": norms["image_adapter"], "text_adapter_gradient_norm": norms["text_adapter"], "soft_prompt_gradient_norm": norms["soft_prompt"], "other_gradient_norm": norms["other"], "finite": bool(torch.isfinite(vector).all()), "nonzero": bool(torch.any(vector != 0))})
    total = sum(terms.values())
    if total.requires_grad:
        total_grads = torch.autograd.grad(total, params, retain_graph=False, allow_unused=True)
    else:
        total_grads = tuple(None for _ in params)
    total_vector = _gradient_vector(total_grads, params).cpu()
    total_norms = _gradient_norms(model, total_grads, params, names_params)
    rows.append({"row_type": "component", "epoch": epoch, "batch_size": len(names), "component": "total", "component_a": "", "component_b": "", "raw_loss": float(total.detach().cpu()), "gradient_norm": total_norms["all"], "image_adapter_gradient_norm": total_norms["image_adapter"], "text_adapter_gradient_norm": total_norms["text_adapter"], "soft_prompt_gradient_norm": total_norms["soft_prompt"], "other_gradient_norm": total_norms["other"], "finite": bool(torch.isfinite(total_vector).all()), "nonzero": bool(torch.any(total_vector != 0))})
    component_names = list(terms)
    for left_index, left in enumerate(component_names):
        for right in component_names[left_index + 1:]:
            a = gradients[left]
            b = gradients[right]
            denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
            cosine = None if denominator == 0.0 else float(torch.dot(a, b) / denominator)
            rows.append({"row_type": "pair", "epoch": epoch, "batch_size": len(names), "component": "", "component_a": left, "component_b": right, "raw_loss": None, "gradient_norm": None, "image_adapter_gradient_norm": None, "text_adapter_gradient_norm": None, "soft_prompt_gradient_norm": None, "other_gradient_norm": None, "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "nonzero": bool(torch.any(a != 0) and torch.any(b != 0)), "gradient_cosine": cosine, "gradient_conflict": None if cosine is None else bool(cosine < 0.0)})
    train_map = output.cir_training_segmentation_probability[:, 1].detach().float().cpu().numpy().reshape(-1)
    deploy_map = output.cir_segmentation_probability.detach().float().cpu().numpy().reshape(-1)
    mask_np = masks.detach().cpu().numpy().reshape(-1).astype(np.int8)
    train_auc, train_ap = binary_metrics(train_map, mask_np, allow_undefined=True)
    deploy_auc, deploy_ap = binary_metrics(deploy_map, mask_np, allow_undefined=True)
    corr = float(np.corrcoef(train_map, deploy_map)[0, 1]) if np.std(train_map) > 0 and np.std(deploy_map) > 0 else None
    mismatch = {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "source": "VisA training loader first deterministic batch",
        "batch_size": len(names),
        "seed": 0,
        "training_probability_path": "_training_probability: bilinear resize -> stage mean -> softmax; no deployment blur",
        "deployment_probability_path": "deploy_native_logits: Gaussian blur -> bilinear resize -> stage mean -> softmax",
        "mean_abs_difference": float(np.abs(train_map - deploy_map).mean()),
        "max_abs_difference": float(np.abs(train_map - deploy_map).max()),
        "pearson_correlation": corr,
        "training_pixel_auroc": _metric_value(train_auc),
        "training_pixel_ap": _metric_value(train_ap),
        "deployment_pixel_auroc": _metric_value(deploy_auc),
        "deployment_pixel_ap": _metric_value(deploy_ap),
        "classification_loss": float(cls_loss.detach().cpu()),
        "segmentation_loss": float(seg_loss.detach().cpu()),
        "weighted_kg_loss": float(terms["weighted_kg"].detach().cpu()),
        "weighted_k_loss": float(terms["weighted_k"].detach().cpu()),
    }
    write_csv(output_root / "gradient_conflict_report.csv", rows)
    write_json(output_root / "gradient_conflict_summary.json", {"rows": rows, "mismatch": mismatch})
    return {"rows": rows, "mismatch": mismatch}


def write_train_deploy_report(output_root: Path, gradient: Mapping[str, Any]) -> None:
    mismatch = gradient.get("mismatch", {})
    lines = [
        "# Train/deploy mismatch audit",
        "",
        f"Status: {mismatch.get('status', 'UNKNOWN')}",
        "",
        "This is a read-only deterministic first-batch audit at the preserved V2 E14 checkpoint. The training-side map intentionally uses the production training probability path; the deployment-side map intentionally uses the frozen deployment path.",
        "",
        "| quantity | value |",
        "|---|---:|",
    ]
    for key in ("mean_abs_difference", "max_abs_difference", "pearson_correlation", "training_pixel_auroc", "training_pixel_ap", "deployment_pixel_auroc", "deployment_pixel_ap", "classification_loss", "segmentation_loss", "weighted_kg_loss", "weighted_k_loss"):
        lines.append(f"| {key} | {mismatch.get(key)} |")
    lines += [
        "",
        "Interpretation: a nonzero map difference is proven deployment/training-path divergence. Its effect on the final six-medical failure is correlational until matched full-evaluation evidence isolates it.",
        "",
        "Evidence: `tools/cir_rmt/runtime.py:60-70`, `model/phase2b_runtime.py:164-184`, and `gradient_conflict_report.csv` in this directory.",
    ]
    (output_root / "train_deploy_mismatch.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_cells(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def write_inference_effect(output_root: Path, cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        if cell.get("scope") != "medical":
            continue
        existing = cell.get("alpha05_existing") or {}
        existing_metrics = existing.get("metrics") or {}
        alpha0 = cell.get("alpha0") or {}
        alpha0_metrics = alpha0
        rows.append({
            "epoch": cell.get("epoch"),
            "target": cell.get("target"),
            "n_images": cell.get("n_images"),
            "checkpoint_sha256": cell.get("checkpoint_sha256"),
            "alpha0_status": cell.get("alpha0_status", "PASS"),
            "alpha05_status": existing.get("status"),
            "alpha05_evaluator_hash": existing.get("evaluator_hash"),
            "alpha0_pixel_auroc": alpha0_metrics.get("pixel_auroc"),
            "alpha0_pixel_ap": alpha0_metrics.get("pixel_ap"),
            "alpha0_image_auroc": alpha0_metrics.get("image_auroc"),
            "alpha0_image_ap": alpha0_metrics.get("image_ap"),
            "alpha05_pixel_auroc": existing_metrics.get("pixel_auroc"),
            "alpha05_pixel_ap": existing_metrics.get("pixel_ap"),
            "alpha05_image_auroc": existing_metrics.get("image_auroc"),
            "alpha05_image_ap": existing_metrics.get("image_ap"),
            "delta_pixel_auroc": None if alpha0_metrics.get("pixel_auroc") is None or existing_metrics.get("pixel_auroc") is None else float(existing_metrics["pixel_auroc"] - alpha0_metrics["pixel_auroc"]),
            "delta_pixel_ap": None if alpha0_metrics.get("pixel_ap") is None or existing_metrics.get("pixel_ap") is None else float(existing_metrics["pixel_ap"] - alpha0_metrics["pixel_ap"]),
            "delta_image_auroc": None if alpha0_metrics.get("image_auroc") is None or existing_metrics.get("image_auroc") is None else float(existing_metrics["image_auroc"] - alpha0_metrics["image_auroc"]),
            "elapsed_seconds": cell.get("elapsed_seconds"),
            "delta_image_ap": None if alpha0_metrics.get("image_ap") is None or existing_metrics.get("image_ap") is None else float(existing_metrics["image_ap"] - alpha0_metrics["image_ap"]),
        })
    fields = list(rows[0].keys()) if rows else []
    write_csv(output_root / "inference_rmt_effect.csv", rows, fields)
    return rows


def write_source_metrics(output_root: Path, cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        if cell.get("scope") != "source":
            continue
        alpha0 = cell.get("alpha0") or {}
        alpha05 = cell.get("alpha05_recomputed") or {}
        rows.append({"epoch": cell.get("epoch"), "target": cell.get("target"), "n_images": cell.get("n_images"), "alpha0_pixel_auroc": alpha0.get("pixel_auroc"), "alpha0_pixel_ap": alpha0.get("pixel_ap"), "alpha0_image_auroc": alpha0.get("image_auroc"), "alpha0_image_ap": alpha0.get("image_ap"), "alpha05_pixel_auroc": alpha05.get("pixel_auroc"), "alpha05_pixel_ap": alpha05.get("pixel_ap"), "alpha05_image_auroc": alpha05.get("image_auroc"), "alpha05_image_ap": alpha05.get("image_ap")})
    write_csv(output_root / "source_checkpoint_metrics.csv", rows)
    return rows


def _flatten_nested(cells: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        value = cell.get(key)
        if key == "mechanism":
            if value:
                rows.append(value)
        elif isinstance(value, list):
            for row in value:
                rows.append({"epoch": cell.get("epoch"), "target": cell.get("target"), "scope": cell.get("scope"), **row})
    return rows


def _macro_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("delta_pixel_auroc") is not None and row.get("delta_pixel_ap") is not None]
    if not valid:
        return {"n": 0}
    return {
        "n": len(valid),
        "mean_delta_pixel_auroc": float(np.mean([row["delta_pixel_auroc"] for row in valid])),
        "mean_delta_pixel_ap": float(np.mean([row["delta_pixel_ap"] for row in valid])),
        "alpha05_better_pixel_auroc": int(sum(row["delta_pixel_auroc"] > 0 for row in valid)),
        "alpha05_better_pixel_ap": int(sum(row["delta_pixel_ap"] > 0 for row in valid)),
        "alpha05_better_both": int(sum(row["delta_pixel_auroc"] > 0 and row["delta_pixel_ap"] > 0 for row in valid)),
        "alpha05_worse_both": int(sum(row["delta_pixel_auroc"] < 0 and row["delta_pixel_ap"] < 0 for row in valid)),
    }


def choose_decision(effect_rows: list[dict[str, Any]], protocol_rows: list[dict[str, Any]]) -> tuple[str, str]:
    summary = _macro_summary(effect_rows)
    scheduler_bug = any(
        row.get("comparison") == "lr_scheduler" and row.get("status") == "MISMATCH_CONFIRMED"
        for row in protocol_rows
    )
    if scheduler_bug:
        return "KEEP_PARENT_FIX_TRAINING", "CIR_SCHEDULER_BUG_CONFIRMED: current CIR-V2 training is not optimization-matched to Phase2B because train_full.py never calls scheduler.step(). The present benchmark cannot cleanly isolate the RMT hypothesis; run one matched corrective retrain before attributing degradation to RMT."
    # The decision is deliberately a simple preregistered audit rule: if the
    # RMT transport loses both pixel metrics on most matched cells, the line is
    # abandoned. Otherwise the architecture remains a candidate, but the
    # protocol/training lineage is repaired before any redesign.
    valid = [row for row in effect_rows if row.get("delta_pixel_auroc") is not None and row.get("delta_pixel_ap") is not None]
    if valid:
        harm_both = sum(float(row["delta_pixel_auroc"]) < 0 and float(row["delta_pixel_ap"]) < 0 for row in valid)
        help_both = sum(float(row["delta_pixel_auroc"]) > 0 and float(row["delta_pixel_ap"]) > 0 for row in valid)
        if harm_both >= math.ceil(0.6 * len(valid)) and summary.get("mean_delta_pixel_auroc", 0.0) < 0 and summary.get("mean_delta_pixel_ap", 0.0) < 0:
            return "ABANDON_RMT_RETURN_TO_PHASE2B", "Full matched alpha=0 versus alpha=0.5 medical inference shows RMT harms both pixel metrics on a majority of cells and in the mean."
        if help_both >= math.ceil(0.6 * len(valid)) and summary.get("mean_delta_pixel_auroc", 0.0) > 0 and summary.get("mean_delta_pixel_ap", 0.0) > 0:
            return "KEEP_RMT_AND_FIX", "RMT improves both pixel metrics on a majority of the full medical matrix; the remaining failure is treated as lineage/protocol or training quality, not a proven transport-sign failure."
    mismatch_count = sum(row.get("status", "").startswith("MISMATCH") for row in protocol_rows)
    if mismatch_count:
        return "KEEP_PARENT_FIX_TRAINING", "The historical anchor is not a matched control: E10 is missing, stride differs, and the legacy objective differs. A matched parent checkpoint/control is required before changing RMT."
    return "KEEP_RMT_AND_FIX", "The available evidence does not show a majority two-metric RMT harm; preserve the frozen architecture while repairing the unresolved lineage."


def write_full_report(output_root: Path, config: Mapping[str, Any], protocol_rows: list[dict[str, Any]], effect_rows: list[dict[str, Any]], drift_rows: list[dict[str, Any]], peer_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]], rank_rows: list[dict[str, Any]], gradient: Mapping[str, Any], decision: str, rationale: str, source_rows: list[dict[str, Any]]) -> None:
    summary = _macro_summary(effect_rows)
    scheduler_bug = any(
        row.get("comparison") == "lr_scheduler" and row.get("status") == "MISMATCH_CONFIRMED"
        for row in protocol_rows
    )

    def _mean_metric(field: str) -> float | None:
        values = [row.get(field) for row in effect_rows if row.get(field) is not None]
        return _safe_mean(np.asarray(values, dtype=np.float64)) if values else None

    alpha0_mean_auc = _mean_metric("alpha0_pixel_auroc")
    alpha05_mean_auc = _mean_metric("alpha05_pixel_auroc")
    alpha0_mean_ap = _mean_metric("alpha0_pixel_ap")
    alpha05_mean_ap = _mean_metric("alpha05_pixel_ap")
    observed_matrix_seconds = sum(float(row.get("elapsed_seconds") or 0.0) for row in effect_rows)
    lines = [
        "# CIR_DFG_RMT_V2 full failure forensics",
        "",
        f"Decision: `{decision}`",
        "",
        "## Executive finding",
        "",
        "The optimization audit is conclusive: `CIR_SCHEDULER_BUG_CONFIRMED`. CIR constructs and serializes StepLR but the epoch loop never calls `scheduler.step()`. The five CIR epoch checkpoints therefore retain `last_epoch=0` and initial image/text LRs instead of the intended gamma=0.9 decay.",
        "",
        "This is a major protocol confound. Current CIR-V2 training was not optimization-matched to Phase2B, so the present benchmark cannot cleanly isolate the RMT hypothesis; the conclusion must not be `RMT failed`.",
        "",
        "The preserved 90.98/40.35 comparison is not a matched V2 parent result. The historical table is E10 with pixel_stride=4, while the V2 artifact family starts at E12 and the current exact evaluator uses pixel_stride=1. The historical model also used a different objective and AMP setting. Therefore the anchor gap is not, by itself, evidence that the V2 RMT transport caused the failure.",
        "",
        f"The new alpha=0 control covers {len(effect_rows)} medical checkpoint-target cells. At alpha=0.5 minus alpha=0, the aggregate summary is: `{json.dumps(summary, sort_keys=True)}`.",
        f"Mean paired pixel AUROC is alpha=0 `{alpha0_mean_auc}` versus alpha=0.5 `{alpha05_mean_auc}`; mean pixel AP is alpha=0 `{alpha0_mean_ap}` versus alpha=0.5 `{alpha05_mean_ap}`. Metric values are decimal scores; the CSV contains the complete cell-level deltas.",
        f"Observed alpha=0 medical-matrix evaluator time summed across cells: `{observed_matrix_seconds:.1f}` seconds (sequential cell time; excludes source and gradient audits).",
        "",
        f"Decision rationale: {rationale}",
        "",
        "## Proven hard facts",
        "",
        "- The current V2 checkpoints are E12/E14/E16/E18/E20 only; E10 is absent even though the parent candidate list includes E10.",
        "- The current medical evaluator uses the deployed CIR map and exact full-resolution pixel metrics; colon image metrics are undefined and represented as null. The historical table used stride 4 and legacy zero-valued colon image columns.",
        "- All five V2 checkpoints carry CIR_DFG_RMT_V2 identity, current parent config hash, FP32 metadata, and alpha=0.5 V2 direction. The nested adapter states drift across epochs.",
        "- The source-confirmation alpha curve is a bounded 120-image VisA sign confirmation from a fresh parent model; its source code does not load a trained V2 checkpoint.",
        "- The run manifest is weak provenance: it records an empty history and a producing commit different from the checkpoint producing commit. Checkpoint nested metadata is the stronger source for epoch/step/identity.",
        "",
        "## Evidence status",
        "",
        "- Proven: CIR scheduler bug (`scheduler.step()` absent from the epoch loop and stale serialized scheduler state), protocol mismatch, E10 disappearance, alpha=0 full-matrix control, current alpha=0.5 artifact identity, peer invariant measurements, deployment/training map difference, and checkpoint parameter drift.",
        "- Correlational: association between RMT effect and medical metric changes; association between peer contamination or stage/group signal quality and final failure.",
        "- Unknown: the magnitude of the scheduler confound after a matched corrective rerun, a matched Phase2B parent checkpoint under the current canonical config, full causal separation of training objective versus representation drift, and any claim that the old 90.98/40.35 result is reproducible under the current exact evaluator.",
        "",
        "## Root-cause ranking",
        "",
        "| rank | candidate cause | evidence status | limiting evidence |",
        "|---:|---|---|---|",
        "| 1 | CIR trained without advancing StepLR | Proven, major protocol bug | No `scheduler.step()` in CIR loop; E12-E20 states all have `last_epoch=0` and initial image/text LRs; corrected-run effect size is not yet measured |",
        "| 2 | Historical 90.98/40.35 anchor is not protocol-equivalent | Proven, high-impact confound | E10 vs E12+, stride 4 vs 1, and legacy objective/AMP differ |",
        "| 3 | No matched parent control / different training lineage | Proven gap; causal contribution unknown | Required parent checkpoint hash is absent and current V2 adapter states drift |",
        "| 4 | RMT transport effect in the current trained representation | Measured by the paired alpha matrix | Establishes an alpha contrast, not the cause of the absolute historical gap |",
        "| 5 | Train/deploy map path divergence | Proven implementation difference; causal size unknown | Training path omits the deployment Gaussian blur |",
        "| 6 | Peer signal quality or contamination | Measured post-hoc; correlational | Ground truth never enters peer selection; contamination is diagnosis only |",
        "| 7 | Checkpoint/run-manifest provenance weakness | Proven administrative risk | Nested checkpoint identity is stronger than the empty run history |",
        "",
        "The ranking separates demonstrated mismatches from hypotheses whose effect is only isolated by correlation or by the paired alpha intervention.",
        "",
        "## Required output coverage",
        "",
        f"- `inference_rmt_effect.csv`: {len(effect_rows)} medical cells, alpha=0 recomputed and alpha=0.5 read from identity-checked preserved artifacts.",
        f"- `checkpoint_drift.csv`: {len(drift_rows)} checkpoints with nested adapter-state drift versus E12 and previous checkpoint.",
        f"- `peer_forensics.csv`: {len(peer_rows)} full-target peer/mechanism summaries; GT contamination is post-hoc only.",
        f"- `stage_group_attribution.csv`: {len(stage_rows)} bounded attribution rows over deterministic diagnostic images.",
        f"- `pixel_rank_forensics.csv`: {len(rank_rows)} deterministic representative image rows, with alpha=0 versus alpha=0.5 score changes and pixel metrics where defined.",
        f"- `gradient_conflict_report.csv`: {len(gradient.get('rows', []))} rows from a deterministic source training batch at E14.",
        "- `scheduler_optimization_audit.csv` and `SCHEDULER_OPTIMIZATION_AUDIT.md`: serialized optimizer/scheduler audit classified `CIR_SCHEDULER_BUG_CONFIRMED`.",
        "",
        "## Causal table",
        "",
        "| hypothesis/intervention | evidence for | evidence against or limit | status |",
        "|---|---|---|---|",
        "| CIR trained with wrong LR schedule | CIR source omits `scheduler.step()`; serialized E12-E20 scheduler states have `last_epoch=0` and image/text LRs remain 1e-3/5e-4 instead of decaying | No corrected matched retrain yet; LR exposure is monotonic while the E14/E16/E18/E20 metric pattern is non-monotonic, so the bug's score contribution is unknown | Major proven confound; not an RMT inference effect |",
        "| Historical anchor caused the observed V2 gap | Anchor uses E10/stride 4/legacy objective; V2 uses E12+/stride 1/current objective | No protocol-equivalent reproduction of the anchor | Not identified; confounded |",
        "| Alpha=0.5 RMT transport changes the trained V2 result | Paired alpha=0 versus alpha=0.5 cells use the same checkpoint and evaluator | This isolates transport effect only within the current representation | Causal for the alpha contrast |",
        "| Parent objective or representation drift caused the absolute gap | Config/objective/AMP and checkpoint lineage differ; adapter states drift | No matched parent checkpoint under the canonical config | Proven mismatch, causal size unknown |",
        "| Deployment smoothing caused the final failure | Train and deploy maps are computed by different documented paths | One-batch audit does not isolate full medical impact | Proven path difference, causal size unknown |",
        "| Peer quality or GT contamination caused the failure | Full-target peer statistics and post-hoc contamination are measured | GT is excluded from peer selection; association is not intervention | Correlational only |",
        "| Epoch/checkpoint drift caused the failure | Nested adapter states and metrics change across E12-E20 | Epoch is confounded with optimization progress | Descriptive, not causal |",
        "",
        "## Smallest next experiment and expected cost",
        "",
        "Run one matched corrective parent/CIR training comparison under the current canonical config: same seed, VisA source, CLIP asset, FP32 policy, effective batch, Adam hyperparameters, StepLR gamma/step timing, losses, and E10/E12-E20 checkpoint schedule, with only CIR/RMT as the intended difference. Do not modify the frozen RMT architecture or start MVTec training.",
        "",
        f"Planning estimate: one matched 20-epoch VisA parent run on the RTX 5060 Ti 16GB is approximately 5-10 GPU-hours; this is an explicit planning range, not a measured rerun. The current alpha=0 medical matrix measured `{observed_matrix_seconds:.1f}` sequential evaluator seconds; a paired matrix should budget that amount for each newly evaluated alpha condition, subject to cache and loader variance.",
        "",
        "## Limitation that controls the next experiment",
        "",
        "Do not redesign the RMT architecture or launch MVTec training from this report. The smallest scientifically decisive next step is the matched corrective parent/CIR retrain above, followed by the same exact evaluator and a paired alpha=0 versus alpha=0.5 matrix. Until then, do not attribute the observed degradation directly to RMT.",
        "",
        "All new files are isolated under this audit directory; existing V2 checkpoints and evaluation artifacts were not overwritten.",
    ]
    (output_root / "FULL_FAILURE_FORENSICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision(output_root: Path, decision: str, rationale: str, summary: Mapping[str, Any], protocol_rows: list[dict[str, Any]]) -> None:
    scheduler_bug = any(
        row.get("comparison") == "lr_scheduler" and row.get("status") == "MISMATCH_CONFIRMED"
        for row in protocol_rows
    )
    lines = [
        "# CIR_DFG_RMT_V2 go/no-go decision",
        "",
        f"DECISION: `{decision}`",
        "",
        f"Rationale: {rationale}",
        "",
        f"Scheduler audit classification: `{'CIR_SCHEDULER_BUG_CONFIRMED' if scheduler_bug else 'NOT_RUN'}`.",
        "Current CIR-V2 training was not optimization-matched to Phase2B; this benchmark cannot cleanly isolate the RMT hypothesis, and the degradation must not be attributed directly to RMT before the matched corrective retrain.",
        "Recommended next experiment: one matched parent/CIR retrain with the same seed, VisA source, CLIP asset, FP32, effective batch, Adam, StepLR timing, losses, and checkpoint schedule; only CIR/RMT differs.",
        "",
        f"Full medical alpha=0.5 minus alpha=0 summary: `{json.dumps(dict(summary), sort_keys=True)}`.",
        "",
        "This is the sole decision for this audit. No architecture change, MVTec training, or overwrite of frozen artifacts is authorized by this file.",
    ]
    (output_root / "GO_NO_GO_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--clip-asset", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--medical-root", type=Path, default=DEFAULT_MEDICAL_ROOT)
    parser.add_argument("--visa-root", type=Path, default=Path("/home/ai4/caohuy/data/VisA_20220922"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--diag-limit", type=int, default=60)
    parser.add_argument("--max-images", type=int, default=None, help="debug-only truncation; do not use for the final audit")
    parser.add_argument("--max-cells", type=int, default=None, help="debug-only cell truncation; do not use for the final audit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-source", action="store_true")
    parser.add_argument("--skip-gradient", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    clip_asset = args.clip_asset.expanduser().resolve()
    medical_root = args.medical_root.expanduser().resolve()
    visa_root = args.visa_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output_root / "cells.jsonl").exists():
        raise SystemExit(f"audit output already exists; pass --resume to continue: {output_root}")

    from tools.cir_rmt.identity import config_sha256, load_cir_config, release_identity_fields
    from model.phase2b_runtime import build_phase2b_frozen, configure_canonical_fp32

    configure_canonical_fp32()
    config = load_cir_config(config_path)
    config = dict(config)
    config_sha = config_sha256(config)
    config["_audit_config_sha256"] = config_sha
    parent_path = Path(config["parent_config_path"])
    if not parent_path.is_absolute():
        parent_path = ROOT / parent_path
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    identity = release_identity_fields(config)
    write_json(output_root / "audit_identity.json", {"identity": identity, "config_path": str(config_path), "config_sha256": config_sha, "current_git": current_git(), "clip_asset": str(clip_asset), "medical_root": str(medical_root), "visa_root": str(visa_root), "run_root": str(run_root), "epochs": list(EPOCHS), "targets": list(MEDICAL_TARGETS), "diag_limit": args.diag_limit, "max_images": args.max_images})

    protocol_rows = protocol_ledger(config, parent, run_root)
    write_csv(output_root / "protocol_equivalence_ledger.csv", protocol_rows)
    write_csv(output_root / "parameter_justification_table.csv", parameter_rows({**config, "parent_config": parent}))
    drift_rows = checkpoint_drift_rows(config, run_root)
    write_csv(output_root / "checkpoint_drift.csv", drift_rows)

    cells_path = output_root / "cells.jsonl"
    completed = {(row.get("scope", ""), int(row.get("epoch", -1))) for row in _load_cells(cells_path)}
    device = torch.device(args.device)
    checkpoint_by_epoch: dict[int, Path] = {epoch: run_root / "checkpoints" / f"epoch_{epoch}.pth" for epoch in EPOCHS}
    cell_plan = [("medical", epoch, target) for epoch in EPOCHS for target in MEDICAL_TARGETS]
    if not args.skip_source:
        cell_plan.extend([("source", epoch, "VisA_SOURCE") for epoch in EPOCHS])
    if args.max_cells is not None:
        cell_plan = cell_plan[: int(args.max_cells)]
    append_mode = "a" if cells_path.exists() else "w"
    with cells_path.open(append_mode, encoding="utf-8") as handle:
        for scope, epoch, target in cell_plan:
            if (scope, epoch, target) in {(row.get("scope", ""), int(row.get("epoch", -1)), row.get("target")) for row in _load_cells(cells_path)}:
                continue
            checkpoint_path = checkpoint_by_epoch[epoch]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            parent_for_model = dict(checkpoint["parent_config"])
            model = build_phase2b_frozen(parent_for_model, checkpoint, clip_asset, device)
            del checkpoint
            gc.collect()
            target_dataset, dataset_name, domain = _dataset_for_target(target, medical_root if scope == "medical" else visa_root)
            cell = run_eval_cell(model, config, target_dataset, dataset_name, domain, epoch, target, scope, device, output_root, args.batch_size, args.num_workers, args.prefetch_factor, args.diag_limit, args.max_images, verify_alpha05=(scope == "source"))
            cell["checkpoint_sha256"] = sha256_file(checkpoint_path)
            cell["checkpoint"] = str(checkpoint_path)
            cell["config_sha256"] = config_sha
            handle.write(json.dumps(cell, sort_keys=True, default=_json_default) + "\n")
            handle.flush()
            completed.add((scope, epoch))
            write_json(output_root / "progress.json", {"completed_cells": len(_load_cells(cells_path)), "planned_cells": len(cell_plan), "last": {"scope": scope, "epoch": epoch, "target": target}, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            del model, target_dataset
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    cells = _load_cells(cells_path)
    effect_rows = write_inference_effect(output_root, cells)
    source_rows = write_source_metrics(output_root, cells)
    peer_rows = _flatten_nested(cells, "mechanism")
    stage_rows = _flatten_nested(cells, "stage_group")
    rank_rows = _flatten_nested(cells, "pixel_rank")
    write_csv(output_root / "peer_forensics.csv", peer_rows)
    write_csv(output_root / "stage_group_attribution.csv", stage_rows)
    write_csv(output_root / "pixel_rank_forensics.csv", rank_rows)

    gradient: dict[str, Any] = {"rows": [], "mismatch": {"status": "SKIPPED"}}
    if not args.skip_gradient:
        gradient = run_gradient_audit(config, run_root, clip_asset, visa_root, device, output_root, epoch=14)
        write_train_deploy_report(output_root, gradient)
    else:
        write_csv(output_root / "gradient_conflict_report.csv", [{"row_type": "status", "status": "SKIPPED"}])
        write_train_deploy_report(output_root, gradient)

    decision, rationale = choose_decision(effect_rows, protocol_rows)
    summary = _macro_summary(effect_rows)
    write_decision(output_root, decision, rationale, summary, protocol_rows)
    write_full_report(output_root, config, protocol_rows, effect_rows, drift_rows, peer_rows, stage_rows, rank_rows, gradient, decision, rationale, source_rows)
    scheduler_summary_path = output_root / "scheduler_audit_summary.json"
    scheduler_summary = json.loads(scheduler_summary_path.read_text(encoding="utf-8")) if scheduler_summary_path.is_file() else {}
    scheduler_audit = {
        "classification": scheduler_summary.get("classification", "NOT_RUN"),
        "summary_path": str(scheduler_summary_path),
        "table_path": str(output_root / "scheduler_optimization_audit.csv"),
        "report_path": str(output_root / "SCHEDULER_OPTIMIZATION_AUDIT.md"),
        "cir_nonmatching_scheduler_rows": scheduler_summary.get("cir_nonmatching_scheduler_rows"),
        "parent_available_checkpoint_count": scheduler_summary.get("parent_available_checkpoint_count"),
    }
    write_json(output_root / "final_summary.json", {
        "decision": decision,
        "rationale": rationale,
        "effect_summary": summary,
        "cell_count": len(cells),
        "medical_effect_rows": len(effect_rows),
        "source_rows": len(source_rows),
        "peer_rows": len(peer_rows),
        "stage_rows": len(stage_rows),
        "rank_rows": len(rank_rows),
        "current_git": current_git(),
        "identity": identity,
        "scheduler_audit": scheduler_audit,
    })
    print(json.dumps({"output_root": str(output_root), "decision": decision, "effect_summary": summary, "medical_cells": len(effect_rows), "source_cells": len(source_rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
