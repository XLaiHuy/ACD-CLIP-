#!/usr/bin/env python3
"""Select one frozen Phase2B checkpoint using development MVTec only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluation.datasets import MVTecDatasetAdapter, resolve_mvtec_root
from evaluation.evaluator import evaluate_records, image_score
from evaluation.metrics import selection_score
from dataset.info import dataset_domain
from model.phase2b_runtime import configure_canonical_fp32, forward_phase2b, load_json_config, load_phase2b_checkpoint

PROTOCOL_VERSION = "PHASE2B_CANONICAL_V1"
DEFAULT_EPOCHS = (10, 12, 14, 16, 18, 20)
METRIC_NAMES = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint_hash(path: Path, expected: str | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected is not None and actual != str(expected):
        raise ValueError(f"checkpoint SHA256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def select_candidate(candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Select by preregistered score; exact ties choose the earlier epoch."""
    rows = []
    required = {"epoch", "path", *METRIC_NAMES}
    for candidate in candidates:
        missing = required - set(candidate)
        if missing:
            raise ValueError(f"candidate missing fields: {sorted(missing)}")
        metrics = {name: float(candidate[name]) for name in METRIC_NAMES}
        if not all(0.0 <= value <= 1.0 for value in metrics.values()):
            raise ValueError("candidate metrics must be in [0,1]")
        rows.append(dict(candidate) | {"score": selection_score(metrics)})
    if not rows:
        raise ValueError("no Phase2B candidates supplied")
    return min(rows, key=lambda row: (-float(row["score"]), int(row["epoch"])))


def _metric_row(path: Path, epoch: int, result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result.get("macro", result)
    if not isinstance(metrics, Mapping) or any(name not in metrics for name in METRIC_NAMES):
        raise ValueError("candidate evaluation must return four exact macro metrics")
    row = {"epoch": int(epoch), "path": str(path)}
    row.update({name: float(metrics[name]) for name in METRIC_NAMES})
    row["sha256"] = verify_checkpoint_hash(path)
    return row


def evaluate_checkpoint_candidates(
    checkpoint_paths: Sequence[Path],
    candidate_epochs: Sequence[int],
    evaluate_one: Callable[[Path, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(checkpoint_paths) != len(candidate_epochs):
        raise ValueError("checkpoint count must match candidate epoch count")
    rows = []
    for path, epoch in zip(checkpoint_paths, candidate_epochs):
        verify_checkpoint_hash(path)
        rows.append(_metric_row(path, int(epoch), evaluate_one(path, int(epoch))))
    return rows


def _evaluate_real_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    adapter: MVTecDatasetAdapter,
    config_path: Path,
    clip_asset: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    checkpoint_sha = verify_checkpoint_hash(checkpoint_path)
    config = load_json_config(config_path)
    model = load_phase2b_checkpoint(checkpoint_path, config, clip_asset, device)
    model.eval()
    loader = DataLoader(adapter, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=device.type == "cuda", persistent_workers=bool(num_workers))
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch in tqdm(loader, desc=f"[Phase2B select] E{epoch}", leave=False):
        images = batch["image"].to(device, non_blocking=device.type == "cuda").float()
        classes = list(batch["class_name"])
        result = forward_phase2b(model, images, classes, device, config, domain="Industrial", require_grad=False, dataset_name="MVTec")
        pixels = result.native_segmentation_probability.detach().cpu().numpy()
        cls = result.classification_probability.detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy().astype(np.int8)
        masks = batch["mask"].detach().cpu().numpy().astype(np.int8)
        paths = list(batch["image_path"])
        for index, class_name in enumerate(classes):
            pixel_scores = pixels[index].reshape(-1)
            records.append({"class_name": class_name, "pixel_scores": pixel_scores, "image_scores": np.asarray([image_score(float(cls[index]), float(pixel_scores.max()), dataset_domain("MVTec"))]), "pixel_labels": masks[index].reshape(-1), "image_labels": np.asarray([labels[index]]), "image_path": paths[index]})
        processed = len(records)
        elapsed = max(time.perf_counter() - started, 1e-9)
        print(f"[Phase2B select] E{epoch} images={processed} img/s={processed/elapsed:.2f}", end="\r", flush=True)
    print()
    result = evaluate_records(records, method="phase2b")
    result["checkpoint_sha256"] = checkpoint_sha
    result["epoch"] = int(epoch)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _output_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"epoch": int(row["epoch"]), "path": str(row["path"]), "sha256": str(row.get("sha256", "")), "pAUROC": float(row["pixel_auroc"]), "pAP": float(row["pixel_ap"]), "iAUROC": float(row["image_auroc"]), "iAP": float(row["image_ap"]), "score": float(row["score"])} for row in candidates]


def _write_selection(output_dir: Path, candidates: list[dict[str, Any]], selected: dict[str, Any], code_sha: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _output_rows(candidates)
    selected_output = next(row for row in outputs if int(row["epoch"]) == int(selected["epoch"]))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN",
        "source_training_dataset": "VisA",
        "development_dataset": "MVTecAD",
        "score": {"unit": "[0,1]", "formula": ".35*pAUROC+.35*pAP+.15*iAUROC+.15*iAP", "weights": {"pixel_auroc": 0.35, "pixel_ap": 0.35, "image_auroc": 0.15, "image_ap": 0.15}},
        "candidates": outputs,
        "selected_epoch": int(selected["epoch"]),
        "selected_checkpoint": str(selected["path"]),
        "selected_checkpoint_sha256": selected_output["sha256"],
        "evaluator": {"exact": True, "external_memory_ready": True, "pixel_stride": 1, "image_score_contract": "domain-specific frozen contract", "code_sha": code_sha},
    }
    (output_dir / "phase2b_selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "phase2b_selection_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "path", "sha256", "pAUROC", "pAP", "iAUROC", "iAP", "score"])
        writer.writeheader(); writer.writerows(outputs)
    print(f"SELECTED E*={selected['epoch']} checkpoint={selected['path']} SHA256={selected_output['sha256']} score={float(selected['score']):.6f}")


def _candidate_paths(args: argparse.Namespace, epochs: Sequence[int]) -> list[Path]:
    paths = list(args.checkpoint)
    if args.checkpoint_dir is not None:
        paths.extend(args.checkpoint_dir / f"adapter_{epoch}.pth" for epoch in epochs)
    if len(paths) != len(epochs):
        raise SystemExit("provide exactly one checkpoint per candidate epoch")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--candidate-epochs", type=str, default=",".join(map(str, DEFAULT_EPOCHS)))
    parser.add_argument("--mvtec-root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2b_canonical_v1.json"))
    parser.add_argument("--clip-asset", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("phase2b_selection"))
    parser.add_argument("--metrics-json", type=Path, help="bounded/debug-only precomputed candidate metrics")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    configure_canonical_fp32()
    if args.preflight_only:
        root = resolve_mvtec_root(args.mvtec_root)
        if root is None:
            print("MVTEC_PREFLIGHT=NOT_RUN_NO_ROOT"); return 0
        report = MVTecDatasetAdapter(root).preflight()
        print(json.dumps(report, indent=2, sort_keys=True)); return 0
    root = resolve_mvtec_root(args.mvtec_root)
    if root is None:
        raise SystemExit("MVTec development root is required; use --mvtec-root <PATH>")
    epochs = tuple(int(value) for value in args.candidate_epochs.split(",") if value.strip())
    if not epochs or epochs != tuple(sorted(set(epochs))):
        raise SystemExit("--candidate-epochs must be unique and ascending")
    paths = _candidate_paths(args, epochs)
    rows: list[dict[str, Any]] = []
    if args.metrics_json is not None:
        raw = json.loads(args.metrics_json.read_text(encoding="utf-8"))
        if not isinstance(raw, list): raise SystemExit("--metrics-json must contain a list")
        by_epoch = {int(row["epoch"]): row for row in raw}
        for path, epoch in zip(paths, epochs):
            if epoch not in by_epoch: raise SystemExit(f"metrics JSON has no candidate epoch {epoch}")
            rows.append(_metric_row(path, epoch, by_epoch[epoch]))
    else:
        if args.clip_asset is None or not args.clip_asset.is_file():
            raise SystemExit("real selector path requires --clip-asset")
        adapter = MVTecDatasetAdapter(root)
        adapter.preflight(inspect_limit=0)
        for index, (path, epoch) in enumerate(zip(paths, epochs), start=1):
            print(f"[Phase2B select] checkpoint {index}/{len(paths)} E{epoch}")
            result = _evaluate_real_checkpoint(path, epoch, adapter, args.config, args.clip_asset, torch.device(args.device), args.batch_size, args.num_workers)
            rows.append(_metric_row(path, epoch, result))
            print("Epoch | pAUROC | pAP | iAUROC | iAP | Score | Time")
            print(f"{epoch} | {rows[-1]['pixel_auroc']:.4f} | {rows[-1]['pixel_ap']:.4f} | {rows[-1]['image_auroc']:.4f} | {rows[-1]['image_ap']:.4f} | {selection_score({name: rows[-1][name] for name in METRIC_NAMES}):.4f}")
    selected = select_candidate(rows)
    _write_selection(args.output_dir, rows, selected, code_sha=sha256_file(Path(__file__).resolve()))
    return 0


def evaluate_candidate_records(records: Iterable[Mapping[str, Any]]) -> dict[str, float | None]:
    return evaluate_records(records, method="phase2b")["macro"]


if __name__ == "__main__":
    raise SystemExit(main())
