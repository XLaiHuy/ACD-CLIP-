#!/usr/bin/env python3
"""Evaluate Safe-Anchor E10--E20 under equal and fixed weighted fusion.

Medical evaluation uses the pinned phase2cd evaluator's exact metric/data
semantics.  The paired loop reuses its bounded-memory pixel metric and prompt
cache while running both fusion modes from the same current-model forward
pass.  MVTec is guarded by the committed Medical selection freeze.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
EXTERNAL_EVALUATOR = Path("/workspace/ACD-CLIP-medical-test")
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(EXTERNAL_EVALUATOR))

import phase2b_anchor_diagnosis as pinned_metrics
from dataset import CLASS_NAMES, DOMAINS, get_text_and_image_dataset
from h2_clean.stage_fusion import (
    H2_EQUAL_STAGE_FUSION_WEIGHTS,
    H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS,
    validate_stage_fusion_weights,
)
from model import adapter as current_adapter
from model.adapter import ACDCLIP
from model.clip import create_model
from phase2b_anchor_diagnosis import (
    IMAGE_DATASETS,
    PIXEL_DATASETS,
    build_text_cache,
    compute_cls_scores,
    metric_or_none,
    prepare_dataset,
)
from phase2cd_medical_eval import evaluator_args


# Keep the pinned evaluator's exact sort/merge implementation, but use larger
# internal chunks for this run's billion-pixel medical datasets.  These values
# change only I/O and working-set size, not ordering, tie handling, or metric
# formulas.
pinned_metrics._PIXEL_SORT_CHUNK = 64_000_000
pinned_metrics._PIXEL_MERGE_CHUNK = 8_000_000


PROTOCOL_ID = "H2_SAFE_ANCHOR_E20_MEDICAL_SELECTED"
RUN_ROOT = Path("/workspace/h2_safe_anchor_e20_medical_selected")
EPOCHS = tuple(range(10, 21))
MODES = {
    "equal": validate_stage_fusion_weights(H2_EQUAL_STAGE_FUSION_WEIGHTS),
    "weighted": validate_stage_fusion_weights(H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS),
}
MEDICAL_PIXEL_DATASETS = tuple(PIXEL_DATASETS)
MEDICAL_IMAGE_DATASETS = tuple(IMAGE_DATASETS)
MVTEC_DATASETS = ("MVTec",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def evaluator_config(batch_size: int, num_workers: int, cuda_device: int):
    return evaluator_args(argparse.Namespace(
        batch_size=batch_size,
        num_workers=num_workers,
        cuda_device=cuda_device,
        img_size=518,
        pixel_stride=1,
        max_samples=None,
    ))


def checkpoint_paths() -> dict[int, Path]:
    paths = {epoch: RUN_ROOT / f"adapter_{epoch}.pth" for epoch in EPOCHS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing retained checkpoints: {missing}")
    return paths


def validate_checkpoint(path: Path, epoch: int, expected_config: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != epoch:
        raise RuntimeError(f"checkpoint epoch mismatch: {path}")
    if sha256_file(path) != expected_config["checkpoint_hashes"][str(epoch)]:
        raise RuntimeError(f"checkpoint hash changed after training summary: {path}")
    scientific = payload.get("resolved_scientific_config")
    if not isinstance(scientific, dict):
        raise RuntimeError(f"missing scientific identity: {path}")
    checks = {
        "dataset": "VisA",
        "model_name": "ViT-L-14-336",
        "img_size": 518,
        "n_groups": 3,
        "batch_size": 6,
        "seed": 0,
        "training_horizon": 20,
        "use_safe_anchor": True,
        "anchor_lambda": 0.0021633926715180626,
        "anchor_family_budget": 0.1,
        "anchor_reference_sha256": "7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35",
        "use_cir_training": False,
        "tf32_enabled": False,
        "precision_protocol": "HISTORICAL_MIXED_FP16_FP32_V1",
    }
    mismatch = {key: (value, scientific.get(key)) for key, value in checks.items() if scientific.get(key) != value}
    if mismatch:
        raise RuntimeError(f"checkpoint scientific identity mismatch: {path}: {mismatch}")
    if payload.get("functional_feature_anchor", False) or scientific.get("use_functional_feature_anchor", False):
        raise RuntimeError(f"functional feature anchor unexpectedly enabled: {path}")
    required = ("image_adapter", "text_adapter", "soft_prompt")
    if any(key not in payload for key in required):
        raise RuntimeError(f"checkpoint missing inference state: {path}")
    return payload


def load_checkpoint(model: ACDCLIP, path: Path, config) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.image_adapter.load_state_dict(payload["image_adapter"])
    model.text_adapter.load_state_dict(payload["text_adapter"])
    model.soft_prompt.load_state_dict(payload["soft_prompt"])
    model.prompt_mode = "hybrid"
    model.use_soft_prompt = False
    model.use_hybrid_soft_prompt = True
    model.hybrid_alpha_current = float(payload["hybrid_alpha_current"])
    model.soft_prompt_ctx_len = int(payload["soft_prompt_ctx_len"])
    model.soft_prompt_freeze_epochs = int(payload["soft_prompt_freeze_epochs"])
    model.dfg_beta_schedule = payload["dfg_beta_schedule"]
    model.dfg_beta_target = float(payload["dfg_beta_target"])
    model.dfg_weight_residual_fp32 = bool(payload["dfg_weight_residual_fp32"])
    model.set_dfg_beta(float(payload["dfg_beta_current"]))
    return int(payload["epoch"])


def make_model(config, device: torch.device) -> ACDCLIP:
    clip_model = create_model(
        model_name=config.model_name,
        img_size=config.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=config.n_groups,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        conv_lora_rank=config.conv_lora_rank,
        conv_lora_alpha=config.conv_lora_alpha,
        conv_kernel_size_list=config.conv_kernel_size_list,
        dfg_mode=config.dfg_mode,
        dfg_attn_dim=config.dfg_attn_dim,
        dfg_attn_tau=config.dfg_attn_tau,
        use_ss2d_dfg=config.use_ss2d_dfg,
        dfg_gamma_max=config.dfg_gamma_max,
        dfg_ss2d_fusion=config.dfg_ss2d_fusion,
        dfg_beta=config.dfg_beta,
        dfg_beta_schedule=config.dfg_beta_schedule,
        dfg_beta_target=config.dfg_beta_target,
        dfg_beta_current=config.dfg_beta,
        dfg_weight_residual_fp32=True,
        use_soft_prompt=True,
        soft_prompt_ctx_len=config.soft_prompt_ctx_len,
        soft_prompt_init=config.soft_prompt_init,
        soft_prompt_init_phrase=config.soft_prompt_init_phrase,
    ).to(device)
    model.eval()
    return model


def exact_pixel_metrics(score_path: Path, target_path: Path, count: int) -> tuple[float, float]:
    # The pinned evaluator owns the exact bounded-memory implementation.
    from phase2b_anchor_diagnosis import _exact_binary_pixel_metrics_from_files
    return _exact_binary_pixel_metrics_from_files(str(score_path), str(target_path), count)


def open_spool() -> tuple[tempfile.NamedTemporaryFile, tempfile.NamedTemporaryFile]:
    score = tempfile.NamedTemporaryFile(prefix="h2_target_scores_", suffix=".float32", delete=False)
    target = tempfile.NamedTemporaryFile(prefix="h2_target_labels_", suffix=".uint8", delete=False)
    score.close()
    target.close()
    return score, target


def unlink_spools(spools) -> None:
    for pair in spools.values():
        for item in pair:
            try:
                Path(item.name).unlink()
            except FileNotFoundError:
                pass


def evaluate_dataset_pair(
        model,
        dataset_name: str,
        class_name: str,
        dataset,
        text_cache,
        device,
        config,
        epoch: int,
        image_score_rule: str,
        metric_executor: ThreadPoolExecutor | None = None,
):
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    spools = {mode: open_spool() for mode in MODES}
    handles = {mode: (open(pair[0].name, "wb"), open(pair[1].name, "wb")) for mode, pair in spools.items()}
    counts = defaultdict(int)
    raw_rows = {mode: [] for mode in MODES}
    class_text = text_cache[class_name]
    try:
        for input_data in tqdm(loader, desc=f"{dataset_name}/{class_name}/e{epoch}"):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device).to(torch.int32)
            labels = input_data["label"].to(torch.int32)
            names = input_data["file_name"]
            class_names = input_data["class_name"]
            if len(set(class_names)) != 1:
                raise RuntimeError("mixed class batch is unsupported")
            with torch.no_grad():
                seg_tokens, det_tokens = model(image)
                seg_features = torch.stack(seg_tokens, dim=0)
                det_features = torch.stack(det_tokens, dim=0)
                batch_size = seg_features.shape[1]
                cls_text = class_text["cls"].unsqueeze(dim=1).repeat(1, batch_size, 1, 1)
                seg_text = class_text["seg"].unsqueeze(dim=1).repeat(1, batch_size, 1, 1)
                cls_scores = compute_cls_scores(det_features, cls_text)
                for mode, weights in MODES.items():
                    model.stage_fusion_weights = weights
                    seg_pred = model.vision_text_fusion_gate_seg(
                        seg_features, seg_text, test_mode=True, domain=DOMAINS[dataset_name]
                    )
                    flat = seg_pred.flatten(start_dim=1)
                    max_pixel = flat.max(dim=1).values
                    top1pct = flat.topk(max(1, math.ceil(flat.shape[1] * 0.01)), dim=1).values.mean(dim=1)
                    handles[mode][0].write(seg_pred.detach().float().cpu().numpy().reshape(-1).tobytes())
                    handles[mode][1].write(mask.detach().to(torch.uint8).cpu().numpy().reshape(-1).tobytes())
                    counts[mode] += int(seg_pred.numel())
                    for index, name in enumerate(names):
                        raw_rows[mode].append({
                            "dataset": dataset_name,
                            "epoch": epoch,
                            "prompt_config": "current_shared",
                            "file_name": name,
                            "label": int(labels[index].item()),
                            "cls_score": float(cls_scores[index].cpu().item()),
                            "max_pixel": float(max_pixel[index].cpu().item()),
                            "top1pct_pixel": float(top1pct[index].cpu().item()),
                        })
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for score, target in handles.values():
            score.close()
            target.close()
    pixel_tasks = {
        mode: (
            metric_executor.submit(
                exact_pixel_metrics,
                Path(spools[mode][0].name),
                Path(spools[mode][1].name),
                counts[mode],
            )
            if metric_executor is not None
            else exact_pixel_metrics(
                Path(spools[mode][0].name), Path(spools[mode][1].name), counts[mode]
            )
        )
        for mode in MODES
    }
    metrics = {}
    for mode in MODES:
        if image_score_rule == "cls_only":
            image_scores = [row["cls_score"] for row in raw_rows[mode]]
        elif image_score_rule == "0.9_cls_0.1_max":
            image_scores = [0.9 * row["cls_score"] + 0.1 * row["max_pixel"] for row in raw_rows[mode]]
        else:
            raise ValueError(f"unknown image score rule: {image_score_rule}")
        labels = [row["label"] for row in raw_rows[mode]]
        image_auc, image_ap = metric_or_none(image_scores, labels, round_result=False)
        if metric_executor is None:
            auc_value, ap_value = pixel_tasks[mode]
        else:
            auc_value = ap_value = None
        metrics[mode] = {
            "mode": mode,
            "dataset": dataset_name,
            "class_name": class_name,
            "epoch": epoch,
            "prompt_config": "current_shared",
            "score_rule": image_score_rule,
            "pixel_auroc": None if auc_value is None else auc_value * 100.0,
            "pixel_ap": None if ap_value is None else ap_value * 100.0,
            "image_auroc": image_auc,
            "image_ap": image_ap,
            "sample_count": len(raw_rows[mode]),
            "pixel_count": counts[mode],
        }
        if metric_executor is not None:
            metrics[mode]["_pixel_future"] = pixel_tasks[mode]
            metrics[mode]["_score_path"] = spools[mode][0].name
            metrics[mode]["_target_path"] = spools[mode][1].name
    if metric_executor is None:
        unlink_spools(spools)
    return metrics


def resolve_pixel_metrics(rows: list[dict]) -> None:
    for row in rows:
        future = row.pop("_pixel_future", None)
        if future is None:
            continue
        try:
            auc_value, ap_value = future.result()
            row["pixel_auroc"] = float(auc_value * 100.0)
            row["pixel_ap"] = float(ap_value * 100.0)
        finally:
            for key in ("_score_path", "_target_path"):
                path = row.pop(key, None)
                if path is not None:
                    try:
                        Path(path).unlink()
                    except FileNotFoundError:
                        pass


def evaluate_medical(model, config, checkpoints: dict[int, Path], summary: dict) -> dict:
    rows = []
    pending = []
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pinned-pixel")
    try:
        for epoch, checkpoint in checkpoints.items():
            validate_checkpoint(checkpoint, epoch, summary)
            load_checkpoint(model, checkpoint, config)
            for dataset_name in MEDICAL_PIXEL_DATASETS:
                datasets = get_text_and_image_dataset(dataset_name, config.img_size, "test")
                device = next(model.parameters()).device
                with torch.no_grad():
                    text_cache = build_text_cache(model, dataset_name, list(datasets), device, "current_shared")
                for class_name, dataset in datasets.items():
                    pair = evaluate_dataset_pair(
                        model, dataset_name, class_name, prepare_dataset(dataset, config),
                        text_cache, device, config, epoch, "cls_only", executor
                    )
                    pending.extend(pair.values())
                    rows.extend(pair.values())
        resolve_pixel_metrics(pending)
    finally:
        executor.shutdown(wait=True)
        for row in pending:
            for key in ("_score_path", "_target_path"):
                path = row.pop(key, None)
                if path is not None:
                    try:
                        Path(path).unlink()
                    except FileNotFoundError:
                        pass
    return build_trajectory(rows, "Medical", summary)


def evaluate_mvtec(model, config, checkpoints: dict[int, Path], summary: dict) -> dict:
    freeze = REPO / "results/H2_MEDICAL_SELECTED_EPOCH_FREEZE.json"
    if not freeze.is_file():
        raise RuntimeError("MVTec is guarded: Medical selection freeze is absent")
    freeze_payload = json.loads(freeze.read_text())
    if freeze_payload.get("mvtec_observed_before_selection") is not False:
        raise RuntimeError("MVTec ordering guard failed")
    rows = []
    pending = []
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pinned-pixel")
    try:
        for epoch, checkpoint in checkpoints.items():
            validate_checkpoint(checkpoint, epoch, summary)
            load_checkpoint(model, checkpoint, config)
            dataset_name = "MVTec"
            datasets = get_text_and_image_dataset(dataset_name, config.img_size, "test")
            device = next(model.parameters()).device
            with torch.no_grad():
                text_cache = build_text_cache(model, dataset_name, list(datasets), device, "current_shared")
            for class_name, dataset in datasets.items():
                pair = evaluate_dataset_pair(
                    model, dataset_name, class_name, prepare_dataset(dataset, config),
                    text_cache, device, config, epoch, "0.9_cls_0.1_max", executor
                )
                pending.extend(pair.values())
                rows.extend(pair.values())
        resolve_pixel_metrics(pending)
    finally:
        executor.shutdown(wait=True)
        for row in pending:
            for key in ("_score_path", "_target_path"):
                path = row.pop(key, None)
                if path is not None:
                    try:
                        Path(path).unlink()
                    except FileNotFoundError:
                        pass
    return build_trajectory(rows, "MVTec", summary, mvtec_image_rule=True)


def build_trajectory(rows: list[dict], domain: str, summary: dict, mvtec_image_rule: bool = False) -> dict:
    # Pair evaluation returns cls-only image metrics for Medical. For MVTec,
    # the final image metric is filled by the dedicated one-pass path below.
    normalized = []
    for row in rows:
        if "mode" not in row:
            raise RuntimeError("internal evaluator row missing inference mode")
        normalized.append(row)
    macro = []
    for epoch in EPOCHS:
        for mode in MODES:
            subset = [row for row in normalized if row["epoch"] == epoch and row["mode"] == mode]
            if domain == "Medical":
                pixel = subset
                images = [row for row in subset if row["dataset"] in MEDICAL_IMAGE_DATASETS and row["image_auroc"] is not None]
            else:
                pixel = subset
                images = [row for row in subset if row["image_auroc"] is not None]
            if not pixel:
                raise RuntimeError(f"missing {domain} metrics for E{epoch} {mode}")
            macro.append({
                "domain": domain,
                "epoch": epoch,
                "inference": mode,
                "scope": "macro",
                "dataset": "ALL_MEDICAL_MACRO" if domain == "Medical" else "ALL_MVTEC_MACRO",
                "pixel_auroc": sum(row["pixel_auroc"] for row in pixel) / len(pixel),
                "pixel_ap": sum(row["pixel_ap"] for row in pixel) / len(pixel),
                "image_auroc": None if not images else sum(row["image_auroc"] for row in images) / len(images),
                "image_ap": None if not images else sum(row["image_ap"] for row in images) / len(images),
                "dataset_count": len(pixel),
                "image_dataset_count": len(images),
                "valid": True,
            })
    return {"protocol_id": PROTOCOL_ID, "domain": domain, "rows": normalized, "macro": macro, "summary": summary}


def write_trajectory(path_csv: Path, path_json: Path, trajectory: dict, config: dict, evaluator_commit: str) -> None:
    rows = []
    for row in trajectory["rows"]:
        rows.append({**row, "scope": "dataset", "valid": True})
    rows.extend(trajectory["macro"])
    fields = [
        "domain", "epoch", "inference", "scope", "dataset", "class_name",
        "prompt_config", "score_rule", "pixel_auroc", "pixel_ap",
        "image_auroc", "image_ap", "sample_count", "pixel_count",
        "dataset_count", "image_dataset_count", "valid",
    ]
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = path_csv.with_name(path_csv.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path_csv)
    payload = {
        **trajectory,
        "evaluation_config": config,
        "evaluator_repo": str(EXTERNAL_EVALUATOR),
        "evaluator_commit": evaluator_commit,
        "medical_before_mvtec": trajectory["domain"] == "Medical",
        "target_selection": "none",
        "target_hyperparameter_sweep": False,
        "fusion_weight_sweep": False,
        "inference_modes": {key: list(value) for key, value in MODES.items()},
    }
    json_dump(path_json, payload)


def read_training_summary() -> dict:
    payload = json.loads((REPO / "audit/H2_SAFE_ANCHOR_E20_TRAINING_SUMMARY.json").read_text())
    if payload.get("milestone_epoch") != 20 or not payload.get("all_retained_checkpoints_numerically_valid"):
        raise RuntimeError("training summary is not complete and numerically valid")
    if payload.get("medical_used_during_training") or payload.get("mvtec_used_during_training"):
        raise RuntimeError("target data was recorded during training")
    payload["checkpoint_hashes"] = {str(row["epoch"]): row["sha256"] for row in payload["checkpoints"]}
    if set(payload["checkpoint_hashes"]) != {str(epoch) for epoch in EPOCHS}:
        raise RuntimeError("training summary does not retain E10-E20")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("medical", "mvtec"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/h2_safe_anchor_target_eval"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    if current_adapter.__file__ is None or Path(current_adapter.__file__).resolve() != (REPO / "model/adapter.py").resolve():
        raise RuntimeError(f"wrong model implementation imported: {current_adapter.__file__}")
    evaluator_commit = subprocess_commit(EXTERNAL_EVALUATOR)
    if evaluator_commit != "6bd932fbce0a425af5c8d3f7230dd7dc041568bd":
        raise RuntimeError(f"pinned evaluator mismatch: {evaluator_commit}")
    if args.mode == "mvtec" and not (REPO / "results/H2_MEDICAL_SELECTED_EPOCH_FREEZE.json").is_file():
        raise RuntimeError("MVTec cannot run before Medical selection freeze")
    training_summary = read_training_summary()
    paths = checkpoint_paths()
    config = evaluator_config(args.batch_size, args.num_workers, args.cuda_device)
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = make_model(config, device)
    trajectory = evaluate_medical(model, config, paths, training_summary) if args.mode == "medical" else evaluate_mvtec(model, config, paths, training_summary)
    suffix = "MEDICAL" if args.mode == "medical" else "MVTEC"
    write_trajectory(
        REPO / f"results/H2_E10_E20_{suffix}_EQUAL_WEIGHTED.csv",
        REPO / f"results/H2_E10_E20_{suffix}_EQUAL_WEIGHTED.json",
        trajectory,
        {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "cuda_device": args.cuda_device,
            "img_size": config.img_size,
            "pixel_stride": 1,
            "metric_precision": "raw_exact",
            "prompt_config": "current_shared",
            "medical_image_score_rule": "cls_only",
            "mvtec_image_score_rule": "0.9_cls_0.1_max",
            "checkpoint_epochs": list(EPOCHS),
        },
        evaluator_commit,
    )
    print(json.dumps({"status": "PASS", "domain": args.mode, "rows": len(trajectory["rows"]), "macro_rows": len(trajectory["macro"])}, sort_keys=True))


def subprocess_commit(root: Path) -> str:
    import subprocess
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


if __name__ == "__main__":
    main()
