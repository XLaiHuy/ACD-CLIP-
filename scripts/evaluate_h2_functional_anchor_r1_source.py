#!/usr/bin/env python3
"""Evaluate the fixed R1 bounded endpoints on a frozen VisA source subset.

This is inference-only. It never reads Medical/MVTec data, performs an
optimizer step, or selects a hyperparameter. The subset is the first four
normal and first four anomalous clean VisA images per category in manifest
order, fixed before any endpoint is loaded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))

from dataset import CLASS_NAMES, DOMAINS, get_text_and_image_dataset
from h2_clean.precision import PrecisionPolicy
from model.adapter import ACDCLIP
from model.clip import create_model
from utils import get_multiple_adapted_text_embedding

IMG = 518
PATCH = 37
ARM_ORDER = ("E1", "A_SHORT_R1", "A_FUNC_SHORT_R1")


def binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.uint8).reshape(-1)
    order = np.argsort(-x, kind="stable")
    sy = y[order]
    tp = np.cumsum(sy, dtype=np.float64)
    fp = np.cumsum(1 - sy, dtype=np.float64)
    positives = float(tp[-1])
    negatives = float(fp[-1])
    ap = float((tp / np.maximum(tp + fp, 1.0) * sy).sum() / positives)
    sx = x[order]
    ends = np.r_[np.flatnonzero(sx[1:] != sx[:-1]), len(sx) - 1]
    tpr = np.r_[0.0, tp[ends] / positives]
    fpr = np.r_[0.0, fp[ends] / negatives]
    return {"auroc": float(np.trapezoid(tpr, fpr)), "ap": ap}


def morphology(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    structure = np.ones((7, 7), dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=structure)
    dilated = ndimage.binary_dilation(mask, structure=structure)
    return mask & ~eroded, eroded, dilated & ~mask, ~dilated


def make_model(device: torch.device) -> ACDCLIP:
    clip = create_model(
        "ViT-L-14-336", img_size=IMG, device=device,
        pretrained="openai", require_pretrained=True,
    )
    model = ACDCLIP(
        clip_model=clip, n_groups=3, image_adapt_weight=.2, text_adapt_weight=.2,
        conv_lora_rank=8, conv_lora_alpha=2., conv_kernel_size_list=[3, 5],
        lora_rank=16, lora_alpha=2., dfg_mode="attn", dfg_attn_dim=256,
        dfg_attn_tau=8., use_ss2d_dfg=True, dfg_gamma_max=.2,
        dfg_ss2d_fusion="weight_residual", dfg_beta=.1,
        dfg_beta_schedule="warmup010", dfg_beta_target=.1,
        dfg_beta_current=.1, dfg_weight_residual_fp32=True,
        use_soft_prompt=False, soft_prompt_ctx_len=4,
        soft_prompt_init="phrase", soft_prompt_init_phrase="a photo of a",
    ).to(device).eval()
    for module in model.modules():
        if hasattr(module, "enable_fp16_numerical_islands"):
            module.enable_fp16_numerical_islands = False
    model.prompt_mode = "hybrid"
    model.use_hybrid_soft_prompt = True
    model.use_soft_prompt = False
    model.requires_grad_(False)
    return model


def endpoint_state(payload: dict) -> dict:
    state = payload.get("model_state")
    return state if state is not None else payload


def load_endpoint(model: ACDCLIP, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = endpoint_state(payload)
    model.image_adapter.load_state_dict(state["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(state["text_adapter"], strict=True)
    model.soft_prompt.load_state_dict(state["soft_prompt"], strict=True)
    model.dfg_beta = float(payload.get("dfg_beta_current", .1))
    model.hybrid_alpha_current = float(payload.get("hybrid_alpha_current", 0.0))
    model.hybrid_alpha_max = float(payload.get("hybrid_alpha_max", .2))
    model.soft_prompt_freeze_epochs = int(payload.get("soft_prompt_freeze_epochs", 3))
    return payload


def fixed_selection(datasets: dict) -> list[tuple[str, int, dict]]:
    selected = []
    for category in CLASS_NAMES["VisA"]:
        dataset = datasets[category]
        by_label = {0: [], 1: []}
        for index, meta in enumerate(dataset.meta):
            by_label[int(meta["label"])].append(index)
        for label in (0, 1):
            for index in by_label[label][:4]:
                selected.append((category, index, dataset.meta[index]))
    return selected


def maps_from_features(model, vision, text):
    batch, patches, _ = vision.shape[1:]
    side = int(math.sqrt(patches))
    group_text = text.unsqueeze(1).repeat(1, batch, 1, 1).permute(1, 0, 2, 3)
    stage_logits = []
    for stage in range(model.n_groups):
        fused = model._vision_text_attention_fusion(vision[stage], group_text, stage)
        logits = torch.matmul(10 * vision[stage], fused).permute(0, 2, 1).view(
            batch, 2, side, side
        )
        stage_logits.append(F.interpolate(
            logits, (IMG, IMG), mode="bilinear", align_corners=True,
        ))
    stage_logits = torch.stack(stage_logits)
    stage_probs = F.softmax(stage_logits, dim=2)[:, :, 1]
    final = model.vision_text_fusion_gate_seg(
        vision, text.unsqueeze(1).repeat(1, batch, 1, 1), img_size=IMG,
        test_mode=True, domain=DOMAINS["VisA"],
    )
    return final, stage_probs


def coverage(scores: np.ndarray, masks: np.ndarray) -> dict[str, dict[str, float]]:
    values = {k: [] for k in ("positive", "negative", "boundary", "interior", "near_background", "far_background")}
    for score, mask in zip(scores, masks):
        boundary, interior, near, far = morphology(mask.astype(bool))
        for key, region in (
            ("positive", mask.astype(bool)), ("negative", ~mask.astype(bool)),
            ("boundary", boundary), ("interior", interior),
            ("near_background", near), ("far_background", far),
        ):
            if region.any():
                values[key].append(score[region].astype(np.float32))
    result = {}
    for key, chunks in values.items():
        x = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
        result[key] = {
            "count": int(x.size),
            "mean": float(x.mean()) if x.size else float("nan"),
            "median": float(np.median(x)) if x.size else float("nan"),
            "p99": float(np.quantile(x, .99)) if x.size else float("nan"),
        }
    return result


def evaluate_arm(model, path: Path, datasets: dict, selected: list, device, policy, e1_features=None):
    payload = load_endpoint(model, path)
    feature_chunks = []
    mask_chunks = []
    final_chunks = []
    stage_chunks = []
    names = []
    geometry = []
    with torch.no_grad():
        for category in CLASS_NAMES["VisA"]:
            rows = [x for x in selected if x[0] == category]
            indices = [x[1] for x in rows]
            loader = DataLoader(Subset(datasets[category], indices), batch_size=8, shuffle=False, num_workers=0)
            for batch in loader:
                image = batch["image"].to(device)
                mask = batch["mask"][:, 0].numpy().astype(np.uint8)
                names.extend(list(batch["file_name"]))
                with policy.autocast(device):
                    text = get_multiple_adapted_text_embedding(model, "VisA", device)[category]
                    seg_tokens, _ = model(image)
                    vision = torch.stack(seg_tokens, dim=0)
                    final, stage_probs = maps_from_features(model, vision, text)
                feature_chunks.append(vision.float().cpu())
                mask_chunks.append(mask)
                final_chunks.append(final.float().cpu().numpy())
                stage_chunks.append(stage_probs.float().cpu().numpy().transpose(1, 0, 2, 3))
    features = torch.cat(feature_chunks, dim=1)
    masks = np.concatenate(mask_chunks, axis=0)
    finals = np.concatenate(final_chunks, axis=0)
    stages = np.concatenate(stage_chunks, axis=0)
    if e1_features is None:
        e1_features = features
    for stage in range(3):
        a = F.normalize(features[stage].float(), dim=-1)
        b = F.normalize(e1_features[stage].float(), dim=-1)
        geometry.append({
            "stage": stage + 1,
            "cosine_to_e1_mean": float((a * b).sum(-1).mean()),
            "cosine_to_e1_median": float((a * b).sum(-1).mean(-1).median()),
        })
    ranking = {"final": binary_metrics(finals, masks)}
    for stage in range(3):
        ranking[f"stage_{stage + 1}"] = binary_metrics(stages[:, stage], masks)
    return {
        "checkpoint": str(path),
        "sample_count": len(names),
        "file_names": names,
        "geometry": geometry,
        "ranking": ranking,
        "coverage": coverage(finals, masks),
        "stage_coverage": [coverage(stages[:, stage], masks) for stage in range(3)],
    }, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    paths = {
        "E1": REPO / "runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth",
        "A_SHORT_R1": root / "A_SHORT_R1/final.pth",
        "A_FUNC_SHORT_R1": root / "A_FUNC_SHORT_R1/final.pth",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    datasets = get_text_and_image_dataset("VisA", IMG, "test")
    selected = fixed_selection(datasets)
    model = make_model(device)
    results = {}
    e1_features = None
    for arm in ARM_ORDER:
        result, features = evaluate_arm(model, paths[arm], datasets, selected, device, policy, e1_features)
        if arm == "E1":
            e1_features = features
        results[arm] = result
        del features
        torch.cuda.empty_cache()
    control = results["A_SHORT_R1"]
    candidate = results["A_FUNC_SHORT_R1"]
    stage2_delta = candidate["geometry"][1]["cosine_to_e1_mean"] - control["geometry"][1]["cosine_to_e1_mean"]
    stage3_delta = candidate["geometry"][2]["cosine_to_e1_mean"] - control["geometry"][2]["cosine_to_e1_mean"]
    result = {
        "scope": "VisA clean source-only fixed subset; no optimizer steps; no Medical/MVTec/target data",
        "selection_rule": "first four normal and first four anomalous clean manifest-order images per VisA category",
        "sample_count": len(selected),
        "categories": CLASS_NAMES["VisA"],
        "arms": results,
        "candidate_minus_control": {
            "stage2_cosine_to_e1_delta": stage2_delta,
            "stage3_cosine_to_e1_delta": stage3_delta,
            "final_ap_delta": candidate["ranking"]["final"]["ap"] - control["ranking"]["final"]["ap"],
            "final_auroc_delta": candidate["ranking"]["final"]["auroc"] - control["ranking"]["final"]["auroc"],
            "positive_median_delta": candidate["coverage"]["positive"]["median"] - control["coverage"]["positive"]["median"],
            "interior_median_delta": candidate["coverage"]["interior"]["median"] - control["coverage"]["interior"]["median"],
            "stage1_ap_delta": candidate["ranking"]["stage_1"]["ap"] - control["ranking"]["stage_1"]["ap"],
        },
        "gate_definitions": {
            "stage2_geometry_improves": stage2_delta > 0.0,
            "stage3_geometry_improves": stage3_delta > 0.0,
            "positive_coverage_improves_or_preserved": result_value(candidate, "positive", "median") >= result_value(control, "positive", "median"),
            "interior_coverage_improves_or_preserved": result_value(candidate, "interior", "median") >= result_value(control, "interior", "median"),
            "stage1_not_degraded_in_ap": candidate["ranking"]["stage_1"]["ap"] >= control["ranking"]["stage_1"]["ap"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    csv_path = output.with_suffix(".csv")
    rows = []
    for arm in ARM_ORDER:
        for kind, metric in results[arm]["ranking"].items():
            rows.append({"arm": arm, "record": "ranking", "region": kind, **metric})
        for region, metric in results[arm]["coverage"].items():
            rows.append({"arm": arm, "record": "coverage", "region": region, **metric})
        for row in results[arm]["geometry"]:
            rows.append({"arm": arm, "record": "geometry", "region": f"stage_{row['stage']}", **row})
    fields = sorted({key for row in rows for key in row})
    import csv
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "PASS", "output": str(output), "csv": str(csv_path), "sample_count": len(selected)}, indent=2))


def result_value(result: dict, region: str, stat: str) -> float:
    return float(result["coverage"][region][stat])


if __name__ == "__main__":
    main()
