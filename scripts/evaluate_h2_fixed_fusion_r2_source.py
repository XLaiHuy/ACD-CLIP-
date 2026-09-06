#!/usr/bin/env python3
"""Evaluate the fixed-fusion R2 endpoints on the frozen VisA endpoint split."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from scipy import ndimage
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataset import CLASS_NAMES, DOMAINS, get_text_and_image_dataset
from h2_clean.contract import anchor_parameter_family, sha256_file
from h2_clean.precision import PrecisionPolicy
from h2_clean.stage_fusion import (
    H2_EQUAL_STAGE_FUSION_WEIGHTS,
    H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS,
    fuse_stage_logits,
)
from model.adapter import ACDCLIP
from model.clip import create_model
from utils import get_hybrid_soft_prompt_single_class_text_embedding


IMG = 518
PATCH = 37
START_CHECKPOINT = REPO / "runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth"
ENDPOINT_SUBSET = REPO / "audit/H2_FUSION_ENDPOINT_EVAL_SUBSET.csv"
SPLIT_IDENTITY = REPO / "audit/H2_FUSION_SPLIT_IDENTITY.json"
ARM_ORDER = ("E1", "A_FUSE_SHORT_R2_CONTROL", "A_FUSE_SHORT_R2_CANDIDATE")


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


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


def coverage(scores: np.ndarray, masks: np.ndarray) -> dict[str, dict[str, float]]:
    values = {
        key: [] for key in (
            "positive", "negative", "boundary", "interior",
            "near_background", "far_background",
        )
    }
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


def make_model(device: torch.device, weights) -> ACDCLIP:
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
        stage_fusion_weights=weights,
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
    model.dfg_beta = float(payload.get("dfg_beta_current", 0.0))
    model.hybrid_alpha_current = float(payload.get("hybrid_alpha_current", 0.0))
    model.hybrid_alpha_max = .2
    model.soft_prompt_freeze_epochs = 3
    return payload


def load_endpoint_selection() -> tuple[list[dict], dict[str, list[dict]]]:
    with ENDPOINT_SUBSET.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 96 or any(row["role"] != "ENDPOINT_EVAL" for row in rows):
        raise RuntimeError("endpoint subset identity is not the frozen 96-row endpoint split")
    by_category = {category: [] for category in CLASS_NAMES["VisA"]}
    for row in rows:
        by_category[row["category"]].append(row)
    if any(len(by_category[category]) != 8 for category in by_category):
        raise RuntimeError("endpoint subset category count mismatch")
    return rows, by_category


def stage_logits_from_features(model, vision, text):
    batch, patches, _ = vision.shape[1:]
    side = int(math.sqrt(patches))
    if side != PATCH:
        raise RuntimeError(f"unexpected patch grid {side}")
    group_text = text.unsqueeze(1).repeat(1, batch, 1, 1).permute(1, 0, 2, 3)
    stage_logits = []
    for stage in range(model.n_groups):
        fused = model._vision_text_attention_fusion(vision[stage], group_text, stage)
        logits = torch.matmul(10 * vision[stage], fused).permute(0, 2, 1).view(
            batch, 2, side, side,
        )
        logits = gaussian_blur2d(logits, (7, 7), (1, 1))
        stage_logits.append(F.interpolate(
            logits, (IMG, IMG), mode="bilinear", align_corners=True,
        ))
    return torch.stack(stage_logits)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    denom = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(np.linalg.norm(cross, "fro") ** 2 / denom) if denom else float("nan")


def parameter_drift(e1_state: dict, endpoint_state_value: dict) -> dict:
    groups = {
        "Conv-LoRA": lambda name: name.startswith("lora_adapters."),
        "image_projection": lambda name: any(name.startswith(prefix) for prefix in ("m_i_w.", "seg_proj.", "det_proj.", "seg_layer_norms.", "det_layer_norms.")),
        "DFG_QK": lambda name: name.startswith("vision_text_q.") or name.startswith("vision_text_k."),
        "SS2D": lambda name: name.startswith("dfg_ss2d_branches.") or name.startswith("dfg_raw_gamma.") or name.startswith("direction_logits."),
    }
    accum = {
        name: {"difference_sq": 0.0, "reference_sq": 0.0, "parameter_count": 0}
        for name in groups
    }
    family = {}
    for name, reference in e1_state.items():
        if name not in endpoint_state_value:
            raise RuntimeError(f"endpoint missing parameter {name}")
        live = endpoint_state_value[name].detach().float()
        ref = reference.detach().float()
        diff_sq = float((live - ref).square().sum().item())
        ref_sq = float(ref.square().sum().item())
        family_name = anchor_parameter_family(name)
        target = family.setdefault(family_name, {"difference_sq": 0.0, "reference_sq": 0.0, "parameter_count": 0})
        for target_map in (target,):
            target_map["difference_sq"] += diff_sq
            target_map["reference_sq"] += ref_sq
            target_map["parameter_count"] += ref.numel()
        for group_name, predicate in groups.items():
            if predicate(name):
                target_map = accum[group_name]
                target_map["difference_sq"] += diff_sq
                target_map["reference_sq"] += ref_sq
                target_map["parameter_count"] += ref.numel()
    def finish(values):
        return {
            "difference_l2": math.sqrt(values["difference_sq"]),
            "reference_l2": math.sqrt(values["reference_sq"]),
            "relative_l2": math.sqrt(values["difference_sq"] / max(values["reference_sq"], 1e-30)),
            "parameter_count": values["parameter_count"],
        }
    return {
        "anchor_families": {name: finish(value) for name, value in sorted(family.items())},
        "requested_families": {name: finish(value) for name, value in accum.items()},
    }


def evaluate_arm(
    model: ACDCLIP,
    checkpoint: Path,
    selected_rows: list[dict],
    selected_by_category: dict[str, list[dict]],
    device: torch.device,
    policy: PrecisionPolicy,
    e1_pooled: np.ndarray | None,
    e1_state: dict,
) -> tuple[dict, np.ndarray]:
    payload = load_endpoint(model, checkpoint)
    dataset_by_category = get_text_and_image_dataset("VisA", IMG, "test")
    expected_names = [row["file_name"] for row in selected_rows]
    names = []
    masks = []
    finals = []
    stages = []
    late_ratios = []
    pooled_chunks = [[] for _ in range(3)]
    with torch.no_grad():
        for category in CLASS_NAMES["VisA"]:
            rows = selected_by_category[category]
            dataset = dataset_by_category[category]
            indices = []
            index_by_name = {meta["image_path"]: index for index, meta in enumerate(dataset.meta)}
            for row in rows:
                if row["file_name"] not in index_by_name:
                    raise RuntimeError(f"endpoint file not found: {row['file_name']}")
                index = index_by_name[row["file_name"]]
                if str(dataset.meta[index]["label"]) != str(1 if row["label_type"] == "anomaly" else 0):
                    raise RuntimeError(f"endpoint label mismatch: {row['file_name']}")
                indices.append(index)
            loader = DataLoader(
                Subset(dataset, indices), batch_size=8, shuffle=False, num_workers=0,
            )
            text, _, _ = get_hybrid_soft_prompt_single_class_text_embedding(
                model, "VisA", category, device, return_kg=False,
            )
            for batch in loader:
                image = batch["image"].to(device)
                mask = batch["mask"][:, 0].numpy().astype(np.uint8)
                names.extend(list(batch["file_name"]))
                with policy.autocast(device):
                    seg_tokens, _ = model(image)
                    vision = torch.stack(seg_tokens)
                    logits = stage_logits_from_features(model, vision, text)
                    fused = fuse_stage_logits(logits, model.stage_fusion_weights)
                    final = F.softmax(fused, dim=1)[:, 1]
                    stage_probs = F.softmax(logits, dim=2)[:, :, 1]
                    pooled = F.normalize(vision.float(), dim=-1).mean(dim=2)
                stage_abs = logits.float().abs().mean(dim=(2, 3, 4))
                weights = torch.as_tensor(model.stage_fusion_weights, device=device, dtype=stage_abs.dtype)
                denominator = (weights[:, None] * stage_abs).sum(dim=0).clamp_min(1e-12)
                late = (weights[1] * stage_abs[1] + weights[2] * stage_abs[2]) / denominator
                masks.append(mask)
                finals.append(final.float().cpu().numpy())
                stages.append(stage_probs.float().cpu().numpy().transpose(1, 0, 2, 3))
                late_ratios.append(late.float().cpu().numpy())
                for stage in range(3):
                    pooled_chunks[stage].append(pooled[stage].cpu().numpy())
    if names != expected_names:
        raise RuntimeError("endpoint evaluation order does not match frozen subset")
    masks_array = np.concatenate(masks, axis=0)
    finals_array = np.concatenate(finals, axis=0)
    stages_array = np.concatenate(stages, axis=0)
    late_array = np.concatenate(late_ratios, axis=0)
    pooled_array = np.stack([np.concatenate(chunks, axis=0) for chunks in pooled_chunks], axis=0)
    ranking = {"final": binary_metrics(finals_array, masks_array)}
    for stage in range(3):
        ranking[f"stage_{stage + 1}"] = binary_metrics(stages_array[:, stage], masks_array)
    geometry = []
    if e1_pooled is not None:
        for stage in range(3):
            left = e1_pooled[stage]
            right = pooled_array[stage]
            left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
            right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
            cosine = (left_norm * right_norm).sum(axis=1)
            geometry.append({
                "stage": stage + 1,
                "pooled_feature_cosine_to_e1_mean": float(cosine.mean()),
                "pooled_feature_cosine_to_e1_median": float(np.median(cosine)),
                "linear_cka_to_e1": linear_cka(left, right),
            })
    state = endpoint_state(payload)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sample_count": len(names),
        "file_names": names,
        "ranking": ranking,
        "coverage": coverage(finals_array, masks_array),
        "stage_coverage": [coverage(stages_array[:, stage], masks_array) for stage in range(3)],
        "R_late": {
            "mean": float(late_array.mean()),
            "median": float(np.median(late_array)),
            "p99": float(np.quantile(late_array, .99)),
            "per_image": late_array.tolist(),
            "formula": "mean((w2*mean_abs(logit_stage2)+w3*mean_abs(logit_stage3))/(w1*mean_abs(logit_stage1)+w2*mean_abs(logit_stage2)+w3*mean_abs(logit_stage3)+1e-12))",
        },
        "geometry": geometry,
        "parameter_drift": parameter_drift(e1_state, state["image_adapter"]),
        "endpoint_weights": list(model.stage_fusion_weights),
        "endpoint_finite": bool(np.isfinite(finals_array).all() and np.isfinite(stages_array).all()),
    }, pooled_array


def result_value(result: dict, region: str, stat: str) -> float:
    return float(result["coverage"][region][stat])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/h2_fixed_fusion_bounded_r2")
    args = parser.parse_args()
    root = Path(args.root)
    selected_rows, selected_by_category = load_endpoint_selection()
    split = json.loads(SPLIT_IDENTITY.read_text())
    if split["intersection_count"] != 0 or split["endpoint_eval_count"] != 96:
        raise RuntimeError("frozen split identity is not disjoint 96-image endpoint")
    paths = {
        "E1": START_CHECKPOINT,
        "A_FUSE_SHORT_R2_CONTROL": root / "A_FUSE_SHORT_R2_CONTROL/final.pth",
        "A_FUSE_SHORT_R2_CANDIDATE": root / "A_FUSE_SHORT_R2_CANDIDATE/final.pth",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError([str(path) for path in paths.values() if not path.is_file()])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    weights = {
        "E1": H2_EQUAL_STAGE_FUSION_WEIGHTS,
        "A_FUSE_SHORT_R2_CONTROL": H2_EQUAL_STAGE_FUSION_WEIGHTS,
        "A_FUSE_SHORT_R2_CANDIDATE": H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS,
    }
    e1_payload = torch.load(START_CHECKPOINT, map_location="cpu", weights_only=False)
    e1_state = endpoint_state(e1_payload)["image_adapter"]
    model = make_model(device, weights["E1"])
    results = {}
    e1_pooled = None
    for arm in ARM_ORDER:
        model.stage_fusion_weights = tuple(weights[arm])
        result, pooled = evaluate_arm(
            model, paths[arm], selected_rows, selected_by_category,
            device, policy, e1_pooled, e1_state,
        )
        if arm == "E1":
            e1_pooled = pooled
            result["geometry"] = [
                {"stage": stage + 1, "pooled_feature_cosine_to_e1_mean": 1.0,
                 "pooled_feature_cosine_to_e1_median": 1.0,
                 "linear_cka_to_e1": 1.0}
                for stage in range(3)
            ]
        results[arm] = result
        del pooled
        torch.cuda.empty_cache()

    control = results["A_FUSE_SHORT_R2_CONTROL"]
    candidate = results["A_FUSE_SHORT_R2_CANDIDATE"]
    control_rows = list(csv.DictReader((REPO / "audit/H2_FUSION_R2_CONTROL.csv").open(newline="")))
    candidate_rows = list(csv.DictReader((REPO / "audit/H2_FUSION_R2_CANDIDATE.csv").open(newline="")))
    identity_keys = ("attempt_index", "epoch", "batch", "file_names", "image_sha256", "mask_sha256", "labels")
    exact_batch_match = (
        len(control_rows) == len(candidate_rows) == 500
        and all(all(left[key] == right[key] for key in identity_keys) for left, right in zip(control_rows, candidate_rows))
    )
    summaries = {}
    for arm in ("A_FUSE_SHORT_R2_CONTROL", "A_FUSE_SHORT_R2_CANDIDATE"):
        summaries[arm] = json.loads((root / arm / "summary.json").read_text())
    numerical_validity = all(
        summary["attempted_steps"] == 500
        and summary["nonfinite_loss_skips"] == 0
        and summary["nonfinite_grad_skips"] <= 1
        and summary["max_consecutive_nonfinite_grad_skips"] == 1
        and summary["optimizer_state_failures"] == 0
        and summary["parameter_corruption"] == 0
        and summary["batch_match_gate"] == "PASS"
        and summary["numerical_failure"] is None
        for summary in summaries.values()
    ) and exact_batch_match and control["endpoint_finite"] and candidate["endpoint_finite"]
    result = {
        "protocol_id": "H2_FIXED_E1_RELIABILITY_RESIDUAL_STAGE_FUSION_R2",
        "scope": "VisA-only source endpoint evaluation; calibration subset excluded; no Medical/MVTec/target evaluation",
        "manifest": "dataset/hub/VisA.jsonl",
        "manifest_sha256": split["manifest_sha256"],
        "endpoint_subset_sha256": sha256_file(ENDPOINT_SUBSET),
        "calibration_subset_evaluated": False,
        "endpoint_sample_count": len(selected_rows),
        "categories": CLASS_NAMES["VisA"],
        "weights": {arm: list(weights[arm]) for arm in ARM_ORDER},
        "arms": results,
        "training_summaries": summaries,
        "exact_batch_identity_match": exact_batch_match,
        "numerical_validity": numerical_validity,
        "candidate_minus_control": {
            "R_late_mean_delta": candidate["R_late"]["mean"] - control["R_late"]["mean"],
            "R_late_median_delta": candidate["R_late"]["median"] - control["R_late"]["median"],
            "final_ap_delta": candidate["ranking"]["final"]["ap"] - control["ranking"]["final"]["ap"],
            "final_auroc_delta": candidate["ranking"]["final"]["auroc"] - control["ranking"]["final"]["auroc"],
            "positive_mean_delta": result_value(candidate, "positive", "mean") - result_value(control, "positive", "mean"),
            "positive_median_delta": result_value(candidate, "positive", "median") - result_value(control, "positive", "median"),
            "interior_mean_delta": result_value(candidate, "interior", "mean") - result_value(control, "interior", "mean"),
            "interior_median_delta": result_value(candidate, "interior", "median") - result_value(control, "interior", "median"),
            "stage1_ap_delta": candidate["ranking"]["stage_1"]["ap"] - control["ranking"]["stage_1"]["ap"],
            "stage1_auroc_delta": candidate["ranking"]["stage_1"]["auroc"] - control["ranking"]["stage_1"]["auroc"],
            "stage2_ap_delta": candidate["ranking"]["stage_2"]["ap"] - control["ranking"]["stage_2"]["ap"],
            "stage3_ap_delta": candidate["ranking"]["stage_3"]["ap"] - control["ranking"]["stage_3"]["ap"],
        },
    }
    output = REPO / "audit/H2_FUSION_R2_ENDPOINT_EVAL.json"
    json_dump(output, result)
    rows = []
    for arm in ARM_ORDER:
        arm_result = results[arm]
        for metric_name, metric in arm_result["ranking"].items():
            rows.append({"arm": arm, "record": "ranking", "metric": metric_name, **metric})
        for region, metric in arm_result["coverage"].items():
            rows.append({"arm": arm, "record": "coverage", "metric": region, **metric})
        for region, metric in arm_result["R_late"].items():
            if region != "per_image" and region != "formula":
                rows.append({"arm": arm, "record": "R_late", "metric": region, "value": metric})
        for metric_name, metric in arm_result["parameter_drift"]["requested_families"].items():
            rows.append({"arm": arm, "record": "parameter_drift", "metric": metric_name, **metric})
        for geometry in arm_result["geometry"]:
            rows.append({"arm": arm, "record": "geometry", "metric": f"stage_{geometry['stage']}", **geometry})
    fields = sorted({key for row in rows for key in row})
    with (REPO / "audit/H2_FUSION_R2_ENDPOINT_EVAL.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "status": "PASS" if numerical_validity else "INVALID_NUMERICAL",
        "output": str(output),
        "csv": str(REPO / "audit/H2_FUSION_R2_ENDPOINT_EVAL.csv"),
        "sample_count": len(selected_rows),
        "exact_batch_identity_match": exact_batch_match,
    }, sort_keys=True))
    if not numerical_validity:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
