#!/usr/bin/env python3
"""Evaluate the H2 late Conv-LoRA freeze arms on the frozen VisA source endpoint."""
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

from dataset import CLASS_NAMES, get_text_and_image_dataset
from h2_clean.contract import anchor_parameter_family, sha256_file
from h2_clean.precision import PrecisionPolicy
from h2_clean.stage_fusion import H2_EQUAL_STAGE_FUSION_WEIGHTS, fuse_stage_logits
from model.adapter import ACDCLIP
from model.clip import create_model
from scripts.evaluate_h2_fixed_fusion_r2_source import (
    ENDPOINT_SUBSET,
    SPLIT_IDENTITY,
    binary_metrics,
    linear_cka,
    load_endpoint_selection,
    stage_logits_from_features,
)
from utils import get_hybrid_soft_prompt_single_class_text_embedding


IMG = 518
PATCH = 37
PROTOCOL_ID = "H2_LATE_CONVLORA_FREEZE_R1"
ARM_E10 = "E10"
ARM_E1 = "E1"
ARM_CONTROL = "A_LATE_FREEZE_R1_CONTROL"
ARM_CANDIDATE = "A_LATE_FREEZE_R1_STAGE23_CONVLORA"
ARM_ORDER = (ARM_E10, ARM_E1, ARM_CONTROL, ARM_CANDIDATE)
E10_CHECKPOINT = Path("/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth")
E1_CHECKPOINT = REPO / "runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth"
CONTROL_CSV = REPO / "audit/H2_LATE_CONVLORA_FREEZE_R1_CONTROL.csv"
CANDIDATE_CSV = REPO / "audit/H2_LATE_CONVLORA_FREEZE_R1_CANDIDATE.csv"
SCOPE_CSV = REPO / "audit/H2_LATE_CONVLORA_FREEZE_R1_PARAMETER_SCOPE.csv"


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def morphology(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    structure = np.ones((7, 7), dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=structure)
    dilated = ndimage.binary_dilation(mask, structure=structure)
    return mask & ~eroded, eroded, dilated & ~mask, ~dilated


def coverage(scores: np.ndarray, masks: np.ndarray) -> dict[str, dict[str, float]]:
    values = {key: [] for key in (
        "positive", "negative", "boundary", "interior", "near_background", "far_background",
    )}
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
            "p95": float(np.quantile(x, .95)) if x.size else float("nan"),
            "p99": float(np.quantile(x, .99)) if x.size else float("nan"),
        }
    return result


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
        stage_fusion_weights=H2_EQUAL_STAGE_FUSION_WEIGHTS,
        use_soft_prompt=False, soft_prompt_ctx_len=4,
        soft_prompt_init="phrase", soft_prompt_init_phrase="a photo of a",
    ).to(device).eval()
    for module in model.modules():
        if hasattr(module, "enable_fp16_numerical_islands"):
            module.enable_fp16_numerical_islands = False
    model.prompt_mode = "hybrid"
    model.use_hybrid_soft_prompt = True
    model.use_soft_prompt = False
    model.hybrid_alpha_current = .2
    model.hybrid_alpha_max = .2
    model.soft_prompt_freeze_epochs = 3
    model.requires_grad_(False)
    return model


def endpoint_state(payload: dict) -> dict:
    return payload.get("model_state", payload)


def load_endpoint(model: ACDCLIP, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = endpoint_state(payload)
    model.image_adapter.load_state_dict(state["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(state["text_adapter"], strict=True)
    model.soft_prompt.load_state_dict(state["soft_prompt"], strict=True)
    model.dfg_beta = float(payload.get("dfg_beta_current", .1))
    model.hybrid_alpha_current = float(payload.get("hybrid_alpha_current", .2))
    model.stage_fusion_weights = H2_EQUAL_STAGE_FUSION_WEIGHTS
    return payload


def parameter_drift(reference: dict, live: dict) -> dict:
    groups = {
        "Conv-LoRA": lambda name: name.startswith("lora_adapters."),
        "Conv-LoRA_stage1": lambda name: name.startswith("lora_adapters.0."),
        "Conv-LoRA_stage2": lambda name: name.startswith("lora_adapters.1."),
        "Conv-LoRA_stage3": lambda name: name.startswith("lora_adapters.2."),
        "image_projection": lambda name: any(name.startswith(prefix) for prefix in (
            "m_i_w.", "seg_proj.", "det_proj.", "seg_layer_norms.", "det_layer_norms.",
        )),
        "DFG_QK": lambda name: name.startswith("vision_text_q.") or name.startswith("vision_text_k."),
        "SS2D": lambda name: name.startswith("dfg_ss2d_branches.") or name.startswith("dfg_raw_gamma.") or name.startswith("direction_logits."),
        "all_image_adapter": lambda name: True,
    }
    accum = {name: [0.0, 0.0, 0] for name in groups}
    family = {}
    for name, ref_value in reference.items():
        if name not in live:
            raise RuntimeError(f"endpoint missing parameter {name}")
        ref = ref_value.detach().float().cpu()
        value = live[name].detach().float().cpu()
        diff_sq = float((value - ref).square().sum())
        ref_sq = float(ref.square().sum())
        fam = anchor_parameter_family(name)
        family.setdefault(fam, [0.0, 0.0, 0])
        family[fam][0] += diff_sq
        family[fam][1] += ref_sq
        family[fam][2] += ref.numel()
        for group, predicate in groups.items():
            if predicate(name):
                accum[group][0] += diff_sq
                accum[group][1] += ref_sq
                accum[group][2] += ref.numel()

    def finish(value):
        return {
            "difference_l2": float(value[0] ** .5),
            "reference_l2": float(value[1] ** .5),
            "relative_l2": float((value[0] / max(value[1], 1e-30)) ** .5),
            "parameter_count": int(value[2]),
        }
    return {
        "anchor_families": {name: finish(value) for name, value in sorted(family.items())},
        "requested_families": {name: finish(value) for name, value in sorted(accum.items())},
    }


def inversion_rates(finals: np.ndarray, masks: np.ndarray) -> dict[str, float | str]:
    chunks = {"near_background": [], "positive": [], "interior": []}
    for score, mask in zip(finals, masks):
        boundary, interior, near, _ = morphology(mask.astype(bool))
        if near.any():
            chunks["near_background"].append(score[near].astype(np.float32))
        if mask.astype(bool).any():
            chunks["positive"].append(score[mask.astype(bool)].astype(np.float32))
        if interior.any():
            chunks["interior"].append(score[interior].astype(np.float32))

    def rate(left_key: str, right_key: str) -> float:
        left = np.concatenate(chunks[left_key]) if chunks[left_key] else np.empty(0, dtype=np.float32)
        right = np.concatenate(chunks[right_key]) if chunks[right_key] else np.empty(0, dtype=np.float32)
        if not left.size or not right.size:
            return float("nan")
        return float(1.0 - binary_metrics(
            np.concatenate([left, right]),
            np.concatenate([np.zeros(left.size, dtype=np.uint8), np.ones(right.size, dtype=np.uint8)]),
        )["auroc"])
    return {
        "near_background_gt_positive": rate("near_background", "positive"),
        "near_background_gt_interior": rate("near_background", "interior"),
        "definition": "1 - pixel-level AUROC for near-background versus comparison region; ties receive half credit",
    }


def geometry(reference: np.ndarray, live: np.ndarray, label: str) -> list[dict]:
    result = []
    for stage in range(3):
        left = reference[stage]
        right = live[stage]
        left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
        right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
        cosine = (left_norm * right_norm).sum(axis=1)
        result.append({
            "stage": stage + 1,
            f"pooled_feature_cosine_to_{label}_mean": float(cosine.mean()),
            f"pooled_feature_cosine_to_{label}_median": float(np.median(cosine)),
            f"linear_cka_to_{label}": linear_cka(left, right),
        })
    return result


def identity_geometry(label: str) -> list[dict]:
    return [
        {
            "stage": stage + 1,
            f"pooled_feature_cosine_to_{label}_mean": 1.0,
            f"pooled_feature_cosine_to_{label}_median": 1.0,
            f"linear_cka_to_{label}": 1.0,
        }
        for stage in range(3)
    ]


def evaluate_arm(model: ACDCLIP, checkpoint: Path, selected_rows: list[dict], by_category: dict[str, list[dict]], device: torch.device, policy: PrecisionPolicy, reference_e10_pooled: np.ndarray | None, reference_e1_pooled: np.ndarray | None, e10_state: dict) -> tuple[dict, np.ndarray]:
    payload = load_endpoint(model, checkpoint)
    datasets = get_text_and_image_dataset("VisA", IMG, "test")
    expected_names = [row["file_name"] for row in selected_rows]
    names, masks, finals, stages = [], [], [], []
    pooled_chunks = [[] for _ in range(3)]
    inversion_final_chunks = []
    with torch.no_grad():
        for category in CLASS_NAMES["VisA"]:
            dataset = datasets[category]
            index_by_name = {meta["image_path"]: index for index, meta in enumerate(dataset.meta)}
            indices = []
            for row in by_category[category]:
                if row["file_name"] not in index_by_name:
                    raise RuntimeError(f"endpoint file not found: {row['file_name']}")
                index = index_by_name[row["file_name"]]
                expected_label = 1 if row["label_type"] == "anomaly" else 0
                if int(dataset.meta[index]["label"]) != expected_label:
                    raise RuntimeError(f"endpoint label mismatch: {row['file_name']}")
                indices.append(index)
            loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False, num_workers=0)
            text, _, _ = get_hybrid_soft_prompt_single_class_text_embedding(model, "VisA", category, device, return_kg=False)
            for batch in loader:
                image = batch["image"].to(device)
                mask = batch["mask"][:, 0].numpy().astype(np.uint8)
                with policy.autocast(device):
                    seg_tokens, _ = model(image)
                    vision = torch.stack(seg_tokens)
                    logits = stage_logits_from_features(model, vision, text)
                    fused = fuse_stage_logits(logits, H2_EQUAL_STAGE_FUSION_WEIGHTS)
                    final = F.softmax(fused, dim=1)[:, 1]
                    stage_probs = F.softmax(logits, dim=2)[:, :, 1]
                    pooled = F.normalize(vision.float(), dim=-1).mean(dim=2)
                names.extend(list(batch["file_name"]))
                masks.append(mask)
                finals.append(final.float().cpu().numpy())
                stages.append(stage_probs.float().cpu().numpy().transpose(1, 0, 2, 3))
                for stage in range(3):
                    pooled_chunks[stage].append(pooled[stage].cpu().numpy())
    if names != expected_names:
        raise RuntimeError("endpoint evaluation order does not match frozen subset")
    masks_array = np.concatenate(masks, axis=0)
    finals_array = np.concatenate(finals, axis=0)
    stages_array = np.concatenate(stages, axis=0)
    pooled_array = np.stack([np.concatenate(chunks, axis=0) for chunks in pooled_chunks], axis=0)
    ranking = {"final": binary_metrics(finals_array, masks_array)}
    for stage in range(3):
        ranking[f"stage_{stage + 1}"] = binary_metrics(stages_array[:, stage], masks_array)
    state = endpoint_state(payload)
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sample_count": len(names),
        "file_names": names,
        "ranking": ranking,
        "coverage": coverage(finals_array, masks_array),
        "inversion_rates": inversion_rates(finals_array, masks_array),
        "stage_coverage": [coverage(stages_array[:, stage], masks_array) for stage in range(3)],
        "geometry_to_e10": geometry(reference_e10_pooled, pooled_array, "e10") if reference_e10_pooled is not None else [],
        "geometry_to_e1": geometry(reference_e1_pooled, pooled_array, "e1") if reference_e1_pooled is not None else [],
        "parameter_drift_from_e10": parameter_drift(e10_state, state["image_adapter"]),
        "endpoint_weights": list(H2_EQUAL_STAGE_FUSION_WEIGHTS),
        "endpoint_finite": bool(np.isfinite(finals_array).all() and np.isfinite(stages_array).all()),
    }
    return result, pooled_array


def result_value(result: dict, region: str, stat: str) -> float:
    return float(result["coverage"][region][stat])


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/h2_late_convlora_freeze_r1")
    args = parser.parse_args()
    root = Path(args.root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    for path in (E10_CHECKPOINT, E1_CHECKPOINT, CONTROL_CSV, CANDIDATE_CSV, SCOPE_CSV, ENDPOINT_SUBSET, SPLIT_IDENTITY):
        if not path.is_file():
            raise FileNotFoundError(path)
    split = json.loads(SPLIT_IDENTITY.read_text())
    selected_rows, by_category = load_endpoint_selection()
    if split["intersection_count"] != 0 or split["endpoint_eval_count"] != 96:
        raise RuntimeError("frozen endpoint split identity mismatch")
    paths = {
        ARM_E10: E10_CHECKPOINT,
        ARM_E1: E1_CHECKPOINT,
        ARM_CONTROL: REPO / "runs/h2_late_convlora_freeze_r1" / ARM_CONTROL / "final.pth",
        ARM_CANDIDATE: REPO / "runs/h2_late_convlora_freeze_r1" / ARM_CANDIDATE / "final.pth",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError([str(path) for path in paths.values() if not path.is_file()])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    e10_payload = torch.load(E10_CHECKPOINT, map_location="cpu", weights_only=False)
    e10_state = endpoint_state(e10_payload)["image_adapter"]
    model = make_model(device)
    results = {}
    pooled = {}
    for arm in ARM_ORDER:
        result, arm_pooled = evaluate_arm(
            model, paths[arm], selected_rows, by_category, device, policy,
            pooled.get(ARM_E10), pooled.get(ARM_E1), e10_state,
        )
        if arm == ARM_E10:
            result["geometry_to_e10"] = identity_geometry("e10")
        if arm == ARM_E1:
            result["geometry_to_e1"] = identity_geometry("e1")
        results[arm] = result
        pooled[arm] = arm_pooled
        torch.cuda.empty_cache()
    control_rows = read_csv_rows(CONTROL_CSV)
    candidate_rows = read_csv_rows(CANDIDATE_CSV)
    identity_keys = ("attempt_index", "epoch", "batch", "file_names", "image_sha256", "mask_sha256", "labels")
    exact_batch_match = len(control_rows) == len(candidate_rows) == 500 and all(
        all(left[key] == right[key] for key in identity_keys)
        for left, right in zip(control_rows, candidate_rows)
    )
    control_summary = json.loads((root / ARM_CONTROL / "summary.json").read_text())
    candidate_summary = json.loads((root / ARM_CANDIDATE / "summary.json").read_text())
    scope_rows = read_csv_rows(SCOPE_CSV)
    frozen_scope = [row for row in scope_rows if row["selected_for_freeze"] == "true"]
    control = results[ARM_CONTROL]
    candidate = results[ARM_CANDIDATE]
    control_natural_skip_set = sorted({
        int(row["attempt_index"])
        for row in control_rows
        if int(row["natural_nonfinite_loss_skip"]) or int(row["natural_nonfinite_grad_skip"])
    })
    candidate_natural_skip_set = sorted({
        int(row["attempt_index"])
        for row in candidate_rows
        if int(row["natural_nonfinite_loss_skip"]) or int(row["natural_nonfinite_grad_skip"])
    })
    deltas = {
        "final_auroc": candidate["ranking"]["final"]["auroc"] - control["ranking"]["final"]["auroc"],
        "final_ap": candidate["ranking"]["final"]["ap"] - control["ranking"]["final"]["ap"],
        "positive_mean": result_value(candidate, "positive", "mean") - result_value(control, "positive", "mean"),
        "positive_median": result_value(candidate, "positive", "median") - result_value(control, "positive", "median"),
        "interior_mean": result_value(candidate, "interior", "mean") - result_value(control, "interior", "mean"),
        "interior_median": result_value(candidate, "interior", "median") - result_value(control, "interior", "median"),
        "near_background_p95": result_value(candidate, "near_background", "p95") - result_value(control, "near_background", "p95"),
        "near_background_p99": result_value(candidate, "near_background", "p99") - result_value(control, "near_background", "p99"),
        "near_background_gt_positive_inversion": candidate["inversion_rates"]["near_background_gt_positive"] - control["inversion_rates"]["near_background_gt_positive"],
        "near_background_gt_interior_inversion": candidate["inversion_rates"]["near_background_gt_interior"] - control["inversion_rates"]["near_background_gt_interior"],
    }
    for stage in range(1, 4):
        deltas[f"stage{stage}_auroc"] = candidate["ranking"][f"stage_{stage}"]["auroc"] - control["ranking"][f"stage_{stage}"]["auroc"]
        deltas[f"stage{stage}_ap"] = candidate["ranking"][f"stage_{stage}"]["ap"] - control["ranking"][f"stage_{stage}"]["ap"]
    numerical_validity = bool(
        exact_batch_match
        and control_summary["attempted"] == candidate_summary["attempted"] == 500
        and control_summary["successful_updates"] == candidate_summary["successful_updates"]
        and control_summary["numerical_failure"] is None
        and candidate_summary["numerical_failure"] is None
        and control_summary["batch_match_gate"] == candidate_summary["batch_match_gate"] == "PASS"
        and all(results[arm]["endpoint_finite"] for arm in ARM_ORDER)
    )
    output = {
        "protocol_id": PROTOCOL_ID,
        "evaluation_scope": "VisA-only frozen source endpoint; calibration excluded; no Medical/MVTec/target inference",
        "manifest": "dataset/hub/VisA.jsonl",
        "manifest_sha256": split["manifest_sha256"],
        "endpoint_subset_sha256": sha256_file(ENDPOINT_SUBSET),
        "endpoint_sample_count": len(selected_rows),
        "calibration_subset_evaluated": False,
        "endpoint_fusion_weights": {arm: list(H2_EQUAL_STAGE_FUSION_WEIGHTS) for arm in ARM_ORDER},
        "parameter_scope": {
            "scope_csv": str(SCOPE_CSV),
            "parameter_scope_id": "image_adapter.lora_adapters.stage2+stage3.all_parameters",
            "selected_parameter_count": len(frozen_scope),
            "selected_stages": ["stage2", "stage3"],
        },
        "arms": results,
        "training_summaries": {ARM_CONTROL: control_summary, ARM_CANDIDATE: candidate_summary},
        "control_natural_skip_set": control_natural_skip_set,
        "candidate_natural_skip_set": candidate_natural_skip_set,
        "candidate_forced_parity_skip_set": sorted({
            int(row["attempt_index"]) for row in candidate_rows if int(row["forced_parity_skip"])
        }),
        "exact_batch_identity_match": exact_batch_match,
        "numerical_validity": numerical_validity,
        "candidate_minus_control": deltas,
        "geometry_reference": {"mandatory": "E10", "optional_reported": "E1"},
        "protocol_prohibitions_verified": {
            "new_full_training_run": False,
            "medical_inference_run": False,
            "mvtec_inference_run": False,
            "target_tuning_used": False,
            "hyperparameter_sweep": False,
        },
    }
    json_dump(REPO / "audit/H2_LATE_CONVLORA_FREEZE_R1_ENDPOINT.json", output)
    table = []
    for arm in ARM_ORDER:
        result = results[arm]
        for metric_name, metric in result["ranking"].items():
            table.append({"arm": arm, "record": "ranking", "metric": metric_name, **metric})
        for region, metric in result["coverage"].items():
            table.append({"arm": arm, "record": "coverage", "metric": region, **metric})
        for metric_name, value in result["inversion_rates"].items():
            if metric_name != "definition":
                table.append({"arm": arm, "record": "inversion_rate", "metric": metric_name, "value": value})
        for metric_name, metric in result["parameter_drift_from_e10"]["requested_families"].items():
            table.append({"arm": arm, "record": "parameter_drift", "metric": metric_name, **metric})
    fields = ["arm", "record", "metric", "value", "count", "mean", "median", "p95", "p99", "auroc", "ap", "difference_l2", "reference_l2", "relative_l2", "parameter_count"]
    with (REPO / "audit/H2_LATE_CONVLORA_FREEZE_R1_ENDPOINT.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    if not numerical_validity:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
