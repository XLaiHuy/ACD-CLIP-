#!/usr/bin/env python3
"""Frozen one-configuration Medical selection and MVTec holdout for NFUR-R2."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("/workspace/h2_nfur_r2_e20_medical_selected")
EPOCHS = tuple(range(1, 21))
PROTOCOL = "H2_NFUR_R2_E20_MEDICAL_SELECTED"
ANCHOR_A = {
    "pixel_auroc": 91.25178605739472,
    "pixel_ap": 39.46836969631676,
    "image_auroc": 75.2827266852061,
    "image_ap": 76.34336352348328,
}
PHASE2B = {"pixel_auroc": 90.98, "pixel_ap": 40.35, "image_auroc": 73.77, "image_ap": 74.24}

import sys
sys.path.insert(0, str(REPO))
sys.path.insert(1, "/workspace/ACD-CLIP-medical-test")
import scripts.run_h2_safe_anchor_equal_weighted_target_eval as base  # noqa: E402
from model.adapter import ACDCLIP  # noqa: E402
from model.clip import create_model  # noqa: E402


def load_training_summary() -> dict:
    payload = json.loads((REPO / "results/H2_NFUR_R2_E20_TRAINING_SUMMARY.json").read_text())
    payload["checkpoint_hashes"] = {str(row["epoch"]): row["sha256"] for row in payload["checkpoints"]}
    if set(payload["checkpoint_hashes"]) != {str(e) for e in EPOCHS}:
        raise RuntimeError("NFUR checkpoint manifest is not E1-E20 complete")
    return payload


def checkpoint_paths() -> dict[int, Path]:
    paths = {epoch: RUN_ROOT / f"adapter_{epoch}.pth" for epoch in EPOCHS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return paths


def validate_checkpoint(path: Path, epoch: int, summary: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != epoch:
        raise RuntimeError(f"epoch mismatch for {path}")
    if base.sha256_file(path) != summary["checkpoint_hashes"][str(epoch)]:
        raise RuntimeError(f"checkpoint hash mismatch for {path}")
    for key in ("image_adapter", "text_adapter", "soft_prompt"):
        if key not in payload:
            raise RuntimeError(f"missing inference state {key} in {path}")
    state = payload.get("model_state", {})
    for value in list(state.get("image_adapter", {}).values()) + list(state.get("text_adapter", {}).values()) + list(state.get("soft_prompt", {}).values()):
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite checkpoint tensor in {path}")
    return payload


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
        use_nfur=True,
        nfur_hidden_channels=32,
        nfur_delta_bound=0.25,
    ).to(device)
    model.eval()
    return model


def load_checkpoint(model: ACDCLIP, path: Path, config) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.image_adapter.load_state_dict(payload["image_adapter"])
    model.text_adapter.load_state_dict(payload["text_adapter"])
    model.soft_prompt.load_state_dict(payload["soft_prompt"])
    state = payload.get("model_state", {})
    if "nfur_refiner" in state:
        model.nfur_refiner.load_state_dict(state["nfur_refiner"])
    elif "nfur_refiner" in payload:
        model.nfur_refiner.load_state_dict(payload["nfur_refiner"])
    model.prompt_mode = "hybrid"
    model.use_soft_prompt = False
    model.use_hybrid_soft_prompt = True
    model.hybrid_alpha_current = float(payload.get("hybrid_alpha_current", 0.2))
    model.soft_prompt_ctx_len = int(payload.get("soft_prompt_ctx_len", config.soft_prompt_ctx_len))
    model.soft_prompt_freeze_epochs = int(payload.get("soft_prompt_freeze_epochs", 3))
    model.dfg_beta_schedule = payload.get("dfg_beta_schedule", config.dfg_beta_schedule)
    model.dfg_beta_target = float(payload.get("dfg_beta_target", config.dfg_beta_target))
    model.dfg_weight_residual_fp32 = bool(payload.get("dfg_weight_residual_fp32", True))
    model.set_dfg_beta(float(payload.get("dfg_beta_current", config.dfg_beta)))
    model.set_nfur_enabled(True)
    return int(payload["epoch"])


def patch_base_globals() -> None:
    base.RUN_ROOT = RUN_ROOT
    base.EPOCHS = EPOCHS
    base.PROTOCOL_ID = PROTOCOL
    base.MODES = {"equal": base.validate_stage_fusion_weights(base.H2_EQUAL_STAGE_FUSION_WEIGHTS)}
    base.validate_checkpoint = validate_checkpoint
    base.load_checkpoint = load_checkpoint
    base.make_model = make_model


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_medical() -> dict:
    patch_base_globals()
    summary = load_training_summary()
    config = base.evaluator_config(32, 0, 0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = make_model(config, device)
    trajectory = base.evaluate_medical(model, config, checkpoint_paths(), summary)
    base.write_trajectory(
        REPO / "results/H2_NFUR_R2_E1_E20_MEDICAL.csv",
        REPO / "results/H2_NFUR_R2_E1_E20_MEDICAL.json",
        trajectory,
        {
            "batch_size": 32,
            "num_workers": 0,
            "cuda_device": 0,
            "img_size": 518,
            "pixel_stride": 1,
            "metric_precision": "raw_exact",
            "prompt_config": "current_shared",
            "score_rule": "cls_only",
            "fusion": "equal_fixed_frozen",
            "checkpoint_epochs": list(EPOCHS),
            "target_hyperparameter_sweep": False,
            "fusion_weight_sweep": False,
        },
        subprocess.check_output(["git", "-C", "/workspace/ACD-CLIP-medical-test", "rev-parse", "HEAD"], text=True).strip(),
    )
    macros = [row for row in trajectory["macro"] if row["inference"] == "equal"]
    selected = max(macros, key=lambda row: (row["pixel_ap"], row["pixel_auroc"], -row["epoch"]))
    selection = {
        "protocol_id": PROTOCOL,
        "selection_metric": "MEAN_SIX_DATASET_PIXEL_AP",
        "selection_rule": "highest six-dataset mean pixel AP; tie pixel AUROC; tie earlier epoch",
        "selected_epoch": int(selected["epoch"]),
        "selected_checkpoint": str(RUN_ROOT / f"adapter_{selected['epoch']}.pth"),
        "selected_checkpoint_sha256": base.sha256_file(RUN_ROOT / f"adapter_{selected['epoch']}.pth"),
        "selected_medical": selected,
        "medical_candidates": macros,
        "medical_role": "TARGET_VALIDATION_FOR_EPOCH_SELECTION",
        "target_validation_used": True,
        "medical_untouched_zero_shot_claim_allowed": False,
        "mvtec_used_for_selection": False,
        "frozen_inference": {"fusion": "equal", "prompt_config": "current_shared", "score_rule": "cls_only", "pixel_stride": 1},
        "evaluator_commit": subprocess.check_output(["git", "-C", "/workspace/ACD-CLIP-medical-test", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_json(REPO / "results/H2_NFUR_R2_MEDICAL_SELECTION_FREEZE.json", selection)
    (REPO / "results/H2_NFUR_R2_MEDICAL_SELECTION_FREEZE.md").write_text("# H2 NFUR-R2 Medical selection freeze\n\n" + json.dumps(selection, indent=2, sort_keys=True) + "\n")
    return selection


def run_mvtec(selection: dict) -> dict:
    patch_base_globals()
    selected_epoch = int(selection["selected_epoch"])
    config = base.evaluator_config(32, 0, 0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = make_model(config, device)
    checkpoint = RUN_ROOT / f"adapter_{selected_epoch}.pth"
    summary = load_training_summary()
    validate_checkpoint(checkpoint, selected_epoch, summary)
    load_checkpoint(model, checkpoint, config)
    rows = []
    pending = []
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nfur-mvtec-pixel")
    try:
        datasets = base.get_text_and_image_dataset("MVTec", config.img_size, "test")
        with torch.no_grad():
            text_cache = base.build_text_cache(model, "MVTec", list(datasets), device, "current_shared")
        for class_name, dataset in datasets.items():
            pair = base.evaluate_dataset_pair(
                model, "MVTec", class_name, base.prepare_dataset(dataset, config),
                text_cache, device, config, selected_epoch, "0.9_cls_0.1_max", executor
            )
            pending.extend(pair.values())
            rows.extend(pair.values())
        base.resolve_pixel_metrics(pending)
    finally:
        executor.shutdown(wait=True)
        for row in pending:
            for key in ("_score_path", "_target_path"):
                path = row.pop(key, None)
                if path:
                    Path(path).unlink(missing_ok=True)
    old_epochs = base.EPOCHS
    base.EPOCHS = (selected_epoch,)
    trajectory = base.build_trajectory(rows, "MVTec", summary, mvtec_image_rule=True)
    base.EPOCHS = old_epochs
    base.write_trajectory(
        REPO / "results/H2_NFUR_R2_MVTEC_FINAL.csv",
        REPO / "results/H2_NFUR_R2_MVTEC_FINAL.json",
        trajectory,
        {
            "batch_size": 32,
            "num_workers": 0,
            "cuda_device": 0,
            "img_size": 518,
            "pixel_stride": 1,
            "metric_precision": "raw_exact",
            "prompt_config": "current_shared",
            "score_rule": "0.9_cls_0.1_max",
            "fusion": "equal_fixed_frozen",
            "selected_epoch_only": selected_epoch,
            "target_hyperparameter_sweep": False,
            "fusion_weight_sweep": False,
            "medical_selection_precedes_mvtec": True,
        },
        subprocess.check_output(["git", "-C", "/workspace/ACD-CLIP-medical-test", "rev-parse", "HEAD"], text=True).strip(),
    )
    macro = trajectory["macro"][0]
    result = {
        "protocol_id": PROTOCOL,
        "role": "POST_SELECTION_HOLDOUT_TARGET",
        "selected_epoch": selected_epoch,
        "selected_checkpoint_sha256": base.sha256_file(checkpoint),
        "selected_equal": macro,
        "medical_selection_used_mvtec": False,
        "evaluator_commit": subprocess.check_output(["git", "-C", "/workspace/ACD-CLIP-medical-test", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_json(REPO / "results/H2_NFUR_R2_MVTEC_FINAL.json", {**json.loads((REPO / "results/H2_NFUR_R2_MVTEC_FINAL.json").read_text()), "final_holdout": result})
    return result


def run_report(selection: dict, mvtec: dict) -> None:
    medical = json.loads((REPO / "results/H2_NFUR_R2_E1_E20_MEDICAL.json").read_text())
    selected = selection["selected_medical"]
    mv = mvtec["selected_equal"]
    report = {
        "protocol_id": PROTOCOL,
        "branch": subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip(),
        "parent_head": "284b12b6dc7802bcbafc02d7809549c40758f94b",
        "nfur": {
            "implemented": True,
            "design": "NFUR-R2",
            "added_parameters": 222401,
            "formulation": "equal-stage native margin disagreement plus low fused-margin uncertainty gates a zero-initialized bounded local residual",
        },
        "medical": {
            "selected_epoch": selection["selected_epoch"],
            "selected": selected,
            "delta_vs_h2_anchor_a_e15": {
                key: float(selected[key] - ANCHOR_A[key]) for key in ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")
            },
            "phase2b_context": PHASE2B,
            "phase2b_comparison_caveat": "descriptive only: older rounded pixel_stride=4 path versus raw exact pixel_stride=1 H2 path",
            "trajectory_artifact": "results/H2_NFUR_R2_E1_E20_MEDICAL.json",
            "target_validation_used": True,
            "medical_untouched_zero_shot_claim_allowed": False,
        },
        "mvtec": {
            "selected": mv,
            "delta_vs_h2_anchor_a_e15": {key: float(mv[key] - ref) for key, ref in {
                "pixel_auroc": 90.041289, "pixel_ap": 45.159349, "image_auroc": 89.816913, "image_ap": 94.779362,
            }.items()},
            "holdout_only_after_medical_selection": True,
        },
        "mechanism": "small native score-space residual addresses contextual coupling while preserving exact OFF identity; no finer-than-37x37 feature was available",
        "compute_overhead": {"parameters": 222401, "head": "Conv2d(772,32,3)+GELU+Conv2d(32,1,1)"},
        "final_interpretation": "MIXED_TRADEOFF",
        "published_surpass_claim_allowed": False,
        "published_surpass_claim_why": "Medical epoch selection used target validation and Phase2B uses a non-identical older evaluator; MVTec is a post-selection holdout, so no clean published-surpass claim is authorized.",
        "medical_artifact": str(REPO / "results/H2_NFUR_R2_E1_E20_MEDICAL.json"),
        "mvtec_artifact": str(REPO / "results/H2_NFUR_R2_MVTEC_FINAL.json"),
    }
    write_json(REPO / "results/H2_NFUR_R2_FINAL_REPORT.json", report)
    (REPO / "results/H2_NFUR_R2_FINAL_REPORT.md").write_text("# H2 NFUR-R2 final report\n\n" + json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("medical", "mvtec"), required=True)
    args = parser.parse_args()
    if args.mode == "medical":
        selection = run_medical()
        print(json.dumps({"status": "OK", "mode": "medical", "selected_epoch": selection["selected_epoch"]}, sort_keys=True))
    else:
        selection = json.loads((REPO / "results/H2_NFUR_R2_MEDICAL_SELECTION_FREEZE.json").read_text())
        mvtec = run_mvtec(selection)
        run_report(selection, mvtec)
        print(json.dumps({"status": "OK", "mode": "mvtec", "selected_epoch": selection["selected_epoch"]}, sort_keys=True))


if __name__ == "__main__":
    main()
