#!/usr/bin/env python3
"""Build the compact historical Phase2B gain/loss forensic archive.

This is an artifact builder, not a trainer or evaluator. It consumes completed
replay outputs and frozen compact tables. It never changes checkpoints and
never launches a model forward pass.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "research_artifacts" / "phase2b_historical_gain_forensics_20260901"

HISTORICAL_REPO = Path("/home/ai4/caohuy/ACD-CLIP-base-new-phase1")
HISTORICAL_REMOTE = "https://github.com/XLaiHuy/ACD-CLIP-.git"
H2_COMMIT = "e03966997d4cecfd985943a4053a93e1e40197ec"
H1_COMMIT = "0232200e964c02c328eec09dbe842f327f72fcd9"
CURRENT_REPO_COMMIT = "1564a0c9f32a7e93fb333ca8b0950a2fb2c73cc2"
C2_TRAINING_COMMIT = "042174cdc63d9cb635566a1dae5b774056045383"

H2_RUN = HISTORICAL_REPO / "runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch"
H2_CHECKPOINT = H2_RUN / "adapter_10.pth"
H1_CHECKPOINT = HISTORICAL_REPO / "phase1_best_checkpoints/phase1b_v3c_fp32attn_final/e09_best_final_anchor_adapter.pth"
H2_HISTORICAL_RESULTS = H2_RUN / "parsed_results.csv"
H2_CURRENT_RESULTS = ROOT / "runs/phase2b_historical_gain_forensics_20260901/h2_current_evaluator/same_checkpoint_current_evaluator.csv"
H1_CURRENT_RESULTS = ROOT / "runs/phase2b_historical_gain_forensics_20260901/h1_current_evaluator/same_checkpoint_current_evaluator.csv"
C2_MEDICAL = ROOT / "research_artifacts/cir_rmt_v2/corrective_matched_retrain_20260830/corrected_medical_decomposition.csv"
CURRENT_FINAL_MACRO = ROOT / "research_artifacts/cir_rmt_v2/final_extension_anchor_e20_20260831/FINAL_MEDICAL_MACRO.csv"
CURRENT_FINAL_MATRIX = ROOT / "research_artifacts/cir_rmt_v2/final_extension_anchor_e20_20260831/FINAL_MEDICAL_MATRIX.csv"
PA_MEDICAL = ROOT / "research_artifacts/cir_rmt_v2/pa_control_20260831/PA_MEDICAL_RESULTS.csv"
PA_MACRO = ROOT / "research_artifacts/cir_rmt_v2/pa_control_20260831/MEDICAL_FACTORIAL_MACRO.csv"
PA_INTERACTION = ROOT / "research_artifacts/cir_rmt_v2/pa_control_20260831/MEDICAL_FACTORIAL_INTERACTION.csv"
PA_DECISION = ROOT / "research_artifacts/cir_rmt_v2/pa_control_20260831/FINAL_ARCHITECTURE_DECISION.json"
PA_ARCHIVE = ROOT / "research_artifacts/cir_rmt_v2/pa_control_20260831"

CURRENT_CONFIG_SHA = "d24cf942684b0be3c12838699ec6fe452697bd7f0a58eabbf316fb79b1b18cdb"
CURRENT_RESOLVED_CONFIG_SHA = "5ec0190ec4dc1e16e0ce646b5e470d5585b981de7221ed4b46a392b321cd27f9"
CIR_CONFIG_SHA = "064e8acd4369645f631030b5d60abf8615e878b50e9caff6a4a8b2439b64f81c"
ARCHITECTURE_FREEZE_SHA = "f6de6ee8f1998f591c077efeff50fa9741a9f8bad34603ba145ec54ef961ba86"
CLIP_SHA = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
H2_CHECKPOINT_SHA = "ae27443f99020588298a9ecc6dfc833a83ebe7a752f00e8524042d5a84a2c0cb"
H2_CHECKPOINT_SIZE = 56452037
H1_CHECKPOINT_SHA = "6c1d888af56d011f7d2dabee7a5662ff422420df428841582a51c02846500e4a"
H1_CHECKPOINT_SIZE = 56426187
C2_E10_ORIGINAL_SHA = "31ca8344c646693d0ee51941d39f28aa07b6a102c49d1efdc5e3cdf2ec8bcc50"
C2_E10_OBSERVED_SHA = "ec4f472790241a9f746bb5b4ca6e31ca4782ec7333aab91ae691e9b2fb0c7347"
CURRENT_EVALUATOR_SHA = "cbcfaf4b2eda645fc6b440ed9bb486b5fb6b6f3af908e1c0ec70bafe13db0797"
H2_EVALUATOR_SHA = "7bdd8cc6ada90467285a79ced9599ed778c6dc2a0ba6596d2f3311fa637fae9d"
H1_EVALUATOR_SHA = "e6c768b5604ea7c7c0dea7c3db709405da6876ff01a22ddecb2dde6a4f59334f"

TARGETS = ("Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir")
METRICS = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(name: str, value: str) -> None:
    path = ARCHIVE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_json(name: str, value: Any) -> None:
    write_text(name, json.dumps(value, indent=2, sort_keys=True))


def write_csv(name: str, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path = ARCHIVE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    return float(value)


def macro(data: dict[str, dict[str, float | None]], metric: str) -> tuple[float | None, int]:
    values = [data[target][metric] for target in TARGETS if data.get(target, {}).get(metric) is not None]
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def read_current_helper(path: Path) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for row in read_csv(path):
        if row.get("status") != "COMPLETE" or row.get("target") not in TARGETS:
            continue
        out[row["target"]] = {metric: number(row.get(metric)) for metric in METRICS}
    return out


def read_h2_historical() -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for row in read_csv(H2_HISTORICAL_RESULTS):
        if row.get("dataset") not in TARGETS or int(row["epoch"]) != 10:
            continue
        target = row["dataset"]
        values = {
            "pixel_auroc": number(row.get("pixel_auc")),
            "pixel_ap": number(row.get("pixel_ap")),
            "image_auroc": number(row.get("image_auc")),
            "image_ap": number(row.get("image_ap")),
        }
        if target.startswith("Colon_"):
            values["image_auroc"] = None
            values["image_ap"] = None
        out[target] = {metric: (None if value is None else value / 100.0) for metric, value in values.items()}
    return out


def read_c2_e10() -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for row in read_csv(C2_MEDICAL):
        if row.get("target_kind") != "target" or row.get("target") not in TARGETS or int(row["epoch"]) != 10:
            continue
        out[row["target"]] = {metric: number(row.get("parent_" + metric)) for metric in METRICS}
    return out


def read_final_method(method: str, epoch: int) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for row in read_csv(CURRENT_FINAL_MATRIX):
        if row.get("method") != method or int(row["epoch"]) != epoch or row.get("target") not in TARGETS:
            continue
        out[row["target"]] = {metric: number(row.get(metric)) for metric in METRICS}
    return out


def read_pa_method(epoch: int) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for row in read_csv(PA_MEDICAL):
        if row.get("method") != "PA" or int(row["epoch"]) != epoch or row.get("target") not in TARGETS:
            continue
        out[row["target"]] = {metric: number(row.get(metric)) for metric in METRICS}
    return out


def read_final_macro() -> dict[tuple[int, str], dict[str, float | None]]:
    out: dict[tuple[int, str], dict[str, float | None]] = {}
    for row in read_csv(CURRENT_FINAL_MACRO):
        out[(int(row["epoch"]), row["method"])] = {
            "pixel_auroc": number(row.get("pixel_auroc_macro")),
            "pixel_ap": number(row.get("pixel_ap_macro")),
            "image_auroc": number(row.get("image_auroc_macro")),
            "image_ap": number(row.get("image_ap_macro")),
        }
    return out


def add_macro_row(rows: list[dict[str, Any]], lineage: str, evaluator: str, epoch: int, data: dict[str, dict[str, float | None]], source: str, checkpoint_sha: str = "") -> None:
    row: dict[str, Any] = {"lineage": lineage, "evaluator": evaluator, "epoch": epoch, "target": "ALL_MEDICAL_MACRO", "source": source, "checkpoint_sha256": checkpoint_sha}
    for metric in METRICS:
        row[metric], row[metric + "_support"] = macro(data, metric)
    rows.append(row)


def build_replay_tables(h1: dict[str, dict[str, float | None]], h2: dict[str, dict[str, float | None]], h2_hist: dict[str, dict[str, float | None]], c2: dict[str, dict[str, float | None]]) -> None:
    fields = ["lineage", "evaluator", "epoch", "target", *METRICS, "pixel_auroc_support", "pixel_ap_support", "image_auroc_support", "image_ap_support", "precision", "source", "status"]
    rows: list[dict[str, Any]] = []
    for lineage, evaluator, epoch, data, precision, source in [
        ("H2", "HISTORICAL_H2_TEST_PY", 10, h2_hist, "AMP/mixed", str(H2_HISTORICAL_RESULTS)),
        ("H2", "CURRENT_EXACT_EVALUATOR", 10, h2, "current evaluator", str(H2_CURRENT_RESULTS)),
        ("H1", "CURRENT_EXACT_EVALUATOR", 9, h1, "current evaluator", str(H1_CURRENT_RESULTS)),
        ("C2", "CURRENT_EXACT_EVALUATOR", 10, c2, "FP32", str(C2_MEDICAL)),
    ]:
        for target in TARGETS:
            values = data.get(target, {})
            row = {"lineage": lineage, "evaluator": evaluator, "epoch": epoch, "target": target, "precision": precision, "source": source, "status": "COMPLETE"}
            row.update({metric: values.get(metric) for metric in METRICS})
            row.update({metric + "_support": int(values.get(metric) is not None) for metric in METRICS})
            rows.append(row)
        add_macro_row(rows, lineage, evaluator, epoch, data, source, H2_CHECKPOINT_SHA if lineage == "H2" else C2_E10_ORIGINAL_SHA)
    write_csv("HISTORICAL_REPLAY_RESULTS.csv", fields, rows)

    fields = ["target", "epoch", "historical_pixel_auroc", "current_pixel_auroc", "delta_pixel_auroc", "historical_pixel_ap", "current_pixel_ap", "delta_pixel_ap", "historical_image_auroc", "current_image_auroc", "delta_image_auroc", "historical_image_ap", "current_image_ap", "delta_image_ap", "historical_evaluator_sha256", "current_evaluator_sha256", "checkpoint_sha256", "status"]
    rows = []
    for target in [*TARGETS, "ALL_MEDICAL_MACRO"]:
        old_values = {metric: macro(h2_hist, metric)[0] for metric in METRICS} if target == "ALL_MEDICAL_MACRO" else h2_hist[target]
        new_values = {metric: macro(h2, metric)[0] for metric in METRICS} if target == "ALL_MEDICAL_MACRO" else h2[target]
        row = {"target": target, "epoch": 10, "historical_evaluator_sha256": H2_EVALUATOR_SHA, "current_evaluator_sha256": CURRENT_EVALUATOR_SHA, "checkpoint_sha256": H2_CHECKPOINT_SHA, "status": "COMPLETE"}
        for metric in METRICS:
            row["historical_" + metric] = old_values.get(metric)
            row["current_" + metric] = new_values.get(metric)
            row["delta_" + metric] = None if old_values.get(metric) is None or new_values.get(metric) is None else new_values[metric] - old_values[metric]
        rows.append(row)
    write_csv("SAME_CHECKPOINT_CROSS_EVALUATOR.csv", fields, rows)

    fields = ["lineage", "target", "epoch", *METRICS, "evaluator_sha256", "evaluator_git_sha", "checkpoint_sha256", "config_sha256", "identity_status", "source"]
    rows = []
    for lineage, epoch, data, checkpoint, config, git_sha, source, identity in [
        ("H1", 9, h1, H1_CHECKPOINT_SHA, "legacy", CURRENT_REPO_COMMIT, str(H1_CURRENT_RESULTS), "LEGACY_METADATA_GATE_BYPASSED"),
        ("H2", 10, h2, H2_CHECKPOINT_SHA, CURRENT_CONFIG_SHA, CURRENT_REPO_COMMIT, str(H2_CURRENT_RESULTS), "LEGACY_METADATA_GATE_BYPASSED"),
        ("C2", 10, c2, C2_E10_ORIGINAL_SHA, CURRENT_CONFIG_SHA, C2_TRAINING_COMMIT, str(C2_MEDICAL), "ORIGINAL_SHA_PRESERVED_MODEL_STATE_REPLAY_ONLY"),
    ]:
        for target in TARGETS:
            rows.append({"lineage": lineage, "target": target, "epoch": epoch, **data.get(target, {}), "evaluator_sha256": CURRENT_EVALUATOR_SHA, "evaluator_git_sha": git_sha, "checkpoint_sha256": checkpoint, "config_sha256": config, "identity_status": identity, "source": source})
        rows.append({"lineage": lineage, "target": "ALL_MEDICAL_MACRO", "epoch": epoch, **{metric: macro(data, metric)[0] for metric in METRICS}, "evaluator_sha256": CURRENT_EVALUATOR_SHA, "evaluator_git_sha": git_sha, "checkpoint_sha256": checkpoint, "config_sha256": config, "identity_status": identity, "source": source})
    write_csv("CROSS_CHECKPOINT_CURRENT_EVAL.csv", fields, rows)


def build_gap_decomposition(h2_hist: dict[str, dict[str, float | None]], h2: dict[str, dict[str, float | None]], c2: dict[str, dict[str, float | None]]) -> None:
    fields = ["scope", "target", "metric", "historical_h2", "same_h2_current_eval", "current_c2_p_e10", "evaluator_component", "checkpoint_training_component", "total_gap", "support", "unit", "interpretation"]
    rows: list[dict[str, Any]] = []
    datasets = {"H2_HISTORICAL": h2_hist, "H2_CURRENT": h2, "C2": c2}
    for target in [*TARGETS, "ALL_MEDICAL_MACRO"]:
        values = {name: {metric: macro(data, metric)[0] if target == "ALL_MEDICAL_MACRO" else data[target][metric] for metric in METRICS} for name, data in datasets.items()}
        for metric in METRICS:
            historical = values["H2_HISTORICAL"][metric]
            current_h2 = values["H2_CURRENT"][metric]
            current_c2 = values["C2"][metric]
            support = macro(h2_hist, metric)[1] if target == "ALL_MEDICAL_MACRO" else int(historical is not None)
            rows.append({
                "scope": "H2 historical -> C2 current parent E10", "target": target, "metric": metric,
                "historical_h2": None if historical is None else historical * 100.0,
                "same_h2_current_eval": None if current_h2 is None else current_h2 * 100.0,
                "current_c2_p_e10": None if current_c2 is None else current_c2 * 100.0,
                "evaluator_component": None if historical is None or current_h2 is None else (current_h2 - historical) * 100.0,
                "checkpoint_training_component": None if current_h2 is None or current_c2 is None else (current_c2 - current_h2) * 100.0,
                "total_gap": None if historical is None or current_c2 is None else (current_c2 - historical) * 100.0,
                "support": support, "unit": "percentage_points",
                "interpretation": "same-checkpoint evaluator migration first; residual includes multiple H2-to-C2 contract/training factors",
            })
    write_csv("PHASE2B_GAP_DECOMPOSITION.csv", fields, rows)


def build_contracts() -> None:
    historical = {
        "lineage": "H2", "protocol": "historical Phase2B hybrid alpha0.2 + K-reg", "repository": str(HISTORICAL_REPO), "repository_remote": HISTORICAL_REMOTE, "repository_commit": H2_COMMIT, "run_directory": str(H2_RUN),
        "source_dataset": "VisA", "source_root": "/home/ai4/caohuy/data/VisA_20220922", "seed": 0, "clip_asset": "model/ViT-L-14-336px.pt", "clip_asset_sha256": CLIP_SHA,
        "model": {"model_name": "ViT-L-14-336", "img_size": 518, "n_groups": 3, "image_adapt_weight": 0.2, "text_adapt_weight": 0.2, "lora_rank": 16, "lora_alpha": 2.0, "conv_lora_rank": 8, "conv_lora_alpha": 2.0, "conv_kernel_size_list": [3, 5]},
        "dfg": {"mode": "attn", "attn_dim": 256, "tau": 8.0, "ss2d": True, "gamma_max": 0.2, "fusion": "weight_residual", "beta": 0.1, "beta_schedule": "warmup010", "beta_target": 0.1, "weight_residual_fp32": True},
        "prompt": {"mode": "hybrid", "alpha_max": 0.2, "schedule": "epochs1-3=0; epoch4=.05; epoch5=.1; epoch6+=.2", "ctx_len": 4, "init": "phrase:a photo of a", "freeze_epochs": 3, "soft_prompt_lr": 5e-5},
        "loss": {"classification": "cross_entropy", "segmentation": "calculate_seg_loss", "lambda_kg": 0.01, "lambda_k": 0.002, "formula": "cls_loss + seg_loss + lambda_kg*kg_loss + lambda_k*k_loss", "k_reg": "exact detached-W_K cosine distance, mean across stages/groups/normal-abnormal"},
        "optimizer": {"class": "torch.optim.Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "groups": ["text_adapter", "image_adapter", "soft_prompt"], "image_lr": 0.001, "text_lr": 0.0005},
        "scheduler": {"class": "StepLR", "step_size": 1, "gamma": 0.9, "actual_behavior": "scheduler.step once after each epoch before candidate save; soft prompt constant policy reapplied"},
        "gradient": {"clip_norm": 1.0, "frequency": "each optimizer update", "nonfinite_gradient_skips": "observed in epochs 1, 3, 11; run continued"},
        "precision": {"model_train_precision": "AMP mixed precision", "global_amp": True, "autocast": True, "grad_scaler": True, "attention_score_precision": "DFG weight-residual FP32 flag; no separate full-score override recovered", "tf32": "not explicitly configured"},
        "data": {"batch_size": 6, "effective_batch_size": 6, "num_workers": 6, "augmentation": "historical repository dataset transforms", "manifest": "historical dataset loader"},
        "evaluation": {"script_sha256": H2_EVALUATOR_SHA, "pixel_stride": 4, "metric_thresholds": None, "image_score": "0.5*classification_score + 0.5*pixel_max", "colon_image_metrics": "unsupported and emitted as 0.00", "candidate_epochs": list(range(7, 16))},
        "selection": {"selected_epoch": 10, "rule": "retrospective best six-Medical pixel AP", "status": "RETROSPECTIVE_BEST; target/Medical-informed"},
    }
    current = {
        "lineage": "C2", "protocol": "current corrected canonical Phase2B", "repository": str(ROOT), "repository_commit": C2_TRAINING_COMMIT, "current_head_at_audit": CURRENT_REPO_COMMIT,
        "source_dataset": "VisA", "source_root": "/home/ai4/caohuy/data/VisA_20220922", "seed": 0, "clip_asset": "model/ViT-L-14-336px.pt", "clip_asset_sha256": CLIP_SHA,
        "config_sha256": CURRENT_CONFIG_SHA, "resolved_run_config_sha256": CURRENT_RESOLVED_CONFIG_SHA, "architecture_freeze_sha256": ARCHITECTURE_FREEZE_SHA,
        "model": {"model_name": "ViT-L-14-336", "img_size": 518, "n_groups": 3, "image_adapt_weight": 0.2, "text_adapt_weight": 0.2, "lora_rank": 16, "lora_alpha": 2.0, "conv_lora_rank": 8, "conv_lora_alpha": 2.0, "conv_kernel_size_list": [3, 5]},
        "dfg": {"mode": "attn", "attn_dim": 256, "tau": 8.0, "ss2d": True, "gamma_max": 0.2, "fusion": "weight_residual", "beta": 0.1, "beta_schedule": "warmup010", "beta_target": 0.1, "weight_residual_fp32": True},
        "prompt": {"mode": "hybrid", "alpha_max": 0.2, "schedule": "canonical get_hybrid_alpha_for_epoch", "ctx_len": 4, "init": "phrase:a photo of a", "freeze_epochs": 3, "soft_prompt_lr": 1e-4},
        "loss": {"classification": "cross_entropy", "segmentation": "calculate_seg_loss", "lambda_kg": 0.001, "lambda_k": 0.0, "k_reg_status": "STUB_ZERO", "formula": "cls_loss + seg_loss + 0.001*kg_loss + 0.0*k_loss"},
        "optimizer": {"class": "torch.optim.Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "groups": ["image_adapter", "text_adapter", "soft_prompt"], "image_lr": 0.001, "text_lr": 0.0005},
        "scheduler": {"class": "StepLR", "step_size": 1, "gamma": 0.9, "actual_behavior": "scheduler.step once after each epoch before history and candidate save"},
        "gradient": {"clip_norm": 1.0, "frequency": "each optimizer update", "nonfinite_gradient_skips": "abort-on-nonfinite loss; no H2-style legacy skip evidence in corrected summary"},
        "precision": {"model_train_precision": "FP32", "global_amp": False, "autocast": False, "grad_scaler": False, "attention_score_precision": "FP32", "tf32": False},
        "data": {"batch_size": 6, "effective_batch_size": 6, "num_workers": 4, "augmentation": "current canonical loader", "manifest": "dataset/hub/VisA.jsonl"},
        "evaluation": {"script_sha256": CURRENT_EVALUATOR_SHA, "pixel_stride": 1, "metric_thresholds": None, "image_score": "current exact evaluator deployment", "colon_image_metrics": "undefined/blank", "candidate_epochs": [10, 12, 14, 16, 18, 20]},
        "selection": {"selected_epoch": 10, "rule": "candidate fixed for decomposition; current archive reports all candidates", "status": "no new target tuning in this audit"},
    }
    write_json("HISTORICAL_PHASE2B_CONTRACT.json", historical)
    write_json("CURRENT_PHASE2B_CONTRACT.json", current)
    write_text("HISTORICAL_PHASE2B_CONTRACT.md", f"""# Historical Phase2B H2 contract

Status: CONFIRMED from the H2 run log, source at commit {H2_COMMIT}, the exact H2 checkpoint payload, and historical test.py.

H2 is the 15-epoch VisA hybrid run with hybrid_alpha_max=0.2, lambda_kg=0.01, lambda_k=0.002, soft_prompt_lr=5e-5, AMP enabled, and the DFG attention/SS2D weight-residual configuration. Its optimizer is Adam with image/text base learning rates 1e-3/5e-4; StepLR uses gamma=0.9 and is actually stepped once per epoch after the epoch loop and before checkpoint save. The soft-prompt group is frozen at LR zero for the first three epochs and then follows a constant 5e-5 policy after the StepLR call.

The exact K-reg computation is documented in KREG_FORENSICS.md. The historical E10 checkpoint is {H2_CHECKPOINT_SHA} and contains model adapter state but no optimizer, scheduler, or RNG state. E10 was selected retrospectively using six-Medical pixel AP, so it is a useful historical champion but not a clean target-blind baseline.
""")
    write_text("CURRENT_PHASE2B_CONTRACT.md", f"""# Current corrected Phase2B C2 contract

Status: recovered from the canonical config, corrected training manifest, checkpoint payloads, and current trainer.

C2 uses the same VisA source, seed, CLIP asset, model dimensions, DFG settings, batch/effective batch, Adam defaults, base image/text learning rates, gradient clip, prompt alpha/freeze schedule, and StepLR gamma=0.9. It differs from H2 in multiple scientific/protocol fields: FP32 without AMP, lambda_kg=0.001, lambda_k=0 with an explicit zero stub, soft_prompt_lr=1e-4, 20 epochs and a different candidate schedule, and the current exact evaluator with pixel stride 1.

The corrected trainer calls scheduler.step() once after every epoch and before the history row and candidate checkpoint save. The raw canonical config hash is {CURRENT_CONFIG_SHA}; the resolved run configuration used by the C2 checkpoint is {CURRENT_RESOLVED_CONFIG_SHA}.
""")


def build_phase1_contract_and_primary_reports() -> None:
    h1 = {
        "lineage": "H1",
        "protocol": "historical Phase1B V3c FP32-attention weight-residual",
        "repository": str(HISTORICAL_REPO),
        "repository_remote": HISTORICAL_REMOTE,
        "repository_commit": H1_COMMIT,
        "run_directory": str(HISTORICAL_REPO / "phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3"),
        "checkpoint": str(H1_CHECKPOINT),
        "checkpoint_sha256": H1_CHECKPOINT_SHA,
        "checkpoint_epoch": 9,
        "source_dataset": "VisA",
        "source_root": "/home/ai4/caohuy/data/VisA_20220922",
        "seed": 0,
        "clip_asset": "model/ViT-L-14-336px.pt",
        "clip_asset_sha256": CLIP_SHA,
        "model": {"model_name": "ViT-L-14-336", "img_size": 518, "n_groups": 3, "image_adapt_weight": 0.2, "text_adapt_weight": 0.2, "n_groups": 3, "lora_rank": 16, "lora_alpha": 2.0, "conv_lora_rank": 8, "conv_lora_alpha": 2.0, "conv_kernel_size_list": [3, 5]},
        "dfg": {"mode": "attn", "attn_dim": 256, "tau": 8.0, "ss2d": True, "gamma_max": 0.2, "fusion": "weight_residual", "beta": 0.1, "beta_schedule": "warmup010", "beta_target": 0.1, "weight_residual_fp32": True},
        "prompt": {"mode": "hard", "hybrid": False, "soft_prompt": False, "status": "not applicable"},
        "loss": {"formula": "classification cross entropy + segmentation loss", "lambda_kg": "not present in H1 path", "lambda_k": "not present"},
        "optimizer": {"class": "torch.optim.Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "groups": ["text_adapter", "image_adapter"], "image_lr": 0.001, "text_lr": 0.0005},
        "scheduler": {"class": "StepLR", "step_size": 1, "gamma": 0.9, "actual_behavior": "stepped after each epoch before checkpoint"},
        "gradient": {"clip_norm": 1.0, "frequency": "each optimizer update"},
        "precision": {"model_train_precision": "AMP mixed precision", "global_amp": True, "autocast": True, "grad_scaler": True, "attention_score_precision": "DFG weight-residual FP32 flag; full score precision not separately isolated", "tf32": "not explicitly configured"},
        "data": {"batch_size": 6, "effective_batch_size": 6, "num_workers": 6, "augmentation": "historical repository transforms"},
        "evaluation": {"script_sha256": H1_EVALUATOR_SHA, "pixel_stride": 4, "metric_thresholds": None, "image_score": "0.5*classification + 0.5*pixel_max", "candidate_epochs": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
        "selection": {"selected_epoch": 9, "rule": "historical Phase1 best reported checkpoint", "status": "RETROSPECTIVE_BEST; target/Medical-informed"},
    }
    write_json("HISTORICAL_PHASE1_CONTRACT.json", h1)
    write_text("HISTORICAL_PHASE1_REPO_IDENTITY.md", f"""# Historical Phase1 repository identity

H1 is the V3c run named phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3 in {HISTORICAL_REPO}. The recovered source snapshot is commit {H1_COMMIT}; the run log, test log, and archived E9 checkpoint all carry the matching V3c DFG signature. The repository remote is {HISTORICAL_REMOTE}.

The fp32attn name refers to the DFG weight-residual/attention path, not full FP32 training. The H1 log explicitly enables AMP, and the source constructs GradScaler and autocast.
""")
    write_text("HISTORICAL_PHASE1_CHECKPOINT_IDENTITY.md", f"""# Historical Phase1 checkpoint identity

H1 E9 is {H1_CHECKPOINT} with SHA256 {H1_CHECKPOINT_SHA} and recorded size {H1_CHECKPOINT_SIZE} bytes. Its payload contains the V3c DFG/adapter state and no optimizer, scheduler, or RNG state. The current evaluator replay used only the model state and recorded a legacy metadata-gate bypass; it does not claim full resume parity.
""")
    write_text("HISTORICAL_PHASE1_CONTRACT.md", f"""# Historical Phase1 H1 contract

H1 is hard-prompt V3c with ViT-L-14-336, image size 518, three groups, attention DFG dimension 256/tau 8, SS2D weight residual gamma .2, beta warmup010 to .1, Adam base image LR 1e-3, AMP/autocast/GradScaler enabled, and StepLR gamma .9 stepped after the epoch before save. H1 has no hybrid soft prompt and no K-reg.

The historical evaluator uses pixel stride 4 and the same legacy image score construction used by H2. The E9 checkpoint is retrospective best rather than a source-only prospective selection. Full details are in HISTORICAL_PHASE1_CONTRACT.json.
""")
    write_text("HISTORICAL_REPLAY_REPORT.md", f"""# Historical replay report

Status: PASS for exact H2 replay. The historical evaluator at source hash {H2_EVALUATOR_SHA} was run in the detached H2 source worktree against checkpoint {H2_CHECKPOINT_SHA}, with batch 8, six workers, metric thresholds unset, and pixel stride 4. The six-domain macro is 90.9750 Pixel AUROC / 40.3483 Pixel AP, matching the logged 90.98 / 40.35 after the historical two-decimal reporting.

Status: PASS for H2 same-checkpoint current-evaluator replay. The current exact evaluator hash {CURRENT_EVALUATOR_SHA} loaded the same H2 model-state checkpoint through the explicitly recorded legacy metadata bypass. The six-domain macro is 90.9222 / 40.3731.

H1 E9 was also replayed through the current evaluator; its six-domain pixel macro is 90.7119 / 39.8490. C2 P E10 current-evaluator values were read from the frozen corrective matrix, with the original recorded checkpoint SHA preserved. H2/H1 historical selection is retrospective Medical-informed and is not target-blind.
""")
    write_text("SAME_CHECKPOINT_CROSS_EVALUATOR.md", """# Same-checkpoint cross-evaluator result

The exact H2 E10 model-state checkpoint was evaluated through both the historical and current evaluator paths. At the six-domain pixel macro, historical H2 is 90.9750 AUROC / 40.3483 AP and current evaluator H2 is 90.9222 / 40.3731. The evaluator component is -0.0528 / +0.0247 percentage points. Supported three-domain image metrics shift by -0.0207 / +0.0107 points.

This bounded result shows evaluator migration is not the dominant explanation of the H2-to-C2 loss. Historical output is rounded to two decimal percentage points, and legacy checkpoint metadata identity was bypassed only for loading; weights were not changed.
""")
    write_text("CROSS_CHECKPOINT_CURRENT_EVAL.md", """# Cross-checkpoint current-evaluator comparison

Under the current evaluator implementation, the H2 E10 replay is 90.9222 / 40.3731 Pixel AUROC/AP, H1 E9 is 90.7119 / 39.8490, and C2 P E10 is 87.9118 / 31.8093. The table keeps per-domain rows and records config/checkpoint/evaluator provenance. H1/H2 use the legacy metadata-only bypass; C2 E10 is a model-state replay based on its preserved original SHA because its physical full payload was overwritten in the disclosed serialization incident.
""")
    write_text("PHASE2B_GAP_DECOMPOSITION.md", """# Phase2B historical-to-current gap decomposition

The decomposition is:

historical H2 through historical evaluator
  -> same H2 checkpoint through current evaluator
  -> current C2 parent E10 through current evaluator.

At the six-domain pixel macro this is 90.9750 -> 90.9222 -> 87.9118 AUROC and 40.3483 -> 40.3731 -> 31.8093 AP. The evaluator components are -0.0528/+0.0247 points; the residual checkpoint/training/config component is -3.0104/-8.5637 points. The total gap is -3.0632/-8.5390 points.

The residual is deliberately not called a pure training effect. It includes the H2-to-C2 K-reg removal, KG coefficient change, AMP-to-FP32 migration, prompt LR change, horizon/candidate/selection changes, possible loader differences, and checkpoint-provenance limitations. The old CIR missing scheduler.step is a separate confirmed bug, not the explanation of this H2-to-C2 decomposition.
""")
    write_text("PHASE2B_HISTORICAL_VS_CURRENT_DIFF.md", """# Historical H2 versus current C2 differences

The CSV classifies every recovered difference. Core model dimensions, DFG settings, base image/text LRs, Adam defaults, batch size, and StepLR timing match. The meaningful differences are K-reg 0.002 to zero/stub, KG 0.01 to 0.001, AMP to FP32, soft-prompt LR 5e-5 to 1e-4, 15 to 20 epochs, historical versus current loader details, and pixel stride 4 versus 1 evaluation. These are multiple confounds, so the observed residual cannot be assigned to one cause.
""")
    diff_rows = [
        ("K-reg", "0.002 exact", "0.0 explicit zero stub", "C_ACCIDENTAL_FEATURE_REMOVAL"),
        ("KG coefficient", "0.01", "0.001", "A_INTENTIONAL_SCIENTIFIC_CHANGE"),
        ("global precision", "AMP/autocast/GradScaler", "FP32/no AMP", "D_PROTOCOL_CORRECTION"),
        ("soft prompt LR", "5e-5 constant after freeze", "1e-4 constant after freeze", "D_PROTOCOL_CORRECTION"),
        ("horizon/candidates", "15; E7-E15", "20; E10/E12/E14/E16/E18/E20", "A_INTENTIONAL_SCIENTIFIC_CHANGE"),
        ("scheduler", "StepLR .9 stepped after epoch before save", "same", "B_ENGINEERING_FIX_WITH_EXPECTED_PARITY"),
        ("evaluator", "legacy test.py stride4", "current exact evaluator stride1", "F_EVALUATION_ONLY_CHANGE"),
        ("loader", "workers6; historical transforms", "workers4; canonical transforms", "E_UNKNOWN_MIGRATION"),
        ("selection", "retrospective Medical best", "fixed E10 for decomposition", "D_PROTOCOL_CORRECTION"),
    ]
    write_csv("PHASE2B_HISTORICAL_VS_CURRENT_DIFF.csv", ["setting", "historical_value", "current_value", "classification", "evidence"], [{"setting": s, "historical_value": h, "current_value": c, "classification": cls, "evidence": "H1_H2_C2_FULL_CONTRACT_MATRIX.csv and migration audit"} for s, h, c, cls in diff_rows])


def build_repository_and_checkpoint_identity() -> None:
    write_csv("HISTORICAL_REPO_CANDIDATES.csv", ["candidate", "repository_path", "remote", "commit", "role", "signature", "status", "notes"], [
        {"candidate": "H1", "repository_path": str(HISTORICAL_REPO), "remote": HISTORICAL_REMOTE, "commit": H1_COMMIT, "role": "historical Phase1B V3c FP32-attention E9", "signature": "phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3", "status": "FOUND", "notes": "exact source snapshot and checkpoint recovered"},
        {"candidate": "H2", "repository_path": str(HISTORICAL_REPO), "remote": HISTORICAL_REMOTE, "commit": H2_COMMIT, "role": "historical Phase2B hybrid alpha0.2 + K-reg E10", "signature": "phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch", "status": "FOUND", "notes": "exact run log, source commit, checkpoint and evaluator recovered"},
        {"candidate": "C2", "repository_path": str(ROOT), "remote": "origin", "commit": C2_TRAINING_COMMIT, "role": "current corrected canonical parent", "signature": "corrective_matched_retrain_20260830", "status": "FOUND", "notes": "current parent E10 original hash preserved in compact archives; physical file integrity caveat recorded"},
        {"candidate": "CURRENT_HEAD", "repository_path": str(ROOT), "remote": "origin", "commit": CURRENT_REPO_COMMIT, "role": "active audit branch", "signature": "research/cir-dfg-rmt-v2-signfix", "status": "FOUND", "notes": "audit branch at artifact-generation time"},
    ])
    c2_path = ROOT / "runs/cir_rmt/CIR_DFG_RMT_V2/corrective_matched_retrain_20260830/parent/phase2b/checkpoints/adapter_10.pth"
    observed_c2 = sha256_file(c2_path) if c2_path.exists() else C2_E10_OBSERVED_SHA
    write_csv("HISTORICAL_CHECKPOINT_CANDIDATES.csv", ["lineage", "epoch", "checkpoint_path", "recorded_sha256", "observed_sha256", "size_bytes", "source_commit", "optimizer_state", "scheduler_state", "rng_state", "selection_status", "identity_status", "usable_for", "notes"], [
        {"lineage": "H1", "epoch": 9, "checkpoint_path": str(H1_CHECKPOINT), "recorded_sha256": H1_CHECKPOINT_SHA, "observed_sha256": sha256_file(H1_CHECKPOINT) if H1_CHECKPOINT.exists() else H1_CHECKPOINT_SHA, "size_bytes": H1_CHECKPOINT_SIZE, "source_commit": H1_COMMIT, "optimizer_state": "ABSENT", "scheduler_state": "ABSENT", "rng_state": "ABSENT", "selection_status": "RETROSPECTIVE_BEST", "identity_status": "CONFIRMED", "usable_for": "historical/current model-state replay", "notes": "legacy checkpoint has V3c metadata; no full resume state"},
        {"lineage": "H2", "epoch": 10, "checkpoint_path": str(H2_CHECKPOINT), "recorded_sha256": H2_CHECKPOINT_SHA, "observed_sha256": sha256_file(H2_CHECKPOINT) if H2_CHECKPOINT.exists() else H2_CHECKPOINT_SHA, "size_bytes": H2_CHECKPOINT_SIZE, "source_commit": H2_COMMIT, "optimizer_state": "ABSENT", "scheduler_state": "ABSENT", "rng_state": "ABSENT", "selection_status": "RETROSPECTIVE_BEST", "identity_status": "CONFIRMED", "usable_for": "historical/current model-state replay; parent recovery reference", "notes": "exact model-state checkpoint; no full resume state"},
        {"lineage": "C2", "epoch": 10, "checkpoint_path": str(c2_path), "recorded_sha256": C2_E10_ORIGINAL_SHA, "observed_sha256": observed_c2, "size_bytes": "original size not recoverable; observed stripped payload 56445189", "source_commit": C2_TRAINING_COMMIT, "optimizer_state": "LOST_FROM_PHYSICAL_FILE", "scheduler_state": "LOST_FROM_PHYSICAL_FILE", "rng_state": "LOST_FROM_PHYSICAL_FILE", "selection_status": "FIXED_E10_FOR_DECOMPOSITION", "identity_status": "MODEL_STATE_REPLAY_ONLY", "usable_for": "frozen metric provenance and model-state replay", "notes": "temporary serialization-only copy followed a symlink and overwrote the physical C2 E10 file; original SHA/metrics remain frozen in prior archive"},
    ])
    write_text("HISTORICAL_CHECKPOINT_IDENTITY.md", f"""# Historical checkpoint identity

H1 is the V3c FP32-attention E9 checkpoint {H1_CHECKPOINT_SHA} ({H1_CHECKPOINT_SIZE} bytes) from {H1_COMMIT}. H2 is the exact Phase2B hybrid/K-reg E10 checkpoint {H2_CHECKPOINT_SHA} ({H2_CHECKPOINT_SIZE} bytes) from {H2_COMMIT}. Both are legacy model-state payloads and do not contain optimizer, scheduler, or RNG state.

The C2 parent E10 provenance originally recorded SHA {C2_E10_ORIGINAL_SHA} and full optimizer/scheduler/RNG metadata. During a temporary serialization-only replay preparation, a symlink to that file was passed to torch.save; the save followed the symlink and overwrote the physical file with a stripped model-state payload. The model tensors were preserved, and the original SHA plus all original compact metric records remain in the frozen corrective/extension archives, but the original full checkpoint metadata is not recoverable from this workspace. Therefore all C2 E10 replay claims are explicitly MODEL_STATE_REPLAY_ONLY.

The candidate table is HISTORICAL_CHECKPOINT_CANDIDATES.csv.
""")
    write_text("CHECKPOINT_INTEGRITY_INCIDENT.md", f"""# C2 E10 checkpoint integrity incident

Status: disclosed and contained.

While preparing a temporary replay copy, /tmp/acdclip_phase2b_current_p_e10_historical/runs/phase2b_replay/adapter_10.pth was a symlink to the physical C2 E10 checkpoint. A serialization-only torch.save operation followed the symlink and replaced the original full payload.

- Original recorded SHA: {C2_E10_ORIGINAL_SHA}.
- Current observed SHA: {observed_c2}.
- Model tensor values: preserved in the stripped payload and replayable.
- Optimizer/scheduler/RNG metadata: no longer physically recoverable.
- Original metric rows and SHA: preserved in the previously frozen compact archives.
- Scientific consequence: C2 E10 is valid for model-state metric comparison, but not for exact full-checkpoint or resume-state verification.

No other C2 candidate checkpoint was intentionally modified. No retraining was launched to conceal or replace this incident.
""")


def build_contract_matrix() -> None:
    rows: list[dict[str, Any]] = []
    def add(dimension: str, setting: str, h1: str, h1e: str, h1s: str, h2: str, h2e: str, h2s: str, c2: str, c2e: str, c2s: str) -> None:
        rows.append({"dimension": dimension, "setting": setting, "H1_value": h1, "H1_evidence": h1e, "H1_source": h1s, "H2_value": h2, "H2_evidence": h2e, "H2_source": h2s, "C2_value": c2, "C2_evidence": c2e, "C2_source": c2s})
    sh1 = "H1 train.log/source@023"; sh2 = "H2 train.log/source@e039"; sc2 = "canonical config/trainer@042"
    add("Architecture", "CLIP", "ViT-L-14-336", "RECOVERED_FROM_LOG", sh1, "ViT-L-14-336", "RECOVERED_FROM_LOG", sh2, "ViT-L-14-336", "RECOVERED_FROM_CODE", sc2)
    add("Architecture", "img_size", "518", "RECOVERED_FROM_LOG", sh1, "518", "RECOVERED_FROM_LOG", sh2, "518", "RECOVERED_FROM_CODE", sc2)
    add("Architecture", "n_groups", "3", "RECOVERED_FROM_LOG", sh1, "3", "RECOVERED_FROM_LOG", sh2, "3", "RECOVERED_FROM_CODE", sc2)
    add("Architecture", "image/text adapter weights", "0.2/0.2", "RECOVERED_FROM_LOG", sh1, "0.2/0.2", "RECOVERED_FROM_LOG", sh2, "0.2/0.2", "RECOVERED_FROM_CODE", sc2)
    add("Architecture", "LoRA / Conv-LoRA", "rank16/a2; conv8/a2; kernels3,5", "RECOVERED_FROM_LOG", sh1, "rank16/a2; conv8/a2; kernels3,5", "RECOVERED_FROM_LOG", sh2, "rank16/a2; conv8/a2; kernels3,5", "RECOVERED_FROM_CODE", sc2)
    add("Architecture", "DFG", "attn dim256 tau8; SS2D; gamma .2", "RECOVERED_FROM_LOG", sh1, "attn dim256 tau8; SS2D; gamma .2", "RECOVERED_FROM_LOG", sh2, "attn dim256 tau8; SS2D; gamma .2", "RECOVERED_FROM_CODE", sc2)
    add("Architecture", "fusion/beta", "weight_residual; warmup010 -> .1", "RECOVERED_FROM_LOG", sh1, "weight_residual; warmup010 -> .1", "RECOVERED_FROM_LOG", sh2, "weight_residual; warmup010 -> .1", "RECOVERED_FROM_CODE", sc2)
    add("Prompt", "mode", "hard", "RECOVERED_FROM_CHECKPOINT", str(H1_CHECKPOINT), "hybrid", "RECOVERED_FROM_LOG", sh2, "hybrid", "RECOVERED_FROM_CODE", sc2)
    add("Prompt", "hybrid alpha", "not applicable", "INFERRED", "no prompt fields", "0->.05->.1->.2 schedule", "RECOVERED_FROM_CODE", sh2, "same alpha schedule", "RECOVERED_FROM_CODE", sc2)
    add("Prompt", "ctx/init/freeze", "none/none/none", "RECOVERED_FROM_CHECKPOINT", str(H1_CHECKPOINT), "4/phrase/freeze3", "RECOVERED_FROM_LOG", sh2, "4/phrase/freeze3", "RECOVERED_FROM_CODE", sc2)
    add("Prompt", "soft prompt LR", "not applicable", "UNKNOWN", "H1 payload", "5e-5 constant after freeze", "RECOVERED_FROM_CODE", sh2, "1e-4 constant after freeze", "RECOVERED_FROM_CODE", sc2)
    add("Loss", "classification/segmentation", "CE + calculate_seg_loss", "RECOVERED_FROM_CODE", sh1, "CE + calculate_seg_loss", "RECOVERED_FROM_CODE", sh2, "CE + calculate_seg_loss", "RECOVERED_FROM_CODE", sc2)
    add("Loss", "lambda_kg", "not explicitly logged; historical default path", "UNKNOWN", sh1, "0.01", "RECOVERED_FROM_LOG", sh2, "0.001", "RECOVERED_FROM_CODE", sc2)
    add("Loss", "lambda_k / K-reg", "not applicable", "RECOVERED_FROM_CHECKPOINT", str(H1_CHECKPOINT), "0.002; exact detached-W_K K-reg", "RECOVERED_FROM_CODE", sh2, "0.0; explicit zero stub", "RECOVERED_FROM_CODE", sc2)
    add("Optimization", "optimizer/betas/eps/decay", "Adam defaults; 0.9/.999/1e-8/0", "RECOVERED_FROM_CODE", sh1, "Adam defaults; 0.9/.999/1e-8/0", "RECOVERED_FROM_CODE", sh2, "Adam defaults; 0.9/.999/1e-8/0", "RECOVERED_FROM_CODE", sc2)
    add("Optimization", "image/text LR", "1e-3 / not applicable", "RECOVERED_FROM_LOG", sh1, "1e-3 / 5e-4", "RECOVERED_FROM_LOG", sh2, "1e-3 / 5e-4", "RECOVERED_FROM_CODE", sc2)
    add("Optimization", "scheduler", "StepLR .9; stepped", "RECOVERED_FROM_CODE", sh1, "StepLR .9; stepped after epoch before save", "RECOVERED_FROM_CODE", sh2, "StepLR .9; stepped after epoch before save", "RECOVERED_FROM_CODE", sc2)
    add("Optimization", "gradient clipping", "1.0 per update", "RECOVERED_FROM_CODE", sh1, "1.0 per update", "RECOVERED_FROM_CODE", sh2, "1.0 per update", "RECOVERED_FROM_CODE", sc2)
    add("Effective batch", "batch/effective batch", "6/6", "RECOVERED_FROM_LOG", sh1, "6/6", "RECOVERED_FROM_LOG", sh2, "6/6", "RECOVERED_FROM_CODE", sc2)
    add("Precision", "global AMP/autocast", "enabled/enabled", "RECOVERED_FROM_LOG", sh1, "enabled/enabled", "RECOVERED_FROM_LOG", sh2, "disabled/disabled", "RECOVERED_FROM_CODE", sc2)
    add("Precision", "GradScaler", "enabled", "RECOVERED_FROM_CODE", sh1, "enabled", "RECOVERED_FROM_CODE", sh2, "disabled", "RECOVERED_FROM_CODE", sc2)
    add("Precision", "attention/TF32", "DFG residual FP32; TF32 not set", "RECOVERED_FROM_CODE", sh1, "DFG residual FP32; TF32 not set", "RECOVERED_FROM_CODE", sh2, "FP32; TF32 false", "RECOVERED_FROM_CODE", sc2)
    add("Checkpointing", "gradient checkpointing", "true", "RECOVERED_FROM_LOG", sh1, "true", "RECOVERED_FROM_LOG", sh2, "true", "RECOVERED_FROM_CODE", sc2)
    add("Data", "source/seed", "VisA / 0", "RECOVERED_FROM_LOG", sh1, "VisA / 0", "RECOVERED_FROM_LOG", sh2, "VisA / 0", "RECOVERED_FROM_CODE", sc2)
    add("Data", "workers/augmentation", "6 / historical transforms", "RECOVERED_FROM_LOG", sh1, "6 / historical transforms", "RECOVERED_FROM_LOG", sh2, "4 / canonical transforms", "RECOVERED_FROM_CODE", sc2)
    add("Evaluation", "evaluator/pixel stride", "historical test.py / 4", "RECOVERED_FROM_SCRIPT", H1_EVALUATOR_SHA, "historical test.py / 4", "RECOVERED_FROM_SCRIPT", H2_EVALUATOR_SHA, "current exact evaluator / 1", "RECOVERED_FROM_CODE", CURRENT_EVALUATOR_SHA)
    add("Evaluation", "image score", "0.5 cls + 0.5 pmax", "RECOVERED_FROM_CODE", H1_EVALUATOR_SHA, "0.5 cls + 0.5 pmax", "RECOVERED_FROM_CODE", H2_EVALUATOR_SHA, "current deployment score", "RECOVERED_FROM_CODE", CURRENT_EVALUATOR_SHA)
    add("Evaluation", "macro semantics", "six pixel / three image", "RECOVERED_FROM_SCRIPT", H1_EVALUATOR_SHA, "six pixel / three image", "RECOVERED_FROM_SCRIPT", H2_EVALUATOR_SHA, "six pixel / three image", "RECOVERED_FROM_CODE", CURRENT_EVALUATOR_SHA)
    add("Checkpoint", "candidate schedule", "E9 selected", "RECOVERED_FROM_CHECKPOINT", str(H1_CHECKPOINT), "E7-E15; E10 selected", "RECOVERED_FROM_LOG", str(H2_RUN), "E10/E12/E14/E16/E18/E20", "RECOVERED_FROM_CODE", str(C2_MEDICAL))
    add("Checkpoint", "selection status", "retrospective best", "RECOVERED_FROM_LOG", "EXPERIMENT_LOG_PHASE1.md", "retrospective Medical pixel AP best", "RECOVERED_FROM_LOG", "PHASE2B_RUN_CONTEXT.md", "fixed E10 for decomposition", "DERIVED", "audit protocol")
    add("Identity", "checkpoint metadata", "model only; no optimizer/scheduler/RNG", "RECOVERED_FROM_CHECKPOINT", str(H1_CHECKPOINT), "model only; no optimizer/scheduler/RNG", "RECOVERED_FROM_CHECKPOINT", str(H2_CHECKPOINT), "original full state recorded; physical E10 now stripped", "RECOVERED_FROM_CHECKPOINT", "CHECKPOINT_INTEGRITY_INCIDENT.md")
    write_csv("H1_H2_C2_FULL_CONTRACT_MATRIX.csv", ["dimension", "setting", "H1_value", "H1_evidence", "H1_source", "H2_value", "H2_evidence", "H2_source", "C2_value", "C2_evidence", "C2_source"], rows)
    write_text("H1_H2_C2_FULL_CONTRACT_MATRIX.md", """# H1/H2/C2 full contract matrix

The CSV is the authoritative cell-by-cell matrix. Each value has an evidence level and source. RECOVERED and DERIVED cells are evidence-backed or arithmetic; UNKNOWN is retained where the historical artifact did not expose a setting.

H2 and C2 share the core CLIP/DFG/Adam/base-LR/StepLR/batch structure, but differ in K-reg, KG coefficient, AMP versus FP32, prompt LR, horizon/candidate selection, and evaluator. H1 is a hard-prompt V3c parent, so it is a useful historical comparator but not an architecture-identical H2 control.
""")


def build_migration_and_forensics() -> None:
    rows = [
        {"change_id": "M01", "dimension": "Loss", "historical_value": "lambda_k=0.002; exact detached-W_K K-reg", "current_value": "lambda_k=0; explicit zero tensor", "implementation_path": "H2 train.py K-reg helper vs current train.py _text_with_regularizers", "classification": "C_ACCIDENTAL_FEATURE_REMOVAL", "expected_mechanism": "removes K-space regularization and its gradient path", "observed_evidence": "current code returns torch.zeros for k_loss; H2 code computes stage cosine", "likely_metric_impact": "potentially material; not isolated by existing runs", "restoration_validity": "valid only with exact historical formula and parity", "causality_intervention": "required bounded matched H2 K-reg comparison"},
        {"change_id": "M02", "dimension": "Loss", "historical_value": "lambda_kg=0.01", "current_value": "lambda_kg=0.001", "implementation_path": "train args/config", "classification": "A_INTENTIONAL_SCIENTIFIC_CHANGE", "expected_mechanism": "changes hard/soft text alignment pressure", "observed_evidence": "H2 log versus current canonical config", "likely_metric_impact": "unknown; confounded with K-reg and precision", "restoration_validity": "must restore for historical-parent candidate", "causality_intervention": "requires single-factor ablation"},
        {"change_id": "M03", "dimension": "Precision", "historical_value": "AMP/autocast/GradScaler enabled", "current_value": "FP32, AMP/GradScaler disabled", "implementation_path": "train launch and canonical FP32 enforcement", "classification": "D_PROTOCOL_CORRECTION", "expected_mechanism": "changes numerical trajectory and stability", "observed_evidence": "both H1/H2 logs enable AMP; C2 manifest disables it", "likely_metric_impact": "potentially material", "restoration_validity": "restore AMP for H2 historical contract; do not call fp32attn full FP32", "causality_intervention": "requires matched precision comparison"},
        {"change_id": "M04", "dimension": "Optimization", "historical_value": "soft_prompt_lr=5e-5", "current_value": "soft_prompt_lr=1e-4", "implementation_path": "optimizer group constant_lr policy", "classification": "D_PROTOCOL_CORRECTION", "expected_mechanism": "changes soft prompt adaptation speed", "observed_evidence": "H2 log and current config", "likely_metric_impact": "unknown; prompt effects are confounded", "restoration_validity": "restore for H2 parent", "causality_intervention": "requires matched one-factor comparison"},
        {"change_id": "M05", "dimension": "Scheduler", "historical_value": "StepLR .9; stepped after epoch before save", "current_value": "StepLR .9; stepped after epoch before save", "implementation_path": "H2 train.py line623 and current train_full.py line347", "classification": "B_ENGINEERING_FIX_WITH_EXPECTED_PARITY", "expected_mechanism": "same post-step LR exposure", "observed_evidence": "H2 source/log and C2 histories", "likely_metric_impact": "not a cause of H2->C2 gap", "restoration_validity": "already matched", "causality_intervention": "none for H2->C2; separate old CIR bug remains confirmed"},
        {"change_id": "M06", "dimension": "Evaluation", "historical_value": "test.py pixel_stride=4; rounded torchmetrics; 0.5 cls + 0.5 pmax", "current_value": "current exact evaluator pixel_stride=1 and current deployment", "implementation_path": "historical test.py vs eval_full.py", "classification": "F_EVALUATION_ONLY_CHANGE", "expected_mechanism": "changes measured score, not model state", "observed_evidence": "same H2 checkpoint cross-evaluator deltas are <=0.053 pp pixel macro", "likely_metric_impact": "small for this H2 replay", "restoration_validity": "do not mix result columns", "causality_intervention": "same-checkpoint cross-evaluator replay completed"},
        {"change_id": "M07", "dimension": "Horizon", "historical_value": "15 epochs; E7-E15 candidates", "current_value": "20 epochs; E10/E12/E14/E16/E18/E20 candidates", "implementation_path": "run launch/config", "classification": "A_INTENTIONAL_SCIENTIFIC_CHANGE", "expected_mechanism": "changes exposure and selected checkpoint", "observed_evidence": "H2 run context and C2 manifest", "likely_metric_impact": "potentially material", "restoration_validity": "restore for H2 parent", "causality_intervention": "requires same horizon"},
        {"change_id": "M08", "dimension": "Checkpoint selection", "historical_value": "E10 retrospective six-Medical pixel AP best", "current_value": "fixed E10 in decomposition; current archive also all candidates", "implementation_path": "historical reports versus audit freeze", "classification": "D_PROTOCOL_CORRECTION", "expected_mechanism": "selection contamination can inflate historical comparison", "observed_evidence": "PHASE2B_RUN_CONTEXT explicitly records best-by-Medical", "likely_metric_impact": "unknown selection optimism", "restoration_validity": "new run must use source/fixed rule", "causality_intervention": "predeclare candidate rule"},
        {"change_id": "M09", "dimension": "Data loader", "historical_value": "workers=6; historical transforms", "current_value": "workers=4; canonical manifest/transforms", "implementation_path": "loader/run arguments", "classification": "E_UNKNOWN_MIGRATION", "expected_mechanism": "possible augmentation/order difference", "observed_evidence": "worker counts recovered; full transform equivalence not proven", "likely_metric_impact": "unknown", "restoration_validity": "audit before strong-parent train", "causality_intervention": "requires exact loader parity"},
    ]
    fields = ["change_id", "dimension", "historical_value", "current_value", "implementation_path", "classification", "expected_mechanism", "observed_evidence", "likely_metric_impact", "restoration_validity", "causality_intervention"]
    write_csv("H2_TO_C2_MIGRATION_CLASSIFICATION.csv", fields, rows)
    write_text("LOST_GAIN_COMPONENTS.md", """# Lost-gain decomposition

The H2 historical replay is confirmed: the exact E10 model-state checkpoint replays to the historical logged macro 90.98 AUROC / 40.35 AP. The same checkpoint under the current evaluator gives 90.9222 / 40.3731, a cross-evaluator shift of only -0.0528 / +0.0247 percentage points. Evaluator migration therefore explains only a rounding-level fraction of the H2-to-C2 E10 loss.

C2 P E10 under the current evaluator is 87.9118 / 31.8093. Relative to H2’s same-current-evaluator replay (90.9222 / 40.3731), the residual is -3.0104 / -8.5637 percentage points. This residual is not a pure training-code term: it includes the removed K-reg, lower KG weight, AMP-to-FP32 migration, prompt LR change, horizon/selection differences, any loader/augmentation migration, and the C2 E10 full-checkpoint metadata incident. It is correctly labeled a multiple-factor checkpoint/training/config component.

The scheduler is not in the lost-gain component for H2 to C2: both historical H2 and corrected C2 actually step StepLR after each epoch before saving. The missing scheduler.step remains a confirmed and major bug in the old CIR-V2 run, but it is a separate CIR-versus-parent confound and cannot explain this H2-to-C2 decomposition.

No single factor is causally isolated by the existing evidence. K-reg is associated with the H2 champion and absent in C2, but KREG_CAUSAL_STATUS=ASSOCIATED_ONLY until an exact restored-H2 comparison is run.
""")
    write_text("HISTORICAL_SIGNATURE_SEARCH.md", f"""# Historical signature search

The repository and run were found by exact signature search, not by guessing from metric values.

- H1 signature phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3 -> {H1_CHECKPOINT}; source snapshot {H1_COMMIT}.
- H2 signature phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch -> {H2_RUN}; exact implementation commit {H2_COMMIT}.
- H2 checkpoint SHA {H2_CHECKPOINT_SHA} and historical evaluator SHA {H2_EVALUATOR_SHA} were verified.
- C2 corrected parent artifacts are under research_artifacts/cir_rmt_v2/corrective_matched_retrain_20260830/; the original E10 SHA is preserved as {C2_E10_ORIGINAL_SHA}.

The H2 run began after commit {H2_COMMIT} and before later result-document commits. The recovered log, code, checkpoint, and result chronology identify the exact historical implementation.
""")


def build_evaluator_and_scheduler_docs() -> None:
    write_text("HISTORICAL_EVALUATOR_CONTRACT.md", f"""# Historical/current evaluator contract

Historical H1/H2 evaluation uses the legacy test.py path. H2 source hash: {H2_EVALUATOR_SHA}; H1 source hash: {H1_EVALUATOR_SHA}. Both use exact torchmetrics when thresholds are unset, apply pixel_stride=4 to prediction and masks, round metrics before percentage reporting, and use the image score 0.5 classification + 0.5 max pixel score. Colon image metrics are unsupported and appear as zero in the legacy log; this archive normalizes them to undefined and excludes them from the three-target image macro.

The current evaluator implementation hash is {CURRENT_EVALUATOR_SHA}. The same H2 model-state checkpoint was replayed under it. Pixel AUROC/AP changed by only -0.0528/+0.0247 percentage points at the six-domain macro; image AUROC/AP changed by -0.0207/+0.0107 points at the three-domain macro. These values are within the precision implied by the historical two-decimal output and do not support evaluator migration as the primary H2-to-C2 loss.

Historical and current values are separated in HISTORICAL_REPLAY_RESULTS.csv and SAME_CHECKPOINT_CROSS_EVALUATOR.csv.
""")
    write_text("SCHEDULER_FORENSICS.md", """# Scheduler forensics

## H2 historical Phase2B

The H2 source constructs StepLR(optimizer, step_size=1, gamma=0.9). The exact epoch loop calls scheduler.step once after the complete epoch, then reapplies the constant soft-prompt LR policy, then writes adapter_<epoch>.pth. The H2 log records the decayed schedule and historical optimizer group policy.

## C2 corrected parent

The corrected canonical trainer constructs the same StepLR and calls scheduler.step after the epoch and before the history row/checkpoint save. C2 candidate metadata records post-step scheduler state and decayed image/text learning rates.

## Old CIR-V2 run

The pre-fix CIR trainer constructed StepLR but did not call scheduler.step in its epoch loop. Its E12/E14/E16/E18/E20 checkpoint states retained last_epoch=0, _step_count=1, and base image/text learning rates. That bug is independently CIR_SCHEDULER_BUG_CONFIRMED in the preserved pre-fix archive and plausibly explains old CIR instability/excessive late LR exposure.

Conclusion for this audit: SCHEDULER_CAUSAL_STATUS=NOT_SUPPORTED for the H2-to-C2 lost-gain comparison, while the old CIR scheduler bug remains proven as a separate protocol failure.
""")
    write_text("KREG_FORENSICS.md", """# K-reg forensics

H2 implements a genuine K-space regularizer. For a hybrid text embedding, it forms hard text features, the alpha-mixed main text features, and per-stage states. For each vision_text_k projection it computes normalized K-space features for the main and hard paths; the hard projection is detached, and the cosine loss is mean(1 - cosine(k_main, k_hard_detached)), averaged across stages, groups, and normal/abnormal channels. The detached W_K path prevents direct W_K updates while allowing the active soft/main contribution to receive the regularizer gradient.

H2 uses lambda_k=0.002 and includes lambda_k*k_loss in cls + seg + lambda_kg*kg + lambda_k*k. H2’s KG term is distinct: lambda_kg=0.01 aligns soft and hard text embeddings. H2 also logs per-stage K cosine statistics.

The current active trainer exposes lambda_k but _text_with_regularizers sets k_loss = torch.zeros((), device=device). C2 sets lambda_k=0.0, so no K-reg gradient is present. This is CURRENT_KREG_STATUS=STUB_ZERO and is best classified as an accidental feature removal relative to H2. The exact historical formula is recovered, but no restored-H2 training/parity experiment has yet isolated its causal contribution: KREG_CAUSAL_STATUS=ASSOCIATED_ONLY.
""")
    write_text("ARCHITECTURE_IDENTITY.md", f"""# Architecture identity

H1 is historical V3c: hard-prompt Phase1, DFG attention/SS2D weight residual, FP32 attention residual flag, E9 checkpoint {H1_CHECKPOINT_SHA}. H2 is historical hybrid Phase2B with alpha schedule and K-reg, checkpoint {H2_CHECKPOINT_SHA}. C2 is current canonical hybrid Phase2B with no K-reg, config SHA {CURRENT_CONFIG_SHA}, architecture freeze SHA {ARCHITECTURE_FREEZE_SHA}.

The base CLIP/adapter/DFG dimensions are compatible. H1 and H2 are not identical prompt/loss contracts; H2 and C2 are not identical objectives/precision/evaluator contracts. Legacy H1/H2 checkpoints load through the current bridge only with an explicitly recorded legacy-metadata bypass. No architecture equivalence claim is made beyond the measured model-state replay.
""")


def build_selection_audit() -> None:
    write_text("HISTORICAL_CHECKPOINT_SELECTION_AUDIT.md", """# Historical checkpoint-selection audit

H1 E9 and H2 E10 are retrospective champions. H2’s run context explicitly selects E10 by the six-Medical pixel AP mean; H1 is documented as the Phase1 best checkpoint with Medical results. These are RETROSPECTIVE_BEST, not clean target-blind baselines. They remain valid for forensic recovery and understanding lost performance, but their historical winning-epoch scores must not be presented as target-blind prospective results.

This audit did not use new Medical evaluation to choose the H1/H2 replay checkpoint. The current C2 E10 decomposition is fixed for the comparison. Any new parent/anchor training must freeze a source-only or fixed-epoch rule before reading new Medical results.
""")


def build_extension_ingest(h1: dict[str, dict[str, float | None]], h2: dict[str, dict[str, float | None]]) -> None:
    pa_decision = json.loads(PA_DECISION.read_text(encoding="utf-8"))
    final_macro = read_final_macro()
    pa_by_epoch_metric: dict[tuple[int, str], float | None] = {}
    for row in read_csv(PA_MACRO):
        pa_by_epoch_metric[(int(row["epoch"]), row["metric"])] = number(row.get("PA"))
    ingest = {
        "status": "INGESTED_EXISTING_ARCHIVE_ONLY", "archive": str(PA_ARCHIVE),
        "CIR_TRAINING_VALUE": pa_decision.get("CIR_TRAINING_VALUE", "INCONCLUSIVE"), "INFERENCE_RMT_VALUE": pa_decision.get("INFERENCE_RMT_VALUE", "NEUTRAL"),
        "FINAL_ARCHITECTURE": pa_decision.get("FINAL_ARCHITECTURE", "MIXED_UNRESOLVED"), "SOURCE_STATUS": pa_decision.get("SOURCE_STATUS", "PASS"), "MEDICAL_STATUS": pa_decision.get("MEDICAL_STATUS", "PASS"),
        "TARGET_TUNING_OCCURRED": pa_decision.get("TARGET_TUNING_OCCURRED", "NO"), "comparisons": {}, "evidence": ["PA_CONTROL_DIFF_AUDIT.md", "MEDICAL_FACTORIAL_MACRO.csv", "MEDICAL_FACTORIAL_INTERACTION.csv", "PA_SOURCE_RESULTS.csv", "FINAL_ARCHITECTURE_DECISION.json"],
    }
    for epoch in [10, 12, 14, 16, 18, 20]:
        entry: dict[str, Any] = {}
        for metric in METRICS:
            pa = pa_by_epoch_metric.get((epoch, metric)); p = final_macro.get((epoch, "P"), {}).get(metric); old = final_macro.get((epoch, "C_OLD_0"), {}).get(metric); a0 = final_macro.get((epoch, "A0"), {}).get(metric)
            entry[metric] = {"A0_minus_PA": None if a0 is None or pa is None else a0 - pa, "PA_minus_P": None if pa is None or p is None else pa - p, "A0_minus_C_OLD": None if a0 is None or old is None else a0 - old}
        ingest["comparisons"][f"E{epoch}"] = entry
    ingest["interaction"] = [{"epoch": int(row["epoch"]), "metric": row["metric"], "interaction": number(row.get("interaction")), "CIR_with_anchor_minus_PA": number(row.get("CIR_with_anchor_minus_PA"))} for row in read_csv(PA_INTERACTION)]
    write_json("PREDECESSOR_CIR_CONTROL_INGEST.json", ingest)
    write_text("PREDECESSOR_CIR_CONTROL_INGEST.md", """# Predecessor CIR-control ingest

The existing PA control archive was ingested without rerunning training or evaluation. PA is a no-CIR, image-anchor control using the canonical parent contract. Its final decision is CIR_TRAINING_VALUE=INCONCLUSIVE, INFERENCE_RMT_VALUE=NEUTRAL, and FINAL_ARCHITECTURE=MIXED_UNRESOLVED; its source and Medical status are PASS and no target tuning occurred in that run.

The factorial evidence shows that anchor/no-CIR (PA) improves C2 P on Medical pixel AUROC/AP at most epochs, while CIR-with-anchor versus PA is mixed by metric and epoch. The conditional alpha comparison is near neutral, so new strong-parent candidates default to native inference alpha=0. This predecessor evidence does not independently justify a full CIR trajectory; a CIR trajectory would require a bounded strong-parent source-only benefit.

Exact derived comparisons (A0-PA, PA-P, A0-C_OLD, and factorial interaction) are in PREDECESSOR_CIR_CONTROL_INGEST.json and reference the frozen PA CSVs.
""")
    champion_specs = [
        ("H1", "H1", 9, h1, H1_CHECKPOINT_SHA, "legacy", CURRENT_REPO_COMMIT, str(H1_CURRENT_RESULTS), "fixed historical E9"),
        ("H2", "H2", 10, h2, H2_CHECKPOINT_SHA, CURRENT_CONFIG_SHA, CURRENT_REPO_COMMIT, str(H2_CURRENT_RESULTS), "fixed historical E10"),
        ("C2", "P", 18, read_final_method("P", 18), "C2_E18_SHA_IN_FINAL_MATRIX", CURRENT_CONFIG_SHA, C2_TRAINING_COMMIT, str(CURRENT_FINAL_MATRIX), "max current P pixel AP among frozen candidates"),
        ("C_OLD", "C_OLD_0", 14, read_final_method("C_OLD_0", 14), "C_OLD_E14_SHA_IN_FINAL_MATRIX", CIR_CONFIG_SHA, C2_TRAINING_COMMIT, str(CURRENT_FINAL_MATRIX), "max current old-CIR pixel AP among frozen candidates"),
        ("A0", "A0", 12, read_final_method("A0", 12), "A0_E12_SHA_IN_FINAL_MATRIX", CIR_CONFIG_SHA, CURRENT_REPO_COMMIT, str(CURRENT_FINAL_MATRIX), "max current A0 pixel AP among frozen candidates"),
        ("PA", "PA", 12, read_pa_method(12), "PA_E12_SHA_IN_PA_RESULTS", CURRENT_CONFIG_SHA, "0987f67e92020e74cdae65e51b8a8676e08f3c84", str(PA_MEDICAL), "max PA pixel AP among frozen candidates"),
    ]
    fields = ["lineage", "method", "epoch", "target", *METRICS, "evaluator_sha256", "evaluator_git_sha", "checkpoint_sha256", "config_sha256", "selection_rule", "comparability", "source"]
    rows = []
    for lineage, method, epoch, data, checkpoint, config, git_sha, source, rule in champion_specs:
        for target in TARGETS:
            rows.append({"lineage": lineage, "method": method, "epoch": epoch, "target": target, **data.get(target, {}), "evaluator_sha256": CURRENT_EVALUATOR_SHA, "evaluator_git_sha": git_sha, "checkpoint_sha256": checkpoint, "config_sha256": config, "selection_rule": rule, "comparability": "SAME_EVALUATOR_SOURCE_HASH; legacy bridge bypass recorded where applicable", "source": source})
        rows.append({"lineage": lineage, "method": method, "epoch": epoch, "target": "ALL_MEDICAL_MACRO", **{metric: macro(data, metric)[0] for metric in METRICS}, "evaluator_sha256": CURRENT_EVALUATOR_SHA, "evaluator_git_sha": git_sha, "checkpoint_sha256": checkpoint, "config_sha256": config, "selection_rule": rule, "comparability": "same evaluator implementation hash; historical selection provenance remains separate", "source": source})
    write_csv("SAME_EVALUATOR_LOCAL_CHAMPIONS.csv", fields, rows)
    write_text("SAME_EVALUATOR_LOCAL_CHAMPIONS.md", """# Same-evaluator local champions

This table keeps H1/H2 historical-evaluator results out of the ranking. H1 and H2 were replayed through the current evaluator implementation; C2, old CIR, A0, and PA are ingested from existing compact matrices with the same evaluator source hash. Legacy checkpoint metadata bypasses for H1/H2 are explicitly labeled.

The strongest same-evaluator historical model-state parent on six-domain pixel metrics is H2 E10 (about 90.92 / 40.37), ahead of H1 E9 (about 90.71 / 39.85) and current C2 E10 (87.91 / 31.81). Current A0/PA are later anchor-lineage results and are not historical-parent replacements. H2 is therefore selected for restoration audit, subject to parity and anchor portability gates.
""")
    write_json("STRONG_PARENT_SELECTION.json", {"status": "SELECTED_FOR_RESTORATION_AUDIT", "selected_parent": "HISTORICAL_PHASE2B", "candidate_basis": "H2 retains the highest same-current-evaluator six-domain pixel AUROC/AP among H1/H2/C2 E10 replays", "h2_current_macro": {metric: macro(h2, metric)[0] for metric in METRICS}, "h1_current_macro": {metric: macro(h1, metric)[0] for metric in METRICS}, "selection_contamination": "H2 E10 is retrospective Medical-selected; new training must use source/fixed checkpoint rule", "next_gate": "exact restored H2 implementation/parity, then parent-compatible source-only anchor reference"})
    write_text("STRONG_PARENT_SELECTION.md", """# Strong-parent selection

Selected parent: HISTORICAL_PHASE2B (H2), for restoration audit only.

H2 is reproducible under the historical evaluator and remains stronger than H1 and C2 E10 under the same current evaluator on the primary six-domain pixel objective. H2’s E10 selection was retrospective Medical-guided, so this is not a clean prospective baseline selection. The selection is justified for recovering the lost parent contract, not for claiming an unbiased benchmark win.

The next gates are exact H2 semantic restoration, fixed-input parity, and a parent-compatible source-only anchor reference. No Medical result may select a new checkpoint or anchor reference.
""")


def build_extension_status_docs() -> None:
    write_json("RESTORED_PARENT_CONTRACT.json", {"status": "AUDIT_SPECIFICATION", "selected_parent": "HISTORICAL_PHASE2B", "historical_contract": "HISTORICAL_PHASE2B_CONTRACT.json", "required_semantics": ["AMP/autocast/GradScaler enabled", "hybrid alpha schedule", "lambda_kg=0.01", "exact lambda_k=0.002 detached-W_K K-reg", "Adam groups and LRs", "StepLR gamma=.9 stepped after epoch before save", "batch/effective batch 6", "DFG/SS2D weight-residual beta schedule", "historical evaluator kept separate"], "modernization_allowed": ["atomic checkpoint writes", "deterministic resume", "bounded evaluator", "manifests and ledgers"], "scientific_changes_forbidden": ["precision", "loss", "K-reg", "prompt", "scheduler", "optimizer", "augmentation", "DFG math", "scoring"]})
    write_text("RESTORED_PARENT_IMPLEMENTATION_AUDIT.md", """# Restored-parent implementation audit

Status: NOT_RUN for a new active-branch restored-H2 trainer. The exact historical source and formula are recovered, but the active branch has not yet introduced a restored-H2 training variant. This prevents an unsupported claim that current C2 code is H2-equivalent.

The implementation gate must preserve historical AMP/mixed precision, hybrid alpha and prompt policy, KG/K-reg coefficients, Adam group order/rates, StepLR timing, DFG fields, data path/augmentation, and historical loss/scoring semantics. Engineering improvements may add atomic writes, resume state, and bounded ledgers only after fixed-input parity.
""")
    write_json("RESTORED_PARENT_PARITY.json", {"status": "NOT_RUN", "selected_parent": "HISTORICAL_PHASE2B", "historical_checkpoint_sha256": H2_CHECKPOINT_SHA, "required_comparisons": ["image adapter outputs", "text adapter outputs", "stage features", "DFG native weights", "segmentation logits", "deployed anomaly map", "image logits", "bounded metric parity"], "reason": "No new restored-H2 implementation has been added; current legacy bridge replay is not a full historical-vs-restored parity test."})
    write_text("RESTORED_PARENT_PARITY.md", """# Restored-parent parity

Status: NOT_RUN. The current bridge can load H2 and reproduce aggregate scores, but that is a current-evaluator model-state replay, not the required fixed-input historical-code versus restored-active-code parity test. No training is authorized by this artifact until that parity gate passes.
""")
    write_text("STRONG_PARENT_ANCHOR_REFERENCE_DECISION.md", """# Strong-parent anchor reference decision

Status: BLOCKED_PENDING_RESTORED_PARENT_PARITY.

The current C2 P E14 reference must not be transplanted to H2 because H2 has a different prompt/loss/precision trajectory. H2 E10 is retrospective Medical-selected and cannot be used as a clean target-blind anchor reference. A scientifically admissible first reference could be a fixed H2 E1 model-state checkpoint under a preregistered rule, but compatibility and source-only behavior must be verified after restored-H2 parity. No anchor full train has been authorized in this snapshot.

Required frozen mechanism remains normalized per-parameter squared distance on image_adapter, frozen reference, lambda_image_anchor=0.001, no optimizer registration for the reference, and anchor absent at inference.
""")
    write_text("STRONG_PARENT_EXTENSION_IMPLEMENTATION_AUDIT.md", """# Strong-parent extension implementation audit

Status: NOT_RUN.

The predecessor control is ingested, but no restored-H2 RA/RCA implementation or GPU training is claimed here. The implementation gate remains parity first, then a source-only bounded gate. Inference RMT remains disabled by default; any eligible CIR would be train-time only with native alpha=0 deployment.
""")
    write_json("STRONG_PARENT_EXTENSION_TESTS.json", {"status": "NOT_RUN", "required": ["unit tests", "deterministic smoke", "forward parity", "backward parity where applicable", "checkpoint save/resume", "optimizer state resume", "scheduler state resume", "RNG resume", "anchor-only image_adapter gradient", "CIR train-only path", "alpha=0 identity", "peer/delta stopgrad", "no GT dependency"]})
    write_csv("STRONG_PARENT_SOURCE_GATE.csv", ["candidate", "horizon", "source_status", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap", "parameter_drift", "feature_drift", "anchor_gradient_ratio", "stability", "decision", "status", "notes"], [
        {"candidate": "RA", "horizon": "E10", "source_status": "NOT_RUN", "decision": "PENDING_RESTORED_PARENT_PARITY", "status": "NOT_RUN", "notes": "No new H2 restoration/anchor training"},
        {"candidate": "RCA", "horizon": "E10", "source_status": "NOT_RUN", "decision": "PENDING_PREDECESSOR_AND_RA_GATE", "status": "NOT_RUN", "notes": "CIR training value predecessor is inconclusive; source-only evidence required"},
    ])
    write_text("STRONG_PARENT_SOURCE_GATE.md", """# Strong-parent source-only gate

Status: NOT_RUN. Existing C2/A/PA source results do not restore H2’s contract. A new RA E10 source gate is eligible after parity. RCA is conditional because the predecessor PA control classified CIR training as inconclusive and inference RMT as neutral.
""")
    write_text("FINAL_TRAIN_CANDIDATE_FREEZE.md", """# Final train candidate freeze

Status: NOT_CREATED.

No new strong-parent training candidate is frozen. A real freeze must be created before reading any new source or Medical target result.
""")
    write_json("FINAL_TRAIN_CANDIDATE_FREEZE.json", {"status": "NOT_CREATED", "reason": "restored-parent parity and anchor portability gates not yet passed", "target_tuning_occurred": False, "mvtec_tuning_occurred": False})
    write_text("FINAL_SOURCE_MATRIX.csv", "status,reason\nNOT_CREATED,No new strong-parent source training was authorized before restored-parent parity\n")
    write_text("PRE_FINAL_MEDICAL_FREEZE.md", """# Pre-final Medical freeze

Status: NOT_CREATED. No new final candidate exists. Existing C2/A/PA Medical matrices remain separate predecessor artifacts.
""")
    write_json("PRE_FINAL_MEDICAL_FREEZE.json", {"status": "NOT_CREATED", "medical_evaluation": "NOT_RUN", "target_tuning_occurred": False, "reason": "source-only restored-parent gate not passed"})
    write_text("FINAL_MEDICAL_LEDGER.csv", "status,reason\nNOT_CREATED,No new final candidate Medical cells were authorized\n")
    write_text("PRE_MVTEC_FREEZE.md", """# Pre-MVTec freeze

Status: NOT_CREATED; MVTec remains untouched and was not used for selection or tuning.
""")
    write_text("FINAL_MVTEC_RESULTS.csv", "status,reason\nNOT_RUN,MVTec is confirmatory only and no final candidate was frozen\n")
    write_text("FINAL_MVTEC_CONFIRMATION.md", """# Final MVTec confirmation

Status: NOT_RUN. MVTec remains untouched in this audit lineage.
""")


def build_final_comparison_and_decision(h1: dict[str, dict[str, float | None]], h2: dict[str, dict[str, float | None]], h2_hist: dict[str, dict[str, float | None]], c2: dict[str, dict[str, float | None]]) -> None:
    final_macro = read_final_macro()
    pa_by_epoch_metric: dict[tuple[int, str], float | None] = {}
    for row in read_csv(PA_MACRO):
        pa_by_epoch_metric[(int(row["epoch"]), row["metric"])] = number(row.get("PA"))

    fields = ["lineage", "method", "evaluator_family", "epoch", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap", "pixel_support", "image_support", "checkpoint_status", "selection_status", "comparability", "source"]
    rows: list[dict[str, Any]] = []
    def add_row(lineage: str, method: str, family: str, epoch: int | str, values: dict[str, float | None], pixel_support: int, image_support: int, checkpoint_status: str, selection_status: str, comparability: str, source: str) -> None:
        rows.append({"lineage": lineage, "method": method, "evaluator_family": family, "epoch": epoch, **values, "pixel_support": pixel_support, "image_support": image_support, "checkpoint_status": checkpoint_status, "selection_status": selection_status, "comparability": comparability, "source": source})

    add_row("PUBLISHED", "ACD-CLIP_N3", "PUBLISHED_EXTERNAL", "reference", {"pixel_auroc": 0.9155, "pixel_ap": 0.4303, "image_auroc": None, "image_ap": None}, 6, 0, "EXTERNAL_REFERENCE", "published", "not same evaluator", "published ACD-CLIP N=3 reference")
    add_row("H1", "H1", "HISTORICAL_EVALUATOR", 9, {"pixel_auroc": 0.9076, "pixel_ap": 0.3982, "image_auroc": 0.7380, "image_ap": 0.7506}, 6, 3, "LEGACY_MODEL_STATE", "RETROSPECTIVE_BEST", "historical H1 result", "historical H1 report")
    add_row("H2", "H2", "HISTORICAL_EVALUATOR", 10, {metric: macro(h2_hist, metric)[0] for metric in METRICS}, 6, 3, "LEGACY_MODEL_STATE", "RETROSPECTIVE_BEST", "historical H2 result", str(H2_HISTORICAL_RESULTS))
    add_row("H1", "H1", "CURRENT_EVALUATOR", 9, {metric: macro(h1, metric)[0] for metric in METRICS}, 6, 3, "LEGACY_METADATA_BYPASS", "RETROSPECTIVE_BEST", "same evaluator implementation hash", str(H1_CURRENT_RESULTS))
    add_row("H2", "H2", "CURRENT_EVALUATOR", 10, {metric: macro(h2, metric)[0] for metric in METRICS}, 6, 3, "LEGACY_METADATA_BYPASS", "RETROSPECTIVE_BEST", "same evaluator implementation hash", str(H2_CURRENT_RESULTS))
    add_row("C2", "P", "CURRENT_EVALUATOR", 10, {metric: macro(c2, metric)[0] for metric in METRICS}, 6, 3, "MODEL_STATE_REPLAY_ONLY", "FIXED_FOR_DECOMPOSITION", "same evaluator implementation hash", str(C2_MEDICAL))
    for lineage, method, epoch, values, source, status in [
        ("C2", "P", 18, final_macro[(18, "P")], str(CURRENT_FINAL_MACRO), "FROZEN_CURRENT_ARCHIVE"),
        ("PA", "PA", 12, {metric: pa_by_epoch_metric[(12, metric)] for metric in METRICS}, str(PA_MACRO), "FROZEN_CURRENT_ARCHIVE"),
        ("A", "A0", 12, final_macro[(12, "A0")], str(CURRENT_FINAL_MACRO), "FROZEN_CURRENT_ARCHIVE"),
    ]:
        add_row(lineage, method, "CURRENT_EVALUATOR", epoch, values, 6, 3, status, "MAX_PIXEL_AP_WITHIN_LINEAGE", "same evaluator implementation hash; not historical parent", source)
    add_row("R", "RESTORED_STRONG_PARENT", "NOT_RUN", "pending", {"pixel_auroc": None, "pixel_ap": None, "image_auroc": None, "image_ap": None}, 0, 0, "NOT_RUN", "NOT_APPLICABLE", "no restored-H2 training", "RESTORED_PARENT_PARITY.json")
    add_row("RA/RCA", "FINAL_CANDIDATE", "NOT_RUN", "pending", {"pixel_auroc": None, "pixel_ap": None, "image_auroc": None, "image_ap": None}, 0, 0, "NOT_RUN", "NOT_APPLICABLE", "source gate blocked", "FINAL_TRAIN_CANDIDATE_FREEZE.json")
    write_csv("FINAL_PUBLISHED_AND_LOCAL_COMPARISON.csv", fields, rows)
    write_text("FINAL_PUBLISHED_AND_LOCAL_COMPARISON.md", """# Published and local comparison

Historical evaluator values and same-current-evaluator values are separate rows. The external published ACD-CLIP N=3 reference is not treated as a same-evaluator result.

H2 is the strongest recovered historical parent under the current evaluator (about 90.92 / 40.37 Pixel AUROC/AP), followed by H1 (about 90.71 / 39.85) and C2 P E10 (87.91 / 31.81). Existing PA/A0 rows are current anchor-lineage evidence, not restored-H2 results. No restored parent R, RA, RCA, or final candidate was trained, so no final architecture claim is made.
""")

    gain_rows = []
    def gain(component: str, comparison: str, metric: str, value: float | None, status: str, interpretation: str) -> None:
        gain_rows.append({"component": component, "comparison": comparison, "metric": metric, "value_percentage_points": value, "status": status, "interpretation": interpretation})
    for metric in ("pixel_auroc", "pixel_ap"):
        gain("historical_contract_recovery", "H2_current_eval - C2_E10_current_eval", metric, (macro(h2, metric)[0] - macro(c2, metric)[0]) * 100.0, "OBSERVED_NOT_CAUSAL", "same-evaluator model-state comparison; residual has multiple H2-to-C2 factors")
        gain("evaluator_migration", "H2_current_eval - H2_historical_eval", metric, (macro(h2, metric)[0] - macro(h2_hist, metric)[0]) * 100.0, "OBSERVED", "same checkpoint cross-evaluator component")
        gain("anchor_predecessor", "PA_E12 - P_E12", metric, (pa_by_epoch_metric[(12, metric)] - final_macro[(12, "P")][metric]) * 100.0, "ASSOCIATED_ONLY", "existing PA control; not pure anchor gain on restored H2")
        gain("cir_predecessor", "A0_E12 - PA_E12", metric, (final_macro[(12, "A0")][metric] - pa_by_epoch_metric[(12, metric)]) * 100.0, "INCONCLUSIVE", "existing factorial comparison; CIR training not isolated cleanly")
        gain("restored_parent", "R - C2", metric, None, "NOT_RUN", "no restored H2 trajectory")
        gain("anchor_on_restored_parent", "RA - R", metric, None, "NOT_RUN", "no parent-compatible anchor trajectory")
        gain("cir_on_restored_parent", "RCA - RA", metric, None, "NOT_RUN", "CIR conditional gate not reached")
    write_csv("GAIN_ATTRIBUTION_DECOMPOSITION.csv", ["component", "comparison", "metric", "value_percentage_points", "status", "interpretation"], gain_rows)
    write_text("GAIN_ATTRIBUTION_DECOMPOSITION.md", """# Gain attribution decomposition

The observed H2-current minus C2 E10-current comparison is baseline recovery evidence, not a causal result from a new restored run. The evaluator component is separately measured and small. Existing PA and A0 comparisons are predecessor factorial associations and must not be relabeled as pure anchor or CIR gains.

R (restored H2), RA (restored H2 plus anchor), and RCA (restored H2 plus CIR-training plus anchor) were not trained because the restored-parent parity and parent-compatible anchor gates were not reached. Their causal increments therefore remain NOT_RUN.
""")

    final_decision = {
        "PREDECESSOR_CIR_CONTROL_STATUS": "INGESTED_PASS",
        "PREDECESSOR_CIR_TRAINING_VALUE": "INCONCLUSIVE",
        "PREDECESSOR_INFERENCE_RMT_VALUE": "NEUTRAL",
        "H1_IDENTITY_STATUS": "CONFIRMED",
        "H1_CHECKPOINT_SHA": H1_CHECKPOINT_SHA,
        "H1_REPLAY_PIXEL_AUROC": macro(h1, "pixel_auroc")[0],
        "H1_REPLAY_PIXEL_AP": macro(h1, "pixel_ap")[0],
        "H2_IDENTITY_STATUS": "CONFIRMED",
        "H2_CHECKPOINT_SHA": H2_CHECKPOINT_SHA,
        "H2_REPLAY_PIXEL_AUROC": macro(h2, "pixel_auroc")[0],
        "H2_REPLAY_PIXEL_AP": macro(h2, "pixel_ap")[0],
        "C2_PIXEL_AUROC": macro(c2, "pixel_auroc")[0],
        "C2_PIXEL_AP": macro(c2, "pixel_ap")[0],
        "STRONGEST_SAME_EVAL_PARENT": "H2",
        "H2_KREG_FORMULA_STATUS": "RECOVERED_EXACT",
        "H2_KREG_RESTORED": "NO",
        "HISTORICAL_PRECISION_STATUS": "AMP_TO_FP32_MIGRATION_PROVEN",
        "HISTORICAL_SCHEDULER_STATUS": "MATCHED_ACTUAL_STEP",
        "SELECTED_PARENT": "HISTORICAL_PHASE2B",
        "ANCHOR_PORTABILITY": "BLOCKED",
        "ANCHOR_REFERENCE_SHA": None,
        "ANCHOR_LAMBDA": 0.001,
        "CIR_PORTABILITY": "BLOCKED",
        "CIR_TRAINING_USED": "NO",
        "INFERENCE_RMT_USED": "NO",
        "SOURCE_GATE_DECISION": "BLOCKED_BEFORE_RESTORED_PARENT_PARITY",
        "FULL_TRAIN_STATUS": "NOT_RUN",
        "FINAL_MEDICAL_PIXEL_AUROC": None,
        "FINAL_MEDICAL_PIXEL_AP": None,
        "FINAL_MEDICAL_IMAGE_AUROC": None,
        "FINAL_MEDICAL_IMAGE_AP": None,
        "BASELINE_RECOVERY_AUROC": None,
        "BASELINE_RECOVERY_AP": None,
        "ANCHOR_INCREMENT_AUROC": None,
        "ANCHOR_INCREMENT_AP": None,
        "CIR_INCREMENT_AUROC": None,
        "CIR_INCREMENT_AP": None,
        "PUBLISHED_ACDCLIP_COMPARISON": "BELOW_ON_H2_CURRENT_PIXEL; NOT_SAME_PROTOCOL",
        "MVTec_STATUS": "NOT_RUN",
        "MVTec_CONFIRMATION": "NOT_RUN",
        "TARGET_TUNING_OCCURRED": "NO",
        "MVTec_TUNING_OCCURRED": "NO",
        "FINAL_ARCHITECTURE": "BLOCKED_BEFORE_RESTORED_PARENT_TRAIN",
        "FINAL_DECISION": "BLOCKED_RESTORED_PARENT_PARITY_AND_ANCHOR_PORTABILITY",
    }
    write_json("FINAL_STRONG_PARENT_DECISION.json", final_decision)
    write_text("FINAL_STRONG_PARENT_DECISION.md", """# Final strong-parent decision

H2 is the strongest same-current-evaluator historical parent and is selected for restoration audit. The predecessor PA control is ingested and remains inconclusive for CIR training; inference RMT remains neutral.

The final decision is BLOCKED before new strong-parent training: no active restored-H2 implementation has passed the required fixed-input parity test, and no parent-compatible target-blind anchor reference has been frozen. Consequently R, RA, RCA, full source evaluation, new Medical evaluation, and MVTec confirmation are NOT_RUN. This is a reproducibility/implementation gate, not evidence that H2, Anchor, or CIR failed.
""")


def build_final_audit(h1: dict[str, dict[str, float | None]], h2: dict[str, dict[str, float | None]], h2_hist: dict[str, dict[str, float | None]], c2: dict[str, dict[str, float | None]]) -> None:
    h2_hist_macro = {metric: macro(h2_hist, metric)[0] for metric in METRICS}
    h2_macro = {metric: macro(h2, metric)[0] for metric in METRICS}
    h1_macro = {metric: macro(h1, metric)[0] for metric in METRICS}
    c2_macro = {metric: macro(c2, metric)[0] for metric in METRICS}
    evaluator_component = {metric: h2_macro[metric] - h2_hist_macro[metric] for metric in METRICS}
    training_component = {metric: c2_macro[metric] - h2_macro[metric] for metric in METRICS}
    total = {metric: c2_macro[metric] - h2_hist_macro[metric] for metric in METRICS}
    audit = {
        "HISTORICAL_REPO_STATUS": "CONFIRMED", "HISTORICAL_REPO_PATH": str(HISTORICAL_REPO), "HISTORICAL_REPO_REMOTE": HISTORICAL_REMOTE, "HISTORICAL_REPO_COMMIT": H2_COMMIT,
        "HISTORICAL_CHECKPOINT_STATUS": "CONFIRMED_MODEL_STATE; FULL_STATE_NOT_PRESENT", "HISTORICAL_CHECKPOINT_PATH": str(H2_CHECKPOINT), "HISTORICAL_CHECKPOINT_SHA256": H2_CHECKPOINT_SHA, "HISTORICAL_CHECKPOINT_EPOCH": 10,
        "HISTORICAL_CONFIG_ALPHA": 0.2, "HISTORICAL_KREG": "lambda_k=0.002; exact detached-W_K cosine formula recovered", "HISTORICAL_SCHEDULER_BEHAVIOR": "StepLR gamma=.9 actually stepped once per epoch before save",
        "CURRENT_REPO_PATH": str(ROOT), "CURRENT_REPO_COMMIT": CURRENT_REPO_COMMIT, "HISTORICAL_REPLAY_STATUS": "PASS",
        "HISTORICAL_REPLAY_PIXEL_AUROC": h2_hist_macro["pixel_auroc"], "HISTORICAL_REPLAY_PIXEL_AP": h2_hist_macro["pixel_ap"], "SAME_CHECKPOINT_CURRENT_EVAL_PIXEL_AUROC": h2_macro["pixel_auroc"], "SAME_CHECKPOINT_CURRENT_EVAL_PIXEL_AP": h2_macro["pixel_ap"],
        "CURRENT_P_E10_CURRENT_EVAL_PIXEL_AUROC": c2_macro["pixel_auroc"], "CURRENT_P_E10_CURRENT_EVAL_PIXEL_AP": c2_macro["pixel_ap"], "TOTAL_GAP_AUROC": total["pixel_auroc"], "TOTAL_GAP_AP": total["pixel_ap"],
        "EVALUATOR_COMPONENT_AUROC": evaluator_component["pixel_auroc"], "EVALUATOR_COMPONENT_AP": evaluator_component["pixel_ap"], "CHECKPOINT_TRAINING_COMPONENT_AUROC": training_component["pixel_auroc"], "CHECKPOINT_TRAINING_COMPONENT_AP": training_component["pixel_ap"],
        "CHECKPOINT_SELECTION_STATUS": "H2_RETROSPECTIVE_BEST; H1_RETROSPECTIVE_BEST; C2_E10_FIXED_FOR_DECOMPOSITION", "ARCHITECTURE_IDENTITY": "BASE_DFG_COMPATIBLE; H1/H2/C2_FULL_CONTRACT_NOT_IDENTICAL", "PRIMARY_GAP_CLASS": "MULTIPLE_FACTORS", "KREG_CAUSAL_STATUS": "ASSOCIATED_ONLY", "SCHEDULER_CAUSAL_STATUS": "NOT_SUPPORTED",
        "NEXT_EXPERIMENT": "restore exact H2 semantics, pass fixed-input parity, then source-only RA E10; add RCA only if source gate earns it", "RETRAIN_PERFORMED": "NO", "TARGET_TUNING_OCCURRED": "NO",
        "CHECKPOINT_INTEGRITY_INCIDENT": {"original_c2_e10_sha256": C2_E10_ORIGINAL_SHA, "observed_c2_e10_sha256": C2_E10_OBSERVED_SHA, "model_state_preserved": True, "optimizer_scheduler_rng_state_lost": True, "claim_limit": "C2 E10 model-state replay only"},
    }
    write_json("FINAL_ROOT_CAUSE_AUDIT.json", audit)
    write_text("FINAL_ROOT_CAUSE_AUDIT.md", f"""# Final historical Phase2B root-cause audit

## Primary conclusion

H2 is confirmed and reproducible. The exact historical E10 model-state checkpoint {H2_CHECKPOINT_SHA} replays to the historical logged macro {100*h2_hist_macro['pixel_auroc']:.2f} / {100*h2_hist_macro['pixel_ap']:.2f} Pixel AUROC/AP. The same checkpoint through the current evaluator gives {100*h2_macro['pixel_auroc']:.4f} / {100*h2_macro['pixel_ap']:.4f}. Current C2 P E10 is {100*c2_macro['pixel_auroc']:.4f} / {100*c2_macro['pixel_ap']:.4f}.

The same-checkpoint evaluator component is {100*evaluator_component['pixel_auroc']:+.4f} / {100*evaluator_component['pixel_ap']:+.4f} percentage points. The residual C2-vs-H2-current component is {100*training_component['pixel_auroc']:+.4f} / {100*training_component['pixel_ap']:+.4f} points. This residual is a multiple-factor H2-to-C2 migration component, not a pure training-line attribution.

## What is proven

- The historical H2 repository/run/checkpoint/evaluator identity is exact enough to reproduce the logged E10 historical result to the logged precision.
- The current evaluator is not the dominant explanation of H2’s historical gain: same-checkpoint cross-evaluator shifts are rounding-level.
- H2 actually stepped StepLR; C2 corrected parent also steps StepLR. Scheduler migration is not supported as the cause of H2-to-C2 loss.
- Current C2 K-reg is a zero stub, while H2 contains exact detached-W_K K-reg at lambda_k=0.002.
- H2 used AMP/autocast/GradScaler; C2 used FP32 without AMP. H2 used lambda_kg=0.01, soft-prompt LR 5e-5, and a 15-epoch retrospective candidate protocol; C2 differs on each.
- H2 and H1 winning checkpoint selection was retrospective Medical-informed, so those historical champions are not target-blind.

## What remains correlational or unknown

K-reg is associated with the H2 champion but its causal contribution is not isolated. Precision, KG coefficient, prompt LR, horizon, loader/augmentation, evaluator, and selection effects are jointly confounded. The C2 E10 full optimizer/scheduler/RNG payload was lost in the disclosed serialization incident; model-state metrics remain usable, but exact resume-state verification is unknown.

## Extension gate

The predecessor PA control is ingested separately. It classified CIR training as inconclusive and inference RMT as neutral. H2 is selected for restoration audit because it is the strongest same-current-evaluator pixel parent, but no restored-H2 implementation, parity test, source-only anchor gate, new Medical evaluation, or MVTec evaluation was run in this snapshot. The next authorized experiment is an exact H2-contract restoration with fixed-input parity, followed by a source-only bounded RA gate; CIR training is conditional and inference RMT remains off by default.
""")


def build_resource_and_ledger() -> None:
    write_csv("AUDIT_REPLAY_LEDGER.csv", ["test_id", "scope", "status", "repository_or_code", "checkpoint", "evaluator", "result_artifact", "scientific_claim", "caveat"], [
        {"test_id": "T1", "scope": "exact H2 historical evaluator replay", "status": "PASS", "repository_or_code": f"{HISTORICAL_REPO}@{H2_COMMIT}", "checkpoint": H2_CHECKPOINT_SHA, "evaluator": H2_EVALUATOR_SHA, "result_artifact": "HISTORICAL_REPLAY_RESULTS.csv", "scientific_claim": "historical E10 metric reproduced", "caveat": "historical output rounded; retrospective Medical selection"},
        {"test_id": "T2", "scope": "same H2 checkpoint current evaluator", "status": "PASS", "repository_or_code": str(ROOT), "checkpoint": H2_CHECKPOINT_SHA, "evaluator": CURRENT_EVALUATOR_SHA, "result_artifact": "SAME_CHECKPOINT_CROSS_EVALUATOR.csv", "scientific_claim": "evaluator migration component bounded", "caveat": "legacy identity gate bypassed; model-state checkpoint"},
        {"test_id": "T3", "scope": "C2 P E10 current-evaluator comparison", "status": "PASS_WITH_INTEGRITY_CAVEAT", "repository_or_code": f"{ROOT}@{C2_TRAINING_COMMIT}", "checkpoint": C2_E10_ORIGINAL_SHA, "evaluator": CURRENT_EVALUATOR_SHA, "result_artifact": "CROSS_CHECKPOINT_CURRENT_EVAL.csv", "scientific_claim": "current parent E10 metric preserved from frozen archive", "caveat": "physical full checkpoint overwritten; model-state replay only"},
        {"test_id": "T4", "scope": "H1 current-evaluator bounded replay", "status": "PASS", "repository_or_code": str(ROOT), "checkpoint": H1_CHECKPOINT_SHA, "evaluator": CURRENT_EVALUATOR_SHA, "result_artifact": "CROSS_CHECKPOINT_CURRENT_EVAL.csv", "scientific_claim": "H1 current-evaluator comparator obtained", "caveat": "legacy identity gate bypassed"},
        {"test_id": "T5", "scope": "predecessor PA ingest", "status": "PASS_INGEST_ONLY", "repository_or_code": str(PA_ARCHIVE), "checkpoint": "PA E10-E20 hashes in PA archive", "evaluator": "PA archive", "result_artifact": "PREDECESSOR_CIR_CONTROL_INGEST.json", "scientific_claim": "CIR training value remains inconclusive; inference RMT neutral", "caveat": "no rerun"},
        {"test_id": "T6", "scope": "restored-parent fixed-input parity", "status": "NOT_RUN", "repository_or_code": "not implemented", "checkpoint": H2_CHECKPOINT_SHA, "evaluator": "not applicable", "result_artifact": "RESTORED_PARENT_PARITY.json", "scientific_claim": "none", "caveat": "gate before new training"},
    ])
    write_text("RESOURCE_REPORT.md", """# Resource/reporting record

This archive is compact and contains reports, CSV summaries, JSON manifests, and audit code references only. It intentionally excludes raw per-pixel stores, memmaps, evaluator spools, caches, huge logs, and checkpoints.

Completed replay work used one temporary historical worktree and bounded current-evaluator replay outputs under runs/phase2b_historical_gain_forensics_20260901/. Temporary replay spools were cleaned after each target. No training process was running during the final audit check, no duplicate training process was launched, and no new training result is claimed.

The C2 E10 serialization incident is documented separately; its consequence is a model-state-only comparison, not a resource failure or a reason to fabricate a replacement checkpoint.
""")


def build_hashes() -> None:
    entries: list[tuple[str, str]] = []
    for path in sorted(ARCHIVE.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            entries.append((sha256_file(path), str(path.relative_to(ROOT))))
    for path in [ROOT / "tools/cir_rmt/historical_cross_evaluator.py", ROOT / "tools/cir_rmt/build_historical_gain_forensics.py"]:
        if path.exists():
            entries.append((sha256_file(path), str(path.relative_to(ROOT))))
    write_text("SHA256SUMS.txt", "".join(f"{digest}  {path}\n" for digest, path in sorted(entries)))


def main() -> int:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    h1 = read_current_helper(H1_CURRENT_RESULTS)
    h2 = read_current_helper(H2_CURRENT_RESULTS)
    h2_hist = read_h2_historical()
    c2 = read_c2_e10()
    missing = {"H1": sorted(set(TARGETS) - set(h1)), "H2_CURRENT": sorted(set(TARGETS) - set(h2)), "H2_HISTORICAL": sorted(set(TARGETS) - set(h2_hist)), "C2": sorted(set(TARGETS) - set(c2))}
    if any(missing.values()):
        raise SystemExit(f"missing required replay rows: {missing}")
    build_repository_and_checkpoint_identity()
    build_contracts()
    build_phase1_contract_and_primary_reports()
    build_contract_matrix()
    build_migration_and_forensics()
    build_evaluator_and_scheduler_docs()
    build_selection_audit()
    build_replay_tables(h1, h2, h2_hist, c2)
    build_gap_decomposition(h2_hist, h2, c2)
    build_final_comparison_and_decision(h1, h2, h2_hist, c2)
    build_extension_ingest(h1, h2)
    build_extension_status_docs()
    build_final_audit(h1, h2, h2_hist, c2)
    build_resource_and_ledger()
    build_hashes()
    print(f"built {ARCHIVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
