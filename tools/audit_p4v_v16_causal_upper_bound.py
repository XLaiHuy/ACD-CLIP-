#!/usr/bin/env python3
"""Inference-only V1.6 protocol-validity and no-spatial-gate audit."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from audit_p4_k1_oracle_utility import DeterministicVisATrainDataset, _sha256
from run_p4v_short64 import _pixel_metrics, _mean_defined, build
from utils import configure_canonical_fp32, get_phase2b_global_text_features

EPS = 1e-6
QUANTILES = (0.05, 0.50, 0.95)


def _load(config: dict, path: Path, device: torch.device):
    model = build(config, device)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.image_adapter.load_state_dict(state["image_adapter"])
    model.text_adapter.load_state_dict(state["text_adapter"])
    model.h6.load_state_dict(state["h6"])
    return model


def _summary(values: torch.Tensor) -> dict:
    values = values.detach().float().flatten().cpu()
    if not values.numel():
        return {"count": 0, "mean": None, "std": None, "p05": None, "p50": None, "p95": None}
    q = torch.quantile(values, torch.tensor(QUANTILES))
    return {"count": int(values.numel()), "mean": float(values.mean()), "std": float(values.std(unbiased=False)), "p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2])}


def _dataset_sha(manifest: dict) -> str:
    payload = json.dumps(manifest["samples"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _predict_from_features(model, features: torch.Tensor, text: torch.Tensor, img_size: int):
    """Reconstruct the predictor's pre-softmax mean group logits exactly."""
    group_text = text.permute(1, 0, 2, 3)
    logits = []
    for group in range(model.n_groups):
        weights = model.compute_dfg_weights(features[group], group_text, group)
        anchors = model.apply_dfg_weights(group_text, weights["normal"], weights["abnormal"])
        raw = (10.0 * features[group]).matmul(anchors)
        side = math.isqrt(raw.shape[1])
        logits.append(F.interpolate(raw.permute(0, 2, 1).view(raw.shape[0], 2, side, side), size=img_size, mode="bilinear", align_corners=True))
    final_logits = torch.stack(logits).mean(0)
    return F.softmax(final_logits, dim=1), final_logits


def _forward(model, image, class_name: str, config: dict, mode: str):
    visual = model(image, return_phase4_features=True)
    original = torch.stack(visual["seg_tokens"])
    text = get_phase2b_global_text_features(model, "VisA", [class_name], image.device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
    base_model_pred, base_group_logits, _ = model.vision_text_fusion_gate_seg(original, text, img_size=config["img_size"], return_details=True)
    base_pred, base_logits = _predict_from_features(model, original, text, config["img_size"])
    if mode == "OFF":
        return base_pred, base_logits, {"predictor_reconstruction_max_abs_error": float((base_pred - base_model_pred).abs().max())}
    state = model.h6.phase4v_state_code(model, visual)["semantic_code"]
    gate = torch.softmax(base_group_logits.float(), dim=-1)[..., 1].detach()
    no_spatial_gate = mode == "NO_GATE"
    outputs = [model.h6.phase4v_adapt(original[group], state, gate[group], enabled=True, semantic_conditioning=True, spatial_gating=not no_spatial_gate) for group in range(model.n_groups)]
    pred, logits = _predict_from_features(model, torch.stack([out["adapted"] for out in outputs]), text, config["img_size"])
    geometry = defaultdict(list)
    for out in outputs:
        geometry["raw_delta_norm"].append(float(out["delta_v"].float().norm(dim=-1).mean()))
        geometry["correction_norm"].append(float(out["correction"].float().norm(dim=-1).mean()))
        geometry["gate_mean"].append(float(out["gate"].float().mean()))
    return pred, logits, {"predictor_reconstruction_max_abs_error": float((base_pred - base_model_pred).abs().max()), "spatial_gating": not no_spatial_gate, **{key: _mean_defined(value) for key, value in geometry.items()}}


@torch.inference_mode()
def _evaluate(model, dataset, config: dict, mode: str):
    model.eval(); model.clipmodel.eval()
    values = {region: defaultdict(list) for region in ("normal", "anomaly")}
    per_image, aps, aucs, reconstruction = [], [], [], []
    geometry_values = defaultdict(list)
    for index in range(len(dataset)):
        raw = dataset[index]
        image = raw["image"].unsqueeze(0).to(next(model.parameters()).device).float()
        target = raw["mask"].unsqueeze(0).to(image.device).float()[:, 0]
        prob, logits, geometry = _forward(model, image, str(raw["class_name"]), config, mode)
        for key, value in geometry.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                geometry_values[key].append(float(value))
        reconstruction.append(geometry["predictor_reconstruction_max_abs_error"])
        z_normal, z_abnormal = logits[:, 0], logits[:, 1]
        margin = z_abnormal - z_normal
        old_bce = F.binary_cross_entropy(prob[:, 1].clamp(EPS, 1.0 - EPS), target, reduction="none")
        stable = F.binary_cross_entropy_with_logits(margin, target, reduction="none")
        ap, auc = _pixel_metrics(target, prob[:, 1])
        aps.append(ap); aucs.append(auc)
        record = {"image_index": index, "image_path": str(raw["file_name"]), "file_name": Path(str(raw["file_name"])).name, "class_name": str(raw["class_name"]), "label": int(raw["label"].item()), "pixel_ap": ap, "pixel_auc": auc}
        for region, selector in (("normal", target < .5), ("anomaly", target >= .5)):
            if not selector.any():
                continue
            for key, tensor in (("probability", prob[:, 1]), ("z_normal", z_normal), ("z_abnormal", z_abnormal), ("margin", margin), ("probability_bce", old_bce), ("stable_logistic_loss", stable)):
                selected = tensor[selector]
                values[region][key].append(selected.detach().cpu())
                record[f"{region}_{key}_mean"] = float(selected.float().mean())
            values[region]["floor"].append((prob[:, 1][selector] <= EPS).detach().cpu())
            values[region]["ceiling"].append((prob[:, 1][selector] >= 1.0 - EPS).detach().cpu())
        record["geometry"] = geometry
        per_image.append(record)
    regions = {}
    for region, region_values in values.items():
        report = {key: _summary(torch.cat(tensors)) for key, tensors in region_values.items() if key not in {"floor", "ceiling"}}
        report["probability_floor_fraction"] = None if not region_values["floor"] else float(torch.cat(region_values["floor"]).float().mean())
        report["probability_ceiling_fraction"] = None if not region_values["ceiling"] else float(torch.cat(region_values["ceiling"]).float().mean())
        regions[region] = report
    return {"mode": mode, "pixel_ap_macro": _mean_defined(aps), "pixel_auc_macro": _mean_defined(aucs), "regions": regions, "geometry": {key: _summary(torch.tensor(rows)) for key, rows in geometry_values.items()}, "per_image": per_image, "predictor_reconstruction_max_abs_error": max(reconstruction)}


def _difference(left: dict, right: dict):
    return {key: left[key] - right[key] for key in ("pixel_ap_macro", "pixel_auc_macro")}


def main():
    config_path = ROOT / "runs/phase4/k1/short64_seed0_attempt5/config.json"
    manifest_path = ROOT / "runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"
    output_path = ROOT / "runs/phase4v/v1_6/V1_6_CAUSAL_UPPER_BOUND.json"
    config, manifest = json.loads(config_path.read_text()), json.loads(manifest_path.read_text())
    configure_canonical_fp32(); device = torch.device("cuda:0")
    dataset = DeterministicVisATrainDataset(manifest, config["img_size"])
    checkpoints = {
        "V1_5_BASE": ROOT / "runs/phase4v/v1_5/paired/base_adapter_state.pth",
        "V1_5_CURRENT_V1": ROOT / "runs/phase4v/v1_5/paired/current_v1_adapter_state.pth",
        "V1A_BASE": ROOT / "runs/phase4v/v1a/paired/base_adapter_state.pth",
        "V1A_CANDIDATE": ROOT / "runs/phase4v/v1a/paired/v1a_adapter_state.pth",
    }
    reports = {}
    for name, path in checkpoints.items():
        model = _load(config, path, device)
        modes = ("OFF",) if name.endswith("BASE") else ("OFF", "ACTIVE", "NO_GATE")
        reports[name] = {mode: _evaluate(model, dataset, config, mode) for mode in modes}
        del model; torch.cuda.empty_cache()
    audit_identity = {}
    for name, checkpoint_report in reports.items():
        if "ACTIVE" not in checkpoint_report:
            continue
        active_correction = checkpoint_report["ACTIVE"]["geometry"]["correction_norm"]["mean"]
        no_gate_correction = checkpoint_report["NO_GATE"]["geometry"]["correction_norm"]["mean"]
        audit_identity[name] = {"active_minus_off": _difference(checkpoint_report["ACTIVE"], checkpoint_report["OFF"]), "no_gate_minus_off": _difference(checkpoint_report["NO_GATE"], checkpoint_report["OFF"]), "active_correction_norm": active_correction, "no_gate_correction_norm": no_gate_correction, "no_gate_to_active_correction_norm_ratio": no_gate_correction / max(active_correction, 1e-30)}
    base = reports["V1A_BASE"]["OFF"]["regions"]["anomaly"]
    exact_floor_degenerate = base["probability_floor_fraction"] == 1.0
    protocol = {"training_dataset": {"size": len(manifest["samples"]), "samples_sha256": _dataset_sha(manifest), "manifest_sha256": _sha256(manifest_path), "cycles": {"V1_5": 88, "V1A": 88}}, "audit_dataset": {"size": len(dataset), "samples_sha256": _dataset_sha(manifest), "manifest_sha256": _sha256(manifest_path)}, "paired_probe_is_audit_subset_training": True, "base_anomaly_prediction_exact_floor_degenerate": exact_floor_degenerate, "health": "UNHEALTHY_BASE_ANOMALY_PROBABILITY_FLOOR_DEGENERACY" if exact_floor_degenerate else "NOT_PROVEN_HEALTHY_BY_THIS_AUDIT"}
    decision = "V1A_TERMINAL_REQUIRES_CAUSAL_UPPER_BOUND_AUDIT"
    report = {"decision": decision, "inference_only": True, "historical_decisions_preserved": ["V1_HARM_DOMINATED_BY_OPTIMIZATION_DRIFT", "PHASE4V_OPTIMIZATION_RECOVERY_NOT_SUPPORTED"], "provenance": {"remote_head_pin": "ab635eea93ff67b611c4e5de713ed5cff8bac83f", "script_sha256": _sha256(Path(__file__).resolve()), "config_sha256": _sha256(config_path), "precision": "strict FP32; TF32 off; AMP off", "optimizer_steps": 0, "checkpoints": {name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for name, path in checkpoints.items()}}, "protocol_validity": protocol, "reports": reports, "causal_upper_bound": audit_identity, "interpretation": "NO_GATE sets spatial_gating=False, which makes VisualAdapter use an all-ones effective gate while retaining the fixed lambda. Probability BCE is retained for continuity; stable_logistic_loss is BCEWithLogitsLoss(z_abnormal-z_normal, target)."}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "protocol": protocol, "causal_upper_bound": audit_identity}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
