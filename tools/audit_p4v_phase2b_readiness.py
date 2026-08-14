#!/usr/bin/env python3
"""Inference-only Phase2B epoch-readiness audit on the fixed 48-image probe."""
from __future__ import annotations

import argparse
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
from audit_p4v_v16_causal_upper_bound import _predict_from_features, _summary
from run_p4v_short64 import _mean_defined, _pixel_metrics, build
from utils import configure_canonical_fp32, get_phase2b_global_text_features

EPS = 1e-6


def load_model(config: dict, path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build(config, device)
    model.image_adapter.load_state_dict(checkpoint["image_adapter"])
    model.text_adapter.load_state_dict(checkpoint["text_adapter"])
    if "soft_prompt" in checkpoint:
        model.soft_prompt.load_state_dict(checkpoint["soft_prompt"])
    model.eval(); model.clipmodel.eval()
    return model, checkpoint


@torch.inference_mode()
def evaluate(model, dataset, config: dict):
    values = {region: defaultdict(list) for region in ("normal", "anomaly")}
    aps, aucs, per_image, reconstruction = [], [], [], []
    for index in range(len(dataset)):
        raw = dataset[index]
        image = raw["image"].unsqueeze(0).to(next(model.parameters()).device).float()
        target = raw["mask"].unsqueeze(0).to(image.device).float()[:, 0]
        visual = model(image, return_phase4_features=True)
        features = torch.stack(visual["seg_tokens"])
        text = get_phase2b_global_text_features(model, "VisA", [str(raw["class_name"])], image.device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
        model_prob, _, _ = model.vision_text_fusion_gate_seg(features, text, img_size=config["img_size"], return_details=True)
        prob, logits = _predict_from_features(model, features, text, config["img_size"])
        reconstruction.append(float((model_prob - prob).abs().max()))
        z_normal, z_abnormal = logits[:, 0], logits[:, 1]
        margin = z_abnormal - z_normal
        old_bce = F.binary_cross_entropy(prob[:, 1].clamp(EPS, 1.0 - EPS), target, reduction="none")
        stable = F.binary_cross_entropy_with_logits(margin, target, reduction="none")
        ap, auc = _pixel_metrics(target, prob[:, 1]); aps.append(ap); aucs.append(auc)
        row = {"image_index": index, "image_path": str(raw["file_name"]), "file_name": Path(str(raw["file_name"])).name, "class_name": str(raw["class_name"]), "label": int(raw["label"].item()), "pixel_ap": ap, "pixel_auc": auc}
        for region, selector in (("normal", target < .5), ("anomaly", target >= .5)):
            if not selector.any():
                continue
            for key, tensor in (("probability", prob[:, 1]), ("z_normal", z_normal), ("z_abnormal", z_abnormal), ("margin", margin), ("probability_bce", old_bce), ("stable_logistic_loss", stable)):
                selected = tensor[selector]
                values[region][key].append(selected.detach().cpu())
                row[f"{region}_{key}_mean"] = float(selected.float().mean())
            values[region]["floor"].append((prob[:, 1][selector] <= EPS).detach().cpu())
            values[region]["ceiling"].append((prob[:, 1][selector] >= 1.0 - EPS).detach().cpu())
        per_image.append(row)
    regions = {}
    for region, collected in values.items():
        report = {key: _summary(torch.cat(rows)) for key, rows in collected.items() if key not in {"floor", "ceiling"}}
        report["probability_floor_fraction"] = float(torch.cat(collected["floor"]).float().mean())
        report["probability_ceiling_fraction"] = float(torch.cat(collected["ceiling"]).float().mean())
        regions[region] = report
    return {"pixel_ap_macro": _mean_defined(aps), "pixel_auc_macro": _mean_defined(aucs), "regions": regions, "per_image": per_image, "predictor_reconstruction_max_abs_error": max(reconstruction)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("runs/phase4/k1/short64_seed0_attempt5/config.json"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_canonical_fp32(); config, manifest = json.loads(args.config.read_text()), json.loads(args.manifest.read_text())
    model, checkpoint = load_model(config, args.checkpoint, torch.device("cuda:0"))
    report = {"decision": "PHASE2B_READINESS_AUDIT_COMPLETE", "inference_only": True, "provenance": {"checkpoint": str(args.checkpoint), "checkpoint_sha256": _sha256(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"), "config_sha256": _sha256(args.config), "manifest_sha256": _sha256(args.manifest), "precision": "strict FP32; TF32 off; AMP off", "optimizer_steps": 0}, "metrics": evaluate(model, DeterministicVisATrainDataset(manifest, config["img_size"]), config)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checkpoint_epoch": report["provenance"]["checkpoint_epoch"], "pixel_ap_macro": report["metrics"]["pixel_ap_macro"], "pixel_auc_macro": report["metrics"]["pixel_auc_macro"], "anomaly_floor_fraction": report["metrics"]["regions"]["anomaly"]["probability_floor_fraction"], "anomaly_margin": report["metrics"]["regions"]["anomaly"]["margin"]}, indent=2))


if __name__ == "__main__":
    main()
