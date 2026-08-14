#!/usr/bin/env python3
"""Inference-only V1.7 same-checkpoint WHERE counterfactual audit."""
from __future__ import annotations
import argparse, json, math, sys
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
MODES = ("OFF", "NO_GATE", "CURRENT_GATE", "ORACLE_REGION", "ORACLE_UTILITY")


def load_model(config: dict, path: Path, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build(config, device)
    model.image_adapter.load_state_dict(state["image_adapter"])
    model.text_adapter.load_state_dict(state["text_adapter"])
    if "soft_prompt" in state:
        model.soft_prompt.load_state_dict(state["soft_prompt"])
    model.h6.load_state_dict(state["h6"])
    model.eval(); model.clipmodel.eval()
    return model


def group_logits(model, features, text, img_size):
    group_text = text.permute(1, 0, 2, 3)
    patch_logits = []
    image_logits = []
    for group in range(model.n_groups):
        weights = model.compute_dfg_weights(features[group], group_text, group)
        anchors = model.apply_dfg_weights(group_text, weights["normal"], weights["abnormal"])
        raw = (10.0 * features[group]).matmul(anchors)
        side = math.isqrt(raw.shape[1])
        patch_logits.append(raw.permute(0, 2, 1))
        image_logits.append(F.interpolate(raw.permute(0, 2, 1).view(raw.shape[0], 2, side, side), size=img_size, mode="bilinear", align_corners=True))
    return patch_logits, torch.stack(image_logits).mean(0)


def gate_for_mode(model, mode, original, text, base_group_logits, visual, target):
    if mode == "NO_GATE":
        return [torch.ones(original[group].shape[:2], device=original.device, dtype=original.dtype) for group in range(model.n_groups)]
    if mode == "CURRENT_GATE":
        return [torch.softmax(base_group_logits.float(), dim=-1)[group, ..., 1].detach() for group in range(model.n_groups)]
    if mode == "ORACLE_REGION":
        return [F.adaptive_avg_pool2d(target.unsqueeze(1), (math.isqrt(original[group].shape[1]), math.isqrt(original[group].shape[1]))).flatten(1).detach() for group in range(model.n_groups)]
    if mode != "ORACLE_UTILITY":
        raise ValueError(mode)
    state = model.h6.phase4v_state_code(model, visual)["semantic_code"]
    base_patch, _ = group_logits(model, original, text, target.shape[-1])
    full = [model.h6.phase4v_adapt(original[group], state, torch.ones(original[group].shape[:2], device=original.device), enabled=True, semantic_conditioning=True, spatial_gating=False) for group in range(model.n_groups)]
    corrected = torch.stack([out["adapted"] for out in full])
    corrected_patch, _ = group_logits(model, corrected, text, target.shape[-1])
    gates = []
    for group in range(model.n_groups):
        side = math.isqrt(original[group].shape[1])
        patch_target = F.adaptive_avg_pool2d(target.unsqueeze(1), (side, side)).flatten(1)
        base_margin = base_patch[group][:, 1] - base_patch[group][:, 0]
        corrected_margin = corrected_patch[group][:, 1] - corrected_patch[group][:, 0]
        base_loss = F.binary_cross_entropy_with_logits(base_margin, patch_target, reduction="none")
        corrected_loss = F.binary_cross_entropy_with_logits(corrected_margin, patch_target, reduction="none")
        gates.append((corrected_loss < base_loss).float().detach())
    return gates


@torch.inference_mode()
def forward_mode(model, image, target, class_name, config, mode):
    visual = model(image, return_phase4_features=True)
    original = torch.stack(visual["seg_tokens"])
    text = get_phase2b_global_text_features(model, "VisA", [class_name], image.device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
    base_prob, base_group_logits, _ = model.vision_text_fusion_gate_seg(original, text, img_size=config["img_size"], return_details=True)
    _, base_logits = _predict_from_features(model, original, text, config["img_size"])
    if mode == "OFF":
        return base_prob, base_logits, {"gate_mean": 0.0, "raw_delta_norm": 0.0, "correction_norm": 0.0}
    state = model.h6.phase4v_state_code(model, visual)["semantic_code"]
    gates = gate_for_mode(model, mode, original, text, base_group_logits, visual, target)
    outputs = [model.h6.phase4v_adapt(original[group], state, gates[group], enabled=True, semantic_conditioning=True, spatial_gating=True) for group in range(model.n_groups)]
    adapted = torch.stack([out["adapted"] for out in outputs])
    _, logits = _predict_from_features(model, adapted, text, config["img_size"])
    return F.softmax(logits, dim=1), logits, {"gate_mean": float(torch.stack([out["gate"].float().mean() for out in outputs]).mean()), "raw_delta_norm": float(torch.stack([out["delta_v"].float().norm(dim=-1).mean() for out in outputs]).mean()), "correction_norm": float(torch.stack([out["correction"].float().norm(dim=-1).mean() for out in outputs]).mean())}


@torch.inference_mode()
def evaluate(model, dataset, config, mode):
    values = {region: defaultdict(list) for region in ("normal", "anomaly")}
    aps, aucs, per_image, geometry = [], [], [], defaultdict(list)
    for index in range(len(dataset)):
        raw = dataset[index]
        device = next(model.parameters()).device
        image = raw["image"].unsqueeze(0).to(device).float()
        target = raw["mask"].unsqueeze(0).to(device).float()[:, 0]
        prob, logits, geom = forward_mode(model, image, target, str(raw["class_name"]), config, mode)
        for key, value in geom.items(): geometry[key].append(value)
        score = prob[:, 1]; margin = logits[:, 1] - logits[:, 0]
        stable = F.binary_cross_entropy_with_logits(margin, target, reduction="none")
        old_bce = F.binary_cross_entropy(score.clamp(EPS, 1.0 - EPS), target, reduction="none")
        ap, auc = _pixel_metrics(target, score); aps.append(ap); aucs.append(auc)
        row = {"image_index": index, "image_path": str(raw["file_name"]), "file_name": Path(str(raw["file_name"])).name, "class_name": str(raw["class_name"]), "label": int(raw["label"].item()), "pixel_ap": ap, "pixel_auc": auc, "geometry": geom}
        for region, selector in (("normal", target < .5), ("anomaly", target >= .5)):
            if not selector.any(): continue
            for key, tensor in (("probability", score), ("margin", margin), ("probability_bce", old_bce), ("stable_logistic_loss", stable)):
                selected = tensor[selector]
                values[region][key].append(selected.detach().cpu())
                row[f"{region}_{key}_mean"] = float(selected.float().mean())
            values[region]["floor"].append((score[selector] <= EPS).detach().cpu())
            values[region]["ceiling"].append((score[selector] >= 1.0 - EPS).detach().cpu())
        per_image.append(row)
    regions = {}
    for region, collected in values.items():
        report = {key: _summary(torch.cat(rows)) for key, rows in collected.items() if key not in {"floor", "ceiling"}}
        report["probability_floor_fraction"] = float(torch.cat(collected["floor"]).float().mean())
        report["probability_ceiling_fraction"] = float(torch.cat(collected["ceiling"]).float().mean())
        regions[region] = report
    return {"mode": mode, "pixel_ap_macro": _mean_defined(aps), "pixel_auc_macro": _mean_defined(aucs), "regions": regions, "geometry": {key: _summary(torch.tensor(rows)) for key, rows in geometry.items()}, "per_image": per_image}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("runs/phase4/k1/short64_seed0_attempt5/config.json"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"))
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--how-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); configure_canonical_fp32(); config, manifest = json.loads(args.config.read_text()), json.loads(args.manifest.read_text()); device = torch.device("cuda:0"); dataset = DeterministicVisATrainDataset(manifest, config["img_size"])
    reports = {}
    base = load_model(config, args.base_checkpoint, device); reports["BASE"] = {"OFF": evaluate(base, dataset, config, "OFF")}; del base; torch.cuda.empty_cache()
    how = load_model(config, args.how_checkpoint, device); reports["HOW"] = {mode: evaluate(how, dataset, config, mode) for mode in MODES}; del how; torch.cuda.empty_cache()
    report = {"decision": "V1_7_COUNTERFACTUAL_AUDIT_COMPLETE", "inference_only": True, "provenance": {"config_sha256": _sha256(args.config), "manifest_sha256": _sha256(args.manifest), "base_checkpoint": str(args.base_checkpoint), "base_checkpoint_sha256": _sha256(args.base_checkpoint), "how_checkpoint": str(args.how_checkpoint), "how_checkpoint_sha256": _sha256(args.how_checkpoint), "precision": "strict FP32; TF32 off; AMP off", "optimizer_steps": 0, "modes": MODES}, "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {mode: {key: value for key, value in metrics.items() if key in {"pixel_ap_macro", "pixel_auc_macro", "geometry", "regions"}} for mode, metrics in modes.items()} for name, modes in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
