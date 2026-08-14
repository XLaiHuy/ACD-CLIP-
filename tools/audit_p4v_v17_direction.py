#!/usr/bin/env python3
"""Inference-only V1.7 offline residual-direction attribution."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from audit_p4_k1_oracle_utility import DeterministicVisATrainDataset, _sha256
from audit_p4v_v16_causal_upper_bound import _predict_from_features, _summary
from audit_p4v_v17_gate_counterfactual import group_logits
from run_p4v_short64 import _mean_defined, _pixel_metrics, build
from utils import configure_canonical_fp32, get_phase2b_global_text_features

DIRECTIONS = ("RAW", "TANGENT_RAW", "PREDICTOR_ALIGNED_TANGENT", "EXISTING_BASE_RESIDUAL")
EPS = 1e-6


def load_model(config, path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build(config, device)
    model.image_adapter.load_state_dict(state["image_adapter"])
    model.text_adapter.load_state_dict(state["text_adapter"])
    model.soft_prompt.load_state_dict(state["soft_prompt"])
    model.h6.load_state_dict(state["h6"])
    model.eval(); model.clipmodel.eval()
    return model


def energy_match(direction, reference):
    norm = direction.float().norm(dim=-1, keepdim=True)
    ref_norm = reference.float().norm(dim=-1, keepdim=True)
    return direction * (ref_norm / norm.clamp_min(1e-12))


def install_hooks(model):
    cache = defaultdict(dict); handles = []
    for group in range(model.n_groups):
        module = model.image_adapter["m_i_w"][group]
        def pre(_module, inputs, group=group): cache[group]["input"] = inputs[0].detach()
        def post(_module, _inputs, output, group=group): cache[group]["output"] = output.detach()
        handles.extend([module.register_forward_pre_hook(pre), module.register_forward_hook(post)])
    return cache, handles


@torch.inference_mode()
def evaluate(model, dataset, config):
    records = {name: {region: [] for region in ("normal", "anomaly")} for name in DIRECTIONS}
    aps = {name: [] for name in DIRECTIONS}; aucs = {name: [] for name in DIRECTIONS}; per_image = {name: [] for name in DIRECTIONS}; geometry = {name: [] for name in DIRECTIONS}
    for index in range(len(dataset)):
        raw = dataset[index]; device = next(model.parameters()).device; image = raw["image"].unsqueeze(0).to(device).float(); target = raw["mask"].unsqueeze(0).to(device).float()[:, 0]
        cache, handles = install_hooks(model); visual = model(image, return_phase4_features=True)
        for handle in handles: handle.remove()
        original = torch.stack(visual["seg_tokens"])
        text = get_phase2b_global_text_features(model, "VisA", [str(raw["class_name"])], device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
        state = model.h6.phase4v_state_code(model, visual)["semantic_code"]
        ones = [torch.ones(original[group].shape[:2], device=device, dtype=original.dtype) for group in range(model.n_groups)]
        outputs = [model.h6.phase4v_adapt(original[group], state, ones[group], enabled=True, semantic_conditioning=True, spatial_gating=False) for group in range(model.n_groups)]
        raw_corr = torch.stack([out["correction"] for out in outputs])
        group_text = text.permute(1, 0, 2, 3)
        directions = {"RAW": raw_corr}
        tangent_groups = []; predictor_groups = []; existing_groups = []
        for group in range(model.n_groups):
            vhat = F.normalize(original[group].float(), dim=-1)
            tangent = raw_corr[group] - (raw_corr[group] * vhat).sum(dim=-1, keepdim=True) * vhat
            tangent_groups.append(energy_match(tangent, raw_corr[group]))
            weights = model.compute_dfg_weights(original[group], group_text, group)
            anchors = model.apply_dfg_weights(group_text, weights["normal"], weights["abnormal"])
            semantic_direction = (anchors[:, :, 1] - anchors[:, :, 0]).unsqueeze(1).expand_as(original[group])
            semantic_tangent = semantic_direction - (semantic_direction * vhat).sum(dim=-1, keepdim=True) * vhat
            predictor_groups.append(energy_match(semantic_tangent, raw_corr[group]))
            t = cache[group]["input"].permute(1, 0, 2)
            raw_proj = model.image_adapter["seg_proj"][group](t)
            raw_norm = model.image_adapter["seg_layer_norms"][group](raw_proj)
            raw_seg = F.normalize(raw_norm.float(), dim=-1)
            existing_groups.append(energy_match(original[group] - raw_seg, raw_corr[group]))
        directions["TANGENT_RAW"] = torch.stack(tangent_groups)
        directions["PREDICTOR_ALIGNED_TANGENT"] = torch.stack(predictor_groups)
        directions["EXISTING_BASE_RESIDUAL"] = torch.stack(existing_groups)
        for name, correction in directions.items():
            features = original + correction
            _, logits = _predict_from_features(model, features, text, config["img_size"])
            prob = F.softmax(logits, dim=1); score = prob[:, 1]; margin = logits[:, 1] - logits[:, 0]
            stable = F.binary_cross_entropy_with_logits(margin, target, reduction="none")
            old_bce = F.binary_cross_entropy(score.clamp(EPS, 1.0 - EPS), target, reduction="none")
            ap, auc = _pixel_metrics(target, score); aps[name].append(ap); aucs[name].append(auc)
            geometry[name].append(float(correction.float().norm(dim=-1).mean()))
            row = {"image_index": index, "class_name": str(raw["class_name"]), "file_name": Path(str(raw["file_name"])).name, "label": int(raw["label"].item()), "pixel_ap": ap, "pixel_auc": auc, "correction_norm": geometry[name][-1]}
            for region, selector in (("normal", target < .5), ("anomaly", target >= .5)):
                if not selector.any(): continue
                for key, tensor in (("stable_logistic_loss", stable), ("probability_bce", old_bce), ("margin", margin)):
                    values = tensor[selector].detach().cpu(); records[name][region].append((key, values)); row[f"{region}_{key}_mean"] = float(values.mean())
            per_image[name].append(row)
    reports = {}
    for name in DIRECTIONS:
        regions = {}
        for region in ("normal", "anomaly"):
            by_key = defaultdict(list)
            for key, values in records[name][region]: by_key[key].append(values)
            regions[region] = {key: _summary(torch.cat(values)) for key, values in by_key.items()}
        reports[name] = {"pixel_ap_macro": _mean_defined(aps[name]), "pixel_auc_macro": _mean_defined(aucs[name]), "regions": regions, "correction_norm": _summary(torch.tensor(geometry[name])), "per_image": per_image[name]}
    return reports


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=Path("runs/phase4/k1/short64_seed0_attempt5/config.json")); parser.add_argument("--manifest", type=Path, default=Path("runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json")); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); configure_canonical_fp32(); config, manifest = json.loads(args.config.read_text()), json.loads(args.manifest.read_text()); device = torch.device("cuda:0"); model = load_model(config, args.checkpoint, device); reports = evaluate(model, DeterministicVisATrainDataset(manifest, config["img_size"]), config)
    report = {"decision": "V1_7_DIRECTION_ATTRIBUTION_COMPLETE", "inference_only": True, "energy_budget": "per-patch correction norm matched to the learned ungated K1 correction", "directions": DIRECTIONS, "provenance": {"checkpoint": str(args.checkpoint), "checkpoint_sha256": _sha256(args.checkpoint), "config_sha256": _sha256(args.config), "manifest_sha256": _sha256(args.manifest), "precision": "strict FP32; TF32 off; AMP off", "optimizer_steps": 0}, "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {"pixel_ap_macro": x["pixel_ap_macro"], "pixel_auc_macro": x["pixel_auc_macro"], "normal_stable": x["regions"]["normal"]["stable_logistic_loss"]["mean"], "anomaly_stable": x["regions"]["anomaly"]["stable_logistic_loss"]["mean"], "correction_norm": x["correction_norm"]["mean"]} for name, x in reports.items()}, indent=2))


if __name__ == "__main__": main()
