#!/usr/bin/env python3
"""One inference-only capture of a bounded, stratified Router-input sample.

The earlier E3 cache omitted router_input_features/queries/keys, preventing a
valid offline separability diagnostic.  This script performs one deterministic
full forward solely to capture those missing tensors for a small fixed sample;
it constructs no optimizer and never calls backward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import get_text_and_image_dataset
from model.h6.utility_routing import utility_teacher
from tools.audit_p1_v83_semantics import _model_from_checkpoint
from tools.audit_p1_v84a_post300 import _IndexedDataset, _seed, _state_hash
from utils import make_dataloader_generator, seed_worker


RHO = 0.05
SCALE = 0.0005203147302381694
STRATA = (
    ("normal_teacher_role0", 0, 0),
    ("normal_teacher_role1", 0, 1),
    ("anomaly_teacher_role0", 1, 0),
    ("anomaly_teacher_role1", 1, 1),
)


def _selected_coordinates(cache: dict, per_stratum: int) -> tuple[dict[int, list[tuple[int, int, int]]], dict[str, int]]:
    residual = torch.stack(cache["residual"]).squeeze(2).float()
    z0 = torch.stack(cache["z0"]).squeeze(2).float()
    target = torch.stack([value.squeeze(0) for value in cache["target"]]).float()
    valid = torch.stack([value.squeeze(0) for value in cache["valid"]]).bool()
    teacher = utility_teacher(
        z0.permute(1, 0, 2), residual.permute(1, 0, 2, 3), target, valid,
        rho=RHO, router_confidence_mode="margin_rel",
        router_margin_rel_threshold=0.10, router_target_mode="patch_zscore_softmax",
        role_topology="r2_normal_anomaly", role_teacher_scale=SCALE,
    )
    q = teacher["q_router_utility"].permute(1, 0, 2, 3)
    informative = teacher["informative"].permute(1, 0, 2)
    hard = q.argmax(dim=-1)
    target_g = target[:, None, :].expand_as(hard)
    selected: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    counts: dict[str, int] = {}
    for name, anomaly, role in STRATA:
        mask = informative & ((target_g >= 0.5) if anomaly else (target_g < 0.5)) & (hard == role)
        coords = torch.nonzero(mask, as_tuple=False)  # [image, group, patch]
        # Deterministic order independent of CUDA/runtime scheduling.
        key = torch.remainder(
            coords[:, 0].long() * 1_103_515_245
            + coords[:, 1].long() * 12_345
            + coords[:, 2].long() * 97_531,
            2_147_483_647,
        )
        take = coords[torch.argsort(key)[: min(int(per_stratum), coords.shape[0])]]
        counts[name] = int(take.shape[0])
        for image, group, patch in take.tolist():
            selected[int(image)].append((int(group), int(patch), int(anomaly)))
    return selected, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-stratum", type=int, default=8000)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this fixed audit environment")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _seed(args.seed)

    source_bytes = args.source_cache.read_bytes()
    source_cache = torch.load(args.source_cache, map_location="cpu", weights_only=False)
    selected, stratum_counts = _selected_coordinates(source_cache, args.per_stratum)
    expected_rows = sum(stratum_counts.values())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("h6_config", {})
    required = {
        "r2_topology": config.get("role_topology") == "r2_normal_anomaly",
        "two_roles": config.get("num_factors") == 2,
        "fixed_scale": abs(float(config.get("role_teacher_scale", 0.0)) - SCALE) < 1e-15,
        "rho_fixed": config.get("rho_fixed") is True,
        "seed": checkpoint.get("seed") == args.seed,
    }
    if not all(required.values()):
        raise RuntimeError(f"checkpoint contract failure: {[k for k, v in required.items() if not v]}")

    device = torch.device("cuda:0")
    model = _model_from_checkpoint(checkpoint, device)
    model.requires_grad_(False)
    model.eval()
    model.clipmodel.eval()
    before = _state_hash(model)
    grads_before = all(parameter.grad is None for parameter in model.parameters())
    dataset = _IndexedDataset(get_text_and_image_dataset("VisA", 518, "train"))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
        worker_init_fn=seed_worker, generator=make_dataloader_generator(args.seed),
    )
    features, queries, probabilities, logits = [], [], [], []
    images, groups, patches, regions, teacher_roles, class_names = [], [], [], [], [], []
    key_snapshot = None
    residual_error = dense_error = rho_error = 0.0
    started = time.monotonic()
    for batch_number, sample in enumerate(loader, 1):
        image_index = int(sample["dataset_index"].item())
        image = sample["image"].to(device, non_blocking=True)
        cls = [str(sample["class_name"][0])]
        with torch.inference_mode():
            visual = model(image, return_phase4_features=True)
            h6 = model.h6.build_batch(
                model, "VisA", cls, visual,
                hybrid_alpha=float(checkpoint["hybrid_alpha_current"]),
                update_load_bias=False,
            )
            residual = h6["factor_residual_logits"].float()
            factor_abs = h6["factor_patch_logits"].float()
            dense = h6["prediction_probabilities"].float()
            act = h6["act_probability"].float()
            residual_error = max(
                residual_error,
                float((residual - (factor_abs - h6["noop_reference_logit"].float().unsqueeze(-1))).abs().max()),
            )
            dense_error = max(
                dense_error,
                float((h6["h6_logits"].float() - act * (dense * residual).sum(-1)).abs().max()),
            )
            rho_error = max(
                rho_error,
                float((h6["rho_scaled_actual_correction"].float() - h6["rho"].float().view(3, 1, 1) * h6["h6_logits"].float()).abs().max()),
            )
            if key_snapshot is None:
                key_snapshot = h6["final_router_keys"].detach().float().cpu()
            for group, patch, region in selected.get(image_index, []):
                features.append(h6["router_patch_features"][group, 0, patch].detach().to(torch.float16).cpu())
                queries.append(h6["queries"][group, 0, patch].detach().to(torch.float16).cpu())
                probabilities.append(dense[group, 0, patch].detach().to(torch.float32).cpu())
                logits.append(h6["prediction_logits"][group, 0, patch].detach().to(torch.float32).cpu())
                images.append(image_index)
                groups.append(group)
                patches.append(patch)
                regions.append(region)
                # Reconstruct this selected row's role deterministically from
                # the source cache's teacher rather than trusting any GT input.
                teacher_roles.append(0)
                class_names.append(cls[0])
        if batch_number % args.progress_every == 0:
            print(json.dumps({"images": batch_number, "total": len(dataset), "rows": len(features), "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)

    # Restore hard teacher role for retained rows from the original cache.
    residual = torch.stack(source_cache["residual"]).squeeze(2).float()
    z0 = torch.stack(source_cache["z0"]).squeeze(2).float()
    target = torch.stack([value.squeeze(0) for value in source_cache["target"]]).float()
    valid = torch.stack([value.squeeze(0) for value in source_cache["valid"]]).bool()
    teacher = utility_teacher(
        z0.permute(1, 0, 2), residual.permute(1, 0, 2, 3), target, valid,
        rho=RHO, router_confidence_mode="margin_rel", router_margin_rel_threshold=0.10,
        router_target_mode="patch_zscore_softmax", role_topology="r2_normal_anomaly", role_teacher_scale=SCALE,
    )
    hard = teacher["q_router_utility"].permute(1, 0, 2, 3).argmax(dim=-1)
    retained_roles = torch.tensor([hard[i, g, p].item() for i, g, p in zip(images, groups, patches)], dtype=torch.long)
    if len(features) != expected_rows:
        raise RuntimeError(f"representation capture expected {expected_rows} rows, got {len(features)}")
    after = _state_hash(model)
    grads_after = all(parameter.grad is None for parameter in model.parameters())
    payload = {
        "router_input_features": torch.stack(features),
        "queries": torch.stack(queries),
        "router_probabilities": torch.stack(probabilities),
        "router_logits": torch.stack(logits),
        "teacher_hard_role": retained_roles,
        "physical_region": torch.tensor(regions, dtype=torch.long),
        "image_index": torch.tensor(images, dtype=torch.long),
        "group_index": torch.tensor(groups, dtype=torch.long),
        "patch_index": torch.tensor(patches, dtype=torch.long),
        "class_name": class_names,
        "final_router_keys": key_snapshot,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report = {
        "status": "PASS",
        "audit": "P1_V84A_ROUTER_REPRESENTATION_CAPTURE",
        "source_cache": str(args.source_cache.resolve()),
        "source_cache_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "contract": required,
        "images": len(dataset),
        "sample_rows": len(features),
        "per_stratum_requested": args.per_stratum,
        "per_stratum_captured": stratum_counts,
        "feature_shape": list(payload["router_input_features"].shape),
        "query_shape": list(payload["queries"].shape),
        "router_key_shape": list(payload["final_router_keys"].shape),
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_state_unchanged": before == after,
        "all_grads_none_before": grads_before,
        "all_grads_none_after": grads_after,
        "invariants": {
            "true_residual_max_abs_error": residual_error,
            "dense_reconstruction_max_abs_error": dense_error,
            "rho_scaled_reconstruction_max_abs_error": rho_error,
        },
        "runtime_seconds": time.monotonic() - started,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "rows": len(features), "feature_shape": report["feature_shape"], "runtime_seconds": round(report["runtime_seconds"], 1)}), flush=True)
    if not (before == after and grads_before and grads_after and residual_error == dense_error == rho_error == 0.0):
        raise RuntimeError("representation capture invariant failure")


if __name__ == "__main__":
    main()
