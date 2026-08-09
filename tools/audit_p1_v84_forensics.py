#!/usr/bin/env python3
"""No-step P1-v8.3 forensic audit for the P1-v8.4 rescue decision.

The audit replays the canonical seed-0 VisA loader against the completed
corrected-300B checkpoint.  It constructs no optimizer and performs no
backward pass.  Current absolute factors, an explicit no-op, and factors
residualized against the exact local fusion-path no-op are compared on the
same valid patch support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset import get_text_and_image_dataset
from model.h6.utility_routing import build_patch_targets
from tools.audit_p1_v83_semantics import _model_from_checkpoint
from utils import get_phase2b_global_text_features, make_dataloader_generator, seed_worker


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    modules = (model.image_adapter, model.text_adapter, model.soft_prompt, model.h6)
    for module in modules:
        for name, value in sorted(module.state_dict().items()):
            cpu = value.detach().contiguous().cpu()
            digest.update(name.encode())
            digest.update(str(cpu.dtype).encode())
            digest.update(str(tuple(cpu.shape)).encode())
            digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class _IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        sample["dataset_index"] = torch.tensor(index, dtype=torch.int64)
        return sample


def _effective_rank(matrix: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(matrix.float())
    total = singular.sum()
    if float(total.item()) <= 1e-12:
        return 0.0
    probability = singular / total
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().item())


def _stage_metrics(values: torch.Tensor, factor_dim: int) -> dict[str, float]:
    moved = values.detach().float().movedim(factor_dim % values.ndim, 0)
    matrix = moved.reshape(moved.shape[0], -1)
    normalized = F.normalize(matrix, dim=-1)
    cosine = normalized @ normalized.T
    mask = torch.triu(
        torch.ones_like(cosine, dtype=torch.bool), diagonal=1
    )
    cos_values = cosine[mask]
    l2_values = torch.cdist(matrix, matrix)[mask]
    return {
        "pairwise_cosine_mean": float(cos_values.mean().item()),
        "pairwise_cosine_min": float(cos_values.min().item()),
        "pairwise_cosine_max": float(cos_values.max().item()),
        "pairwise_l2_mean": float(l2_values.mean().item()),
        "pairwise_l2_min": float(l2_values.min().item()),
        "pairwise_l2_max": float(l2_values.max().item()),
        "effective_rank": _effective_rank(matrix),
        "factorwise_std_mean": float(matrix.std(dim=0, unbiased=False).mean().item()),
    }


def _append_stage(
    accumulator: dict[str, dict[str, list[float]]],
    name: str,
    values: torch.Tensor,
    factor_dim: int,
) -> None:
    metrics = _stage_metrics(values, factor_dim)
    for key, value in metrics.items():
        accumulator[name][key].append(value)


def _summarize_stages(
    accumulator: dict[str, dict[str, list[float]]]
) -> dict[str, dict[str, float]]:
    return {
        stage: {
            metric: float(np.mean(values))
            for metric, values in metrics.items()
        }
        for stage, metrics in accumulator.items()
    }


def _correlation_summary(matrix: torch.Tensor) -> dict[str, Any]:
    """Pairwise Pearson correlation across patch observations [N,M]."""
    matrix = matrix.float()
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered
    scale = centered.square().sum(dim=0).sqrt()
    denominator = scale[:, None] * scale[None, :]
    correlation = covariance / denominator.clamp_min(1e-12)
    mask = torch.triu(torch.ones_like(correlation, dtype=torch.bool), diagonal=1)
    values = correlation[mask]
    return {
        "matrix": correlation.tolist(),
        "pairwise_mean": float(values.mean().item()),
        "pairwise_min": float(values.min().item()),
        "pairwise_max": float(values.max().item()),
    }


def _region_report(
    name: str,
    region: torch.Tensor,
    z0: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    absolute: torch.Tensor,
    residual: torch.Tensor,
) -> dict[str, Any]:
    z = z0[region]
    y = target[region]
    ref = reference[region]
    absolute_values = absolute[region]
    residual_values = residual[region]
    base_loss = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    expanded_target = y.unsqueeze(-1).expand_as(absolute_values)
    absolute_loss = F.binary_cross_entropy_with_logits(
        z.unsqueeze(-1) + 0.05 * absolute_values,
        expanded_target,
        reduction="none",
    )
    residual_loss = F.binary_cross_entropy_with_logits(
        z.unsqueeze(-1) + 0.05 * residual_values,
        expanded_target,
        reduction="none",
    )
    denominator = base_loss.unsqueeze(-1).clamp_min(0.1)
    absolute_gain = (base_loss.unsqueeze(-1) - absolute_loss) / denominator
    residual_gain = (base_loss.unsqueeze(-1) - residual_loss) / denominator
    current_oracle = absolute_loss.min(dim=-1).values
    residual_oracle = residual_loss.min(dim=-1).values
    absolute_with_noop = torch.minimum(base_loss, current_oracle)
    residual_with_noop = torch.minimum(base_loss, residual_oracle)
    absolute_winners = torch.cat((base_loss.unsqueeze(-1), absolute_loss), dim=-1).argmin(dim=-1)
    residual_winners = torch.cat((base_loss.unsqueeze(-1), residual_loss), dim=-1).argmin(dim=-1)

    factors = []
    for factor in range(absolute_values.shape[-1]):
        factor_absolute = absolute_values[:, factor]
        factor_residual = residual_values[:, factor]
        factors.append({
            "factor": factor + 1,
            "mean_l_m": float(factor_absolute.mean().item()),
            "mean_delta_m": float(factor_residual.mean().item()),
            "std_delta_m": float(factor_residual.std(unbiased=False).item()),
            "p_l_m_positive": float((factor_absolute > 0).float().mean().item()),
            "p_l_m_negative": float((factor_absolute < 0).float().mean().item()),
            "p_delta_m_positive": float((factor_residual > 0).float().mean().item()),
            "p_delta_m_negative": float((factor_residual < 0).float().mean().item()),
        })

    winner_fractions = {
        "Base": float((residual_winners == 0).float().mean().item())
    }
    winner_fractions.update({
        f"F{factor + 1}": float((residual_winners == factor + 1).float().mean().item())
        for factor in range(absolute_values.shape[-1])
    })
    base_mean = base_loss.mean()
    residual_factor_means = residual_loss.mean(dim=0)
    residual_best_single = residual_factor_means.min()
    residual_oracle_mean = residual_oracle.mean()
    return {
        "region": name,
        "count": int(region.sum().item()),
        "mean_l_ref": float(ref.mean().item()),
        "factors": factors,
        "best_absolute_factor_gain_mean": float(absolute_gain.max(dim=-1).values.mean().item()),
        "best_residual_factor_gain_mean": float(residual_gain.max(dim=-1).values.mean().item()),
        "current_all_harm_fraction": float((absolute_gain.max(dim=-1).values <= 0).float().mean().item()),
        "residual_all_harm_fraction": float((residual_gain.max(dim=-1).values <= 0).float().mean().item()),
        "noop_selected_fraction": float((absolute_winners == 0).float().mean().item()),
        "residual_noop_selected_fraction": float((residual_winners == 0).float().mean().item()),
        "Base": float(base_mean.item()),
        "CurrentOracle": float(current_oracle.mean().item()),
        "OraclePlusNoOp": float(absolute_with_noop.mean().item()),
        "ResidualOraclePlusNoOp": float(residual_with_noop.mean().item()),
        "ResidualOracle": float(residual_oracle_mean.item()),
        "ResidualBestSingle": float(residual_best_single.item()),
        "ResidualG_local": float(((base_mean - residual_oracle_mean) / base_mean.clamp_min(1e-12)).item()),
        "ResidualG_multi": float(((residual_best_single - residual_oracle_mean) / base_mean.clamp_min(1e-12)).item()),
        "residual_noop_winner_fractions": winner_fractions,
        "absolute_factor_correlation": _correlation_summary(absolute_values),
        "residual_factor_correlation": _correlation_summary(residual_values),
        "mean_subtracted_factor_correlation": _correlation_summary(
            absolute_values - absolute_values.mean(dim=-1, keepdim=True)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/p1_v83_dev/corrected_300b_primary_anchored_attempt1/adapter_1.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/p1_v83_dev/v84_forensic_audit"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the forensic forward replay")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if (args.output_dir / "forensic_summary.json").exists():
        raise FileExistsError(
            f"refusing to overwrite completed forensic audit: {args.output_dir}"
        )

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _seed(args.seed)
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = _model_from_checkpoint(checkpoint, device)
    model.requires_grad_(False)
    model.eval()
    model.clipmodel.eval()
    state_hash_before = _state_hash(model)
    all_grad_none_before = all(parameter.grad is None for parameter in model.parameters())

    dataset = _IndexedDataset(
        get_text_and_image_dataset("VisA", int(checkpoint["img_size"]), "train")
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(args.seed),
    )

    absolute_records: list[torch.Tensor] = []
    residual_records: list[torch.Tensor] = []
    reference_records: list[torch.Tensor] = []
    base_records: list[torch.Tensor] = []
    target_records: list[torch.Tensor] = []
    stage_accumulator: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    dataset_indices: list[int] = []
    noop_bank_max_error = 0.0
    noop_factor_logit_max_error = 0.0
    factor_logit_formula_max_error = 0.0
    started = time.monotonic()

    for batch_number, sample in enumerate(loader, start=1):
        if batch_number > args.max_batches:
            break
        dataset_indices.append(int(sample["dataset_index"].item()))
        image = sample["image"].to(device, non_blocking=True)
        mask = sample["mask"].to(device, non_blocking=True)
        local_valid = sample["local_mask_valid"].to(device, non_blocking=True)
        class_names = list(sample["class_name"])

        with torch.no_grad():
            visual = model(image, return_phase4_features=True)
            h6_batch = model.h6.build_batch(
                model,
                "VisA",
                class_names,
                visual,
                hybrid_alpha=float(checkpoint["hybrid_alpha_current"]),
                update_load_bias=False,
            )
            seg_features = torch.stack(visual["seg_tokens"], dim=0)
            text_global = get_phase2b_global_text_features(
                model,
                "VisA",
                class_names,
                device,
                use_hybrid_soft_prompt=True,
                use_soft_prompt=False,
            ).to(dtype=seg_features.dtype)
            _, _, z0 = model.vision_text_fusion_gate_seg(
                seg_features,
                text_global,
                img_size=int(checkpoint["img_size"]),
                h6_patch_logits=h6_batch["h6_logits"],
                return_details=True,
            )
            patch_count = int(h6_batch["factor_patch_logits"].shape[2])
            y_patch, valid_patch = build_patch_targets(mask, patch_count, local_valid)

            patches = F.normalize(seg_features.float(), dim=-1)
            active_bank = h6_batch["active_factor_bank"].float()
            noop_bank = h6_batch["expected_noop_pre_expert_bank"].float()
            expected_noop = F.normalize(h6_batch["hard_adapted"].float(), dim=2)
            expected_noop = expected_noop.unsqueeze(2).expand_as(noop_bank)
            noop_bank_max_error = max(
                noop_bank_max_error,
                float((noop_bank - expected_noop).abs().max().item()),
            )
            direct_absolute = 10.0 * torch.einsum(
                "gbpd,gbmd->gbpm",
                patches,
                active_bank[..., 1] - active_bank[..., 0],
            )
            factor_logit_formula_max_error = max(
                factor_logit_formula_max_error,
                float(
                    (direct_absolute - h6_batch["factor_patch_logits"].float())
                    .abs()
                    .max()
                    .item()
                ),
            )
            noop_logits_all = 10.0 * torch.einsum(
                "gbpd,gbmd->gbpm",
                patches,
                noop_bank[..., 1] - noop_bank[..., 0],
            )
            noop_factor_logit_max_error = max(
                noop_factor_logit_max_error,
                float(
                    (noop_logits_all - noop_logits_all[..., :1])
                    .abs()
                    .max()
                    .item()
                ),
            )
            reference = noop_logits_all[..., 0]
            residual = h6_batch["factor_patch_logits"].float() - reference.unsqueeze(-1)
            valid = valid_patch.unsqueeze(0).expand_as(z0)
            targets = y_patch.unsqueeze(0).expand_as(z0).float()

            absolute_records.append(
                h6_batch["factor_patch_logits"].float()[valid].cpu()
            )
            residual_records.append(residual[valid].cpu())
            reference_records.append(reference[valid].cpu())
            base_records.append(z0.float()[valid].cpu())
            target_records.append(targets[valid].cpu())

            if batch_number == 1:
                _append_stage(
                    stage_accumulator,
                    "concept_slots",
                    h6_batch["concept_slots"],
                    0,
                )
            for stage_name, key, factor_dim in (
                ("normal_queries", "normal_queries", 1),
                ("abnormal_queries", "abnormal_queries", 1),
                ("normal_prototypes", "prototype_normal", 1),
                ("abnormal_prototypes", "prototype_abnormal", 1),
                ("state_delta_raw", "state_delta_raw", 1),
                ("state_delta_generated", "state_delta_generated", 1),
                ("state_delta_with_identity", "state_delta_with_identity", 1),
                ("state_tokens", "state_tokens", 1),
                ("structured_prompt_contexts_used", "structured_contexts", 1),
                ("legacy_dynamic_contexts_not_used_by_v83_prompt", "dynamic_contexts", 1),
                ("dynamic_text_raw", "dynamic_text_raw", 2),
                ("dynamic_text_normalized", "dynamic_text", 2),
                ("factor_bank", "active_factor_bank", 2),
            ):
                _append_stage(
                    stage_accumulator, stage_name, h6_batch[key], factor_dim
                )
            _append_stage(
                stage_accumulator,
                "abnormal_minus_normal_factor_direction",
                active_bank[..., 1] - active_bank[..., 0],
                2,
            )

        if batch_number % args.progress_every == 0 or batch_number == args.max_batches:
            _write_json(
                args.output_dir / "progress.json",
                {
                    "status": "RUNNING" if batch_number < args.max_batches else "FORWARD_COMPLETE",
                    "batches": batch_number,
                    "valid_group_patches": int(sum(item.shape[0] for item in base_records)),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            print(
                json.dumps({
                    "batches": batch_number,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }),
                flush=True,
            )

    absolute = torch.cat(absolute_records, dim=0)
    residual = torch.cat(residual_records, dim=0)
    reference = torch.cat(reference_records, dim=0)
    z0 = torch.cat(base_records, dim=0)
    target = torch.cat(target_records, dim=0)
    normal = target < 0.5
    anomaly = target >= 0.5
    factor_mean = absolute.mean(dim=-1, keepdim=True)
    mean_subtracted = absolute - factor_mean
    total_energy = absolute.square().sum()
    common_energy = factor_mean.expand_as(absolute).square().sum()
    residual_energy = mean_subtracted.square().sum()

    regions = {
        "overall": _region_report(
            "overall", torch.ones_like(normal), z0, target, reference, absolute, residual
        ),
        "normal": _region_report(
            "normal", normal, z0, target, reference, absolute, residual
        ),
        "anomaly": _region_report(
            "anomaly", anomaly, z0, target, reference, absolute, residual
        ),
    }
    common_mode = {
        "common_mode_energy_fraction": float((common_energy / total_energy.clamp_min(1e-12)).item()),
        "factor_residual_energy_fraction": float((residual_energy / total_energy.clamp_min(1e-12)).item()),
        "orthogonal_decomposition_relative_error": float(
            ((total_energy - common_energy - residual_energy).abs() / total_energy.clamp_min(1e-12)).item()
        ),
        "variance_across_factors_mean": float(absolute.var(dim=-1, unbiased=False).mean().item()),
        "all_factor_sign_agreement_fraction": float(
            (((absolute > 0).all(dim=-1)) | ((absolute < 0).all(dim=-1))).float().mean().item()
        ),
        "absolute_factor_correlation": _correlation_summary(absolute),
        "noop_residual_factor_correlation": _correlation_summary(residual),
        "mean_subtracted_factor_correlation": _correlation_summary(mean_subtracted),
    }
    stages = _summarize_stages(stage_accumulator)
    stages["absolute_patch_function"] = {
        "effective_rank": _effective_rank(absolute.T),
        "functional_correlation_mean": regions["overall"]["absolute_factor_correlation"]["pairwise_mean"],
        "functional_correlation_min": regions["overall"]["absolute_factor_correlation"]["pairwise_min"],
        "functional_correlation_max": regions["overall"]["absolute_factor_correlation"]["pairwise_max"],
        "factorwise_std_mean": float(absolute.std(dim=-1, unbiased=False).mean().item()),
    }
    stages["noop_residual_patch_function"] = {
        "effective_rank": _effective_rank(residual.T),
        "functional_correlation_mean": regions["overall"]["residual_factor_correlation"]["pairwise_mean"],
        "functional_correlation_min": regions["overall"]["residual_factor_correlation"]["pairwise_min"],
        "functional_correlation_max": regions["overall"]["residual_factor_correlation"]["pairwise_max"],
        "factorwise_std_mean": float(residual.std(dim=-1, unbiased=False).mean().item()),
    }

    state_hash_after = _state_hash(model)
    all_grad_none_after = all(parameter.grad is None for parameter in model.parameters())
    noop_vs_base = _correlation_summary(
        torch.stack((reference, z0), dim=-1)
    )
    output = {
        "status": "PASS",
        "forensic_case": "PENDING_DECISION_REVIEW",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_git_sha": checkpoint.get("git_sha"),
        "audit_source_git_head": _git_head(),
        "probe_kind": "seed0_visa_forward_only_no_step",
        "seed": args.seed,
        "batches": len(dataset_indices),
        "dataset_indices": dataset_indices,
        "loader": {
            "batch_size": 1,
            "shuffle": True,
            "num_workers": 0,
            "generator_seed": args.seed,
            "replay_scope": (
                "same canonical loader seed/order/augmentation policy; final checkpoint "
                "is evaluated statically rather than recreating historical intermediate weights"
            ),
        },
        "optimizer_constructed": False,
        "backward_executed": False,
        "optimizer_steps": 0,
        "all_grad_none_before": all_grad_none_before,
        "all_grad_none_after": all_grad_none_after,
        "model_state_hash_before": state_hash_before,
        "model_state_hash_after": state_hash_after,
        "model_state_unchanged": state_hash_before == state_hash_after,
        "rho": model.h6.rho_values().detach().cpu().tolist(),
        "noop_reference": {
            "source": "expected_noop_pre_expert_bank",
            "formula": "normalize(hard_adapted) broadcast over factors via legacy fusion alpha=0",
            "distinct_from_z_base": bool(
                float((reference - z0).abs().mean().item()) > 1e-8
            ),
            "noop_bank_max_absolute_error": noop_bank_max_error,
            "noop_factor_logit_max_absolute_error": noop_factor_logit_max_error,
            "factor_logit_formula_max_absolute_error": factor_logit_formula_max_error,
            "l_ref_vs_z_base_correlation": noop_vs_base,
            "l_ref_minus_z_base_abs_mean": float((reference - z0).abs().mean().item()),
        },
        "regions": regions,
        "common_mode": common_mode,
        "collapse_trace": stages,
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json(args.output_dir / "oracle_decomposition.json", {
        "regions": regions,
        "common_mode": common_mode,
    })
    _write_json(args.output_dir / "collapse_trace.json", {
        "stages": stages,
        "classification": "PENDING_DECISION_REVIEW",
    })
    _write_json(args.output_dir / "forensic_summary.json", output)
    _write_json(args.output_dir / "progress.json", {
        "status": "COMPLETE",
        "batches": len(dataset_indices),
        "valid_group_patches": int(z0.numel()),
        "elapsed_seconds": output["runtime_seconds"],
    })
    print(json.dumps({
        "status": output["status"],
        "batches": output["batches"],
        "normal_group_patches": regions["normal"]["count"],
        "anomaly_group_patches": regions["anomaly"]["count"],
        "runtime_seconds": round(output["runtime_seconds"], 3),
    }))


if __name__ == "__main__":
    main()
