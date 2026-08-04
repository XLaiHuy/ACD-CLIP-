"""Structural early-stop gates for H6 Progress-1.

The gates catch architectural collapse only.  They deliberately ignore ordinary
metric noise and keep independent consecutive counters so one bad epoch can
recover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


STRUCTURAL_GATE_VERSION = 1
HARD_GATE_NAMES = (
    "query_collapse",
    "key_anchor_failure",
    "factor_collapse",
    "semantic_drift",
    "sparse_collapse",
)


@dataclass(frozen=True)
class StructuralGateConfig:
    enabled: bool = False
    patience: int = 2
    dense_start_epoch: int = 8
    require_all_levels: bool = True
    query_rank_max: float = 1.10
    query_top1_energy_min: float = 0.995
    query_cosine_min: float = 0.9999
    logit_std_max: float = 1e-6
    key_cosine_max: float = 0.95
    key_l2_min: float = 0.05
    dynamic_cosine_min: float = 0.999
    dynamic_orth_center: float = 0.75
    dynamic_orth_tolerance: float = 0.005
    hard_anchor_cosine_min: float = 0.30
    sparse_min_ratio: float = 0.50
    max_sparse_dead_factors: int = 1
    min_unique_topk_pairs: int = 2

    @classmethod
    def from_args(cls, args) -> "StructuralGateConfig":
        return cls(
            enabled=bool(args.h6_structural_gate_enabled),
            patience=int(args.h6_structural_gate_patience),
            dense_start_epoch=int(args.h6_structural_gate_dense_start_epoch),
            require_all_levels=bool(args.h6_structural_gate_require_all_levels),
            query_rank_max=float(args.h6_gate_query_rank_max),
            query_top1_energy_min=float(args.h6_gate_query_top1_energy_min),
            query_cosine_min=float(args.h6_gate_query_cosine_min),
            logit_std_max=float(args.h6_gate_logit_std_max),
            key_cosine_max=float(args.h6_gate_key_cosine_max),
            key_l2_min=float(args.h6_gate_key_l2_min),
            dynamic_cosine_min=float(args.h6_gate_dynamic_cosine_min),
            dynamic_orth_center=float(args.h6_gate_dynamic_orth_center),
            dynamic_orth_tolerance=float(args.h6_gate_dynamic_orth_tolerance),
            hard_anchor_cosine_min=float(args.h6_gate_hard_anchor_cosine_min),
            sparse_min_ratio=float(args.h6_gate_sparse_min_ratio),
            max_sparse_dead_factors=int(args.h6_gate_max_sparse_dead_factors),
            min_unique_topk_pairs=int(args.h6_gate_min_unique_topk_pairs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "structural_gate_enabled": self.enabled,
            "structural_gate_version": STRUCTURAL_GATE_VERSION,
            "structural_gate_patience": self.patience,
            "structural_gate_dense_start_epoch": self.dense_start_epoch,
            "structural_gate_require_all_levels": self.require_all_levels,
            "h6_gate_query_rank_max": self.query_rank_max,
            "h6_gate_query_top1_energy_min": self.query_top1_energy_min,
            "h6_gate_query_cosine_min": self.query_cosine_min,
            "h6_gate_logit_std_max": self.logit_std_max,
            "h6_gate_key_cosine_max": self.key_cosine_max,
            "h6_gate_key_l2_min": self.key_l2_min,
            "h6_gate_dynamic_cosine_min": self.dynamic_cosine_min,
            "h6_gate_dynamic_orth_center": self.dynamic_orth_center,
            "h6_gate_dynamic_orth_tolerance": self.dynamic_orth_tolerance,
            "h6_gate_hard_anchor_cosine_min": self.hard_anchor_cosine_min,
            "h6_gate_sparse_min_ratio": self.sparse_min_ratio,
            "h6_gate_max_sparse_dead_factors": self.max_sparse_dead_factors,
            "h6_gate_min_unique_topk_pairs": self.min_unique_topk_pairs,
            "warning_thresholds": {
                "teacher_entropy": 0.98,
                "query_rank": 2.0,
                "query_top1_energy": 0.95,
                "final_key_cosine": 0.80,
                "dynamic_residual_cosine": 0.99,
                "dynamic_mean_hard_cosine": 0.70,
                "task_loss_rolling_median_multiplier": 2.0,
            },
        }


@dataclass
class GateDecision:
    hard_failure: bool = False
    abort_reason: str | None = None
    soft_warnings: list[str] = field(default_factory=list)
    failed: dict[str, bool] = field(default_factory=dict)
    per_level: dict[str, list[bool]] = field(default_factory=dict)
    fatal_metrics: list[str] = field(default_factory=list)

    @property
    def state_label(self) -> str:
        if self.hard_failure:
            return "hard_failure"
        if self.soft_warnings:
            return "warning"
        return "ok"


class H6StructuralGateState:
    def __init__(self, config: StructuralGateConfig, counters: Mapping[str, int] | None = None):
        self.config = config
        self.counters = {name: int((counters or {}).get(name, 0)) for name in HARD_GATE_NAMES}
        self.last_decision = GateDecision()

    def reset(self) -> None:
        for name in HARD_GATE_NAMES:
            self.counters[name] = 0
        self.last_decision = GateDecision()

    def state_dict(self) -> dict[str, Any]:
        return {
            "structural_gate_version": STRUCTURAL_GATE_VERSION,
            "structural_gate_counters": dict(self.counters),
            "structural_gate_last_decision": self.decision_to_dict(self.last_decision),
            "structural_gate_last_soft_warnings": list(self.last_decision.soft_warnings),
        }

    @staticmethod
    def decision_to_dict(decision: GateDecision) -> dict[str, Any]:
        return {
            "gate_state": decision.state_label,
            "gate_hard_failure": decision.hard_failure,
            "gate_abort_reason": decision.abort_reason,
            "gate_soft_warnings": list(decision.soft_warnings),
            "gate_failed": dict(decision.failed),
            "gate_per_level": {key: list(value) for key, value in decision.per_level.items()},
            "gate_fatal_metrics": list(decision.fatal_metrics),
        }

    @staticmethod
    def tensor_to_python(value: Any) -> Any:
        if torch.is_tensor(value):
            data = value.detach().cpu()
            return data.item() if data.ndim == 0 else data.tolist()
        return value

    @staticmethod
    def finite_metric_names(metrics: Mapping[str, Any]) -> list[str]:
        bad = []
        for key, value in metrics.items():
            if torch.is_tensor(value):
                if not torch.isfinite(value.detach().float()).all().item():
                    bad.append(key)
            elif isinstance(value, (float, int)) and not torch.isfinite(torch.tensor(float(value))).item():
                bad.append(key)
        return bad

    @staticmethod
    def _float(value: Any, default: float = 0.0, reducer: str = "mean") -> float:
        if value is None:
            return default
        if torch.is_tensor(value):
            data = value.detach().float()
            if data.numel() == 0:
                return default
            if reducer == "max":
                return float(data.max().item())
            if reducer == "min":
                return float(data.min().item())
            return float(data.mean().item())
        if isinstance(value, (list, tuple)):
            tensor = torch.tensor(value, dtype=torch.float32)
            if tensor.numel() == 0:
                return default
            if reducer == "max":
                return float(tensor.max().item())
            if reducer == "min":
                return float(tensor.min().item())
            return float(tensor.mean().item())
        return float(value)

    @staticmethod
    def _level_list(value: Any) -> list[float]:
        if value is None:
            return []
        if torch.is_tensor(value):
            data = value.detach().float().cpu()
            if data.ndim == 0:
                return [float(data.item())]
            if data.ndim > 1:
                data = data.reshape(data.shape[0], -1)
                data = data.max(dim=1).values
            return [float(v) for v in data.tolist()]
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                if isinstance(item, (list, tuple)):
                    out.append(max(float(v) for v in item))
                else:
                    out.append(float(item))
            return out
        return [float(value)]

    def _level_gate(self, decisions: list[bool]) -> bool:
        active = [bool(value) for value in decisions]
        if not active:
            return False
        return all(active) if self.config.require_all_levels else any(active)

    def _update_counter(self, name: str, failed: bool) -> bool:
        self.counters[name] = self.counters[name] + 1 if failed else 0
        return failed and self.counters[name] >= self.config.patience

    def evaluate(
        self,
        *,
        epoch: int,
        diagnostics: Mapping[str, Any],
        epoch_metrics: Mapping[str, float],
        teacher_diag: Mapping[str, Any],
        sparse_ratio: float,
        hybrid_alpha: float,
        hybrid_alpha_max: float,
        router_teacher_weight: float,
        router_teacher_target: float,
        dynamic_mean_anchor_weight: float,
        dynamic_mean_anchor_target: float,
        query_mode: str,
        tangent_enabled: bool,
    ) -> GateDecision:
        decision = GateDecision()
        if not self.config.enabled:
            self.last_decision = decision
            return decision

        finite_inputs = dict(diagnostics)
        finite_inputs.update({
            "task_loss": float(epoch_metrics.get("task", 0.0)),
            "total_loss": float(epoch_metrics.get("total", 0.0)),
            "dynamic_mean_anchor_weight": float(dynamic_mean_anchor_weight),
        })
        fatal = self.finite_metric_names(finite_inputs)
        if fatal:
            decision.hard_failure = True
            decision.abort_reason = "h6_nonfinite_structural_gate_metric"
            decision.fatal_metrics = fatal
            self.last_decision = decision
            return decision

        teacher_entropy = self._float(teacher_diag.get("teacher_entropy"), 0.0, reducer="mean")
        teacher_fraction = self._float(teacher_diag.get("teacher_informative_patch_fraction"), 1.0, reducer="mean")
        if teacher_entropy >= 0.98:
            decision.soft_warnings.append("teacher_entropy_high")
        if teacher_fraction <= 0.05:
            decision.soft_warnings.append("teacher_informative_fraction_low")

        task_loss = float(epoch_metrics.get("task", 0.0))
        rolling_median = float(epoch_metrics.get("task_rolling_median", task_loss))
        if rolling_median > 0.0 and task_loss > 2.0 * rolling_median:
            decision.soft_warnings.append("task_loss_spike")

        query_rank = self._level_list(diagnostics.get("final_query_effective_rank"))
        query_top1 = self._level_list(diagnostics.get("final_query_top1_energy_ratio"))
        query_cos = self._level_list(diagnostics.get("final_query_pairwise_cos_mean"))
        logit_std = self._level_list(diagnostics.get("per_factor_logit_std_across_patches"))
        unique_pairs = self._level_list(diagnostics.get("unique_topk_pairs"))
        if any(value < 2.0 for value in query_rank):
            decision.soft_warnings.append("query_low_rank")
        if any(value > 0.95 for value in query_top1):
            decision.soft_warnings.append("query_top1_energy_high")
        if any(value < self.config.min_unique_topk_pairs for value in unique_pairs):
            decision.soft_warnings.append("unique_topk_pairs_low")

        alpha_full = float(hybrid_alpha) >= float(hybrid_alpha_max) - 1e-8
        teacher_full = float(router_teacher_weight) >= float(router_teacher_target) - 1e-12
        query_eligible = (
            epoch >= self.config.dense_start_epoch
            and teacher_full
            and alpha_full
            and query_mode != "raw"
            and bool(query_rank)
        )
        query_failed_levels = []
        for index, rank in enumerate(query_rank):
            top1 = query_top1[index] if index < len(query_top1) else 0.0
            cos = query_cos[index] if index < len(query_cos) else 0.0
            std = logit_std[index] if index < len(logit_std) else float("inf")
            pairs = unique_pairs[index] if index < len(unique_pairs) else self.config.min_unique_topk_pairs
            query_failed_levels.append(
                query_eligible
                and rank <= self.config.query_rank_max
                and top1 >= self.config.query_top1_energy_min
                and cos >= self.config.query_cosine_min
                and std <= self.config.logit_std_max
                and pairs < self.config.min_unique_topk_pairs
            )
        query_failed = self._level_gate(query_failed_levels)
        decision.per_level["query_failed_levels"] = query_failed_levels
        decision.failed["query_collapse"] = query_failed

        key_cos = self._float(diagnostics.get("final_router_key_cos_max"), 0.0, reducer="max")
        key_l2 = self._float(diagnostics.get("final_router_key_l2_min"), float("inf"), reducer="min")
        raw_key_cos = self._float(diagnostics.get("raw_concept_key_cos_max"), 0.0, reducer="max")
        if raw_key_cos >= 0.80:
            decision.soft_warnings.append("raw_concept_key_geometry_similar")
        if key_cos >= 0.80:
            decision.soft_warnings.append("final_key_cosine_high")
        key_failed = epoch >= 2 and (key_cos >= self.config.key_cosine_max or key_l2 <= self.config.key_l2_min)
        decision.failed["key_anchor_failure"] = key_failed

        dyn_res_cos = self._float(diagnostics.get("dynamic_residual_cos_mean"), 0.0, reducer="max")
        dyn_text_cos = self._float(diagnostics.get("stage_dynamic_text_norm_cos_mean"), 0.0, reducer="max")
        tangent_pair_cos = self._float(diagnostics.get("factor_identity_tangent_pair_cos_max"), 0.0, reducer="max")
        orth = float(epoch_metrics.get("orth", 0.0))
        if dyn_res_cos >= 0.99:
            decision.soft_warnings.append("dynamic_residual_cosine_high")
        dynamic_eligible = tangent_enabled and alpha_full and epoch >= self.config.dense_start_epoch
        factor_failed = (
            dynamic_eligible
            and dyn_res_cos >= self.config.dynamic_cosine_min
            and dyn_text_cos >= self.config.dynamic_cosine_min
            and abs(orth - self.config.dynamic_orth_center) <= self.config.dynamic_orth_tolerance
            and tangent_pair_cos >= self.config.dynamic_cosine_min
        )
        decision.failed["factor_collapse"] = factor_failed

        mean_hard_cos = self._float(diagnostics.get("dynamic_mean_hard_cos"), 1.0, reducer="min")
        if mean_hard_cos < 0.70:
            decision.soft_warnings.append("dynamic_mean_below_trust_floor")
        trust_full = float(dynamic_mean_anchor_weight) >= float(dynamic_mean_anchor_target) - 1e-12
        semantic_failed = trust_full and alpha_full and mean_hard_cos < self.config.hard_anchor_cosine_min
        decision.failed["semantic_drift"] = semantic_failed

        sparse_dead = self._level_list(diagnostics.get("sparse_dead_factors", diagnostics.get("sparse_dead")))
        if not sparse_dead and "sparse_factor_usage" in diagnostics:
            usage = diagnostics["sparse_factor_usage"]
            sparse_dead = self._level_list((usage.detach().float() < 0.01).sum(dim=-1) if torch.is_tensor(usage) else None)
        if sparse_ratio == 0.25 and any(value > 1 for value in sparse_dead):
            decision.soft_warnings.append("sparse_dead_transition_epoch")
        sparse_failed_levels = []
        for index, dead in enumerate(sparse_dead):
            pairs = unique_pairs[index] if index < len(unique_pairs) else self.config.min_unique_topk_pairs
            sparse_failed_levels.append(
                sparse_ratio >= self.config.sparse_min_ratio
                and dead > self.config.max_sparse_dead_factors
                and pairs < self.config.min_unique_topk_pairs
            )
        sparse_failed = self._level_gate(sparse_failed_levels)
        decision.per_level["sparse_failed_levels"] = sparse_failed_levels
        decision.failed["sparse_collapse"] = sparse_failed

        abort_map = {
            "query_collapse": "h6_dense_patch_query_collapse",
            "key_anchor_failure": "h6_final_router_key_anchor_failure",
            "factor_collapse": "h6_dynamic_factor_directional_collapse",
            "semantic_drift": "h6_catastrophic_clip_semantic_drift",
            "sparse_collapse": "h6_sparse_single_pair_collapse",
        }
        for name in HARD_GATE_NAMES:
            if self._update_counter(name, bool(decision.failed.get(name, False))):
                decision.hard_failure = True
                decision.abort_reason = abort_map[name]
                break

        self.last_decision = decision
        return decision

