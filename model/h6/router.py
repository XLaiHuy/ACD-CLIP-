"""Patch-wise sparse text routing for the H6 dynamic factor bank."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .semantic_bank import apply_relative_slot_offsets_, deterministic_slot_directions


class PatchRouter(nn.Module):
    def __init__(
        self,
        n_groups: int,
        num_factors: int = 4,
        text_dim: int = 768,
        bank_dim: int = 256,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        soft_routing_epochs: int = 2,
        sparse_transition_epochs: int = 1,
        top_k: int = 2,
        load_bias_enabled: bool = False,
        load_bias_momentum: float = 0.9,
        load_bias_step: float = 0.001,
        load_bias_max: float = 0.03,
        slot_init_enabled: bool = False,
        slot_init_scale: float = 0.02,
        slot_init_seed_offset: int = 6100,
        router_query_mode: str = "local_global_bypass",
        router_query_global_weight: float = 0.10,
        router_local_bypass_scale: float = 0.10,
        router_local_bypass_max_ratio: float = 0.20,
        router_local_projection_seed_offset: int = 7200,
        router_key_anchor_enabled: bool = True,
        router_key_anchor_seed_offset: int = 7300,
        router_key_adaptation_initial_ratio: float = 0.10,
        router_key_adaptation_max_ratio: float = 0.25,
    ):
        super().__init__()
        if not 0 < top_k <= num_factors:
            raise ValueError("top_k must be in [1, num_factors]")
        if temperature <= 0:
            raise ValueError("router temperature must be positive")
        if router_query_mode not in {"raw", "local_residual", "local_global_bypass"}:
            raise ValueError("router_query_mode must be raw, local_residual, or local_global_bypass")
        self.n_groups = int(n_groups)
        self.num_factors = int(num_factors)
        self.text_dim = int(text_dim)
        self.bank_dim = int(bank_dim)
        self.temperature = float(temperature)
        self.soft_routing_epochs = int(soft_routing_epochs)
        self.sparse_transition_epochs = max(1, int(sparse_transition_epochs))
        self.top_k = int(top_k)
        self.load_bias_enabled = bool(load_bias_enabled)
        self.load_bias_momentum = float(load_bias_momentum)
        self.load_bias_step = float(load_bias_step)
        self.load_bias_max = float(load_bias_max)
        self.slot_init_enabled = bool(slot_init_enabled)
        self.slot_init_scale = float(slot_init_scale)
        self.slot_init_seed_offset = int(slot_init_seed_offset)
        self.router_query_mode = str(router_query_mode)
        self.router_query_global_weight = float(router_query_global_weight)
        self.router_local_bypass_scale = float(router_local_bypass_scale)
        self.router_local_bypass_max_ratio = float(router_local_bypass_max_ratio)
        self.router_local_projection_seed_offset = int(router_local_projection_seed_offset)
        self.router_key_anchor_enabled = bool(router_key_anchor_enabled)
        self.router_key_anchor_seed_offset = int(router_key_anchor_seed_offset)
        self.router_key_adaptation_initial_ratio = float(router_key_adaptation_initial_ratio)
        self.router_key_adaptation_max_ratio = float(router_key_adaptation_max_ratio)
        self.level_embedding = nn.Parameter(torch.empty(n_groups, text_dim))
        self.query_projector = nn.Sequential(
            nn.Linear(3 * text_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, bank_dim)
        )
        self.local_query_projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, bank_dim)
        )
        self.local_norm = nn.LayerNorm(text_dim)
        self.global_norm = nn.LayerNorm(text_dim)
        self.fallback_concept_keys = nn.Parameter(torch.empty(num_factors, bank_dim))
        self.register_buffer(
            "frozen_local_projection",
            deterministic_slot_directions(
                bank_dim,
                text_dim,
                self.router_local_projection_seed_offset,
            ).T.contiguous(),
        )
        self.register_buffer(
            "router_key_anchors",
            deterministic_slot_directions(
                num_factors,
                bank_dim,
                self.router_key_anchor_seed_offset,
            ),
        )
        self.register_buffer("ema_topk_usage", torch.full((n_groups, num_factors), 1.0 / num_factors))
        self.register_buffer("load_bias", torch.zeros(n_groups, num_factors))
        nn.init.normal_(self.level_embedding, std=0.02)
        nn.init.normal_(self.fallback_concept_keys, std=0.02)
        self.slot_init_applied_components: list[str] = []
        if self.slot_init_enabled:
            apply_relative_slot_offsets_(
                self.fallback_concept_keys,
                scale=self.slot_init_scale,
                seed=self.slot_init_seed_offset + 17,
            )
            self.slot_init_applied_components.append("router.fallback_concept_keys")

    @staticmethod
    def _stack(level_tokens: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        return level_tokens if torch.is_tensor(level_tokens) else torch.stack(list(level_tokens), dim=0)

    def forward(
        self,
        level_tokens: Sequence[torch.Tensor] | torch.Tensor,
        epoch_one_based: int,
        concept_keys: torch.Tensor | None = None,
        update_load_bias: bool = False,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        input_alias = False
        if not torch.is_tensor(level_tokens):
            token_list = list(level_tokens)
            input_alias = len({id(t) for t in token_list}) != len(token_list)
            level_tokens = token_list
        stacked = self._stack(level_tokens)
        with torch.autocast(device_type=stacked.device.type, enabled=False):
            output = self._forward_fp32(stacked, epoch_one_based, concept_keys, update_load_bias, valid_patch_mask)
        output["level_input_alias"] = torch.tensor(input_alias, device=stacked.device)
        return output

    def sparse_ratio(self, epoch_one_based: int) -> float:
        """Deterministic dense-to-sparse interpolation; cannot stay dense forever."""
        epoch = int(epoch_one_based)
        if epoch <= self.soft_routing_epochs:
            return 0.0
        step = epoch - self.soft_routing_epochs
        return min(1.0, float(step) / float(self.sparse_transition_epochs))

    def _forward_fp32(
        self,
        tokens: torch.Tensor,
        epoch_one_based: int,
        concept_keys: torch.Tensor | None = None,
        update_load_bias: bool = False,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if tokens.ndim != 4 or tokens.shape[0] != self.n_groups or tokens.shape[-1] != self.text_dim:
            raise ValueError("router input must be [n_groups, B, P, 768]")
        if concept_keys is None:
            concept_keys = self.fallback_concept_keys
        if concept_keys.ndim != 2 or concept_keys.shape != (self.num_factors, self.bank_dim):
            raise ValueError(f"concept_keys must be [{self.num_factors}, {self.bank_dim}]")
        tokens = F.normalize(tokens.float(), dim=-1)
        groups, batch, patches, _ = tokens.shape
        local_patch, local_input, combined_input = self._local_query_inputs(tokens, valid_patch_mask)
        level = self.level_embedding[:, None, None, :].expand(groups, batch, patches, -1)
        if self.router_query_mode == "raw":
            context = tokens.mean(dim=2, keepdim=True).expand(-1, -1, patches, -1)
            raw_query = self.query_projector(torch.cat([tokens, context, level], dim=-1)).float()
            learned_query = raw_query
            local_bypass_residual = torch.zeros_like(learned_query)
            query_pre_norm = learned_query
        elif self.router_query_mode == "local_residual":
            raw_query = self.local_query_projector(tokens).float()
            learned_query = self.local_query_projector(local_input).float()
            local_bypass_residual = torch.zeros_like(learned_query)
            query_pre_norm = learned_query
        else:
            raw_query = self.local_query_projector(tokens).float()
            learned_query = self.local_query_projector(combined_input).float()
            local_bypass_raw = local_input @ self.frozen_local_projection.to(
                device=local_input.device,
                dtype=local_input.dtype,
            )
            local_bypass_direction = F.normalize(local_bypass_raw.float(), dim=-1)
            learned_query_norm = learned_query.detach().float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            local_bypass_residual = (
                float(self.router_local_bypass_scale) * learned_query_norm * local_bypass_direction
            )
            max_norm = float(self.router_local_bypass_max_ratio) * learned_query_norm
            residual_norm = local_bypass_residual.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            local_bypass_residual = local_bypass_residual * torch.clamp(max_norm / residual_norm, max=1.0)
            zero_local = local_bypass_raw.detach().float().norm(dim=-1, keepdim=True) < 1e-8
            local_bypass_residual = torch.where(zero_local, torch.zeros_like(local_bypass_residual), local_bypass_residual)
            query_pre_norm = learned_query + local_bypass_residual.to(learned_query.dtype)
        query = F.normalize(query_pre_norm.float(), dim=-1)
        keys, key_diagnostics = self.final_router_keys(concept_keys)
        logits = torch.einsum("gbpd,md->gbpm", query, keys) / self.temperature
        dense_probabilities = F.softmax(logits, dim=-1)
        ratio = self.sparse_ratio(epoch_one_based)
        bias_active = self.load_bias_enabled and ratio > 0.0
        selection_logits = logits
        if bias_active:
            selection_logits = selection_logits + self.load_bias[:, None, None, :].detach()
        _, topk_indices = torch.topk(selection_logits, k=self.top_k, dim=-1)
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(-1, topk_indices, logits.gather(-1, topk_indices))
        sparse_probabilities = F.softmax(masked_logits, dim=-1)
        st_sparse_probabilities = dense_probabilities + (sparse_probabilities - dense_probabilities).detach()
        prediction_probabilities = (1.0 - ratio) * dense_probabilities + ratio * st_sparse_probabilities
        sparse_active = bool(ratio >= 1.0)
        topk_frequency = self.topk_frequency(topk_indices, self.num_factors).detach()
        if self.training and update_load_bias and bias_active:
            self.update_load_bias(topk_frequency)
        level_query_difference = self._level_pairwise_l2_min(query)
        level_logit_difference = self._level_pairwise_l2_min(logits)
        level_input_difference = self._level_pairwise_l2_min(tokens)
        query_patch = self.query_patch_diagnostics(
            raw_query,
            local_input,
            query,
            logits,
            valid_patch_mask=valid_patch_mask,
            local_bypass_residual=local_bypass_residual,
            learned_query=learned_query,
        )
        return {
            "logits": logits,
            "selection_logits": selection_logits,
            "queries": query,
            "raw_queries": raw_query,
            "local_query_inputs": local_input,
            "raw_concept_keys": concept_keys,
            "concept_keys": keys,
            "final_router_keys": keys,
            "dense_probabilities": dense_probabilities,
            "sparse_probabilities": sparse_probabilities,
            "st_sparse_probabilities": st_sparse_probabilities,
            "prediction_probabilities": prediction_probabilities,
            "probabilities": prediction_probabilities,
            "topk_indices": topk_indices,
            "topk_frequency": topk_frequency,
            "sparse_ratio": torch.tensor(ratio, device=logits.device, dtype=logits.dtype),
            "sparse_active": torch.tensor(sparse_active, device=logits.device),
            "sparse": torch.tensor(sparse_active, device=logits.device),
            "load_bias": self.load_bias.detach().clone(),
            "ema_topk_usage": self.ema_topk_usage.detach().clone(),
            "level_input_difference": level_input_difference.detach(),
            "level_query_difference": level_query_difference.detach(),
            "level_logit_difference": level_logit_difference.detach(),
            "router_patch_count": torch.tensor(patches, device=logits.device, dtype=torch.long),
            "router_softmax_dim": torch.tensor(logits.ndim - 1, device=logits.device, dtype=torch.long),
            "router_topk_dim": torch.tensor(logits.ndim - 1, device=logits.device, dtype=torch.long),
            **key_diagnostics,
            **query_patch,
        }

    def _local_query_inputs(
        self,
        tokens: torch.Tensor,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if valid_patch_mask is None:
            image_center = tokens.mean(dim=2, keepdim=True).detach()
        else:
            mask = valid_patch_mask
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            if mask.ndim != 3 or tuple(mask.shape) != tuple(tokens.shape[:3]):
                raise ValueError("valid_patch_mask must be [B,P] or [G,B,P]")
            weights = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
            valid_count = weights.sum(dim=2, keepdim=True)
            masked_center = (tokens * weights).sum(dim=2, keepdim=True) / valid_count.clamp_min(1.0)
            fallback_center = tokens.mean(dim=2, keepdim=True)
            image_center = torch.where(valid_count > 0, masked_center, fallback_center).detach()
        local_patch = tokens - image_center
        local_input = self.local_norm(local_patch)
        global_input = self.global_norm(tokens)
        combined_input = local_input + float(self.router_query_global_weight) * global_input
        return local_patch, local_input, combined_input

    def final_router_keys(self, concept_keys: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw = concept_keys.float()
        raw_diag = self.concept_key_diagnostics(raw, prefix="raw_concept_key")
        if not self.router_key_anchor_enabled:
            final = F.normalize(raw, dim=-1)
            zero = raw.sum() * 0.0
            return final, {
                **raw_diag,
                **self.concept_key_diagnostics(final, prefix="final_router_key"),
                "router_key_adaptation_ratio_mean": zero.detach(),
                "router_key_adaptation_ratio_max": zero.detach(),
            }
        anchor = self.router_key_anchors.to(device=raw.device, dtype=raw.dtype)
        adaptation_direction = F.normalize(raw, dim=-1)
        anchor_norm = anchor.detach().float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        adaptation = float(self.router_key_adaptation_initial_ratio) * anchor_norm * adaptation_direction
        max_norm = float(self.router_key_adaptation_max_ratio) * anchor_norm
        adaptation_norm = adaptation.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        adaptation = adaptation * torch.clamp(max_norm / adaptation_norm, max=1.0)
        final = F.normalize(anchor + adaptation.to(anchor.dtype), dim=-1)
        ratio = adaptation.float().norm(dim=-1, keepdim=True) / anchor_norm
        return final, {
            **raw_diag,
            **self.concept_key_diagnostics(final, prefix="final_router_key"),
            "router_key_adaptation_ratio_mean": ratio.mean().detach(),
            "router_key_adaptation_ratio_max": ratio.max().detach(),
        }

    @staticmethod
    def _level_pairwise_l2_min(values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] < 2:
            return torch.tensor(float("inf"), device=values.device, dtype=values.dtype)
        flattened = values.float().reshape(values.shape[0], -1)
        distances = torch.cdist(flattened, flattened)
        mask = ~torch.eye(values.shape[0], device=values.device, dtype=torch.bool)
        return distances[mask].min()

    @staticmethod
    def query_patch_diagnostics(
        raw_query: torch.Tensor,
        local_query: torch.Tensor,
        query: torch.Tensor,
        logits: torch.Tensor,
        valid_patch_mask: torch.Tensor | None = None,
        local_bypass_residual: torch.Tensor | None = None,
        learned_query: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        def summarize(values: torch.Tensor, prefix: str) -> Dict[str, torch.Tensor]:
            groups, batch, patches, rank_dim = values.shape
            cos_means = []
            cos_maxs = []
            variances = []
            effective_ranks = []
            sv_ratios = []
            masks = PatchRouter._diagnostic_mask(valid_patch_mask, values)
            for group in range(groups):
                group_values = values[group].detach().float()
                group_mask = masks[group] if masks is not None else None
                per_image_rows = []
                per_image_var = []
                per_image_cos = []
                per_image_cos_max = []
                for image_idx in range(batch):
                    rows = group_values[image_idx]
                    if group_mask is not None:
                        rows = rows[group_mask[image_idx]]
                    if rows.shape[0] == 0:
                        rows = group_values[image_idx]
                    if rows.shape[0] > 1:
                        normed = F.normalize(rows, dim=-1)
                        gram = normed @ normed.T
                        offdiag = gram[~torch.eye(rows.shape[0], device=rows.device, dtype=torch.bool)]
                        per_image_cos.append(offdiag.mean())
                        per_image_cos_max.append(offdiag.abs().max())
                    else:
                        zero = rows.sum() * 0.0
                        per_image_cos.append(zero)
                        per_image_cos_max.append(zero)
                    centered = rows - rows.mean(dim=0, keepdim=True)
                    per_image_rows.append(centered)
                    per_image_var.append(centered.var(dim=0, unbiased=False).mean())
                matrix = torch.cat(per_image_rows, dim=0)
                singular = torch.linalg.svdvals(matrix.float())
                energy = singular.clamp_min(0)
                prob = energy / energy.sum().clamp_min(1e-8)
                entropy = -(prob * prob.clamp_min(1e-8).log()).sum()
                cos_means.append(torch.stack(per_image_cos).mean())
                cos_maxs.append(torch.stack(per_image_cos_max).max())
                variances.append(torch.stack(per_image_var).mean())
                effective_ranks.append(torch.exp(entropy))
                sv_ratios.append((singular[0] ** 2) / singular.pow(2).sum().clamp_min(1e-8))
            return {
                f"{prefix}_pairwise_cos_mean": torch.stack(cos_means).detach(),
                f"{prefix}_pairwise_cos_max": torch.stack(cos_maxs).detach(),
                f"{prefix}_variance_across_patches": torch.stack(variances).detach(),
                f"{prefix}_effective_rank": torch.stack(effective_ranks).detach(),
                f"{prefix}_top1_energy_ratio": torch.stack(sv_ratios).detach(),
            }

        out = {}
        out.update(summarize(raw_query, "raw_query"))
        out.update(summarize(local_query, "local_query"))
        out.update(summarize(query, "final_query"))
        bypass = torch.zeros_like(query) if local_bypass_residual is None else local_bypass_residual.float()
        learned = query if learned_query is None else learned_query.float()
        learned_norm = learned.detach().float().norm(dim=-1).clamp_min(1e-6)
        bypass_norm = bypass.float().norm(dim=-1)
        bypass_ratio = bypass_norm / learned_norm
        out.update({
            "local_bypass_norm_mean": bypass_norm.mean().detach(),
            "local_bypass_to_learned_ratio_mean": bypass_ratio.mean().detach(),
            "local_bypass_to_learned_ratio_max": bypass_ratio.max().detach(),
        })
        out["query_pairwise_cos_mean_across_patches"] = out["final_query_pairwise_cos_mean"]
        out["query_pairwise_cos_max_across_patches"] = out["final_query_pairwise_cos_max"]
        out["query_variance_across_patches"] = out["final_query_variance_across_patches"]
        out["query_effective_rank"] = out["final_query_effective_rank"]
        out["query_singular_value_ratio"] = out["final_query_top1_energy_ratio"]
        out["per_factor_logit_std_across_patches"] = logits.float().std(dim=2, unbiased=False).mean(dim=1).detach()
        return out

    @staticmethod
    def _diagnostic_mask(valid_patch_mask: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor | None:
        if valid_patch_mask is None:
            return None
        mask = valid_patch_mask.to(device=values.device, dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(values.shape[0], -1, -1)
        if mask.ndim != 3 or tuple(mask.shape) != tuple(values.shape[:3]):
            raise ValueError("valid_patch_mask must be [B,P] or [G,B,P]")
        return mask

    @staticmethod
    def legacy_query_patch_diagnostics(query: torch.Tensor, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        groups, batch, patches, rank_dim = query.shape
        cos_means = []
        cos_maxs = []
        effective_ranks = []
        sv_ratios = []
        for group in range(groups):
            q = query[group].float()
            q_norm = F.normalize(q, dim=-1)
            gram = torch.einsum("bpd,bqd->bpq", q_norm, q_norm)
            if patches > 1:
                mask = ~torch.eye(patches, device=query.device, dtype=torch.bool)
                offdiag = gram[:, mask]
                cos_means.append(offdiag.mean())
                cos_maxs.append(offdiag.abs().max())
            else:
                zero = q.sum() * 0.0
                cos_means.append(zero)
                cos_maxs.append(zero)
            flat = q.reshape(batch * patches, rank_dim)
            flat = flat - flat.mean(dim=0, keepdim=True)
            singular = torch.linalg.svdvals(flat)
            energy = singular.clamp_min(0)
            prob = energy / energy.sum().clamp_min(1e-8)
            entropy = -(prob * prob.clamp_min(1e-8).log()).sum()
            effective_ranks.append(torch.exp(entropy))
            sv_ratios.append(singular[0] / singular.sum().clamp_min(1e-8))
        return {
            "query_pairwise_cos_mean_across_patches": torch.stack(cos_means).detach(),
            "query_pairwise_cos_max_across_patches": torch.stack(cos_maxs).detach(),
            "query_variance_across_patches": query.float().var(dim=2, unbiased=False).mean(dim=(1, 2)).detach(),
            "query_effective_rank": torch.stack(effective_ranks).detach(),
            "query_singular_value_ratio": torch.stack(sv_ratios).detach(),
            "per_factor_logit_std_across_patches": logits.float().std(dim=2, unbiased=False).mean(dim=1).detach(),
        }

    @staticmethod
    def topk_frequency(topk_indices: torch.Tensor, num_factors: int) -> torch.Tensor:
        selected = F.one_hot(topk_indices.long(), num_classes=num_factors).float()
        return selected.mean(dim=(1, 2, 3))

    @torch.no_grad()
    def update_load_bias(self, topk_frequency: torch.Tensor) -> None:
        target = torch.full_like(topk_frequency, 1.0 / float(self.num_factors))
        self.ema_topk_usage.mul_(self.load_bias_momentum).add_(
            topk_frequency.to(self.ema_topk_usage.device) * (1.0 - self.load_bias_momentum)
        )
        self.load_bias.add_(self.load_bias_step * (target - self.ema_topk_usage))
        self.load_bias.clamp_(min=-self.load_bias_max, max=self.load_bias_max)

    @staticmethod
    def aggregate_global(probabilities: torch.Tensor, factor_bank: torch.Tensor) -> torch.Tensor:
        """Return image-wise text [G,B,768,2] from the one shared factor bank."""
        routing_global = probabilities.float().mean(dim=2)
        text_global = torch.einsum("gbm,gbmds->gbds", routing_global, factor_bank.float())
        return F.normalize(text_global, dim=2)

    @staticmethod
    def local_text(probabilities: torch.Tensor, factor_bank: torch.Tensor) -> torch.Tensor:
        """Return patch-local normal/abnormal text [G,B,P,768,2]."""
        text = torch.einsum("gbpm,gbmds->gbpds", probabilities.float(), factor_bank.float())
        return F.normalize(text, dim=3)

    @staticmethod
    def unique_topk_pair_counts(topk_indices: torch.Tensor) -> torch.Tensor:
        groups = topk_indices.shape[0]
        counts = []
        pairs = torch.sort(topk_indices.detach().long(), dim=-1).values
        for group in range(groups):
            flattened = pairs[group].reshape(-1, pairs.shape[-1])
            counts.append(torch.unique(flattened, dim=0).shape[0])
        return torch.tensor(counts, device=topk_indices.device, dtype=torch.long)

    @staticmethod
    def concept_key_diagnostics(concept_keys: torch.Tensor, prefix: str = "concept_key") -> Dict[str, torch.Tensor]:
        keys = F.normalize(concept_keys.float(), dim=-1)
        cosine = keys @ keys.T
        offdiag = cosine[~torch.eye(cosine.shape[0], device=cosine.device, dtype=torch.bool)]
        distances = torch.cdist(concept_keys.float(), concept_keys.float())
        offdiag_l2 = distances[~torch.eye(distances.shape[0], device=distances.device, dtype=torch.bool)]
        return {
            f"{prefix}_cosine": cosine.detach(),
            f"{prefix}_cos_mean": offdiag.mean().detach(),
            f"{prefix}_cos_max": offdiag.abs().max().detach(),
            f"{prefix}_l2_min": offdiag_l2.min().detach(),
            f"{prefix}_norm": concept_keys.float().norm(dim=-1).detach(),
        }

    @staticmethod
    def _probability_diagnostics(probabilities: torch.Tensor) -> Dict[str, torch.Tensor]:
        probabilities = probabilities.float()
        factors = probabilities.shape[-1]
        usage = probabilities.mean(dim=(1, 2))
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy.mean(dim=(1, 2)) / torch.log(torch.tensor(float(factors), device=probabilities.device))
        return {
            "usage": usage.detach(),
            "entropy": entropy.detach(),
            "top1_share": usage.max(dim=-1).values.detach(),
            "dead_factors": (usage < 0.01).sum(dim=-1).detach(),
            "max_factor_usage": usage.max(dim=-1).values.detach(),
        }

    @staticmethod
    def diagnostics(
        prediction_probabilities: torch.Tensor,
        dense_probabilities: torch.Tensor | None = None,
        sparse_probabilities: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        prediction = PatchRouter._probability_diagnostics(prediction_probabilities)
        output = {
            "factor_usage": prediction["usage"],
            "normalized_entropy": prediction["entropy"],
            "top1_share": prediction["top1_share"],
            "dead_factors": prediction["dead_factors"],
            "max_factor_usage": prediction["max_factor_usage"],
            "prediction_factor_usage": prediction["usage"],
            "prediction_normalized_entropy": prediction["entropy"],
        }
        if dense_probabilities is not None:
            dense = PatchRouter._probability_diagnostics(dense_probabilities)
            output["dense_factor_usage"] = dense["usage"]
            output["dense_normalized_entropy"] = dense["entropy"]
        if sparse_probabilities is not None:
            sparse = PatchRouter._probability_diagnostics(sparse_probabilities)
            output["sparse_factor_usage"] = sparse["usage"]
            output["sparse_normalized_entropy"] = sparse["entropy"]
        if topk_indices is not None:
            output["selected_topk_frequency"] = PatchRouter.topk_frequency(
                topk_indices, prediction_probabilities.shape[-1]
            ).detach()
            output["unique_topk_pairs"] = PatchRouter.unique_topk_pair_counts(topk_indices).detach()
        return output
