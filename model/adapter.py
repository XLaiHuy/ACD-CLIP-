import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torch import nn
from torch.utils.checkpoint import checkpoint

from .adapter_modules import TextLoraAdapter, MLPAdapter, ConvLoraAdapter, DFGSS2DResidualBranch
from .h6.model import H6Progress1
from .soft_prompt import SoftPromptLearner


class AddWeight(nn.Module):
    def __init__(
            self,
            image_adapt_weight,
            is_text=False,
            change_x=True,
    ):
        super().__init__()
        d_model = 768 if is_text else 1024
        self.i_w = nn.Parameter(torch.ones(1, 1, d_model) * image_adapt_weight)
        self.change_x = change_x

    def forward(self, x, adapt_out):
        if not self.change_x:
            return x + self.i_w * adapt_out
        return (1 - self.i_w) * x + self.i_w * adapt_out


class ACDCLIP(nn.Module):
    def __init__(
            self,
            clip_model,
            text_adapt_weight: float = 0.15,
            image_adapt_weight: float = 0.15,
            n_groups: int = 3,
            lora_rank: int = 16,
            lora_alpha: float = 2,
            conv_lora_rank: int = 8,
            conv_lora_alpha: float = 2,
            conv_kernel_size_list=(3, 5),
            dfg_mode: str = "mlp",
            dfg_attn_dim: int = 256,
            dfg_attn_tau: float = 4.0,
            use_ss2d_dfg: bool = False,
            dfg_gamma_max: float = 0.2,
            dfg_ss2d_fusion: str = "feature_residual",
            dfg_beta: float = 0.10,
            dfg_beta_schedule: str = "fixed",
            dfg_beta_target: float | None = None,
            dfg_beta_current: float | None = None,
            dfg_weight_residual_fp32: bool = True,
            use_soft_prompt: bool = False,
            soft_prompt_ctx_len: int = 4,
            soft_prompt_init: str = "phrase",
            soft_prompt_init_phrase: str = "a photo of a",
            h6_progress: int = 0,
            h6_num_factors: int = 4,
            h6_top_k: int = 2,
            h6_bank_dim: int = 256,
            h6_router_dim: int = 128,
            h6_router_temperature: float = 1.0,
            h6_router_soft_epochs: int = 2,
            h6_sparse_transition_epochs: int = 1,
            h6_load_bias_enabled: bool = False,
            h6_load_bias_momentum: float = 0.9,
            h6_load_bias_step: float = 0.001,
            h6_load_bias_max: float = 0.03,
            h6_vae_hidden_dim: int = 512,
            h6_vae_latent_dim: int = 256,
            h6_vae_class_ratio: float = 0.25,
            h6_slot_init_enabled: bool = False,
            h6_slot_init_scale: float = 0.02,
            h6_slot_init_seed_offset: int = 6100,
            h6_factor_grad_diagnostics: bool = False,
            h6_late_factor_identity_enabled: bool = False,
            h6_factor_id_scale: float = 0.02,
            h6_factor_id_max_ratio: float = 0.05,
            h6_factor_generator_specialization_enabled: bool = False,
            h6_factor_head_init_scale: float = 1e-3,
            h6_factor_local_dynamic_mix: float = 0.0,
            h6_router_query_mode: str = "local_global_bypass",
            h6_router_query_global_weight: float = 0.10,
            h6_router_local_bypass_scale: float = 0.10,
            h6_router_local_bypass_max_ratio: float = 0.20,
            h6_router_local_projection_seed_offset: int = 7200,
            h6_router_key_anchor_enabled: bool = True,
            h6_router_key_anchor_seed_offset: int = 7300,
            h6_router_key_adaptation_initial_ratio: float = 0.10,
            h6_router_key_adaptation_max_ratio: float = 0.25,
            h6_factor_context_anchor_enabled: bool = True,
            h6_factor_context_anchor_seed_offset: int = 7400,
            h6_factor_context_adaptation_initial_ratio: float = 0.10,
            h6_factor_context_adaptation_max_ratio: float = 0.25,
            h6_factor_identity_tangent_projection_enabled: bool = True,
            lambda_h6_dynamic_mean_anchor: float = 0.001,
            h6_dynamic_mean_anchor_min_cosine: float = 0.70,
            h6_dynamic_mean_anchor_start_epoch: int = 4,
            h6_dynamic_mean_anchor_warmup_epochs: int = 3,
            h6_router_teacher_mode: str = "raw_cosine",
            h6_progress_version: str = "P1-v6",
            h6_local_factor_mode: str = "center_spread",
            h6_local_center_mix: float = 0.05,
            h6_local_factor_spread: float = 0.10,
            h6_expert_enabled: bool = False,
            h6_expert_bottleneck: int = 64,
            h6_expert_fofs_seed_offset: int = 7500,
            h6_expert_state_condition_scale: float = 0.25,
            h6_expert_scale_target: float = 0.10,
            h6_expert_scale_start_epoch: int = 1,
            h6_expert_scale_warmup_epochs: int = 6,
            h6_expert_max_relative_ratio: float = 0.10,
            **kwargs,
    ):
        super().__init__()
        assert n_groups in [2, 3, 4, 6], "n_groups must be one of [2, 3, 4, 6]"
        assert dfg_mode in ["mlp", "attn"], "dfg_mode must be one of ['mlp', 'attn']"
        assert dfg_attn_tau > 0, "dfg_attn_tau must be positive"
        assert dfg_gamma_max >= 0, "dfg_gamma_max must be non-negative"
        assert dfg_ss2d_fusion in ["feature_residual", "weight_residual"], (
            "dfg_ss2d_fusion must be one of ['feature_residual', 'weight_residual']"
        )
        assert dfg_beta_schedule in ["fixed", "warmup010"], (
            "dfg_beta_schedule must be one of ['fixed', 'warmup010']"
        )
        assert 0 <= dfg_beta <= 1, "dfg_beta must be in [0, 1]"
        if dfg_beta_target is None:
            dfg_beta_target = dfg_beta
        if dfg_beta_current is None:
            dfg_beta_current = dfg_beta
        assert 0 <= dfg_beta_target <= 1, "dfg_beta_target must be in [0, 1]"
        assert 0 <= dfg_beta_current <= 1, "dfg_beta_current must be in [0, 1]"
        if use_ss2d_dfg and dfg_mode != "attn":
            raise ValueError("use_ss2d_dfg is only supported when dfg_mode='attn'")
        if not use_ss2d_dfg and dfg_ss2d_fusion != "feature_residual":
            raise ValueError("dfg_ss2d_fusion='weight_residual' requires use_ss2d_dfg=True")
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        text_step = 12 // n_groups
        image_step = 24 // n_groups
        self.image_levels = [t for t in range(image_step, 24 + 1, image_step)]
        self.text_levels = [t for t in range(text_step, 12 + 1, text_step)]
        self.n_groups = n_groups
        self.image_adapt_weight = image_adapt_weight
        self.text_adapt_weight = text_adapt_weight
        self.dfg_mode = dfg_mode
        self.dfg_attn_dim = dfg_attn_dim
        self.dfg_attn_tau = dfg_attn_tau
        self.use_ss2d_dfg = use_ss2d_dfg
        self.dfg_gamma_max = dfg_gamma_max
        self.dfg_ss2d_fusion = dfg_ss2d_fusion
        self.dfg_beta = dfg_beta_current
        self.dfg_beta_schedule = dfg_beta_schedule
        self.dfg_beta_target = dfg_beta_target
        self.dfg_weight_residual_fp32 = dfg_weight_residual_fp32
        self.use_soft_prompt = use_soft_prompt
        self.soft_prompt_ctx_len = soft_prompt_ctx_len
        self.soft_prompt_init = soft_prompt_init
        self.soft_prompt_init_phrase = soft_prompt_init_phrase
        self._frozen_anchor_cache = {}
        self._last_dfg_stats = {}

        image_adapt_weights = nn.ModuleList(
            [AddWeight(image_adapt_weight) for _ in range(n_groups)]
        )
        image_lora_adapters = nn.ModuleList(
            [
                ConvLoraAdapter(1024, 1024, lora_rank, lora_alpha, conv_lora_rank, conv_lora_alpha,
                                conv_kernel_size_list)
                for _ in
                range(n_groups)
            ]
        )
        seg_proj = nn.ModuleList(
            [MLPAdapter(1024, 768, 256) for _ in range(n_groups)]
        )
        det_proj = nn.ModuleList(
            MLPAdapter(1024, 768, 256) for _ in range(n_groups)
        )
        seg_image_layer_norms = nn.ModuleList(
            [nn.LayerNorm(768) for _ in range(n_groups)]
        )
        det_image_layer_norms = nn.ModuleList(
            [nn.LayerNorm(768) for _ in range(n_groups)]
        )
        image_adapter_dict = {
            "m_i_w": image_adapt_weights,
            "lora_adapters": image_lora_adapters,
            "seg_layer_norms": seg_image_layer_norms,
            "seg_proj": seg_proj,
            "det_layer_norms": det_image_layer_norms,
            "det_proj": det_proj,
        }
        if dfg_mode == "mlp":
            # Original DFG: use GAP visual tokens to predict text-level weights.
            image_adapter_dict["vision_text_gate"] = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(768, 256),
                    nn.GELU(),
                    nn.Linear(256, n_groups * 2)
                ) for _ in range(n_groups)
            ])
        else:
            # Phase 1A DFG: W_Q/W_K only compute attention scores; values stay in CLIP text space.
            image_adapter_dict["vision_text_q"] = nn.ModuleList(
                [nn.Linear(768, dfg_attn_dim, bias=False) for _ in range(n_groups)]
            )
            image_adapter_dict["vision_text_k"] = nn.ModuleList(
                [nn.Linear(768, dfg_attn_dim, bias=False) for _ in range(n_groups)]
            )
            if use_ss2d_dfg:
                image_adapter_dict["dfg_ss2d_branches"] = nn.ModuleList(
                    [DFGSS2DResidualBranch(768) for _ in range(n_groups)]
                )
                image_adapter_dict["dfg_raw_gamma"] = nn.ParameterList(
                    [nn.Parameter(torch.zeros(())) for _ in range(n_groups)]
                )
        self.image_adapter = nn.ModuleDict(image_adapter_dict)
        if dfg_mode == "attn":
            self._init_dfg_attention()
        text_adapt_weights = nn.ModuleList(
            [AddWeight(text_adapt_weight, is_text=True) for _ in range(n_groups)]
        )
        text_lora_adapters = nn.ModuleList(
            [TextLoraAdapter(768, 768, r=lora_rank, alpha=lora_alpha) for _ in range(n_groups)]
        )
        text_layer_norms = nn.ModuleList(
            [nn.LayerNorm(768) for _ in range(n_groups)]
        )
        self.text_adapter = nn.ModuleDict(
            {
                "m_t_w": text_adapt_weights,
                "layer_norms": text_layer_norms,
                "lora_adapters": text_lora_adapters,
            }
        )
        init_phrase = soft_prompt_init_phrase if soft_prompt_init == "phrase" else "random"
        self.soft_prompt = SoftPromptLearner(
            ctx_len=soft_prompt_ctx_len,
            text_dim=768,
            init_phrase=init_phrase,
            clip_model=clip_model,
        )
        if h6_progress not in (0, 1):
            raise ValueError("this branch implements only H6 Progress 1; use h6_progress=0 or 1")
        self.h6_enabled = bool(h6_progress == 1)
        self.h6_progress = int(h6_progress)
        self.h6 = None
        if self.h6_enabled:
            self.h6 = H6Progress1(
                n_groups=n_groups,
                num_factors=h6_num_factors,
                top_k=h6_top_k,
                role_topology=kwargs.get("h6_role_topology", "flat"),
                role_teacher_scale=kwargs.get("h6_role_teacher_scale", None),
                intrinsic_factor_responsibility=kwargs.get("h6_intrinsic_factor_responsibility", False),
                prediction_routing=kwargs.get("h6_prediction_routing", "dense"),
                bank_dim=h6_bank_dim,
                router_dim=h6_router_dim,
                router_temperature=h6_router_temperature,
                router_soft_epochs=h6_router_soft_epochs,
                sparse_transition_epochs=h6_sparse_transition_epochs,
                load_bias_enabled=h6_load_bias_enabled,
                load_bias_momentum=h6_load_bias_momentum,
                load_bias_step=h6_load_bias_step,
                load_bias_max=h6_load_bias_max,
                vae_hidden_dim=h6_vae_hidden_dim,
                vae_latent_dim=h6_vae_latent_dim,
                vae_class_ratio=h6_vae_class_ratio,
                slot_init_enabled=h6_slot_init_enabled,
                slot_init_scale=h6_slot_init_scale,
                slot_init_seed_offset=h6_slot_init_seed_offset,
                factor_grad_diagnostics=h6_factor_grad_diagnostics,
                diagnostics_mode=kwargs.get("diagnostics_mode", "light"),
                diagnostics_interval=kwargs.get("diagnostics_interval", 1),
                test_rho_override=kwargs.get("test_rho_override", None),
                late_factor_identity_enabled=h6_late_factor_identity_enabled,
                factor_id_scale=h6_factor_id_scale,
                factor_id_max_ratio=h6_factor_id_max_ratio,
                factor_generator_specialization_enabled=h6_factor_generator_specialization_enabled,
                factor_head_init_scale=h6_factor_head_init_scale,
                factor_local_dynamic_mix=h6_factor_local_dynamic_mix,
                cluster_responsibility_enabled=kwargs.get("h6_cluster_responsibility", False),
                cluster_temperature=kwargs.get("h6_cluster_temperature", 0.10),
                router_query_mode=h6_router_query_mode,
                router_query_global_weight=h6_router_query_global_weight,
                router_local_bypass_scale=h6_router_local_bypass_scale,
                router_local_bypass_max_ratio=h6_router_local_bypass_max_ratio,
                router_local_projection_seed_offset=h6_router_local_projection_seed_offset,
                router_key_anchor_enabled=h6_router_key_anchor_enabled,
                router_key_anchor_seed_offset=h6_router_key_anchor_seed_offset,
                router_key_adaptation_initial_ratio=h6_router_key_adaptation_initial_ratio,
                router_key_adaptation_max_ratio=h6_router_key_adaptation_max_ratio,
                router_boundary_mode=kwargs.get("h6_router_boundary_mode", "none"),
                router_boundary_trust_scale=kwargs.get("h6_router_boundary_trust_scale", None),
                factor_context_anchor_enabled=h6_factor_context_anchor_enabled,
                factor_context_anchor_seed_offset=h6_factor_context_anchor_seed_offset,
                factor_context_adaptation_initial_ratio=h6_factor_context_adaptation_initial_ratio,
                factor_context_adaptation_max_ratio=h6_factor_context_adaptation_max_ratio,
                factor_identity_tangent_projection_enabled=h6_factor_identity_tangent_projection_enabled,
                lambda_dynamic_mean_anchor=lambda_h6_dynamic_mean_anchor,
                dynamic_mean_anchor_min_cosine=h6_dynamic_mean_anchor_min_cosine,
                dynamic_mean_anchor_start_epoch=h6_dynamic_mean_anchor_start_epoch,
                dynamic_mean_anchor_warmup_epochs=h6_dynamic_mean_anchor_warmup_epochs,
                router_teacher_mode=h6_router_teacher_mode,
                progress_version=h6_progress_version,
                local_factor_mode=h6_local_factor_mode,
                local_center_mix=h6_local_center_mix,
                local_factor_spread=h6_local_factor_spread,
                expert_enabled=h6_expert_enabled,
                expert_bottleneck=h6_expert_bottleneck,
                expert_fofs_seed_offset=h6_expert_fofs_seed_offset,
                expert_state_condition_scale=h6_expert_state_condition_scale,
                expert_scale_target=h6_expert_scale_target,
                expert_scale_start_epoch=h6_expert_scale_start_epoch,
                expert_scale_warmup_epochs=h6_expert_scale_warmup_epochs,
                expert_max_relative_ratio=h6_expert_max_relative_ratio,
                text_dim=768,
                ctx_len=soft_prompt_ctx_len,
                phase4v_bottleneck=kwargs.get("phase4v_bottleneck", 64),
                phase4v_lambda=kwargs.get("phase4v_lambda", 0.05),
            )

    def _init_dfg_attention(self):
        if self.dfg_attn_dim == 768:
            for q_proj, k_proj in zip(
                    self.image_adapter["vision_text_q"],
                    self.image_adapter["vision_text_k"],
            ):
                nn.init.eye_(q_proj.weight)
                nn.init.eye_(k_proj.weight)
            return
        for q_proj, k_proj in zip(
                self.image_adapter["vision_text_q"],
                self.image_adapter["vision_text_k"],
        ):
            nn.init.xavier_uniform_(q_proj.weight)
            nn.init.xavier_uniform_(k_proj.weight)

    def set_dfg_beta(self, beta: float):
        if not 0 <= beta <= 1:
            raise ValueError(f"dfg beta must be in [0, 1], got {beta}")
        self.dfg_beta = float(beta)

    def forward_original(self, x, modality="visual"):
        if modality == "visual":
            cls_features, patch_features = self.clipmodel.encode_image(x, [24])
            patch_features = [
                self.clipmodel.visual._global_pool(t)[1] for t in patch_features
            ]
            patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
            patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
            return patch_features, cls_features
        else:
            raise ValueError("modality must be visual")


    def frozen_router_reference(self, image: torch.Tensor) -> torch.Tensor:
        """Frozen adapter-independent final CLIP patches for Router-only routing."""
        with torch.no_grad():
            tokens = self.image_encoder.conv1(image)
            tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1).permute(0, 2, 1)
            tokens = torch.cat([self.image_encoder.class_embedding.to(tokens.dtype) + torch.zeros(tokens.shape[0], 1, tokens.shape[-1], dtype=tokens.dtype, device=tokens.device), tokens], dim=1)
            tokens = self.image_encoder.ln_pre(self.image_encoder.patch_dropout(tokens + self.image_encoder.positional_embedding.to(tokens.dtype))).permute(1, 0, 2)
            for resblock in self.image_encoder.transformer.resblocks:
                tokens, _ = resblock(tokens, attn_mask=None)
            patches = self.image_encoder.ln_post(tokens[1:].permute(1, 0, 2))
            if self.image_encoder.proj is not None:
                patches = patches @ self.image_encoder.proj
        return F.normalize(patches.float(), dim=-1).detach()

    def forward(self, x, return_phase4_features: bool = False):
        input_image = x
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        group_outs = []
        cls24 = None
        for i in range(24):
            resblock = self.image_encoder.transformer.resblocks[i]
            if self.image_encoder.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x, attn = checkpoint(resblock, x, None, None, None, use_reentrant=False)
            else:
                x, attn = resblock(x, attn_mask=None)
            if i == 23:
                # Reuse the final block CLS, never invoke the visual tower a second time.
                cls24 = self.image_encoder.ln_post(x[0])
                if self.image_encoder.proj is not None:
                    cls24 = cls24 @ self.image_encoder.proj
                cls24 = F.normalize(cls24.float(), dim=-1)
            # [1370, bs, 1024]
            index = -1
            for j in range(self.n_groups):
                if i + 1 == self.image_levels[j]:
                    index = j
                    break
            if index != -1:
                t = x[1:, :, :]
                adapt_out = self.image_adapter["lora_adapters"][index](t)
                adapt_out = (
                        adapt_out
                        * t.norm(dim=-1, keepdim=True)
                        / adapt_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                )
                # [1369, bs, 1024]
                group_out = self.image_adapter["m_i_w"][index](t, adapt_out)
                # group_out = t * (1 - self.image_adapt_weight) + adapt_out * self.image_adapt_weight
                group_outs.append(group_out)
                x = torch.cat(
                    [
                        x[0, :, :].unsqueeze(0),
                        group_out
                    ],
                    dim=0,
                )

        # [batch_size, seq_len, dim]
        group_tokens = [t.permute(1, 0, 2) for t in group_outs]  # [bs, 1369, 1024]

        # 1: Segmentation Tokens
        seg_tokens_proj = [
            self.image_adapter["seg_proj"][i](t) for i, t in enumerate(group_tokens)
        ]  # 投影到分割空间
        seg_tokens_norm = [
            self.image_adapter["seg_layer_norms"][i](t) for i, t in enumerate(seg_tokens_proj)
        ]  # 层归一化
        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens_norm]  # L2归一化

        # 2: Detection Tokens
        det_tokens_proj = [
            self.image_adapter["det_proj"][i](t) for i, t in enumerate(group_tokens)
        ]
        det_tokens_norm = [
            self.image_adapter["det_layer_norms"][i](t) for i, t in enumerate(det_tokens_proj)
        ]
        det_tokens = [F.normalize(t, dim=-1).mean(1) for t in det_tokens_norm]  # L2归一化 + 全局平均池化

        if return_phase4_features:
            if cls24 is None:
                raise RuntimeError("failed to capture the final visual CLS token")
            output = {
                "seg_tokens": seg_tokens,
                "seg_tokens_pre_l2": seg_tokens_norm,
                "det_tokens": det_tokens,
                "cls24": cls24,
            }
            if self.h6_enabled and self.h6 is not None and self.h6.router.boundary_mode == "frozen_reference_direct_router":
                output["frozen_router_reference"] = self.frozen_router_reference(input_image)
            return output
        return seg_tokens, det_tokens

    def vision_text_fusion_gate_seg(
            self,
            vision_tokens: torch.Tensor,
            text_features: torch.Tensor,
            img_size: int = 518,
            test_mode: bool = False,
            domain: str = "Industrial",
            h6_patch_logits: torch.Tensor | None = None,
            return_details: bool = False,
    ):
        """
        Fuse vision and text features using a gating mechanism.
        :param vision_tokens: vision tokens from the image encoder. [n_groups, bs, patch_num, 768]
        :param text_features: text features from the text encoder. [n_groups, bs, 768, 2]
        :return: fused seg features.
        """
        B, patch_size, _ = vision_tokens.shape[1:]
        H = int(np.sqrt(patch_size))
        group_seg_preds = []
        base_group_logits_list = []
        for i in range(self.n_groups):
            img_feat = vision_tokens[i]  # [bs, patch_num, 768]
            img_feat = 10 * img_feat
            # [n_groups, bs, 768, 2] -> [bs, n_groups, 768, 2]
            group_text_features = text_features.permute(1, 0, 2, 3)
            dfg_weights = self.compute_dfg_weights(
                vision_tokens[i], group_text_features, i
            )
            group_text_features = self.apply_dfg_weights(
                group_text_features,
                dfg_weights["normal"],
                dfg_weights["abnormal"],
            )
            fused_feature = torch.matmul(img_feat, group_text_features)  # [bs, patch_num, 2]
            if return_details:
                base_group_logits_list.append(fused_feature.clone())

            if h6_patch_logits is not None:
                if not self.h6_enabled or self.h6 is None:
                    raise ValueError("H6 residual logits require an H6-enabled ACDCLIP model")
                local_logits = h6_patch_logits[i]
                if tuple(local_logits.shape) != (B, patch_size):
                    raise ValueError(
                        f"H6 patch logits at level {i} must be [B, P]={B, patch_size}, got {tuple(local_logits.shape)}"
                    )
                correction = self.h6.rho_values()[i].to(local_logits.dtype) * local_logits
                # Adding to the abnormal logit is exactly an additive correction to
                # the Phase2B normal-vs-abnormal patch logit, while retaining all
                # existing DFG, smoothing, interpolation, and softmax behavior.
                fused_feature = torch.stack(
                    [fused_feature[..., 0], fused_feature[..., 1].float() + correction.float()], dim=-1
                )
            seg_logits = fused_feature.permute(0, 2, 1).view(B, 2, H, H)  # [bs, 2, H, H]
            group_seg_preds.append(seg_logits)  # [bs, 2, H, H]
        if test_mode:
            sigma = 1 if domain == "Industrial" else 1.5
            kernel_size = 7 if domain == "Industrial" else 9
            group_seg_preds = [
                gaussian_blur2d(
                    seg_pred,
                    (kernel_size, kernel_size),
                    (sigma, sigma)
                ) for seg_pred in group_seg_preds
            ]
        group_seg_preds = [
            F.interpolate(
                seg_pred,
                size=img_size,
                mode="bilinear",
                align_corners=True
            ).unsqueeze(0) for seg_pred in group_seg_preds
            # [1, bs, 2, img_size, img_size]
        ]  # [1, bs, 2, img_size, img_size] * n_groups
        all_group_preds = torch.cat(group_seg_preds, dim=0)  # [n_groups, bs, 2, img_size, img_size]
        final_seg_pred = torch.mean(all_group_preds, dim=0)  # [bs, 2, img_size, img_size]
        final_seg_pred = F.softmax(final_seg_pred, dim=1)  # [bs, 2, img_size, img_size]
        if test_mode:
            # [bs, img_size, img_size]
            final_seg_pred = final_seg_pred[:, 1, :, :]
            # final_seg_pred = (final_seg_pred[:, 1, :, :] + 1 - final_seg_pred[:, 0, :, :]) / 2
            
        if return_details:
            base_group_logits = torch.stack(base_group_logits_list, dim=0) # [G, B, P, 2]
            base_abnormal_minus_normal = base_group_logits[..., 1] - base_group_logits[..., 0] # [G, B, P]
            return final_seg_pred, base_group_logits, base_abnormal_minus_normal
            
        return final_seg_pred

    def compute_dfg_weights(
            self,
            img_feat: torch.Tensor,
            group_text_features: torch.Tensor,
            group_index: int,
    ):
        """Compute auditable base DFG weights without applying them to text.

        Returned weights have shape ``[B, n_groups]`` for Normal and
        Abnormal independently. The method preserves the historical Phase2B
        computation for both MLP and attention DFG modes.
        """
        if group_text_features.ndim != 4 or group_text_features.shape[-1] != 2:
            raise ValueError("group_text_features must be [B,G,D,2]")
        if self.dfg_mode == "mlp":
            gate_weights = self.image_adapter["vision_text_gate"][group_index](
                img_feat.mean(dim=1, keepdim=True)
            ).squeeze(1)
            gate_weights = gate_weights.view(img_feat.shape[0], self.n_groups, 2)
            gate_weights = F.softmax(gate_weights, dim=1)
            return {
                "normal": gate_weights[..., 0],
                "abnormal": gate_weights[..., 1],
            }

        v_gap = img_feat.mean(dim=1)  # [bs, 768]
        v_global = v_gap
        gamma = None
        ss2d_ratio = None
        v_ss2d = None
        if self.use_ss2d_dfg:
            v_ss2d = self.image_adapter["dfg_ss2d_branches"][group_index](img_feat)
            raw_gamma = self.image_adapter["dfg_raw_gamma"][group_index]
            gamma = self.dfg_gamma_max * torch.tanh(raw_gamma)
            if self.dfg_ss2d_fusion == "feature_residual":
                v_global = v_gap + gamma * v_ss2d
            ss2d_ratio = (gamma.abs() * v_ss2d.norm(dim=-1).mean()) / v_gap.norm(dim=-1).mean().clamp_min(1e-6)
        text_normal = group_text_features[..., 0]  # [bs, n_groups, 768]
        text_abnormal = group_text_features[..., 1]  # [bs, n_groups, 768]
        k_normal = self.image_adapter["vision_text_k"][group_index](text_normal)  # [bs, n_groups, d_attn]
        k_abnormal = self.image_adapter["vision_text_k"][group_index](text_abnormal)

        scale = (self.dfg_attn_dim ** 0.5) * self.dfg_attn_tau
        q = None
        q_gap = None
        q_ss2d = None
        scores_normal = None
        scores_abnormal = None
        scores_gap_normal = None
        scores_gap_abnormal = None
        scores_ss2d_normal = None
        scores_ss2d_abnormal = None
        if self.use_ss2d_dfg and self.dfg_ss2d_fusion == "weight_residual":
            q_gap = self.image_adapter["vision_text_q"][group_index](v_gap)
            q_ss2d = self.image_adapter["vision_text_q"][group_index](v_ss2d)
            if self.dfg_weight_residual_fp32:
                q_gap_for_attn = q_gap.float()
                q_ss2d_for_attn = q_ss2d.float()
                k_normal_for_attn = k_normal.float()
                k_abnormal_for_attn = k_abnormal.float()
            else:
                q_gap_for_attn = q_gap
                q_ss2d_for_attn = q_ss2d
                k_normal_for_attn = k_normal
                k_abnormal_for_attn = k_abnormal
            scores_gap_normal = self._attention_scores(q_gap_for_attn, k_normal_for_attn, scale)
            scores_gap_abnormal = self._attention_scores(q_gap_for_attn, k_abnormal_for_attn, scale)
            scores_ss2d_normal = self._attention_scores(q_ss2d_for_attn, k_normal_for_attn, scale)
            scores_ss2d_abnormal = self._attention_scores(q_ss2d_for_attn, k_abnormal_for_attn, scale)
            weights_gap_normal = F.softmax(scores_gap_normal, dim=1)
            weights_gap_abnormal = F.softmax(scores_gap_abnormal, dim=1)
            weights_ss2d_normal = F.softmax(scores_ss2d_normal, dim=1)
            weights_ss2d_abnormal = F.softmax(scores_ss2d_abnormal, dim=1)
            weights_normal = (1 - self.dfg_beta) * weights_gap_normal + self.dfg_beta * weights_ss2d_normal
            weights_abnormal = (1 - self.dfg_beta) * weights_gap_abnormal + self.dfg_beta * weights_ss2d_abnormal
            weights_normal = weights_normal.to(dtype=text_normal.dtype)
            weights_abnormal = weights_abnormal.to(dtype=text_abnormal.dtype)
        else:
            q = self.image_adapter["vision_text_q"][group_index](v_global)  # [bs, d_attn]
            scores_normal = self._attention_scores(q, k_normal, scale)
            scores_abnormal = self._attention_scores(q, k_abnormal, scale)
            weights_normal = F.softmax(scores_normal, dim=1)
            weights_abnormal = F.softmax(scores_abnormal, dim=1)
        self._last_dfg_stats[group_index] = {
            "raw_gamma": None if gamma is None else raw_gamma.detach().float().cpu(),
            "gamma": None if gamma is None else gamma.detach().float().cpu(),
            "ss2d_ratio": None if ss2d_ratio is None else ss2d_ratio.detach().float().cpu(),
            "dfg_beta": torch.tensor(self.dfg_beta).float().cpu(),
            "entropy_normal": (-(weights_normal * weights_normal.clamp_min(1e-8).log()).sum(dim=1).mean()).detach().float().cpu(),
            "entropy_abnormal": (-(weights_abnormal * weights_abnormal.clamp_min(1e-8).log()).sum(dim=1).mean()).detach().float().cpu(),
            "weights_normal_sum_error": (weights_normal.sum(dim=1) - 1).abs().max().detach().float().cpu(),
            "weights_abnormal_sum_error": (weights_abnormal.sum(dim=1) - 1).abs().max().detach().float().cpu(),
            "weights_normal": weights_normal.mean(dim=0).detach().float().cpu(),
            "weights_abnormal": weights_abnormal.mean(dim=0).detach().float().cpu(),
            "v_gap_finite": torch.isfinite(v_gap).all().detach().cpu(),
            "v_gap_absmax": v_gap.detach().float().abs().max().cpu(),
            "v_gap_norm": v_gap.detach().float().norm(dim=-1).mean().cpu(),
            "v_ss2d_finite": None if v_ss2d is None else torch.isfinite(v_ss2d).all().detach().cpu(),
            "v_ss2d_absmax": None if v_ss2d is None else v_ss2d.detach().float().abs().max().cpu(),
            "v_ss2d_norm": None if v_ss2d is None else v_ss2d.detach().float().norm(dim=-1).mean().cpu(),
            "q_finite": None if q is None else torch.isfinite(q).all().detach().cpu(),
            "q_absmax": None if q is None else q.detach().float().abs().max().cpu(),
            "q_norm": None if q is None else q.detach().float().norm(dim=-1).mean().cpu(),
            "q_gap_finite": None if q_gap is None else torch.isfinite(q_gap).all().detach().cpu(),
            "q_gap_absmax": None if q_gap is None else q_gap.detach().float().abs().max().cpu(),
            "q_gap_norm": None if q_gap is None else q_gap.detach().float().norm(dim=-1).mean().cpu(),
            "q_ss2d_finite": None if q_ss2d is None else torch.isfinite(q_ss2d).all().detach().cpu(),
            "q_ss2d_absmax": None if q_ss2d is None else q_ss2d.detach().float().abs().max().cpu(),
            "q_ss2d_norm": None if q_ss2d is None else q_ss2d.detach().float().norm(dim=-1).mean().cpu(),
            "k_normal_finite": torch.isfinite(k_normal).all().detach().cpu(),
            "k_normal_absmax": k_normal.detach().float().abs().max().cpu(),
            "k_normal_norm": k_normal.detach().float().norm(dim=-1).mean().cpu(),
            "k_abnormal_finite": torch.isfinite(k_abnormal).all().detach().cpu(),
            "k_abnormal_absmax": k_abnormal.detach().float().abs().max().cpu(),
            "k_abnormal_norm": k_abnormal.detach().float().norm(dim=-1).mean().cpu(),
            "scores_normal_finite": None if scores_normal is None else torch.isfinite(scores_normal).all().detach().cpu(),
            "scores_normal_absmax": None if scores_normal is None else scores_normal.detach().float().abs().max().cpu(),
            "scores_abnormal_finite": None if scores_abnormal is None else torch.isfinite(scores_abnormal).all().detach().cpu(),
            "scores_abnormal_absmax": None if scores_abnormal is None else scores_abnormal.detach().float().abs().max().cpu(),
            "scores_gap_normal_finite": None if scores_gap_normal is None else torch.isfinite(scores_gap_normal).all().detach().cpu(),
            "scores_gap_normal_absmax": None if scores_gap_normal is None else scores_gap_normal.detach().float().abs().max().cpu(),
            "scores_gap_abnormal_finite": None if scores_gap_abnormal is None else torch.isfinite(scores_gap_abnormal).all().detach().cpu(),
            "scores_gap_abnormal_absmax": None if scores_gap_abnormal is None else scores_gap_abnormal.detach().float().abs().max().cpu(),
            "scores_ss2d_normal_finite": None if scores_ss2d_normal is None else torch.isfinite(scores_ss2d_normal).all().detach().cpu(),
            "scores_ss2d_normal_absmax": None if scores_ss2d_normal is None else scores_ss2d_normal.detach().float().abs().max().cpu(),
            "scores_ss2d_abnormal_finite": None if scores_ss2d_abnormal is None else torch.isfinite(scores_ss2d_abnormal).all().detach().cpu(),
            "scores_ss2d_abnormal_absmax": None if scores_ss2d_abnormal is None else scores_ss2d_abnormal.detach().float().abs().max().cpu(),
        }
        return {"normal": weights_normal, "abnormal": weights_abnormal}

    def apply_dfg_weights(
            self,
            group_text_features: torch.Tensor,
            weights_normal: torch.Tensor,
            weights_abnormal: torch.Tensor,
    ) -> torch.Tensor:
        """Apply precomputed cross-level DFG weights and normalize the result."""
        if group_text_features.ndim != 4 or group_text_features.shape[-1] != 2:
            raise ValueError("group_text_features must be [B,G,D,2]")
        expected = group_text_features.shape[:2]
        if weights_normal.shape != expected or weights_abnormal.shape != expected:
            raise ValueError(
                f"DFG weights must be {expected}, got "
                f"{tuple(weights_normal.shape)} and {tuple(weights_abnormal.shape)}"
            )
        text_normal = torch.einsum(
            "bn,bnd->bd", weights_normal, group_text_features[..., 0]
        )
        text_abnormal = torch.einsum(
            "bn,bnd->bd", weights_abnormal, group_text_features[..., 1]
        )
        # Historical Phase2B normalized attention-DFG values, while the
        # original MLP-DFG path used the weighted sum directly.
        if self.dfg_mode != "mlp":
            text_normal = F.normalize(text_normal, dim=-1)
            text_abnormal = F.normalize(text_abnormal, dim=-1)
        return torch.stack([text_normal, text_abnormal], dim=-1)  # [bs, 768, 2]

    def _vision_text_attention_fusion(
            self,
            img_feat: torch.Tensor,
            group_text_features: torch.Tensor,
            group_index: int,
    ):
        """Backward-compatible wrapper around the explicit DFG interface."""
        weights = self.compute_dfg_weights(img_feat, group_text_features, group_index)
        return self.apply_dfg_weights(
            group_text_features, weights["normal"], weights["abnormal"]
        )

    @staticmethod
    def _attention_scores(query: torch.Tensor, key: torch.Tensor, scale: float):
        return torch.einsum("bd,bnd->bn", query, key) / scale

    @staticmethod
    def _attention_weights(query: torch.Tensor, key: torch.Tensor, scale: float):
        return F.softmax(ACDCLIP._attention_scores(query, key, scale), dim=1)

    def get_dfg_diagnostics(self):
        diagnostics = {}
        for group_index, stats in self._last_dfg_stats.items():
            prefix = f"stage{group_index + 1}"
            for key, value in stats.items():
                diagnostics[f"{prefix}_{key}"] = value
        return diagnostics

    def _encode_text_from_embeddings(
            self,
            text,
            x,
            adapt_text=True,
            adapter_layer_norm=True,
            functional_layer_norm=False,
    ):
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        out_features = []
        for i in range(12):
            x, attn = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )
            index = -1
            for j in range(self.n_groups):
                if i + 1 == self.text_levels[j]:
                    index = j
                    break
            if index != -1:
                if adapt_text:
                    adapt_out = self.text_adapter["lora_adapters"][index](x)
                    adapt_out = (
                            adapt_out
                            * x.norm(dim=-1, keepdim=True)
                            / adapt_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    )
                    x = self.text_adapter["m_t_w"][index](x, adapt_out)
                    # x = x * (1 - self.text_adapt_weight) + adapt_out * self.text_adapt_weight
                out_features.append(x)

        indices = text.argmax(dim=-1)
        out_features = [t.permute(1, 0, 2) for t in out_features]
        if adapter_layer_norm:
            out_features = [self.text_adapter["layer_norms"][i](t) for i, t in enumerate(out_features)]
        elif functional_layer_norm:
            out_features = [F.layer_norm(t.float(), (t.shape[-1],)).to(dtype=t.dtype) for t in out_features]
        out_features = [t[torch.arange(t.shape[0], device=t.device), indices] for t in out_features]
        return out_features

    def encode_text(self, text, adapt_text=True):
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(
            cast_dtype
        )  # [batch_size, n_ctx, d_model]
        return self._encode_text_from_embeddings(text, x, adapt_text=adapt_text)

    def encode_frozen_anchor_text(self, text):
        """Encode hard anchors without any trainable text adapter dependency."""
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        with torch.no_grad():
            x = self.clipmodel.token_embedding(text).to(cast_dtype)
            levels = self._encode_text_from_embeddings(
                text,
                x,
                adapt_text=False,
                adapter_layer_norm=False,
                functional_layer_norm=True,
            )
        return [F.normalize(level.detach().float(), dim=-1) for level in levels]

    def encode_soft_prompt_text(self, text, ctx, adapt_text=False):
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(cast_dtype)
        ctx = ctx.to(device=x.device, dtype=cast_dtype)
        if ctx.shape[0] != self.soft_prompt_ctx_len:
            raise ValueError(f"ctx length {ctx.shape[0]} != expected {self.soft_prompt_ctx_len}")
        if text.shape[1] < 1 + ctx.shape[0]:
            raise ValueError("tokenized prompt is shorter than soft context length")
        x = x.clone()
        x[:, 1:1 + ctx.shape[0], :] = ctx.unsqueeze(0)
        return self._encode_text_from_embeddings(text, x, adapt_text=adapt_text)

    def encode_dynamic_prompt_text(
            self, token_ids, dynamic_context, state_token=None, class_token=None, adapt_text=True
    ):
        """Encode per-sample dynamic contexts in one vectorized text forward.

        ``dynamic_context`` is [N, ctx_len, 768], so a Progress 1 batch only
        encodes B * 2 * M prompts and never loops over patches.
        """
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(token_ids).to(cast_dtype)
        dynamic_context = dynamic_context.to(device=x.device, dtype=cast_dtype)
        if dynamic_context.ndim != 3:
            raise ValueError("dynamic_context must be [N, ctx_len, 768]")
        if dynamic_context.shape[0] != x.shape[0]:
            raise ValueError("token_ids and dynamic_context must have the same batch length")
        if dynamic_context.shape[1] != self.soft_prompt_ctx_len:
            raise ValueError(
                f"dynamic ctx length {dynamic_context.shape[1]} != expected {self.soft_prompt_ctx_len}"
            )
        if dynamic_context.shape[2] != x.shape[2] or token_ids.shape[1] < 1 + dynamic_context.shape[1]:
            raise ValueError("dynamic context shape is incompatible with CLIP token embeddings")
        x = x.clone()
        x[:, 1:1 + dynamic_context.shape[1], :] = dynamic_context
        if (state_token is None) != (class_token is None):
            raise ValueError("STATE and CLASS tokens must be provided together")
        if state_token is not None:
            state_position = 1 + dynamic_context.shape[1]
            class_position = state_position + 1
            if token_ids.shape[1] <= class_position:
                raise ValueError("structured prompt is too short for STATE and CLASS tokens")
            if state_token.shape != (x.shape[0], x.shape[2]) or class_token.shape != state_token.shape:
                raise ValueError("STATE and CLASS tokens must be [N, text_dim]")
            x[:, state_position, :] = state_token.to(device=x.device, dtype=cast_dtype)
            x[:, class_position, :] = class_token.to(device=x.device, dtype=cast_dtype)
        return self._encode_text_from_embeddings(token_ids, x, adapt_text=adapt_text)
