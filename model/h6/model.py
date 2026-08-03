"""Progress 1 integration: one dynamic semantic bank shared by both paths."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .router import PatchRouter
from .semantic_bank import BoundedPositiveGate, CoPSSemanticCore


class H6Progress1(nn.Module):
    """The complete Progress 1 semantic core, without visual experts.

    ``build_batch`` deliberately accepts the owning ACDCLIP model.  H6 owns the
    image-conditioned state and router, while ACDCLIP remains the sole owner of
    the Phase2B text encoder and hard-prompt helpers.  That keeps a single text
    bank and avoids a second visual forward for dynamic prompts.
    """

    progress = 1

    def __init__(
        self,
        n_groups: int,
        num_factors: int = 4,
        top_k: int = 2,
        bank_dim: int = 256,
        router_dim: int = 128,
        router_temperature: float = 1.0,
        router_soft_epochs: int = 2,
        vae_hidden_dim: int = 512,
        vae_latent_dim: int = 256,
        text_dim: int = 768,
        ctx_len: int = 4,
        h6_logit_temperature: float = 10.0,
    ):
        super().__init__()
        self.n_groups = int(n_groups)
        self.num_factors = int(num_factors)
        self.top_k = int(top_k)
        self.bank_dim = int(bank_dim)
        self.router_dim = int(router_dim)
        self.router_temperature = float(router_temperature)
        self.router_soft_epochs = int(router_soft_epochs)
        self.vae_hidden_dim = int(vae_hidden_dim)
        self.vae_latent_dim = int(vae_latent_dim)
        self.text_dim = int(text_dim)
        self.ctx_len = int(ctx_len)
        self.h6_logit_temperature = float(h6_logit_temperature)
        self.semantic_core = CoPSSemanticCore(
            n_groups=n_groups,
            num_factors=num_factors,
            bank_dim=bank_dim,
            text_dim=text_dim,
            ctx_len=ctx_len,
            vae_hidden_dim=vae_hidden_dim,
            vae_latent_dim=vae_latent_dim,
        )
        self.router = PatchRouter(
            n_groups=n_groups,
            num_factors=num_factors,
            text_dim=text_dim,
            hidden_dim=router_dim,
            temperature=router_temperature,
            soft_routing_epochs=router_soft_epochs,
            top_k=top_k,
        )
        self.rho = BoundedPositiveGate(initial=0.05, maximum=0.50, count=n_groups)
        self.epoch_one_based = 1

    def config_dict(self) -> Dict[str, int | float]:
        return {
            "progress": self.progress,
            "n_groups": self.n_groups,
            "num_factors": self.num_factors,
            "top_k": self.top_k,
            "bank_dim": self.bank_dim,
            "router_dim": self.router_dim,
            "router_temperature": self.router_temperature,
            "router_soft_epochs": self.router_soft_epochs,
            "vae_hidden_dim": self.vae_hidden_dim,
            "vae_latent_dim": self.vae_latent_dim,
            "text_dim": self.text_dim,
            "ctx_len": self.ctx_len,
            "h6_logit_temperature": self.h6_logit_temperature,
            "rho_init": 0.05,
            "rho_max": 0.50,
        }

    def set_epoch(self, epoch_one_based: int) -> None:
        self.epoch_one_based = int(epoch_one_based)

    def rho_cap(self) -> float:
        return min(0.50, 0.10 * max(1, self.epoch_one_based))

    def rho_values(self) -> torch.Tensor:
        return self.rho(cap=self.rho_cap())

    def forward_core(self, visual_output: Dict[str, torch.Tensor], ctx_normal: torch.Tensor, ctx_abnormal: torch.Tensor, debug: bool = False) -> Dict[str, torch.Tensor]:
        return self.semantic_core(
            visual_output["seg_tokens_pre_l2"],
            visual_output["cls24"],
            ctx_normal,
            ctx_abnormal,
            debug=debug,
        )

    @staticmethod
    def _batch_hard_embeddings(base_model, dataset_name: str, class_names: Sequence[str], device: torch.device):
        # Local import avoids an adapter -> h6 -> utils import cycle during module import.
        from utils import get_hard_anchor_single_class_text_embedding, get_hard_phase1_single_class_text_embedding

        adapted_cache = {}
        frozen_cache = {}
        adapted = []
        frozen = []
        for class_name in class_names:
            if class_name not in adapted_cache:
                adapted_cache[class_name] = get_hard_phase1_single_class_text_embedding(
                    base_model, dataset_name, class_name, device, adapt_text=True
                )
                frozen_cache[class_name] = get_hard_anchor_single_class_text_embedding(
                    base_model, dataset_name, class_name, device
                )
            adapted.append(adapted_cache[class_name])
            frozen.append(frozen_cache[class_name])
        hard_adapted = torch.stack(adapted, dim=1).float()  # [G,B,D,2]
        hard_frozen = torch.stack(frozen, dim=1).detach().float()  # [G,B,D,2]
        return hard_adapted, hard_frozen

    def _encode_dynamic_bank(self, base_model, dataset_name: str, class_names: Sequence[str], dynamic_contexts: torch.Tensor) -> torch.Tensor:
        from utils import get_real_name, get_soft_prompt_sentence
        from model.tokenizer import tokenize

        batch, factors, states, ctx_len, text_dim = dynamic_contexts.shape
        if factors != self.num_factors or states != 2 or ctx_len != self.ctx_len or text_dim != self.text_dim:
            raise ValueError("dynamic contexts do not match the H6 configuration")
        sentences = []
        for class_name in class_names:
            real_name = get_real_name(dataset_name, class_name)
            for _ in range(self.num_factors):
                for state in range(2):
                    sentences.append(get_soft_prompt_sentence(real_name, state, self.ctx_len))
        token_ids = tokenize(sentences).to(dynamic_contexts.device)
        contexts = dynamic_contexts.reshape(batch * factors * states, ctx_len, text_dim)
        text_levels = base_model.encode_dynamic_prompt_text(token_ids, contexts, adapt_text=False)
        dynamic = torch.stack(text_levels, dim=0).view(
            self.n_groups, batch, factors, states, self.text_dim
        )
        return dynamic.permute(0, 1, 2, 4, 3).contiguous().float()

    def build_batch(
        self,
        base_model,
        dataset_name: str,
        class_names: Sequence[str],
        visual_output: Dict[str, torch.Tensor],
        hybrid_alpha: float,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Create the unique dynamic factor bank and route every patch through it."""
        if len(class_names) != visual_output["cls24"].shape[0]:
            raise ValueError("class_names must have one entry per image")
        core = self.forward_core(
            visual_output,
            base_model.soft_prompt.ctx_normal,
            base_model.soft_prompt.ctx_abnormal,
            debug=debug,
        )
        dynamic = self._encode_dynamic_bank(base_model, dataset_name, class_names, core["dynamic_contexts"])
        hard_adapted, hard_frozen = self._batch_hard_embeddings(
            base_model, dataset_name, class_names, visual_output["cls24"].device
        )
        factor_bank = F.normalize(
            (1.0 - float(hybrid_alpha)) * hard_adapted.unsqueeze(2)
            + float(hybrid_alpha) * dynamic,
            dim=3,
        )
        anchor = hard_frozen.unsqueeze(2).expand_as(dynamic)
        kg_loss = (1.0 - F.cosine_similarity(dynamic.float(), anchor, dim=3)).mean()
        routing = self.router(visual_output["seg_tokens"], epoch_one_based=self.epoch_one_based)
        local_text = self.router.local_text(routing["probabilities"], factor_bank)
        patches = torch.stack(visual_output["seg_tokens"], dim=0).float()
        patches = F.normalize(patches, dim=-1)
        h6_logits = self.h6_logit(patches, local_text)
        return {
            **core,
            **routing,
            "hard_adapted": hard_adapted,
            "hard_frozen": hard_frozen,
            "dynamic_text": dynamic,
            "factor_bank": factor_bank,
            "kg_loss": kg_loss,
            "text_global": self.router.aggregate_global(routing["probabilities"], factor_bank),
            "local_text": local_text,
            "h6_logits": h6_logits,
            "rho": self.rho_values(),
            "router_diagnostics": self.router.diagnostics(routing["probabilities"]),
        }

    def h6_logit(self, normalized_patches: torch.Tensor, local_text: torch.Tensor) -> torch.Tensor:
        normal = local_text[..., 0]
        abnormal = local_text[..., 1]
        normal_similarity = (normalized_patches.float() * normal.float()).sum(dim=-1)
        abnormal_similarity = (normalized_patches.float() * abnormal.float()).sum(dim=-1)
        return self.h6_logit_temperature * (abnormal_similarity - normal_similarity)

    def parameter_partitions(self) -> Dict[str, Iterable[nn.Parameter]]:
        """Named optimizer partitions matching the Progress 1 protocol."""
        return {
            "h6_projectors": self.semantic_core.level_projectors.parameters(),
            "h6_concepts": list(self.semantic_core.normal_query.parameters())
            + list(self.semantic_core.abnormal_query.parameters())
            + list(self.semantic_core.router_key.parameters())
            + [self.semantic_core.concept_slots, self.semantic_core.level_embedding],
            "h6_prototype": list(self.semantic_core.prototype_attention.parameters())
            + list(self.semantic_core.normal_state_update.parameters())
            + list(self.semantic_core.abnormal_state_update.parameters()),
            "h6_vae": self.semantic_core.class_vae.parameters(),
            "h6_router": self.router.parameters(),
            "h6_dynamic_prompt": list(self.semantic_core.state_to_context_normal.parameters())
            + list(self.semantic_core.state_to_context_abnormal.parameters())
            + list(self.semantic_core.class_to_context.parameters()),
            "h6_gates": list(self.semantic_core.gamma_state.parameters())
            + list(self.semantic_core.gamma_class.parameters())
            + list(self.rho.parameters()),
        }
