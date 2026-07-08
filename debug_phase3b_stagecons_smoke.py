import types

import torch
from torch import nn

from model.adapter import ACDCLIP
from model.adapter_modules import DFGSS2DResidualBranch
from train import compute_stage_routing_consistency


class DummyDfgModel(nn.Module):
    def __init__(self, n_groups=3, dim=768, dfg_attn_dim=32):
        super().__init__()
        self.n_groups = n_groups
        self.dfg_mode = "attn"
        self.dfg_attn_dim = dfg_attn_dim
        self.dfg_attn_tau = 8.0
        self.use_ss2d_dfg = True
        self.dfg_gamma_max = 0.2
        self.dfg_ss2d_fusion = "weight_residual"
        self.dfg_beta = 0.1
        self.dfg_weight_residual_fp32 = True
        self.image_adapter = nn.ModuleDict({
            "vision_text_q": nn.ModuleList([nn.Linear(dim, dfg_attn_dim, bias=False) for _ in range(n_groups)]),
            "vision_text_k": nn.ModuleList([nn.Linear(dim, dfg_attn_dim, bias=False) for _ in range(n_groups)]),
            "dfg_ss2d_branches": nn.ModuleList([DFGSS2DResidualBranch(dim) for _ in range(n_groups)]),
            "dfg_raw_gamma": nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(n_groups)]),
        })
        self._linear_projection = ACDCLIP._linear_projection
        self._attention_scores = ACDCLIP._attention_scores
        self._vision_text_attention_routing_weights = types.MethodType(
            ACDCLIP._vision_text_attention_routing_weights,
            self,
        )


def grad_norm(module):
    grads = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.stack(grads).sum().item())


def main():
    torch.manual_seed(7)
    model = DummyDfgModel()
    seg_features = torch.randn(3, 2, 4, 768, requires_grad=True)
    hard_text = torch.randn(3, 2, 768, 2, requires_grad=True)
    hard_text = torch.nn.functional.normalize(hard_text, dim=2)
    soft_text = torch.randn(3, 2, 768, 2, requires_grad=True)
    soft_text = torch.nn.functional.normalize(soft_text, dim=2)
    soft_text.retain_grad()

    zero_loss, zero_stats = compute_stage_routing_consistency(
        model=model,
        seg_features=seg_features,
        hard_text_features=hard_text,
        soft_text_features=soft_text,
        alpha=0.2,
        loss_type="none",
        margin=0.02,
        detach_visual=True,
        detach_qk=True,
    )
    assert float(zero_loss.item()) == 0.0
    assert zero_stats == {}

    loss, stats = compute_stage_routing_consistency(
        model=model,
        seg_features=seg_features,
        hard_text_features=hard_text,
        soft_text_features=soft_text,
        alpha=1.0,
        loss_type="js",
        margin=0.02,
        detach_visual=True,
        detach_qk=True,
    )
    assert torch.isfinite(loss).all(), "stage loss is not finite"
    assert stats["mean_stage_js"] >= -1e-7, "JS should be non-negative up to numerical error"
    assert max(v for k, v in stats.items() if k.endswith("_sum_error")) < 1e-5

    loss.backward()
    q_grad = grad_norm(model.image_adapter["vision_text_q"])
    k_grad = grad_norm(model.image_adapter["vision_text_k"])
    ss2d_grad = grad_norm(model.image_adapter["dfg_ss2d_branches"])
    raw_gamma_grad = sum(
        0.0 if p.grad is None else float(p.grad.detach().float().abs().item())
        for p in model.image_adapter["dfg_raw_gamma"]
    )
    assert soft_text.grad is not None and float(soft_text.grad.float().norm().item()) > 0
    assert seg_features.grad is None
    assert q_grad == 0.0
    assert k_grad == 0.0
    assert ss2d_grad == 0.0
    assert raw_gamma_grad == 0.0
    print(
        "phase3b_stagecons_smoke ok "
        f"loss={float(loss.item()):.8f} "
        f"mean_js={stats['mean_stage_js']:.8f} "
        f"active={stats['stage_active_fraction']:.4f} "
        f"soft_grad={float(soft_text.grad.float().norm().item()):.8f}"
    )


if __name__ == "__main__":
    main()
