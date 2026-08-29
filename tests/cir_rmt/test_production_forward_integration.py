from pathlib import Path

import pytest
import torch
from torch import nn

import tools.cir_rmt.runtime as cir_runtime
from tools.cir_rmt.core import transport_pair
from tools.cir_rmt.identity import load_cir_config
from tools.cir_rmt.runtime import forward_cir


class SyntheticParent(nn.Module):
    """Small parent-shaped module exposing only the production CIR contract."""

    def __init__(self, batch: int = 2, groups: int = 3, dim: int = 768, patches: int = 1369):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.tensor([[[-0.8, 0.4], [0.1, -0.2], [0.7, 0.3]], [[0.5, -0.4], [-0.3, 0.6], [0.2, -0.1]], [[-0.2, 0.8], [0.4, -0.5], [0.0, 0.2]]], dtype=torch.float32))
        generator = torch.Generator().manual_seed(31415)
        seg = torch.randn(3, patches, dim, generator=generator)
        self.register_buffer("seg_base", torch.nn.functional.normalize(seg, dim=-1))
        self.register_buffer("det_base", torch.nn.functional.normalize(torch.randn(3, dim, generator=generator), dim=-1))
        self.batch = batch
        self.groups = groups
        self.fusion_calls = 0

    def forward(self, image, return_phase4_features=False):
        batch = int(image.shape[0])
        return {
            "seg_tokens": [value.unsqueeze(0).expand(batch, -1, -1) for value in self.seg_base],
            "det_tokens": [value.unsqueeze(0).expand(batch, -1) for value in self.det_base],
        }

    def compute_dfg_weights(self, img_feat, group_text_features, group_index):
        values = torch.softmax(self.gate_logits[int(group_index)], dim=0)
        weights = values.unsqueeze(0).expand(img_feat.shape[0], -1, -1)
        return {"normal": weights[..., 0], "abnormal": weights[..., 1]}
    def apply_dfg_weights(self, group_text_features, weights_normal, weights_abnormal):
        normal_text = torch.einsum("bg,bgd->bd", weights_normal, group_text_features[..., 0])
        abnormal_text = torch.einsum("bg,bgd->bd", weights_abnormal, group_text_features[..., 1])
        return torch.stack([normal_text, abnormal_text], dim=-1)


    def vision_text_fusion_gate_seg(self, vision_tokens, text_features, img_size=518, test_mode=False, domain="Industrial", return_details=False):
        self.fusion_calls += 1
        group_text = text_features.permute(1, 0, 2, 3)
        logits = []
        for group_index in range(vision_tokens.shape[0]):
            image_features = 10.0 * vision_tokens[group_index]
            weights = self.compute_dfg_weights(image_features, group_text, group_index)
            fused_text = self.apply_dfg_weights(group_text, weights["normal"], weights["abnormal"])
            logits.append(torch.matmul(image_features, fused_text))
        base = torch.stack(logits, dim=0)
        margins = base[..., 1] - base[..., 0]
        dummy = torch.zeros((vision_tokens.shape[1], 2, img_size, img_size), device=vision_tokens.device)
        return dummy, base, margins


def _text_features(batch: int = 2, groups: int = 3, dim: int = 768) -> torch.Tensor:
    text = torch.zeros(batch, groups, dim, 2)
    for group in range(groups):
        text[:, group, 2 * group, 0] = 1.0
        text[:, group, 2 * group + 1, 1] = 1.0
    return text


def test_production_forward_executes_cir_transport_and_backpropagates():
    torch.manual_seed(2718)
    config = dict(load_cir_config())
    batch, groups, patches, dim = 2, 3, 1369, 768
    model = SyntheticParent(batch=batch, groups=groups, dim=dim, patches=patches)
    image = torch.zeros(batch, 3, 518, 518)
    text = _text_features(batch=batch, groups=groups, dim=dim)
    calls = []
    original = cir_runtime.cir_logits_from_native_weights

    def spy(*args, **kwargs):
        calls.append({"alpha": args[4], "score_mode": kwargs.get("score_mode")})
        return original(*args, **kwargs)

    original_names = [name for name, _ in model.named_parameters()]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cir_runtime, "cir_logits_from_native_weights", spy)
        output = forward_cir(model, image, ["class_a", "class_b"], torch.device("cpu"), config, require_grad=True, dataset_name="VisA", precomputed_text_features=text)

    assert calls == [{"alpha": config["rmt_transport_alpha"], "score_mode": "optimized"}]
    assert model.fusion_calls == 0
    assert output.native_group_margin.shape == (3, batch, patches, groups)
    assert output.peer_margins.shape == (3, batch, patches, 8, groups)
    assert output.delta.shape == (3, batch, patches, groups)
    assert output.delta_stats["center"].shape == (3, batch, patches, groups)
    assert output.delta_stats["mad"].shape == (3, batch, patches, groups)
    assert output.delta_stats["z"].shape == (3, batch, patches, groups)
    _, parent_logits, parent_margin = model.vision_text_fusion_gate_seg(
        output.seg_features, text.permute(1, 0, 2, 3), return_details=True
    )
    assert torch.allclose(output.native_logits, parent_logits)
    assert torch.allclose(output.native_margin, parent_margin)
    assert torch.isfinite(output.delta).all()
    assert not output.delta.requires_grad
    assert output.peer_valid.any()
    assert output.delta_stats["transport"]["delta_abs_mean"] >= 0.0
    assert output.cir_logits.requires_grad
    assert output.native_weights.requires_grad

    evidence = output.delta
    native_normal = output.native_weights[..., 0].unsqueeze(2).expand(3, batch, patches, groups)
    native_abnormal = output.native_weights[..., 1].unsqueeze(2).expand(3, batch, patches, groups)
    transported_normal, transported_abnormal = transport_pair(native_normal, native_abnormal, evidence, float(config["rmt_transport_alpha"]))
    assert torch.max((transported_normal - native_normal).abs()).item() > 1e-7
    assert torch.max((transported_abnormal - native_abnormal).abs()).item() > 1e-7

    loss = output.cir_training_segmentation_probability[:, 1].mean() + output.classification_logits[:, 1].mean()
    loss.backward()
    assert model.gate_logits.grad is not None
    assert torch.isfinite(model.gate_logits.grad).all()
    assert model.gate_logits.grad.abs().sum().item() > 0.0
    assert original_names == ["gate_logits"]
    assert [name for name, _ in model.named_parameters()] == original_names

    repo_root = Path(__file__).resolve().parents[2]
    train_source = (repo_root / "scripts/cir_rmt/train_full.py").read_text(encoding="utf-8")
    eval_source = (repo_root / "scripts/cir_rmt/eval_full.py").read_text(encoding="utf-8")
    runtime_source = (repo_root / "tools/cir_rmt/runtime.py").read_text(encoding="utf-8")
    assert "vision_text_fusion_gate_seg(" not in runtime_source
    assert "_per_group_margins(" in runtime_source
    assert "forward_cir(" in train_source
    assert "forward_cir(" in eval_source
    assert "forward_phase2b(" not in train_source
    assert "SABRA" not in train_source
    assert "SABRA" not in eval_source
    runner_source = (repo_root / "scripts/cir_rmt/run_full_cir_v1.sh").read_text(encoding="utf-8")
    assert "RELEASE_LOCK=TRUE" not in runner_source
    assert "SABRA" not in runner_source
