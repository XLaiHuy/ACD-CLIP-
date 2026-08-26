from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tools.sabra_v2.correction_teacher import build_source_teacher_region_target
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import (
    CacheProvenance,
    CachedSourceDataset,
    TierADataset,
    stable_sample_id,
    validate_tier_a_shard,
    write_tier_a_shard,
    write_tier_b_shard,
)
from tools.sabra_v2.student_forward import forward_region_student
from utils import calculate_seg_loss


def _row(class_name: str, suffix: str) -> dict[str, object]:
    return {"class_name": class_name, "image_path": f"{class_name}/{suffix}.JPG", "label": 0}


def _frozen(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn((3, 1369, 768), generator=generator, dtype=torch.float32),
        torch.randn((3, 1369, 2), generator=generator, dtype=torch.float32),
    )


def _provenance(name: str = "execution") -> CacheProvenance:
    return CacheProvenance(scientific_execution_base_sha=name, metadata_sha256="metadata")


def _write_a(root: Path, rows: list[dict[str, object]], values: list[tuple[torch.Tensor, torch.Tensor]], provenance: CacheProvenance) -> None:
    by_class: dict[str, list[tuple[dict[str, object], tuple[torch.Tensor, torch.Tensor]]]] = {}
    for row, value in zip(rows, values):
        by_class.setdefault(str(row["class_name"]), []).append((row, value))
    for class_name, pairs in by_class.items():
        write_tier_a_shard(
            root,
            class_name,
            [stable_sample_id(row) for row, _ in pairs],
            (value for _, value in pairs),
            provenance,
        )


def test_frozen_features_native_logits_and_order_roundtrip_exactly(tmp_path: Path) -> None:
    rows = [_row("capsules", "b"), _row("candle", "a"), _row("capsules", "c")]
    values = [_frozen(1), _frozen(2), _frozen(3)]
    provenance = _provenance()
    _write_a(tmp_path, rows, values, provenance)

    cached = TierADataset(rows, tmp_path, provenance)

    assert [cached[index]["sample_id"] for index in range(3)] == [stable_sample_id(row) for row in rows]
    for index, (seg_features, native_logits) in enumerate(values):
        assert torch.equal(cached[index]["seg_features"], seg_features)
        assert torch.equal(cached[index]["native_logits"], native_logits)


def test_teacher_student_loss_gradient_and_optimizer_step_cache_parity(tmp_path: Path) -> None:
    row = _row("capsules", "source")
    seg_features, native_logits = _frozen(4)
    provenance = _provenance()
    _write_a(tmp_path, [row], [(seg_features, native_logits)], provenance)
    source_mask = torch.zeros((1, 518, 518), dtype=torch.float32)
    source_mask[:, 100:220, 200:340] = 1.0
    teacher = build_source_teacher_region_target(native_logits.unsqueeze(1), source_mask.unsqueeze(0))[0]
    write_tier_b_shard(
        tmp_path,
        "candle",
        [row],
        [(source_mask, teacher)],
        provenance,
        source_mask_file_reads=1,
    )
    cached = CachedSourceDataset([row], "candle", tmp_path, provenance)[0]
    assert torch.equal(cached["teacher_region"], teacher)

    direct_adapter = RegionResidualAdapter()
    cached_adapter = RegionResidualAdapter()
    cached_adapter.load_state_dict(direct_adapter.state_dict())
    direct_optimizer = torch.optim.AdamW(direct_adapter.parameters(), lr=1e-3)
    cached_optimizer = torch.optim.AdamW(cached_adapter.parameters(), lr=1e-3)

    def step(adapter: RegionResidualAdapter, optimizer: torch.optim.Optimizer, seg: torch.Tensor, native: torch.Tensor, mask: torch.Tensor, target: torch.Tensor):
        output = forward_region_student(adapter, seg.unsqueeze(1), native.unsqueeze(1))
        distillation = F.smooth_l1_loss(output.region_residual, target.unsqueeze(0).unsqueeze(0).expand(3, -1, -1, -1))
        localization = calculate_seg_loss(output.deployed_probability, mask.unsqueeze(0))
        loss = distillation + localization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = {name: parameter.grad.detach().clone() for name, parameter in adapter.named_parameters()}
        optimizer.step()
        return output, loss.detach(), gradients

    direct = step(direct_adapter, direct_optimizer, seg_features, native_logits, source_mask, teacher)
    from_cache = step(
        cached_adapter,
        cached_optimizer,
        cached["seg_features"],
        cached["native_logits"],
        cached["mask"],
        cached["teacher_region"],
    )
    assert torch.equal(direct[0].region_residual, from_cache[0].region_residual)
    assert torch.equal(direct[0].corrected_logits, from_cache[0].corrected_logits)
    assert torch.equal(direct[1], from_cache[1])
    for name in direct[2]:
        assert torch.equal(direct[2][name], from_cache[2][name])
    for name, parameter in direct_adapter.state_dict().items():
        assert torch.equal(parameter, cached_adapter.state_dict()[name])


def test_cached_source_inventory_structurally_rejects_held_class_and_records_zero_held_reads(tmp_path: Path) -> None:
    source = _row("capsules", "source")
    held = _row("candle", "held")
    provenance = _provenance()
    _write_a(tmp_path, [source, held], [_frozen(5), _frozen(6)], provenance)
    mask = torch.zeros((1, 518, 518), dtype=torch.float32)
    teacher = torch.zeros((9, 9), dtype=torch.float32)
    manifest = write_tier_b_shard(tmp_path, "candle", [source], [(mask, teacher)], provenance, source_mask_file_reads=0)

    assert manifest["source_classes"] == ["capsules"]
    assert manifest["held_mask_reads"] == 0
    assert len(CachedSourceDataset([source], "candle", tmp_path, provenance)) == 1
    with pytest.raises(RuntimeError, match="held class"):
        CachedSourceDataset([source, held], "candle", tmp_path, provenance)


def test_cached_source_can_omit_unused_mask_and_native_logit_copies(tmp_path: Path) -> None:
    source = _row("capsules", "source")
    provenance = _provenance()
    _write_a(tmp_path, [source], [_frozen(9)], provenance)
    source_mask = torch.zeros((1, 518, 518), dtype=torch.float32)
    teacher = torch.ones((9, 9), dtype=torch.float32)
    write_tier_b_shard(
        tmp_path,
        "candle",
        [source],
        [(source_mask, teacher)],
        provenance,
        source_mask_file_reads=1,
    )

    cached = CachedSourceDataset(
        [source],
        "candle",
        tmp_path,
        provenance,
        load_source_mask=False,
        load_native_logits=False,
    )[0]
    assert "mask" not in cached
    assert "native_logits" not in cached
    assert torch.equal(cached["teacher_region"], teacher)


def test_cache_rejects_wrong_provenance_and_incomplete_shards(tmp_path: Path) -> None:
    row = _row("candle", "sample")
    expected = _provenance("expected")
    _write_a(tmp_path, [row], [_frozen(7)], expected)
    shard = tmp_path / "tier_a" / "candle"
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        validate_tier_a_shard(shard, "candle", [stable_sample_id(row)], _provenance("wrong"))

    manifest_path = shard / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["completion_status"] = "INCOMPLETE"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="incomplete cache"):
        validate_tier_a_shard(shard, "candle", [stable_sample_id(row)], expected)


def test_adapter_checkpoint_reload_is_exact_after_cached_step(tmp_path: Path) -> None:
    adapter = RegionResidualAdapter()
    with torch.no_grad():
        adapter.output.bias.copy_(torch.tensor([0.25, -0.5, 0.75]))
    checkpoint = tmp_path / "adapter.pt"
    torch.save({"state_dict": adapter.state_dict()}, checkpoint)
    restored = RegionResidualAdapter()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"])
    features = _frozen(8)[0].unsqueeze(1)
    assert torch.equal(adapter(features), restored(features))
