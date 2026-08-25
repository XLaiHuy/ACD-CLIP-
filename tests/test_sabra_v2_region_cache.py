from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import (
    CACHE_SCHEMA,
    CachedRegionDataset,
    CacheProvenance,
    RegionCacheWriter,
    preserve_rng_state,
    source_inventory_digest,
    validate_region_cache,
)
from tools.sabra_v2.student_forward import forward_region_student
from tools.sabra_v2.run_p27_science import FOLD_ORDER, _aggregate
from utils import calculate_seg_loss


SHAPES = {
    "seg_features": (3, 1369, 768),
    "native_logits": (3, 1369, 2),
    "teacher_region": (1, 9, 9),
    "source_mask": (1, 518, 518),
}


def _rows() -> list[dict[str, object]]:
    return [
        {"class_name": "capsules", "image_path": "capsules/a.png", "mask_path": "capsules/a-mask.png", "label": 1},
        {"class_name": "cashew", "image_path": "cashew/b.png", "label": 0},
    ]


def _provenance(**changes: object) -> CacheProvenance:
    values = dict(
        held_class="candle",
        source_classes=("capsules", "cashew"),
        source_inventory_sha256=source_inventory_digest(_rows()),
        source_files_sha256="source-files",
        p26_checkpoint_sha256="p26",
        clip_asset_sha256="clip",
        config_sha256="config",
        protocol_sha256="protocol",
        dataset_root="/data/VisA",
    )
    values.update(changes)
    return CacheProvenance(**values)


def _sample(seed: int = 3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    sample = {name: torch.randn(shape, generator=generator, dtype=torch.float32) for name, shape in SHAPES.items()}
    sample["source_mask"] = (sample["source_mask"] > 0).to(torch.float32)
    return sample


def _write_cache(path: Path, *, complete: bool = True) -> None:
    writer = RegionCacheWriter(path, _provenance(), _rows())
    writer.append(_sample())
    writer.append(_sample(4))
    if complete:
        writer.finalize()


def _losses(adapter: RegionResidualAdapter, sample: dict[str, torch.Tensor]):
    features = sample["seg_features"].unsqueeze(1)
    logits = sample["native_logits"].unsqueeze(1)
    teacher = sample["teacher_region"].unsqueeze(0)
    mask = sample["source_mask"].unsqueeze(0)
    student = forward_region_student(adapter, features, logits)
    distillation = F.smooth_l1_loss(student.region_residual, teacher.expand(3, -1, -1, -1))
    localization = calculate_seg_loss(student.deployed_probability, mask)
    return student, distillation, localization, distillation + localization


def test_exact_cached_tensor_and_student_loss_gradient_optimizer_parity(tmp_path: Path) -> None:
    _write_cache(tmp_path)
    cached = CachedRegionDataset(validate_region_cache(tmp_path, _provenance(), _rows()))[0]
    direct = _sample()
    for key in SHAPES:
        assert torch.equal(cached[key], direct[key]), key

    torch.manual_seed(9)
    direct_adapter = RegionResidualAdapter()
    cached_adapter = copy.deepcopy(direct_adapter)
    direct_optimizer = torch.optim.AdamW(direct_adapter.parameters(), lr=1e-3)
    cached_optimizer = torch.optim.AdamW(cached_adapter.parameters(), lr=1e-3)
    direct_output, direct_distill, direct_localize, direct_total = _losses(direct_adapter, direct)
    cached_output, cached_distill, cached_localize, cached_total = _losses(cached_adapter, cached)

    assert torch.equal(direct_output.region_residual, cached_output.region_residual)
    assert torch.equal(direct_distill, cached_distill)
    assert torch.equal(direct_localize, cached_localize)
    assert torch.equal(direct_total, cached_total)
    direct_total.backward(); cached_total.backward()
    for left, right in zip(direct_adapter.parameters(), cached_adapter.parameters()):
        assert torch.equal(left.grad, right.grad)
    direct_optimizer.step(); cached_optimizer.step()
    for left, right in zip(direct_adapter.parameters(), cached_adapter.parameters()):
        assert torch.equal(left, right)


def test_short_multistep_and_checkpoint_reload_parity(tmp_path: Path) -> None:
    _write_cache(tmp_path)
    dataset = CachedRegionDataset(validate_region_cache(tmp_path, _provenance(), _rows()))
    torch.manual_seed(12)
    left = RegionResidualAdapter(); right = copy.deepcopy(left)
    left_opt = torch.optim.AdamW(left.parameters(), lr=1e-3)
    right_opt = torch.optim.AdamW(right.parameters(), lr=1e-3)
    direct_samples = [_sample(), _sample(4), _sample()]
    cached_samples = [dataset[0], dataset[1], dataset[0]]
    for direct, cached in zip(direct_samples, cached_samples):
        left_opt.zero_grad(set_to_none=True); right_opt.zero_grad(set_to_none=True)
        _losses(left, direct)[-1].backward(); _losses(right, cached)[-1].backward()
        left_opt.step(); right_opt.step()
    for left_parameter, right_parameter in zip(left.parameters(), right.parameters()):
        assert torch.equal(left_parameter, right_parameter)
    restored = RegionResidualAdapter(); restored.load_state_dict(right.state_dict())
    assert torch.equal(restored(dataset[0]["seg_features"].unsqueeze(1)), right(dataset[0]["seg_features"].unsqueeze(1)))


def test_cache_is_source_only_and_records_zero_held_reads(tmp_path: Path) -> None:
    _write_cache(tmp_path)
    validated = validate_region_cache(tmp_path, _provenance(), _rows())
    assert validated.manifest["schema_version"] == CACHE_SCHEMA
    assert validated.manifest["held_class"] == "candle"
    assert validated.manifest["source_classes"] == ["capsules", "cashew"]
    assert validated.manifest["held_gt_reads"] == 0
    assert validated.manifest["held_mask_reads"] == 0
    assert all(row["class_name"] != "candle" for row in validated.manifest["records"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("held_class", "capsules", "held class"),
        ("p26_checkpoint_sha256", "wrong", "P26"),
        ("clip_asset_sha256", "wrong", "CLIP"),
        ("config_sha256", "wrong", "config"),
        ("source_files_sha256", "wrong", "source file inventory"),
    ],
)
def test_cache_rejects_wrong_fold_or_asset_hash(tmp_path: Path, field: str, value: str, message: str) -> None:
    _write_cache(tmp_path)
    with pytest.raises(RuntimeError, match=message):
        validate_region_cache(tmp_path, _provenance(**{field: value}), _rows())


def test_cache_rejects_incomplete_cache(tmp_path: Path) -> None:
    _write_cache(tmp_path, complete=False)
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_region_cache(tmp_path, _provenance(), _rows())


def test_cache_rejects_source_inventory_mismatch(tmp_path: Path) -> None:
    _write_cache(tmp_path)
    changed = _rows() + [{"class_name": "pcb1", "image_path": "pcb1/c.png", "label": 0}]
    with pytest.raises(RuntimeError, match="source inventory"):
        validate_region_cache(tmp_path, _provenance(), changed)


def test_cache_rejects_corrupted_tensor_file(tmp_path: Path) -> None:
    _write_cache(tmp_path)
    tensor_path = tmp_path / "native_logits.bin"
    with tensor_path.open("r+b") as handle:
        handle.seek(0); handle.write(b"bad!")
    with pytest.raises(RuntimeError, match="checksum"):
        validate_region_cache(tmp_path, _provenance(), _rows())


def test_cache_construction_preserves_python_numpy_torch_and_cuda_rng() -> None:
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    with preserve_rng_state():
        random.random(); np.random.rand(); torch.rand(2)
        if torch.cuda.is_available(): torch.rand(2, device="cuda")
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    if torch.cuda.is_available():
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_before))


def test_complete_aggregation_reports_paired_metrics_breadth_and_concentration() -> None:
    metrics = {
        held: {
            "native_metrics": {"pAP": 0.5, "pAUROC": 0.8},
            "p27_metrics": {"pAP": 0.5 + (index - 5) / 1000, "pAUROC": 0.8 + index / 10000},
        }
        for index, held in enumerate(FOLD_ORDER)
    }
    aggregate = _aggregate(metrics)
    assert len(aggregate["per_class"]) == 12
    assert aggregate["breadth"] == {"improving_pAP": 6, "non_regressing_pAP": 7, "regressing_pAP": 5}
    assert aggregate["best_category"]["held_class"] == FOLD_ORDER[-1]
    assert aggregate["worst_category"]["held_class"] == FOLD_ORDER[0]
    assert aggregate["gain_concentration"]["top_1_fraction_of_net_gain"] is not None
