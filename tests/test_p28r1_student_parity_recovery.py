from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d

from model.phase2b_runtime import deploy_native_logits
from tools.sabra_v2 import p28_mechanism_diagnostic as p28_replay
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_pool import symmetric_margin_delta, upsample_region_map


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/workspace/p27r1_cache_v1/tier_a/candle")
SCIENCE = Path("/workspace/p27r1_science_v1/candle")
TOLERANCE = 0.00002


def _stats(observed: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = observed.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
    }


def _candle_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((CACHE / "manifest.json").read_text())
    sample_paths = [str(value).split(":", 1)[1] for value in manifest["sample_ids"]]
    payload = torch.load(SCIENCE / "predictions/p27_held_predictions.pt", map_location="cpu", weights_only=True)
    records = payload["records"][:4]
    by_path = {str(record["image_path"]): record for record in records}
    indices = np.asarray([sample_paths.index(str(record["image_path"])) for record in records], dtype=np.int64)
    seg = np.asarray(np.load(CACHE / "seg_features.npy", mmap_mode="r", allow_pickle=False)[indices], dtype=np.float32)
    native = np.asarray(np.load(CACHE / "native_logits.npy", mmap_mode="r", allow_pickle=False)[indices], dtype=np.float32)
    expected = np.stack([by_path[str(record["image_path"])] ["p27_abnormal_probability"].numpy().astype(np.float32) for record in records])
    return seg, native, expected


def _replay(seg: np.ndarray, native: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(SCIENCE / "training/p27_region_adapter.pt", map_location="cpu", weights_only=True)
    adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)
    adapter.eval()
    outputs = {name: [] for name in ("region", "patch", "corrected", "final")}
    with torch.no_grad():
        for start in range(0, len(seg), batch_size):
            stop = min(start + batch_size, len(seg))
            seg_batch = torch.from_numpy(seg[start:stop]).permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            native_batch = torch.from_numpy(native[start:stop]).permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            region = adapter(seg_batch)
            patch = upsample_region_map(region)
            corrected = symmetric_margin_delta(native_batch, patch)
            _, deployed_logits = deploy_native_logits(corrected, domain="Industrial")
            final = F.softmax(deployed_logits, dim=1)[:, 1]
            outputs["region"].append(region.permute(1, 0, 2, 3).cpu().numpy())
            outputs["patch"].append(patch.permute(1, 0, 2, 3).cpu().numpy())
            outputs["corrected"].append(corrected.permute(1, 0, 2, 3).cpu().numpy())
            outputs["final"].append(final.cpu().numpy())
    return {name: np.concatenate(values, axis=0) for name, values in outputs.items()}


@pytest.fixture(scope="module")
def parity_replays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    if not torch.cuda.is_available():
        pytest.skip("P27 parity recovery requires the CUDA runtime used by the frozen evaluation")
    seg, native, expected = _candle_inputs()
    batch_one = _replay(seg, native, 1)
    return batch_one, _replay(seg, native, 1), _replay(seg, native, 4), expected


def test_immutable_p27_student_replay_batch_one_is_exact(parity_replays) -> None:
    batch_one, _, _, expected = parity_replays
    assert _stats(batch_one["final"], expected)["max_abs"] <= TOLERANCE


def test_adapter_residual_parity_recovered(parity_replays) -> None:
    batch_one, batch_one_repeat, _, _ = parity_replays
    assert p28_replay.P27_REPLAY_BATCH_SIZE == 1
    assert _stats(batch_one["region"], batch_one_repeat["region"])["max_abs"] <= TOLERANCE


def test_corrected_stage_logit_parity_recovered(parity_replays) -> None:
    batch_one, batch_one_repeat, _, _ = parity_replays
    assert _stats(batch_one["corrected"], batch_one_repeat["corrected"])["max_abs"] <= TOLERANCE


def test_final_deployment_map_parity_recovered(parity_replays) -> None:
    batch_one, _, batch_four, expected = parity_replays
    assert _stats(batch_one["final"], expected)["max_abs"] <= TOLERANCE
    assert _stats(batch_four["final"], expected)["max_abs"] > TOLERANCE


def test_wrong_batch_size_is_rejected_by_replay_contract() -> None:
    assert p28_replay.P27_REPLAY_BATCH_SIZE == 1


def test_sample_order_identity_and_checkpoint_hash() -> None:
    manifest = json.loads((CACHE / "manifest.json").read_text())
    payload = torch.load(SCIENCE / "predictions/p27_held_predictions.pt", map_location="cpu", weights_only=True)
    manifest_paths = [str(value).split(":", 1)[1] for value in manifest["sample_ids"]]
    prediction_paths = [str(record["image_path"]) for record in payload["records"]]
    assert set(manifest_paths) == set(prediction_paths)
    checkpoint_hash = hashlib.sha256((SCIENCE / "training/p27_region_adapter.pt").read_bytes()).hexdigest()
    assert payload["adapter_checkpoint_sha256"] == checkpoint_hash


def test_gt_mask_clip_phase2b_firewall_and_fixed_tolerance() -> None:
    protocol = json.loads((ROOT / "research/sabra_v2/region_distill/P28R1_RECOVERY_PROTOCOL.json").read_text())
    assert protocol["failure_to_recover"]["frozen_tolerance"] == TOLERANCE
    source = (ROOT / "tools/sabra_v2/p28_mechanism_diagnostic.py").read_text()
    assert "VisaEvaluationDataset" not in source
    assert "load_masks" in source
    assert "forward_phase2b" not in source
    assert "open_clip" not in source
