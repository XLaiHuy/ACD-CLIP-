from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from tools.sabra_v2 import cuda_runtime
from tools.sabra_v2.p26_parent import (
    P26_CLIP_ASSET_SHA256,
    P26_PHASE2B_CHECKPOINT_SHA256,
    P26_RUNTIME_CONFIG_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
HOST_DRIVER = "/usr/lib/x86_64-linux-gnu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_runtime_environment_places_verified_host_driver_first() -> None:
    source = {
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/compat:/safe/lib:/usr/lib/x86_64-linux-gnu:/safe/lib",
        "CUDA_HOME": "/usr/local/cuda",
        "CUDA_PATH": "/usr/local/cuda",
    }
    result = cuda_runtime.build_p27_cuda_environment(source)

    assert result["LD_LIBRARY_PATH"].split(":") == [HOST_DRIVER, "/safe/lib"]
    assert result["CUDA_HOME"] == source["CUDA_HOME"]
    assert result["CUDA_PATH"] == source["CUDA_PATH"]
    assert result["P27_CUDA_RUNTIME_RECOVERY"] == "HOST_DRIVER_FIRST_V1"
    assert result["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_runtime_environment_removes_compat_and_stub_precedence() -> None:
    source = {
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64/stubs:/x/compatibility/lib:/x/compat:/ordinary/stubs-extra:/ok"
    }
    result = cuda_runtime.build_p27_cuda_environment(source)
    components = result["LD_LIBRARY_PATH"].split(":")

    assert components == [HOST_DRIVER, "/x/compatibility/lib", "/ordinary/stubs-extra", "/ok"]
    assert not any(Path(component).name in {"compat", "stubs"} for component in components)


def test_libcuda_validation_accepts_host_and_rejects_compat_or_stub() -> None:
    assert cuda_runtime.validate_resolved_libcuda(["/usr/lib/x86_64-linux-gnu/libcuda.so.570.172.08"]) == Path(
        "/usr/lib/x86_64-linux-gnu/libcuda.so.570.172.08"
    )
    with pytest.raises(RuntimeError, match="compat"):
        cuda_runtime.validate_resolved_libcuda(["/usr/local/cuda-12.9/compat/libcuda.so.575.57.08"])
    with pytest.raises(RuntimeError, match="stub"):
        cuda_runtime.validate_resolved_libcuda(["/usr/local/cuda/lib64/stubs/libcuda.so"])


def test_exact_runner_constructs_recovered_environment_before_attempt_marker() -> None:
    from tools.sabra_v2 import run_region_distill_science

    source = inspect.getsource(run_region_distill_science.run)
    assert source.index("build_p27_cuda_environment") < source.index("atomic_write_json(attempt_path")
    assert source.index("probe_cuda_subprocess") < source.index("atomic_write_json(attempt_path")
    assert source.index("_prediction_gate") < source.index("score_region_distill_frozen")


def test_frozen_assets_and_protocol_are_unchanged() -> None:
    assert _sha256(ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth") == P26_PHASE2B_CHECKPOINT_SHA256
    assert _sha256(ROOT / "model/ViT-L-14-336px.pt") == P26_CLIP_ASSET_SHA256
    assert _sha256(ROOT / "configs/phase2b_canonical_v1.json") == P26_RUNTIME_CONFIG_SHA256
    assert _sha256(ROOT / "research/sabra_v2/region_distill/P27_PROTOCOL.json") == "992c58ff7cf8b612a701e63a04013703900e37f8222d39e2c4282d6d7536aca7"
